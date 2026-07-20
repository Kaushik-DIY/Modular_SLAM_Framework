"""Offline semantic-mapping pipeline over a finished fusion2 run.

Called by the batch runner (``--semantic``) and by tools/semantic_map_video.py.
Independent of the SLAM mode: it needs only the run dir (anchored
trajectory.tum + map.npy + map_meta.json) and the dataset (RGB-D + LiDAR).

Pipeline: dense segmentation cadence (default one frame every 0.4 s of
dataset time, poses interpolated between keyframes) -> depth back-projection
-> tilt-corrected base frame -> weighted class votes on the run's map grid
-> LiDAR-gated + despeckled semantic raster.

Outputs written into the run dir:
  semantic_map.mp4     4-panel video: RGB POV | semantic POV (top) and
                       LiDAR occupancy building | semantic map building (bottom)
  semantic.png         final clean semantic floor plan + legend
  class_votes.npz      vote tensor (re-render without re-segmenting)
  semantic_gmaps.npz   segmentation cache (re-pace video without re-segmenting)
  semantic_meta.json   params + sanity metrics
"""
from __future__ import annotations

import json
import math
import time
from bisect import bisect_right
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from slam_core.fusion2.Dependencies.dataset import LabHybridStream
from slam_core.fusion2.Dependencies.semantics import (
    FLOOR_GID, GROUPS, PALETTE_BGR, WALL_LIKE_GIDS, SegmenterBackend,
    SemanticVoteGrid, backproject_labeled, cam_to_base, clean_raster,
    content_bbox, estimate_tilt_from_floor, interp_pose, occupied_mask,
    render_semantic_png)

BG = (24, 24, 24)
STRIP_H = 64
WALL_BGR = (35, 35, 35)
TRAJ_BGR = (200, 140, 30)


def _put(img, text, org, scale=0.55, color=(235, 235, 235), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def _load_traj(run_dir: Path) -> np.ndarray:
    rows = []
    for line in open(run_dir / "trajectory.tum"):
        p = line.split()
        if len(p) >= 8:
            rows.append((float(p[0]), float(p[1]), float(p[2]),
                         2.0 * math.atan2(float(p[6]), float(p[7]))))
    return np.array(rows, dtype=np.float64)


def run_semantic_pipeline(run_dir: Path, dataset: Path,
                          model_dir: Path = Path("third_party/models/segformer_b2_ade20k"),
                          speed: float = 4.0, fps: int = 10,
                          seg_interval: float = 0.4, stride: int = 4,
                          min_votes: int = 8, max_depth: float = 4.0,
                          obstacle_max_depth: float = 3.0, alpha: float = 0.5,
                          video: bool = True, refresh: bool = False,
                          lidar_gate: bool = True,
                          out_mp4: Optional[Path] = None) -> dict:
    run_dir, dataset = Path(run_dir), Path(dataset)
    meta = json.load(open(run_dir / "map_meta.json"))
    prob = np.load(run_dir / "map.npy")
    traj = _load_traj(run_dir)
    stream = LabHybridStream(dataset)
    cam = yaml.safe_load(open(dataset / "sensor_config.yaml"))["camera"]
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    depth_factor = float(cam.get("depth_map_factor", 1000.0))
    backend = SegmenterBackend(model_dir)
    grid = SemanticVoteGrid(meta)
    t0, t_end = float(traj[0, 0]), float(traj[-1, 0])

    # ---- dense segmentation set: one RGB-D frame every seg_interval -------
    entries = stream.rgbd_entries()                  # (t, rgb_path, depth_path)
    ets = np.array([e[0] for e in entries])
    seg_idx = []
    last = -1e18
    for j in range(len(entries)):
        t = float(ets[j])
        if t < t0 - 0.1 or t > t_end + 0.1:
            continue
        if t - last >= seg_interval:
            seg_idx.append(j)
            last = t
    print(f"[semantic] run {run_dir.name}: {len(traj)} kf, "
          f"{len(seg_idx)} frames to segment (every {seg_interval}s)")

    # ---- segmentation cache: RAW ADE20K ids keyed by association row ------
    # (raw ids, not group ids, so the class->group mapping can be retuned
    # without re-running the model)
    cache_path = run_dir / "semantic_gmaps.npz"
    raw_maps: dict = {}
    seg_ms: list = []
    if cache_path.exists() and not refresh:
        z = np.load(cache_path)
        if ("raw_aidx" in z.files
                and str(z.get("model", "")) == Path(model_dir).name
                and set(seg_idx) <= set(int(i) for i in z["raw_aidx"])):
            raw_maps = {int(i): g for i, g in zip(z["raw_aidx"], z["raw"])}
            print(f"[semantic] loaded {len(raw_maps)} cached segmentations "
                  f"({Path(model_dir).name})")
    if not all(j in raw_maps for j in seg_idx):
        for n, j in enumerate(seg_idx):
            if j in raw_maps:
                continue
            tt = time.perf_counter()
            raw_maps[j] = backend.segment_raw(cv2.imread(str(entries[j][1])))
            seg_ms.append((time.perf_counter() - tt) * 1e3)
            if n % 100 == 0:
                print(f"[semantic]   segmenting {n}/{len(seg_idx)} "
                      f"(~{np.mean(seg_ms):.0f} ms)", flush=True)
        aidx = np.array(sorted(raw_maps), dtype=np.int32)
        np.savez_compressed(cache_path, raw_aidx=aidx,
                            raw=np.stack([raw_maps[i] for i in aidx]),
                            model=Path(model_dir).name)
        print(f"[semantic] cached {len(raw_maps)} segmentations "
              f"(mean {np.mean(seg_ms):.0f} ms)")
    gmaps = {j: backend.lut[raw_maps[j]] for j in seg_idx}

    # ---- tilt calibration from floor-labeled points ------------------------
    floor_pts = []
    for j in seg_idx:
        if len(floor_pts) >= 30:
            break
        depth = cv2.imread(str(entries[j][2]), cv2.IMREAD_UNCHANGED)
        pc, g = backproject_labeled(gmaps[j], depth, K, stride=stride,
                                    max_depth=max_depth, depth_factor=depth_factor)
        fp = cam_to_base(pc)[g == FLOOR_GID]
        if len(fp):
            floor_pts.append(fp)
    tilt_R, z_floor, inl = (np.eye(3), None, 0.0)
    if floor_pts:
        tilt_R, z_floor, inl = estimate_tilt_from_floor(np.vstack(floor_pts))
    tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, tilt_R[2, 2]))))
    grid.z_floor = z_floor
    print(f"[semantic] tilt {tilt_deg:.2f} deg, floor z="
          f"{'n/a' if z_floor is None else f'{z_floor:.3f} m'}, inliers {inl*100:.0f}%")

    def vote(j):
        depth = cv2.imread(str(entries[j][2]), cv2.IMREAD_UNCHANGED)
        pose = interp_pose(traj, float(ets[j]))
        grid.add_keyframe(gmaps[j], depth, K, pose, stride=stride,
                          max_depth=max_depth, depth_factor=depth_factor,
                          tilt_R=tilt_R, obstacle_max_depth=obstacle_max_depth)

    # ------------------------------------------------------------------ video
    if not video:
        for j in seg_idx:
            vote(j)
    else:
        W, H = 1920, 1080
        ph, pw = (H - STRIP_H) // 2, W // 2
        occ_final = occupied_mask(prob)
        r0, r1, c0, c1 = content_bbox(np.full(prob.shape, -1, np.int16), occ_final)
        ch, cw = r1 - r0, c1 - c0
        ms_ = min((pw - 24) / cw, (ph - 56) / ch)
        mox, moy = int((pw - cw * ms_) / 2), 44 + int((ph - 56 - ch * ms_) / 2)
        res, ox, oy = grid.res, grid.ox, grid.oy

        # progressive LiDAR hit canvas (scans at interpolated poses)
        hits = np.zeros(prob.shape, np.float32)
        scan_iter = iter(stream.lidar_stream())
        pending_scan = next(scan_iter, None)

        def integrate_scans(upto_t):
            nonlocal pending_scan
            while pending_scan is not None and pending_scan[0] <= upto_t:
                st, pts = pending_scan
                if st >= t0:
                    x, y, th = interp_pose(traj, float(st))
                    c, s = math.cos(th), math.sin(th)
                    wx = c * pts[:, 0] - s * pts[:, 1] + x
                    wy = s * pts[:, 0] + c * pts[:, 1] + y
                    ix = np.floor((wx - ox) / res).astype(np.int64)
                    iy = np.floor((wy - oy) / res).astype(np.int64)
                    m = (ix >= 0) & (ix < grid.w) & (iy >= 0) & (iy < grid.h)
                    np.add.at(hits, (iy[m], ix[m]), 1.0)
                pending_scan = next(scan_iter, None)

        def cell_to_px(px0, py0, wx, wy):
            col = (wx - ox) / res - c0
            row = (wy - oy) / res - r0
            return (int(px0 + mox + col * ms_),
                    int(py0 + moy + (ch - 1 - row) * ms_))

        def map_panel(px0, py0, frame, body, title, upto_kf):
            """body: (ch,cw,3) uint8 already flipped for display."""
            rs = cv2.resize(body, (int(cw * ms_), int(ch * ms_)),
                            interpolation=cv2.INTER_NEAREST)
            frame[py0 + moy:py0 + moy + rs.shape[0],
                  px0 + mox:px0 + mox + rs.shape[1]] = rs
            pts = [cell_to_px(px0, py0, traj[k, 1], traj[k, 2])
                   for k in range(0, upto_kf + 1, 2)]
            if len(pts) > 1:
                cv2.polylines(frame, [np.array(pts, np.int32)], False, TRAJ_BGR, 2)
            if pts:
                cv2.circle(frame, pts[-1], 5, (50, 50, 230), -1)
            _put(frame, title, (px0 + 14, py0 + 30), 0.62, thick=2)

        def wall_mask():
            """Progressive reveal of the final (log-odds, free-space-carved,
            despeckled) occupancy: a cell is drawn once the LiDAR has observed
            it at least once, and its occupied state comes from the final
            grid. Raw hit accumulation without free-space carving blackens
            every transient over the run; this keeps reveal timing honest
            while inheriting the final map's crispness."""
            return occ_final & (hits >= 1.0)

        def wall_img(mask):
            return cv2.flip(mask[r0:r1, c0:c1].astype(np.uint8), 0)

        writer_path = Path(out_mp4) if out_mp4 else (run_dir / "semantic_map.mp4")
        writer = cv2.VideoWriter(str(writer_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        speed_txt = ("real-time" if abs(speed - 1.0) < 1e-6
                     else f"{speed:g}x real-time")

        seg_ptr = 0
        tr_panel = None                                     # cached TR composite
        sem_raster_cache = np.full((*prob.shape, 3), 255, np.uint8)  # BR colors
        t_now = t0
        dt = speed / fps
        n_frames = 0
        while t_now <= t_end + 1e-9:
            # advance state to t_now
            integrate_scans(t_now)
            new_seg = False
            while seg_ptr < len(seg_idx) and float(ets[seg_idx[seg_ptr]]) <= t_now:
                j = seg_idx[seg_ptr]
                vote(j)
                rgb_j = cv2.imread(str(entries[j][1]))
                seg_c = backend.colorize(gmaps[j])
                tr_img = cv2.addWeighted(rgb_j, 1 - alpha, seg_c, alpha, 0)
                tr_panel = tr_img
                seg_ptr += 1
                new_seg = True
            wm = wall_mask()
            if new_seg:
                raster = clean_raster(grid.argmax_raster(
                    min_votes, occ_prob=prob if lidar_gate else None))
                body = np.full((*raster.shape, 3), 255, np.uint8)
                for gid, name in enumerate(GROUPS):
                    body[raster == gid] = PALETTE_BGR[name]
                sem_raster_cache = body
            sem_body = sem_raster_cache.copy()
            sem_body[wm] = WALL_BGR
            sem_body = cv2.flip(sem_body[r0:r1, c0:c1], 0)

            kf_i = max(0, int(np.searchsorted(traj[:, 0], t_now)) - 1)
            frame = np.full((H, W, 3), BG, np.uint8)

            # TL: raw RGB at video cadence (continuous POV — no keyframe lag)
            jr = min(len(entries) - 1, max(0, bisect_right(ets, t_now) - 1))
            rgb_now = cv2.imread(str(entries[jr][1]))
            s = min((pw - 24) / rgb_now.shape[1], (ph - 56) / rgb_now.shape[0])
            rs = cv2.resize(rgb_now, (int(rgb_now.shape[1] * s),
                                      int(rgb_now.shape[0] * s)))
            ox_, oy_ = (pw - rs.shape[1]) // 2, 44 + (ph - 56 - rs.shape[0]) // 2
            frame[oy_:oy_ + rs.shape[0], ox_:ox_ + rs.shape[1]] = rs
            _put(frame, "Robot camera (POV)", (14, 30), 0.62, thick=2)

            # TR: semantic segmentation view (latest segmented frame)
            if tr_panel is not None:
                rs = cv2.resize(tr_panel, (int(tr_panel.shape[1] * s),
                                           int(tr_panel.shape[0] * s)))
                frame[oy_:oy_ + rs.shape[0],
                      pw + ox_:pw + ox_ + rs.shape[1]] = rs
            _put(frame, "Semantic segmentation view", (pw + 14, 30), 0.62, thick=2)

            # BL: LiDAR occupancy building
            occ_body = np.full((ch, cw, 3), 255, np.uint8)
            occ_body[wall_img(wm) > 0] = WALL_BGR
            map_panel(0, ph, frame, occ_body,
                      "Geometric map (LiDAR occupancy)", kf_i)

            # BR: semantic map building
            map_panel(pw, ph, frame, sem_body, "Semantic map", kf_i)

            # panel separators + speed badge + legend strip
            cv2.line(frame, (pw, 0), (pw, H - STRIP_H), (70, 70, 70), 1)
            cv2.line(frame, (0, ph), (W, ph), (70, 70, 70), 1)
            _put(frame, f"{speed_txt}   t={t_now - t0:6.1f}s",
                 (W - 360, 30), 0.6, (120, 220, 255), 2)
            lx = 14
            for name in GROUPS:
                if name == "person":
                    continue
                cv2.rectangle(frame, (lx, H - STRIP_H + 22),
                              (lx + 20, H - STRIP_H + 42), PALETTE_BGR[name], -1)
                _put(frame, name, (lx + 25, H - STRIP_H + 39), 0.46,
                     (205, 205, 205))
                lx += 32 + 10 * len(name)
            cv2.rectangle(frame, (lx, H - STRIP_H + 22),
                          (lx + 20, H - STRIP_H + 42), WALL_BGR, -1)
            _put(frame, "LiDAR wall", (lx + 25, H - STRIP_H + 39), 0.46,
                 (205, 205, 205))
            _put(frame, f"kf {kf_i + 1}/{len(traj)}",
                 (W - 170, H - STRIP_H + 39), 0.5, (170, 170, 170))
            writer.write(frame)
            n_frames += 1
            t_now += dt
        writer.release()
        print(f"[semantic] video: {n_frames} frames @ {fps} fps ({speed_txt}) "
              f"-> {writer_path.name}")

    # ---- final outputs + sanity metrics ------------------------------------
    # lidar_gate=False -> pure camera-only semantics (no LiDAR gating in the
    # raster, no LiDAR walls in the PNG). The LiDAR map is STILL used below as
    # the independent evaluation reference, which is exactly what the
    # cross-modal ablation needs. Ungated outputs get a _nogate suffix so both
    # variants can coexist in one run dir.
    suffix = "" if lidar_gate else "_nogate"
    raster = clean_raster(grid.argmax_raster(
        min_votes, occ_prob=prob if lidar_gate else None))
    floor_m = raster == FLOOR_GID
    wall_m = np.isin(raster, WALL_LIKE_GIDS)
    metrics = dict(
        keyframes=len(traj), seg_frames=len(seg_idx),
        seg_ms_mean=round(float(np.mean(seg_ms)), 1) if seg_ms else None,
        tilt_deg=round(tilt_deg, 2),
        z_floor=None if z_floor is None else round(z_floor, 3),
        lidar_gate=bool(lidar_gate),
        floor_cells=int(floor_m.sum()),
        wall_like_cells=int(wall_m.sum()),
        floor_free_agreement=round(float((prob[floor_m] < 0.35).mean()), 3)
        if floor_m.any() else None,
        wall_near_occupied=round(float(
            (cv2.distanceTransform((~occupied_mask(prob)).astype(np.uint8),
                                   cv2.DIST_L2, 3)[wall_m] * grid.res <= 0.3
             ).mean()), 3) if wall_m.any() else None)
    print("[semantic] sanity:", json.dumps(metrics))

    grid.save(run_dir / "class_votes.npz")
    render_semantic_png(raster, prob if lidar_gate else None, meta,
                        run_dir / f"semantic{suffix}.png",
                        f"Semantic map — {run_dir.name} "
                        f"({len(traj)} kf, {len(seg_idx)} frames"
                        f"{'' if lidar_gate else ', no LiDAR gating'})",
                        traj_xyt=traj[:, 1:4])
    json.dump(dict(meta, groups=GROUPS,
                   palette_bgr={k: list(v) for k, v in PALETTE_BGR.items()},
                   params=dict(seg_interval=seg_interval, stride=stride,
                               min_votes=min_votes, max_depth=max_depth,
                               obstacle_max_depth=obstacle_max_depth,
                               speed=speed, fps=fps, model=str(model_dir),
                               lidar_gate=lidar_gate),
                   metrics=metrics),
              open(run_dir / f"semantic_meta{suffix}.json", "w"), indent=2)
    print(f"[semantic] wrote semantic{suffix}.png class_votes.npz "
          f"semantic_meta{suffix}.json in {run_dir}")
    return metrics
