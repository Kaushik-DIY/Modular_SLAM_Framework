"""Fusion v2 unified runner.

All four modes insert every synced keyframe into ONE shared C++ map
(fusion_core: Signature store + STM/WM/LTM MemoryManager + FusionGraph2D);
modes differ only in which payloads drive tracking and loop verification.

This file is deliberately thin: dataset feeding, keyframe normalization, loop
proposing, accept/reject bookkeeping, and outputs. Everything hot runs in
fusion_core with the GIL released.

Usage:
  .venv/bin/python -m slam_core.fusion2.runner --mode lidar --dataset datasets/lab_hybrid
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

import fusion_core as fc

from slam_core.common.types import Pose2 as PyPose2
from slam_core.fusion2.config import FusionV2Config
from slam_core.fusion2.dataset import LabHybridStream


# ---------------------------------------------------------------------------
# SE(2) helpers on fusion_core.Pose2
# ---------------------------------------------------------------------------

def _rel(a: fc.Pose2, b: fc.Pose2) -> fc.Pose2:
    """T_a^{-1} ∘ T_b."""
    return a.inverse().compose(b)


def _fc_pose(p) -> fc.Pose2:
    return fc.Pose2(float(p.x), float(p.y), float(p.theta))


def _rel_sane(rel: fc.Pose2, pred: fc.Pose2, max_m: float, max_rad: float) -> bool:
    """Loop-edge sanity: verified rel pose must roughly agree with the graph
    prediction. Rejects rotational-ambiguity false positives that score well
    on the matcher but disagree wildly with the (drift-bounded) odometry."""
    return (math.hypot(rel.x - pred.x, rel.y - pred.y) <= max_m
            and abs(math.atan2(math.sin(rel.theta - pred.theta),
                               math.cos(rel.theta - pred.theta))) <= max_rad)


# ---------------------------------------------------------------------------
# Shared map assembly
# ---------------------------------------------------------------------------

@dataclass
class SharedMap:
    memory: "fc.MemoryManager"
    store: "fc.InRamLtmStore"
    graph: "fc.FusionGraph2D"
    grid_cfg: "fc.GridConfig"
    bnb_cfg: "fc.BnbConfig"


def build_shared_map(cfg: FusionV2Config) -> SharedMap:
    mc = fc.MemoryConfig()
    mc.stm_size = cfg.stm_size
    mc.wm_cap = cfg.wm_cap
    mc.rehearsal_similarity = cfg.rehearsal_similarity
    store = fc.InRamLtmStore()
    memory = fc.MemoryManager(mc, store)

    gc = fc.GraphConfig()
    gc.huber_scale = cfg.huber_scale
    gc.spine_trans_weight = cfg.spine_trans_weight
    gc.spine_rot_weight = cfg.spine_rot_weight
    graph = fc.FusionGraph2D(gc)

    grid_cfg = fc.GridConfig()
    grid_cfg.resolution = cfg.grid_resolution
    grid_cfg.l_occ = cfg.grid_l_occ
    grid_cfg.l_free = cfg.grid_l_free

    bnb = fc.BnbConfig()
    bnb.linear_search_window = cfg.bnb_window_xy
    bnb.angular_search_window = cfg.bnb_window_th
    bnb.depth = cfg.bnb_depth
    return SharedMap(memory, store, graph, grid_cfg, bnb)


# ---------------------------------------------------------------------------
# Proximity loop proposer + candidate-local B&B verification (LiDAR side)
# ---------------------------------------------------------------------------

def propose_candidates(shared: SharedMap, cfg: FusionV2Config, query_id: int,
                       query_pose: fc.Pose2) -> List[int]:
    """Bounded proximity proposer over WM only (STM hidden per RTAB)."""
    cands = []
    for cid in shared.memory.wm_ids():
        if abs(query_id - cid) < cfg.min_kf_separation:
            continue
        if not shared.graph.has_node(cid):
            continue
        p = shared.graph.get_pose(cid)
        if math.hypot(p.x - query_pose.x, p.y - query_pose.y) <= cfg.proposal_radius_m:
            cands.append((math.hypot(p.x - query_pose.x, p.y - query_pose.y), cid))
    cands.sort()
    return [cid for _, cid in cands[: cfg.max_candidates_per_query]]


def _candidate_neighborhood(shared: SharedMap, cfg: FusionV2Config, query_id: int,
                            cand_id: int):
    """Scans grouped around the candidate, query's temporal trail excluded."""
    nb = fc.retrieve_neighborhood(shared.memory, shared.graph, cand_id,
                                  graph_depth=cfg.retrieval_graph_depth,
                                  metric_radius=cfg.retrieval_metric_radius,
                                  scans_only=True)
    sigs, poses = [], []
    for s, p in zip(nb.signatures, nb.poses):
        if abs(s.id - query_id) < cfg.min_kf_separation:
            continue  # never let the query (or its recent trail) verify itself
        sigs.append(s)
        poses.append(p)
    return sigs, poses


def verify_candidate_bnb(shared: SharedMap, cfg: FusionV2Config, query_id: int,
                         query_scan: np.ndarray, query_pose: fc.Pose2,
                         cand_id: int):
    """Candidate-local B&B: grid from scans grouped around the candidate
    (query's own temporal neighborhood excluded), then bounded search."""
    sigs, poses = _candidate_neighborhood(shared, cfg, query_id, cand_id)
    if len(sigs) < 2:
        return None
    grid = fc.assemble_local_grid(sigs, poses, shared.grid_cfg)
    return fc.bnb_match(grid, query_scan, query_pose, shared.bnb_cfg)


def verify_candidate_icp(shared: SharedMap, cfg: FusionV2Config, query_id: int,
                         query_scan: np.ndarray, query_pose: fc.Pose2,
                         cand_id: int):
    """Standalone ICP loop verifier (small_gicp), RTAB-Map RegistrationIcp style
    and INDEPENDENT of B&B. Align the query scan against the candidate-local
    neighbourhood cloud, seeded ONLY by the graph-predicted relative pose (the
    "guess from odometry" RTAB uses), and accept on the ICP's OWN quality
    metrics: the correspondence ratio (fraction of query points with a target
    neighbour within icp_fitness_dist) and the inlier RMSE. Returns a
    BnbResult-shaped object with coarse_score=correspondence_ratio,
    refined_score=inlier_rmse (metres, LOWER is better).

    No B&B coarse seed and no occupancy-grid cross-check. The "result must stay
    near the guess" slide-lock guard is RTAB's max-translation/rotation bound,
    applied by the caller as the rel-sanity gate. Consequence by design: ICP
    needs a good initial guess, so high-drift front-ends (e.g. visual-led
    orb_lidar, where VO drift exceeds GICP's convergence basin) will verify
    fewer/no loops with ICP than with B&B — that is a property of metric ICP to
    be studied comparatively, not a fault to be patched with a B&B seed."""
    import small_gicp

    sigs, poses = _candidate_neighborhood(shared, cfg, query_id, cand_id)
    if len(sigs) < 2:
        return None

    tgt_pts = []
    for s, p in zip(sigs, poses):
        sc = np.asarray(s.scan_xy, dtype=np.float64)
        c, sn = math.cos(p.theta), math.sin(p.theta)
        tgt_pts.append(sc @ np.array([[c, sn], [-sn, c]]) + [p.x, p.y])
    tgt3 = np.c_[np.vstack(tgt_pts), np.zeros(sum(len(t) for t in tgt_pts))]

    # seed = graph-predicted pose (RTAB's odometry guess); NO coarse pre-search.
    seed_pose = query_pose
    q = np.asarray(query_scan, dtype=np.float64)
    c, sn = math.cos(seed_pose.theta), math.sin(seed_pose.theta)
    q_world = q @ np.array([[c, sn], [-sn, c]]) + [seed_pose.x, seed_pose.y]
    src3 = np.c_[q_world, np.zeros(len(q_world))]

    res = small_gicp.align(tgt3, src3, registration_type="GICP",
                           max_correspondence_distance=cfg.icp_max_corr_dist,
                           num_threads=2)
    T = np.asarray(res.T_target_source)
    dx, dy = float(T[0, 3]), float(T[1, 3])
    dth = float(math.atan2(T[1, 0], T[0, 0]))

    # ICP-native quality: correspondence ratio + inlier RMSE over the aligned
    # query (RTAB's correspondencesRatio and inlier residual).
    corr = (src3 @ T[:3, :3].T) + T[:3, 3]
    tree = small_gicp.KdTree(tgt3)
    d = np.sqrt(np.array([tree.nearest_neighbor_search(p)[2] for p in corr]))
    inl = d < cfg.icp_fitness_dist
    corr_ratio = float(inl.mean())
    inlier_rmse = float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else cfg.icp_max_corr_dist

    cx = T[0, 0] * seed_pose.x + T[0, 1] * seed_pose.y + dx
    cy = T[1, 0] * seed_pose.x + T[1, 1] * seed_pose.y + dy
    corrected_pose = fc.Pose2(cx, cy, seed_pose.theta + dth)

    class _R:  # BnbResult-compatible shape
        success = bool(res.converged)
        coarse_score = corr_ratio       # gated by icp_accept_fitness (>=)
        refined_score = inlier_rmse     # gated by icp_accept_rmse (<=), metres
        pose = corrected_pose
        refined = True
    return _R()


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def _anchor_poses(poses: np.ndarray) -> np.ndarray:
    """Express every node pose in the robot-START frame so EVERY mode renders
    in the same standard convention: the first keyframe sits at the origin
    facing +x. Without this the VO front-end starts at heading 90° (the
    BASE_T_CAM ∘ CAMERA_GROUND_TRANSFORM projection of identity), rotating the
    orb maps 90° vs the LiDAR maps. A pure rigid re-frame — map/trajectory
    geometry is unchanged, only the global orientation is normalized.
    `poses` columns: [nid, x, y, theta], assumed nid-sorted (gauge = row 0)."""
    if len(poses) == 0:
        return poses
    ax, ay, ath = float(poses[0, 1]), float(poses[0, 2]), float(poses[0, 3])
    c, s = math.cos(ath), math.sin(ath)
    out = poses.copy()
    dx = poses[:, 1] - ax
    dy = poses[:, 2] - ay
    out[:, 1] = c * dx + s * dy          # R(-ath) @ (p - anchor)
    out[:, 2] = -s * dx + c * dy
    out[:, 3] = np.arctan2(np.sin(poses[:, 3] - ath), np.cos(poses[:, 3] - ath))
    return out


def render_fused_occupancy(render_sigs, render_poses, traj_xyt, grid_cfg, out_png,
                           title, npy_path=None, meta_path=None):
    """Fuse the given signatures' scans (at their optimized poses) into ONE
    log-odds occupancy grid (C++ assemble_local_grid) and render it grayscale
    with the trajectory overlay. Shared by the batch runner (write_outputs) and
    the real-time runner (V5) so the map convention is identical. `traj_xyt` is
    an (N,3) x/y/theta array for the blue path + start/end markers. Returns
    (prob, extent) or (None, None) if there is nothing with scans to render."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not render_sigs:
        return None, None
    grid = fc.assemble_local_grid(render_sigs, render_poses, grid_cfg)
    prob = np.asarray(grid.probability())
    extent = [grid.origin_x, grid.origin_x + grid.width * grid.resolution,
              grid.origin_y, grid.origin_y + grid.height * grid.resolution]
    if npy_path is not None:
        np.save(npy_path, prob)
    if meta_path is not None:
        with open(meta_path, "w") as f:
            json.dump(dict(origin_x=grid.origin_x, origin_y=grid.origin_y,
                           resolution=grid.resolution, width=grid.width,
                           height=grid.height, extent=extent), f, indent=2)
    traj = np.asarray(traj_xyt, dtype=float)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(prob, cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower",
              extent=extent, interpolation="nearest")
    if len(traj):
        ax.plot(traj[:, 0], traj[:, 1], "-", lw=1.0, color="tab:blue", alpha=0.9)
        ax.scatter(traj[0, 0], traj[0, 1], c="g", s=50, zorder=5, label="start")
        ax.scatter(traj[-1, 0], traj[-1, 1], c="r", s=50, zorder=5, label="end")
    ax.set_title(title)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.grid(alpha=0.15); ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return prob, extent


def write_outputs(shared: SharedMap, cfg: FusionV2Config, run_dir: Path,
                  kf_stamps: dict, stats: dict,
                  skip_scan_ids: Optional[set] = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    poses = _anchor_poses(np.asarray(shared.graph.poses()))

    with open(run_dir / "trajectory.tum", "w") as f:
        for nid, x, y, th in poses:
            t = kf_stamps.get(int(nid), float(nid))
            qz, qw = math.sin(th / 2.0), math.cos(th / 2.0)
            f.write(f"{t:.6f} {x:.6f} {y:.6f} 0.0 0.0 0.0 {qz:.9f} {qw:.9f}\n")

    # Collect renderable signatures (scans at optimized poses).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    render_sigs, render_poses, pts_all = [], [], []
    for nid, x, y, th in poses:
        if skip_scan_ids and int(nid) in skip_scan_ids:
            continue   # blind-pose keyframes (REINIT) must not paint the map
        sig = shared.memory.get(int(nid))
        if sig is None or not sig.has_scan:
            continue
        render_sigs.append(sig)
        render_poses.append(fc.Pose2(float(x), float(y), float(th)))
        sc = np.asarray(sig.scan_xy, dtype=np.float64)
        c, s = math.cos(th), math.sin(th)
        pts_all.append(sc @ np.array([[c, -s], [s, c]]).T + [x, y])

    title = (f"fusion2 --mode {cfg.mode}: {len(poses)} keyframes, "
             f"{stats.get('loops_accepted', 0)} loops "
             f"(STM {shared.memory.stm_count()} / WM {shared.memory.wm_count()} / "
             f"LTM {shared.memory.ltm_count()})")

    # Primary output: FUSED log-odds occupancy grid (V4.2, thesis-grade).
    # Reuses the same C++ integration the B&B verifier trusts; overlapping
    # observations reinforce walls instead of smearing as a scatter band.
    render_fused_occupancy(
        render_sigs, render_poses, poses[:, 1:4] if len(poses) else np.zeros((0, 3)),
        shared.grid_cfg, run_dir / "occupancy.png", title,
        npy_path=run_dir / "map.npy", meta_path=run_dir / "map_meta.json")

    # Secondary debug output: raw scan scatter (the pre-V4.2 rendering).
    fig, ax = plt.subplots(figsize=(12, 7))
    if pts_all:
        P = np.vstack(pts_all)
        ax.scatter(P[:, 0], P[:, 1], s=0.2, c="k", alpha=0.25, linewidths=0)
    ax.plot(poses[:, 1], poses[:, 2], "-", lw=0.8, color="tab:blue", alpha=0.9)
    ax.scatter(poses[0, 1], poses[0, 2], c="g", s=50, zorder=5, label="start")
    ax.scatter(poses[-1, 1], poses[-1, 2], c="r", s=50, zorder=5, label="end")
    ax.set_title(title)
    ax.axis("equal"); ax.grid(alpha=0.2); ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "scan_overlay.png", dpi=110)
    plt.close(fig)

    with open(run_dir / "run_summary.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)


def _rss_gb() -> float:
    import os
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024.0 / 1024.0
    return -1.0


# ---------------------------------------------------------------------------
# Mode: lidar
# ---------------------------------------------------------------------------

def run_lidar_mode(cfg: FusionV2Config) -> dict:
    """Modes `lidar` (B&B/ICP scan verification) and `lidar_orb` (visual PnP
    verification of proximity proposals)."""
    import cv2

    from slam_core.fusion2.lidar_frontend import make_lidar_frontend
    from slam_core.fusion2.visual_features import (cam_rel_to_base_se2,
                                                   extract_orb_rgbd, pnp_verify)

    visual_backend = cfg.mode == "lidar_orb"
    K = None
    if visual_backend:
        import yaml
        sc = yaml.safe_load(open(Path(cfg.dataset) / "sensor_config.yaml"))["camera"]
        K = np.array([[sc["fx"], 0, sc["cx"]], [0, sc["fy"], sc["cy"]], [0, 0, 1]])

    stream = LabHybridStream(cfg.dataset, cfg.sync_tolerance_s)
    fe = make_lidar_frontend(
        cfg.lidar_frontend,
        dataset_name="lab_hybrid",
        imu_path=str(Path(cfg.dataset) / "imu.csv") if cfg.lidar_use_imu else None,
        kf_min_dist_m=cfg.kf_min_dist_m,
        kf_min_angle_rad=cfg.kf_min_angle_rad,
        kf_min_dt_s=cfg.kf_min_dt_s,
    )
    shared = build_shared_map(cfg)

    kf_id = -1
    kf_stamps: dict = {}
    last_fe_pose: Optional[fc.Pose2] = None     # front-end odometry frame
    last_graph_pose: Optional[fc.Pose2] = None  # optimized graph frame
    stats = dict(mode=cfg.mode, frontend=cfg.lidar_frontend,
                 verifier=("pnp" if cfg.mode == "lidar_orb" else cfg.scan_verifier),
                 imu_active=cfg.lidar_use_imu,
                 scans=0, keyframes=0, fallbacks=0, proposals=0, verified=0,
                 loops_accepted=0, rehearsal_merges=0, optimize_calls=0)
    t_start = time.perf_counter()
    verify_ms: List[float] = []
    verify_log: List[dict] = []  # per-verification diagnostics -> verifications.csv

    for t, scan in stream.lidar_stream(cfg.max_scans):
        stats["scans"] += 1
        fe_pose_py, pts, is_kf = fe.process(t, scan)
        if not is_kf:
            continue
        fe_pose = _fc_pose(fe_pose_py)
        # Signatures carry the RAW scan (plan §8): ~560 valid beams give
        # contiguous walls in candidate-local grids; the voxel-filtered `pts`
        # stay the front-end's matching diet. B&B subsamples the query itself.
        raw_scan = scan

        kf_id += 1
        kf_stamps[kf_id] = t
        # Initialize the node in the OPTIMIZED frame: continue from the last
        # graph pose using the front-end's relative motion (so global loop
        # corrections never fight new odometry).
        if last_fe_pose is None:
            node_pose = fe_pose
        else:
            node_pose = last_graph_pose.compose(_rel(last_fe_pose, fe_pose))

        if visual_backend:
            # lidar_orb: signatures additionally carry the synced visual payload
            # so loop candidates can be verified by ORB matching + PnP.
            kpts = des = pts3d = None
            pair = stream.nearest_rgbd(t)
            if pair is not None:
                rgb = cv2.imread(str(pair[0]), cv2.IMREAD_COLOR)
                depth = cv2.imread(str(pair[1]), cv2.IMREAD_UNCHANGED)
                if rgb is not None and depth is not None:
                    kpts, des, pts3d = extract_orb_rgbd(rgb, depth, K)
            if kpts is not None:
                sig = fc.Signature(kf_id, t, kpts=kpts, des=des,
                                   pts3d=pts3d,
                                   scan_xy=raw_scan)
            else:
                sig = fc.Signature(kf_id, t, scan_xy=raw_scan)
        else:
            sig = fc.Signature(kf_id, t, scan_xy=raw_scan)
        sig.pose = node_pose
        # Rehearsal gate: keyframes here are ALREADY motion-filtered, so only a
        # stationary (time-triggered) keyframe may rehearsal-merge with its
        # predecessor (RTAB collapses repeated observations of the SAME spot;
        # consecutive moving scans in a room overlap ~85% and must NOT merge).
        if last_fe_pose is not None:
            d = _rel(last_fe_pose, fe_pose)
            stationary = math.hypot(d.x, d.y) < 0.05 and abs(d.theta) < math.radians(2.0)
        else:
            stationary = False
        res = shared.memory.insert(sig, similarity=-1.0 if stationary else 0.0)
        if res.rehearsal_merged:
            stats["rehearsal_merges"] += 1
            # The merged predecessor's graph node remains (full trajectory kept);
            # only its payload is deduplicated out of memory.
        shared.graph.add_node(kf_id, node_pose)
        if last_fe_pose is not None:
            sig.add_link(kf_id - 1, fc.LinkType.NEIGHBOR, _rel(last_fe_pose, fe_pose),
                         cfg.spine_trans_weight, cfg.spine_rot_weight)
            shared.graph.add_spine_edge(kf_id - 1, kf_id, _rel(last_fe_pose, fe_pose))
        last_fe_pose = fe_pose
        last_graph_pose = node_pose
        stats["keyframes"] += 1

        # ---- loop closure: propose (bounded proximity) -> verify per mode:
        #   lidar:     candidate-local B&B (or --verifier icp) on scans
        #   lidar_orb: ORB descriptor match + PnP RANSAC on visual payloads
        if kf_id % cfg.propose_every_n_kf == 0 and kf_id > cfg.min_kf_separation:
            for cand in propose_candidates(shared, cfg, kf_id, node_pose):
                stats["proposals"] += 1
                t0 = time.perf_counter()
                accepted = False
                rel = None
                row = dict(query=kf_id, cand=cand, rx=None, ry=None, rth=None)
                if visual_backend:
                    cand_sig = shared.memory.get(cand)
                    if (cand_sig is None or not cand_sig.has_visual
                            or not sig.has_visual):
                        verify_ms.append((time.perf_counter() - t0) * 1000)
                        continue
                    stats["verified"] += 1
                    ok, T_tq, n_in, ratio = pnp_verify(
                        np.asarray(sig.kpts), np.asarray(sig.des),
                        np.asarray(cand_sig.des), np.asarray(cand_sig.pts3d), K,
                        nndr=cfg.pnp_nndr, min_inliers=cfg.pnp_min_inliers)
                    row.update(coarse=n_in, refined=round(ratio, 3))
                    if ok:
                        rx, ry, rth = cam_rel_to_base_se2(T_tq)
                        rel = fc.Pose2(rx, ry, rth)
                        # sanity vs predicted relative pose (drift-bounded gate)
                        pred = _rel(shared.graph.get_pose(cand), node_pose)
                        if (math.hypot(rel.x - pred.x, rel.y - pred.y)
                                <= cfg.pnp_rel_sanity_m
                                and abs(math.atan2(math.sin(rel.theta - pred.theta),
                                                   math.cos(rel.theta - pred.theta)))
                                <= cfg.pnp_rel_sanity_rad):
                            accepted = True
                    row.update(dx=None, dy=None, dth=None)
                else:
                    verify_fn = (verify_candidate_icp if cfg.scan_verifier == "icp"
                                 else verify_candidate_bnb)
                    r = verify_fn(shared, cfg, kf_id, raw_scan, node_pose, cand)
                    if r is None:
                        verify_ms.append((time.perf_counter() - t0) * 1000)
                        continue
                    stats["verified"] += 1
                    if cfg.scan_verifier == "icp":
                        # RTAB-style: correspondence ratio + inlier RMSE (metres,
                        # lower=better). Standalone, no B&B grid cross-check.
                        accepted = (r.success
                                    and r.coarse_score >= cfg.icp_accept_fitness
                                    and r.refined_score <= cfg.icp_accept_rmse)
                    else:
                        accepted = (r.success
                                    and r.coarse_score >= cfg.accept_coarse_min
                                    and r.refined_score >= cfg.accept_refined_min)
                    row.update(coarse=round(r.coarse_score, 4),
                               refined=round(r.refined_score, 4),
                               dx=round(r.pose.x - node_pose.x, 3),
                               dy=round(r.pose.y - node_pose.y, 3),
                               dth=round(r.pose.theta - node_pose.theta, 4))
                    if accepted:
                        # V4.1: rel-sanity gate (see run_orb_mode_native note)
                        cand_pose = shared.graph.get_pose(cand)
                        rel = _rel(cand_pose, r.pose)
                        pred = _rel(cand_pose, node_pose)
                        if not _rel_sane(rel, pred, cfg.scan_rel_sanity_m,
                                         cfg.scan_rel_sanity_rad):
                            accepted = False
                            rel = None
                verify_ms.append((time.perf_counter() - t0) * 1000)
                row["accepted"] = accepted
                verify_log.append(row)
                if accepted and rel is not None:
                    # measured loop transform (cand->query), for offline true/false
                    # loop analysis (tools/analyze_fusion_loops.py)
                    row.update(rx=round(rel.x, 4), ry=round(rel.y, 4),
                               rth=round(rel.theta, 5))
                    shared.graph.add_loop_edge(cand, kf_id, rel,
                                               cfg.loop_trans_weight,
                                               cfg.loop_rot_weight)
                    sig.add_link(cand, fc.LinkType.LOOP, rel,
                                 cfg.loop_trans_weight, cfg.loop_rot_weight)
                    shared.memory.on_loop_confirmed(kf_id, cand)
                    stats["loops_accepted"] += 1

        if kf_id > 0 and kf_id % cfg.optimize_every_n_kf == 0:
            shared.graph.optimize()
            stats["optimize_calls"] += 1
            last_graph_pose = shared.graph.get_pose(kf_id)

        if stats["scans"] % cfg.print_every == 0:
            print(f"[{stats['scans']:5d}] kf={stats['keyframes']} "
                  f"loops={stats['loops_accepted']} "
                  f"tiers S/W/L={shared.memory.stm_count()}/"
                  f"{shared.memory.wm_count()}/{shared.memory.ltm_count()} "
                  f"rss={_rss_gb():.2f}GB", flush=True)

    shared.graph.optimize()
    stats["optimize_calls"] += 1
    stats["fallbacks"] = int(getattr(fe, "fallback_count", 0))
    stats["elapsed_sec"] = round(time.perf_counter() - t_start, 1)
    stats["peak_rss_gb"] = round(_rss_gb(), 2)
    stats["verify_ms_mean"] = round(float(np.mean(verify_ms)), 1) if verify_ms else None
    stats["verify_ms_p95"] = round(float(np.percentile(verify_ms, 95)), 1) if verify_ms else None
    stats["stm"] = shared.memory.stm_count()
    stats["wm"] = shared.memory.wm_count()
    stats["ltm"] = shared.memory.ltm_count()
    stats["map_payload_mb"] = round(shared.memory.payload_bytes() / 1e6, 1)

    run_dir = Path(cfg.output_dir) / f"{cfg.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    write_outputs(shared, cfg, run_dir, kf_stamps, stats)
    if verify_log:
        import csv as _csv
        with open(run_dir / "verifications.csv", "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(verify_log[0].keys()))
            w.writeheader()
            w.writerows(verify_log)
    stats["run_dir"] = str(run_dir)
    return stats


# ---------------------------------------------------------------------------
# Mode: orb (shared map; DBoW propose logged — visual verification lands in C7)
# ---------------------------------------------------------------------------

def run_orb_mode(cfg: FusionV2Config) -> dict:
    import cv2

    from slam_core.fusion.signature import (CAMERA_GROUND_TRANSFORM,
                                            project_pose3d_to_pose2)
    from slam_core.fusion.orbslam_frontend import OrbSlamFrontendBackend
    from visual_slam.orbslam.io.rgbd_dataset import make_rgbd_camera

    from slam_core.fusion.adapters.orb_loop_proposer import OrbLoopProposer
    from slam_core.fusion2.visual_features import cam_rel_to_base_se2, pnp_verify

    stream = LabHybridStream(cfg.dataset, cfg.sync_tolerance_s)
    camera = make_rgbd_camera(cfg.dataset)
    fe = OrbSlamFrontendBackend(camera)
    shared = build_shared_map(cfg)
    # DBoW appearance proposer — propose-only (v1 adapter; no ORB source edits).
    proposer = OrbLoopProposer(fe.make_loop_detector(),
                               min_index_separation=cfg.min_kf_separation)
    K = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]],
                 dtype=np.float64)
    scan_backend = cfg.mode == "orb_lidar"

    kf_id = -1
    kf_stamps: dict = {}
    orb_to_fusion: dict = {}
    last_pose: Optional[fc.Pose2] = None
    stats = dict(mode=cfg.mode, frames=0, keyframes=0, proposals=0, verified=0,
                 loops_accepted=0, rehearsal_merges=0, optimize_calls=0)
    t_start = time.perf_counter()
    verify_ms: List[float] = []
    verify_log: List[dict] = []

    for t, rgb_p, depth_p, scan in stream.rgbd_stream(cfg.max_scans):
        stats["frames"] += 1
        rgb = cv2.imread(str(rgb_p), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_p), cv2.IMREAD_UNCHANGED)
        okf = fe.track(rgb, depth, t)
        if okf is None:
            continue

        # REP-103 node frame (base_T_cam) so PnP loop-edge translations live in
        # the same body frame as the spine — see the note in run_orb_mode_native.
        from slam_core.fusion2.visual_features import BASE_T_CAM as _B
        p2 = project_pose3d_to_pose2(okf.pose, base_T_cam=_B,
                                     world_transform=CAMERA_GROUND_TRANSFORM)
        node_pose = fc.Pose2(p2.x, p2.y, p2.theta)
        kf_id += 1
        kf_stamps[kf_id] = t
        # Key by the ORB keyframe 'kid' — that is what the DBoW detector's
        # candidate_idxs contain (falls back to frame id).
        orb_to_fusion[int(getattr(okf.source, 'kid', okf.id))] = kf_id

        kpts = np.ascontiguousarray(okf.keypoints, dtype=np.float32)
        des = okf.descriptors if okf.descriptors is not None else None
        # Per-keypoint camera-frame 3D from the ORB KF's depths (for PnP):
        pts3d = np.zeros((0, 3), np.float32)
        kf_depths = np.asarray(getattr(okf.source, "depths", np.zeros(0)), np.float32)
        if des is not None and len(kf_depths) == len(kpts):
            z = kf_depths
            pts3d = np.full((len(kpts), 3), np.nan, np.float32)
            ok_z = z > 0.05
            pts3d[ok_z, 0] = (kpts[ok_z, 0] - camera.cx) / camera.fx * z[ok_z]
            pts3d[ok_z, 1] = (kpts[ok_z, 1] - camera.cy) / camera.fy * z[ok_z]
            pts3d[ok_z, 2] = z[ok_z]
        sig = fc.Signature(kf_id, t, kpts=kpts, des=des, pts3d=pts3d,
                           scan_xy=scan if scan is not None else np.zeros((0, 2), np.float32))
        sig.pose = node_pose
        res = shared.memory.insert(sig, similarity=0.0)
        if res.rehearsal_merged:
            stats["rehearsal_merges"] += 1
        shared.graph.add_node(kf_id, node_pose)
        if last_pose is not None:
            rel = _rel(last_pose, node_pose)
            sig.add_link(kf_id - 1, fc.LinkType.NEIGHBOR, rel,
                         cfg.spine_trans_weight, cfg.spine_rot_weight)
            shared.graph.add_spine_edge(kf_id - 1, kf_id, rel)
        last_pose = node_pose
        stats["keyframes"] += 1

        # ---- loop closure: DBoW propose (appearance) -> verify per mode:
        #   orb:       ORB descriptor match + PnP RANSAC (visual/visual)
        #   orb_lidar: candidate-local B&B (or ICP) on the synced scans
        proposer.register(okf.source)
        for prop in proposer.poll_candidates(okf.source):
            cand = orb_to_fusion.get(int(prop.candidate_id), None)
            if cand is None or abs(kf_id - cand) < cfg.min_kf_separation:
                continue
            stats["proposals"] += 1
            t0 = time.perf_counter()
            accepted = False
            rel_lp = None
            row = dict(query=kf_id, cand=cand, rx=None, ry=None, rth=None)
            if scan_backend:
                if sig.has_scan:
                    r = (verify_candidate_icp if cfg.scan_verifier == "icp"
                         else verify_candidate_bnb)(
                             shared, cfg, kf_id,
                             np.asarray(sig.scan_xy), node_pose, cand)
                    if r is not None:
                        stats["verified"] += 1
                        if cfg.scan_verifier == "icp":
                            accepted = (r.success
                                        and r.coarse_score >= cfg.icp_accept_fitness
                                        and r.refined_score >= cfg.accept_refined_min)
                        else:
                            accepted = (r.success
                                        and r.coarse_score >= cfg.accept_coarse_min
                                        and r.refined_score >= cfg.accept_refined_min)
                        row.update(coarse=round(r.coarse_score, 4),
                                   refined=round(r.refined_score, 4))
                        if accepted:
                            # V4.1: scan verifiers need the same rel-sanity gate
                            # as PnP — high-scoring wrong-rotation alignments
                            # otherwise warp the graph at full loop weight.
                            cand_pose = shared.graph.get_pose(cand)
                            rel_lp = _rel(cand_pose, r.pose)
                            pred = _rel(cand_pose, node_pose)
                            if not _rel_sane(rel_lp, pred, cfg.scan_rel_sanity_m,
                                             cfg.scan_rel_sanity_rad):
                                accepted = False
                                rel_lp = None
            else:
                cand_sig = shared.memory.get(cand)
                if cand_sig is not None and cand_sig.has_visual and sig.has_visual:
                    stats["verified"] += 1
                    okp, T_tq, n_in, ratio = pnp_verify(
                        np.asarray(sig.kpts), np.asarray(sig.des),
                        np.asarray(cand_sig.des), np.asarray(cand_sig.pts3d), K,
                        nndr=cfg.pnp_nndr, min_inliers=cfg.pnp_min_inliers)
                    row.update(coarse=n_in, refined=round(ratio, 3))
                    if okp:
                        rx, ry, rth = cam_rel_to_base_se2(T_tq)
                        rel_lp = fc.Pose2(rx, ry, rth)
                        pred = _rel(shared.graph.get_pose(cand), node_pose)
                        sane = (math.hypot(rel_lp.x - pred.x, rel_lp.y - pred.y)
                                <= cfg.pnp_rel_sanity_m
                                and abs(math.atan2(
                                    math.sin(rel_lp.theta - pred.theta),
                                    math.cos(rel_lp.theta - pred.theta)))
                                <= cfg.pnp_rel_sanity_rad)
                        # strong PnP is self-validating (bootstrap loops happen
                        # exactly when drift breaks the sanity prediction)
                        accepted = sane or n_in >= cfg.pnp_strong_inliers
            verify_ms.append((time.perf_counter() - t0) * 1000)
            row["accepted"] = accepted
            verify_log.append(row)
            if accepted and rel_lp is not None:
                shared.graph.add_loop_edge(cand, kf_id, rel_lp,
                                           cfg.loop_trans_weight,
                                           cfg.loop_rot_weight)
                sig.add_link(cand, fc.LinkType.LOOP, rel_lp,
                             cfg.loop_trans_weight, cfg.loop_rot_weight)
                row.update(rx=round(rel_lp.x, 4), ry=round(rel_lp.y, 4),
                           rth=round(rel_lp.theta, 5))
                shared.memory.on_loop_confirmed(kf_id, cand)
                stats["loops_accepted"] += 1

        if kf_id > 0 and kf_id % cfg.optimize_every_n_kf == 0:
            shared.graph.optimize()
            stats["optimize_calls"] += 1

        if stats["frames"] % cfg.print_every == 0:
            print(f"[{stats['frames']:5d}] kf={stats['keyframes']} "
                  f"tiers S/W/L={shared.memory.stm_count()}/"
                  f"{shared.memory.wm_count()}/{shared.memory.ltm_count()} "
                  f"rss={_rss_gb():.2f}GB", flush=True)

    shared.graph.optimize()
    stats["optimize_calls"] += 1
    stats["elapsed_sec"] = round(time.perf_counter() - t_start, 1)
    stats["peak_rss_gb"] = round(_rss_gb(), 2)
    stats["stm"] = shared.memory.stm_count()
    stats["wm"] = shared.memory.wm_count()
    stats["ltm"] = shared.memory.ltm_count()
    stats["map_payload_mb"] = round(shared.memory.payload_bytes() / 1e6, 1)
    stats["verify_ms_mean"] = round(float(np.mean(verify_ms)), 1) if verify_ms else None

    run_dir = Path(cfg.output_dir) / f"{cfg.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    write_outputs(shared, cfg, run_dir, kf_stamps, stats)
    if verify_log:
        import csv as _csv
        with open(run_dir / "verifications.csv", "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(verify_log[0].keys()))
            w.writeheader()
            w.writerows(verify_log)
    stats["run_dir"] = str(run_dir)
    return stats


# ---------------------------------------------------------------------------
# Modes orb / orb_lidar with the NATIVE windowed C++ VO front-end (fusion v3)
# ---------------------------------------------------------------------------

def run_orb_mode_native(cfg: FusionV2Config) -> dict:
    import cv2

    from slam_core.fusion.signature import (CAMERA_GROUND_TRANSFORM,
                                            project_pose3d_to_pose2)
    from slam_core.fusion2.appearance_index import AppearanceIndex
    from slam_core.fusion2.visual_features import (BASE_T_CAM,
                                                   cam_rel_to_base_se2,
                                                   pnp_verify)
    from slam_core.fusion2.vo_orb_frontend import NativeOrbFrontend

    # SE(2) node poses MUST be REP-103 (x-forward) — the same frame the PnP
    # loop edges (cam_rel_to_base_se2) and LiDAR scans live in. Projecting
    # with world_transform alone leaves the heading on the camera RIGHT axis:
    # loop-edge translations then disagree with spine edges by a 90° body
    # rotation and the optimizer warps the trajectory (the v3.4 orb defect).
    def cam_to_se2(Twc) -> fc.Pose2:
        p = project_pose3d_to_pose2(Twc, base_T_cam=BASE_T_CAM,
                                    world_transform=CAMERA_GROUND_TRANSFORM)
        return fc.Pose2(p.x, p.y, p.theta)

    stream = LabHybridStream(cfg.dataset, cfg.sync_tolerance_s)
    fe = NativeOrbFrontend(cfg.dataset, imu_path=str(Path(cfg.dataset) / "imu.csv"),
                           depth_max=cfg.vo_depth_max,
                           imu_dropout=cfg.vo_imu_dropout,
                           vo_overrides=cfg.vo_overrides)
    shared = build_shared_map(cfg)
    index = AppearanceIndex(min_score=cfg.dbow_min_score,
                            min_separation=cfg.min_kf_separation,
                            max_candidates=cfg.max_candidates_per_query)
    K = fe.K
    scan_backend = cfg.mode == "orb_lidar"
    if scan_backend:
        shared.bnb_cfg.angular_search_window = cfg.orb_bnb_window_th

    kf_id = -1
    kf_stamps: dict = {}
    last_graph_pose: Optional[fc.Pose2] = None
    blind_ids: set = set()
    stats = dict(mode=cfg.mode, frontend="native",
                 verifier=("pnp" if cfg.mode == "orb" else cfg.scan_verifier),
                 imu_active=cfg.vo_imu_dropout, frames=0, keyframes=0,
                 proposals=0, verified=0, loops_accepted=0, rehearsal_merges=0,
                 optimize_calls=0, reinits=0)
    t_start = time.perf_counter()
    track_ms: List[float] = []
    verify_ms: List[float] = []
    verify_log: List[dict] = []

    for t, rgb_p, depth_p, scan in stream.rgbd_stream(cfg.max_scans):
        stats["frames"] += 1
        rgb = cv2.imread(str(rgb_p), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_p), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            continue
        Twc, state, nkf = fe.track(rgb, depth, t)
        track_ms.append(fe.last_track_ms)
        if state == fc.VoState.REINIT:
            stats["reinits"] += 1
        if nkf is None:
            continue

        fe_pose = cam_to_se2(nkf.Twc)
        kf_id += 1
        kf_stamps[kf_id] = t
        # A REINIT keyframe's pose is dead-reckoning, not measurement: its
        # spine edge must be SOFT so accepted loops can bend the blind segment
        # back, and its scan must not paint the occupancy map.
        blind = nkf.state == fc.VoState.REINIT
        if blind:
            blind_ids.add(kf_id)
        # Spine = the local-BA-refined relative motion between consecutive
        # keyframes (both poses from the SAME BA epoch). This is the fix for the
        # smeared map: motion-only odometry was not drift-bounded; the windowed
        # BA now ties consecutive keyframe poses through co-observed points.
        if nkf.prev_Twc is None:
            node_pose = fe_pose
            rel = None
        else:
            rel = _rel(cam_to_se2(nkf.prev_Twc), fe_pose)
            node_pose = last_graph_pose.compose(rel)

        sig = fc.Signature(kf_id, t, kpts=nkf.kpts, des=nkf.des, pts3d=nkf.pts3d_cam,
                           scan_xy=scan if scan is not None else np.zeros((0, 2), np.float32))
        sig.pose = node_pose
        res = shared.memory.insert(sig, similarity=0.0)
        if res.rehearsal_merged:
            stats["rehearsal_merges"] += 1
        shared.graph.add_node(kf_id, node_pose)
        if rel is not None:
            tw = cfg.blind_spine_weight if blind else cfg.spine_trans_weight
            rw = cfg.blind_spine_weight if blind else cfg.spine_rot_weight
            sig.add_link(kf_id - 1, fc.LinkType.NEIGHBOR, rel, tw, rw)
            shared.graph.add_spine_edge(kf_id - 1, kf_id, rel, tw, rw)
        last_graph_pose = node_pose
        stats["keyframes"] += 1

        # appearance propose (query BEFORE adding self) -> verify per mode
        cands = index.query(kf_id, nkf.des)
        index.add(kf_id, nkf.des)
        any_loop_this_kf = False
        for cand, dbow_score in cands:
            stats["proposals"] += 1
            t0 = time.perf_counter()
            accepted = False
            rel_lp = None
            row = dict(query=kf_id, cand=cand, dbow=round(dbow_score, 4),
                       rx=None, ry=None, rth=None)
            if scan_backend:
                if sig.has_scan:
                    # Standalone ICP (RTAB-style): seeded only by the graph
                    # prediction. Visual-led VO drift can exceed GICP's basin, so
                    # orb_lidar+icp may verify fewer loops than +bnb -- an accepted
                    # comparative property, not a defect (see verify_candidate_icp).
                    if cfg.scan_verifier == "icp":
                        r = verify_candidate_icp(shared, cfg, kf_id,
                                                 np.asarray(sig.scan_xy),
                                                 node_pose, cand)
                    else:
                        r = verify_candidate_bnb(shared, cfg, kf_id,
                                                 np.asarray(sig.scan_xy),
                                                 node_pose, cand)
                    if r is not None:
                        stats["verified"] += 1
                        if cfg.scan_verifier == "icp":
                            accepted = (r.success
                                        and r.coarse_score >= cfg.icp_accept_fitness
                                        and r.refined_score <= cfg.icp_accept_rmse)
                        else:
                            accepted = (r.success
                                        and r.coarse_score >= cfg.accept_coarse_min
                                        and r.refined_score >= cfg.accept_refined_min)
                        row.update(coarse=round(r.coarse_score, 4),
                                   refined=round(r.refined_score, 4))
                        if accepted:
                            # V4.1: scan verifiers need the same rel-sanity gate
                            # as PnP — high-scoring wrong-rotation alignments
                            # otherwise warp the graph at full loop weight.
                            cand_pose = shared.graph.get_pose(cand)
                            rel_lp = _rel(cand_pose, r.pose)
                            pred = _rel(cand_pose, node_pose)
                            if not _rel_sane(rel_lp, pred, cfg.scan_rel_sanity_m,
                                             cfg.scan_rel_sanity_rad):
                                accepted = False
                                rel_lp = None
            else:
                cand_sig = shared.memory.get(cand)
                if cand_sig is not None and cand_sig.has_visual and sig.has_visual:
                    stats["verified"] += 1
                    okp, T_tq, n_in, ratio = pnp_verify(
                        np.asarray(sig.kpts), np.asarray(sig.des),
                        np.asarray(cand_sig.des), np.asarray(cand_sig.pts3d), K,
                        nndr=cfg.pnp_nndr, min_inliers=cfg.pnp_min_inliers)
                    row.update(coarse=n_in, refined=round(ratio, 3))
                    if okp:
                        rx, ry, rth = cam_rel_to_base_se2(T_tq)
                        rel_lp = fc.Pose2(rx, ry, rth)
                        pred = _rel(shared.graph.get_pose(cand), node_pose)
                        sane = (math.hypot(rel_lp.x - pred.x, rel_lp.y - pred.y)
                                <= cfg.pnp_rel_sanity_m
                                and abs(math.atan2(
                                    math.sin(rel_lp.theta - pred.theta),
                                    math.cos(rel_lp.theta - pred.theta)))
                                <= cfg.pnp_rel_sanity_rad)
                        # strong PnP is self-validating (bootstrap loops happen
                        # exactly when drift breaks the sanity prediction)
                        accepted = sane or n_in >= cfg.pnp_strong_inliers
            verify_ms.append((time.perf_counter() - t0) * 1000)
            row["accepted"] = accepted
            verify_log.append(row)
            if accepted and rel_lp is not None:
                shared.graph.add_loop_edge(cand, kf_id, rel_lp,
                                           cfg.loop_trans_weight, cfg.loop_rot_weight)
                sig.add_link(cand, fc.LinkType.LOOP, rel_lp,
                             cfg.loop_trans_weight, cfg.loop_rot_weight)
                row.update(rx=round(rel_lp.x, 4), ry=round(rel_lp.y, 4),
                           rth=round(rel_lp.theta, 5))
                shared.memory.on_loop_confirmed(kf_id, cand)
                stats["loops_accepted"] += 1
                any_loop_this_kf = True

        # optimize immediately on accepted loops (snaps blind segments back —
        # effective relocalization), else on the periodic cadence.
        if (any_loop_this_kf and kf_id > 0) or \
                (kf_id > 0 and kf_id % cfg.optimize_every_n_kf == 0):
            shared.graph.optimize()
            stats["optimize_calls"] += 1
            last_graph_pose = shared.graph.get_pose(kf_id)

        if stats["frames"] % cfg.print_every == 0:
            print(f"[{stats['frames']:5d}] kf={stats['keyframes']} "
                  f"loops={stats['loops_accepted']} reinits={stats['reinits']} "
                  f"track={np.mean(track_ms[-200:]):.1f}ms "
                  f"tiers S/W/L={shared.memory.stm_count()}/"
                  f"{shared.memory.wm_count()}/{shared.memory.ltm_count()} "
                  f"rss={_rss_gb():.2f}GB", flush=True)

    shared.graph.optimize()
    stats["optimize_calls"] += 1
    stats["elapsed_sec"] = round(time.perf_counter() - t_start, 1)
    stats["fps"] = round(stats["frames"] / max(stats["elapsed_sec"], 1e-9), 2)
    stats["peak_rss_gb"] = round(_rss_gb(), 2)
    stats["track_ms_mean"] = round(float(np.mean(track_ms)), 1) if track_ms else None
    stats["verify_ms_mean"] = round(float(np.mean(verify_ms)), 1) if verify_ms else None
    stats["stm"] = shared.memory.stm_count()
    stats["wm"] = shared.memory.wm_count()
    stats["ltm"] = shared.memory.ltm_count()
    stats["map_payload_mb"] = round(shared.memory.payload_bytes() / 1e6, 1)
    stats["blind_kfs"] = len(blind_ids)
    stats["imu_calibration"] = fe.imu_calibration

    run_dir = Path(cfg.output_dir) / f"{cfg.mode}_native_{time.strftime('%Y%m%d_%H%M%S')}"
    write_outputs(shared, cfg, run_dir, kf_stamps, stats, skip_scan_ids=blind_ids)
    if verify_log:
        import csv as _csv
        with open(run_dir / "verifications.csv", "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(verify_log[0].keys()))
            w.writeheader()
            w.writerows(verify_log)
    stats["run_dir"] = str(run_dir)
    return stats


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Fusion v2 unified runner (C++ shared map)")
    ap.add_argument("--mode", choices=("orb", "lidar", "orb_lidar", "lidar_orb"),
                    default="lidar")
    ap.add_argument("--dataset", type=Path, default=Path("datasets/lab_hybrid"))
    ap.add_argument("--output", type=Path, default=Path("fusion2_outputs"))
    ap.add_argument("--max-scans", type=int, default=0)
    ap.add_argument("--print-every", type=int, default=200)
    ap.add_argument("--verifier", choices=("bnb", "icp"), default="bnb",
                    help="scan-side loop verifier (modes lidar / orb_lidar)")
    ap.add_argument("--frontend", choices=("native", "legacy"), default="native",
                    help="visual front-end for modes orb/orb_lidar: native windowed "
                         "C++ VO (default) or the legacy Python ORB-SLAM stack")
    ap.add_argument("--lidar-frontend",
                    choices=("native_s2s", "native_s2m", "legacy_s2s", "legacy_s2m"),
                    default=None,
                    help="LiDAR local-mapping variant for modes lidar/lidar_orb "
                         "(scan_to_submap or scan_to_map, C++ or Python); default "
                         "comes from FusionV2Config.lidar_frontend")
    args = ap.parse_args(argv)

    # Mode x option validity (V4.3): fail fast with a clear message.
    if args.mode in ("orb", "orb_lidar") and args.lidar_frontend is not None:
        raise SystemExit(f"--lidar-frontend is not applicable to mode {args.mode} "
                         "(its front-end is the visual one; use --frontend)")
    if args.mode in ("orb", "lidar_orb") and args.verifier != "bnb":
        # default is "bnb"; an explicit icp here signals a misunderstanding
        import sys
        if "--verifier" in (argv or sys.argv):
            raise SystemExit(f"--verifier is not applicable to mode {args.mode} "
                             "(orb and lidar_orb verify loops with ORB+PnP)")

    cfg = FusionV2Config(mode=args.mode, dataset=args.dataset,
                         output_dir=args.output, max_scans=args.max_scans,
                         print_every=args.print_every,
                         scan_verifier=args.verifier)
    if args.lidar_frontend is not None:
        cfg.lidar_frontend = args.lidar_frontend
    cfg.visual_frontend = args.frontend
    cfg.extra["frontend"] = args.frontend

    if args.mode in ("lidar", "lidar_orb"):
        stats = run_lidar_mode(cfg)
    elif args.mode in ("orb", "orb_lidar"):
        stats = (run_orb_mode_native(cfg) if args.frontend == "native"
                 else run_orb_mode(cfg))
    else:
        raise SystemExit(f"unknown mode {args.mode}")

    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
