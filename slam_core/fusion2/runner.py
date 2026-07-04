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
from pathlib import Path
from typing import List, Optional

import numpy as np

import fusion_core as fc

from slam_core.common.types import Pose2 as PyPose2
from slam_core.fusion2.config import FusionV2Config
from slam_core.fusion2.Dependencies.dataset import LabHybridStream
from slam_core.fusion2.Dependencies.pose_utils import _rel, _fc_pose, _rel_sane
from slam_core.fusion2.Dependencies.outputs import (_anchor_poses,
                                                    render_fused_occupancy,
                                                    write_outputs, _rss_gb)
from slam_core.fusion2.backend import SharedMap, build_shared_map
from slam_core.fusion2.Loop_Proposer.proximity import propose_candidates
from slam_core.fusion2.Loop_Verifier.bnb import verify_candidate_bnb
from slam_core.fusion2.Loop_Verifier.icp import verify_candidate_icp


# ---------------------------------------------------------------------------
# Mode: lidar
# ---------------------------------------------------------------------------

def run_lidar_mode(cfg: FusionV2Config) -> dict:
    """Modes `lidar` (B&B/ICP scan verification) and `lidar_orb` (visual PnP
    verification of proximity proposals)."""
    import cv2

    from slam_core.fusion2.Front_End.lidar_frontend import make_lidar_frontend
    from slam_core.fusion2.Dependencies.visual_features import (cam_rel_to_base_se2,
                                                                extract_orb_rgbd)
    from slam_core.fusion2.Loop_Verifier.pnp import pnp_verify

    visual_backend = cfg.mode == "lidar_orb"
    K = None
    if visual_backend:
        # PnP verification needs camera intrinsics for LiDAR-attached RGB-D frames.
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
        # Store raw scans for mapping; filtered points are only for matching.
        raw_scan = scan

        kf_id += 1
        kf_stamps[kf_id] = t
        # Chain new odometry from the last optimized graph pose.
        if last_fe_pose is None:
            node_pose = fe_pose
        else:
            node_pose = last_graph_pose.compose(_rel(last_fe_pose, fe_pose))

        if visual_backend:
            # LiDAR-led tracking, visual loop verification.
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
        # Rehearsal merges only stationary repeated observations.
        if last_fe_pose is not None:
            d = _rel(last_fe_pose, fe_pose)
            stationary = math.hypot(d.x, d.y) < 0.05 and abs(d.theta) < math.radians(2.0)
        else:
            stationary = False
        res = shared.memory.insert(sig, similarity=-1.0 if stationary else 0.0)
        if res.rehearsal_merged:
            stats["rehearsal_merges"] += 1
            # Graph node remains; only duplicate payload is merged in memory.
        shared.graph.add_node(kf_id, node_pose)
        if last_fe_pose is not None:
            # The odometry spine is always consecutive keyframe relative motion.
            sig.add_link(kf_id - 1, fc.LinkType.NEIGHBOR, _rel(last_fe_pose, fe_pose),
                         cfg.spine_trans_weight, cfg.spine_rot_weight)
            shared.graph.add_spine_edge(kf_id - 1, kf_id, _rel(last_fe_pose, fe_pose))
        last_fe_pose = fe_pose
        last_graph_pose = node_pose
        stats["keyframes"] += 1

        # Loop closure: propose by proximity, then verify by the selected modality.
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
                        # Gate visual loop against the graph-predicted relative pose.
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
                        cand_pose = shared.graph.get_pose(cand)
                        rel = _rel(cand_pose, r.pose)
                        pred = _rel(cand_pose, node_pose)
                        if not _rel_sane(rel, pred, cfg.scan_rel_sanity_m,
                                         cfg.scan_rel_sanity_rad,
                                         cfg.scan_abs_max_m):
                            accepted = False
                            rel = None
                verify_ms.append((time.perf_counter() - t0) * 1000)
                row["accepted"] = accepted
                verify_log.append(row)
                if accepted and rel is not None:
                    # Store measured cand->query transform for loop analysis.
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
            # Periodic graph solve keeps odometry chaining near the corrected graph.
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
    from slam_core.fusion2.Dependencies.visual_features import cam_rel_to_base_se2
    from slam_core.fusion2.Loop_Verifier.pnp import pnp_verify

    stream = LabHybridStream(cfg.dataset, cfg.sync_tolerance_s)
    camera = make_rgbd_camera(cfg.dataset)
    fe = OrbSlamFrontendBackend(camera)
    shared = build_shared_map(cfg)
    # Legacy visual mode uses the v1 ORB loop proposer as a propose-only source.
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

        # Convert camera pose to the planar base frame used by the graph.
        from slam_core.fusion2.Dependencies.visual_features import BASE_T_CAM as _B
        p2 = project_pose3d_to_pose2(okf.pose, base_T_cam=_B,
                                     world_transform=CAMERA_GROUND_TRANSFORM)
        node_pose = fc.Pose2(p2.x, p2.y, p2.theta)
        kf_id += 1
        kf_stamps[kf_id] = t
        # DBoW candidates are reported in ORB keyframe ids; map them to fusion ids.
        orb_to_fusion[int(getattr(okf.source, 'kid', okf.id))] = kf_id

        kpts = np.ascontiguousarray(okf.keypoints, dtype=np.float32)
        des = okf.descriptors if okf.descriptors is not None else None
        # Back-project ORB keypoint depths for visual PnP verification.
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

        # Loop closure: DBoW proposes, then PnP or scan verification confirms.
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
                            # Reject scan matches that disagree with graph prediction.
                            cand_pose = shared.graph.get_pose(cand)
                            rel_lp = _rel(cand_pose, r.pose)
                            pred = _rel(cand_pose, node_pose)
                            if not _rel_sane(rel_lp, pred, cfg.scan_rel_sanity_m,
                                             cfg.scan_rel_sanity_rad,
                                             cfg.scan_abs_max_m):
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
                        # Strong PnP may bootstrap through large pre-loop drift.
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
    from slam_core.fusion2.Loop_Proposer.dbow import AppearanceIndex
    from slam_core.fusion2.Dependencies.visual_features import (BASE_T_CAM,
                                                                cam_rel_to_base_se2)
    from slam_core.fusion2.Loop_Verifier.pnp import pnp_verify
    from slam_core.fusion2.Front_End.vo_orb_frontend import NativeOrbFrontend

    # Convert camera poses to REP-103 planar base poses for graph consistency.
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
        # REINIT pose is dead-reckoned, so it gets a soft edge and no map painting.
        blind = nkf.state == fc.VoState.REINIT
        if blind:
            blind_ids.add(kf_id)
        # Use local-BA-refined relative motion between consecutive visual KFs.
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

        # Query DBoW before inserting the current descriptors to avoid self-match.
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
                    # ICP is prediction-seeded; B&B performs a bounded scan search.
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
                            # Reject scan matches that disagree with graph prediction.
                            cand_pose = shared.graph.get_pose(cand)
                            rel_lp = _rel(cand_pose, r.pose)
                            pred = _rel(cand_pose, node_pose)
                            if not _rel_sane(rel_lp, pred, cfg.scan_rel_sanity_m,
                                             cfg.scan_rel_sanity_rad,
                                             cfg.scan_abs_max_m):
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
                        # Strong PnP may bootstrap through large pre-loop drift.
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

        # Optimize on accepted loops immediately; otherwise follow the cadence.
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

    # Validate module choices before building the selected pipeline.
    if args.mode in ("orb", "orb_lidar") and args.lidar_frontend is not None:
        raise SystemExit(f"--lidar-frontend is not applicable to mode {args.mode} "
                         "(its front-end is the visual one; use --frontend)")
    if args.mode in ("orb", "lidar_orb") and args.verifier != "bnb":
        # Default "bnb" is ignored here; explicit ICP would be misleading.
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
