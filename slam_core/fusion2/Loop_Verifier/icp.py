"""Standalone ICP scan loop verifier (small_gicp), RTAB-Map RegistrationIcp style.

Independent of B&B: align the query scan against the candidate-local neighbourhood
cloud, seeded only by the graph-predicted relative pose, and accept on ICP's own
quality metrics (correspondence ratio + inlier RMSE).
"""
from __future__ import annotations

import math

import numpy as np

import fusion_core as fc

from slam_core.fusion2.config import FusionV2Config
from slam_core.fusion2.backend import SharedMap
from slam_core.fusion2.Loop_Verifier.neighborhood import candidate_neighborhood


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

    sigs, poses = candidate_neighborhood(shared, cfg, query_id, cand_id)
    if len(sigs) < 2:
        return None

    tgt_pts = []
    for s, p in zip(sigs, poses):
        sc = np.asarray(s.scan_xy, dtype=np.float64)
        c, sn = math.cos(p.theta), math.sin(p.theta)
        # Transform candidate-neighbour scans into the graph frame.
        tgt_pts.append(sc @ np.array([[c, sn], [-sn, c]]) + [p.x, p.y])
    tgt3 = np.c_[np.vstack(tgt_pts), np.zeros(sum(len(t) for t in tgt_pts))]

    # seed = graph-predicted pose (RTAB's odometry guess); NO coarse pre-search.
    seed_pose = query_pose
    q = np.asarray(query_scan, dtype=np.float64)
    c, sn = math.cos(seed_pose.theta), math.sin(seed_pose.theta)
    # Seed the source cloud at the graph prediction before ICP refinement.
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
    # Convert ICP's correction back to the absolute query pose expected by callers.
    corrected_pose = fc.Pose2(cx, cy, seed_pose.theta + dth)

    class _R:  # BnbResult-compatible shape
        success = bool(res.converged)
        coarse_score = corr_ratio       # gated by icp_accept_fitness (>=)
        refined_score = inlier_rmse     # gated by icp_accept_rmse (<=), metres
        pose = corrected_pose
        refined = True
    return _R()
