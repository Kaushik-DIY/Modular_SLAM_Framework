"""
Dataset-agnostic post-run PGO for Hector SLAM.

Supports:
  - datasets: lab_run_2, fr079, intel
  - matchers: scan_to_map, scan_to_submap

The script auto-detects dataset and matcher type from the trajectory path or
from the sidecar debug file when available.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hector.config as cfg

from slam_core.common.types import Pose2
from slam_core.common.se2 import wrap_angle
from slam_core.matching.scan_to_map import ScanToMapMatcher, _transform_points
from slam_core.matching.scan_to_submap_old import (
    CartoRefinementProblem,
    GaussNewtonLM,
    GNLMConfig,
    ProbabilityGrid,
    Submap2D,
    SubmapBuilder2D,
    correlative_match_two_stage,
)

from hector.eval._generic_eval_common import (
    configure_dataset,
    dataset_tag,
    default_map_size_m,
    ensure_dir,
    load_aligned_scan_points,
    parse_trajectory_context,
    resolve_latest_local_traj,
)


def _se2_arr(x: float, y: float, th: float) -> np.ndarray:
    return np.array([x, y, wrap_angle(th)], dtype=float)


def _se2_inv(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a[2]), np.sin(a[2])
    t = -np.array([[c, s], [-s, c]]) @ a[:2]
    return np.array([t[0], t[1], wrap_angle(-a[2])], dtype=float)


def _se2_compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c, s = np.cos(a[2]), np.sin(a[2])
    R = np.array([[c, -s], [s, c]])
    t = a[:2] + R @ b[:2]
    return np.array([t[0], t[1], wrap_angle(a[2] + b[2])], dtype=float)


def _se2_between(xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
    return _se2_compose(_se2_inv(xi), xj)


def _pose2_from_arr(a: np.ndarray) -> Pose2:
    return Pose2(float(a[0]), float(a[1]), float(a[2]))


def _bresenham_cells(gx0: int, gy0: int, gx1: int, gy1: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    dx = abs(gx1 - gx0)
    dy = abs(gy1 - gy0)
    sx = 1 if gx0 < gx1 else -1
    sy = 1 if gy0 < gy1 else -1
    err = dx - dy
    x, y = gx0, gy0
    while True:
        points.append((x, y))
        if x == gx1 and y == gy1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return points


def _integrate_scan_probability_grid(
    grid: ProbabilityGrid,
    pose_local: np.ndarray,
    scan_points_local: np.ndarray,
    ray_steps: int,
) -> None:
    endpoints_local = _transform_points(pose_local, scan_points_local)
    gx0, gy0 = grid.world_to_grid(float(pose_local[0]), float(pose_local[1]))
    if not grid.in_bounds(gx0, gy0):
        return

    for ex, ey in endpoints_local:
        gx1, gy1 = grid.world_to_grid(float(ex), float(ey))
        if not grid.in_bounds(gx1, gy1):
            continue

        if ray_steps > 0:
            cells = _bresenham_cells(gx0, gy0, gx1, gy1)
            if len(cells) > ray_steps:
                idx = np.linspace(0, len(cells) - 1, ray_steps, dtype=int)
                cells = [cells[int(k)] for k in idx]
        else:
            cells = _bresenham_cells(gx0, gy0, gx1, gy1)

        for cx, cy in cells[:-1]:
            grid.update_cell(cx, cy, grid.l_free)
        grid.update_cell(gx1, gy1, grid.l_occ)


def _build_anchor_local_grid(
    center_idx: int,
    poses: np.ndarray,
    scan_pts_list: List[np.ndarray],
    seed_half: int,
    map_res: float,
    map_size_m: float,
    l_free: float,
    l_occ: float,
    ray_steps: int,
) -> ProbabilityGrid:
    lo = max(0, center_idx - seed_half)
    hi = min(len(poses) - 1, center_idx + seed_half)

    grid = ProbabilityGrid(
        size_m=map_size_m,
        resolution=map_res,
        l0=cfg.L0,
        l_occ=l_occ,
        l_free=l_free,
        l_min=cfg.L_MIN,
        l_max=cfg.L_MAX,
    )

    anchor = poses[center_idx]
    anchor_inv = _se2_inv(anchor)

    for idx in range(lo, hi + 1):
        pts = scan_pts_list[idx]
        if pts.shape[0] == 0:
            continue
        pose_local = _se2_compose(anchor_inv, poses[idx])
        _integrate_scan_probability_grid(
            grid=grid,
            pose_local=pose_local,
            scan_points_local=pts,
            ray_steps=ray_steps,
        )

    return grid


def _scan_context_descriptor(
    pts: np.ndarray,
    *,
    num_rings: int,
    num_sectors: int,
    max_radius: float,
) -> np.ndarray:
    desc = np.zeros((num_rings, num_sectors), dtype=np.float32)
    if pts.shape[0] == 0:
        return desc

    r = np.hypot(pts[:, 0], pts[:, 1])
    a = (np.arctan2(pts[:, 1], pts[:, 0]) + np.pi) / (2.0 * np.pi)
    mask = (r > 0.05) & (r <= max_radius)
    if not np.any(mask):
        return desc

    ring = np.floor((r[mask] / max_radius) * num_rings).astype(np.int32)
    sector = np.floor(a[mask] * num_sectors).astype(np.int32)
    ring = np.clip(ring, 0, num_rings - 1)
    sector = np.mod(sector, num_sectors)
    desc[ring, sector] = 1.0
    return desc


def _scan_context_best_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    best_score = -1.0
    best_shift = 0
    a_norm = float(np.linalg.norm(a))
    if a_norm <= 1e-9:
        return best_score, best_shift

    for shift in range(a.shape[1]):
        b_shift = np.roll(b, shift=shift, axis=1)
        denom = a_norm * float(np.linalg.norm(b_shift))
        if denom <= 1e-9:
            continue
        score = float(np.sum(a * b_shift) / denom)
        if score > best_score:
            best_score = score
            best_shift = shift
    return best_score, best_shift


def load_trajectory_full(path: str) -> Tuple[np.ndarray, np.ndarray]:
    stamps_list, poses_list = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                t, x, y, th, sc = map(float, parts[:5])
            except ValueError:
                continue
            if sc >= 0.0:
                stamps_list.append(t)
                poses_list.append([x, y, wrap_angle(th)])
    if not stamps_list:
        raise RuntimeError(f"No accepted poses in {path}")
    return np.array(stamps_list, dtype=float), np.array(poses_list, dtype=float)


class LoopClosure(NamedTuple):
    i: int
    j: int
    z_ij: np.ndarray
    score: float
    source: str = "scan_match"


def _loop_is_consistent(
    poses: np.ndarray,
    i: int,
    j: int,
    z_ij: np.ndarray,
    *,
    max_trans_error: float,
    max_rot_error: float,
) -> tuple[bool, float, float]:
    z_pred = _se2_between(poses[i], poses[j])
    err = _se2_between(z_ij, z_pred)
    trans_err = float(np.hypot(err[0], err[1]))
    rot_err = float(abs(wrap_angle(err[2])))
    return (
        trans_err <= float(max_trans_error) and rot_err <= float(max_rot_error),
        trans_err,
        rot_err,
    )


class Relation2D(NamedTuple):
    a: float
    b: float
    dx: float
    dy: float
    dtheta: float


def _nearest_index(stamps: np.ndarray, t: float) -> int:
    idx = int(np.searchsorted(stamps, float(t)))
    if idx <= 0:
        return 0
    if idx >= len(stamps):
        return len(stamps) - 1
    if abs(stamps[idx] - t) < abs(stamps[idx - 1] - t):
        return idx
    return idx - 1


def load_relations(path: str) -> List[Relation2D]:
    relations: List[Relation2D] = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                a = float(parts[0])
                b = float(parts[1])
                dx = float(parts[2])
                dy = float(parts[3])
                dtheta = float(parts[-1])
            except ValueError:
                continue
            relations.append(Relation2D(a=a, b=b, dx=dx, dy=dy, dtheta=wrap_angle(dtheta)))
    return relations


def relation_residual_summary(
    poses: np.ndarray,
    stamps: np.ndarray,
    relations: List[Relation2D],
    *,
    max_time_error_s: float,
) -> dict:
    trans_err: List[float] = []
    rot_err: List[float] = []
    skipped = 0

    for rel in relations:
        if rel.a < stamps[0] or rel.a > stamps[-1] or rel.b < stamps[0] or rel.b > stamps[-1]:
            skipped += 1
            continue

        i = _nearest_index(stamps, rel.a)
        j = _nearest_index(stamps, rel.b)
        if abs(stamps[i] - rel.a) > max_time_error_s or abs(stamps[j] - rel.b) > max_time_error_s:
            skipped += 1
            continue

        z_ref = _se2_arr(rel.dx, rel.dy, rel.dtheta)
        z_hat = _se2_between(poses[i], poses[j])
        e = _se2_between(z_ref, z_hat)
        trans_err.append(float(np.hypot(e[0], e[1])))
        rot_err.append(float(abs(wrap_angle(e[2]))))

    if not trans_err:
        return {
            "used": 0,
            "skipped": skipped,
            "rmse_trans_m": float("nan"),
            "rmse_rot_deg": float("nan"),
        }

    trans_arr = np.array(trans_err, dtype=float)
    rot_arr = np.rad2deg(np.array(rot_err, dtype=float))
    return {
        "used": int(trans_arr.size),
        "skipped": int(skipped),
        "rmse_trans_m": float(np.sqrt(np.mean(trans_arr**2))),
        "rmse_rot_deg": float(np.sqrt(np.mean(rot_arr**2))),
    }


def relation_closures_from_file(
    path: str,
    stamps: np.ndarray,
    *,
    max_time_error_s: float,
    min_index_gap: int,
) -> tuple[List[LoopClosure], dict]:
    relations = load_relations(path)
    closures: List[LoopClosure] = []
    skipped_time = 0
    skipped_gap = 0

    for rel in relations:
        if rel.a < stamps[0] or rel.a > stamps[-1] or rel.b < stamps[0] or rel.b > stamps[-1]:
            skipped_time += 1
            continue

        i = _nearest_index(stamps, rel.a)
        j = _nearest_index(stamps, rel.b)
        if abs(stamps[i] - rel.a) > max_time_error_s or abs(stamps[j] - rel.b) > max_time_error_s:
            skipped_time += 1
            continue
        if i == j or abs(j - i) < int(min_index_gap):
            skipped_gap += 1
            continue

        z_ij = _se2_arr(rel.dx, rel.dy, rel.dtheta)
        if i > j:
            i, j = j, i
            z_ij = _se2_inv(z_ij)

        closures.append(LoopClosure(i=i, j=j, z_ij=z_ij, score=1.0, source="relations"))

    stats = {
        "relations_total": len(relations),
        "relations_used": len(closures),
        "relations_skipped_time": skipped_time,
        "relations_skipped_gap": skipped_gap,
        "relations_min_index_gap": int(min_index_gap),
        "relations_max_time_error_s": float(max_time_error_s),
    }
    return closures, stats


def _build_mini_map(
    center_idx: int,
    poses: np.ndarray,
    scan_pts_list: List[np.ndarray],
    seed_half: int,
    map_res: float,
    map_size_m: float,
    l_free: float,
    l_occ: float,
    ray_steps: int,
) -> ScanToMapMatcher:
    lo = max(0, center_idx - seed_half)
    hi = min(len(poses) - 1, center_idx + seed_half)

    map_params = dict(
        base_res=map_res,
        size_m=map_size_m,
        num_levels=cfg.PYRAMID_LEVELS,
        l0=cfg.L0,
        l_min=cfg.L_MIN,
        l_max=cfg.L_MAX,
        l_free=l_free,
        l_occ=l_occ,
        ray_steps=ray_steps,
        bootstrap_scans=1,
    )
    corr_params = dict(
        gn_iters_per_level=cfg.GN_ITERS_PER_LEVEL,
        gn_damping=cfg.GN_DAMPING,
        min_points=max(20, cfg.CORR_MAP_MIN_POINTS),
        min_inliers_accept=max(20, cfg.CORR_MAP_MIN_INLIERS),
        min_score=min(0.50, cfg.CORR_MAP_MIN_SCORE),
        step_clip_xy=max(0.20, cfg.CORR_MAP_STEP_CLIP_XY),
        step_clip_th=np.deg2rad(8.0),
    )

    matcher = ScanToMapMatcher(map_params=map_params, corr_params=corr_params)

    # Build the mini-map in a local frame anchored at poses[center_idx].
    # GridMap is always centered at world (0,0), so inserting scans at their
    # global world positions fails whenever the robot is >size_m/2 from origin.
    # Transforming to the anchor-local frame keeps all inserts near (0,0).
    anchor = poses[center_idx]
    c_a, s_a = np.cos(anchor[2]), np.sin(anchor[2])
    R_anchor_T = np.array([[c_a, s_a], [-s_a, c_a]])  # R(anchor_th)^T

    for idx in range(lo, hi + 1):
        pose_world = poses[idx].copy()
        pts = scan_pts_list[idx]
        if pts.shape[0] == 0:
            continue
        pts_world = _transform_points(pose_world, pts)
        # transform pose and points into anchor-local frame
        pose_local = _se2_compose(_se2_inv(anchor), pose_world)
        pts_local = (pts_world - anchor[:2]) @ R_anchor_T
        for grid in matcher.pyr.levels:
            grid.integrate_scan_simple(
                pose=pose_local,
                pts_world=pts_local,
                l_free=l_free,
                l_occ=l_occ,
                ray_steps=ray_steps,
            )
    matcher.initialized = True
    matcher._is_initialized = True
    matcher._bootstrap_count = matcher._bootstrap_scans
    return matcher


def _build_mini_submap(
    center_idx: int,
    poses: np.ndarray,
    scan_pts_list: List[np.ndarray],
    seed_half: int,
    map_res: float,
    map_size_m: float,
    l_free: float,
    l_occ: float,
    ray_steps: int,
) -> Submap2D:
    lo = max(0, center_idx - seed_half)
    hi = min(len(poses) - 1, center_idx + seed_half)

    builder = SubmapBuilder2D(
        submap_size_m=map_size_m,
        resolution=map_res,
        scans_per_submap=10000,
        ray_steps=ray_steps,
        l0=cfg.L0,
        l_free=l_free,
        l_occ=l_occ,
        l_min=cfg.L_MIN,
        l_max=cfg.L_MAX,
    )

    for idx in range(lo, hi + 1):
        pts = scan_pts_list[idx]
        if pts.shape[0] == 0:
            continue
        pose_arr = poses[idx]
        builder.insert_scan(Pose2(float(pose_arr[0]), float(pose_arr[1]), float(pose_arr[2])), pts)
    return builder.get_active_submaps()[0]


def detect_loop_closures_map(
    poses: np.ndarray,
    scan_pts_list: List[np.ndarray],
    search_radius: float,
    min_index_gap: int,
    lc_min_score: float,
    keyframe_stride: int,
    seed_half: int,
    map_res: float,
    map_size_m: float,
    l_free: float,
    l_occ: float,
    ray_steps: int,
    max_per_query: int = 1,
    consistency_max_trans: float = 0.75,
    consistency_max_rot: float = np.deg2rad(20.0),
    max_measurement_trans: Optional[float] = None,
    max_measurement_rot: Optional[float] = None,
) -> List[LoopClosure]:
    closures: List[LoopClosure] = []
    xy = poses[:, :2]
    kf_indices = list(range(0, len(poses), keyframe_stride))

    # Cache mini-maps by center index — the same reference pose i can be the
    # nearest candidate for many different query keyframes j.  Building once
    # and reusing cuts wall-clock time roughly proportional to the average
    # number of j keyframes that share each i candidate.
    mini_map_cache: dict = {}

    for j in kf_indices:
        dists = np.linalg.norm(xy[:max(0, j - min_index_gap)] - xy[j], axis=1)
        candidates = np.where(dists <= search_radius)[0]
        if len(candidates) == 0:
            continue
        candidates = candidates[np.argsort(dists[candidates])[:10]]
        accepted_for_query = 0

        for i in [int(x) for x in candidates]:
            pts_j = scan_pts_list[j]
            if pts_j.shape[0] < 30:
                continue

            anchor = poses[i]
            if i not in mini_map_cache:
                mini_map_cache[i] = _build_mini_map(
                    center_idx=i,
                    poses=poses,
                    scan_pts_list=scan_pts_list,
                    seed_half=seed_half,
                    map_res=map_res,
                    map_size_m=map_size_m,
                    l_free=l_free,
                    l_occ=l_occ,
                    ray_steps=ray_steps,
                )
            mini_map = mini_map_cache[i]

            # The mini-map is in anchor-local frame (centered at poses[i]).
            # Transform poses[j] into that frame for the GN initial guess.
            pred_local = _se2_compose(_se2_inv(anchor), poses[j])
            result = mini_map.match(
                t=0.0,
                scan_points_local=pts_j,
                predicted_pose_world=_pose2_from_arr(pred_local),
            )
            if result.success and result.score >= lc_min_score:
                # Back-transform matched pose from anchor-local to world frame
                matched_local = _se2_arr(
                    result.pose_world.x, result.pose_world.y, result.pose_world.theta
                )
                matched_world = _se2_compose(anchor, matched_local)
                z_ij = _se2_between(poses[i], matched_world)
                if max_measurement_trans is not None and np.hypot(z_ij[0], z_ij[1]) > max_measurement_trans:
                    continue
                if max_measurement_rot is not None and abs(wrap_angle(z_ij[2])) > max_measurement_rot:
                    continue
                ok, terr, rerr = _loop_is_consistent(
                    poses,
                    i,
                    j,
                    z_ij,
                    max_trans_error=consistency_max_trans,
                    max_rot_error=consistency_max_rot,
                )
                if not ok:
                    print(
                        f"[pgo_any]   LC rejected: ({i},{j}) inconsistent  "
                        f"score={result.score:.3f}  terr={terr:.3f}m  "
                        f"rerr={np.rad2deg(rerr):.1f}deg"
                    )
                    continue
                closures.append(LoopClosure(i=i, j=j, z_ij=z_ij, score=result.score))
                accepted_for_query += 1
                print(
                    f"[pgo_any]   LC accepted: ({i},{j})  score={result.score:.3f}  "
                    f"dxy={np.hypot(z_ij[0], z_ij[1]):.3f}m  dth={np.rad2deg(z_ij[2]):.1f}deg"
                )
                if accepted_for_query >= int(max_per_query):
                    break
    return closures


def detect_loop_closures_submap(
    poses: np.ndarray,
    scan_pts_list: List[np.ndarray],
    search_radius: float,
    min_index_gap: int,
    lc_min_score: float,
    keyframe_stride: int,
    seed_half: int,
    map_res: float,
    map_size_m: float,
    l_free: float,
    l_occ: float,
    ray_steps: int,
    max_per_query: int = 1,
    consistency_max_trans: float = 0.75,
    consistency_max_rot: float = np.deg2rad(20.0),
    max_measurement_trans: Optional[float] = None,
    max_measurement_rot: Optional[float] = None,
) -> List[LoopClosure]:
    closures: List[LoopClosure] = []
    xy = poses[:, :2]
    kf_indices = list(range(0, len(poses), keyframe_stride))

    for j in kf_indices:
        dists = np.linalg.norm(xy[:max(0, j - min_index_gap)] - xy[j], axis=1)
        candidates = np.where(dists <= search_radius)[0]
        if len(candidates) == 0:
            continue
        candidates = candidates[np.argsort(dists[candidates])[:3]]
        accepted_for_query = 0

        for i in [int(x) for x in candidates]:
            pts_j = scan_pts_list[j]
            if pts_j.shape[0] < 30:
                continue

            mini_submap = _build_mini_submap(
                center_idx=i,
                poses=poses,
                scan_pts_list=scan_pts_list,
                seed_half=seed_half,
                map_res=map_res,
                map_size_m=map_size_m,
                l_free=l_free,
                l_occ=l_occ,
                ray_steps=ray_steps,
            )

            sub_pts = pts_j
            if pts_j.shape[0] > cfg.SUBMAP_MAX_MATCH_POINTS:
                np.random.seed(j)
                idx = np.random.choice(pts_j.shape[0], cfg.SUBMAP_MAX_MATCH_POINTS, replace=False)
                sub_pts = pts_j[idx]

            # Use poses[j] as initial guess.  The correlative search window
            # (coarse_xy_window ≈ 0.5 m) must be centred on the scan's expected
            # location; poses[i] can be up to search_radius away, which puts
            # scan j entirely outside the search window on fr079/intel.
            predicted = _pose2_from_arr(poses[j])
            best_pose, score = correlative_match_two_stage(
                prob_img=mini_submap.grid.probability(),
                grid_origin_xy=mini_submap.grid.origin_world,
                res=mini_submap.grid.res,
                points_local=sub_pts,
                initial_submap_pose=predicted,
                min_valid=max(20, cfg.SUBMAP_MIN_VALID),
                coarse_xy_window=max(0.5, cfg.SUBMAP_COARSE_XY_WINDOW),
                coarse_th_window=max(np.deg2rad(15.0), cfg.SUBMAP_COARSE_TH_WINDOW),
                coarse_xy_step=max(0.15, cfg.SUBMAP_COARSE_XY_STEP),
                coarse_th_step=max(np.deg2rad(4.0), cfg.SUBMAP_COARSE_TH_STEP),
            )
            if score < lc_min_score:
                continue

            problem = CartoRefinementProblem(
                grid=mini_submap.grid,
                pts_local=sub_pts,
                pred_pose_sub=np.array([best_pose.x, best_pose.y, best_pose.theta]),
                min_points=max(20, cfg.SUBMAP_REFINE_MIN_POINTS),
                w_trans=cfg.SUBMAP_REFINE_W_TRANS,
                w_rot=cfg.SUBMAP_REFINE_W_ROT,
            )
            solver = GaussNewtonLM(GNLMConfig(iters=10, damping=1e-4, eps_stop=1e-4, verbose=False))
            x0 = np.array([best_pose.x, best_pose.y, best_pose.theta], dtype=float)
            x_opt = solver.solve(x0, problem.compute_r_J).reshape(3)
            matched_arr = _se2_arr(x_opt[0], x_opt[1], x_opt[2])
            z_ij = _se2_between(poses[i], matched_arr)
            if max_measurement_trans is not None and np.hypot(z_ij[0], z_ij[1]) > max_measurement_trans:
                continue
            if max_measurement_rot is not None and abs(wrap_angle(z_ij[2])) > max_measurement_rot:
                continue
            ok, terr, rerr = _loop_is_consistent(
                poses,
                i,
                j,
                z_ij,
                max_trans_error=consistency_max_trans,
                max_rot_error=consistency_max_rot,
            )
            if not ok:
                print(
                    f"[pgo_any]   LC rejected: ({i},{j}) inconsistent  "
                    f"score={score:.3f}  terr={terr:.3f}m  "
                    f"rerr={np.rad2deg(rerr):.1f}deg"
                )
                continue
            closures.append(LoopClosure(i=i, j=j, z_ij=z_ij, score=score))
            accepted_for_query += 1
            print(
                f"[pgo_any]   LC accepted: ({i},{j})  score={score:.3f}  "
                f"dxy={np.hypot(z_ij[0], z_ij[1]):.3f}m  dth={np.rad2deg(z_ij[2]):.1f}deg"
            )
            if accepted_for_query >= int(max_per_query):
                break
    return closures


def detect_loop_closures_scan_context(
    poses: np.ndarray,
    scan_pts_list: List[np.ndarray],
    *,
    min_index_gap: int,
    keyframe_stride: int,
    seed_half: int,
    map_res: float,
    map_size_m: float,
    l_free: float,
    l_occ: float,
    ray_steps: int,
    descriptor_min_score: float,
    lc_min_score: float,
    top_k: int,
    num_rings: int = 20,
    num_sectors: int = 60,
    max_radius: float = 20.0,
    coarse_xy_window: float = 1.5,
    coarse_th_window: float = np.pi,
    coarse_xy_step: float = 0.15,
    coarse_th_step: float = np.deg2rad(10.0),
    max_match_xy: float = 1.2,
    max_abs_yaw: Optional[float] = None,
    reciprocal_check: bool = True,
    reciprocal_xy_tol: float = 0.5,
    reciprocal_yaw_tol: float = np.deg2rad(20.0),
) -> List[LoopClosure]:
    closures: List[LoopClosure] = []
    kf_indices = list(range(0, len(poses), keyframe_stride))
    desc_by_idx = {
        idx: _scan_context_descriptor(
            scan_pts_list[idx],
            num_rings=num_rings,
            num_sectors=num_sectors,
            max_radius=max_radius,
        )
        for idx in kf_indices
    }
    grid_cache: dict[int, ProbabilityGrid] = {}
    accepted_for_j: set[int] = set()

    print(
        "[pgo_any] Scan-context LC: "
        f"keyframes={len(kf_indices)}  top_k={top_k}  "
        f"descriptor_min={descriptor_min_score:.2f}  geom_min={lc_min_score:.2f}"
    )

    for jj, j in enumerate(kf_indices):
        if j in accepted_for_j:
            continue

        candidates: list[tuple[float, int, int]] = []
        desc_j = desc_by_idx[j]
        for i in kf_indices[:jj]:
            if j - i < min_index_gap:
                continue
            score, shift = _scan_context_best_shift(desc_by_idx[i], desc_j)
            if score >= descriptor_min_score:
                candidates.append((score, i, shift))

        if not candidates:
            continue

        candidates.sort(reverse=True, key=lambda item: item[0])
        for desc_score, i, shift in candidates[:top_k]:
            pts_j = scan_pts_list[j]
            if pts_j.shape[0] < 30:
                continue

            if i not in grid_cache:
                grid_cache[i] = _build_anchor_local_grid(
                    center_idx=i,
                    poses=poses,
                    scan_pts_list=scan_pts_list,
                    seed_half=seed_half,
                    map_res=map_res,
                    map_size_m=map_size_m,
                    l_free=l_free,
                    l_occ=l_occ,
                    ray_steps=ray_steps,
                )
            grid = grid_cache[i]

            # A positive circular shift means scan j's sectors had to rotate
            # forward to best overlap i, so the pose yaw starts at -shift bins.
            yaw_guess = wrap_angle(-2.0 * np.pi * float(shift) / float(num_sectors))
            initial = Pose2(0.0, 0.0, yaw_guess)
            best_pose, geom_score = correlative_match_two_stage(
                prob_img=grid.probability(),
                grid_origin_xy=grid.origin_world,
                res=grid.res,
                points_local=pts_j,
                initial_submap_pose=initial,
                min_valid=max(20, cfg.SUBMAP_MIN_VALID),
                precomp_levels=3,
                coarse_level=2,
                coarse_xy_window=coarse_xy_window,
                coarse_th_window=coarse_th_window,
                coarse_xy_step=coarse_xy_step,
                coarse_th_step=coarse_th_step,
                fine_level=0,
                fine_xy_window=0.25,
                fine_th_window=np.deg2rad(12.0),
                fine_xy_step=0.05,
                fine_th_step=np.deg2rad(2.0),
            )
            if geom_score < lc_min_score:
                continue

            problem = CartoRefinementProblem(
                grid=grid,
                pts_local=pts_j,
                pred_pose_sub=np.array([best_pose.x, best_pose.y, best_pose.theta]),
                min_points=max(20, cfg.SUBMAP_REFINE_MIN_POINTS),
                w_trans=0.1,
                w_rot=0.5,
            )
            solver = GaussNewtonLM(GNLMConfig(iters=10, damping=1e-4, eps_stop=1e-4, verbose=False))
            x0 = np.array([best_pose.x, best_pose.y, best_pose.theta], dtype=float)
            x_opt = solver.solve(x0, problem.compute_r_J).reshape(3)
            z_ij = _se2_arr(x_opt[0], x_opt[1], x_opt[2])
            if np.hypot(z_ij[0], z_ij[1]) > max_match_xy:
                continue
            if max_abs_yaw is not None and abs(wrap_angle(z_ij[2])) > max_abs_yaw:
                continue

            if reciprocal_check:
                if j not in grid_cache:
                    grid_cache[j] = _build_anchor_local_grid(
                        center_idx=j,
                        poses=poses,
                        scan_pts_list=scan_pts_list,
                        seed_half=seed_half,
                        map_res=map_res,
                        map_size_m=map_size_m,
                        l_free=l_free,
                        l_occ=l_occ,
                        ray_steps=ray_steps,
                    )
                inv_guess = _se2_inv(z_ij)
                back_pose, back_score = correlative_match_two_stage(
                    prob_img=grid_cache[j].probability(),
                    grid_origin_xy=grid_cache[j].origin_world,
                    res=grid_cache[j].res,
                    points_local=scan_pts_list[i],
                    initial_submap_pose=_pose2_from_arr(inv_guess),
                    min_valid=max(20, cfg.SUBMAP_MIN_VALID),
                    precomp_levels=3,
                    coarse_level=2,
                    coarse_xy_window=0.6,
                    coarse_th_window=np.deg2rad(30.0),
                    coarse_xy_step=0.10,
                    coarse_th_step=np.deg2rad(6.0),
                    fine_level=0,
                    fine_xy_window=0.20,
                    fine_th_window=np.deg2rad(10.0),
                    fine_xy_step=0.05,
                    fine_th_step=np.deg2rad(2.0),
                )
                if back_score < lc_min_score:
                    continue
                z_ji = _se2_arr(back_pose.x, back_pose.y, back_pose.theta)
                consistency = _se2_compose(z_ij, z_ji)
                if np.hypot(consistency[0], consistency[1]) > reciprocal_xy_tol:
                    continue
                if abs(wrap_angle(consistency[2])) > reciprocal_yaw_tol:
                    continue

            closures.append(
                LoopClosure(
                    i=i,
                    j=j,
                    z_ij=z_ij,
                    score=float(geom_score),
                    source="scan_context",
                )
            )
            accepted_for_j.add(j)
            print(
                f"[pgo_any]   SC-LC accepted: ({i},{j})  "
                f"desc={desc_score:.3f}  score={geom_score:.3f}  "
                f"dxy={np.hypot(z_ij[0], z_ij[1]):.3f}m  "
                f"dth={np.rad2deg(z_ij[2]):.1f}deg"
            )
            break

    return closures


def build_and_optimize(
    poses: np.ndarray,
    closures: List[LoopClosure],
    odom_sig_xy: float,
    odom_sig_th: float,
    lc_sig_xy: float,
    lc_sig_th: float,
    pgo_iters: int,
    pgo_damping: float,
) -> np.ndarray:
    n = len(poses)
    n_vars = 3 * n
    omega_odom = np.diag([1.0 / odom_sig_xy**2, 1.0 / odom_sig_xy**2, 1.0 / odom_sig_th**2])
    omega_lc = np.diag([1.0 / lc_sig_xy**2, 1.0 / lc_sig_xy**2, 1.0 / lc_sig_th**2])

    HUBER_DELTA = 3.0  # whitened residual threshold; matches slam/pose_graph.py

    edges_init = [(k, k + 1, _se2_between(poses[k], poses[k + 1]), omega_odom) for k in range(n - 1)]
    edges_init.extend((lc.i, lc.j, lc.z_ij, omega_lc) for lc in closures)

    x = poses.reshape(-1).copy()
    print(f"[pgo_any] Optimising {n} nodes, {n - 1} seq edges, {len(closures)} LC edges")

    for iteration in range(pgo_iters):
        H = sp.lil_matrix((n_vars, n_vars), dtype=float)
        g = np.zeros(n_vars, dtype=float)

        for ei, ej, z, omega in edges_init:
            xi = x[3 * ei : 3 * ei + 3]
            xj = x[3 * ej : 3 * ej + 3]

            ci, si = np.cos(xi[2]), np.sin(xi[2])
            Rth_i_T = np.array([[ci, si], [-si, ci]])
            dp = xj[:2] - xi[:2]
            t_hat = Rth_i_T @ dp
            th_hat = wrap_angle(xj[2] - xi[2])
            z_hat = np.array([t_hat[0], t_hat[1], th_hat], dtype=float)
            r_k = z_hat - z
            r_k[2] = wrap_angle(r_k[2])

            # Huber robust weighting — protects against false loop closures
            whitened = float(np.sqrt(r_k @ omega @ r_k))
            w = 1.0 if whitened <= HUBER_DELTA else (HUBER_DELTA / whitened)
            omega_eff = w * omega

            dRdth_dp = np.array([-si * dp[0] + ci * dp[1], -ci * dp[0] - si * dp[1]], dtype=float)
            A = np.zeros((3, 3), dtype=float)   # de/d xi
            A[:2, :2] = -Rth_i_T
            A[:2, 2] = dRdth_dp
            A[2, 2] = -1.0

            B = np.zeros((3, 3), dtype=float)   # de/d xj
            B[:2, :2] = Rth_i_T
            B[2, 2] = 1.0

            ii = slice(3 * ei, 3 * ei + 3)
            jj = slice(3 * ej, 3 * ej + 3)

            H[ii, ii] = H[ii, ii] + A.T @ omega_eff @ A
            H[ii, jj] = H[ii, jj] + A.T @ omega_eff @ B
            H[jj, ii] = H[jj, ii] + B.T @ omega_eff @ A
            H[jj, jj] = H[jj, jj] + B.T @ omega_eff @ B

            g[ii] += A.T @ omega_eff @ r_k
            g[jj] += B.T @ omega_eff @ r_k

        # Structural anchoring: fix node 0 by solving the reduced (3N-3)×(3N-3) system
        H_csr = H.tocsr() + pgo_damping * sp.eye(n_vars, format="csr")
        H_red = H_csr[3:, 3:]
        g_red = g[3:]

        try:
            dx_red = -spla.spsolve(H_red, g_red)
        except Exception as exc:
            print(f"[pgo_any] WARNING: sparse solve failed at iter {iteration}: {exc}")
            break

        dx = np.concatenate([[0.0, 0.0, 0.0], dx_red])
        x += dx
        for k in range(n):
            x[3 * k + 2] = wrap_angle(x[3 * k + 2])

        step_norm = float(np.linalg.norm(dx))
        if (iteration % 5 == 0) or (iteration == pgo_iters - 1):
            print(f"[pgo_any]   iter {iteration:3d}: |dx|={step_norm:.4e}")
        if step_norm < 1e-7:
            print(f"[pgo_any]   Converged at iter {iteration}")
            break

    return x.reshape(n, 3)


def rebuild_map_from_poses(
    matcher_type: str,
    poses_opt: np.ndarray,
    scan_pts_list: List[np.ndarray],
    map_res: float,
    map_size_m: float,
    l_free: float,
    l_occ: float,
    ray_steps: int,
):
    if matcher_type == "scan_to_map":
        from slam_core.matching.scan_to_map import GridMap

        grid = GridMap(res=map_res, size_m=map_size_m, l_min=cfg.L_MIN, l_max=cfg.L_MAX)
        for pose_arr, pts in zip(poses_opt, scan_pts_list):
            if pts.shape[0] == 0:
                continue
            pts_world = _transform_points(pose_arr, pts)
            grid.integrate_scan_simple(
                pose=pose_arr,
                pts_world=pts_world,
                l_free=l_free,
                l_occ=l_occ,
                ray_steps=ray_steps,
            )
        return grid

    builder = SubmapBuilder2D(
        submap_size_m=cfg.SUBMAP_SIZE_METERS,
        resolution=map_res,
        scans_per_submap=cfg.SCANS_PER_SUBMAP,
        ray_steps=ray_steps,
        l0=cfg.L0,
        l_free=l_free,
        l_occ=l_occ,
        l_min=cfg.L_MIN,
        l_max=cfg.L_MAX,
    )
    for pose_arr, pts in zip(poses_opt, scan_pts_list):
        if pts.shape[0] == 0:
            continue
        builder.insert_scan(Pose2(float(pose_arr[0]), float(pose_arr[1]), float(pose_arr[2])), pts)

    global_grid = ProbabilityGrid(
        size_m=map_size_m,
        resolution=map_res,
        l0=cfg.L0,
        l_occ=l_occ,
        l_free=l_free,
        l_min=cfg.L_MIN,
        l_max=cfg.L_MAX,
    )
    for sm in builder.finished_submaps + builder.active:
        sm_grid = sm.grid
        gx, gy = global_grid.world_to_grid(*sm_grid.origin_world)
        gx0, gy0 = max(0, gx), max(0, gy)
        gx1 = min(global_grid.w, gx + sm_grid.w)
        gy1 = min(global_grid.h, gy + sm_grid.h)
        sx0, sy0 = gx0 - gx, gy0 - gy
        sx1 = sx0 + (gx1 - gx0)
        sy1 = sy0 + (gy1 - gy0)
        if gx1 > gx0 and gy1 > gy0:
            global_patch = global_grid.L[gy0:gy1, gx0:gx1]
            sm_patch = sm_grid.L[sy0:sy1, sx0:sx1]
            global_grid.L[gy0:gy1, gx0:gx1] = np.clip(global_patch + sm_patch, global_grid.l_min, global_grid.l_max)
    return global_grid


def _grid_prob(grid) -> np.ndarray:
    if hasattr(grid, "prob"):
        return grid.prob().astype(np.float32)
    return grid.probability().astype(np.float32)


def _grid_xy(grid, traj_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if traj_xy.shape[0] == 0:
        return np.array([]), np.array([])
    if hasattr(grid, "size"):
        gxy = grid.world_to_grid(traj_xy)
        return gxy[:, 0], grid.size - 1 - gxy[:, 1]
    gxy = np.array([grid.world_to_grid(row[0], row[1]) for row in traj_xy], dtype=int)
    return gxy[:, 0], grid.h - 1 - gxy[:, 1]


def save_map_png(
    prob_before: np.ndarray,
    prob_after: np.ndarray,
    traj_xy_before: np.ndarray,
    traj_xy_after: np.ndarray,
    grid_before,
    grid_after,
    out_path: str,
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
    except ImportError:
        print("[pgo_any] matplotlib not available, skipping PNG")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=120)
    fig.patch.set_facecolor("#1a1a2e")
    for ax, label, prob, traj_xy, grid in [
        (axes[0], "Before PGO", prob_before, traj_xy_before, grid_before),
        (axes[1], "After PGO", prob_after, traj_xy_after, grid_after),
    ]:
        ax.set_facecolor("#1a1a2e")
        ax.imshow(np.flipud(prob), cmap="binary_r", vmin=0.2, vmax=0.8, interpolation="nearest", origin="upper")
        px, py = _grid_xy(grid, traj_xy)
        if len(px) > 0:
            colours = cm.cool(np.linspace(0, 1, len(px)))
            ax.scatter(px, py, c=colours, s=4, linewidths=0, zorder=3, alpha=0.85)
            ax.plot(px[0], py[0], "o", color="#00ff88", markersize=10, zorder=4, label="Start")
            ax.plot(px[-1], py[-1], "X", color="#ff4466", markersize=12, zorder=4, label="End")
        ax.set_title(label, color="white", fontsize=12, pad=8)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444455")
        ax.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e", labelcolor="white", framealpha=0.8)
    plt.suptitle(title, color="white", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[pgo_any] Map PNG saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-run PGO + loop closure for Hector SLAM across datasets.")
    ap.add_argument("--traj", default=None, help="Input local trajectory. Auto-discovered if omitted.")
    ap.add_argument("--dataset", default=None, choices=["lab_run_2", "fr079", "intel"])
    ap.add_argument("--scan-variant", dest="scan_variant", default=None, choices=["raw", "360"])
    ap.add_argument("--matcher", default=None, choices=["scan_to_map", "scan_to_submap"])
    ap.add_argument("--hector_out", default="hector_outputs")
    ap.add_argument("--out_dir", default="hector_outputs")
    ap.add_argument("--lc_radius", type=float, default=None)
    ap.add_argument("--lc_min_gap", type=int, default=None)
    ap.add_argument("--lc_min_score", type=float, default=None)
    ap.add_argument("--kf_stride", type=int, default=None)
    ap.add_argument("--seed_half", type=int, default=10)
    ap.add_argument("--odom_sig_xy", type=float, default=None)
    ap.add_argument("--odom_sig_th_deg", type=float, default=None)
    ap.add_argument("--lc_sig_xy", type=float, default=None)
    ap.add_argument("--lc_sig_th_deg", type=float, default=None)
    ap.add_argument("--lc_max_per_query", type=int, default=1)
    ap.add_argument("--lc_consistency_max_trans", type=float, default=None)
    ap.add_argument("--lc_consistency_max_rot_deg", type=float, default=None)
    ap.add_argument("--lc_max_measurement_trans", type=float, default=None)
    ap.add_argument("--lc_max_measurement_rot_deg", type=float, default=None)
    ap.add_argument("--pgo_iters", type=int, default=None)
    ap.add_argument("--pgo_damping", type=float, default=None)
    ap.add_argument("--map_res", type=float, default=None)
    ap.add_argument("--map_size_m", type=float, default=None)
    ap.add_argument("--ray_steps", type=int, default=None)
    ap.add_argument("--no_filter", action="store_true")
    ap.add_argument(
        "--relations",
        default=None,
        help="Optional Freiburg-style .relations file used only for evaluation metrics.",
    )
    ap.add_argument(
        "--use_relations_as_constraints",
        action="store_true",
        help="Benchmark mode: add --relations as trusted graph constraints.",
    )
    ap.add_argument(
        "--relations_only",
        action="store_true",
        help="Benchmark mode: use only --relations constraints and skip automatic loop detection.",
    )
    ap.add_argument(
        "--relations_max_time_error",
        type=float,
        default=0.25,
        help="Max timestamp association error in seconds for --relations evaluation/benchmark constraints.",
    )
    ap.add_argument(
        "--relations_min_gap",
        type=int,
        default=1,
        help="Minimum index separation when --use_relations_as_constraints is enabled.",
    )
    ap.add_argument(
        "--global_scan_context",
        action="store_true",
        help="Add automatic drift-independent scan-context loop candidates.",
    )
    ap.add_argument("--sc_lc_min_score", type=float, default=0.55)
    ap.add_argument("--sc_descriptor_min_score", type=float, default=0.35)
    ap.add_argument("--sc_top_k", type=int, default=5)
    ap.add_argument("--sc_min_gap", type=int, default=None)
    ap.add_argument("--sc_map_size_m", type=float, default=16.0)
    ap.add_argument("--sc_coarse_xy_window", type=float, default=1.5)
    ap.add_argument("--sc_coarse_xy_step", type=float, default=0.15)
    ap.add_argument("--sc_coarse_th_step_deg", type=float, default=10.0)
    ap.add_argument("--sc_max_match_xy", type=float, default=1.2)
    ap.add_argument("--sc_max_abs_yaw_deg", type=float, default=None)
    ap.add_argument("--no_sc_reciprocal_check", action="store_true")
    ap.add_argument("--sc_reciprocal_xy_tol", type=float, default=0.5)
    ap.add_argument("--sc_reciprocal_yaw_tol_deg", type=float, default=20.0)
    args = ap.parse_args()

    if args.relations_only:
        args.use_relations_as_constraints = True
        if args.relations is None:
            ap.error("--relations_only requires --relations")

    traj_path = args.traj
    if traj_path is None:
        traj_path = resolve_latest_local_traj(
            out_dir=args.hector_out,
            dataset_name=args.dataset,
            scan_variant=args.scan_variant,
            matcher_type=args.matcher,
        )
        if traj_path is None:
            ap.error("No matching local trajectory found. Pass --traj explicitly.")

    ctx = parse_trajectory_context(traj_path)
    dataset_name = args.dataset or ctx["dataset_name"]
    scan_variant = args.scan_variant or ctx["scan_variant"]
    matcher_type = args.matcher or ctx["matcher_type"]

    configure_dataset(dataset_name)
    map_res = args.map_res if args.map_res is not None else cfg.MAP_RESOLUTION
    map_size_m = args.map_size_m if args.map_size_m is not None else default_map_size_m(dataset_name, matcher_type)
    ray_steps = args.ray_steps if args.ray_steps is not None else cfg.RAY_STEPS
    odom_sig_xy = args.odom_sig_xy if args.odom_sig_xy is not None else cfg.ODOM_SIGMA_XY
    odom_sig_th = np.deg2rad(args.odom_sig_th_deg) if args.odom_sig_th_deg is not None else cfg.ODOM_SIGMA_TH
    pgo_iters = args.pgo_iters if args.pgo_iters is not None else cfg.PGO_ITERS
    pgo_damping = args.pgo_damping if args.pgo_damping is not None else cfg.PGO_DAMPING
    kf_stride = args.kf_stride if args.kf_stride is not None else cfg.KEYFRAME_STRIDE
    if dataset_name == "lab_run_2":
        lc_radius = 1.2 if args.lc_radius is None else args.lc_radius
        lc_min_gap = 120 if args.lc_min_gap is None else args.lc_min_gap
        lc_sig_xy = 0.30 if args.lc_sig_xy is None else args.lc_sig_xy
        lc_sig_th = np.deg2rad(12.0 if args.lc_sig_th_deg is None else args.lc_sig_th_deg)
        lc_consistency_max_trans = (
            0.50 if args.lc_consistency_max_trans is None else args.lc_consistency_max_trans
        )
        lc_consistency_max_rot = np.deg2rad(
            12.0 if args.lc_consistency_max_rot_deg is None else args.lc_consistency_max_rot_deg
        )
        if args.lc_max_measurement_trans is None:
            args.lc_max_measurement_trans = 0.80
        if args.lc_max_measurement_rot_deg is None:
            args.lc_max_measurement_rot_deg = 15.0
    else:
        lc_radius = 2.0 if args.lc_radius is None else args.lc_radius
        lc_min_gap = 80 if args.lc_min_gap is None else args.lc_min_gap
        lc_sig_xy = 0.03 if args.lc_sig_xy is None else args.lc_sig_xy
        lc_sig_th = np.deg2rad(2.0 if args.lc_sig_th_deg is None else args.lc_sig_th_deg)
        lc_consistency_max_trans = (
            0.75 if args.lc_consistency_max_trans is None else args.lc_consistency_max_trans
        )
        lc_consistency_max_rot = np.deg2rad(
            20.0 if args.lc_consistency_max_rot_deg is None else args.lc_consistency_max_rot_deg
        )
    lc_min_score = args.lc_min_score
    if lc_min_score is None:
        lc_min_score = 0.60 if matcher_type == "scan_to_submap" else max(0.65, cfg.CORR_MAP_MIN_SCORE)

    print(f"[pgo_any] Trajectory: {traj_path}")
    print(f"[pgo_any] Dataset={dataset_name}  variant={scan_variant}  matcher={matcher_type}")

    stamps, poses_before = load_trajectory_full(traj_path)
    print(f"[pgo_any] Loaded {len(poses_before)} accepted poses")

    print("[pgo_any] Loading and preprocessing scans ...")
    _, scan_pts_list = load_aligned_scan_points(
        dataset_name=dataset_name,
        scan_variant=scan_variant,
        stamps=stamps,
        voxel_filter=(False if args.no_filter else None),
    )
    print(f"[pgo_any] Scans ready: {len(scan_pts_list)}  mean_pts={np.mean([p.shape[0] for p in scan_pts_list]):.0f}")

    closures: List[LoopClosure] = []

    if not args.relations_only:
        print(
            f"[pgo_any] Loop closure search: N={len(poses_before)}  stride={kf_stride}  "
            f"radius={lc_radius}m  min_gap={lc_min_gap}"
        )
        if matcher_type == "scan_to_submap":
            closures.extend(detect_loop_closures_submap(
                poses=poses_before,
                scan_pts_list=scan_pts_list,
                search_radius=lc_radius,
                min_index_gap=lc_min_gap,
                lc_min_score=lc_min_score,
                keyframe_stride=kf_stride,
                seed_half=args.seed_half,
                map_res=map_res,
                map_size_m=min(map_size_m, cfg.SUBMAP_SIZE_METERS),
                l_free=cfg.L_FREE,
                l_occ=cfg.L_OCC,
                ray_steps=ray_steps,
                max_per_query=args.lc_max_per_query,
                consistency_max_trans=lc_consistency_max_trans,
                consistency_max_rot=lc_consistency_max_rot,
                max_measurement_trans=args.lc_max_measurement_trans,
                max_measurement_rot=(
                    None if args.lc_max_measurement_rot_deg is None
                    else np.deg2rad(args.lc_max_measurement_rot_deg)
                ),
            ))
        else:
            closures.extend(detect_loop_closures_map(
                poses=poses_before,
                scan_pts_list=scan_pts_list,
                search_radius=lc_radius,
                min_index_gap=lc_min_gap,
                lc_min_score=lc_min_score,
                keyframe_stride=kf_stride,
                seed_half=args.seed_half,
                map_res=map_res,
                map_size_m=min(map_size_m, 20.0),
                l_free=cfg.L_FREE,
                l_occ=cfg.L_OCC,
                ray_steps=ray_steps,
                max_per_query=args.lc_max_per_query,
                consistency_max_trans=lc_consistency_max_trans,
                consistency_max_rot=lc_consistency_max_rot,
                max_measurement_trans=args.lc_max_measurement_trans,
                max_measurement_rot=(
                    None if args.lc_max_measurement_rot_deg is None
                    else np.deg2rad(args.lc_max_measurement_rot_deg)
                ),
            ))

        if args.global_scan_context:
            closures.extend(detect_loop_closures_scan_context(
                poses=poses_before,
                scan_pts_list=scan_pts_list,
                min_index_gap=(args.sc_min_gap if args.sc_min_gap is not None else lc_min_gap),
                keyframe_stride=kf_stride,
                seed_half=args.seed_half,
                map_res=map_res,
                map_size_m=args.sc_map_size_m,
                l_free=cfg.L_FREE,
                l_occ=cfg.L_OCC,
                ray_steps=ray_steps,
                descriptor_min_score=args.sc_descriptor_min_score,
                lc_min_score=args.sc_lc_min_score,
                top_k=args.sc_top_k,
                coarse_xy_window=args.sc_coarse_xy_window,
                coarse_xy_step=args.sc_coarse_xy_step,
                coarse_th_step=np.deg2rad(args.sc_coarse_th_step_deg),
                max_match_xy=args.sc_max_match_xy,
                max_abs_yaw=(
                    None if args.sc_max_abs_yaw_deg is None
                    else np.deg2rad(args.sc_max_abs_yaw_deg)
                ),
                reciprocal_check=(not args.no_sc_reciprocal_check),
                reciprocal_xy_tol=args.sc_reciprocal_xy_tol,
                reciprocal_yaw_tol=np.deg2rad(args.sc_reciprocal_yaw_tol_deg),
            ))

    relation_list: Optional[List[Relation2D]] = None
    if args.relations is not None:
        relation_list = load_relations(args.relations)
        before_rel = relation_residual_summary(
            poses_before,
            stamps,
            relation_list,
            max_time_error_s=args.relations_max_time_error,
        )
        print(
            "[pgo_any] Relations residual before: "
            f"used={before_rel['used']}  "
            f"trans_rmse={before_rel['rmse_trans_m']:.3f}m  "
            f"rot_rmse={before_rel['rmse_rot_deg']:.2f}deg"
        )
        if args.use_relations_as_constraints:
            relation_closures, relation_stats = relation_closures_from_file(
                args.relations,
                stamps,
                max_time_error_s=args.relations_max_time_error,
                min_index_gap=args.relations_min_gap,
            )
            closures.extend(relation_closures)
            print(
                "[pgo_any] BENCHMARK: using relations as graph constraints: "
                f"used={relation_stats['relations_used']} / {relation_stats['relations_total']}  "
                f"skipped_time={relation_stats['relations_skipped_time']}  "
                f"skipped_gap={relation_stats['relations_skipped_gap']}"
            )
        else:
            print("[pgo_any] Relations are evaluation-only; no relation edges were added.")

    by_source: dict[str, int] = {}
    for lc in closures:
        by_source[lc.source] = by_source.get(lc.source, 0) + 1
    print(f"[pgo_any] Loop closures found: {len(closures)}")
    if by_source:
        print("[pgo_any] Loop closure sources: " + ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())))

    poses_after = build_and_optimize(
        poses=poses_before,
        closures=closures,
        odom_sig_xy=odom_sig_xy,
        odom_sig_th=odom_sig_th,
        lc_sig_xy=lc_sig_xy,
        lc_sig_th=lc_sig_th,
        pgo_iters=pgo_iters,
        pgo_damping=pgo_damping,
    )

    delta_xy = np.linalg.norm(poses_after[:, :2] - poses_before[:, :2], axis=1)
    print(
        f"[pgo_any] PGO pose correction: max={delta_xy.max():.4f}m  "
        f"mean={delta_xy.mean():.4f}m  std={delta_xy.std():.4f}m"
    )
    if relation_list is not None:
        after_rel = relation_residual_summary(
            poses_after,
            stamps,
            relation_list,
            max_time_error_s=args.relations_max_time_error,
        )
        print(
            "[pgo_any] Relations residual after : "
            f"used={after_rel['used']}  "
            f"trans_rmse={after_rel['rmse_trans_m']:.3f}m  "
            f"rot_rmse={after_rel['rmse_rot_deg']:.2f}deg"
        )

    ensure_dir(args.out_dir)
    stem = Path(traj_path).stem
    pgo_traj_path = os.path.join(args.out_dir, f"{stem}_pgo.txt")
    with open(pgo_traj_path, "w") as f:
        f.write("# timestamp x y theta [pgo-corrected]\n")
        for t, p in zip(stamps, poses_after):
            f.write(f"{t:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    print(f"[pgo_any] PGO trajectory saved: {pgo_traj_path}")

    print("[pgo_any] Rebuilding maps ...")
    grid_after = rebuild_map_from_poses(
        matcher_type=matcher_type,
        poses_opt=poses_after,
        scan_pts_list=scan_pts_list,
        map_res=map_res,
        map_size_m=map_size_m,
        l_free=cfg.L_FREE,
        l_occ=cfg.L_OCC,
        ray_steps=ray_steps,
    )
    grid_before = rebuild_map_from_poses(
        matcher_type=matcher_type,
        poses_opt=poses_before,
        scan_pts_list=scan_pts_list,
        map_res=map_res,
        map_size_m=map_size_m,
        l_free=cfg.L_FREE,
        l_occ=cfg.L_OCC,
        ray_steps=ray_steps,
    )
    prob_after = _grid_prob(grid_after)
    prob_before = _grid_prob(grid_before)

    npy_path = os.path.join(args.out_dir, f"map_{stem}_pgo.npy")
    np.save(npy_path, prob_after)
    print(f"[pgo_any] PGO map .npy saved: {npy_path}")

    png_path = os.path.join(args.out_dir, f"map_{stem}_pgo.png")
    save_map_png(
        prob_before=prob_before,
        prob_after=prob_after,
        traj_xy_before=poses_before[:, :2],
        traj_xy_after=poses_after[:, :2],
        grid_before=grid_before,
        grid_after=grid_after,
        out_path=png_path,
        title=f"Hector PGO — {dataset_tag(dataset_name, scan_variant)} — {matcher_type} ({len(closures)} loop closures)",
    )

    occ = int((prob_after > 0.65).sum())
    free = int((prob_after < 0.35).sum())
    unk = int(prob_after.size) - occ - free
    print(f"[pgo_any] PGO map stats: {occ} occupied, {free} free, {unk} unknown")
    print("[pgo_any] Done.")


if __name__ == "__main__":
    main()
