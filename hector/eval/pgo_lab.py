"""
pgo_lab.py  —  Post-run Pose-Graph Optimisation for Hector SLAM (lab_run_2)
============================================================================

This script takes the *local* SLAM trajectory produced by
``hector.run_local_slam_new`` and applies a full pose-graph optimisation
pipeline:

  1. **Build pose graph** — every accepted scan is a node.  Sequential
     edges encode the local SLAM relative transforms (odometry-like).

  2. **Detect loop closures** — for each keyframe node we query all earlier
     nodes within a spatial radius.  Candidate pairs that are far enough
     apart in graph index (to avoid self-matching) are verified with a
     scan-to-map ICP match against a locally-built submatch map seeded
     with the candidate node's scan.  Pairs that pass the score threshold
     become loop-closure edges.

  3. **GN/LM optimisation** — the standard information-matrix formulation
     over all sequential + loop edges, anchored at node 0.

  4. **Map rebuild** — replays all scans into a fresh GridMap at the
     optimised poses and saves the PNG.

Usage
-----
    # After running the SLAM pipeline:
    python -m hector.eval.pgo_lab

    # With explicit paths:
    python -m hector.eval.pgo_lab \\
        --traj hector_outputs/trajectory_lab_run_2_raw_scan_to_map_1087.txt \\
        --variant raw \\
        --out_dir hector_outputs

Algorithm Notes
---------------
The loop-closure verifier uses a mini scan-to-map ICP:
  - Seeds a temporary GridMap with N_SEED_SCANS scans centred on the
    candidate node (scans [i-seed//2 .. i+seed//2]).
  - Runs GN matching of the query scan at the predicted loop pose.
  - Accepts the closure if score >= lc_min_score.

This avoids any external solver dependency (no PyCeres needed).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from slam_core.common.types import Pose2
from slam_core.common.se2 import wrap_angle, pose_inverse, pose_compose
from slam_core.matching.scan_to_map import (
    ScanToMapMatcher,
    _transform_points,
)
from carto.local_slam.range_to_points import ranges_to_points


# ─── SE(2) helpers ────────────────────────────────────────────────────────────

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


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_trajectory_full(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load trajectory file with columns: timestamp x y theta score ...
    Returns:
      stamps  (N,)   — timestamps of accepted poses
      poses   (N,3)  — [x, y, theta]
    """
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
                t, x, y, th, sc = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                if sc >= 0.0:  # skip bootstrap scans (score=-1)
                    stamps_list.append(t)
                    poses_list.append([x, y, wrap_angle(th)])
            except ValueError:
                continue
    if not stamps_list:
        raise RuntimeError(f"No accepted poses in {path}")
    return np.array(stamps_list, dtype=float), np.array(poses_list, dtype=float)


def load_scans_for_poses(
    stamps: np.ndarray,
    scan_variant: str,
) -> Tuple[object, List[dict]]:
    """
    Load dataset scans and align them to our pose timestamps.
    Returns (profile, aligned_scans) where aligned_scans[i] corresponds to stamps[i].
    """
    from slam_core.dataio.dataset_catalog import load_dataset_scans
    profile, all_scans = load_dataset_scans("lab_run_2", scan_variant=scan_variant)

    # Build a timestamp array for all scans
    scan_stamps = np.array([s["t"] for s in all_scans], dtype=float)

    aligned = []
    for t in stamps:
        idx = int(np.argmin(np.abs(scan_stamps - t)))
        aligned.append(all_scans[idx])

    return profile, aligned


# ─── Scan point extraction ─────────────────────────────────────────────────────

def scan_to_points(scan: dict, profile) -> np.ndarray:
    return ranges_to_points(
        scan["ranges"],
        profile.angle_min,
        profile.angle_inc,
        profile.range_min,
        profile.range_max,
    )


# ─── Loop closure verifier ─────────────────────────────────────────────────────

class LoopClosure(NamedTuple):
    i: int           # reference (anchor) node index
    j: int           # query node index
    z_ij: np.ndarray # relative transform [dx, dy, dtheta] : T_i^{-1} * T_j
    score: float


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
) -> "ScanToMapMatcher":
    """
    Seed a temporary ScanToMapMatcher with scans around `center_idx`.
    Returns the matcher after integrating seed scans.
    """
    lo = max(0, center_idx - seed_half)
    hi = min(len(poses) - 1, center_idx + seed_half)

    map_params = dict(
        base_res=map_res,
        size_m=map_size_m,
        num_levels=3,
        l0=0.0,
        l_min=-5.0,
        l_max=5.0,
        l_free=l_free,
        l_occ=l_occ,
        ray_steps=ray_steps,
        bootstrap_scans=1,
    )
    corr_params = dict(
        gn_iters_per_level=[20, 15, 10],
        gn_damping=1e-4,
        min_points=30,
        min_inliers_accept=40,
        min_score=0.50,
        step_clip_xy=0.20,
        step_clip_th=np.deg2rad(8.0),
        max_translation_jump=2.0,   # generous for LC search
        max_rotation_jump=np.deg2rad(45.0),
    )

    matcher = ScanToMapMatcher(map_params=map_params, corr_params=corr_params)

    for idx in range(lo, hi + 1):
        pose_arr = poses[idx].copy()
        pts = scan_pts_list[idx]
        if pts.shape[0] == 0:
            continue
        pts_world = _transform_points(pose_arr, pts)
        for grid in matcher.pyr.levels:
            grid.integrate_scan_simple(
                pose=pose_arr,
                pts_world=pts_world,
                l_free=l_free,
                l_occ=l_occ,
                ray_steps=ray_steps,
            )
    matcher.initialized = True
    matcher._is_initialized = True
    matcher._bootstrap_count = matcher._bootstrap_scans  # skip bootstrap

    return matcher


def detect_loop_closures(
    poses: np.ndarray,
    scan_pts_list: List[np.ndarray],
    search_radius: float = 2.0,
    min_index_gap: int = 100,
    lc_min_score: float = 0.65,
    keyframe_stride: int = 15,
    seed_half: int = 12,
    map_res: float = 0.05,
    map_size_m: float = 20.0,
    l_free: float = -0.1,
    l_occ: float = 1.0,
    ray_steps: int = 20,
) -> List[LoopClosure]:
    """
    For every keyframe j, find earlier candidates i where:
      - j - i >= min_index_gap
      - ||pos_j - pos_i|| <= search_radius
    Verify each candidate pair with scan-to-map ICP.
    """
    closures: List[LoopClosure] = []
    N = len(poses)
    xy = poses[:, :2]

    print(f"[pgo_lab] Loop closure search: {N} poses, stride={keyframe_stride}, "
          f"radius={search_radius}m, min_gap={min_index_gap}")

    kf_indices = list(range(0, N, keyframe_stride))

    for j in kf_indices:
        pos_j = xy[j]
        pose_j = poses[j]

        # Find spatial candidates
        dists = np.linalg.norm(xy[:max(0, j - min_index_gap)] - pos_j, axis=1)
        candidates = np.where(dists <= search_radius)[0]

        if len(candidates) == 0:
            continue

        # Take at most 3 closest candidates
        candidates = candidates[np.argsort(dists[candidates])[:3]]

        for i in int_arr(candidates):
            # Build a mini map seeded around node i
            mini_map = _build_mini_map(
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

            # Try to match query scan j into this mini map, starting from predicted loop pose
            pts_j = scan_pts_list[j]
            if pts_j.shape[0] < 30:
                continue
            predicted = _pose2_from_arr(poses[i])  # initialise at anchor pose

            result = mini_map.match(
                t=0.0,
                scan_points_local=pts_j,
                predicted_pose_world=predicted,
            )

            if result.success and result.score >= lc_min_score:
                # Compute relative transform in the original world frame
                matched_arr = _se2_arr(
                    result.pose_world.x,
                    result.pose_world.y,
                    result.pose_world.theta,
                )
                z_ij = _se2_between(poses[i], matched_arr)
                closures.append(LoopClosure(i=i, j=j, z_ij=z_ij, score=result.score))
                print(f"[pgo_lab]   LC accepted: ({i},{j})  score={result.score:.3f}  "
                      f"dxy={np.hypot(z_ij[0], z_ij[1]):.3f}m  "
                      f"dth={np.rad2deg(z_ij[2]):.1f}°")

    print(f"[pgo_lab] Loop closures found: {len(closures)}")
    return closures


def int_arr(a) -> List[int]:
    return [int(x) for x in a]


# ─── Pose graph optimisation ──────────────────────────────────────────────────

def build_and_optimize(
    poses: np.ndarray,
    closures: List[LoopClosure],
    odom_sig_xy: float = 0.05,
    odom_sig_th: float = 0.05,
    lc_sig_xy: float = 0.03,
    lc_sig_th: float = 0.03,
    pgo_iters: int = 30,
    pgo_damping: float = 1e-6,
) -> np.ndarray:
    """
    Sparse pose-graph GN/LM optimisation over sequential + loop edges.

    State vector: x = [x0, y0, th0,  x1, y1, th1, ... xN-1, yN-1, thN-1]
    First node is fixed (anchor).
    """
    N = len(poses)
    n_vars = 3 * N

    Omega_odom = np.diag([1.0/odom_sig_xy**2, 1.0/odom_sig_xy**2, 1.0/odom_sig_th**2])
    Omega_lc   = np.diag([1.0/lc_sig_xy**2,   1.0/lc_sig_xy**2,   1.0/lc_sig_th**2])

    def _Rth(th: float) -> np.ndarray:
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s], [s, c]], dtype=float)

    def _residual_jacobian(x: np.ndarray):
        """
        Compute stacked residuals r and Jacobian J for all edges.
        Returns (r, J) with shapes (m,) and (m, n_vars).
        """
        r_list, J_list = [], []

        def _pose(k: int) -> np.ndarray:
            return x[3*k: 3*k+3].copy()

        # Sequential edges
        edges = [(k, k+1, _se2_between(poses[k], poses[k+1]), Omega_odom)
                 for k in range(N - 1)]
        # Loop edges
        for lc in closures:
            edges.append((lc.i, lc.j, lc.z_ij, Omega_lc))

        for (i, j, z, Omega) in edges:
            xi = _pose(i)
            xj = _pose(j)

            # z_hat = T_i^{-1} * T_j
            dth_i = xi[2]
            ci, si = np.cos(dth_i), np.sin(dth_i)
            Rth_i_T = np.array([[ci, si], [-si, ci]])
            dp = xj[:2] - xi[:2]
            t_hat = Rth_i_T @ dp
            th_hat = wrap_angle(xj[2] - xi[2])
            z_hat = np.array([t_hat[0], t_hat[1], th_hat], dtype=float)

            # residual (3,)
            r_k = z_hat - z
            r_k[2] = wrap_angle(r_k[2])

            # Jacobian wrt xi (3×3) and xj (3×3)
            # d(t_hat)/d(xi_xy) = -Rth_i_T
            # d(t_hat)/d(xi_th) = d/dth_i [ Rth_i_T @ dp ]
            #                   = [[-si, ci], [-ci, -si]] @ dp (column vec)
            dRdth_dp = np.array([-si * dp[0] + ci * dp[1],
                                  -ci * dp[0] - si * dp[1]], dtype=float)
            jac_xi = np.zeros((3, 3), dtype=float)
            jac_xi[:2, :2] = -Rth_i_T
            jac_xi[:2,  2] = dRdth_dp
            jac_xi[2,   2] = -1.0

            jac_xj = np.zeros((3, 3), dtype=float)
            jac_xj[:2, :2] = Rth_i_T
            jac_xj[2,   2] = 1.0

            # Information-weighted residual and jacobian
            r_w = Omega @ r_k
            J_w_xi = Omega @ jac_xi
            J_w_xj = Omega @ jac_xj

            r_list.append(r_w)

            J_row = np.zeros((3, n_vars), dtype=float)
            J_row[:, 3*i: 3*i+3] = J_w_xi
            J_row[:, 3*j: 3*j+3] = J_w_xj
            J_list.append(J_row)

        return np.concatenate(r_list), np.vstack(J_list)

    # Initialise from local SLAM poses
    x = poses.reshape(-1).copy()

    print(f"[pgo_lab] Optimising {N} nodes, {N-1} seq edges, {len(closures)} LC edges")
    print(f"[pgo_lab] GN/LM: iters={pgo_iters}  damping={pgo_damping:.1e}")

    for iteration in range(pgo_iters):
        r, J = _residual_jacobian(x)

        # Normal equations H dx = -g, skip rows/cols for fixed node 0
        H = J.T @ J
        g = J.T @ r

        # Anchor node 0: remove first 3 DOF
        H_red = H[3:, 3:]
        g_red = g[3:]
        H_red += pgo_damping * np.eye(H_red.shape[0])

        try:
            dx_red = -np.linalg.solve(H_red, g_red)
        except np.linalg.LinAlgError:
            print(f"[pgo_lab] WARNING: singular H at iter {iteration}, stopping")
            break

        dx_full = np.concatenate([[0.0, 0.0, 0.0], dx_red])
        x = x + dx_full

        # Wrap angle components
        for k in range(N):
            x[3*k+2] = wrap_angle(x[3*k+2])

        step_norm = float(np.linalg.norm(dx_full))
        cost = float(0.5 * np.dot(r, r))
        if (iteration % 5 == 0) or (iteration == pgo_iters - 1):
            print(f"[pgo_lab]   iter {iteration:3d}: cost={cost:.4f}  |dx|={step_norm:.4e}")

        if step_norm < 1e-7:
            print(f"[pgo_lab]   Converged at iter {iteration}")
            break

    poses_opt = x.reshape(N, 3)
    return poses_opt


# ─── Map reconstruction ────────────────────────────────────────────────────────

def rebuild_map_from_poses(
    poses_opt: np.ndarray,
    scan_pts_list: List[np.ndarray],
    map_res: float = 0.05,
    map_size_m: float = 40.0,
    l_free: float = -0.1,
    l_occ: float = 1.0,
    ray_steps: int = 20,
):
    from slam_core.matching.scan_to_map import GridMap
    grid = GridMap(res=map_res, size_m=map_size_m, l_min=-5.0, l_max=5.0)

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


def save_map_png(
    prob_before: np.ndarray,
    prob_after: np.ndarray,
    traj_xy_before: np.ndarray,
    traj_xy_after: np.ndarray,
    grid_before,
    grid_after,
    out_path: str,
    title: str = "Lab Room — Hector SLAM + PGO",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("[pgo_lab] matplotlib not available — skipping PNG")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=120)
    fig.patch.set_facecolor("#1a1a2e")

    for ax, label, prob, traj_xy, grid in [
        (axes[0], "Before PGO (local SLAM only)", prob_before, traj_xy_before, grid_before),
        (axes[1], "After PGO (loop-corrected)",  prob_after,  traj_xy_after,  grid_after),
    ]:
        ax.set_facecolor("#1a1a2e")
        ax.imshow(np.flipud(prob), cmap="binary_r", vmin=0.2, vmax=0.8,
                  interpolation="nearest", origin="upper")

        if traj_xy.shape[0] > 0:
            gxy = grid.world_to_grid(traj_xy)
            px = gxy[:, 0]
            py = grid.size - 1 - gxy[:, 1]
            n = len(px)
            colours = cm.cool(np.linspace(0, 1, n))
            ax.scatter(px, py, c=colours, s=4, linewidths=0, zorder=3, alpha=0.85)
            ax.plot(px[0], py[0], "o", color="#00ff88", markersize=10, zorder=4, label="Start")
            ax.plot(px[-1], py[-1], "X", color="#ff4466", markersize=12, zorder=4, label="End")

        ax.set_title(label, color="white", fontsize=12, pad=8)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444455")
        ax.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e",
                  labelcolor="white", framealpha=0.8)

    plt.suptitle(title, color="white", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[pgo_lab] Map PNG saved: {out_path}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def _latest_traj(out_dir: str, variant: str) -> Optional[str]:
    p = Path(out_dir)
    # Search for both scan_to_map and scan_to_submap trajectories
    candidates = sorted(
        list(p.glob(f"trajectory_lab_run_2_{variant}_scan_to_map_*.txt")) +
        list(p.glob(f"trajectory_lab_run_2_{variant}_scan_to_submap_*.txt")),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        if "_debug" not in c.name and "_pgo" not in c.name:
            return str(c)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Post-run PGO + loop closure for Hector SLAM lab_run_2."
    )
    ap.add_argument("--traj", default=None)
    ap.add_argument("--variant", default="raw", choices=["raw", "360"])
    ap.add_argument("--hector_out", default="hector_outputs")
    ap.add_argument("--out_dir", default="hector_outputs")

    # Loop closure params
    ap.add_argument("--lc_radius", type=float, default=1.5,
                    help="Spatial radius (m) for loop candidate search (default: 1.5)")
    ap.add_argument("--lc_min_gap", type=int, default=80,
                    help="Min node index gap to consider as loop closure (default: 80)")
    ap.add_argument("--lc_min_score", type=float, default=0.65,
                    help="Min ICP score to accept a LC constraint (default: 0.65)")
    ap.add_argument("--kf_stride", type=int, default=15,
                    help="Only check every k-th node as LC query (default: 15)")
    ap.add_argument("--seed_half", type=int, default=10,
                    help="Half-window of scans used to seed LC mini-map (default: 10)")

    # PGO params
    ap.add_argument("--odom_sig_xy", type=float, default=0.05)
    ap.add_argument("--odom_sig_th_deg", type=float, default=3.0)
    ap.add_argument("--lc_sig_xy", type=float, default=0.03)
    ap.add_argument("--lc_sig_th_deg", type=float, default=2.0)
    ap.add_argument("--pgo_iters", type=int, default=30)
    ap.add_argument("--pgo_damping", type=float, default=1e-6)

    # Map params
    ap.add_argument("--map_res", type=float, default=0.05)
    ap.add_argument("--map_size_m", type=float, default=40.0)
    ap.add_argument("--ray_steps", type=int, default=20)

    args = ap.parse_args()

    # ── 1. Load trajectory ──────────────────────────────────────────────────
    traj_path = args.traj
    if traj_path is None:
        traj_path = _latest_traj(args.hector_out, args.variant)
        if traj_path is None:
            ap.error(f"No trajectory found in '{args.hector_out}'. Pass --traj explicitly.")
    print(f"[pgo_lab] Trajectory: {traj_path}")

    stamps, poses_before = load_trajectory_full(traj_path)
    N = len(poses_before)
    print(f"[pgo_lab] Loaded {N} accepted poses")

    # ── 2. Load aligned scans ───────────────────────────────────────────────
    print("[pgo_lab] Loading scans ...")
    profile, scans = load_scans_for_poses(stamps, args.variant)

    from slam_core.matching.preprocessing import PointCloudProcessor, PointCloudProcessorConfig
    proc = PointCloudProcessor(PointCloudProcessorConfig(
        fixed_voxel_size=0.03,
        adaptive_voxel_max_size=0.10,
        adaptive_min_num_points=200,
        adaptive_num_iterations=6,
        enabled=True,
    ))

    scan_pts_list: List[np.ndarray] = []
    for s in scans:
        pts_raw = ranges_to_points(
            s["ranges"], profile.angle_min, profile.angle_inc,
            profile.range_min, profile.range_max,
        )
        pts, _ = proc.process(pts_raw)
        scan_pts_list.append(pts)

    print(f"[pgo_lab] Scans loaded: {N}, mean pts/scan = "
          f"{np.mean([p.shape[0] for p in scan_pts_list]):.0f}")

    # ── 3. Detect loop closures ──────────────────────────────────────────────
    closures = detect_loop_closures(
        poses=poses_before,
        scan_pts_list=scan_pts_list,
        search_radius=args.lc_radius,
        min_index_gap=args.lc_min_gap,
        lc_min_score=args.lc_min_score,
        keyframe_stride=args.kf_stride,
        seed_half=args.seed_half,
        map_res=args.map_res,
        map_size_m=min(args.map_size_m, 20.0),   # LC maps are small local windows
        l_free=-0.1,
        l_occ=1.0,
        ray_steps=args.ray_steps,
    )

    # ── 4. Pose-graph optimisation ───────────────────────────────────────────
    poses_after = build_and_optimize(
        poses=poses_before,
        closures=closures,
        odom_sig_xy=args.odom_sig_xy,
        odom_sig_th=np.deg2rad(args.odom_sig_th_deg),
        lc_sig_xy=args.lc_sig_xy,
        lc_sig_th=np.deg2rad(args.lc_sig_th_deg),
        pgo_iters=args.pgo_iters,
        pgo_damping=args.pgo_damping,
    )

    # Stats
    delta_xy = np.linalg.norm(poses_after[:, :2] - poses_before[:, :2], axis=1)
    print(f"[pgo_lab] PGO pose correction: "
          f"max={delta_xy.max():.4f}m  mean={delta_xy.mean():.4f}m  "
          f"std={delta_xy.std():.4f}m")

    # ── 5. Save PGO trajectory ───────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    stem = Path(traj_path).stem
    pgo_traj_path = os.path.join(args.out_dir, f"{stem}_pgo.txt")

    with open(pgo_traj_path, "w") as f:
        f.write("# timestamp x y theta [pgo-corrected]\n")
        for t, p in zip(stamps, poses_after):
            f.write(f"{t:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    print(f"[pgo_lab] PGO trajectory saved: {pgo_traj_path}")

    # ── 6. Rebuild map from PGO poses ───────────────────────────────────────
    print("[pgo_lab] Rebuilding map from PGO poses ...")
    grid_after = rebuild_map_from_poses(
        poses_opt=poses_after,
        scan_pts_list=scan_pts_list,
        map_res=args.map_res,
        map_size_m=args.map_size_m,
        l_free=-0.1,
        l_occ=1.0,
        ray_steps=args.ray_steps,
    )

    prob_after = grid_after.prob().astype(np.float32)
    npy_path = os.path.join(args.out_dir, f"map_{stem}_pgo.npy")
    np.save(npy_path, prob_after)
    print(f"[pgo_lab] PGO map .npy saved: {npy_path}")

    # Also rebuild before-PGO map for comparison
    print("[pgo_lab] Rebuilding before-PGO map for comparison ...")
    grid_before = rebuild_map_from_poses(
        poses_opt=poses_before,
        scan_pts_list=scan_pts_list,
        map_res=args.map_res,
        map_size_m=args.map_size_m,
        l_free=-0.1,
        l_occ=1.0,
        ray_steps=args.ray_steps,
    )
    prob_before = grid_before.prob().astype(np.float32)

    png_path = os.path.join(args.out_dir, f"map_{stem}_pgo.png")
    save_map_png(
        prob_before=prob_before,
        prob_after=prob_after,
        traj_xy_before=poses_before[:, :2],
        traj_xy_after=poses_after[:, :2],
        grid_before=grid_before,
        grid_after=grid_after,
        out_path=png_path,
        title=f"Lab Room — Hector SLAM + PGO  ({len(closures)} loop closures)",
    )

    occ = (prob_after > 0.65).sum()
    free = (prob_after < 0.35).sum()
    total = prob_after.size
    print(f"[pgo_lab] PGO map stats: {occ} occupied, {free} free, "
          f"{total - occ - free} unknown  ({args.map_res}m res, {grid_after.size}x{grid_after.size})")
    print(f"[pgo_lab] Done! PGO corrected {len(closures)} loop(s).")


if __name__ == "__main__":
    main()
