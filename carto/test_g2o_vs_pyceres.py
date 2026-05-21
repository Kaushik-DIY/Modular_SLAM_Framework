"""
Unit test: g2o SE2 pose-graph backend vs PyCeres backend.

Builds an identical square-loop pose graph (intra-submap constraints +
consecutive-node local-trajectory regularization + one loop closure) for each
backend through the real PoseGraph2D API, then asserts the optimized poses agree.

This proves G2oBackend2D is a drop-in replacement for PyCeresBackend2D and that
the loop closure actually corrects the injected drift.

Run: .venv/bin/python -m carto.test_g2o_vs_pyceres
"""
from __future__ import annotations

import numpy as np

from carto.common.types import Pose2
from carto.common.se2 import inverse_pose, pose_compose, wrap_angle
from carto.pose_graph.pose_graph_2d import PoseGraph2D
from carto.pose_graph.backends.pyceres_backend_2d import PyCeresBackend2D
from carto.pose_graph.backends.g2o_backend_2d import G2oBackend2D


class _SM:
    """Minimal submap stub exposing .id and .pose_world."""

    def __init__(self, sid: int, pose: Pose2):
        self.id = sid
        self.pose_world = pose


def _square_groundtruth(n_per_side: int = 6, side: float = 4.0):
    """Ground-truth poses around a closed square loop, heading tangent."""
    poses = []
    corners = [(0, 0), (side, 0), (side, side), (0, side)]
    headings = [0.0, np.pi / 2, np.pi, -np.pi / 2]
    for c in range(4):
        x0, y0 = corners[c]
        x1, y1 = corners[(c + 1) % 4]
        for s in range(n_per_side):
            f = s / float(n_per_side)
            x = x0 + (x1 - x0) * f
            y = y0 + (y1 - y0) * f
            poses.append(Pose2(x, y, headings[c]))
    return poses


def _build_graph(backend, gt_poses, drift_per_step=0.04, drift_ang=0.01, seed=0):
    """
    Build a pose graph on `backend` from ground-truth poses.

    - Node initial estimates accumulate odometry drift (so the loop does NOT
      close before optimization).
    - INTRA constraints use the (accurate) relative pose between consecutive
      ground-truth poses, anchored to a per-segment submap.
    - One loop-closure (INTER) constraint ties the last node back to submap 0
      using the accurate relative geometry.
    """
    pg = PoseGraph2D(backend=backend)

    n = len(gt_poses)

    # Single submap anchored at GT[0]. Using one submap keeps the local-trajectory
    # spine (local_pose is defined relative to the primary submap) fully GT-
    # consistent across the whole loop, so the unique optimum is exactly GT.
    submap = _SM(0, gt_poses[0])

    # Build from GT poses -> all relative measurements (intra + spine) consistent.
    for i in range(n):
        pg.add_node_with_intra_constraints(
            t=float(i),
            node_pose_world=gt_poses[i],
            active_submaps=[submap],
        )

    # Loop closure (INTER): last node vs submap 0, accurate geometry. Exercises
    # the inter-submap + Huber robust-kernel path.
    rel_loop = pose_compose(inverse_pose(submap.pose_world), gt_poses[-1])
    pg.add_loop_submap_node_constraint(
        submap_id=0,
        node_id=n - 1,
        relative_pose=rel_loop,
        translation_weight=1.1e4,
        rotation_weight=1e5,
        match_score=1.0,
    )

    # Inject odometry drift into the INITIAL ESTIMATES only (not the
    # measurements): the optimizer must converge back to GT from a drifted start.
    est = Pose2(gt_poses[0].x, gt_poses[0].y, gt_poses[0].theta)
    drifted = [est]
    for i in range(1, n):
        rel = pose_compose(inverse_pose(gt_poses[i - 1]), gt_poses[i])
        rel_drift = Pose2(rel.x + drift_per_step, rel.y, rel.theta + drift_ang)
        est = pose_compose(est, rel_drift)
        drifted.append(est)

    # Overwrite backend initial estimates with drift (anchor submap 0 held fixed).
    for i in range(n):
        backend.nodes[i] = drifted[i]

    return pg, drifted


def _node_array(backend, n):
    return np.array(
        [[backend.nodes[i].x, backend.nodes[i].y, backend.nodes[i].theta] for i in range(n)]
    )


def main() -> int:
    gt = _square_groundtruth()
    n = len(gt)
    gt_arr = np.array([[p.x, p.y, p.theta] for p in gt])

    # --- PyCeres ---
    pyc = PyCeresBackend2D(huber_scale=1e1, max_num_iterations=100)
    pyc.set_fixed("submap", 0)
    pg_pyc, drifted = _build_graph(pyc, gt)
    pg_pyc.solve(max_iters=100)
    pyc_arr = _node_array(pyc, n)

    # --- g2o ---
    g2 = G2oBackend2D(huber_scale=1e1, max_num_iterations=100)
    g2.set_fixed("submap", 0)
    pg_g2, _ = _build_graph(g2, gt)
    pg_g2.solve(max_iters=100)
    g2_arr = _node_array(g2, n)

    drift_arr = np.array([[p.x, p.y, p.theta] for p in drifted])

    def rmse(a, b):
        d = a[:, :2] - b[:, :2]
        return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))

    drift_rmse = rmse(drift_arr, gt_arr)
    pyc_rmse = rmse(pyc_arr, gt_arr)
    g2_rmse = rmse(g2_arr, gt_arr)

    # Agreement between the two solvers (the drop-in claim).
    diff = np.abs(g2_arr - pyc_arr)
    diff[:, 2] = np.abs((np.mod(g2_arr[:, 2] - pyc_arr[:, 2] + np.pi, 2 * np.pi)) - np.pi)
    max_xy = float(np.max(diff[:, :2]))
    max_th = float(np.max(diff[:, 2]))

    print(f"nodes={n}")
    print(f"drift  trajectory RMSE vs GT : {drift_rmse:.4f} m")
    print(f"pyceres trajectory RMSE vs GT: {pyc_rmse:.4f} m")
    print(f"g2o     trajectory RMSE vs GT: {g2_rmse:.4f} m")
    print(f"g2o-vs-pyceres  max |dxy|={max_xy:.6f} m  max |dtheta|={np.rad2deg(max_th):.6f} deg")

    ok = True
    # 1. Loop closure must reduce drift substantially for both solvers.
    if not (pyc_rmse < 0.5 * drift_rmse):
        print("FAIL: pyceres did not reduce drift")
        ok = False
    if not (g2_rmse < 0.5 * drift_rmse):
        print("FAIL: g2o did not reduce drift")
        ok = False
    # 2. The two solvers must agree (drop-in equivalence).
    if not (max_xy < 1e-2 and max_th < np.deg2rad(0.5)):
        print("FAIL: g2o and pyceres disagree beyond tolerance")
        ok = False

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
