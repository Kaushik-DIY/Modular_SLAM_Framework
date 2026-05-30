"""
ICPLoopVerifier — cross-modal loop verification with small_gicp.

RTAB_inspired_implementation_plan.md §10.1. Implements the ``LoopVerifier``
Protocol from ``slam_core/loop_closure.py``. Used in Mode C: ORB-SLAM proposes
loop candidates, this verifier confirms them by aligning the candidate's 2D
LiDAR scan against the target's scan with GICP.

small_gicp operates on 3D clouds, so the 2D scans are promoted to z=0. The
alignment returns ``T_target_source`` (maps the query/source scan into the
target frame); the query node's corrected global pose is then
``T_target_world ∘ T_target_source``.

The target's scan is read from ``ClosureTarget.map_view`` (an ndarray, or any
object exposing ``.scan`` / ``.scan_points``).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from slam_core.common.types import Pose2
from slam_core.common.se2 import pose_compose, pose_inverse, wrap_angle
from slam_core.loop_closure import ClosureTarget, LoopMatchResult, LoopNode

import small_gicp


def _se2_to_mat(p: Pose2) -> np.ndarray:
    c, s = math.cos(p.theta), math.sin(p.theta)
    return np.array(
        [[c, -s, 0.0, p.x],
         [s, c, 0.0, p.y],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _mat_to_se2(T: np.ndarray) -> Pose2:
    return Pose2(float(T[0, 3]), float(T[1, 3]),
                 float(wrap_angle(math.atan2(T[1, 0], T[0, 0]))))


def _promote_3d(scan_xy: np.ndarray) -> np.ndarray:
    scan_xy = np.asarray(scan_xy, dtype=np.float64)
    return np.column_stack([scan_xy[:, 0], scan_xy[:, 1], np.zeros(len(scan_xy))])


def _extract_scan(obj) -> Optional[np.ndarray]:
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj
    for attr in ("scan", "scan_points"):
        val = getattr(obj, attr, None)
        if val is not None:
            return np.asarray(val)
    return None


class ICPLoopVerifier:
    def __init__(
        self,
        max_correspondence_distance: float = 0.5,
        fitness_threshold: float = 0.6,
        inlier_rmse_threshold: float = 0.10,
        downsampling_resolution: float = 0.05,
        registration_type: str = "GICP",
        max_iterations: int = 30,
        min_points: int = 10,
    ):
        self.max_corr = float(max_correspondence_distance)
        self.fitness_threshold = float(fitness_threshold)
        self.inlier_rmse_threshold = float(inlier_rmse_threshold)
        self.downsampling_resolution = float(downsampling_resolution)
        self.registration_type = registration_type
        self.max_iterations = int(max_iterations)
        self.min_points = int(min_points)

    def verify(self, node: LoopNode, target: ClosureTarget) -> LoopMatchResult:
        query_scan = _extract_scan(node.scan_points)
        target_scan = _extract_scan(target.map_view)

        if (query_scan is None or target_scan is None
                or len(query_scan) < self.min_points
                or len(target_scan) < self.min_points):
            return LoopMatchResult(False, 0.0, None, status="matcher_failed")

        src = _promote_3d(query_scan)      # query == source
        tgt = _promote_3d(target_scan)

        # init guess: query expressed in target frame = T_target^{-1} · T_query
        init_rel = pose_compose(pose_inverse(target.pose_global), node.pose_guess_global)
        init_mat = _se2_to_mat(init_rel)

        try:
            res = small_gicp.align(
                tgt, src, init_mat,
                registration_type=self.registration_type,
                downsampling_resolution=self.downsampling_resolution,
                max_correspondence_distance=self.max_corr,
                max_iterations=self.max_iterations,
            )
        except Exception:
            return LoopMatchResult(False, 0.0, None, status="matcher_failed")

        if not res.converged:
            return LoopMatchResult(False, 0.0, None, status="matcher_failed")

        T_ts = np.asarray(res.T_target_source, dtype=np.float64)
        refined_rel = _mat_to_se2(T_ts)

        fitness, inlier_rmse = self._score(src, tgt, T_ts)
        score = float(fitness)

        if fitness < self.fitness_threshold or inlier_rmse > self.inlier_rmse_threshold:
            return LoopMatchResult(False, score, None, status="score_failed")

        matched_global = pose_compose(target.pose_global, refined_rel)
        trans_corr = math.hypot(refined_rel.x - init_rel.x, refined_rel.y - init_rel.y)
        rot_corr = abs(wrap_angle(refined_rel.theta - init_rel.theta))

        return LoopMatchResult(
            success=True,
            score=score,
            matched_node_pose_global=matched_global,
            status="accepted",
            translation_residual_m=float(trans_corr),
            rotation_residual_rad=float(rot_corr),
        )

    def _score(self, src: np.ndarray, tgt: np.ndarray, T_ts: np.ndarray):
        """Overlap fitness + inlier RMSE after applying T_target_source to source."""
        from scipy.spatial import cKDTree

        src_aligned = (T_ts[:3, :3] @ src.T).T + T_ts[:3, 3]
        tree = cKDTree(tgt)
        dist, _ = tree.query(src_aligned, k=1)
        inliers = dist <= self.max_corr
        fitness = float(np.mean(inliers))
        if not np.any(inliers):
            return fitness, float("inf")
        inlier_rmse = float(np.sqrt(np.mean(dist[inliers] ** 2)))
        return fitness, inlier_rmse
