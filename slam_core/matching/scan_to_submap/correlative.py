"""
Discrete correlative scan matching for scan-to-submap.

Canonical home for the two-stage (coarse multi-resolution + fine) brute-force
correlative matcher. Previously lived in ``scan_to_submap_old.py``; relocated
here so the package is the single source of truth for the scan-to-submap
front-end shared by both Hector and Cartographer pipelines.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from slam_core.common.types import Pose2
from slam_core.common.se2 import wrap_angle, transform_points_pose


def _max_pool_2x2(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    h2 = h // 2
    w2 = w // 2
    img = img[: 2 * h2, : 2 * w2]
    a = img[0::2, 0::2]
    b = img[0::2, 1::2]
    c = img[1::2, 0::2]
    d = img[1::2, 1::2]
    return np.maximum(np.maximum(a, b), np.maximum(c, d)).astype(np.float32)


class PrecomputationGridStack:
    """
    Multi-resolution probability stack for correlative scan matching.
    """

    def __init__(self, prob_img: np.ndarray, num_levels: int = 3):
        assert prob_img.ndim == 2
        self.levels = [prob_img.astype(np.float32)]
        for _ in range(1, int(num_levels)):
            self.levels.append(_max_pool_2x2(self.levels[-1]))

    def score_points_nearest(
        self,
        level: int,
        pts_g_level: np.ndarray,
        min_valid: int = 20,
    ) -> float:
        img = self.levels[level]
        h, w = img.shape

        gx = np.rint(pts_g_level[:, 0]).astype(np.int32)
        gy = np.rint(pts_g_level[:, 1]).astype(np.int32)

        mask = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        n = int(mask.sum())
        if n < min_valid:
            return -1e9

        return float(img[gy[mask], gx[mask]].mean())


def _score_candidate(
    stack: PrecomputationGridStack,
    level: int,
    grid_origin_xy: np.ndarray,
    res: float,
    points_local: np.ndarray,
    pose_sub: Pose2,
    min_valid: int,
) -> float:
    pts_sub = transform_points_pose(pose_sub, points_local)
    gx0 = (pts_sub[:, 0] - grid_origin_xy[0]) / res
    gy0 = (pts_sub[:, 1] - grid_origin_xy[1]) / res

    scale = 2 ** int(level)
    pts_g_level = np.stack([gx0 / scale, gy0 / scale], axis=1)
    return stack.score_points_nearest(int(level), pts_g_level, min_valid=min_valid)


def _bruteforce_search(
    stack: PrecomputationGridStack,
    level: int,
    grid_origin_xy: np.ndarray,
    res: float,
    points_local: np.ndarray,
    center_pose: Pose2,
    xy_window: float,
    th_window: float,
    xy_step: float,
    th_step: float,
    min_valid: int,
) -> Tuple[Pose2, float]:
    best_pose = center_pose
    best_score = -1e9

    xs = np.arange(-xy_window, xy_window + 1e-9, xy_step)
    ys = np.arange(-xy_window, xy_window + 1e-9, xy_step)
    ths = np.arange(-th_window, th_window + 1e-9, th_step)

    for dth in ths:
        th = wrap_angle(center_pose.theta + float(dth))
        for dx in xs:
            for dy in ys:
                pose = Pose2(
                    center_pose.x + float(dx),
                    center_pose.y + float(dy),
                    th,
                )
                s = _score_candidate(
                    stack,
                    level,
                    grid_origin_xy,
                    res,
                    points_local,
                    pose,
                    min_valid,
                )
                if s > best_score:
                    best_score = s
                    best_pose = pose

    return best_pose, best_score


def correlative_match_two_stage(
    prob_img: np.ndarray,
    grid_origin_xy: np.ndarray,
    res: float,
    points_local: np.ndarray,
    initial_submap_pose: Pose2,
    min_valid: int = 20,
    precomp_levels: int = 3,
    coarse_level: int = 2,
    coarse_xy_window: float = 0.8,
    coarse_th_window: float = 0.3,
    coarse_xy_step: float = 0.20,
    coarse_th_step: float = 0.08,
    fine_level: int = 0,
    fine_xy_window: float = 0.25,
    fine_th_window: float = 0.12,
    fine_xy_step: float = 0.05,
    fine_th_step: float = 0.02,
) -> Tuple[Pose2, float]:
    stack = PrecomputationGridStack(prob_img, num_levels=int(precomp_levels))

    coarse_pose, _ = _bruteforce_search(
        stack=stack,
        level=int(coarse_level),
        grid_origin_xy=grid_origin_xy,
        res=res,
        points_local=points_local,
        center_pose=initial_submap_pose,
        xy_window=coarse_xy_window,
        th_window=coarse_th_window,
        xy_step=coarse_xy_step,
        th_step=coarse_th_step,
        min_valid=min_valid,
    )

    fine_pose, fine_score = _bruteforce_search(
        stack=stack,
        level=int(fine_level),
        grid_origin_xy=grid_origin_xy,
        res=res,
        points_local=points_local,
        center_pose=coarse_pose,
        xy_window=fine_xy_window,
        th_window=fine_th_window,
        xy_step=fine_xy_step,
        th_step=fine_th_step,
        min_valid=min_valid,
    )

    return fine_pose, fine_score
