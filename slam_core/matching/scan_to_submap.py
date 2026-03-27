from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from slam_core.common.types import Pose2
from slam_core.common.se2 import (
    wrap_angle,
    pose_compose,
    inverse_pose,
    transform_points_pose,
)
from slam_core.matching.core import ScanMatcherBase, MatchResult, BufferedScan

# Shared optimizer (already generic)
from slam_core.optimisers.gn_lm import GNLMConfig, GaussNewtonLM


# ============================================================
# Basic submap data container
# ============================================================

@dataclass
class Submap2D:
    id: int
    grid: "ProbabilityGrid"
    pose_world: Pose2
    num_inserted: int = 0
    finished: bool = False


# ============================================================
# Probability grid
# ============================================================

class ProbabilityGrid:
    def __init__(
        self,
        size_m: float,
        resolution: float,
        l0: float,
        l_occ: float,
        l_free: float,
        l_min: float,
        l_max: float,
    ):
        self.res = float(resolution)
        self.size_m = float(size_m)
        self.w = int(np.ceil(size_m / resolution))
        self.h = int(np.ceil(size_m / resolution))

        self.l0 = float(l0)
        self.l_occ = float(l_occ)
        self.l_free = float(l_free)
        self.l_min = float(l_min)
        self.l_max = float(l_max)

        self.L = np.full((self.h, self.w), self.l0, dtype=np.float32)

        half = 0.5 * self.size_m
        self.origin_world = np.array([-half, -half], dtype=float)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(np.floor((x - self.origin_world[0]) / self.res))
        gy = int(np.floor((y - self.origin_world[1]) / self.res))
        return gx, gy

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.w and 0 <= gy < self.h

    def update_cell(self, gx: int, gy: int, delta: float) -> None:
        if not self.in_bounds(gx, gy):
            return
        self.L[gy, gx] = np.clip(self.L[gy, gx] + delta, self.l_min, self.l_max)

    def probability(self) -> np.ndarray:
        # Keep output identical in meaning to old implementation
        return 1.0 / (1.0 + np.exp(-self.L))


# ============================================================
# Multi-resolution precomputation stack
# ============================================================

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


# ============================================================
# Correlative scan matcher
# ============================================================

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


# ============================================================
# Refinement problem
# ============================================================

def _transform_points_se2(pose: np.ndarray, pts: np.ndarray) -> np.ndarray:
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return (pts @ R.T) + np.array([x, y], dtype=float)


def _d_world_d_theta(theta: float, pts_local: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    dR = np.array([[-s, -c], [c, -s]], dtype=float)
    return pts_local @ dR.T


def _bilinear(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    x = xy[:, 0]
    y = xy[:, 1]
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    dx = x - x0
    dy = y - y0

    v00 = img[y0, x0]
    v10 = img[y0, x1]
    v01 = img[y1, x0]
    v11 = img[y1, x1]

    return (
        v00 * (1 - dx) * (1 - dy)
        + v10 * dx * (1 - dy)
        + v01 * (1 - dx) * dy
        + v11 * dx * dy
    )


def _bilinear_grad(img: np.ndarray, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = xy[:, 0]
    y = xy[:, 1]
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    dx = x - x0
    dy = y - y0

    v00 = img[y0, x0]
    v10 = img[y0, x1]
    v01 = img[y1, x0]
    v11 = img[y1, x1]

    dpdx = (v10 - v00) * (1 - dy) + (v11 - v01) * dy
    dpdy = (v01 - v00) * (1 - dx) + (v11 - v10) * dx
    return dpdx, dpdy


@dataclass
class CartoRefinementProblem:
    grid: ProbabilityGrid
    pts_local: np.ndarray
    pred_pose_sub: np.ndarray
    min_points: int = 20
    w_trans: float = 1.0
    w_rot: float = 1.0

    def compute_r_J(self, x: np.ndarray):
        self._last_inliers = 0

        pose = np.array(x, dtype=float).reshape(3)
        pose[2] = wrap_angle(pose[2])

        prob_img = self.grid.probability()

        pts_sub = _transform_points_se2(pose, self.pts_local)

        gx = (pts_sub[:, 0] - float(self.grid.origin_world[0])) / float(self.grid.res)
        gy = (pts_sub[:, 1] - float(self.grid.origin_world[1])) / float(self.grid.res)
        gxy = np.stack([gx, gy], axis=1)

        mask = (
            (gx >= 1.0)
            & (gx < float(self.grid.w - 2))
            & (gy >= 1.0)
            & (gy < float(self.grid.h - 2))
        )
        self._last_inliers = int(np.sum(mask))

        if self._last_inliers < int(self.min_points):
            return None, None

        gxy_use = gxy[mask]
        pts_use = self.pts_local[mask]

        p = _bilinear(prob_img, gxy_use)
        r_map = (1.0 - p).astype(float)

        dpdx_img, dpdy_img = _bilinear_grad(prob_img, gxy_use)

        dpdx_m = dpdx_img / float(self.grid.res)
        dpdy_m = dpdy_img / float(self.grid.res)
        dP_dsub = np.stack([dpdx_m, dpdy_m], axis=1)

        dPs_dth = _d_world_d_theta(pose[2], pts_use)

        J_map = np.zeros((gxy_use.shape[0], 3), dtype=float)
        J_map[:, 0] = -dP_dsub[:, 0]
        J_map[:, 1] = -dP_dsub[:, 1]
        J_map[:, 2] = -(dP_dsub * dPs_dth).sum(axis=1)

        xp = np.array(self.pred_pose_sub, dtype=float).reshape(3)
        rp = np.array(
            [
                np.sqrt(self.w_trans) * (pose[0] - xp[0]),
                np.sqrt(self.w_trans) * (pose[1] - xp[1]),
                np.sqrt(self.w_rot) * wrap_angle(pose[2] - xp[2]),
            ],
            dtype=float,
        )

        Jp = np.array(
            [
                [np.sqrt(self.w_trans), 0.0, 0.0],
                [0.0, np.sqrt(self.w_trans), 0.0],
                [0.0, 0.0, np.sqrt(self.w_rot)],
            ],
            dtype=float,
        )

        r = np.concatenate([r_map, rp], axis=0)
        J = np.vstack([J_map, Jp])

        return r, J


# ============================================================
# Submap builder
# ============================================================

class SubmapBuilder2D:
    """
    Cartographer-style:
      - keep 2 active submaps
      - insert each accepted scan into both
    """

    def __init__(
        self,
        submap_size_m: float,
        resolution: float,
        scans_per_submap: int,
        ray_steps: int,
        l0: float,
        l_occ: float,
        l_free: float,
        l_min: float,
        l_max: float,
    ):
        self.submap_size_m = float(submap_size_m)
        self.resolution = float(resolution)
        self.scans_per_submap = int(scans_per_submap)
        self.ray_steps = int(ray_steps)

        self._grid_params = dict(
            l0=l0,
            l_occ=l_occ,
            l_free=l_free,
            l_min=l_min,
            l_max=l_max,
        )
        self._next_id = 0
        self.active: List[Submap2D] = []
        self._initialized = False

    @staticmethod
    def _bresenham(gx0: int, gy0: int, gx1: int, gy1: int):
        points = []
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

    def _new_submap(self, pose_world: Pose2) -> Submap2D:
        grid = ProbabilityGrid(self.submap_size_m, self.resolution, **self._grid_params)
        sm = Submap2D(id=self._next_id, grid=grid, pose_world=pose_world)
        self._next_id += 1
        return sm

    def _ensure_two_active(self, pose_world: Pose2) -> None:
        while len(self.active) < 2:
            self.active.append(self._new_submap(pose_world))

    def _maybe_finish_oldest(self, pose_world: Pose2) -> None:
        while self.active and self.active[0].num_inserted >= self.scans_per_submap:
            self.active[0].finished = True
            self.active.pop(0)
            self.active.append(self._new_submap(pose_world))

    def insert_scan(self, pose_world: Pose2, scan_points_local: np.ndarray) -> bool:
        if not self._initialized:
            self._ensure_two_active(pose_world)
            self._initialized = True

        endpoints_world = transform_points_pose(pose_world, scan_points_local)

        for sm in self.active:
            T_ws_inv = inverse_pose(sm.pose_world)

            origin_sub = transform_points_pose(
                T_ws_inv,
                np.array([[pose_world.x, pose_world.y]], dtype=float),
            )[0]
            endpoints_sub = transform_points_pose(T_ws_inv, endpoints_world)

            self._integrate_submap_frame(sm, origin_sub, endpoints_sub)
            sm.num_inserted += 1

        self._maybe_finish_oldest(pose_world)
        self._ensure_two_active(pose_world)

        return True

    def _integrate_submap_frame(
        self,
        sm: Submap2D,
        origin_sub: np.ndarray,
        endpoints_sub: np.ndarray,
    ) -> None:
        grid = sm.grid
        ox, oy = float(origin_sub[0]), float(origin_sub[1])
        gx0, gy0 = grid.world_to_grid(ox, oy)

        for ex, ey in endpoints_sub:
            gx1, gy1 = grid.world_to_grid(float(ex), float(ey))

            if not grid.in_bounds(gx0, gy0):
                continue
            if not grid.in_bounds(gx1, gy1):
                continue

            cells = self._bresenham(gx0, gy0, gx1, gy1)

            for gx, gy in cells[:-1]:
                grid.update_cell(gx, gy, grid.l_free)

            grid.update_cell(gx1, gy1, grid.l_occ)

    def get_active_submaps(self) -> List[Submap2D]:
        return list(self.active)

    def clear(self) -> None:
        self.active = []
        self._initialized = False
        self._next_id = 0


# ============================================================
# Core scan-to-submap matcher
# ============================================================

class ScanToSubmapMatcher(ScanMatcherBase):
    """
    Self-contained occupancy-grid scan-to-submap matcher.

    Owns:
      - submap builder
      - correlative scan matching
      - local refinement
    Does NOT own:
      - extrapolator
      - motion filter
      - pose graph
    """

    def __init__(
        self,
        submap_builder: SubmapBuilder2D,
        corr_params: dict,
    ):
        super().__init__(name="scan_to_submap")

        self.submap_builder = submap_builder
        self.corr_params = dict(corr_params)

        self._last_score: Optional[float] = None
        self._last_refine_delta: Optional[np.ndarray] = None
        self._last_refine_inliers: Optional[int] = None

        refine_iters = int(self.corr_params.get("refine_iters", 8))
        refine_damping = float(self.corr_params.get("refine_damping", 1e-3))
        refine_clip_xy = float(self.corr_params.get("refine_step_clip_xy", 0.10))
        refine_clip_th = float(self.corr_params.get("refine_step_clip_th", np.deg2rad(5.0)))

        self.refine_solver = GaussNewtonLM(
            GNLMConfig(
                iters=refine_iters,
                damping=refine_damping,
                eps_stop=float(self.corr_params.get("refine_eps_stop", 1e-6)),
                step_clip=np.array(
                    [refine_clip_xy, refine_clip_xy, refine_clip_th],
                    dtype=float,
                ),
                verbose=bool(self.corr_params.get("refine_verbose", False)),
            )
        )

    # --------------------------------------------------------
    # Initialization from rolling matched buffer
    # --------------------------------------------------------
    def initialize_from_buffer(self, scans: Sequence[BufferedScan]) -> None:
        self.submap_builder.clear()

        if len(scans) == 0:
            self._is_initialized = False
            return

        for item in scans:
            self.submap_builder.insert_scan(item.pose_world, item.scan_points_local)

        self._is_initialized = True

    # --------------------------------------------------------
    # Matching
    # --------------------------------------------------------
    def match(
        self,
        t: float,
        scan_points_local: np.ndarray,
        predicted_pose_world: Pose2,
        odom_pose_world: Optional[Pose2] = None,
    ) -> MatchResult:
        self._last_score = None
        self._last_refine_delta = None
        self._last_refine_inliers = None

        active = self.submap_builder.get_active_submaps()
        if len(active) == 0:
            fallback_pose = odom_pose_world if odom_pose_world is not None else predicted_pose_world
            return MatchResult(
                pose_world=fallback_pose,
                score=-1.0,
                success=False,
                method=self.name,
                refine_delta=None,
                inliers=None,
                debug_info={"reason": "no_active_submaps"},
            )

        target = active[-1]

        T_ws_inv = inverse_pose(target.pose_world)
        pred_sub = pose_compose(T_ws_inv, predicted_pose_world)

        pts_match = scan_points_local
        max_match_pts = int(self.corr_params.get("max_match_points", 60))
        if pts_match.shape[0] > max_match_pts:
            stride = max(1, pts_match.shape[0] // max_match_pts)
            pts_match = pts_match[::stride]

        prob_img = target.grid.probability()

        best_sub, best_score = correlative_match_two_stage(
            prob_img=prob_img,
            grid_origin_xy=target.grid.origin_world,
            res=target.grid.res,
            points_local=pts_match,
            initial_submap_pose=pred_sub,
            min_valid=int(self.corr_params.get("min_valid", 20)),
            precomp_levels=int(self.corr_params.get("precomp_levels", 3)),
            coarse_level=int(self.corr_params.get("coarse_level", 2)),
            coarse_xy_window=float(self.corr_params.get("coarse_xy_window", 0.8)),
            coarse_th_window=float(self.corr_params.get("coarse_th_window", 0.3)),
            coarse_xy_step=float(self.corr_params.get("coarse_xy_step", 0.20)),
            coarse_th_step=float(self.corr_params.get("coarse_th_step", 0.08)),
            fine_level=int(self.corr_params.get("fine_level", 0)),
            fine_xy_window=float(self.corr_params.get("fine_xy_window", 0.25)),
            fine_th_window=float(self.corr_params.get("fine_th_window", 0.12)),
            fine_xy_step=float(self.corr_params.get("fine_xy_step", 0.05)),
            fine_th_step=float(self.corr_params.get("fine_th_step", 0.02)),
        )

        self._last_score = float(best_score)

        min_score = float(self.corr_params.get("min_score", 0.52))
        if best_score < min_score:
            fallback_pose = odom_pose_world if odom_pose_world is not None else predicted_pose_world
            return MatchResult(
                pose_world=fallback_pose,
                score=float(best_score),
                success=False,
                method=self.name,
                refine_delta=None,
                inliers=None,
                debug_info={"reason": "score_below_threshold"},
            )

        refined_sub = best_sub

        do_refine = self.corr_params.get("do_refine", True)
        if isinstance(do_refine, str):
            do_refine = do_refine.lower() in ("1", "true", "yes", "y")
        do_refine = bool(do_refine)

        if do_refine:
            x0 = np.array([best_sub.x, best_sub.y, best_sub.theta], dtype=float)
            xpred = np.array([pred_sub.x, pred_sub.y, pred_sub.theta], dtype=float)

            refine_pts = scan_points_local
            max_refine_pts = int(self.corr_params.get("max_refine_points", 180))
            if refine_pts.shape[0] > max_refine_pts:
                stride = max(1, refine_pts.shape[0] // max_refine_pts)
                refine_pts = refine_pts[::stride]

            problem = CartoRefinementProblem(
                grid=target.grid,
                pts_local=refine_pts,
                pred_pose_sub=xpred,
                min_points=int(self.corr_params.get("refine_min_points", 20)),
                w_trans=float(self.corr_params.get("refine_w_trans", 1.0)),
                w_rot=float(self.corr_params.get("refine_w_rot", 1.0)),
            )

            x_opt = self.refine_solver.solve(x0, problem.compute_r_J).reshape(3)
            x_opt[2] = wrap_angle(x_opt[2])

            self._last_refine_delta = x_opt - x0
            self._last_refine_inliers = getattr(problem, "_last_inliers", None)

            refined_sub = Pose2(float(x_opt[0]), float(x_opt[1]), float(x_opt[2]))

        matched_world = pose_compose(target.pose_world, refined_sub)

        return MatchResult(
            pose_world=matched_world,
            score=float(self._last_score if self._last_score is not None else best_score),
            success=True,
            method=self.name,
            refine_delta=self._last_refine_delta,
            inliers=self._last_refine_inliers,
            debug_info={"target_submap_id": int(target.id)},
        )

    # --------------------------------------------------------
    # Target update
    # --------------------------------------------------------
    def update_target(
        self,
        pose_world: Pose2,
        scan_points_local: np.ndarray,
        t: Optional[float] = None,
    ) -> bool:
        return self.submap_builder.insert_scan(pose_world, scan_points_local)

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------
    def shutdown(self) -> None:
        self.submap_builder.clear()
        self._is_initialized = False

    def get_active_submaps(self):
        return self.submap_builder.get_active_submaps()

    # --------------------------------------------------------
    # Optional debug helpers
    # --------------------------------------------------------
    def last_score(self) -> Optional[float]:
        return self._last_score

    def last_refine_delta(self) -> Optional[np.ndarray]:
        return self._last_refine_delta

    def last_refine_inliers(self) -> Optional[int]:
        return self._last_refine_inliers