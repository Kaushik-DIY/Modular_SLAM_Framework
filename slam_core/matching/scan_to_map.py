from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, List, Tuple

import numpy as np

from slam_core.common.types import Pose2
from slam_core.common.se2 import wrap_angle
from slam_core.matching.core import ScanMatcherBase, MatchResult, BufferedScan


# ============================================================
# Generic SE(2) helpers for scan-to-map
# ============================================================

def _transform_points(pose: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """
    pts_xy: (N,2) in local frame -> (N,2) in world frame
    pose = [x, y, theta]
    """
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return pts_xy @ R.T + np.array([x, y], dtype=float)


def _d_world_d_theta(theta: float, pts_local: np.ndarray) -> np.ndarray:
    """
    d/dtheta of R(theta)*p, p in local frame. Returns (N,2).
    """
    c, s = np.cos(theta), np.sin(theta)
    dR = np.array([[-s, -c], [c, -s]], dtype=float)
    return pts_local @ dR.T


# ============================================================
# Hector-style GridMap
# ============================================================

@dataclass
class GridMap:
    """
    2D occupancy grid in log-odds form.
    World frame is meters.
    Grid coords are float (gx, gy).
    Storage is array [gy, gx].
    """
    res: float
    size_m: float
    l0: float = 0.0
    l_min: float = -5.0
    l_max: float = 5.0

    def __post_init__(self):
        self.size = int(np.ceil(self.size_m / self.res))
        if self.size % 2 == 1:
            self.size += 1

        # world (0,0) at grid center
        self.origin = np.array([self.size / 2.0, self.size / 2.0], dtype=float)
        self.logodds = np.full((self.size, self.size), self.l0, dtype=np.float32)

    # ---------------- coordinate transforms ----------------

    def world_to_grid(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        gx = xy[:, 0] / self.res + self.origin[0]
        gy = xy[:, 1] / self.res + self.origin[1]
        return np.stack([gx, gy], axis=1)

    def grid_to_world(self, gxy: np.ndarray) -> np.ndarray:
        gxy = np.asarray(gxy, dtype=float)
        x = (gxy[:, 0] - self.origin[0]) * self.res
        y = (gxy[:, 1] - self.origin[1]) * self.res
        return np.stack([x, y], axis=1)

    def in_bounds(self, gxy: np.ndarray) -> np.ndarray:
        # 1-cell margin for bilinear interpolation
        x = gxy[:, 0]
        y = gxy[:, 1]
        return (x >= 1.0) & (y >= 1.0) & (x < self.size - 2.0) & (y < self.size - 2.0)

    # ---------------- occupancy representation ----------------

    def prob(self) -> np.ndarray:
        l = self.logodds.astype(np.float32)
        return 1.0 / (1.0 + np.exp(-l))

    @staticmethod
    def _bilinear_sample(grid: np.ndarray, gxy: np.ndarray) -> np.ndarray:
        x = gxy[:, 0]
        y = gxy[:, 1]

        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)

        dx = x - x0
        dy = y - y0

        v00 = grid[y0, x0]
        v10 = grid[y0, x0 + 1]
        v01 = grid[y0 + 1, x0]
        v11 = grid[y0 + 1, x0 + 1]

        v0 = v00 * (1.0 - dx) + v10 * dx
        v1 = v01 * (1.0 - dx) + v11 * dx
        return v0 * (1.0 - dy) + v1 * dy

    def sample_prob(self, gxy: np.ndarray, prob_grid: np.ndarray | None = None) -> np.ndarray:
        if prob_grid is None:
            prob_grid = self.prob()
        return self._bilinear_sample(prob_grid, gxy)

    def gradients_prob(self) -> Tuple[np.ndarray, np.ndarray]:
        p = self.prob().astype(np.float32)

        gx = 0.5 * (p[:, 2:] - p[:, :-2])
        gx = np.pad(gx, ((0, 0), (1, 1)), mode="edge")

        gy = 0.5 * (p[2:, :] - p[:-2, :])
        gy = np.pad(gy, ((1, 1), (0, 0)), mode="edge")

        return gx, gy

    # ---------------- mapping update ----------------

    def add_logodds_at(self, gxy_int: np.ndarray, delta: float) -> None:
        x = gxy_int[:, 0].astype(np.int32)
        y = gxy_int[:, 1].astype(np.int32)

        m = (x >= 0) & (y >= 0) & (x < self.size) & (y < self.size)
        x, y = x[m], y[m]

        self.logodds[y, x] = np.clip(
            self.logodds[y, x] + delta,
            self.l_min,
            self.l_max,
        )

    def integrate_scan_simple(
        self,
        pose: np.ndarray,
        pts_world: np.ndarray,
        l_free: float,
        l_occ: float,
        ray_steps: int = 40,
    ) -> None:
        origin_w = np.array([[pose[0], pose[1]]], dtype=float)
        g0 = self.world_to_grid(origin_w)[0]
        x0, y0 = float(g0[0]), float(g0[1])

        g_end = self.world_to_grid(pts_world)
        mask = self.in_bounds(g_end)
        g_end = g_end[mask]

        if g_end.shape[0] == 0:
            return

        for i in range(g_end.shape[0]):
            x1, y1 = float(g_end[i, 0]), float(g_end[i, 1])
            xs = np.linspace(x0, x1, ray_steps, dtype=float)
            ys = np.linspace(y0, y1, ray_steps, dtype=float)

            xi = np.round(xs[:-1]).astype(int)
            yi = np.round(ys[:-1]).astype(int)
            pts = np.stack([xi, yi], axis=1)

            self.add_logodds_at(pts, l_free)

        xe = np.round(g_end[:, 0]).astype(int)
        ye = np.round(g_end[:, 1]).astype(int)
        end_pts = np.stack([xe, ye], axis=1)
        self.add_logodds_at(end_pts, l_occ)


# ============================================================
# Map pyramid
# ============================================================

@dataclass
class MapPyramid:
    levels: List[GridMap]  # coarse -> fine

    @staticmethod
    def create(
        base_res: float,
        size_m: float,
        num_levels: int,
        l0: float,
        l_min: float,
        l_max: float,
    ) -> "MapPyramid":
        grids = []
        for i in range(num_levels):
            res = base_res * (2 ** (num_levels - 1 - i))
            grids.append(
                GridMap(
                    res=res,
                    size_m=size_m,
                    l0=l0,
                    l_min=l_min,
                    l_max=l_max,
                )
            )
        return MapPyramid(levels=grids)

    def finest(self) -> GridMap:
        return self.levels[-1]

    def clear(self) -> None:
        for g in self.levels:
            g.logodds.fill(g.l0)


# ============================================================
# Hector-style Gauss-Newton scan-to-map alignment
# ============================================================

def align_pose_gauss_newton(
    grid: GridMap,
    init_pose: np.ndarray,
    pts_lidar: np.ndarray,
    iters: int = 10,
    damping: float = 1e-3,
    min_points: int = 20,
    step_clip_xy: float = 0.03,
    step_clip_th: float = np.deg2rad(1.0),
) -> Tuple[np.ndarray, dict]:
    pose = np.array(init_pose, dtype=float).reshape(3)
    pose[2] = wrap_angle(pose[2])

    prob_grid = grid.prob()
    grad_x, grad_y = grid.gradients_prob()

    last_valid = 0
    last_mean_prob = 0.0
    last_residual_mean = 1.0
    total_delta = np.zeros(3, dtype=float)

    for _ in range(iters):
        pts_w = _transform_points(pose, pts_lidar)
        gxy = grid.world_to_grid(pts_w)

        mask = grid.in_bounds(gxy)
        valid = int(mask.sum())
        last_valid = valid

        if valid < int(min_points):
            break

        gxy_use = gxy[mask]
        pts_use = pts_lidar[mask]

        m = grid.sample_prob(gxy_use, prob_grid)
        r = (1.0 - m).reshape(-1, 1)

        last_mean_prob = float(np.mean(m))
        last_residual_mean = float(np.mean(r))

        gx = GridMap._bilinear_sample(grad_x, gxy_use)
        gy = GridMap._bilinear_sample(grad_y, gxy_use)

        dM_dworld = np.stack([gx, gy], axis=1) / grid.res
        dPw_dth = _d_world_d_theta(pose[2], pts_use)

        J = np.zeros((gxy_use.shape[0], 3), dtype=float)
        J[:, 0] = -dM_dworld[:, 0]
        J[:, 1] = -dM_dworld[:, 1]
        J[:, 2] = -(dM_dworld * dPw_dth).sum(axis=1)

        H = (J.T @ J) + (damping * np.eye(3))
        g = (J.T @ r).reshape(3)

        try:
            delta = -np.linalg.solve(H, g)
            delta[0] = np.clip(delta[0], -step_clip_xy, step_clip_xy)
            delta[1] = np.clip(delta[1], -step_clip_xy, step_clip_xy)
            delta[2] = np.clip(delta[2], -step_clip_th, step_clip_th)
        except np.linalg.LinAlgError:
            break

        pose[0] += float(delta[0])
        pose[1] += float(delta[1])
        pose[2] = wrap_angle(pose[2] + float(delta[2]))

        total_delta += delta

        if float(np.linalg.norm(delta)) < 1e-6:
            break

    pose[2] = wrap_angle(pose[2])

    debug = {
        "valid_points": int(last_valid),
        "mean_prob": float(last_mean_prob),
        "mean_residual": float(last_residual_mean),
        "total_delta": total_delta.copy(),
    }

    return pose, debug


# ============================================================
# Unified matcher wrapper
# ============================================================

class ScanToMapMatcher(ScanMatcherBase):
    """
    Hector-style scan-to-map matcher with multi-resolution map pyramid.
    """

    def __init__(self, map_params: dict, corr_params: dict):
        super().__init__(name="scan_to_map")

        self.map_params = dict(map_params)
        self.corr_params = dict(corr_params)

        self.pyr = MapPyramid.create(
            base_res=float(self.map_params["base_res"]),
            size_m=float(self.map_params["size_m"]),
            num_levels=int(self.map_params["num_levels"]),
            l0=float(self.map_params["l0"]),
            l_min=float(self.map_params["l_min"]),
            l_max=float(self.map_params["l_max"]),
        )

        self.l_free = float(self.map_params["l_free"])
        self.l_occ = float(self.map_params["l_occ"])
        self.ray_steps = int(self.map_params["ray_steps"])

        self.pose = np.array([0.0, 0.0, 0.0], dtype=float)
        self.initialized = False

        self._last_score: Optional[float] = None
        self._last_inliers: Optional[int] = None
        self._last_delta: Optional[np.ndarray] = None
        self._last_debug_info: Optional[dict] = None

    def initialize_from_buffer(self, scans: Sequence[BufferedScan]) -> None:
        self.pyr = MapPyramid.create(
            base_res=float(self.map_params["base_res"]),
            size_m=float(self.map_params["size_m"]),
            num_levels=int(self.map_params["num_levels"]),
            l0=float(self.map_params["l0"]),
            l_min=float(self.map_params["l_min"]),
            l_max=float(self.map_params["l_max"]),
        )

        self.initialized = False
        self._last_score = None
        self._last_inliers = None
        self._last_delta = None
        self._last_debug_info = None

        if len(scans) == 0:
            self._is_initialized = False
            return

        for item in scans:
            pose_arr = np.array(
                [item.pose_world.x, item.pose_world.y, item.pose_world.theta],
                dtype=float,
            )
            pts_world = _transform_points(pose_arr, item.scan_points_local)

            for grid in self.pyr.levels:
                grid.integrate_scan_simple(
                    pose=pose_arr,
                    pts_world=pts_world,
                    l_free=self.l_free,
                    l_occ=self.l_occ,
                    ray_steps=self.ray_steps,
                )

            self.pose = pose_arr.copy()
            self.initialized = True

        self._is_initialized = True

    def match(
        self,
        t: float,
        scan_points_local: np.ndarray,
        predicted_pose_world: Pose2,
        odom_pose_world: Optional[Pose2] = None,
    ) -> MatchResult:
        pose_pred = np.array(
            [
                predicted_pose_world.x,
                predicted_pose_world.y,
                predicted_pose_world.theta,
            ],
            dtype=float,
        )
        pose_pred[2] = wrap_angle(pose_pred[2])

        self._last_score = None
        self._last_inliers = None
        self._last_delta = None
        self._last_debug_info = None

        if not self.initialized:
            fallback_pose = odom_pose_world if odom_pose_world is not None else predicted_pose_world
            self._last_score = -1.0
            self._last_inliers = 0
            self._last_delta = np.zeros(3, dtype=float)
            self._last_debug_info = {"reason": "map_not_initialized"}
            return MatchResult(
                pose_world=fallback_pose,
                score=-1.0,
                success=False,
                method=self.name,
                debug_info=self._last_debug_info,
            )

        # ----------------------------------------------------
        # True Hector-style coarse-to-fine scan-to-map matching
        # ----------------------------------------------------
        levels = sorted(self.pyr.levels, key=lambda g: g.res, reverse=True)

        gn_iters_per_level = self.corr_params.get("gn_iters_per_level", [15, 12, 10, 8])
        gn_damping = float(self.corr_params.get("gn_damping", 1e-3))
        min_points = int(self.corr_params.get("min_points", 20))
        step_clip_xy = float(self.corr_params.get("step_clip_xy", 0.02))
        step_clip_th = float(self.corr_params.get("step_clip_th", np.deg2rad(0.7)))

        min_score = float(self.corr_params.get("min_score", 0.45))
        min_inliers_accept = int(self.corr_params.get("min_inliers_accept", min_points))

        level_debug = []
        pose_est = pose_pred.copy()

        for lvl, grid in enumerate(levels):
            iters = int(gn_iters_per_level[min(lvl, len(gn_iters_per_level) - 1)])

            pose_before = pose_est.copy()

            pose_est, dbg = align_pose_gauss_newton(
                grid=grid,
                init_pose=pose_est,
                pts_lidar=scan_points_local,
                iters=iters,
                damping=gn_damping,
                min_points=min_points,
                step_clip_xy=step_clip_xy,
                step_clip_th=step_clip_th,
            )

            lvl_delta = pose_est - pose_before
            lvl_delta[2] = wrap_angle(lvl_delta[2])

            level_debug.append({
                "level": int(lvl),
                "res": float(grid.res),
                "valid_points": int(dbg["valid_points"]),
                "mean_prob": float(dbg["mean_prob"]),
                "mean_residual": float(dbg["mean_residual"]),
                "delta": np.array(lvl_delta, dtype=float),
            })

        pose_est[2] = wrap_angle(pose_est[2])
        pose_world = Pose2(float(pose_est[0]), float(pose_est[1]), float(pose_est[2]))

        delta = pose_est - pose_pred
        delta[2] = wrap_angle(delta[2])

        finest_grid = self.pyr.finest()
        pts_world = _transform_points(pose_est, scan_points_local)
        gxy = finest_grid.world_to_grid(pts_world)
        mask = finest_grid.in_bounds(gxy)

        if int(mask.sum()) >= min_points:
            prob_grid = finest_grid.prob()
            probs = finest_grid.sample_prob(gxy[mask], prob_grid)
            score = float(np.mean(probs))
            inliers = int(mask.sum())
        else:
            score = -1.0
            inliers = int(mask.sum())

        self._last_score = score
        self._last_inliers = inliers
        self._last_delta = delta.copy()

        success = (inliers >= min_inliers_accept) and (score >= min_score)

        self._last_debug_info = {
            "reason": "hector_scan_to_map" if success else "scan_to_map_rejected",
            "valid_points_finest": int(inliers),
            "delta": delta.copy(),
            "level_debug": level_debug,
            "score": float(score),
            "min_score": float(min_score),
            "min_inliers_accept": int(min_inliers_accept),
        }

        if not success:
            fallback_pose = odom_pose_world if odom_pose_world is not None else predicted_pose_world
            return MatchResult(
                pose_world=fallback_pose,
                score=float(score),
                success=False,
                method=self.name,
                refine_delta=delta.copy(),
                inliers=int(inliers),
                debug_info=self._last_debug_info,
            )

        return MatchResult(
            pose_world=pose_world,
            score=float(score),
            success=True,
            method=self.name,
            refine_delta=delta.copy(),
            inliers=int(inliers),
            debug_info=self._last_debug_info,
        )

    def update_target(
        self,
        pose_world: Pose2,
        scan_points_local: np.ndarray,
        t: Optional[float] = None,
    ) -> bool:
        pose_arr = np.array([pose_world.x, pose_world.y, pose_world.theta], dtype=float)
        pts_world = _transform_points(pose_arr, scan_points_local)

        for grid in self.pyr.levels:
            grid.integrate_scan_simple(
                pose=pose_arr,
                pts_world=pts_world,
                l_free=self.l_free,
                l_occ=self.l_occ,
                ray_steps=self.ray_steps,
            )

        self.pose = pose_arr.copy()
        self.initialized = True
        return True

    def shutdown(self) -> None:
        self.pyr.clear()
        self.initialized = False
        self._is_initialized = False
        self._last_score = None
        self._last_inliers = None
        self._last_delta = None
        self._last_debug_info = None

    def last_score(self) -> Optional[float]:
        return self._last_score

    def last_inliers(self) -> Optional[int]:
        return self._last_inliers

    def last_delta(self) -> Optional[np.ndarray]:
        return self._last_delta

    def last_debug_info(self) -> Optional[dict]:
        return self._last_debug_info