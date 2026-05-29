from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

from slam_core.common.types import Pose2
from slam_core.common.se2 import inverse_pose, transform_points_pose


class ProbabilityGrid:
    def __init__(self, size_m: float, res: float = None, l0=0.0, l_occ=0.85, l_free=-0.1, l_min=-5.0, l_max=5.0, resolution: float = None):
        # `resolution` is a backward-compatible alias for `res` (eval scripts
        # historically constructed ProbabilityGrid with resolution=...).
        if res is None:
            res = resolution
        if res is None:
            raise TypeError("ProbabilityGrid requires `res` (or `resolution`)")
        self.size_m = float(size_m)
        self.res = float(res)
        self.w = int(np.round(size_m / res))
        self.h = int(np.round(size_m / res))
        self.origin_world = np.array([-size_m / 2.0, -size_m / 2.0], dtype=float)

        self.l0 = float(l0)
        self.l_occ = float(l_occ)
        self.l_free = float(l_free)
        self.l_min = float(l_min)
        self.l_max = float(l_max)

        self.L = np.full((self.h, self.w), self.l0, dtype=np.float32)

    def world_to_grid(self, x: float, y: float):
        gx = int(np.floor((x - self.origin_world[0]) / self.res))
        gy = int(np.floor((y - self.origin_world[1]) / self.res))
        return gx, gy

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.w and 0 <= gy < self.h

    def update_cell(self, gx: int, gy: int, dl: float):
        if self.in_bounds(gx, gy):
            self.L[gy, gx] = np.clip(self.L[gy, gx] + dl, self.l_min, self.l_max)

    def probability(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.L))


@dataclass
class Submap2D:
    id: int
    grid: ProbabilityGrid
    pose_world: Pose2
    num_inserted: int = 0
    finished: bool = False


class SubmapBuilder2D:
    """
    Cartographer-style submap manager with active and finished submaps.
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
        fast_insert: bool = False,
    ):
        self.submap_size_m = float(submap_size_m)
        self.resolution = float(resolution)
        self.scans_per_submap = int(scans_per_submap)
        self.ray_steps = int(ray_steps)
        # When True, use the vectorized occupancy integration (adds all log-odds at
        # once + a single clip) instead of the per-cell scalar-clip loop. ~30x faster
        # insertion; values match within float rounding. Default False keeps the
        # original per-cell behaviour byte-for-byte.
        self.fast_insert = bool(fast_insert)

        self._grid_params = dict(
            l0=l0,
            l_occ=l_occ,
            l_free=l_free,
            l_min=l_min,
            l_max=l_max,
        )
        self._next_id = 0
        self.active: List[Submap2D] = []
        self.finished_submaps: List[Submap2D] = []
        self._newly_finished_ids: List[int] = []
        self._last_inserted_submaps: List[Submap2D] = []
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
        """Cartographer-faithful submap rotation (active_submaps_2d.cc).

        The original Cartographer invariant:
          active[0] = ALWAYS a full/mature submap (>=N/2 scans).
          active[1] = the currently-growing submap.

        The finished submap stays in active[] as the matching target for an
        entire second submap cycle (N/2 scans). It is only removed once
        active[1] has accumulated N/2 scans, at which point active[1] becomes
        the new reliable matching target and a fresh empty submap is appended.

        Our previous code (pop immediately on reaching N) switched the matcher
        to a 0-scan submap, causing systematic score=-1 FALLBACKs.
        """
        # Step 1: Mark the oldest active submap as finished when full.
        # Do NOT remove it — it becomes the continued match target.
        if (self.active
                and not self.active[0].finished
                and self.active[0].num_inserted >= self.scans_per_submap):
            self.active[0].finished = True
            self.finished_submaps.append(self.active[0])
            self._newly_finished_ids.append(int(self.active[0].id))

        # Step 2: Rotate ONLY when the second submap is mature (>=N/2 scans).
        # This guarantees active[0] always has >=N/2 scans at match time.
        if (len(self.active) >= 2
                and self.active[0].finished
                and self.active[1].num_inserted >= self.scans_per_submap // 2):
            self.active.pop(0)
            self.active.append(self._new_submap(pose_world))
            # Handle the startup simultaneous-fill edge case: if the new [0]
            # also happens to be full (both submaps started at the same time),
            # mark it finished immediately so the next rotate fires correctly.
            if (self.active
                    and not self.active[0].finished
                    and self.active[0].num_inserted >= self.scans_per_submap):
                self.active[0].finished = True
                self.finished_submaps.append(self.active[0])
                self._newly_finished_ids.append(int(self.active[0].id))

    def insert_scan(self, pose_world: Pose2, scan_points_local: np.ndarray) -> bool:
        if not self._initialized:
            self._ensure_two_active(pose_world)
            self._initialized = True

        # Only insert range data into non-finished submaps.
        # Cartographer's Submap2D::InsertRangeData asserts CHECK(!finished_).
        insertable = [sm for sm in self.active if not sm.finished]
        # _last_inserted_submaps returns ALL active submaps (including the
        # finished match-target) for pose-graph intra-submap constraint tracking.
        self._last_inserted_submaps = list(self.active)

        endpoints_world = transform_points_pose(pose_world, scan_points_local)

        for sm in insertable:
            T_ws_inv = inverse_pose(sm.pose_world)

            origin_sub = transform_points_pose(
                T_ws_inv,
                np.array([[pose_world.x, pose_world.y]], dtype=float),
            )[0]
            endpoints_sub = transform_points_pose(T_ws_inv, endpoints_world)

            if self.fast_insert:
                self._integrate_submap_frame_fast(sm, origin_sub, endpoints_sub)
            else:
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

    def _integrate_submap_frame_fast(
        self,
        sm: Submap2D,
        origin_sub: np.ndarray,
        endpoints_sub: np.ndarray,
    ) -> None:
        """Vectorized occupancy integration.

        Collects all free/occupied cell indices from the Bresenham rays, accumulates the
        log-odds in one shot (np.bincount over flattened indices), and clips once. A cell
        hit by K rays still gets K·l_free (same as the per-cell loop); the only difference
        from `_integrate_submap_frame` is end-clip vs per-update-clip, which is identical
        for same-sign accumulation and differs only negligibly at the clamp bounds.
        """
        grid = sm.grid
        W, H = int(grid.w), int(grid.h)
        gx0, gy0 = grid.world_to_grid(float(origin_sub[0]), float(origin_sub[1]))
        if not grid.in_bounds(gx0, gy0):
            return

        free_flat: list = []
        occ_flat: list = []
        for ex, ey in endpoints_sub:
            gx1, gy1 = grid.world_to_grid(float(ex), float(ey))
            if not grid.in_bounds(gx1, gy1):
                continue
            cells = self._bresenham(gx0, gy0, gx1, gy1)
            for (cx, cy) in cells[:-1]:
                free_flat.append(cy * W + cx)
            occ_flat.append(gy1 * W + gx1)

        flat = grid.L.reshape(-1)
        n = W * H
        if free_flat:
            flat += np.bincount(np.asarray(free_flat, dtype=np.intp), minlength=n).astype(np.float32) * grid.l_free
        if occ_flat:
            flat += np.bincount(np.asarray(occ_flat, dtype=np.intp), minlength=n).astype(np.float32) * grid.l_occ
        np.clip(grid.L, grid.l_min, grid.l_max, out=grid.L)

    def get_active_submaps(self) -> List[Submap2D]:
        return list(self.active)

    def get_last_inserted_submaps(self) -> List[Submap2D]:
        return list(self._last_inserted_submaps)

    def get_finished_submaps(self) -> List[Submap2D]:
        return list(self.finished_submaps)

    def consume_newly_finished_ids(self) -> List[int]:
        ids = list(self._newly_finished_ids)
        self._newly_finished_ids.clear()
        return ids

    def get_submap_by_id(self, submap_id: int) -> Submap2D:
        submap_id = int(submap_id)
        for sm in self.active:
            if int(sm.id) == submap_id:
                return sm
        for sm in self.finished_submaps:
            if int(sm.id) == submap_id:
                return sm
        raise KeyError(f"Unknown submap id: {submap_id}")

    def clear(self) -> None:
        self.active = []
        self.finished_submaps = []
        self._newly_finished_ids = []
        self._last_inserted_submaps = []
        self._initialized = False
        self._next_id = 0