from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Tuple
import numpy as np

from slam_core.common.types import Pose2
from slam_core.common.se2 import transform_points_pose


@dataclass
class MapTile:
    """
    Generic local map tile used for post-optimization global map composition.
    """
    tile_id: str
    pose_global: Pose2
    evidence_grid: np.ndarray
    resolution: float
    origin_local_xy: np.ndarray
    neutral_value: float = 0.0


@dataclass
class ReconstructionConfig:
    """
    Configuration for global map composition from local map tiles.
    """
    global_resolution: float
    informative_evidence_threshold: float = 0.05
    evidence_clip_min: float = -10.0
    evidence_clip_max: float = 10.0
    tile_cell_stride: int = 1
    map_margin_m: float = 1.0


@dataclass
class ReconstructionResult:
    """
    Output of the post-optimization map reconstruction stage.
    """
    probability_grid: np.ndarray
    evidence_grid: np.ndarray
    extent_xy: Tuple[float, float, float, float]


class TileProvider(Protocol):
    """
    Backend-specific provider of map tiles.
    """

    def get_tiles(self, use_optimized_poses: bool) -> List[MapTile]:
        ...


class MapReconstructionManager:
    """
    Generic global map composer.

    The manager is intentionally independent of the SLAM backend. It only
    assumes that the backend can provide local map tiles together with their
    global poses.
    """

    def __init__(self, config: ReconstructionConfig, provider: TileProvider) -> None:
        self.config = config
        self.provider = provider

    def reconstruct(self, use_optimized_poses: bool) -> ReconstructionResult:
        tiles = self.provider.get_tiles(use_optimized_poses=use_optimized_poses)
        if not tiles:
            raise RuntimeError("No map tiles available for reconstruction.")

        min_x, min_y, max_x, max_y = self._compute_bounds(tiles)
        res = float(self.config.global_resolution)

        width = int(np.ceil((max_x - min_x) / res)) + 1
        height = int(np.ceil((max_y - min_y) / res)) + 1

        global_evidence = np.zeros((height, width), dtype=np.float32)

        for tile in tiles:
            self._accumulate_tile(
                tile=tile,
                global_evidence=global_evidence,
                min_x=min_x,
                min_y=min_y,
                global_res=res,
            )

        global_evidence = np.clip(
            global_evidence,
            self.config.evidence_clip_min,
            self.config.evidence_clip_max,
        )
        global_prob = 1.0 / (1.0 + np.exp(-global_evidence))

        return ReconstructionResult(
            probability_grid=global_prob,
            evidence_grid=global_evidence,
            extent_xy=(min_x, max_x, min_y, max_y),
        )

    def _compute_bounds(self, tiles: List[MapTile]) -> Tuple[float, float, float, float]:
        xs = []
        ys = []

        for tile in tiles:
            h, w = tile.evidence_grid.shape
            res = float(tile.resolution)
            x0, y0 = float(tile.origin_local_xy[0]), float(tile.origin_local_xy[1])

            corners_local = np.array(
                [
                    [x0, y0],
                    [x0 + w * res, y0],
                    [x0, y0 + h * res],
                    [x0 + w * res, y0 + h * res],
                ],
                dtype=float,
            )

            corners_world = transform_points_pose(tile.pose_global, corners_local)
            xs.extend(corners_world[:, 0].tolist())
            ys.extend(corners_world[:, 1].tolist())

        margin = float(self.config.map_margin_m)
        return (
            min(xs) - margin,
            min(ys) - margin,
            max(xs) + margin,
            max(ys) + margin,
        )

    def _accumulate_tile(
        self,
        tile: MapTile,
        global_evidence: np.ndarray,
        min_x: float,
        min_y: float,
        global_res: float,
    ) -> None:
        grid = tile.evidence_grid
        h, w = grid.shape
        stride = max(1, int(self.config.tile_cell_stride))

        delta = grid - float(tile.neutral_value)
        mask = np.abs(delta) > float(self.config.informative_evidence_threshold)

        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return

        ys = ys[::stride]
        xs = xs[::stride]

        local_x = float(tile.origin_local_xy[0]) + (xs.astype(float) + 0.5) * float(tile.resolution)
        local_y = float(tile.origin_local_xy[1]) + (ys.astype(float) + 0.5) * float(tile.resolution)
        pts_local = np.column_stack([local_x, local_y])

        pts_world = transform_points_pose(tile.pose_global, pts_local)

        gx = np.floor((pts_world[:, 0] - min_x) / global_res).astype(int)
        gy = np.floor((pts_world[:, 1] - min_y) / global_res).astype(int)

        valid = (
            (gx >= 0) & (gx < global_evidence.shape[1]) &
            (gy >= 0) & (gy < global_evidence.shape[0])
        )
        if not np.any(valid):
            return

        gx = gx[valid]
        gy = gy[valid]
        vals = delta[ys[valid], xs[valid]].astype(np.float32)

        np.add.at(global_evidence, (gy, gx), vals)