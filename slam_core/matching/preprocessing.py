from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class PointCloudProcessorConfig:
    """
    Shared preprocessing configuration for 2D scan points.
    """
    fixed_voxel_size: float = 0.02
    adaptive_voxel_max_size: float = 0.10
    adaptive_min_num_points: int = 160
    adaptive_num_iterations: int = 8
    enabled: bool = True


class PointCloudProcessor:
    """
    Shared 2D point-cloud preprocessing.

    The processor applies:
        1. fixed voxel filtering
        2. adaptive voxel filtering

    This layer is matcher-agnostic and can therefore be used by both
    scan-to-map and scan-to-submap pipelines.
    """

    def __init__(self, config: PointCloudProcessorConfig):
        self.config = config

    def process(self, points_xy: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Apply preprocessing and return:
            filtered_points, debug_info
        """
        pts = np.asarray(points_xy, dtype=float)
        debug = {
            "n_input": int(len(pts)),
            "n_after_fixed": int(len(pts)),
            "n_after_adaptive": int(len(pts)),
        }

        if not self.config.enabled:
            return pts, debug

        if pts.size == 0:
            return pts.reshape(0, 2), debug

        pts_fixed = self._fixed_voxel_filter(
            pts,
            voxel_size=float(self.config.fixed_voxel_size),
        )
        debug["n_after_fixed"] = int(len(pts_fixed))

        pts_adapt = self._adaptive_voxel_filter(
            pts_fixed,
            max_voxel_size=float(self.config.adaptive_voxel_max_size),
            min_num_points=int(self.config.adaptive_min_num_points),
            num_iterations=int(self.config.adaptive_num_iterations),
        )
        debug["n_after_adaptive"] = int(len(pts_adapt))

        return pts_adapt, debug

    @staticmethod
    def _fixed_voxel_filter(points_xy: np.ndarray, voxel_size: float) -> np.ndarray:
        """
        Aggregate points inside each voxel using the centroid.
        """
        if voxel_size <= 0.0 or len(points_xy) == 0:
            return np.asarray(points_xy, dtype=float)

        pts = np.asarray(points_xy, dtype=float)
        voxel_idx = np.floor(pts / voxel_size).astype(np.int64)

        unique_voxels, inverse = np.unique(voxel_idx, axis=0, return_inverse=True)

        sums = np.zeros((len(unique_voxels), 2), dtype=float)
        counts = np.zeros(len(unique_voxels), dtype=np.int64)

        np.add.at(sums, inverse, pts)
        np.add.at(counts, inverse, 1)

        centroids = sums / counts[:, None]
        return centroids.astype(float)

    @classmethod
    def _adaptive_voxel_filter(
        cls,
        points_xy: np.ndarray,
        max_voxel_size: float,
        min_num_points: int,
        num_iterations: int,
    ) -> np.ndarray:
        """
        Find the largest voxel size that still retains at least the requested
        minimum number of points.
        """
        pts = np.asarray(points_xy, dtype=float)
        if len(pts) == 0:
            return pts.reshape(0, 2)

        if max_voxel_size <= 0.0:
            return pts

        if len(pts) <= min_num_points:
            return pts

        low = 0.0
        high = float(max_voxel_size)
        best = pts

        for _ in range(max(1, int(num_iterations))):
            mid = 0.5 * (low + high)
            if mid <= 1e-9:
                break

            filtered = cls._fixed_voxel_filter(pts, voxel_size=mid)

            if len(filtered) >= int(min_num_points):
                best = filtered
                low = mid
            else:
                high = mid

        return best