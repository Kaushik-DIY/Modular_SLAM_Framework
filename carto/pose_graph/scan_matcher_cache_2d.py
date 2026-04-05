from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from slam_core.common.types import Pose2
from slam_core.matching.scan_to_submap import (
    Submap2D,
    PrecomputationGridStack,
    _bruteforce_search,
)


@dataclass
class CachedSubmapMatcher2D:
    """
    Cached matching data for one finished submap.
    """
    submap_id: int
    submap: Submap2D
    prob_img: np.ndarray
    stack: PrecomputationGridStack


class SubmapScanMatcherCache2D:
    """
    Per-submap scan matcher cache.

    Cartographer constructs and reuses scan matcher state per finished submap
    instead of rebuilding search structures for every loop-closure candidate.
    This class mirrors that idea for the current Python framework.
    """

    def __init__(self, precomp_levels: int = 3) -> None:
        self.precomp_levels = int(precomp_levels)
        self._cache: Dict[int, CachedSubmapMatcher2D] = {}

    def has(self, submap_id: int) -> bool:
        return int(submap_id) in self._cache

    def clear(self) -> None:
        self._cache.clear()

    def get_or_build(self, submap: Submap2D) -> CachedSubmapMatcher2D:
        submap_id = int(submap.id)
        if submap_id in self._cache:
            return self._cache[submap_id]

        prob_img = submap.grid.probability().astype(np.float32)
        stack = PrecomputationGridStack(prob_img, num_levels=self.precomp_levels)

        entry = CachedSubmapMatcher2D(
            submap_id=submap_id,
            submap=submap,
            prob_img=prob_img,
            stack=stack,
        )
        self._cache[submap_id] = entry
        return entry

    def coarse_match(
        self,
        submap: Submap2D,
        points_local: np.ndarray,
        initial_submap_pose: Pose2,
        *,
        min_valid: int,
        coarse_level: int,
        coarse_xy_window: float,
        coarse_th_window: float,
        coarse_xy_step: float,
        coarse_th_step: float,
    ):
        """
        Run only the coarse correlative stage against a cached finished submap.
        """
        entry = self.get_or_build(submap)

        best_pose, best_score = _bruteforce_search(
            stack=entry.stack,
            level=int(coarse_level),
            grid_origin_xy=submap.grid.origin_world,
            res=float(submap.grid.res),
            points_local=points_local,
            center_pose=initial_submap_pose,
            xy_window=float(coarse_xy_window),
            th_window=float(coarse_th_window),
            xy_step=float(coarse_xy_step),
            th_step=float(coarse_th_step),
            min_valid=int(min_valid),
        )
        return best_pose, float(best_score)