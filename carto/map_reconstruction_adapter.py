from __future__ import annotations

from typing import List
import numpy as np
import matplotlib.pyplot as plt

from slam_core.map_reconstruction import (
    MapTile,
    ReconstructionConfig,
    ReconstructionResult,
    MapReconstructionManager,
)
from slam_core.common.types import Pose2


class CartoTileProvider:
    """
    Provides Cartographer-style submaps as generic reconstruction tiles.
    """

    def __init__(self, matcher, pose_graph) -> None:
        self.matcher = matcher
        self.pose_graph = pose_graph

    def get_tiles(self, use_optimized_poses: bool) -> List[MapTile]:
        submaps = []
        submaps.extend(self.matcher.submap_builder.get_finished_submaps())
        submaps.extend(self.matcher.submap_builder.get_active_submaps())

        tiles: List[MapTile] = []
        seen_ids = set()

        for sm in submaps:
            sm_id = int(sm.id)
            if sm_id in seen_ids:
                continue
            seen_ids.add(sm_id)

            try:
                pose = self.pose_graph.get_submap_pose(sm_id, use_optimized=use_optimized_poses)
            except (KeyError, AttributeError):
                pose = sm.pose_world

            tiles.append(
                MapTile(
                    tile_id=str(sm_id),
                    pose_global=Pose2(float(pose.x), float(pose.y), float(pose.theta)),
                    evidence_grid=sm.grid.L.copy(),
                    resolution=float(sm.grid.res),
                    origin_local_xy=np.asarray(sm.grid.origin_world, dtype=float).copy(),
                    neutral_value=float(sm.grid.l0),
                )
            )

        return tiles


class CartoMapReconstructionAdapter:
    """
    Carto-specific wrapper around the generic map reconstruction manager.
    """

    def __init__(self, matcher, pose_graph, config: ReconstructionConfig) -> None:
        self.matcher = matcher
        self.pose_graph = pose_graph
        self.config = config

        self.provider = CartoTileProvider(matcher=self.matcher, pose_graph=self.pose_graph)
        self.manager = MapReconstructionManager(config=self.config, provider=self.provider)

    def reconstruct_before_after(self):
        before_result = self.manager.reconstruct(use_optimized_poses=False)
        after_result = self.manager.reconstruct(use_optimized_poses=True)
        return before_result, after_result

    def save_before_after_plot(self, out_prefix: str) -> None:
        before_result, after_result = self.reconstruct_before_after()

        self._save_result_arrays(before_result, f"{out_prefix}_before")
        self._save_result_arrays(after_result, f"{out_prefix}_after")

        self._plot_before_after(before_result, after_result, f"{out_prefix}_before_after.png")

    @staticmethod
    def _save_result_arrays(result: ReconstructionResult, out_prefix: str) -> None:
        np.save(f"{out_prefix}_prob.npy", result.probability_grid)
        np.save(f"{out_prefix}_evidence.npy", result.evidence_grid)

    @staticmethod
    def _plot_before_after(
        before_result: ReconstructionResult,
        after_result: ReconstructionResult,
        out_path: str,
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        extent_before = [
            before_result.extent_xy[0],
            before_result.extent_xy[1],
            before_result.extent_xy[2],
            before_result.extent_xy[3],
        ]
        extent_after = [
            after_result.extent_xy[0],
            after_result.extent_xy[1],
            after_result.extent_xy[2],
            after_result.extent_xy[3],
        ]

        axes[0].imshow(
            before_result.probability_grid,
            origin="lower",
            extent=extent_before,
            cmap="gray",
        )
        axes[0].set_title("Map Before Loop Closure")
        axes[0].set_xlabel("x [m]")
        axes[0].set_ylabel("y [m]")
        axes[0].axis("equal")
        axes[0].grid(True)

        axes[1].imshow(
            after_result.probability_grid,
            origin="lower",
            extent=extent_after,
            cmap="gray",
        )
        axes[1].set_title("Map After Loop Closure")
        axes[1].set_xlabel("x [m]")
        axes[1].set_ylabel("y [m]")
        axes[1].axis("equal")
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.show()

        print("Wrote:", out_path)
        print("Wrote:", out_path.replace("_before_after.png", "_before_prob.npy"))
        print("Wrote:", out_path.replace("_before_after.png", "_before_evidence.npy"))
        print("Wrote:", out_path.replace("_before_after.png", "_after_prob.npy"))
        print("Wrote:", out_path.replace("_before_after.png", "_after_evidence.npy"))