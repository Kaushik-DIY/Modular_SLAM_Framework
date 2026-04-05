from __future__ import annotations

from typing import Optional
import numpy as np

from slam_core.common.types import Pose2
from carto.pose_graph.pose_graph_2d import PoseGraph2D
from carto.pose_graph.constraint_builder_2d import (
    ConstraintBuilder2D,
    ConstraintBuilder2DConfig,
)


class CartoGlobalSlam2D:
    def __init__(
        self,
        matcher,
        pose_graph: PoseGraph2D,
        config: Optional[ConstraintBuilder2DConfig] = None,
    ) -> None:
        self.matcher = matcher
        self.pose_graph = pose_graph
        self.config = config or ConstraintBuilder2DConfig()

        self.constraint_builder = ConstraintBuilder2D(
            matcher=self.matcher,
            pose_graph=self.pose_graph,
            config=self.config,
        )

    def on_node_inserted(
        self,
        node_id: int,
        timestamp: float,
        scan_points: np.ndarray,
        pose_global: Pose2,
        insertion_submaps,
    ) -> None:
        insertion_submap_ids = [int(sm.id) for sm in insertion_submaps]

        self.constraint_builder.add_node(
            node_id=int(node_id),
            timestamp=float(timestamp),
            scan_points=np.asarray(scan_points, dtype=float),
            pose_global=pose_global,
            insertion_submap_ids=insertion_submap_ids,
        )

        self.constraint_builder.maybe_add_constraints_for_new_node(int(node_id))

        finished_ids = self.matcher.submap_builder.consume_newly_finished_ids()
        for sid in finished_ids:
            self.constraint_builder.maybe_add_constraints_for_finished_submap(int(sid))

        self.constraint_builder.maybe_optimize()

    def finalize(self) -> None:
        self.pose_graph.solve()

    def get_stats(self) -> dict:
        return self.constraint_builder.get_stats()

    def get_recent_events(self, n: int = 20):
        return self.constraint_builder.get_recent_events(n)