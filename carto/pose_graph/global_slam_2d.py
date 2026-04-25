from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from carto.adapter import CartoLocalSlamAdapter

from slam_core.matching.scan_to_submap.submaps import Submap2D
from carto.pose_graph.pose_graph_2d import PoseGraph2D
from carto.loop_closure_adapter import CartoLoopClosureAdapter


class CartoGlobalSlam2D:
    """
    Orchestration layer for Cartographer-style global SLAM (back-end).

    Coordinates:
    - The CartoLoopClosureAdapter (two-way constraint search)
    - The PoseGraph2D (graph structure + solve)
    - Optional CartoLocalSlamAdapter reference (for extrapolator correction)

    After each optimization round, optimized submap poses are pushed back into
    the live SubmapBuilder2D (via PoseGraph2D._sync_submaps_to_builder), and
    the extrapolator state is softly corrected toward the most recent optimized
    node pose (via CartoLocalSlamAdapter.apply_optimization_correction).
    """

    def __init__(
        self,
        loop_closure_adapter: CartoLoopClosureAdapter,
        pose_graph: PoseGraph2D,
        optimize_every_n_nodes: int = 90,
        adapter: Optional["CartoLocalSlamAdapter"] = None,
        correction_alpha: float = 0.5,
    ) -> None:
        self.loop_closure_adapter = loop_closure_adapter
        self.pose_graph = pose_graph
        self.optimize_every_n_nodes = int(optimize_every_n_nodes)

        # Optional reference to the local SLAM adapter for extrapolator correction
        self._adapter: Optional["CartoLocalSlamAdapter"] = adapter
        self._correction_alpha = float(correction_alpha)

        self._last_node_id: int = -1
        self._nodes_since_last_opt: int = 0

    def set_adapter(self, adapter: "CartoLocalSlamAdapter") -> None:
        """Attach the local SLAM adapter for post-solve extrapolator correction."""
        self._adapter = adapter

    def on_node_inserted(
        self,
        node_id: int,
        timestamp: float,
        scan_points,
        pose_global,
        insertion_submaps: List[Submap2D],
    ) -> None:
        """
        Called after each new node is added to the pose graph.

        - Forwards node to loop closure manager for candidate search.
        - Triggers optimization after every optimize_every_n_nodes.
        - Processes any newly finished submaps (second search direction).
        """
        self._last_node_id = int(node_id)
        self._nodes_since_last_opt += 1

        # Register node with the loop closure manager
        local_submap_ids = [str(sm.id) for sm in insertion_submaps]
        self.loop_closure_adapter.on_new_node(
            node_id=node_id,
            scan_points=scan_points,
            pose_guess_global=pose_global,
            timestamp=float(timestamp),
            local_submap_ids=local_submap_ids,
        )

        # Process any submaps that were finished during this scan insertion
        self.loop_closure_adapter.process_finished_submaps()

        # Cartographer triggers optimization every N nodes regardless of whether
        # loop constraints exist yet (the intra-submap constraints alone form a valid
        # graph to solve). Requiring num_loop_constraints > 0 meant the optimizer never
        # ran during the first ~100 nodes, leaving submap poses un-refined when loop
        # closure eventually started. Use num_intra_constraints as the guard instead.
        if (self.optimize_every_n_nodes > 0 and
                self._nodes_since_last_opt >= self.optimize_every_n_nodes and
                self.pose_graph.num_intra_constraints > 0):
            self._run_optimization()

    def _run_optimization(self) -> None:
        """
        Run one round of Ceres pose-graph optimization.

        After solve:
        1. Optimized submap poses are written back to SubmapBuilder2D
           (via PoseGraph2D._sync_submaps_to_builder).
        2. The extrapolator state is softly corrected toward the last
           optimized node pose (if an adapter reference is set).
        """
        optimized = self.pose_graph.solve()
        self._nodes_since_last_opt = 0

        # Apply extrapolator correction (Work Item 8)
        if self._adapter is not None and self._last_node_id >= 0 and optimized:
            self._adapter.apply_optimization_correction(
                optimized=optimized,
                last_node_id=self._last_node_id,
                correction_alpha=self._correction_alpha,
            )

    def finalize(self) -> None:
        """
        Full optimization pass at the end of the mapping session.

        Cartographer runs max_num_final_iterations=200 here. We run a single
        solve with the maximum iteration budget set in the backend.
        """
        if self.pose_graph.num_intra_constraints > 0:
            self.loop_closure_adapter.finalize()
            optimized = self.pose_graph.solve()
            if self._adapter is not None and self._last_node_id >= 0 and optimized:
                # Full correction at finalize — safe since no more live matching
                self._adapter.apply_optimization_correction(
                    optimized=optimized,
                    last_node_id=self._last_node_id,
                    correction_alpha=1.0,   # Full correction at end
                )

    def get_stats(self):
        lc_stats = self.loop_closure_adapter.get_stats()
        pg_counts = self.pose_graph.get_constraint_counts()
        return {
            # Flat fields matching the original API expected by run_loop_closure_slam.py
            "candidate_pairs": int(lc_stats.candidate_pairs),
            "accepted_pairs": int(lc_stats.accepted_pairs),
            "rejected_pairs": int(lc_stats.rejected_pairs),
            "duplicate_pairs": int(lc_stats.duplicate_pairs),
            # Nested structs for richer access
            "loop_closure": lc_stats,
            "pose_graph": pg_counts,
        }

    def get_recent_events(self, n: int = 20):
        """Delegate to the loop closure adapter's event log."""
        return self.loop_closure_adapter.get_recent_events(n)

    def get_diagnostics_summary(self) -> dict:
        """Delegate to the active loop-closure adapter diagnostics."""
        return self.loop_closure_adapter.get_diagnostics_summary()
