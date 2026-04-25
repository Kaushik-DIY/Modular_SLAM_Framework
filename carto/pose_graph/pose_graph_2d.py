import numpy as np
from carto.common.se2 import inverse_pose, pose_compose
from carto.common.types import Pose2
from carto.pose_graph.constraint import (
    PoseGraphNode,
    PoseGraphSubmap,
    PoseGraphConstraint,
    ConstraintPose2D,
    INTRA_SUBMAP,
    INTER_SUBMAP,
)


class PoseGraph2D:
    """
    Cartographer-style 2D pose graph.

    Holds the graph structure (nodes, submaps, constraints) and delegates the
    nonlinear optimization to a pluggable backend (PyCeresBackend2D is primary).

    Weight defaults match the original Cartographer pose_graph.lua:
        matcher_translation_weight = 5e2
        matcher_rotation_weight    = 1.6e3

    These are used for INTRA_SUBMAP constraints (local scan-to-submap).
    Loop closure (INTER_SUBMAP) weights are set per-constraint by the
    ConstraintBuilder2D, using the Cartographer lua defaults:
        loop_closure_translation_weight = 1.1e4
        loop_closure_rotation_weight    = 1e5
    """

    def __init__(
        self,
        backend,
        submap_builder=None,
        intra_translation_weight: float = 5e2,
        intra_rotation_weight: float = 1.6e3,
        # Legacy sig_xy / sig_theta kept for backwards compat but ignored
        sig_xy: float = None,
        sig_theta: float = None,
    ):
        self.backend = backend

        # Canonical Cartographer intra-submap weights (matcher_translation/rotation_weight)
        self.intra_translation_weight = float(intra_translation_weight)
        self.intra_rotation_weight = float(intra_rotation_weight)

        self.nodes: list[PoseGraphNode] = []
        self.submaps: dict[int, PoseGraphSubmap] = {}

        self.next_node_id = 0

        self.num_intra_constraints = 0
        self.num_loop_constraints = 0

        # Optional reference to the live submap builder so optimized poses
        # can be written back after solve().
        self._submap_builder = submap_builder

        # --- FIX: State Preservation for 'Before' Mapping ---
        # node_id -> drifted_pose_world (never updated by solver)
        self.drifted_nodes: dict[int, Pose2] = {}
        # submap_id -> drifted_pose_world (never updated by solver)
        self.drifted_submaps: dict[int, Pose2] = {}

    # ------------------------------------------------------------------
    # Submap builder management
    # ------------------------------------------------------------------

    def set_submap_builder(self, submap_builder) -> None:
        """Attach (or replace) the live submap builder reference."""
        self._submap_builder = submap_builder

    # ------------------------------------------------------------------
    # Graph mutation
    # ------------------------------------------------------------------

    def add_submap_if_needed(self, submap_id: int, submap_pose_world: Pose2):
        submap_id = int(submap_id)
        if submap_id in self.submaps:
            return

        sm = PoseGraphSubmap(id=submap_id, pose=submap_pose_world)
        self.submaps[submap_id] = sm
        self.drifted_submaps[submap_id] = submap_pose_world
        self.backend.add_submap(sm)

    def get_submap_pose(self, submap_id: int, use_optimized: bool = True) -> Pose2:
        submap_id = int(submap_id)
        if not use_optimized:
            return self.drifted_submaps[submap_id]
        if submap_id in self.backend.submaps:
            return self.backend.submaps[submap_id]
        if submap_id in self.submaps:
            return self.submaps[submap_id].pose
        raise KeyError(f"Unknown submap id: {submap_id}")

    def get_node_pose(self, node_id: int) -> Pose2:
        node_id = int(node_id)
        if node_id in self.backend.nodes:
            return self.backend.nodes[node_id]
        for node in self.nodes:
            if int(node.id) == node_id:
                return node.pose
        raise KeyError(f"Unknown node id: {node_id}")

    def add_node_with_intra_constraints(self, t, node_pose_world, active_submaps):
        node_id = int(self.next_node_id)

        # Compute local_pose: T_submap^{-1} * T_node_world for the primary
        # (oldest) active submap.  This is NodeSpec2D.local_pose_2d in the
        # original Cartographer, and is the measurement used for consecutive-node
        # local-trajectory regularization.
        local_pose = None
        if active_submaps:
            primary_sm = active_submaps[0]
            local_pose = pose_compose(inverse_pose(primary_sm.pose_world), node_pose_world)

        node = PoseGraphNode(id=node_id, time=float(t), pose=node_pose_world, local_pose=local_pose)
        self.nodes.append(node)
        self.backend.add_node(node)

        # If the backend supports local-pose tracking (PyCeresBackend2D), register it.
        if local_pose is not None and hasattr(self.backend, "update_node_local_pose"):
            self.backend.update_node_local_pose(node_id, local_pose)

        self.drifted_nodes[node_id] = node_pose_world

        for sm in active_submaps:
            self.add_submap_if_needed(sm.id, sm.pose_world)

            rel = pose_compose(inverse_pose(sm.pose_world), node_pose_world)

            c = PoseGraphConstraint(
                submap_id=int(sm.id),
                node_id=node_id,
                pose=ConstraintPose2D(
                    relative_pose=rel,
                    translation_weight=float(self.intra_translation_weight),
                    rotation_weight=float(self.intra_rotation_weight),
                    match_score=1.0,
                ),
                tag=INTRA_SUBMAP,
            )
            self.backend.add_constraint(c)
            self.num_intra_constraints += 1

        self.next_node_id += 1
        return node_id

    def add_loop_submap_node_constraint(
        self,
        submap_id: int,
        node_id: int,
        relative_pose: Pose2,
        translation_weight: float,
        rotation_weight: float,
        match_score: float = 1.0,
    ) -> None:
        c = PoseGraphConstraint(
            submap_id=int(submap_id),
            node_id=int(node_id),
            pose=ConstraintPose2D(
                relative_pose=relative_pose,
                translation_weight=float(translation_weight),
                rotation_weight=float(rotation_weight),
                match_score=float(match_score),
            ),
            tag=INTER_SUBMAP,
        )
        self.backend.add_constraint(c)
        self.num_loop_constraints += 1

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def _sync_from_backend(self, optimized):
        """Write optimized poses from backend dict back into graph node/submap objects."""
        if not isinstance(optimized, dict):
            return

        for (kind, idx), pose in optimized.items():
            if kind == "submap":
                if int(idx) in self.submaps:
                    self.submaps[int(idx)].pose = pose
                if int(idx) in self.backend.submaps:
                    self.backend.submaps[int(idx)] = pose
            elif kind == "node":
                if int(idx) in self.backend.nodes:
                    self.backend.nodes[int(idx)] = pose
                if 0 <= int(idx) < len(self.nodes) and int(self.nodes[int(idx)].id) == int(idx):
                    self.nodes[int(idx)].pose = pose
                else:
                    for node in self.nodes:
                        if int(node.id) == int(idx):
                            node.pose = pose
                            break

    def _sync_submaps_to_builder(self, optimized: dict) -> None:
        """
        Push optimized submap poses back into the live SubmapBuilder2D.

        After each solve() the optimized poses in the backend dict are the
        ground truth for global positions. The submap builder must be updated
        so that loop-closure distance checks and future scan insertions use
        the correct post-optimization positions.

        This is the Python equivalent of the global_submap_poses_2d update
        that Cartographer's HandleWorkQueue() triggers after RunOptimization().
        """
        if self._submap_builder is None:
            return

        for (kind, idx), pose in optimized.items():
            if kind != "submap":
                continue
            # Update finished submaps
            for sm in self._submap_builder.finished_submaps:
                if int(sm.id) == int(idx):
                    sm.pose_world = pose
                    break
            # Update active submaps
            for sm in self._submap_builder.active:
                if int(sm.id) == int(idx):
                    sm.pose_world = pose
                    break

    def solve(self, max_iters: int = 50):
        if hasattr(self.backend, "solve"):
            try:
                optimized = self.backend.solve(max_iters=max_iters)
            except TypeError:
                optimized = self.backend.solve()
            self._sync_from_backend(optimized)
            self._sync_submaps_to_builder(optimized)
            return optimized
        raise RuntimeError("Backend does not implement solve().")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_optimized_poses(self):
        return self.backend.get_optimized_poses()

    def get_constraint_counts(self):
        return {
            "total": int(len(self.backend.constraints)),
            "intra": int(self.num_intra_constraints),
            "loop": int(self.num_loop_constraints),
        }