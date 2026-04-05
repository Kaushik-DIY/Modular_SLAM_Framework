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
    def __init__(self, backend, sig_xy=0.05, sig_theta=np.deg2rad(1.0)):
        self.backend = backend

        self.sig_xy = float(sig_xy)
        self.sig_theta = float(sig_theta)

        self.nodes: list[PoseGraphNode] = []
        self.submaps: dict[int, PoseGraphSubmap] = {}

        self.next_node_id = 0

        self.num_intra_constraints = 0
        self.num_loop_constraints = 0

        # Cartographer-style weights are closer to per-axis residual scaling
        # than generic information matrices. These are base weights for local
        # insertion constraints.
        self.intra_translation_weight = 1.0 / max(self.sig_xy, 1e-6)
        self.intra_rotation_weight = 1.0 / max(self.sig_theta, 1e-6)

    def add_submap_if_needed(self, submap_id: int, submap_pose_world: Pose2):
        submap_id = int(submap_id)
        if submap_id in self.submaps:
            return

        sm = PoseGraphSubmap(id=submap_id, pose=submap_pose_world)
        self.submaps[submap_id] = sm
        self.backend.add_submap(sm)

    def get_submap_pose(self, submap_id: int) -> Pose2:
        submap_id = int(submap_id)
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

        node = PoseGraphNode(id=node_id, time=float(t), pose=node_pose_world)
        self.nodes.append(node)
        self.backend.add_node(node)

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
        match_score: float,
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

    def _sync_from_backend(self, optimized):
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

    def solve(self, max_iters: int = 50):
        if hasattr(self.backend, "solve"):
            try:
                optimized = self.backend.solve(max_iters=max_iters)
            except TypeError:
                optimized = self.backend.solve()
            self._sync_from_backend(optimized)
            return optimized
        raise RuntimeError("Backend does not implement solve().")

    def get_optimized_poses(self):
        return self.backend.get_optimized_poses()

    def get_constraint_counts(self):
        return {
            "total": int(len(self.backend.constraints)),
            "intra": int(self.num_intra_constraints),
            "loop": int(self.num_loop_constraints),
        }