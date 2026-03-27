import numpy as np
from carto.common.se2 import inverse_pose, pose_compose
from carto.pose_graph.constraint import PoseGraphNode, PoseGraphSubmap, PoseGraphConstraint


class PoseGraph2D:
    """
    Solver-agnostic pose graph manager.
    Owns the graph structure:
      - node variables (trajectory nodes)
      - submap variables
      - constraints between them

    Backend owns:
      - parameterization + solver implementation
    """

    def __init__(self, backend, sig_xy=0.05, sig_theta=np.deg2rad(1.0)):
        self.backend = backend

        self.sig_xy = float(sig_xy)
        self.sig_theta = float(sig_theta)

        self.nodes: list[PoseGraphNode] = []
        self.submaps: dict[int, PoseGraphSubmap] = {}

        self.next_node_id = 0

        # Default information for intra constraints (submap <-> node)
        self.default_information = np.diag([
            1.0 / (self.sig_xy ** 2),
            1.0 / (self.sig_xy ** 2),
            1.0 / (self.sig_theta ** 2),
        ])

        # Inter-submap constraints usually weaker
        self.inter_submap_information = np.diag([
            1.0 / (0.20 ** 2),              # 20 cm
            1.0 / (0.20 ** 2),
            1.0 / (np.deg2rad(5.0) ** 2),   # 5 deg
        ])

    # -------------------------
    # Submaps
    # -------------------------
    def add_submap_if_needed(self, submap_id: int, submap_pose_world):
        submap_id = int(submap_id)
        if submap_id in self.submaps:
            return

        sm = PoseGraphSubmap(id=submap_id, pose=submap_pose_world)
        self.submaps[submap_id] = sm

        # BackendBase-style
        self.backend.add_submap(sm)

    def add_inter_submap_constraint(self, prev_submap_id: int, new_submap_id: int):
        """
        Adds constraint: prev_submap -> new_submap
        relative = T_prev^-1 ⊕ T_new  (both in world)
        """
        prev_submap_id = int(prev_submap_id)
        new_submap_id = int(new_submap_id)

        sm_prev = self.submaps[prev_submap_id]
        sm_new = self.submaps[new_submap_id]

        rel = pose_compose(inverse_pose(sm_prev.pose), sm_new.pose)

        c = PoseGraphConstraint(
            type_from="submap",
            id_from=prev_submap_id,
            type_to="submap",
            id_to=new_submap_id,
            relative_pose=rel,
            information=self.inter_submap_information,
        )
        self.backend.add_constraint(c)

    # -------------------------
    # Nodes + intra constraints
    # -------------------------
    def add_node_with_intra_constraints(self, t, node_pose_world, active_submaps):
        node_id = int(self.next_node_id)

        # 1) Add node variable
        node = PoseGraphNode(id=node_id, time=float(t), pose=node_pose_world)
        self.nodes.append(node)
        self.backend.add_node(node)

        # 2) Ensure submaps exist + add constraints submap -> node
        for sm in active_submaps:
            self.add_submap_if_needed(sm.id, sm.pose_world)

            rel = pose_compose(inverse_pose(sm.pose_world), node_pose_world)

            c = PoseGraphConstraint(
                type_from="submap",
                id_from=int(sm.id),
                type_to="node",
                id_to=node_id,
                relative_pose=rel,
                information=self.default_information,
            )
            self.backend.add_constraint(c)

        self.next_node_id += 1
        return node_id

    # -------------------------
    def solve(self, max_iters: int = 50):
        # allow backend to ignore max_iters if not supported
        if hasattr(self.backend, "solve"):
            try:
                return self.backend.solve(max_iters=max_iters)
            except TypeError:
                return self.backend.solve()
        raise RuntimeError("Backend does not implement solve().")

    def get_optimized_poses(self):
        return self.backend.get_optimized_poses()