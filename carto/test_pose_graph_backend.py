# test_pose_graph_backend.py

import numpy as np

from carto.pose_graph.pose_graph_2d import PoseGraph2D
from carto.pose_graph.backends.scipy_backend_2d import SciPyBackend2D
from carto.pose_graph.constraint import PoseGraphNode, PoseGraphSubmap, PoseGraphConstraint
from carto.common.types import Pose2
from carto.common.se2 import inverse_pose, pose_compose


def main():
    print("Creating pose graph...")

    pg = PoseGraph2D(
        backend=SciPyBackend2D(),
        sig_xy=0.1,
        sig_theta=np.deg2rad(5.0),
    )

    # ---- Add submap 0 at origin ----
    submap0 = PoseGraphSubmap(id=0, pose=Pose2(0.0, 0.0, 0.0))
    pg.backend.add_submap(submap0)

    # ---- Add node 0 at (1, 0, 0) ----
    node0 = PoseGraphNode(id=0, time=0.0, pose=Pose2(1.0, 0.0, 0.0))
    pg.backend.add_node(node0)

    # Constraint: submap0 -> node0 = (1,0,0)
    info = np.diag([100.0, 100.0, 100.0])
    c0 = PoseGraphConstraint(
        type_from="submap",
        id_from=0,
        type_to="node",
        id_to=0,
        relative_pose=Pose2(1.0, 0.0, 0.0),
        information=info,
    )
    pg.backend.add_constraint(c0)

    # Node 1 initial guess (slightly wrong on purpose)
    node1 = PoseGraphNode(id=1, time=1.0, pose=Pose2(2.2, 0.1, 0.05))
    pg.backend.add_node(node1)

    # Constraint: submap0 -> node1 should be (2,0,0)
    c1 = PoseGraphConstraint(
        type_from="submap",
        id_from=0,
        type_to="node",
        id_to=1,
        relative_pose=Pose2(2.0, 0.0, 0.0),
        information=info,
    )
    pg.backend.add_constraint(c1)

    print("Solving...")
    pg.solve()

    print("Optimized poses:")
    result = pg.backend.get_optimized_poses()
    for key, pose in result.items():
        print(key, pose)


if __name__ == "__main__":
    main()