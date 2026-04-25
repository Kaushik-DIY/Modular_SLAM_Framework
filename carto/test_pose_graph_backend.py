import numpy as np

from carto.pose_graph.pose_graph_2d import PoseGraph2D
from carto.pose_graph.backends.pyceres_backend_2d import PyCeresBackend2D
from carto.pose_graph.constraint import (
    PoseGraphNode,
    PoseGraphSubmap,
    PoseGraphConstraint,
    ConstraintPose2D,
    INTRA_SUBMAP,
    INTER_SUBMAP,
)
from carto.common.types import Pose2


def main():
    print("Creating pose graph...")

    backend = PyCeresBackend2D(
        huber_scale=1.0,
        linear_solver_type="DENSE_QR",
        num_threads=1,
        minimizer_progress_to_stdout=False,
    )
    backend.set_fixed("submap", 0)

    pg = PoseGraph2D(
        backend=backend,
        sig_xy=0.1,
        sig_theta=np.deg2rad(5.0),
    )

    submap0 = PoseGraphSubmap(id=0, pose=Pose2(0.0, 0.0, 0.0))
    pg.backend.add_submap(submap0)

    node0 = PoseGraphNode(id=0, time=0.0, pose=Pose2(0.8, 0.1, 0.05))
    pg.backend.add_node(node0)

    c0 = PoseGraphConstraint(
        submap_id=0,
        node_id=0,
        pose=ConstraintPose2D(
            relative_pose=Pose2(1.0, 0.0, 0.0),
            translation_weight=10.0,
            rotation_weight=10.0,
            match_score=1.0,
        ),
        tag=INTRA_SUBMAP,
    )
    pg.backend.add_constraint(c0)

    node1 = PoseGraphNode(id=1, time=1.0, pose=Pose2(2.2, -0.2, 0.08))
    pg.backend.add_node(node1)

    c1 = PoseGraphConstraint(
        submap_id=0,
        node_id=1,
        pose=ConstraintPose2D(
            relative_pose=Pose2(2.0, 0.0, 0.0),
            translation_weight=10.0,
            rotation_weight=10.0,
            match_score=1.0,
        ),
        tag=INTRA_SUBMAP,
    )
    pg.backend.add_constraint(c1)

    print("Solving...")
    pg.solve(max_iters=50)

    print("Optimized poses:")
    result = pg.backend.get_optimized_poses()
    for key, pose in result.items():
        print(key, pose)

    summary = pg.backend.get_last_summary()
    if summary is not None:
        try:
            print(summary.BriefReport())
        except Exception:
            print(summary)


if __name__ == "__main__":
    main()