from dataclasses import dataclass
from typing import Optional
from carto.common.types import Pose2


INTRA_SUBMAP = "INTRA_SUBMAP"
INTER_SUBMAP = "INTER_SUBMAP"


@dataclass
class PoseGraphNode:
    id: int
    time: float
    pose: Pose2
    # local_pose: pose of this node in its primary insertion submap's local frame.
    # Equivalent to NodeSpec2D.local_pose_2d in optimization_problem_2d.cc.
    # Used for local trajectory regularization: z_ij = local_i^{-1} * local_j
    # between consecutive nodes. Fixed measurement — never optimized.
    local_pose: Optional[Pose2] = None


@dataclass
class PoseGraphSubmap:
    id: int
    pose: Pose2


@dataclass
class ConstraintPose2D:
    """
    Cartographer-style relative pose constraint payload.
    """
    relative_pose: Pose2
    translation_weight: float
    rotation_weight: float
    match_score: float = 1.0


@dataclass
class PoseGraphConstraint:
    """
    Cartographer-style 2D constraint:
      submap_id -> node_id
    tagged as either INTRA_SUBMAP or INTER_SUBMAP.
    """
    submap_id: int
    node_id: int
    pose: ConstraintPose2D
    tag: str