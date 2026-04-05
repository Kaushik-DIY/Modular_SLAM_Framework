from dataclasses import dataclass
from carto.common.types import Pose2


INTRA_SUBMAP = "INTRA_SUBMAP"
INTER_SUBMAP = "INTER_SUBMAP"


@dataclass
class PoseGraphNode:
    id: int
    time: float
    pose: Pose2


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