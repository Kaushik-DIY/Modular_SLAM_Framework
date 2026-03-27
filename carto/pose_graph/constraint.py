from dataclasses import dataclass
import numpy as np
from carto.common.types import Pose2


@dataclass
class PoseGraphNode:
    """Trajectory node (scan)."""
    id: int
    time: float
    pose: Pose2


@dataclass
class PoseGraphSubmap:
    """Submap node (optimized globally)."""
    id: int
    pose: Pose2


@dataclass
class PoseGraphConstraint:
    """
    Generic constraint between either:
      - submap -> node   (intra)
      - submap -> submap (inter)
      - node -> node     (optional)
    """
    type_from: str  # 'node' or 'submap'
    id_from: int
    type_to: str    # 'node' or 'submap'
    id_to: int

    relative_pose: Pose2
    information: np.ndarray  # 3x3