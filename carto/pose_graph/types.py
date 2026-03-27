# carto/pose_graph/types.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np
from carto.common.types import Pose2


class ConstraintType(str, Enum):
    INTRA_SUBMAP = "INTRA_SUBMAP"
    INTER_SUBMAP = "INTER_SUBMAP"   # loop closure later
    ODOM = "ODOM"                   # optional later


@dataclass
class NodeSpec2D:
    node_id: int
    time: float
    initial_pose_world: Pose2


@dataclass
class SubmapSpec2D:
    submap_id: int
    initial_pose_world: Pose2
    finished: bool = False


@dataclass
class Constraint2D:
    submap_id: int
    node_id: int
    # Measurement: z = (submap^-1 ⊕ node)  i.e., pose of node in submap frame
    z_submap_node: Pose2
    # Information matrix (3x3) for [x,y,theta]
    information: np.ndarray
    ctype: ConstraintType = ConstraintType.INTRA_SUBMAP