"""Shared adapter-layer record types (Phase 6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from slam_core.common.types import Pose2
from slam_core.loop_closure import ClosureTarget


@dataclass
class LoopProposal:
    """A proposed loop candidate, emitted *without* any geometric verification."""

    candidate_id: Union[int, str]
    score: float
    source: str                      # "orb" | "lidar"
    target: Optional[ClosureTarget] = None  # populated by the LiDAR proposer


@dataclass
class KeyframeRecord:
    """Normalized visual front-end keyframe output."""

    id: int
    timestamp: float
    pose: np.ndarray                 # 4x4 SE(3) camera pose (T_world_cam)
    keypoints: Optional[np.ndarray] = None
    descriptors: Optional[np.ndarray] = None
    points3d: Optional[np.ndarray] = None
    is_keyframe: bool = True


@dataclass
class FrontendStep:
    """Normalized LiDAR front-end per-scan output."""

    timestamp: float
    pose: Pose2                      # world pose from the front-end
    rel_pose: Pose2                  # relative to the previous step
    scan: Optional[np.ndarray]       # (M, 2) scan in sensor frame
    is_keyframe: bool = False
