"""
RTAB-inspired multi-modal SLAM fusion package (v1).

Phase 1 (Foundation) exports: configuration, the keyframe Signature, the
SE(3)->SE(2) projection, soft timestamp synchronization, the synthetic 2D
LiDAR utility, and the TUM RGB-D + synthetic-LiDAR dataset loader.

See CLAUDE.md §4 and RTAB_inspired_implementation_plan.md §13 for the phase plan.
"""

from __future__ import annotations

from slam_core.fusion.config import FusionConfig, Mode
from slam_core.fusion.signature import Signature, project_pose3d_to_pose2
from slam_core.fusion.sync import FusedSample, SoftSync
from slam_core.fusion.lidar_synth import synthesize_2d_scan
from slam_core.fusion.dataset import FusionDataset, FusionFrame
from slam_core.fusion.memory import MemoryManager, InsertResult, TickResult

__all__ = [
    "FusionConfig",
    "Mode",
    "Signature",
    "project_pose3d_to_pose2",
    "FusedSample",
    "SoftSync",
    "synthesize_2d_scan",
    "FusionDataset",
    "FusionFrame",
    "MemoryManager",
    "InsertResult",
    "TickResult",
]
