"""
Fusion adapter layer (Phase 6).

Thin wrappers that drive the existing ORB-SLAM and LiDAR pipelines as services
and expose their loop-candidate detectors in **propose-only** mode — the
upstream pipelines' own geometric verification and graph optimization are not
invoked in fusion modes (CLAUDE.md §2.7, §6.2). Underlying pipeline code is not
modified; these adapters intercept at clean seams via dependency injection.
"""

from __future__ import annotations

from slam_core.fusion.adapters.types import (
    LoopProposal,
    KeyframeRecord,
    FrontendStep,
)
from slam_core.fusion.adapters.orb_loop_proposer import OrbLoopProposer
from slam_core.fusion.adapters.lidar_loop_proposer import LidarLoopProposer
from slam_core.fusion.adapters.visual_frontend_service import VisualFrontendService
from slam_core.fusion.adapters.lidar_frontend_service import LidarFrontendService

__all__ = [
    "LoopProposal",
    "KeyframeRecord",
    "FrontendStep",
    "OrbLoopProposer",
    "LidarLoopProposer",
    "VisualFrontendService",
    "LidarFrontendService",
]
