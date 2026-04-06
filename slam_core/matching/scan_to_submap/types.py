from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
import numpy as np

from slam_core.common.types import Pose2
from slam_core.matching.core import MatchResult


SearchBackendType = Literal["two_stage_bruteforce", "branch_and_bound"]


@dataclass
class SubmapSearchWindow:
    """
    Discrete search-window description used by scan-to-submap backends.
    """
    xy_window: float
    theta_window: float
    xy_step: float
    theta_step: float
    level: int = 0


@dataclass
class ScanToSubmapBackendConfig:
    """
    Shared configuration for scan-to-submap matching backends.

    This configuration is intentionally backend-agnostic at the top level.
    Backend-specific fields are included where required so that the public
    matcher API remains stable across multiple internal implementations.
    """
    backend_type: SearchBackendType = "two_stage_bruteforce"

    # Generic acceptance / validity
    min_score: float = 0.58
    min_valid: int = 20

    # Generic precomputation support
    precomp_levels: int = 3

    # Refinement
    do_refine: bool = True
    max_match_points: int = 60
    max_refine_points: int = 180
    refine_min_points: int = 20
    refine_w_trans: float = 1.0
    refine_w_rot: float = 1.0

    refine_iters: int = 8
    refine_damping: float = 1e-3
    refine_eps_stop: float = 1e-6
    refine_step_clip_xy: float = 0.10
    refine_step_clip_th: float = float(np.deg2rad(5.0))
    refine_verbose: bool = False

    # Two-stage search windows
    coarse: Optional[SubmapSearchWindow] = None
    fine: Optional[SubmapSearchWindow] = None

    # Branch-and-bound specific
    bnb_depth_limit: int = 8
    bnb_min_rotational_step: float = 0.02
    bnb_branching: int = 4


@dataclass
class SubmapMatchRequest:
    """
    Explicit scan-to-submap match request.

    The caller must specify the chosen target submap and the initial pose
    estimate in the world frame. This keeps the matching layer generic and
    independent of SLAM-specific candidate-generation policy.
    """
    scan_points_local: np.ndarray
    predicted_pose_world: Pose2
    submap_pose_world: Pose2
    submap: Any
    timestamp: float = 0.0
    odom_pose_world: Optional[Pose2] = None


@dataclass
class SubmapMatchDebug:
    """
    Backend debug information for analysis and benchmarking.
    """
    backend_type: SearchBackendType
    coarse_score: float
    refined: bool
    num_points_match: int
    num_points_refine: int
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubmapMatchResponse:
    """
    Explicit backend response for one chosen submap target.
    """
    success: bool
    score: float
    pose_world: Pose2
    debug: SubmapMatchDebug


def response_to_match_result(resp: SubmapMatchResponse) -> MatchResult:
    """
    Convert the explicit backend response into the framework-level MatchResult.
    """
    refine_delta = resp.debug.extra.get("refine_delta", None)
    refine_inliers = resp.debug.extra.get("refine_inliers", None)

    return MatchResult(
        method="scan_to_submap",
        success=bool(resp.success),
        pose_world=resp.pose_world,
        score=float(resp.score),
        refine_delta=refine_delta,
        inliers=refine_inliers,
    )