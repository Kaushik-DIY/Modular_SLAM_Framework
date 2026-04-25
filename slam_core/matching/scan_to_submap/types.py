from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional
import numpy as np

from slam_core.common.types import Pose2
from slam_core.matching.core import MatchResult


SearchBackendType = Literal["two_stage_bruteforce", "branch_and_bound"]


@dataclass
class SubmapSearchWindow:
    """
    Search window specification for discrete correlative scan matching.
    """
    xy_window: float = 1.0
    theta_window: float = 0.35
    xy_step: float = 0.05
    theta_step: float = 0.02
    level: int = 0


@dataclass
class ScanToSubmapBackendConfig:
    """
    Shared configuration for scan-to-submap matching backends.

    Notes
    -----
    The same structure is used for both local tracking and loop-closure
    verification. The backend chooses the appropriate acceptance threshold
    depending on whether the request is a constrained local match or a
    broad full-submap search.
    """
    backend_type: SearchBackendType = "two_stage_bruteforce"

    # Local constrained-search threshold.
    min_score: float = 0.66

    # Full-submap / global-localization threshold.
    global_localization_min_score: float = 0.72

    min_valid: int = 20
    precomp_levels: int = 3

    # Refinement controls.
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

    # Search windows.
    coarse: Optional[SubmapSearchWindow] = None
    fine: Optional[SubmapSearchWindow] = None

    # Branch-and-bound controls.
    bnb_depth_limit: int = 8
    bnb_min_rotational_step: float = 0.02
    bnb_branching: int = 4


@dataclass
class SubmapMatchRequest:
    """
    Explicit scan-to-submap match request.

    Parameters
    ----------
    match_full_submap:
        False -> constrained local search around the predicted pose.
        True  -> broad full-submap search for loop-closure style
                 global localization.
    """
    scan_points_local: np.ndarray
    predicted_pose_world: Pose2
    submap_pose_world: Pose2
    submap: Any
    timestamp: float = 0.0
    odom_pose_world: Optional[Pose2] = None
    match_full_submap: bool = False


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
    Convert the explicit backend response into the framework-level matcher output.
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