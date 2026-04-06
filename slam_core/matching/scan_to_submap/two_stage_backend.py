from __future__ import annotations

import numpy as np

from slam_core.matching.scan_to_submap.backend_base import IScanToSubmapBackend
from slam_core.matching.scan_to_submap.types import (
    ScanToSubmapBackendConfig,
    SubmapMatchDebug,
    SubmapMatchRequest,
    SubmapMatchResponse,
)
from slam_core.matching.scan_to_submap_old import ScanToSubmapMatcher as LegacyScanToSubmapMatcher


class TwoStageBruteForceSubmapBackend(IScanToSubmapBackend):
    """
    Compatibility backend that delegates to the original stable
    scan_to_submap implementation.

    This preserves the exact local scan-to-submap behavior that was already
    working in the thesis framework, while the new package provides the stable
    public API and allows the branch-and-bound backend to coexist cleanly.
    """

    def __init__(self, submap_builder, config: ScanToSubmapBackendConfig, refine_solver=None) -> None:
        self.submap_builder = submap_builder
        self.config = config
        self.refine_solver = refine_solver

        corr_params = {
            "min_score": float(config.min_score),
            "max_match_points": int(config.max_match_points),
            "min_valid": int(config.min_valid),
            "precomp_levels": int(config.precomp_levels),
            "do_refine": bool(config.do_refine),
            "max_refine_points": int(config.max_refine_points),
            "refine_min_points": int(config.refine_min_points),
            "refine_w_trans": float(config.refine_w_trans),
            "refine_w_rot": float(config.refine_w_rot),
            "refine_iters": int(config.refine_iters),
            "refine_damping": float(config.refine_damping),
            "refine_eps_stop": float(config.refine_eps_stop),
            "refine_step_clip_xy": float(config.refine_step_clip_xy),
            "refine_step_clip_th": float(config.refine_step_clip_th),
            "refine_verbose": bool(config.refine_verbose),
        }

        if config.coarse is not None:
            corr_params.update(
                {
                    "coarse_level": int(config.coarse.level),
                    "coarse_xy_window": float(config.coarse.xy_window),
                    "coarse_th_window": float(config.coarse.theta_window),
                    "coarse_xy_step": float(config.coarse.xy_step),
                    "coarse_th_step": float(config.coarse.theta_step),
                }
            )

        if config.fine is not None:
            corr_params.update(
                {
                    "fine_level": int(config.fine.level),
                    "fine_xy_window": float(config.fine.xy_window),
                    "fine_th_window": float(config.fine.theta_window),
                    "fine_xy_step": float(config.fine.xy_step),
                    "fine_th_step": float(config.fine.theta_step),
                }
            )

        self.legacy = LegacyScanToSubmapMatcher(
            submap_builder=self.submap_builder,
            corr_params=corr_params,
        )

        if self.refine_solver is not None:
            self.legacy.refine_solver = self.refine_solver

    def match(self, request: SubmapMatchRequest) -> SubmapMatchResponse:
        legacy_result = self.legacy.match(
            t=float(request.timestamp),
            scan_points_local=np.asarray(request.scan_points_local, dtype=float),
            predicted_pose_world=request.predicted_pose_world,
            odom_pose_world=request.odom_pose_world,
        )

        refine_delta = getattr(legacy_result, "refine_delta", None)
        refine_inliers = getattr(legacy_result, "inliers", None)

        debug = SubmapMatchDebug(
            backend_type="two_stage_bruteforce",
            coarse_score=float(legacy_result.score),
            refined=refine_delta is not None,
            num_points_match=int(np.asarray(request.scan_points_local).shape[0]),
            num_points_refine=0 if refine_inliers is None else int(refine_inliers),
            extra={
                "legacy_method": getattr(legacy_result, "method", "scan_to_submap"),
                "debug_info": getattr(legacy_result, "debug_info", {}),
                "refine_delta": refine_delta,
                "refine_inliers": refine_inliers,
            },
        )

        return SubmapMatchResponse(
            success=bool(legacy_result.success),
            score=float(legacy_result.score),
            pose_world=legacy_result.pose_world,
            debug=debug,
        )