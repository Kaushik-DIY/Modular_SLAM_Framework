from __future__ import annotations

import numpy as np

from slam_core.common.se2 import inverse_pose, pose_compose, wrap_angle
from slam_core.common.types import Pose2
from slam_core.matching.scan_to_submap.backend_base import IScanToSubmapBackend
from slam_core.matching.scan_to_submap.local_pyceres_matcher import (
    PyCeresLocalScanMatcher2D,
)
from slam_core.matching.scan_to_submap.types import (
    ScanToSubmapBackendConfig,
    SubmapMatchDebug,
    SubmapMatchRequest,
    SubmapMatchResponse,
    SubmapSearchWindow,
)
from slam_core.matching.scan_to_submap_old import correlative_match_two_stage


class TwoStageBruteForceSubmapBackend(IScanToSubmapBackend):
    """
    Active local scan-to-submap backend.

    Cartographer-like local flow:
      1. transform the predicted world pose into the target submap frame
      2. optionally improve that estimate with constrained correlative search
      3. refine continuously with a PyCeres local scan matcher
         (occupied-space + translation prior + rotation prior)

    The correlative stage provides:
      - an initializer
      - a confidence signal

    The refined pose is the actual local estimate.
    """

    def __init__(
        self,
        submap_builder,
        config: ScanToSubmapBackendConfig,
        refine_solver=None,
    ) -> None:
        self.submap_builder = submap_builder
        self.config = config
        self.refine_solver = refine_solver

        self.coarse_window = config.coarse or SubmapSearchWindow(
            xy_window=0.8,
            theta_window=0.3,
            xy_step=0.20,
            theta_step=0.08,
            level=2,
        )
        self.fine_window = config.fine or SubmapSearchWindow(
            xy_window=0.25,
            theta_window=0.12,
            xy_step=0.05,
            theta_step=0.02,
            level=0,
        )

        # Local continuous matcher.
        self.local_matcher = PyCeresLocalScanMatcher2D(
            translation_weight=float(self.config.refine_w_trans),
            rotation_weight=float(self.config.refine_w_rot),
            max_num_iterations=int(self.config.refine_iters),
            num_threads=1,
            minimizer_progress_to_stdout=bool(self.config.refine_verbose),
            function_tolerance=float(self.config.refine_eps_stop),
            gradient_tolerance=1e-10,
            parameter_tolerance=1e-8,
            linear_solver_type="DENSE_QR",
            invalid_point_residual=1.0,
        )

    def match(self, request: SubmapMatchRequest) -> SubmapMatchResponse:
        submap = request.submap
        submap_pose_world = request.submap_pose_world

        # Search policy:
        # - local tracking uses a constrained window around prediction
        # - full-submap mode is reserved for broad-search use cases
        if bool(request.match_full_submap):
            pred_sub = Pose2(0.0, 0.0, 0.0)
            coarse_window = SubmapSearchWindow(
                xy_window=0.5 * float(submap.grid.size_m),
                theta_window=float(np.pi),
                xy_step=float(self.coarse_window.xy_step),
                theta_step=float(self.coarse_window.theta_step),
                level=int(self.coarse_window.level),
            )
            fine_window = self.fine_window
        else:
            pred_sub = pose_compose(
                inverse_pose(submap_pose_world),
                request.predicted_pose_world,
            )
            coarse_window = self.coarse_window
            fine_window = self.fine_window

        points_local = np.asarray(request.scan_points_local, dtype=float)

        # Downsample only for the discrete correlative stage.
        points_match = points_local
        max_match_pts = int(self.config.max_match_points)
        if points_match.shape[0] > max_match_pts:
            stride = max(1, points_match.shape[0] // max_match_pts)
            points_match = points_match[::stride]

        prob_img = submap.grid.probability().astype(np.float32)

        # Discrete correlative search refines only the initializer.
        coarse_pose_sub, coarse_score = correlative_match_two_stage(
            prob_img=prob_img,
            grid_origin_xy=np.asarray(submap.grid.origin_world, dtype=float),
            res=float(submap.grid.res),
            points_local=points_match,
            initial_submap_pose=pred_sub,
            min_valid=int(self.config.min_valid),
            precomp_levels=int(self.config.precomp_levels),
            coarse_level=int(coarse_window.level),
            coarse_xy_window=float(coarse_window.xy_window),
            coarse_th_window=float(coarse_window.theta_window),
            coarse_xy_step=float(coarse_window.xy_step),
            coarse_th_step=float(coarse_window.theta_step),
            fine_level=int(fine_window.level),
            fine_xy_window=float(fine_window.xy_window),
            fine_th_window=float(fine_window.theta_window),
            fine_xy_step=float(fine_window.xy_step),
            fine_th_step=float(fine_window.theta_step),
        )

        raw_coarse_score = float(coarse_score) if np.isfinite(float(coarse_score)) else -1.0
        coarse_score_f = float(raw_coarse_score if raw_coarse_score >= 0.0 else -1.0)

        coarse_available = (
            coarse_pose_sub is not None
            and coarse_score_f >= 0.0
            and np.all(
                np.isfinite(
                    np.array(
                        [coarse_pose_sub.x, coarse_pose_sub.y, coarse_pose_sub.theta],
                        dtype=float,
                    )
                )
            )
        )

        coarse_trusted = bool(coarse_available and (coarse_score_f >= float(self.config.min_score)))

        if coarse_trusted:
            initial_refine_pose_sub = coarse_pose_sub
            used_pred_initializer = False
        else:
            initial_refine_pose_sub = pred_sub
            used_pred_initializer = True

        # Continuous local refinement: this is the actual local estimate.
        pyceres_result = self.local_matcher.match(
            grid=submap.grid,
            points_local=points_local,
            pred_pose_sub=pred_sub,
            initial_pose_sub=initial_refine_pose_sub,
            min_valid_points=int(self.config.refine_min_points),
        )

        final_pose_sub = pyceres_result.pose_sub
        final_pose_world = pose_compose(submap_pose_world, final_pose_sub)

        refine_delta = np.array(
            [
                float(final_pose_sub.x - pred_sub.x),
                float(final_pose_sub.y - pred_sub.y),
                float(wrap_angle(final_pose_sub.theta - pred_sub.theta)),
            ],
            dtype=float,
        )

        summary = self.local_matcher.get_last_summary()
        if summary is not None:
            try:
                summary_brief = summary.BriefReport()
            except Exception:
                summary_brief = str(summary)
        else:
            summary_brief = None

        debug = SubmapMatchDebug(
            backend_type="two_stage_bruteforce",
            coarse_score=float(coarse_score_f),
            refined=True,
            num_points_match=int(points_match.shape[0]),
            num_points_refine=min(int(points_local.shape[0]), int(self.config.max_refine_points)),
            extra={
                "match_full_submap": bool(request.match_full_submap),
                "coarse_trusted": bool(coarse_available),
                "coarse_valid": bool(coarse_trusted),
                "used_pred_initializer": bool(used_pred_initializer),
                "raw_coarse_score": float(raw_coarse_score),
                "coarse_pose_sub": coarse_pose_sub,
                "initial_refine_pose_sub": initial_refine_pose_sub,
                "final_pose_sub": final_pose_sub,
                "refined_score": float(pyceres_result.refined_score),
                "refine_delta": refine_delta,
                "refine_inliers": int(pyceres_result.valid_points),
                "pyceres_reason": str(pyceres_result.reason),
                "pyceres_summary": summary_brief,
            },
        )

        if not pyceres_result.success:
            return SubmapMatchResponse(
                success=False,
                score=-1.0,
                pose_world=request.predicted_pose_world,
                debug=debug,
            )

        # Top-level score remains the correlative confidence signal.
        # The refined pose is the actual local estimate.
        return SubmapMatchResponse(
            success=True,
            score=float(coarse_score_f),
            pose_world=final_pose_world,
            debug=debug,
        )