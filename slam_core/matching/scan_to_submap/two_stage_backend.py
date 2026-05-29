from __future__ import annotations

import numpy as np

from slam_core.common.se2 import inverse_pose, pose_compose, wrap_angle
from slam_core.common.types import Pose2
from slam_core.optimisers.gn_lm import GaussNewtonLM, GNLMConfig
from slam_core.matching.scan_to_submap.backend_base import IScanToSubmapBackend
from slam_core.matching.scan_to_submap.types import (
    ScanToSubmapBackendConfig,
    SubmapMatchDebug,
    SubmapMatchRequest,
    SubmapMatchResponse,
    SubmapSearchWindow,
)
from slam_core.matching.scan_to_submap.refine import (
    CartoRefinementProblem,
    refine_pose_submap,
)
from slam_core.matching.scan_to_submap.correlative import correlative_match_two_stage


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
        # Default: native GaussNewtonLM on the occupancy grid (no pyceres dep).
        # g2o is reserved for the pose graph; the 3-DOF local match stays native.
        self.local_refine_backend = str(
            getattr(self.config, "local_refine_backend", "native")
        )
        # Vectorized correlative search (same window/scoring, batched NumPy) — opt-in.
        self.use_vectorized_search = bool(
            getattr(self.config, "use_vectorized_search", False)
        )

        self._native_solver = refine_solver or GaussNewtonLM(
            GNLMConfig(
                iters=int(self.config.refine_iters),
                damping=float(self.config.refine_damping),
                eps_stop=float(self.config.refine_eps_stop),
                step_clip=np.array(
                    [
                        float(self.config.refine_step_clip_xy),
                        float(self.config.refine_step_clip_xy),
                        float(self.config.refine_step_clip_th),
                    ],
                    dtype=float,
                ),
                verbose=bool(self.config.refine_verbose),
            )
        )

        self.local_matcher = None
        if self.local_refine_backend == "pyceres":
            # Optional legacy path. Imported lazily so the package has no hard
            # pyceres dependency for the default native flow.
            from slam_core.matching.scan_to_submap.local_pyceres_matcher import (
                PyCeresLocalScanMatcher2D,
            )

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

    @staticmethod
    def _score_pose_on_grid(grid, points_local: np.ndarray, pose_sub: Pose2):
        """Mean occupancy probability at the scan endpoints under pose_sub.

        Returns (mean_prob, valid_point_count). Mirrors the diagnostic score
        the pyceres local matcher reported, so downstream behaviour is unchanged.
        """
        pts = np.asarray(points_local, dtype=float)
        if pts.shape[0] == 0:
            return -1.0, 0

        prob_grid = grid.probability()
        c = np.cos(float(pose_sub.theta))
        s = np.sin(float(pose_sub.theta))

        qx = c * pts[:, 0] - s * pts[:, 1] + float(pose_sub.x)
        qy = s * pts[:, 0] + c * pts[:, 1] + float(pose_sub.y)

        gx = np.floor((qx - float(grid.origin_world[0])) / float(grid.res)).astype(int)
        gy = np.floor((qy - float(grid.origin_world[1])) / float(grid.res)).astype(int)

        mask = (gx >= 0) & (gx < int(grid.w)) & (gy >= 0) & (gy < int(grid.h))
        n = int(mask.sum())
        if n == 0:
            return -1.0, 0
        return float(prob_grid[gy[mask], gx[mask]].mean()), n

    def _refine_native(
        self,
        grid,
        points_local: np.ndarray,
        pred_pose_sub: Pose2,
        initial_pose_sub: Pose2,
        min_valid_points: int,
    ):
        """Native GaussNewtonLM local refinement on the occupancy grid.

        Returns (success, pose_sub, valid_points, refined_score, reason).
        """
        pts = np.asarray(points_local, dtype=float)
        if pts.shape[0] == 0:
            return False, pred_pose_sub, 0, -1.0, "empty_scan"

        problem = CartoRefinementProblem(
            grid=grid,
            pts_local=pts,
            pred_pose_sub=np.array(
                [pred_pose_sub.x, pred_pose_sub.y, pred_pose_sub.theta], dtype=float
            ),
            min_points=int(min_valid_points),
            w_trans=float(self.config.refine_w_trans),
            w_rot=float(self.config.refine_w_rot),
        )

        pose_sub = refine_pose_submap(self._native_solver, problem, initial_pose_sub)

        finite = np.all(
            np.isfinite([pose_sub.x, pose_sub.y, pose_sub.theta])
        )
        refined_score, valid_points = self._score_pose_on_grid(grid, pts, pose_sub)

        if not finite:
            return False, pred_pose_sub, int(valid_points), float(refined_score), "nonfinite_solution"
        if int(valid_points) < int(min_valid_points):
            return False, pred_pose_sub, int(valid_points), float(refined_score), "insufficient_valid_support"

        return True, pose_sub, int(valid_points), float(refined_score), "ok"

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
            vectorized=self.use_vectorized_search,
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

        # Legacy Hector behaviour: reject low-confidence matches (FALLBACK to the
        # predicted pose) instead of refining from the prediction. Avoids accepting
        # ambiguous matches during fast turns. Off by default (Cartographer refines).
        if (not coarse_trusted) and bool(getattr(self.config, "reject_below_min_score", False)) \
                and not bool(request.match_full_submap):
            debug = SubmapMatchDebug(
                backend_type="two_stage_bruteforce",
                coarse_score=float(coarse_score_f),
                refined=False,
                num_points_match=int(points_match.shape[0]),
                num_points_refine=0,
                extra={
                    "match_full_submap": bool(request.match_full_submap),
                    "coarse_trusted": bool(coarse_available),
                    "coarse_valid": bool(coarse_trusted),
                    "raw_coarse_score": float(raw_coarse_score),
                    "refine_reason": "coarse_below_min_score",
                },
            )
            return SubmapMatchResponse(
                success=False,
                score=float(coarse_score_f),
                pose_world=request.predicted_pose_world,
                debug=debug,
            )

        if coarse_trusted:
            initial_refine_pose_sub = coarse_pose_sub
            used_pred_initializer = False
        else:
            initial_refine_pose_sub = pred_sub
            used_pred_initializer = True

        # Continuous local refinement: this is the actual local estimate.
        if self.local_refine_backend == "native":
            # Downsample for the native GN refine (matches legacy Hector behaviour).
            refine_pts = points_local
            max_refine = int(self.config.max_refine_points)
            if refine_pts.shape[0] > max_refine:
                stride = max(1, refine_pts.shape[0] // max_refine)
                refine_pts = refine_pts[::stride]

            (
                refine_success,
                final_pose_sub,
                refine_valid,
                refined_score,
                refine_reason,
            ) = self._refine_native(
                grid=submap.grid,
                points_local=refine_pts,
                pred_pose_sub=pred_sub,
                initial_pose_sub=initial_refine_pose_sub,
                min_valid_points=int(self.config.refine_min_points),
            )
            summary_brief = None
        else:
            pyceres_result = self.local_matcher.match(
                grid=submap.grid,
                points_local=points_local,
                pred_pose_sub=pred_sub,
                initial_pose_sub=initial_refine_pose_sub,
                min_valid_points=int(self.config.refine_min_points),
            )
            refine_success = bool(pyceres_result.success)
            final_pose_sub = pyceres_result.pose_sub
            refine_valid = int(pyceres_result.valid_points)
            refined_score = float(pyceres_result.refined_score)
            refine_reason = str(pyceres_result.reason)
            summary = self.local_matcher.get_last_summary()
            if summary is not None:
                try:
                    summary_brief = summary.BriefReport()
                except Exception:
                    summary_brief = str(summary)
            else:
                summary_brief = None

        final_pose_world = pose_compose(submap_pose_world, final_pose_sub)

        refine_delta = np.array(
            [
                float(final_pose_sub.x - pred_sub.x),
                float(final_pose_sub.y - pred_sub.y),
                float(wrap_angle(final_pose_sub.theta - pred_sub.theta)),
            ],
            dtype=float,
        )

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
                "refined_score": float(refined_score),
                "refine_delta": refine_delta,
                "refine_inliers": int(refine_valid),
                "refine_backend": self.local_refine_backend,
                "refine_reason": str(refine_reason),
                "refine_summary": summary_brief,
            },
        )

        if not refine_success:
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