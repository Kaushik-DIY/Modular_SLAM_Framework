from __future__ import annotations

from typing import Optional
import numpy as np

from slam_core.common.types import Pose2
from slam_core.matching.core import MatchResult
from slam_core.optimisers.gn_lm import GaussNewtonLM, GNLMConfig
from slam_core.matching.scan_to_submap.types import (
    ScanToSubmapBackendConfig,
    SubmapMatchRequest,
    SubmapMatchResponse,
)
from slam_core.matching.scan_to_submap.backend_base import IScanToSubmapBackend
from slam_core.matching.scan_to_submap.submaps import SubmapBuilder2D
from slam_core.matching.scan_to_submap.two_stage_backend import TwoStageBruteForceSubmapBackend
from slam_core.matching.scan_to_submap.branch_and_bound_backend import BranchAndBoundSubmapBackend


class ScanToSubmapMatcher:
    name = "scan_to_submap"

    def __init__(
        self,
        submap_builder: SubmapBuilder2D,
        backend_config: ScanToSubmapBackendConfig,
        refine_solver=None,
    ) -> None:
        self.submap_builder = submap_builder
        self.backend_config = backend_config

        # Retained for compatibility with backends that still use the legacy
        # refinement path. The active local path is now PyCeres-backed inside
        # TwoStageBruteForceSubmapBackend.
        self.refine_solver = refine_solver or GaussNewtonLM(
            GNLMConfig(
                iters=int(self.backend_config.refine_iters),
                damping=float(self.backend_config.refine_damping),
                eps_stop=float(self.backend_config.refine_eps_stop),
                step_clip=np.array(
                    [
                        float(self.backend_config.refine_step_clip_xy),
                        float(self.backend_config.refine_step_clip_xy),
                        float(self.backend_config.refine_step_clip_th),
                    ],
                    dtype=float,
                ),
                verbose=bool(self.backend_config.refine_verbose),
            )
        )

        self.backend = self._build_backend()

    def _build_backend(self) -> IScanToSubmapBackend:
        if self.backend_config.backend_type == "two_stage_bruteforce":
            return TwoStageBruteForceSubmapBackend(
                submap_builder=self.submap_builder,
                config=self.backend_config,
                refine_solver=self.refine_solver,
            )

        if self.backend_config.backend_type == "branch_and_bound":
            return BranchAndBoundSubmapBackend(
                config=self.backend_config,
                refine_solver=self.refine_solver,
            )

        raise ValueError(f"Unsupported backend type: {self.backend_config.backend_type}")

    def match_against_submap(self, request: SubmapMatchRequest) -> SubmapMatchResponse:
        return self.backend.match(request)

    def _response_to_match_result(self, response: SubmapMatchResponse) -> MatchResult:
        """
        Convert backend response into the framework-level MatchResult while
        preserving backend diagnostics for runner-level introspection.
        """
        extra = response.debug.extra if response.debug is not None else {}

        refine_delta = extra.get("refine_delta", None)
        refine_inliers = extra.get("refine_inliers", None)

        return MatchResult(
            method="scan_to_submap",
            success=bool(response.success),
            pose_world=response.pose_world,
            score=float(response.score),
            refine_delta=refine_delta,
            inliers=refine_inliers,
            debug=response.debug,
        )

    def match(
        self,
        t: float,
        scan_points_local: np.ndarray,
        predicted_pose_world: Pose2,
        odom_pose_world: Optional[Pose2] = None,
    ) -> MatchResult:
        active_submaps = self.submap_builder.get_active_submaps()
        if not active_submaps:
            return MatchResult(
                method="scan_to_submap",
                success=False,
                pose_world=predicted_pose_world,
                score=-1.0,
                refine_delta=None,
                inliers=None,
                debug=None,
            )

        # Cartographer convention:
        # match against the OLDEST active submap, not the newest one.
        # The oldest active submap is richer because it has accumulated more
        # scans and therefore provides the more reliable matching target.
        target_submap = active_submaps[0]

        request = SubmapMatchRequest(
            scan_points_local=np.asarray(scan_points_local, dtype=float),
            predicted_pose_world=predicted_pose_world,
            submap_pose_world=target_submap.pose_world,
            submap=target_submap,
            timestamp=float(t),
            odom_pose_world=odom_pose_world,
        )

        response = self.match_against_submap(request)
        return self._response_to_match_result(response)

    def update_submaps(
        self,
        pose_world: Pose2,
        scan_points_local: np.ndarray,
        t: float,
    ) -> bool:
        _ = t
        return self.submap_builder.insert_scan(pose_world, scan_points_local)

    def update_target(
        self,
        pose_world: Pose2,
        scan_points_local: np.ndarray,
        t: float,
    ) -> bool:
        return self.update_submaps(
            pose_world=pose_world,
            scan_points_local=scan_points_local,
            t=t,
        )

    def get_active_submaps(self):
        return self.submap_builder.get_active_submaps()

    def get_last_inserted_submaps(self):
        return self.submap_builder.get_last_inserted_submaps()

    def get_active_targets(self):
        return self.get_active_submaps()

    def get_last_inserted_targets(self):
        return self.get_last_inserted_submaps()