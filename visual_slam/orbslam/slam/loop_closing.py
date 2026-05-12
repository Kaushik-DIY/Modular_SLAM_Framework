"""
Loop-closing orchestration for the sparse map.
This module detects loops, verifies geometry, corrects drift, and triggers map optimization.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import threading
from typing import Optional

import g2o
import numpy as np

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.bow_matcher import BoWGuidedMatcher
from visual_slam.orbslam.slam.essential_graph import (
    EssentialGraphResult,
    apply_correction_to_map_points,
    optimize_essential_graph_se3,
)
from visual_slam.orbslam.slam.feature_tracker_shared import FeatureTrackerShared
from visual_slam.orbslam.slam.frame import ensure_frame_feature_arrays
from visual_slam.orbslam.slam.geometry_matchers import ProjectionFuseDiagnostics, ProjectionMatcher
from visual_slam.orbslam.slam.global_ba import GlobalBAResult, GlobalBundleAdjuster
from visual_slam.orbslam.slam.keyframe import KeyFrame
from visual_slam.orbslam.slam.loop_detector import LoopDetector, LoopDetectorOutput
from visual_slam.orbslam.slam.rotation_histogram import RotationHistogram
from visual_slam.orbslam.slam.sim3_solver import Sim3Estimate, estimate_scale_fixed_sim3


# Track one covisibility group and its accumulated loop-consistency score.
@dataclass
class ConsistencyGroup:
    keyframes: set = field(default_factory=set)
    consistency: int = 0


# Store diagnostics for one loop-detection and correction attempt.
@dataclass
class LoopDiagnostics:
    candidates: int = 0
    accepted: int = 0
    rejected_by_bow: int = 0
    rejected_by_consistency: int = 0
    rejected_by_geometry: int = 0
    corrected_keyframes: int = 0
    corrected_points: int = 0
    fused_points: int = 0
    fusion_diagnostics: Optional[ProjectionFuseDiagnostics] = None
    optimization_result: Optional[EssentialGraphResult] = None
    global_ba_result: Optional[GlobalBAResult] = None
    global_ba_started: bool = False
    global_ba_success: bool = False
    global_ba_aborted: bool = False
    global_ba_reason: str = ""
    global_ba_num_keyframes: int = 0
    global_ba_num_map_points: int = 0
    global_ba_num_edges: int = 0
    global_ba_num_inliers: int = 0
    global_ba_num_outliers: int = 0
    global_ba_mean_error_before: Optional[float] = None
    global_ba_mean_error_after: Optional[float] = None
    global_ba_elapsed_sec: float = 0.0
    unavailable_reason: Optional[str] = None
    loop_debug_records: list[dict] = field(default_factory=list)
    candidate_pair_reports: list[dict] = field(default_factory=list)


# Accumulate consistent loop groups across successive loop queries.
class LoopGroupConsistencyChecker:
    def __init__(self, consistency_threshold: int = 3):
        self.consistent_groups: list[ConsistencyGroup] = []
        self.consistency_threshold = int(consistency_threshold)
        self.enough_consistent_candidates: list[KeyFrame] = []
        self.last_candidate_debug: dict[int, dict] = {}

    def clear_consistency_groups(self) -> None:
        self.consistent_groups = []
        self.enough_consistent_candidates = []
        self.last_candidate_debug = {}

    def check_candidates(self, current_keyframe: KeyFrame, candidate_keyframes: list[KeyFrame]) -> bool:
        self.enough_consistent_candidates = []
        self.last_candidate_debug = {}
        current_consistent_groups = []
        group_updated = [False] * len(self.consistent_groups)
        previous_group_ids = [
            sorted(int(getattr(kf, "kid", getattr(kf, "id", -1))) for kf in group.keyframes)
            for group in self.consistent_groups
        ]

        for candidate in candidate_keyframes:
            if candidate is None or candidate.is_bad():
                continue

            candidate_group = set(candidate.get_connected_keyframes())
            candidate_group.add(candidate)
            candidate_group_ids = sorted(
                int(getattr(kf, "kid", getattr(kf, "id", -1))) for kf in candidate_group
            )

            enough_consistent = False
            consistent_for_some_group = False
            best_overlap_count = 0
            best_consistency = 0

            for idx, previous_group in enumerate(self.consistent_groups):
                overlap = candidate_group.intersection(previous_group.keyframes)
                if overlap:
                    consistent_for_some_group = True
                    best_overlap_count = max(best_overlap_count, len(overlap))
                    current_consistency = previous_group.consistency + 1
                    best_consistency = max(best_consistency, current_consistency)

                    if not group_updated[idx]:
                        current_consistent_groups.append(
                            ConsistencyGroup(candidate_group, current_consistency)
                        )
                        group_updated[idx] = True

                    if (
                        current_consistency >= self.consistency_threshold
                        and not enough_consistent
                    ):
                        self.enough_consistent_candidates.append(candidate)
                        enough_consistent = True

            if not consistent_for_some_group:
                current_consistent_groups.append(ConsistencyGroup(candidate_group, 0))
                if self.consistency_threshold <= 0:
                    self.enough_consistent_candidates.append(candidate)
                    enough_consistent = True

            candidate_key = int(getattr(candidate, "kid", getattr(candidate, "id", -1)))
            self.last_candidate_debug[candidate_key] = {
                "current_group_kf_ids": candidate_group_ids,
                "candidate_group_kf_ids": candidate_group_ids,
                "previous_consistency_group_ids": previous_group_ids,
                "consistency_overlap_count": best_overlap_count,
                "consistency_count": best_consistency,
                "consistency_required": self.consistency_threshold,
                "passed_consistency": bool(enough_consistent),
            }

        self.consistent_groups = current_consistent_groups
        return len(self.enough_consistent_candidates) > 0


# Verify a loop candidate through matching, rigid alignment, and refinement.
class LoopGeometryChecker:
    def __init__(
        self,
        min_matches: int = Parameters.kLoopClosingGeometryCheckerMinKpsMatches,
        keyframe_database=None,
    ):
        self.min_matches = int(min_matches)
        self.keyframe_database = keyframe_database
        self.success_loop_kf: Optional[KeyFrame] = None
        self.success_sim3: Optional[Sim3Estimate] = None
        self.success_map_point_matches: list = []
        self.success_map_point_matches_idxs = np.array([], dtype=np.int32)
        self.success_loop_map_points = set()
        self.num_last_matches = 0
        self.num_last_inliers = 0
        self.last_error = None
        self.last_bow_guided_matching_available = False
        self.last_fallback_descriptor_matching = False
        self.last_match_diagnostics = None
        self.last_match_distances = np.array([], dtype=np.float32)
        self.last_guided_projection_matches = 0
        self.last_final_matches = 0
        self.last_candidate_reports: dict[int, dict] = {}

    def check_candidates(self, current_keyframe: KeyFrame, candidate_keyframes: list[KeyFrame]) -> bool:
        self.success_loop_kf = None
        self.success_sim3 = None
        self.success_map_point_matches = []
        self.success_map_point_matches_idxs = np.array([], dtype=np.int32)
        self.success_loop_map_points = set()
        self.num_last_matches = 0
        self.num_last_inliers = 0
        self.last_error = None
        self.last_bow_guided_matching_available = False
        self.last_fallback_descriptor_matching = False
        self.last_match_diagnostics = None
        self.last_match_distances = np.array([], dtype=np.float32)
        self.last_guided_projection_matches = 0
        self.last_final_matches = 0
        self.last_candidate_reports = {}

        for candidate in candidate_keyframes:
            if candidate is None or candidate is current_keyframe or candidate.is_bad():
                continue

            candidate_key = int(getattr(candidate, "kid", getattr(candidate, "id", -1)))
            report = self._base_candidate_report(current_keyframe, candidate)
            self.last_candidate_reports[candidate_key] = report
            idxs_current, idxs_candidate = self.match_keyframes(current_keyframe, candidate)
            self.num_last_matches = max(self.num_last_matches, len(idxs_current))
            report["bow_match_pairs"] = [
                [int(i), int(j)] for i, j in zip(idxs_current[:250], idxs_candidate[:250])
            ]
            report["descriptor_distances"] = _distance_summary(self.last_match_distances)
            report["bow_matches_after_orientation"] = int(len(idxs_current))
            report["bow_matches_with_valid_mappoints"] = int(len(idxs_current))
            self._merge_match_diagnostics(report)

            if len(idxs_current) < self.min_matches:
                self.last_error = "too few loop geometry matches"
                report["rejection_stage"] = "geometry"
                report["rejection_reason"] = self.last_error
                continue

            points_current, points_loop, idxs_current, idxs_candidate = self._matched_3d_points(
                current_keyframe,
                candidate,
                idxs_current,
                idxs_candidate,
            )
            report["geometry_input_correspondences"] = int(len(points_current))
            if len(points_current) < self.min_matches:
                self.last_error = "too few valid 3D loop correspondences"
                report["bow_matches_with_valid_mappoints"] = int(len(points_current))
                report["rejection_stage"] = "geometry"
                report["rejection_reason"] = self.last_error
                continue

            estimate = estimate_scale_fixed_sim3(
                points_current,
                points_loop,
                max_error=Parameters.kLoopClosingSE3RansacMaxError,
                ransac_iterations=Parameters.kLoopClosingSE3RansacIterations,
            )
            seed_inliers = int(np.sum(estimate.inlier_mask))
            seed_min_inliers = int(
                getattr(
                    Parameters,
                    "kLoopClosingSE3GuidedMinSeedInliers",
                    max(3, self.min_matches // 2),
                )
            )
            self.num_last_inliers = max(self.num_last_inliers, seed_inliers)
            report["geometry_ransac_inliers"] = seed_inliers
            report["geometry_refined_inliers"] = seed_inliers
            report["geometry_reprojection_rmse"] = _estimate_rmse(estimate)

            if not estimate.success or seed_inliers < seed_min_inliers:
                self.last_error = estimate.error
                report["rejection_stage"] = "geometry"
                report["rejection_reason"] = estimate.error or "not enough SE3 RANSAC seed inliers"
                continue

            pose_distance, pose_rotation_deg = self._estimated_pose_delta(current_keyframe, candidate)
            max_pose_distance = float(
                getattr(Parameters, "kLoopClosingMaxEstimatedPoseDistanceForGuidedSE3", 0.0) or 0.0
            )
            max_pose_rotation = float(
                getattr(Parameters, "kLoopClosingMaxEstimatedPoseRotationDegForGuidedSE3", 0.0) or 0.0
            )
            report["estimated_pose_distance"] = pose_distance
            report["estimated_pose_distance_threshold"] = max_pose_distance
            report["estimated_pose_rotation_deg"] = pose_rotation_deg
            report["estimated_pose_rotation_threshold_deg"] = max_pose_rotation
            use_estimated_pose_prior = seed_inliers < self.min_matches
            if use_estimated_pose_prior and max_pose_distance > 0.0 and (
                pose_distance is None
                or not np.isfinite(float(pose_distance))
                or float(pose_distance) > max_pose_distance
            ):
                self.last_error = "estimated pose distance too large for guided SE3 loop seed"
                report["rejection_stage"] = "geometry"
                report["rejection_reason"] = self.last_error
                continue
            if use_estimated_pose_prior and max_pose_rotation > 0.0 and (
                pose_rotation_deg is None
                or not np.isfinite(float(pose_rotation_deg))
                or float(pose_rotation_deg) > max_pose_rotation
            ):
                self.last_error = "estimated pose rotation too large for guided SE3 loop seed"
                report["rejection_stage"] = "geometry"
                report["rejection_reason"] = self.last_error
                continue

            matches = [None] * len(current_keyframe.points)
            match_idxs = np.full(len(current_keyframe.points), -1, dtype=np.int32)
            inlier_current = idxs_current[estimate.inlier_mask]
            inlier_candidate = idxs_candidate[estimate.inlier_mask]
            report["inlier_match_pairs"] = [
                [int(i), int(j)] for i, j in zip(inlier_current[:250], inlier_candidate[:250])
            ]

            for idx_cur, idx_cand in zip(inlier_current, inlier_candidate):
                if 0 <= idx_cur < len(matches) and 0 <= idx_cand < len(candidate.points):
                    matches[int(idx_cur)] = candidate.points[int(idx_cand)]
                    match_idxs[int(idx_cur)] = int(idx_cand)

            guided_added = self._guided_projection_refinement(
                current_keyframe,
                candidate,
                estimate,
                matches,
                match_idxs,
            )
            final_matches = int(sum(match is not None for match in matches))
            self.last_guided_projection_matches = max(self.last_guided_projection_matches, guided_added)
            self.last_final_matches = max(self.last_final_matches, final_matches)
            self.num_last_inliers = max(self.num_last_inliers, final_matches)
            report["guided_projection_matches"] = int(guided_added)
            report["guided_projection_total_matches"] = int(final_matches)

            if final_matches < self.min_matches:
                self.last_error = "not enough final guided loop matches"
                report["rejection_stage"] = "geometry"
                report["rejection_reason"] = self.last_error
                continue

            (
                final_points_current,
                final_points_loop,
                final_idxs_current,
                final_idxs_candidate,
            ) = self._matched_3d_points_from_matches(current_keyframe, candidate, matches, match_idxs)
            if len(final_points_current) < self.min_matches:
                self.last_error = "too few valid guided 3D loop correspondences"
                report["rejection_stage"] = "geometry"
                report["rejection_reason"] = self.last_error
                continue

            refined_estimate = estimate_scale_fixed_sim3(
                final_points_current,
                final_points_loop,
                max_error=Parameters.kLoopClosingSE3RansacMaxError,
                ransac_iterations=Parameters.kLoopClosingSE3RansacIterations,
            )
            refined_inliers = int(np.sum(refined_estimate.inlier_mask))
            report["geometry_refined_inliers"] = refined_inliers
            report["final_inliers"] = refined_inliers
            report["geometry_reprojection_rmse"] = _estimate_rmse(refined_estimate)
            self.num_last_inliers = max(self.num_last_inliers, refined_inliers)

            if not refined_estimate.success or refined_inliers < self.min_matches:
                self.last_error = refined_estimate.error or "not enough refined guided loop inliers"
                report["rejection_stage"] = "geometry"
                report["rejection_reason"] = self.last_error
                continue

            refined_mask = refined_estimate.inlier_mask.astype(bool)
            refined_current = final_idxs_current[refined_mask]
            refined_candidate = final_idxs_candidate[refined_mask]
            cleaned_matches = [None] * len(current_keyframe.points)
            cleaned_match_idxs = np.full(len(current_keyframe.points), -1, dtype=np.int32)
            for idx_cur, idx_cand in zip(refined_current, refined_candidate):
                if 0 <= idx_cur < len(cleaned_matches) and 0 <= idx_cand < len(candidate.points):
                    cleaned_matches[int(idx_cur)] = candidate.points[int(idx_cand)]
                    cleaned_match_idxs[int(idx_cur)] = int(idx_cand)
            matches = cleaned_matches
            match_idxs = cleaned_match_idxs
            estimate = refined_estimate
            report["inlier_match_pairs"] = [
                [int(i), int(j)] for i, j in zip(refined_current[:250], refined_candidate[:250])
            ]

            candidate_group = candidate.get_covisible_keyframes()
            candidate_group.append(candidate)
            loop_map_points = set()
            for kf in candidate_group:
                for point in kf.get_matched_good_points():
                    if point is not None and not point.is_bad():
                        loop_map_points.add(point)

            # Reproject the loop-side covisibility points with the corrected current pose.
            Tcw_current = np.asarray(current_keyframe.Tcw(), dtype=np.float64).reshape(4, 4)
            T12 = np.asarray(estimate.T, dtype=np.float64).reshape(4, 4)
            try:
                T12_inv = np.linalg.inv(T12)
            except np.linalg.LinAlgError:
                self.last_error = "singular SE3 loop estimate"
                report["rejection_stage"] = "geometry"
                report["rejection_reason"] = self.last_error
                self.success_loop_kf = None
                continue
            Tcw_current_corrected = Tcw_current @ T12_inv

            new_projection_matches, matches, search_more_diag = (
                ProjectionMatcher.search_more_map_points_by_projection(
                    loop_map_points,
                    current_keyframe,
                    Tcw_current_corrected,
                    matches,
                    match_idxs,
                    max_reproj_distance=Parameters.kLoopClosingMaxReprojectionDistanceMapSearch,
                    return_diagnostics=True,
                )
            )

            num_matched_map_points = sum(m is not None for m in matches)

            # Diagnostics surfaced into the candidate report / loop debug CSV.
            report["seed_inliers"] = int(seed_inliers)
            report["candidate_covisible_points"] = int(len(loop_map_points))
            report["projected_visible_points"] = int(
                search_more_diag.get("projected_visible_points", 0)
            )
            report["new_projection_matches"] = int(new_projection_matches)
            report["total_final_matches"] = int(num_matched_map_points)
            report["final_gate_threshold"] = int(
                Parameters.kLoopClosingMinNumMatchedMapPoints
            )
            report["guided_projection_total_matches"] = num_matched_map_points
            if num_matched_map_points < Parameters.kLoopClosingMinNumMatchedMapPoints:
                self.last_error = (
                    f"too few matched map points after covisibility expansion "
                    f"({num_matched_map_points} < {Parameters.kLoopClosingMinNumMatchedMapPoints})"
                )
                report["rejection_stage"] = "geometry"
                report["rejection_reason"] = self.last_error
                report["accepted_or_rejected"] = "rejected"
                self.success_loop_kf = None
                continue

            self.success_loop_map_points = loop_map_points
            self.success_loop_kf = candidate
            self.success_sim3 = estimate
            self.success_map_point_matches = matches
            self.success_map_point_matches_idxs = match_idxs
            report["final_inliers"] = num_matched_map_points
            report["accepted"] = True
            report["accepted_or_rejected"] = "accepted"
            report["rejection_stage"] = ""
            report["rejection_reason"] = ""
            return True

        return False

    def match_keyframes(self, current_keyframe: KeyFrame, candidate: KeyFrame):
        ensure_frame_feature_arrays(current_keyframe)
        ensure_frame_feature_arrays(candidate)

        current_idxs = np.asarray(
            [
                idx
                for idx, point in enumerate(current_keyframe.points)
                if point is not None and not point.is_bad()
            ],
            dtype=np.int32,
        )
        candidate_idxs = np.asarray(
            [
                idx
                for idx, point in enumerate(candidate.points)
                if point is not None and not point.is_bad()
            ],
            dtype=np.int32,
        )

        if len(current_idxs) == 0 or len(candidate_idxs) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        if self.keyframe_database is not None and getattr(self.keyframe_database, "available", False):
            try:
                self.keyframe_database.compute_bow(current_keyframe)
                self.keyframe_database.compute_bow(candidate)
                bow_result = BoWGuidedMatcher(self.keyframe_database.voc).match(
                    current_keyframe,
                    candidate,
                    valid_idxs1=current_idxs,
                    valid_idxs2=candidate_idxs,
                    max_descriptor_distance=Parameters.kMaxDescriptorDistance,
                    ratio_test=Parameters.kLoopClosingFeatureMatchRatioTest,
                    orientation_check=True,
                )
                self.last_match_diagnostics = bow_result.diagnostics
                self.last_match_distances = np.asarray(bow_result.distances, dtype=np.float32)
                self.last_bow_guided_matching_available = bow_result.available
                if bow_result.available:
                    return bow_result.idxs1, bow_result.idxs2
            except Exception as exc:
                self.last_error = f"BoW-guided loop matching unavailable: {exc}"

        self.last_fallback_descriptor_matching = True
        if FeatureTrackerShared.feature_matcher is None:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        result = FeatureTrackerShared.feature_matcher.match(
            current_keyframe.img,
            candidate.img,
            current_keyframe.des[current_idxs],
            candidate.des[candidate_idxs],
            kps1=[current_keyframe.kps[i] for i in current_idxs],
            kps2=[candidate.kps[i] for i in candidate_idxs],
            ratio_test=Parameters.kLoopClosingFeatureMatchRatioTest,
        )

        if result.idxs1 is None or len(result.idxs1) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        idxs_current = current_idxs[np.asarray(result.idxs1, dtype=np.int32)]
        idxs_candidate = candidate_idxs[np.asarray(result.idxs2, dtype=np.int32)]

        if FeatureTrackerShared.oriented_features and len(idxs_current) > 0:
            valid = RotationHistogram.filter_matches_with_histogram_orientation(
                idxs_current,
                idxs_candidate,
                current_keyframe.angles,
                candidate.angles,
            )
            idxs_current = idxs_current[valid]
            idxs_candidate = idxs_candidate[valid]

        return idxs_current, idxs_candidate

    @staticmethod
    def _base_candidate_report(current_keyframe: KeyFrame, candidate: KeyFrame) -> dict:
        return {
            "current_kf_id": int(getattr(current_keyframe, "kid", getattr(current_keyframe, "id", -1))),
            "candidate_kf_id": int(getattr(candidate, "kid", getattr(candidate, "id", -1))),
            "current_frame_id": int(getattr(current_keyframe, "id", -1)),
            "candidate_frame_id": int(getattr(candidate, "id", -1)),
            "geometry_method": "rgbd_se3_ransac",
            "common_words": 0,
            "bow_matches_raw": 0,
            "bow_matches_after_ratio": 0,
            "bow_matches_after_orientation": 0,
            "bow_matches_with_valid_mappoints": 0,
            "geometry_input_correspondences": 0,
            "geometry_ransac_inliers": 0,
            "geometry_refined_inliers": 0,
            "geometry_reprojection_rmse": None,
            "estimated_pose_distance": None,
            "estimated_pose_distance_threshold": float(
                getattr(Parameters, "kLoopClosingMaxEstimatedPoseDistanceForGuidedSE3", 0.0) or 0.0
            ),
            "estimated_pose_rotation_deg": None,
            "estimated_pose_rotation_threshold_deg": float(
                getattr(Parameters, "kLoopClosingMaxEstimatedPoseRotationDegForGuidedSE3", 0.0) or 0.0
            ),
            "guided_projection_matches": 0,
            "guided_projection_total_matches": 0,
            "final_inliers": 0,
            "accept_threshold_inliers": int(Parameters.kLoopClosingMinNumMatchedMapPoints),
            "accepted": False,
            "rejection_stage": "",
            "rejection_reason": "",
        }

    def _merge_match_diagnostics(self, report: dict) -> None:
        diagnostics = self.last_match_diagnostics
        if diagnostics is None:
            return
        report["common_words"] = int(getattr(diagnostics, "shared_words", 0) or 0)
        report["bow_matches_raw"] = int(getattr(diagnostics, "raw_matches", 0) or 0)
        report["bow_matches_after_ratio"] = int(
            getattr(diagnostics, "matches_after_ratio", getattr(diagnostics, "raw_matches", 0)) or 0
        )
        report["bow_matches_after_orientation"] = int(
            getattr(
                diagnostics,
                "matches_after_orientation",
                report.get("bow_matches_after_orientation", 0),
            )
            or 0
        )
        report["threshold_rejects"] = int(getattr(diagnostics, "threshold_rejects", 0) or 0)
        report["ratio_rejects"] = int(getattr(diagnostics, "ratio_rejects", 0) or 0)
        report["duplicate_train_rejects"] = int(getattr(diagnostics, "duplicate_train_rejects", 0) or 0)
        report["orientation_rejects"] = int(getattr(diagnostics, "orientation_rejects", 0) or 0)

    @staticmethod
    def _guided_projection_refinement(
        current_keyframe: KeyFrame,
        candidate: KeyFrame,
        estimate: Sim3Estimate,
        matches: list,
        match_idxs: np.ndarray,
    ) -> int:
        ensure_frame_feature_arrays(current_keyframe)
        ensure_frame_feature_arrays(candidate)

        if len(current_keyframe.points) == 0 or len(candidate.points) == 0:
            return 0

        Tcw_candidate = np.asarray(candidate.Tcw(), dtype=np.float64).reshape(4, 4)
        Rcw = Tcw_candidate[:3, :3]
        tcw = Tcw_candidate[:3, 3]
        taken_candidate_idxs = {int(idx) for idx in match_idxs.tolist() if int(idx) >= 0}
        max_descriptor_distance = float(Parameters.kMaxDescriptorDistance or 100)
        search_radius = float(Parameters.kLoopClosingMaxReprojectionDistanceMapSearch)
        added = 0

        candidate_uvs = np.asarray(
            [kp.pt if hasattr(kp, "pt") else kp for kp in candidate.kpsu],
            dtype=np.float64,
        ).reshape(-1, 2)

        for idx_cur, point_current in enumerate(current_keyframe.points):
            if idx_cur < len(matches) and matches[idx_cur] is not None:
                continue
            if point_current is None or point_current.is_bad():
                continue
            position = point_current.get_position()
            if not np.all(np.isfinite(position)):
                continue

            aligned_position = estimate.R @ position + estimate.t
            point_c = Rcw @ aligned_position + tcw
            if not np.all(np.isfinite(point_c)) or point_c[2] <= Parameters.kMinDepth:
                continue
            uv, depth = candidate.camera.project(point_c.reshape(1, 3))
            uv = np.asarray(uv, dtype=np.float64).reshape(1, 2)
            depth = np.asarray(depth, dtype=np.float64).reshape(1)
            if not bool(candidate.are_in_image(uv, depth)[0]):
                continue

            pixel_dists = np.linalg.norm(candidate_uvs - uv.reshape(1, 2), axis=1)
            nearby = np.flatnonzero(pixel_dists <= search_radius)
            if len(nearby) == 0:
                continue

            best_idx = -1
            best_distance = float("inf")
            for idx_cand in nearby:
                idx_cand = int(idx_cand)
                if idx_cand in taken_candidate_idxs:
                    continue
                if idx_cand < 0 or idx_cand >= len(candidate.points):
                    continue
                point_loop = candidate.points[idx_cand]
                if point_loop is None or point_loop.is_bad():
                    continue
                descriptor_distance = point_current.min_des_distance(candidate.des[idx_cand])
                if descriptor_distance < best_distance:
                    best_distance = float(descriptor_distance)
                    best_idx = idx_cand

            if best_idx >= 0 and best_distance <= max_descriptor_distance:
                matches[int(idx_cur)] = candidate.points[best_idx]
                match_idxs[int(idx_cur)] = int(best_idx)
                taken_candidate_idxs.add(int(best_idx))
                added += 1

        return added

    @staticmethod
    def _estimated_pose_delta(current_keyframe: KeyFrame, candidate: KeyFrame) -> tuple[Optional[float], Optional[float]]:
        try:
            Twc_current = np.asarray(current_keyframe.Twc(), dtype=np.float64).reshape(4, 4)
            Twc_candidate = np.asarray(candidate.Twc(), dtype=np.float64).reshape(4, 4)
        except Exception:
            return None, None
        position_current = Twc_current[:3, 3]
        position_candidate = Twc_candidate[:3, 3]
        distance = float(np.linalg.norm(position_current - position_candidate))
        if not np.isfinite(distance):
            distance = None
        rotation_deg = _rotation_angle_deg(Twc_current[:3, :3], Twc_candidate[:3, :3])
        return distance, rotation_deg

    @staticmethod
    def _matched_3d_points_from_matches(current_keyframe, candidate, matches, match_idxs):
        idxs_current = []
        idxs_candidate = []
        for idx_cur, point_loop in enumerate(matches):
            if point_loop is None:
                continue
            idx_cand = int(match_idxs[idx_cur]) if idx_cur < len(match_idxs) else -1
            if idx_cand < 0:
                continue
            idxs_current.append(int(idx_cur))
            idxs_candidate.append(idx_cand)
        return LoopGeometryChecker._matched_3d_points(
            current_keyframe,
            candidate,
            np.asarray(idxs_current, dtype=np.int32),
            np.asarray(idxs_candidate, dtype=np.int32),
        )

    @staticmethod
    def _matched_3d_points(current_keyframe, candidate, idxs_current, idxs_candidate):
        points_current = []
        points_loop = []
        valid_current = []
        valid_candidate = []

        for idx_cur, idx_cand in zip(idxs_current, idxs_candidate):
            if idx_cur < 0 or idx_cur >= len(current_keyframe.points):
                continue
            if idx_cand < 0 or idx_cand >= len(candidate.points):
                continue
            point_current = current_keyframe.points[int(idx_cur)]
            point_loop = candidate.points[int(idx_cand)]
            if point_current is None or point_loop is None:
                continue
            if point_current.is_bad() or point_loop.is_bad():
                continue
            p1 = point_current.get_position()
            p2 = point_loop.get_position()
            if not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)):
                continue
            points_current.append(p1)
            points_loop.append(p2)
            valid_current.append(int(idx_cur))
            valid_candidate.append(int(idx_cand))

        return (
            np.asarray(points_current, dtype=np.float64).reshape(-1, 3),
            np.asarray(points_loop, dtype=np.float64).reshape(-1, 3),
            np.asarray(valid_current, dtype=np.int32),
            np.asarray(valid_candidate, dtype=np.int32),
        )


# Apply loop corrections to poses, points, and post-loop optimization.
class LoopCorrector:
    def __init__(self, slam, geometry_checker: LoopGeometryChecker):
        self.slam = slam
        self.loop_geometry_checker = geometry_checker
        self.mean_graph_chi2_error = None
        self.last_result: Optional[EssentialGraphResult] = None
        self.last_num_fused_points = 0
        self.last_num_corrected_points = 0
        self.last_fusion_diagnostics = ProjectionFuseDiagnostics()
        self.last_global_ba_result = GlobalBAResult(started=False, reason="disabled")

    @property
    def map(self):
        return getattr(self.slam, "map", None)

    def correct_loop(self, current_keyframe: KeyFrame) -> EssentialGraphResult:
        loop_keyframe = self.loop_geometry_checker.success_loop_kf
        estimate = self.loop_geometry_checker.success_sim3
        if loop_keyframe is None or estimate is None or not estimate.success:
            result = EssentialGraphResult(False, float("inf"), float("inf"), 0, "missing loop geometry")
            self.last_result = result
            return result

        correction_T = estimate.T
        if not np.all(np.isfinite(correction_T)):
            result = EssentialGraphResult(False, float("inf"), float("inf"), 0, "non-finite correction")
            self.last_result = result
            return result

        current_group = current_keyframe.get_connected_keyframes()
        current_group.append(current_keyframe)
        corrected_poses = self._make_corrected_pose_map(current_group, correction_T)

        lock = self.map.update_lock if self.map is not None else _NullLock()
        with lock:
            self.last_fusion_diagnostics = ProjectionFuseDiagnostics()
            self.last_num_corrected_points = 0
            direct_fused = self._fuse_loop_matches(current_keyframe)
            projection_fused = self.search_and_fuse_corrected_keyframes(
                current_keyframe,
                loop_keyframe,
                corrected_poses,
            )
            self.last_num_fused_points = direct_fused + projection_fused

            loop_connections = self._build_loop_connections_after_fusion(current_group)
            if loop_keyframe not in loop_connections.get(current_keyframe, []):
                loop_connections.setdefault(current_keyframe, []).append(loop_keyframe)

            result = optimize_essential_graph_se3(
                current_group,
                loop_keyframe,
                current_keyframe,
                correction_T,
                map_object=self.map,
                corrected_poses=corrected_poses,
                loop_connections=loop_connections,
            )

            if result.success:
                self.last_num_corrected_points = getattr(result, "corrected_points", 0)
                loop_keyframe.add_loop_edge(current_keyframe)
                current_keyframe.add_loop_edge(loop_keyframe)
                for keyframe in current_group:
                    keyframe.update_connections()
                loop_keyframe.update_connections()
                self.last_global_ba_result = self._run_global_ba_after_loop(loop_keyframe)
                self._reset_tracking_after_loop(current_keyframe)
            else:
                self.last_global_ba_result = GlobalBAResult(started=False, reason="loop correction failed")

        self.last_result = result
        self.mean_graph_chi2_error = result.after_error
        return result

    def _reset_tracking_after_loop(self, current_keyframe: KeyFrame) -> None:
        tracking = getattr(self.slam, "tracking", None)
        if tracking is None:
            return
        motion_model = getattr(tracking, "motion_model", None)
        if motion_model is not None and hasattr(motion_model, "reset"):
            motion_model.reset()
        tracking.kf_ref = current_keyframe
        tracking.kf_last = current_keyframe
        try:
            tracking.map.update_local_map(current_keyframe)
        except Exception:
            pass

    def _run_global_ba_after_loop(self, loop_keyframe: KeyFrame) -> GlobalBAResult:
        enabled = bool(
            getattr(self.slam, "enable_global_ba", False)
            and getattr(self.slam, "global_ba_after_loop", False)
        )
        if not enabled:
            return GlobalBAResult(started=False, reason="disabled")

        adjuster = GlobalBundleAdjuster(
            self.map,
            rounds=int(getattr(self.slam, "global_ba_iterations", Parameters.kGlobalBAIterations)),
            use_robust_kernel=Parameters.kGBAUseRobustKernel,
            min_inlier_edges=Parameters.kGlobalBAMinInlierEdges,
        )
        return adjuster.run(loop_kf_id=int(getattr(loop_keyframe, "kid", 0)))

    def _fuse_loop_matches(self, current_keyframe: KeyFrame) -> int:
        fused = 0
        for idx, loop_point in enumerate(self.loop_geometry_checker.success_map_point_matches):
            if loop_point is None or loop_point.is_bad():
                if loop_point is not None:
                    self.last_fusion_diagnostics.rejected_bad_point += 1
                continue
            current_point = current_keyframe.get_point_match(idx)
            if current_point is None:
                if loop_point.add_observation(current_keyframe, idx):
                    loop_point.update_info()
                    self.last_fusion_diagnostics.added_observations += 1
                    self.last_fusion_diagnostics.fused_points += 1
                    fused += 1
            elif current_point is not loop_point:
                current_point.replace_with(loop_point)
                self.last_fusion_diagnostics.replaced_points += 1
                self.last_fusion_diagnostics.fused_points += 1
                fused += 1
            else:
                self.last_fusion_diagnostics.rejected_duplicate += 1
        return fused

    def search_and_fuse_corrected_keyframes(
        self,
        current_keyframe: KeyFrame,
        loop_keyframe: KeyFrame,
        corrected_poses: dict[KeyFrame, np.ndarray],
    ) -> int:
        loop_map_points = self._collect_loop_map_points(loop_keyframe)
        corrected_keyframes = self._collect_current_corrected_keyframes(current_keyframe)

        fused = 0
        added_before = self.last_fusion_diagnostics.added_observations
        affected_points = set()
        affected_keyframes = set(corrected_keyframes)

        for keyframe in corrected_keyframes:
            corrected_pose = corrected_poses.get(keyframe, keyframe.Tcw())
            replace_points = [None] * len(loop_map_points)
            local_diag = ProjectionFuseDiagnostics()
            ProjectionMatcher.search_and_fuse_for_loop_correction(
                keyframe,
                corrected_pose,
                loop_map_points,
                replace_points,
                diagnostics=local_diag,
            )
            self.last_fusion_diagnostics.merge(local_diag)

            for idx, replacement_source in enumerate(replace_points):
                if replacement_source is None:
                    continue
                loop_point = loop_map_points[idx]
                if loop_point is None or loop_point.is_bad():
                    self.last_fusion_diagnostics.rejected_bad_point += 1
                    continue
                if replacement_source is loop_point:
                    self.last_fusion_diagnostics.rejected_duplicate += 1
                    continue
                if replacement_source.is_bad():
                    self.last_fusion_diagnostics.rejected_bad_point += 1
                    continue
                replacement_source.replace_with(loop_point)
                affected_points.add(loop_point)
                affected_keyframes.update(loop_point.keyframes())
                fused += 1

        for point in loop_map_points:
            if point is not None and not point.is_bad():
                affected_points.add(point)
        for point in affected_points:
            point.update_info()
        for keyframe in affected_keyframes:
            if keyframe is not None and not keyframe.is_bad():
                keyframe.update_connections()

        added_here = self.last_fusion_diagnostics.added_observations - added_before
        return fused + added_here

    def _collect_loop_map_points(self, loop_keyframe: KeyFrame) -> list:
        keyframes = []
        for keyframe in [loop_keyframe] + loop_keyframe.get_best_covisible_keyframes(
            Parameters.kNumBestCovisibilityKeyFrames
        ):
            if keyframe is not None and not keyframe.is_bad() and keyframe not in keyframes:
                keyframes.append(keyframe)

        points = []
        seen = set()
        for keyframe in keyframes:
            for point in keyframe.get_matched_good_points():
                if point is None or point.is_bad() or point in seen:
                    continue
                points.append(point)
                seen.add(point)
        return points

    def _collect_current_corrected_keyframes(self, current_keyframe: KeyFrame) -> list[KeyFrame]:
        keyframes = []
        for keyframe in [current_keyframe] + current_keyframe.get_best_covisible_keyframes(
            Parameters.kNumBestCovisibilityKeyFrames
        ):
            if keyframe is not None and not keyframe.is_bad() and keyframe not in keyframes:
                keyframes.append(keyframe)
        return keyframes

    @staticmethod
    def _make_corrected_pose_map(keyframes: list[KeyFrame], correction_T: np.ndarray) -> dict[KeyFrame, np.ndarray]:
        correction_T = np.asarray(correction_T, dtype=np.float64).reshape(4, 4)
        correction_inv = np.linalg.inv(correction_T)
        corrected = {}
        for keyframe in keyframes:
            Tcw = np.asarray(keyframe.Tcw(), dtype=np.float64).reshape(4, 4)
            corrected[keyframe] = Tcw @ correction_inv
        return corrected

    @staticmethod
    def _build_loop_connections_after_fusion(current_group: list[KeyFrame]) -> dict[KeyFrame, list[KeyFrame]]:
        previous_neighbors = {keyframe: set(keyframe.get_covisible_keyframes()) for keyframe in current_group}
        current_set = set(current_group)
        loop_connections = {}

        for keyframe in current_group:
            keyframe.update_connections()
            new_connections = set(keyframe.get_connected_keyframes())
            new_connections.difference_update(previous_neighbors.get(keyframe, set()))
            new_connections.difference_update(current_set)
            loop_connections[keyframe] = [
                connected for connected in new_connections if connected is not None and not connected.is_bad()
            ]
        return loop_connections


# Manage the loop-closing queue and execute one loop event at a time.
class LoopClosing:
    def __init__(self, slam, keyframe_database=None, consistency_threshold: int = 3):
        self.slam = slam
        self.keyframe_database = keyframe_database or getattr(slam, "keyframe_database", None)
        self.loop_detector = LoopDetector(self.keyframe_database)
        self.loop_consistency_checker = LoopGroupConsistencyChecker(consistency_threshold)
        self.loop_geometry_checker = LoopGeometryChecker(keyframe_database=self.keyframe_database)
        self.loop_corrector = LoopCorrector(slam, self.loop_geometry_checker)
        self.queue = deque()
        self._queue_lock = threading.Lock()
        self.last_loop_kf_id = 0
        self.last_diagnostics = LoopDiagnostics()
        self.mean_graph_chi2_error = None
        self._is_correcting = False
        self._is_correcting_lock = threading.Lock()

    def is_correcting(self) -> bool:
        with self._is_correcting_lock:
            return bool(self._is_correcting)

    def insert_keyframe(self, keyframe: KeyFrame) -> None:
        self.add_keyframe(keyframe)

    def add_keyframe(self, keyframe: KeyFrame, img=None) -> None:
        if img is not None:
            keyframe.img = img
        with self._queue_lock:
            self.queue.append(keyframe)

    def queue_size(self) -> int:
        with self._queue_lock:
            return len(self.queue)

    def pop_keyframe(self):
        with self._queue_lock:
            if not self.queue:
                return None
            return self.queue.popleft()

    def step(self) -> bool:
        keyframe = self.pop_keyframe()
        if keyframe is None:
            return False
        return self.process_keyframe(keyframe)

    def process_keyframe(self, keyframe: KeyFrame) -> bool:
        diagnostics = LoopDiagnostics()
        self.last_diagnostics = diagnostics

        if self.keyframe_database is None or not self.keyframe_database.available:
            diagnostics.unavailable_reason = (
                "keyframe database is not configured"
                if self.keyframe_database is None
                else self.keyframe_database.unavailable_reason()
            )
            diagnostics.rejected_by_bow = 1
            return False

        self.keyframe_database.add(keyframe)
        detection_output: LoopDetectorOutput = self.loop_detector.detect(keyframe)
        diagnostics.candidates = len(detection_output.candidate_keyframes)
        diagnostics.loop_debug_records = self._build_loop_debug_records(keyframe, detection_output)

        if len(detection_output.candidate_keyframes) == 0:
            diagnostics.rejected_by_bow = 1
            self.loop_consistency_checker.clear_consistency_groups()
            return False

        got_consistent = self.loop_consistency_checker.check_candidates(
            keyframe,
            detection_output.candidate_keyframes,
        )
        self._merge_consistency_debug(diagnostics.loop_debug_records)

        if not got_consistent:
            diagnostics.rejected_by_consistency = diagnostics.candidates
            for record in diagnostics.loop_debug_records:
                record["rejection_stage"] = "consistency"
                record["rejection_reason"] = "rejected_by_consistency"
            return False

        consistent_candidates = [
            candidate
            for candidate in self.loop_consistency_checker.enough_consistent_candidates
            if not candidate.is_bad()
        ]
        got_geometry = self.loop_geometry_checker.check_candidates(keyframe, consistent_candidates)
        self._merge_geometry_debug(diagnostics.loop_debug_records)
        diagnostics.candidate_pair_reports = list(self.loop_geometry_checker.last_candidate_reports.values())

        for record in diagnostics.loop_debug_records:
            if not bool(record.get("passed_consistency")) and not record.get("rejection_stage"):
                record["rejection_stage"] = "consistency"
                record["rejection_reason"] = "rejected_by_consistency"

        if not got_geometry:
            diagnostics.rejected_by_geometry = len(consistent_candidates)
            for record in diagnostics.loop_debug_records:
                if bool(record.get("passed_consistency")) and not record.get("rejection_stage"):
                    record["rejection_stage"] = "geometry"
                    record["rejection_reason"] = record.get("rejection_reason") or "rejected_by_geometry"
            return False

        with self._is_correcting_lock:
            self._is_correcting = True
        try:
            result = self.loop_corrector.correct_loop(keyframe)
        finally:
            with self._is_correcting_lock:
                self._is_correcting = False
        diagnostics.optimization_result = result
        diagnostics.corrected_keyframes = result.corrected_keyframes
        diagnostics.corrected_points = self.loop_corrector.last_num_corrected_points
        diagnostics.fused_points = self.loop_corrector.last_num_fused_points
        diagnostics.fusion_diagnostics = self.loop_corrector.last_fusion_diagnostics
        diagnostics.global_ba_result = self.loop_corrector.last_global_ba_result
        self._copy_global_ba_diagnostics(diagnostics, diagnostics.global_ba_result)

        if result.success:
            diagnostics.accepted = 1
            for record in diagnostics.loop_debug_records:
                if int(record.get("candidate_kf_id", -1)) == int(getattr(self.loop_geometry_checker.success_loop_kf, "kid", -999)):
                    record["rejection_stage"] = ""
                    record["rejection_reason"] = ""
            self.last_loop_kf_id = keyframe.kid
            self.mean_graph_chi2_error = result.after_error
            return True

        diagnostics.rejected_by_geometry = len(consistent_candidates)
        return False

    def _build_loop_debug_records(self, keyframe: KeyFrame, detection_output: LoopDetectorOutput) -> list[dict]:
        records = []
        scores = list(getattr(detection_output, "candidate_scores", []) or [])
        for rank, candidate in enumerate(detection_output.candidate_keyframes, start=1):
            if candidate is None:
                continue
            score = scores[rank - 1] if rank - 1 < len(scores) else None
            candidate_kid = int(getattr(candidate, "kid", getattr(candidate, "id", -1)))
            current_kid = int(getattr(keyframe, "kid", getattr(keyframe, "id", -1)))
            records.append(
                {
                    "frame_id": int(getattr(keyframe, "id", current_kid)),
                    "current_kf_id": current_kid,
                    "candidate_kf_id": candidate_kid,
                    "current_timestamp": getattr(keyframe, "timestamp", None),
                    "candidate_timestamp": getattr(candidate, "timestamp", None),
                    "candidate_score": score,
                    "candidate_rank": rank,
                    "candidate_source": "keyframe_database",
                    "temporal_separation_kf": abs(current_kid - candidate_kid),
                    "temporal_separation_frames": abs(int(getattr(keyframe, "id", current_kid)) - int(getattr(candidate, "id", candidate_kid))),
                    "current_group_kf_ids": [],
                    "candidate_group_kf_ids": [],
                    "previous_consistency_group_ids": [],
                    "consistency_overlap_count": 0,
                    "consistency_count": 0,
                    "consistency_required": self.loop_consistency_checker.consistency_threshold,
                    "passed_consistency": False,
                    "common_words": 0,
                    "bow_score_raw": score,
                    "bow_score_normalized": score,
                    "bow_matches_raw": 0,
                    "bow_matches_after_ratio": 0,
                    "bow_matches_after_orientation": 0,
                    "bow_matches_with_valid_mappoints": 0,
                    "geometry_method": "rgbd_se3_ransac",
                    "geometry_input_correspondences": 0,
                    "geometry_ransac_inliers": 0,
                    "geometry_refined_inliers": 0,
                    "geometry_reprojection_rmse": None,
                    "estimated_pose_distance": None,
                    "estimated_pose_distance_threshold": float(
                        getattr(Parameters, "kLoopClosingMaxEstimatedPoseDistanceForGuidedSE3", 0.0) or 0.0
                    ),
                    "estimated_pose_rotation_deg": None,
                    "estimated_pose_rotation_threshold_deg": float(
                        getattr(Parameters, "kLoopClosingMaxEstimatedPoseRotationDegForGuidedSE3", 0.0) or 0.0
                    ),
                    "guided_projection_matches": 0,
                    "guided_projection_total_matches": 0,
                    "final_inliers": 0,
                    "accept_threshold_inliers": int(Parameters.kLoopClosingMinNumMatchedMapPoints),
                    "rejection_stage": "",
                    "rejection_reason": "",
                }
            )
        return records

    def _merge_consistency_debug(self, records: list[dict]) -> None:
        for record in records:
            candidate_kid = int(record.get("candidate_kf_id", -1))
            info = self.loop_consistency_checker.last_candidate_debug.get(candidate_kid)
            if info:
                record.update(info)

    def _merge_geometry_debug(self, records: list[dict]) -> None:
        for record in records:
            candidate_kid = int(record.get("candidate_kf_id", -1))
            info = self.loop_geometry_checker.last_candidate_reports.get(candidate_kid)
            if info:
                record.update({k: v for k, v in info.items() if k not in {"current_kf_id", "candidate_kf_id"}})

    @staticmethod
    def _copy_global_ba_diagnostics(diagnostics: LoopDiagnostics, result: Optional[GlobalBAResult]) -> None:
        if result is None:
            result = GlobalBAResult(started=False, reason="not run")
        diagnostics.global_ba_started = bool(result.started)
        diagnostics.global_ba_success = bool(result.success)
        diagnostics.global_ba_aborted = bool(result.aborted)
        diagnostics.global_ba_reason = result.reason
        diagnostics.global_ba_num_keyframes = int(result.num_keyframes)
        diagnostics.global_ba_num_map_points = int(result.num_map_points)
        diagnostics.global_ba_num_edges = int(result.num_edges)
        diagnostics.global_ba_num_inliers = int(result.num_inliers)
        diagnostics.global_ba_num_outliers = int(result.num_outliers)
        diagnostics.global_ba_mean_error_before = result.mean_error_before
        diagnostics.global_ba_mean_error_after = result.mean_error_after
        diagnostics.global_ba_elapsed_sec = float(result.elapsed_sec)


# Provide a no-op lock interface when the map has no update lock.
class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _estimate_rmse(estimate: Sim3Estimate) -> Optional[float]:
    if estimate is None or estimate.inlier_mask is None:
        return None
    if not np.isfinite(estimate.mean_error):
        return None
    return float(estimate.mean_error)


def _rotation_angle_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> Optional[float]:
    try:
        Ra = np.asarray(rotation_a, dtype=np.float64).reshape(3, 3)
        Rb = np.asarray(rotation_b, dtype=np.float64).reshape(3, 3)
        trace_value = float((np.trace(Ra.T @ Rb) - 1.0) * 0.5)
        trace_value = max(-1.0, min(1.0, trace_value))
        angle = float(np.degrees(np.arccos(trace_value)))
    except Exception:
        return None
    return angle if np.isfinite(angle) else None


def _distance_summary(distances) -> dict:
    values = np.asarray(distances, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }
