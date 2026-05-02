"""
pySLAM-aligned loop detection, verification, and correction scaffold.

Reference:
- pySLAM: pyslam/loop_closing/loop_closing.py

This is the RGB-D sparse path. The control flow follows pySLAM:
candidate query, covisibility consistency, geometry verification, loop
correction, map-point fusion, and diagnostics. The geometry solver is a
scale-fixed Sim3/SE3 compatibility path because pySLAM's C++ sim3solver module
is not available locally.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
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


@dataclass
class ConsistencyGroup:
    keyframes: set = field(default_factory=set)
    consistency: int = 0


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


class LoopGroupConsistencyChecker:
    def __init__(self, consistency_threshold: int = 3):
        self.consistent_groups: list[ConsistencyGroup] = []
        self.consistency_threshold = int(consistency_threshold)
        self.enough_consistent_candidates: list[KeyFrame] = []

    def clear_consistency_groups(self) -> None:
        self.consistent_groups = []
        self.enough_consistent_candidates = []

    def check_candidates(self, current_keyframe: KeyFrame, candidate_keyframes: list[KeyFrame]) -> bool:
        self.enough_consistent_candidates = []
        current_consistent_groups = []
        group_updated = [False] * len(self.consistent_groups)

        for candidate in candidate_keyframes:
            if candidate is None or candidate.is_bad():
                continue

            candidate_group = set(candidate.get_connected_keyframes())
            candidate_group.add(candidate)

            enough_consistent = False
            consistent_for_some_group = False

            for idx, previous_group in enumerate(self.consistent_groups):
                if candidate_group.intersection(previous_group.keyframes):
                    consistent_for_some_group = True
                    current_consistency = previous_group.consistency + 1

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

        self.consistent_groups = current_consistent_groups
        return len(self.enough_consistent_candidates) > 0


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

        for candidate in candidate_keyframes:
            if candidate is None or candidate is current_keyframe or candidate.is_bad():
                continue

            idxs_current, idxs_candidate = self.match_keyframes(current_keyframe, candidate)
            self.num_last_matches = max(self.num_last_matches, len(idxs_current))

            if len(idxs_current) < self.min_matches:
                self.last_error = "too few loop geometry matches"
                continue

            points_current, points_loop, idxs_current, idxs_candidate = self._matched_3d_points(
                current_keyframe,
                candidate,
                idxs_current,
                idxs_candidate,
            )
            if len(points_current) < self.min_matches:
                self.last_error = "too few valid 3D loop correspondences"
                continue

            estimate = estimate_scale_fixed_sim3(
                points_current,
                points_loop,
                max_error=0.10,
            )
            self.num_last_inliers = max(self.num_last_inliers, int(np.sum(estimate.inlier_mask)))

            if not estimate.success or np.sum(estimate.inlier_mask) < self.min_matches:
                self.last_error = estimate.error
                continue

            matches = [None] * len(current_keyframe.points)
            match_idxs = np.full(len(current_keyframe.points), -1, dtype=np.int32)
            inlier_current = idxs_current[estimate.inlier_mask]
            inlier_candidate = idxs_candidate[estimate.inlier_mask]

            for idx_cur, idx_cand in zip(inlier_current, inlier_candidate):
                if 0 <= idx_cur < len(matches) and 0 <= idx_cand < len(candidate.points):
                    matches[int(idx_cur)] = candidate.points[int(idx_cand)]
                    match_idxs[int(idx_cur)] = int(idx_cand)

            candidate_group = candidate.get_covisible_keyframes()
            candidate_group.append(candidate)
            self.success_loop_map_points = set()
            for keyframe in candidate_group:
                for point in keyframe.get_matched_good_points():
                    if point is not None and not point.is_bad():
                        self.success_loop_map_points.add(point)

            self.success_loop_kf = candidate
            self.success_sim3 = estimate
            self.success_map_point_matches = matches
            self.success_map_point_matches_idxs = match_idxs
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
            else:
                self.last_global_ba_result = GlobalBAResult(started=False, reason="loop correction failed")

        self.last_result = result
        self.mean_graph_chi2_error = result.after_error
        return result

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


class LoopClosing:
    def __init__(self, slam, keyframe_database=None, consistency_threshold: int = 3):
        self.slam = slam
        self.keyframe_database = keyframe_database or getattr(slam, "keyframe_database", None)
        self.loop_detector = LoopDetector(self.keyframe_database)
        self.loop_consistency_checker = LoopGroupConsistencyChecker(consistency_threshold)
        self.loop_geometry_checker = LoopGeometryChecker(keyframe_database=self.keyframe_database)
        self.loop_corrector = LoopCorrector(slam, self.loop_geometry_checker)
        self.queue = deque()
        self.last_loop_kf_id = 0
        self.last_diagnostics = LoopDiagnostics()
        self.mean_graph_chi2_error = None
        self._is_correcting = False

    def is_correcting(self) -> bool:
        return bool(self._is_correcting)

    def insert_keyframe(self, keyframe: KeyFrame) -> None:
        self.add_keyframe(keyframe)

    def add_keyframe(self, keyframe: KeyFrame, img=None) -> None:
        if img is not None:
            keyframe.img = img
        self.queue.append(keyframe)

    def queue_size(self) -> int:
        return len(self.queue)

    def pop_keyframe(self):
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

        if len(detection_output.candidate_keyframes) == 0:
            diagnostics.rejected_by_bow = 1
            self.loop_consistency_checker.clear_consistency_groups()
            return False

        got_consistent = self.loop_consistency_checker.check_candidates(
            keyframe,
            detection_output.candidate_keyframes,
        )

        if not got_consistent:
            diagnostics.rejected_by_consistency = diagnostics.candidates
            return False

        consistent_candidates = [
            candidate
            for candidate in self.loop_consistency_checker.enough_consistent_candidates
            if not candidate.is_bad()
        ]
        got_geometry = self.loop_geometry_checker.check_candidates(keyframe, consistent_candidates)

        if not got_geometry:
            diagnostics.rejected_by_geometry = len(consistent_candidates)
            return False

        self._is_correcting = True
        try:
            result = self.loop_corrector.correct_loop(keyframe)
        finally:
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
            self.last_loop_kf_id = keyframe.kid
            self.mean_graph_chi2_error = result.after_error
            return True

        diagnostics.rejected_by_geometry = len(consistent_candidates)
        return False

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


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
