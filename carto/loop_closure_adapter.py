from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from slam_core.common.types import Pose2
from slam_core.common.se2 import inverse_pose, pose_compose, wrap_angle
from slam_core.loop_closure import (
    ClosureTarget,
    ConstraintSink,
    LoopClosureConfig,
    LoopClosureManager,
    LoopConstraint,
    LoopMatchResult,
    LoopNode,
    LoopVerifier,
    TargetProvider,
    pose_translation_distance,
)
from slam_core.matching.scan_to_submap import ScanToSubmapMatcher, Submap2D
from slam_core.matching.scan_to_submap.types import SubmapMatchRequest

from carto.pose_graph.pose_graph_2d import PoseGraph2D


def _relative_pose(from_pose: Pose2, to_pose: Pose2) -> Pose2:
    """
    Compute T_from^{-1} * T_to.
    """
    return pose_compose(inverse_pose(from_pose), to_pose)


def _pose_residual(pred_rel: Pose2, match_rel: Pose2) -> tuple[float, float]:
    """
    Residual between two relative poses in SE(2).

    Returns:
        translation_residual_m, rotation_residual_rad
    """
    delta = _relative_pose(pred_rel, match_rel)
    trans_residual = float(np.hypot(delta.x, delta.y))
    rot_residual = float(abs(wrap_angle(delta.theta)))
    return trans_residual, rot_residual


class CartoTargetProvider(TargetProvider):
    def __init__(self, matcher: ScanToSubmapMatcher, pose_graph: PoseGraph2D) -> None:
        self.matcher = matcher
        self.pose_graph = pose_graph

    def get_candidate_targets_for_node(
        self,
        node: LoopNode,
        all_nodes: Dict[int, LoopNode],
        config: LoopClosureConfig,
    ) -> List[ClosureTarget]:
        if len(all_nodes) <= int(config.min_node_index_separation):
            return []

        finished_submaps = self.matcher.submap_builder.get_finished_submaps()
        if not finished_submaps:
            return []

        newest_finished_id = max(int(sm.id) for sm in finished_submaps)

        ranked = []
        for submap in finished_submaps:
            submap_id_int = int(submap.id)
            submap_id = str(submap_id_int)

            if submap_id in node.local_target_ids:
                continue
            if (newest_finished_id - submap_id_int) < int(config.recent_finished_submap_exclusion):
                continue
            if getattr(submap, "num_inserted", 0) < 20:
                continue

            target_pose = self.pose_graph.get_submap_pose(submap_id_int)
            dist = pose_translation_distance(node.pose_guess_global, target_pose)
            if dist > float(config.spatial_search_radius):
                continue

            ranked.append((dist, submap, target_pose))

        ranked.sort(key=lambda item: float(item[0]))

        max_targets = int(config.max_candidate_targets_per_new_node)
        if max_targets > 0:
            ranked = ranked[:max_targets]

        targets: List[ClosureTarget] = []
        for _, submap, target_pose in ranked:
            targets.append(
                ClosureTarget(
                    target_id=str(int(submap.id)),
                    target_type="submap",
                    pose_global=target_pose,
                    is_finished=True,
                    is_fixed=False,
                    map_view=submap,
                    match_full_submap=False,
                    search_source="new_node_search",
                )
            )
        return targets

    def get_finished_target(self, target_id: str) -> ClosureTarget:
        submap = self.matcher.submap_builder.get_submap_by_id(int(target_id))
        target_pose = self.pose_graph.get_submap_pose(int(target_id))
        return ClosureTarget(
            target_id=str(target_id),
            target_type="submap",
            pose_global=target_pose,
            is_finished=True,
            is_fixed=False,
            map_view=submap,
            match_full_submap=False,
            search_source="finished_submap_search",
        )

    def get_candidate_nodes_for_finished_target(
        self,
        target: ClosureTarget,
        all_nodes: Dict[int, LoopNode],
        config: LoopClosureConfig,
    ) -> List[LoopNode]:
        if not all_nodes:
            return []

        latest_node_id = max(all_nodes.keys())
        filtered: List[LoopNode] = []

        for node_id in sorted(all_nodes.keys()):
            node = all_nodes[node_id]

            if target.target_id in node.local_target_ids:
                continue
            if (latest_node_id - node.node_id) < int(config.min_node_index_separation):
                continue

            dist = pose_translation_distance(node.pose_guess_global, target.pose_global)
            if dist > float(config.spatial_search_radius):
                continue

            filtered.append(node)

        filtered.sort(
            key=lambda node: (
                pose_translation_distance(node.pose_guess_global, target.pose_global),
                int(node.node_id),
            )
        )

        stride = max(1, int(config.historical_node_stride))
        filtered = filtered[::stride]

        max_candidates = int(config.max_candidate_nodes_per_finished_target)
        if max_candidates > 0:
            filtered = filtered[:max_candidates]

        return filtered


class CartoLoopVerifier(LoopVerifier):
    """
    Geometric verification of loop-closure candidates via the active matcher backend.

    Acceptance order:
      1. matcher success
      2. score threshold
      3. geometric consistency w.r.t. current graph estimate
    """

    def __init__(self, matcher: ScanToSubmapMatcher, config: LoopClosureConfig) -> None:
        self.matcher = matcher
        self.config = config

    def verify(self, node: LoopNode, target: ClosureTarget) -> LoopMatchResult:
        request = SubmapMatchRequest(
            scan_points_local=np.asarray(node.scan_points, dtype=float),
            predicted_pose_world=node.pose_guess_global,
            submap_pose_world=target.pose_global,
            submap=target.map_view,
            timestamp=float(node.timestamp),
            match_full_submap=bool(target.match_full_submap),
        )

        try:
            response = self.matcher.match_against_submap(request)
        except Exception:
            return LoopMatchResult(
                success=False,
                score=0.0,
                matched_node_pose_global=None,
                status="matcher_failed",
                used_full_submap=bool(target.match_full_submap),
                translation_residual_m=None,
                rotation_residual_rad=None,
            )

        score = float(response.score)
        if not response.success:
            return LoopMatchResult(
                success=False,
                score=score,
                matched_node_pose_global=None,
                status="matcher_failed",
                used_full_submap=bool(target.match_full_submap),
                translation_residual_m=None,
                rotation_residual_rad=None,
            )

        required_score = (
            float(self.config.global_localization_min_score)
            if target.match_full_submap
            else float(self.config.min_score)
        )
        if score < required_score:
            return LoopMatchResult(
                success=False,
                score=score,
                matched_node_pose_global=None,
                status="score_failed",
                used_full_submap=bool(target.match_full_submap),
                translation_residual_m=None,
                rotation_residual_rad=None,
            )

        matched_pose_global = response.pose_world

        # --------------------------------------------------------------
        # Geometry-consistency gate
        # --------------------------------------------------------------
        pred_rel = _relative_pose(target.pose_global, node.pose_guess_global)
        match_rel = _relative_pose(target.pose_global, matched_pose_global)

        trans_residual_m, rot_residual_rad = _pose_residual(pred_rel, match_rel)

        if trans_residual_m > float(self.config.max_loop_translation_residual_m):
            return LoopMatchResult(
                success=False,
                score=score,
                matched_node_pose_global=None,
                status="geometry_failed",
                used_full_submap=bool(target.match_full_submap),
                translation_residual_m=trans_residual_m,
                rotation_residual_rad=rot_residual_rad,
            )

        if rot_residual_rad > float(self.config.max_loop_rotation_residual_rad):
            return LoopMatchResult(
                success=False,
                score=score,
                matched_node_pose_global=None,
                status="geometry_failed",
                used_full_submap=bool(target.match_full_submap),
                translation_residual_m=trans_residual_m,
                rotation_residual_rad=rot_residual_rad,
            )

        return LoopMatchResult(
            success=True,
            score=score,
            matched_node_pose_global=matched_pose_global,
            status="accepted",
            used_full_submap=bool(target.match_full_submap),
            translation_residual_m=trans_residual_m,
            rotation_residual_rad=rot_residual_rad,
        )


class CartoConstraintSink(ConstraintSink):
    def __init__(self, pose_graph: PoseGraph2D) -> None:
        self.pose_graph = pose_graph

    def add_loop_constraint(self, constraint: LoopConstraint) -> None:
        self.pose_graph.add_loop_submap_node_constraint(
            submap_id=int(constraint.target_id),
            node_id=int(constraint.node_id),
            relative_pose=constraint.relative_pose,
            translation_weight=constraint.translation_weight,
            rotation_weight=constraint.rotation_weight,
            match_score=float(getattr(constraint, "match_score", 1.0)),
        )

    def maybe_optimize(self, node_count: int, config: LoopClosureConfig) -> None:
        _ = node_count, config
        # Optimization scheduling is owned by CartoGlobalSlam2D so that every
        # solve is followed by synchronized submap write-back and live-state
        # correction. The active loop path therefore never solves directly here.
        return


class CartoLoopClosureAdapter:
    def __init__(
        self,
        matcher: ScanToSubmapMatcher,
        pose_graph: PoseGraph2D,
        config: Optional[LoopClosureConfig] = None,
    ) -> None:
        self.matcher = matcher
        self.pose_graph = pose_graph
        self.config = config or LoopClosureConfig()

        provider = CartoTargetProvider(matcher=self.matcher, pose_graph=self.pose_graph)
        verifier = CartoLoopVerifier(matcher=self.matcher, config=self.config)
        sink = CartoConstraintSink(pose_graph=self.pose_graph)

        self.manager = LoopClosureManager(
            config=self.config,
            provider=provider,
            verifier=verifier,
            sink=sink,
        )

    def on_new_node(
        self,
        node_id: int,
        scan_points,
        pose_guess_global: Pose2,
        timestamp: float,
        local_submap_ids: Optional[List[str]] = None,
    ) -> None:
        node = LoopNode(
            node_id=int(node_id),
            scan_points=np.asarray(scan_points, dtype=float),
            pose_guess_global=pose_guess_global,
            timestamp=float(timestamp),
            local_target_ids=list(local_submap_ids or []),
        )
        self.manager.on_new_node(node)

    def on_submap_finished(self, submap_id: int | str) -> None:
        self.manager.enqueue_finished_target(str(submap_id))

    def process_finished_submaps(self) -> int:
        finished_ids = self.matcher.submap_builder.consume_newly_finished_ids()
        for submap_id in finished_ids:
            self.on_submap_finished(submap_id)

        return self.manager.drain_pending_finished_targets(
            max_verifications=int(self.config.finished_submap_verification_budget_per_tick)
        )

    def finalize(self) -> int:
        return self.manager.finalize_pending_finished_targets()

    def get_stats(self):
        return self.manager.get_stats()

    def get_recent_events(self, n: int = 20):
        return self.manager.get_recent_events(n)

    def get_diagnostics_summary(self) -> dict:
        return self.manager.get_diagnostics_summary()