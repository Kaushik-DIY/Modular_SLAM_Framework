from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random
import numpy as np

from slam_core.common.types import Pose2
from slam_core.common.se2 import inverse_pose, pose_compose, wrap_angle
from slam_core.loop_closure import pose_relative, pose_translation_distance
from slam_core.matching.scan_to_submap import (
    CartoRefinementProblem,
    ScanToSubmapMatcher,
    Submap2D,
)
from carto.pose_graph.pose_graph_2d import PoseGraph2D
from carto.pose_graph.scan_matcher_cache_2d import SubmapScanMatcherCache2D


@dataclass
class TrajectoryNodeRecord:
    node_id: int
    timestamp: float
    scan_points: np.ndarray
    pose_global: Pose2
    insertion_submap_ids: List[int]


@dataclass
class ConstraintBuilderEvent:
    node_id: int
    target_id: int
    score: float
    accepted: bool
    status: str


@dataclass
class ConstraintBuilder2DConfig:
    # Cartographer-style candidate policy
    max_constraint_distance: float = 4.5
    sampling_ratio: float = 0.60
    min_score: float = 0.66
    optimize_every_n_nodes: int = 100

    # Candidate eligibility
    min_node_index_separation: int = 40
    recent_finished_submap_exclusion: int = 3

    # Coarse scan matching
    min_valid: int = 20
    coarse_level: int = 2
    coarse_xy_window: float = 0.8
    coarse_th_window: float = 0.3
    coarse_xy_step: float = 0.20
    coarse_th_step: float = 0.08

    # Refinement
    do_refine: bool = True
    max_match_points: int = 40
    max_refine_points: int = 120
    refine_min_points: int = 25

    # Consistency gate between predicted and measured relative pose
    consistency_max_translation_delta: float = 1.0
    consistency_max_rotation_delta: float = np.deg2rad(20.0)

    # Base loop weights
    loop_translation_weight: float = 10.0
    loop_rotation_weight: float = 50.0

    # Score-aware weighting
    min_score_for_weight: float = 0.66
    max_score_for_weight: float = 0.90
    min_weight_scale: float = 0.50
    max_weight_scale: float = 1.50

    sampler_seed: int = 7


class FixedRatioSampler:
    def __init__(self, ratio: float, seed: int = 0) -> None:
        self.ratio = float(ratio)
        self._rng = random.Random(int(seed))

    def pulse(self) -> bool:
        if self.ratio >= 1.0:
            return True
        if self.ratio <= 0.0:
            return False
        return self._rng.random() < self.ratio


class ConstraintBuilder2D:
    """
    Cartographer-style 2D constraint builder.

    This module is responsible for:
      1. candidate gating,
      2. per-submap sampling,
      3. scan-to-finished-submap verification,
      4. consistency gating,
      5. loop-constraint creation,
      6. batched optimization triggering.
    """

    def __init__(
        self,
        matcher: ScanToSubmapMatcher,
        pose_graph: PoseGraph2D,
        config: Optional[ConstraintBuilder2DConfig] = None,
    ) -> None:
        self.matcher = matcher
        self.pose_graph = pose_graph
        self.config = config or ConstraintBuilder2DConfig()

        self.cache = SubmapScanMatcherCache2D(
            precomp_levels=int(self.matcher.corr_params.get("precomp_levels", 3))
        )

        self.nodes: Dict[int, TrajectoryNodeRecord] = {}
        self._accepted_pairs: set[tuple[int, int]] = set()
        self._per_submap_samplers: Dict[int, FixedRatioSampler] = {}

        self.num_loop_constraints = 0
        self.num_candidate_pairs = 0
        self.num_rejected_pairs = 0
        self.num_duplicate_pairs = 0

        self.events: List[ConstraintBuilderEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_node(
        self,
        node_id: int,
        timestamp: float,
        scan_points: np.ndarray,
        pose_global: Pose2,
        insertion_submap_ids: List[int],
    ) -> None:
        self.nodes[int(node_id)] = TrajectoryNodeRecord(
            node_id=int(node_id),
            timestamp=float(timestamp),
            scan_points=np.asarray(scan_points, dtype=float),
            pose_global=pose_global,
            insertion_submap_ids=[int(x) for x in insertion_submap_ids],
        )

    def maybe_add_constraints_for_new_node(self, node_id: int) -> None:
        node = self.nodes[int(node_id)]
        finished_submaps = self.matcher.submap_builder.get_finished_submaps()
        if not finished_submaps:
            return

        newest_finished_id = max(int(sm.id) for sm in finished_submaps)

        for submap in finished_submaps:
            sid = int(submap.id)
            pair_key = (int(node.node_id), sid)

            if sid in node.insertion_submap_ids:
                continue
            if pair_key in self._accepted_pairs:
                self.num_duplicate_pairs += 1
                self._record_event(node.node_id, sid, None, False, "duplicate")
                continue
            if (newest_finished_id - sid) < int(self.config.recent_finished_submap_exclusion):
                continue
            if int(node.node_id) < int(self.config.min_node_index_separation):
                continue

            submap_pose = self.pose_graph.get_submap_pose(sid)
            if pose_translation_distance(node.pose_global, submap_pose) > float(self.config.max_constraint_distance):
                continue

            self.num_candidate_pairs += 1

            if not self._per_submap_sampler(sid).pulse():
                self.num_rejected_pairs += 1
                self._record_event(node.node_id, sid, None, False, "sampled_out")
                continue

            verified = self._verify_pair(node=node, submap=submap)
            if verified is None:
                self.num_rejected_pairs += 1
                self._record_event(node.node_id, sid, None, False, "verify_failed")
                continue

            score, matched_pose_world = verified

            ok, status = self._passes_consistency_gate(
                node_pose_world=node.pose_global,
                submap_id=sid,
                matched_pose_world=matched_pose_world,
            )
            if not ok:
                self.num_rejected_pairs += 1
                self._record_event(node.node_id, sid, score, False, status)
                continue

            tw, rw = self._score_to_weights(score)

            rel = pose_relative(
                from_pose=self.pose_graph.get_submap_pose(sid),
                to_pose=matched_pose_world,
            )

            self.pose_graph.add_loop_submap_node_constraint(
                submap_id=sid,
                node_id=int(node.node_id),
                relative_pose=rel,
                translation_weight=tw,
                rotation_weight=rw,
                match_score=float(score),
            )

            self._accepted_pairs.add(pair_key)
            self.num_loop_constraints += 1
            self._record_event(node.node_id, sid, score, True, "accepted")

    def maybe_add_constraints_for_finished_submap(self, submap_id: int) -> None:
        sid = int(submap_id)
        submap = self.matcher.submap_builder.get_submap_by_id(sid)
        latest_node_id = max(self.nodes.keys()) if self.nodes else -1

        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            pair_key = (int(node.node_id), sid)

            if sid in node.insertion_submap_ids:
                continue
            if pair_key in self._accepted_pairs:
                self.num_duplicate_pairs += 1
                self._record_event(node.node_id, sid, None, False, "duplicate")
                continue
            if (latest_node_id - int(node.node_id)) < int(self.config.min_node_index_separation):
                continue

            submap_pose = self.pose_graph.get_submap_pose(sid)
            if pose_translation_distance(node.pose_global, submap_pose) > float(self.config.max_constraint_distance):
                continue

            self.num_candidate_pairs += 1

            if not self._per_submap_sampler(sid).pulse():
                self.num_rejected_pairs += 1
                self._record_event(node.node_id, sid, None, False, "sampled_out")
                continue

            verified = self._verify_pair(node=node, submap=submap)
            if verified is None:
                self.num_rejected_pairs += 1
                self._record_event(node.node_id, sid, None, False, "verify_failed")
                continue

            score, matched_pose_world = verified

            ok, status = self._passes_consistency_gate(
                node_pose_world=node.pose_global,
                submap_id=sid,
                matched_pose_world=matched_pose_world,
            )
            if not ok:
                self.num_rejected_pairs += 1
                self._record_event(node.node_id, sid, score, False, status)
                continue

            tw, rw = self._score_to_weights(score)

            rel = pose_relative(
                from_pose=self.pose_graph.get_submap_pose(sid),
                to_pose=matched_pose_world,
            )

            self.pose_graph.add_loop_submap_node_constraint(
                submap_id=sid,
                node_id=int(node.node_id),
                relative_pose=rel,
                translation_weight=tw,
                rotation_weight=rw,
                match_score=float(score),
            )

            self._accepted_pairs.add(pair_key)
            self.num_loop_constraints += 1
            self._record_event(node.node_id, sid, score, True, "accepted")

    def maybe_optimize(self) -> None:
        if len(self.nodes) > 0 and (len(self.nodes) % int(self.config.optimize_every_n_nodes)) == 0:
            self.pose_graph.solve()

    def get_stats(self) -> dict:
        return {
            "candidate_pairs": int(self.num_candidate_pairs),
            "accepted_pairs": int(self.num_loop_constraints),
            "rejected_pairs": int(self.num_rejected_pairs),
            "duplicate_pairs": int(self.num_duplicate_pairs),
        }

    def get_recent_events(self, n: int = 20) -> List[ConstraintBuilderEvent]:
        return list(self.events[-int(n):])

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    def _verify_pair(
        self,
        node: TrajectoryNodeRecord,
        submap: Submap2D,
    ) -> Optional[Tuple[float, Pose2]]:
        sid = int(submap.id)
        submap_pose = self.pose_graph.get_submap_pose(sid)

        # Predicted node pose in the chosen submap frame.
        pred_sub = pose_compose(inverse_pose(submap_pose), node.pose_global)

        pts_match = node.scan_points
        max_match_pts = int(self.config.max_match_points)
        if pts_match.shape[0] > max_match_pts:
            stride = max(1, pts_match.shape[0] // max_match_pts)
            pts_match = pts_match[::stride]

        coarse_pose_sub, coarse_score = self.cache.coarse_match(
            submap=submap,
            points_local=pts_match,
            initial_submap_pose=pred_sub,
            min_valid=int(self.config.min_valid),
            coarse_level=int(self.config.coarse_level),
            coarse_xy_window=float(self.config.coarse_xy_window),
            coarse_th_window=float(self.config.coarse_th_window),
            coarse_xy_step=float(self.config.coarse_xy_step),
            coarse_th_step=float(self.config.coarse_th_step),
        )

        if coarse_score < float(self.config.min_score):
            return None

        refined_sub = coarse_pose_sub

        if bool(self.config.do_refine):
            x0 = np.array([coarse_pose_sub.x, coarse_pose_sub.y, coarse_pose_sub.theta], dtype=float)
            xpred = np.array([pred_sub.x, pred_sub.y, pred_sub.theta], dtype=float)

            refine_pts = node.scan_points
            max_refine_pts = int(self.config.max_refine_points)
            if refine_pts.shape[0] > max_refine_pts:
                stride = max(1, refine_pts.shape[0] // max_refine_pts)
                refine_pts = refine_pts[::stride]

            problem = CartoRefinementProblem(
                grid=submap.grid,
                pts_local=refine_pts,
                pred_pose_sub=xpred,
                min_points=int(self.config.refine_min_points),
                w_trans=float(self.matcher.corr_params.get("refine_w_trans", 1.0)),
                w_rot=float(self.matcher.corr_params.get("refine_w_rot", 1.0)),
            )

            x_opt = self.matcher.refine_solver.solve(x0, problem.compute_r_J).reshape(3)
            x_opt[2] = wrap_angle(x_opt[2])
            refined_sub = Pose2(float(x_opt[0]), float(x_opt[1]), float(x_opt[2]))

        matched_pose_world = pose_compose(submap_pose, refined_sub)
        return float(coarse_score), matched_pose_world

    # ------------------------------------------------------------------
    # Acceptance improvements
    # ------------------------------------------------------------------
    def _passes_consistency_gate(
        self,
        node_pose_world: Pose2,
        submap_id: int,
        matched_pose_world: Pose2,
    ) -> Tuple[bool, str]:
        submap_pose = self.pose_graph.get_submap_pose(int(submap_id))

        predicted_rel = pose_compose(inverse_pose(submap_pose), node_pose_world)
        measured_rel = pose_compose(inverse_pose(submap_pose), matched_pose_world)

        dx = measured_rel.x - predicted_rel.x
        dy = measured_rel.y - predicted_rel.y
        dtrans = float(np.hypot(dx, dy))
        drot = float(abs(wrap_angle(measured_rel.theta - predicted_rel.theta)))

        if dtrans > float(self.config.consistency_max_translation_delta):
            return False, "consistency_translation_failed"
        if drot > float(self.config.consistency_max_rotation_delta):
            return False, "consistency_rotation_failed"

        return True, "accepted"

    def _score_to_weights(self, score: float) -> Tuple[float, float]:
        s0 = float(self.config.min_score_for_weight)
        s1 = float(self.config.max_score_for_weight)

        if s1 <= s0:
            alpha = 1.0
        else:
            alpha = (float(score) - s0) / (s1 - s0)
            alpha = float(np.clip(alpha, 0.0, 1.0))

        scale = (
            float(self.config.min_weight_scale)
            + alpha * (float(self.config.max_weight_scale) - float(self.config.min_weight_scale))
        )

        tw = float(self.config.loop_translation_weight) * scale
        rw = float(self.config.loop_rotation_weight) * scale
        return tw, rw

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _record_event(
        self,
        node_id: int,
        target_id: int,
        score: Optional[float],
        accepted: bool,
        status: str,
    ) -> None:
        score_value = float(score) if score is not None else float("nan")
        self.events.append(
            ConstraintBuilderEvent(
                node_id=int(node_id),
                target_id=int(target_id),
                score=score_value,
                accepted=bool(accepted),
                status=str(status),
            )
        )
        if len(self.events) > 5000:
            self.events = self.events[-5000:]

    def _per_submap_sampler(self, submap_id: int) -> FixedRatioSampler:
        submap_id = int(submap_id)
        if submap_id not in self._per_submap_samplers:
            self._per_submap_samplers[submap_id] = FixedRatioSampler(
                ratio=float(self.config.sampling_ratio),
                seed=int(self.config.sampler_seed + submap_id),
            )
        return self._per_submap_samplers[submap_id]