from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random
import numpy as np

from slam_core.common.types import Pose2
from slam_core.loop_closure import pose_relative, pose_translation_distance
from slam_core.matching.scan_to_submap import (
    ScanToSubmapMatcher,
    SubmapMatchRequest,
)
from carto.pose_graph.pose_graph_2d import PoseGraph2D


@dataclass
class TrajectoryNodeRecord:
    """
    Persistent trajectory-node record used for loop-closure search.
    """
    node_id: int
    timestamp: float
    scan_points: np.ndarray
    pose_global: Pose2
    insertion_submap_ids: List[int]


@dataclass
class ConstraintBuilderEvent:
    """
    Diagnostic record for one candidate node-to-submap evaluation.
    """
    node_id: int
    target_id: int
    score: float
    accepted: bool
    status: str
    source: str


@dataclass
class ConstraintBuilder2DConfig:
    """
    Configuration of the loop-constraint generation policy.

    All defaults are calibrated to the official Cartographer pose_graph.lua values.
    Tuned for 2D indoor LIDAR SLAM (e.g., fr079 benchmark dataset).

    References:
        cartographer/configuration_files/pose_graph.lua
        cartographer/mapping/internal/2d/pose_graph_2d.cc
        cartographer/mapping/internal/constraints/constraint_builder_2d.cc
    """

    # === Candidate filtering (from constraint_builder options in pose_graph.lua) ===
    # Original: max_constraint_distance = 15.0m
    max_constraint_distance: float = 15.0

    # Original: sampling_ratio = 0.3 (random thinning of candidates per submap)
    sampling_ratio: float = 0.3

    # Original: min_score = 0.55 (constrained local-search acceptance threshold)
    min_score: float = 0.55

    # Original: global_localization_min_score = 0.6 (full-submap search threshold)
    global_localization_min_score: float = 0.60

    # Original: optimize_every_n_nodes = 90
    optimize_every_n_nodes: int = 90

    # === Candidate budgeting ===
    # 0 = no cap (Cartographer does not hard-cap targets per new node)
    new_node_max_targets: int = 0
    finished_submap_max_candidates: int = 0

    # Original: search all nodes, no stride (stride=1 equivalent)
    finished_submap_node_stride: int = 1

    # === Separation gates ===
    min_node_index_separation: int = 0
    recent_finished_submap_exclusion: int = 0

    # === Loop closure constraint weights (from pose_graph.lua) ===
    # Original: loop_closure_translation_weight = 1.1e4
    # Original: loop_closure_rotation_weight    = 1e5
    # These must be in the same scale as matcher weights (5e2/1.6e3) and
    # local-trajectory regularization weights (1e5/1e5).
    loop_translation_weight: float = 1.1e4
    loop_rotation_weight: float = 1.0e5

    sampler_seed: int = 0


class FixedRatioSampler:
    """
    Deterministic fixed-ratio sampler for candidate thinning.
    """

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
    Cartographer-like 2D loop-constraint builder with explicit diagnostics.

    The builder evaluates two search paths:
    1. new node -> old finished submaps
    2. newly finished submap -> old nodes
    """

    def __init__(
        self,
        loop_matcher: ScanToSubmapMatcher,
        pose_graph: PoseGraph2D,
        config: Optional[ConstraintBuilder2DConfig] = None,
    ) -> None:
        self.loop_matcher = loop_matcher
        self.pose_graph = pose_graph
        self.config = config or ConstraintBuilder2DConfig()

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
        """
        Register a newly inserted trajectory node.
        """
        self.nodes[int(node_id)] = TrajectoryNodeRecord(
            node_id=int(node_id),
            timestamp=float(timestamp),
            scan_points=np.asarray(scan_points, dtype=float),
            pose_global=pose_global,
            insertion_submap_ids=[int(x) for x in insertion_submap_ids],
        )

    def maybe_add_constraints_for_new_node(self, node_id: int) -> None:
        """
        Evaluate one newly inserted node against older finished submaps.
        """
        source = "new_node_search"

        node = self.nodes[int(node_id)]
        finished_submaps = self.loop_matcher.submap_builder.get_finished_submaps()
        if not finished_submaps:
            return

        newest_finished_id = max(int(sm.id) for sm in finished_submaps)

        # Prefer the nearest finished submaps for incremental loop search.
        finished_submaps = sorted(
            finished_submaps,
            key=lambda sm: pose_translation_distance(
                node.pose_global,
                self.pose_graph.get_submap_pose(int(sm.id)),
            ),
        )

        max_targets = int(self.config.new_node_max_targets)
        if max_targets > 0:
            finished_submaps = finished_submaps[:max_targets]

        for submap in finished_submaps:
            sid = int(submap.id)
            pair_key = (int(node.node_id), sid)

            if sid in node.insertion_submap_ids:
                continue

            if pair_key in self._accepted_pairs:
                self.num_duplicate_pairs += 1
                self._record_event(node.node_id, sid, np.nan, False, "duplicate", source)
                continue

            if (
                int(self.config.recent_finished_submap_exclusion) > 0
                and (newest_finished_id - sid) < int(self.config.recent_finished_submap_exclusion)
            ):
                continue

            if (
                int(self.config.min_node_index_separation) > 0
                and int(node.node_id) < int(self.config.min_node_index_separation)
            ):
                continue

            submap_pose = self.pose_graph.get_submap_pose(sid)
            if pose_translation_distance(node.pose_global, submap_pose) > float(self.config.max_constraint_distance):
                continue

            self.num_candidate_pairs += 1

            if not self._per_submap_sampler(sid).pulse():
                self.num_rejected_pairs += 1
                self._record_event(node.node_id, sid, np.nan, False, "sampled_out", source)
                continue

            status, score, matched_pose_world = self._verify_pair(
                node=node,
                submap=submap,
                source=source,
            )

            if matched_pose_world is None:
                self.num_rejected_pairs += 1
                self._record_event(node.node_id, sid, score, False, status, source)
                continue

            rel = pose_relative(
                from_pose=submap_pose,
                to_pose=matched_pose_world,
            )

            self.pose_graph.add_loop_submap_node_constraint(
                submap_id=sid,
                node_id=int(node.node_id),
                relative_pose=rel,
                translation_weight=float(self.config.loop_translation_weight),
                rotation_weight=float(self.config.loop_rotation_weight),
                match_score=float(score),
            )

            self._accepted_pairs.add(pair_key)
            self.num_loop_constraints += 1
            self._record_event(node.node_id, sid, score, True, "accepted", source)

    def maybe_add_constraints_for_finished_submap(self, submap_id: int) -> None:
        """
        Evaluate one newly finished submap against older trajectory nodes.
        """
        source = "finished_submap_search"

        sid = int(submap_id)
        submap = self.loop_matcher.submap_builder.get_submap_by_id(sid)
        latest_node_id = max(self.nodes.keys()) if self.nodes else -1
        submap_pose = self.pose_graph.get_submap_pose(sid)

        # Build a prefiltered candidate list before expensive verification.
        candidate_nodes: List[TrajectoryNodeRecord] = []

        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]

            if sid in node.insertion_submap_ids:
                continue

            if (
                int(self.config.min_node_index_separation) > 0
                and (latest_node_id - int(node.node_id)) < int(self.config.min_node_index_separation)
            ):
                continue

            if pose_translation_distance(node.pose_global, submap_pose) > float(self.config.max_constraint_distance):
                continue

            candidate_nodes.append(node)

        # Prefer nodes whose current pose estimate lies closest to the
        # newly finished submap.
        candidate_nodes.sort(
            key=lambda node: pose_translation_distance(node.pose_global, submap_pose)
        )

        stride = max(1, int(self.config.finished_submap_node_stride))
        candidate_nodes = candidate_nodes[::stride]

        max_candidates = int(self.config.finished_submap_max_candidates)
        if max_candidates > 0:
            candidate_nodes = candidate_nodes[:max_candidates]

        for node in candidate_nodes:
            pair_key = (int(node.node_id), sid)

            if pair_key in self._accepted_pairs:
                self.num_duplicate_pairs += 1
                self._record_event(node.node_id, sid, np.nan, False, "duplicate", source)
                continue

            self.num_candidate_pairs += 1

            if not self._per_submap_sampler(sid).pulse():
                self.num_rejected_pairs += 1
                self._record_event(node.node_id, sid, np.nan, False, "sampled_out", source)
                continue

            status, score, matched_pose_world = self._verify_pair(
                node=node,
                submap=submap,
                source=source,
            )

            if matched_pose_world is None:
                self.num_rejected_pairs += 1
                self._record_event(node.node_id, sid, score, False, status, source)
                continue

            rel = pose_relative(
                from_pose=submap_pose,
                to_pose=matched_pose_world,
            )

            self.pose_graph.add_loop_submap_node_constraint(
                submap_id=sid,
                node_id=int(node.node_id),
                relative_pose=rel,
                translation_weight=float(self.config.loop_translation_weight),
                rotation_weight=float(self.config.loop_rotation_weight),
                match_score=float(score),
            )

            self._accepted_pairs.add(pair_key)
            self.num_loop_constraints += 1
            self._record_event(node.node_id, sid, score, True, "accepted", source)

    def maybe_optimize(self) -> None:
        """
        Trigger batch graph optimization at the configured interval.
        """
        if len(self.nodes) > 0 and (len(self.nodes) % int(self.config.optimize_every_n_nodes)) == 0:
            self.pose_graph.solve()

    def get_stats(self) -> dict:
        """
        Return accumulated loop-closure counts.
        """
        return {
            "candidate_pairs": int(self.num_candidate_pairs),
            "accepted_pairs": int(self.num_loop_constraints),
            "rejected_pairs": int(self.num_rejected_pairs),
            "duplicate_pairs": int(self.num_duplicate_pairs),
        }

    def get_recent_events(self, n: int = 20) -> List[ConstraintBuilderEvent]:
        """
        Return the most recent diagnostic events.
        """
        return list(self.events[-int(n):])

    def get_diagnostics_summary(self) -> dict:
        """
        Return a compact diagnostic summary for accepted and rejected pairs.
        """
        accepted_events = [ev for ev in self.events if ev.accepted]
        rejected_events = [ev for ev in self.events if not ev.accepted]

        accepted_scores = np.asarray(
            [ev.score for ev in accepted_events if np.isfinite(ev.score)],
            dtype=float,
        )
        score_failed_scores = np.asarray(
            [
                ev.score
                for ev in rejected_events
                if ev.status == "score_failed" and np.isfinite(ev.score)
            ],
            dtype=float,
        )

        def _count(events: List[ConstraintBuilderEvent], **conds) -> int:
            c = 0
            for ev in events:
                ok = True
                for k, v in conds.items():
                    if getattr(ev, k) != v:
                        ok = False
                        break
                if ok:
                    c += 1
            return c

        summary = {
            "accepted_from_new_node_search": _count(
                accepted_events, source="new_node_search", status="accepted"
            ),
            "accepted_from_finished_submap_search": _count(
                accepted_events, source="finished_submap_search", status="accepted"
            ),
            "rejected_sampled_out": _count(
                rejected_events, status="sampled_out"
            ),
            "rejected_matcher_failed": _count(
                rejected_events, status="matcher_failed"
            ),
            "rejected_score_failed": _count(
                rejected_events, status="score_failed"
            ),
            "rejected_duplicate": _count(
                rejected_events, status="duplicate"
            ),
            "accepted_score_count": int(accepted_scores.size),
            "score_failed_count": int(score_failed_scores.size),
        }

        if accepted_scores.size > 0:
            summary.update(
                {
                    "accepted_score_min": float(np.min(accepted_scores)),
                    "accepted_score_mean": float(np.mean(accepted_scores)),
                    "accepted_score_median": float(np.median(accepted_scores)),
                    "accepted_score_max": float(np.max(accepted_scores)),
                    "accepted_near_min_score_count": int(
                        np.sum(accepted_scores <= (float(self.config.min_score) + 1e-3))
                    ),
                }
            )

        if score_failed_scores.size > 0:
            summary.update(
                {
                    "score_failed_min": float(np.min(score_failed_scores)),
                    "score_failed_mean": float(np.mean(score_failed_scores)),
                    "score_failed_median": float(np.median(score_failed_scores)),
                    "score_failed_max": float(np.max(score_failed_scores)),
                }
            )

        return summary

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    def _verify_pair(
        self,
        node: TrajectoryNodeRecord,
        submap,
        source: str,
    ) -> Tuple[str, float, Optional[Pose2]]:
        """
        Verify one candidate node-to-submap pair.

        Search mode depends on the source path:
        - new_node_search: constrained local search
        - finished_submap_search: broader full-submap search
        """
        use_full_submap = (source == "finished_submap_search")

        request = SubmapMatchRequest(
            scan_points_local=node.scan_points,
            predicted_pose_world=node.pose_global,
            submap_pose_world=self.pose_graph.get_submap_pose(int(submap.id)),
            submap=submap,
            timestamp=float(node.timestamp),
            odom_pose_world=None,
            match_full_submap=use_full_submap,
        )

        response = self.loop_matcher.match_against_submap(request)

        score = float(response.score) if hasattr(response, "score") else float("nan")

        if not response.success:
            return "matcher_failed", score, None

        matched_pose_world = response.pose_world

        required_score = (
            float(self.config.global_localization_min_score)
            if use_full_submap
            else float(self.config.min_score)
        )

        if score < required_score:
            return "score_failed", score, None

        return "accepted", score, matched_pose_world

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _record_event(
        self,
        node_id: int,
        target_id: int,
        score: float,
        accepted: bool,
        status: str,
        source: str,
    ) -> None:
        """
        Append one bounded diagnostic event record.
        """
        self.events.append(
            ConstraintBuilderEvent(
                node_id=int(node_id),
                target_id=int(target_id),
                score=float(score),
                accepted=bool(accepted),
                status=str(status),
                source=str(source),
            )
        )

        if len(self.events) > 5000:
            self.events = self.events[-5000:]

    def _per_submap_sampler(self, submap_id: int) -> FixedRatioSampler:
        """
        Return the persistent sampler associated with one target submap.
        """
        submap_id = int(submap_id)
        if submap_id not in self._per_submap_samplers:
            self._per_submap_samplers[submap_id] = FixedRatioSampler(
                ratio=float(self.config.sampling_ratio),
                seed=int(self.config.sampler_seed + submap_id),
            )
        return self._per_submap_samplers[submap_id]