from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Protocol
import math

import numpy as np

from slam_core.common.se2 import pose_compose, pose_inverse
from slam_core.common.types import Pose2


def pose_relative(from_pose: Pose2, to_pose: Pose2) -> Pose2:
    """
    Compute the relative planar transform between two poses.

    The returned pose corresponds to:
        T_rel = T_from^{-1} * T_to
    """
    return pose_compose(pose_inverse(from_pose), to_pose)


def pose_translation_distance(a: Pose2, b: Pose2) -> float:
    """Euclidean distance between the translation parts of two planar poses."""
    return math.hypot(a.x - b.x, a.y - b.y)


@dataclass
class LoopNode:
    """Persistent node record used by the loop-closure layer."""

    node_id: int
    scan_points: np.ndarray
    pose_guess_global: Pose2
    timestamp: float
    local_target_ids: List[str] = field(default_factory=list)


@dataclass
class ClosureTarget:
    """
    Generic loop-closure target.

    `match_full_submap` drives the verifier's search mode:
      False -> constrained matching around the predicted pose
      True  -> broad full-submap matching
    """

    target_id: str
    target_type: str
    pose_global: Pose2
    is_finished: bool
    is_fixed: bool
    map_view: Any
    match_full_submap: bool = False
    search_source: Optional[str] = None


@dataclass
class LoopMatchResult:
    """Result of geometric verification for one node-target pair."""

    success: bool
    score: float
    matched_node_pose_global: Optional[Pose2]
    status: str = "matcher_failed"
    used_full_submap: bool = False
    translation_residual_m: Optional[float] = None
    rotation_residual_rad: Optional[float] = None


@dataclass
class LoopConstraint:
    """
    Generic loop-closure constraint stored in target-relative form:
        z_tj = T_t^{-1} * T_j
    """

    node_id: int
    target_id: str
    target_type: str
    relative_pose: Pose2
    translation_weight: float
    rotation_weight: float
    match_score: float = 1.0
    constraint_kind: str = "loop"
    is_fixed_target: bool = False


@dataclass
class LoopClosureEvent:
    node_id: int
    target_id: str
    score: float
    accepted: bool
    status: str
    source: str
    used_full_submap: bool


@dataclass
class LoopClosureStats:
    candidate_pairs: int = 0
    accepted_pairs: int = 0
    rejected_pairs: int = 0
    duplicate_pairs: int = 0


@dataclass
class LoopClosureConfig:
    """Configuration parameters governing loop-closure scheduling and gating."""

    min_score: float = 0.62
    global_localization_min_score: float = 0.72
    translation_weight: float = 25.0
    rotation_weight: float = 200.0
    optimize_every_n_nodes: int = 30
    check_every_n_nodes: int = 1
    min_node_index_separation: int = 20
    spatial_search_radius: float = 6.0
    max_candidate_targets_per_new_node: int = 4
    max_accepted_targets_per_new_node: int = 1
    recent_finished_submap_exclusion: int = 2
    historical_node_stride: int = 3
    max_candidate_nodes_per_finished_target: int = 0
    finished_submap_verification_budget_per_tick: int = 24
    force_full_submap_for_finished_submap_search: bool = False
    finished_submap_full_search_failure_threshold: int = 3
    max_loop_translation_residual_m: float = 1.0
    max_loop_rotation_residual_rad: float = 0.20944


class TargetProvider(Protocol):
    """Backend-specific interface for exposing loop-closure targets."""

    def get_candidate_targets_for_node(
        self,
        node: LoopNode,
        all_nodes: Dict[int, LoopNode],
        config: LoopClosureConfig,
    ) -> List[ClosureTarget]:
        ...

    def get_finished_target(self, target_id: str) -> ClosureTarget:
        ...

    def get_candidate_nodes_for_finished_target(
        self,
        target: ClosureTarget,
        all_nodes: Dict[int, LoopNode],
        config: LoopClosureConfig,
    ) -> List[LoopNode]:
        ...


class LoopVerifier(Protocol):
    """Backend-specific geometric verification interface."""

    def verify(self, node: LoopNode, target: ClosureTarget) -> LoopMatchResult:
        ...


class ConstraintSink(Protocol):
    """Backend-specific constraint insertion and optimization scheduling interface."""

    def add_loop_constraint(self, constraint: LoopConstraint) -> None:
        ...

    def maybe_optimize(self, node_count: int, config: LoopClosureConfig) -> None:
        ...


@dataclass
class _PendingFinishedTargetWork:
    target_id: str
    candidate_node_ids: List[int]
    next_index: int = 0
    consecutive_constrained_failures: int = 0


class LoopClosureManager:
    """
    Generic loop-closure manager.

    This manager preserves Cartographer-style two-way search semantics:
      1. new node -> old finished targets
      2. new finished target -> old nodes
    """

    def __init__(
        self,
        config: LoopClosureConfig,
        provider: TargetProvider,
        verifier: LoopVerifier,
        sink: ConstraintSink,
    ) -> None:
        self.config = config
        self.provider = provider
        self.verifier = verifier
        self.sink = sink

        self.nodes: Dict[int, LoopNode] = {}
        self._accepted_pairs: set[tuple[int, str]] = set()
        self.stats = LoopClosureStats()
        self.events: List[LoopClosureEvent] = []
        self._pending_finished_targets: List[_PendingFinishedTargetWork] = []

    def on_new_node(self, node: LoopNode) -> None:
        """Process a newly added node against historical finished targets."""
        self.nodes[node.node_id] = node

        processed_node_count = len(self.nodes)
        if (processed_node_count % self.config.check_every_n_nodes) != 0:
            return

        candidate_targets = self.provider.get_candidate_targets_for_node(
            node=node,
            all_nodes=self.nodes,
            config=self.config,
        )

        successful_matches = []
        for target in candidate_targets:
            pair_key = (node.node_id, target.target_id)
            self.stats.candidate_pairs += 1

            if pair_key in self._accepted_pairs:
                self.stats.duplicate_pairs += 1
                self._record_event(
                    node_id=node.node_id,
                    target_id=target.target_id,
                    score=float("nan"),
                    accepted=False,
                    status="duplicate",
                    source=target.search_source or "new_node_search",
                    used_full_submap=bool(target.match_full_submap),
                )
                continue

            match_result = self.verifier.verify(node=node, target=target)
            if (not match_result.success) or (match_result.matched_node_pose_global is None):
                self.stats.rejected_pairs += 1
                self._record_event(
                    node_id=node.node_id,
                    target_id=target.target_id,
                    score=float(match_result.score),
                    accepted=False,
                    status=str(match_result.status),
                    source=target.search_source or "new_node_search",
                    used_full_submap=bool(match_result.used_full_submap),
                )
                continue

            successful_matches.append((target, pair_key, match_result))

        successful_matches.sort(key=lambda x: float(x[2].score), reverse=True)
        max_keep = max(1, int(self.config.max_accepted_targets_per_new_node))

        for rank, (target, pair_key, match_result) in enumerate(successful_matches):
            if rank >= max_keep:
                self.stats.rejected_pairs += 1
                self._record_event(
                    node_id=node.node_id,
                    target_id=target.target_id,
                    score=float(match_result.score),
                    accepted=False,
                    status="accepted_limit",
                    source=target.search_source or "new_node_search",
                    used_full_submap=bool(match_result.used_full_submap),
                )
                continue

            constraint = self._build_constraint(
                node=node,
                target=target,
                matched_node_pose_global=match_result.matched_node_pose_global,
                match_score=float(match_result.score),
            )
            self.sink.add_loop_constraint(constraint)
            self._accepted_pairs.add(pair_key)
            self.stats.accepted_pairs += 1
            self._record_event(
                node_id=node.node_id,
                target_id=target.target_id,
                score=float(match_result.score),
                accepted=True,
                status="accepted",
                source=target.search_source or "new_node_search",
                used_full_submap=bool(match_result.used_full_submap),
            )

        self.sink.maybe_optimize(node_count=len(self.nodes), config=self.config)

    def enqueue_finished_target(self, target_id: str) -> None:
        """Queue a newly finished target for bounded retrospective verification."""
        target = self.provider.get_finished_target(target_id=target_id)
        if not target.is_finished:
            return

        candidate_nodes = self.provider.get_candidate_nodes_for_finished_target(
            target=target,
            all_nodes=self.nodes,
            config=self.config,
        )
        if not candidate_nodes:
            return

        self._pending_finished_targets.append(
            _PendingFinishedTargetWork(
                target_id=str(target.target_id),
                candidate_node_ids=[int(node.node_id) for node in candidate_nodes],
            )
        )

    def drain_pending_finished_targets(self, max_verifications: Optional[int] = None) -> int:
        """
        Drain bounded retrospective finished-target work.

        Returns the number of candidate verifications consumed in this call.
        """
        if max_verifications is None:
            remaining = None
        else:
            remaining = int(max_verifications)
            if remaining <= 0:
                return 0

        processed = 0
        threshold = int(self.config.finished_submap_full_search_failure_threshold)
        force_full = bool(self.config.force_full_submap_for_finished_submap_search)

        while self._pending_finished_targets and (remaining is None or remaining > 0):
            work = self._pending_finished_targets[0]

            if work.next_index >= len(work.candidate_node_ids):
                self._pending_finished_targets.pop(0)
                continue

            target = self.provider.get_finished_target(target_id=work.target_id)
            if not target.is_finished:
                self._pending_finished_targets.pop(0)
                continue

            node_id = int(work.candidate_node_ids[work.next_index])
            work.next_index += 1

            node = self.nodes.get(node_id)
            if node is None:
                continue

            use_full_submap = force_full or (
                threshold > 0 and work.consecutive_constrained_failures >= threshold
            )
            target_for_match = replace(
                target,
                match_full_submap=bool(use_full_submap),
                search_source="finished_submap_search",
            )

            pair_key = (node.node_id, target_for_match.target_id)
            self.stats.candidate_pairs += 1
            processed += 1
            if remaining is not None:
                remaining -= 1

            if pair_key in self._accepted_pairs:
                self.stats.duplicate_pairs += 1
                self._record_event(
                    node_id=node.node_id,
                    target_id=target_for_match.target_id,
                    score=float("nan"),
                    accepted=False,
                    status="duplicate",
                    source="finished_submap_search",
                    used_full_submap=bool(use_full_submap),
                )
                continue

            match_result = self.verifier.verify(node=node, target=target_for_match)
            if (not match_result.success) or (match_result.matched_node_pose_global is None):
                self.stats.rejected_pairs += 1
                self._record_event(
                    node_id=node.node_id,
                    target_id=target_for_match.target_id,
                    score=float(match_result.score),
                    accepted=False,
                    status=str(match_result.status),
                    source="finished_submap_search",
                    used_full_submap=bool(match_result.used_full_submap),
                )
                if not use_full_submap and match_result.status in {"matcher_failed", "score_failed"}:
                    work.consecutive_constrained_failures += 1
                continue

            constraint = self._build_constraint(
                node=node,
                target=target_for_match,
                matched_node_pose_global=match_result.matched_node_pose_global,
                match_score=float(match_result.score),
            )
            self.sink.add_loop_constraint(constraint)
            self._accepted_pairs.add(pair_key)
            self.stats.accepted_pairs += 1
            self._record_event(
                node_id=node.node_id,
                target_id=target_for_match.target_id,
                score=float(match_result.score),
                accepted=True,
                status="accepted",
                source="finished_submap_search",
                used_full_submap=bool(match_result.used_full_submap),
            )
            if not use_full_submap:
                work.consecutive_constrained_failures = 0

            if work.next_index >= len(work.candidate_node_ids):
                self._pending_finished_targets.pop(0)

        self.sink.maybe_optimize(node_count=len(self.nodes), config=self.config)
        return processed

    def finalize_pending_finished_targets(self) -> int:
        """Drain all queued retrospective verification work."""
        return self.drain_pending_finished_targets(max_verifications=None)

    def _build_constraint(
        self,
        node: LoopNode,
        target: ClosureTarget,
        matched_node_pose_global: Pose2,
        match_score: float = 1.0,
    ) -> LoopConstraint:
        """Construct a target-relative loop-closure constraint."""
        relative_pose = pose_relative(
            from_pose=target.pose_global,
            to_pose=matched_node_pose_global,
        )

        return LoopConstraint(
            node_id=node.node_id,
            target_id=target.target_id,
            target_type=target.target_type,
            relative_pose=relative_pose,
            translation_weight=self.config.translation_weight,
            rotation_weight=self.config.rotation_weight,
            match_score=float(match_score),
            constraint_kind="loop",
            is_fixed_target=target.is_fixed,
        )

    def _record_event(
        self,
        node_id: int,
        target_id: str,
        score: float,
        accepted: bool,
        status: str,
        source: str,
        used_full_submap: bool,
    ) -> None:
        self.events.append(
            LoopClosureEvent(
                node_id=int(node_id),
                target_id=str(target_id),
                score=float(score),
                accepted=bool(accepted),
                status=str(status),
                source=str(source),
                used_full_submap=bool(used_full_submap),
            )
        )
        if len(self.events) > 5000:
            self.events = self.events[-5000:]

    def get_stats(self) -> LoopClosureStats:
        return LoopClosureStats(
            candidate_pairs=int(self.stats.candidate_pairs),
            accepted_pairs=int(self.stats.accepted_pairs),
            rejected_pairs=int(self.stats.rejected_pairs),
            duplicate_pairs=int(self.stats.duplicate_pairs),
        )

    def get_recent_events(self, n: int = 20) -> List[LoopClosureEvent]:
        return list(self.events[-int(n):])

    def get_diagnostics_summary(self) -> dict:
        """Return compact diagnostics mirroring the legacy builder summary."""
        accepted_events = [ev for ev in self.events if ev.accepted]
        rejected_events = [ev for ev in self.events if not ev.accepted]

        accepted_scores = np.asarray(
            [ev.score for ev in accepted_events if np.isfinite(ev.score)],
            dtype=float,
        )
        score_failed_scores = np.asarray(
            [ev.score for ev in rejected_events if ev.status == "score_failed" and np.isfinite(ev.score)],
            dtype=float,
        )

        def _count(events: List[LoopClosureEvent], **conds) -> int:
            total = 0
            for event in events:
                if all(getattr(event, key) == value for key, value in conds.items()):
                    total += 1
            return total

        summary = {
            "accepted_from_new_node_search": _count(
                accepted_events, source="new_node_search", status="accepted"
            ),
            "accepted_from_finished_submap_search": _count(
                accepted_events, source="finished_submap_search", status="accepted"
            ),
            "rejected_sampled_out": _count(rejected_events, status="sampled_out"),
            "rejected_matcher_failed": _count(rejected_events, status="matcher_failed"),
            "rejected_score_failed": _count(rejected_events, status="score_failed"),
            "rejected_duplicate": _count(rejected_events, status="duplicate"),
            "retrospective_full_submap_attempts": _count(
                self.events, source="finished_submap_search", used_full_submap=True
            ),
            "retrospective_constrained_attempts": _count(
                self.events, source="finished_submap_search", used_full_submap=False
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
