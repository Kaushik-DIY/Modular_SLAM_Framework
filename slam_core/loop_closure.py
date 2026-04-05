from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol
import math
import numpy as np
from slam_core.common.se2 import pose_compose, pose_inverse
from slam_core.common.types import Pose2
# =============================================================================
# Basic SE(2) pose utilities
# =============================================================================
# The loop-closure layer must remain self-contained and independent of any
# specific SLAM backend implementation. For this reason, a lightweight SE(2)
# pose container and a small set of helper functions are defined here.
# If your project already exposes an equivalent SE(2) type in
# slam_core.common.se2, this local dataclass can later be replaced cleanly.


def pose_relative(from_pose: Pose2, to_pose: Pose2) -> Pose2:
    """
    Compute the relative planar transform between two poses.

    The returned pose corresponds to:
        T_rel = T_from^{-1} * T_to

    This is the standard Cartographer-style relation used for node-to-submap
    loop-closure constraint construction.
    """
    return pose_compose(pose_inverse(from_pose), to_pose)


def pose_translation_distance(a: Pose2, b: Pose2) -> float:
    """
    Euclidean distance between the translation parts of two planar poses.
    """
    return math.hypot(a.x - b.x, a.y - b.y)

# =============================================================================
# Loop-closure data structures
# =============================================================================


@dataclass
class LoopNode:
    """
    Persistent node record used by the loop-closure layer.

    A node corresponds to one accepted scan insertion event. The loop-closure
    module stores the minimal information required for later geometric
    verification against historical targets.

    Attributes
    ----------
    node_id:
        Unique node identifier in the current SLAM run.
    scan_points:
        2D scan points expressed in the node's local sensor / tracking frame.
        Shape is expected to be (N, 2).
    pose_guess_global:
        Current estimated global pose of the node before loop-closure
        verification.
    timestamp:
        Acquisition time of the node. This is useful for temporal gating.
    local_target_ids:
        IDs of targets into which the node was already inserted locally.
        For Cartographer-style submap SLAM, this is typically the pair of
        active submaps. These targets are excluded from loop-closure search.
    """
    node_id: int
    scan_points: np.ndarray
    pose_guess_global: Pose2
    timestamp: float
    local_target_ids: List[str] = field(default_factory=list)


@dataclass
class ClosureTarget:
    """
    Generic loop-closure target.

    This abstraction is intentionally neutral. In Cartographer-style SLAM,
    the target is a finished submap. In scan-to-map style SLAM, the target can
    later be defined as a global map anchor or frozen map snapshot.

    Attributes
    ----------
    target_id:
        Unique target identifier.
    target_type:
        Semantic type of the target, e.g. "submap" or "global_map".
    pose_global:
        Global pose of the target frame.
    is_finished:
        Whether the target is eligible for loop closure.
    is_fixed:
        Whether the target pose is fixed in optimization.
    map_view:
        Opaque handle to the data structure required by the matcher.
        This may be a submap grid, map query object, or other backend-specific
        representation.
    """
    target_id: str
    target_type: str
    pose_global: Pose2
    is_finished: bool
    is_fixed: bool
    map_view: Any


@dataclass
class LoopMatchResult:
    """
    Result of geometric verification for one node-target pair.

    Attributes
    ----------
    success:
        Whether the candidate passed the geometric verification stage.
    score:
        Matching quality score returned by the verifier.
    matched_node_pose_global:
        Refined node pose in the global frame after successful verification.
    """
    success: bool
    score: float
    matched_node_pose_global: Optional[Pose2]


@dataclass
class LoopConstraint:
    """
    Generic loop-closure constraint.

    The constraint is stored in target-relative form:
        z_tj = T_t^{-1} * T_j

    This exactly matches the Cartographer-style node-to-submap formulation.

    Attributes
    ----------
    node_id:
        Identifier of the constrained node.
    target_id:
        Identifier of the matched target.
    target_type:
        Semantic type of the matched target.
    relative_pose:
        Measured target-to-node relative pose.
    translation_weight:
        Weight assigned to translational residual components.
    rotation_weight:
        Weight assigned to rotational residual components.
    constraint_kind:
        Constraint label. "loop" is used here to distinguish these edges from
        local insertion constraints.
    is_fixed_target:
        Indicates whether the target pose is fixed in the backend graph.
    """
    node_id: int
    target_id: str
    target_type: str
    relative_pose: Pose2
    translation_weight: float
    rotation_weight: float
    constraint_kind: str = "loop"
    is_fixed_target: bool = False

@dataclass
class LoopClosureEvent:
    node_id: int
    target_id: str
    score: float
    accepted: bool


@dataclass
class LoopClosureStats:
    candidate_pairs: int = 0
    accepted_pairs: int = 0
    rejected_pairs: int = 0
    duplicate_pairs: int = 0

@dataclass
class LoopClosureConfig:
    """
    Configuration parameters governing loop-closure scheduling and gating.
    """
    min_score: float = 0.62
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


# =============================================================================
# Abstract provider / verifier / sink protocols
# =============================================================================
# These protocol classes define the minimal backend-specific hooks required by
# the generic loop-closure manager. The goal is to preserve a single abstract
# loop-closure implementation while allowing the target type to vary.


class TargetProvider(Protocol):
    """
    Backend-specific interface for exposing loop-closure targets and node
    candidates to the abstract loop-closure manager.
    """

    def get_candidate_targets_for_node(
        self,
        node: LoopNode,
        all_nodes: Dict[int, LoopNode],
        config: LoopClosureConfig,
    ) -> List[ClosureTarget]:
        """
        Return eligible finished targets to be tested against a newly added node.
        """
        ...

    def get_finished_target(self, target_id: str) -> ClosureTarget:
        """
        Return one finished target by identifier.
        """
        ...

    def get_candidate_nodes_for_finished_target(
        self,
        target: ClosureTarget,
        all_nodes: Dict[int, LoopNode],
        config: LoopClosureConfig,
    ) -> List[LoopNode]:
        """
        Return historical nodes to be tested against a newly finished target.
        """
        ...


class LoopVerifier(Protocol):
    """
    Backend-specific geometric verification interface.

    A verifier is responsible for running the actual node-to-target matching
    procedure, which may include:
        1. coarse correlative scan matching,
        2. score-based rejection,
        3. nonlinear refinement.
    """

    def verify(self, node: LoopNode, target: ClosureTarget) -> LoopMatchResult:
        """
        Geometrically verify a node-target pair.
        """
        ...


class ConstraintSink(Protocol):
    """
    Backend-specific constraint insertion and optimization scheduling interface.
    """

    def add_loop_constraint(self, constraint: LoopConstraint) -> None:
        """
        Insert one accepted loop-closure constraint into the backend graph.
        """
        ...

    def maybe_optimize(self, node_count: int, config: LoopClosureConfig) -> None:
        """
        Trigger batched optimization if the scheduling condition is satisfied.
        """
        ...


# =============================================================================
# Generic loop-closure manager
# =============================================================================


class LoopClosureManager:
    """
    Generic loop-closure manager.

    This class implements the abstract loop-closure policy and preserves the
    Cartographer-style two-way search semantics:

        1. new node  -> old finished targets
        2. new target -> old nodes

    Importantly, this manager does not know whether the target is a finished
    submap or another map representation. That backend-specific knowledge is
    delegated to the target provider and verifier.
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

        # A lightweight archive of all nodes generated during the current run.
        # This archive is essential for Cartographer-style two-way loop-closure
        # search, where newly finished targets must be matched against older
        # historical nodes.
        self.nodes: Dict[int, LoopNode] = {}
        self._accepted_pairs: set[tuple[int, str]] = set()
        self.stats = LoopClosureStats()
        self.events: List[LoopClosureEvent] = []

    def on_new_node(self, node: LoopNode) -> None:
        """
        Process a newly added node.

        The function performs the first half of the Cartographer-style loop-
        closure strategy:
            new node -> old finished targets
        """
        self.nodes[node.node_id] = node

        # Loop-closure checks may be decimated if required. The default value
        # of 1 means that every node is considered.
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
            self.stats.candidate_pairs += 1

            pair_key = (node.node_id, target.target_id)
            if pair_key in self._accepted_pairs:
                self.stats.duplicate_pairs += 1
                continue

            match_result = self.verifier.verify(node=node, target=target)
            if (not match_result.success) or (match_result.matched_node_pose_global is None):
                self.stats.rejected_pairs += 1
                self._record_event(
                    node_id=node.node_id,
                    target_id=target.target_id,
                    score=float(match_result.score),
                    accepted=False,
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
                )
                continue

            constraint = self._build_constraint(
                node=node,
                target=target,
                matched_node_pose_global=match_result.matched_node_pose_global,
            )
            self.sink.add_loop_constraint(constraint)
            self._accepted_pairs.add(pair_key)
            self.stats.accepted_pairs += 1
            self._record_event(
                node_id=node.node_id,
                target_id=target.target_id,
                score=float(match_result.score),
                accepted=True,
            )

        self.sink.maybe_optimize(node_count=len(self.nodes), config=self.config)

    def on_target_finished(self, target_id: str) -> None:
        """
        Process a newly finished target.

        This function implements the second half of the Cartographer-style
        two-way search:
            newly finished target -> old nodes
        """
        target = self.provider.get_finished_target(target_id=target_id)

        if not target.is_finished:
            return

        candidate_nodes = self.provider.get_candidate_nodes_for_finished_target(
            target=target,
            all_nodes=self.nodes,
            config=self.config,
        )

        for node in candidate_nodes:
            self.stats.candidate_pairs += 1

            pair_key = (node.node_id, target.target_id)
            if pair_key in self._accepted_pairs:
                self.stats.duplicate_pairs += 1
                continue

            match_result = self.verifier.verify(node=node, target=target)
            if (not match_result.success) or (match_result.matched_node_pose_global is None):
                self.stats.rejected_pairs += 1
                self._record_event(
                    node_id=node.node_id,
                    target_id=target.target_id,
                    score=float(match_result.score),
                    accepted=False,
                )
                continue

            constraint = self._build_constraint(
                node=node,
                target=target,
                matched_node_pose_global=match_result.matched_node_pose_global,
            )
            self.sink.add_loop_constraint(constraint)
            self._accepted_pairs.add(pair_key)
            self.stats.accepted_pairs += 1
            self._record_event(
                node_id=node.node_id,
                target_id=target.target_id,
                score=float(match_result.score),
                accepted=True,
            )

        self.sink.maybe_optimize(node_count=len(self.nodes), config=self.config)

    def _build_constraint(
        self,
        node: LoopNode,
        target: ClosureTarget,
        matched_node_pose_global: Pose2,
    ) -> LoopConstraint:
        """
        Construct a target-relative loop-closure constraint.

        This follows the same geometric form as Cartographer's 2D
        node-to-submap constraint construction:
            z_tj = T_t^{-1} * T_j

        where T_t is the target pose and T_j is the refined node pose.
        """
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
            constraint_kind="loop",
            is_fixed_target=target.is_fixed,
        )
    def _record_event(self, node_id: int, target_id: str, score: float, accepted: bool) -> None:
        self.events.append(
            LoopClosureEvent(
                node_id=int(node_id),
                target_id=str(target_id),
                score=float(score),
                accepted=bool(accepted),
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