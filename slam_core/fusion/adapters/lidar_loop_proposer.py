"""
LidarLoopProposer — propose-only wrapper over the LiDAR candidate provider.

RTAB_inspired_implementation_plan.md §6.2. Wraps a ``TargetProvider`` (in
fusion: ``carto.loop_closure_adapter.CartoTargetProvider``) and exposes only
its B&B / proximity candidate proposal. It never calls the LiDAR
``LoopVerifier`` and never touches a ``ConstraintSink`` (so the LiDAR g2o PGO
does not solve in fusion modes — CLAUDE.md §2.7). The fusion graph owns
optimization.

The verifier/sink are intentionally not referenced here; propose-only is
therefore structural, not a runtime flag.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from slam_core.fusion.adapters.types import LoopProposal
from slam_core.loop_closure import ClosureTarget, LoopClosureConfig, LoopNode


def _target_candidate_id(target: ClosureTarget):
    tid = target.target_id
    try:
        return int(tid)
    except (TypeError, ValueError):
        return str(tid)


class LidarLoopProposer:
    def __init__(self, provider, config: Optional[LoopClosureConfig] = None,
                 propose_only: bool = True):
        if not propose_only:
            raise NotImplementedError(
                "LidarLoopProposer is propose-only in v1; the LiDAR pipeline's "
                "own B&B verification and g2o backend are bypassed in fusion modes."
            )
        self.provider = provider
        self.config = config or LoopClosureConfig()
        self.propose_only = True

    def poll_candidates(self, node: LoopNode,
                        all_nodes: Dict[int, LoopNode]) -> List[LoopProposal]:
        """Return B&B / proximity candidate targets without verifying them."""
        targets = self.provider.get_candidate_targets_for_node(
            node=node, all_nodes=all_nodes, config=self.config
        )
        proposals: List[LoopProposal] = []
        for t in targets:
            proposals.append(
                LoopProposal(
                    candidate_id=_target_candidate_id(t),
                    score=1.0,  # unscored pre-verification; proximity-ranked by provider
                    source="lidar",
                    target=t,
                )
            )
        return proposals
