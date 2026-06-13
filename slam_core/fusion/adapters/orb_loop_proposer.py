"""
OrbLoopProposer — propose-only wrapper over ORB-SLAM's loop-candidate detector.

RTAB_inspired_implementation_plan.md §6.2; CLAUDE.md §2.3. In fusion modes the
proposer ONLY retrieves loop candidates from ORB-SLAM's appearance detector
(``LoopDetector`` over ``KeyFrameDatabase``); it never invokes ORB-SLAM's
Sim(3) verification or loop-correction (those live in ``loop_closing.py`` and
are simply not called). This is the clean adapter-interception path, so no
``propose_only`` gate needs to be added inside ``loop_closing.py`` (CLAUDE.md §8
risk row).

The detector is injected: in Mode C the runner passes the real
``LoopDetector``; tests pass a lightweight stub. The detector must expose
``detect(keyframe) -> output`` where ``output`` has ``candidate_idxs`` and
``candidate_scores`` (the ``LoopDetectorOutput`` contract), and optionally
``add(keyframe)`` to register keyframes for future retrieval.
"""

from __future__ import annotations

from typing import List, Optional

from slam_core.fusion.adapters.types import LoopProposal


def _kid(keyframe) -> int:
    for attr in ("kid", "id"):
        v = getattr(keyframe, attr, None)
        if v is not None:
            return int(v)
    raise AttributeError("keyframe has neither .kid nor .id")


class OrbLoopProposer:
    def __init__(self, detector, propose_only: bool = True, min_index_separation: int = 20):
        if not propose_only:
            raise NotImplementedError(
                "OrbLoopProposer is propose-only in v1; full ORB-SLAM verification "
                "is intentionally bypassed in fusion modes."
            )
        self.detector = detector
        self.propose_only = True
        self.min_index_separation = int(min_index_separation)
        self._registered: List[int] = []

    def register(self, keyframe) -> None:
        """Add a keyframe to the underlying database for future retrieval.

        Per RTAB-Map, a node becomes loop-searchable only once it leaves STM and
        enters Working Memory — so the caller registers on STM->WM aging, not at
        creation, keeping the search index bounded to WM.
        """
        add = getattr(self.detector, "add", None)
        if add is None:
            db = getattr(self.detector, "keyframe_database", None)
            add = getattr(db, "add", None)
        if add is not None:
            add(keyframe)
        self._registered.append(_kid(keyframe))

    def erase(self, keyframe) -> None:
        """Remove a keyframe from the search index (on WM->LTM transfer)."""
        er = getattr(self.detector, "erase", None)
        if er is None:
            db = getattr(self.detector, "keyframe_database", None)
            er = getattr(db, "erase", None)
        if er is not None:
            er(keyframe)

    def poll_candidates(self, keyframe) -> List[LoopProposal]:
        """Retrieve loop candidates for ``keyframe`` without verifying them."""
        output = self.detector.detect(keyframe)
        ids = list(getattr(output, "candidate_idxs", []) or [])
        scores = list(getattr(output, "candidate_scores", []) or [])
        if len(scores) != len(ids):
            scores = [float("nan")] * len(ids)

        qid = _kid(keyframe)
        proposals: List[LoopProposal] = []
        for cid, score in zip(ids, scores):
            cid = int(cid)
            if abs(qid - cid) < self.min_index_separation:
                continue
            proposals.append(LoopProposal(candidate_id=cid, score=float(score), source="orb"))
        return proposals
