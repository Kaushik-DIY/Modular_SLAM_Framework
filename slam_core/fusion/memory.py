"""
MemoryManager — RTAB-Map-style STM / WM / LTM tiers (in-RAM for v1).

Faithful to RTAB_review.md §3 and RTAB_inspired_implementation_plan.md §8:

- **STM** (`_stm`): bounded queue of the most recent signatures; protected from
  loop-closure scoring; the site of the rehearsal merge. When full, the oldest
  STM signature ages out into WM.
- **WM** (`_wm`): id -> last-access "timestamp" (a monotonic clock here). Scored
  for loop closure and included in the optimized graph. When it exceeds
  ``wm_cap``, the oldest of the lowest-weighted nodes is transferred to LTM.
- **LTM** (`_ltm`): in-RAM in v1 (SQLite is v2). A safety cap drops the oldest
  entry with a warning once ``ltm_cap`` is exceeded.

Weight rule (RTAB_review.md §3.4): new signatures start at weight 0; a rehearsal
merge does ``w_t += w_c + 1`` and inherits the predecessor's neighbour links; a
confirmed loop closure does ``w_t += w_i + 1`` (see :meth:`confirm_loop`).

Rehearsal similarity is multi-modal (plan §8.2): if both the new signature and
its STM predecessor carry ORB descriptors, similarity is the matched-descriptor
ratio; otherwise, if both carry scans, similarity is a nearest-neighbour scan
overlap (a lightweight ICP-fit proxy — the real ``small_gicp`` fit can be
injected via ``scan_similarity_fn`` once Phase 4 lands).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional

import numpy as np

from slam_core.fusion.signature import Signature

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Default multi-modal rehearsal similarity functions
# --------------------------------------------------------------------------

def default_orb_similarity(desc_a: np.ndarray, desc_b: np.ndarray,
                           nndr: float = 0.7) -> float:
    """Matched-descriptor ratio in [0, 1] via Hamming + Lowe ratio test."""
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return 0.0
    import cv2

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(np.asarray(desc_a, dtype=np.uint8),
                      np.asarray(desc_b, dtype=np.uint8), k=2)
    good = 0
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < nndr * n.distance:
            good += 1
    return good / float(min(len(desc_a), len(desc_b)))


def default_scan_similarity(scan_a: np.ndarray, scan_b: np.ndarray,
                            inlier_dist: float = 0.10) -> float:
    """Nearest-neighbour overlap ratio in [0, 1] (cheap ICP-fit proxy)."""
    if scan_a is None or scan_b is None or len(scan_a) == 0 or len(scan_b) == 0:
        return 0.0
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(scan_b, dtype=float))
    dist, _ = tree.query(np.asarray(scan_a, dtype=float), k=1)
    return float(np.mean(dist <= inlier_dist))


# --------------------------------------------------------------------------
# Result records
# --------------------------------------------------------------------------

@dataclass
class InsertResult:
    sig_id: int
    merged_predecessor: Optional[int] = None  # id removed by rehearsal merge
    aged_out: Optional[int] = None            # id moved STM -> WM on overflow


@dataclass
class TickResult:
    transferred: List[int] = field(default_factory=list)  # WM -> LTM
    dropped: List[int] = field(default_factory=list)       # dropped from full LTM


# --------------------------------------------------------------------------
# MemoryManager
# --------------------------------------------------------------------------

class MemoryManager:
    def __init__(
        self,
        stm_size: int = 30,
        wm_cap: int = 200,
        ltm_cap: int = 1000,
        rehearsal_sim: float = 0.2,
        recent_wm_ratio: float = 0.2,
        max_retrieved_neighbors: int = 2,
        orb_similarity_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        scan_similarity_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ):
        if stm_size < 1:
            raise ValueError("stm_size must be >= 1")
        self.stm_size = stm_size
        self.wm_cap = wm_cap
        self.ltm_cap = ltm_cap
        self.rehearsal_sim = rehearsal_sim
        self.recent_wm_ratio = recent_wm_ratio
        self.max_retrieved_neighbors = max_retrieved_neighbors

        self._orb_sim = orb_similarity_fn or default_orb_similarity
        self._scan_sim = scan_similarity_fn or default_scan_similarity

        # All RAM-resident signatures (STM + WM). LTM is held separately.
        self._signatures: Dict[int, Signature] = {}
        self._stm: Deque[int] = deque()
        self._wm: Dict[int, float] = {}   # id -> last-access clock
        self._ltm: Dict[int, Signature] = {}

        self._clock: int = 0  # monotonic access counter (deterministic "time")

    # ---- tier sizes ----
    @property
    def stm_count(self) -> int:
        return len(self._stm)

    @property
    def wm_count(self) -> int:
        return len(self._wm)

    @property
    def ltm_count(self) -> int:
        return len(self._ltm)

    @property
    def live_count(self) -> int:
        """Signatures resident anywhere (STM + WM + LTM)."""
        return len(self._signatures) + len(self._ltm)

    # ---- tier membership ----
    def is_in_stm(self, sig_id: int) -> bool:
        return sig_id in self._stm

    def is_in_wm(self, sig_id: int) -> bool:
        return sig_id in self._wm

    def is_in_ltm(self, sig_id: int) -> bool:
        return sig_id in self._ltm

    def get(self, sig_id: int) -> Optional[Signature]:
        if sig_id in self._signatures:
            return self._signatures[sig_id]
        return self._ltm.get(sig_id)

    # ---- insertion + rehearsal ----
    def insert(self, sig: Signature) -> InsertResult:
        """Add a new signature: rehearsal-merge, then STM push with aging."""
        self._clock += 1
        merged = self._rehearse(sig)
        self._signatures[sig.id] = sig

        aged_out: Optional[int] = None
        # A merge keeps STM size flat; only age out when no merge filled the slot.
        if merged is None and len(self._stm) >= self.stm_size:
            aged_out = self._stm.popleft()
            self._move_stm_to_wm(aged_out)

        self._stm.append(sig.id)
        return InsertResult(sig.id, merged_predecessor=merged, aged_out=aged_out)

    def _rehearse(self, new_sig: Signature) -> Optional[int]:
        if not self._stm:
            return None
        prev_id = self._stm[-1]
        prev = self._signatures[prev_id]
        if self._similarity(new_sig, prev) < self.rehearsal_sim:
            return None

        # Weight rule: w_t += w_c + 1; inherit predecessor's neighbour links.
        new_sig.weight += prev.weight + 1
        for nb in prev.neighbors:
            if nb != new_sig.id and nb not in new_sig.neighbors:
                new_sig.neighbors.append(nb)

        self._stm.remove(prev_id)
        self._signatures.pop(prev_id, None)
        return prev_id

    def _similarity(self, a: Signature, b: Signature) -> float:
        if a.descriptors is not None and b.descriptors is not None:
            return self._orb_sim(a.descriptors, b.descriptors)
        if a.scan is not None and b.scan is not None:
            return self._scan_sim(a.scan, b.scan)
        return 0.0

    def _move_stm_to_wm(self, sig_id: int) -> None:
        self._wm[sig_id] = self._clock

    # ---- aging / transfer ----
    def tick(self) -> TickResult:
        """Run the WM->LTM transfer step and enforce the LTM safety cap."""
        result = TickResult()

        while len(self._wm) > self.wm_cap:
            victim = self._select_transfer_victim()
            if victim is None:
                break
            self._transfer_to_ltm(victim)
            result.transferred.append(victim)

        while len(self._ltm) > self.ltm_cap:
            oldest = min(self._ltm.keys())  # smallest id == oldest created
            self._ltm.pop(oldest)
            result.dropped.append(oldest)
            logger.warning("LTM cap %d exceeded; dropped oldest signature %d",
                           self.ltm_cap, oldest)

        return result

    def _select_transfer_victim(self) -> Optional[int]:
        if not self._wm:
            return None
        # Protect the most-recently-accessed fraction of WM (RTAB _recentWmRatio).
        by_access = sorted(self._wm.items(), key=lambda kv: kv[1])  # oldest first
        n_protect = int(self.recent_wm_ratio * len(by_access))
        eligible = by_access if n_protect == 0 else by_access[: len(by_access) - n_protect]
        if not eligible:
            return None
        # Lowest weight; tie-break oldest (smallest id).
        victim = min(eligible, key=lambda kv: (self._signatures[kv[0]].weight, kv[0]))
        return victim[0]

    def _transfer_to_ltm(self, sig_id: int) -> None:
        sig = self._signatures.pop(sig_id)
        self._wm.pop(sig_id, None)
        self._ltm[sig_id] = sig

    # ---- reactivation (LTM -> WM) ----
    def reactivate(self, ids: List[int]) -> List[Signature]:
        """Pull listed signatures (and up to N LTM neighbours each) into WM."""
        out: List[Signature] = []
        seen = set()
        for sig_id in ids:
            sig = self._reactivate_one(sig_id)
            if sig is None or sig.id in seen:
                continue
            seen.add(sig.id)
            out.append(sig)
            pulled = 0
            for nb in sig.neighbors:
                if pulled >= self.max_retrieved_neighbors:
                    break
                if nb in self._ltm:
                    nsig = self._reactivate_one(nb)
                    if nsig is not None and nsig.id not in seen:
                        seen.add(nsig.id)
                        out.append(nsig)
                        pulled += 1
        return out

    def _reactivate_one(self, sig_id: int) -> Optional[Signature]:
        if sig_id in self._ltm:
            sig = self._ltm.pop(sig_id)
            self._signatures[sig_id] = sig
            self._clock += 1
            self._wm[sig_id] = self._clock
            return sig
        if sig_id in self._signatures:
            self.touch(sig_id)
            return self._signatures[sig_id]
        return None

    def touch(self, sig_id: int) -> None:
        """Refresh a WM signature's last-access time."""
        if sig_id in self._wm:
            self._clock += 1
            self._wm[sig_id] = self._clock

    # ---- loop-closure weight bump ----
    def confirm_loop(self, target_id: int, matched_id: int) -> None:
        """Apply the loop-closure weight bump: w_target += w_matched + 1."""
        target = self.get(target_id)
        matched = self.get(matched_id)
        if target is None or matched is None:
            return
        target.weight += matched.weight + 1
        if target_id not in target.loops:
            target.loops.append(matched_id)
