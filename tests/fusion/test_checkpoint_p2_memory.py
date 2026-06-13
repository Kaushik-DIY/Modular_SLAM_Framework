"""
Phase 2 checkpoint — Memory tier.

Exercises the RTAB-Map STM/WM/LTM semantics (RTAB_review.md §3): STM aging,
rehearsal merge on both the ORB-descriptor and scan (ICP-fit) paths, weight
inheritance, oldest-of-lowest-weight transfer, LTM reactivation, and the
500-keyframe synthetic stream with tier-bound invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

from slam_core.common.types import Pose2
from slam_core.fusion.memory import (
    MemoryManager,
    default_orb_similarity,
    default_scan_similarity,
)
from slam_core.fusion.signature import Signature


def _orb_sig(sig_id, rng, n=120, like=None):
    """ORB-only signature; if `like` given, copies its descriptors (a duplicate)."""
    desc = like.copy() if like is not None else rng.integers(0, 256, (n, 32), np.uint8)
    return Signature(id=sig_id, timestamp=float(sig_id), pose=Pose2(0, 0, 0),
                     descriptors=desc)


def _scan_sig(sig_id, rng, n=150, like=None):
    """Scan-only signature; if `like` given, copies its scan (a duplicate)."""
    scan = like.copy() if like is not None else rng.normal(0, 1, (n, 2))
    return Signature(id=sig_id, timestamp=float(sig_id), pose=Pose2(0, 0, 0),
                     scan=scan)


# --------------------------------------------------------------------------
# similarity helpers
# --------------------------------------------------------------------------

def test_orb_similarity_identical_vs_random():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (120, 32), np.uint8)
    b = rng.integers(0, 256, (120, 32), np.uint8)
    assert default_orb_similarity(a, a) > 0.9      # identical -> high
    assert default_orb_similarity(a, b) < 0.2      # unrelated -> low


def test_scan_similarity_identical_vs_disjoint():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, (150, 2))
    b = a + 100.0  # far away -> no overlap
    assert default_scan_similarity(a, a) > 0.9
    assert default_scan_similarity(a, b) < 0.1


# --------------------------------------------------------------------------
# STM aging
# --------------------------------------------------------------------------

def test_stm_aging_moves_oldest_to_wm():
    rng = np.random.default_rng(2)
    mem = MemoryManager(stm_size=3, wm_cap=100, ltm_cap=100)
    results = [mem.insert(_orb_sig(i, rng)) for i in range(5)]

    assert mem.stm_count == 3
    assert mem.wm_count == 2
    # first two inserts aged out nothing; later inserts aged out 0 then 1.
    aged = [r.aged_out for r in results]
    assert aged[:3] == [None, None, None]
    assert aged[3] == 0 and aged[4] == 1
    assert mem.is_in_wm(0) and mem.is_in_wm(1)
    assert mem.is_in_stm(2) and mem.is_in_stm(3) and mem.is_in_stm(4)
    # tier invariant: every RAM signature is in exactly STM or WM
    assert len(mem._signatures) == mem.stm_count + mem.wm_count


# --------------------------------------------------------------------------
# rehearsal merge — ORB path + weight inheritance
# --------------------------------------------------------------------------

def test_rehearsal_orb_merge_and_weight_inheritance():
    rng = np.random.default_rng(3)
    mem = MemoryManager(stm_size=30, wm_cap=100, rehearsal_sim=0.2)

    base = _orb_sig(0, rng)
    mem.insert(base)
    # three consecutive duplicates of the running STM tail -> chained merges.
    r1 = mem.insert(_orb_sig(1, rng, like=base.descriptors))
    assert r1.merged_predecessor == 0
    r2 = mem.insert(_orb_sig(2, rng, like=base.descriptors))
    assert r2.merged_predecessor == 1
    r3 = mem.insert(_orb_sig(3, rng, like=base.descriptors))
    assert r3.merged_predecessor == 2

    # weights: id1 = 0+0+1 = 1; id2 = 0+1+1 = 2; id3 = 0+2+1 = 3
    assert mem.get(3).weight == 3
    # merged predecessors are gone from RAM; only the survivor remains in STM.
    assert mem.get(0) is None and mem.get(1) is None and mem.get(2) is None
    assert mem.is_in_stm(3)
    assert mem.stm_count == 1


def test_rehearsal_does_not_merge_unrelated():
    rng = np.random.default_rng(4)
    mem = MemoryManager(stm_size=30, rehearsal_sim=0.2)
    mem.insert(_orb_sig(0, rng))
    r = mem.insert(_orb_sig(1, rng))  # independent random descriptors
    assert r.merged_predecessor is None
    assert mem.stm_count == 2


# --------------------------------------------------------------------------
# rehearsal merge — scan (ICP-fit) path
# --------------------------------------------------------------------------

def test_rehearsal_scan_merge_path():
    rng = np.random.default_rng(5)
    mem = MemoryManager(stm_size=30, rehearsal_sim=0.2)
    base = _scan_sig(0, rng)
    mem.insert(base)
    r = mem.insert(_scan_sig(1, rng, like=base.scan))
    assert r.merged_predecessor == 0
    assert mem.get(1).weight == 1


def test_mixed_modality_boundary_no_merge():
    rng = np.random.default_rng(6)
    mem = MemoryManager(stm_size=30, rehearsal_sim=0.2)
    mem.insert(_orb_sig(0, rng))          # ORB-only
    r = mem.insert(_scan_sig(1, rng))     # scan-only -> no common modality
    assert r.merged_predecessor is None


# --------------------------------------------------------------------------
# transfer — oldest of lowest weight
# --------------------------------------------------------------------------

def test_transfer_picks_lowest_weight_oldest():
    rng = np.random.default_rng(7)
    # stm_size=1 forces everything into WM quickly; recent_wm_ratio=0 = no protection.
    mem = MemoryManager(stm_size=1, wm_cap=3, ltm_cap=100, recent_wm_ratio=0.0)

    for i in range(5):
        s = _orb_sig(i, rng)
        mem.insert(s)
    # Give id 2 a high weight so it is NOT chosen for transfer.
    if mem.get(2) is not None:
        mem.get(2).weight = 10

    res = mem.tick()
    assert mem.wm_count <= 3
    assert len(res.transferred) >= 1
    # the high-weight node must survive in RAM, not be transferred.
    assert not mem.is_in_ltm(2)
    # transferred nodes are the low-weight oldest ones (ids 0/1...).
    assert min(res.transferred) == 0


# --------------------------------------------------------------------------
# LTM safety cap
# --------------------------------------------------------------------------

def test_ltm_cap_drops_oldest():
    rng = np.random.default_rng(8)
    mem = MemoryManager(stm_size=1, wm_cap=1, ltm_cap=3, recent_wm_ratio=0.0)
    for i in range(10):
        mem.insert(_orb_sig(i, rng))
        mem.tick()
    assert mem.ltm_count <= 3
    # the very first signatures must have been dropped from the bounded LTM.
    assert mem.get(0) is None


# --------------------------------------------------------------------------
# reactivation
# --------------------------------------------------------------------------

def test_reactivation_moves_ltm_to_wm_with_neighbors():
    rng = np.random.default_rng(9)
    mem = MemoryManager(stm_size=1, wm_cap=2, ltm_cap=100, recent_wm_ratio=0.0,
                        max_retrieved_neighbors=2)
    for i in range(8):
        mem.insert(_orb_sig(i, rng))
        mem.tick()

    # find a node currently in LTM and wire a neighbour that is also in LTM
    ltm_ids = list(mem._ltm.keys())
    assert len(ltm_ids) >= 2
    target, neigh = ltm_ids[0], ltm_ids[1]
    mem.get(target).neighbors.append(neigh)

    out = mem.reactivate([target])
    assert mem.is_in_wm(target) and not mem.is_in_ltm(target)
    assert mem.is_in_wm(neigh) and not mem.is_in_ltm(neigh)
    assert {s.id for s in out} >= {target, neigh}


# --------------------------------------------------------------------------
# loop-closure weight bump
# --------------------------------------------------------------------------

def test_confirm_loop_weight_bump():
    rng = np.random.default_rng(10)
    mem = MemoryManager(stm_size=10)
    mem.insert(_orb_sig(0, rng))
    mem.insert(_orb_sig(1, rng))
    mem.get(0).weight = 4
    mem.confirm_loop(target_id=1, matched_id=0)
    assert mem.get(1).weight == 0 + 4 + 1  # w_t += w_i + 1
    assert 0 in mem.get(1).loops


# --------------------------------------------------------------------------
# 500-keyframe synthetic stream with tier-bound invariants
# --------------------------------------------------------------------------

def test_500_keyframe_stream_invariants():
    rng = np.random.default_rng(123)
    stm_size, wm_cap, ltm_cap = 30, 200, 1000
    mem = MemoryManager(stm_size=stm_size, wm_cap=wm_cap, ltm_cap=ltm_cap,
                        rehearsal_sim=0.2, recent_wm_ratio=0.2)

    n = 500
    merges = 0
    max_weight_seen = 0
    prev_desc = None
    prev_scan = None

    for i in range(n):
        # Interleave modalities in blocks; sprinkle duplicates to force merges.
        scan_block = (i // 50) % 2 == 1  # every other block of 50 is scan-only
        duplicate = (i % 25 == 0) and i > 0  # periodic near-stationary repeat

        if scan_block:
            sig = _scan_sig(i, rng, like=prev_scan if (duplicate and prev_scan is not None) else None)
            prev_scan, prev_desc = sig.scan, None
        else:
            sig = _orb_sig(i, rng, like=prev_desc if (duplicate and prev_desc is not None) else None)
            prev_desc, prev_scan = sig.descriptors, None

        res = mem.insert(sig)
        if res.merged_predecessor is not None:
            merges += 1
        mem.tick()

        # Invariants that must hold after every step.
        assert mem.stm_count <= stm_size
        assert mem.wm_count <= wm_cap
        assert mem.ltm_count <= ltm_cap
        assert len(mem._signatures) == mem.stm_count + mem.wm_count
        survivor = mem.get(sig.id)
        if survivor is not None:
            max_weight_seen = max(max_weight_seen, survivor.weight)

    # The stream must have exercised every transition.
    assert merges > 0, "rehearsal merges never fired"
    assert max_weight_seen >= 1, "weight inheritance never observed"
    assert mem.ltm_count > 0, "no WM->LTM transfers occurred"

    # Reactivate a batch from LTM and confirm the tier move.
    ltm_sample = list(mem._ltm.keys())[:5]
    mem.reactivate(ltm_sample)
    for sid in ltm_sample:
        assert mem.is_in_wm(sid)
        assert not mem.is_in_ltm(sid)
