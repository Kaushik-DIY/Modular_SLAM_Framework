"""C2 — MemoryManager RTAB invariants (RTAB_review.md §3): STM bound, WM cap,
rehearsal merge + weight inheritance, oldest-of-lowest-weight transfer, reactivation."""
import numpy as np
import pytest

fusion_core = pytest.importorskip("fusion_core")


def _sig(i, scan=None, n_scan=50, seed=None):
    rng = np.random.default_rng(i if seed is None else seed)
    if scan is None:
        scan = rng.uniform(-8, 8, (n_scan, 2)).astype(np.float32)
    return fusion_core.Signature(i, float(i), scan_xy=scan)


def _mgr(stm=5, wm=10, rehearsal=0.2, enabled=True, depth=1):
    cfg = fusion_core.MemoryConfig()
    cfg.stm_size = stm
    cfg.wm_cap = wm
    cfg.rehearsal_similarity = rehearsal
    cfg.rehearsal_enabled = enabled
    cfg.reactivation_neighbor_depth = depth
    store = fusion_core.InRamLtmStore()
    return fusion_core.MemoryManager(cfg, store), store


def test_stm_bound_and_aging_order():
    mgr, _ = _mgr(stm=5, wm=100)
    for i in range(20):
        res = mgr.insert(_sig(i), similarity=0.0)
        assert mgr.stm_count() <= 5
    # first aged-out ids are the oldest
    assert mgr.tier(0) == fusion_core.Tier.WM
    assert mgr.tier(19) == fusion_core.Tier.STM
    assert sorted(mgr.stm_ids()) == [15, 16, 17, 18, 19]


def test_wm_cap_and_transfer_to_ltm():
    mgr, store = _mgr(stm=2, wm=5)
    for i in range(30):
        mgr.insert(_sig(i), similarity=0.0)
    assert mgr.wm_count() <= 5
    assert mgr.stm_count() <= 2
    assert mgr.ltm_count() == 30 - mgr.wm_count() - mgr.stm_count()
    # transferred ids actually live in the store
    for tid in store.ids():
        assert mgr.tier(tid) == fusion_core.Tier.LTM


def test_transfer_picks_oldest_of_lowest_weight():
    mgr, store = _mgr(stm=1, wm=3)
    # Insert 4 sigs; once WM exceeds 3, the victim must be the lowest-weight,
    # oldest member. Give id=1 a high weight via a confirmed loop.
    for i in range(4):
        mgr.insert(_sig(i), similarity=0.0)
    # WM now holds 0,1,2 (id 3 in STM). Bump weight of id 1; this also TOUCHES
    # ids 1 and 0 (loop ends are recently-useful, per RTAB).
    mgr.on_loop_confirmed(1, 0)  # w1 += w0+1 = 1
    mgr.insert(_sig(4), similarity=0.0)  # pushes 3 into WM -> over cap -> transfer
    # Victim must be id 2: weight 0 and the oldest-accessed among weight-0
    # members (0 was touched by the loop; 1 has weight 1).
    assert mgr.tier(2) == fusion_core.Tier.LTM
    assert mgr.tier(1) == fusion_core.Tier.WM
    assert mgr.tier(0) == fusion_core.Tier.WM


def test_rehearsal_merge_weight_and_links():
    mgr, _ = _mgr(stm=10, wm=10, rehearsal=0.2)
    base = np.random.default_rng(0).uniform(-5, 5, (80, 2)).astype(np.float32)
    s0 = _sig(0, scan=base)
    s0.add_link(99, fusion_core.LinkType.NEIGHBOR, fusion_core.Pose2(0, 0, 0))
    mgr.insert(s0, similarity=0.0)
    # Nearly identical scan -> auto similarity ~1.0 -> merge
    s1 = _sig(1, scan=base + np.float32(0.001))
    res = mgr.insert(s1)  # similarity=auto
    assert res.similarity > 0.9
    assert res.rehearsal_merged and res.merged_predecessor_id == 0
    assert mgr.tier(0) == fusion_core.Tier.NONE  # predecessor deleted
    assert s1.weight == 1  # w_t += w_c + 1 with w_c = 0
    link_targets = [l.to_id for l in s1.links]
    assert 99 in link_targets  # predecessor's links inherited
    assert mgr.stm_count() == 1  # merge did not grow STM


def test_rehearsal_chain_accumulates_weight():
    mgr, _ = _mgr(stm=10, wm=10)
    base = np.random.default_rng(1).uniform(-5, 5, (80, 2)).astype(np.float32)
    mgr.insert(_sig(0, scan=base), similarity=0.0)
    last = None
    for i in range(1, 5):  # 4 consecutive merges (stationary robot)
        last = _sig(i, scan=base)
        mgr.insert(last)
    assert last.weight == 4  # 0->1->2->3->4
    assert mgr.stm_count() == 1


def test_loop_confirmed_weight_bump():
    mgr, _ = _mgr(stm=3, wm=10, enabled=False)
    for i in range(6):
        mgr.insert(_sig(i), similarity=0.0)
    q = mgr.get(5)
    t = mgr.get(0)
    t.weight = 7
    mgr.on_loop_confirmed(5, 0)
    assert q.weight == 8  # w_q += w_t + 1


def test_reactivation_with_neighbors():
    mgr, store = _mgr(stm=1, wm=2, depth=1)
    sigs = []
    for i in range(8):
        s = _sig(i)
        if i > 0:
            s.add_link(i - 1, fusion_core.LinkType.NEIGHBOR, fusion_core.Pose2(0.1, 0, 0))
        sigs.append(s)
        mgr.insert(s, similarity=0.0)
    assert mgr.ltm_count() >= 3
    assert mgr.tier(2) == fusion_core.Tier.LTM
    assert mgr.tier(1) == fusion_core.Tier.LTM
    moved = mgr.reactivate(2)
    assert 2 in moved
    assert mgr.tier(2) == fusion_core.Tier.WM
    assert mgr.tier(1) == fusion_core.Tier.WM  # depth-1 neighbor pulled too
    assert not store.contains(2)


def test_500_signature_stream_invariants():
    """The plan's C2 gate: 500-signature stream, all invariants hold throughout."""
    mgr, store = _mgr(stm=30, wm=200)
    rng = np.random.default_rng(42)
    inserted, merged_away = [], set()
    pos = np.zeros(2, np.float32)
    scan_prev = None
    for i in range(500):
        # simulate motion with occasional stationary stretches (-> rehearsal)
        stationary = (i % 50) > 44
        if not stationary or scan_prev is None:
            pos = pos + rng.uniform(-0.5, 0.5, 2).astype(np.float32)
            scan = (rng.uniform(-8, 8, (60, 2)) + pos).astype(np.float32)
        else:
            scan = scan_prev
        scan_prev = scan
        res = mgr.insert(_sig(i, scan=scan))
        inserted.append(i)
        if res.rehearsal_merged:
            merged_away.add(res.merged_predecessor_id)
        # invariants at every step
        assert mgr.stm_count() <= 30
        assert mgr.wm_count() <= 200
        # occasional loop confirmations + reactivations
        if i % 97 == 0 and i > 200:
            lids = store.ids()
            if lids:
                mgr.reactivate(lids[0])
                assert mgr.tier(lids[0]) == fusion_core.Tier.WM
    # accounting: every inserted id is in exactly one tier or was merged away
    for i in inserted:
        t = mgr.tier(i)
        if i in merged_away:
            assert t == fusion_core.Tier.NONE
        else:
            assert t in (fusion_core.Tier.STM, fusion_core.Tier.WM, fusion_core.Tier.LTM)
    total = mgr.stm_count() + mgr.wm_count() + mgr.ltm_count()
    assert total == 500 - len(merged_away)
    print(f"\n  stream: 500 inserted, {len(merged_away)} rehearsal-merged, "
          f"STM={mgr.stm_count()} WM={mgr.wm_count()} LTM={mgr.ltm_count()}, "
          f"payload={mgr.payload_bytes()/1e6:.1f} MB")
