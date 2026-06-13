"""C4 — neighborhood retrieval (graph + metric, LTM reactivation) and
local log-odds grid assembly (pixel-match vs numpy reference)."""
import math

import numpy as np
import pytest

fusion_core = pytest.importorskip("fusion_core")


def _sig(i, scan):
    return fusion_core.Signature(i, float(i), scan_xy=scan.astype(np.float32))


def _chain_world(n=12, step=1.0, scan_pts=40, seed=0):
    """Linear chain of signatures with small local scans; consecutive links."""
    rng = np.random.default_rng(seed)
    sigs, poses = [], []
    for i in range(n):
        scan = rng.uniform(-1.5, 1.5, (scan_pts, 2))
        s = _sig(i, scan)
        if i > 0:
            s.add_link(i - 1, fusion_core.LinkType.NEIGHBOR,
                       fusion_core.Pose2(-step, 0, 0))
        sigs.append(s)
        poses.append(fusion_core.Pose2(i * step, 0.0, 0.0))
    return sigs, poses


def _build_mem_graph(sigs, poses, stm=2, wm=4):
    cfg = fusion_core.MemoryConfig()
    cfg.stm_size = stm
    cfg.wm_cap = wm
    cfg.rehearsal_enabled = False
    store = fusion_core.InRamLtmStore()
    mem = fusion_core.MemoryManager(cfg, store)
    graph = fusion_core.FusionGraph2D(fusion_core.GraphConfig())
    for s, p in zip(sigs, poses):
        mem.insert(s, similarity=0.0)
        graph.add_node(s.id, p)
    return mem, graph, store


def test_graph_bfs_membership_and_reactivation():
    sigs, poses = _chain_world(n=12)
    mem, graph, store = _build_mem_graph(sigs, poses, stm=2, wm=4)
    # early chain members are in LTM now
    assert mem.tier(2) == fusion_core.Tier.LTM
    nb = fusion_core.retrieve_neighborhood(mem, graph, 2, graph_depth=2,
                                           metric_radius=0.0)
    ids = sorted(s.id for s in nb.signatures)
    # BFS depth 2 over the chain links: {2, 1, 0} (links point backwards only)
    assert ids == [0, 1, 2]
    assert 2 in nb.reactivated  # was LTM, pulled back
    assert mem.tier(2) == fusion_core.Tier.WM


def test_metric_radius_membership():
    sigs, poses = _chain_world(n=12)
    mem, graph, _ = _build_mem_graph(sigs, poses, stm=12, wm=12)  # all resident
    nb = fusion_core.retrieve_neighborhood(mem, graph, 5, graph_depth=0,
                                           metric_radius=2.1)
    ids = sorted(s.id for s in nb.signatures)
    # nodes at x within [5-2.1, 5+2.1]: 3,4,5,6,7
    assert ids == [3, 4, 5, 6, 7]


def _numpy_reference_grid(scans_world, sensors_world, origin, w, h, res,
                          l_occ=0.85, l_free=-0.4, l_min=-5.0, l_max=5.0):
    L = np.zeros((h, w), np.float64)

    def upd(gx, gy, dl):
        if 0 <= gx < w and 0 <= gy < h:
            L[gy, gx] = np.clip(L[gy, gx] + dl, l_min, l_max)

    for pts, (sx, sy) in zip(scans_world, sensors_world):
        sgx = int(math.floor((sx - origin[0]) / res))
        sgy = int(math.floor((sy - origin[1]) / res))
        for px, py in pts:
            gx = int(math.floor((px - origin[0]) / res))
            gy = int(math.floor((py - origin[1]) / res))
            # bresenham (same integer variant as C++)
            x, y = sgx, sgy
            dx, dy = abs(gx - x), -abs(gy - y)
            stx = 1 if x < gx else -1
            sty = 1 if y < gy else -1
            err = dx + dy
            while x != gx or y != gy:
                upd(x, y, l_free)
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x += stx
                if e2 <= dx:
                    err += dx
                    y += sty
            upd(gx, gy, l_occ)
    return L


def test_grid_pixel_match_vs_numpy_reference():
    rng = np.random.default_rng(7)
    # two signatures at different poses with rotation
    scans = [rng.uniform(0.3, 4.0, (60, 2)).astype(np.float32) for _ in range(2)]
    poses = [fusion_core.Pose2(0.0, 0.0, 0.0), fusion_core.Pose2(1.0, 0.5, math.pi / 6)]
    sigs = [_sig(i, s) for i, s in enumerate(scans)]

    cfg = fusion_core.GridConfig()
    cfg.resolution = 0.05
    grid = fusion_core.assemble_local_grid(sigs, poses, cfg)
    L_cpp = np.asarray(grid.log_odds, dtype=np.float64)

    # transform scans to world with the same float32 math order as C++
    def tf(scan, p):
        c, s = math.cos(p.theta), math.sin(p.theta)
        out = []
        for x, y in scan:
            out.append((p.x + c * float(x) - s * float(y),
                        p.y + s * float(x) + c * float(y)))
        return out

    scans_world = [tf(s, p) for s, p in zip(scans, poses)]
    sensors = [(p.x, p.y) for p in poses]
    L_ref = _numpy_reference_grid(scans_world, sensors,
                                  (grid.origin_x, grid.origin_y),
                                  grid.width, grid.height, grid.resolution)
    np.testing.assert_allclose(L_cpp, L_ref, atol=1e-5)
    # sanity: grid has occupied and free content
    assert (L_cpp > 0.5).sum() > 50
    assert (L_cpp < 0).sum() > 500


def test_candidate_local_grid_end_to_end():
    sigs, poses = _chain_world(n=12)
    mem, graph, _ = _build_mem_graph(sigs, poses, stm=12, wm=12)
    grid, nb = fusion_core.candidate_local_grid(mem, graph, 6, graph_depth=1,
                                                metric_radius=1.5)
    assert len(nb) >= 3
    L = np.asarray(grid.log_odds)
    assert L.size > 100 and (L != 0).any()
    p = np.asarray(grid.probability())
    assert p.min() >= 0.0 and p.max() <= 1.0
