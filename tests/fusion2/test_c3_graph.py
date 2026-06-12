"""C3 — FusionGraph2D: parity vs Python g2o on identical problems + 10k-node timing."""
import math
import time

import numpy as np
import pytest

fusion_core = pytest.importorskip("fusion_core")
g2o = pytest.importorskip("g2o")

HUBER = 10.0
SPINE_W = (1e5, 1e5)
LOOP_W = (1.1e4, 1e5)


def _toy_problem():
    """5-node chain with accumulated drift + 1 loop edge back to node 0.
    Returns (initial_poses, spine_edges, loop_edges)."""
    # ground truth: square-ish path returning near start
    init = [
        (0, (0.0, 0.0, 0.0)),
        (1, (1.0, 0.05, 0.1)),
        (2, (2.1, 0.2, math.pi / 2 + 0.1)),
        (3, (2.2, 1.3, math.pi + 0.15)),
        (4, (1.0, 1.5, -math.pi / 2 + 0.2)),  # drifted; loop says it's near (1,1)
    ]
    spine = [
        (0, 1, (1.0, 0.0, 0.0)),
        (1, 2, (1.0, 0.0, math.pi / 2)),
        (2, 3, (1.0, 0.0, math.pi / 2)),
        (3, 4, (1.0, 0.0, math.pi / 2)),
    ]
    loops = [
        (4, 0, (1.0, 0.0, math.pi / 2)),  # closing the square
    ]
    return init, spine, loops


def _solve_cpp(init, spine, loops, iters=50):
    cfg = fusion_core.GraphConfig()
    cfg.huber_scale = HUBER
    g = fusion_core.FusionGraph2D(cfg)
    for nid, p in init:
        g.add_node(nid, fusion_core.Pose2(*p))
    g.set_fixed(0)
    for a, b, z in spine:
        g.add_spine_edge(a, b, fusion_core.Pose2(*z), *SPINE_W)
    for a, b, z in loops:
        g.add_loop_edge(a, b, fusion_core.Pose2(*z), *LOOP_W)
    g.optimize(iters)
    return {int(r[0]): (r[1], r[2], r[3]) for r in np.asarray(g.poses())}


def _solve_pyg2o(init, spine, loops, iters=50):
    opt = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE2(g2o.LinearSolverEigenSE2())
    opt.set_algorithm(g2o.OptimizationAlgorithmLevenberg(solver))
    for nid, p in init:
        v = g2o.VertexSE2()
        v.set_id(nid)
        v.set_estimate(g2o.SE2(*p))
        if nid == 0:
            v.set_fixed(True)
        opt.add_vertex(v)

    def add_edge(a, b, z, w, huber):
        e = g2o.EdgeSE2()
        e.set_vertex(0, opt.vertex(a))
        e.set_vertex(1, opt.vertex(b))
        e.set_measurement(g2o.SE2(*z))
        e.set_information(np.diag([w[0], w[0], w[1]]).astype(np.float64))
        if huber:
            rk = g2o.RobustKernelHuber()
            rk.set_delta(HUBER)
            e.set_robust_kernel(rk)
        opt.add_edge(e)

    for a, b, z in spine:
        add_edge(a, b, z, SPINE_W, False)
    for a, b, z in loops:
        add_edge(a, b, z, LOOP_W, True)
    opt.initialize_optimization()
    opt.optimize(iters)
    out = {}
    for nid, _ in init:
        v = opt.vertex(nid).estimate().to_vector()
        out[nid] = (float(v[0]), float(v[1]), float(v[2]))
    return out


def test_parity_with_python_g2o():
    init, spine, loops = _toy_problem()
    cpp = _solve_cpp(init, spine, loops)
    ref = _solve_pyg2o(init, spine, loops)
    for nid in ref:
        for k in range(3):
            d = abs(cpp[nid][k] - ref[nid][k])
            if k == 2:
                d = abs(math.atan2(math.sin(d), math.cos(d)))
            assert d < 1e-6, f"node {nid} comp {k}: cpp={cpp[nid][k]} ref={ref[nid][k]}"


def test_anchor_stays_fixed_and_graph_deforms():
    init, spine, loops = _toy_problem()
    before = {nid: p for nid, p in init}
    cpp = _solve_cpp(init, spine, loops)
    assert cpp[0] == pytest.approx(before[0], abs=1e-12)  # anchor untouched
    moved = sum(np.hypot(cpp[n][0] - before[n][0], cpp[n][1] - before[n][1]) > 1e-3
                for n in before if n != 0)
    assert moved >= 2  # loop actually deformed the chain


def test_no_edges_is_noop():
    g = fusion_core.FusionGraph2D(fusion_core.GraphConfig())
    g.add_node(0, fusion_core.Pose2(1, 2, 0.5))
    assert g.optimize() == 0.0
    p = g.get_pose(0)
    assert (p.x, p.y, p.theta) == (1, 2, 0.5)


def test_10k_node_timing():
    rng = np.random.default_rng(0)
    cfg = fusion_core.GraphConfig()
    g = fusion_core.FusionGraph2D(cfg)
    n = 10_000
    x = y = th = 0.0
    g.add_node(0, fusion_core.Pose2(0, 0, 0))
    for i in range(1, n):
        dx, dth = 0.1, float(rng.normal(0, 0.02))
        x += dx * math.cos(th)
        y += dx * math.sin(th)
        th += dth
        g.add_node(i, fusion_core.Pose2(x + rng.normal(0, 0.01), y + rng.normal(0, 0.01), th))
        g.add_spine_edge(i - 1, i, fusion_core.Pose2(0.1, 0.0, dth))
        if i % 100 == 0 and i > 200:
            g.add_loop_edge(i, i - 200, fusion_core.Pose2(*(-rng.uniform(0, 0.05, 3))),
                            1.1e4, 1e5)
    t0 = time.perf_counter()
    g.optimize(20)
    dt = time.perf_counter() - t0
    print(f"\n  10k nodes / {g.num_edges} edges ({g.num_loop_edges} loops): "
          f"optimize(20) = {dt*1000:.0f} ms")
    assert dt < 5.0
