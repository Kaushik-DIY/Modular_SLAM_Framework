"""
Phase 3 checkpoint — Fusion graph.

Verifies FusionGraph (wrapping G2oBackend2D) per CLAUDE.md §4 Phase 3:
optimizer converges, the first node stays fixed, node poses update under a
synthetic loop, and the optimizable subgraph is recoverable. Also checks the
ConstraintSink Protocol path and memory-gated write-back.
"""

from __future__ import annotations

import numpy as np
import pytest

from slam_core.common.types import Pose2
from slam_core.fusion.signature import Signature
from slam_core.fusion.graph import FusionGraph, keyframe_target_id
from slam_core.fusion.memory import MemoryManager
from slam_core.loop_closure import LoopConstraint, LoopClosureConfig


def _sig(i, x, y=0.0, theta=0.0):
    return Signature(id=i, timestamp=float(i), pose=Pose2(x, y, theta))


def _line_graph(xs):
    """Build a spine over keyframes initialized at the given x positions."""
    g = FusionGraph()
    sigs = [_sig(i, x) for i, x in enumerate(xs)]
    for s in sigs:
        g.add_node(s)
    for a, b in zip(sigs[:-1], sigs[1:]):
        g.add_neighbor_link(a, b, rel_pose=Pose2(1.0, 0.0, 0.0))
    return g, sigs


# --------------------------------------------------------------------------
# anchor + subgraph
# --------------------------------------------------------------------------

def test_anchor_fixed_and_subgraph_recoverable():
    g, sigs = _line_graph([0, 1, 2, 3, 4])
    g.solve()
    p0 = g.get_pose(0)
    assert p0.x == pytest.approx(0.0, abs=1e-6)
    assert p0.y == pytest.approx(0.0, abs=1e-6)
    assert p0.theta == pytest.approx(0.0, abs=1e-6)
    assert g.get_subgraph_for_optimization() == {0, 1, 2, 3, 4}
    assert g.node_count == 5


# --------------------------------------------------------------------------
# deformation from spine consistency (a kink gets straightened)
# --------------------------------------------------------------------------

def test_spine_straightens_kink():
    g = FusionGraph()
    # node 2 starts off the line at y=0.5; spine says the path is straight.
    sigs = [_sig(0, 0, 0), _sig(1, 1, 0), _sig(2, 2, 0.5), _sig(3, 3, 0), _sig(4, 4, 0)]
    for s in sigs:
        g.add_node(s)
    for a, b in zip(sigs[:-1], sigs[1:]):
        g.add_neighbor_link(a, b, rel_pose=Pose2(1.0, 0.0, 0.0))

    g.solve(max_iters=50)
    assert abs(g.get_pose(2).y) < 0.1          # kink pulled back toward the line
    assert g.get_pose(0).y == pytest.approx(0.0, abs=1e-6)   # anchor unmoved


# --------------------------------------------------------------------------
# loop closure deforms the graph
# --------------------------------------------------------------------------

def test_loop_constraint_deforms_graph():
    g, sigs = _line_graph([0, 1, 2, 3, 4])
    before = g.solve()
    assert before[4].x == pytest.approx(4.0, abs=1e-3)  # spine-only: stays at 4

    # Loop: node 4 should be only 2 units from node 0 (conflicts with the spine).
    g.add_loop_edge(target_id=0, source_id=4, rel_pose=Pose2(2.0, 0.0, 0.0))
    after = g.solve(max_iters=80)

    assert g.loop_count == 1
    assert after[0].x == pytest.approx(0.0, abs=1e-6)   # anchor still fixed
    assert after[4].x < before[4].x - 1e-3              # pulled toward the loop
    assert after[2].x != pytest.approx(before[2].x, abs=1e-4)  # interior moved
    for p in after.values():
        assert np.isfinite([p.x, p.y, p.theta]).all()


# --------------------------------------------------------------------------
# ConstraintSink Protocol path
# --------------------------------------------------------------------------

def test_constraint_sink_add_loop_constraint():
    g, sigs = _line_graph([0, 1, 2, 3, 4])
    g.solve()
    # target-relative form z = T_target^{-1} T_source
    c = LoopConstraint(
        node_id=4,
        target_id=keyframe_target_id(0),
        target_type="keyframe",
        relative_pose=Pose2(2.0, 0.0, 0.0),
        translation_weight=1.1e4,
        rotation_weight=1e5,
    )
    g.add_loop_constraint(c)
    assert g.loop_count == 1
    after = g.solve(max_iters=80)
    assert after[4].x < 4.0 - 1e-3


def test_maybe_optimize_respects_cadence():
    g, sigs = _line_graph([0, 1, 2, 3, 4])
    g.add_loop_edge(0, 4, Pose2(2.0, 0.0, 0.0))
    cfg = LoopClosureConfig(optimize_every_n_nodes=5)
    # node_count not a multiple of 5 -> no solve (poses unchanged from init)
    g.maybe_optimize(node_count=3, config=cfg)
    assert g.get_pose(4).x == pytest.approx(4.0, abs=1e-9)
    # multiple of 5 -> solve fires, deforms
    g.maybe_optimize(node_count=5, config=cfg)
    assert g.get_pose(4).x < 4.0 - 1e-3


# --------------------------------------------------------------------------
# memory-gated subgraph + write-back
# --------------------------------------------------------------------------

def test_memory_gated_subgraph():
    mem = MemoryManager(stm_size=2, wm_cap=100, recent_wm_ratio=0.0)
    g = FusionGraph(memory=mem)
    sigs = [_sig(i, i) for i in range(5)]
    for s in sigs:
        mem.insert(s)
        g.add_node(s)
    # stm_size=2 -> ids 0..2 aged into WM, 3..4 in STM; all are optimizable.
    sub = g.get_subgraph_for_optimization()
    assert sub == {0, 1, 2, 3, 4}
    # move some to LTM and confirm they drop out of the subgraph.
    mem._transfer_to_ltm(0)
    assert 0 not in g.get_subgraph_for_optimization()
