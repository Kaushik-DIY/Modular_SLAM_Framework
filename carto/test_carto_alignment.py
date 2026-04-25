from __future__ import annotations

import math

import numpy as np
import pytest

from carto.adapter import CartoLocalSlamAdapter
from carto.common.se2 import pose_compose
from carto.common.types import Pose2
from carto.local_slam.pose_extrapolator import PoseExtrapolatorCV
from carto.pose_graph.global_slam_2d import CartoGlobalSlam2D
from carto.pose_graph.pose_graph_2d import PoseGraph2D
from slam_core.loop_closure import (
    ClosureTarget,
    ConstraintSink,
    LoopClosureConfig,
    LoopClosureManager,
    LoopMatchResult,
    LoopNode,
    LoopVerifier,
    TargetProvider,
)
from slam_core.matching.scan_to_submap.branch_and_bound_backend import BranchAndBoundSubmapBackend
from slam_core.matching.scan_to_submap.submaps import ProbabilityGrid, Submap2D, SubmapBuilder2D
from slam_core.matching.scan_to_submap.two_stage_backend import TwoStageBruteForceSubmapBackend
from slam_core.matching.scan_to_submap.types import (
    ScanToSubmapBackendConfig,
    SubmapMatchRequest,
    SubmapSearchWindow,
)
from slam_core.optimisers.gn_lm import GNLMConfig, GaussNewtonLM


def _make_grid(size_m: float = 8.0, res: float = 0.1) -> ProbabilityGrid:
    return ProbabilityGrid(
        size_m=size_m,
        res=res,
        l0=0.0,
        l_occ=0.85,
        l_free=-0.1,
        l_min=-5.0,
        l_max=5.0,
    )


def _paint_scan(grid: ProbabilityGrid, pose_sub: Pose2, scan_points_local: np.ndarray) -> None:
    ca = math.cos(pose_sub.theta)
    sa = math.sin(pose_sub.theta)
    for px, py in np.asarray(scan_points_local, dtype=float):
        x = ca * float(px) - sa * float(py) + float(pose_sub.x)
        y = sa * float(px) + ca * float(py) + float(pose_sub.y)
        gx, gy = grid.world_to_grid(x, y)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if grid.in_bounds(gx + dx, gy + dy):
                    grid.L[gy + dy, gx + dx] = grid.l_max


def _pose_error(a: Pose2, b: Pose2) -> float:
    return (
        math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))
        + abs(float(a.theta) - float(b.theta))
    )


class _NoopMatcherManager:
    pass


def _make_refine_solver() -> GaussNewtonLM:
    return GaussNewtonLM(
        GNLMConfig(
            iters=12,
            damping=1e-3,
            eps_stop=1e-6,
            step_clip=np.array([0.1, 0.1, math.radians(5.0)], dtype=float),
            verbose=False,
        )
    )


def test_apply_optimization_correction_uses_live_adapter_state() -> None:
    extrap = PoseExtrapolatorCV(max_dt=0.5)
    adapter = CartoLocalSlamAdapter(
        matcher_manager=_NoopMatcherManager(),
        extrapolator=extrap,
    )

    t = 1.0
    initial_pose = Pose2(1.0, 2.0, 0.1)
    adapter.initialize_extrapolator(t, initial_pose)
    adapter.last_insert_time = t
    adapter.last_insert_pose = initial_pose

    optimized = {("node", 7): Pose2(2.0, 4.0, 0.3)}
    adapter.apply_optimization_correction(optimized=optimized, last_node_id=7, correction_alpha=0.5)

    predicted = extrap.predict(t)
    assert adapter.last_insert_pose is not None
    assert adapter.last_insert_pose.x == pytest.approx(1.5)
    assert adapter.last_insert_pose.y == pytest.approx(3.0)
    assert adapter.last_insert_pose.theta == pytest.approx(0.2)
    assert predicted.x == pytest.approx(1.5)
    assert predicted.y == pytest.approx(3.0)
    assert predicted.theta == pytest.approx(0.2)


class _StubPoseGraphBackend:
    def __init__(self, optimized):
        self.optimized = optimized
        self.nodes = {}
        self.submaps = {}
        self.constraints = []

    def add_node(self, node):
        self.nodes[int(node.id)] = node.pose

    def add_submap(self, submap):
        self.submaps[int(submap.id)] = submap.pose

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def solve(self, max_iters=50):
        _ = max_iters
        return dict(self.optimized)

    def get_optimized_poses(self):
        return dict(self.optimized)


class _LoopAdapterStub:
    def on_new_node(self, **kwargs):
        _ = kwargs

    def process_finished_submaps(self):
        return 0

    def finalize(self):
        return 0

    def get_stats(self):
        class _Stats:
            candidate_pairs = 0
            accepted_pairs = 0
            rejected_pairs = 0
            duplicate_pairs = 0

        return _Stats()

    def get_recent_events(self, n=20):
        _ = n
        return []

    def get_diagnostics_summary(self):
        return {}


class _CorrectionSpy:
    def __init__(self, builder: SubmapBuilder2D, expected):
        self.builder = builder
        self.expected = expected
        self.called = False

    def apply_optimization_correction(self, optimized, last_node_id, correction_alpha):
        _ = optimized, last_node_id, correction_alpha
        self.called = True
        assert self.builder.active[0].pose_world == self.expected[("submap", 0)]
        assert self.builder.finished_submaps[0].pose_world == self.expected[("submap", 1)]


def test_global_slam_applies_live_correction_after_pose_sync() -> None:
    builder = SubmapBuilder2D(
        submap_size_m=8.0,
        resolution=0.1,
        scans_per_submap=10,
        ray_steps=10,
        l0=0.0,
        l_occ=0.85,
        l_free=-0.1,
        l_min=-5.0,
        l_max=5.0,
    )
    builder.active = [Submap2D(id=0, grid=_make_grid(), pose_world=Pose2(0.0, 0.0, 0.0))]
    builder.finished_submaps = [Submap2D(id=1, grid=_make_grid(), pose_world=Pose2(1.0, 1.0, 0.0), finished=True)]

    optimized = {
        ("submap", 0): Pose2(0.5, 0.0, 0.0),
        ("submap", 1): Pose2(2.0, 2.0, 0.1),
        ("node", 0): Pose2(3.0, 3.0, 0.2),
    }
    backend = _StubPoseGraphBackend(optimized=optimized)
    pose_graph = PoseGraph2D(backend=backend, submap_builder=builder)
    pose_graph.add_submap_if_needed(0, Pose2(0.0, 0.0, 0.0))
    pose_graph.add_submap_if_needed(1, Pose2(1.0, 1.0, 0.0))
    pose_graph.num_intra_constraints = 1

    spy = _CorrectionSpy(builder=builder, expected=optimized)
    global_slam = CartoGlobalSlam2D(
        loop_closure_adapter=_LoopAdapterStub(),
        pose_graph=pose_graph,
        optimize_every_n_nodes=1,
        adapter=spy,
    )

    global_slam.on_node_inserted(
        node_id=0,
        timestamp=0.0,
        scan_points=np.zeros((4, 2), dtype=float),
        pose_global=Pose2(0.0, 0.0, 0.0),
        insertion_submaps=builder.active,
    )

    assert spy.called is True
    assert pose_graph.get_submap_pose(0) == optimized[("submap", 0)]
    assert pose_graph.get_submap_pose(1) == optimized[("submap", 1)]


def test_two_stage_backend_refines_toward_occupied_space_with_prior() -> None:
    scan_points = np.array(
        [[1.0, 0.0], [1.1, 0.2], [0.9, -0.2], [1.2, 0.05], [0.8, 0.25]],
        dtype=float,
    )
    truth_sub = Pose2(1.4, 0.9, 0.15)
    predicted_world = Pose2(1.0, 0.6, 0.0)

    submap = Submap2D(id=0, grid=_make_grid(), pose_world=Pose2(0.0, 0.0, 0.0))
    _paint_scan(submap.grid, truth_sub, scan_points)

    backend = TwoStageBruteForceSubmapBackend(
        submap_builder=None,
        config=ScanToSubmapBackendConfig(
            min_score=0.1,
            min_valid=3,
            max_match_points=80,
            max_refine_points=80,
            refine_min_points=3,
            refine_w_trans=10.0,
            refine_w_rot=40.0,
            coarse=SubmapSearchWindow(
                xy_window=0.8,
                theta_window=math.radians(15.0),
                xy_step=0.1,
                theta_step=math.radians(4.0),
                level=2,
            ),
            fine=SubmapSearchWindow(
                xy_window=0.2,
                theta_window=math.radians(5.0),
                xy_step=0.05,
                theta_step=math.radians(1.0),
                level=0,
            ),
        ),
        refine_solver=_make_refine_solver(),
    )

    response = backend.match(
        SubmapMatchRequest(
            scan_points_local=scan_points,
            predicted_pose_world=predicted_world,
            submap_pose_world=submap.pose_world,
            submap=submap,
        )
    )

    assert response.success is True
    assert response.debug.refined is True
    assert _pose_error(response.pose_world, truth_sub) < _pose_error(predicted_world, truth_sub)


def test_branch_and_bound_response_score_uses_coarse_score() -> None:
    scan_points = np.array(
        [[1.0, 0.0], [1.1, 0.15], [0.9, -0.15], [1.2, 0.1], [0.8, 0.2]],
        dtype=float,
    )
    truth_sub = Pose2(1.3, 1.0, 0.1)
    predicted_world = Pose2(1.2, 0.9, 0.0)

    submap = Submap2D(id=5, grid=_make_grid(), pose_world=Pose2(0.0, 0.0, 0.0), finished=True)
    _paint_scan(submap.grid, truth_sub, scan_points)

    backend = BranchAndBoundSubmapBackend(
        config=ScanToSubmapBackendConfig(
            backend_type="branch_and_bound",
            min_score=0.1,
            global_localization_min_score=0.1,
            max_match_points=80,
            max_refine_points=80,
            refine_min_points=3,
            refine_w_trans=10.0,
            refine_w_rot=40.0,
            coarse=SubmapSearchWindow(
                xy_window=0.5,
                theta_window=math.radians(10.0),
                xy_step=0.05,
                theta_step=math.radians(2.0),
                level=0,
            ),
        ),
        refine_solver=_make_refine_solver(),
    )

    response = backend.match(
        SubmapMatchRequest(
            scan_points_local=scan_points,
            predicted_pose_world=predicted_world,
            submap_pose_world=submap.pose_world,
            submap=submap,
        )
    )

    assert response.success is True
    assert response.score == pytest.approx(response.debug.coarse_score)
    assert "refined_score" in response.debug.extra


class _LoopProvider(TargetProvider):
    def get_candidate_targets_for_node(self, node, all_nodes, config):
        _ = all_nodes, config
        if int(node.node_id) != 100:
            return []
        return [
            ClosureTarget(
                target_id="new_target",
                target_type="submap",
                pose_global=Pose2(0.0, 0.0, 0.0),
                is_finished=True,
                is_fixed=False,
                map_view="new_target",
                match_full_submap=False,
                search_source="new_node_search",
            )
        ]

    def get_finished_target(self, target_id: str) -> ClosureTarget:
        return ClosureTarget(
            target_id=str(target_id),
            target_type="submap",
            pose_global=Pose2(0.0, 0.0, 0.0),
            is_finished=True,
            is_fixed=False,
            map_view=str(target_id),
            match_full_submap=False,
            search_source="finished_submap_search",
        )

    def get_candidate_nodes_for_finished_target(self, target, all_nodes, config):
        _ = target, config
        return [all_nodes[node_id] for node_id in (10, 11, 12, 13)]


class _LoopVerifier(LoopVerifier):
    def verify(self, node: LoopNode, target: ClosureTarget) -> LoopMatchResult:
        if target.search_source == "new_node_search":
            return LoopMatchResult(
                success=True,
                score=0.7,
                matched_node_pose_global=node.pose_guess_global,
                status="accepted",
                used_full_submap=bool(target.match_full_submap),
            )

        if int(node.node_id) in (10, 11, 12):
            assert target.match_full_submap is False
            return LoopMatchResult(
                success=False,
                score=0.2,
                matched_node_pose_global=None,
                status="score_failed",
                used_full_submap=False,
            )

        assert int(node.node_id) == 13
        return LoopMatchResult(
            success=bool(target.match_full_submap),
            score=0.65 if target.match_full_submap else 0.2,
            matched_node_pose_global=node.pose_guess_global if target.match_full_submap else None,
            status="accepted" if target.match_full_submap else "score_failed",
            used_full_submap=bool(target.match_full_submap),
        )


class _LoopSink(ConstraintSink):
    def __init__(self):
        self.constraints = []

    def add_loop_constraint(self, constraint) -> None:
        self.constraints.append(constraint)

    def maybe_optimize(self, node_count: int, config: LoopClosureConfig) -> None:
        _ = node_count, config


def _make_loop_manager(threshold: int) -> LoopClosureManager:
    config = LoopClosureConfig(
        min_score=0.55,
        global_localization_min_score=0.60,
        finished_submap_verification_budget_per_tick=10,
        finished_submap_full_search_failure_threshold=threshold,
    )
    manager = LoopClosureManager(
        config=config,
        provider=_LoopProvider(),
        verifier=_LoopVerifier(),
        sink=_LoopSink(),
    )
    for node_id in (10, 11, 12, 13):
        manager.nodes[node_id] = LoopNode(
            node_id=node_id,
            scan_points=np.zeros((4, 2), dtype=float),
            pose_guess_global=Pose2(0.0, 0.0, 0.0),
            timestamp=float(node_id),
        )
    return manager


def test_loop_manager_reports_diagnostics_and_escalates_finished_submap_search() -> None:
    manager = _make_loop_manager(threshold=3)

    manager.on_new_node(
        LoopNode(
            node_id=100,
            scan_points=np.zeros((4, 2), dtype=float),
            pose_guess_global=Pose2(0.0, 0.0, 0.0),
            timestamp=100.0,
        )
    )
    manager.enqueue_finished_target("finished_target")
    processed = manager.drain_pending_finished_targets(max_verifications=10)

    summary = manager.get_diagnostics_summary()

    assert processed == 4
    assert summary["accepted_from_new_node_search"] == 1
    assert summary["accepted_from_finished_submap_search"] == 1
    assert summary["rejected_score_failed"] == 3
    assert summary["retrospective_full_submap_attempts"] == 1


def test_loop_manager_disables_full_submap_escalation_for_non_positive_threshold() -> None:
    manager = _make_loop_manager(threshold=0)
    manager.enqueue_finished_target("finished_target")
    manager.drain_pending_finished_targets(max_verifications=10)

    recent_events = manager.get_recent_events(10)
    assert any(event.status == "score_failed" for event in recent_events)
    assert all(event.used_full_submap is False for event in recent_events)
