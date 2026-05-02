from pathlib import Path

import numpy as np

from visual_slam.orbslam.slam.essential_graph import EssentialGraph, EssentialGraphResult
from visual_slam.orbslam.slam.global_ba import GlobalBAResult, GlobalBundleAdjuster
from visual_slam.orbslam.slam.optimizer_g2o import OptimizerResult, local_bundle_adjustment, pose_optimization

from tests.visual_slam.orbslam.test_checkpoint_2_8_optimizer_g2o import (
    make_Tcw,
    make_synthetic_frame,
    setup_tracker,
)
from tests.visual_slam.orbslam.test_checkpoint_2_23_essential_graph import _prepare_graph_scene
from tests.visual_slam.orbslam.test_checkpoint_2_24_global_ba import _build_gba_scene


class StopFlag:
    def __init__(self, value=False):
        self.value = bool(value)


def test_optimizer_result_objects_have_required_diagnostics():
    optimizer_result = OptimizerResult(1, 1, 0, 0.0, True)
    gba_result = GlobalBAResult(started=True)
    eg_result = EssentialGraphResult(True, 1.0, 0.5, 2)
    for obj in (optimizer_result, gba_result, eg_result):
        assert hasattr(obj, "success")
    assert hasattr(optimizer_result, "num_edges")
    assert hasattr(optimizer_result, "num_inliers")
    assert hasattr(gba_result, "elapsed_sec")
    assert hasattr(eg_result, "graph_edges")


def test_pose_optimization_rejects_nonfinite_observations():
    setup_tracker()
    frame = make_synthetic_frame(make_Tcw(), make_Tcw(), n=12)
    frame.kps[0].pt = (float("nan"), 10.0)
    num_inliers, mse = pose_optimization(frame)
    assert num_inliers >= 10
    assert np.isfinite(mse)


def test_local_ba_preserves_map_on_failure():
    slam_map, _, kf1, points, _ = _build_gba_scene(noisy=True)
    old_pose = kf1.Tcw().copy()
    old_point = points[0].get_position()
    flag = StopFlag(True)
    result = local_bundle_adjustment(kf1, abort_flag=flag, rounds=3)
    assert not result.success
    assert result.aborted
    np.testing.assert_allclose(kf1.Tcw(), old_pose)
    np.testing.assert_allclose(points[0].get_position(), old_point)


def test_global_ba_preserves_map_on_failure():
    slam_map, _, kf1, points, _ = _build_gba_scene(noisy=True)
    old_pose = kf1.Tcw().copy()
    old_point = points[0].get_position()
    result = GlobalBundleAdjuster(slam_map, min_inlier_edges=9999).run(loop_kf_id=1)
    assert not result.success
    np.testing.assert_allclose(kf1.Tcw(), old_pose)
    np.testing.assert_allclose(points[0].get_position(), old_point)


def test_global_ba_reduces_error_on_synthetic_problem():
    slam_map, _, kf1, _, Tcw1_true = _build_gba_scene(noisy=True)
    before = np.linalg.norm(kf1.Tcw()[:3, 3] - Tcw1_true[:3, 3])
    result = GlobalBundleAdjuster(slam_map, rounds=10).run(loop_kf_id=1)
    after = np.linalg.norm(kf1.Tcw()[:3, 3] - Tcw1_true[:3, 3])
    assert result.success
    assert after < before


def test_essential_graph_information_matrices_are_not_identity_for_all_edges():
    slam_map, _, loop_kf, current_kf, _, corrected, loops = _prepare_graph_scene()
    graph = EssentialGraph(
        map_object=slam_map,
        keyframes_to_correct=[current_kf],
        loop_keyframe=loop_kf,
        current_keyframe=current_kf,
        corrected_poses=corrected,
        loop_connections=loops,
        min_covisibility_weight=10,
    ).build_from_map()
    assert graph.edge_information
    assert any(not np.allclose(info, np.eye(6)) for info in graph.edge_information)


def test_covisibility_edge_weight_scales_with_connection_strength():
    slam_map, _, loop_kf, current_kf, _, corrected, loops = _prepare_graph_scene()
    current_kf.parent = None
    loop_kf.children.clear()
    loop_kf.add_connection(current_kf, 300)
    current_kf.add_connection(loop_kf, 300)
    graph = EssentialGraph(
        map_object=slam_map,
        keyframes_to_correct=[current_kf],
        loop_keyframe=loop_kf,
        current_keyframe=current_kf,
        corrected_poses=corrected,
        loop_connections={},
        min_covisibility_weight=10,
    ).build_from_map()
    assert graph.edge_weights["covisibility"]
    assert max(graph.edge_weights["covisibility"]) > 1.0


def test_loop_edge_weight_applied():
    slam_map, _, loop_kf, current_kf, _, corrected, loops = _prepare_graph_scene()
    graph = EssentialGraph(
        map_object=slam_map,
        keyframes_to_correct=[current_kf],
        loop_keyframe=loop_kf,
        current_keyframe=current_kf,
        corrected_poses=corrected,
        loop_connections=loops,
    ).build_from_map()
    assert graph.edge_weights["loop"]
    assert max(graph.edge_weights["loop"]) >= 10.0


def test_positive_depth_filtering_in_ba():
    slam_map, _, _, points, _ = _build_gba_scene(noisy=False)
    points[0].update_position(np.array([0.0, 0.0, -1.0]))
    result = GlobalBundleAdjuster(slam_map, rounds=3).run(loop_kf_id=2)
    assert result.success
    assert result.num_edges < len(points) * 2


def test_optimizer_writeback_is_atomic(monkeypatch):
    slam_map, _, kf1, points, _ = _build_gba_scene(noisy=True)
    old_pose = kf1.Tcw().copy()
    old_point = points[0].get_position()
    adjuster = GlobalBundleAdjuster(slam_map)
    monkeypatch.setattr(adjuster, "validate_updates", lambda *args, **kwargs: (False, "forced invalid"))
    result = adjuster.run(loop_kf_id=3)
    assert not result.success
    np.testing.assert_allclose(kf1.Tcw(), old_pose)
    np.testing.assert_allclose(points[0].get_position(), old_point)


def test_optimizer_parity_audit_file_exists():
    path = Path("visual_slam/reference_audit/checkpoint_2_25/OPTIMIZER_PARITY_AUDIT.md")
    assert path.exists()
    text = path.read_text()
    assert "optimizer_g2o.py" in text
    assert "pySLAM" in text
