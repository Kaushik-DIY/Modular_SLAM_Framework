import numpy as np

from visual_slam.orbslam.slam.global_ba import GlobalBAResult, GlobalBundleAdjuster
from visual_slam.orbslam.slam.loop_closing import LoopClosing

from tests.visual_slam.orbslam.test_checkpoint_2_8_optimizer_g2o import (
    attach_observations,
    make_Tcw,
    make_synthetic_keyframe,
    setup_tracker,
)
from tests.visual_slam.orbslam.test_checkpoint_2_21_loop_closing import (
    build_loop_scene,
    make_slam_namespace,
)


class StopFlag:
    def __init__(self, value=False):
        self.value = bool(value)


def _build_gba_scene(noisy=True, n=25):
    from visual_slam.orbslam.slam.map import Map
    from visual_slam.orbslam.slam.map_point import MapPoint

    setup_tracker()
    slam_map = Map()
    Tcw0_true = make_Tcw(0.0, 0.0, 0.0)
    Tcw1_true = make_Tcw(0.12, 0.0, 0.0)
    Tcw1_init = make_Tcw(0.20, -0.04, 0.08) if noisy else Tcw1_true

    kf0 = make_synthetic_keyframe(Tcw0_true, Tcw0_true, kid=0, timestamp=1.0)
    kf1 = make_synthetic_keyframe(Tcw1_true, Tcw1_init, kid=1, timestamp=2.0)
    kf0.points = []
    kf1.points = []

    points = []
    for i in range(n):
        true_position = np.array([-0.35 + 0.035 * i, 0.03 * (i % 5), 2.0 + 0.01 * (i % 3)])
        position = true_position + (np.array([0.015, -0.01, 0.02]) if noisy else 0.0)
        points.append(MapPoint(position))

    attach_observations(kf0, points, Tcw0_true)
    attach_observations(kf1, points, Tcw1_true)
    slam_map.add_keyframe(kf0)
    slam_map.add_keyframe(kf1)
    kf1.set_parent(kf0)
    for point in points:
        slam_map.add_point(point)
        point.update_info()
    kf0.update_connections()
    kf1.update_connections()
    return slam_map, kf0, kf1, points, Tcw1_true


def test_global_ba_imports():
    assert GlobalBAResult is not None
    assert GlobalBundleAdjuster is not None


def test_global_ba_collects_valid_graph():
    slam_map, _, _, points, _ = _build_gba_scene()
    adjuster = GlobalBundleAdjuster(slam_map)
    keyframes, graph_points = adjuster.collect_graph()
    assert len(keyframes) == 2
    assert len(graph_points) == len(points)


def test_global_ba_fixes_origin_gauge():
    slam_map, _, _, _, _ = _build_gba_scene()
    result = GlobalBundleAdjuster(slam_map, rounds=3).run(loop_kf_id=1)
    assert result.success
    assert 0 in result.fixed_keyframe_ids


def test_global_ba_reduces_synthetic_reprojection_error():
    slam_map, _, kf1, _, Tcw1_true = _build_gba_scene(noisy=True)
    before = np.linalg.norm(kf1.Tcw()[:3, 3] - Tcw1_true[:3, 3])
    result = GlobalBundleAdjuster(slam_map, rounds=10).run(loop_kf_id=1)
    after = np.linalg.norm(kf1.Tcw()[:3, 3] - Tcw1_true[:3, 3])
    assert result.success
    assert after < before
    assert result.mean_error_after <= result.mean_error_before


def test_global_ba_updates_keyframes_and_map_points_on_success():
    slam_map, _, kf1, points, _ = _build_gba_scene(noisy=True)
    old_pose = kf1.Tcw().copy()
    result = GlobalBundleAdjuster(slam_map, rounds=8).run(loop_kf_id=2)
    assert result.success
    assert not np.allclose(kf1.Tcw(), old_pose)
    assert points[0].is_pt_GBA_valid
    assert np.all(np.isfinite(points[0].get_position()))


def test_global_ba_recomputes_map_point_info(monkeypatch):
    slam_map, _, _, points, _ = _build_gba_scene(noisy=True)
    called = {"value": False}
    original = points[0].update_info

    def mark():
        called["value"] = True
        original()

    monkeypatch.setattr(points[0], "update_info", mark)
    result = GlobalBundleAdjuster(slam_map, rounds=5).run(loop_kf_id=3)
    assert result.success
    assert called["value"]


def test_global_ba_failure_does_not_corrupt_map():
    slam_map, _, kf1, points, _ = _build_gba_scene(noisy=True)
    old_pose = kf1.Tcw().copy()
    old_point = points[0].get_position()
    result = GlobalBundleAdjuster(slam_map, min_inlier_edges=10_000).run(loop_kf_id=4)
    assert not result.success
    np.testing.assert_allclose(kf1.Tcw(), old_pose)
    np.testing.assert_allclose(points[0].get_position(), old_point)


def test_global_ba_abort_flag_returns_aborted():
    slam_map, _, kf1, _, _ = _build_gba_scene(noisy=True)
    old_pose = kf1.Tcw().copy()
    flag = StopFlag(True)
    result = GlobalBundleAdjuster(slam_map).run(loop_kf_id=5, stop_flag=flag)
    assert result.aborted
    assert not result.success
    np.testing.assert_allclose(kf1.Tcw(), old_pose)


def test_loop_closing_triggers_global_ba_when_enabled(monkeypatch):
    slam_map, cam, loop_kf, current_kf = build_loop_scene(n=80)
    database = __import__("visual_slam.orbslam.slam", fromlist=["KeyFrameDatabase", "load_default_vocabulary"])
    kf_db = database.KeyFrameDatabase(database.load_default_vocabulary())
    kf_db.add(loop_kf)
    slam = make_slam_namespace(slam_map, cam, kf_db)
    slam.enable_global_ba = True
    slam.global_ba_after_loop = True
    slam.global_ba_iterations = 1
    closing = LoopClosing(slam, kf_db, consistency_threshold=0)

    def fake_run(self, loop_kf_id=0, stop_flag=None, verbose=False):
        return GlobalBAResult(started=True, success=True, num_keyframes=2, num_map_points=3, reason="ok")

    monkeypatch.setattr(GlobalBundleAdjuster, "run", fake_run)
    assert closing.process_keyframe(current_kf)
    assert closing.last_diagnostics.global_ba_started
    assert closing.last_diagnostics.global_ba_success


def test_loop_closing_does_not_trigger_global_ba_when_disabled():
    slam_map, cam, loop_kf, current_kf = build_loop_scene(n=80)
    database = __import__("visual_slam.orbslam.slam", fromlist=["KeyFrameDatabase", "load_default_vocabulary"])
    kf_db = database.KeyFrameDatabase(database.load_default_vocabulary())
    kf_db.add(loop_kf)
    slam = make_slam_namespace(slam_map, cam, kf_db)
    slam.enable_global_ba = False
    slam.global_ba_after_loop = False
    closing = LoopClosing(slam, kf_db, consistency_threshold=0)

    assert closing.process_keyframe(current_kf)
    assert not closing.last_diagnostics.global_ba_started


def test_global_ba_diagnostics_present():
    result = GlobalBAResult()
    diagnostics = result.to_diagnostics()
    required = {
        "started",
        "success",
        "num_keyframes",
        "num_map_points",
        "num_edges",
        "num_inliers",
        "num_outliers",
        "mean_error_before",
        "mean_error_after",
        "elapsed_sec",
        "aborted",
        "reason",
    }
    assert required.issubset(diagnostics)
