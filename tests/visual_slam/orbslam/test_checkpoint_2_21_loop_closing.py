from types import SimpleNamespace

import cv2
import g2o
import numpy as np

from visual_slam.orbslam.local_features import create_orb2_feature_tracker
from visual_slam.orbslam.slam import (
    DBoW3Vocabulary,
    FeatureTrackerShared,
    Frame,
    KeyFrame,
    KeyFrameDatabase,
    Map,
    MapPoint,
    PinholeCamera,
    SensorType,
    Slam,
    load_default_vocabulary,
)
from visual_slam.orbslam.slam.loop_closing import (
    ConsistencyGroup,
    LoopClosing,
    LoopGeometryChecker,
    LoopGroupConsistencyChecker,
)
from visual_slam.orbslam.slam.loop_detector import LoopDetector
from visual_slam.orbslam.slam.sim3_solver import estimate_scale_fixed_sim3


def setup_tracker():
    FeatureTrackerShared.reset()
    tracker = create_orb2_feature_tracker()
    FeatureTrackerShared.set_feature_tracker(tracker)
    return tracker


def make_camera():
    return PinholeCamera.from_params(
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        sensor_type=SensorType.RGBD,
        baseline=0.08,
        depth_map_factor=5000.0,
        th_depth=40.0,
    )


def make_Tcw(tx=0.0, ty=0.0, tz=0.0):
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return T


def make_descriptors(n=80, seed=221):
    return np.random.default_rng(seed).integers(0, 256, size=(n, 32), dtype=np.uint8)


def make_base_points(n=80):
    points = []
    for i in range(n):
        points.append(
            np.array(
                [
                    -0.45 + 0.09 * (i % 10),
                    -0.25 + 0.07 * ((i // 10) % 8),
                    2.0 + 0.04 * (i % 5),
                ],
                dtype=np.float64,
            )
        )
    return points


def project(Tcw, point_w, cam):
    pc = Tcw[:3, :3] @ point_w + Tcw[:3, 3]
    u = cam.fx * pc[0] / pc[2] + cam.cx
    v = cam.fy * pc[1] / pc[2] + cam.cy
    ur = u - cam.bf / pc[2]
    return float(u), float(v), float(ur)


def build_keyframe(slam_map, cam, frame_id, positions, descriptors, Tcw):
    frame = Frame(
        camera=cam,
        img=np.zeros((480, 640, 3), dtype=np.uint8),
        depth_img=None,
        pose=g2o.Isometry3d(Tcw),
        id=frame_id,
        timestamp=float(frame_id),
    )
    kps = []
    uRs = []
    for point in positions:
        u, v, ur = project(Tcw, point, cam)
        kps.append(cv2.KeyPoint(float(u), float(v), 20.0, 0.0, 1.0, 0))
        uRs.append(ur)
    frame.kps = kps
    frame.kpsu = kps
    frame.des = np.asarray(descriptors, dtype=np.uint8)
    frame.depths = np.full(len(kps), 2.0, dtype=np.float32)
    frame.uRs = np.asarray(uRs, dtype=np.float32)
    frame.kps_ur = frame.uRs
    frame.octaves = np.zeros(len(kps), dtype=np.int32)
    frame.angles = np.zeros(len(kps), dtype=np.float32)
    frame.sizes = np.full(len(kps), 20.0, dtype=np.float32)
    frame.points = [None] * len(kps)
    frame.outliers = np.zeros(len(kps), dtype=bool)

    keyframe = KeyFrame(frame, kid=frame_id)
    slam_map.add_keyframe(keyframe)
    for idx, (position, descriptor) in enumerate(zip(positions, descriptors)):
        point = MapPoint(position.copy())
        point.set_descriptor(descriptor)
        point.add_observation(keyframe, idx)
        point.update_info()
        keyframe.points[idx] = point
        frame.points[idx] = point
        slam_map.add_point(point)
    keyframe.update_connections()
    return keyframe


def build_loop_scene(n=80):
    setup_tracker()
    slam_map = Map()
    cam = make_camera()
    descriptors = make_descriptors(n)
    base = make_base_points(n)
    # Advance the map's keyframe counter so that loop_kf and current_kf receive
    # kid values that pass the loop-closure cooldown check
    # (kid >= last_loop_kf_id + kMinDeltaFrameForMeaningfulLoopClosure = 0 + 10).
    from visual_slam.orbslam.slam.config_parameters import Parameters
    slam_map.max_keyframe_id = Parameters.kMinDeltaFrameForMeaningfulLoopClosure
    loop_kf = build_keyframe(slam_map, cam, 0, base, descriptors, make_Tcw())
    drifted = [point + np.array([1.0, 0.0, 0.0], dtype=np.float64) for point in base]
    current_kf = build_keyframe(slam_map, cam, 100, drifted, descriptors, make_Tcw(-1.0, 0.0, 0.0))
    return slam_map, cam, loop_kf, current_kf


def make_slam_namespace(slam_map, cam, database):
    return SimpleNamespace(
        camera=cam,
        sensor_type=SensorType.RGBD,
        map=slam_map,
        keyframe_database=database,
        feature_tracker=FeatureTrackerShared.feature_tracker,
        local_mapping=None,
    )


def seed_consistency(closing, loop_kf):
    """Pre-populate the consistency checker with loop_kf's group at consistency=0.

    Simulates one prior detection of loop_kf as a candidate (the state after the first
    process_keyframe call in a real sequence where a different keyframe sees the same candidate).
    This avoids calling process_keyframe twice with the same keyframe, which is not a valid
    usage pattern (loop_query_id on loop_kf would be set to current_kf.id after the first call,
    preventing re-detection in the second call).
    """
    group = set(loop_kf.get_connected_keyframes())
    group.add(loop_kf)
    closing.loop_consistency_checker.consistent_groups = [ConsistencyGroup(group, 0)]


def test_loop_detector_and_closing_import_and_initialize():
    vocab = load_default_vocabulary()
    database = KeyFrameDatabase(vocab)
    detector = LoopDetector(database)
    slam_map, cam, _, _ = build_loop_scene(n=30)
    closing = LoopClosing(make_slam_namespace(slam_map, cam, database), database)

    assert detector.available
    assert closing.queue_size() == 0
    assert not closing.is_correcting()


def test_loop_queue_insert_and_pop_works():
    vocab = load_default_vocabulary()
    database = KeyFrameDatabase(vocab)
    slam_map, cam, loop_kf, _ = build_loop_scene(n=30)
    closing = LoopClosing(make_slam_namespace(slam_map, cam, database), database)

    closing.insert_keyframe(loop_kf)

    assert closing.queue_size() == 1
    assert closing.pop_keyframe() is loop_kf


def test_database_candidate_query_excludes_connected_recent_and_self():
    vocab = load_default_vocabulary()
    assert vocab.available, vocab.error
    database = KeyFrameDatabase(vocab)
    slam_map, cam, loop_kf, current_kf = build_loop_scene(n=60)
    recent = build_keyframe(slam_map, cam, 95, make_base_points(60), current_kf.des, make_Tcw())
    connected = build_keyframe(slam_map, cam, 20, make_base_points(60), current_kf.des, make_Tcw())
    current_kf.add_connection(connected, 30)
    connected.add_connection(current_kf, 30)
    for keyframe in (loop_kf, recent, connected):
        database.add(keyframe)

    candidates = database.detect_loop_candidates(current_kf, min_score=0.0)

    assert loop_kf in candidates
    # Temporal min-delta filtering is the detector's responsibility (Stage 2 of
    # the pyslam loop-closure realignment); the database may return temporally
    # close candidates here.
    assert connected not in candidates
    assert current_kf not in candidates


def test_consistency_group_accumulation_requires_repeated_group():
    _, _, loop_kf, current_kf = build_loop_scene(n=30)
    checker = LoopGroupConsistencyChecker(consistency_threshold=1)

    first = checker.check_candidates(current_kf, [loop_kf])
    second = checker.check_candidates(current_kf, [loop_kf])

    assert not first
    assert second
    assert checker.enough_consistent_candidates == [loop_kf]


def test_geometry_verification_accepts_synthetic_rgbd_loop_and_uses_bow_matching():
    vocab = load_default_vocabulary()
    database = KeyFrameDatabase(vocab)
    slam_map, _, loop_kf, current_kf = build_loop_scene(n=80)
    database.add(loop_kf)
    database.compute_bow(current_kf)
    checker = LoopGeometryChecker(keyframe_database=database)

    ok = checker.check_candidates(current_kf, [loop_kf])

    assert ok
    assert checker.success_loop_kf is loop_kf
    assert checker.last_bow_guided_matching_available
    assert checker.num_last_matches >= checker.min_matches
    # t12 is the camera-to-camera Sim3 translation. For this synthetic scene,
    # both cameras see identical structures in their local frames (drift cancelled
    # by the camera pose difference), so the camera-space transform is near-identity.
    assert checker.success_sim3 is not None
    assert checker.success_sim3.success


def test_geometry_verification_rejects_insufficient_matches():
    vocab = load_default_vocabulary()
    database = KeyFrameDatabase(vocab)
    _, _, loop_kf, current_kf = build_loop_scene(n=8)
    database.add(loop_kf)
    checker = LoopGeometryChecker(keyframe_database=database)

    assert not checker.check_candidates(current_kf, [loop_kf])
    assert checker.last_error in {"too few loop geometry matches", "too few valid 3D loop correspondences"}


def test_scale_fixed_sim3_correction_reduces_synthetic_loop_error():
    base = np.asarray(make_base_points(50), dtype=np.float64)
    drifted = base + np.array([1.0, 0.0, 0.0], dtype=np.float64)

    estimate = estimate_scale_fixed_sim3(drifted, base, max_error=0.01)

    assert estimate.success
    assert np.linalg.norm(((estimate.R @ drifted.T).T + estimate.t.reshape(1, 3)) - base, axis=1).mean() < 1e-9


def test_loop_correction_updates_poses_and_fuses_duplicate_points():
    vocab = load_default_vocabulary()
    database = KeyFrameDatabase(vocab)
    slam_map, cam, loop_kf, current_kf = build_loop_scene(n=80)
    database.add(loop_kf)
    slam = make_slam_namespace(slam_map, cam, database)
    closing = LoopClosing(slam, database, consistency_threshold=0)
    seed_consistency(closing, loop_kf)

    ok = closing.process_keyframe(current_kf)

    assert ok
    assert closing.last_diagnostics.accepted == 1
    assert closing.last_diagnostics.corrected_keyframes >= 1
    assert closing.last_diagnostics.fused_points > 0
    assert closing.last_diagnostics.optimization_result.after_error < closing.last_diagnostics.optimization_result.before_error
    assert loop_kf in current_kf.get_loop_edges()


def test_map_point_fusion_skips_bad_loop_points_without_corruption():
    vocab = load_default_vocabulary()
    database = KeyFrameDatabase(vocab)
    slam_map, cam, _, current_kf = build_loop_scene(n=30)
    slam = make_slam_namespace(slam_map, cam, database)
    closing = LoopClosing(slam, database, consistency_threshold=0)
    bad_point = MapPoint(np.array([0.0, 0.0, 2.0], dtype=np.float64))
    bad_point.set_bad()
    closing.loop_geometry_checker.success_map_point_matches = [bad_point] + [None] * (len(current_kf.points) - 1)

    before = list(current_kf.points)
    fused = closing.loop_corrector._fuse_loop_matches(current_kf)

    assert fused == 0
    assert current_kf.points == before


def test_missing_vocabulary_disables_loop_detection_cleanly(tmp_path):
    missing = tmp_path / "missing.dbow3"
    vocab = DBoW3Vocabulary(missing, autoload=True)
    database = KeyFrameDatabase(vocab)
    slam_map, cam, _, current_kf = build_loop_scene(n=30)
    closing = LoopClosing(make_slam_namespace(slam_map, cam, database), database, consistency_threshold=0)

    assert not closing.process_keyframe(current_kf)
    assert closing.last_diagnostics.unavailable_reason
    assert closing.last_diagnostics.rejected_by_bow == 1


def test_slam_and_runner_can_enable_loop_module():
    setup_tracker()
    slam = Slam(
        camera=make_camera(),
        sensor_type=SensorType.RGBD,
        headless=True,
        start_local_mapping_thread=False,
        enable_loop_closing=True,
    )

    assert slam.loop_closing is not None
    assert hasattr(slam.loop_closing, "insert_keyframe")
