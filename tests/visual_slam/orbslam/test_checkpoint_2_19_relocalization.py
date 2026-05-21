from types import SimpleNamespace

import cv2
import g2o
import numpy as np

from visual_slam.orbslam.local_features import create_orb2_feature_tracker
from visual_slam.orbslam.slam import (
    FeatureTrackerShared,
    Frame,
    KeyFrame,
    Map,
    MapPoint,
    PinholeCamera,
    Relocalizer,
    SensorType,
    SlamState,
    TemporaryRelocalizationKeyFrameDatabase,
    Tracking,
)


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


def project(Tcw, point_w, cam):
    pc = Tcw[:3, :3] @ point_w + Tcw[:3, 3]
    u = cam.fx * pc[0] / pc[2] + cam.cx
    v = cam.fy * pc[1] / pc[2] + cam.cy
    ur = u - cam.bf / pc[2]
    return float(u), float(v), float(ur)


def make_points_and_descriptors(n=70):
    rng = np.random.default_rng(219)
    points = []
    descriptors = rng.integers(0, 256, size=(n, 32), dtype=np.uint8)

    for i in range(n):
        x = -0.45 + 0.09 * (i % 10)
        y = -0.20 + 0.08 * ((i // 10) % 7)
        z = 2.0 + 0.08 * (i % 5)
        point = MapPoint(np.array([x, y, z], dtype=np.float64))
        point.set_descriptor(descriptors[i])
        points.append(point)

    return points, descriptors


def make_synthetic_frame(cam, Tcw_obs, Tcw_pose, points, descriptors, timestamp=1.0, frame_id=None):
    frame = Frame(
        camera=cam,
        img=None,
        depth_img=None,
        pose=g2o.Isometry3d(Tcw_pose),
        id=frame_id,
        timestamp=timestamp,
    )

    kps = []
    uRs = []
    for point in points:
        u, v, ur = project(Tcw_obs, point.get_position(), cam)
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
    frame.idxs = np.arange(len(kps), dtype=np.int32)
    return frame


def build_reference_map(n=70):
    setup_tracker()
    slam_map = Map()
    cam = make_camera()
    points, descriptors = make_points_and_descriptors(n)
    f0 = make_synthetic_frame(cam, make_Tcw(), make_Tcw(), points, descriptors, timestamp=1.0, frame_id=0)
    kf0 = KeyFrame(f0, kid=0)
    slam_map.add_keyframe(kf0)

    for idx, point in enumerate(points):
        f0.points[idx] = point
        kf0.points[idx] = point
        point.add_observation(kf0, idx)
        point.add_frame_view(f0, idx)
        point.update_info()
        slam_map.add_point(point)

    kf0.update_connections()
    return slam_map, cam, kf0, points, descriptors


def make_slam_namespace(slam_map, camera):
    return SimpleNamespace(
        camera=camera,
        sensor_type=SensorType.RGBD,
        map=slam_map,
        feature_tracker=FeatureTrackerShared.feature_tracker,
        local_mapping=None,
    )


def test_relocalizer_imports_and_initializes():
    relocalizer = Relocalizer()

    assert relocalizer.num_relocalization_candidates == 0
    assert hasattr(relocalizer, "detect_relocalization_candidates")
    assert hasattr(relocalizer, "estimate_pose_pnp")


def test_relocalizer_candidate_retrieval_interface_exists():
    slam_map, cam, kf0, points, descriptors = build_reference_map(n=20)
    frame = make_synthetic_frame(cam, make_Tcw(0.1), make_Tcw(0.1), points, descriptors, timestamp=2.0, frame_id=1)
    database = TemporaryRelocalizationKeyFrameDatabase(slam_map)

    candidates = database.detect_relocalization_candidates(frame)

    assert candidates == [kf0]


def test_pnp_pose_recovery_works_on_synthetic_correspondences():
    slam_map, cam, kf0, points, descriptors = build_reference_map(n=60)
    T_true = make_Tcw(0.12, -0.03, 0.06)
    frame = make_synthetic_frame(cam, T_true, make_Tcw(), points, descriptors, timestamp=2.0, frame_id=1)
    relocalizer = Relocalizer(slam_map)

    idxs_frame = np.arange(len(points), dtype=np.int32)
    idxs_kf = np.arange(len(points), dtype=np.int32)
    points_3d, points_2d, _, _, _ = relocalizer.prepare_input_data_for_pnp(frame, kf0, idxs_frame, idxs_kf)

    result = relocalizer.estimate_pose_pnp(frame, points_3d, points_2d)

    assert result.success
    assert result.num_inliers >= 50
    np.testing.assert_allclose(result.Tcw[:3, 3], T_true[:3, 3], atol=1e-4)


def test_relocalization_candidate_with_enough_matches_recovers_pose():
    slam_map, cam, kf0, points, descriptors = build_reference_map(n=70)
    T_true = make_Tcw(0.10, 0.02, -0.03)
    frame = make_synthetic_frame(cam, T_true, make_Tcw(0.4, -0.1, 0.2), points, descriptors, timestamp=2.0, frame_id=1)
    relocalizer = Relocalizer(slam_map)

    ok = relocalizer.relocalize(frame, keyframes_map=slam_map.keyframes_map)

    assert ok
    assert frame.kf_ref is kf0
    assert relocalizer.last_relocalization_success
    assert relocalizer.num_relocalization_candidates == 1
    assert relocalizer.num_relocalization_matches >= 50
    assert relocalizer.num_relocalization_inliers >= 50
    np.testing.assert_allclose(frame.Tcw()[:3, 3], T_true[:3, 3], atol=1e-3)


def test_relocalization_candidate_with_too_few_matches_fails_cleanly():
    slam_map, cam, kf0, points, descriptors = build_reference_map(n=8)
    frame = make_synthetic_frame(cam, make_Tcw(0.1), make_Tcw(), points, descriptors, timestamp=2.0, frame_id=1)
    relocalizer = Relocalizer(slam_map)

    ok = relocalizer.relocalize(frame, keyframes_map=slam_map.keyframes_map)

    assert not ok
    assert not relocalizer.last_relocalization_success
    assert relocalizer.last_relocalization_error == "all candidates rejected"


def test_tracking_lost_state_calls_relocalization_path():
    slam_map, cam, kf0, points, descriptors = build_reference_map(n=20)
    slam = make_slam_namespace(slam_map, cam)
    tracking = Tracking(slam)
    tracking.state = SlamState.LOST
    tracking.kf_ref = kf0
    tracking.kf_last = kf0
    tracking.need_new_keyframe = lambda: False

    previous = make_synthetic_frame(cam, make_Tcw(), make_Tcw(), points, descriptors, timestamp=1.0)
    slam_map.add_frame(previous)

    class DummyRelocalizer:
        def __init__(self):
            self.calls = 0

        def relocalize(self, frame, **kwargs):
            self.calls += 1
            frame.kf_ref = kf0
            frame.update_pose(g2o.Isometry3d(make_Tcw(0.1)))
            return True

    dummy = DummyRelocalizer()
    tracking.relocalizer = dummy

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.zeros((480, 640), dtype=np.uint16)
    ok = tracking.track(image, depth=depth, img_id=2, timestamp=2.0)

    assert ok
    assert dummy.calls == 1
    assert tracking.state == SlamState.OK


def test_successful_relocalization_restores_tracking_state_to_ok():
    slam_map, cam, kf0, points, descriptors = build_reference_map(n=70)
    slam = make_slam_namespace(slam_map, cam)
    tracking = Tracking(slam)
    frame = make_synthetic_frame(cam, make_Tcw(0.1), make_Tcw(), points, descriptors, timestamp=2.0, frame_id=1)

    tracking.state = SlamState.LOST
    ok = tracking.relocalize(frame)
    if ok:
        tracking.last_reloc_frame_id = frame.id
        tracking.state = SlamState.OK
        tracking.pose_is_ok = True
        tracking.kf_ref = frame.kf_ref
        slam_map.update_local_map(tracking.kf_ref)

    assert ok
    assert tracking.state == SlamState.OK
    assert tracking.kf_ref is kf0


def test_failed_relocalization_leaves_tracking_state_lost():
    slam_map, cam, kf0, points, descriptors = build_reference_map(n=8)
    slam = make_slam_namespace(slam_map, cam)
    tracking = Tracking(slam)
    frame = make_synthetic_frame(cam, make_Tcw(0.1), make_Tcw(), points, descriptors, timestamp=2.0, frame_id=1)

    tracking.state = SlamState.LOST
    ok = tracking.relocalize(frame)
    if not ok:
        tracking.state = SlamState.LOST

    assert not ok
    assert tracking.state == SlamState.LOST


def test_failed_relocalization_does_not_corrupt_map():
    slam_map, cam, kf0, points, descriptors = build_reference_map(n=8)
    frame = make_synthetic_frame(cam, make_Tcw(0.1), make_Tcw(), points, descriptors, timestamp=2.0, frame_id=1)
    relocalizer = Relocalizer(slam_map)

    keyframes_before = slam_map.num_keyframes()
    points_before = slam_map.num_points()
    kf_points_before = list(kf0.points)

    ok = relocalizer.relocalize(frame, keyframes_map=slam_map.keyframes_map)

    assert not ok
    assert slam_map.num_keyframes() == keyframes_before
    assert slam_map.num_points() == points_before
    assert kf0.points == kf_points_before
