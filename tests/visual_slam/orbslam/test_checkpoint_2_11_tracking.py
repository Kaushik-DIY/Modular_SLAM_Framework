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
    SensorType,
    SlamState,
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


def make_slam():
    tracker = setup_tracker()
    cam = make_camera()
    slam_map = Map()
    return SimpleNamespace(
        camera=cam,
        sensor_type=SensorType.RGBD,
        map=slam_map,
        feature_tracker=tracker,
        local_mapping=None,
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


def make_points_and_descriptors(n=40):
    points = []
    descriptors = []

    for i in range(n):
        p = MapPoint(np.array([
            -0.45 + 0.025 * i,
            -0.15 + 0.02 * (i % 8),
            2.0 + 0.03 * (i % 4),
        ], dtype=np.float64))
        d = np.full(32, i % 255, dtype=np.uint8)
        p.set_descriptor(d)
        points.append(p)
        descriptors.append(d)

    return points, np.asarray(descriptors, dtype=np.uint8)


def make_synthetic_frame(cam, Tcw_obs, Tcw_pose, points, descriptors, timestamp=1.0):
    frame = Frame(
        camera=cam,
        img=None,
        depth_img=None,
        pose=g2o.Isometry3d(Tcw_pose),
        timestamp=timestamp,
    )

    kps = []
    uRs = []
    for p in points:
        u, v, ur = project(Tcw_obs, p.get_position(), cam)
        kps.append(cv2.KeyPoint(float(u), float(v), 20.0, 0.0, 1.0, 0))
        uRs.append(ur)

    frame.kps = kps
    frame.kpsu = kps
    frame.des = np.asarray(descriptors, dtype=np.uint8)
    frame.depths = np.full(len(kps), 2.0, dtype=np.float32)
    frame.uRs = np.asarray(uRs, dtype=np.float32)
    frame.kps_ur = frame.uRs
    frame.points = [None] * len(kps)
    frame.outliers = np.zeros(len(kps), dtype=bool)
    frame.idxs = np.arange(len(kps), dtype=np.int32)
    return frame


def build_reference_keyframe(slam, points, descriptors):
    cam = slam.camera
    f0 = make_synthetic_frame(cam, make_Tcw(0.0), make_Tcw(0.0), points, descriptors, timestamp=1.0)
    kf0 = KeyFrame(f0, kid=0)
    slam.map.add_keyframe(kf0)

    for i, p in enumerate(points):
        # pySLAM RGB-D initialization attaches created map points to both
        # the initial frame and the first keyframe.
        f0.points[i] = p
        kf0.points[i] = p
        p.add_observation(kf0, i)
        p.add_frame_view(f0, i)
        p.update_info()
        slam.map.add_point(p)

    kf0.update_connections()
    return f0, kf0


def test_tracking_pose_optimization_wrapper():
    slam = make_slam()
    tracking = Tracking(slam)

    points, descriptors = make_points_and_descriptors(30)
    cam = slam.camera

    T_true = make_Tcw(0.1, 0.0, 0.0)
    T_init = make_Tcw(0.2, -0.04, 0.08)

    frame = make_synthetic_frame(cam, T_true, T_init, points, descriptors, timestamp=2.0)
    frame.points = list(points)

    ok, mse = tracking.pose_optimization(frame, "unit-test")

    assert ok
    assert np.isfinite(mse)
    np.testing.assert_allclose(frame.Tcw()[:3, 3], T_true[:3, 3], atol=1e-3)


def test_track_reference_frame_matches_and_optimizes_pose():
    slam = make_slam()
    tracking = Tracking(slam)

    points, descriptors = make_points_and_descriptors(35)
    f0, kf0 = build_reference_keyframe(slam, points, descriptors)

    cam = slam.camera
    f1 = make_synthetic_frame(
        cam,
        make_Tcw(0.10),
        make_Tcw(0.20, -0.04, 0.08),
        points,
        descriptors,
        timestamp=2.0,
    )

    ok = tracking.track_reference_frame(kf0, f1, "match-frame-keyframe")

    assert ok
    assert tracking.pose_is_ok
    assert tracking.num_matched_map_points >= 20
    np.testing.assert_allclose(f1.Tcw()[:3, 3], np.array([0.10, 0.0, 0.0]), atol=1e-3)


def test_track_previous_frame_projection_path():
    slam = make_slam()
    tracking = Tracking(slam)

    points, descriptors = make_points_and_descriptors(35)
    f0, kf0 = build_reference_keyframe(slam, points, descriptors)

    tracking.kf_ref = kf0
    tracking.kf_last = kf0
    tracking.f_ref = f0
    tracking.motion_model.update_pose_from_matrix(1.0, f0.pose())

    cam = slam.camera
    f1 = make_synthetic_frame(
        cam,
        make_Tcw(0.10),
        make_Tcw(0.20, -0.04, 0.08),
        points,
        descriptors,
        timestamp=2.0,
    )

    ok = tracking.track_previous_frame(f0, f1)

    assert ok
    assert tracking.pose_is_ok
    assert tracking.num_matched_map_points >= 20
    np.testing.assert_allclose(f1.Tcw()[:3, 3], np.array([0.10, 0.0, 0.0]), atol=1e-3)


def test_track_local_map_after_reference_tracking():
    slam = make_slam()
    tracking = Tracking(slam)

    points, descriptors = make_points_and_descriptors(35)
    f0, kf0 = build_reference_keyframe(slam, points, descriptors)

    tracking.kf_ref = kf0
    tracking.kf_last = kf0
    tracking.f_ref = f0

    cam = slam.camera
    f1 = make_synthetic_frame(
        cam,
        make_Tcw(0.10),
        make_Tcw(0.20, -0.04, 0.08),
        points,
        descriptors,
        timestamp=2.0,
    )

    tracking.track_reference_frame(kf0, f1, "match-frame-keyframe")
    tracking.f_cur = f1

    ok = tracking.track_local_map()

    assert ok
    assert tracking.num_matched_map_points >= 20


def test_create_new_keyframe_adds_keyframe_and_points():
    slam = make_slam()
    tracking = Tracking(slam)

    points, descriptors = make_points_and_descriptors(35)
    f0, kf0 = build_reference_keyframe(slam, points, descriptors)

    cam = slam.camera
    f1 = make_synthetic_frame(
        cam,
        make_Tcw(0.10),
        make_Tcw(0.10),
        points,
        descriptors,
        timestamp=2.0,
    )
    f1.points = list(points)

    tracking.f_cur = f1
    tracking.kf_ref = kf0
    tracking.kf_last = kf0

    num_kfs_before = slam.map.num_keyframes()
    kf1 = tracking.create_new_keyframe()

    assert kf1 is not None
    assert slam.map.num_keyframes() == num_kfs_before + 1
    assert tracking.kf_last is kf1
    assert tracking.kf_ref is kf1
    assert kf1.map is slam.map


def test_tracking_history_update():
    slam = make_slam()
    tracking = Tracking(slam)

    points, descriptors = make_points_and_descriptors(20)
    f0, kf0 = build_reference_keyframe(slam, points, descriptors)

    tracking.f_cur = f0
    tracking.kf_ref = kf0
    tracking.state = SlamState.OK

    tracking.update_history()

    assert len(tracking.poses) == 1
    assert len(tracking.tracking_history.relative_frame_poses) == 1
    assert tracking.tracking_history.kf_references[0] is kf0
