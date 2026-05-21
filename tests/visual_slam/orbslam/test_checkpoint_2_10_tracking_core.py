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
    TrackingCore,
)
from visual_slam.orbslam.utilities.geometry import inv_T, poseRt, skew
from visual_slam.orbslam.utilities.geom_2views import computeF12, estimate_pose_ess_mat


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


def make_points_and_descriptors(n=30):
    points = []
    descriptors = []

    for i in range(n):
        p = MapPoint(np.array([
            -0.45 + 0.03 * i,
            -0.15 + 0.02 * (i % 6),
            2.0 + 0.03 * (i % 4),
        ], dtype=np.float64))
        d = np.full(32, i % 255, dtype=np.uint8)
        p.set_descriptor(d)
        points.append(p)
        descriptors.append(d)

    return points, np.asarray(descriptors, dtype=np.uint8)


def test_ported_geometry_helpers():
    R = np.eye(3, dtype=np.float64)
    t = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    T = poseRt(R, t)
    Tinv = inv_T(T)

    np.testing.assert_allclose(T[:3, :3], R)
    np.testing.assert_allclose(T[:3, 3], t)
    np.testing.assert_allclose(Tinv @ T, np.eye(4), atol=1e-12)

    S = skew(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(S + S.T, np.zeros((3, 3)), atol=1e-12)


def test_computeF12_returns_valid_shapes():
    setup_tracker()
    cam = make_camera()
    points, descriptors = make_points_and_descriptors(20)

    f1 = make_synthetic_frame(cam, make_Tcw(0.0), make_Tcw(0.0), points, descriptors)
    f2 = make_synthetic_frame(cam, make_Tcw(0.1), make_Tcw(0.1), points, descriptors)

    F12, H21 = computeF12(f1, f2)

    assert F12.shape == (3, 3)
    assert H21.shape == (3, 3)
    assert np.all(np.isfinite(F12))
    assert np.all(np.isfinite(H21))


def test_estimate_pose_ess_mat_handles_insufficient_points():
    Trc, mask = estimate_pose_ess_mat(
        np.empty((0, 2), dtype=np.float64),
        np.empty((0, 2), dtype=np.float64),
    )

    assert Trc is None
    assert mask is None


def test_find_homography_with_ransac_filters_outliers():
    setup_tracker()
    cam = make_camera()

    points, descriptors = make_points_and_descriptors(25)
    f_ref = make_synthetic_frame(cam, make_Tcw(0.0), make_Tcw(0.0), points, descriptors)
    f_cur = make_synthetic_frame(cam, make_Tcw(0.0), make_Tcw(0.0), points, descriptors)

    for i in range(3):
        f_cur.kps[i].pt = (f_cur.kps[i].pt[0] + 120.0, f_cur.kps[i].pt[1] - 90.0)

    idxs = np.arange(25, dtype=np.int32)

    ok, idxs_cur, idxs_ref, num_inliers, num_outliers = TrackingCore.find_homography_with_ransac(
        f_cur,
        f_ref,
        idxs,
        idxs,
        reproj_threshold=5,
        min_num_inliers=15,
    )

    assert ok
    assert num_inliers >= 20
    assert num_outliers >= 1
    assert len(idxs_cur) == num_inliers
    assert len(idxs_ref) == num_inliers


def test_propagate_map_point_matches():
    setup_tracker()
    cam = make_camera()

    points, descriptors = make_points_and_descriptors(30)

    f_ref = make_synthetic_frame(cam, make_Tcw(0.0), make_Tcw(0.0), points, descriptors)
    f_cur = make_synthetic_frame(cam, make_Tcw(0.05), make_Tcw(0.05), points, descriptors)

    f_ref.points = list(points)

    idxs = np.arange(30, dtype=np.int32)

    count, idx_ref_out, idx_cur_out = TrackingCore.propagate_map_point_matches(
        f_ref,
        f_cur,
        idxs,
        idxs,
        max_descriptor_distance=100,
    )

    assert count == 30
    assert len(idx_ref_out) == 30
    assert len(idx_cur_out) == 30
    assert sum(p is not None for p in f_cur.points) == 30


def test_create_vo_points_from_rgbd_depth():
    setup_tracker()
    cam = make_camera()

    points, descriptors = make_points_and_descriptors(30)
    frame = make_synthetic_frame(cam, make_Tcw(0.0), make_Tcw(0.0), points, descriptors)

    frame.points = [None] * len(frame.kps)

    created = TrackingCore.create_vo_points(frame, max_num_points=10)

    assert len(created) >= 10
    assert sum(p is not None for p in frame.points) >= 10
    assert all(p.is_in_frame(frame) for p in created)


def test_create_and_add_stereo_map_points_on_new_keyframe():
    setup_tracker()
    cam = make_camera()
    points, descriptors = make_points_and_descriptors(30)

    frame = make_synthetic_frame(cam, make_Tcw(0.0), make_Tcw(0.0), points, descriptors)
    kf = KeyFrame(frame, kid=0)

    slam_map = Map()
    slam_map.add_keyframe(kf)

    kf.points = [None] * len(kf.kps)
    frame.points = [None] * len(frame.kps)

    count = TrackingCore.create_and_add_stereo_map_points_on_new_kf(
        frame,
        kf,
        slam_map,
        img=None,
    )

    assert count >= 30
    assert slam_map.num_points() >= 30
    assert sum(p is not None for p in kf.points) >= 30


def test_count_tracked_and_non_tracked_close_points():
    setup_tracker()
    cam = make_camera()

    points, descriptors = make_points_and_descriptors(20)
    frame = make_synthetic_frame(cam, make_Tcw(0.0), make_Tcw(0.0), points, descriptors)

    frame.points = [None] * len(frame.kps)
    frame.outliers = np.zeros(len(frame.kps), dtype=bool)

    for i in range(5):
        frame.points[i] = points[i]

    num_tracked, num_non_tracked, tracked_mask = TrackingCore.count_tracked_and_non_tracked_close_points(
        frame,
        SensorType.RGBD,
    )

    assert num_tracked == 5
    assert num_non_tracked == len(frame.kps) - 5
    assert tracked_mask.dtype == bool
    assert np.count_nonzero(tracked_mask) == 5


def test_estimate_pose_by_fitting_ess_mat_handles_empty_input():
    setup_tracker()
    cam = make_camera()

    points, descriptors = make_points_and_descriptors(10)
    f_ref = make_synthetic_frame(cam, make_Tcw(0.0), make_Tcw(0.0), points, descriptors)
    f_cur = make_synthetic_frame(cam, make_Tcw(0.1), make_Tcw(0.1), points, descriptors)

    idx_ref, idx_cur, num_inliers = TrackingCore.estimate_pose_by_fitting_ess_mat(
        f_ref,
        f_cur,
        [],
        [],
    )

    assert len(idx_ref) == 0
    assert len(idx_cur) == 0
    assert num_inliers == 0
