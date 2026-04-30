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
    MotionModel,
    PinholeCamera,
    ProjectionMatcher,
    RotationHistogram,
    SensorType,
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


def make_Tcw(tx=0.0):
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.array([tx, 0.0, 0.0])
    return T


def project(Tcw, point, cam):
    pc = Tcw[:3, :3] @ point + Tcw[:3, 3]
    u = cam.fx * pc[0] / pc[2] + cam.cx
    v = cam.fy * pc[1] / pc[2] + cam.cy
    ur = u - cam.bf / pc[2]
    return float(u), float(v), float(ur)


def make_synthetic_frame(cam, Tcw, points, descriptors):
    frame = Frame(camera=cam, img=None, depth_img=None, pose=g2o.Isometry3d(Tcw), timestamp=1.0)

    kps = []
    uRs = []
    for point in points:
        u, v, ur = project(Tcw, point.get_position(), cam)
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


def make_points_and_descriptors(n=25):
    points = []
    descriptors = []

    for i in range(n):
        point = MapPoint(np.array([-0.4 + 0.04 * i, -0.1 + 0.01 * (i % 5), 2.0 + 0.02 * (i % 3)]))
        desc = np.full(32, i % 255, dtype=np.uint8)
        point.set_descriptor(desc)
        points.append(point)
        descriptors.append(desc)

    return points, np.asarray(descriptors, dtype=np.uint8)


def test_rotation_histogram_valid_invalid_indices():
    hist = RotationHistogram(histogram_length=12)

    # Top 3 orientation bins should remain valid.
    for idx in range(10):
        hist.push(5.0, idx)
    for idx in range(10, 13):
        hist.push(90.0, idx)
    for idx in range(20, 22):
        hist.push(180.0, idx)

    # Fourth weaker bin should be rejected.
    hist.push(270.0, 99)

    valid = hist.get_valid_idxs()
    invalid = hist.get_invalid_idxs()

    assert set(range(10)).issubset(set(valid))
    assert set(range(10, 13)).issubset(set(valid))
    assert set(range(20, 22)).issubset(set(valid))
    assert 99 in invalid


def test_motion_model_constant_velocity_matrix_prediction():
    model = MotionModel()

    T0 = make_Tcw(0.0)
    T1 = make_Tcw(0.1)

    model.update_pose_from_matrix(timestamp=0.0, Tcw=T0)
    model.update_pose_from_matrix(timestamp=1.0, Tcw=T1)

    predicted, _ = model.predict_pose(timestamp=2.0)

    np.testing.assert_allclose(predicted.matrix()[:3, 3], np.array([0.2, 0.0, 0.0]), atol=1e-12)


def test_search_frame_by_projection_matches_reference_points_to_current_frame():
    setup_tracker()

    cam = make_camera()
    points, descriptors = make_points_and_descriptors(25)

    f_ref = make_synthetic_frame(cam, make_Tcw(0.0), points, descriptors)
    f_ref.points = list(points)

    f_cur = make_synthetic_frame(cam, make_Tcw(0.05), points, descriptors)

    idxs_ref, idxs_cur, count = ProjectionMatcher.search_frame_by_projection(
        f_ref,
        f_cur,
        max_reproj_distance=10,
        max_descriptor_distance=100,
        is_monocular=False,
    )

    assert count >= 20
    assert len(idxs_ref) == count
    assert len(idxs_cur) == count
    assert sum(p is not None for p in f_cur.points) >= 20


def test_search_map_by_projection_matches_points_to_frame():
    setup_tracker()

    cam = make_camera()
    points, descriptors = make_points_and_descriptors(25)

    f_cur = make_synthetic_frame(cam, make_Tcw(0.05), points, descriptors)

    count, frame_idxs = ProjectionMatcher.search_map_by_projection(
        points,
        f_cur,
        max_reproj_distance=10,
        max_descriptor_distance=100,
    )

    assert count >= 20
    assert len(frame_idxs) == count
    assert sum(p is not None for p in f_cur.points) >= 20


def test_search_keyframe_by_projection_and_fuse():
    setup_tracker()

    cam = make_camera()
    points, descriptors = make_points_and_descriptors(25)

    ref_frame = make_synthetic_frame(cam, make_Tcw(0.0), points, descriptors)
    ref_frame.points = list(points)

    kf_ref = KeyFrame(ref_frame, kid=0)
    for i, p in enumerate(points):
        p.add_observation(kf_ref, i)

    f_cur = make_synthetic_frame(cam, make_Tcw(0.05), points, descriptors)

    idxs_ref, idxs_cur, count = ProjectionMatcher.search_keyframe_by_projection(
        kf_ref,
        f_cur,
        max_reproj_distance=10,
        max_descriptor_distance=100,
    )

    assert count >= 20
    assert len(idxs_ref) == count
    assert len(idxs_cur) == count

    fuse_frame = make_synthetic_frame(cam, make_Tcw(0.0), points, descriptors)
    kf_fuse = KeyFrame(fuse_frame, kid=1)

    fused = ProjectionMatcher.search_and_fuse(
        points,
        kf_fuse,
        max_reproj_distance=10,
        max_descriptor_distance=100,
    )

    assert fused >= 20


def test_search_all_map_by_projection_uses_map_points():
    setup_tracker()

    cam = make_camera()
    points, descriptors = make_points_and_descriptors(25)

    slam_map = Map()
    for p in points:
        slam_map.add_point(p)

    f_cur = make_synthetic_frame(cam, make_Tcw(0.05), points, descriptors)

    count, frame_idxs = ProjectionMatcher.search_all_map_by_projection(
        slam_map,
        f_cur,
        max_descriptor_distance=100,
    )

    assert count >= 20
    assert len(frame_idxs) == count
