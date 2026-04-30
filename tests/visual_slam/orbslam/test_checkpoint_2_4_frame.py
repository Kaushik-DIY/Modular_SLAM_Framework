from types import SimpleNamespace

import cv2
import g2o
import numpy as np

from visual_slam.orbslam.local_features import create_orb2_feature_tracker
from visual_slam.orbslam.slam import (
    FeatureTrackerShared,
    Frame,
    FrameBase,
    PinholeCamera,
    SensorType,
    are_map_points_visible_in_frame,
    match_frames,
)


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


def make_image():
    image = np.zeros((480, 640), dtype=np.uint8)
    for x in range(80, 600, 80):
        cv2.circle(image, (x, 240), 18, 255, 2)
    for y in range(80, 420, 80):
        cv2.line(image, (60, y), (580, y), 180, 2)
    return image


def setup_feature_tracker():
    FeatureTrackerShared.reset()
    tracker = create_orb2_feature_tracker()
    FeatureTrackerShared.set_feature_tracker(tracker)
    return tracker


def test_framebase_tcw_projection_helpers():
    cam = make_camera()

    Tcw = np.eye(4, dtype=np.float64)
    Tcw[:3, 3] = np.array([0.0, 0.0, 0.0])

    frame = FrameBase(camera=cam, pose=g2o.Isometry3d(Tcw), timestamp=1.0)

    pw = np.array([0.2, -0.1, 2.0], dtype=np.float64)
    pc = frame.transform_point(pw)
    np.testing.assert_allclose(pc, pw)

    uv, z = frame.project_point(pw)
    np.testing.assert_allclose(uv, np.array([370.0, 215.0]))
    assert z == 2.0

    uvu, z_st = frame.project_point(pw, do_stereo_project=True)
    np.testing.assert_allclose(uvu, np.array([370.0, 215.0, 350.0]))
    assert z_st == 2.0

    assert frame.is_in_image(uv, z)


def test_frame_extracts_features_and_depths_from_rgbd():
    setup_feature_tracker()
    cam = make_camera()

    image = make_image()
    depth = np.full((480, 640), 10000, dtype=np.uint16)  # 2 m with factor 1/5000

    frame = Frame(camera=cam, img=image, depth_img=depth, timestamp=1.0)

    assert len(frame.kps) > 20
    assert frame.des.dtype == np.uint8
    assert frame.des.shape[1] == 32
    assert len(frame.points) == len(frame.kps)
    assert len(frame.depths) == len(frame.kps)
    assert np.all(frame.depths[frame.depths > 0] > 0.0)

    valid = frame.depths > 0
    assert np.count_nonzero(valid) > 20
    np.testing.assert_allclose(frame.depths[valid], 2.0)

    # uR = uL - bf/depth
    first = int(np.flatnonzero(valid)[0])
    expected_ur = frame.kps[first].pt[0] - cam.bf / 2.0
    assert abs(frame.uRs[first] - expected_ur) < 1e-4


def test_frame_point_association_helpers():
    setup_feature_tracker()
    cam = make_camera()
    image = make_image()
    depth = np.full((480, 640), 10000, dtype=np.uint16)

    frame = Frame(camera=cam, img=image, depth_img=depth, timestamp=1.0)

    p0 = object()
    p1 = object()

    frame.set_point_match(p0, 0)
    assert frame.get_point_match(0) is p0

    frame.set_point_match(p1, 1)
    assert len(frame.get_matched_points()) == 2
    assert set(frame.get_matched_points_idxs().tolist()) == {0, 1}

    frame.remove_point_match(0)
    assert frame.get_point_match(0) is None
    assert frame.get_matched_points_idxs().tolist() == [1]

    frame.reset_points()
    assert len(frame.get_matched_points()) == 0


def test_are_map_points_visible_in_frame():
    cam = make_camera()
    frame = FrameBase(camera=cam, pose=g2o.Isometry3d(np.eye(4)), timestamp=1.0)

    map_points = [
        SimpleNamespace(position=np.array([0.0, 0.0, 2.0])),
        SimpleNamespace(position=np.array([10.0, 0.0, 2.0])),
        SimpleNamespace(position=np.array([0.0, 0.0, -1.0])),
    ]

    visible, projs, depths = are_map_points_visible_in_frame(frame, map_points)

    assert visible.tolist() == [True, False, False]
    np.testing.assert_allclose(projs[0], np.array([320.0, 240.0]))
    assert depths[0] == 2.0


def test_match_frames_self_image():
    setup_feature_tracker()
    cam = make_camera()
    image = make_image()
    depth = np.full((480, 640), 10000, dtype=np.uint16)

    f1 = Frame(camera=cam, img=image, depth_img=depth, timestamp=1.0)
    f2 = Frame(camera=cam, img=image, depth_img=depth, timestamp=2.0)

    result = match_frames(f1, f2, ratio_test=0.9)

    assert len(result.idxs1) > 20
    assert len(result.idxs1) == len(result.idxs2)
