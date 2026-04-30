import cv2
import g2o
import numpy as np

from visual_slam.orbslam.local_features import create_orb2_feature_tracker
from visual_slam.orbslam.slam import (
    FeatureTrackerShared,
    Frame,
    MapPoint,
    PinholeCamera,
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


def make_image():
    image = np.zeros((480, 640), dtype=np.uint8)
    for x in range(80, 600, 80):
        cv2.circle(image, (x, 240), 18, 255, 2)
    for y in range(80, 420, 80):
        cv2.line(image, (60, y), (580, y), 180, 2)
    return image


def make_frame(timestamp=1.0, is_keyframe=False):
    cam = make_camera()
    image = make_image()
    depth = np.full((480, 640), 10000, dtype=np.uint16)
    frame = Frame(
        camera=cam,
        img=image,
        depth_img=depth,
        pose=g2o.Isometry3d(np.eye(4, dtype=np.float64)),
        timestamp=timestamp,
    )
    frame.is_keyframe = is_keyframe
    frame.kid = int(timestamp * 10)
    return frame


def first_valid_depth_idx(frame):
    valid = np.flatnonzero(frame.depths > 0)
    assert len(valid) > 0
    return int(valid[0])


def test_map_point_observation_lifecycle_and_descriptor():
    setup_tracker()
    kf = make_frame(is_keyframe=True)
    idx = first_valid_depth_idx(kf)

    mp = MapPoint(np.array([0.0, 0.0, 2.0]), keyframe=kf, idx=idx)

    assert mp.id >= 0
    assert mp.is_in_keyframe(kf)
    assert mp.get_observation_idx(kf) == idx
    assert kf.get_point_match(idx) is mp
    assert mp.num_observations() in (1, 2)

    descriptor = mp.get_descriptor()
    assert descriptor is not None
    assert descriptor.dtype == np.uint8
    assert descriptor.shape == (32,)

    assert not mp.is_bad()
    mp.remove_observation(kf, idx)
    assert mp.is_bad()
    assert kf.get_point_match(idx) is None


def test_map_point_frame_views():
    setup_tracker()
    frame = make_frame(is_keyframe=False)
    idx = first_valid_depth_idx(frame)

    mp = MapPoint(np.array([0.0, 0.0, 2.0]))

    assert mp.add_frame_view(frame, idx)
    assert mp.is_in_frame(frame)
    assert mp.get_frame_view_idx(frame) == idx
    assert frame.get_point_match(idx) is mp

    mp.remove_frame_view(frame, idx)
    assert not mp.is_in_frame(frame)
    assert frame.get_point_match(idx) is None


def test_map_point_update_normal_depth_and_scale_prediction():
    setup_tracker()
    kf = make_frame(is_keyframe=True)
    idx = first_valid_depth_idx(kf)

    mp = MapPoint(np.array([0.0, 0.0, 2.0]), keyframe=kf, idx=idx)

    mp.update_normal_and_depth()

    np.testing.assert_allclose(mp.get_normal(), np.array([0.0, 0.0, 1.0]), atol=1e-6)
    assert mp.max_distance > 0.0
    assert mp.min_distance > 0.0
    assert mp.min_distance <= mp.max_distance

    scale = mp.predict_scale(dist=2.0, frame_or_keyframe=kf)
    assert 0 <= scale < FeatureTrackerShared.feature_manager.num_levels


def test_map_point_found_visible_ratio_and_replacement():
    setup_tracker()

    kf1 = make_frame(timestamp=1.0, is_keyframe=True)
    kf2 = make_frame(timestamp=2.0, is_keyframe=True)
    idx1 = first_valid_depth_idx(kf1)
    idx2 = first_valid_depth_idx(kf2)

    mp1 = MapPoint(np.array([0.0, 0.0, 2.0]), keyframe=kf1, idx=idx1)
    mp2 = MapPoint(np.array([0.1, 0.0, 2.0]), keyframe=kf2, idx=idx2)

    mp1.increase_visible(4)
    mp1.increase_found(2)

    assert abs(mp1.get_found_ratio() - 3.0 / 5.0) < 1e-12

    mp1.replace_with(mp2)

    assert mp1.is_bad()
    assert mp1.get_replaced() is mp2
    assert mp2.num_times_visible >= 5
    assert mp2.num_times_found >= 3
