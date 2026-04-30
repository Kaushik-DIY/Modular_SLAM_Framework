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
    Parameters,
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
    for x in range(60, 620, 40):
        cv2.circle(image, (x, 240), 10, 255, 2)
    for y in range(60, 460, 40):
        cv2.line(image, (40, y), (600, y), 180, 2)
    return image


def make_frame(timestamp=1.0, tx=0.0):
    cam = make_camera()
    image = make_image()
    depth = np.full((480, 640), 10000, dtype=np.uint16)

    Tcw = np.eye(4, dtype=np.float64)
    Tcw[:3, 3] = np.array([tx, 0.0, 0.0], dtype=np.float64)

    return Frame(
        camera=cam,
        img=image,
        depth_img=depth,
        pose=g2o.Isometry3d(Tcw),
        timestamp=timestamp,
    )


def valid_indices(frame, n):
    idxs = np.flatnonzero(frame.depths > 0)
    assert len(idxs) >= n
    return [int(i) for i in idxs[:n]]


def test_map_add_remove_frames_keyframes_points_and_ids():
    setup_tracker()

    slam_map = Map()

    f0 = make_frame(timestamp=1.0)
    f1 = make_frame(timestamp=2.0)

    assert slam_map.add_frame(f0) == f0.id
    assert slam_map.add_frame(f1) == f1.id
    assert slam_map.num_frames() == 2
    assert slam_map.get_frame(0) is f0
    assert slam_map.get_frame(1) is f1

    kf0 = KeyFrame(f0, kid=100)
    kf1 = KeyFrame(f1, kid=101)

    assert slam_map.add_keyframe(kf0) == 0
    assert slam_map.add_keyframe(kf1) == 1
    assert kf0.kid == 0
    assert kf1.kid == 1
    assert kf0.map is slam_map
    assert kf1.map is slam_map
    assert slam_map.num_keyframes() == 2
    assert slam_map.get_first_keyframe() is kf0
    assert slam_map.get_last_keyframe() is kf1
    assert slam_map.get_keyframe_by_frame_id(kf0.id) is kf0

    p0 = MapPoint(np.array([0.0, 0.0, 2.0]))
    p1 = MapPoint(np.array([0.1, 0.0, 2.0]))

    assert slam_map.add_point(p0) == 0
    assert slam_map.add_point(p1) == 1
    assert p0.id == 0
    assert p1.id == 1
    assert p0.map is slam_map
    assert slam_map.num_points() == 2

    slam_map.remove_point(p0)
    assert slam_map.num_points() == 1
    assert p0.map is None

    slam_map.remove_keyframe(kf1)
    assert slam_map.num_keyframes() == 1
    assert kf1.map is None


def test_map_reset_and_delete():
    setup_tracker()

    slam_map = Map()
    f0 = make_frame(timestamp=1.0)
    kf0 = KeyFrame(f0, kid=0)
    p0 = MapPoint(np.array([0.0, 0.0, 2.0]))

    slam_map.add_frame(f0)
    slam_map.add_keyframe(kf0)
    slam_map.add_point(p0)

    assert slam_map.num_frames() == 1
    assert slam_map.num_keyframes() == 1
    assert slam_map.num_points() == 1

    slam_map.reset()

    assert slam_map.num_frames() == 0
    assert slam_map.num_keyframes() == 0
    assert slam_map.num_points() == 0
    assert slam_map.max_frame_id == 0
    assert slam_map.max_keyframe_id == 0
    assert slam_map.max_point_id == 0


def test_map_last_keyframes_window():
    setup_tracker()

    slam_map = Map()

    keyframes = []
    for i in range(5):
        f = make_frame(timestamp=float(i))
        kf = KeyFrame(f, kid=i)
        slam_map.add_keyframe(kf)
        keyframes.append(kf)

    last_three = slam_map.get_last_keyframes(3).to_list()

    assert last_three == keyframes[-3:]


def test_local_covisibility_map_collects_keyframes_and_points():
    setup_tracker()

    slam_map = Map()

    f0 = make_frame(timestamp=1.0)
    f1 = make_frame(timestamp=2.0)
    f2 = make_frame(timestamp=3.0)

    kf0 = KeyFrame(f0, kid=0)
    kf1 = KeyFrame(f1, kid=1)
    kf2 = KeyFrame(f2, kid=2)

    slam_map.add_keyframe(kf0)
    slam_map.add_keyframe(kf1)
    slam_map.add_keyframe(kf2)

    n_shared = Parameters.kMinNumOfCovisiblePointsForCreatingConnection + 1
    idxs0 = valid_indices(kf0, n_shared)
    idxs1 = valid_indices(kf1, n_shared)
    idxs2 = valid_indices(kf2, n_shared)

    points = []
    for j, (i0, i1, i2) in enumerate(zip(idxs0, idxs1, idxs2)):
        p = MapPoint(np.array([0.01 * j, 0.0, 2.0]), keyframe=kf0, idx=i0)
        p.add_observation(kf1, i1)

        if j % 2 == 0:
            p.add_observation(kf2, i2)

        p.update_info()
        slam_map.add_point(p)
        points.append(p)

    kf0.update_connections()
    kf1.update_connections()
    kf2.update_connections()

    slam_map.update_local_map(kf0)

    local_kfs = slam_map.get_local_keyframes().to_list()
    local_points = slam_map.get_local_points().to_list()

    assert kf0 in local_kfs
    assert kf1 in local_kfs
    assert len(local_points) > 0
    assert all(p in points for p in local_points)


def test_map_remove_point_called_from_mappoint_set_bad_does_not_recurse():
    setup_tracker()

    slam_map = Map()
    p0 = MapPoint(np.array([0.0, 0.0, 2.0]))

    slam_map.add_point(p0)
    assert slam_map.num_points() == 1

    p0.set_bad()

    assert p0.is_bad()
    assert slam_map.num_points() == 0
