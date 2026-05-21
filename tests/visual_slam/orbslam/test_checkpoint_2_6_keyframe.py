import cv2
import g2o
import numpy as np

from visual_slam.orbslam.local_features import create_orb2_feature_tracker
from visual_slam.orbslam.slam import (
    FeatureTrackerShared,
    Frame,
    KeyFrame,
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


def test_keyframe_constructs_from_frame_without_reextracting():
    setup_tracker()
    frame = make_frame(timestamp=1.0)

    kf = KeyFrame(frame, kid=0)

    assert kf.is_keyframe is True
    assert kf.kid == 0
    assert kf.id == frame.id
    assert kf.camera is frame.camera
    assert kf.kps is frame.kps
    assert kf.des is frame.des
    assert kf.depths is frame.depths
    assert kf.uRs is frame.uRs
    assert len(kf.points) == len(frame.points)
    assert kf.is_bad() is False


def test_keyframe_graph_children_loop_edges_and_connections():
    setup_tracker()
    kf0 = KeyFrame(make_frame(timestamp=1.0), kid=0)
    kf1 = KeyFrame(make_frame(timestamp=2.0), kid=1)

    kf1.set_parent(kf0)
    assert kf1.get_parent() is kf0
    assert kf0.has_child(kf1)

    kf0.add_loop_edge(kf1)
    assert kf1 in kf0.get_loop_edges()
    assert kf0.not_to_erase is True

    kf0.add_connection(kf1, 5)
    assert kf1 in kf0.get_connected_keyframes()
    assert kf0.get_weight(kf1) == 5
    assert kf0.get_best_covisible_keyframes(1) == [kf1]

    kf0.erase_connection(kf1)
    assert kf1 not in kf0.get_connected_keyframes()


def test_keyframe_init_observations_and_update_connections():
    setup_tracker()
    frame0 = make_frame(timestamp=1.0)
    frame1 = make_frame(timestamp=2.0)

    kf0 = KeyFrame(frame0, kid=0)
    kf1 = KeyFrame(frame1, kid=1)

    n_shared = Parameters.kMinNumOfCovisiblePointsForCreatingConnection + 1
    idxs0 = valid_indices(kf0, n_shared)
    idxs1 = valid_indices(kf1, n_shared)

    points = []
    for j, (i0, i1) in enumerate(zip(idxs0, idxs1)):
        p = MapPoint(np.array([0.01 * j, 0.0, 2.0]), keyframe=kf0, idx=i0)
        p.add_observation(kf1, i1)
        p.update_info()
        points.append(p)

    # Make sure keyframes point arrays contain the observations.
    for p, i0, i1 in zip(points, idxs0, idxs1):
        assert kf0.get_point_match(i0) is p
        assert kf1.get_point_match(i1) is p

    kf0.update_connections()
    kf1.update_connections()

    assert kf1 in kf0.get_connected_keyframes()
    assert kf0 in kf1.get_connected_keyframes()
    assert kf0.get_weight(kf1) == n_shared
    assert kf1.get_weight(kf0) == n_shared

    assert kf1.get_parent() is kf0


def test_keyframe_set_bad_removes_observations_for_non_root():
    setup_tracker()
    kf0 = KeyFrame(make_frame(timestamp=1.0), kid=0)
    kf1 = KeyFrame(make_frame(timestamp=2.0), kid=1)

    kf1.set_parent(kf0)

    idxs0 = valid_indices(kf0, 3)
    idxs1 = valid_indices(kf1, 3)

    points = []
    for j, (i0, i1) in enumerate(zip(idxs0, idxs1)):
        p = MapPoint(np.array([0.01 * j, 0.0, 2.0]), keyframe=kf0, idx=i0)
        p.add_observation(kf1, i1)
        points.append(p)

    kf0.update_connections()
    kf1.update_connections()

    assert kf0.has_child(kf1)
    kf1.set_bad()

    assert kf1.is_bad()
    assert not kf0.has_child(kf1)
    assert all(p.get_observation_idx(kf1) == -1 for p in points)
