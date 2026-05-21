import cv2
import g2o
import numpy as np

from visual_slam.orbslam.slam import Frame, MapPoint, PinholeCamera, SensorType


def make_camera():
    return PinholeCamera.from_params(
        width=640,
        height=480,
        fx=517.3,
        fy=516.5,
        cx=318.6,
        cy=255.3,
        sensor_type=SensorType.RGBD,
        baseline=0.08,
        depth_map_factor=5000.0,
    )


def make_frame(num_kps=5):
    frame = Frame(
        camera=make_camera(),
        img=None,
        depth_img=None,
        pose=g2o.Isometry3d(np.eye(4, dtype=np.float64)),
        timestamp=0.0,
        img_id=0,
    )

    frame.kps = [cv2.KeyPoint(float(20 + i), float(30 + i), 1.0) for i in range(num_kps)]
    frame.kpsu = list(frame.kps)
    frame.des = np.zeros((num_kps, 32), dtype=np.uint8)
    frame.depths = np.ones(num_kps, dtype=np.float32)
    frame.uRs = np.full(num_kps, -1.0, dtype=np.float32)
    frame.kps_ur = frame.uRs
    frame.points = [None] * num_kps
    frame.outliers = np.zeros(num_kps, dtype=bool)
    frame.idxs = np.arange(num_kps, dtype=np.int32)
    return frame


def test_frame_has_pyslam_matched_point_api_methods():
    assert hasattr(Frame, "get_matched_good_points_and_idxs")
    assert hasattr(Frame, "get_matched_good_points")
    assert hasattr(Frame, "get_matched_good_points_idxs")
    assert hasattr(Frame, "num_tracked_points")
    assert hasattr(Frame, "check_replaced_map_points")


def test_frame_matched_good_points_keep_original_keypoint_indices():
    frame = make_frame(5)

    mp0 = MapPoint(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    mp3 = MapPoint(np.array([0.3, 0.0, 1.0], dtype=np.float64))

    frame.points[0] = mp0
    frame.points[3] = mp3
    frame.outliers[3] = True

    pairs = frame.get_matched_good_points_and_idxs()

    assert pairs == [(mp0, 0)]
    assert frame.get_matched_good_points() == [mp0]
    assert list(frame.get_matched_good_points_idxs()) == [0]
    assert frame.num_tracked_points(min_num_observations=0) == 1


def test_frame_check_replaced_map_points_updates_frame_association():
    frame = make_frame(4)

    old_mp = MapPoint(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    new_mp = MapPoint(np.array([0.1, 0.0, 1.0], dtype=np.float64))

    frame.points[2] = old_mp
    old_mp.replacement = new_mp

    replaced = frame.check_replaced_map_points()

    assert replaced == 1
    assert frame.points[2] is new_mp
