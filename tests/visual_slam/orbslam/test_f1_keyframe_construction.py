"""F1b — C++ KeyFrame construction parity from a real Python Frame.

Proves the construction recipe (build_cpp_keyframe_from_frame) before it is wired
into the live KeyFrame class: a C++ KeyFrame built from a real RGB-D Frame must
match the source Frame on identity, features, pose, and map-point associations,
and expose a working covisibility graph.
"""
import sys, os
sys.path.insert(0, "/home/kaushik/slam_ws")
os.chdir("/home/kaushik/slam_ws")

import cv2
import numpy as np
import pytest

from visual_slam.orbslam.local_features import create_orb2_feature_tracker
from visual_slam.orbslam.slam import Frame, FeatureTrackerShared, PinholeCamera, SensorType
from visual_slam.orbslam.slam.keyframe import build_cpp_keyframe_from_frame, _CppKeyFrameBase

pytestmark = pytest.mark.skipif(_CppKeyFrameBase is None,
                                reason="cpp_slam_core.KeyFrame unavailable")


def _make_camera():
    return PinholeCamera.from_params(
        width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0,
        sensor_type=SensorType.RGBD, baseline=0.08, depth_map_factor=5000.0, th_depth=40.0)


def _make_image():
    image = np.zeros((480, 640), dtype=np.uint8)
    for x in range(80, 600, 80):
        cv2.circle(image, (x, 240), 18, 255, 2)
    for y in range(80, 420, 80):
        cv2.line(image, (60, y), (580, y), 180, 2)
    return image


def _make_real_frame(timestamp=1.0):
    FeatureTrackerShared.reset()
    FeatureTrackerShared.set_feature_tracker(create_orb2_feature_tracker())
    cam = _make_camera()
    depth = np.full((480, 640), 10000, dtype=np.uint16)  # 2 m at factor 5000
    return Frame(camera=cam, img=_make_image(), depth_img=depth, timestamp=timestamp)


def test_construction_feature_parity():
    frame = _make_real_frame()
    n = len(frame.kps)
    assert n > 20, "need a non-trivial frame"

    kf = build_cpp_keyframe_from_frame(frame, kid=7)

    assert kf.kid == 7
    assert kf.kpsu.shape == (n, 2)
    assert len(kf.octaves) == n
    # init_feature_arrays keeps only kpsu; the cv2.KeyPoint list must be retained too
    # (consumers read kf.kps; an empty kps breaks ensure_frame_feature_arrays).
    assert len(kf.kps) == n
    # octaves must match the source frame exactly (deterministic feature mapping)
    np.testing.assert_array_equal(np.asarray(kf.octaves, dtype=np.int32),
                                  np.asarray(frame.octaves, dtype=np.int32))
    # descriptors preserved
    assert kf.des.shape == (n, 32)
    np.testing.assert_array_equal(np.asarray(kf.des, dtype=np.uint8),
                                  np.asarray(frame.des, dtype=np.uint8))


def test_construction_projection_family_present():
    """The projection family must exist on the C++ KeyFrame. Its absence was swallowed
    by fuse_map_points' bare `except`, silently disabling map-point fusion (-> 2x
    duplicate points, under-culling). are_visible must project + return correct shapes."""
    import cpp_slam_core
    frame = _make_real_frame()
    kf = build_cpp_keyframe_from_frame(frame, kid=13)
    for m in ("are_visible", "are_in_image", "project_points", "project_point",
              "project_map_points", "transform_points", "transform_point", "unproject_points_3d"):
        assert hasattr(kf, m), f"C++ KeyFrame missing {m} (fuse/triangulation needs it)"
    mps = [cpp_slam_core.MapPoint([0.05 * i, 0.0, 2.0]) for i in range(6)]
    vis, projs, depths, dists = kf.are_visible(mps, kf.camera.is_stereo())
    assert len(vis) == len(mps) and projs.shape[0] == len(mps)


def test_construction_timestamp_preserved():
    """timestamp must transfer (C++ ctor doesn't set it) — a 0 timestamp disables
    keyframe culling (the dt-to-parent guard sees dt=0 < max_time_dist)."""
    frame = _make_real_frame(timestamp=123.456)
    kf = build_cpp_keyframe_from_frame(frame, kid=12)
    assert abs(float(kf.timestamp) - 123.456) < 1e-6


def test_construction_kps_ur_stereo_preserved():
    """Stereo right-coords (uRs) must transfer — a stale all-(-1) frame.kps_ur must
    NOT win, else stereo observations score weight 1 -> num_tracked_points collapses."""
    frame = _make_real_frame()
    n_valid = int(np.sum(np.asarray(frame.uRs) >= 0))
    assert n_valid > 20, "RGB-D frame should have many valid stereo coords"
    kf = build_cpp_keyframe_from_frame(frame, kid=11)
    np.testing.assert_allclose(np.asarray(kf.kps_ur, dtype=np.float32),
                               np.asarray(frame.uRs, dtype=np.float32), atol=1e-3)
    assert int(np.sum(np.asarray(kf.kps_ur) >= 0)) == n_valid


def test_construction_kpsu_matches_frame():
    frame = _make_real_frame()
    kf = build_cpp_keyframe_from_frame(frame, kid=8)
    # frame.kpsu is a cv2.KeyPoint list; compare (x,y) against the C++ (N,2) array
    frame_xy = np.array([kp.pt for kp in frame.kpsu], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(kf.kpsu, dtype=np.float32), frame_xy, atol=1e-4)


def test_construction_pose_parity():
    frame = _make_real_frame()
    T = np.eye(4); T[0, 3] = 1.5; T[1, 3] = -0.3
    frame.update_pose(T)
    kf = build_cpp_keyframe_from_frame(frame, kid=9)
    np.testing.assert_allclose(np.asarray(kf.Tcw()), np.asarray(frame.Tcw()), atol=1e-9)


def test_construction_point_associations():
    frame = _make_real_frame()
    import cpp_slam_core
    # attach a few MapPoints to the source frame
    attached = {}
    for idx in (0, 3, 5):
        if idx < len(frame.points):
            mp = cpp_slam_core.MapPoint([float(idx), 0.0, 2.0])
            frame.points[idx] = mp
            attached[idx] = mp

    kf = build_cpp_keyframe_from_frame(frame, kid=10)
    for idx, mp in attached.items():
        assert kf.get_point_match(idx) is mp


def test_constructed_kf_has_working_covisibility():
    f1 = _make_real_frame(timestamp=1.0)
    f2 = _make_real_frame(timestamp=2.0)
    kf1 = build_cpp_keyframe_from_frame(f1, kid=20)
    kf2 = build_cpp_keyframe_from_frame(f2, kid=21)
    kf1.add_connection(kf2, 33)
    assert kf2 in kf1.get_connected_keyframes()
    assert kf1.get_weight(kf2) == 33
    # hashable / equality-by-kid (consumers put KFs in sets)
    assert kf1 in {kf1, kf2}
