"""F1b (2/n) — the LIVE KeyFrame wired on the C++ base (USE_CPP_KEYFRAME).

The flag selects the KeyFrame base class at import time, so the C++-path checks run
in a SUBPROCESS with SLAM_USE_CPP_KEYFRAME=1. Verifies: the live KeyFrame is both a
KeyFrame and a C++ KeyFrame (isinstance preserved), construction from a real Frame
populates features/pose, and the delegated graph/point methods + the Python-side Tcp
on set_bad all work.
"""
import os
import subprocess
import sys

import pytest

try:
    import cpp_slam_core  # noqa: F401
    _HAVE_CPP = hasattr(cpp_slam_core, "KeyFrame")
except Exception:
    _HAVE_CPP = False

pytestmark = pytest.mark.skipif(not _HAVE_CPP, reason="cpp_slam_core.KeyFrame unavailable")


def _smoke():
    import cv2
    import numpy as np
    from visual_slam.orbslam.local_features import create_orb2_feature_tracker
    from visual_slam.orbslam.slam import Frame, FeatureTrackerShared, PinholeCamera, SensorType
    from visual_slam.orbslam.slam.keyframe import KeyFrame, _USE_CPP_KF, _CppKeyFrameBase

    assert _USE_CPP_KF, "flag not picked up at import"

    def make_frame(ts):
        FeatureTrackerShared.reset()
        FeatureTrackerShared.set_feature_tracker(create_orb2_feature_tracker())
        cam = PinholeCamera.from_params(width=640, height=480, fx=500., fy=500., cx=320., cy=240.,
                                        sensor_type=SensorType.RGBD, baseline=0.08,
                                        depth_map_factor=5000., th_depth=40.)
        img = np.zeros((480, 640), np.uint8)
        for x in range(80, 600, 80):
            cv2.circle(img, (x, 240), 18, 255, 2)
        for y in range(80, 420, 80):
            cv2.line(img, (60, y), (580, y), 180, 2)
        return Frame(camera=cam, img=img, depth_img=np.full((480, 640), 10000, np.uint16), timestamp=ts)

    f1, f2 = make_frame(1.0), make_frame(2.0)
    kf1, kf2 = KeyFrame(f1, kid=5), KeyFrame(f2, kid=6)

    # single class with C++ base -> isinstance must hold for BOTH
    assert isinstance(kf1, KeyFrame) and isinstance(kf1, _CppKeyFrameBase)
    assert kf1.kid == 5
    assert kf1.kpsu.shape == (len(f1.kps), 2)
    assert kf1.uRs.shape[0] == len(f1.kps)          # readonly C++ property, readable
    assert kf1.is_bad() is False
    assert kf1.num_tracked_points() == 0            # no map points attached
    assert len(kf1.get_matched_good_points()) == 0

    kf1.update_connections()                        # delegated to C++, no crash
    kf1.add_connection(kf2, 17)
    assert kf1.get_weight(kf2) == 17

    # Projection family must be present + functional on the LIVE C++ KeyFrame —
    # its absence was silently swallowed by fuse_map_points, disabling fusion.
    import cpp_slam_core
    for m in ("are_visible", "are_in_image", "project_points", "project_point",
              "project_map_points", "transform_points", "transform_point", "unproject_points_3d"):
        assert hasattr(kf1, m), f"live C++ KeyFrame missing {m}"
    mps = [cpp_slam_core.MapPoint([0.05 * i, 0.0, 2.0]) for i in range(6)]
    vis, projs, depths, dists = kf1.are_visible(mps, kf1.camera.is_stereo())
    assert len(vis) == len(mps) and projs.shape[0] == len(mps)

    # spanning tree + set_bad must populate Tcp (Python-side; C++ set_bad stubs it)
    kf2.set_parent(kf1)
    T = np.eye(4); T[0, 3] = 2.0
    kf2.update_pose(T)
    kf2.set_bad()
    assert kf2.is_bad() is True
    assert np.asarray(kf2.Tcp()).shape == (4, 4)

    assert kf1 in {kf1, kf2}                         # hash-by-kid
    print("F1B_CPP_WIRED_OK")


def test_live_keyframe_on_cpp_base():
    """Run the C++-path smoke in a subprocess with the import-time flag set."""
    env = dict(os.environ, SLAM_USE_CPP_KEYFRAME="1")
    proc = subprocess.run(
        [sys.executable, __file__], cwd="/home/kaushik/slam_ws",
        env=env, capture_output=True, text=True, timeout=300)
    assert "F1B_CPP_WIRED_OK" in proc.stdout, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"


if __name__ == "__main__":
    sys.path.insert(0, "/home/kaushik/slam_ws")
    _smoke()
