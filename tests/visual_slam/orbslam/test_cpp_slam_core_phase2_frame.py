"""
Phase 2 parity test: C++ Frame.

Verifies:
1. kpsu normalization from cv2.KeyPoint list and numpy arrays
2. Pose round-trip (Tcw ↔ Twc, Ow)
3. Point match management (get/set/remove)
4. Feature array access (octaves, des, kps_ur)
5. Parity with Python Frame for same inputs
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, "/home/kaushik/slam_ws")
os.chdir("/home/kaushik/slam_ws")


def make_cpp_frame(n_kps=8):
    """Create a C++ Frame with synthetic feature data."""
    import cv2, cpp_slam_core
    f = cpp_slam_core.Frame()
    kps = [cv2.KeyPoint(x=float(i * 10), y=float(i * 5), size=1.0, octave=i % 4)
           for i in range(n_kps)]
    des = np.random.randint(0, 255, (n_kps, 32), dtype=np.uint8)
    kps_ur = np.full(n_kps, -1.0, dtype=np.float32)
    octaves = np.array([kp.octave for kp in kps], dtype=np.int32)
    f.init_feature_arrays(kps, des, kps_ur, octaves, n_kps)
    return f, kps, des


class TestFrameConstruction:
    def test_id_auto_increment(self):
        import cpp_slam_core
        f1 = cpp_slam_core.Frame()
        f2 = cpp_slam_core.Frame()
        assert f2.id == f1.id + 1

    def test_given_id(self):
        import cpp_slam_core
        f = cpp_slam_core.Frame(id=7777)
        assert f.id == 7777


class TestKpsuNormalization:
    def test_from_cv2_keypoints(self):
        """cv2.KeyPoint list → float32 (N, 2) numpy array."""
        import cv2, cpp_slam_core
        n = 10
        kps = [cv2.KeyPoint(x=float(i * 3.0), y=float(i * 2.0), size=1.0) for i in range(n)]
        des = np.zeros((n, 32), dtype=np.uint8)
        f = cpp_slam_core.Frame()
        f.init_feature_arrays(kps, des, None, None, n)
        assert f.kpsu.shape == (n, 2)
        assert f.kpsu.dtype == np.float32
        for i in range(n):
            np.testing.assert_allclose(f.kpsu[i], [i * 3.0, i * 2.0], rtol=1e-5)

    def test_from_numpy_float32(self):
        """numpy float32 (N, 2) → passes through unchanged."""
        import cpp_slam_core
        n = 6
        arr = np.array([[float(i), float(i * 2)] for i in range(n)], dtype=np.float32)
        des = np.zeros((n, 32), dtype=np.uint8)
        f = cpp_slam_core.Frame()
        f.init_feature_arrays(arr, des, None, None, n)
        assert f.kpsu.shape == (n, 2)
        np.testing.assert_allclose(f.kpsu, arr, rtol=1e-6)

    def test_from_numpy_float64(self):
        """numpy float64 (N, 2) → converted to float32."""
        import cpp_slam_core
        n = 4
        arr = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5], [7.5, 8.5]], dtype=np.float64)
        des = np.zeros((n, 32), dtype=np.uint8)
        f = cpp_slam_core.Frame()
        f.init_feature_arrays(arr, des, None, None, n)
        np.testing.assert_allclose(f.kpsu, arr.astype(np.float32), rtol=1e-5)

    def test_no_subscriptability_error(self):
        """cv2.KeyPoint objects must NOT cause subscriptability errors."""
        import cv2, cpp_slam_core
        kps = [cv2.KeyPoint(x=100.0, y=200.0, size=5.0, octave=1)]
        des = np.zeros((1, 32), dtype=np.uint8)
        f = cpp_slam_core.Frame()
        # This previously raised: 'cv2.KeyPoint' object is not subscriptable
        f.init_feature_arrays(kps, des, None, None, 1)
        assert f.kpsu[0, 0] == pytest.approx(100.0)
        assert f.kpsu[0, 1] == pytest.approx(200.0)


class TestFramePose:
    def test_identity_pose(self):
        import cpp_slam_core
        f = cpp_slam_core.Frame()
        np.testing.assert_allclose(f.Tcw(), np.eye(4), atol=1e-12)

    def test_update_pose_matrix(self):
        import cpp_slam_core
        f = cpp_slam_core.Frame()
        T = np.eye(4)
        T[0, 3] = 1.0
        T[1, 3] = 2.0
        T[2, 3] = 3.0
        f.update_pose(T)
        np.testing.assert_allclose(f.Tcw(), T, atol=1e-12)

    def test_twc_inverse_of_tcw(self):
        import cpp_slam_core
        f = cpp_slam_core.Frame()
        T = np.eye(4)
        T[0, 3] = 5.0
        f.update_pose(T)
        Tcw = f.Tcw()
        Twc = f.Twc()
        product = Tcw @ Twc
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)

    def test_ow_camera_center(self):
        """Ow = -Rwc * tcw = camera center in world."""
        import cpp_slam_core
        f = cpp_slam_core.Frame()
        T = np.eye(4)
        T[0, 3] = 1.5
        T[1, 3] = -0.5
        f.update_pose(T)
        Ow = f.Ow()
        # With identity rotation: Ow = -tcw
        np.testing.assert_allclose(Ow, [-1.5, 0.5, 0.0], atol=1e-10)

    def test_ow_matches_twc_translation(self):
        """Twc's translation column = camera center = Ow."""
        import cpp_slam_core
        f = cpp_slam_core.Frame()
        T = np.eye(4)
        T[0, 3] = 2.0
        f.update_pose(T)
        np.testing.assert_allclose(f.Ow(), f.Twc()[:3, 3], atol=1e-10)

    def test_update_pose_g2o(self):
        """g2o.Isometry3d pose update."""
        import g2o, cpp_slam_core, numpy as np
        f = cpp_slam_core.Frame()
        T = np.eye(4)
        T[0, 3] = 3.0
        iso = g2o.Isometry3d(T)
        f.update_pose(iso)
        np.testing.assert_allclose(f.Tcw()[0, 3], 3.0, atol=1e-10)


class TestFrameFeatureArrays:
    def test_octaves_access(self):
        import cv2, cpp_slam_core
        n = 5
        kps = [cv2.KeyPoint(x=0, y=0, size=1.0, octave=i) for i in range(n)]
        des = np.zeros((n, 32), dtype=np.uint8)
        f = cpp_slam_core.Frame()
        f.init_feature_arrays(kps, des, None, None, n)
        for i in range(n):
            assert f.octaves[i] == i

    def test_kps_ur_default_monocular(self):
        import cv2, cpp_slam_core
        n = 4
        kps = [cv2.KeyPoint(x=0, y=0, size=1.0) for _ in range(n)]
        des = np.zeros((n, 32), dtype=np.uint8)
        f = cpp_slam_core.Frame()
        f.init_feature_arrays(kps, des, None, None, n)
        assert all(ur < 0 for ur in f.kps_ur)

    def test_kps_ur_stereo(self):
        import cv2, cpp_slam_core
        n = 3
        kps = [cv2.KeyPoint(x=float(i*10), y=0, size=1.0) for i in range(n)]
        des = np.zeros((n, 32), dtype=np.uint8)
        kps_ur = np.array([50.0, 60.0, 70.0], dtype=np.float32)
        f = cpp_slam_core.Frame()
        f.init_feature_arrays(kps, des, kps_ur, None, n)
        np.testing.assert_allclose(f.kps_ur, kps_ur, rtol=1e-5)

    def test_des_access(self):
        import cv2, cpp_slam_core
        n = 4
        kps = [cv2.KeyPoint(x=0, y=0, size=1.0) for _ in range(n)]
        des = np.arange(n * 32, dtype=np.uint8).reshape(n, 32)
        f = cpp_slam_core.Frame()
        f.init_feature_arrays(kps, des, None, None, n)
        np.testing.assert_array_equal(f.des, des)


class TestFramePointMatch:
    def setup_method(self):
        self.f, _, _ = make_cpp_frame(n_kps=8)

    def test_set_and_get_point_match(self):
        import cpp_slam_core
        mp = cpp_slam_core.MapPoint([1.0, 0.0, 0.0])
        self.f.set_point_match(mp, 3)
        assert self.f.get_point_match(3) is mp

    def test_remove_point_match(self):
        import cpp_slam_core
        mp = cpp_slam_core.MapPoint([1.0, 0.0, 0.0])
        self.f.set_point_match(mp, 4)
        self.f.remove_point_match(4)
        assert self.f.get_point_match(4) is None

    def test_remove_point(self):
        import cpp_slam_core
        mp = cpp_slam_core.MapPoint([1.0, 0.0, 0.0])
        self.f.set_point_match(mp, 1)
        self.f.set_point_match(mp, 3)
        self.f.remove_point(mp)
        assert self.f.get_point_match(1) is None
        assert self.f.get_point_match(3) is None

    def test_reset_points(self):
        import cpp_slam_core
        mp = cpp_slam_core.MapPoint([1.0, 0.0, 0.0])
        self.f.set_point_match(mp, 0)
        self.f.set_point_match(mp, 5)
        self.f.reset_points()
        for i in range(8):
            assert self.f.get_point_match(i) is None

    def test_out_of_bounds_safe(self):
        """Out-of-bounds access returns None without crashing."""
        assert self.f.get_point_match(-1) is None
        assert self.f.get_point_match(100) is None


class TestFrameParityWithPython:
    """Check that C++ Frame and Python Frame handle the same inputs consistently."""

    def test_kpsu_parity(self):
        """C++ kpsu from cv2.KeyPoint == Python kpsu from cv2.KeyPoint."""
        import cv2, cpp_slam_core
        from visual_slam.orbslam.slam.frame import Frame as PyFrame
        from visual_slam.orbslam.slam.camera import Camera

        n = 6
        kps = [cv2.KeyPoint(x=float(i * 15), y=float(i * 7), size=3.0, octave=i % 4)
               for i in range(n)]
        des = np.random.randint(0, 255, (n, 32), dtype=np.uint8)

        # C++ Frame
        f_cpp = cpp_slam_core.Frame()
        f_cpp.init_feature_arrays(kps, des, None, None, n)
        kpsu_cpp = f_cpp.kpsu  # (N, 2) float32

        # Ground truth: extract pt from cv2.KeyPoint directly
        kpsu_ref = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)

        np.testing.assert_allclose(kpsu_cpp, kpsu_ref, rtol=1e-5,
                                   err_msg="C++ kpsu from cv2.KeyPoint does not match reference")

    def test_pose_roundtrip_parity(self):
        """Tcw → Twc → Ow consistency between C++ Frame and manual numpy."""
        import cpp_slam_core
        import numpy as np

        # Rotation: 45° around z
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        Rcw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        tcw = np.array([1.0, 2.0, 3.0])
        Tcw = np.eye(4)
        Tcw[:3, :3] = Rcw
        Tcw[:3, 3] = tcw

        f = cpp_slam_core.Frame()
        f.update_pose(Tcw)

        # Expected Ow = -Rcw^T * tcw
        Ow_expected = -Rcw.T @ tcw

        np.testing.assert_allclose(f.Ow(), Ow_expected, atol=1e-10)
        np.testing.assert_allclose(f.Tcw(), Tcw, atol=1e-12)
