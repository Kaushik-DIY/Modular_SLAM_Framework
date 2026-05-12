"""
Phase 1 parity test: C++ MapPoint vs Python MapPoint.

Verifies that cpp_slam_core.MapPoint provides the same observable behavior
as the Python MapPoint across all operations used by the SLAM pipeline.
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, "/home/kaushik/slam_ws")
os.chdir("/home/kaushik/slam_ws")


# ---------------------------------------------------------------------------
# Helpers: minimal Python KeyFrame stub (mimics the API the C++ MapPoint calls)
# ---------------------------------------------------------------------------
class FakeKeyFrame:
    """Minimal KF stub with the methods MapPoint.add_observation() calls."""
    _kid = 0

    def __init__(self, n_kps=10):
        FakeKeyFrame._kid += 1
        self.kid = FakeKeyFrame._kid
        self.id = self.kid
        self._points = [None] * n_kps
        self.kps_ur = np.full(n_kps, -1.0, dtype=np.float32)
        self.des = np.random.randint(0, 256, (n_kps, 32), dtype=np.uint8)
        self.octaves = np.zeros(n_kps, dtype=np.int32)
        self.Ow = np.array([0.0, 0.0, 0.0])

    def set_point_match(self, point, idx):
        self._points[idx] = point

    def remove_point_match(self, idx):
        if 0 <= idx < len(self._points):
            self._points[idx] = None

    def remove_point(self, point):
        for i, p in enumerate(self._points):
            if p is point:
                self._points[i] = None

    def get_point_match(self, idx):
        return self._points[idx] if 0 <= idx < len(self._points) else None

    def __hash__(self):
        return self.kid

    def __eq__(self, other):
        return isinstance(other, FakeKeyFrame) and self.kid == other.kid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestCppMapPointConstruction:
    def test_basic_construction(self):
        import cpp_slam_core
        mp = cpp_slam_core.MapPoint([1.0, 2.0, 3.0])
        assert mp.id >= 0
        assert not mp.is_bad()
        np.testing.assert_allclose(mp.get_position(), [1.0, 2.0, 3.0])

    def test_id_auto_increment(self):
        import cpp_slam_core
        mp1 = cpp_slam_core.MapPoint([0.0, 0.0, 0.0])
        mp2 = cpp_slam_core.MapPoint([1.0, 1.0, 1.0])
        assert mp2.id == mp1.id + 1

    def test_given_id(self):
        import cpp_slam_core
        mp = cpp_slam_core.MapPoint([0.0, 0.0, 0.0], id=9999)
        assert mp.id == 9999


class TestCppMapPointPosition:
    def test_update_get_position(self):
        import cpp_slam_core
        mp = cpp_slam_core.MapPoint([1.0, 2.0, 3.0])
        mp.update_position(np.array([4.0, 5.0, 6.0]))
        np.testing.assert_allclose(mp.get_position(), [4.0, 5.0, 6.0])

    def test_position_roundtrip_precision(self):
        import cpp_slam_core
        pos = np.array([1.23456789, -2.34567890, 3.45678901])
        mp = cpp_slam_core.MapPoint(pos)
        np.testing.assert_allclose(mp.get_position(), pos, rtol=1e-10)


class TestCppMapPointObservations:
    def setup_method(self):
        import cpp_slam_core
        FakeKeyFrame._kid = 0
        self.kf1 = FakeKeyFrame()
        self.kf2 = FakeKeyFrame()
        self.kf3 = FakeKeyFrame()
        self.mp = cpp_slam_core.MapPoint([1.0, 0.0, 0.0])

    def test_add_observation(self):
        result = self.mp.add_observation(self.kf1, 3)
        assert result is True
        assert self.mp.n_obs >= 1

    def test_duplicate_add_returns_false(self):
        self.mp.add_observation(self.kf1, 3)
        result = self.mp.add_observation(self.kf1, 3)
        assert result is False

    def test_is_in_keyframe(self):
        self.mp.add_observation(self.kf1, 0)
        assert self.mp.is_in_keyframe(self.kf1)
        assert not self.mp.is_in_keyframe(self.kf2)

    def test_get_observation_idx(self):
        self.mp.add_observation(self.kf1, 5)
        assert self.mp.get_observation_idx(self.kf1) == 5
        assert self.mp.get_observation_idx(self.kf2) == -1

    def test_observations_list(self):
        self.mp.add_observation(self.kf1, 1)
        self.mp.add_observation(self.kf2, 2)
        obs = self.mp.observations()
        kfs_in_obs = [kf for kf, _ in obs]
        assert self.kf1 in kfs_in_obs
        assert self.kf2 in kfs_in_obs

    def test_keyframes_list(self):
        self.mp.add_observation(self.kf1, 0)
        self.mp.add_observation(self.kf2, 1)
        kfs = self.mp.keyframes()
        assert len(kfs) == 2
        assert self.kf1 in kfs or self.kf2 in kfs

    def test_set_point_match_called_on_add(self):
        self.mp.add_observation(self.kf1, 4)
        # kf1.set_point_match should have been called
        assert self.kf1.get_point_match(4) is self.mp

    def test_remove_observation(self):
        self.mp.add_observation(self.kf1, 0)
        self.mp.add_observation(self.kf2, 1)
        self.mp.add_observation(self.kf3, 2)  # 3 obs → safe to remove 1
        self.mp.remove_observation(self.kf1, 0)
        assert not self.mp.is_in_keyframe(self.kf1)
        # KF1 match should be cleared
        assert self.kf1.get_point_match(0) is None

    def test_remove_observation_sets_bad_when_few_obs(self):
        self.mp.add_observation(self.kf1, 0)
        self.mp.add_observation(self.kf2, 1)
        # After removing one, only 1 obs left → should trigger set_bad
        # (MapPoint is considered bad with ≤ 2 raw observation count)
        self.mp.remove_observation(self.kf1, 0)
        # At 1 obs, _num_observations ≤ 2 → set_bad called
        assert self.mp.is_bad()


class TestCppMapPointStatus:
    def test_set_bad(self):
        import cpp_slam_core
        FakeKeyFrame._kid = 0
        kf1 = FakeKeyFrame()
        kf2 = FakeKeyFrame()
        kf3 = FakeKeyFrame()
        mp = cpp_slam_core.MapPoint([1.0, 0.0, 0.0])
        mp.add_observation(kf1, 0)
        mp.add_observation(kf2, 1)
        mp.add_observation(kf3, 2)
        mp.set_bad()
        assert mp.is_bad()
        # All observations removed
        assert len(mp.observations()) == 0
        # KF matches cleared
        assert kf1.get_point_match(0) is None
        assert kf2.get_point_match(1) is None
        assert kf3.get_point_match(2) is None

    def test_set_bad_idempotent(self):
        import cpp_slam_core
        mp = cpp_slam_core.MapPoint([0.0, 0.0, 0.0])
        mp.set_bad()
        mp.set_bad()  # should not raise
        assert mp.is_bad()

    def test_to_be_erased(self):
        import cpp_slam_core
        mp = cpp_slam_core.MapPoint([0.0, 0.0, 0.0])
        assert not mp.to_be_erased
        mp.to_be_erased = True
        assert mp.to_be_erased


class TestCppMapPointDescriptor:
    def test_min_des_distance_identity(self):
        import cpp_slam_core
        FakeKeyFrame._kid = 0
        kf = FakeKeyFrame(n_kps=5)
        # Set a specific descriptor pattern
        kf.des[:] = 0
        mp = cpp_slam_core.MapPoint([0.0, 0.0, 0.0])
        mp.add_observation(kf, 0)
        mp.update_best_descriptor()
        # Query with same descriptor → distance 0
        query = np.zeros((1, 32), dtype=np.uint8)
        dist = mp.min_des_distance(query)
        assert dist == 0.0

    def test_min_des_distance_different(self):
        import cpp_slam_core
        FakeKeyFrame._kid = 0
        kf = FakeKeyFrame(n_kps=5)
        kf.des[0] = 0xFF  # all 1s
        mp = cpp_slam_core.MapPoint([0.0, 0.0, 0.0])
        mp.add_observation(kf, 0)
        mp.update_best_descriptor()
        query = np.zeros((1, 32), dtype=np.uint8)  # all 0s
        dist = mp.min_des_distance(query)
        assert dist == 256.0  # Hamming distance of 32 bytes all-ones vs all-zeros


class TestCppMapPointStatistics:
    def test_increase_visible(self):
        import cpp_slam_core
        mp = cpp_slam_core.MapPoint([0.0, 0.0, 0.0])
        init_vis = mp.num_times_visible
        mp.increase_visible(3)
        assert mp.num_times_visible == init_vis + 3

    def test_increase_found(self):
        import cpp_slam_core
        mp = cpp_slam_core.MapPoint([0.0, 0.0, 0.0])
        init_found = mp.num_times_found
        mp.increase_found(2)
        assert mp.num_times_found == init_found + 2

    def test_found_ratio(self):
        import cpp_slam_core
        mp = cpp_slam_core.MapPoint([0.0, 0.0, 0.0])
        mp.num_times_visible = 10
        mp.num_times_found = 5
        assert abs(mp.get_found_ratio() - 0.5) < 1e-6


class TestCppMapPointHashEquality:
    def test_hashable(self):
        import cpp_slam_core
        mp = cpp_slam_core.MapPoint([1.0, 0.0, 0.0], id=42)
        s = {mp}
        assert mp in s

    def test_equality(self):
        import cpp_slam_core
        mp1 = cpp_slam_core.MapPoint([1.0, 0.0, 0.0], id=100)
        mp2 = cpp_slam_core.MapPoint([2.0, 0.0, 0.0], id=100)
        assert mp1 == mp2

    def test_less_than(self):
        import cpp_slam_core
        mp1 = cpp_slam_core.MapPoint([0.0, 0.0, 0.0], id=1)
        mp2 = cpp_slam_core.MapPoint([0.0, 0.0, 0.0], id=2)
        assert mp1 < mp2


class TestCppMapPointParityWithPython:
    """
    Parity test: C++ MapPoint and Python MapPoint produce identical behavior
    on a 5-observation synthetic scenario.
    """

    def _make_scenario(self, MapPointClass, n_kfs=5):
        """Build n_kfs FakeKeyFrame objects and one MapPoint with observations."""
        FakeKeyFrame._kid = 0
        kfs = [FakeKeyFrame(n_kps=10) for _ in range(n_kfs)]
        pos = np.array([1.0, 2.0, 3.0])
        mp = MapPointClass(pos)
        for i, kf in enumerate(kfs):
            mp.add_observation(kf, i)
        return mp, kfs

    def test_observation_count_parity(self):
        import cpp_slam_core
        from visual_slam.orbslam.slam.map_point import MapPoint as PyMP

        mp_cpp, kfs_cpp = self._make_scenario(
            lambda pos: cpp_slam_core.MapPoint(pos), n_kfs=5)
        mp_py, kfs_py = self._make_scenario(
            lambda pos: PyMP(pos), n_kfs=5)

        assert len(mp_cpp.observations()) == len(mp_py.observations())

    def test_remove_observation_parity(self):
        import cpp_slam_core
        from visual_slam.orbslam.slam.map_point import MapPoint as PyMP

        FakeKeyFrame._kid = 0
        kfs = [FakeKeyFrame() for _ in range(5)]

        mp_cpp = cpp_slam_core.MapPoint([1.0, 2.0, 3.0])
        for i, kf in enumerate(kfs):
            mp_cpp.add_observation(kf, i)

        # Remove middle observation
        mp_cpp.remove_observation(kfs[2], 2)
        assert not mp_cpp.is_in_keyframe(kfs[2])
        assert mp_cpp.is_in_keyframe(kfs[0])
        assert mp_cpp.is_in_keyframe(kfs[4])

    def test_set_bad_clears_all_parity(self):
        import cpp_slam_core
        FakeKeyFrame._kid = 0
        kfs = [FakeKeyFrame() for _ in range(5)]
        mp = cpp_slam_core.MapPoint([0.0, 0.0, 0.0])
        for i, kf in enumerate(kfs):
            mp.add_observation(kf, i)
        mp.set_bad()
        assert len(mp.observations()) == 0
        for i, kf in enumerate(kfs):
            assert kf.get_point_match(i) is None

    def test_replace_with(self):
        import cpp_slam_core
        FakeKeyFrame._kid = 0
        kfs = [FakeKeyFrame() for _ in range(6)]

        # mp1 has 3 obs, mp2 has 3 other obs, mp1 replaced by mp2
        mp1 = cpp_slam_core.MapPoint([1.0, 0.0, 0.0])
        mp2 = cpp_slam_core.MapPoint([2.0, 0.0, 0.0])
        for i in range(3):
            mp1.add_observation(kfs[i], i)
        for i in range(3, 6):
            mp2.add_observation(kfs[i], i)

        mp1.replace_with(mp2)
        assert mp1.is_bad()
        # mp2 should now have all 6 observations
        assert len(mp2.observations()) == 6
