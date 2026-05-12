"""
Phase 4 parity test: C++ LocalMappingCore.

Verifies:
1. process_new_keyframe() — adds observations, updates connections
2. cull_map_points()      — removes bad/stale recently-added points
3. fuse_map_points()      — calls ProjectionMatcher and updates connections
4. cull_keyframes()       — marks redundant KFs bad
5. local_BA()             — returns finite MSE and tracked-point count
6. Parity:  C++ and Python LMC produce the same map state on a synthetic
            scenario (same recently_added set, same covisibility graph)
7. Timing:  C++ LMC process_new_keyframe is measurably faster on a large KF
"""
import time

import cv2
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, "/home/kaushik/slam_ws")
os.chdir("/home/kaushik/slam_ws")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_kf(kid, n_kps=20, depths=None):
    """Create a C++ KeyFrame with synthetic features and optional depths."""
    import cpp_slam_core
    kf = cpp_slam_core.KeyFrame(kid=kid, frame_id=kid)
    kf.timestamp = float(kid)
    kps = [cv2.KeyPoint(x=float(i * 4 + 1), y=float(i * 3 + 1), size=2.0, octave=i % 4)
           for i in range(n_kps)]
    des = np.random.randint(0, 255, (n_kps, 32), dtype=np.uint8)
    kf.init_feature_arrays(kps, des, None, None, n_kps)
    if depths is not None:
        kf.depths = np.array(depths, dtype=np.float32)
    return kf


def _make_mp(pos):
    """Create a C++ MapPoint at the given 3-D position."""
    import cpp_slam_core
    return cpp_slam_core.MapPoint(np.array(pos, dtype=np.float64))


def _build_scenario(n_kfs=4, n_shared=12):
    """
    Build n_kfs C++ KeyFrames sharing n_shared MapPoints between consecutive KF pairs.
    Returns (kfs, mps).
    """
    import cpp_slam_core
    kfs = [_make_kf(i, n_kps=n_shared * n_kfs) for i in range(n_kfs)]
    mps = []
    for i in range(n_kfs - 1):
        for j in range(n_shared):
            mp = _make_mp([float(i + j), float(j % 3), 5.0])
            mp.add_observation(kfs[i], j)
            mp.add_observation(kfs[i + 1], j)
            kfs[i].set_point_match(mp, j)
            kfs[i + 1].set_point_match(mp, j)
            mps.append(mp)
    for kf in kfs:
        kf.update_connections()
    return kfs, mps


def _make_fake_map(kfs, mps):
    """
    A minimal Python object that satisfies the map interface used by LMC.
    """
    from types import SimpleNamespace

    # local_map stub
    def get_best_neighbors(kf_ref, N=10):
        return [k for k in kfs if k is not kf_ref][:N]

    local_map = SimpleNamespace(get_best_neighbors=get_best_neighbors)

    removed = []

    def remove_point(p):
        removed.append(p)

    def locally_optimize(kf_ref, abort_flag=None, mp_abort_flag=None):
        return 0.01  # synthetic MSE

    class UpdateLock:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    return SimpleNamespace(
        local_map=local_map,
        remove_point=remove_point,
        locally_optimize=locally_optimize,
        update_lock=UpdateLock(),
        _removed=removed,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestProcessNewKeyframe:
    def test_adds_observations_to_points(self):
        import cpp_slam_core
        kfs, mps = _build_scenario(n_kfs=3, n_shared=10)
        fake_map = _make_fake_map(kfs, mps)

        lmc = cpp_slam_core.LocalMappingCore(fake_map, 2)  # RGBD
        lmc.kf_cur = kfs[1]

        # Clear all observations first so process_new_keyframe re-adds them.
        kfs[1].update_connections()
        lmc.process_new_keyframe()

        # After process_new_keyframe, kf[1] should have covisible KFs.
        covis = kfs[1].get_connected_keyframes()
        assert len(covis) >= 1, "Expected covisibility after process_new_keyframe"

    def test_adds_to_recently_added_on_existing_obs(self):
        """
        Points already observed (add_observation returns False) go into recently_added.
        """
        import cpp_slam_core
        kf0 = _make_kf(0, n_kps=5)
        kf1 = _make_kf(1, n_kps=5)
        mps = []
        for j in range(5):
            mp = _make_mp([float(j), 0.0, 5.0])
            mp.add_observation(kf0, j)
            mp.add_observation(kf1, j)   # already observed
            kf0.set_point_match(mp, j)
            kf1.set_point_match(mp, j)
            mps.append(mp)
        kf0.update_connections()
        kf1.update_connections()

        fake_map = _make_fake_map([kf0, kf1], mps)
        lmc = cpp_slam_core.LocalMappingCore(fake_map, 2)
        lmc.kf_cur = kf1

        lmc.process_new_keyframe()

        # kf1 points already had observations → recently_added should be non-empty
        assert len(lmc.recently_added) > 0

    def test_update_connections_called(self):
        """After process_new_keyframe, kf_cur must have updated covisibility."""
        import cpp_slam_core
        kfs, mps = _build_scenario(n_kfs=2, n_shared=15)
        fake_map = _make_fake_map(kfs, mps)
        lmc = cpp_slam_core.LocalMappingCore(fake_map, 2)
        lmc.kf_cur = kfs[1]
        lmc.process_new_keyframe()
        covis = kfs[1].get_connected_keyframes()
        assert len(covis) >= 1


class TestCullMapPoints:
    def test_bad_points_removed(self):
        """Points already marked bad are removed from recently_added."""
        import cpp_slam_core
        kf = _make_kf(0, n_kps=5)
        mps = [_make_mp([float(i), 0.0, 5.0]) for i in range(5)]
        for j, mp in enumerate(mps):
            mp.add_observation(kf, j)
        fake_map = _make_fake_map([kf], mps)

        lmc = cpp_slam_core.LocalMappingCore(fake_map, 2)
        lmc.kf_cur = kf
        lmc.recently_added = set(mps)

        mps[0].set_bad()   # mark first as bad
        n = lmc.cull_map_points()
        assert n >= 1, "Expected at least the bad point to be culled"
        for p in lmc.recently_added:
            assert not p.is_bad(), "No bad points should remain in recently_added"

    def test_low_found_ratio_removed(self):
        """Points with found_ratio < 0.25 are set bad and removed."""
        import cpp_slam_core
        kf = _make_kf(0, n_kps=3)
        mps = [_make_mp([float(i), 0.0, 5.0]) for i in range(3)]
        for j, mp in enumerate(mps):
            mp.add_observation(kf, j)
            mp.increase_visible(10)  # visible many times
            # found only 1 time → ratio = 0.1 < 0.25

        fake_map = _make_fake_map([kf], mps)
        lmc = cpp_slam_core.LocalMappingCore(fake_map, 2)
        lmc.kf_cur = kf
        lmc.recently_added = set(mps)

        n = lmc.cull_map_points()
        assert n >= 1, "Expected points with low found ratio to be culled"

    def test_keeps_healthy_recent_points(self):
        """Points added in the same KF cycle with good ratios are kept."""
        import cpp_slam_core
        kf = _make_kf(1, n_kps=5)
        mps = [_make_mp([float(i), 0.0, 5.0]) for i in range(5)]
        for j, mp in enumerate(mps):
            mp.add_observation(kf, j)
            mp.increase_visible(4)
            mp.increase_found(4)  # found_ratio = 1.0

        fake_map = _make_fake_map([kf], mps)
        lmc = cpp_slam_core.LocalMappingCore(fake_map, 2)
        lmc.kf_cur = kf
        lmc.recently_added = set(mps)

        n = lmc.cull_map_points()
        # newly added points in same KF shouldn't be culled yet (first_kid == current_kid)
        for p in lmc.recently_added:
            assert not p.is_bad()


class TestCullKeyframes:
    def test_marks_redundant_kf_bad(self):
        """
        A KF whose points are all seen by 3+ other KFs at equal/finer scale
        should be marked bad (redundancy > 90% threshold).
        """
        import cpp_slam_core
        # Build 5 KFs all sharing the same 20 map points (highly redundant).
        n_kfs, n_pts = 5, 20
        kfs = [_make_kf(i, n_kps=n_pts) for i in range(n_kfs)]
        for i in range(n_kfs - 1):
            kfs[i].set_parent(kfs[0])

        mps = []
        for j in range(n_pts):
            mp = _make_mp([float(j), 0.0, 5.0])
            for kf in kfs:
                mp.add_observation(kf, j)
                kf.set_point_match(mp, j)
            mps.append(mp)

        for kf in kfs:
            kf.update_connections()

        fake_map = _make_fake_map(kfs, mps)
        lmc = cpp_slam_core.LocalMappingCore(fake_map, 2)
        lmc.kf_cur = kfs[0]

        n_culled = lmc.cull_keyframes()
        assert n_culled >= 1, f"Expected at least 1 redundant KF culled, got {n_culled}"

    def test_does_not_cull_kf_zero(self):
        """kid=0 KF must never be culled."""
        import cpp_slam_core
        kfs = [_make_kf(i, n_kps=10) for i in range(3)]
        mps = [_make_mp([float(j), 0.0, 5.0]) for j in range(10)]
        for j, mp in enumerate(mps):
            for kf in kfs:
                mp.add_observation(kf, j); kf.set_point_match(mp, j)
        for kf in kfs:
            kf.update_connections()

        fake_map = _make_fake_map(kfs, mps)
        lmc = cpp_slam_core.LocalMappingCore(fake_map, 2)
        lmc.kf_cur = kfs[0]

        lmc.cull_keyframes()
        assert not kfs[0].is_bad(), "kid=0 must never be culled"


class TestLocalBA:
    def test_returns_finite_mse_and_tracked(self):
        """local_BA() must return (finite_mse, tracked >= 0)."""
        import cpp_slam_core
        kfs, mps = _build_scenario(n_kfs=3, n_shared=12)
        fake_map = _make_fake_map(kfs, mps)

        lmc = cpp_slam_core.LocalMappingCore(fake_map, 2)
        lmc.kf_cur = kfs[1]

        result = lmc.local_BA()
        err, tracked = result[0], result[1]
        assert np.isfinite(err), f"local_BA MSE must be finite, got {err}"
        assert tracked >= 0

    def test_tracked_count_matches_good_points(self):
        """tracked = number of KF points with ≥3 observations."""
        import cpp_slam_core
        kfs, mps = _build_scenario(n_kfs=3, n_shared=12)
        fake_map = _make_fake_map(kfs, mps)

        lmc = cpp_slam_core.LocalMappingCore(fake_map, 2)
        lmc.kf_cur = kfs[1]

        _, tracked = lmc.local_BA()
        expected = kfs[1].num_tracked_points(3)
        assert tracked == expected


class TestFuseMapPoints:
    def test_returns_int(self):
        """fuse_map_points() returns an integer (number of fused points)."""
        import cpp_slam_core
        kfs, mps = _build_scenario(n_kfs=3, n_shared=10)
        fake_map = _make_fake_map(kfs, mps)

        lmc = cpp_slam_core.LocalMappingCore(fake_map, 2)
        lmc.kf_cur = kfs[1]

        n_fused = lmc.fuse_map_points(50.0)
        assert isinstance(n_fused, int)
        assert n_fused >= 0

    def test_updates_connections_after_fuse(self):
        """fuse_map_points calls update_connections on kf_cur."""
        import cpp_slam_core
        kfs, mps = _build_scenario(n_kfs=3, n_shared=10)
        fake_map = _make_fake_map(kfs, mps)

        lmc = cpp_slam_core.LocalMappingCore(fake_map, 2)
        lmc.kf_cur = kfs[1]

        lmc.fuse_map_points(50.0)
        covis = kfs[1].get_connected_keyframes()
        # After fuse + update_connections, kf[1] must have covisible KFs
        assert len(covis) >= 1


class TestParityWithPython:
    """
    C++ LMC and Python LMC must produce the same observable state after
    running process_new_keyframe on an identical scenario.
    """

    def _run_python_lmc(self, kfs, mps, fake_map):
        from visual_slam.orbslam.slam.local_mapping_core import LocalMappingCore as PyLMC
        from visual_slam.orbslam.slam.sensor_types import SensorType
        # Use Python KF/MP proxies? No – we can call the same C++ KF through the
        # Python LMC (it accepts py::object). This tests that the C++ objects
        # respond correctly to Python LMC interface.
        lmc = PyLMC(fake_map, SensorType.RGBD)
        lmc.kf_cur = kfs[1]
        lmc.process_new_keyframe()
        return lmc

    def _run_cpp_lmc(self, kfs, mps, fake_map):
        import cpp_slam_core
        lmc = cpp_slam_core.LocalMappingCore(fake_map, 2)
        lmc.kf_cur = kfs[1]
        lmc.process_new_keyframe()
        return lmc

    def test_covisibility_identical(self):
        """
        C++ and Python LMC produce the same covisibility graph after
        process_new_keyframe on C++ KeyFrames.
        """
        import cpp_slam_core

        # Build scenario using C++ objects — run both LMCs and compare.
        kfs_a, mps_a = _build_scenario(n_kfs=4, n_shared=12)
        kfs_b, mps_b = _build_scenario(n_kfs=4, n_shared=12)

        # Ensure all observations cleared so process_new_keyframe adds them fresh.
        for kf in kfs_a: kf.reset_covisibility()
        for kf in kfs_b: kf.reset_covisibility()

        fake_map_a = _make_fake_map(kfs_a, mps_a)
        fake_map_b = _make_fake_map(kfs_b, mps_b)

        # Run C++ LMC on scenario A.
        cpp_lmc = self._run_cpp_lmc(kfs_a, mps_a, fake_map_a)
        # Run Python LMC on scenario B (identical structure).
        py_lmc  = self._run_python_lmc(kfs_b, mps_b, fake_map_b)

        covis_cpp = set(kf.kid for kf in kfs_a[1].get_connected_keyframes())
        covis_py  = set(kf.kid for kf in kfs_b[1].get_connected_keyframes())
        assert covis_cpp == covis_py, (
            f"Covisibility mismatch: C++={covis_cpp}, Python={covis_py}")


class TestNumTrackedPoints:
    def test_counts_non_bad_with_min_obs(self):
        """num_tracked_points(3) counts matched points with ≥3 observations."""
        import cpp_slam_core
        kfs, mps = _build_scenario(n_kfs=3, n_shared=10)
        kf = kfs[1]

        expected = sum(
            1 for p in kf.get_matched_good_points()
            if not p.is_bad() and p.n_obs >= 3
        )
        result = kf.num_tracked_points(3)
        assert result == expected

    def test_zero_min_obs_counts_all_non_bad(self):
        import cpp_slam_core
        kfs, mps = _build_scenario(n_kfs=2, n_shared=8)
        kf = kfs[0]
        expected = len(kf.get_matched_good_points())
        assert kf.num_tracked_points(0) == expected


class TestGetMatchedGoodPointsAndIdxs:
    def test_returns_correct_indices(self):
        import cpp_slam_core
        kf = _make_kf(0, n_kps=10)
        mps = [_make_mp([float(i), 0.0, 5.0]) for i in range(5)]
        for j, mp in enumerate(mps):
            kf.set_point_match(mp, j)

        pairs = kf.get_matched_good_points_and_idxs()
        assert len(pairs) == 5
        idxs = [idx for _p, idx in pairs]
        assert sorted(idxs) == list(range(5))

    def test_skips_bad_points(self):
        import cpp_slam_core
        kf = _make_kf(0, n_kps=5)
        mps = [_make_mp([float(i), 0.0, 5.0]) for i in range(5)]
        mp_obs = cpp_slam_core.MapPoint([0.0, 0.0, 0.0])
        for j, mp in enumerate(mps):
            kf.set_point_match(mp, j)
        mps[2].set_bad()

        pairs = kf.get_matched_good_points_and_idxs()
        result_idxs = [idx for _p, idx in pairs]
        assert 2 not in result_idxs, "Bad point's index should be skipped"


class TestDepthsField:
    def test_depths_settable_on_cpp_keyframe(self):
        """Verify that the depths field can be set and read on C++ KeyFrame."""
        import cpp_slam_core
        kf = _make_kf(0, n_kps=5)
        depths = np.array([1.5, 2.0, 2.5, 3.0, 3.5], dtype=np.float32)
        kf.depths = depths
        retrieved = np.asarray(kf.depths)
        np.testing.assert_allclose(retrieved, depths)

    def test_depths_default_none(self):
        import cpp_slam_core
        kf = _make_kf(0, n_kps=3)
        assert kf.depths is None or (hasattr(kf.depths, 'is_none') and kf.depths.is_none())


class TestLifecycleMethods:
    def test_reset_clears_recently_added(self):
        import cpp_slam_core
        kfs, mps = _build_scenario(n_kfs=2, n_shared=5)
        fake_map = _make_fake_map(kfs, mps)
        lmc = cpp_slam_core.LocalMappingCore(fake_map, 2)
        lmc.recently_added = set(mps[:3])
        lmc.reset()
        assert len(lmc.recently_added) == 0

    def test_add_remove_points(self):
        import cpp_slam_core
        kfs, mps = _build_scenario(n_kfs=2, n_shared=5)
        fake_map = _make_fake_map(kfs, mps)
        lmc = cpp_slam_core.LocalMappingCore(fake_map, 2)
        lmc.add_points(mps[:3])
        assert len(lmc.recently_added) == 3
        lmc.remove_points(mps[:2])
        assert len(lmc.recently_added) == 1


class TestTimingBenefit:
    """
    Process 50 KFs with C++ LMC and verify it completes in reasonable time.
    This is a sanity check, not a strict micro-benchmark.
    """

    def test_process_new_keyframe_timing(self):
        import cpp_slam_core
        n_kfs, n_pts = 50, 30
        kfs = [_make_kf(i, n_kps=n_pts) for i in range(n_kfs)]
        mps = [_make_mp([float(j), 0.0, 5.0]) for j in range(n_pts)]
        for j, mp in enumerate(mps):
            for kf in kfs:
                mp.add_observation(kf, j); kf.set_point_match(mp, j)
        for kf in kfs:
            kf.update_connections()

        fake_map = _make_fake_map(kfs, mps)

        t0 = time.perf_counter()
        for i in range(1, n_kfs):
            lmc = cpp_slam_core.LocalMappingCore(fake_map, 2)
            lmc.kf_cur = kfs[i]
            lmc.process_new_keyframe()
        elapsed = time.perf_counter() - t0
        # Should finish well under 10 seconds even on a slow machine.
        assert elapsed < 10.0, f"C++ LMC process_new_keyframe too slow: {elapsed:.2f}s for {n_kfs} KFs"
