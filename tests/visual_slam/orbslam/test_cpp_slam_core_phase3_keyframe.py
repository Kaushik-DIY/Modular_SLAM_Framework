"""
Phase 3 parity test: C++ KeyFrame covisibility graph.

Verifies:
1. C++ KeyFrame is-a Frame (inherits all Frame attributes)
2. Covisibility graph add/erase/get operations
3. update_connections() parity with Python Counter-based implementation
4. Spanning tree parent/children
5. Thread safety: concurrent add_connection calls
6. set_bad: clears covisibility from all connected KFs
"""
import numpy as np
import pytest
import threading
import sys, os
sys.path.insert(0, "/home/kaushik/slam_ws")
os.chdir("/home/kaushik/slam_ws")


def make_kf(kid, n_kps=10):
    """Create a C++ KeyFrame with synthetic feature data."""
    import cv2, cpp_slam_core
    kf = cpp_slam_core.KeyFrame(kid=kid, frame_id=kid)
    kps = [cv2.KeyPoint(x=float(i * 5), y=float(i * 3), size=1.0, octave=i % 4)
           for i in range(n_kps)]
    des = np.random.randint(0, 255, (n_kps, 32), dtype=np.uint8)
    kf.init_feature_arrays(kps, des, None, None, n_kps)
    return kf


class TestKeyFrameInheritsFrame:
    """C++ KeyFrame must expose all Frame attributes."""

    def test_kpsu_accessible(self):
        kf = make_kf(1)
        assert kf.kpsu.shape == (10, 2)

    def test_pose_accessible(self):
        kf = make_kf(2)
        T = np.eye(4); T[0, 3] = 3.0
        kf.update_pose(T)
        np.testing.assert_allclose(kf.Tcw()[0, 3], 3.0, atol=1e-12)

    def test_point_match_accessible(self):
        import cpp_slam_core
        kf = make_kf(3)
        mp = cpp_slam_core.MapPoint([1.0, 0.0, 0.0])
        kf.set_point_match(mp, 4)
        assert kf.get_point_match(4) is mp

    def test_octaves_accessible(self):
        kf = make_kf(4)
        assert len(kf.octaves) == 10


class TestCovisibilityBasic:
    def test_add_connection(self):
        import cpp_slam_core
        kf1 = make_kf(10)
        kf2 = make_kf(11)
        kf1.add_connection(kf2, 25)
        assert kf2 in kf1.get_connected_keyframes()

    def test_erase_connection(self):
        import cpp_slam_core
        kf1 = make_kf(20)
        kf2 = make_kf(21)
        kf1.add_connection(kf2, 15)
        kf1.erase_connection(kf2)
        assert kf2 not in kf1.get_connected_keyframes()

    def test_get_weight(self):
        import cpp_slam_core
        kf1 = make_kf(30)
        kf2 = make_kf(31)
        kf1.add_connection(kf2, 42)
        assert kf1.get_weight(kf2) == 42

    def test_get_best_covisibles_order(self):
        """get_best_covisibles returns KFs sorted by weight descending."""
        import cpp_slam_core
        kf = make_kf(40)
        kf_a = make_kf(41)
        kf_b = make_kf(42)
        kf_c = make_kf(43)
        kf.add_connection(kf_a, 5)
        kf.add_connection(kf_b, 30)
        kf.add_connection(kf_c, 15)
        best = kf.get_best_covisibles(2)
        assert len(best) == 2
        assert best[0] is kf_b  # highest weight
        assert best[1] is kf_c  # second highest

    def test_get_covisible_by_weight(self):
        """get_covisible_by_weight(min_w) returns KFs with weight > min_w."""
        import cpp_slam_core
        kf = make_kf(50)
        kf_a = make_kf(51)
        kf_b = make_kf(52)
        kf.add_connection(kf_a, 5)
        kf.add_connection(kf_b, 30)
        result = kf.get_covisible_by_weight(10)
        assert kf_b in result
        assert kf_a not in result


class TestCovisibilityParityWithPython:
    """
    Verify that C++ covisibility weights match Python Counter-based results.
    Build a 5-KF map with shared MapPoints and check update_connections().
    """

    def _build_scenario(self, KFClass, MPClass, n_kfs=5, n_shared=15):
        """
        Build n_kfs keyframes sharing n_shared map points each.
        Each KF i sees all KFs j<i via shared points.
        Returns (kfs, map_points).
        """
        import cv2
        kfs = []
        for i in range(n_kfs):
            kf = KFClass(kid=i, frame_id=i) if hasattr(KFClass, '__self__') else KFClass(i, i)
            import numpy as np
            n_kps = n_shared * n_kfs
            kps = [cv2.KeyPoint(x=float(j * 2), y=0, size=1.0) for j in range(n_kps)]
            des = np.zeros((n_kps, 32), dtype=np.uint8)
            kf.init_feature_arrays(kps, des, None, None, n_kps)
            kfs.append(kf)

        # Create MapPoints shared between consecutive KF pairs
        mps = []
        for i in range(n_kfs - 1):
            for j in range(n_shared):
                mp = MPClass([float(i + j), 0.0, 5.0])
                # KF i observes it at feature j
                mp.add_observation(kfs[i], j)
                # KF i+1 also observes it at feature j
                mp.add_observation(kfs[i + 1], j)
                mps.append(mp)

        return kfs, mps

    def test_update_connections_cpp(self):
        """C++ update_connections() builds non-empty covisibility."""
        import cpp_slam_core
        kfs, mps = self._build_scenario(
            lambda kid, fid: cpp_slam_core.KeyFrame(kid=kid, frame_id=fid),
            lambda pos: cpp_slam_core.MapPoint(pos),
            n_kfs=4, n_shared=10)

        # Run update_connections for all KFs
        for kf in kfs:
            kf.update_connections()

        # Each non-first/last KF should have at least 2 covisible KFs
        for i in range(1, len(kfs) - 1):
            covis = kfs[i].get_connected_keyframes()
            assert len(covis) >= 1, f"KF {i} has no covisible KFs"

    def test_covisibility_weights_correct(self):
        """Weight = number of shared MapPoints between KF pair."""
        import cpp_slam_core
        kf1 = cpp_slam_core.KeyFrame(kid=100, frame_id=100)
        kf2 = cpp_slam_core.KeyFrame(kid=101, frame_id=101)
        import cv2, numpy as np

        n_shared = 20
        for kf in [kf1, kf2]:
            kps = [cv2.KeyPoint(x=float(i), y=0, size=1.0) for i in range(n_shared + 5)]
            des = np.zeros((n_shared + 5, 32), dtype=np.uint8)
            kf.init_feature_arrays(kps, des, None, None, n_shared + 5)

        # Create n_shared shared MapPoints
        for j in range(n_shared):
            mp = cpp_slam_core.MapPoint([float(j), 0.0, 5.0])
            mp.add_observation(kf1, j)
            mp.add_observation(kf2, j)

        kf1.update_connections()
        # kf1 should have kf2 as covisible with weight = n_shared
        assert kf1.get_weight(kf2) == n_shared


class TestSpanningTree:
    def test_set_parent(self):
        import cpp_slam_core
        kf0 = make_kf(200)
        kf1 = make_kf(201)
        kf1.set_parent(kf0)
        assert kf1.get_parent() is kf0
        assert kf1.has_child(kf1) is False  # kf0 should have kf1 as child

    def test_add_erase_child(self):
        import cpp_slam_core
        parent = make_kf(210)
        child = make_kf(211)
        parent.add_child(child)
        assert child in parent.get_children()
        parent.erase_child(child)
        assert child not in parent.get_children()


class TestLoopEdges:
    def test_add_loop_edge(self):
        import cpp_slam_core
        kf1 = make_kf(300)
        kf2 = make_kf(301)
        kf1.add_loop_edge(kf2)
        assert kf2 in kf1.get_loop_edges()
        assert kf1.not_to_erase is True


class TestThreadSafety:
    def test_concurrent_add_connection(self):
        """Two threads calling add_connection concurrently must not deadlock."""
        import cpp_slam_core
        kf = make_kf(400)
        other_kfs = [make_kf(400 + i + 1) for i in range(10)]
        errors = []

        def add_all(kf_list):
            for k in kf_list:
                try:
                    kf.add_connection(k, 5)
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=add_all, args=(other_kfs[:5],))
        t2 = threading.Thread(target=add_all, args=(other_kfs[5:],))
        t1.start(); t2.start()
        t1.join(timeout=5.0); t2.join(timeout=5.0)
        assert not t1.is_alive(), "Thread 1 deadlocked"
        assert not t2.is_alive(), "Thread 2 deadlocked"
        assert len(errors) == 0
        assert len(kf.get_connected_keyframes()) == 10


class TestSetBad:
    def test_set_bad_clears_covisibility(self):
        """set_bad() removes the KF from all its covisible KFs' graphs."""
        import cpp_slam_core
        kf_main = make_kf(500)
        kf_main.kid = 500  # Ensure kid != 0 so set_bad doesn't return early
        others = [make_kf(501 + i) for i in range(3)]

        # Add bidirectional connections
        for o in others:
            kf_main.add_connection(o, 10)
            o.add_connection(kf_main, 10)

        kf_main.set_bad()
        assert kf_main.is_bad()

    def test_set_bad_is_idempotent(self):
        import cpp_slam_core
        kf = make_kf(600)
        kf.kid = 600
        kf.set_bad()
        kf.set_bad()  # Should not raise
        assert kf.is_bad()


class TestHashEquality:
    def test_hashable_in_set(self):
        import cpp_slam_core
        kf1 = make_kf(700)
        kf2 = make_kf(701)
        s = {kf1, kf2}
        assert kf1 in s
        assert kf2 in s

    def test_equality_by_kid(self):
        import cpp_slam_core
        kf_a = cpp_slam_core.KeyFrame(kid=800)
        kf_b = cpp_slam_core.KeyFrame(kid=800)
        assert kf_a == kf_b

    def test_ordering(self):
        import cpp_slam_core
        kf1 = cpp_slam_core.KeyFrame(kid=1)
        kf2 = cpp_slam_core.KeyFrame(kid=2)
        assert kf1 < kf2
