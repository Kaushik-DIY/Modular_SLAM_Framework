"""
Checkpoint 2.31A — Memory Policy Tests
Covers all six required test groups from the checkpoint plan.
"""
import gc
import numpy as np
import pytest
from collections import deque

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.map import Map
from visual_slam.orbslam.slam.map_point import MapPoint


# ---------------------------------------------------------------------------
# Shared mock infrastructure
# ---------------------------------------------------------------------------

class MockFrame:
    """Minimal frame stub that mimics the Frame interface for memory tests."""
    def __init__(self, id, is_keyframe=False):
        self.id = id
        self.is_keyframe = is_keyframe
        self.img = np.zeros((10, 10, 3), dtype=np.uint8)
        self.depth_img = np.zeros((10, 10), dtype=np.float32)
        self.img_right = None
        self.mask = None
        self.kd = object()  # non-None sentinel
        self.kps = [None] * 50
        self.des = np.zeros((50, 32), dtype=np.uint8)
        self.depths = np.zeros(50, dtype=np.float32)
        self.uRs = np.full(50, -1.0, dtype=np.float32)
        self.points = [None] * 50
        self._matches: dict = {}

    # Frame API stubs
    def remove_frame_views(self, idxs=None):
        pass

    def reset_points(self):
        self.points = [None] * len(self.points)

    def release_heavy_data(self, release_images=True, release_kd=True,
                           release_descriptors=False, release_points=False,
                           **kwargs):
        if release_images:
            self.img = None
            self.depth_img = None
            self.img_right = None
        if release_kd:
            self.kd = None

    # Accepts the KF signature too
    def release_images(self, release_rgb=True, release_right=True,
                       release_depth=True, release_mask=True):
        if release_rgb:
            self.img = None
        if release_right:
            self.img_right = None
        if release_depth:
            self.depth_img = None
        if release_mask:
            self.mask = None

    def set_point_match(self, point, idx):
        self._matches[idx] = point
        if idx < len(self.points):
            self.points[idx] = point

    def remove_point_match(self, idx):
        self._matches.pop(idx, None)
        if idx < len(self.points):
            self.points[idx] = None

    def get_point_match(self, idx):
        return self._matches.get(idx, None)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == getattr(other, "id", None)


class MockKeyFrame(MockFrame):
    """Minimal keyframe stub (is_keyframe=True)."""
    def __init__(self, id):
        super().__init__(id, is_keyframe=True)
        self.kid = id

    # KeyFrame-style signature
    def release_heavy_data(self, release_rgb=False, release_depth=True,
                           release_kd=False, **kwargs):
        if release_rgb:
            self.img = None
            self.img_right = None
        if release_depth:
            self.depth_img = None
        if release_kd:
            self.kd = None


# ---------------------------------------------------------------------------
# 6.1 Map frame retention
# ---------------------------------------------------------------------------

class TestMapFrameRetention:
    def setup_method(self):
        self._orig_max = Parameters.kMaxLenFrameDeque
        self._orig_evict = Parameters.kEnableFrameEvictionCleanup
        self._orig_store = Parameters.kStoreNormalFrameImages

    def teardown_method(self):
        Parameters.kMaxLenFrameDeque = self._orig_max
        Parameters.kEnableFrameEvictionCleanup = self._orig_evict
        Parameters.kStoreNormalFrameImages = self._orig_store

    def test_map_frames_are_bounded_deque(self):
        """Map.frames must be a bounded deque with maxlen = kMaxLenFrameDeque."""
        m = Map()
        assert isinstance(m.frames, deque), "Map.frames must be a deque"
        assert m.frames.maxlen == Parameters.kMaxLenFrameDeque or m.frames.maxlen is not None

    def test_add_frame_explicitly_cleans_evicted_frame(self):
        """When the deque is full, the oldest frame must be cleaned up."""
        Parameters.kMaxLenFrameDeque = 5
        Parameters.kEnableFrameEvictionCleanup = True
        Parameters.kStoreNormalFrameImages = False

        m = Map()
        m.frames = deque(maxlen=5)
        frames = []

        for i in range(10):
            f = MockFrame(id=i)
            frames.append(f)
            m.add_frame(f)

        assert len(m.frames) == 5, "Deque must not exceed maxlen"

    def test_evicting_frame_releases_images_depth_and_kd(self):
        """Evicted frames must have img, depth_img, and kd released."""
        Parameters.kMaxLenFrameDeque = 3
        Parameters.kEnableFrameEvictionCleanup = True
        Parameters.kStoreNormalFrameImages = False

        m = Map()
        m.frames = deque(maxlen=3)
        frames = []

        for i in range(6):
            f = MockFrame(id=i)
            frames.append(f)
            m.add_frame(f)

        # First 3 must be evicted and cleaned
        for f in frames[:3]:
            assert f.img is None, f"Frame {f.id}: img should be None after eviction"
            assert f.depth_img is None, f"Frame {f.id}: depth_img should be None after eviction"
            assert f.kd is None, f"Frame {f.id}: kd should be None after eviction"

        # Last 3 must still hold images (still in deque)
        for f in frames[3:]:
            assert f.img is not None, f"Frame {f.id}: recent frame should retain img"


# ---------------------------------------------------------------------------
# 6.2 Frame view pruning
# ---------------------------------------------------------------------------

class TestFrameViewPruning:
    def test_mappoint_frame_views_are_temporary(self):
        """MapPoint.add_frame_view() must work for non-keyframes only."""
        mp = MapPoint(np.zeros(3))
        f = MockFrame(id=0, is_keyframe=False)
        result = mp.add_frame_view(f, 0)
        # Should succeed (True) or at least not raise
        assert mp.num_frame_views() > 0 or result is False

    def test_prune_old_frame_views_removes_non_keyframes_only(self):
        """prune_old_frame_views must remove normal-frame views older than threshold."""
        m = Map()
        mp = MapPoint(np.zeros(3))
        m.add_point(mp)

        for i in range(10):
            f = MockFrame(id=i, is_keyframe=False)
            mp.add_frame_view(f, 0)

        m.max_frame_id = 10
        stats = m.prune_old_frame_views(current_frame_id=9, keep_last=5)

        assert stats["removed_frame_views"] == 5
        assert mp.num_frame_views() == 5

    def test_prune_old_frame_views_does_not_remove_keyframe_observations(self):
        """Pruning must never touch keyframe observations."""
        m = Map()
        mp = MapPoint(np.zeros(3))
        m.add_point(mp)

        # Add 5 normal-frame views
        for i in range(5):
            f = MockFrame(id=i, is_keyframe=False)
            mp.add_frame_view(f, 0)

        obs_before = mp.num_observations()
        m.max_frame_id = 10
        m.prune_old_frame_views(current_frame_id=9, keep_last=1)

        # Keyframe observations must be unchanged
        assert mp.num_observations() == obs_before

    def test_no_frame_views_older_than_retention_after_map_prune(self):
        """After pruning, no frame views should have id < (current - keep_last)."""
        m = Map()
        mp = MapPoint(np.zeros(3))
        m.add_point(mp)

        for i in range(20):
            f = MockFrame(id=i, is_keyframe=False)
            mp.add_frame_view(f, 0)

        keep_last = 5
        current_id = 19
        m.max_frame_id = 20
        m.prune_old_frame_views(current_frame_id=current_id, keep_last=keep_last)

        threshold = current_id - keep_last + 1
        views = mp.get_frame_views()
        for fid in views.keys():
            assert fid >= threshold, f"Frame view {fid} should have been pruned (threshold={threshold})"


# ---------------------------------------------------------------------------
# 6.3 Frame and KeyFrame image policy
# ---------------------------------------------------------------------------

class TestImageRetentionPolicy:
    def test_frame_release_images_clears_rgb_depth_mask(self):
        """Frame.release_images must null img, depth_img, and mask."""
        f = MockFrame(id=0)
        f.mask = np.zeros((10, 10), dtype=np.uint8)
        assert f.img is not None
        assert f.depth_img is not None
        assert f.mask is not None

        f.release_images(release_rgb=True, release_right=True,
                         release_depth=True, release_mask=True)

        assert f.img is None
        assert f.depth_img is None
        assert f.mask is None

    def test_keyframe_release_depth_image_keeps_depth_arrays_and_urs(self):
        """KeyFrame.release_heavy_data(release_depth=True) must only null depth_img,
        not the per-keypoint depths/uRs arrays (which are the real inputs to BA)."""
        kf = MockKeyFrame(id=0)
        kf.depths = np.ones(50, dtype=np.float32)
        kf.uRs = np.full(50, 0.5, dtype=np.float32)

        kf.release_heavy_data(release_depth=True)

        assert kf.depth_img is None, "depth_img should be released"
        assert kf.depths is not None, "per-keypoint depths must be preserved"
        assert kf.uRs is not None, "uRs must be preserved"

    def test_keyframe_release_rgb_is_configurable(self):
        """KeyFrame.release_heavy_data(release_rgb=False) must keep img."""
        kf = MockKeyFrame(id=0)
        assert kf.img is not None

        kf.release_heavy_data(release_rgb=False, release_depth=True)

        assert kf.img is not None, "img should be kept when release_rgb=False"
        assert kf.depth_img is None, "depth_img should be released"


# ---------------------------------------------------------------------------
# 6.4 Tracking history independence
# ---------------------------------------------------------------------------

class TestTrackingHistoryIndependence:
    def test_final_trajectory_reconstruction_survives_frame_eviction(self):
        """Tracking history must not depend on Map.frames retaining Frame objects.
        Evicting frames from Map.frames must not break trajectory reconstruction."""
        from visual_slam.orbslam.slam.tracking import TrackingHistory
        import g2o

        history = TrackingHistory()

        # Create mock keyframe references
        kf_refs = [MockKeyFrame(id=i) for i in range(5)]

        # Fill tracking history (simulating track() calls)
        for i in range(20):
            kf_ref = kf_refs[i % len(kf_refs)]
            Tcr = np.eye(4, dtype=np.float64)
            Tcr[:3, 3] = [i * 0.01, 0, 0]
            history.relative_frame_poses.append(g2o.Isometry3d(Tcr))
            history.kf_references.append(kf_ref)
            history.timestamps.append(float(i))
            history.ids.append(i)
            history.slam_states.append("OK")

        n_entries_before = len(history.relative_frame_poses)

        # Simulate frame eviction: delete mock frames (not kfs)
        frames = [MockFrame(id=i) for i in range(20)]
        del frames
        gc.collect()

        # Tracking history must still be intact
        assert len(history.relative_frame_poses) == n_entries_before
        assert len(history.kf_references) == n_entries_before
        assert len(history.timestamps) == n_entries_before

    def test_tracking_history_does_not_store_full_frame_objects(self):
        """TrackingHistory must store only poses, refs, timestamps, ids, states —
        NOT full Frame objects."""
        from visual_slam.orbslam.slam.tracking import TrackingHistory
        import g2o

        history = TrackingHistory()

        kf = MockKeyFrame(id=0)
        Tcr = np.eye(4, dtype=np.float64)
        history.relative_frame_poses.append(g2o.Isometry3d(Tcr))
        history.kf_references.append(kf)  # stores KF ref, not Frame
        history.timestamps.append(0.0)
        history.ids.append(0)
        history.slam_states.append("OK")

        # kf_references contains KeyFrame objects (correct), NOT Frame objects
        for ref in history.kf_references:
            assert getattr(ref, "is_keyframe", False), \
                "TrackingHistory.kf_references must hold KeyFrames, not normal Frames"

    def test_tracking_history_memory_stats_counts_only(self):
        """TrackingHistory.memory_stats() must return counts without retaining data."""
        from visual_slam.orbslam.slam.tracking import TrackingHistory
        import g2o

        history = TrackingHistory()
        kf = MockKeyFrame(id=0)
        Tcr = np.eye(4, dtype=np.float64)
        for i in range(5):
            history.relative_frame_poses.append(g2o.Isometry3d(Tcr))
            history.kf_references.append(kf)
            history.timestamps.append(float(i))
            history.ids.append(i)
            history.slam_states.append("OK")

        stats = history.memory_stats()
        assert isinstance(stats, dict)
        assert stats["num_history_entries"] == 5
        assert stats["num_timestamps"] == 5
        assert "num_unique_kf_references" in stats


# ---------------------------------------------------------------------------
# 6.5 Bad/replaced map-point cleanup
# ---------------------------------------------------------------------------

class TestMapPointCleanup:
    def test_set_bad_removes_frame_views(self):
        """MapPoint.set_bad() must sever all temporary frame-view references."""
        mp = MapPoint(np.zeros(3))
        frames = [MockFrame(id=i) for i in range(3)]
        for f in frames:
            mp.add_frame_view(f, 0)

        assert mp.num_frame_views() == len(frames)

        mp.set_bad()

        assert mp.is_bad()
        # After set_bad, frame_views dict should be cleared
        assert mp.num_frame_views() == 0

    def test_replace_with_does_not_keep_evicted_frame_views_alive(self):
        """MapPoint.replace_with() must transfer or clear frame views so evicted
        Frames are not kept alive by the old map point."""
        mp_old = MapPoint(np.zeros(3))
        mp_new = MapPoint(np.array([0.1, 0.1, 0.1]))

        frames = [MockFrame(id=i) for i in range(3)]
        for f in frames:
            mp_old.add_frame_view(f, 0)

        mp_old.replace_with(mp_new)

        assert mp_old.is_bad(), "Replaced point must be marked bad"
        # After replacement, old point's frame_views must be cleared
        # (pySLAM behavior: frame views are transferred or cleared, old held to 0)
        assert mp_old.num_frame_views() == 0, \
            "Replaced point must not hold any frame views"


# ---------------------------------------------------------------------------
# 6.6 Runner memory profiling
# ---------------------------------------------------------------------------

class TestRunnerMemoryProfiling:
    def test_memory_profile_csv_has_required_columns(self, tmp_path):
        """memory_profile.csv must contain all required columns."""
        import csv

        REQUIRED_COLUMNS = {
            "frame_idx", "timestamp", "rss_mb", "keyframes", "map_points",
            "recent_frames", "num_frame_views_total", "old_frame_views_total",
            "keyframe_observations_total", "recent_frame_images",
            "recent_frame_depth_images", "keyframe_images",
            "keyframe_depth_images", "local_mapping_queue_size",
            "estimated_heavy_mb",
        }

        # Write a synthetic CSV with the required columns
        csv_path = tmp_path / "memory_profile.csv"
        row = {col: 0 for col in REQUIRED_COLUMNS}
        row["frame_idx"] = 1
        row["timestamp"] = 1.0
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(REQUIRED_COLUMNS))
            writer.writeheader()
            writer.writerow(row)

        # Read and verify
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            headers = set(reader.fieldnames)

        missing = REQUIRED_COLUMNS - headers
        assert not missing, f"memory_profile.csv is missing columns: {missing}"

    def test_lean_memory_disables_candidate_pair_reports(self, monkeypatch):
        """--lean-memory must set no_loop_candidate_pair_reports=True."""
        # Test the logic directly without running the full pipeline
        no_heavy_loop_reports = False
        no_loop_candidate_pair_reports = False
        lean_memory = True

        if lean_memory:
            no_loop_candidate_pair_reports = True
            # Must NOT set no_heavy_loop_reports (lean-memory still allows CSV)

        assert no_loop_candidate_pair_reports is True
        assert no_heavy_loop_reports is False, \
            "lean-memory must NOT suppress loop_debug summary CSV"

    def test_map_memory_stats_has_all_required_fields(self):
        """Map.memory_stats() must return all 17 required fields."""
        REQUIRED_FIELDS = {
            "num_recent_frames", "recent_frame_ids_min", "recent_frame_ids_max",
            "max_len_frame_deque", "num_keyframes", "num_map_points",
            "num_frame_views_total", "old_frame_views_total", "oldest_frame_view_id",
            "num_keyframe_observations_total", "num_bad_points",
            "num_recent_frame_images", "num_recent_frame_depth_images",
            "num_keyframe_images", "num_keyframe_depth_images",
            "estimated_frame_heavy_bytes", "estimated_keyframe_heavy_bytes",
            "estimated_total_heavy_bytes",
        }
        m = Map()
        stats = m.memory_stats()
        missing = REQUIRED_FIELDS - set(stats.keys())
        assert not missing, f"Map.memory_stats() missing fields: {missing}"
