from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from visual_slam.orbslam.run_rgbd_slam import (
    FRAME_TIMING_COLUMNS,
    run_rgbd_slam,
)
from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.local_mapping_core import LocalMappingCore
from visual_slam.orbslam.slam.map import Map
from visual_slam.orbslam.slam.runtime_profiler import RuntimeProfiler
from visual_slam.orbslam.slam.sensor_types import SensorType
from visual_slam.orbslam.slam.slam_commons import SlamState


class ParameterGuard:
    def __init__(self, *names: str):
        self.names = names
        self.values = {name: getattr(Parameters, name) for name in names}

    def restore(self) -> None:
        for name, value in self.values.items():
            setattr(Parameters, name, value)


class FakeAssociationsEntry:
    def __init__(self, rgb_path: Path, depth_path: Path, timestamp: float):
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.timestamp = timestamp


class FakeTrackingHistory:
    def __init__(self):
        self.timestamps = []


class FakeTracking:
    def __init__(self):
        self.poses = []
        self.tracking_history = FakeTrackingHistory()
        self.num_matched_map_points = 8
        self.mean_pose_opt_chi2_error = 0.1


class FakeMap:
    def __init__(self):
        self._keyframes = 2
        self._points = 25
        self._frames = 1
        self.memory_stats_modes: list[str] = []
        self.prune_calls = 0

    def num_keyframes(self):
        return self._keyframes

    def num_points(self):
        return self._points

    def num_frames(self):
        return self._frames

    def prune_old_frame_views(self, current_frame_id=None, keep_last=None):
        self.prune_calls += 1
        return {
            "checked_points": 3,
            "removed_frame_views": 1,
            "remaining_frame_views": 2,
            "oldest_remaining_frame_view_id": 0,
        }

    def memory_stats(self, mode="deep"):
        self.memory_stats_modes.append(str(mode))
        if str(mode) == "deep":
            return {
                "num_recent_frames": self._frames,
                "recent_frame_ids_min": 0,
                "recent_frame_ids_max": 0,
                "max_len_frame_deque": Parameters.kMaxLenFrameDeque,
                "num_keyframes": self._keyframes,
                "num_map_points": self._points,
                "num_frame_views_total": 4,
                "old_frame_views_total": 1,
                "oldest_frame_view_id": 0,
                "num_keyframe_observations_total": 6,
                "num_bad_points": 0,
                "num_recent_frame_images": 0,
                "num_recent_frame_depth_images": 0,
                "num_keyframe_images": 0,
                "num_keyframe_depth_images": 0,
                "estimated_frame_heavy_bytes": 0,
                "estimated_keyframe_heavy_bytes": 0,
                "estimated_total_heavy_bytes": 0,
            }
        return {
            "num_recent_frames": self._frames,
            "recent_frame_ids_min": 0,
            "recent_frame_ids_max": 0,
            "max_len_frame_deque": Parameters.kMaxLenFrameDeque,
            "num_keyframes": self._keyframes,
            "num_map_points": self._points,
            "num_frame_views_total": 2,
            "old_frame_views_total": 0,
            "oldest_frame_view_id": 0,
            "num_keyframe_observations_total": 0,
            "num_bad_points": 0,
            "num_recent_frame_images": 0,
            "num_recent_frame_depth_images": 0,
            "num_keyframe_images": 0,
            "num_keyframe_depth_images": 0,
            "estimated_frame_heavy_bytes": 0,
            "estimated_keyframe_heavy_bytes": 0,
            "estimated_total_heavy_bytes": 0,
        }


class FakeLocalMapping:
    def __init__(self):
        self._pending = 0
        self.last_num_fused_points = 0
        self.last_num_triangulated_points = 0

    def queue_size(self):
        return self._pending

    def step(self):
        self._pending = max(0, self._pending - 1)

    def wait_idle(self, timeout=0.5):
        return True

    def queue_memory_stats(self):
        return {
            "queue_size": self._pending,
            "estimated_queue_heavy_bytes": 0,
            "active_img": False,
            "active_depth": False,
        }


class FakeLoopClosing:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._pending = 1 if enabled else 0
        self.last_diagnostics = SimpleNamespace(
            global_ba_started=False,
            global_ba_success=False,
            global_ba_reason="",
            global_ba_num_edges=0,
            global_ba_num_inliers=0,
            global_ba_mean_error_after=None,
            candidate_pair_reports=[{"candidate_kf_id": 1}],
        )

    def queue_size(self):
        return self._pending

    def step(self):
        self._pending = max(0, self._pending - 1)
        return True


class FakeSlam:
    def __init__(
        self,
        *,
        enable_loop_closing=False,
        raise_on_track=False,
        recorded_maps: list[FakeMap] | None = None,
        **kwargs,
    ):
        self.map = FakeMap()
        if recorded_maps is not None:
            recorded_maps.append(self.map)
        self.tracking = FakeTracking()
        self.local_mapping = FakeLocalMapping()
        self.loop_closing = FakeLoopClosing(enable_loop_closing)
        self.start_local_mapping_thread = False
        self.runtime_profiler = None
        self._state = SlamState.OK
        self._raise_on_track = raise_on_track
        self._track_calls = 0

    def track(self, img, img_right=None, depth=None, img_id=None, timestamp=None):
        if self._raise_on_track:
            raise RuntimeError("synthetic track failure")
        self._track_calls += 1
        self.tracking.poses.append(np.eye(4))
        self.tracking.tracking_history.timestamps.append(timestamp)
        self.map._frames = 1
        return True

    def get_tracking_state(self):
        return self._state

    def get_final_trajectory(self):
        return {
            "poses": [np.eye(4)],
            "timestamps": [0.0],
            "slam_states": [SlamState.OK],
        }

    def shutdown(self):
        return None


def _install_fake_runner_environment(
    monkeypatch,
    tmp_path: Path,
    *,
    enable_loop_closing: bool = False,
    raise_on_track: bool = False,
):
    recorded_maps: list[FakeMap] = []
    dataset = tmp_path / "rgbd_dataset_freiburg1_room"
    dataset.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "visual_slam.orbslam.run_rgbd_slam.detect_dataset_type",
        lambda dataset_path: "tum_rgbd",
    )
    monkeypatch.setattr(
        "visual_slam.orbslam.run_rgbd_slam.make_rgbd_camera",
        lambda *args, **kwargs: SimpleNamespace(
            fx=517.3,
            fy=516.5,
            cx=318.6,
            cy=255.3,
            width=640,
            height=480,
            depth_factor=0.0002,
        ),
    )
    monkeypatch.setattr(
        "visual_slam.orbslam.run_rgbd_slam.resolve_camera_metadata",
        lambda *args, **kwargs: {
            "dataset_name": "rgbd_dataset_freiburg1_room",
            "camera_source": "synthetic_camera",
            "depth_map_factor": 5000.0,
            "baseline_m": 0.08,
            "baseline_source": "synthetic",
            "depth_threshold": 3.2,
            "depth_threshold_source": "synthetic",
            "bf": 41.384,
            "width": 640,
            "height": 480,
            "fx": 517.3,
            "fy": 516.5,
            "cx": 318.6,
            "cy": 255.3,
        },
    )
    monkeypatch.setattr(
        "visual_slam.orbslam.run_rgbd_slam.load_rgbd_associations",
        lambda *args, **kwargs: [
            FakeAssociationsEntry(dataset / "rgb/0001.png", dataset / "depth/0001.png", 0.0),
            FakeAssociationsEntry(dataset / "rgb/0002.png", dataset / "depth/0002.png", 0.1),
        ],
    )
    monkeypatch.setattr(
        "visual_slam.orbslam.run_rgbd_slam._load_rgb",
        lambda path: np.zeros((4, 4, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "visual_slam.orbslam.run_rgbd_slam._load_depth",
        lambda path: np.zeros((4, 4), dtype=np.uint16),
    )
    monkeypatch.setattr(
        "visual_slam.orbslam.run_rgbd_slam.save_tum_trajectory",
        lambda poses, timestamps, path: Path(path).write_text("0 0 0 0 0 0 0 1\n"),
    )
    monkeypatch.setattr(
        "visual_slam.orbslam.run_rgbd_slam.export_orbslam_map",
        lambda slam, output_dir: {
            "map_points_ply": None,
            "keyframes_json": None,
            "keyframe_graph_json": None,
        },
    )

    def _make_fake_slam(**kwargs):
        kwargs.pop("enable_loop_closing", None)
        return FakeSlam(
            enable_loop_closing=enable_loop_closing,
            raise_on_track=raise_on_track,
            recorded_maps=recorded_maps,
            **kwargs,
        )

    monkeypatch.setattr("visual_slam.orbslam.run_rgbd_slam.Slam", _make_fake_slam)
    return dataset, recorded_maps


def test_map_frames_uses_parameters_maxlen():
    guard = ParameterGuard("kMaxLenFrameDeque")
    try:
        Parameters.kMaxLenFrameDeque = 3
        slam_map = Map()
        frames = [SimpleNamespace(id=i, remove_frame_views=lambda idxs=None: None, reset_points=lambda: None,
                                  release_heavy_data=lambda **kwargs: None, release_images=lambda **kwargs: None)
                  for i in range(5)]
        for frame in frames:
            slam_map.add_frame(frame)
        assert slam_map.frames.maxlen == 3
        assert len(slam_map.frames) == 3
    finally:
        guard.restore()


def test_lean_memory_disables_pair_report_dumping(monkeypatch, tmp_path):
    dataset, _ = _install_fake_runner_environment(monkeypatch, tmp_path, enable_loop_closing=True)
    pair_report_calls = []
    monkeypatch.setattr(
        "visual_slam.orbslam.run_rgbd_slam._write_candidate_pair_reports",
        lambda *args, **kwargs: pair_report_calls.append((args, kwargs)),
    )

    run_rgbd_slam(
        dataset=dataset,
        output_dir=tmp_path / "out",
        max_frames=1,
        feature_backend="auto",
        enable_loop_closing=True,
        dump_loop_candidate_reports=True,
        lean_memory=True,
        no_map_export=True,
    )

    assert pair_report_calls == []


def test_run_rgbd_slam_restores_parameters_after_lean_memory(monkeypatch, tmp_path):
    dataset, _ = _install_fake_runner_environment(monkeypatch, tmp_path)
    guard = ParameterGuard("kStoreNormalFrameImages", "kStoreKeyFrameImages", "kWaitForLocalMappingTimeout", "kFrameViewPruneEveryNFrames")
    try:
        run_rgbd_slam(
            dataset=dataset,
            output_dir=tmp_path / "out_restore",
            max_frames=1,
            lean_memory=True,
            lm_wait_timeout=1.75,
            frame_view_prune_every=7,
            no_map_export=True,
        )
        for name, value in guard.values.items():
            assert getattr(Parameters, name) == value
    finally:
        guard.restore()


def test_run_rgbd_slam_restores_parameters_after_exception(monkeypatch, tmp_path):
    dataset, _ = _install_fake_runner_environment(monkeypatch, tmp_path, raise_on_track=True)
    guard = ParameterGuard("kStoreNormalFrameImages", "kStoreKeyFrameImages", "kWaitForLocalMappingTimeout")
    try:
        with pytest.raises(RuntimeError, match="synthetic track failure"):
            run_rgbd_slam(
                dataset=dataset,
                output_dir=tmp_path / "out_exception",
                max_frames=1,
                lean_memory=True,
                lm_wait_timeout=1.5,
                no_map_export=True,
            )
        for name, value in guard.values.items():
            assert getattr(Parameters, name) == value
    finally:
        guard.restore()


class IndexGuard:
    def __init__(self, valid_indices: set[int], value: float):
        self.valid_indices = set(valid_indices)
        self.value = value
        self.accessed: list[int] = []

    def __getitem__(self, idx):
        self.accessed.append(int(idx))
        assert int(idx) in self.valid_indices
        return self.value


class FakeObservedKeyFrame:
    def __init__(self, octave_value: int):
        self._bad = False
        self.octaves = {1: octave_value, 10: octave_value}

    def is_bad(self):
        return self._bad


class FakeCullPoint:
    def __init__(self, obs_idx: int):
        self.obs_idx = obs_idx
        self._observers = [(FakeObservedKeyFrame(0), obs_idx) for _ in range(3)]

    def is_bad(self):
        return False

    def num_observations(self):
        return 4

    def observations(self):
        return list(self._observers)


class FakeCullKeyFrame:
    def __init__(self):
        self.kid = 1
        self.timestamp = 2.0
        self.parent = SimpleNamespace(timestamp=0.0)
        self.camera = SimpleNamespace(depth_threshold=100.0)
        self.depths = IndexGuard({1, 10}, 1.0)
        self.octaves = IndexGuard({1, 10}, 0)
        self._points = [None] * 12
        self._points[1] = FakeCullPoint(1)
        self._points[10] = FakeCullPoint(10)
        self.bad = False

    def is_bad(self):
        return self.bad

    def set_bad(self):
        self.bad = True

    def get_matched_good_points_and_idxs(self):
        return [(self._points[1], 1), (self._points[10], 10)]


def test_cull_keyframes_uses_original_keypoint_indices():
    candidate_kf = FakeCullKeyFrame()
    core = LocalMappingCore(map=SimpleNamespace(), sensor_type=SensorType.RGBD)
    core.kf_cur = SimpleNamespace(get_covisible_keyframes=lambda: [candidate_kf])
    core.cull_keyframes()
    assert set(candidate_kf.depths.accessed) == {1, 10}
    assert set(candidate_kf.octaves.accessed) == {1, 10}


def test_runtime_profiler_records_sections():
    profiler = RuntimeProfiler(enabled=True)
    with profiler.section("frame.total"):
        pass
    profiler.start("tracking.track")
    profiler.stop("tracking.track")
    payload = profiler.to_dict()
    assert payload["frame.total"]["calls"] == 1
    assert payload["tracking.track"]["calls"] == 1


def test_runtime_profiler_writes_csv_and_json(tmp_path):
    profiler = RuntimeProfiler(enabled=True)
    with profiler.section("frame.total"):
        pass
    csv_path = profiler.write_csv(tmp_path / "runtime_profile.csv")
    json_path = profiler.write_json(tmp_path / "runtime_profile.json")
    assert csv_path.exists()
    assert json_path.exists()
    rows = list(csv.DictReader(csv_path.open()))
    assert rows[0]["section"] == "frame.total"
    payload = json.loads(json_path.read_text())
    assert payload["sections"][0]["section"] == "frame.total"


def test_frame_timing_csv_columns(monkeypatch, tmp_path):
    dataset, _ = _install_fake_runner_environment(monkeypatch, tmp_path)
    run_rgbd_slam(
        dataset=dataset,
        output_dir=tmp_path / "out_timing",
        max_frames=1,
        profile_runtime=True,
        no_map_export=True,
    )
    csv_path = tmp_path / "out_timing" / "frame_timing.csv"
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == FRAME_TIMING_COLUMNS


def test_memory_profile_disabled_does_not_call_deep_memory_stats(monkeypatch, tmp_path):
    dataset, recorded_maps = _install_fake_runner_environment(monkeypatch, tmp_path)
    run_rgbd_slam(
        dataset=dataset,
        output_dir=tmp_path / "out_no_mem_profile",
        max_frames=1,
        no_map_export=True,
    )
    assert recorded_maps
    assert "deep" not in recorded_maps[0].memory_stats_modes
    assert "cheap" in recorded_maps[0].memory_stats_modes


def test_prune_old_frame_views_runtime_section_exists_when_enabled(monkeypatch, tmp_path):
    dataset, _ = _install_fake_runner_environment(monkeypatch, tmp_path)
    run_rgbd_slam(
        dataset=dataset,
        output_dir=tmp_path / "out_runtime",
        max_frames=1,
        profile_runtime=True,
        no_map_export=True,
    )
    runtime_csv = tmp_path / "out_runtime" / "runtime_profile.csv"
    rows = list(csv.DictReader(runtime_csv.open()))
    sections = {row["section"] for row in rows}
    assert "memory.prune_old_frame_views" in sections
