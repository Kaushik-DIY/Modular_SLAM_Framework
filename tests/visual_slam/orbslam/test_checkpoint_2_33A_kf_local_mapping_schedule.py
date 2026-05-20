from types import SimpleNamespace

import numpy as np
import pytest

from visual_slam.orbslam.run_rgbd_slam import (
    KEYFRAME_DECISION_COLUMNS,
    LOCAL_MAPPING_SCHEDULE_COLUMNS,
    build_run_summary,
)
from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.local_mapping import LocalMapping
from visual_slam.orbslam.slam.sensor_types import SensorType
from visual_slam.orbslam.slam.tracking import Tracking


class FakeCamera:
    fps = 30
    depth_threshold = 4.0


class FakeMap:
    def __init__(self, keyframes=None):
        self.keyframes = list(keyframes or [])

    def num_keyframes(self):
        return len(self.keyframes)

    def get_keyframes(self):
        return list(self.keyframes)


class FakeKeyFrame:
    def __init__(self, kid=0, tracked=100):
        self.id = kid
        self.kid = kid
        self.timestamp = float(kid)
        self.tracked = tracked

    def is_bad(self):
        return False

    def num_tracked_points(self, min_num_observations=0):
        return self.tracked


class FakeFrame:
    def __init__(self, frame_id=10, tracked_points=50, total_points=150):
        self.id = frame_id
        self.timestamp = float(frame_id)
        self.camera = FakeCamera()
        self.points = [object()] * tracked_points + [None] * (total_points - tracked_points)
        self.outliers = np.zeros(total_points, dtype=bool)
        self.depths = np.full(total_points, 1.0, dtype=np.float32)


class FakeLocalMapping:
    def __init__(self, accepting=True, idle=True, queue_size=0):
        self._accepting = accepting
        self._idle = idle
        self._queue_size = queue_size
        self.interrupts = 0
        self.stopped = False
        self.stop_requested = False
        self.local_mapping_core = SimpleNamespace(opt_abort_flag=SimpleNamespace(value=False))

    def is_idle(self):
        return self._idle

    def accept_keyframes(self):
        return self._accepting

    def keyframes_in_queue(self):
        return self._queue_size

    def queue_size(self):
        return self._queue_size

    def interrupt_optimization(self):
        self.interrupts += 1
        self.local_mapping_core.opt_abort_flag.value = True


class FakeSlam:
    def __init__(self, local_mapping=None, keyframes=None):
        self.camera = FakeCamera()
        self.sensor_type = SensorType.RGBD
        self.map = FakeMap(keyframes)
        self.local_mapping = local_mapping
        self.runtime_profiler = None


def make_tracking(
    *,
    local_mapping=None,
    frame_id=10,
    last_kf_id=0,
    num_keyframes=3,
    num_matched=50,
    ref_tracked=100,
    tracked_close=50,
    total_points=150,
):
    kf_last = FakeKeyFrame(last_kf_id, tracked=ref_tracked)
    kfs = [FakeKeyFrame(i, tracked=ref_tracked) for i in range(num_keyframes - 1)] + [kf_last]
    tracking = object.__new__(Tracking)
    tracking.slam = FakeSlam(local_mapping=local_mapping, keyframes=kfs)
    tracking.kf_ref = kf_last
    tracking.kf_last = kf_last
    tracking.f_cur = FakeFrame(frame_id=frame_id, tracked_points=tracked_close, total_points=total_points)
    tracking.num_matched_map_points = num_matched
    tracking.max_frames_between_kfs = 30
    tracking.max_frames_between_kfs_after_reloc = 30
    tracking.min_frames_between_kfs = 0
    tracking.last_reloc_frame_id = -10_000
    tracking.profile_keyframes = True
    tracking.keyframe_decision_rows = []
    return tracking


def make_local_mapping(monkeypatch, *, threaded=False, queue_pending=False):
    slam = SimpleNamespace(
        map=FakeMap([FakeKeyFrame(0)]),
        sensor_type=SensorType.RGBD,
        tracking=SimpleNamespace(descriptor_distance_sigma=30.0, num_kf_ref_tracked_points=0),
        loop_closing=None,
    )
    lm = LocalMapping(slam)
    lm.profile_keyframes = True
    lm.kf_cur = FakeKeyFrame(2)
    monkeypatch.setattr(lm, "_is_single_thread", lambda: not threaded)
    monkeypatch.setattr(lm, "process_new_keyframe", lambda: None)
    monkeypatch.setattr(lm, "cull_map_points", lambda: 1)
    monkeypatch.setattr(lm, "create_new_map_points", lambda: 2)
    monkeypatch.setattr(lm, "fuse_map_points", lambda: 3)
    monkeypatch.setattr(lm, "local_BA", lambda: None)
    monkeypatch.setattr(lm, "cull_keyframes", lambda: 4)
    if queue_pending:
        lm.queue.put((FakeKeyFrame(3), None, None, None))
    return lm


def test_keyframe_decision_log_has_required_columns():
    assert set(KEYFRAME_DECISION_COLUMNS) == {
        "frame_id", "timestamp", "num_keyframes", "last_keyframe_id",
        "frames_since_last_kf", "min_frames_between_kfs", "max_frames_between_kfs",
        "sensor_type", "local_mapping_idle", "local_mapping_accepting",
        "local_mapping_queue_size", "local_mapping_abort_requested",
        "num_ref_tracked", "ref_min_obs", "num_matched_cur", "num_tracked_close",
        "num_non_tracked_close", "need_to_insert_close", "ref_ratio", "th_ref_ratio",
        "c1a", "c1b", "c1c", "c2", "inserted", "insert_reason", "reject_reason",
    }


def test_local_mapping_schedule_log_has_required_columns():
    assert set(LOCAL_MAPPING_SCHEDULE_COLUMNS) == {
        "kf_id", "timestamp", "queue_size_before", "queue_size_after",
        "accept_keyframes_before", "accept_keyframes_after", "is_single_thread",
        "processed_new_keyframe", "ran_cull_map_points", "ran_create_new_map_points",
        "ran_fuse_map_points", "ran_local_BA", "ran_cull_keyframes",
        "skipped_fuse_reason", "skipped_local_BA_reason", "local_BA_started",
        "local_BA_completed", "local_BA_aborted", "local_BA_forced_due_starvation",
        "keyframes_since_last_successful_ba", "local_BA_sec", "total_step_sec",
    }


def test_run_summary_has_local_ba_schedule_counters():
    summary = build_run_summary(
        dataset_name="d", dataset_type="tum_rgbd", frames_attempted=1,
        tracking_ok_count=1, tracking_lost_count=0, errors=0, final_state="OK",
        keyframes=1, map_points=2, trajectory_poses=1, elapsed_sec=1.0, avg_fps=1.0,
        feature_backend="pyslam_orb2", enable_loop_closing=False, enable_global_ba=False,
        global_ba_after_loop=False, loop_debug_events=0, accepted_loops=0,
        completed_timestamp="now", standardized_output_stem="stem",
        output_files={
            "local_ba_started_count": 1,
            "local_ba_completed_count": 1,
            "local_ba_aborted_count": 0,
            "local_ba_skipped_due_queue_count": 2,
            "local_ba_forced_due_starvation_count": 3,
            "last_successful_local_ba_kid": 4,
            "keyframes_since_last_successful_ba": 5,
            "consecutive_local_ba_aborts": 6,
        },
    )
    assert summary["local_ba_started_count"] == 1
    assert summary["local_ba_completed_count"] == 1
    assert summary["local_ba_skipped_due_queue_count"] == 2
    assert summary["local_ba_forced_due_starvation_count"] == 3
    assert summary["consecutive_local_ba_aborts"] == 6


def test_cull_keyframes_uses_original_keypoint_indices():
    import inspect
    from visual_slam.orbslam.slam.local_mapping_core import LocalMappingCore

    source = inspect.getsource(LocalMappingCore.cull_keyframes)
    assert "get_matched_good_points_and_idxs" in source
    assert "kf.depths[i]" in source
    assert "kf.octaves[i]" in source


def test_cull_keyframes_does_not_use_compact_get_points_index():
    import inspect
    from visual_slam.orbslam.slam.local_mapping_core import LocalMappingCore

    source = inspect.getsource(LocalMappingCore.cull_keyframes)
    assert "enumerate(kf.get_points())" not in source


def test_local_mapping_accept_keyframes_state_changes_during_processing(monkeypatch):
    seen = []
    lm = make_local_mapping(monkeypatch)
    monkeypatch.setattr(lm, "process_new_keyframe", lambda: seen.append(lm.accept_keyframes()))
    lm.do_local_mapping()
    assert seen == [False]
    assert lm.accept_keyframes() is True


def test_keyframes_in_queue_reports_pending_queue_size(monkeypatch):
    lm = make_local_mapping(monkeypatch, queue_pending=True)
    assert lm.keyframes_in_queue() == 1
    assert lm.check_new_keyframes() is True


def test_accept_keyframes_restored_after_exception(monkeypatch):
    lm = make_local_mapping(monkeypatch)
    monkeypatch.setattr(lm, "process_new_keyframe", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        lm.do_local_mapping()
    assert lm.accept_keyframes() is True


def test_sequential_min_keyframe_spacing_uses_parameter_or_fps(monkeypatch):
    tracking = make_tracking()
    monkeypatch.setattr(Parameters, "kMinFramesBetweenKeyframesSequentialRgbd", 7)
    assert tracking._compute_min_frames_between_keyframes(is_threaded=False) == 7


def test_sequential_mode_no_longer_forces_min_frames_three(monkeypatch):
    tracking = make_tracking()
    monkeypatch.setattr(Parameters, "kMinFramesBetweenKeyframesSequentialRgbd", -1)
    monkeypatch.setattr(Parameters, "kUseFpsAwareKeyframeSpacing", True)
    monkeypatch.setattr(Parameters, "kMinKeyframeSpacingSeconds", 0.30)
    assert tracking._compute_min_frames_between_keyframes(is_threaded=False) == 9


def test_emergency_close_point_condition_can_still_request_keyframe(monkeypatch):
    monkeypatch.setattr(Parameters, "kMinFramesBetweenKeyframesSequentialRgbd", 9)
    lm = FakeLocalMapping(accepting=True, idle=True)
    tracking = make_tracking(local_mapping=lm, frame_id=1, last_kf_id=0, num_matched=50, tracked_close=0, total_points=200)
    assert tracking.need_new_keyframe() is True
    assert tracking.keyframe_decision_rows[-1]["need_to_insert_close"] is True


def test_keyframe_inserted_when_mapper_accepts_and_conditions_true(monkeypatch):
    monkeypatch.setattr(Parameters, "kMinFramesBetweenKeyframesSequentialRgbd", 0)
    tracking = make_tracking(local_mapping=FakeLocalMapping(accepting=True, idle=True))
    assert tracking.need_new_keyframe() is True
    assert tracking.keyframe_decision_rows[-1]["insert_reason"] == "local_mapping_accepting"


def test_keyframe_rejected_when_mapper_busy_and_queue_too_large(monkeypatch):
    monkeypatch.setattr(Parameters, "kMinFramesBetweenKeyframesSequentialRgbd", 0)
    monkeypatch.setattr(Parameters, "kLocalMappingMaxQueueForForcedInsert", 3)
    lm = FakeLocalMapping(accepting=False, idle=False, queue_size=3)
    tracking = make_tracking(local_mapping=lm)
    assert tracking.need_new_keyframe() is False
    assert tracking.keyframe_decision_rows[-1]["reject_reason"] == "local_mapping_busy_queue_pressure"


def test_rgbd_forced_insert_allowed_when_queue_below_threshold(monkeypatch):
    monkeypatch.setattr(Parameters, "kMinFramesBetweenKeyframesSequentialRgbd", 0)
    monkeypatch.setattr(Parameters, "kLocalMappingMaxQueueForForcedInsert", 3)
    lm = FakeLocalMapping(accepting=False, idle=False, queue_size=2)
    tracking = make_tracking(local_mapping=lm)
    assert tracking.need_new_keyframe() is True
    assert tracking.keyframe_decision_rows[-1]["insert_reason"] == "busy_rgbd_queue_below_threshold"


def test_interrupt_optimization_called_when_mapper_busy(monkeypatch):
    monkeypatch.setattr(Parameters, "kMinFramesBetweenKeyframesSequentialRgbd", 0)
    lm = FakeLocalMapping(accepting=False, idle=False, queue_size=3)
    tracking = make_tracking(local_mapping=lm)
    tracking.need_new_keyframe()
    assert lm.interrupts == 1


def test_fuse_runs_when_queue_empty(monkeypatch):
    lm = make_local_mapping(monkeypatch, threaded=True, queue_pending=False)
    lm.do_local_mapping()
    assert lm.schedule_log_rows[-1]["ran_fuse_map_points"] is True


def test_fuse_skipped_when_queue_pending_in_threaded_mode(monkeypatch):
    lm = make_local_mapping(monkeypatch, threaded=True, queue_pending=True)
    lm.do_local_mapping()
    assert lm.schedule_log_rows[-1]["ran_fuse_map_points"] is False
    assert lm.schedule_log_rows[-1]["skipped_fuse_reason"] == "queue_pending_threaded"


def test_fuse_runs_in_single_thread_mode_even_if_queue_pending_if_pyslam_policy_requires(monkeypatch):
    lm = make_local_mapping(monkeypatch, threaded=False, queue_pending=True)
    lm.do_local_mapping()
    assert lm.schedule_log_rows[-1]["ran_fuse_map_points"] is True


def test_local_ba_runs_when_queue_empty(monkeypatch):
    lm = make_local_mapping(monkeypatch, threaded=True, queue_pending=False)
    lm.do_local_mapping()
    assert lm.schedule_log_rows[-1]["ran_local_BA"] is True


def test_local_ba_skipped_when_queue_pending_in_threaded_mode_unless_forced(monkeypatch):
    lm = make_local_mapping(monkeypatch, threaded=True, queue_pending=True)
    lm.do_local_mapping()
    assert lm.schedule_log_rows[-1]["ran_local_BA"] is False
    assert lm.local_ba_skipped_due_queue_count == 1


def test_local_ba_starvation_guard_forces_ba_after_threshold(monkeypatch):
    monkeypatch.setattr(Parameters, "kMaxKeyframesWithoutLocalBA", 1)
    lm = make_local_mapping(monkeypatch, threaded=True, queue_pending=True)
    lm.keyframes_since_last_successful_ba = 1
    lm.do_local_mapping()
    assert lm.schedule_log_rows[-1]["ran_local_BA"] is True
    assert lm.schedule_log_rows[-1]["local_BA_forced_due_starvation"] is True


def test_consecutive_ba_aborts_trigger_ba_completion_window(monkeypatch):
    monkeypatch.setattr(Parameters, "kMaxConsecutiveLocalBAAborts", 1)
    lm = make_local_mapping(monkeypatch)

    def aborting_ba():
        lm.set_opt_abort_flag(True)

    monkeypatch.setattr(lm, "local_BA", aborting_ba)
    lm.do_local_mapping()
    assert lm._local_ba_completion_window is True
    assert lm.accept_keyframes() is False


def test_local_ba_counters_update_for_completed_skipped_aborted(monkeypatch):
    completed = make_local_mapping(monkeypatch)
    completed.do_local_mapping()
    assert completed.local_ba_started_count == 1
    assert completed.local_ba_completed_count == 1

    skipped = make_local_mapping(monkeypatch, threaded=True, queue_pending=True)
    skipped.do_local_mapping()
    assert skipped.local_ba_skipped_due_queue_count == 1

    aborted = make_local_mapping(monkeypatch)
    monkeypatch.setattr(aborted, "local_BA", lambda: aborted.set_opt_abort_flag(True))
    aborted.do_local_mapping()
    assert aborted.local_ba_aborted_count == 1


def test_local_ba_not_starved_when_keyframes_arrive_frequently(monkeypatch):
    monkeypatch.setattr(Parameters, "kMaxKeyframesWithoutLocalBA", 5)
    lm = make_local_mapping(monkeypatch, threaded=True, queue_pending=True)
    lm.keyframes_since_last_successful_ba = 0
    lm.do_local_mapping()
    assert lm.schedule_log_rows[-1]["local_BA_forced_due_starvation"] is False


def test_ref_min_obs_auto_uses_two_for_small_map(monkeypatch):
    monkeypatch.setattr(Parameters, "kNewKeyframeRefMinObs", -1)
    assert Tracking._new_keyframe_ref_min_obs(2) == 2


def test_ref_min_obs_auto_uses_three_for_larger_map(monkeypatch):
    monkeypatch.setattr(Parameters, "kNewKeyframeRefMinObs", -1)
    assert Tracking._new_keyframe_ref_min_obs(3) == 3


def test_ref_min_obs_override_is_respected(monkeypatch):
    monkeypatch.setattr(Parameters, "kNewKeyframeRefMinObs", 1)
    assert Tracking._new_keyframe_ref_min_obs(10) == 1


def test_need_new_keyframe_computes_c1_c2_conditions(monkeypatch):
    monkeypatch.setattr(Parameters, "kMinFramesBetweenKeyframesSequentialRgbd", 0)
    tracking = make_tracking(local_mapping=FakeLocalMapping(accepting=True, idle=True))
    assert tracking.need_new_keyframe() is True
    row = tracking.keyframe_decision_rows[-1]
    assert row["c1b"] is True
    assert row["c2"] is True


def test_need_new_keyframe_reason_is_logged(monkeypatch):
    monkeypatch.setattr(Parameters, "kMinFramesBetweenKeyframesSequentialRgbd", 0)
    tracking = make_tracking(local_mapping=FakeLocalMapping(accepting=True, idle=True))
    tracking.need_new_keyframe()
    assert tracking.keyframe_decision_rows[-1]["insert_reason"]


def test_need_new_keyframe_respects_mapper_backpressure(monkeypatch):
    monkeypatch.setattr(Parameters, "kMinFramesBetweenKeyframesSequentialRgbd", 0)
    tracking = make_tracking(local_mapping=FakeLocalMapping(accepting=False, idle=False, queue_size=99))
    assert tracking.need_new_keyframe() is False


def test_need_new_keyframe_allows_max_frame_interval(monkeypatch):
    monkeypatch.setattr(Parameters, "kMinFramesBetweenKeyframesSequentialRgbd", 9)
    tracking = make_tracking(
        local_mapping=FakeLocalMapping(accepting=True, idle=False),
        frame_id=31,
        last_kf_id=0,
    )
    assert tracking.need_new_keyframe() is True
    assert tracking.keyframe_decision_rows[-1]["c1a"] is True
