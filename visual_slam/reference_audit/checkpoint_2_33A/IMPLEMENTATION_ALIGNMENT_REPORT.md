# Checkpoint 2.33A Implementation Alignment Report

## 1. Task/Checkpoint

Checkpoint 2.33A - pySLAM-aligned keyframe insertion and LocalMapping scheduling with BA starvation protection.

## 2. Files Inspected

- `visual_slam/orbslam/slam/tracking.py`
- `visual_slam/orbslam/slam/local_mapping.py`
- `visual_slam/orbslam/slam/local_mapping_core.py`
- `visual_slam/orbslam/slam/config_parameters.py`
- `visual_slam/orbslam/run_rgbd_slam.py`
- `visual_slam/orbslam/slam/keyframe.py`
- `visual_slam/orbslam/slam/map.py`
- `visual_slam/orbslam/slam/map_point.py`

## 3. pySLAM Files Inspected

- `third_party/pyslam_reference/pyslam/slam/tracking.py` at `a95db3a0e95764b8c68b81fade544bdd6ecb912e`
- `third_party/pyslam_reference/pyslam/slam/local_mapping.py`
- `third_party/pyslam_reference/pyslam/slam/local_mapping_core.py`
- `third_party/pyslam_reference/pyslam/slam/map.py`
- `third_party/pyslam_reference/pyslam/slam/keyframe.py`
- `third_party/pyslam_reference/pyslam/slam/map_point.py`
- `third_party/pyslam_reference/pyslam/slam/optimizer_g2o.py`

## 4. Root Cause

The local sequential runner previously made LocalMapping look idle before nearly every keyframe decision, hard-clamped RGB-D keyframe spacing to 3 frames, and ran fusion plus Local BA after every new keyframe. This produced dense keyframe insertion and made LocalMapping dominate runtime.

## 5. Exact Changes Made

- Added keyframe decision and LocalMapping scheduling CSV diagnostics.
- Added Local BA counters to `run_summary.json`.
- Added configurable FPS-aware sequential RGB-D keyframe spacing.
- Added LocalMapping `accept_keyframes`, `set_accept_keyframes`, `keyframes_in_queue`, and `check_new_keyframes`.
- Reworked `Tracking.need_new_keyframe` to use pySLAM-style condition ordering, automatic reference min observations, mapper acceptance, queue pressure, and interrupt requests.
- Made LocalMapping fusion and Local BA conditional on queue/thread state while preserving the pySLAM single-thread exception.
- Added Local BA starvation guard counters and forced BA windows.
- Verified keyframe culling already uses original keypoint indices and added tests.

## 6. Gap Alignment Scores

Gap H - diagnostics complete:

- pySLAM reference: diagnostic-free, but scheduling state comes from `Tracking.need_new_keyframe` and `LocalMapping.do_local_mapping`.
- Local before: no keyframe-decision or LocalMapping schedule CSVs.
- Local after: `keyframe_decision_log.csv`, `local_mapping_schedule_log.csv`, and run-summary BA counters.
- Alignment score: 99%.
- Deviation: diagnostics are local additions.
- Test evidence: `test_keyframe_decision_log_has_required_columns`, `test_local_mapping_schedule_log_has_required_columns`, `test_run_summary_has_local_ba_schedule_counters`.

Gap G - keyframe culling original-index correctness:

- pySLAM reference: `LocalMappingCore.cull_keyframes`.
- Local before: current source already used `get_matched_good_points_and_idxs()`.
- Local after: unchanged behavior, regression tested.
- Alignment score: 100%.
- Deviation: none in current local source.
- Test evidence: `test_cull_keyframes_uses_original_keypoint_indices`, `test_cull_keyframes_does_not_use_compact_get_points_index`.

Gap A - LocalMapping backpressure state:

- pySLAM/ORB-SLAM2 concept: mapper idle/queue/acceptance state gates keyframe insertion.
- Local before: `need_new_keyframe` used `is_idle()` only.
- Local after: LocalMapping exposes `accept_keyframes`, `set_accept_keyframes`, `keyframes_in_queue`, and `check_new_keyframes`; accept state is false during processing and restored in `finally`.
- Alignment score: 98%.
- Deviation: explicit accept flag is an ORB-SLAM2-style addition beyond this pySLAM reference's idle flag.
- Test evidence: `test_local_mapping_accept_keyframes_state_changes_during_processing`, `test_keyframes_in_queue_reports_pending_queue_size`, `test_accept_keyframes_restored_after_exception`.

Gap B - sequential min-frame spacing:

- pySLAM reference: spacing uses min/max frame intervals, mapper state, and tracking quality; local pySLAM copy also has a 3-frame single-thread note.
- Local before: hardcoded sequential RGB-D spacing of 3.
- Local after: `kMinFramesBetweenKeyframesSequentialRgbd`, `kMinFramesBetweenKeyframesThreadedRgbd`, `kUseFpsAwareKeyframeSpacing`, and `kMinKeyframeSpacingSeconds`; default sequential RGB-D spacing is `max(3, int(0.30 * fps))`.
- Alignment score: 97%.
- Deviation: Python sequential mode intentionally uses FPS-aware spacing to compensate for slow local mapping.
- Test evidence: `test_sequential_min_keyframe_spacing_uses_parameter_or_fps`, `test_sequential_mode_no_longer_forces_min_frames_three`, `test_emergency_close_point_condition_can_still_request_keyframe`.

Gap C - queue-size keyframe admission:

- pySLAM/ORB-SLAM2 concept: if mapper is not accepting, interrupt optimization and only admit under sensor/queue pressure rules.
- Local before: busy LocalMapping could force insert on time/close conditions without queue-size admission.
- Local after: final insertion requires mapper acceptance, or RGB-D queue size below `kLocalMappingMaxQueueForForcedInsert` after interrupting optimization.
- Alignment score: 98%.
- Deviation: RGB-D forced insert under low queue pressure follows the checkpoint target and is more permissive than this local pySLAM reference's non-monocular busy branch.
- Test evidence: `test_keyframe_inserted_when_mapper_accepts_and_conditions_true`, `test_keyframe_rejected_when_mapper_busy_and_queue_too_large`, `test_rgbd_forced_insert_allowed_when_queue_below_threshold`, `test_interrupt_optimization_called_when_mapper_busy`.

Gap D - conditional expensive LocalMapping stages:

- pySLAM reference: `fuse_map_points` and `local_BA` run only if no new keyframes are waiting or in single-thread mode.
- Local before: fusion and Local BA were unconditional.
- Local after: fusion and Local BA are skipped under threaded queue pressure, while single-thread mode still runs them as pySLAM requires.
- Alignment score: 98%.
- Deviation: Local BA can also be forced by the starvation guard.
- Test evidence: `test_fuse_runs_when_queue_empty`, `test_fuse_skipped_when_queue_pending_in_threaded_mode`, `test_fuse_runs_in_single_thread_mode_even_if_queue_pending_if_pyslam_policy_requires`, `test_local_ba_runs_when_queue_empty`, `test_local_ba_skipped_when_queue_pending_in_threaded_mode_unless_forced`.

Gap E - Local BA abort and starvation protection:

- pySLAM reference: new keyframes set the abort flag; Local BA is skipped under queue pressure in threaded mode but always allowed in single-thread mode.
- Local before: abort flag existed, but no completion/skip/abort/starvation accounting existed.
- Local after: Local BA counters track started/completed/aborted/skipped/forced; starvation threshold forces BA; repeated aborts open a completion window that temporarily stops normal acceptance.
- Alignment score: 96%.
- Deviation: starvation protection is a deliberate Python-safe adaptation.
- Test evidence: `test_local_ba_starvation_guard_forces_ba_after_threshold`, `test_consecutive_ba_aborts_trigger_ba_completion_window`, `test_local_ba_counters_update_for_completed_skipped_aborted`, `test_local_ba_not_starved_when_keyframes_arrive_frequently`.

Gap F - reference tracked count `nMinObs`:

- pySLAM reference: `nMinObs = 2` for maps with two or fewer keyframes, else `3`.
- Local before: hardcoded `nMinObs = 1`.
- Local after: `kNewKeyframeRefMinObs = -1` enables automatic pySLAM behavior; explicit overrides remain available for experiments.
- Alignment score: 99%.
- Deviation: configurable override exists but default is pySLAM automatic.
- Test evidence: `test_ref_min_obs_auto_uses_two_for_small_map`, `test_ref_min_obs_auto_uses_three_for_larger_map`, `test_ref_min_obs_override_is_respected`.

Gap A/B/C combined - pySLAM-style keyframe decision:

- pySLAM reference: stopped mapper and relocalization guards, reference/current tracked counts, close-point need, `c1a/c1b/c1c`, `c2`, mapper state, queue/backpressure.
- Local before: same broad conditions but hardcoded spacing, `nMinObs=1`, and no acceptance/queue admission.
- Local after: condition computation and diagnostic rows follow the target ordering and include accept/queue final admission.
- Alignment score: 98%.
- Deviation: Python sequential spacing and RGB-D queue-forced insert are checkpoint-directed adaptations.
- Test evidence: `test_need_new_keyframe_computes_c1_c2_conditions`, `test_need_new_keyframe_reason_is_logged`, `test_need_new_keyframe_respects_mapper_backpressure`, `test_need_new_keyframe_allows_max_frame_interval`.

## 7. Tests Added/Updated

- `tests/visual_slam/orbslam/test_checkpoint_2_33A_kf_local_mapping_schedule.py`

## 8. Test Commands Run

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_33A_kf_local_mapping_schedule.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam -k "not cpp_slam_core"
```

Results:

- Targeted: 31 passed.
- Non-C++ visual SLAM slice: 309 passed, 1 skipped, 94 deselected.

## 9. Dataset Validation

Baseline and post-change validation reports are written separately in this folder.

## 10. Remaining Risks

- The C++ LocalMappingCore path remains out of scope and unmodified.
- Sequential mode still runs Local BA for each inserted keyframe by design; runtime improvement comes primarily from pySLAM-style insertion selectivity.
- Loop closure ran during the 300-frame validation but no loop was accepted; loop thresholds were intentionally not changed.

## 11. Next Recommended Action

Use the 300-frame profile to decide whether the next checkpoint should address remaining `cull_map_points` cost or loop-recall diagnostics.
