# Checkpoint 2.33A Pre-Change pySLAM KF/LocalMapping Audit

## 1. pySLAM Files/Functions Inspected

- `third_party/pyslam_reference/pyslam/slam/tracking.py` at commit `a95db3a0e95764b8c68b81fade544bdd6ecb912e`
  - `Tracking.need_new_keyframe`
  - `Tracking.create_new_keyframe`
- `third_party/pyslam_reference/pyslam/slam/local_mapping.py` at the same commit
  - `LocalMapping.push_keyframe`
  - `LocalMapping.pop_keyframe`
  - `LocalMapping.queue_size`
  - `LocalMapping.is_idle`
  - `LocalMapping.set_idle`
  - `LocalMapping.interrupt_optimization`
  - `LocalMapping.set_do_not_stop`
  - `LocalMapping.step`
  - `LocalMapping.do_local_mapping`
  - `LocalMapping.local_BA`
  - `LocalMapping.cull_map_points`
  - `LocalMapping.fuse_map_points`
  - `LocalMapping.cull_keyframes`
- `third_party/pyslam_reference/pyslam/slam/local_mapping_core.py`
  - `LocalMappingCore.cull_keyframes`
  - `LocalMappingCore.local_BA`
- `third_party/pyslam_reference/pyslam/slam/keyframe.py`
  - `KeyFrame.num_tracked_points`
- `third_party/pyslam_reference/pyslam/slam/map.py`
  - local optimization entry points
- `third_party/pyslam_reference/pyslam/slam/map_point.py`
  - observation semantics relevant to tracked-point counts
- `third_party/pyslam_reference/pyslam/slam/optimizer_g2o.py`
  - Local BA abort-flag behavior

## 2. Local Files/Functions Inspected

- `visual_slam/orbslam/slam/tracking.py`
  - `Tracking.__init__`
  - `Tracking.need_new_keyframe`
  - `Tracking.create_new_keyframe`
- `visual_slam/orbslam/slam/local_mapping.py`
  - `LocalMapping.push_keyframe`
  - `LocalMapping.pop_keyframe`
  - `LocalMapping.queue_size`
  - `LocalMapping.is_idle`
  - `LocalMapping.step`
  - `LocalMapping.do_local_mapping`
  - `LocalMapping.local_BA`
  - `LocalMapping.fuse_map_points`
  - `LocalMapping.cull_keyframes`
- `visual_slam/orbslam/slam/local_mapping_core.py`
  - `LocalMappingCore.local_BA`
  - `LocalMappingCore.cull_keyframes`
- `visual_slam/orbslam/slam/config_parameters.py`
- `visual_slam/orbslam/run_rgbd_slam.py`
- `visual_slam/orbslam/slam/frame.py`
- `visual_slam/orbslam/slam/keyframe.py`
- `visual_slam/orbslam/slam/map.py`
- `visual_slam/orbslam/slam/map_point.py`

## 3. pySLAM Keyframe Insertion Workflow

`Tracking.need_new_keyframe` first rejects insertion when LocalMapping is stopped or stop-requested. It suppresses insertion after recent relocalization when the map is already large enough. It then computes the reference keyframe tracked-point count with `nMinObs = 2` for small maps and `3` otherwise, computes current inlier map-point matches, close-point starvation, reference ratio thresholds, and conditions `cond1a`, `cond1b`, `cond1c`, optional coverage `cond1d`, optional FOV-center `cond3`, and `cond2`.

When the compound condition passes, pySLAM inserts immediately if LocalMapping is idle. If LocalMapping is not idle, it may interrupt optimization depending on configuration/C++ use and applies queue/backpressure policy before returning. The local reference code in this workspace still contains a single-thread RGB-D spacing clamp to 3 frames; the checkpoint request intentionally asks the local implementation to improve this for the slower Python sequential runner while preserving the same gating concepts.

## 4. pySLAM LocalMapping Scheduling Workflow

`LocalMapping.push_keyframe` queues a keyframe and sets the local optimizer abort flag. `LocalMapping.step` pops work, marks the mapper non-idle while processing, and marks it idle afterward. `LocalMapping.do_local_mapping` runs:

1. `process_new_keyframe`
2. `cull_map_points`
3. `create_new_map_points`
4. `fuse_map_points` only when the queue is empty or in single-thread mode
5. reset optimizer abort flag
6. `local_BA` only when the queue is empty and no stop is requested, or in single-thread mode
7. keyframe culling inside the same Local BA branch
8. downstream semantic/loop/integration queueing

This preserves throughput in threaded mode but keeps full local refinement in single-thread mode.

## 5. pySLAM Local BA Abort/Single-Thread Behavior

pySLAM sets the optimizer abort flag when a new keyframe is queued and clears it immediately before Local BA. In threaded mode, newly queued keyframes can interrupt Local BA. In single-thread mode, the explicit `or is_single_thread` branches still run fusion and Local BA even when more work is pending. The local Python implementation is much slower, so this checkpoint needs a guarded deviation: Local BA may be skipped/deferred due to queue pressure only under controlled conditions, and starvation counters must force successful Local BA regularly.

## 6. Local Current Keyframe Insertion Workflow

The local `Tracking.need_new_keyframe` currently:

- rejects monocular and stopped LocalMapping cases
- uses only `LocalMapping.is_idle()` for mapper state
- hard-codes `nMinObs = 1` for the reference tracked count
- computes close-point starvation and `c1a`, `c1b`, `c1c`, `c2`
- hard-clamps non-threaded spacing to `self.min_frames_between_kfs = 3`
- inserts immediately if `is_idle`
- if busy, interrupts optimization and force-inserts for max-frame fallback or close-point starvation

It does not emit keyframe-decision diagnostics and does not apply queue-size admission in a pySLAM/ORB-SLAM2-style way.

## 7. Local Current LocalMapping Scheduling Workflow

The local `LocalMapping.do_local_mapping` currently runs `process_new_keyframe`, point culling, new point creation, fusion, Local BA, and keyframe culling unconditionally for every popped keyframe. It exposes `queue_size()` and `is_idle()` but not explicit accept-keyframe state, `check_new_keyframes()`, `keyframes_in_queue()`, or Local BA scheduling counters. It does not emit schedule diagnostics.

## 8. Gap-by-Gap Confirmation/Refutation

- Gap A confirmed: Tracking sees `is_idle()` only. No explicit accept-keyframes/backpressure state exists.
- Gap B confirmed: non-threaded RGB-D currently forces `min_frames_between_kfs = 3`.
- Gap C confirmed: busy LocalMapping admission is not gated by queue size in the requested pySLAM-style structure.
- Gap D confirmed: local fusion and Local BA run unconditionally.
- Gap E confirmed: abort flag exists, but Local BA completion/abort/skip/starvation counters and forced BA windows do not exist.
- Gap F confirmed: local reference tracked count currently uses `nMinObs = 1`.
- Gap G refuted for current source: `LocalMappingCore.cull_keyframes` already uses `kf.get_matched_good_points_and_idxs()` and indexes `depths`/`octaves` with original keypoint indices. A regression test is still needed.
- Gap H confirmed: keyframe decision and LocalMapping schedule logs are missing.

## 9. Exact Implementation Plan

1. Add diagnostics first, without behavior changes:
   - keyframe decision rows in `Tracking`
   - LocalMapping schedule rows and Local BA counters in `LocalMapping`
   - `--profile-keyframes` runner flag, automatic enable with `--profile-runtime`
   - `keyframe_decision_log.csv`, `local_mapping_schedule_log.csv`, and run-summary counters
2. Run required 3-frame and 30-frame baseline diagnostics.
3. Implement Gap H tests for log schemas and counters.
4. Add Gap G regression tests proving original keypoint indices are used.
5. Add LocalMapping accept/backpressure methods and restore accept state with `finally`.
6. Replace sequential min-frame hard clamp with FPS-aware/configurable spacing.
7. Rework `need_new_keyframe` to use pySLAM-style condition ordering plus mapper accept/queue admission.
8. Make LocalMapping fusion and Local BA conditional on queue/thread state, while preserving pySLAM single-thread behavior.
9. Add Local BA starvation guard counters and forced completion windows.
10. Restore pySLAM automatic `nMinObs` behavior with an override parameter.
11. Add targeted tests for all scheduling gates and counters.
12. Run targeted and non-C++ visual SLAM test slices.
13. Run post-change 30-frame validation and then, if stable, 300-frame validation.

## 10. Risks and Safeguards

- Risk: FPS-aware sequential spacing may reduce keyframe density too much on difficult RGB-D sections. Safeguard: keep max-frame fallback, weak-tracking, and close-point starvation insertion paths.
- Risk: Local BA skips in threaded queue-pressure cases could starve BA. Safeguard: counters, `kMaxKeyframesWithoutLocalBA`, and forced BA windows.
- Risk: abort handling is less meaningful in sequential Python than pySLAM threaded mode. Safeguard: clear abort only at controlled BA start and count aborted/skipped/completed decisions explicitly.
- Risk: diagnostics could perturb runtime. Safeguard: append small dict rows in memory and write once per run when profiling is enabled.
