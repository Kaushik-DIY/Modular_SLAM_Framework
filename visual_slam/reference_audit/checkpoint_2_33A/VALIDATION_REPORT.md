# Checkpoint 2.33A Validation Report

## 1. Task/Checkpoint

Checkpoint 2.33A - pySLAM-aligned keyframe insertion and LocalMapping scheduling with BA starvation protection.

## 2. Files Inspected

- `visual_slam/orbslam/slam/tracking.py`
- `visual_slam/orbslam/slam/local_mapping.py`
- `visual_slam/orbslam/slam/local_mapping_core.py`
- `visual_slam/orbslam/slam/config_parameters.py`
- `visual_slam/orbslam/run_rgbd_slam.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_33A_kf_local_mapping_schedule.py`

## 3. pySLAM Files Inspected

- `third_party/pyslam_reference/pyslam/slam/tracking.py`
- `third_party/pyslam_reference/pyslam/slam/local_mapping.py`
- `third_party/pyslam_reference/pyslam/slam/local_mapping_core.py`
- `third_party/pyslam_reference/pyslam/slam/map.py`
- `third_party/pyslam_reference/pyslam/slam/keyframe.py`
- `third_party/pyslam_reference/pyslam/slam/map_point.py`
- `third_party/pyslam_reference/pyslam/slam/optimizer_g2o.py`

## 4. Root Cause

Sequential LocalMapping was effectively idle before most decisions, `min_frames_between_kfs` was forced to 3, and LocalMapping performed expensive stages after almost every inserted keyframe. This created dense keyframes and excessive LocalMapping runtime.

## 5. Exact Changes Made

- Added scheduler diagnostics and run-summary BA counters.
- Added pySLAM/ORB-SLAM2-style LocalMapping acceptance and queue state.
- Replaced hardcoded sequential 3-frame spacing with configurable FPS-aware spacing.
- Reworked keyframe admission around mapper acceptance, queue pressure, and automatic reference min observations.
- Made expensive LocalMapping stages queue-aware while preserving single-thread Local BA behavior.
- Added Local BA starvation guard counters and completion-window behavior.
- Added targeted tests.

## 6. Why Structurally Correct

The implementation follows pySLAM's condition ordering and LocalMapping stage ordering, while keeping a documented Python-safe adaptation for sequential RGB-D spacing and BA starvation protection. Local BA is not blindly skipped: in the validated sequential runs it completed for every inserted keyframe.

## 7. Tests Added/Updated

- `tests/visual_slam/orbslam/test_checkpoint_2_33A_kf_local_mapping_schedule.py`

## 8. Test Commands and Results

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_33A_kf_local_mapping_schedule.py
```

Result: 31 passed.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam -k "not cpp_slam_core"
```

Result: 309 passed, 1 skipped, 94 deselected.

Known C++ segfault status: treated as out of scope per task instructions. The non-C++ slice was used, and no C++ code was modified.

## 9. Dataset Validation Commands and Results

Baseline 3-frame:

- Output: `visual_slam_outputs/checkpoint_2_33A/baseline_fr1_desk_3_scheduler_profile`
- Result: 3/3 OK, final state OK, keyframes 1, map points 1879, trajectory poses 3.

Baseline 30-frame:

- Output: `visual_slam_outputs/checkpoint_2_33A/baseline_fr1_desk_30_scheduler_profile`
- Result: 30/30 OK, keyframes 10, map points 4265, elapsed 101.731 sec, Local BA completed 9/9.

Post-change 30-frame:

- Output: `visual_slam_outputs/checkpoint_2_33A/postchange_fr1_desk_30_scheduler_profile`
- Result: 30/30 OK, keyframes 4, map points 3503, elapsed 32.328 sec, Local BA completed 3/3.

Post-change 300-frame:

- Output: `visual_slam_outputs/checkpoint_2_33A/fr1_room_300_scheduler_profile`
- Result: 300/300 OK, keyframes 16, map points 3953, elapsed 206.815 sec, Local BA completed 15/15.

## 10. Runtime Comparison

30-frame fr1/desk:

- Keyframes: 10 -> 4.
- `local_mapping.step`: 73.892 sec -> 15.614 sec.
- `local_BA`: 58.289 sec -> 11.244 sec.
- `fuse_map_points`: 4.859 sec -> 0.608 sec.
- Tracking stayed 30/30 OK.

300-frame fr1/room compared to 2.32A context:

- Keyframes: about 99 -> 16.
- `local_mapping.step`: about 1384 sec -> 80.426 sec.
- `local_BA`: about 879 sec -> 42.937 sec.
- `cull_map_points`: about 319 sec -> 33.401 sec.
- `fuse_map_points`: about 141 sec -> 2.655 sec.
- Tracking stayed 300/300 OK.

## 11. Local BA Completion/Starvation Summary

- 30-frame post-change: started 3, completed 3, skipped 0, aborted 0, forced 0.
- 300-frame post-change: started 15, completed 15, skipped 0, aborted 0, forced 0.
- `keyframes_since_last_successful_ba` ended at 0 in both post-change validations.
- Local BA was not starved.

## 12. Remaining Risks

- Loop closing was enabled in the 300-frame run, but no loop was accepted; this checkpoint did not tune loop thresholds.
- Map point count is lower after more selective keyframes but did not collapse and tracking remained stable.
- C++ tests remain out of scope due the known segfault exception.

## 13. Next Recommended Action

Use this as the new scheduling baseline. Remaining optimization work should profile and target `cull_map_points` or loop-recall diagnostics without threshold forcing.
