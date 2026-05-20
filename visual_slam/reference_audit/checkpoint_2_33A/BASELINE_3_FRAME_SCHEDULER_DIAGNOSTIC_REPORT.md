# Checkpoint 2.33A Baseline 3-Frame Scheduler Diagnostic Report

## 1. Task/Checkpoint

Checkpoint 2.33A - diagnostics-only baseline before scheduling behavior changes.

## 2. Files Inspected

- `visual_slam/orbslam/run_rgbd_slam.py`
- `visual_slam/orbslam/slam/tracking.py`
- `visual_slam/orbslam/slam/local_mapping.py`

## 3. pySLAM Files Inspected

- `third_party/pyslam_reference/pyslam/slam/tracking.py`
- `third_party/pyslam_reference/pyslam/slam/local_mapping.py`
- `third_party/pyslam_reference/pyslam/slam/local_mapping_core.py`

## 4. Root Cause or Current Hypothesis

This run was diagnostic-only. The confirmed hypothesis remains that local keyframe scheduling uses `is_idle()` plus a sequential 3-frame clamp and lacks explicit queue/backpressure diagnostics.

## 5. Exact Changes Made Before This Run

- Added `--profile-keyframes`.
- Added `keyframe_decision_log.csv`.
- Added `local_mapping_schedule_log.csv`.
- Added Local BA scheduling counters to `run_summary.json`.
- No keyframe admission, LocalMapping scheduling, culling, fusion, Local BA, feature, camera/depth, loop, or Global BA behavior was intentionally changed before this baseline.

## 6. Why the Changes Are Structurally Correct

The added code only records decision/schedule state and writes CSV/summary artifacts when keyframe profiling is enabled, or automatically with runtime profiling. It does not change the current scheduling conditions.

## 7. Tests Added/Updated

No tests were added before the required baseline; tests will be added after baseline capture as requested.

## 8. Test Commands Run

```bash
source .venv/bin/activate
python -m py_compile visual_slam/orbslam/run_rgbd_slam.py visual_slam/orbslam/slam/tracking.py visual_slam/orbslam/slam/local_mapping.py
```

Result: passed.

## 9. Dataset Validation Command and Results

```bash
python -m visual_slam.orbslam.run_rgbd_slam \
  "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" \
  --dataset-type tum_rgbd \
  --camera-profile auto \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_33A/baseline_fr1_desk_3_scheduler_profile" \
  --feature-backend pyslam_orb2 \
  --max-frames 3 \
  --disable-loop-closing \
  --disable-global-ba \
  --no-map-export \
  --profile-runtime \
  --profile-keyframes \
  --print-every 1
```

Results:

- `frames_attempted`: 3
- `tracking_ok_count`: 3
- `tracking_lost_count`: 0
- `final_state`: OK
- `keyframes`: 1
- `map_points`: 1879
- `trajectory_poses`: 3
- `elapsed_sec`: 2.144
- `avg_fps`: 1.40
- `local_ba_started_count`: 0
- `local_ba_completed_count`: 0
- `local_ba_aborted_count`: 0

Artifacts:

- `visual_slam_outputs/checkpoint_2_33A/baseline_fr1_desk_3_scheduler_profile/run_summary.json`
- `visual_slam_outputs/checkpoint_2_33A/baseline_fr1_desk_3_scheduler_profile/keyframe_decision_log.csv`
- `visual_slam_outputs/checkpoint_2_33A/baseline_fr1_desk_3_scheduler_profile/local_mapping_schedule_log.csv`

`keyframe_decision_log.csv` exists and contains the required columns. `local_mapping_schedule_log.csv` exists and contains the required header; no LocalMapping step ran in this 3-frame baseline because no new keyframe was queued after initialization.

## 10. Remaining Risks

- The 3-frame baseline does not exercise LocalMapping scheduling because only the initial keyframe exists.
- LocalMapping stage/counter diagnostics need confirmation in the 30-frame baseline where new keyframes are expected.

## 11. Next Recommended Action

Run the required 30-frame baseline before any behavior changes.
