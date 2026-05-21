# Checkpoint 2.33A Baseline 30-Frame Scheduler Report

## 1. Task/Checkpoint

Checkpoint 2.33A - pre-behavior-change 30-frame scheduler baseline on TUM fr1/desk.

## 2. Files Inspected

- `visual_slam/orbslam/run_rgbd_slam.py`
- `visual_slam/orbslam/slam/tracking.py`
- `visual_slam/orbslam/slam/local_mapping.py`
- `visual_slam/orbslam/slam/local_mapping_core.py`
- `visual_slam/orbslam/slam/config_parameters.py`

## 3. pySLAM Files Inspected

- `third_party/pyslam_reference/pyslam/slam/tracking.py`
- `third_party/pyslam_reference/pyslam/slam/local_mapping.py`
- `third_party/pyslam_reference/pyslam/slam/local_mapping_core.py`

## 4. Root Cause or Current Hypothesis

The 30-frame baseline confirms the suspected scheduling issue: after the initial keyframe, keyframes are inserted at frames 5, 8, 11, 14, 17, 20, 23, 26, and 29, driven by `local_mapping_idle` with the sequential `min_frames_between_kfs = 3` clamp. LocalMapping then runs fusion and Local BA for every queued keyframe.

## 5. Exact Changes Made Before This Run

Diagnostics only:

- keyframe decision CSV
- LocalMapping schedule CSV
- Local BA scheduling counters in run summary
- `--profile-keyframes`

No scheduling behavior was intentionally changed before this baseline.

## 6. Why the Changes Are Structurally Correct

The diagnostics capture existing state and decisions without changing the return paths in `Tracking.need_new_keyframe` or the unconditional LocalMapping stage order.

## 7. Tests Added/Updated

No targeted tests were added before the required baseline; they will be added during implementation.

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
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_33A/baseline_fr1_desk_30_scheduler_profile" \
  --feature-backend pyslam_orb2 \
  --max-frames 30 \
  --disable-loop-closing \
  --disable-global-ba \
  --no-map-export \
  --profile-runtime \
  --profile-keyframes \
  --print-every 10
```

Run summary:

- `frames_attempted`: 30
- `tracking_ok_count`: 30
- `tracking_lost_count`: 0
- `final_state`: OK
- `final_keyframes`: 10
- `final_map_points`: 4265
- `trajectory_poses`: 30
- `elapsed_sec`: 101.731
- `avg_fps`: 0.295

Keyframe insertion:

- Insertions after initialization: 9
- Insert frames: 5, 8, 11, 14, 17, 20, 23, 26, 29
- Average spacing from initial frame 0: 3.22 frames
- Insert reasons: `local_mapping_idle` = 9
- Reject reasons: `conditions_not_met` = 20

LocalMapping and Local BA:

- `local_mapping.step`: 9 calls, 73.892 sec total, 8.210 sec mean
- `local_mapping.local_BA`: 9 calls, 58.289 sec total, 6.477 sec mean
- `local_mapping.cull_map_points`: 9 calls, 9.372 sec total, 1.041 sec mean
- `local_mapping.fuse_map_points`: 9 calls, 4.859 sec total, 0.540 sec mean
- `tracking.track_local_map`: 29 calls, 3.854 sec total, 0.133 sec mean
- Local BA counters: started 9, completed 9, aborted 0, skipped due queue 0, forced due starvation 0

Artifacts:

- `visual_slam_outputs/checkpoint_2_33A/baseline_fr1_desk_30_scheduler_profile/run_summary.json`
- `visual_slam_outputs/checkpoint_2_33A/baseline_fr1_desk_30_scheduler_profile/keyframe_decision_log.csv`
- `visual_slam_outputs/checkpoint_2_33A/baseline_fr1_desk_30_scheduler_profile/local_mapping_schedule_log.csv`
- `visual_slam_outputs/checkpoint_2_33A/baseline_fr1_desk_30_scheduler_profile/runtime_profile.csv`

## 10. Remaining Risks

- This baseline validates the logs but also confirms that the slow unconditional LocalMapping path dominates runtime.
- Behavior changes must preserve the 30/30 OK tracking result and avoid Local BA starvation.

## 11. Next Recommended Action

Implement gaps A-H with targeted tests, starting from diagnostics tests and culling-index regression tests.
