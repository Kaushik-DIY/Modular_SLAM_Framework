# Checkpoint 2.33A Post-Change 30-Frame Comparison Report

## 1. Task/Checkpoint

Checkpoint 2.33A - post-change 30-frame validation against the pre-change scheduler baseline.

## 2. Files Inspected

- `visual_slam/orbslam/slam/tracking.py`
- `visual_slam/orbslam/slam/local_mapping.py`
- `visual_slam/orbslam/slam/config_parameters.py`
- `visual_slam/orbslam/run_rgbd_slam.py`
- `visual_slam_outputs/checkpoint_2_33A/baseline_fr1_desk_30_scheduler_profile/*`
- `visual_slam_outputs/checkpoint_2_33A/postchange_fr1_desk_30_scheduler_profile/*`

## 3. pySLAM Files Inspected

- `third_party/pyslam_reference/pyslam/slam/tracking.py`
- `third_party/pyslam_reference/pyslam/slam/local_mapping.py`
- `third_party/pyslam_reference/pyslam/slam/local_mapping_core.py`

## 4. Root Cause

The baseline inserted keyframes every 3 frames after frame 5 because sequential LocalMapping appeared idle and `min_frames_between_kfs` was clamped to 3. The post-change run uses FPS-aware spacing and mapper acceptance/queue diagnostics, reducing keyframe density while preserving tracking.

## 5. Exact Changes Made

See `IMPLEMENTATION_ALIGNMENT_REPORT.md`; the relevant runtime changes are FPS-aware keyframe spacing, pySLAM-style `nMinObs`, mapper acceptance/backpressure, conditional LocalMapping stages, and Local BA starvation counters.

## 6. Why Structurally Correct

The post-change keyframe decisions are still driven by pySLAM-style `c1a/c1b/c1c` and `c2`; the change is that sequential RGB-D no longer claims the minimum interval has elapsed every 3 frames. Local BA still completes for every inserted keyframe in single-thread mode.

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

## 9. Dataset Validation Command and Results

```bash
python -m visual_slam.orbslam.run_rgbd_slam \
  "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" \
  --dataset-type tum_rgbd \
  --camera-profile auto \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_33A/postchange_fr1_desk_30_scheduler_profile" \
  --feature-backend pyslam_orb2 \
  --max-frames 30 \
  --disable-loop-closing \
  --disable-global-ba \
  --no-map-export \
  --profile-runtime \
  --profile-keyframes \
  --print-every 10
```

Comparison:

| Metric | Baseline 30 | Post-change 30 |
|---|---:|---:|
| frames_attempted | 30 | 30 |
| tracking_ok_count | 30 | 30 |
| tracking_lost_count | 0 | 0 |
| final_state | OK | OK |
| final_keyframes | 10 | 4 |
| final_map_points | 4265 | 3503 |
| trajectory_poses | 30 | 30 |
| elapsed_sec | 101.731 | 32.328 |
| avg_fps | 0.295 | 0.928 |
| average frames between inserted KFs | 3.22 | 9.67 |
| local_mapping.step total/mean | 73.892 / 8.210 | 15.614 / 5.205 |
| local_BA total/mean | 58.289 / 6.477 | 11.244 / 3.748 |
| cull_map_points total/mean | 9.372 / 1.041 | 3.347 / 1.116 |
| fuse_map_points total/mean | 4.859 / 0.540 | 0.608 / 0.203 |
| track_local_map total/mean | 3.854 / 0.133 | 2.979 / 0.103 |
| peak_rss_mb | 1013.4 | 937.9 |

Keyframe insertion:

- Baseline insert frames: 5, 8, 11, 14, 17, 20, 23, 26, 29.
- Post-change insert frames: 11, 20, 29.
- Baseline insert reason: `local_mapping_idle` = 9.
- Post-change insert reason: `local_mapping_accepting` = 3.
- Rejections after change: `conditions_not_met` = 26.

Local BA:

- Baseline: started 9, completed 9, skipped 0, aborted 0, forced 0.
- Post-change: started 3, completed 3, skipped 0, aborted 0, forced 0.
- Local BA was not starved.

## 10. Remaining Risks

- Map points decreased from 4265 to 3503, but did not collapse and tracking stayed stable.
- This 30-frame slice does not prove long-horizon loop behavior; the 300-frame validation addresses longer scheduling stability.

## 11. Next Recommended Action

Proceed to 300-frame validation. This was completed automatically after the 30-frame gate passed.
