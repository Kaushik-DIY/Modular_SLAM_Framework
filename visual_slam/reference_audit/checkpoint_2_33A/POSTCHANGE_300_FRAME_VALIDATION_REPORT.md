# Checkpoint 2.33A Post-Change 300-Frame Validation Report

## 1. Task/Checkpoint

Checkpoint 2.33A - automatic 300-frame validation after stable 30-frame post-change run.

## 2. Files Inspected

- `visual_slam/orbslam/slam/tracking.py`
- `visual_slam/orbslam/slam/local_mapping.py`
- `visual_slam/orbslam/slam/config_parameters.py`
- `visual_slam/orbslam/run_rgbd_slam.py`
- `visual_slam_outputs/checkpoint_2_33A/fr1_room_300_scheduler_profile/*`

## 3. pySLAM Files Inspected

- `third_party/pyslam_reference/pyslam/slam/tracking.py`
- `third_party/pyslam_reference/pyslam/slam/local_mapping.py`
- `third_party/pyslam_reference/pyslam/slam/local_mapping_core.py`

## 4. Root Cause

The previous 2.32A context showed excessive keyframe density and LocalMapping cost. This run validates that the scheduling changes reduce keyframes and LocalMapping time while preserving tracking stability on a longer RGB-D sequence.

## 5. Exact Changes Made

See `IMPLEMENTATION_ALIGNMENT_REPORT.md`.

## 6. Why Structurally Correct

The 300-frame result shows that selective keyframe insertion did not starve Local BA: every inserted post-initialization keyframe completed Local BA in sequential mode, and tracking remained OK for all frames.

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
  "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room" \
  --dataset-type tum_rgbd \
  --camera-profile auto \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_33A/fr1_room_300_scheduler_profile" \
  --feature-backend pyslam_orb2 \
  --max-frames 300 \
  --enable-loop-closing \
  --disable-global-ba \
  --no-map-export \
  --profile-runtime \
  --profile-memory \
  --profile-keyframes \
  --memory-profile-every 30 \
  --print-every 50
```

Results:

- `frames_attempted`: 300
- `tracking_ok_count`: 300
- `tracking_lost_count`: 0
- `final_state`: OK
- `final_keyframes`: 16
- `final_map_points`: 3953
- `trajectory_poses`: 300
- `elapsed_sec`: 206.815
- `avg_fps`: 1.451
- `peak_rss_mb`: 1129.6
- `accepted_loops`: 0
- `loop_debug_events`: 0

Keyframe insertion:

- Inserted after initialization: 15
- Insert frames: 9, 18, 35, 87, 98, 107, 134, 173, 189, 196, 204, 213, 225, 262, 297
- Average spacing from initial frame 0: 19.8 frames
- Insert reasons: `local_mapping_accepting` = 15
- Reject reasons: `conditions_not_met` = 284

Runtime:

- `local_mapping.step`: 15 calls, 80.426 sec total, 5.362 sec mean
- `local_mapping.local_BA`: 15 calls, 42.937 sec total, 2.862 sec mean
- `local_mapping.cull_map_points`: 15 calls, 33.401 sec total, 2.227 sec mean
- `local_mapping.fuse_map_points`: 15 calls, 2.655 sec total, 0.177 sec mean
- `tracking.track_local_map`: 299 calls, 35.960 sec total, 0.120 sec mean
- `slam.track`: 300 calls, 121.157 sec total, 0.404 sec mean
- `frame.total`: 300 calls, 206.544 sec total, 0.688 sec mean

Local BA:

- started: 15
- completed: 15
- aborted: 0
- skipped due queue: 0
- forced due starvation: 0
- last successful Local BA keyframe: 15
- keyframes since last successful BA: 0

Comparison to 2.32A context:

- 2.32A keyframes: about 99; 2.33A keyframes: 16.
- 2.32A `local_mapping.step`: about 1384 sec; 2.33A: 80.426 sec.
- 2.32A `local_BA`: about 879 sec; 2.33A: 42.937 sec.
- 2.32A `cull_map_points`: about 319 sec; 2.33A: 33.401 sec.
- 2.32A `fuse_map_points`: about 141 sec; 2.33A: 2.655 sec.
- Tracking remained 300/300 OK.

## 10. Remaining Risks

- No loop was accepted in this 300-frame run, so this checkpoint validates scheduling stability rather than loop recall.
- `cull_map_points` remains a meaningful cost among the remaining LocalMapping stages.
- The known C++ test segfault remains out of scope and no C++ code was modified.

## 11. Next Recommended Action

Use this profile as the new scheduling baseline. A good next checkpoint would target remaining LocalMapping point-culling cost or loop-detection recall diagnostics without changing loop thresholds blindly.
