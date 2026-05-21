# Checkpoint 2.32A - fr1_room 300-Frame Monitoring Report

## 1. Task/checkpoint name

Checkpoint 2.32A - pySLAM-aligned local map construction and projection-search workflow.

## 2. Run command

```bash
python -m visual_slam.orbslam.run_rgbd_slam \
  "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room" \
  --dataset-type tum_rgbd \
  --camera-profile auto \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_32A/fr1_room_300_localmap_profile" \
  --feature-backend pyslam_orb2 \
  --max-frames 300 \
  --enable-loop-closing \
  --disable-global-ba \
  --no-map-export \
  --profile-runtime \
  --profile-memory \
  --profile-local-map \
  --memory-profile-every 20 \
  --print-every 20
```

## 3. Output artifacts

- Output directory: `visual_slam_outputs/checkpoint_2_32A/fr1_room_300_localmap_profile`
- `run_summary.json`
- `runtime_profile.csv`
- `runtime_profile_live.csv`
- `local_map_profile.csv`
- `memory_profile.csv`
- `frame_log_rgbd_dataset_freiburg1_room.csv`
- `frame_timing.csv`
- `trajectory_rgbd_dataset_freiburg1_room.txt`

## 4. Final run summary

| Metric | Value |
| --- | ---: |
| frames_attempted | 300 |
| tracking_ok_count | 300 |
| tracking_lost_count | 0 |
| errors | 0 |
| final_state | OK |
| keyframes | 99 |
| map_points | 15704 |
| trajectory_poses | 300 |
| elapsed_sec | 1908.060 |
| avg_fps | 0.157 |
| peak_rss_mb | 2637.340 |
| final_rss_mb | 2637.590 |
| accepted_loops | 0 |
| loop_debug_events | 0 |

## 5. Runtime profile summary

| Section | Calls | Total sec | Mean sec | Max sec |
| --- | ---: | ---: | ---: | ---: |
| frame.total | 300 | 1907.744 | 6.359 | 27.992 |
| slam.track | 300 | 516.863 | 1.723 | 8.032 |
| tracking.track_local_map | 299 | 99.388 | 0.332 | 0.622 |
| tracking.track_previous_frame | 299 | 78.132 | 0.261 | 0.505 |
| tracking.create_new_keyframe | 98 | 324.464 | 3.311 | 7.127 |
| local_mapping.step | 98 | 1384.315 | 14.126 | 20.494 |
| local_mapping.local_BA | 98 | 878.933 | 8.969 | 16.084 |
| local_mapping.cull_map_points | 98 | 319.429 | 3.259 | 6.955 |
| local_mapping.fuse_map_points | 98 | 141.376 | 1.443 | 2.879 |
| loop_closing.step | 98 | 1.279 | 0.013 | 0.038 |

## 6. Local map profile summary

| Metric | Average | Max | Last |
| --- | ---: | ---: | ---: |
| local keyframes/frame | 47.502 | 80 | 80 |
| voted local keyframes/frame | 27.428 | 77 | 67 |
| expanded local keyframes/frame | 20.074 | 60 | 13 |
| local points/frame | 8461.860 | 14196 | 13714 |
| already-seen rejections/frame | 537.117 | 797 | 326 |
| visible projected points/frame | 4666.087 | 11163 | 9425 |
| KD candidates/frame | 8665.786 | 17703 | 13481 |
| descriptor comparisons/frame | 997.819 | 1680 | 1620 |
| projection matches/frame | 477.753 | 673 | 535 |
| track_local_map_sec | 0.332 | 0.621 | 0.575 |
| search_map_by_projection_sec | 0.236 | 0.490 | 0.451 |
| pose_optimization_sec | 0.024 | 0.079 | 0.019 |
| local_map_build_sec | 0.067 | 0.142 | 0.101 |

## 7. Memory profile summary

| Metric | Average | Max | Last |
| --- | ---: | ---: | ---: |
| rss_mb | 1738.792 | 2637.340 | 2637.340 |
| keyframes | 49.688 | 99 | 99 |
| map_points | 8917.375 | 15704 | 15704 |
| recent_frames | 18.813 | 20 | 20 |
| num_frame_views_total | 11370.250 | 14575 | 14186 |
| old_frame_views_total | 0.000 | 0 | 0 |

Memory grew steadily with keyframes/map points but frame retention stayed bounded at 20 recent frames, and old frame views stayed at 0 in the cheap memory profile.

## 8. Monitoring checkpoints

| Approx frame | track_local_map mean sec | RSS | Notes |
| ---: | ---: | ---: | --- |
| 19 | 0.115 | ~956 MB | Run healthy; local mapping already dominant. |
| 42 | 0.135 | ~1.04 GB | Local mapping mean ~8.7s. |
| 60 | 0.146 | ~1.15 GB | Tracking local-map still low. |
| 88 | 0.164 | ~1.33 GB | Local BA/map-point culling growing. |
| 108 | 0.178 | ~1.45 GB | Passed original 100-frame point. |
| 135 | 0.197 | ~1.62 GB | Local mapping step mean ~12.0s. |
| 165 | 0.225 | ~1.80 GB | LBA and culling dominate. |
| 207 | 0.253 | ~2.04 GB | Tracking remains OK. |
| 249 | 0.291 | ~2.27 GB | Local mapping mean ~13.2s. |
| 285 | 0.321 | ~2.48 GB | Near finish; no tracking loss. |
| 300 | 0.332 | 2.64 GB | Completed OK. |

## 9. Interpretation

The 2.32A local-map reconstruction scales through 300 frames without tracking loss. `tracking.track_local_map` remains controlled at 0.332 seconds mean, with max 0.622 seconds. This is much lower than the pre-change 30-frame desk baseline of 6.638 seconds mean.

The current dominant runtime bottleneck at 300 frames is no longer tracking local-map construction. It is local mapping, especially:

- `local_mapping.local_BA`: 8.969 sec mean.
- `local_mapping.cull_map_points`: 3.259 sec mean.
- `local_mapping.step`: 14.126 sec mean.

## 10. Remaining risks

- This run did not accept loops in the first 300 frames of `fr1_room`, so it does not validate post-loop correction/GBA behavior.
- Memory reached about 2.64 GB RSS by frame 300. This is not a failure, but longer runs should use memory limits and continue profiling.
- Local keyframes hit the configured `kMaxNumOfKeyframesInLocalMap` of 80 near the end. This is the pySLAM max-local-keyframe bound, not a blind local-map-point cap.

## 11. Next recommended action

For the next optimization checkpoint, focus on local mapping/LBA/map-point culling rather than tracking-local-map construction. For loop-closure validation, use a loop-window or longer `fr1_room` run with memory guardrails.
