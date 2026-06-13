# Checkpoint 2.42 Full Lab - Native Triangulated Point Batch

Date: 2026-06-08

## Run

```bash
SLAM_USE_CPP_KEYFRAME=1 PYTHONPATH=. .venv/bin/python tools/run_lab_cpp.py \
  datasets/lab_rgbd_run_2 \
  --output visual_slam/reference_audit/checkpoint_2_42_full_lab_native_triangulated_point_batch_threaded_20260608/run \
  --start-local-mapping-thread \
  --disable-loop-closing \
  --print-every 300 \
  --profile-runtime
```

Console log:

`visual_slam/reference_audit/checkpoint_2_42_full_lab_native_triangulated_point_batch_threaded_20260608/FULL_RUN_CONSOLE.log`

## Summary

- `frames_attempted`: 4494
- `tracking_ok_count`: 4418
- `tracking_lost_count`: 76
- `errors`: 0
- `final_state`: OK
- `keyframes`: 206
- `map_points`: 36810
- `trajectory_poses`: 4418
- `elapsed_sec`: 633.117
- `avg_fps`: 7.10

Artifacts:

- `trajectory_file`: `run/trajectory_lab_rgbd_run_2.txt`
- `frame_log_file`: `run/frame_log_lab_rgbd_run_2.csv`
- `frame_timing_file`: `run/frame_timing.csv`
- `map_points_ply`: `run/map_points.ply`
- `keyframes_json`: `run/keyframes.json`
- `keyframe_graph_json`: `run/keyframe_graph.json`
- `runtime_profile_json`: `run/runtime_profile.json`
- `runtime_profile_csv`: `run/runtime_profile.csv`

## Runtime Profile

| Section | 2.41 mean sec | 2.42 mean sec | Delta |
|---|---:|---:|---:|
| `frame.total` | 0.136187 | 0.139768 | +2.6% |
| `slam.track` | 0.079840 | 0.083576 | +4.7% |
| `tracking.track_previous_frame` | 0.013589 | 0.014332 | +5.5% |
| `tracking.track_local_map` | 0.021853 | 0.022125 | +1.2% |
| `local_mapping.step` | 0.036224 | 0.036628 | +1.1% |
| `local_mapping.process_new_keyframe` | 0.007137 | 0.008462 | +18.6% |
| `local_mapping.cull_map_points` | 0.004767 | 0.005612 | +17.7% |
| `local_mapping.create_new_map_points` | 0.109056 | 0.068118 | -37.5% |
| `local_mapping.fuse_map_points` | 0.139129 | 0.145679 | +4.7% |
| `local_mapping.local_BA` | 0.315709 | 0.325937 | +3.2% |

## Comparison To Full-Run References

Against checkpoint 2.41:

- `avg_fps`: 7.29 -> 7.10
- `elapsed_sec`: 616.430 -> 633.117
- `tracking_lost_count`: 83 -> 76
- `tracking_ok_count`: 4411 -> 4418
- `keyframes`: 202 -> 206
- `map_points`: 36277 -> 36810
- `local_mapping.create_new_map_points` mean: 109.1 ms -> 68.1 ms

Against checkpoint 2.39:

- `avg_fps`: 6.18 -> 7.10
- `tracking_lost_count`: 60 -> 76
- `local_mapping.create_new_map_points` mean: 110.4 ms -> 68.1 ms
- `local_mapping.local_BA` mean: 606.5 ms -> 325.9 ms

## Assessment

Checkpoint 2.42 validates the native triangulated point creation milestone on the full 4494-frame
lab run. The target section improved materially: `local_mapping.create_new_map_points` dropped by
about 37.5% versus checkpoint 2.41.

Overall full-run FPS did not improve over 2.41 because this run created more keyframes/map points
and other sections grew slightly. Tracking loss improved relative to 2.41 but remains above the
2.39 quality reference.

Use checkpoint 2.41 as the fastest full-run performance reference, checkpoint 2.42 as the validated
native point-creation reference, and checkpoint 2.39 as the quality/reference checkpoint until the
tracking-loss flag is resolved.
