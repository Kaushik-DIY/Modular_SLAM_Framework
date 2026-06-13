# Checkpoint 2.41 Full Lab - Local BA Batch Write-Back

Date: 2026-06-08

## Run

```bash
SLAM_USE_CPP_KEYFRAME=1 PYTHONPATH=. .venv/bin/python tools/run_lab_cpp.py \
  datasets/lab_rgbd_run_2 \
  --output visual_slam/reference_audit/checkpoint_2_41_full_lab_batch_writeback_threaded_20260608/run \
  --start-local-mapping-thread \
  --disable-loop-closing \
  --print-every 300 \
  --profile-runtime
```

Console log:

`visual_slam/reference_audit/checkpoint_2_41_full_lab_batch_writeback_threaded_20260608/FULL_RUN_CONSOLE.log`

## Summary

- `frames_attempted`: 4494
- `tracking_ok_count`: 4411
- `tracking_lost_count`: 83
- `errors`: 0
- `final_state`: OK
- `keyframes`: 202
- `map_points`: 36277
- `trajectory_poses`: 4411
- `elapsed_sec`: 616.430
- `avg_fps`: 7.29

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

| Section | 2.39 mean sec | 2.41 mean sec | Delta |
|---|---:|---:|---:|
| `frame.total` | 0.160817 | 0.136187 | -15.3% |
| `slam.track` | 0.083779 | 0.079840 | -4.7% |
| `tracking.track_previous_frame` | 0.014009 | 0.013589 | -3.0% |
| `tracking.track_local_map` | 0.023831 | 0.021853 | -8.3% |
| `local_mapping.step` | 0.055580 | 0.036224 | -34.8% |
| `local_mapping.process_new_keyframe` | 0.006596 | 0.007137 | +8.2% |
| `local_mapping.cull_map_points` | 0.005208 | 0.004767 | -8.5% |
| `local_mapping.create_new_map_points` | 0.110355 | 0.109056 | -1.2% |
| `local_mapping.fuse_map_points` | 0.142488 | 0.139129 | -2.4% |
| `local_mapping.local_BA` | 0.606480 | 0.315709 | -47.9% |

## Comparison To Checkpoint 2.39

- `avg_fps`: 6.18 -> 7.29
- `elapsed_sec`: 726.994 -> 616.430
- `tracking_lost_count`: 60 -> 83
- `tracking_ok_count`: 4434 -> 4411
- `keyframes`: 213 -> 202
- `map_points`: 36687 -> 36277

## Assessment

Checkpoint 2.41 is a clear full-run runtime improvement, especially in local BA, and completes the
full 4494-frame lab run with `final_state=OK` and no errors.

However, `tracking_lost_count` increased from checkpoint 2.39's 60 to 83. This should be treated as
a tracking-stability flag before promoting 2.41 as the quality reference. The runtime work can
continue, but any further increase in tracking loss should be handled as high priority.

The current quality/reference checkpoint remains 2.39; checkpoint 2.41 is the current full-run
performance reference.
