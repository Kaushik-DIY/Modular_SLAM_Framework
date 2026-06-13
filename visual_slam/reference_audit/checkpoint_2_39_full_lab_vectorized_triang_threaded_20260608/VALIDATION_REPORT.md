# Checkpoint 2.39 Full Lab Vectorized Triangulation Threaded Validation

Date: 2026-06-08
Commit: `25448ed`
Dataset: `datasets/lab_rgbd_run_2`
Mode: threaded local mapping, loop closing disabled, runtime profiling enabled

## Command

```bash
SLAM_USE_CPP_KEYFRAME=1 PYTHONPATH=. .venv/bin/python tools/run_lab_cpp.py datasets/lab_rgbd_run_2 \
  --output visual_slam/reference_audit/checkpoint_2_39_full_lab_vectorized_triang_threaded_20260608/run \
  --start-local-mapping-thread --disable-loop-closing --print-every 500 --profile-runtime \
  | tee visual_slam/reference_audit/checkpoint_2_39_full_lab_vectorized_triang_threaded_20260608/FULL_RUN_CONSOLE.log
```

Console log:

```text
visual_slam/reference_audit/checkpoint_2_39_full_lab_vectorized_triang_threaded_20260608/FULL_RUN_CONSOLE.log
```

## Result

The run completed all 4494 frames, exited without crash or deadlock, recovered from the known tracking-loss region, and ended with `final_state=OK`.

- `frames_attempted`: 4494
- `tracking_ok_count`: 4434
- `tracking_lost_count`: 60
- `errors`: 0
- `final_state`: `OK`
- `keyframes`: 213
- `map_points`: 36687
- `trajectory_poses`: 4434
- `elapsed_sec`: 726.994
- `avg_fps`: 6.18
- `peak_rss_mb`: 4832.574
- `final_rss_mb`: 4836.023

## Comparison To Checkpoint 2.38

Reference compared against:

```text
visual_slam/reference_audit/checkpoint_2_38_full_lab_f3b_build_mark_search_threaded_20260608/
```

| Metric | 2.38 F3b build-mark-search | 2.39 vectorized triangulation | Delta |
|---|---:|---:|---:|
| `frames_attempted` | 4494 | 4494 | same |
| `tracking_lost_count` | 76 | 60 | -16 |
| `final_state` | `OK` | `OK` | same |
| `elapsed_sec` | 745.471 | 726.994 | -18.477 sec |
| `avg_fps` | 6.03 | 6.18 | +2.5% |
| `keyframes` | 196 | 213 | +17 |
| `map_points` | 36227 | 36687 | +460 |
| `peak_rss_mb` | 4769.973 | 4832.574 | +62.602 MB |
| `final_rss_mb` | 4772.457 | 4836.023 | +63.566 MB |

## Runtime Profile Comparison

From `run/runtime_profile.csv`:

| Section | 2.38 mean sec | 2.39 mean sec | Delta |
|---|---:|---:|---:|
| `tracking.track_local_map` | 0.024000 | 0.023831 | -0.7% |
| `slam.track` | 0.085543 | 0.083779 | -2.1% |
| `frame.total` | 0.164505 | 0.160817 | -2.2% |
| `local_mapping.process_new_keyframe` | 0.013518 | 0.006596 | -51.2% |
| `local_mapping.cull_map_points` | 0.013989 | 0.005208 | -62.8% |
| `local_mapping.create_new_map_points` | 0.114624 | 0.110355 | -3.7% |
| `local_mapping.fuse_map_points` | 0.169278 | 0.142488 | -15.8% |
| `local_mapping.local_BA` | 0.624272 | 0.606480 | -2.8% |
| `memory.prune_old_frame_views` | 0.069667 | 0.067323 | -3.4% |

## Tracking Loss Flag

Status: `MONITOR`, improved.

The tracking-loss count decreased relative to checkpoint 2.38:

- Checkpoint 2.38 tracking loss: 76 frames
- Checkpoint 2.39 tracking loss: 60 frames

Observed recovery events in this run:

- Relocalization failures from frame 1906 through frame 1935.
- Depth reinitialization at frame 1935 after 30 failed relocalizations.
- Relocalization failures from frame 1952 through frame 1981.
- Depth reinitialization at frame 1981 after 30 failed relocalizations.
- The run recovered by frame 2000 progress output and ended with `final_state=OK`.

Per current policy, this remains a monitored stability issue rather than a blocker. Raise it to high priority if future full lab runs exceed this range materially, or revisit after the runtime-efficiency porting is complete.

## Saved Artifacts

Primary runner outputs are under:

```text
visual_slam/reference_audit/checkpoint_2_39_full_lab_vectorized_triang_threaded_20260608/run/
```

Important files:

- `run/run_summary.json`
- `run/runtime_profile.csv`
- `run/runtime_profile.json`
- `run/frame_log_lab_rgbd_run_2.csv`
- `run/frame_timing.csv`
- `run/local_map_profile.csv`
- `run/local_mapping_schedule_log.csv`
- `run/keyframe_decision_log.csv`
- `run/trajectory_lab_rgbd_run_2.txt`
- `run/map_points.ply`
- `run/keyframes.json`
- `run/keyframe_graph.json`

The runner also wrote standardized timestamped copies with stem:

```text
lab_rgbd__lab_rgbd_run_2__completed_20260608_091750
```

## Interpretation

Checkpoint 2.39 is a small full-run runtime improvement over checkpoint 2.38 and improves the known tracking-loss count from 76 to 60 frames. The vectorized triangulation and native local-mapping cleanup remain stable over the full dataset.

The larger remaining runtime bottleneck is still local mapping, especially local BA and late-run local-map maintenance. The full-run `avg_fps=6.18` remains below the 10-12 FPS target, so the next implementation target should be local BA packing/write-back and/or deeper native create-new-map-points / map insertion work.
