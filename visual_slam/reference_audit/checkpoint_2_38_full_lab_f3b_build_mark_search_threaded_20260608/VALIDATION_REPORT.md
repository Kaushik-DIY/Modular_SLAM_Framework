# Checkpoint 2.38 Full Lab F3b Build-Mark-Search Threaded Validation

Date: 2026-06-08
Commit: `a5f6403`
Dataset: `datasets/lab_rgbd_run_2`
Mode: threaded local mapping, loop closing disabled, runtime profiling enabled

## Command

```bash
SLAM_USE_CPP_KEYFRAME=1 PYTHONPATH=. .venv/bin/python tools/run_lab_cpp.py datasets/lab_rgbd_run_2 \
  --output visual_slam/reference_audit/checkpoint_2_38_full_lab_f3b_build_mark_search_threaded_20260608/run \
  --start-local-mapping-thread --disable-loop-closing --print-every 500 --profile-runtime
```

Console log:

```text
visual_slam/reference_audit/checkpoint_2_38_full_lab_f3b_build_mark_search_threaded_20260608/FULL_RUN_CONSOLE.log
```

## Result

The run completed all 4494 frames, exited without crash or deadlock, recovered from the known tracking-loss segment, and ended with `final_state=OK`.

- `frames_attempted`: 4494
- `tracking_ok_count`: 4418
- `tracking_lost_count`: 76
- `errors`: 0
- `final_state`: `OK`
- `keyframes`: 196
- `map_points`: 36227
- `trajectory_poses`: 4418
- `elapsed_sec`: 745.471
- `avg_fps`: 6.03
- `peak_rss_mb`: 4769.973
- `final_rss_mb`: 4772.457

## Comparison To Checkpoint 2.37

Reference compared against:

```text
visual_slam/reference_audit/checkpoint_2_37_full_lab_native_lm_fuse_triang_threaded_20260607/
```

| Metric | 2.37 native LM fuse/triang | 2.38 F3b build-mark-search | Delta |
|---|---:|---:|---:|
| `frames_attempted` | 4494 | 4494 | same |
| `tracking_lost_count` | 76 | 76 | same |
| `final_state` | `OK` | `OK` | same |
| `elapsed_sec` | 1569.855 | 745.471 | -824.384 sec |
| `avg_fps` | 2.86 | 6.03 | 2.11x |
| `keyframes` | 205 | 196 | -9 |
| `map_points` | 36098 | 36227 | +129 |
| `peak_rss_mb` | 4804.844 | 4769.973 | -34.871 MB |
| `final_rss_mb` | 4809.098 | 4772.457 | -36.641 MB |

## Runtime Profile Comparison

From `run/runtime_profile.csv`:

| Section | 2.37 mean sec | 2.38 mean sec | Delta |
|---|---:|---:|---:|
| `tracking.track_local_map` | 0.219151 | 0.024000 | -89.0% |
| `tracking.track_previous_frame` | 0.014447 | 0.014458 | flat |
| `tracking.create_new_keyframe` | 0.142029 | 0.146930 | +3.5% |
| `local_mapping.fuse_map_points` | 0.272480 | 0.169278 | -37.9% |
| `local_mapping.create_new_map_points` | 0.158217 | 0.114624 | -27.6% |
| `local_mapping.local_BA` | 0.780459 | 0.624272 | -20.0% |
| `memory.prune_old_frame_views` | 0.064038 | 0.069667 | +8.8% |
| `slam.track` | 0.277544 | 0.085543 | -69.2% |
| `frame.total` | 0.348108 | 0.164505 | -52.7% |

## Tracking Loss Flag

Status: `MONITOR`, unchanged.

The tracking-loss count did not grow relative to checkpoint 2.37:

- Checkpoint 2.37 tracking loss: 76 frames
- Checkpoint 2.38 tracking loss: 76 frames
- Lost region remains concentrated around frames 1904-2010.
- The run recovered and ended with `final_state=OK`.

Notable events in this run:

- Repeated relocalization failures from frame 1904 through frame 1933.
- Depth reinitialization at frame 1933 after 30 failed relocalizations.
- Relocalization failures from frame 1951 through frame 1965, followed by successful relocalization at frame 1966.
- Relocalization failures from frame 1981 through frame 2010.
- Frame 2001 progress line reported `state=LOST`.
- Depth reinitialization at frame 2010 after 30 failed relocalizations.
- The run recovered and ended with `final_state=OK`.

Per current policy, this remains a monitored stability issue rather than a blocker. Raise it to high priority if future full lab runs exceed this range materially, or revisit after the runtime-efficiency porting is complete.

## Saved Artifacts

Primary runner outputs are under:

```text
visual_slam/reference_audit/checkpoint_2_38_full_lab_f3b_build_mark_search_threaded_20260608/run/
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
lab_rgbd__lab_rgbd_run_2__completed_20260608_075609
```

## Interpretation

Checkpoint 2.38 is a strong runtime improvement over checkpoint 2.37 while preserving the known tracking-loss range. The combined native F3b tracking local-map path reduced `tracking.track_local_map` from about 219 ms to 24 ms on average over the full dataset, and full-run average FPS improved from 2.86 to 6.03.

The remaining runtime gap is now less about local-map projection search and more about full-frame cost, local mapping, BA, and late-run map/maintenance growth. Recommended next step: profile the 2.38 full-run artifacts to choose the next porting target, with likely candidates including local BA write-back / optimizer bridge overhead, remaining local-mapping Python/GIL work, and map-density/maintenance costs.
