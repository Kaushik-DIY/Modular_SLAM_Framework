# Checkpoint 2.43 Full Lab Validation - Native Local BA Pack

Date: 2026-06-09

Commit under validation: `77c3ca82` (`F4b: pack local BA natively`)

Dataset: `datasets/lab_rgbd_run_2`

Run folder:
`visual_slam/reference_audit/checkpoint_2_43_full_lab_native_local_ba_pack_threaded_20260608/run/`

Command:

```bash
SLAM_USE_CPP_KEYFRAME=1 PYTHONPATH=. .venv/bin/python tools/run_lab_cpp.py datasets/lab_rgbd_run_2 \
  --output visual_slam/reference_audit/checkpoint_2_43_full_lab_native_local_ba_pack_threaded_20260608/run \
  --start-local-mapping-thread --disable-loop-closing --print-every 500 --profile-runtime
```

## Result

The full lab run completed all 4494 frames with no process errors and final state OK.

| Metric | Value |
| --- | ---: |
| Frames attempted | 4494 |
| Tracking OK | 4434 |
| Tracking lost | 60 |
| Errors | 0 |
| Final state | OK |
| Keyframes | 221 |
| Map points | 36277 |
| Trajectory poses | 4434 |
| Elapsed time | 784.15 s |
| Average FPS | 5.73 |

## Comparison

| Checkpoint | Main change | FPS | Lost | Keyframes | Map points | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2.39 | Vectorized triangulation | 6.18 | 60 | - | - | Current quality/reference checkpoint |
| 2.41 | Local BA batch write-back | 7.29 | 83 | 202 | 36277 | Current fastest full-run performance reference |
| 2.42 | Native triangulated point batch | 7.10 | 76 | 206 | 36810 | Validated native point-creation reference |
| 2.43 | Native local BA pack | 5.73 | 60 | 221 | 36277 | Stability/quality checkpoint, not a speed reference |

Checkpoint 2.43 restores the monitored tracking-loss count to the 2.39 quality-reference range
(`60` lost frames), improving over 2.41 (`83`) and 2.42 (`76`). It is therefore acceptable from a
stability perspective.

It should not be promoted as the performance reference. This run produced more keyframes and slower
tracking/load timing, so end-to-end throughput regressed from 2.42 (`7.10 FPS`) and 2.41
(`7.29 FPS`) to `5.73 FPS`.

## Runtime Profile Highlights

| Section | 2.41 mean | 2.42 mean | 2.43 mean |
| --- | ---: | ---: | ---: |
| `frame.total` | 136.2 ms | 139.8 ms | 173.5 ms |
| `slam.track` | 79.8 ms | 83.6 ms | 110.7 ms |
| `tracking.track_local_map` | 21.9 ms | 22.1 ms | 28.7 ms |
| `tracking.track_previous_frame` | 13.6 ms | 14.3 ms | 18.9 ms |
| `tracking.create_new_keyframe` | 135.3 ms | 140.0 ms | 173.8 ms |
| `local_mapping.create_new_map_points` | 109.1 ms | 68.1 ms | 82.7 ms |
| `local_mapping.fuse_map_points` | 139.1 ms | 145.7 ms | 172.5 ms |
| `local_mapping.local_BA` | 315.7 ms | 325.9 ms | 275.2 ms |
| `local_mapping.cull_keyframes` | 82.7 ms | 82.1 ms | 105.2 ms |
| `local_mapping.step` | 36.2 ms | 36.6 ms | 38.7 ms |

The accepted 2.43 local-BA pack change is visible in the full-run `local_mapping.local_BA` mean:
`325.9 ms -> 275.2 ms` versus 2.42. The whole pipeline still slowed because the run had higher
tracking time, more keyframes, and higher local-map fuse/cull/create-new-keyframe costs.

## Decision

Keep 2.43 as a saved full-run stability/quality checkpoint. Continue implementation rather than
tuning tracking loss now, because the lost-frame count is back at the monitored 60-frame range. The
next implementation work should target the remaining per-frame/GIL-held sections that block runtime:
tracking local-map growth, keyframe creation, and local-mapping fuse/cull orchestration.
