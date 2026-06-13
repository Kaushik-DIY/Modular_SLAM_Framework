# Checkpoint 2.40 - Local BA Packing Cache Smoke

Date: 2026-06-08

## Change Under Test

`visual_slam/orbslam/slam/slam_optimizer_bridge.py::pack_local_ba` now caches per-keyframe
feature arrays and matched-point lists while packing local bundle adjustment inputs. It also
materializes map-point positions once and reuses a finite-position mask when collecting
observations.

The intent is to reduce Python dispatch and repeated attribute/list lookups before the native
`slam_optimizer_core.run_local_ba` solve. Solver behavior and BA write-back are unchanged.

## Validation

Focused tests:

```bash
PYTHONPATH=/home/kaushik/slam_ws .venv/bin/python -m pytest \
  tests/visual_slam/orbslam/test_slam_optimizer_core_parity.py -q
# 11 passed

PYTHONPATH=/home/kaushik/slam_ws .venv/bin/python -m pytest \
  tests/visual_slam/orbslam/test_checkpoint_2_12_local_mapping.py -q
# 6 passed

PYTHONPATH=/home/kaushik/slam_ws .venv/bin/python -m pytest \
  tests/visual_slam/orbslam/test_cpp_slam_core_phase4_local_mapping.py -q
# 24 passed
```

Full ORB-SLAM suite:

```bash
PYTHONPATH=/home/kaushik/slam_ws .venv/bin/python -m pytest tests/visual_slam/orbslam/ -q
# 514 passed, 1 skipped
```

Threaded 600-frame lab smoke:

```bash
SLAM_USE_CPP_KEYFRAME=1 PYTHONPATH=. .venv/bin/python tools/run_lab_cpp.py \
  datasets/lab_rgbd_run_2 \
  --output visual_slam/reference_audit/checkpoint_2_40_local_ba_pack_cache_smoke_20260608/run \
  --max-frames 600 \
  --start-local-mapping-thread \
  --disable-loop-closing \
  --print-every 200 \
  --profile-runtime
```

Console log:

`visual_slam/reference_audit/checkpoint_2_40_local_ba_pack_cache_smoke_20260608/SMOKE_RUN_CONSOLE.log`

## Smoke Summary

- `frames_attempted`: 600
- `tracking_ok_count`: 600
- `tracking_lost_count`: 0
- `errors`: 0
- `final_state`: OK
- `keyframes`: 11
- `map_points`: 3858
- `trajectory_poses`: 600
- `elapsed_sec`: 87.425
- `avg_fps`: 6.86

Runtime profile highlights:

- `frame.total` mean: 144.929 ms
- `slam.track` mean: 96.632 ms
- `tracking.track_local_map` mean: 24.840 ms
- `local_mapping.process_new_keyframe` mean: 7.244 ms
- `local_mapping.cull_map_points` mean: 3.322 ms
- `local_mapping.create_new_map_points` mean: 240.936 ms
- `local_mapping.fuse_map_points` mean: 155.131 ms
- `local_mapping.local_BA` mean: 376.670 ms

This is a stability-preserving local BA packing cleanup. It does not replace checkpoint 2.39 as the
current full-lab reference.
