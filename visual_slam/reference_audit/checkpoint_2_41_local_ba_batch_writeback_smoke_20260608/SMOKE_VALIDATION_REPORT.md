# Checkpoint 2.41 - Local BA Batch Write-Back Smoke

Date: 2026-06-08

## Change Under Test

Local BA result write-back now uses `cpp_slam_core` batch helpers when available:

- `update_local_ba_poses_batch`
- `update_local_ba_points_batch`

The helpers apply the same pose, point-position, and normal/depth updates as the previous Python
loops, but avoid one Python method dispatch per updated keyframe/map point for native
`cpp_slam_core` objects. The bridge keeps the old Python write-back as a fallback.

Outlier pruning in `unpack_local_ba` also now iterates only flagged outlier rows and caches
keyframe point lists before checking observation identity.

This is still GIL-held because map-point normal/depth refresh touches Python keyframe attributes.
It is a dispatch-reduction milestone, not the larger GIL-overlap port.

## Build

```bash
cmake --build third_party/cpp_slam_core/build -j4
cp third_party/cpp_slam_core/build/cpp_slam_core.cpython-311-x86_64-linux-gnu.so \
  .venv/lib/python3.11/site-packages/cpp_slam_core.cpython-311-x86_64-linux-gnu.so
.venv/bin/python -c "import cpp_slam_core as c; print(hasattr(c, 'update_local_ba_poses_batch'), hasattr(c, 'update_local_ba_points_batch'))"
# True True
```

## Validation

Focused tests:

```bash
PYTHONPATH=/home/kaushik/slam_ws .venv/bin/python -m pytest \
  tests/visual_slam/orbslam/test_slam_optimizer_core_parity.py -q
# 11 passed

PYTHONPATH=/home/kaushik/slam_ws .venv/bin/python -m pytest \
  tests/visual_slam/orbslam/test_checkpoint_2_12_local_mapping.py \
  tests/visual_slam/orbslam/test_cpp_slam_core_phase4_local_mapping.py -q
# 30 passed
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
  --output visual_slam/reference_audit/checkpoint_2_41_local_ba_batch_writeback_smoke_20260608/run \
  --max-frames 600 \
  --start-local-mapping-thread \
  --disable-loop-closing \
  --print-every 200 \
  --profile-runtime
```

Console log:

`visual_slam/reference_audit/checkpoint_2_41_local_ba_batch_writeback_smoke_20260608/SMOKE_RUN_CONSOLE.log`

## Smoke Summary

- `frames_attempted`: 600
- `tracking_ok_count`: 600
- `tracking_lost_count`: 0
- `errors`: 0
- `final_state`: OK
- `keyframes`: 11
- `map_points`: 3753
- `trajectory_poses`: 600
- `elapsed_sec`: 73.950
- `avg_fps`: 8.11

Runtime profile highlights:

- `frame.total` mean: 122.558 ms
- `slam.track` mean: 82.782 ms
- `tracking.track_local_map` mean: 21.791 ms
- `local_mapping.process_new_keyframe` mean: 6.964 ms
- `local_mapping.cull_map_points` mean: 3.079 ms
- `local_mapping.create_new_map_points` mean: 221.226 ms
- `local_mapping.fuse_map_points` mean: 139.396 ms
- `local_mapping.local_BA` mean: 299.859 ms

## Comparison To Checkpoint 2.40 Smoke

- `avg_fps`: 6.86 -> 8.11
- `frame.total` mean: 144.929 ms -> 122.558 ms
- `slam.track` mean: 96.632 ms -> 82.782 ms
- `tracking.track_local_map` mean: 24.840 ms -> 21.791 ms
- `local_mapping.local_BA` mean: 376.670 ms -> 299.859 ms
- `local_mapping.fuse_map_points` mean: 155.131 ms -> 139.396 ms
- `local_mapping.create_new_map_points` mean: 240.936 ms -> 221.226 ms

This is a focused 600-frame smoke improvement and does not replace checkpoint 2.39 as the current
full-lab reference.
