# Checkpoint 2.42 - Native Triangulated Point Batch Smoke

Date: 2026-06-08

## Change Under Test

`create_new_map_points` now routes native C++ keyframe pairs through
`cpp_slam_core.add_triangulated_map_points_batch` after epipolar matching and triangulation.

The helper performs the candidate validity checks, far-point filtering, C++ `MapPoint`
construction, keyframe observation insertion, Python map insertion, and point info update in one
binding call. The existing Python `Map.add_points` path remains the fallback.

The native epipolar binding also accepts contiguous NumPy angle arrays directly, avoiding Python
float-list construction for every neighbor pair.

This is a native local-mapping dispatch-reduction milestone. It still holds the GIL for Python map
insertion and observation callbacks, but moves the hot per-candidate point-creation loop out of
Python.

## Build

```bash
cmake --build third_party/cpp_slam_core/build -j4
cp third_party/cpp_slam_core/build/cpp_slam_core.cpython-311-x86_64-linux-gnu.so \
  .venv/lib/python3.11/site-packages/cpp_slam_core.cpython-311-x86_64-linux-gnu.so
.venv/bin/python - <<'PY'
import cpp_slam_core as c
print(c.hello(), hasattr(c, "add_triangulated_map_points_batch"))
PY
# cpp_slam_core ok True
```

## Validation

Focused tests:

```bash
PYTHONPATH=/home/kaushik/slam_ws .venv/bin/python -m pytest \
  tests/visual_slam/orbslam/test_checkpoint_2_12_local_mapping.py \
  tests/visual_slam/orbslam/test_cpp_slam_core_phase4_local_mapping.py -q
# 30 passed

PYTHONPATH=/home/kaushik/slam_ws .venv/bin/python -m pytest \
  tests/visual_slam/orbslam/test_f2_local_map.py \
  tests/visual_slam/orbslam/test_cpp_slam_core_phase1_map_point.py \
  tests/visual_slam/orbslam/test_f1_keyframe_cpp_wired.py -q
# 34 passed
```

F4 invariant audit:

```bash
# map_point.cpp CLEAN
# keyframe.cpp CLEAN
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
  --output visual_slam/reference_audit/checkpoint_2_42_native_triangulated_point_batch_smoke2_20260608/run \
  --max-frames 600 \
  --start-local-mapping-thread \
  --disable-loop-closing \
  --print-every 200 \
  --profile-runtime
```

Console log:

`visual_slam/reference_audit/checkpoint_2_42_native_triangulated_point_batch_smoke2_20260608/SMOKE_RUN_CONSOLE.log`

## Smoke Summary

- `frames_attempted`: 600
- `tracking_ok_count`: 600
- `tracking_lost_count`: 0
- `errors`: 0
- `final_state`: OK
- `keyframes`: 11
- `map_points`: 3805
- `trajectory_poses`: 600
- `elapsed_sec`: 71.246
- `avg_fps`: 8.42

Runtime profile highlights:

- `frame.total` mean: 118.015 ms
- `slam.track` mean: 80.526 ms
- `tracking.track_local_map` mean: 20.015 ms
- `local_mapping.process_new_keyframe` mean: 6.422 ms
- `local_mapping.cull_map_points` mean: 5.624 ms
- `local_mapping.create_new_map_points` mean: 148.588 ms
- `local_mapping.fuse_map_points` mean: 146.955 ms
- `local_mapping.local_BA` mean: 299.570 ms

## Comparison To Checkpoint 2.41 Smoke

- `avg_fps`: 8.11 -> 8.42
- `frame.total` mean: 122.558 ms -> 118.015 ms
- `slam.track` mean: 82.782 ms -> 80.526 ms
- `tracking.track_local_map` mean: 21.791 ms -> 20.015 ms
- `local_mapping.create_new_map_points` mean: 221.226 ms -> 148.588 ms
- `local_mapping.local_BA` mean: 299.859 ms -> 299.570 ms

This is a focused 600-frame smoke improvement. A full lab run is still required before promoting
checkpoint 2.42 beyond a smoke milestone.
