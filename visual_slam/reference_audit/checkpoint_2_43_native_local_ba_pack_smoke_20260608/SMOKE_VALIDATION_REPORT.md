# Checkpoint 2.43 - Native Local BA Packing Smoke

Date: 2026-06-08

## Change Under Test

`pack_local_ba()` now tries `cpp_slam_core.pack_local_ba_native()` before the
Python packing path. The native helper packs C++ keyframe poses, fixed flags, map
point positions, reprojection observation rows, camera intrinsics, and outlier
write-back triples for `slam_optimizer_core.run_local_ba()`.

The existing Python packer remains the fallback for mixed/Python objects or native
packing failures. Solver behavior and BA write-back are unchanged.

## Validation

Build/import:

```bash
cmake --build third_party/cpp_slam_core/build -j4
cp third_party/cpp_slam_core/build/cpp_slam_core.cpython-311-x86_64-linux-gnu.so \
  .venv/lib/python3.11/site-packages/cpp_slam_core.cpython-311-x86_64-linux-gnu.so
.venv/bin/python -c "import cpp_slam_core as c; print(c.hello(), hasattr(c, 'pack_local_ba_native'))"
# cpp_slam_core ok True
```

Focused tests:

```bash
PYTHONPATH=/home/kaushik/slam_ws .venv/bin/python -m pytest \
  tests/visual_slam/orbslam/test_cpp_slam_core_phase4_local_mapping.py \
  tests/visual_slam/orbslam/test_slam_optimizer_core_parity.py \
  tests/visual_slam/orbslam/test_checkpoint_2_25_optimizer_parity.py -q
# 47 passed
```

Full ORB-SLAM suite:

```bash
PYTHONPATH=/home/kaushik/slam_ws .venv/bin/python -m pytest tests/visual_slam/orbslam/ -q
# 515 passed, 1 skipped
```

Threaded 600-frame standard-wait smoke:

```bash
SLAM_USE_CPP_KEYFRAME=1 PYTHONPATH=. .venv/bin/python tools/run_lab_cpp.py \
  datasets/lab_rgbd_run_2 \
  --output visual_slam/reference_audit/checkpoint_2_43_native_local_ba_pack_smoke_20260608/run \
  --max-frames 600 \
  --start-local-mapping-thread \
  --disable-loop-closing \
  --print-every 200 \
  --profile-runtime
```

No-wait policy probe:

```bash
SLAM_USE_CPP_KEYFRAME=1 PYTHONPATH=. .venv/bin/python tools/run_lab_cpp.py \
  datasets/lab_rgbd_run_2 \
  --output visual_slam/reference_audit/checkpoint_2_43_native_local_ba_pack_no_wait_smoke_20260608/run \
  --max-frames 600 \
  --start-local-mapping-thread \
  --lm-wait-timeout 0.0 \
  --disable-loop-closing \
  --print-every 200 \
  --profile-runtime
```

## Smoke Summary

Standard wait (`0.5s`):

- `frames_attempted`: 600
- `tracking_lost_count`: 0
- `errors`: 0
- `final_state`: OK
- `keyframes`: 11
- `map_points`: 3742
- `avg_fps`: 5.91

No-wait probe (`0.0s`):

- `frames_attempted`: 600
- `tracking_lost_count`: 0
- `errors`: 0
- `final_state`: OK
- `keyframes`: 12
- `map_points`: 4080
- `avg_fps`: 6.21

## Same-Condition Comparison

The current machine state was slower than the earlier checkpoint 2.42 smoke, so
this checkpoint is compared against clean restored-code reruns made immediately
before this change.

Standard wait (`0.5s`):

- `avg_fps`: 5.96 -> 5.91
- `frame.total` mean: 166.8 ms -> 168.1 ms
- `slam.track` mean: 118.4 ms -> 122.6 ms
- `local_mapping.local_BA` mean: 434.1 ms -> 293.3 ms
- `local_mapping.step` mean: 29.6 ms -> 26.2 ms

No-wait probe (`0.0s`):

- `avg_fps`: 6.13 -> 6.21
- `frame.total` mean: 161.0 ms -> 159.5 ms
- `slam.track` mean: 138.8 ms -> 138.4 ms
- `local_mapping.local_BA` mean: 732.4 ms -> 521.5 ms

## Decision

Accept as a local-BA packing milestone: quality stayed stable on both smoke runs
and the local BA section improved materially. This is not a new fastest full-run
reference because total FPS remained noisy/near-flat under the current runtime
conditions.
