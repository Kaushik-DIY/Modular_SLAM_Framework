# Checkpoint 2.32A - Validation Report

## 1. Task/checkpoint name

Checkpoint 2.32A - pySLAM-aligned local map construction and projection-search workflow.

## 2. Files inspected

- `visual_slam/orbslam/slam/tracking.py`
- `visual_slam/orbslam/slam/map.py`
- `visual_slam/orbslam/slam/keyframe.py`
- `visual_slam/orbslam/slam/map_point.py`
- `visual_slam/orbslam/slam/geometry_matchers.py`
- `visual_slam/orbslam/slam/local_mapping.py`
- `visual_slam/orbslam/slam/local_mapping_core.py`
- `visual_slam/orbslam/slam/config_parameters.py`
- `visual_slam/orbslam/run_rgbd_slam.py`

## 3. pySLAM files inspected

- `third_party/pyslam_reference/pyslam/slam/tracking.py`
- `third_party/pyslam_reference/pyslam/slam/map.py`
- `third_party/pyslam_reference/pyslam/slam/keyframe.py`
- `third_party/pyslam_reference/pyslam/slam/map_point.py`
- `third_party/pyslam_reference/pyslam/slam/geometry_matchers.py`

Reference commit: `a95db3a0e95764b8c68b81fade544bdd6ecb912e`.

## 4. Root cause found

The baseline confirmed that local-map construction dominated `track_local_map`:

- `tracking.track_local_map`: mean 6.638 sec, total 192.493 sec over 29 calls.
- `local_map_build_sec`: mean 6.504 sec.
- `search_map_by_projection_sec`: mean 0.105 sec.

The root structural gap was that local tracking built the local map from a stale/reference keyframe and all reference covisibles, then selected the best reference afterward. pySLAM starts from current-frame matched map points, votes observing keyframes, selects the max-vote reference before constructing local points, and expands a bounded graph neighborhood.

## 5. Exact changes made

- Added `--profile-local-map` and `local_map_profile.csv`.
- Added optional projection diagnostics for local projection search.
- Added tracking local-keyframe voting from current-frame matched map points.
- Selected reference keyframe from max vote before local keyframes and local points are built.
- Built local keyframes from the voted set.
- Added bounded one-best-covisible, one-child, and parent expansion.
- Honored `Parameters.kNumBestCovisibilityKeyFramesTracking`.
- Added `MapPoint.last_track_reference_frame_id` for per-frame local-point uniqueness.
- Marked already-matched current-frame points as seen before projection search.
- Moved projection search's bad/already-seen rejection before expensive projection/KD query.

## 6. Tests added/updated

Added:

- `tests/visual_slam/orbslam/test_checkpoint_2_32A_local_map_pyslam_alignment.py`

Coverage includes all 17 checkpoint-required tests.

## 7. Test commands and results

### Targeted checkpoint test

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/visual_slam/orbslam/test_checkpoint_2_32A_local_map_pyslam_alignment.py
```

Result:

```text
17 passed in 0.79s
```

### Full visual SLAM test slice

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam
```

Result:

```text
Fatal Python error: Segmentation fault
Current test: tests/visual_slam/orbslam/test_cpp_slam_core_phase2_frame.py::TestFramePointMatch::test_reset_points
```

This native C++ extension segfault reproduces standalone and is explicitly waived for Checkpoint 2.32A by user confirmation because C++ extension changes are out of scope for this checkpoint.

### Standalone native-test repro

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/visual_slam/orbslam/test_cpp_slam_core_phase2_frame.py::TestFramePointMatch::test_reset_points
```

Result:

```text
Fatal Python error: Segmentation fault
```

### Python/non-C++ visual SLAM slice

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam -k 'not cpp_slam_core'
```

Result:

```text
278 passed, 1 skipped, 94 deselected in 27.39s
```

## 8. Dataset validation commands and results

### Baseline 30-frame run

Completed before behavior changes:

```text
Output: visual_slam_outputs/checkpoint_2_32A/baseline_fr1_desk_30_localmap_profile
frames_attempted: 30
tracking_ok_count: 30
tracking_lost_count: 0
final_state: OK
keyframes: 10
map_points: 4288
trajectory_poses: 30
elapsed_sec: 290.892
avg_fps: 0.103
```

### Post-change 30-frame run

Completed after the unrelated standalone C++ segfault was explicitly waived:

```text
Output: visual_slam_outputs/checkpoint_2_32A/postchange_fr1_desk_30_localmap_profile
frames_attempted: 30
tracking_ok_count: 30
tracking_lost_count: 0
final_state: OK
keyframes: 10
map_points: 4239
trajectory_poses: 30
elapsed_sec: 100.814
avg_fps: 0.298
```

### Baseline vs post-change summary

```text
tracking_lost_count: 0 -> 0
final_state: OK -> OK
trajectory_poses: 30 -> 30
map_points: 4288 -> 4239
track_local_map mean: 6.638 sec -> 0.132 sec
search_map_by_projection mean: 0.105 sec -> 0.089 sec
local_points/frame: 3122.379 -> 3106.724
visible_projected_points/frame: 2833.724 -> 2273.690
descriptor_comparisons/frame: 1132.897 -> 821.793
projection_matches/frame: 490.655 -> 490.759
```

### User-requested 300-frame fr1_room run

After the successful 30-frame comparison, the user requested a 300-frame run instead of the proposed 100-frame run. Command:

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

Result:

```text
Output: visual_slam_outputs/checkpoint_2_32A/fr1_room_300_localmap_profile
frames_attempted: 300
tracking_ok_count: 300
tracking_lost_count: 0
errors: 0
final_state: OK
keyframes: 99
map_points: 15704
trajectory_poses: 300
elapsed_sec: 1908.060
avg_fps: 0.157
peak_rss_mb: 2637.340
accepted_loops: 0
```

300-frame runtime highlights:

```text
tracking.track_local_map mean: 0.332 sec, max: 0.622 sec
search_map_by_projection mean: 0.236 sec, max: 0.490 sec
local_map_build mean: 0.067 sec, max: 0.142 sec
local_mapping.step mean: 14.126 sec, max: 20.494 sec
local_mapping.local_BA mean: 8.969 sec, max: 16.084 sec
local_mapping.cull_map_points mean: 3.259 sec, max: 6.955 sec
```

The 300-frame run confirms that the repaired tracking local-map path remains controlled at medium-run scale. The active bottleneck has shifted to local mapping, especially local BA and map-point culling.

## 9. Correctness/benchmarkability/runtime status

- Correctness of the Python local-map reconstruction is supported by targeted parity tests and the non-C++ visual SLAM slice.
- Benchmarkability is improved by the new `local_map_profile.csv` output.
- Runtime improved substantially in the required 30-frame comparison: average `track_local_map_sec` dropped by about 98.0%, from 6.638 seconds to 0.132 seconds.
- The user-requested 300-frame `fr1_room` run completed with 300/300 OK frames and `tracking.track_local_map` at 0.332 seconds mean.

## 10. Remaining risks

- Full test-slice completion remains blocked by a native C++ extension segfault unrelated to the Python local-map implementation.
- The C++ extension segfault remains unresolved outside this checkpoint and should be handled separately before relying on the C++ core test suite as a gate.
- Child expansion order follows local set iteration; this is bounded and tested, but exact child identity can vary if multiple valid children exist.
- The 300-frame run did not accept loops, so loop-correction behavior is still not exercised by this validation.
- Memory reached 2.64 GB RSS by frame 300; longer runs should use explicit memory guardrails.

## 11. Next recommended action

For the next optimization checkpoint, focus on local mapping/LBA/map-point culling. For loop behavior, use a loop-window or longer `fr1_room` run with memory limits.
