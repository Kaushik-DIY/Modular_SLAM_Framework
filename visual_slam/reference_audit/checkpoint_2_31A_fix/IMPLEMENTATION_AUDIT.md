# Checkpoint 2.31A Fix — Implementation Audit

## 1. Task/Checkpoint Name
Checkpoint 2.31A fix — complete memory-policy integration gaps and add lightweight runtime/memory profiling.

## 2. Files Modified
- `visual_slam/orbslam/run_rgbd_slam.py`
- `visual_slam/orbslam/run_tum_rgbd_smoke.py`
- `visual_slam/orbslam/slam/runtime_profiler.py`
- `visual_slam/orbslam/slam/map.py`
- `visual_slam/orbslam/slam/config_parameters.py`
- `visual_slam/orbslam/slam/local_mapping_core.py`
- `visual_slam/orbslam/slam/local_mapping.py`
- `visual_slam/orbslam/slam/tracking.py`
- `visual_slam/orbslam/slam/loop_closing.py`
- `visual_slam/orbslam/slam/slam.py`
- `visual_slam/orbslam/slam/essential_graph.py`
- `visual_slam/orbslam/slam/geometry_matchers.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_31A_fix_memory_runtime.py`

## 3. Remaining Gaps Fixed
### 3.1 `Map.frames` now uses the configured parameter
- `Map.__init__()` now constructs `self.frames` from `Parameters.kMaxLenFrameDeque`.
- This removes the duplicated local source of truth for new map creation.

### 3.2 Lean-memory now suppresses heavy pair reports correctly
- Lean-memory now forces `no_heavy_loop_reports=True` and `no_loop_candidate_pair_reports=True`.
- The actual write site is now gated by those suppression flags, so heavy per-candidate pair reports are skipped even when dumping is otherwise enabled.
- Lightweight loop-debug CSV output remains allowed.

### 3.3 Global parameters are restored after runner execution
- Added `ParameterSnapshot` and `temporary_parameters()`.
- Runner overrides are now scoped to one run and restored on both success and exception.
- This covers image-retention flags, frame-view pruning cadence, recent-frame retention length, and local-mapping wait timeout.

### 3.4 Legacy smoke runner is now safe
- `run_tum_rgbd_smoke.py` is now a thin backward-compatible wrapper around `run_rgbd_slam.py`.
- Old imports remain valid, but the actual execution path now inherits the final runner’s cleanup and profiling behavior.

### 3.5 Keyframe culling original-index bug is fixed
- `LocalMappingCore.cull_keyframes()` now iterates `kf.get_matched_good_points_and_idxs()`.
- Depth and octave reads now use original keypoint indices instead of compact `get_points()` positions.

## 4. Runtime Profiler Design
### 4.1 Utility
- Added `visual_slam/orbslam/slam/runtime_profiler.py`.
- It supports:
  - `start(section)`
  - `stop(section)`
  - `with profiler.section(section):`
  - `to_dict()`
  - `write_csv(path)`
  - `write_json(path)`

### 4.2 Runner outputs
- Added:
  - `frame_timing.csv`
  - `runtime_profile.csv`
  - `runtime_profile.json`
  - `runtime_profile_live.csv` during long runs

### 4.3 Instrumented sections
- High-level:
  - `frame.total`
  - `frame.load_rgb`
  - `frame.load_depth`
  - `slam.track`
  - `local_mapping.step`
  - `loop_closing.step`
  - `memory.prune_old_frame_views`
  - `memory.profile_snapshot`
  - `frame.log_write`
- Tracking internals:
  - `tracking.track_previous_frame`
  - `tracking.track_reference_frame`
  - `tracking.track_local_map`
  - `tracking.need_new_keyframe`
  - `tracking.create_new_keyframe`
- Local mapping internals:
  - `local_mapping.process_new_keyframe`
  - `local_mapping.cull_map_points`
  - `local_mapping.create_new_map_points`
  - `local_mapping.fuse_map_points`
  - `local_mapping.local_BA`
  - `local_mapping.cull_keyframes`
- Loop internals:
  - `loop.detect_candidates`
  - `loop.compute_geometry`
  - `loop.search_more_projection`
  - `loop.correct_loop`
  - `essential_graph.optimize`
  - `global_ba.run`

## 5. Memory Profiler Cheap/Deep Behavior
### 5.1 Cheap mode
- Cheap mode returns:
  - recent frame count
  - keyframe count
  - map point count
  - recent image/depth retention counts
  - cached frame-view totals when available
- It avoids the full map-point scan.

### 5.2 Deep mode
- Deep mode performs the full scan and computes:
  - total frame views
  - old frame views
  - keyframe observation totals
  - heavy-byte estimates

### 5.3 Cache behavior
- `Map.prune_old_frame_views()` now updates a frame-view stats cache.
- Cheap mode can therefore expose the latest known frame-view totals without rescanning the whole map.

## 6. Additional Safe Runtime Changes
- Default `Parameters.kFrameViewPruneEveryNFrames` was relaxed from `5` to `20`.
- Added runner flag `--frame-view-prune-every`.
- Added runner flags:
  - `--profile-runtime`
  - `--runtime-profile-every`
  - `--memory-profile-mode {cheap,deep}`

## 7. Why These Changes Are Structurally Correct
- No loop thresholds were tuned.
- No tracking thresholds were tuned.
- No BA settings were tuned.
- No feature extraction or camera/depth logic was changed.
- The local-mapping/keyframe-culling fix only corrects index semantics; it does not alter the culling rule itself.
- Runner changes are limited to memory policy, profiling, logging, and parameter lifecycle management.

## 8. Tests Added / Updated
- Added `tests/visual_slam/orbslam/test_checkpoint_2_31A_fix_memory_runtime.py` covering:
  - map frame deque parameter usage
  - lean-memory pair-report suppression
  - parameter restoration on success and exception
  - original-index keyframe culling
  - runtime profiler serialization
  - frame timing CSV schema
  - cheap-vs-deep memory profiling behavior
  - pruning runtime-section presence
- Existing documentation-guard tests also required docstring updates in:
  - `essential_graph.py`
  - `geometry_matchers.py`

## 9. Test Commands Run
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_31A_fix_memory_runtime.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam -k 'not cpp_slam_core and not test_cpp'`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_23_essential_graph.py::test_rgbd_se3_policy_documented`

## 10. Test Results
- New checkpoint regression file: passed.
- Non-C++ `tests/visual_slam/orbslam` slice: passed after docstring fixes.
- Full slice including native C++ tests remains blocked by pre-existing extension segfaults in `test_cpp_slam_core_*`.

## 11. Dataset Validation
- 30-frame `fr1_desk` sanity run completed successfully.
- 100-frame `fr1_room` profiling run is documented in `VALIDATION_REPORT.md`.

## 12. Remaining Risks
- The profiling data already shows strong runtime growth in local mapping, especially `local_BA` and `track_local_map`, but the full 100-frame profiling result remains the best short-run evidence.
- Native C++ core tests still crash outside the Python-only scope of this checkpoint.

## 13. Next Recommended Action
- Use the completed short-run profiling outputs to choose the first focused runtime investigation.
- Based on the early live profile, prioritize:
  1. `tracking.track_local_map`
  2. `local_mapping.local_BA`
  3. overall `local_mapping.step`
