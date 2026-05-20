# Checkpoint 2.31A Fix — pySLAM Comparison

## 1. Task/Checkpoint Name
Checkpoint 2.31A fix — memory-policy integration gaps and lightweight runtime/memory profiling.

## 2. Local Files Inspected
- `visual_slam/orbslam/slam/map.py`
- `visual_slam/orbslam/slam/local_mapping_core.py`
- `visual_slam/orbslam/slam/local_mapping.py`
- `visual_slam/orbslam/slam/tracking.py`
- `visual_slam/orbslam/slam/loop_closing.py`
- `visual_slam/orbslam/run_rgbd_slam.py`
- `visual_slam/orbslam/run_tum_rgbd_smoke.py`

## 3. pySLAM Files Inspected
- `third_party/pyslam_reference/pyslam/slam/map.py`
- `third_party/pyslam_reference/pyslam/slam/local_mapping_core.py`
- `third_party/pyslam_reference/pyslam/slam/local_mapping.py`
- `third_party/pyslam_reference/pyslam/slam/slam.py`
- `third_party/pyslam_reference/pyslam/slam/frame.py`

## 4. pySLAM Behavior
### 4.1 Map / recent frames
- Python pySLAM still constructs `Map.frames` from a module constant `kMaxLenFrameDeque`.
- The C++ pySLAM map path uses `Parameters::kMaxLenFrameDeque`.
- Both retain only a short recent-frame window and rely on keyframes/map points for long-term structure.

### 4.2 Keypoint-index semantics
- pySLAM explicitly exposes `get_matched_good_points_and_idxs()` in `Frame`.
- `process_new_keyframe()` already uses original keypoint indices.
- The inspected Python `cull_keyframes()` path still enumerates `kf.get_points()` and therefore carries the compact-index risk.

### 4.3 Runner / global parameters
- pySLAM is generally configured once per process and does not try to support repeated benchmark-style runs in one interpreter with per-run parameter restoration.
- Our runner/test environment does need that isolation because multiple tests and runs execute in-process.

### 4.4 Runtime profiling
- pySLAM has timer/reporting utilities in several subsystems, but not the lightweight single-file runner outputs required by this checkpoint (`runtime_profile.csv`, `runtime_profile.json`, `frame_timing.csv`).

## 5. Local Behavior Before Change
- `Map()` used a duplicated local `kMaxLenFrameDeque` constant rather than `Parameters.kMaxLenFrameDeque`.
- `run_rgbd_slam.py` set lean-memory parameter overrides globally and left them mutated after the run.
- Lean-memory suppressed the flag state for pair reports, but not the actual writer call.
- Keyframe culling still consumed compact `get_points()` indices while indexing `depths`/`octaves`.
- Memory profiling always called the deep map scan once profiling was enabled.
- There was no lightweight runtime profiler or per-frame timing CSV.
- `run_tum_rgbd_smoke.py` was still a separate execution path.

## 6. Gap
- The local code had the right 2.31A building blocks, but not the last mile of integration and repeatability.
- The main pySLAM-aligned principle is bounded normal-frame retention with safe temporary observation cleanup.
- The local benchmark harness additionally needs deterministic parameter restoration and cheap diagnostics, which are runner responsibilities rather than SLAM-geometry changes.

## 7. Implemented Fixes / Deliberate Deviations
### 7.1 `Map.frames` parameter source of truth
- Local fix: `Map.__init__()` now uses `deque(maxlen=Parameters.kMaxLenFrameDeque)`.
- Rationale: this aligns the active map construction path with the configured runtime parameter rather than a duplicated constant.

### 7.2 Keyframe culling original-index fix
- Local fix: `LocalMappingCore.cull_keyframes()` now iterates `kf.get_matched_good_points_and_idxs()`.
- Rationale: this is a deliberate tightening beyond the inspected Python pySLAM implementation, but it matches the intended original-keypoint-index semantics already used elsewhere in pySLAM and avoids incorrect `depths[idx]` / `octaves[idx]` reads.

### 7.3 Runner parameter restoration
- Local fix: added `ParameterSnapshot` / `temporary_parameters()` in `run_rgbd_slam.py`.
- Rationale: this is a local runner requirement for repeated tests and multi-run experiments; it does not change SLAM behavior, only process hygiene.

### 7.4 Lean-memory pair-report suppression
- Local fix: heavy per-candidate pair report writing is now gated by both `dump_loop_candidate_reports` and the suppression flags.
- Rationale: preserves lightweight CSV diagnostics while preventing heavy dumps in lean-memory mode.

### 7.5 Lightweight runtime profiling
- Local fix: added `runtime_profiler.py` plus runner and subsystem instrumentation for tracking, local mapping, loop closing, pruning, and memory snapshots.
- Rationale: this is additive observability and does not modify thresholds, feature extraction, or geometry.

### 7.6 Cheap vs deep memory profiling
- Local fix: `Map.memory_stats(mode=...)` now supports `cheap` and `deep`.
- Rationale: frequent deep scans are not part of pySLAM’s core algorithm; they are local diagnostics. Cheap mode keeps profiling overhead low for medium/overnight runs.

### 7.7 Legacy smoke runner safety
- Local fix: `run_tum_rgbd_smoke.py` now forwards into `run_rgbd_slam.py`.
- Rationale: keeps old imports working while preventing a stale long-run path from bypassing new memory/profiling behavior.

## 8. Why The Result Is pySLAM-Aligned
- Normal-frame retention remains bounded and configurable.
- Temporary frame-view cleanup remains part of the memory policy.
- Long-term graph/map semantics remain keyframe/map-point driven.
- No loop thresholds, tracking thresholds, BA settings, feature extraction, camera behavior, or geometry logic were tuned.
- The deliberate deviations are in runner hygiene, diagnostics, and original-index correctness, all of which support reliable RGB-D benchmarking.

## 9. Test Evidence
- `tests/visual_slam/orbslam/test_checkpoint_2_31A_fix_memory_runtime.py`
- `tests/visual_slam/orbslam` non-C++ slice: `260 passed, 1 skipped, 95 deselected`
- Additional real-run evidence is recorded in `VALIDATION_REPORT.md`.

## 10. Remaining Risks
- The full `tests/visual_slam/orbslam` slice is still blocked by pre-existing native C++ core segfaults in `test_cpp_slam_core_*`, which are outside the Python memory-policy/profiling scope of this checkpoint.
- The 100-frame runtime profile is expected to show whether the dominant cost is tracking/local-map growth, but the 30-frame run already indicates rising local-mapping cost per keyframe.
