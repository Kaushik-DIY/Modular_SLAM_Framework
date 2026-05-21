# Checkpoint 2.31A Fix — Pre-Change Memory/Runtime Gap Audit

## 1. Task/Checkpoint Name
Checkpoint 2.31A fix — remaining memory-policy gaps plus lightweight runtime/memory profiling.

## 2. Files Inspected
- `CODEX_CHECKPOINT_2_31A_FIX_RUNTIME_MEMORY_PROFILE.md`
- `visual_slam/orbslam/slam/map.py`
- `visual_slam/orbslam/slam/local_mapping_core.py`
- `visual_slam/orbslam/slam/local_mapping.py`
- `visual_slam/orbslam/slam/tracking.py`
- `visual_slam/orbslam/slam/slam.py`
- `visual_slam/orbslam/slam/config_parameters.py`
- `visual_slam/orbslam/run_rgbd_slam.py`
- `visual_slam/orbslam/run_tum_rgbd_smoke.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_31A_memory_policy.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_29A_rgbd_runner.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_12_local_mapping.py`
- `visual_slam/reference_audit/checkpoint_2_31A/PRE_CHANGE_PYSLAM_MEMORY_POLICY_AUDIT.md`
- `visual_slam/reference_audit/checkpoint_2_31A/IMPLEMENTATION_AUDIT.md`

## 3. pySLAM Files Inspected
- `third_party/pyslam_reference/pyslam/slam/map.py`
- `third_party/pyslam_reference/pyslam/slam/local_mapping.py`
- `third_party/pyslam_reference/pyslam/slam/local_mapping_core.py`
- `third_party/pyslam_reference/pyslam/slam/slam.py`
- `third_party/pyslam_reference/pyslam/slam/frame.py`

## 4. Current Checkpoint 2.31A Memory-Policy Implementation Summary
The local codebase already contains the main 2.31A building blocks:

- `Map.add_frame()` manually evicts the oldest normal frame and calls `_cleanup_evicted_frame()`.
- `Map.prune_old_frame_views()` removes stale non-keyframe frame views from map points.
- `Map.memory_stats()` reports counts for recent frames, keyframes, frame views, and image retention.
- `Frame`/`KeyFrame` expose `release_heavy_data()` and `heavy_memory_bytes()`.
- `LocalMapping.step()` and the background thread path explicitly clear `img_cur`, `img_cur_right`, and `depth_cur`.
- `run_rgbd_slam.py` already supports `--profile-memory`, `--memory-profile-every`, `--memory-limit-gb`, `--lean-memory`, `--no-heavy-loop-reports`, and `--no-loop-candidate-pair-reports`.

This means the current work is integration-focused rather than a fresh policy implementation.

## 5. Remaining Gaps Found In Code
### 5.1 `Map.frames` source of truth mismatch
- `visual_slam/orbslam/slam/map.py` still defines a local module constant `kMaxLenFrameDeque = 20`.
- `Map.__init__()` constructs `self.frames = deque(maxlen=kMaxLenFrameDeque)` instead of reading `Parameters.kMaxLenFrameDeque`.
- `Map.add_frame()` uses `Parameters.kMaxLenFrameDeque` as fallback during eviction, so construction and eviction are not using the same source of truth.

### 5.2 Lean-memory suppression is incomplete
- `run_rgbd_slam.py` sets `no_loop_candidate_pair_reports = True` in lean-memory mode.
- The actual write path still uses `if dump_loop_candidate_reports:` and ignores `no_loop_candidate_pair_reports`.
- Result: heavy per-candidate pair reports can still be written in lean-memory mode if dumping is otherwise enabled.

### 5.3 Global `Parameters` mutation leaks across runs
- `run_rgbd_slam.py` mutates `Parameters` directly for lean-memory execution.
- The function does not snapshot/restore the mutated globals.
- If the run raises an exception, later tests and later in-process runs inherit the modified settings.

### 5.4 Old smoke runner is a separate execution path
- `run_tum_rgbd_smoke.py` still implements its own dataset loop, loop-report writing, and artifact generation.
- It does not pick up the new profiling behavior or parameter-restoration fix automatically.
- It is therefore still a bypass path for the final runner behavior.

### 5.5 Keyframe culling still uses compact point indices
- `LocalMappingCore.cull_keyframes()` currently builds:
  `[(i, p) for i, p in enumerate(kf.get_points()) if p is not None and not p.is_bad()]`
- This is safe only if `kf.get_points()` preserves original keypoint indexing.
- The task requirement explicitly identifies the remaining bug pattern as compact-index misuse and requires switching to original keypoint indices via `get_matched_good_points_and_idxs()`.
- The fix is structurally important because `depths[i]` and `octaves[i]` must always use original feature indices.

### 5.6 Runtime profiling is missing
- There is no reusable runtime profiler utility.
- `run_rgbd_slam.py` does not write `runtime_profile.csv`, `runtime_profile.json`, or per-frame timing CSV output.
- The current run artifacts cannot explain whether the 300-frame slowdown is dominated by tracking, local mapping, loop work, frame-view pruning, or memory scans.

### 5.7 Memory profiling is deep and unconditional once enabled
- `run_rgbd_slam.py` calls `slam.map.memory_stats()` every profiling interval.
- `Map.memory_stats()` always scans all map points and all keyframes, and always computes heavy byte estimates.
- This is acceptable for deep diagnostics but too expensive as the default profiling mode for longer runs.

### 5.8 Frame-view pruning frequency may be too aggressive
- `Parameters.kFrameViewPruneEveryNFrames` defaults to `5`.
- The runner calls `slam.map.prune_old_frame_views()` synchronously from the main loop.
- Because pruning scans all map points, its overhead could become significant on longer sequences.

## 6. Current Runner Behavior
### 6.1 `--max-frames`
- Applied by slicing the loaded RGB-D association list before execution.
- Behavior is correct and straightforward.

### 6.2 `--lean-memory`
- Disables several debug outputs and forces frame/keyframe image-retention flags off.
- Sets `no_loop_candidate_pair_reports = True`.
- Does not restore the modified `Parameters`.
- Does not currently suppress the pair-report writer at the actual write site.

### 6.3 `--profile-memory`
- When enabled, the runner samples RSS and calls `slam.map.memory_stats()` and `slam.local_mapping.queue_memory_stats()` every `memory_profile_every` frames.
- Writes `memory_profile.csv`.
- Stops the run if RSS exceeds `--memory-limit-gb`.

### 6.4 Loop candidate reports
- Loop debug CSV rows are appended when `--loop-debug` is enabled.
- Heavy per-candidate pair reports are written when `dump_loop_candidate_reports` is true.
- The writer currently does not check `no_loop_candidate_pair_reports`, which is the main suppression gap.

### 6.5 Map export
- Export still happens by default through `export_orbslam_map()` unless `--no-map-export` is used.
- This matches the task instruction to keep export disabled during short profiling validation.

### 6.6 Parameter mutation
- The runner mutates global `Parameters` inline.
- No restore path exists in success or failure cases.

## 7. Current Keyframe-Culling Implementation And Index-Bug Status
### Local behavior before change
- `LocalMappingCore.cull_keyframes()` iterates keyframe points and uses the resulting index to access `kf.depths[i]` and `kf.octaves[i]`.
- The intended semantics are “original keypoint index,” not “compact matched-point index.”

### pySLAM behavior
- The inspected `pyslam/slam/local_mapping_core.py` still uses the same compact-style enumeration in Python.
- `process_new_keyframe()` already uses `get_matched_good_points_and_idxs()`, which shows the upstream codebase already recognizes the need for original indices in some critical paths.

### Gap
- For culling, the local implementation needs the stronger invariant required by this checkpoint: use original keypoint indices explicitly.

## 8. Root Cause / Current Hypothesis
The remaining memory and runtime issues are primarily integration issues:

- global configuration is being mutated without isolation;
- lean-memory suppression is applied in argument state but not enforced at the write site;
- profiling currently measures memory but not time;
- the existing memory profiling path is too deep for routine runtime diagnostics;
- frame-view pruning and memory scans likely contribute measurable overhead during medium runs;
- keyframe culling may retain too many keyframes because it can read the wrong depth/octave slots.

## 9. Profiling Design To Implement
### 9.1 Runtime profiler
- Add a small `runtime_profiler.py` utility using `time.perf_counter()`.
- Provide `start()`, `stop()`, `section()`, `to_dict()`, `write_csv()`, and `write_json()`.
- Default to a no-op mode when profiling is disabled.

### 9.2 Per-frame timing output
- Add `frame_timing.csv` with:
  `frame_idx`, `timestamp`, `frame_total_sec`, `load_rgb_sec`, `load_depth_sec`, `slam_track_sec`, `local_mapping_sec`, `loop_closing_sec`, `prune_old_frame_views_sec`, `memory_profile_sec`, `rss_mb`, `keyframes`, `map_points`, `recent_frames`, `num_frame_views_total`, `old_frame_views_total`.

### 9.3 High-level instrumented sections
- `frame.total`
- `frame.load_rgb`
- `frame.load_depth`
- `slam.track`
- `local_mapping.step`
- `loop_closing.step`
- `memory.prune_old_frame_views`
- `memory.profile_snapshot`
- `frame.log_write`

### 9.4 Internal sections where practical
- tracking: previous/reference/local map/new-keyframe decisions
- local mapping: process/cull/triangulate/fuse/local BA/keyframe cull
- loop closing: detect candidates/correct loop/essential graph/global BA

### 9.5 Cheap vs deep memory profiling
- Cheap mode default:
  RSS, keyframes, map points, recent frames, queue size, cached view counters when already available.
- Deep mode opt-in:
  full `Map.memory_stats()` scan including frame-view totals and heavy-byte estimates.

## 10. Why The Planned Changes Are Structurally Correct
- They do not alter tracking thresholds, BA settings, feature extraction, camera logic, or SLAM geometry.
- They isolate runner-only configuration changes so tests and repeated runs are deterministic.
- They make profiling explicit and lightweight enough for medium/overnight runs.
- They tighten keyframe-culling indexing semantics without changing the culling policy itself.

## 11. Tests Planned
- `test_map_frames_uses_parameters_maxlen`
- `test_lean_memory_disables_pair_report_dumping`
- `test_run_rgbd_slam_restores_parameters_after_lean_memory`
- `test_run_rgbd_slam_restores_parameters_after_exception`
- `test_cull_keyframes_uses_original_keypoint_indices`
- `test_runtime_profiler_records_sections`
- `test_runtime_profiler_writes_csv_and_json`
- `test_frame_timing_csv_columns`
- `test_memory_profile_disabled_does_not_call_deep_memory_stats`
- `test_prune_old_frame_views_runtime_section_exists_when_enabled`

## 12. Next Recommended Action
Implement the runner parameter snapshot/restore, fix the pair-report suppression gate, wire in lightweight runtime profiling plus `frame_timing.csv`, switch memory profiling to cheap-by-default, fix keyframe-culling original-index access, then run the targeted tests and the requested 30-frame/100-frame validations.
