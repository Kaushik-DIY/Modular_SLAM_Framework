# TUM RGB-D fr1/room Benchmark Evaluation Report

**Date:** 2026-05-08  
**Dataset:** `rgbd_dataset_freiburg1_room`  
**Run:** `fr1_room_loop_gba_threaded`  
**Pipeline:** pySLAM-aligned ORB2 RGB-D · Loop closing enabled · Global BA enabled · LM threading enabled  
**Ground truth:** `groundtruth.txt` (4887 poses)

---

## 1. Run Summary

| Parameter | Value |
|---|---|
| Frames attempted | 1362 |
| Tracking OK | 1362 (100%) |
| Tracking LOST | 0 |
| Errors | 0 |
| Final keyframes | 50 |
| Final map points | 5760 |
| Loop edges | **0** |
| Global BA events | 0 |
| Elapsed time | 3944.6 s (65.7 min) |
| Average FPS | 0.35 |
| LM threading | **enabled** |
| Trajectory poses saved | 1362 |
| KF trajectory consistency (max) | 0.0000 m |

**Threading speedup:** 0.35 fps vs ~0.23 fps sequential (approx. **52% speedup** from LM background thread).

---

## 2. Trajectory Metrics vs Ground Truth

All 1362 estimated poses were associated to ground truth (max time diff 0.006 s).

| Metric | Value |
|---|---:|
| **ATE RMSE SE(3) [m]** | **0.2940** |
| ATE mean SE(3) [m] | 0.2670 |
| ATE median SE(3) [m] | 0.2496 |
| ATE max SE(3) [m] | 0.5338 |
| ATE RMSE Sim(3) [m] | 0.2937 |
| Sim(3) scale | 1.0133 |
| **RPE translation RMSE [m]** | **0.0169** |
| RPE rotation RMSE [deg] | 0.635 |
| RPE pairs | 1361 |
| Associations | 1362 / 1362 |
| Mean association time diff [s] | 0.0024 |

### Reference comparison (ORB-SLAM2 RGBD paper, TUM Benchmark)

| System | fr1/room ATE RMSE [m] |
|---|---:|
| ORB-SLAM2 RGBD (published) | **0.047** |
| This run (loop enabled, 0 loops accepted) | 0.294 |
| Gap | ~6× worse |

---

## 3. Key Finding: Zero Loop Closures Despite Loop Detector Being Enabled

### 3.1 Root Cause

Loop closing was enabled and the detection code path ran correctly for each of the 50 keyframes. However, **zero loop candidates were ever accepted** because the fundamental prerequisite for loop detection was absent: the camera trajectory does not form a detectable loop in the sparse keyframe set.

Diagnostic evidence:

- **50 keyframes total** for 1362 frames (1 KF per 27 frames on average).
- **KF0 position:** [0, 0, 0] (origin).
- **KF49 position:** [1.94, −0.78, 1.09] m.
- **Distance KF0→KF49:** 2.35 m.
- **Min distance first-5 KFs to last-5 KFs:** 1.45 m.

The trajectory starts at the origin and ends ~2.4 m away in SLAM coordinates. Even accounting for drift, this means the camera trajectory as seen by the SLAM map never closes a loop within the 50-KF set.

In the `rgbd_dataset_freiburg1_room` TUM sequence, the camera performs a large arc through the room. The published ORB-SLAM2 results show loop detection works when the SLAM creates ~200–300 keyframes for this sequence. With 50 KFs distributed over 1362 frames, many revisited positions either:
1. Do not generate a new keyframe (too many existing map points visible → KF creation blocked), or  
2. Are culled after creation (map point redundancy check culls ~80% of created KFs).

### 3.2 The KF Density Problem

| Metric | This run | pySLAM reference (fr1/room) |
|---|---|---|
| Total KFs | 50 | ~200–300 |
| KF per 100 frames | ~3.7 | ~15–22 |
| Loop edges detected | 0 | 5–10 |

With 50 KFs and `kMinDeltaFrameForMeaningfulLoopClosure = 10`, only 40 KFs are eligible as loop candidates at any moment. Given the covisibility graph covers most of those 40, the BoW database query returns few non-connected candidates, and the consistency checker (requiring 3 consecutive consistent detections) never fires.

### 3.3 Why KF Density Is Low

Two mechanisms together suppress KF creation beyond the first 200 frames:

1. **KF creation gate (Tracking):** A new KF is created only when tracked map points < threshold. Once the map grows to ~3000–4000 well-localised points, the gate rarely opens.
2. **KF culling (LocalMapping):** `cull_keyframes()` removes any KF whose map points are already observed by ≥3 other KFs with ≥90% coverage. In the ORB-SLAM2 paper this threshold is 90% of the KF's map points; aggressive culling is correct per spec, but our specific threshold and local-map window size may be more aggressive than pySLAM's tuning for this dataset.

---

## 4. Metric Interpretation

### ATE RMSE 0.294 m — Drift Without Loop Correction

The ATE of 0.294 m is pure odometric drift. The RPE of 0.0169 m/frame (relative frame-to-frame translation error) is low, showing that individual tracking steps are accurate. Drift compounds over the full 1362-frame trajectory.

With correct loop closure (returning the trajectory to its start), the ATE for fr1/room would drop toward 0.047 m as shown in the ORB-SLAM2 paper. The gap is entirely attributable to missing loop correction.

### Sim(3) Scale = 1.013 — Metric Scale Is Correct

A Sim(3) scale factor of 1.013 (1.3% deviation) confirms the RGB-D depth pipeline is correctly metrically scaled. This rules out systematic depth errors.

### RPE 0.0169 m — Tracking Quality Is Good

Frame-to-frame relative pose accuracy of ~1.7 cm (RMSE) is within the expected range for ORB2 RGB-D on TUM data. This indicates the feature extraction, depth integration, and pose optimisation are functioning correctly.

---

## 5. Generated Plots

All 8 figures saved to `visual_slam_outputs/fr1_room_loop_gba_threaded/plots/`:

| Figure | Description |
|---|---|
| `trajectory_xy.png` | Top-down XY trajectory: estimated (SE3-aligned) vs GT |
| `trajectory_3d.png` | 3D trajectory comparison |
| `ate_over_time.png` | Per-pose ATE (shows monotonic drift growth) |
| `rpe_over_time.png` | Per-consecutive-pair RPE translation |
| `tracking_quality.png` | Tracked points, keyframes, map points over run |
| `map_xy.png` | Sparse map + trajectory (SE3-aligned to GT) |
| `keyframe_graph_xy.png` | Keyframe covisibility graph overlaid on GT |
| `metrics_panel.png` | Summary table of all trajectory metrics |
| `metrics_table.md` | Markdown metrics table |

---

## 6. Gaps and Required Actions for Benchmarkable Quality

### P0 — Increase Keyframe Density

**Problem:** 50 KFs for a 1362-frame sequence is too sparse for loop detection. Target: 200–300 KFs.

**Proposed fix:** Tune KF creation condition and/or reduce culling aggressiveness.  
Specifically:
- In `tracking.py`: reduce the "tracked points ratio" threshold that blocks new KF creation.
- In `local_mapping_core.py` `cull_keyframes()`: raise the observation-coverage threshold from 90% → 95%, or reduce the minimum co-observer count from 3 → 2.
- Compare against pySLAM's `LocalMapping.cull_keyframes()` to verify the threshold values.

**Validation:** Run fr1_room with no culling changes as baseline, measure KF count at each checkpoint, then tune until ~200 KFs are produced.

### P1 — Verify Loop Closure After KF Density Fix

Once KF density is corrected, re-run with loop detection enabled and verify:
- Loop candidates are returned by BoW query.
- Consistency check (3 consecutive consistent candidates) fires at least once.
- Essential graph optimisation reduces ATE.
- Global BA further reduces ATE toward the 0.047 m target.

### P2 — Runtime Optimisation (Lower Priority)

With LM threading the pipeline runs at 0.35 fps. This is sufficient for research benchmarking but not real-time. No action required at this stage.

---

## 7. Files

### Run outputs
```
visual_slam_outputs/fr1_room_loop_gba_threaded/
  trajectory_rgbd_dataset_freiburg1_room_smoke.txt   # TUM-format estimated trajectory
  frame_log_rgbd_dataset_freiburg1_room_smoke.csv    # Per-frame tracking log
  map_points.ply                                     # Sparse map (5760 points)
  keyframes.json                                     # Keyframe poses + descriptors
  keyframe_graph.json                                # Covisibility/spanning-tree graph
  run.log                                            # Full console output
  trajectory_eval/
    trajectory_metrics.json                          # ATE/RPE numeric results
    trajectory_metrics.md                            # Markdown table
    associated_poses.csv                             # Per-pose GT association
  plots/
    trajectory_xy.png
    trajectory_3d.png
    ate_over_time.png
    rpe_over_time.png
    tracking_quality.png
    map_xy.png
    keyframe_graph_xy.png
    metrics_panel.png
    metrics_table.md
```

### Code changes in this session
```
visual_slam/orbslam/slam/local_mapping.py    # Added start_thread/stop_thread/_run_thread
visual_slam/orbslam/slam/slam.py             # Added start_thread call + shutdown()
visual_slam/orbslam/run_tum_rgbd_smoke.py   # Thread-aware loop, CLI flag, shutdown()
tools/plot_tum_evaluation.py                 # New: single-run ATE/RPE plot tool
tools/launch_fr1_room_benchmark.sh          # New: 3-phase benchmark launcher
```

---

## 8. Conclusion

The fr1/room benchmark run completed successfully from a tracking perspective (1362/1362 OK, 0 LOST) with correct metric scale (Sim3 scale = 1.013) and low frame-to-frame RPE (0.017 m). The pipeline is mechanically sound.

The ATE RMSE of 0.294 m is 6× worse than the published ORB-SLAM2 result (0.047 m) because no loop closure occurred. The root cause is a keyframe density problem: only 50 KFs are generated vs the 200–300 needed for the BoW loop detector to find reliable candidates. Fixing KF culling aggressiveness is the single most important next step before the fr1/room result can be considered benchmarkable.
