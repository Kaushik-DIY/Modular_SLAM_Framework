# Checkpoint 2.24 Global BA Audit

## 1. pySLAM Global BA Files Inspected

- `pyslam/loop_closing/loop_closing.py`
- `pyslam/slam/global_bundle_adjustment.py`
- `pyslam/slam/optimizer_g2o.py`
- `pyslam/slam/optimizer_gtsam.py`
- `pyslam/slam/map.py`
- `pyslam/slam/keyframe.py`
- `pyslam/slam/map_point.py`
- `pyslam/config_parameters.py`

## 2. Current Local Files Inspected

- `visual_slam/orbslam/slam/loop_closing.py`
- `visual_slam/orbslam/slam/global_ba.py`
- `visual_slam/orbslam/slam/optimizer_g2o.py`
- `visual_slam/orbslam/slam/essential_graph.py`
- `visual_slam/orbslam/slam/map.py`
- `visual_slam/orbslam/slam/keyframe.py`
- `visual_slam/orbslam/slam/map_point.py`
- `visual_slam/orbslam/slam/slam.py`
- `visual_slam/orbslam/run_tum_rgbd_smoke.py`

## 3. How pySLAM Triggers Global BA

pySLAM accepts a loop, corrects/fuses map points, updates covisibility, optimizes the essential graph, inserts loop edges, then starts `GlobalBundleAdjustment` when `Parameters.kUseGBA` is enabled.

## 4. What Was Implemented

Added `GlobalBundleAdjuster`, a synchronous RGB-D GBA coordinator that collects valid keyframes and map points, runs g2o Global BA with deferred optimizer updates, validates results, writes back atomically, recomputes point info, and refreshes covisibility.

## 5. Synchronous-vs-Threaded Decision

pySLAM uses a background process/thread and later correction propagation. This port is synchronous for deterministic tests and validation. The optimizer accepts a stop/abort flag and keeps deferred write-back semantics.

## 6. Keyframe/Map-Point Selection

All non-bad keyframes are selected. Non-bad map points with sufficient observations and finite positions are selected.

## 7. Edge Construction

The g2o BA core builds SE3 pose vertices, XYZ point vertices, and mono or RGB-D virtual-stereo reprojection edges. RGB-D edges use valid `uR` observations and octave inverse variance.

## 8. Robust Kernel and Outlier Policy

Global BA follows pySLAM's robust-then-final schedule when robust kernels are enabled. Outlier observations are disabled for optimization diagnostics but are not pruned in Global BA. Local BA remains responsible for observation pruning.

## 9. Abort/Stop Flag Behavior

The synchronous wrapper checks the flag before graph construction, after optimizer return, and before write-back. The low-level g2o path also receives the flag when the binding supports it.

## 10. Safe Write-Back Behavior

Optimized poses and points are validated for finite SE3, orthonormal rotations, bounded translation jumps, finite point positions, positive-depth observers, and enough inlier edges. Write-back occurs only after all validation passes. On failure, old map state is preserved.

## 11. Loop Integration

`LoopCorrector` triggers GBA after successful SE3 essential graph correction and loop-edge insertion only when `Slam.enable_global_ba` and `Slam.global_ba_after_loop` are true. Runtime defaults keep GBA disabled unless explicitly enabled.

## 12. Diagnostics

Loop diagnostics now include:

- `global_ba_started`
- `global_ba_success`
- `global_ba_aborted`
- `global_ba_reason`
- `global_ba_num_keyframes`
- `global_ba_num_map_points`
- `global_ba_num_edges`
- `global_ba_num_inliers`
- `global_ba_num_outliers`
- `global_ba_mean_error_before`
- `global_ba_mean_error_after`
- `global_ba_elapsed_sec`

## 13. Unit Tests

`tests/visual_slam/orbslam/test_checkpoint_2_24_global_ba.py` covers imports, graph collection, gauge fixing, synthetic error reduction, write-back, point info recomputation, failure preservation, abort handling, loop trigger behavior, disabled path, and diagnostics.

Result: `11 passed in 2.18s`

## 14. 100-Frame Validation

- Output: `visual_slam_outputs/checkpoint_2_24_fr1_desk_100_pyslam_orb2_gba`
- Backend: `pyslam_orb2`
- Frames attempted: 100
- Tracking OK: 100
- Tracking lost: 0
- Errors: 0
- Final state: OK
- Final keyframes: 7
- Final map points: 3625
- Elapsed: 466.880 seconds
- Bad-pattern log scan: clean
- Runtime GBA trigger count: 0 accepted-loop GBA starts in this 100-frame log; loop-triggered GBA is validated by synthetic loop tests.

## 15. Deviations from pySLAM

- Synchronous instead of background process/thread.
- RGB-D SE3 scope only; no monocular Sim3 GBA parity.
- No new-keyframe propagation during background GBA because local mapping is sequential in this port.
- Local Python g2o binding uses parameter-based projection edges through `visual_slam.g2o_compat`.

## 16. Remaining Gaps

- No background GBA correction propagation for keyframes inserted during GBA.
- No GTSAM backend port for this local implementation.
- Full benchmark remains for Checkpoint 2.26.
