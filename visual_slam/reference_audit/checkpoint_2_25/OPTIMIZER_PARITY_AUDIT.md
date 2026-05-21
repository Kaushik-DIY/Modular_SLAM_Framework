# Checkpoint 2.25 Optimizer Parity Audit

## 1. pySLAM Optimizer Files Inspected

- `pyslam/slam/optimizer_g2o.py`
- `pyslam/slam/optimizer_gtsam.py`
- `pyslam/slam/global_bundle_adjustment.py`
- `pyslam/loop_closing/loop_closing.py`
- `pyslam/slam/map.py`
- `pyslam/slam/keyframe.py`
- `pyslam/slam/map_point.py`
- `pyslam/config_parameters.py`

## 2. Local Optimizer Files Inspected

- `visual_slam/orbslam/slam/optimizer_g2o.py`
- `visual_slam/orbslam/slam/global_ba.py`
- `visual_slam/orbslam/slam/essential_graph.py`
- `visual_slam/orbslam/slam/loop_closing.py`
- `visual_slam/orbslam/slam/map.py`

## 3. Pose-Only Optimization Comparison

pySLAM builds fixed XYZ-only pose edges and rejects outliers over multiple rounds. The local implementation keeps the same four-round structure and now skips non-finite or negative-depth observations before edge creation.

## 4. Local BA Comparison

pySLAM local BA optimizes local keyframes and points, fixes reference keyframes, robustly classifies outliers, removes bad observations, then writes back. The local path preserves this behavior and keeps outlier pruning enabled for local BA only.

## 5. Global BA Comparison

pySLAM Global BA optimizes all valid keyframes and map points, fixes the origin/root, and stores updates for later safe correction. The local checkpoint adds a synchronous wrapper with deferred optimizer updates and atomic write-back.

## 6. Essential Graph Comparison

pySLAM uses Sim3 graph optimization with fixed scale for non-monocular sensors. The local RGB-D path remains SE3-only and uses finite-pose validation before write-back.

## 7. Edge Type Comparison

- Mono/RGB-D BA edges: local g2o compatibility edges mirror pySLAM projection constraints.
- RGB-D uses virtual-stereo `uR` when available.
- Essential graph uses SE3 pose-pose edges instead of Sim3 edges.

## 8. Information Matrix/Weighting Comparison

BA edges use octave inverse variance. Essential graph edges now use conservative non-identity weights:

- spanning tree: structural base weight
- covisibility: clamped weight based on connection strength
- loop edges: stronger loop-constraint weight

## 9. Robust Kernel Comparison

Pose-only and local BA use Huber kernels with chi-square deltas. Global BA supports pySLAM-style robust-first/final optimization when enabled.

## 10. Chi-Square/Outlier Policy Comparison

Mono uses `5.991`; stereo/RGB-D uses `7.815`. Local BA removes final outlier observations. Global BA only disables/rejects outliers for diagnostics and safe result selection.

## 11. Positive-Depth Check Comparison

Pose-only, local BA, and global BA now avoid non-finite/negative-depth observations before edge creation and re-check positive depth during outlier classification or safe GBA validation.

## 12. Write-Back/Failure Safety Comparison

Local BA writes back after successful optimization. Global BA validates all updates first and applies them under the map update lock only after all checks pass. Failure preserves old poses and map-point positions.

## 13. Changes Implemented

- Added `GlobalBAResult` and `GlobalBundleAdjuster`.
- Added deferred low-level BA write-back support.
- Split local BA pruning from global BA non-pruning.
- Added positive-depth/non-finite pre-edge filtering.
- Added essential graph weighting parameters and edge information matrices.
- Added loop and smoke-run diagnostics for GBA.

## 14. Tests and Results

- `test_checkpoint_2_25_optimizer_parity.py`: `11 passed in 0.95s`
- Full ORB-SLAM unit folder: `176 passed, 1 skipped in 6.26s`

## 15. 100-Frame Validation Results

- Output: `visual_slam_outputs/checkpoint_2_25_fr1_desk_100_pyslam_orb2_loop_gba`
- Backend: `pyslam_orb2`
- Frames attempted: 100
- Tracking OK: 100
- Tracking lost: 0
- Errors: 0
- Final state: OK
- Final keyframes: 7
- Final map points: 3631
- Elapsed: 468.779 seconds
- Bad-pattern log scan: clean
- Runtime GBA trigger count: 0 accepted-loop GBA starts in the 100-frame log; loop-triggered GBA path is covered by synthetic loop tests.

## 16. Full 596-Frame pyslam_orb2-Only Result

- Output: `visual_slam_outputs/checkpoint_2_25_fr1_desk_full_pyslam_orb2_loop_gba`
- Backend: `pyslam_orb2`
- Frames attempted: 596
- Tracking OK: 596
- Tracking lost: 0
- Errors: 0
- Final state: OK
- Final keyframes: 18
- Final map points: 4020
- Trajectory poses: 596
- Elapsed: 1728.695 seconds
- Bad-pattern log scan: clean
- Runtime GBA trigger count: 0 accepted-loop GBA starts in the full-sequence log; no real loop closure was accepted on this run.
- Evaluation: ATE RMSE SE3 `0.027362178948210913 m`; ATE RMSE Sim3 `0.02734044771005415 m`; RPE trans RMSE `0.011095358381673403 m`; RPE rot RMSE `0.6358914310615538 deg`; associations `596`

## 17. Deviations from pySLAM

- RGB-D SE3 path only; no monocular Sim3 parity.
- Synchronous GBA instead of background process/thread.
- No C++ core optimizer backend or GTSAM backend in this local port.

## 18. Remaining Gaps

- Full quantitative benchmark is deferred to Checkpoint 2.26.
- Background GBA/new-keyframe propagation remains future work if local mapping becomes threaded.
