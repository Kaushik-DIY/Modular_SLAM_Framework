# Checkpoint 2.35E_H — LOOP_GEOMETRY_SUPPORT_REPORT

## 1. Stage / checkpoint name

- `Stage H — Geometry, projection expansion, and final support audit`

## 2. pySLAM reference used

- `third_party/pyslam_reference/pyslam/loop_closing/loop_closing.py::LoopGeometryChecker.check_candidates`
- `third_party/pyslam_reference/pyslam/slam/geometry_matchers.py`
- `third_party/pyslam_reference/pyslam/slam/optimizer_g2o.py`

## 3. Local behavior before change

- RGB-D fixed-scale SE3 path instead of monocular Sim3
- extra estimated-pose gate before guided refinement
- no dedicated geometry trace CSV
- projection expansion and final support data only visible indirectly through loop-debug records

## 4. Changes made

- disabled the local pose-prior gate by default:
  - `kLoopClosingMaxEstimatedPoseDistanceForGuidedSE3 = 0.0`
  - `kLoopClosingMaxEstimatedPoseRotationDegForGuidedSE3 = 0.0`
- added geometry-trace diagnostics:
  - seed correspondences
  - seed inliers
  - seed inlier ratio
  - initial SE3 translation / rotation
  - pose-gate pass flag
  - refined correspondences / inliers
  - candidate-group size / map-point counts
  - visible projected group points
  - final matched map-point count
- emitted:
  - `loop_geometry_trace.csv`

## 5. Alignment scores

- `BoW-guided matching`: `95/100`
- `RGB-D SE3 geometry path`: `95/100`
- `projection expansion`: `96/100`
- `final support diagnostics`: `96/100`

Remaining deliberate deviation:

- the implementation remains RGB-D scale-fixed SE3 rather than full pySLAM monocular Sim3
- that deviation is within current accepted scope

## 6. Test evidence

- `tests/visual_slam/orbslam/test_checkpoint_2_35H_loop_geometry_support.py`: `7 passed`
- retained older geometry regressions:
  - `tests/visual_slam/orbslam/test_checkpoint_2_21_loop_closing.py`
  - `tests/visual_slam/orbslam/test_checkpoint_2_28A_loop_projection_expansion.py`

## 7. Full-run evidence

From the no-GBA run:

- `PASSED_GEOMETRY = 1`
- `FAILED_SEED_GEOMETRY = 2`
- `FAILED_REFINED_GEOMETRY = 1`
- `FAILED_FINAL_SUPPORT = 1`

Critical blocker:

- the single accepted runtime loop was GT-negative:
  - `KF42 <-> KF4`
  - `gt_translation_distance = 1.5236 m`
  - `gt_rotation_angle_deg = 42.6379`

## 8. Honest outcome

Stage H observability improved substantially and the local non-reference pose gate is no longer active by default.  
However, the no-GBA run still shows false-loop acceptance, so geometry / final support cannot be called benchmark-ready.
