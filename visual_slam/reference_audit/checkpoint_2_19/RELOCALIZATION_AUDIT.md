# Checkpoint 2.19 Relocalization Audit

## pySLAM files inspected

- `third_party/pyslam_reference/pyslam/slam/relocalizer.py`
- `third_party/pyslam_reference/pyslam/slam/tracking.py`
- `third_party/pyslam_reference/pyslam/slam/frame.py`
- `third_party/pyslam_reference/pyslam/slam/keyframe.py`
- `third_party/pyslam_reference/pyslam/slam/map.py`
- `third_party/pyslam_reference/pyslam/slam/map_point.py`
- `third_party/pyslam_reference/pyslam/slam/optimizer_g2o.py`
- `third_party/pyslam_reference/pyslam/local_features/feature_matcher.py`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_closing.py`
- `third_party/pyslam_reference/pyslam/loop_closing/keyframe_database.py`

## Current files inspected

- `visual_slam/orbslam/slam/tracking.py`
- `visual_slam/orbslam/slam/frame.py`
- `visual_slam/orbslam/slam/keyframe.py`
- `visual_slam/orbslam/slam/map.py`
- `visual_slam/orbslam/slam/map_point.py`
- `visual_slam/orbslam/slam/geometry_matchers.py`
- `visual_slam/orbslam/slam/optimizer_g2o.py`
- `visual_slam/orbslam/slam/slam.py`
- `visual_slam/orbslam/run_tum_rgbd_smoke.py`

## pySLAM relocalization flow summary

pySLAM enters relocalization from `Tracking` when tracking is not OK. It queries loop/BoW relocalization candidates, matches the current frame to each candidate keyframe, filters orientation, prepares 3D-2D correspondences from candidate map points, estimates pose with the PnP solver, runs pose-only optimization, removes outliers, expands matches by projection, and optimizes again before accepting a candidate. On success it sets the current frame reference keyframe and restores tracking to OK; on failure it leaves tracking LOST and does not write a bad pose into the map.

## Implementation files changed

- `visual_slam/orbslam/slam/relocalizer.py`
- `visual_slam/orbslam/slam/tracking.py`
- `visual_slam/orbslam/slam/__init__.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_19_relocalization.py`
- `visual_slam/reference_audit/checkpoint_2_19/RELOCALIZATION_AUDIT.md`

## PnP backend used

The local implementation uses OpenCV `cv2.solvePnPRansac` with `cv2.SOLVEPNP_EPNP`, followed by `cv2.solvePnPRefineLM` when available. The pySLAM C++ `pnpsolver.MLPnPsolver` wrapper is not available in this workspace, so this is a compatibility backend.

## Temporary candidate database behavior

`TemporaryRelocalizationKeyFrameDatabase` exposes the same `detect_relocalization_candidates(frame)` interface expected from the Checkpoint 2.20 database. Before BoW is available, it returns non-bad map keyframes in recent-first order. Tracking only calls the database-shaped interface through `Relocalizer`, so it is not coupled to the fallback.

## Implementation summary

- Added `Relocalizer` with pySLAM-style candidate loop, descriptor matching, rotation filtering, PnP, pose-only optimization, projection match expansion, second optimization, and success/failure bookkeeping.
- Added instrumentation fields: candidate count, match count, inlier count, last success flag, last keyframe id, and last error.
- Wired `Tracking.relocalize()` and the LOST/RELOCALIZE branch to call `Relocalizer`.
- On relocalization success, tracking restores `SlamState.OK`, updates the reference keyframe/local map, records `last_reloc_frame_id`, and suppresses immediate motion-model reuse.
- On relocalization failure, tracking leaves the state LOST and restores the frame pose/points for each rejected candidate.

## Deviations from pySLAM and reasons

- OpenCV PnP replaces pySLAM `pnpsolver.MLPnPsolver` because the local C++ wrapper is unavailable.
- BoW candidate retrieval now uses `KeyFrameDatabase` when the DBOW backend is available. The temporary map-scan fallback remains only for missing/unavailable BoW and is explicit.
- BoW-guided descriptor matching is used when the database is available. Fallback descriptor matching is explicit through `last_fallback_descriptor_matching`.
- Relocalization runs synchronously in the tracking thread. pySLAM routes relocalization through loop closing when loop closing is active; that integration is deferred to Checkpoints 2.20 and 2.21.

## Validation output summary

- Baseline before edits:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/visual_slam/orbslam`
  - Result: `100 passed, 1 skipped`
  - `tools/validate_orbslam_pyslam_port.py ... checkpoint_2_19_baseline_validation`
  - Result: `VALIDATION PASSED`
- Checkpoint 2.19 focused tests:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_19_relocalization.py`
  - Result: `9 passed`
- Full ORB-SLAM suite:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/visual_slam/orbslam`
  - Result: `109 passed, 1 skipped`
- TUM no-regression validation:
  - `.venv/bin/python tools/validate_orbslam_pyslam_port.py --dataset "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_19_validation"`
  - Result: `VALIDATION PASSED`
  - Smoke 30 summary: `tracking_ok_count=30`, `tracking_lost_count=0`, `errors=0`, `final_state=OK`, `final_keyframes=6`, `final_map_points=3473`
- Revalidation after 2.20/2.21 work:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_19_relocalization.py tests/visual_slam/orbslam/test_checkpoint_2_20_bow_keyframe_database.py tests/visual_slam/orbslam/test_checkpoint_2_20_bow_guided_matching.py`
  - Result: `23 passed`
  - Full suite after final edits: `134 passed, 1 skipped`
  - Default TUM validation after final edits: `VALIDATION PASSED`

## Remaining relocalization gaps

- PnP uses OpenCV instead of pySLAM's MLPnP wrapper.
- Relocalization is synchronous and in-process rather than routed through pySLAM's loop detector process.

## Files changed

- `visual_slam/orbslam/slam/relocalizer.py`
- `visual_slam/orbslam/slam/tracking.py`
- `visual_slam/orbslam/slam/__init__.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_19_relocalization.py`
- `visual_slam/reference_audit/checkpoint_2_19/RELOCALIZATION_AUDIT.md`
