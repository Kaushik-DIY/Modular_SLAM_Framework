# Checkpoint 2.21 Loop Closing Audit

## pySLAM files inspected

- `third_party/pyslam_reference/pyslam/loop_closing/loop_closing.py`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow3.py`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow2.py`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_base.py`
- `third_party/pyslam_reference/pyslam/loop_closing/keyframe_database.py`
- `third_party/pyslam_reference/pyslam/slam/geometry_matchers.py`
- `third_party/pyslam_reference/pyslam/slam/optimizer_g2o.py`
- `third_party/pyslam_reference/pyslam/slam/relocalizer.py`
- `third_party/pyslam_reference/pyslam/slam/frame.py`
- `third_party/pyslam_reference/pyslam/slam/keyframe.py`

## Current files inspected

- `visual_slam/orbslam/slam/loop_detector.py`
- `visual_slam/orbslam/slam/loop_closing.py`
- `visual_slam/orbslam/slam/sim3_solver.py`
- `visual_slam/orbslam/slam/essential_graph.py`
- `visual_slam/orbslam/slam/bow_matcher.py`
- `visual_slam/orbslam/slam/keyframe_database.py`
- `visual_slam/orbslam/slam/optimizer_g2o.py`
- `visual_slam/orbslam/slam/geometry_matchers.py`
- `visual_slam/orbslam/run_tum_rgbd_smoke.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_21_loop_closing.py`

## Loop architecture implemented

- `LoopClosing` owns a keyframe queue and processes one keyframe per `step()`.
- `LoopDetector` computes/query BoW candidates through `KeyFrameDatabase`.
- `LoopGroupConsistencyChecker` follows pySLAM's covisibility-group consistency accumulation.
- `LoopGeometryChecker` verifies candidates with BoW-guided descriptor matches and 3D-3D RGB-D transform estimation.
- `LoopCorrector` applies validated pose/map-point correction, fuses duplicate map points, adds loop edges, and updates connections.
- `run_tum_rgbd_smoke.py` supports `--enable-loop-closing` and `--disable-loop-closing`.

## DBoW and BoW-guided matching integration

- New keyframes are added to `KeyFrameDatabase` during tracking.
- Loop detection rejects unavailable BoW clearly through diagnostics.
- Loop geometry verification uses `BoWGuidedMatcher` when the database is available.
- Missing vocabulary/DBoW disables loop detection without breaking imports or normal tracking.

## Consistency check implementation

The checker expands each candidate into its covisibility group plus itself. A candidate is accepted only after repeated overlap with prior groups, matching the pySLAM control flow. Tests cover repeated-group accumulation and the threshold-zero synthetic acceptance path.

## Geometry verification implementation

The local path uses RGB-D map-point correspondences:

1. BoW-guided keyframe descriptor matching.
2. ORB orientation histogram filtering.
3. Matched map-point extraction.
4. Scale-fixed Sim3/SE3 estimation with Kabsch and inlier filtering.
5. Success only when enough finite 3D inliers remain.

## Sim3 availability check

Local `g2o` exposes Sim3 symbols:

```text
Sim3: Sim3, EdgeSim3, EdgeSim3ProjectXYZ, EdgeInverseSim3ProjectXYZ, VertexSim3Expmap, BlockSolverSim3, LinearSolver*Sim3
SE3: SE3Quat, EdgeSE3*, VertexSE3*
```

The pySLAM `sim3solver` RANSAC wrapper is not available locally. For RGB-D, metric scale is known, so the implemented correction is a documented scale-fixed SE3 fallback. This does not claim monocular Sim3 parity.

## Essential graph / pose graph implementation

- `essential_graph.py` validates finite correction matrices before writing poses.
- It applies the RGB-D correction to the current covisibility group and associated map points.
- It records before/after camera-center error and corrected keyframe count.
- This is a conservative SE3 pose-graph correction, not pySLAM's full Sim3 essential graph optimizer.

## Map-point fusion/replacement behavior

- Matched current points are replaced with loop-side map points when duplicated.
- Missing current observations are added to loop-side points.
- Bad loop points are skipped.
- Tests cover replacement/fusion and bad-point no-corruption behavior.

## Unit tests

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_21_loop_closing.py`
- Result: `11 passed`
- Final full ORB-SLAM suite:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/visual_slam/orbslam`
  - Result: `134 passed, 1 skipped`

## Synthetic pose-graph result

The synthetic loop test builds a current keyframe with a +1 m map drift and a corresponding `Tcw` drift. The scale-fixed estimate returns translation approximately `[-1, 0, 0]`, loop correction reduces camera-center error, updates at least one keyframe, and fuses duplicate map points.

## TUM no-regression result

- Command:
  - `.venv/bin/python tools/validate_orbslam_pyslam_port.py --dataset "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_21_validation"`
- Result: `VALIDATION PASSED`
- Smoke 30 summary: `tracking_ok_count=30`, `tracking_lost_count=0`, `errors=0`, `final_state=OK`, `final_keyframes=6`, `final_map_points=3473`.

## Backend durability result

- Command:
  - `.venv/bin/python tools/run_orb_backend_durability.py --dataset "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_21_backend_durability" --frame-counts 100 300 full --backends opencv_orb pyslam_orb2`
- Result summary:
  - `opencv_orb 100`: `100/100 OK`, `0 lost`, eval `ok`, ATE SE3 `0.025173217`.
  - `opencv_orb 300`: `300/300 OK`, `0 lost`, eval `ok`, ATE SE3 `0.026261712`.
  - `opencv_orb full`: `596/596 OK`, `0 lost`, eval `ok`, ATE SE3 `0.048299007`.
  - `pyslam_orb2 100`: `100/100 OK`, `0 lost`, eval `ok`, ATE SE3 `0.021865343`.
  - `pyslam_orb2 300`: `300/300 OK`, `0 lost`, eval `ok`, ATE SE3 `0.032148861`.
  - `pyslam_orb2 full`: `596/596 OK`, `0 lost`, eval `ok`, ATE SE3 `0.029251704`.

## Loop-enabled smoke result

- Command:
  - `.venv/bin/python -m visual_slam.orbslam.run_tum_rgbd_smoke "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_21_loop_enabled_fr1_desk" --max-frames 30 --feature-backend opencv_orb --enable-loop-closing --print-every 5`
- Result: `30/30 OK`, `0 lost`, `errors=0`, `final_state=OK`, `final_keyframes=6`, `final_map_points=3473`.

## Real loop dataset result

Local loop-capable datasets found:

- `rgbd_dataset_freiburg1_room`
- `rgbd_dataset_freiburg2_desk`

Validation run:

- Command:
  - `.venv/bin/python -m visual_slam.orbslam.run_tum_rgbd_smoke "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room" --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_21_loop_dataset_fr1_room" --max-frames 60 --feature-backend opencv_orb --enable-loop-closing --print-every 10`
- Result: `60/60 OK`, `0 lost`, `errors=0`, `final_state=OK`, `final_keyframes=5`, `final_map_points=2749`.

This is a bounded real-dataset loop-enabled smoke, not a full-sequence loop-closure benchmark.

## Log inspection

Validation logs were checked for tracebacks, RuntimeWarnings, NaNs, overflows, and repeated `0 vertices to optimize`. None were found. Generated logs from runs before the final smoke-format patch contain first-frame `ba_mse=inf` as a sentinel before pose optimization; source now writes that as `NA`/blank for future runs.

## Deviations from pySLAM

- The full pySLAM multiprocessing loop detector is not ported; this implementation is in-process.
- The pySLAM `sim3solver` wrapper is unavailable, so RGB-D correction uses a scale-fixed SE3 estimate.
- The full Sim3 essential graph optimizer is not ported; `essential_graph.py` applies finite SE3 correction conservatively.
- Projection search expansion/fusion is narrower than pySLAM's full `search_more_map_points_by_projection` plus `search_and_fuse_for_loop_correction`.
- Global BA after loop closure is deferred.

## Remaining gaps

- Full monocular Sim3 parity.
- Full essential graph optimizer parity.
- Full loop-correction projection search/fusion parity.
- Global BA after loop closure.
- Full-sequence real loop-closure benchmark on `fr1_room` or `fr2_desk`.

## Files changed

- `visual_slam/orbslam/slam/loop_detector.py`
- `visual_slam/orbslam/slam/loop_closing.py`
- `visual_slam/orbslam/slam/sim3_solver.py`
- `visual_slam/orbslam/slam/essential_graph.py`
- `visual_slam/orbslam/slam/bow_matcher.py`
- `visual_slam/orbslam/slam/slam.py`
- `visual_slam/orbslam/slam/tracking.py`
- `visual_slam/orbslam/run_tum_rgbd_smoke.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_21_loop_closing.py`
- `visual_slam/reference_audit/checkpoint_2_21/LOOP_CLOSING_AUDIT.md`
