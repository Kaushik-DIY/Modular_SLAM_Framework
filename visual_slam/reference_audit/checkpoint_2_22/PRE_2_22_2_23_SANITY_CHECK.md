# Checkpoints 2.22-2.23 Pre-Implementation Sanity Check

## Current tests run and results

- Python environment: `/home/kaushik/slam_ws/.venv/bin/python`.
- Checkpoint revalidation:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_19_relocalization.py tests/visual_slam/orbslam/test_checkpoint_2_20_bow_keyframe_database.py tests/visual_slam/orbslam/test_checkpoint_2_20_bow_guided_matching.py tests/visual_slam/orbslam/test_checkpoint_2_21_loop_closing.py`
  - Result: `34 passed in 2.08s`.
- Full ORB-SLAM test suite:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam`
  - Result: `134 passed, 1 skipped in 4.59s`.
- Baseline RGB-D validation:
  - `python tools/validate_orbslam_pyslam_port.py --dataset "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_22_baseline_validation"`
  - Result: `VALIDATION PASSED`.
  - Built-in smoke summaries: 3/3, 10/10, and 30/30 frames tracked OK.

## Current DBOW/FeatureVector status

- `pydbow3`: `/home/kaushik/slam_ws/third_party/local/pydbow3/pydbow3.cpython-311-x86_64-linux-gnu.so`.
- Exposed DBOW3 symbols include `BowVector`, `FeatureVector`, and `Vocabulary`.
- `orbslam2_features`: `/home/kaushik/slam_ws/third_party/local/orbslam2_features/orbslam2_features.cpython-311-x86_64-linux-gnu.so`.
- Vocabulary present locally: `third_party/vocabs/ORBvoc.dbow3`, 48M.

## Current loop-closing implementation summary

- Local loop closing follows the 2.21 scaffold: DBOW3 candidate retrieval, consistency groups, BoW-guided keyframe matching, RGB-D scale-fixed Sim3/SE3 estimation, then local pose/point correction.
- `LoopCorrector.correct_loop()` currently corrects the current keyframe covisibility group and then fuses only the directly matched current keyframe slots from `success_map_point_matches`.
- Loop edges are added after a successful correction, and current/loop covisibility is refreshed.

## Current loop-fusion limitation

- `ProjectionMatcher.search_and_fuse_for_loop_correction()` is still a `NotImplementedError`.
- Loop correction does not project map points from the loop keyframe plus loop covisible neighbors into the current corrected keyframe group.
- Direct matched-point replacement exists, but it is one-keyframe scoped and does not mirror pySLAM's wider search-and-fuse pass.
- `MapPoint.replace_with()` transfers observations and then calls `set_bad()`, which can clear keyframe slots that were just transferred. This must be fixed before wider fusion can be safe.

## Current essential graph limitation

- `essential_graph.py` applies a single RGB-D SE3 correction to the current covisibility group.
- It does not yet build an explicit essential graph with vertices, spanning tree edges, strong covisibility edges, and loop edges.
- Pose write-back is conservative for finite matrices, but it is not yet graph-optimized or edge-diagnostic.

## pySLAM files inspected

- `third_party/pyslam_reference/pyslam/loop_closing/loop_closing.py`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow3.py`
- `third_party/pyslam_reference/pyslam/loop_closing/keyframe_database.py`
- `third_party/pyslam_reference/pyslam/slam/geometry_matchers.py`
- `third_party/pyslam_reference/pyslam/slam/optimizer_g2o.py`
- `third_party/pyslam_reference/pyslam/slam/optimizer_gtsam.py`
- `third_party/pyslam_reference/pyslam/slam/keyframe.py`
- `third_party/pyslam_reference/pyslam/slam/map_point.py`
- `third_party/pyslam_reference/pyslam/slam/map.py`
- `third_party/pyslam_reference/pyslam/slam/sim3_pose.py`
- `third_party/pyslam_reference/pyslam/config_parameters.py`

## Local files inspected

- `visual_slam/orbslam/slam/loop_closing.py`
- `visual_slam/orbslam/slam/loop_detector.py`
- `visual_slam/orbslam/slam/essential_graph.py`
- `visual_slam/orbslam/slam/sim3_solver.py`
- `visual_slam/orbslam/slam/optimizer_g2o.py`
- `visual_slam/orbslam/slam/geometry_matchers.py`
- `visual_slam/orbslam/slam/bow_matcher.py`
- `visual_slam/orbslam/slam/keyframe.py`
- `visual_slam/orbslam/slam/map_point.py`
- `visual_slam/orbslam/slam/map.py`
- `visual_slam/orbslam/slam/slam.py`
- `visual_slam/orbslam/run_tum_rgbd_smoke.py`
- Existing checkpoint tests under `tests/visual_slam/orbslam/`.

## Exact plan for 2.22 and 2.23

1. Fix `MapPoint.replace_with()` so replacement transfers observations safely, preserves keyframe slots, marks the replaced point bad/replaced, and removes it from the map without undoing transferred observations.
2. Implement loop-correction projection fusion in `ProjectionMatcher.search_and_fuse_for_loop_correction()` with pySLAM-style projection, scale, descriptor, duplicate, bad-point, and visibility checks.
3. Add `LoopCorrector.search_and_fuse_corrected_keyframes()` to collect loop-side points from the loop keyframe and covisible neighbors, collect current-side corrected keyframes from the current keyframe and covisible neighbors, project/fuse, recompute point info, and update covisibility.
4. Replace the one-keyframe-only fusion path in loop correction with direct matched fusion plus the wider projection/fusion pass.
5. Implement an RGB-D SE3 essential graph builder/optimizer with keyframe vertices, fixed root/origin gauge, spanning tree edges, strong covisibility edges, loop edges, finite relative-pose constraints, safety validation, and atomic pose/map-point write-back.
6. Add required 2.22 and 2.23 tests, audits, and final review update.
7. Run required checkpoint-specific tests, full test suite, TUM validation, loop-enabled smoke, and staged backend durability.

## Expected deviations from pySLAM for RGB-D-only scope

- pySLAM uses Sim3 loop correction and `g2o.EdgeSim3` in `optimizer_g2o.optimize_essential_graph()`.
- This port remains RGB-D only. Metric scale is observable from depth, so the implementation will use scale-fixed SE3 correction and document that monocular Sim3 parity is not claimed.
- The installed `g2o` exposes Sim3 symbols (`Sim3`, `VertexSim3Expmap`, `EdgeSim3`), but the local project policy for this checkpoint is to use the safer RGB-D SE3 path unless a complete tested Sim3 path already exists locally.
- Global BA is not implemented in this stage and is deferred to Checkpoint 2.24.
