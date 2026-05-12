# Checkpoint 2.23 RGB-D SE3 Essential Graph Audit

## pySLAM essential graph / optimizer files inspected

- `third_party/pyslam_reference/pyslam/loop_closing/loop_closing.py`
- `third_party/pyslam_reference/pyslam/slam/optimizer_g2o.py`
- `third_party/pyslam_reference/pyslam/slam/optimizer_gtsam.py`
- `third_party/pyslam_reference/pyslam/slam/keyframe.py`
- `third_party/pyslam_reference/pyslam/slam/map_point.py`
- `third_party/pyslam_reference/pyslam/slam/sim3_pose.py`

## Current files inspected

- `visual_slam/orbslam/slam/essential_graph.py`
- `visual_slam/orbslam/slam/loop_closing.py`
- `visual_slam/orbslam/slam/optimizer_g2o.py`
- `visual_slam/orbslam/slam/keyframe.py`
- `visual_slam/orbslam/slam/map_point.py`
- `visual_slam/orbslam/slam/map.py`

## Sim3/SE3 capability check

The local `g2o` binding exposes Sim3 and SE3 symbols:

- `Sim3`, `VertexSim3Expmap`, `EdgeSim3`, `BlockSolverSim3`, `LinearSolverEigenSim3`
- `SE3Quat`, `VertexSE3Expmap`, `EdgeSE3Expmap`, `EdgeSE3`

## RGB-D SE3 correction decision

RGB-D SE3 fallback used. Metric scale is observable from depth. Monocular Sim3 loop correction parity is not claimed.

Even though the binding exposes Sim3 classes, the project scope for this stage is RGB-D only and the local pySLAM `sim3solver` path is not fully ported. The implementation therefore uses fixed-scale SE3 pose graph optimization.

## Graph vertices and edges implemented

- Added `EssentialGraph` in `visual_slam/orbslam/slam/essential_graph.py`.
- Vertices include map keyframes plus the loop/current keyframes and connected graph neighbors.
- Root/origin keyframe (`kid == 0`) is fixed for gauge freedom when available; otherwise the loop keyframe is fixed.
- Edges are finite SE3 relative-pose constraints.

## Spanning tree edges

- `EssentialGraph.add_spanning_tree_edges()` adds parent-child edges using keyframe graph parent links.
- Measurements are computed from pre-correction SE3 poses, matching the pySLAM essential graph structure conceptually.

## Covisibility edges

- `EssentialGraph.add_covisibility_edges()` adds strong covisibility edges above the configured minimum weight.
- Parent/child and already inserted loop pairs are skipped to avoid duplicate constraints.

## Loop edges

- Loop edges are added from:
  - newly created loop connections after fusion,
  - persistent keyframe loop-edge sets.
- Loop-connection measurements use corrected poses where available, mirroring pySLAM's use of corrected Sim3 loop constraints.

## Optimization safety checks

- g2o SE3 optimizer must return positive iterations.
- No write-back occurs if graph construction has no vertices/edges, g2o raises, optimized poses are non-finite, rotations are not orthonormal enough, determinants drift from 1, or translations jump unreasonably.
- On failure, old keyframe poses and map-point positions remain unchanged.

## Map point correction/write-back

- Write-back is performed only after optimizer success and pose validation.
- Corrected map points are transformed through their reference corrected keyframe:
  - old world point -> old keyframe camera frame -> optimized keyframe world frame.
- Updated points recompute normal/depth and record loop-correction provenance.
- Keyframe covisibility is refreshed after successful write-back.

## Tests and results

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_23_essential_graph.py`
  - Result: `10 passed in 2.15s`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam`
  - Result: `154 passed, 1 skipped in 5.83s`.

## TUM validation results

- `python tools/validate_orbslam_pyslam_port.py --dataset "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_23_validation"`
  - Result: `VALIDATION PASSED`.
- `fr1_desk`, loop enabled, 100 frames:
  - Result: 100/100 tracking OK, 0 lost, 0 errors, final state OK.
- `fr1_room`, loop enabled, 100 frames:
  - Result: 100/100 tracking OK, 0 lost, 0 errors, final state OK.

## Deviations from pySLAM

- pySLAM uses Sim3 vertices and `EdgeSim3` for essential graph optimization. This checkpoint uses `VertexSE3Expmap` and `EdgeSE3Expmap` with fixed metric scale.
- pySLAM can trigger Global BA after loop pose graph optimization. This stage intentionally does not start Global BA.
- The implementation is synchronous and Python-level; pySLAM has optional C++ and GTSAM acceleration paths.

## Remaining gaps

- Monocular Sim3 parity remains unimplemented.
- More robust loop-edge information matrices can be tuned in a later checkpoint.
- Full loop-triggered Global BA remains outside this stage.

## Global BA

Global BA is deferred to Checkpoint 2.24.
