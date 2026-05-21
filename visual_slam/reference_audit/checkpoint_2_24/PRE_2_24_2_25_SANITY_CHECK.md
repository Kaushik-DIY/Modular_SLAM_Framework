# Checkpoint 2.24-2.25 Pre-Implementation Sanity Check

## 1. Current Test Results

- Prior checkpoint tests: `20 passed in 2.53s`
- Full ORB-SLAM unit folder: `154 passed, 1 skipped in 5.23s`

Commands used the project venv:

```bash
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_22_loop_fusion.py tests/visual_slam/orbslam/test_checkpoint_2_23_essential_graph.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam
```

## 2. Current 100-Frame pyslam_orb2 Smoke Result

- Dataset: `datasets/tum/rgbd_dataset_freiburg1_desk`
- Backend: `pyslam_orb2`
- Loop closing: enabled
- Frames attempted: 100
- Tracking OK: 100
- Tracking lost: 0
- Errors: 0
- Final state: OK
- Final keyframes: 7
- Final map points: 3666
- Elapsed: 482.509 seconds

## 3. pySLAM Files Inspected

- `third_party/pyslam_reference/pyslam/loop_closing/loop_closing.py`
- `third_party/pyslam_reference/pyslam/slam/global_bundle_adjustment.py`
- `third_party/pyslam_reference/pyslam/slam/optimizer_g2o.py`
- `third_party/pyslam_reference/pyslam/slam/optimizer_gtsam.py`
- `third_party/pyslam_reference/pyslam/slam/keyframe.py`
- `third_party/pyslam_reference/pyslam/slam/map_point.py`
- `third_party/pyslam_reference/pyslam/slam/map.py`
- `third_party/pyslam_reference/pyslam/slam/slam.py`
- `third_party/pyslam_reference/pyslam/config_parameters.py`

## 4. Current Local Files Inspected

- `visual_slam/orbslam/slam/optimizer_g2o.py`
- `visual_slam/orbslam/slam/loop_closing.py`
- `visual_slam/orbslam/slam/essential_graph.py`
- `visual_slam/orbslam/slam/map.py`
- `visual_slam/orbslam/slam/keyframe.py`
- `visual_slam/orbslam/slam/map_point.py`
- `visual_slam/orbslam/slam/slam.py`
- `visual_slam/orbslam/run_tum_rgbd_smoke.py`

## 5. Existing global_bundle_adjustment Implementation

The local `optimizer_g2o.global_bundle_adjustment()` is a thin wrapper around the shared BA core. It can optimize all keyframes and points, but it does not provide a loop-triggered coordinator, pySLAM-style deferred correction, loop diagnostics, or safe atomic write-back.

## 6. Missing Global BA Integration

Loop correction currently stops after SE3 essential graph optimization, loop-edge insertion, and covisibility refresh. It does not trigger Global BA after accepted loop closure.

## 7. Existing Optimizer Gaps

- Global BA result diagnostics are minimal.
- BA write-back is immediate in the low-level optimizer path.
- Local BA and global BA use the same pruning behavior, while pySLAM prunes outliers in local BA but only disables them during global BA.
- Essential graph information matrices are identity for every edge.
- Loop diagnostics do not include Global BA fields.

## 8. Planned 2.24 Implementation

Add a synchronous RGB-D `GlobalBundleAdjuster` wrapper that collects valid keyframes/map points, runs all-keyframe g2o BA with robust kernels and abort support, validates optimized poses/points, writes back atomically, recomputes point info, refreshes covisibility, and reports loop diagnostics.

## 9. Planned 2.25 Implementation

Refine optimizer parity by standardizing diagnostics, filtering non-finite/negative-depth observations before BA edge creation, keeping local BA outlier cleanup separate from global BA, and weighting SE3 essential graph edges by structural type and covisibility/loop confidence.

## 10. RGB-D-Only Policy and Sim3 Decision

pySLAM uses Sim3 for loop pose graph and monocular scale handling. This checkpoint keeps the existing RGB-D SE3 loop correction path and does not claim monocular Sim3 parity.

## 11. Validation Plan

- Run 2.24 and 2.25 checkpoint tests.
- Run full `tests/visual_slam/orbslam`.
- Run 100-frame fr1_desk `pyslam_orb2` loop+GBA smoke first.
- Run one full fr1_desk `pyslam_orb2` loop+GBA sequence only after 100-frame checks pass.
- Generate trajectory evaluation for the full fr1_desk run.
