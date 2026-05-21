# Checkpoint 2.14: pySLAM Local BA Port Audit

## pySLAM references inspected

- `third_party/pyslam_reference/pyslam/slam/optimizer_g2o.py`
  - `bundle_adjustment`
  - `local_bundle_adjustment`
  - `pose_optimization`
- `third_party/pyslam_reference/pyslam/slam/map.py`
  - `Map.locally_optimize`
  - `LocalMapBase.update_from_keyframes`
  - `LocalCovisibilityMap.update`
- `third_party/pyslam_reference/pyslam/slam/local_mapping.py`
  - `LocalMapping.local_BA`
- `third_party/pyslam_reference/pyslam/slam/local_mapping_core.py`
  - `LocalMappingCore.local_BA`
- `third_party/pyslam_reference/pyslam/slam/map_point.py`
  - `MapPoint.remove_observation`
  - `MapPoint.update_position`
  - `MapPoint.update_normal_and_depth`
- `third_party/pyslam_reference/pyslam/slam/keyframe.py`
  - covisibility accessors and point-match helpers
- `third_party/pyslam_reference/pyslam/slam/frame.py`
  - pose/update and point-match helpers

## Current files modified

- `visual_slam/orbslam/slam/optimizer_g2o.py`
- `visual_slam/orbslam/slam/map.py`
- `visual_slam/orbslam/slam/map_point.py`
- `visual_slam/orbslam/slam/__init__.py`
- `tools/validate_orbslam_pyslam_port.py`

## Structural changes

- Reworked `_bundle_adjustment_core()` into the pySLAM local BA sequence:
  1. filter bad keyframes and map points,
  2. build stable even keyframe vertices from `kf.kid * 2`,
  3. build stable odd map-point vertices from `point.id * 2 + 1`,
  4. add mono/stereo reprojection edges with octave-dependent information,
  5. use robust kernels for the initial pass,
  6. run initial optimization,
  7. classify bad edges by chi-square and positive-depth checks,
  8. deactivate bad edges with `edge.set_level(1)`,
  9. remove robust kernels,
  10. run final optimization,
  11. collect final outlier observations,
  12. remove bad observations via `point.remove_observation(kf, idx, map_no_lock=True)`,
  13. only then write back local keyframe poses and map-point positions,
  14. update map-point normal/depth.
- Updated `LocalCovisibilityMap.update()` to return pySLAM-style `(local_keyframes, local_points, ref_keyframes)`.
- Updated `Map.locally_optimize()` to call `local_bundle_adjustment(keyframes, points, ref_keyframes, ...)` instead of rediscovering the local window inside the optimizer.
- Changed map-point position storage to copy incoming arrays so optimized `g2o`/Eigen memory is not retained after optimizer teardown.
- Changed `Slam` and `SlamMode` package exports to lazy imports to avoid the validation-time import cycle.
- Fixed `tools/validate_orbslam_pyslam_port.py` so it inserts the repo root on `sys.path` when run as a script from `tools/`.

## Deviations from pySLAM and reasons

- The installed `g2o` binding exposes `EdgeProjectXYZ2UV` / `EdgeProjectXYZ2UVU` with `g2o.CameraParameters` rather than pySLAM's direct `edge.fx`, `edge.fy`, `edge.cx`, `edge.cy`, and `edge.bf` fields.
  - Reason: binding compatibility. The port continues to use `visual_slam.g2o_compat`.
- The installed projection edges do not expose `edge.is_depth_positive()`.
  - Reason: binding compatibility. The port computes positive depth manually from the optimized pose vertex and point vertex.
- Edge outlier classification uses a manual ORB-SLAM camera-model reprojection chi-square.
  - Reason: binding compatibility. The parameterized `g2o.CameraParameters` edge uses a single focal length and does not exactly match the RGB-D camera model used by the rest of this port. Manual classification preserves pySLAM's intended chi-square/depth gating with local `fx/fy/cx/cy/bf`.
- `optimizer.set_force_stop_flag()` is only used when the abort flag is an actual `g2o` object.
  - Reason: this workspace's `g2o` module does not expose `g2o.Flag`; local mapping uses a small Python fallback flag for abort bookkeeping.
- `local_bundle_adjustment()` still supports the older test-facing call shape `local_bundle_adjustment(reference_keyframe, ...)`.
  - Reason: compatibility with the existing unit-test interface while supporting the pySLAM-style `(keyframes, points, keyframes_ref, ...)` path.
- `MapPoint.set_position()` copies the incoming array.
  - Reason: this matches pySLAM's `np.array(v_p.estimate())` writeback behavior and prevents retaining invalid borrowed memory from the Python `g2o` binding.

## Validation commands run

```bash
source .venv/bin/activate
python -c "import sys; print(sys.executable)"
```

Result: `/home/kaushik/slam_ws/.venv/bin/python`

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_8_optimizer_g2o.py
```

Result: `4 passed in 0.48s`

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam
```

Result: `75 passed in 3.28s` inside the final validation harness.

```bash
python tools/validate_orbslam_pyslam_port.py \
    --dataset "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" \
    --output "$HOME/slam_ws/visual_slam_outputs/codex_validation"
```

Result: `VALIDATION PASSED`

## Validation summary

- Local BA consistency:
  - Before local mapping / `KF1`: `chi2 p90=6.718`, `z_min=0.784531`, `z_max=1.834`.
  - After local mapping / `KF1`: `chi2 p90=2.178`, `z_min=0.570648`, `z_max=1.834`.
  - No non-finite geometry reported.
- TUM RGB-D 3-frame smoke:
  - `tracking_ok_count=3`, `tracking_lost_count=0`, `final_state=OK`.
- TUM RGB-D 10-frame smoke:
  - `tracking_ok_count=10`, `tracking_lost_count=0`, `final_state=OK`.
- TUM RGB-D 30-frame smoke:
  - `tracking_ok_count=30`, `tracking_lost_count=0`, `final_state=OK`.
- Logs:
  - Final validation reported no traceback, NaN, overflow warning, or repeated `0 vertices to optimize` failures.

