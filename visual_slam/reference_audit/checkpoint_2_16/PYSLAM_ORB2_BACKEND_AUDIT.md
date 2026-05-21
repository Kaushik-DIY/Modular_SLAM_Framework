# Checkpoint 2.16 - Optional pySLAM ORB2 Backend Audit

## pySLAM files inspected

- `third_party/pyslam_reference/pyslam/local_features/feature_orbslam2.py`
- `third_party/pyslam_reference/pyslam/local_features/feature_manager.py`
- `third_party/pyslam_reference/thirdparty/orbslam2_features/ORBextractor.h`
- `third_party/pyslam_reference/thirdparty/orbslam2_features/orb_extractor.cpp`
- `third_party/pyslam_reference/thirdparty/orbslam2_features/CMakeLists.txt`

Searches confirmed pySLAM imports `ORBextractor` and
`ORBextractorDeterministic` from the external `orbslam2_features` module.

## Availability result

Checked inside the project venv with:

```bash
.venv/bin/python - <<'PY'
try:
    import orbslam2_features
    print("[OK] orbslam2_features importable:", orbslam2_features)
except Exception as exc:
    print("[MISSING] orbslam2_features:", type(exc).__name__, exc)
PY
```

Result:

```text
[MISSING] orbslam2_features: ModuleNotFoundError No module named 'orbslam2_features'
```

## What was implemented

- `PySLAMORB2Backend` was added as an optional backend.
- `PySLAMORB2Backend.is_available()` dynamically checks whether
  `orbslam2_features` can be imported.
- Explicit `pyslam_orb2` selection raises `BackendUnavailableError` with a
  clear local-venv build/install message when unavailable.
- The default backend remains `opencv_orb`.
- Importing `visual_slam.orbslam.local_features` and `visual_slam.orbslam.slam`
  does not require `orbslam2_features`.

## Adaptations and deviations

- No C++ build was attempted. The pySLAM C++ source exists under
  `third_party/pyslam_reference/thirdparty/orbslam2_features/`, but the
  workflow requires stopping before heavy build commands unless the user
  approves.
- Missing pySLAM ORB2 does not silently fall back when explicitly requested.
  Fallback is only available through the backend factory's `auto` mode.
- Keypoint tuple conversion follows pySLAM's `cv2.KeyPoint(*kp)` structure.

## Validation results

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_16_pyslam_orb2_optional.py`
  - `4 passed, 1 skipped`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/visual_slam/orbslam`
  - `90 passed, 1 skipped`
- `.venv/bin/python tools/validate_orbslam_pyslam_port.py --dataset "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_16_validation" --skip-smoke30`
  - validation passed
  - TUM 3-frame smoke: `3/3 OK`, lost `0`
  - TUM 10-frame smoke: `10/10 OK`, lost `0`

## Build/install status

- C++ build attempted: no
- Global install attempted: no
- Local install attempted: no

