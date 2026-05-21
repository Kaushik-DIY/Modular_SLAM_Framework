# Checkpoint 2.15 - Feature Backend Abstraction Audit

## pySLAM files inspected

- `third_party/pyslam_reference/pyslam/local_features/feature_orbslam2.py`
- `third_party/pyslam_reference/pyslam/local_features/feature_manager.py`
- `third_party/pyslam_reference/pyslam/local_features/feature_tracker.py`
- `third_party/pyslam_reference/pyslam/local_features/feature_tracker_configs.py`
- `third_party/pyslam_reference/pyslam/slam/frame.py`
- `third_party/pyslam_reference/pyslam/slam/feature_tracker_shared.py`

## Current files modified

- `visual_slam/orbslam/local_features/extractor_backends.py`
- `visual_slam/orbslam/local_features/feature_orbslam2.py`
- `visual_slam/orbslam/local_features/feature_manager.py`
- `visual_slam/orbslam/local_features/feature_tracker.py`
- `visual_slam/orbslam/local_features/feature_tracker_configs.py`
- `visual_slam/orbslam/local_features/__init__.py`
- `visual_slam/orbslam/slam/frame.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_15_feature_backend_contract.py`

## What was ported

pySLAM keeps detector/descriptor ownership inside `FeatureManager`, exposes
`detectAndCompute()`, and lets `FeatureTracker` and `Frame` consume only the
resulting keypoints and descriptors. This checkpoint preserves that structure
while inserting a backend contract below `FeatureManager`.

The new backend contract is:

- `FeatureExtractionResult`
- `FeatureExtractorBackend`
- `OpenCVORBBackend`
- `create_extractor_backend()`

`FeatureExtractionResult` carries:

- `keypoints: list[cv2.KeyPoint]`
- `descriptors: np.ndarray` with `uint8` ORB descriptors shaped `(N, 32)`
- `octaves`
- `angles`
- `sizes`
- `backend_name`
- `success/message` metadata

## Adaptations and deviations

- OpenCV ORB remains the default backend as `opencv_orb`.
  The existing port already used OpenCV ORB to stand in for the unavailable
  pySLAM C++ ORB2 extractor. Keeping this default preserves current tracking
  behavior and avoids dependency churn.
- `FeatureManager.detectAndCompute()` still returns `(keypoints, descriptors)`
  for pySLAM compatibility. The richer backend result is available through
  `FeatureManager.extract()` and `FeatureTracker.extract()`.
- `Frame` now stores `sizes` alongside `octaves` and `angles` so downstream
  code can depend on stable feature metadata without knowing the backend.

## Validation results

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_15_feature_backend_contract.py`
  - `4 passed`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/visual_slam/orbslam`
  - `90 passed, 1 skipped`
- `.venv/bin/python tools/validate_orbslam_pyslam_port.py --dataset "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_15_validation" --skip-smoke30`
  - validation passed
  - local BA p90 improved from `6.718` before local mapping to `2.178` after local mapping on the keyframe check
  - TUM 3-frame smoke: `3/3 OK`, lost `0`, final keyframes `2`, map points `2234`
  - TUM 10-frame smoke: `10/10 OK`, lost `0`, final keyframes `3`, map points `2484`

