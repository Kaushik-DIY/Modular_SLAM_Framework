# Checkpoint 2.20 BoW / KeyFrame Database Audit

## pySLAM files inspected

- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_vocabulary.py`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow3.py`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow2.py`
- `third_party/pyslam_reference/pyslam/loop_closing/keyframe_database.py`
- `third_party/pyslam_reference/pyslam/slam/relocalizer.py`
- `third_party/pyslam_reference/pyslam/slam/frame.py`
- `third_party/pyslam_reference/pyslam/slam/keyframe.py`
- `third_party/pyslam_reference/pyslam/local_features/feature_matcher.py`
- `third_party/pyslam_reference/thirdparty/pydbow3/src/py_dbow3.cpp`
- `third_party/pyslam_reference/thirdparty/pydbow3/modules/dbow3/src/Vocabulary.h`
- `third_party/pyslam_reference/thirdparty/pydbow3/modules/dbow3/src/FeatureVector.h`
- `third_party/pyslam_reference/thirdparty/pydbow2/src/py_dbow2.cpp`

## Current files inspected

- `visual_slam/orbslam/slam/bow.py`
- `visual_slam/orbslam/slam/bow_matcher.py`
- `visual_slam/orbslam/slam/keyframe_database.py`
- `visual_slam/orbslam/slam/relocalizer.py`
- `visual_slam/orbslam/slam/loop_closing.py`
- `visual_slam/orbslam/slam/frame.py`
- `visual_slam/orbslam/slam/keyframe.py`
- `tools/install_pyslam_vocabulary_local.sh`
- `tools/build_pyslam_pydbow3_local.sh`
- `tests/visual_slam/orbslam/test_checkpoint_2_20_bow_keyframe_database.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_20_bow_guided_matching.py`

## Vocabulary and backend status

- Vocabulary source: local pySLAM reference `third_party/pyslam_reference/thirdparty/pydbow3/modules/dbow3/orbvoc.dbow3`.
- Local vocabulary path: `third_party/vocabs/ORBvoc.dbow3`.
- Installed size: `49,417,196` bytes.
- pySLAM source URL recorded by the installer: `https://drive.google.com/uc?id=13xmRtop_ow3aPtv3qCT5beG19_mlogqI`.
- pydbow3 build root: `third_party/build/pydbow3`.
- pydbow3 import path: `third_party/local/pydbow3/pydbow3.cpython-311-x86_64-linux-gnu.so`.
- venv path file: `.venv/lib/python3.11/site-packages/slam_ws_pydbow3_local.pth`.
- Vocabulary loads locally with `use_boost=False`; `use_boost=True` is not used because local loading can fail with allocation errors on this `.dbow3` file.

No system package manager, `sudo`, global Python install, or global path write was used. No Python packages were installed for this checkpoint.

## Implementation summary

- `DBoW3Vocabulary` wraps pySLAM's `pydbow3.Vocabulary`, records explicit backend availability, and computes both `bow_vector/g_des` and `feature_vector/f_des`.
- `tools/build_pyslam_pydbow3_local.sh` now patches only the project-local copied pydbow3 source under `third_party/build/pydbow3/source` to expose DBoW3 `FeatureVector` through `TransformResult.featureVector`.
- `Frame.compute_bow()` and `KeyFrame.compute_bow()` store pySLAM-compatible BoW fields.
- `KeyFrameDatabase` implements add, erase, clear/reset, relocalization candidates, and loop candidates using pySLAM shared-word filtering, BoW score, covisibility score accumulation, connected-keyframe exclusion, recent-keyframe exclusion, and 0.75-best retention.
- `Relocalizer` uses database-backed retrieval when BoW is available.

## BoW-guided descriptor matching parity

Full direct-file matching parity is now implemented for the local DBOW3 backend:

- `pydbow3` exposes `FeatureVector.to_dict()` and `Vocabulary.transform_with_feature_vector(descriptors, levelsup=4)`.
- `visual_slam/orbslam/slam/bow_matcher.py` implements a pySLAM/ORB-SLAM-style BoW matcher:
  - computes/uses direct-file node groups,
  - intersects shared node IDs,
  - compares descriptors only inside shared node groups,
  - applies Hamming distance threshold,
  - applies nearest/second-nearest ratio when a second candidate exists,
  - enforces one-to-one train descriptor assignment,
  - applies ORB orientation histogram filtering when oriented features are active,
  - returns explicit diagnostics.
- `Relocalizer` uses BoW-guided matching when the database is available and records explicit fallback diagnostics when it is unavailable.
- `LoopGeometryChecker` uses BoW-guided matching for loop verification when the database is available.

Remaining deviation: this is DBOW3 direct-file parity, not DBOW2 `KeyFrameOrbDatabase` C++ direct-file storage. The matching structure and feature-vector semantics follow pySLAM/ORB-SLAM, but the in-process database remains Python-side for this checkpoint.

## Validation output summary

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_19_relocalization.py tests/visual_slam/orbslam/test_checkpoint_2_20_bow_keyframe_database.py tests/visual_slam/orbslam/test_checkpoint_2_20_bow_guided_matching.py`
- Result: `23 passed`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/visual_slam/orbslam`
- Result after final edits: `134 passed, 1 skipped`
- `.venv/bin/python tools/validate_orbslam_pyslam_port.py --dataset "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_21_validation"`
- Result: `VALIDATION PASSED`

## Deviations from pySLAM and reasons

- The local pydbow3 source is copied and patched under `third_party/build/pydbow3/source` instead of modifying the pySLAM reference clone or installing globally.
- The local `Slam` path uses an in-process Python `KeyFrameDatabase`; pySLAM's full loop detector process and persistence helpers are outside 2.20 and handled structurally in 2.21.
- Vocabulary loading uses `use_boost=False` for the local DBOW3 file because the boost path is unstable in this environment.

## Remaining gaps

- Persistent database save/load parity is not implemented.
- The loop detector process/multiprocessing architecture is not ported; the local loop detector is in-process.

## Files changed

- `visual_slam/orbslam/slam/bow.py`
- `visual_slam/orbslam/slam/bow_matcher.py`
- `visual_slam/orbslam/slam/keyframe_database.py`
- `visual_slam/orbslam/slam/relocalizer.py`
- `visual_slam/orbslam/slam/loop_closing.py`
- `visual_slam/orbslam/slam/frame.py`
- `visual_slam/orbslam/slam/keyframe.py`
- `visual_slam/orbslam/slam/slam.py`
- `visual_slam/orbslam/slam/tracking.py`
- `visual_slam/orbslam/slam/__init__.py`
- `tools/install_pyslam_vocabulary_local.sh`
- `tools/build_pyslam_pydbow3_local.sh`
- `tests/visual_slam/orbslam/test_checkpoint_2_20_bow_keyframe_database.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_20_bow_guided_matching.py`
