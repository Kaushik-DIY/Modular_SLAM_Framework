# Checkpoint 2.18A - ORBSLAM2 Features Build and Integration Audit

## 1. Goal

Safely build and validate pySLAM's `orbslam2_features` C++ ORB-SLAM2 extractor module for use as the optional `pyslam_orb2` backend, without installing or building anything globally and without changing the default extractor backend without approval.

## 2. Initial Environment

- Python executable: `/home/kaushik/slam_ws/.venv/bin/python`
- venv path: `/home/kaushik/slam_ws/.venv`
- `orbslam2_features` before build: not importable (`ModuleNotFoundError: No module named 'orbslam2_features'`)
- Build helpers already present in `.venv` before work:
  - `pybind11==3.0.4`
  - `wheel==0.47.0`
  - `setuptools==82.0.1`
  - `cmake`: not installed in `.venv`
  - `ninja`: not installed in `.venv`
- System build tools used but not installed by this checkpoint:
  - `cmake 3.22.1`
  - GNU C/C++ 11.4.0
- CMake-discovered C++ dependencies:
  - OpenCV C++ headers/libs from `/usr`, version `4.5.4`
  - Python interpreter `/home/kaushik/slam_ws/.venv/bin/python`
  - Python link library `/usr/lib/x86_64-linux-gnu/libpython3.11.so`

Initial `git status --short` before work:

```text
 M visual_slam/orbslam/local_features/__init__.py
 M visual_slam/orbslam/local_features/feature_manager.py
 M visual_slam/orbslam/local_features/feature_orbslam2.py
 M visual_slam/orbslam/local_features/feature_tracker.py
 M visual_slam/orbslam/local_features/feature_tracker_configs.py
 M visual_slam/orbslam/slam/__init__.py
 M visual_slam/orbslam/slam/frame.py
 M visual_slam/orbslam/slam/geometry_matchers.py
 M visual_slam/orbslam/slam/keyframe.py
 M visual_slam/orbslam/slam/map.py
 M visual_slam/orbslam/slam/map_point.py
 M visual_slam/orbslam/slam/optimizer_g2o.py
 M visual_slam/orbslam/slam/tracking.py
?? .venv_backup_before_orbslam2_features/
?? AGENTS.md
?? tests/visual_slam/orbslam/test_checkpoint_2_14_tum_smoke_runner.py
?? tests/visual_slam/orbslam/test_checkpoint_2_15_feature_backend_contract.py
?? tests/visual_slam/orbslam/test_checkpoint_2_16_pyslam_orb2_optional.py
?? tests/visual_slam/orbslam/test_checkpoint_2_17_extractor_comparison.py
?? tools/
?? visual_slam.zip
?? visual_slam/orbslam/io/
?? visual_slam/orbslam/local_features/extractor_backends.py
?? visual_slam/orbslam/run_tum_rgbd_smoke.py
?? visual_slam/reference_audit/checkpoint_2_14/
?? visual_slam/reference_audit/checkpoint_2_15/
?? visual_slam/reference_audit/checkpoint_2_16/
?? visual_slam/reference_audit/checkpoint_2_17/
?? visual_slam/reference_audit/checkpoint_2_18A/
```

## 3. pySLAM Source Paths Inspected

- `third_party/pyslam_reference/pyslam/local_features/feature_orbslam2.py`
- `third_party/pyslam_reference/pyslam/local_features/feature_manager.py`
- `third_party/pyslam_reference/pyslam/local_features/feature_tracker.py`
- `third_party/pyslam_reference/thirdparty/orbslam2_features/CMakeLists.txt`
- `third_party/pyslam_reference/thirdparty/orbslam2_features/build.sh`
- `third_party/pyslam_reference/thirdparty/orbslam2_features/orb_extractor.cpp`
- `third_party/pyslam_reference/thirdparty/orbslam2_features/test_orb_determinism.py`
- `third_party/pyslam_reference/scripts/install_thirdparty.sh`
- `third_party/pyslam_reference/cpp/casters/opencv_type_casters.h`

## 4. Discovered Build System

`third_party/pyslam_reference/thirdparty/orbslam2_features` uses CMake:

- `CMakeLists.txt` calls `find_package(OpenCV 4 REQUIRED)`.
- `CMakeLists.txt` calls `add_subdirectory(pybind11)`.
- `pybind11_add_module(orbslam2_features orb_extractor.cpp ORBextractor.cpp)` builds the Python extension.
- The pySLAM `build.sh` creates an in-source `build/` directory and builds with `make -j 4`.
- The CMake file writes the extension to `${PROJECT_SOURCE_DIR}/lib`.

Because pySLAM's default build writes generated files into the source tree, this checkpoint used a copied source tree under `third_party/build/orbslam2_features/source`.

## 5. Packages Installed Into `.venv`

No Python packages were installed during this checkpoint.

Already-present `.venv` packages used:

| Package | Version | Reason |
|---|---:|---|
| `pybind11` | `3.0.4` | Python build helper already installed; pySLAM also used vendored pybind11 for CMake. |
| `wheel` | `0.47.0` | Existing build helper, not modified. |
| `setuptools` | `82.0.1` | Existing build helper, not modified. |

## 6. Build Commands Used

```bash
cd /home/kaushik/slam_ws
.venv/bin/python -c "import sys; print(sys.executable)"
find third_party/pyslam_reference -iname "*orbslam2*" -o -iname "*features*" | head -100
find third_party/pyslam_reference -iname "CMakeLists.txt" | grep -i "orb\|feature\|third" || true
grep -R "orbslam2_features" -n third_party/pyslam_reference | head -100
grep -R "ORBextractor" -n third_party/pyslam_reference | head -100
```

```bash
mkdir -p third_party/build/orbslam2_features
cp -a third_party/pyslam_reference/thirdparty/orbslam2_features third_party/build/orbslam2_features/source
ln -s ../../pyslam_reference/thirdparty/pybind11 third_party/build/orbslam2_features/pybind11
```

```bash
cmake -S third_party/build/orbslam2_features/source \
  -B third_party/build/orbslam2_features/cmake-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=/home/kaushik/slam_ws/.venv/bin/python \
  -DPython_EXECUTABLE=/home/kaushik/slam_ws/.venv/bin/python \
  -DPYTHON_EXECUTABLE=/home/kaushik/slam_ws/.venv/bin/python \
  -DCMAKE_INSTALL_PREFIX=/home/kaushik/slam_ws/third_party/local/orbslam2_features
```

First build attempt:

```bash
cmake --build third_party/build/orbslam2_features/cmake-build --config Release -j"$(nproc)"
```

Local fix for copied-source broken symlink:

```bash
rm third_party/build/orbslam2_features/source/opencv_type_casters.h
cp third_party/pyslam_reference/cpp/casters/opencv_type_casters.h \
  third_party/build/orbslam2_features/source/opencv_type_casters.h
```

Successful rebuild:

```bash
cmake --build third_party/build/orbslam2_features/cmake-build --config Release -j"$(nproc)"
cmake --install third_party/build/orbslam2_features/cmake-build
```

Manual local install because the CMake project has no install rules:

```bash
mkdir -p third_party/local/orbslam2_features
cp third_party/build/orbslam2_features/source/lib/orbslam2_features.cpython-311-x86_64-linux-gnu.so \
  third_party/local/orbslam2_features/
```

Durable import path:

```text
.venv/lib/python3.11/site-packages/orbslam2_features_local.pth
```

with contents:

```text
/home/kaushik/slam_ws/third_party/local/orbslam2_features
```

## 7. Install/Import Path

- Build output: `/home/kaushik/slam_ws/third_party/build/orbslam2_features/source/lib/orbslam2_features.cpython-311-x86_64-linux-gnu.so`
- Local import/install copy: `/home/kaushik/slam_ws/third_party/local/orbslam2_features/orbslam2_features.cpython-311-x86_64-linux-gnu.so`
- Durable venv path hook: `/home/kaushik/slam_ws/.venv/lib/python3.11/site-packages/orbslam2_features_local.pth`

## 8. Build Errors Encountered

First build failed while compiling `orb_extractor.cpp`:

```text
/home/kaushik/slam_ws/third_party/build/orbslam2_features/source/orb_extractor.cpp:28:10: fatal error: opencv_type_casters.h: No such file or directory
   28 | #include "opencv_type_casters.h"
      |          ^~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
```

Classification: source path mismatch caused by a broken relative symlink after copying the pySLAM source to the local build tree.

## 9. Debugging Steps and Fixes

- Inspected the copied header path.
- Found `third_party/build/orbslam2_features/source/opencv_type_casters.h` was a copied symlink to `../../cpp/casters/opencv_type_casters.h`.
- In the copied build tree, that relative symlink resolved outside the copied source and did not point to a real file.
- Replaced only the copied build-tree symlink with a real copy of `third_party/pyslam_reference/cpp/casters/opencv_type_casters.h`.
- Re-ran `cmake --build`; the extension built successfully.

No system packages were installed. No files under `/usr`, `/usr/local`, `/opt`, `/root`, or `/home/kaushik` outside the repo were modified.

## 10. Final Import Verification

```text
python: /home/kaushik/slam_ws/.venv/bin/python
orbslam2_features: <module 'orbslam2_features' from '/home/kaushik/slam_ws/third_party/local/orbslam2_features/orbslam2_features.cpython-311-x86_64-linux-gnu.so'>
file: /home/kaushik/slam_ws/third_party/local/orbslam2_features/orbslam2_features.cpython-311-x86_64-linux-gnu.so
symbols: ['ORBextractor', 'ORBextractorDeterministic']
```

## 11. Backend Integration Changes

- `visual_slam/orbslam/run_tum_rgbd_smoke.py` now accepts:

```text
--feature-backend {opencv_orb,pyslam_orb2,auto}
```

- If omitted, the smoke runner leaves the existing default backend path unchanged.
- If provided, the runner passes `feature_tracker_config={"extractor_backend": <backend>}` into `Slam`.
- Explicit `pyslam_orb2` now exercises the optional C++ backend through the normal SLAM path.

No downstream tracking, matching, or optimizer thresholds were changed for pySLAM ORB2.

## 12. Tests Run

Baseline before build:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/visual_slam/orbslam
90 passed, 1 skipped in 2.86s
```

Baseline shortened validation:

```text
.venv/bin/python tools/validate_orbslam_pyslam_port.py \
  --dataset "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_18A_baseline" \
  --skip-smoke30
VALIDATION PASSED
```

Optional pySLAM ORB2 backend tests after import:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/visual_slam/orbslam/test_checkpoint_2_16_pyslam_orb2_optional.py
4 passed, 1 skipped in 0.39s
```

Post-build full ORB-SLAM tests:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/visual_slam/orbslam
90 passed, 1 skipped in 3.20s
```

Full default validation:

```text
.venv/bin/python tools/validate_orbslam_pyslam_port.py \
  --dataset "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_18A_full_default_validation"
VALIDATION PASSED
```

## 13. Extractor Comparison Result

Command:

```bash
.venv/bin/python tools/compare_orb_extractors.py \
  --dataset "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_18A_orb2_comparison" \
  --max-frames 30
```

Summary:

| Backend | Available | Avg features | Descriptor | Grid coverage | Match count | Avg FPS | Smoke 3/10/30 |
|---|---:|---:|---|---:|---:|---:|---|
| `opencv_orb` | yes | 1864.2 | `uint8 1928x32` | 0.898 | 827.1 | 73.83 | 3/3, 10/10, 30/30 OK |
| `pyslam_orb2` | yes | 1999.5 | `uint8 2000x32` | 1.000 | 722.5 | 47.26 | 3/3, 10/10, 30/30 OK |

Generated files:

- `visual_slam_outputs/checkpoint_2_18A_orb2_comparison/extractor_comparison_summary.md`
- `visual_slam_outputs/checkpoint_2_18A_orb2_comparison/extractor_frame_metrics.csv`

## 14. TUM Smoke Result

Baseline/default OpenCV backend:

- 3 frames: 3/3 OK, 0 lost, final state OK, 2 keyframes, 2234 map points.
- 10 frames: 10/10 OK, 0 lost, final state OK, 3 keyframes, 2484 map points.
- 30 frames: 30/30 OK, 0 lost, final state OK, 6 keyframes, 3473 map points.

Explicit OpenCV CLI smoke:

- Command used `--feature-backend opencv_orb`.
- 3 frames: 3/3 OK, 0 lost, final state OK, 2 keyframes, 2234 map points.

Explicit pySLAM ORB2 CLI smoke:

- Command used `--feature-backend pyslam_orb2`.
- 30 frames: 30/30 OK, 0 lost, final state OK, 5 keyframes, 3247 map points, average FPS 0.16.

Output directory:

- `visual_slam_outputs/checkpoint_2_18A_pyslam_orb2_smoke_30`

## 15. Default Backend Decision

Keep `opencv_orb` as the default backend.

Rationale:

- pySLAM ORB2 is now available and passed the smoke gates.
- OpenCV ORB remains faster in extractor-only metrics (`73.83` avg FPS vs `47.26` avg FPS).
- OpenCV ORB had higher average descriptor match count (`827.1` vs `722.5`).
- pySLAM ORB2 had stronger grid coverage and feature count, but the checkpoint requires user approval before switching defaults.

## 16. Remaining Risks

- The built extension links against system OpenCV C++ `4.5.4`, while the Python venv has `opencv-contrib-python==4.10.0.84`. The backend passed contract and smoke validation, but this C++/Python OpenCV version split should remain documented.
- CMake linked against `/usr/lib/x86_64-linux-gnu/libpython3.11.so` while using the venv interpreter. This is normal for the local interpreter ABI on this machine, but it should be rechecked if Python is upgraded.
- `third_party/build/orbslam2_features` is a generated local build tree and should not be committed unless repository policy explicitly wants build artifacts.
- The `.pth` file inside `.venv` is a local environment change and is intentionally not represented in `git diff`.

## 17. Exact Deviations From pySLAM

- Did not run `third_party/pyslam_reference/scripts/install_thirdparty.sh`.
- Did not run pySLAM's `thirdparty/orbslam2_features/build.sh` in-place.
- Copied `thirdparty/orbslam2_features` into `third_party/build/orbslam2_features/source` to keep generated files out of the pySLAM reference source.
- Replaced the copied build-tree `opencv_type_casters.h` symlink with a real copied header because the original relative symlink is only valid in the pySLAM source layout.
- Installed/imported the extension from `third_party/local/orbslam2_features` through a venv `.pth` file instead of pySLAM's local `sys.path.append("./lib/")` test pattern.
- Added a local smoke-runner CLI backend switch; pySLAM's reference wrapper does not provide this exact repo-specific smoke CLI.

## 18. Files Changed

Tracked or untracked repo files/directories affected by this checkpoint:

- `visual_slam/orbslam/run_tum_rgbd_smoke.py`
- `visual_slam/reference_audit/checkpoint_2_18A/ORBSLAM2_FEATURES_BUILD_AND_INTEGRATION_AUDIT.md`
- `third_party/local/orbslam2_features/orbslam2_features.cpython-311-x86_64-linux-gnu.so`

Local environment/build artifacts affected:

- `third_party/build/orbslam2_features/`
- `.venv/lib/python3.11/site-packages/orbslam2_features_local.pth`

Validation artifacts generated:

- `visual_slam_outputs/checkpoint_2_18A_baseline/`
- `visual_slam_outputs/checkpoint_2_18A_orb2_comparison/`
- `visual_slam_outputs/checkpoint_2_18A_full_default_validation/`
- `visual_slam_outputs/checkpoint_2_18A_opencv_orb_cli_smoke_3/`
- `visual_slam_outputs/checkpoint_2_18A_pyslam_orb2_smoke_30/`
