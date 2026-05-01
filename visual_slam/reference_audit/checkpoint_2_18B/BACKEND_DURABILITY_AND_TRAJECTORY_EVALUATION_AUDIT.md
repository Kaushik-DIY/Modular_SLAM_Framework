# Checkpoint 2.18B - Backend Durability and Trajectory Evaluation Audit

## 1. Goal

Add a reproducible local `orbslam2_features` build script, prevent generated artifacts from becoming commit candidates, add TUM trajectory ATE/RPE evaluation, and run backend durability checks for `opencv_orb` and `pyslam_orb2`.

The default backend must remain `opencv_orb` unless durability and trajectory evidence clearly justify switching and the user approves.

## 2. Baseline Environment

- Python executable: `/home/kaushik/slam_ws/.venv/bin/python`
- `orbslam2_features` import path: `/home/kaushik/slam_ws/third_party/local/orbslam2_features/orbslam2_features.cpython-311-x86_64-linux-gnu.so`
- Exported symbols: `ORBextractor`, `ORBextractorDeterministic`
- Python OpenCV version: `4.10.0`
- C++ OpenCV linked libraries checked with `ldd`:
  - `/lib/x86_64-linux-gnu/libopencv_core.so.4.5d`
  - `/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.5d`
  - `/lib/x86_64-linux-gnu/libopencv_features2d.so.4.5d`
  - `/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.5d`
  - plus related OpenCV 4.5d libraries

Baseline verification before edits:

```text
orbslam2_features import: OK
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/visual_slam/orbslam
90 passed, 1 skipped in 3.01s
```

Short baseline validation:

```text
.venv/bin/python tools/validate_orbslam_pyslam_port.py \
  --dataset "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_18B_baseline_validation" \
  --skip-smoke30
VALIDATION PASSED
```

## 3. Files Inspected

- `AGENTS.md`
- `/home/kaushik/Downloads/CODEX_CHECKPOINT_2_18B_BACKEND_DURABILITY_EVALUATION.md`
- `visual_slam/reference_audit/checkpoint_2_18A/ORBSLAM2_FEATURES_BUILD_AND_INTEGRATION_AUDIT.md`
- `visual_slam/orbslam/run_tum_rgbd_smoke.py`
- `tools/compare_orb_extractors.py`
- `tools/validate_orbslam_pyslam_port.py`
- `third_party/pyslam_reference/pyslam/local_features/feature_orbslam2.py`
- `third_party/pyslam_reference/thirdparty/orbslam2_features/CMakeLists.txt`
- `third_party/pyslam_reference/thirdparty/orbslam2_features/build.sh`
- TUM dataset files:
  - `datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt`
  - `datasets/tum/rgbd_dataset_freiburg1_desk/associations.txt`

## 4. Build-Script Implementation

Added:

```text
tools/build_orbslam2_features_local.sh
```

Behavior:

- Verifies repo root is `/home/kaushik/slam_ws`.
- Verifies `.venv/bin/python` exists and reports `/home/kaushik/slam_ws/.venv/bin/python`.
- Verifies pySLAM `orbslam2_features`, pybind11, and OpenCV caster sources exist locally.
- Builds only under `third_party/build/orbslam2_features/`.
- Copies the extension only under `third_party/local/orbslam2_features/`.
- Writes only `.venv/lib/python3.11/site-packages/orbslam2_features_local.pth`.
- Supports `--help` and `--clean`.
- Reproduces the 2.18A symlink fix by replacing the copied `opencv_type_casters.h` symlink with a real local copy in the build tree.

Validation:

```text
bash tools/build_orbslam2_features_local.sh --help
bash tools/build_orbslam2_features_local.sh
```

Result:

```text
python: /home/kaushik/slam_ws/.venv/bin/python
orbslam2_features: /home/kaushik/slam_ws/third_party/local/orbslam2_features/orbslam2_features.cpython-311-x86_64-linux-gnu.so
symbols: ['ORBextractor', 'ORBextractorDeterministic']
```

No packages were installed and no system/global files were modified.

## 5. `.gitignore` Cleanup

Added ignore coverage for:

```gitignore
third_party/build/
third_party/local/orbslam2_features/*.so
third_party/local/orbslam2_features/*.pyd
third_party/local/orbslam2_features/*.dylib
.venv_backup_before_orbslam2_features/
.venv_backup*/
visual_slam_outputs/
visual_slam.zip
slam_ws*.zip
```

Confirmed ignored:

```text
!! .venv_backup_before_orbslam2_features/
!! third_party/build/
!! third_party/local/
!! visual_slam.zip
!! visual_slam_outputs/checkpoint_2_18B_backend_durability/
!! visual_slam_outputs/checkpoint_2_18B_baseline_validation/
!! visual_slam_outputs/checkpoint_2_18B_default_validation/
```

## 6. Trajectory Evaluator Implementation

Added:

```text
tools/evaluate_tum_trajectory.py
```

Functionality:

- Parses TUM trajectory files: `timestamp tx ty tz qx qy qz qw`.
- Associates estimated poses to nearest ground-truth timestamp with configurable `--max-time-diff`.
- Computes ATE RMSE after SE(3) alignment.
- Computes ATE RMSE after Sim(3) alignment.
- Computes consecutive-pose RPE translational RMSE and rotational RMSE in degrees.
- Writes:
  - `trajectory_metrics.json`
  - `trajectory_metrics.md`
  - `associated_poses.csv`

Real 30-frame evaluator smoke on default validation trajectory:

```text
num_associations: 30
ATE RMSE SE(3): 0.010075874 m
ATE RMSE Sim(3): 0.009266177 m
RPE translational RMSE: 0.007701415 m
RPE rotational RMSE: 0.576575263 deg
```

## 7. Backend Durability Tool Implementation

Added:

```text
tools/run_orb_backend_durability.py
```

Functionality:

- Runs `visual_slam.orbslam.run_tum_rgbd_smoke` for each backend/frame-count pair.
- Supports frame counts like `100`, `300`, and `full`.
- Maps `full` to `--max-frames 0`, which uses all associated frames.
- Collects tracking, runtime, map, trajectory, and frame-log statistics.
- Runs `tools/evaluate_tum_trajectory.py` for each completed trajectory when ground truth exists.
- Writes:
  - `backend_durability_summary.md`
  - `backend_durability_metrics.csv`
  - `backend_durability_metrics.json`
  - per-run command logs and trajectory evaluation folders

The tool updates summary files incrementally after each run so long durability checks leave partial evidence.

## 8. Unit Tests Run

Focused tests:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/visual_slam/orbslam/test_checkpoint_2_18B_trajectory_evaluation.py \
  tests/visual_slam/orbslam/test_checkpoint_2_18B_backend_durability.py
10 passed in 0.22s
```

Final full ORB-SLAM test suite:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/visual_slam/orbslam
100 passed, 1 skipped in 3.84s
```

## 9. Build-Script Validation Result

Passed.

Command:

```bash
bash tools/build_orbslam2_features_local.sh
```

Final import verification:

```text
python: /home/kaushik/slam_ws/.venv/bin/python
orbslam2_features: /home/kaushik/slam_ws/third_party/local/orbslam2_features/orbslam2_features.cpython-311-x86_64-linux-gnu.so
symbols: ['ORBextractor', 'ORBextractorDeterministic']
```

## 10. Default Validation Result

Command:

```bash
.venv/bin/python tools/validate_orbslam_pyslam_port.py \
  --dataset "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_18B_default_validation"
```

Result:

```text
VALIDATION PASSED
```

Smoke details:

| Frames | OK | Lost | Final | Keyframes | Map points |
|---:|---:|---:|---|---:|---:|
| 3 | 3 | 0 | OK | 2 | 2234 |
| 10 | 10 | 0 | OK | 3 | 2484 |
| 30 | 30 | 0 | OK | 6 | 3473 |

Forbidden-log scan found no `Traceback`, `RuntimeWarning`, `NaN/nan`, `overflow`, or `0 vertices to optimize` hits in 2.18B validation/durability outputs.

## 11. OpenCV ORB Durability Result

Command:

```bash
.venv/bin/python tools/run_orb_backend_durability.py \
  --dataset "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_18B_backend_durability" \
  --frame-counts 100 300 full \
  --backends opencv_orb pyslam_orb2
```

OpenCV ORB results:

| Frames | OK/Lost | Final | KF | MP | FPS | ATE SE3 m | ATE Sim3 m | RPE trans m | RPE rot deg |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | 100/0 | OK | 7 | 3991 | 0.140 | 0.025173217 | 0.020735896 | 0.011146809 | 0.708365958 |
| 300 | 300/0 | OK | 11 | 2908 | 0.190 | 0.026261712 | 0.026257972 | 0.013858959 | 0.824498821 |
| full 596 | 596/0 | OK | 17 | 3271 | 0.230 | 0.048299007 | 0.047398223 | 0.012840473 | 0.755723259 |

All OpenCV ORB runs completed with return code `0` and trajectory evaluation status `ok`.

## 12. pySLAM ORB2 Durability Result

pySLAM ORB2 results:

| Frames | OK/Lost | Final | KF | MP | FPS | ATE SE3 m | ATE Sim3 m | RPE trans m | RPE rot deg |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | 100/0 | OK | 7 | 3565 | 0.200 | 0.018958119 | 0.017966244 | 0.009063484 | 0.586953943 |
| 300 | 300/0 | OK | 11 | 3109 | 0.310 | 0.030370815 | 0.022269358 | 0.011186454 | 0.660353098 |
| full 596 | 596/0 | OK | 17 | 3816 | 0.320 | 0.049295297 | 0.049283102 | 0.011074600 | 0.646989150 |

All pySLAM ORB2 runs completed with return code `0` and trajectory evaluation status `ok`.

## 13. ATE/RPE Results

Both backends associated every estimated pose with ground truth for the 100, 300, and full runs.

Observations:

- `pyslam_orb2` had better 100-frame ATE/RPE than `opencv_orb`.
- `opencv_orb` had better SE(3) ATE at 300 and full sequence.
- `pyslam_orb2` had better Sim(3) ATE at 300, but not full sequence.
- `pyslam_orb2` had better RPE translational/rotational metrics at 300 and full sequence.
- Both backends remained tracking-stable with `OK ratio = 1.000` for 100, 300, and full.

Metric files:

- `visual_slam_outputs/checkpoint_2_18B_backend_durability/backend_durability_summary.md`
- `visual_slam_outputs/checkpoint_2_18B_backend_durability/backend_durability_metrics.csv`
- per-run `trajectory_eval/trajectory_metrics.json`

## 14. Backend Recommendation

Keep `opencv_orb` as the default backend for now.

Evidence:

- Both backends are durable over the full 596-frame sequence with no lost frames.
- pySLAM ORB2 is stable and faster in the full durability run (`0.320 FPS` vs `0.230 FPS`) and has better full-sequence RPE.
- OpenCV ORB has slightly better full-sequence SE(3)/Sim(3) ATE:
  - OpenCV full SE(3) ATE RMSE: `0.048299007 m`
  - pySLAM ORB2 full SE(3) ATE RMSE: `0.049295297 m`
  - OpenCV full Sim(3) ATE RMSE: `0.047398223 m`
  - pySLAM ORB2 full Sim(3) ATE RMSE: `0.049283102 m`
- The evidence supports keeping pySLAM ORB2 available and validated, but does not justify changing the default without broader sequence coverage and user approval.

## 15. Remaining Risks

- The C++ `orbslam2_features` extension links against system OpenCV 4.5d while Python imports OpenCV 4.10.0. This checkpoint validated the combination on `rgbd_dataset_freiburg1_desk`, but the ABI split should remain documented.
- No loop closing/relocalization is included in the current sparse RGB-D checkpoint scope. Full-sequence durability passed here, but broader TUM sequences may expose drift or recovery limitations.
- The build script depends on local CMake and compiler availability. It does not install system packages.
- Generated validation outputs are intentionally ignored and should not be committed.

## 16. Files Changed

New or modified 2.18B commit candidates:

- `.gitignore`
- `tools/build_orbslam2_features_local.sh`
- `tools/evaluate_tum_trajectory.py`
- `tools/run_orb_backend_durability.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_18B_trajectory_evaluation.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_18B_backend_durability.py`
- `visual_slam/reference_audit/checkpoint_2_18B/BACKEND_DURABILITY_AND_TRAJECTORY_EVALUATION_AUDIT.md`

Pre-existing uncommitted checkpoint files remain in the worktree and should be grouped deliberately if committing.

## 17. Files Intentionally Not Committed

- `.venv/`
- `.venv_backup_before_orbslam2_features/`
- `third_party/build/`
- `third_party/local/orbslam2_features/*.so`
- `.venv/lib/python3.11/site-packages/orbslam2_features_local.pth`
- `visual_slam_outputs/`
- `visual_slam.zip`
- any generated command logs, trajectory files, trajectory metrics, or smoke outputs
