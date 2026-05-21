# Checkpoint 2.29A — RGB-D Runner Implementation Audit

## Purpose

Create a final dataset-agnostic RGB-D SLAM runner that preserves the working
baseline behavior from `visual_slam/orbslam/run_tum_rgbd_smoke.py` while
adding a clean dataset/camera configuration layer for:

- TUM RGB-D datasets
- lab/JetRacer RGB-D datasets with `camera.yaml`

This checkpoint intentionally avoids changes to tracking, local mapping, loop
closing, Global BA, optimizer, and map-point logic.

## Files Inspected

- `visual_slam/orbslam/run_tum_rgbd_smoke.py`
- `visual_slam/orbslam/io/rgbd_dataset.py`
- `visual_slam/orbslam/io/tum_rgbd.py`
- `visual_slam/orbslam/io/__init__.py`
- `visual_slam/orbslam/slam/camera.py`
- `tools/export_orbslam_map.py`
- `tools/run_fr1_room_full_evaluation.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_14_tum_smoke_runner.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_26A_fr1_room_evaluation_tools.py`

## Current `run_tum_rgbd_smoke.py` Behavior Reused

The new runner keeps the smoke runner’s working execution path:

- RGB load via `cv2.IMREAD_COLOR`
- depth load via `cv2.IMREAD_UNCHANGED`
- `slam.track(...)` per frame
- sequential local mapping stepping unless threading is enabled
- loop-closing queue stepping
- optional loop-debug record collection
- optional loop candidate report dumping
- optional stop-after loop debug controls
- final trajectory export from `slam.get_final_trajectory()`
- export of only `SlamState.OK` poses to TUM trajectory format
- sparse map export through `export_orbslam_map(...)`
- per-frame CSV logging
- keyframe/trajectory consistency printout when available

## New `run_rgbd_slam.py` Design

New file:

- `visual_slam/orbslam/run_rgbd_slam.py`

Main additions:

- dataset-type aware CLI
- explicit camera-profile and camera-config handling
- dataset-agnostic effective run config export
- final output naming based on dataset name instead of smoke suffixes
- `run_summary.json` generation for each run

## Dataset Type Handling

Implemented dataset types:

- `tum_rgbd`
- `lab_rgbd`
- `auto`

Auto-detection rules:

- `rgbd_dataset_freiburg*` folder names or `groundtruth.txt` imply `tum_rgbd`
- `camera.yaml` + `rgb/` + `depth/` + `associations.txt` imply `lab_rgbd`
- ambiguous cases raise a clear error and require explicit `--dataset-type`

## Camera Config Handling

Updated helper module:

- `visual_slam/orbslam/io/rgbd_dataset.py`

Implemented support for:

- TUM camera auto/profile selection using existing Freiburg logic
- lab camera loading from `camera.yaml`
- explicit `--camera-config`
- automatic `dataset/camera.yaml` fallback for lab datasets
- clear failure when lab data is requested without a camera config

The YAML reader is dependency-free and intentionally minimal so the task stays
inside the existing project venv without adding `PyYAML`.

## Depth Factor Handling

For lab datasets:

- `depth.depth_map_factor` is read from `camera.yaml`
- effective `depth_factor` is computed as `1.0 / depth_map_factor`
- both values are printed and written to `effective_run_config.json`

This preserves the required convention:

- TUM depth map factor is typically `5000.0`
- many lab/RealSense datasets use `1000.0`

## Output Files

Each run now writes:

- `effective_run_config.json`
- `trajectory_<dataset_name>.txt`
- `frame_log_<dataset_name>.csv`
- `map_points.ply`
- `keyframes.json`
- `keyframe_graph.json`
- `loop_debug_candidates.csv` when loop debug is enabled
- `loop_candidate_pair_reports/` when requested
- `run_summary.json`

## Tests Run

Targeted checkpoint tests:

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/visual_slam/orbslam/test_checkpoint_2_29A_rgbd_runner.py
```

Status:

- Passed: 11/11 tests

Broader ORB-SLAM suite:

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam
```

Status:

- Started successfully
- Reached roughly 67% before an unrelated segmentation fault in:
  `tests/visual_slam/orbslam/test_cpp_slam_core_phase2_frame.py::test_reset_points`

## Validation Runs

Requested small TUM validation:

```bash
python -m visual_slam.orbslam.run_rgbd_slam \
  "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" \
  --dataset-type tum_rgbd \
  --camera-profile auto \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_29A/fr1_desk_30" \
  --feature-backend pyslam_orb2 \
  --max-frames 30 \
  --disable-loop-closing \
  --print-every 1
```

Runtime status at audit drafting time:

- Completed successfully

Observed results:

- `frames_attempted = 30`
- `tracking_ok_count = 30`
- `tracking_lost_count = 0`
- `final_state = OK`
- `keyframes = 10`
- `map_points = 4295`
- `trajectory_poses = 30`
- `errors = 0`
- output directory:
  `visual_slam_outputs/checkpoint_2_29A/fr1_desk_30`

Artifacts verified:

- `effective_run_config.json`
- `trajectory_rgbd_dataset_freiburg1_desk.txt`
- `frame_log_rgbd_dataset_freiburg1_desk.csv`
- `map_points.ply`
- `keyframes.json`
- `keyframe_graph.json`
- `run_summary.json`

Requested small lab validation:

```bash
python -m visual_slam.orbslam.run_rgbd_slam \
  "$HOME/slam_ws/datasets/lab_rgbd_run_2" \
  --dataset-type lab_rgbd \
  --camera-config "$HOME/slam_ws/datasets/lab_rgbd_run_2/camera.yaml" \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_29A/lab_rgbd_run_2_30" \
  --feature-backend pyslam_orb2 \
  --max-frames 30 \
  --disable-loop-closing \
  --print-every 1
```

Status:

- Completed successfully on the real local dataset

Observed dataset properties:

- dataset root: `datasets/lab_rgbd_run_2`
- layout verified: `rgb/`, `depth/`, `rgb.txt`, `depth.txt`, `associations.txt`, `camera.yaml`
- association count: `4494`
- image size: `640x480`
- RGB dtype: `uint8`
- depth dtype: `uint16`
- `DepthMapFactor = 1000.0`
- effective `depth_factor = 0.001`
- camera intrinsics from `camera.yaml`:
  `fx=609.883300781`, `fy=609.177246094`,
  `cx=324.920776367`, `cy=229.748153687`

Observed 30-frame runtime result:

- `frames_attempted = 30`
- `tracking_ok_count = 30`
- `tracking_lost_count = 0`
- `final_state = OK`
- `keyframes = 1`
- `map_points = 780`
- `trajectory_poses = 30`
- `errors = 0`
- `avg_fps ≈ 1.85`

Artifacts verified:

- `visual_slam_outputs/checkpoint_2_29A/lab_rgbd_run_2_30/effective_run_config.json`
- `visual_slam_outputs/checkpoint_2_29A/lab_rgbd_run_2_30/trajectory_lab_rgbd_run_2.txt`
- `visual_slam_outputs/checkpoint_2_29A/lab_rgbd_run_2_30/frame_log_lab_rgbd_run_2.csv`
- `visual_slam_outputs/checkpoint_2_29A/lab_rgbd_run_2_30/map_points.ply`
- `visual_slam_outputs/checkpoint_2_29A/lab_rgbd_run_2_30/keyframes.json`
- `visual_slam_outputs/checkpoint_2_29A/lab_rgbd_run_2_30/keyframe_graph.json`
- `visual_slam_outputs/checkpoint_2_29A/lab_rgbd_run_2_30/run_summary.json`

Also verified:

- auto dataset detection works for `datasets/lab_rgbd_run_2`
- automatic `dataset/camera.yaml` pickup works without passing `--camera-config`
- the saved run config now records when baseline/depth-threshold values are
  coming from RGB-D defaults rather than explicit camera.yaml keys

## Remaining Limitations

- The broader `tests/visual_slam/orbslam` suite currently has an unrelated
  native-code segmentation fault outside this checkpoint’s file set.
- The custom YAML reader is intentionally narrow and targeted at the expected
  project camera config shapes.
- The 30-frame TUM validation completed successfully but is slow with
  `pyslam_orb2` on this machine at roughly `0.10 FPS`.
- The flat ORB-SLAM2-style `camera.yaml` used by `lab_rgbd_run_2` does not
  specify an explicit virtual RGB-D baseline or `ThDepth`, so the runner
  currently records and reports the default RGB-D values it applies:
  baseline `0.08 m` and `ThDepth = 40`.

## Next Recommended Step

1. Inspect a longer `lab_rgbd_run_2` segment with loop closing enabled once
   you want to evaluate lab loop behavior rather than runner integration.
2. If needed, add an explicit lab-side camera key for virtual RGB-D baseline
   and/or `ThDepth` so those no longer rely on the current defaults.
3. Investigate the unrelated `test_cpp_slam_core_phase2_frame.py` native-code
   segmentation fault separately from this runner checkpoint.
