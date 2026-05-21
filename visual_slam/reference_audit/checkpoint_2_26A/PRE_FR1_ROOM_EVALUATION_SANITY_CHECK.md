# Checkpoint 2.26A Pre fr1_room Evaluation Sanity Check

## Purpose

Prepare a full `rgbd_dataset_freiburg1_room` evaluation of the existing RGB-D ORB2 pipeline with benchmark logs, sparse map export, GT-reference point-cloud generation, and thesis-ready figures.

This checkpoint is evaluation and visualization only. No new core SLAM algorithm work is planned.

## Dataset Presence

Dataset checked:

```text
/home/kaushik/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room
```

Required files/folders are present:

```text
groundtruth.txt
associations.txt
rgb/
depth/
```

## Backend Status

Python executable was verified as:

```text
/home/kaushik/slam_ws/.venv/bin/python
```

Native dependencies loaded from local/project paths:

```text
pydbow3
orbslam2_features
g2o
```

Full-run backend will be:

```text
pyslam_orb2
```

## Current Tests

Baseline targeted tests before editing:

```text
tests/visual_slam/orbslam/test_checkpoint_2_24_global_ba.py
tests/visual_slam/orbslam/test_checkpoint_2_25_optimizer_parity.py
```

Result:

```text
22 passed
```

Full ORB-SLAM test suite before editing:

```text
tests/visual_slam/orbslam
```

Result:

```text
176 passed, 1 skipped
```

## Current Runner Capabilities

`visual_slam/orbslam/run_tum_rgbd_smoke.py` can run real TUM RGB-D frames, select `pyslam_orb2`, enable loop closing and Global BA, save a TUM trajectory, and write a basic per-frame CSV.

## Current Logging Gaps

The existing smoke runner does not produce the checkpoint-required artifact set:

```text
run_summary.json
frame_log.csv
loop_events.csv
global_ba_events.csv
```

Loop and Global BA diagnostics already exist in the current loop-closing/GBA code and can be exported without changing core SLAM behavior.

## Current Map Export Gaps

The map/keyframe/map-point objects expose enough state to export:

```text
map_points.ply
keyframes.json
keyframe_graph.json
```

No standalone exporter existed before this checkpoint.

## Planned Tools

Add or update:

```text
tools/run_fr1_room_full_evaluation.py
tools/build_tum_reference_cloud.py
tools/export_orbslam_map.py
tools/plot_fr1_room_evaluation.py
```

## Planned Full-Run Validation

Execution plan:

```text
1. 100-frame dry runs A/B/C.
2. Full Run C only by default:
   loop_closing = ON
   global_ba = ON
   backend = pyslam_orb2
3. Generate trajectory metrics, map exports, reference cloud, plots, and final reports.
4. Report clearly whether real loop-triggered Global BA was exercised.
```
