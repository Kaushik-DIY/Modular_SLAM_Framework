# Modular SLAM Framework

This repository contains a modular SLAM thesis framework built around a shared
fusion layer. The main idea is that a user can choose a front end, loop proposer,
and loop verifier, then run that combination through the same shared SE(2) map.

The current main runner is:

```bash
.venv/bin/python run_fusion.py
```

Run all commands from the repository root:

```bash
cd /home/kaushik/slam_ws
```

Use the project virtual environment for every command:

```bash
.venv/bin/python ...
```

## Architecture

All fusion2 modes write keyframes into one shared C++ backend:

```text
Front End -> Loop Proposer -> Loop Verifier -> Shared SE(2) Map
```

The shared map is implemented through `fusion_core` and contains:

- Signature storage
- STM / WM / LTM memory tiers
- SE(2) pose graph
- Candidate neighbourhood retrieval
- Occupancy grid assembly
- Scan-based branch-and-bound verification

The Python fusion layer lives in:

```text
slam_core/fusion2/
```

Important submodules:

| Folder | Purpose |
|---|---|
| `Front_End/` | LiDAR and visual odometry front ends |
| `Loop_Proposer/` | Loop candidate proposal modules |
| `Loop_Verifier/` | Loop verification modules |
| `Dependencies/` | Dataset, ROS, pose, visual feature, and output helpers |
| `backend.py` | Shared map construction |
| `runner.py` | Batch runner implementation |
| `run_realtime.py` | Real-time switching runner implementation |
| `config.py` | Central fusion configuration |

## Runner Script

The batch runner executes one fixed SLAM configuration end to end:

```bash
.venv/bin/python run_fusion.py \
  --dataset datasets/lab_hybrid \
  --mode lidar \
  --lidar-frontend native_s2s \
  --verifier bnb \
  --output fusion2_outputs
```

Equivalent module form:

```bash
.venv/bin/python -m slam_core.fusion2.runner \
  --dataset datasets/lab_hybrid \
  --mode lidar
```

Show all options:

```bash
.venv/bin/python run_fusion.py --help
```

## Available Modes

Mode names describe the tracking front end and the verification modality.

| Mode | Tracking Front End | Loop Proposer | Loop Verifier | Typical Use |
|---|---|---|---|---|
| `lidar` | LiDAR | Proximity | B&B or ICP on scans | LiDAR-only SLAM |
| `orb` | Visual VO / ORB | DBoW | PnP on visual features | Visual-only SLAM |
| `orb_lidar` | Visual VO / ORB | DBoW | B&B or ICP on synced LiDAR scans | Visual-led, scan-verified SLAM |
| `lidar_orb` | LiDAR | Proximity | PnP on synced visual features | LiDAR-led, visual-verified SLAM |

## Module Choices

### Front End Modules

For LiDAR-led modes:

| Option | Description |
|---|---|
| `native_s2s` | Native C++ scan-to-submap LiDAR front end. Default. |
| `native_s2m` | Native C++ scan-to-map LiDAR front end. |
| `legacy_s2s` | Python Hector scan-to-submap front end. Debug/parity path. |
| `legacy_s2m` | Python Hector scan-to-map front end. Debug/parity path. |

Use with:

```bash
--lidar-frontend native_s2s
```

For visual-led modes:

| Option | Description |
|---|---|
| `native` | Native C++ windowed RGB-D visual odometry. Default. |
| `legacy` | Legacy Python ORB-SLAM based front end. |

Use with:

```bash
--frontend native
```

### Loop Proposer Modules

| Proposer | Used By | Description |
|---|---|---|
| `proximity` | `lidar`, `lidar_orb` | Uses graph pose proximity and retrieves old places from LTM when needed. |
| `dbow` | `orb`, `orb_lidar` | Uses DBoW appearance matching over ORB descriptors. |

The batch runner selects the proposer automatically from `--mode`.

### Loop Verifier Modules

| Verifier | Used By | Description |
|---|---|---|
| `bnb` | `lidar`, `orb_lidar` | Candidate-local branch-and-bound scan matching. Default scan verifier. |
| `icp` | `lidar`, `orb_lidar` | Standalone GICP scan matching seeded by graph prediction. |
| `pnp` | `orb`, `lidar_orb` | ORB descriptor matching plus PnP RANSAC. Selected automatically. |

Use scan verifier with:

```bash
--verifier bnb
```

or:

```bash
--verifier icp
```

`--verifier` applies only to scan-verified modes (`lidar`, `orb_lidar`).

## Common Batch Runs

### LiDAR SLAM, native scan-to-submap, B&B verifier

```bash
.venv/bin/python run_fusion.py \
  --dataset datasets/lab_hybrid \
  --mode lidar \
  --lidar-frontend native_s2s \
  --verifier bnb \
  --output fusion2_outputs/lab_lidar_s2s_bnb
```

### LiDAR SLAM, native scan-to-map, ICP verifier

```bash
.venv/bin/python run_fusion.py \
  --dataset datasets/lab_hybrid \
  --mode lidar \
  --lidar-frontend native_s2m \
  --verifier icp \
  --output fusion2_outputs/lab_lidar_s2m_icp
```

### Visual SLAM, native visual front end, PnP verifier

```bash
.venv/bin/python run_fusion.py \
  --dataset datasets/lab_hybrid \
  --mode orb \
  --frontend native \
  --output fusion2_outputs/lab_orb
```

### Visual-led SLAM with LiDAR scan verification

```bash
.venv/bin/python run_fusion.py \
  --dataset datasets/lab_hybrid \
  --mode orb_lidar \
  --frontend native \
  --verifier bnb \
  --output fusion2_outputs/lab_orb_lidar_bnb
```

### LiDAR-led SLAM with visual PnP verification

```bash
.venv/bin/python run_fusion.py \
  --dataset datasets/lab_hybrid \
  --mode lidar_orb \
  --lidar-frontend native_s2s \
  --output fusion2_outputs/lab_lidar_orb
```

### Quick smoke run

Limit the number of scans/frames:

```bash
.venv/bin/python run_fusion.py \
  --dataset datasets/lab_hybrid \
  --mode lidar \
  --max-scans 500
```

## Real-Time Runner

The real-time runner supports live module switching while keeping the same shared
map alive:

```bash
.venv/bin/python run_fusion_realtime.py \
  --dataset datasets/lab_hybrid \
  --mode lidar \
  --lidar-frontend native_s2s \
  --verifier bnb \
  --proposer proximity \
  --attach-visual
```

Live commands typed into the terminal:

```text
verifier bnb
verifier icp
verifier pnp
proposer proximity
proposer dbow
fe s2s
fe s2m
fe vo
status
quit
```

Notes:

- `pnp` and `dbow` require visual descriptors.
- LiDAR keyframes carry visual descriptors only when using `--attach-visual` or
  starting from `--mode lidar_orb`.
- Visual front-end keyframes always carry visual descriptors.

Deterministic switching schedule example:

```bash
.venv/bin/python run_fusion_realtime.py \
  --dataset datasets/lab_hybrid \
  --mode lidar \
  --lidar-frontend native_s2s \
  --verifier bnb \
  --proposer proximity \
  --attach-visual \
  --switch-schedule "180:proposer dbow;330:fe vo;470:verifier pnp" \
  --speed 1.0
```

## Outputs

Batch and real-time runs write results under the selected output directory,
normally `fusion2_outputs/`.

Typical output files:

| File | Description |
|---|---|
| `trajectory.tum` | Optimized trajectory in TUM format |
| `occupancy.png` | Rendered fused occupancy map |
| `map.npy` | Occupancy probability grid |
| `map_meta.json` | Map origin, resolution, size, and extent |
| `scan_overlay.png` | Raw scan overlay debug figure |
| `run_summary.json` | Run statistics |
| `verifications.csv` | Loop proposal / verification log |

Generated outputs are ignored by git.

## Datasets

The expected local datasets are:

```text
datasets/lab_hybrid
datasets/lab_hybrid_small
datasets/lab_hybrid_3
datasets/lab_hybrid_3_slow
```

Example:

```bash
.venv/bin/python run_fusion.py --dataset datasets/lab_hybrid --mode lidar
```

## Configuration

Most fusion parameters are in:

```text
slam_core/fusion2/config.py
```

LiDAR matcher profiles are defined in:

```text
hector/config.py
```

For implementation details, see:

```text
slam_core/fusion2/README.md
```

## Basic Validation

Compile-check the fusion2 Python files:

```bash
.venv/bin/python -m py_compile $(find slam_core/fusion2 -path '*/__pycache__' -prune -o -name '*.py' -print | sort)
```

Run the fusion2 test suite:

```bash
.venv/bin/python -m pytest tests/fusion2
```

