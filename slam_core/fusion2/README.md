# Fusion v2 — C++-core RTAB-inspired multi-modal shared map

One shared map, four SLAM configurations. All hot-path state lives in the
`fusion_core` C++ extension (`third_party/fusion_core/`); this package is the
thin Python UI: dataset feeding, mode dispatch, loop bookkeeping, outputs.

## Why v2 exists

Python could not carry the hot data path at lab scale: the ORB object map needed
~10.4 GB for 839 keyframes, and the Python branch-and-bound loop search stalled
the Hector+PGO pipeline (see `FUSION2_STATUS.md` and the plan in CLAUDE.md
history). v2 moves Signatures, STM/WM/LTM memory tiers, the SE(2) pose graph,
neighborhood retrieval, occupancy grids, and the correlative B&B verifier into
C++ (zero-copy numpy views, GIL released). Measured wins: **18.7× smaller map
memory**, **85× faster loop verification (26 ms vs 2.2 s)**, full-lab runs in
**~0.12 GB RSS** vs ~10 GB.

## Running (V4: single entry point + full modularity)

```bash
.venv/bin/python run_fusion.py --dataset datasets/lab_hybrid \
    --mode {orb,lidar,orb_lidar,lidar_orb} \
    [--lidar-frontend {native_s2s,native_s2m,legacy_s2s,legacy_s2m}] \
    [--verifier {bnb,icp}] [--max-scans N]
```

- `--lidar-frontend` (modes lidar/lidar_orb): native C++ scan_to_submap (default,
  ~11 ms/scan) or scan_to_map (~15 ms/scan); legacy_* = the Python hector stack
  kept for parity/debugging.
- `--verifier` (modes lidar/orb_lidar): candidate-local B&B (default) or GICP
  with the occupancy-grid cross-check (orb_lidar additionally seeds GICP from
  the B&B coarse pose — VO drift exceeds GICP's convergence basin).
- ALL tunables live in `slam_core/fusion2/config.py` (FusionV2Config); LiDAR
  matcher profiles come from `hector/config.py` per-dataset profiles.
- IMU is always active as the pose-prior fallback (extrapolator gyro+yaw in
  LiDAR modes; dropout dead-reckoning with online self-calibration in VO modes).

Outputs land in `fusion2_outputs/<mode>_<timestamp>/`: `trajectory.tum`,
`occupancy.png` (fused log-odds grid, thesis-grade) + `map.npy`/`map_meta.json`,
`scan_overlay.png` (debug scatter), `run_summary.json`, `verifications.csv`.

## The four modes (all via the SAME shared C++ map)

| Mode | Front-end (odometry/keyframes) | Loop proposing | Loop verification |
|---|---|---|---|
| `orb` | ORB-SLAM (loop closing disabled, propose-only) | DBoW appearance | ORB descriptor match + PnP RANSAC |
| `lidar` | hector scan_to_submap (PGO disabled) | proximity (WM-only) | **candidate-local C++ B&B** (or `--verifier icp`) |
| `orb_lidar` | ORB-SLAM | DBoW appearance | candidate-local C++ B&B on synced scans |
| `lidar_orb` | hector scan_to_submap | proximity (WM-only) | ORB match + PnP on synced visual payloads |

"Candidate-local" is the core idea: scans are retrieved from the shared map
*around the proposed candidate* (graph BFS + metric radius, with RTAB-style LTM
reactivation), assembled into one small occupancy grid in C++, and the query is
verified against that bounded grid — never a dataset-wide search.

## Memory tiers (RTAB semantics, `RTAB_review.md` §3)

STM (30) → WM (200) → LTM (in-RAM behind the `LtmStore` interface; SQLite is a
drop-in later). Rehearsal merges only fire for stationary keyframes (motion-
filtered keyframes must not merge); weights via `w_t += w_c + 1`; transfer picks
the oldest of the lowest-weight; loop hypotheses reactivate LTM members + graph
neighbors into WM.

## Key implementation notes

- Signatures store **raw** LiDAR scans (~560 valid beams), not voxel-filtered
  points — voxel-thin scans make dotted walls and suppress verification scores.
- Local grids use hector's log-odds balance (`l_occ=0.85`, `l_free=-0.1`);
  aggressive free-carving erodes wall hits.
- New graph nodes are initialized in the optimized frame by composing the last
  graph pose with the front-end's relative motion, so loop corrections never
  fight fresh odometry.
- Acceptance: B&B `coarse ≥ 0.55 AND refined ≥ 0.60`; ICP `fitness ≥ 0.6`;
  PnP `inliers ≥ 15` + relative-pose sanity gate vs the graph prediction.

## Datasets

`datasets/lab_hybrid` (two rooms, 4147 scans / 6468 RGB-D / IMU) — loop-closure
testbed. `datasets/lab_hybrid_small` (one room, 1581 scans) — fast validation.
Both: 909-beam 360° LiDAR @9.6 Hz, RGB-D @15 fps, IMU @30 Hz, soft-synced.

## Native C++ VO front-end (v3, default for orb/orb_lidar)

`fusion_core.VoFrontend`: lean windowed RGB-D visual odometry (ORB + grid
bucketing, 7-keyframe sliding window, projection matching, SE(3) Gauss-Newton,
depth re-init safety net). Replaces the legacy Python ORB-SLAM stack as the
default (`--frontend legacy` keeps the old path selectable). Per-frame ~14-19 ms.

**Local bundle adjustment (v3.5):** after each keyframe the window runs a
Schur-complement + Levenberg-Marquardt local BA that jointly refines the window
poses (oldest fixed) and the map points they co-observe — this is what bounds
inter-keyframe drift and is the difference between a coherent and a smeared
vision map (the reference ORB-SLAM2 makes its map purely from local BA, with loop
closing and global BA OFF). `reinit_patience` coasts through brief tracking
dropouts so a single blurred frame no longer nukes the window. Pure Gauss-Newton
diverges here (weakly-constrained cameras jump metres); LM with a cost-decrease
guard keeps corrections at mm scale. Vision maps are coherent but still less crisp
than the LiDAR-led modes (no persistent covisibility map / relocalization yet).
Appearance proposing for native runs uses `AppearanceIndex` (DBoW3 directly over
Signature descriptors — no legacy keyframe objects needed).

## Validated results (see FUSION2_STATUS.md for the full table)

On lab_hybrid (BIG, two rooms):
- `lidar`: 382 kf, **74 B&B loops**, clean 2-room map, 0.12 GB RSS.
- `lidar_orb`: **26 PnP cross-modal loops** @7.5 ms, clean 2-room map, 0.29 GB.
- `orb` (native, v3.6): **38 PnP loops, clean 2-room map (end≈start)**, 32.6 fps, 0.72 GB.
- `orb_lidar` (native, v3.6): **43 cross-modal loops, 2-room map**, 31.9 fps, 0.72 GB.
- `lidar --verifier icp` (small): 30 loops @10.6 ms.

v3.6 made the vision-led maps coherent by fixing four root causes (see
`FUSION2_STATUS.md` §V3.6): REP-103-consistent node frames (PnP loop edges were
90°-rotated vs spine edges and warped the graph), pose-free brute-force + PnP
recovery on fast turns (failures were turn-bursts, not featureless areas),
IMU dead-reckoning scoped to dropouts with online sign/up-axis self-calibration
(~8° mount tilt measured from data), and soft spine edges + map exclusion for
dead-reckoned REINIT keyframes.
