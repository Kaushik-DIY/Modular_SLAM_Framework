# `slam_core/fusion/` — RTAB-Map-inspired multi-modal SLAM (v1)

A unified SLAM front-end that fuses the existing **ORB-SLAM** (RGB-D visual) and
**LiDAR** (Hector / scan-to-submap) pipelines under an RTAB-Map-inspired
architecture: STM/WM/LTM memory tiers, a single SE(2) pose graph, and
**cross-modal loop verification** (one sensor proposes loop candidates, the
other confirms them geometrically).

> Design source of truth: `RTAB_inspired_implementation_plan.md` (root).
> Phase status: `FUSION_STATUS.md` (root). Upstream reference: `RTAB_review.md`.

## The four modes

`python -m slam_core.fusion.runner --mode {orb,lidar,vlmain,lvmain} ...`

| Mode | Name | What it does |
|------|------|--------------|
| `orb`    | Visual-only (pass-through) | Delegates verbatim to `visual_slam/orbslam/run_rgbd_slam.py`. Byte-equal to running it directly. |
| `lidar`  | LiDAR-only (pass-through)  | Delegates verbatim to `hector/run_local_slam_new.py`. |
| `vlmain` | **Visual-main + LiDAR verifier** (Mode C) | ORB-SLAM proposes loop candidates (appearance); **ICP** (`small_gicp`) confirms them on the 2D LiDAR scans. |
| `lvmain` | **LiDAR-main + Visual verifier** (Mode D) | LiDAR proximity/B&B proposes; **ORB + PnP** confirms on the RGB features. |

In both fusion modes the upstream pipelines run **propose-only**: ORB-SLAM's own
Sim(3) loop verification and the LiDAR g2o PGO are *not* run — the unified
`FusionGraph` owns optimization.

## Pipeline (Mode C / D)

```
FusionDataset ─► SoftSync ─► front-end service ─► Signature
   (RGB-D + synth 2D LiDAR)   (visual | lidar)        │
                                                      ├─► MemoryManager  (STM ► WM ► LTM)
                                                      ├─► FusionGraph    (SE(2), g2o)
                                                      └─► LoopProposer ─► LoopVerifier ─► loop edge
                                                          (propose only)   (ICP | Visual)
                                                                                 │
                                                          FusionGraph.solve() ◄──┘
                                                                                 │
                                          map_output ─► trajectory.tum + occupancy.png
```

## Module map

| File | Role |
|------|------|
| `runner.py` | Entry point; mode dispatch; `run_mode_c` / `run_mode_d` orchestration. |
| `config.py` | `Mode` enum + `FusionConfig` (all tunables). |
| `signature.py` | `Signature` keyframe payload + SE(3)→SE(2) projection. |
| `sync.py` | `SoftSync` — ±tolerance RGB-D↔scan timestamp matching. |
| `lidar_synth.py` | Synthetic 2D LiDAR from a TUM depth row. |
| `dataset.py` | `FusionDataset` — TUM RGB-D + synthetic LiDAR loader. |
| `memory.py` | `MemoryManager` — STM/WM/LTM, rehearsal merge, weight rule, transfer. |
| `graph.py` | `FusionGraph` — keyframe-only SE(2) graph on `G2oBackend2D`. |
| `icp_verifier.py` | `ICPLoopVerifier` (`small_gicp`) — `LoopVerifier` Protocol. |
| `visual_verifier.py` | `VisualLoopVerifier` (ORB + PnP) — `LoopVerifier` Protocol. |
| `frontends.py` | Runnable stand-ins: `OrbRgbdVoBackend`, `BruteForceOrbDetector`, `ProximityTargetProvider`. |
| `adapters/` | Propose-only loop proposers + front-end services (dependency-injected). |
| `map_output.py` | TUM trajectory writer + occupancy grid assembler. |

## Reused (not reimplemented)

- `carto/pose_graph/backends/g2o_backend_2d.py` — the SE(2) g2o solver.
- `slam_core/loop_closure.py` — `LoopVerifier` / `TargetProvider` / `ConstraintSink` Protocols.
- `slam_core/common/{types,se2,types3d}.py` — `Pose2` / SE(2) algebra / `Pose3D`.
- `visual_slam/orbslam/io/tum_rgbd.py` — TUM RGB-D loader.

## Running

```bash
# Mode C (visual-main, LiDAR-verified) on TUM fr1_room, emit trajectory + grid
.venv/bin/python -m slam_core.fusion.runner --mode vlmain \
    --dataset datasets/tum/rgbd_dataset_freiburg1_room \
    --output fusion_outputs --max-frames 300 --optimize-every 20

# Pass-through (identical to invoking the underlying runner directly)
.venv/bin/python -m slam_core.fusion.runner --mode orb -- <run_rgbd_slam.py args>
.venv/bin/python -m slam_core.fusion.runner --mode lidar -- <run_local_slam_new.py args>

# Phase-1 dataset/sync probe
.venv/bin/python tools/probe_fusion_dataset.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_room --num-frames 20
```

Outputs land in `fusion_outputs/<run_id>/{trajectory.tum, occupancy.png}`.

## Config flags (CLI, Mode C)

| Flag | Default | Meaning |
|------|---------|---------|
| `--dataset` | — | TUM RGB-D sequence directory (required). |
| `--output` | `fusion_outputs` | Output root; a `<run_id>/` subdir is created. |
| `--max-frames` | `0` (all) | Cap on frames processed. |
| `--stm-size` | `30` | Short-Term Memory size (rehearsal window). |
| `--wm-cap` | `200` | Working Memory cap before WM→LTM transfer. |
| `--ltm-cap` | `1000` | In-RAM LTM safety cap (oldest dropped past it). |
| `--optimize-every` | `30` | Run g2o every N keyframes (when loops exist). |

`FusionConfig` (in `config.py`) holds the remaining knobs — sync tolerance,
ICP thresholds (`icp_max_corr`/`icp_fitness`/`icp_inlier_rmse`), visual
thresholds (`visual_nndr`/`visual_min_inliers`), and `rehearsal_similarity`.

## Memory tiers (RTAB-Map semantics)

- **STM**: bounded queue of recent keyframes; site of the **rehearsal merge**
  (near-identical consecutive observations collapse into one node whose weight
  grows: `w_t += w_c + 1`). Protected from loop scoring.
- **WM**: scored for loop closure and optimized. When it exceeds `wm_cap`, the
  **oldest of the lowest-weighted** nodes transfer to LTM (recent-WM protected).
- **LTM**: in-RAM in v1 (SQLite is v2). Reactivation pulls a node and up to N
  neighbours back into WM.

## Status, profile & known limitations

- **Tests:** `.venv/bin/pytest tests/fusion/ -q` (per-phase checkpoints p1–p10).
- **Profile (dev x86_64, fr1_room, 60 keyframes, Mode C):** ~**61 ms median /
  63 ms mean** per keyframe (p95 ~109 ms on the periodic-optimization keyframes).
  Comfortably under the 100 ms v1 target for the typical per-keyframe cost.
- **Stand-ins:** the runnable Mode C/D front-ends (`OrbRgbdVoBackend`,
  `BruteForceOrbDetector`, `ProximityTargetProvider`) are lightweight
  placeholders behind the Phase 6 adapter seams. Production swaps in the full
  ORB-SLAM tracking + DBoW `KeyFrameDatabase` and the `scan_to_submap` front-end
  + `CartoTargetProvider` — a backend change only.
- **Deferred (see `FUSION_STATUS.md` open follow-ups):** SQLite LTM persistence,
  loop-edge residual rollback (plan §10.3), graph-side rehearsal-merge node
  removal, log-odds free-space in the grid, and the real-front-end Mode D CLI.
  All v2/v3 per `CLAUDE.md` §9.
```
