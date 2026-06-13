# RTAB-Inspired Multi-Modal SLAM — v1 Implementation Plan

> **Companion to `RTAB_review.md`.** This plan describes the v1 we will actually build, grounded in the decisions captured below. Read `RTAB_review.md` first if you need the upstream-architecture context for any choice referenced here.

---

## 1. Context

The thesis workspace already contains three production-quality SLAM pipelines:
- **ORB-SLAM** (RGB-D visual SLAM) — `visual_slam/orbslam/`
- **Hector** (2D LiDAR scan-to-map / scan-to-submap) — `hector/`
- **Cartographer-style** 2D LiDAR with online g2o PGO — `carto/`

Each runs standalone. None of them shares data with another, and the visual pipeline cannot benefit from the LiDAR's geometric reliability or vice versa.

The next thesis phase is to make these pipelines collaborate, taking **RTAB-Map** as the architectural inspiration. The core gain we want is **cross-modal loop-closure verification**: a candidate revisit found by one modality is confirmed by the other before becoming a graph constraint. RTAB-Map shows that this style of fusion, when combined with a disciplined three-tier memory manager, is real-time-feasible on modest hardware. We want the same property for our system, on a Jetson Nano.

This plan covers **v1 only**: cross-modal verification working, in-RAM memory tiers, four user-selectable runner modes. SQLite persistence, multi-session resumption, ROS integration on the Jetson, and ICP-as-LiDAR-loop-verifier are explicitly deferred to v2/v3 (briefly mentioned in §16).

## 2. Goals

| # | Goal | Acceptance evidence |
|---|---|---|
| G1 | A new **unified runner** lets the user pick one of four operating modes (Visual-only / LiDAR-only / Visual-main+LiDAR-verifier / LiDAR-main+Visual-verifier) via a single CLI flag. | `python -m slam_core.fusion.runner --mode {orb,lidar,vlmain,lvmain} ...` produces a trajectory + 2D occupancy grid for each mode. |
| G2 | A **shared graph + signature payload** holds, per keyframe, the pose + ORB descriptors + most-recent LiDAR scan + local occupancy grid. | The same data structure is read by the visual verifier in Mode D and the LiDAR verifier in Mode C, without rerunning the front-ends. |
| G3 | A **three-tier memory manager** (STM / WM / LTM) faithfully implements RTAB-Map's rehearsal merge, weight-based transfer, and bounded WM, fully in RAM. | Run on TUM fr1_room with `WM_cap=200`; observe oldest-lowest-weight signatures transferred to LTM; total live signatures ≤ cap. |
| G4 | A **small_gicp ICP verifier** confirms loop candidates from the visual side, and a **visual PnP verifier** confirms candidates from the LiDAR side. | Both verifiers are interchangeable behind the existing `LoopVerifier` Protocol in `slam_core/loop_closure.py`. |
| G5 | All four modes produce a trajectory (TUM format) and a 2D occupancy grid built from optimized poses. | End-to-end runs on TUM RGB-D + synthesized 2D LiDAR with measured ATE/RPE vs ground truth. |
| G6 | The implementation **reuses** existing modules wherever possible — ORB-SLAM's loop detector, the g2o pose-graph back-end, the LiDAR front-ends, the generic `LoopClosureManager`. New code is added only where no existing module fits. | The "Reuse map" in §6 names each reused module and the adapter (if any) that connects it. |

## 3. Non-goals (deferred)

- SQLite persistence and multi-session resumption (v2).
- ROS integration on the Jetson — sensors are loaded from offline data in v1 (v3).
- ICP as a LiDAR loop-closure verifier inside the LiDAR-only pipeline (v2).
- ICP inside per-scan front-end refinement (out of scope).
- 3D OctoMap / dense point cloud outputs (v2, after Jetson validation).
- A new appearance-based BoW from scratch — we reuse ORB-SLAM's detector.
- Online realtime visualization — the existing `run_realtime_viz.py` is not touched in v1.

## 4. Constraints

| Constraint | How it shapes the plan |
|---|---|
| **Python-only application code** | All orchestration, graph management, memory tiers in Python. Heavy numerical work delegated to existing C++-backed bindings: `g2o` (already used), `small_gicp` (new), OpenCV. |
| **No ROS in v1** | Data is loaded offline from TUM RGB-D-style folders + a synthesized 2D LiDAR derived from depth-image slices. ROS integration is a v3 concern; the dataset loader is a clean Python abstraction. |
| **Jetson Nano (4 GB RAM, low-power GPU)** | Conservative memory tier defaults (STM=30, WM cap=200, configurable). No GPU-dependent libraries (Open3D, learned descriptors). small_gicp chosen for low-overhead C++ ICP. |
| **Existing pipelines must keep working unchanged when run standalone** | Modes A (`orb`) and B (`lidar`) are pass-throughs to the existing runners. Fusion code never modifies the standalone code paths. |
| **Planar indoor robot** | Unified graph is SE(2). ORB-SLAM's SE(3) keyframe poses are projected to SE(2) at the fusion-layer boundary, using a known camera→base-link TF. |

## 5. Architecture overview

```
                       ┌─────────────────────────────────────────┐
                       │  slam_core/fusion/runner.py             │
                       │  Unified runner — mode selector         │
                       │   --mode orb     -> delegate Mode A     │
                       │   --mode lidar   -> delegate Mode B     │
                       │   --mode vlmain  -> Mode C  (V→L verify)│
                       │   --mode lvmain  -> Mode D  (L→V verify)│
                       └────────────────┬────────────────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────────┐
              ▼                         ▼                             ▼
     ┌────────────────┐       ┌────────────────┐         ┌──────────────────────┐
     │  Mode A: ORB   │       │ Mode B: LiDAR  │         │  Mode C / D (FUSION) │
     │  ────────────  │       │ ────────────── │         │  ──────────────────── │
     │  Delegate to   │       │ Delegate to    │         │ Both front-ends are  │
     │  visual_slam/  │       │ hector/        │         │ services. Fusion     │
     │  orbslam/      │       │ run_local_slam_│         │ layer owns the graph,│
     │  run_rgbd_slam │       │ new.py         │         │ memory tiers, loop   │
     │  .py           │       │                │         │ verification.        │
     │  (unchanged)   │       │ (unchanged)    │         │                      │
     └────────────────┘       └────────────────┘         └──────────────────────┘

      ┌─────────────────────────────────────────────────────────────────────────────┐
      │  Mode C / D — internal architecture                                         │
      │                                                                             │
      │   ┌──────────────────┐    ┌──────────────────────┐                          │
      │   │ Dataset loader   │───▶│ SoftSync             │                          │
      │   │ (TUM + synth     │    │ ±50 ms window        │                          │
      │   │  2D lidar)       │    │ (RGB-D, scan, t)     │                          │
      │   └──────────────────┘    └──────┬───────────────┘                          │
      │                                  │                                          │
      │            ┌─────────────────────┴────────────────────────┐                 │
      │            ▼                                              ▼                 │
      │   ┌─────────────────────┐                       ┌─────────────────────┐    │
      │   │ Visual front-end    │                       │ LiDAR front-end     │    │
      │   │ (ORB-SLAM           │                       │ (scan_to_submap     │    │
      │   │  tracking)          │                       │  or scan_to_map)    │    │
      │   │ produces            │                       │ produces            │    │
      │   │ - keyframes         │                       │ - per-scan rel pose │    │
      │   │ - relative pose     │                       │ - LiDAR scan        │    │
      │   │ - ORB descriptors   │                       │                     │    │
      │   └──────────┬──────────┘                       └──────────┬──────────┘    │
      │              │                                             │               │
      │              └────────────────────┬────────────────────────┘               │
      │                                   ▼                                        │
      │                  ┌───────────────────────────────────────┐                 │
      │                  │  FusionGraph + MemoryManager          │                 │
      │                  │  Per fused keyframe -> Signature      │                 │
      │                  │   { pose_SE2, orb_kpts+desc, scan,    │                 │
      │                  │     local_occ_grid, weight, links }   │                 │
      │                  │  STM -> WM -> LTM lifecycle           │                 │
      │                  └────────────────────┬──────────────────┘                 │
      │                                       │                                    │
      │                  Mode-specific candidate flow:                             │
      │                                                                            │
      │   Mode C (visual-main):                                                    │
      │     ORB candidates from KeyFrameDatabase ──▶ ICPLoopVerifier(small_gicp)   │
      │                                                                            │
      │   Mode D (lidar-main):                                                     │
      │     LiDAR candidates from B&B/proximity   ──▶ VisualLoopVerifier(PnP)      │
      │                                                                            │
      │                                       │                                    │
      │                                       ▼                                    │
      │                  ┌───────────────────────────────────────┐                 │
      │                  │ Unified g2o SE(2) backend             │                 │
      │                  │ (reuses carto/pose_graph/backends/    │                 │
      │                  │  g2o_backend_2d.py)                   │                 │
      │                  │ Constraints:                          │                 │
      │                  │  - spine (consecutive keyframes)      │                 │
      │                  │  - cross-modal loop edges             │                 │
      │                  └────────────────────┬──────────────────┘                 │
      │                                       │                                    │
      │                                       ▼                                    │
      │                  ┌───────────────────────────────────────┐                 │
      │                  │ Map outputs:                          │                 │
      │                  │  - trajectory.tum                     │                 │
      │                  │  - occupancy_grid.png                 │                 │
      │                  └───────────────────────────────────────┘                 │
      └─────────────────────────────────────────────────────────────────────────────┘
```

## 6. Reuse map — what we extend vs what we build new

### 6.1 Reused (no modification needed)

| Existing module | Role in v1 | Citation |
|---|---|---|
| `carto/pose_graph/backends/g2o_backend_2d.py` | The g2o SE(2) solver, vertex/edge mapping, Huber kernel, spine regularization | Used as-is by `FusionGraph` |
| `carto/pose_graph/pose_graph_2d.py` | Reference implementation of an SE(2) PGO that we mirror at the fusion-graph layer | Pattern source |
| `slam_core/loop_closure.py` | The generic `LoopClosureManager`, `LoopVerifier`, `TargetProvider`, `ConstraintSink` Protocols | Becomes the abstraction layer behind both Mode C and Mode D |
| `slam_core/matching/scan_to_submap/` | LiDAR front-end (matcher + submaps) — produces per-scan relative pose for fusion's neighbor links | Front-end service in fusion modes |
| `slam_core/matching/scan_to_map.py` | Alternative LiDAR front-end (selectable via existing flag) | Front-end service in fusion modes |
| `carto/local_slam/range_to_points.py` + `pose_extrapolator.py` | LiDAR scan preprocessing + odometry prediction | Same path as standalone LiDAR runner |
| `visual_slam/orbslam/io/tum_rgbd.py` + `rgbd_dataset.py` | TUM RGB-D loader | Used by fusion dataset loader |
| `visual_slam/orbslam/slam/keyframe_database.py` | ORB-SLAM's DBoW-style loop candidate detector | Polled in propose-only mode for Mode C |
| `slam_core/common/types.py` (`Pose2`) + `se2.py` | The SE(2) algebra we already use everywhere on the LiDAR side | All fusion poses are `Pose2` |
| `slam_core/common/types3d.py` (`Pose3D`) | ORB-SLAM keyframe pose type | Projected to `Pose2` at fusion boundary |

### 6.2 Extended (adapter layer added; original code untouched)

| Existing module | Adapter we add | Purpose |
|---|---|---|
| `visual_slam/orbslam/slam/loop_closing.py` | `slam_core/fusion/adapters/orb_loop_proposer.py` | Wraps ORB-SLAM's loop closing so it **only proposes** candidates and **does not verify or add Sim(3) constraints** when fusion mode is active. A configuration flag `propose_only=True` is passed in via the adapter; the existing loop_closing.py is not edited (the adapter intercepts the proposal callback before verification runs). |
| `carto/loop_closure_adapter.py` | `slam_core/fusion/adapters/lidar_loop_proposer.py` | Wraps the LiDAR `CartoLoopClosureAdapter` so it **only proposes** B&B candidates and **does not verify or add intra/inter constraints** when fusion mode is active. Mirrors the same `propose_only` pattern. |
| `hector/adapter.py` (and `carto/adapter.py`) | `slam_core/fusion/adapters/lidar_frontend_service.py` | Wraps the LiDAR adapter to expose a clean "give me the next per-scan relative pose + raw scan" service interface. Underlying behaviour unchanged. |
| `visual_slam/orbslam/slam/slam.py` | `slam_core/fusion/adapters/visual_frontend_service.py` | Same idea for ORB-SLAM: a clean "give me the next keyframe pose + ORB descriptors + tracking status" service interface. |

### 6.3 New modules under `slam_core/fusion/` (no existing equivalent)

```
slam_core/fusion/
├── __init__.py
├── runner.py               # Unified runner; mode selection; main loop
├── config.py               # Mode enum + tier sizes + sync tolerance + verifier params
├── signature.py            # Signature class: SE(2) pose, ORB desc, scan, local grid, weight
├── memory.py               # MemoryManager: STM + WM + LTM with rehearsal & transfer
├── graph.py                # FusionGraph: SE(2) graph built on top of G2oBackend2D
├── sync.py                 # SoftSync: timestamp matching with tolerance window
├── lidar_synth.py          # Depth-slice → synthetic 2D LiDAR scan utility (for TUM)
├── dataset.py              # FusionDataset: TUM RGB-D + synthesized LiDAR
├── icp_verifier.py         # small_gicp wrapper implementing LoopVerifier Protocol
├── visual_verifier.py      # ORB descriptor match + PnP wrapper implementing LoopVerifier
├── map_output.py           # Trajectory writer + occupancy grid assembler
└── adapters/
    ├── __init__.py
    ├── orb_loop_proposer.py
    ├── lidar_loop_proposer.py
    ├── visual_frontend_service.py
    └── lidar_frontend_service.py
```

## 7. Mode-by-mode data flow

### 7.1 Mode A — `--mode orb` (Visual-only)

Pass-through. The unified runner calls into `visual_slam/orbslam/run_rgbd_slam.py` via subprocess or by importing its `main()`. No fusion code is exercised. This mode exists so the unified runner is the single entry point for all SLAM operations going forward.

### 7.2 Mode B — `--mode lidar` (LiDAR-only)

Pass-through. The unified runner calls into `hector/run_local_slam_new.py` (or `carto/run_loop_closure_slam.py`) selected by a sub-flag. No fusion code is exercised.

### 7.3 Mode C — `--mode vlmain` (Visual-main + LiDAR verifier)

```
loop step:
1. FusionDataset yields (rgb, depth, t_rgb) and an optional (scan, t_scan) within ±50 ms
2. VisualFrontendService.step(rgb, depth, t_rgb)
     -> ORB-SLAM tracking updates the visual map and may publish a new keyframe
3. If a new visual keyframe is published:
     a. Project keyframe pose SE(3) -> SE(2) using camera→base TF
     b. Construct Signature {pose_SE2, orb_kpts+desc, scan (if present), local_occ_grid}
     c. MemoryManager.insert(signature)   -> goes into STM
     d. FusionGraph.add_node(signature)
     e. FusionGraph.add_neighbor_link(prev_signature, signature, odom_rel_pose)
4. OrbLoopProposer.poll_candidates(signature)
     -> returns list of (candidate_signature_id, score) without verifying or adding constraints
5. For each candidate:
     ICPLoopVerifier.verify(query_scan = signature.scan,
                            target_scan = candidate_signature.scan,
                            initial_guess = relative pose between current SE(2) estimates)
     -> if accepted: emit LoopConstraint(target_id, transform, info_matrix)
6. FusionGraph applies LoopConstraints; G2oBackend2D.solve() runs.
7. Optimized poses are written back into signatures (in WM); LiDAR PGO is NOT invoked.
8. MemoryManager.tick() — rehearsal, STM aging, WM transfer if cap exceeded.
9. Map outputs updated at the configured cadence.
```

Key property: ORB-SLAM's own loop verification (Sim(3), local BA fold-in) is **not run** in this mode. The adapter installs a `propose_only=True` flag at fusion runner init that the proposer respects.

### 7.4 Mode D — `--mode lvmain` (LiDAR-main + Visual verifier)

```
loop step:
1. FusionDataset yields (rgb, depth, t_rgb) and (scan, t_scan)
2. LiDARFrontendService.step(scan, t_scan)
     -> LiDAR front-end (scan_to_submap or scan_to_map) updates and may publish a keyframe
3. If a new LiDAR keyframe is published:
     a. Pose is already SE(2)
     b. Construct Signature {pose_SE2, scan, local_occ_grid,
                             orb_kpts+desc (from the nearest synced RGB-D, if any)}
     c. MemoryManager.insert(signature)   -> goes into STM
     d. FusionGraph.add_node(signature) + neighbor link (odom from front-end)
4. LiDARLoopProposer.poll_candidates(signature)
     -> returns B&B + proximity candidates without verifying or adding constraints
5. For each candidate:
     VisualLoopVerifier.verify(query_desc = signature.orb_desc,
                               target_desc = candidate_signature.orb_desc,
                               initial_guess = relative pose between SE(2) estimates)
     -> ORB descriptor matching + PnP RANSAC
     -> if accepted: emit LoopConstraint
6-9 as above.
```

Key property: the LiDAR pipeline's B&B verification and its g2o backend are **not run** for graph updates — the fusion graph owns optimization.

## 8. Memory management design (the deep part)

This is the single most important subsystem after the unified graph. It directly implements the rules quoted in `RTAB_review.md` §3.

### 8.1 `Signature` class — `slam_core/fusion/signature.py`

```python
@dataclass
class Signature:
    id: int                          # Sequential unique ID
    map_id: int                      # Session ID (always 0 in v1; v2 multi-session)
    stamp: float                     # Timestamp
    weight: int                      # RTAB-style; init 0; grows via rehearsal & loop closure
    pose: Pose2                      # SE(2) pose, optimized by the unified graph
    # Visual payload (None if visual front-end not active)
    orb_kpts:  Optional[np.ndarray]      # (N, 2)
    orb_desc:  Optional[np.ndarray]      # (N, 32) ORB descriptors (uint8)
    orb_3d_pts: Optional[np.ndarray]     # (N, 3) for PnP backprojection
    # LiDAR payload (None if no scan in sync window)
    scan_xy:   Optional[np.ndarray]      # (M, 2) preprocessed scan in sensor frame
    local_grid: Optional[ProbabilityGrid] # small per-node grid (~5 m wide)
    # Graph connectivity
    links: list[Link]                 # using the Link concept already in carto/pose_graph
```

Compression: in v1 no compression; the v2 SQLite path will introduce zlib + cv2.imencode for the persisted payload. Note this in §16.

### 8.2 `MemoryManager` class — `slam_core/fusion/memory.py`

Three tiers, mirroring RTAB-Map's `_stMem`, `_workingMem`, and (in v1) an in-RAM `_ltm`:

```python
class MemoryManager:
    def __init__(self, stm_size=30, wm_cap=200, rehearsal_sim=0.2):
        self._signatures: dict[int, Signature] = {}   # all RAM-resident
        self._stm: deque[int] = deque(maxlen=stm_size)
        self._wm:  dict[int, float] = {}             # id -> last-access timestamp
        self._ltm: dict[int, Signature] = {}         # in-RAM in v1; SQLite in v2

    def insert(self, sig: Signature) -> None:
        # 1. rehearsal-merge against STM tail if similar
        # 2. push into STM; if STM full, oldest STM -> WM
        # 3. apply weight rule from RTAB §3.4
    def tick(self) -> None:
        # 1. if WM size > wm_cap: transfer oldest of lowest-weighted -> LTM
        # 2. update word references (none in v1; ORB descriptor refs in v2 if BoW added)
    def reactivate(self, ids: list[int]) -> list[Signature]:
        # LTM -> WM; pulls up to N neighbors from the graph
    def is_in_stm(self, id) -> bool: ...
    def is_in_wm(self, id) -> bool: ...
    def is_in_ltm(self, id) -> bool: ...
```

Rehearsal in v1 uses two configurable similarity metrics depending on what payload the signature has:
- If both signatures have ORB descriptors, similarity = (#matched_descriptors / min(N1,N2)) above `rehearsal_sim`
- Else (LiDAR-only signatures), similarity = ICP fit score above `rehearsal_sim` between scans

This matches RTAB-Map's intent (collapse near-stationary observations into one node whose weight grows) while staying multi-modal.

Weight rule (verbatim from RTAB §3.4): rehearsal merge → `w_t += w_c + 1`; loop closure confirmed → `w_t += w_i + 1`; transfer policy → oldest among lowest-weighted.

### 8.3 Defaults for Jetson Nano (configurable)

| Parameter | Default | Effect | Memory cost (rough) |
|---|---|---|---|
| `stm_size` | 30 | RTAB-Map default | 30 × ~2 MB = 60 MB |
| `wm_cap` | 200 | Bounded WM under fusion load | 200 × ~2 MB = 400 MB |
| `rehearsal_sim` | 0.2 | RTAB-Map default | — |
| `ltm_in_ram_cap` | 1000 | v1 safety cap; once exceeded, oldest LTM is **dropped** (logged warning) | 1000 × ~2 MB = ~2 GB worst case |

The user can shrink any of these via `--stm-size`, `--wm-cap`, `--ltm-cap` flags. The trade-off (smaller cap → faster but loses long-term recall) is documented in the runner's `--help`.

## 9. Shared graph design

### 9.1 `FusionGraph` class — `slam_core/fusion/graph.py`

A thin wrapper around `carto/pose_graph/backends/g2o_backend_2d.py`. It is **not** the same object as `PoseGraph2D` (which is submap-aware) — fusion uses keyframe-only graph semantics:

```python
class FusionGraph:
    def __init__(self, backend: G2oBackend2D, memory: MemoryManager): ...
    def add_node(self, sig: Signature) -> int:
        # Add an SE(2) vertex; first node is fixed
    def add_neighbor_link(self, sig_a: Signature, sig_b: Signature,
                          rel_pose: Pose2, info: np.ndarray) -> None:
        # spine edge between two consecutive keyframes (weight ~ 1e5)
    def add_loop_constraint(self, c: LoopConstraint) -> None:
        # cross-modal loop edge (weight ~ 1.1e4, Huber kernel via existing g2o backend)
    def solve(self, max_iters: int = 30) -> dict[int, Pose2]:
        # Run g2o; write optimized poses back into Signatures in WM
        # LTM signatures are NOT touched; corrected lazily on reactivate()
    def get_subgraph_for_optimization(self) -> set[int]:
        # WM IDs + any LTM IDs reactivated this iteration
```

Why not extend `PoseGraph2D` directly: `PoseGraph2D` carries submap vertices (`PoseGraphSubmap`) and INTRA/INTER constraint logic specific to the LiDAR pipeline. Fusion is keyframe-only and that logic doesn't apply. We reuse `G2oBackend2D` (the actual solver) but not `PoseGraph2D`.

### 9.2 Edge weights

Match the existing LiDAR PGO defaults so behavior is consistent when only LiDAR constraints are in play:

| Edge type | Translation weight | Rotation weight | Robust kernel |
|---|---|---|---|
| Neighbor (spine) | 1e5 | 1e5 | none |
| Cross-modal loop (ICP) | 1.1e4 | 1e5 | Huber, δ=10 |
| Cross-modal loop (visual PnP) | 1.1e4 | 1e5 | Huber, δ=10 |

The first node is fixed (`G2oBackend2D.set_fixed("submap", 0)` — repurposing the anchor mechanism that already exists).

## 10. ICP verifier design

### 10.1 `ICPLoopVerifier` — `slam_core/fusion/icp_verifier.py`

Implements the `LoopVerifier` Protocol from `slam_core/loop_closure.py`. Single point of contact with `small_gicp`:

```python
class ICPLoopVerifier:
    def __init__(self, max_correspondence_distance=0.5,
                 fitness_threshold=0.6,
                 inlier_rmse_threshold=0.10):
        self._backend = small_gicp  # imported at module load

    def verify(self, query: LoopNode, target: ClosureTarget) -> LoopMatchResult:
        # 1. get query.scan (Mx2) and target.scan (Nx2)
        # 2. compute initial guess = T_target_world^{-1} · T_query_world (SE2 -> 3D rigid)
        # 3. small_gicp.align(query, target, init=guess, downsampling_resolution=0.05)
        # 4. if fitness >= threshold and inlier_rmse < threshold:
        #      success, transform, info_matrix
        # 5. info matrix from inlier_rmse and overlap (MAD-style, matches RTAB-Map)
```

Note: `small_gicp` operates on 3D point clouds. We promote the 2D scan to 3D by appending `z=0`, run the alignment, and project the returned transform back to SE(2) by extracting (x, y, yaw) from the 4×4 result.

### 10.2 `VisualLoopVerifier` — `slam_core/fusion/visual_verifier.py`

Also implements `LoopVerifier`. Reuses ORB-SLAM's already-imported OpenCV stack:

```python
class VisualLoopVerifier:
    def __init__(self, nndr=0.7, min_inliers=15):
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def verify(self, query: LoopNode, target: ClosureTarget) -> LoopMatchResult:
        # 1. KNN-match query.orb_desc against target.orb_desc, ratio test
        # 2. PnP RANSAC: 2D query keypoints vs target.orb_3d_pts (in target frame)
        # 3. if inliers >= min_inliers:
        #      success, transform (SE(3) -> project to SE(2)), info_matrix
```

ORB-SLAM already has the PnP routine; this verifier wraps that call rather than reimplementing.

### 10.3 Both verifiers gate by `RGBD/OptimizeMaxError`-style rule

Following RTAB-Map's outlier rejection: after `FusionGraph.solve()`, if a freshly-added loop edge's residual exceeds `optimize_max_error_factor × sqrt(translation_variance)`, reject the loop edge and roll back the graph (`G2oBackend2D` keeps the pre-solve estimate). Default factor = 1.0.

## 11. Sensor synchronization and dataset

### 11.1 `SoftSync` — `slam_core/fusion/sync.py`

```python
class SoftSync:
    def __init__(self, tolerance_s: float = 0.050): ...
    def push_rgbd(self, t, rgb, depth): ...
    def push_scan(self, t, scan): ...
    def try_pop(self) -> Optional[FusedSample]:
        # FusedSample(t, rgb, depth, scan_or_None)
        # if no scan in window, scan = None (Mode C still proceeds; LiDAR-verify skipped)
```

Tolerance default ±50 ms. Configurable.

### 11.2 `lidar_synth.py` — synthetic 2D LiDAR from TUM depth

Approach: pick a horizontal row of the depth image (default: the row closest to the camera optical axis); for each column compute the 3D point via inverse projection; project to the (x, y) plane; treat as a 2D scan. Add Gaussian noise (default σ=2 cm) to match real LiDAR. Configurable beam count, angle range, range cap.

```python
def synthesize_2d_scan(depth: np.ndarray, K: np.ndarray,
                       row_index: Optional[int] = None,
                       num_beams: int = 360,
                       range_min: float = 0.3,
                       range_max: float = 10.0,
                       noise_sigma: float = 0.02) -> np.ndarray:
    # Returns (M, 2) scan points in sensor frame
```

### 11.3 `FusionDataset` — `slam_core/fusion/dataset.py`

Wraps `visual_slam/orbslam/io/tum_rgbd.py` (already loads TUM RGB-D) and the synthetic LiDAR. Yields tuples `(t, rgb, depth, scan)` that `SoftSync` consumes.

Test target: `tum_fr1_room` (long enough trajectory for loops, has ground truth).

## 12. Configuration

```python
@dataclass
class FusionConfig:
    # Mode
    mode: Mode  # ORB | LIDAR | VLMAIN | LVMAIN
    # Front-end choice for LiDAR side (Mode B / Mode C / Mode D)
    lidar_frontend: Literal["scan_to_submap", "scan_to_map"] = "scan_to_submap"
    # Sync
    sync_tolerance_s: float = 0.050
    # Memory tiers
    stm_size: int = 30
    wm_cap:   int = 200
    ltm_cap:  int = 1000
    rehearsal_similarity: float = 0.2
    # ICP verifier
    icp_max_corr: float = 0.5
    icp_fitness:  float = 0.6
    icp_inlier_rmse: float = 0.10
    # Visual verifier
    visual_nndr: float = 0.7
    visual_min_inliers: int = 15
    # Optimization
    optimize_every_n_keyframes: int = 30
    optimize_max_error_factor: float = 1.0
    # I/O
    dataset_path: str
    output_dir: str
```

The runner exposes one CLI flag per important knob; the rest stay at defaults.

## 13. Implementation phases (sequencing)

Each phase ends with a runnable artifact and a verification step. Stop and review before moving to the next.

### Phase 1 — Foundation (no fusion behaviour yet)
- Create `slam_core/fusion/` scaffolding (empty modules with stubs)
- Implement `Signature`, `Pose2` projection from `Pose3D`
- Implement `SoftSync` + unit tests
- Implement `lidar_synth.py` + a quick visualization of synthesized vs real LiDAR
- Implement `FusionDataset` for TUM fr1_room
- **Verify**: a small script `tools/probe_fusion_dataset.py` plays through TUM, prints sync rate, dumps a few synthetic scans alongside their RGB frames

### Phase 2 — Memory tier
- Implement `MemoryManager` with STM, WM, in-RAM LTM
- Rehearsal merge (ORB-descriptor similarity path AND ICP-fit similarity path)
- Weight rule + transfer policy
- **Verify**: unit tests for insert/tick/reactivate; integration test on a synthetic 500-keyframe stream that exercises all transitions

### Phase 3 — Fusion graph
- Implement `FusionGraph` wrapping `G2oBackend2D`
- Add neighbour links from a recorded keyframe stream (no fusion sources yet)
- Add a synthetic loop edge; verify the graph deforms correctly
- **Verify**: comparison against a known-good g2o output for a small toy graph

### Phase 4 — ICP verifier (Mode C dependency)
- Implement `ICPLoopVerifier` with small_gicp
- Build small_gicp on the dev machine (verify Python bindings load)
- **Verify**: a self-loop test (align a scan to itself with noise added) returns identity within tolerance; two real TUM-derived scans with known overlap align correctly

### Phase 5 — Visual verifier (Mode D dependency)
- Implement `VisualLoopVerifier` calling ORB-SLAM's PnP routine
- **Verify**: two ORB-feature sets from known-overlapping TUM frames return a transform within tolerance of ground truth

### Phase 6 — Adapters
- `orb_loop_proposer.py` + `lidar_loop_proposer.py` with `propose_only=True`
- `visual_frontend_service.py` + `lidar_frontend_service.py`
- **Verify**: each adapter, called independently, returns the expected proposal stream on a short run

### Phase 7 — Mode A and Mode B (pass-through)
- `runner.py` skeleton + mode dispatch
- Modes A and B just import-and-call existing runners
- **Verify**: outputs are byte-equal (or trajectory-equal within float noise) to running the existing runners directly

### Phase 8 — Mode C end-to-end
- Wire visual front-end service → MemoryManager + FusionGraph
- Wire OrbLoopProposer → ICPLoopVerifier → FusionGraph
- **Verify**: run on TUM fr1_room; ATE vs ground truth ≤ standalone ORB-SLAM baseline within 5%; loop count > 0

### Phase 9 — Mode D end-to-end
- Wire LiDAR front-end service → MemoryManager + FusionGraph
- Wire LidarLoopProposer → VisualLoopVerifier → FusionGraph
- **Verify**: same TUM dataset (with synthesized LiDAR); ATE comparable to LiDAR-standalone; loop count > 0

### Phase 10 — Map output
- Trajectory TUM writer
- Occupancy grid assembler from accumulated scans transformed by optimized poses
- **Verify**: visual inspection of the grid against ground truth; check that re-running after a late loop closure correctly redraws the grid

### Phase 11 — Hardening & docs
- Profile on dev machine, document expected Jetson runtime
- README inside `slam_core/fusion/` summarizing the modes and config
- Update root `CLAUDE.md` with a one-line pointer to this plan

## 14. Verification

### 14.1 Per-phase verification
Listed inline in §13.

### 14.2 End-to-end acceptance (after Phase 10)

| Check | How |
|---|---|
| Mode A trajectory matches standalone ORB-SLAM | ATE within float-noise of `run_rgbd_slam.py` output |
| Mode B trajectory matches standalone LiDAR | ATE within float-noise of `run_local_slam_new.py` output |
| Mode C accepts ≥1 cross-modal loop on fr1_room | Log shows ≥1 `kGlobalClosure`-equivalent emitted by ICP verifier |
| Mode D accepts ≥1 cross-modal loop on fr1_room | Log shows ≥1 emitted by Visual verifier |
| Mode C / D ATE no worse than Mode A baseline | Reported ATE table after each run |
| Memory tiers behave per spec | Diagnostic log shows STM size bounded, WM cap respected, LTM growth tracked |
| Live signature count ≤ WM cap + STM size + LTM cap | Periodic memory diagnostics print stays under documented total |
| Runtime per keyframe on dev machine ≤ 100 ms | Profile printed at end-of-run |

### 14.3 Jetson-target verification (deferred to a follow-up plan)

End-to-end runtime profile, memory headroom, real ROS-input integration. Not part of v1 acceptance — v1 is offline only.

## 15. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `small_gicp` Python bindings fail to build on Jetson Nano (ARM aarch64) | Medium | High — blocks Mode C verifier | Verify build on dev x86_64 in Phase 4; v2 fallback to pure NumPy point-to-point ICP if needed; this is acceptable because loop verification runs at ~1 Hz, not per scan |
| Disabling ORB-SLAM's loop verification cleanly without editing `loop_closing.py` is harder than expected | Medium | Medium — may force a small edit to the original file | Read `loop_closing.py` carefully in Phase 6; if a clean adapter hook isn't possible, add a single `propose_only` config flag in `loop_closing.py` (one-line gate around the verifier call) — minimal modification, preserves standalone behaviour by default |
| Synthesized 2D LiDAR quality from TUM depth slice is too noisy/clean to represent real LiDAR | Medium | Medium | Tunable noise model in `lidar_synth.py`; cross-check by recording a quick handheld run with real LiDAR + RGB-D before Phase 8 if available |
| Fusion graph's SE(2) projection of ORB-SLAM SE(3) poses loses too much info on slightly uneven floors | Low | Medium | Document assumption; in Phase 8 measure ATE delta vs Mode A baseline; if regression > 10%, plan an SE(3) backend in v2 |
| Memory tier overhead exceeds expectation on Jetson Nano | Medium | Medium | v1 acceptance is dev-machine only; Jetson profile is a follow-up plan; conservative defaults reduce risk |
| The four-mode runner becomes a maintenance burden | Low | Low | Modes A and B are thin pass-throughs (≤30 lines each); Modes C and D share ~80% of code via the `LoopVerifier` Protocol abstraction |

## 16. Future work (v2 / v3)

Not scoped here, but anchored to v1 decisions:

- **v2 — SQLite persistence**: Replace in-RAM LTM with a SQLite database. Add async write thread. Add multi-session resumption (each Signature already carries `map_id`).
- **v2 — ICP as LiDAR loop verifier**: Add an `IcpLidarLoopVerifier` alongside the existing B&B verifier in `carto/`, selectable via flag. Same `small_gicp` module is reused.
- **v2 — Compression of Signature payload**: zlib + cv2.imencode for RGB-D images and scans on transfer to LTM (reduces RAM and prepares for SQLite).
- **v3 — ROS integration on Jetson**: Replace `FusionDataset` with a ROS-message-driven source; `SoftSync` is reused unchanged.
- **v3 — Dense 3D point cloud / OctoMap output**: After Jetson-runtime headroom is established.
- **v3 — Visual BoW + Bayes filter**: If reusing ORB-SLAM's detector proves limiting (e.g., poor recall on visually monotonous corridors), build a separate RTAB-style BoW + Bayes filter (deferred from v1 by choice).

## 17. Files (v1 final inventory)

### To be created (in this order)

```
slam_core/fusion/__init__.py
slam_core/fusion/config.py
slam_core/fusion/signature.py
slam_core/fusion/sync.py
slam_core/fusion/lidar_synth.py
slam_core/fusion/dataset.py
slam_core/fusion/memory.py
slam_core/fusion/graph.py
slam_core/fusion/icp_verifier.py
slam_core/fusion/visual_verifier.py
slam_core/fusion/map_output.py
slam_core/fusion/adapters/__init__.py
slam_core/fusion/adapters/orb_loop_proposer.py
slam_core/fusion/adapters/lidar_loop_proposer.py
slam_core/fusion/adapters/visual_frontend_service.py
slam_core/fusion/adapters/lidar_frontend_service.py
slam_core/fusion/runner.py
tools/probe_fusion_dataset.py            # development probe (Phase 1 verify)
```

### To be touched (minimal, only if adapter alone can't avoid it)

| File | Modification | Reason |
|---|---|---|
| `visual_slam/orbslam/slam/loop_closing.py` | At most one config-gated branch around the verifier call (`if not self.propose_only: ...`); the gate defaults to `False`, preserving standalone behaviour | Only added if the adapter pattern cannot intercept cleanly |
| `carto/loop_closure_adapter.py` | Same pattern as above for the LiDAR loop proposer | Same reason |

### Not modified

All other existing files. The standalone visual and LiDAR runners are left exactly as they are.

---

## Appendix A — Cross-reference to `RTAB_review.md`

| `RTAB_review.md` section | What we implement in v1 |
|---|---|
| §3 Memory management | Full STM/WM/LTM in RAM (§8 here) |
| §4 Graph map | FusionGraph + Signature (§6, §8.1, §9 here) |
| §5 Sensor input | SoftSync + FusionDataset (§11 here) |
| §6 Signature creation | `Signature` class with ORB + scan + local grid (§8.1 here) |
| §7 Odometry sources | Front-end services emit per-keyframe relative pose; no new odometry written |
| §8 Loop closure detector | Mode C: ORB-SLAM detector reused. Mode D: LiDAR B&B reused. No new BoW. |
| §9 Retrieval mechanism | `MemoryManager.reactivate(ids)` in v1; no DB I/O |
| §10 Geometric verification | ICP (Mode C) + Visual PnP (Mode D) — §10 here |
| §11 Graph optimization | `G2oBackend2D` reused, SE(2) — §9 here |
| §12 Map regeneration | Trajectory + 2D occupancy grid only — §13 Phase 10 here |
| §13 Database persistence | **Deferred to v2** |

---

*Plan author: Claude. Inputs validated through four rounds of `AskUserQuestion`. No assumptions made beyond what the answers cover; every other implementation detail flagged in §15 or §16.*
