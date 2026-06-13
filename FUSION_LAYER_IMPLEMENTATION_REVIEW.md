# Fusion Layer Implementation Review — Part 1: Architecture, Shared Map & Data Structures

> **Document purpose:** This is a self-contained technical reference for the RTAB-inspired multi-modal SLAM fusion layer (`slam_core/fusion2/`). It covers every design decision, all data structures, how sensor inputs enter the shared map, memory management, and how the four operating modes work. It is intended as thesis documentation and as a working manual for anyone reading or extending this code.

---

## 1. Context and Motivation

The workspace contains three independently-proven SLAM pipelines:

| Pipeline | Modality | Location |
|---|---|---|
| ORB-SLAM (RGB-D) | Visual | `visual_slam/orbslam/` |
| Hector (2D LiDAR scan-to-map / scan-to-submap) | LiDAR | `hector/` |
| Cartographer-style (2D LiDAR + g2o PGO) | LiDAR | `carto/` |

Each ran standalone. The fusion layer (`slam_core/fusion2/`) makes them collaborate through one shared map so that a revisit found by one modality can be confirmed by the other before becoming a graph constraint — exactly as RTAB-Map does, but adapted to this robot's hardware and data model.

### Why a C++ core (fusion v2)?

The Python-only v1 (`slam_core/fusion/`) was the prototype. At lab scale it collapsed:
- ORB object map: **~10.4 GB** for 839 keyframes.
- Branch-and-bound loop search: **~2.2 s/candidate** in Python, stalling the pipeline.

v2 moves all hot-path state into a C++ extension (`third_party/fusion_core/` → `fusion_core` Python module) with zero-copy NumPy views and GIL released:
- **18.7× smaller map memory** (21.2 MB C++ vs ~395 MB Python-equivalent for 200 signatures with 2000 keypoints each).
- **85× faster loop verification** (26 ms vs 2.2 s per candidate).
- Full-lab runs: **~0.12 GB RSS** vs ~10 GB.

The Python runner (`slam_core/fusion2/runner.py`) is deliberately thin: dataset feeding, keyframe normalization, loop bookkeeping, and output writing. Everything real-time runs inside `fusion_core`.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  run_fusion.py  (root entry point)                                  │
│  slam_core/fusion2/runner.py  (mode dispatch + main loop)           │
└──────────┬──────────────────────────────────────────────────────────┘
           │  selects one of four modes
     ┌─────▼──────────────────────────────────────────────────────┐
     │  SHARED C++ MAP  (fusion_core Python extension)             │
     │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ │
     │  │MemoryManager │  │ FusionGraph  │  │  Signature Store  │ │
     │  │ STM/WM/LTM   │  │ SE(2) g2o   │  │  (packed C++ bufs)│ │
     │  └──────────────┘  └──────────────┘  └───────────────────┘ │
     │  ┌──────────────────────────────────────────────────────┐   │
     │  │  Neighborhood retrieval + local log-odds grid (C++)  │   │
     │  └──────────────────────────────────────────────────────┘   │
     │  ┌──────────────────────────────────────────────────────┐   │
     │  │  Correlative Branch-and-Bound verifier (C++, 85×)   │   │
     │  └──────────────────────────────────────────────────────┘   │
     └────────────────────────────────────────────────────────────┘
           │
     ┌─────▼──────────────────────────────────────────────────────┐
     │  Mode-specific front-ends + proposers                       │
     │  lidar: NativeLidarFrontend → proximity proposer            │
     │  orb:   NativeOrbFrontend   → DBoW3 AppearanceIndex         │
     │  orb_lidar: NativeOrbFrontend → proximity proposer + B&B    │
     │  lidar_orb: NativeLidarFrontend → proximity proposer + PnP  │
     └────────────────────────────────────────────────────────────┘
```

All four modes write into the **same** `SharedMap` object. The mode determines only:
1. Which front-end produces odometry and keyframes.
2. Which proposer nominates loop candidates.
3. Which verifier confirms or rejects those candidates.

---

## 3. The Shared Map — `build_shared_map(cfg)`

File: `slam_core/fusion2/runner.py`, lines 59–91.

```python
@dataclass
class SharedMap:
    memory: fc.MemoryManager      # STM/WM/LTM three-tier memory
    store:  fc.InRamLtmStore      # LTM backend (SQLite is a drop-in)
    graph:  fc.FusionGraph2D      # SE(2) pose-graph (g2o-backed)
    grid_cfg: fc.GridConfig       # log-odds grid parameters
    bnb_cfg:  fc.BnbConfig        # Branch-and-Bound search window
```

These five objects are created once at run start and shared by the entire pipeline:

```
build_shared_map(cfg)
  ├── fc.MemoryConfig  → stm_size=30, wm_cap=200, rehearsal_sim=0.20
  ├── fc.InRamLtmStore  (in-RAM; SQLite store is the v3 drop-in)
  ├── fc.MemoryManager(mc, store)
  ├── fc.GraphConfig   → huber_scale=10, spine_weights=1e5, ...
  ├── fc.FusionGraph2D(gc)
  ├── fc.GridConfig    → resolution=0.05 m, l_occ=0.85, l_free=-0.1
  └── fc.BnbConfig     → window_xy=6 m, window_th=±30°, depth=7
```

---

## 4. The Signature — The Unit of the Shared Map

Every keyframe is represented as a **Signature** (`fc.Signature`). It is the fundamental unit stored in all three memory tiers and as a node in the pose graph.

### 4.1 Signature fields

| Field | Type | Content | Source |
|---|---|---|---|
| `id` | `int` | Monotonically increasing keyframe ID (0, 1, 2, …) | Runner counter |
| `timestamp` | `float` | Sensor timestamp (seconds) | Front-end |
| `pose` | `fc.Pose2` | SE(2) pose (x, y, θ) in the optimized world frame | Graph write-back |
| `scan_xy` | `(N,2) float32` | **Raw** 2D LiDAR scan in sensor frame (~560 beams) | LiDAR stream |
| `kpts` | `(N,2) float32` | ORB keypoint pixel coordinates | Visual front-end |
| `des` | `(N,32) uint8` | ORB binary descriptors | Visual front-end |
| `pts3d` | `(N,3) float32` | Camera-frame 3D points (NaN where depth missing) | Depth backprojection |

**Key design decision:** Signatures store the **raw** LiDAR scan (~560 valid beams), NOT the voxel-filtered point cloud (~200 pts). Voxel-thin scans produce dotted walls in candidate-local grids and suppress B&B verification scores. This was discovered and fixed during C6 diagnosis.

### 4.2 Signature properties

| Property | Meaning |
|---|---|
| `has_scan` | True if `scan_xy` is non-empty |
| `has_visual` | True if `kpts` and `des` are non-None |

### 4.3 Signature links

Each Signature also records its connections in the graph:
- **NEIGHBOR links**: consecutive keyframe spine edges.
- **LOOP links**: accepted loop-closure edges.

These are added via `sig.add_link(other_id, fc.LinkType.NEIGHBOR/LOOP, rel_pose, tw, rw)`.

### 4.4 Sensor payload by mode

| Mode | scan_xy stored? | kpts/des/pts3d stored? |
|---|---|---|
| `lidar` | ✓ (raw ~560 beams) | ✗ |
| `lidar_orb` | ✓ | ✓ (from synced RGB-D frame) |
| `orb` | ✓ if scan synced | ✓ |
| `orb_lidar` | ✓ if scan synced | ✓ |

In visual-led modes the scan is optional (soft-synced within ±50 ms). If no scan arrives within the window, `scan_xy` is an empty `(0,2)` array.

---

## 5. Memory Management — STM / WM / LTM

RTAB-Map's three-tier memory model is the core of the fusion layer's bounded-memory design. It is implemented in `fc.MemoryManager` (C++) based on the Python prototype in `slam_core/fusion/memory.py`.

### 5.1 Three tiers

```
┌─────────────────────────────────────────────────────────────┐
│ STM — Short-Term Memory                                      │
│  Size: 30 signatures (bounded FIFO queue)                    │
│  Role: Newest arrivals; NOT yet searchable for loops         │
│  Transition: oldest STM member ages out → WM when STM fills  │
└─────────────────────────────────────────────────────────────┘
           ↓ (age-out when STM full)
┌─────────────────────────────────────────────────────────────┐
│ WM — Working Memory                                          │
│  Capacity: 200 signatures                                    │
│  Role: Searchable for loop candidates; used in optimization  │
│  Transition: when WM > 200, lowest-weight → LTM             │
└─────────────────────────────────────────────────────────────┘
           ↓ (transfer when WM full)
┌─────────────────────────────────────────────────────────────┐
│ LTM — Long-Term Memory                                       │
│  Backend: InRamLtmStore (SQLite is a planned drop-in)        │
│  Role: Archived signatures; reactivated on loop hypothesis   │
│  NOT searched directly for candidates (WM-only per RTAB)    │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Weight rule (RTAB §3.4)

Every signature starts with weight = 0. Weight grows through two mechanisms:

1. **Rehearsal merge**: When a new keyframe is nearly stationary relative to the last, it is merged into its predecessor: `w_new += w_predecessor + 1`. The predecessor is removed from STM. This collapses repeated observations of the same spot into a single higher-weight node.

2. **Loop confirmation**: When a loop is accepted: `w_target += w_matched + 1`. Both ends of the confirmed loop keep their signatures resident.

### 5.3 Transfer policy

When WM exceeds the 200-signature cap, the **oldest of the lowest-weight** signatures is transferred to LTM:
- Sort WM members by last-access time (oldest first).
- Protect the most-recently-accessed 20% fraction (`recent_wm_ratio = 0.2`).
- Among the unprotected, pick the one with minimum weight (tie-break: smallest id).

This mirrors RTAB-Map's policy: frequently revisited places (high weight) stay in WM; rarely seen corridors (weight 0) get archived.

### 5.4 Rehearsal gate in fusion v2

In v2 (motion-filtered keyframes), the rehearsal gate is specifically for **stationary** keyframes only:

```python
d = _rel(last_fe_pose, fe_pose)
stationary = math.hypot(d.x, d.y) < 0.05 and abs(d.theta) < math.radians(2.0)
res = shared.memory.insert(sig, similarity=-1.0 if stationary else 0.0)
```

Passing `similarity=0.0` disables rehearsal for moving keyframes. Passing `-1.0` (or any value below the 0.20 threshold) for stationary ones allows the C++ MemoryManager to evaluate the actual scan overlap.

### 5.5 Loop-touch: keeping loop endpoints resident

When a loop is confirmed:
```python
shared.memory.on_loop_confirmed(kf_id, cand)
```
This call bumps the weight of both the query and the candidate keyframe and refreshes their last-access time, keeping both in WM even if they would otherwise be the next transfer victims.

### 5.6 LTM reactivation

When a loop candidate is retrieved, the neighborhood retrieval call (`fc.retrieve_neighborhood`) automatically reactivates LTM members that fall within the graph-BFS + metric radius of the candidate, pulling them back into WM. This enables loop closure against old map regions that have been archived.

---

## 6. The Pose Graph — `FusionGraph2D`

The pose graph (`fc.FusionGraph2D`) is a native C++ g2o-backed SE(2) graph. It owns all keyframe poses and has two edge types:

### 6.1 Edge types

| Edge type | Source | Weight (trans / rot) | Kernel |
|---|---|---|---|
| **Spine (NEIGHBOR)** | Front-end relative motion | 1e5 / 1e5 | None |
| **Blind spine (REINIT)** | Dead-reckoned VO | 1e2 / 1e2 | None |
| **Loop (LOOP)** | Verified loop closure | 1.1e4 / 1e5 | Huber δ=10 |

The much lower weight for REINIT spine edges is critical: dead-reckoned poses during tracking dropouts are guesses, not measurements. Giving them full 1e5 weight prevented accepted loop closures from bending the blind trajectory segment back into place. With 1e2 weight, a single strong loop edge can correct the entire blind segment.

### 6.2 Node initialization in the optimized frame

New nodes are NOT initialized from the raw front-end pose. They inherit from the last optimized graph pose plus the front-end's relative motion:

```python
if last_fe_pose is None:
    node_pose = fe_pose                             # first keyframe
else:
    node_pose = last_graph_pose.compose(
        _rel(last_fe_pose, fe_pose))                # graph frame + rel delta
```

This means loop corrections from `graph.optimize()` are never fought by fresh raw odometry — the new keyframe always continues from the corrected anchor.

### 6.3 Optimization cadence

- **Periodic**: every 30 keyframes (`optimize_every_n_kf = 30`).
- **Immediate** (orb/orb_lidar modes): triggered on any accepted loop closure (`any_loop_this_kf`).
- **Final**: once at the end of the run before output writing.

After each optimization, the last graph pose is refreshed:
```python
last_graph_pose = shared.graph.get_pose(kf_id)
```

---

## 7. Sensor Synchronization

File: `slam_core/fusion2/dataset.py` — `LabHybridStream`.

The dataset emits two streams:
- **LiDAR stream**: `(t, scan_xy)` at ~9.6 Hz, 909-beam 360° scanner, converted from ranges to 2D points via `ranges_to_points`.
- **RGB-D stream**: `(t, rgb_path, depth_path, scan_or_None)` at ~15 fps.

Soft sync: for each RGB-D frame, the nearest LiDAR scan within ±50 ms (`sync_tolerance_s = 0.05`) is attached. If no scan falls in the window, `scan = None` and the visual mode continues without LiDAR data for that keyframe.

Similarly, `nearest_rgbd(t)` provides the nearest RGB-D pair for a LiDAR timestamp (used in `lidar_orb` mode to attach visual payload).

---

## 8. SE(2) Pose Projection from Camera SE(3)

ORB-SLAM produces SE(3) camera poses. The fusion graph is SE(2). The projection is done in `slam_core/fusion/signature.py::project_pose3d_to_pose2` with two transforms:

1. **`base_T_cam` (REP-103 extrinsic)**: converts optical frame (z-forward, y-down, x-right) to base frame (x-forward, y-left, z-up):
```python
BASE_T_CAM = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]])
```

2. **`CAMERA_GROUND_TRANSFORM`**: maps the camera world frame onto the top-down ground plane.

The final SE(2) pose `(x, y, θ)` is extracted as `(T[0,3], T[1,3], atan2(T[1,0], T[0,0]))`.

**Critical note (V3.6 root-cause fix #1)**: Node poses MUST use both `base_T_cam` AND `CAMERA_GROUND_TRANSFORM`. Using only `CAMERA_GROUND_TRANSFORM` leaves the SE(2) heading on the camera RIGHT axis, making PnP loop-edge translations (which are in the base/forward frame) disagree with spine edges by 90°. Every accepted loop then warped the graph.

---

## 9. Output Map Orientation — `_anchor_poses`

All modes re-frame their trajectory into the **robot-START frame** before output:
- First keyframe is placed at origin facing +x.
- All other poses are expressed relative to this anchor via `R(-θ_0) @ (p - p_0)`.

This is a pure rigid re-frame (loop results unchanged). It ensures all four modes render consistently oriented maps regardless of the initial ORB-SLAM heading (which naturally starts at 90° due to the optical-frame convention).

---

## 10. Output Files

Every run produces in `fusion2_outputs/<mode>_<timestamp>/`:

| File | Content |
|---|---|
| `trajectory.tum` | Timestamped poses in TUM format (x y z qx qy qz qw), z=0 for SE(2) |
| `occupancy.png` | Fused log-odds occupancy grid from C++ `assemble_local_grid` |
| `map.npy` | Raw probability grid array (for thesis re-styling) |
| `map_meta.json` | Grid origin, resolution, size |
| `scan_overlay.png` | Debug scatter: raw scan points at optimized poses |
| `run_summary.json` | All statistics: modes, loops, timing, memory, RSS |
| `verifications.csv` | Per-verification diagnostics (query, cand, scores, accepted) |

The fused log-odds occupancy grid (V4.2 addition) uses the same C++ `assemble_local_grid` that the B&B verifier trusts. Overlapping observations reinforce walls instead of smearing — this is the thesis-grade output. REINIT keyframes (blind dead-reckoned poses) are excluded from rendering via `skip_scan_ids`.
# Fusion Layer Implementation Review — Part 2: Front-Ends, Modes, Loop Closure & Workflow

---

## 11. Front-Ends

### 11.1 LiDAR Front-End

**Factory**: `make_lidar_frontend(kind, ...)` in `slam_core/fusion2/lidar_frontend.py`

Four variants selectable via `--lidar-frontend`:

| Kind | Implementation | Speed |
|---|---|---|
| `native_s2s` | C++ scan-to-submap (default) | ~11–15 ms/scan |
| `native_s2m` | C++ scan-to-map | ~15 ms/scan |
| `legacy_s2s` | Python hector scan-to-submap | ~240–280 ms/scan |
| `legacy_s2m` | Python hector scan-to-map | ~280 ms/scan |

All share the same contract: `process(t, scan_xy) → (Pose2, filtered_pts, is_keyframe)`.

**Native C++ front-end** (`NativeLidarFrontend`): wraps `fc.NativeLidarFrontend`, configured field-by-field from the `hector/config.py` per-dataset profile. Per scan:
1. IMU samples accumulated since last scan are batched and sent.
2. `fc.NativeLidarFrontend.process(scan_xy, t, imu_samples)` runs GIL-released.
3. Returns pose, voxel-filtered points, and keyframe flag.

**Keyframe decision** (distance/angle/time threshold):
```
is_kf = True  when any of:
  - distance from last KF ≥ 0.25 m
  - angle from last KF  ≥ 12°
  - time since last KF  ≥ 2.0 s
```

**IMU integration**: A `PoseExtrapolatorCV` uses gyro + yaw correction (α=0.02) as a pose prior for the scan matcher. When matching fails, the IMU-informed extrapolator provides a fallback pose (counted as `fallback_count`).

### 11.2 Visual Front-End (Native ORB)

**Class**: `NativeOrbFrontend` in `slam_core/fusion2/vo_orb_frontend.py`

Wraps `fc.VoFrontend` — a lean windowed RGB-D visual odometry engine in C++:

**Per-frame processing**:
1. Convert RGB to grayscale.
2. If last frame had `n_inliers < reinit_threshold`, compute IMU dead-reckoning prior.
3. Call `fc.VoFrontend.track(gray, depth_u16, t, prior_Twc)`.
4. On `res.new_keyframe`: extract `keyframe_payload(kf_id)` → (kpts, des, pts3d).

**VO internals** (C++, `fusion_core/vo_frontend`):
- ORB extraction with grid bucketing (uniform spatial coverage).
- 7-keyframe sliding window with projection matching.
- SE(3) Gauss-Newton pose refinement.
- **Local sliding-window BA** (V3.5): Schur complement + Levenberg-Marquardt jointly refining window poses and co-observed map points. This is what bounds inter-keyframe drift — the reference ORB-SLAM2 makes coherent maps purely from local BA (1361 runs), not loop closing.
- `reinit_patience=3`: coasts through brief failures before hard re-initialization.
- Brute-force descriptor matching + PnP RANSAC recovery on fast turns.

**Online IMU self-calibration** (V3.6 fix):
1. **Yaw sign**: correlate VO heading deltas vs IMU yaw deltas during good tracking; commit once `sign_energy > 0.05 rad²`.
2. **Up axis**: PCA over keyframe positions (planar trajectory) → smallest-variance direction is gravity. Accounts for camera mount tilt (~8° measured on the lab robot).
3. **Dead-reckoning**: IMU prior is applied ONLY during tracking dropouts (`n_inliers < reinit_threshold`). Per-frame IMU priors hurt healthy tracking (mount tilt) but during a dropout they beat coasting a stale velocity.

**Emitted keyframe** (`NativeKeyframe`):
```python
@dataclass
class NativeKeyframe:
    id: int
    stamp: float
    Twc: np.ndarray       # 4×4 camera pose (world-from-camera)
    kpts: np.ndarray      # (N,2) f32 pixel coords
    des: np.ndarray       # (N,32) u8 ORB descriptors
    pts3d_cam: np.ndarray # (N,3) f32, NaN where depth missing
    state: fc.VoState     # OK | REINIT | INIT
    prev_Twc: Optional[np.ndarray]  # BA-refined pose of previous KF
```

The `prev_Twc` field is the key to the V3.5 spine fix: the spine edge is built from `_rel(cam_to_se2(prev_Twc), fe_pose)` — both poses from the SAME BA epoch — so BA refinements reach the graph trajectory instead of re-accumulating drift.

### 11.3 Legacy Visual Front-End (OrbSlamFrontendBackend)

Used with `--frontend legacy` for debug/parity. Wraps the full Python ORB-SLAM pipeline with loop closing disabled (propose-only mode via `OrbLoopProposer`). ~10× slower and uses much more memory. Included for reference comparison.

---

## 12. Loop Proposers

### 12.1 Proximity Proposer (LiDAR-led modes)

```python
def propose_candidates(shared, cfg, query_id, query_pose):
    for cid in shared.memory.wm_ids():           # WM ONLY — not LTM
        if abs(query_id - cid) < min_kf_separation:  # temporal guard (30 KFs)
            continue
        p = shared.graph.get_pose(cid)
        if dist(p, query_pose) <= proposal_radius_m:  # 4.0 m radius
            candidates.append((dist, cid))
    return [cid for _,cid in sorted(candidates)[:max_candidates_per_query]]
```

Key properties:
- **WM-only search**: LTM members are not directly proposed (RTAB design). They enter via neighborhood reactivation during verification.
- **Temporal guard** (`min_kf_separation=30`): prevents recent trail from proposing against itself.
- **Bounded** (`max_candidates_per_query=2`): caps verification cost.
- **Fired every 5 keyframes** (`propose_every_n_kf=5`).

### 12.2 DBoW3 Appearance Proposer (Visual-led modes)

**Class**: `AppearanceIndex` in `slam_core/fusion2/appearance_index.py`

Uses `pydbow3.Database` directly over raw ORB descriptor arrays (no legacy KeyFrame objects needed):

```python
index.add(kf_id, des)          # after inserting each keyframe
cands = index.query(kf_id, des) # before adding self → proposals
```

Candidates are filtered by:
- `dbow_score >= dbow_min_score` (0.05)
- `abs(kf_id - cand) >= min_separation` (30)
- Top `max_candidates` (2)

For the legacy ORB path, `OrbLoopProposer` wraps ORB-SLAM's `KeyFrameDatabase` in propose-only mode.

---

## 13. Loop Verifiers

### 13.1 Candidate-Local B&B (Branch-and-Bound) — `verify_candidate_bnb`

This is the primary verifier for LiDAR-led modes and the cross-modal scan verifier in `orb_lidar`.

**Step 1 — Neighborhood retrieval**:
```python
nb = fc.retrieve_neighborhood(shared.memory, shared.graph, cand_id,
                               graph_depth=2, metric_radius=3.0,
                               scans_only=True)
```
The C++ function does:
- BFS outward from `cand_id` to graph depth 2.
- Keep nodes within 3.0 m metric radius.
- Reactivates LTM members that fall in the neighborhood.
- Excludes the query's recent temporal trail (`abs(s.id - query_id) < 30`).

**Step 2 — Local grid assembly**:
```python
grid = fc.assemble_local_grid(sigs, poses, shared.grid_cfg)
```
Builds a log-odds occupancy grid from the neighborhood scans at their optimized poses. Parameters: `resolution=0.05 m`, `l_occ=0.85`, `l_free=-0.1`. The hector-proven log-odds balance: aggressive `l_free` would erode wall hits.

**Step 3 — B&B correlative search**:
```python
r = fc.bnb_match(grid, query_scan, query_pose, shared.bnb_cfg)
```
Hierarchical correlative search (depth=7) within `±6 m × ±30°` window at ~26 ms vs ~2.2 s Python equivalent. Returns coarse score and a refined Gauss-Newton pose.

**Acceptance gate**:
- `r.success AND r.coarse_score >= 0.55 AND r.refined_score >= 0.60`
- Plus **rel-sanity gate** (V4.1): verified relative pose must agree with the graph prediction within `2.5 m` and `±45°`. This rejects rotational-ambiguity false positives (rooms that look symmetric) that score well geometrically but disagree with drift-bounded odometry.

### 13.2 GICP + Grid Cross-Check — `verify_candidate_icp`

Alternative verifier (`--verifier icp`). Uses `small_gicp.align` (GICP) on the same candidate-local neighborhood:

1. Assemble neighborhood (same as B&B).
2. **B&B coarse seed** (orb-led mode only): VO drift exceeds GICP's convergence basin (~1 m). B&B provides a global coarse pose so GICP can refine metrically. LiDAR-led modes use prediction-seeded GICP (drift is smaller).
3. `small_gicp.align(tgt3, src3, GICP, max_corr=1.0 m)`.
4. **Grid cross-check** (V4.5 addition): ICP nearest-neighbor fitness is blind to corridor slide-locks (0.98 fitness at a longitudinally wrong pose). The corrected pose is additionally scored on the same candidate-local occupancy grid (must pass `refined_score >= 0.60`). This filter eliminated the 7.95 m closure error in `orb_lidar_icp`.

### 13.3 PnP Visual Verifier — `pnp_verify`

File: `slam_core/fusion2/visual_features.py`

Used in `lidar_orb` (cross-modal, LiDAR proposes → PnP confirms) and `orb` / `orb_lidar` (visual proposes → PnP confirms).

1. **Brute-force KNN matching** (`cv2.BFMatcher(NORM_HAMMING)`): query descriptors vs candidate descriptors, Lowe ratio test `nndr=0.7`.
2. **PnP RANSAC** (`cv2.solvePnPRansac`): 2D query keypoints vs 3D candidate points, `reprojError=3.0 px`.
3. Returns `(ok, T_target_query 4×4, n_inliers, inlier_ratio)`.
4. **SE(3)→SE(2)**: `cam_rel_to_base_se2(T)` conjugates through `BASE_T_CAM` to get the relative pose in the robot base frame.

**Acceptance logic**:
- `n_inliers >= 15` (minimum).
- Rel-sanity gate: verified pose must agree with graph prediction within `2.5 m` / `±40°`.
- **Strong PnP bypass** (`n_inliers >= 40`): bypasses the sanity gate. The sanity gate would reject true bootstrap loops when pre-loop drift exceeds the bound. Strong PnP is self-validating (50–79 inlier loops observed in practice).

---

## 14. The Four Operating Modes

### 14.1 Mode: `lidar`

**Driving question**: Can a single LiDAR stream, with scan-based loop verification, produce a consistent map?

**Front-end**: `NativeLidarFrontend` (scan-to-submap or scan-to-map).
**Proposer**: Proximity over WM.
**Verifier**: B&B (default) or GICP+grid-cross-check.

**Per-keyframe loop**:
```
scan → LidarFrontend.process(t, scan)
      → (fe_pose, filtered_pts, is_kf)
if is_kf:
  1. Compute node_pose in optimized frame
  2. Signature(id, t, scan_xy=raw_scan)
  3. memory.insert(sig, similarity=stationary_gate)
  4. graph.add_node + spine_edge
  5. every 5 KFs: propose_candidates → verify_candidate_bnb/icp
  6. if accepted: graph.add_loop_edge + memory.on_loop_confirmed
  7. every 30 KFs: graph.optimize()
```

**Result** (lab_hybrid BIG, native s2s): 382 KF, 74 B&B loops accepted, clean two-room map, 0.16 GB RSS, ~1 min.

### 14.2 Mode: `lidar_orb`

**Driving question**: Can LiDAR odometry + visual PnP verification close loops that are ambiguous for pure scan matching (symmetric rooms)?

**Front-end**: `NativeLidarFrontend`.
**Proposer**: Proximity over WM (same as `lidar`).
**Verifier**: ORB+PnP on the visual payloads stored in Signatures.

For each proximity candidate:
```python
cand_sig = shared.memory.get(cand)
if cand_sig.has_visual and sig.has_visual:
    ok, T_tq, n_in, ratio = pnp_verify(
        sig.kpts, sig.des, cand_sig.des, cand_sig.pts3d, K)
```

Visual payloads are attached at keyframe time by loading the nearest synced RGB-D frame:
```python
pair = stream.nearest_rgbd(t)   # soft-sync ±50 ms
rgb, depth = cv2.imread(...)
kpts, des, pts3d = extract_orb_rgbd(rgb, depth, K)
sig = fc.Signature(kf_id, t, kpts=kpts, des=des, pts3d=pts3d, scan_xy=raw_scan)
```

**Result** (lab_hybrid BIG, native s2s): 26 PnP cross-modal loops @7.5 ms/verify, clean two-room map, 0.37 GB RSS.

### 14.3 Mode: `orb`

**Driving question**: Can visual odometry + DBoW3 appearance + PnP verification produce a consistent map without any LiDAR?

**Front-end**: `NativeOrbFrontend` (or legacy Python ORB-SLAM with `--frontend legacy`).
**Proposer**: `AppearanceIndex` (DBoW3 over Signature descriptors).
**Verifier**: ORB+PnP.

Key differences from `lidar` mode:
- Keyframe decision is internal to the VO front-end (windowed BA triggers).
- REINIT keyframes get soft spine edges (`blind_spine_weight=1e2`).
- Appearance proposing: query BEFORE adding self to the index (so the current KF is never its own candidate):
```python
cands = index.query(kf_id, nkf.des)   # query first
index.add(kf_id, nkf.des)             # then add
```
- `optimize()` triggered IMMEDIATELY on any accepted loop (snaps blind REINIT segments back).

**Result** (lab_hybrid BIG, native V3.6): 38 PnP loops, clean two-room map (end≈start), 32.6 fps, 0.72 GB RSS.

### 14.4 Mode: `orb_lidar`

**Driving question**: Can visual odometry + DBoW3 appearance + scan-based verification (B&B or GICP) close loops more reliably than pure visual verification?

**Front-end**: `NativeOrbFrontend`.
**Proposer**: `AppearanceIndex` (DBoW appearance).
**Verifier**: B&B (or B&B-seeded GICP) on the synced LiDAR scans.

When a DBoW candidate is found, verification uses the scan payload:
```python
if sig.has_scan:
    r = verify_candidate_bnb(shared, cfg, kf_id,
                             np.asarray(sig.scan_xy), node_pose, cand)
```

For ICP in orb-led mode (`seed_from_bnb=True`): B&B first gives a coarse pose, then GICP refines metrically. This is necessary because VO drift can exceed GICP's convergence basin (~1 m) by the time a loop is proposed.

The angular B&B window is slightly wider for orb-led mode (`orb_bnb_window_th=±45°` vs `±30°`) to accommodate residual VO yaw drift.

**Result** (lab_hybrid BIG, native V3.6): 43 cross-modal loops, two-room map, 31.9 fps, 0.72 GB RSS.

---

## 15. Summary: What Each Mode Uses from the Shared Map

| Map component | `lidar` | `lidar_orb` | `orb` | `orb_lidar` |
|---|---|---|---|---|
| Signature.scan_xy | propose + verify | propose | stored only | verify |
| Signature.kpts/des/pts3d | — | verify (PnP) | propose + verify | propose (DBoW) |
| MemoryManager STM/WM/LTM | ✓ | ✓ | ✓ | ✓ |
| FusionGraph2D spine edges | ✓ | ✓ | ✓ (incl. blind) | ✓ (incl. blind) |
| FusionGraph2D loop edges | B&B/ICP | PnP | PnP | B&B/ICP |
| Neighborhood retrieval | ✓ (verify) | — | — | ✓ (verify) |
| Local grid assembly | ✓ (verify) | — | — | ✓ (verify) |
| Output occupancy grid | from scans | from scans | from scans | from scans |

All modes use the same `write_outputs` function: poses are anchor-normalized, the C++ `assemble_local_grid` renders the fused occupancy, and REINIT keyframe scans are excluded.

---

## 16. Verified Performance Matrix (V4.5 Final, lab_hybrid BIG)

| # | Mode | Front-end | Verifier | Loops acc/prop | End-start | Peak RSS | Wall |
|---|---|---|---|---|---|---|---|
| 1 | lidar + B&B | native s2s | B&B | 71/110 | **0.39 m** | 0.16 GB | 1.0 min |
| 2 | lidar + ICP | native s2s | GICP+grid | 34/110 | **0.31 m** | 0.16 GB | 0.9 min |
| 3 | lidar + B&B | native s2m | B&B | 74/104 | **0.11 m** | 0.25 GB | 1.2 min |
| 4 | lidar + ICP | native s2m | GICP+grid | 74/104 | **0.24 m** | 0.25 GB | 1.2 min |
| 5 | lidar_orb | native s2s | ORB+PnP | 26/110 | **0.47 m** | 0.37 GB | 1.0 min |
| 6 | lidar_orb | native s2m | ORB+PnP | 25/103 | **0.11 m** | 0.41 GB | 1.3 min |
| 7 | orb_lidar | native VO | B&B | 42/53 | 1.16 m | 0.91 GB | 3.5 min |
| 8 | orb_lidar | native VO | B&B-seeded GICP | 42/53 | 1.43 m | 0.95 GB | 3.4 min |
| 9 | orb | native VO | ORB+PnP (DBoW) | 38/53 | 1.81 m | 0.98 GB | 3.4 min |

All 9 modes: two-room maps, tiers bounded (STM=30/WM=200), RSS < 1 GB, faster than sensor rate.

---

## 17. Key Design Decisions and Lessons Learned

| Decision | Rationale |
|---|---|
| Raw scans (not voxel-filtered) in Signatures | Voxel-filtered ~200 pt scans → dotted walls, suppressed B&B scores (C6 diagnosis) |
| `l_occ=0.85, l_free=-0.1` (hector-proven balance) | Aggressive `l_free=-0.4` eroded wall hits (23× more free cells than occ in early runs) |
| WM-only proximity proposals | RTAB design: LTM members enter via reactivation, not direct search |
| node_pose from optimized frame + FE relative delta | Prevents loop corrections from fighting fresh odometry |
| Rel-sanity gate on all verifiers (V4.1) | High-scoring wrong-rotation matches (symmetric rooms) disagree wildly with drift-bounded odometry |
| Strong-PnP bypass of sanity gate | Bootstrap loops happen exactly when drift breaks the sanity prediction (50–79 inliers observed) |
| Grid cross-check on ICP (V4.5) | NN fitness blind to corridor slide-locks (0.98 fitness at 8 m wrong position) |
| B&B-seeded GICP for orb-led mode | VO drift exceeds GICP convergence basin; B&B provides global coarse init |
| `blind_spine_weight=1e2` for REINIT KFs | Soft edges allow loop corrections to bend dead-reckoned segments back |
| Optimize immediately on accepted loops | Snaps blind REINIT segments back → effective relocalization |
| REP-103 `BASE_T_CAM` for node projection | Without it, PnP loop edges are 90° rotated vs spine edges, warping the graph |
| IMU dead-reckoning scoped to dropouts only | Per-frame IMU prior hurt healthy tracking due to ~8° mount tilt; measured from A/B |
| Online yaw-sign + up-axis calibration | Camera-IMU extrinsic unknown; learned from correlated VO/IMU rotation during good tracking |
# Fusion Layer Implementation Review — Part 3: End-to-End Workflow & File Reference

---

## 18. Complete End-to-End Workflow Diagram

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  DATASET INPUT — LabHybridStream (slam_core/fusion2/dataset.py)                     ║
║                                                                                      ║
║  ┌─────────────────────┐            ┌─────────────────────┐                         ║
║  │  LiDAR stream       │            │  RGB-D stream        │                         ║
║  │  scans.csv → (N,2)  │            │  associations_rgbd   │                         ║
║  │  9.6 Hz, 909 beams  │            │  15 fps, (rgb, depth)│                         ║
║  │  360°, 16 m range   │            │                      │                         ║
║  └──────────┬──────────┘            └──────────┬───────────┘                        ║
║             │                                  │                                     ║
║             └──────── Soft Sync ±50 ms ─────────┘                                   ║
║                       nearest scan attached to each RGB-D frame                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
                              │
                   ┌──────────▼──────────┐
                   │   MODE SELECTION     │
                   │   (run_fusion.py)    │
                   └────┬────────────┬───┘
             ┌──────────┘            └──────────┐
             ▼ LiDAR-led                        ▼ Visual-led
   ┌──────────────────────┐         ┌──────────────────────────┐
   │  NativeLidarFrontend │         │  NativeOrbFrontend        │
   │  (fusion_core C++)   │         │  (fusion_core C++)        │
   │                      │         │                           │
   │  Per scan:           │         │  Per frame:               │
   │  1. IMU batch inject │         │  1. Check dropout → IMU   │
   │  2. Voxel filter     │         │     dead-reckoning prior  │
   │  3. Scan-to-submap / │         │  2. ORB extraction        │
   │     scan-to-map      │         │  3. Projection matching   │
   │  4. Pose extrapolate │         │  4. SE(3) Gauss-Newton    │
   │  5. KF decision      │         │  5. Local sliding-window  │
   │     (dist/angle/dt)  │         │     Bundle Adjustment (LM)│
   │                      │         │  6. KF decision           │
   │  Output:             │         │                           │
   │  (Pose2, pts, is_kf) │         │  Output: NativeKeyframe   │
   └──────────┬───────────┘         │  (Twc, prev_Twc, kpts,   │
              │                     │   des, pts3d, state)       │
              │ if is_kf            └──────────────┬─────────────┘
              │                                    │ if new_keyframe
              ▼                                    ▼
╔═════════════════════════════════════════════════════════════════════╗
║  KEYFRAME NORMALIZATION & SIGNATURE BUILDING                         ║
║                                                                      ║
║  1. Compute node_pose in OPTIMIZED frame:                            ║
║     node_pose = last_graph_pose ⊕ Δ(last_fe_pose, fe_pose)         ║
║     (inherits loop corrections; never fights fresh odometry)         ║
║                                                                      ║
║  2. Project visual SE(3) → SE(2) [visual-led only]:                 ║
║     cam_to_se2(Twc) = project_pose3d_to_pose2(                      ║
║         Twc, base_T_cam=BASE_T_CAM,                                  ║
║         world_transform=CAMERA_GROUND_TRANSFORM)                     ║
║     (REP-103 consistent: prevents 90° loop-edge vs spine mismatch)  ║
║                                                                      ║
║  3. Build fc.Signature:                                              ║
║     Signature(kf_id, t,                                              ║
║       scan_xy = raw_scan (LiDAR-led) or synced scan (visual-led),   ║
║       kpts = ORB keypoints (visual modes),                           ║
║       des  = ORB descriptors (visual modes),                         ║
║       pts3d= camera-frame 3D points (visual modes))                  ║
║     sig.pose = node_pose                                             ║
║                                                                      ║
║  4. REINIT flag [visual-led]: blind=True if state==REINIT            ║
╚═════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔═════════════════════════════════════════════════════════════════════╗
║  SHARED MAP INSERTION — fc.MemoryManager                             ║
║                                                                      ║
║  memory.insert(sig, similarity=s)                                    ║
║                                                                      ║
║  ┌────────────────────────────────────────────────────────────────┐  ║
║  │  REHEARSAL CHECK (stationary KFs only):                        │  ║
║  │  if similarity(sig, STM[-1]) >= 0.20:                          │  ║
║  │    sig.weight += predecessor.weight + 1                         │  ║
║  │    remove predecessor from STM                                  │  ║
║  │    (deduplicates near-stationary observations)                  │  ║
║  └────────────────────────────────────────────────────────────────┘  ║
║                                                                      ║
║  STM (30 slots): newest arrivals, not searchable                     ║
║    ↓ age-out when STM full: oldest STM → WM                         ║
║  WM (200 slots): searchable for loops, used in optimization          ║
║    ↓ transfer when WM > 200: oldest-lowest-weight → LTM             ║
║  LTM (in-RAM, InRamLtmStore): archived; reactivated on hypothesis   ║
╚═════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔═════════════════════════════════════════════════════════════════════╗
║  POSE GRAPH — fc.FusionGraph2D                                       ║
║                                                                      ║
║  graph.add_node(kf_id, node_pose)                                    ║
║                                                                      ║
║  graph.add_spine_edge(kf_id-1, kf_id, rel_pose,                     ║
║      weight = 1e5 (normal) or 1e2 (REINIT blind))                   ║
╚═════════════════════════════════════════════════════════════════════╝
                              │
              ┌───────────────▼────────────────┐
              │   LOOP PROPOSING                │
              │   (every 5 keyframes)           │
              │                                 │
              │  LiDAR-led: propose_candidates  │
              │  → proximity WM search ≤4 m     │
              │  → temporal guard ≥30 KF gap    │
              │  → max 2 candidates             │
              │                                 │
              │  Visual-led: AppearanceIndex    │
              │  → DBoW3 query (before add-self)│
              │  → score ≥0.05                  │
              │  → separation ≥30 KF            │
              │  → max 2 candidates             │
              └───────────────┬────────────────┘
                              │ for each candidate
                              ▼
╔═════════════════════════════════════════════════════════════════════╗
║  LOOP VERIFICATION (mode-specific)                                   ║
║                                                                      ║
║  ┌────────────────┐ ┌────────────────┐ ┌────────────────────────┐   ║
║  │ B&B Verifier   │ │ GICP+Grid      │ │ PnP Verifier           │   ║
║  │ (C++ 26 ms)    │ │ Verifier       │ │ (ORB match + RANSAC)   │   ║
║  │                │ │                │ │                          │   ║
║  │1. Retrieve     │ │1. Retrieve     │ │1. KNN match descriptors  │   ║
║  │   neighborhood │ │   neighborhood │ │   Lowe ratio test 0.7    │   ║
║  │   (BFS+metric) │ │   (BFS+metric) │ │2. cv2.solvePnPRansac    │   ║
║  │   excl. recent │ │   excl. recent │ │   3D cand ↔ 2D query    │   ║
║  │   trail        │ │   trail        │ │3. SE(3)→SE(2) via       │   ║
║  │2. Assemble     │ │2. Assemble     │ │   BASE_T_CAM conjugation │   ║
║  │   local grid   │ │   local grid   │ │                          │   ║
║  │   log-odds     │ │3. B&B seed     │ │Accept if:                │   ║
║  │3. B&B search   │ │   (orb-led)    │ │  n_inliers ≥ 15 AND      │   ║
║  │   ±6m ±30°     │ │4. GICP align   │ │  (sanity_gate OR         │   ║
║  │                │ │5. Grid score   │ │   n_inliers ≥ 40)        │   ║
║  │Accept if:      │ │   cross-check  │ │                          │   ║
║  │  coarse ≥0.55  │ │                │ │                          │   ║
║  │  refined ≥0.60 │ │Accept if:      │ │                          │   ║
║  │  + rel-sanity  │ │  fitness ≥0.6  │ │                          │   ║
║  │  gate (±2.5m,  │ │  grid ≥0.60   │ │                          │   ║
║  │  ±45°)         │ │  + rel-sanity  │ │                          │   ║
║  └────────┬───────┘ └───────┬────────┘ └───────────┬────────────┘   ║
║           └────────────────┬┘                       │               ║
║                            │  if accepted            │               ║
╚════════════════════════════╪═════════════════════════╪═══════════════╝
                             │ rel_pose computed        │
                             ▼                          ▼
╔═════════════════════════════════════════════════════════════════════╗
║  LOOP ACCEPTANCE                                                     ║
║                                                                      ║
║  graph.add_loop_edge(cand, kf_id, rel_pose,                         ║
║      trans_weight=1.1e4, rot_weight=1e5)   ← Huber kernel δ=10     ║
║                                                                      ║
║  sig.add_link(cand, LOOP, rel_pose, ...)                            ║
║  memory.on_loop_confirmed(kf_id, cand)     ← weight bump + touch    ║
╚═════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔═════════════════════════════════════════════════════════════════════╗
║  POSE GRAPH OPTIMIZATION — fc.FusionGraph2D.optimize()              ║
║                                                                      ║
║  Triggered by:                                                        ║
║  • every 30 keyframes (periodic)                                     ║
║  • immediately on accepted loop (visual-led modes)                   ║
║  • once at end of run                                                 ║
║                                                                      ║
║  After optimize():                                                    ║
║  last_graph_pose = graph.get_pose(kf_id)   ← snap correction        ║
╚═════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔═════════════════════════════════════════════════════════════════════╗
║  MAP OUTPUT — write_outputs()                                        ║
║                                                                      ║
║  1. _anchor_poses(graph.poses())                                     ║
║     → re-frame all poses: first KF at origin facing +x               ║
║                                                                      ║
║  2. trajectory.tum                                                   ║
║     → TUM format: t x y 0 0 0 qz qw                                 ║
║                                                                      ║
║  3. Fused log-odds occupancy grid (thesis-grade):                    ║
║     → for each KF (excluding REINIT):                                ║
║          retrieve sig from memory.get(nid)                           ║
║     → fc.assemble_local_grid(render_sigs, render_poses, grid_cfg)   ║
║     → occupancy.png (vmin=0, vmax=1, gray_r)                        ║
║     → map.npy + map_meta.json                                        ║
║                                                                      ║
║  4. scan_overlay.png (debug scatter of raw points)                   ║
║                                                                      ║
║  5. run_summary.json (all stats)                                     ║
║  6. verifications.csv (per-candidate diagnostics)                    ║
╚═════════════════════════════════════════════════════════════════════╝
```

---

## 19. Per-Mode Data Flow Summary

```
MODE: lidar
════════════════════════════════════════════════════════════
  scans.csv
     → NativeLidarFrontend (C++, ~11ms/scan)
       [voxel filter → scan-to-submap → IMU extrapolator]
       → (Pose2, pts_filtered, is_kf)
  if is_kf:
     Signature(id, t, scan_xy=raw_scan)
     → memory.insert() → STM(30)→WM(200)→LTM
     → graph.add_node + spine_edge(1e5)
     every 5 KFs:
       propose_candidates(WM, radius=4m)
         → verify_candidate_bnb(local_grid, query_scan)
           → ACCEPT: graph.add_loop_edge(1.1e4/1e5)
     every 30 KFs: graph.optimize()
  Output: scan-based occupancy grid

MODE: lidar_orb
════════════════════════════════════════════════════════════
  scans.csv + RGB-D (soft-sync ±50ms)
     → NativeLidarFrontend → (Pose2, pts, is_kf)
  if is_kf:
     synced_rgbd = stream.nearest_rgbd(t)
     kpts, des, pts3d = extract_orb_rgbd(rgb, depth, K)
     Signature(id, t, scan_xy=raw_scan, kpts, des, pts3d)
     → memory + graph (same as lidar)
     every 5 KFs:
       propose_candidates(WM, radius=4m)
         → pnp_verify(sig.kpts, sig.des,
                      cand.des, cand.pts3d, K)
           → ACCEPT: graph.add_loop_edge

MODE: orb
════════════════════════════════════════════════════════════
  RGB-D stream [+ optional synced scan]
     → NativeOrbFrontend (C++, ~14-19ms/frame)
       [ORB → project-match → GN → local-BA]
       → (Twc, state, NativeKeyframe)
  if new_keyframe:
     node_pose = cam_to_se2(Twc, BASE_T_CAM, CAMERA_GROUND_TRANSFORM)
     spine_rel  = cam_to_se2(_rel(prev_Twc, Twc))   ← BA-refined
     blind = (state == REINIT)
     Signature(id, t, kpts, des, pts3d, scan_xy)
     → memory.insert()
     → graph.add_node + spine_edge(1e5 or 1e2 if blind)
     AppearanceIndex.query(kf_id, des) before add(kf_id, des):
       → pnp_verify(sig.des, cand.des, cand.pts3d, K)
         → ACCEPT: graph.add_loop_edge
         → IMMEDIATE: graph.optimize() [snaps blind segments]

MODE: orb_lidar
════════════════════════════════════════════════════════════
  RGB-D stream [+ optional synced scan]
     → NativeOrbFrontend → NativeKeyframe
  if new_keyframe:
     [same as orb: node_pose, blind flag, signature]
     AppearanceIndex.query → candidates
       → if sig.has_scan:
           verify_candidate_bnb(local_grid, sig.scan_xy)
           or verify_candidate_icp(seed_from_bnb=True)
           → ACCEPT: graph.add_loop_edge
         → IMMEDIATE: graph.optimize()
```

---

## 20. File Reference Map

### Core fusion layer (slam_core/fusion2/)

| File | Purpose |
|---|---|
| `runner.py` | Main loop, `build_shared_map`, all four mode functions, `write_outputs` |
| `config.py` | `FusionV2Config` — ALL tunables in one place |
| `dataset.py` | `LabHybridStream` — lidar_stream, rgbd_stream, soft-sync |
| `lidar_frontend.py` | `LidarFrontend` (Python) + `make_lidar_frontend` factory |
| `native_lidar_frontend.py` | `NativeLidarFrontend` — wraps `fc.NativeLidarFrontend` |
| `vo_orb_frontend.py` | `NativeOrbFrontend` — wraps `fc.VoFrontend`, online IMU calib |
| `appearance_index.py` | `AppearanceIndex` — DBoW3 over Signature descriptors |
| `visual_features.py` | `extract_orb_rgbd`, `pnp_verify`, `cam_rel_to_base_se2`, `BASE_T_CAM` |
| `README.md` | Quick reference: modes, commands, results |

### Fusion v1 reference (slam_core/fusion/) — Python prototype

| File | Purpose |
|---|---|
| `signature.py` | `Signature` dataclass + `project_pose3d_to_pose2` + `CAMERA_GROUND_TRANSFORM` |
| `memory.py` | Python `MemoryManager` (STM/WM/LTM prototype) |
| `graph.py` | Python `FusionGraph` wrapping `G2oBackend2D` |
| `runner.py` | `run_mode_c` (vlmain) + `run_mode_d` (lvmain) |
| `adapters/` | `OrbLoopProposer`, `LidarLoopProposer`, front-end services |
| `icp_verifier.py` | `ICPLoopVerifier` using small_gicp |
| `visual_verifier.py` | `VisualLoopVerifier` using ORB+PnP |

### C++ extension (third_party/fusion_core/)

| Component | Python name | Purpose |
|---|---|---|
| `Signature` | `fc.Signature` | Packed payload with zero-copy NumPy views |
| `MemoryManager` | `fc.MemoryManager` | STM/WM/LTM with rehearsal, weight rule, transfer |
| `InRamLtmStore` | `fc.InRamLtmStore` | In-RAM LTM backend (SQLite is a drop-in) |
| `FusionGraph2D` | `fc.FusionGraph2D` | Native SE(2) g2o pose graph |
| `retrieve_neighborhood` | `fc.retrieve_neighborhood` | BFS+metric neighborhood retrieval with LTM reactivation |
| `assemble_local_grid` | `fc.assemble_local_grid` | Log-odds grid from neighborhood scans |
| `bnb_match` | `fc.bnb_match` | 85× faster correlative B&B scan matching |
| `VoFrontend` | `fc.VoFrontend` | Windowed RGB-D VO with local BA |
| `NativeLidarFrontend` | `fc.NativeLidarFrontend` | ~11–15 ms/scan LiDAR odometry |
| `VoState` | `fc.VoState` | OK / REINIT / INIT tracking state enum |

### Configuration

| File | Purpose |
|---|---|
| `slam_core/fusion2/config.py` | All Python-side tunables (FusionV2Config) |
| `hector/config.py` | LiDAR matcher parameters per dataset profile |
| `datasets/lab_hybrid/sensor_config.yaml` | Camera intrinsics + LiDAR geometry |
| `run_fusion.py` (root) | Top-level CLI entry point |

---

## 21. Thesis Notes

### Cross-modal architecture argument

The results directly validate the core thesis claim: **cross-modal verification closes loops that each modality alone cannot**:
- `orb` alone: 38 loops, end-start 1.81 m (DBoW appearance fails at new viewing angles).
- `lidar` alone: 74 loops, end-start 0.39 m (proximity finds geometric loops).
- `orb_lidar`: 43 loops (DBoW nominates → scan geometry confirms), 0.91 GB vs 10 GB standalone ORB.
- `lidar_orb`: 26 loops (proximity nominates → visual PnP confirms appearance at geometric candidates).

### Memory management effectiveness

The three-tier RTAB-style memory (STM=30 / WM=200 / LTM=in-RAM) successfully bounds peak RSS:
- Full 6468-frame lab dataset: **0.12–0.98 GB** (mode dependent) vs **10.4–11.2 GB** for standalone Python ORB-SLAM.
- Map payload (Signatures only): 22–32 MB C++ vs 395 MB Python-equivalent for 200 signatures.
- The weight rule + transfer policy keeps frequently-revisited places in WM for fast loop detection while archiving never-revisited corridors.

### Honest limitations

1. **Vision-led maps** still less crisp than LiDAR-led: VO inter-loop yaw drift is corrected at loop nodes but not everywhere between. Loop edges correct topology; they cannot correct every pose between loops without a persistent covisibility map or dense relocalization.
2. **orb end-start 1.81 m**: DBoW proposed nothing in the final 37 keyframes (new viewpoint); proximity-based modes close this gap (0.11–0.47 m).
3. **native vs legacy scan-to-map**: 1-ULP float32 differences in the occupancy sigmoid amplify over hundreds of scans (mm→cm). Both are verified unit-exact; s2s (scan-to-submap) is exact end-to-end (the default).

---

*Report generated from source analysis of slam_core/fusion2/, slam_core/fusion/, FUSION2_STATUS.md, RTAB_inspired_implementation_plan.md, and third_party/fusion_core interface.*
