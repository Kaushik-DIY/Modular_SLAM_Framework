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
