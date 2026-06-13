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
