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
