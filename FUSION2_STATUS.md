# Fusion v2 — C++-core RTAB-inspired shared map: status tracker

Plan: `~/.claude/plans/lazy-napping-snail.md` (approved 2026-06-10). Python v1
(`slam_core/fusion/`) untouched as reference. Read this file at session start.

**Cross-cutting gates every phase must report:** memory (RSS measurement), runtime
(per-op timing), end-result quality where applicable (trajectory + map vs references:
fixed Test-3 LiDAR map 183 loops; IMU-aided ORB 0-lost trajectory).

| Phase | Scope | Status | Completed | Commit | Notes |
|---|---|---|---|---|---|
| C0 | Module skeleton, build script, g2o static link + coexistence check | DONE | 2026-06-10 | — | 4/4 tests; fusion_core+g2o+slam_optimizer_core+cpp_slam_core coexist in one process; Pose2 + SE2 roundtrip OK |
| C1 | Signature (packed payloads, zero-copy views) + LtmStore interface + InRamLtmStore | DONE | 2026-06-10 | — | 7/7 tests; **memory gate 18.7×** (200 sigs × 2000 kpts: 21.2 MB C++ ≈ raw payload vs 395.6 MB Python-equivalent); zero-copy views lifetime-safe; copy-in semantics verified |
| C2 | MemoryManager (STM/WM/LTM, rehearsal, weight rule, transfer, reactivation) | DONE | 2026-06-10 | — | 8/8 tests; 500-sig stream: 50 rehearsal merges, STM=30 WM=200 LTM=220, exact accounting; auto scan-overlap similarity; loop-touch keeps matched ends resident |
| C3 | FusionGraph2D (native SE(2) g2o), batch write-back | DONE | 2026-06-10 | — | 4/4 tests; parity vs Python g2o ≤1e-6; **10k nodes / 10k edges / 97 loops optimize(20) = 358 ms**; needed GL+GLU link (g2o static libs built with draw actions) |
| C4 | Neighborhood retrieval (BFS/metric) + local log-odds grid assembly | DONE | 2026-06-10 | — | 4/4 tests; BFS+metric membership exact; LTM reactivation on retrieval; grid pixel-matches numpy reference (1e-5); GIL-released candidate_local_grid() one-call API |
| C5 | Correlative B&B + GN refine in C++ | DONE | 2026-06-10 | — | 4/4 tests; **score parity EXACT (0.493=0.493)**, pose ≤2cm/0.5°; self-loop recovery OK; **85× vs Python (26 ms vs 2225 ms)**, under the 50 ms/candidate C7 budget |
| C6 | slam_core/fusion2 orchestration; modes orb + lidar on lab_hybrid | DONE | 2026-06-11 | — | **lidar on lab_hybrid: 382 kf, 74 accepted loops (110 verified, p50 coarse 0.69), clean 2-room map, 0.12 GB RSS, 17 min** (`fusion2_outputs/lidar_20260611_021436`); orb on lab_hybrid_small: 2466 frames, 428 kf through shared map, tiers exact, map payload 25.9 MB |
| C7 | Modes orb_lidar + lidar_orb; --verifier icp option | DONE (big-run pending) | 2026-06-11 | — | **lidar_orb/small: 8 PnP loops @7 ms** (gate met, ≤50 ms); **icp/small: 30 loops @10.6 ms**; **orb_lidar/small: 49 DBoW proposals → 1 accepted B&B loop** (gate met; acceptance limited by unaided ORB rotation drift — levers: wider angular window, IMU-aided FE); lidar_orb on lab_hybrid running |
| C8 | Benchmarks, README, Jetson notes | DONE | 2026-06-11 | — | README at `slam_core/fusion2/README.md`; runs index `fusion2_outputs/INDEX.md`; final table below |

## C8 final benchmark table (dev machine, lab_hybrid = BIG two-room dataset)

| Configuration | Loops accepted | Verify ms (mean) | Peak RSS | Wall time | Map |
|---|---|---|---|---|---|
| fusion2 `lidar` (BIG) | **74** B&B | 52 | **0.12 GB** | 17 min | clean 2-room |
| fusion2 `lidar_orb` (BIG) | **26** PnP cross-modal | **7.5** | 0.29 GB | 17 min | clean 2-room |
| fusion2 `lidar --verifier icp` (small) | 30 ICP | 10.6 | 0.10 GB | 7 min | coherent room |
| fusion2 `orb_lidar` (small) | 1 B&B cross-modal (gate ≥1 met) | 160 | 5.6 GB* | 30 min | — |
| *reference:* fixed Test-3 (Python PGO, BIG) | 183 | n/a (stalled twice pre-fix) | ~0.13 GB | 34 min | clean 2-room |
| *reference:* standalone ORB Python (BIG) | 9–13 visual | n/a | **10.4–11.2 GB** | 52 min | drift/teleports |

*` orb`/`orb_lidar` RSS is dominated by the legacy Python ORB front-end's own map,
not the fusion core (fusion map payload: 22–32 MB). Migrating the ORB front-end's
data model into C++ is the headline v3 item.

### Jetson notes
- fusion_core builds with CMake + pybind11 + Eigen/OpenCV/g2o-static; no x86-only
  dependencies besides `-march=native` (set `-mcpu=native` or drop on aarch64).
- g2o static libs in third_party/g2opy must be rebuilt for aarch64; GL/GLU link
  required (or rebuild g2o with draw actions off).
- small_gicp aarch64 build still unverified (risk carried from v1 plan §15).

## V3 — native windowed C++ VO front-end (in progress 2026-06-11)

| Phase | Status | Notes |
|---|---|---|
| V3.1 VoFrontend C++ | DONE | 4/4 tests; ORB+grid-bucket extraction, 7-KF window, projection match, SE(3) GN, REINIT safety net; 14–19 ms/frame |
| V3.2 Integration + A/B gate | DONE | `--frontend {native,legacy}` + DBoW3-over-signatures AppearanceIndex (pydbow3 addFeatures/query_local_des); **A/B on small: 34 fps vs 1.4 (24×), 0.66 GB vs 5.6 (8.5×), 436 kf ≈ legacy 428** |
| V3.3 Acceptance tuning | DONE | strong-PnP (≥40 inliers) bypasses rel-sanity gate (bootstrap loops were all rejected at 50–79 inliers) → **orb: 10 loops small / 13 loops BIG (gate ≥5 MET)**; orb_lidar: ±45° no change, **±180° (DBoW vouches place, B&B finds rotation) → 22 cross-modal loops BIG (gate ≥10 MET)** at 25.9 fps overall |
| V3.4 Close-out | DONE | final table below; timing gate re-verified contention-free (18.4 ms) |
| V3.5 Local BA (map-quality fix) | DONE | **Root cause of smeared vision maps found**: the lean VO did motion-only tracking with NO bundle adjustment. The reference ORB-SLAM2 made its coherent map with loops+global-BA OFF, purely from **1361 local-BA runs**. Added a local sliding-window BA (Schur-complement, LM trust-region, oldest KF fixed, points co-observed across the window) + reinit patience (coast brief dropouts) + full per-KF point pool. See V3.5 notes below. |

## V3.5 — local bundle adjustment + reinit robustness (2026-06-11)

The smeared `orb`/`orb_lidar` maps were NOT featureless-zone / IMU-extrinsic limited.
Proof: the reference full ORB-SLAM2 (`visual_slam_outputs/lab_hybrid_cpp_full`,
`enable_loop_closing=false`, `enable_global_ba=false`, **`local_ba_completed=1361`**)
makes a coherent 76k-point map. Its `tracking_lost=362/6468` (5.6%) is no better than
ours — local BA, not loss rate, is what bounds drift. Our VO had **no BA at all** (only
motion-only pose GN against depth points frozen at creation pose).

**Fixes (`third_party/fusion_core/src/vo_frontend.{h,cpp}`):**
1. **Local sliding-window BA** — map points are now shared across keyframes (a later
   frame matching a point adds an observation). After each KF, jointly refine the window
   poses (oldest fixed as gauge) + co-observed points via **Schur complement + Levenberg-
   Marquardt with a cost-decrease guard**. Pure GN diverged (a weakly-constrained camera
   jumped 2.58 m in one step); LM keeps corrections at a healthy **median 7 mm / max 35 mm**.
   Active points capped (`local_ba_max_points=300`, most-observed kept) → bounded cost;
   BA overhead is now a few ms/frame (gated in `test_v31_vo`).
2. **Reinit patience** (`reinit_patience=3`) — brief tracking failures (blur/occlusion)
   COAST on the prediction and keep the window alive; hard depth-reinit only after N
   consecutive failures. Cut reinits **11.6%→6.6%** on small (now better than the
   reference's 5.6–8.5% loss rate). Non-destructive reinit (keep window) was tried and
   REVERTED — stale points poisoned matching (162→296 reinits).
3. **Full per-KF point pool restored** — every KF re-seeds ~150 fresh depth points (the
   original behaviour) so the tracking pool stays rich; cross-KF observations are attached
   separately for BA. (Only seeding unmatched keypoints had starved the pool: 13.7%→11.6%.)
4. **Spine uses BA-refined relative poses** — runner builds each spine edge from the two
   keyframes' poses at the SAME BA epoch (`VoResult.prev_Twc`), so refinements actually
   reach the trajectory instead of re-accumulating drift.

**Results on lab_hybrid (BIG), native VO:**
| Mode | Before (V3.4) | After (V3.5) |
|---|---|---|
| `orb` | 13 loops, 328 reinits, round smear-blob map | **20 loops, 177 reinits, structured two-room corridor**, 30 fps, 0.70 GB |
| `orb_lidar` | 22 loops, 328 reinits, dense swirl | **11 loops, 260 reinits, two visible room clusters**, 23.9 fps, 0.71 GB |

39/39 fusion2 tests pass. (V3.5 alone was NOT sufficient — see V3.6 below, which found
the remaining root causes by probe-driven debugging.)

## V3.6 — the orb/orb_lidar root causes, found and fixed (2026-06-11)

User correctly rejected the V3.5 maps. Systematic debugging (per-frame probe replay +
frame-convention audit + reference-config forensics) found FOUR defects; each is backed
by evidence and each fix was measured:

1. **PnP loop edges were rotated 90° from spine edges.** Node poses were projected with
   `CAMERA_GROUND_TRANSFORM` only → SE(2) heading = bearing of the camera *right* axis;
   PnP loop edges (`cam_rel_to_base_se2`, REP-103) express translation along the camera
   *forward* axis. Numerically proven: the same physical motion produced spine rel
   (0,+1) vs loop rel (+1,0). Every accepted loop **warped** the graph — more loops made
   orb maps worse, and the V3.3 "sanity gate rejects true loops" mystery was this bug
   (the gate compared translations across mismatched frames; the strong-PnP bypass then
   admitted the corrupted edges). Fix: project node poses with `base_T_cam=BASE_T_CAM`
   too (both native and legacy orb paths). Verified: spine rel == loop rel to 1e-9.
2. **Tracking died on fast turns, NOT in featureless areas.** Probe: REINIT frames have
   median **912 keypoints, 232 with depth<4 m** (only 32/260 reinits truly featureless;
   depth_max=4 starvation disproven — 0 frames with <15 close + ≥30 far). Failures come
   in 2-4 s bursts during turns: const-velocity prediction goes wrong → 7/15 px
   projection windows search the wrong place → 0 matches despite abundant features.
   The reference survives via pose-free `track_reference_keyframe`. Fix: brute-force
   descriptor matching (TH_LOW=50, ratio 0.75) + PnP RANSAC recovery in the C++ VO when
   projection matching fails. **Reinits 260 → 122** by itself.
3. **IMU was disabled outright** (V3.3) instead of scoped to dropouts. Per-frame IMU
   priors hurt healthy tracking (mount tilt), but during a dropout there is nothing to
   match — integrating real rotation beats coasting a stale velocity. Fix: IMU dead-
   reckoning ONLY while `n_inliers < reinit threshold`, with the two extrinsic unknowns
   learned online from data: yaw **sign** by correlating VO vs IMU yaw deltas during good
   tracking, and the **up axis** by PCA over keyframe positions (planar trajectory).
   The calibration converged on-run to `up=[0.140,-0.990,0.012]` — a real **~8° mount
   tilt**, which is exactly why the naive (0,-1,0) prior failed in V3.3.
4. **REINIT keyframes polluted graph + map at full trust.** Blind dead-reckoned spine
   edges carried the same 1e5 weight as tracked ones (loops could not bend them back),
   and their scans painted the occupancy. Fix: `blind_spine_weight=1e2` for REINIT spine
   edges, REINIT scans excluded from rendering, and optimize-on-accepted-loop so loop
   corrections snap blind segments immediately (cheap relocalization).

**Results on lab_hybrid (BIG):**
| Mode | V3.5 (rejected) | V3.6 |
|---|---|---|
| `orb` | 20 loops, 177 reinits, smear | **38 loops, 117 reinits, clean two-room map with straight walls, end≈start**, 32.6 fps, 0.72 GB |
| `orb_lidar` | 11 loops, 260 reinits, swirl | **43 cross-modal loops, 117 reinits, two-room map**, 31.9 fps, 0.72 GB |

Montage: `fusion2_outputs/orb_native_20260611_210119/before_after_montage.png`.
LiDAR-led modes untouched (their frame conventions were already consistent — which is
precisely why they always mapped cleanly while orb didn't).

## V3 final benchmark (lab_hybrid BIG, native front-end, dev machine)

| Mode | Loops | fps | Peak RSS | Wall time | vs legacy front-end |
|---|---|---|---|---|---|
| `orb` (native) | **13 PnP** | **34.5** | **0.71 GB** | 3.1 min | legacy: 1.4 fps, 5.6 GB, 52 min, map spaghetti |
| `orb_lidar` (native, ±180° B&B) | **22 cross-modal** | **25.9** | **0.70 GB** | 4.2 min | was 1 loop / 30 min on small with legacy FE |
| `lidar` (unchanged v2) | 74 B&B | sensor rate | 0.12 GB | 17 min* | — |
| `lidar_orb` (unchanged v2) | 26 PnP | sensor rate | 0.29 GB | 17 min* | — |

*LiDAR-led wall time is dominated by the Python scan_to_submap front-end (~240 ms/scan);
porting it natively is a v4 option (same recipe as the VO).

**Honest quality note:** vision-led maps (orb / orb_lidar) still render smeared despite
accepted loops — the VO's inter-loop yaw drift dominates; loop edges correct topology, not
every pose between loops. Clean maps come from the LiDAR-led modes (by design — this is the
cross-modal architecture's purpose). The documented path to better vision-led maps:
camera-IMU extrinsic calibration for a usable IMU prior, denser loop acceptance, and/or a
small sliding-window BA in the VO.

**V3 empirical findings:** (1) raw ImuFallback prior rotates about world-z = roll in the
camera world → REINIT storm (1680!); removed. (2) Correctly-conjugated up-axis yaw prior
ALSO hurts (both signs: 633/839 reinits vs 205 const-velocity) — the low camera mount tilts
the up-axis; proper IMU fusion needs the camera-IMU extrinsic (future). Constant velocity
is the best prior on this data. (3) Vision-only mode `orb` has a yaw-drift map ceiling on
the big dataset (13 loops accepted but smeared map) — by design the cross-modal modes are
the answer; orb-mode numbers still far better than legacy Test-1 spaghetti at 24× speed.

### v3 levers (ordered)
1. ORB front-end data model in C++ (kills the remaining 5 GB / 1–2 fps ceiling;
   IMU-aided flag already works on the shared Tracking orchestrator).
2. orb_lidar acceptance: wider angular window / IMU-aided FE prior (rejects
   cluster at p50 0.526 vs 0.55 gate).
3. SQLite LtmStore (interface ready) → bounded RAM forever + multi-session.
4. PrecompStack caching per neighborhood between optimize epochs (verify p95
   ~106 ms → target <50 ms on BIG).

## C6 diagnosis log (loop acceptance on lab_hybrid)
- First full lidar run: pipeline + tiers + RSS all healthy (382 kf, STM 30/WM 200/LTM 139,
  **0.12 GB RSS**, map payload 0.6 MB) but 111/111 verifications rejected, scores plateaued
  at ~0.48 (unknown-cell level).
- Probe on lab_hybrid_small (real-data self-verification) found TWO real-data defects the
  synthetic tests could not see: (1) grid `l_free=-0.4` eroded wall hits (823 occ vs 21k free
  cells) — fixed to hector-proven -0.1; (2) Signatures stored voxel-filtered ~200-pt scans →
  dotted walls at 0.05 m cells — fixed to store RAW scans (~560 valid beams) per plan §8.
- After fixes: true-pose score 0.724 vs negative 0.515 (clean separation); perturbed query
  recovers pose. Acceptance thresholds set to coarse ≥0.55, refined ≥0.60.
- Datasets: `lab_hybrid_small` sanity-checked clean (2466 RGB-D / 1581 LiDAR / 4935 IMU,
  synced) — used for fast validation; `lab_hybrid` (two rooms) is the loop-closure testbed.

## Open follow-ups
- SQLite LtmStore implementation (v3; interface designed in C1).
- Async WM→LTM transfer thread (C2 ships synchronous first).
- Per-candidate verify time 57-73 ms mean (budget 50 ms): cache PrecompStack per
  candidate-neighborhood between optimization epochs; trim grid margin.
- LTM members are excluded from proximity proposals (WM-only per RTAB); consider
  RTAB-style retrieval triggers from weak WM hypotheses in C7/C8.

## V4 — close-out: native LiDAR, full modularity, master config, thesis-grade results (2026-06-12)

Plan: `~/.claude/plans/lazy-napping-snail.md` (approved). All 5 user-mandated tasks complete.

| Phase | Status | Notes |
|---|---|---|
| V4.1 orb_lidar fix | DONE | Scan verifiers (B&B/ICP) gained the same rel-sanity gate as PnP (`scan_rel_sanity_m/rad`); `orb_bnb_window_th` 180°→45° (the wide window let rotationally-ambiguous rooms produce high-scoring WRONG rotations). orb_lidar BIG: clean two-room map, no star-knots. |
| V4.2 fused map rendering | DONE | `write_outputs` renders a fused log-odds occupancy grid (reuses C++ `assemble_local_grid`) + `map.npy`/`map_meta.json` for thesis re-styling; raw scatter kept as `scan_overlay.png`; `tools/rerender_fused_map.py` re-renders old runs. Crisp single-cell walls vs the old fuzzy bands. |
| V4.3 master config + runner | DONE | ALL tunables in `slam_core/fusion2/config.py` (grid l_occ/l_free, ICP thresholds, vo_depth_max, IMU toggles moved in); dataset LiDAR geometry read from `sensor_config.yaml`; root `run_fusion.py` entry point with mode×option validity checks; `make_lidar_frontend` factory (native_s2s/native_s2m/legacy_s2s/legacy_s2m). |
| V4.4 native C++ LiDAR FE | DONE | `third_party/fusion_core`: scan_match_common (refine generalized initial≠prior; test_c5 bit-identical), PoseExtrapolatorCV (<1e-12 parity), voxel filters (<1e-12 incl. row order), scan_to_submap (grids BIT-identical incl. rotation invariant; correlative ≤1e-9 vs BOTH Python paths; **E2E 100% of scans within 2mm/2mrad, 0.00mm final drift**), scan_to_map (grids bit-identical; closed-loop ULP wander documented). **~11-15 ms/scan vs ~280 ms Python (≈20×; sensor budget 104 ms)**. |
| V4.5 final 9-combo matrix | DONE | Table below; 2 issues found by per-test analysis, root-caused, fixed, re-run (per acceptance mandate). |
| V4.6 docs | DONE | This section; INDEX/README/memory updated. |

### V4.7 — standard map orientation (all modes)

The VO front-end starts at SE(2) heading 90° (the `BASE_T_CAM ∘ CAMERA_GROUND_TRANSFORM`
projection of camera identity), which rotated the orb maps 90° vs the LiDAR maps. Fixed in
`write_outputs::_anchor_poses`: every mode's trajectory + occupancy is re-framed into the
robot-START frame (first keyframe at origin facing +x) before rendering — the standard SLAM
convention. Pure rigid re-frame (loop results unchanged); all 9 maps now render identically
oriented (robot starts bottom-left facing +x; corridor extends along +x). Canonical run:
`fusion2_outputs/final_matrix_20260612_154851/`.

### V4.5 final matrix (lab_hybrid BIG, all IMU-assisted, run folder `fusion2_outputs/final_matrix_20260612_154851`)

| # | Combo | Front-end | Verifier | Loops (acc/prop) | End-start | Peak RSS | Wall |
|---|---|---|---|---|---|---|---|
| 1 | lidar + B&B | native scan_to_submap | B&B | 71/110 | **0.39 m** | 0.16 GB | 1.0 min |
| 2 | lidar + ICP | native scan_to_submap | GICP+grid-check | 34/110 | **0.31 m** | 0.16 GB | 0.9 min |
| 3 | lidar + B&B | native scan_to_map | B&B | 74/104 | **0.11 m** | 0.25 GB | 1.2 min |
| 4 | lidar + ICP | native scan_to_map | GICP+grid-check | 74/104 | **0.24 m** | 0.25 GB | 1.2 min |
| 5 | lidar_orb | native scan_to_submap | ORB+PnP | 26/110 | **0.47 m** | 0.37 GB | 1.0 min |
| 6 | lidar_orb | native scan_to_map | ORB+PnP | 25/103 | **0.11 m** | 0.41 GB | 1.3 min |
| 7 | orb_lidar | native VO | B&B | 42/53 | 1.16 m | 0.91 GB | 3.5 min (32 fps) |
| 8 | orb_lidar | native VO | B&B-seeded GICP | 42/53 | 1.43 m | 0.95 GB | 3.4 min (32 fps) |
| 9 | orb | native VO | ORB+PnP (DBoW) | 38/53 | 1.81 m | 0.98 GB | 3.4 min (33 fps) |

All 9: two-room maps (LiDAR-led crisp; vision-led correct topology), tiers bounded 30/200,
RSS < 1 GB, faster than sensor rate. Montages + REPORT.md in the run folder.

### V4.5 issues found by per-test analysis (the fix-rerun loop in action)
1. **orb_lidar_icp closed 7.95 m off (FAIL)** → two stacked root causes: (a) nearest-neighbour
   ICP *fitness* is blind to corridor slide-locks (0.98-fitness loops at longitudinally wrong
   poses that B&B rightly rejected) → added a **grid cross-check**: the ICP-corrected pose must
   also pass the candidate-local occupancy-grid score gate (same gate as B&B). (b) With the
   cross-check, orb-led ICP accepted ZERO loops — prediction-seeded GICP cannot converge when
   VO drift exceeds `icp_max_corr_dist` (1 m) → **B&B-seeded GICP** for orb-led mode (B&B
   global coarse init within a few cm, robust to drift → GICP metric refine within its basin
   → grid cross-check gate). Result: 42 loops, 1.43 m closure.
2. **lidar_s2s_icp at 1.30 m** (marginal) → same grid cross-check filtered its sloppy edges:
   34 clean loops, **0.31 m**.

### Honest notes for the thesis
- **orb (vision-only) closes at 1.81 m**: DBoW proposed NOTHING in the final 37 keyframes
  (appearance matching is viewpoint-dependent; the final approach views the room from a new
  angle), so the tail after the last loop (KF 608) drifts freely. This is the inherent
  vision-only-proposer ceiling — and precisely the argument for the cross-modal modes
  (proximity proposers close the same gap at 0.11-0.47 m).
- **scan_to_map vs scan_to_submap**: s2m gives the tightest closures (0.11 m) on this dataset
  but logs 75 low-score fallbacks (score gate 0.10 with IMU-informed prediction covering);
  s2s matches every scan (1 fallback). Both are valid; s2s is the default.
- **native-vs-legacy s2m trajectories** slowly diverge (mm→cm over hundreds of scans) from
  1-ULP float32-sigmoid differences amplified by the closed map-feedback loop; all operators
  are unit-proven verbatim (grids bit-identical). s2s is exact end-to-end.

**FUSION LAYER IMPLEMENTATION COMPLETE.** Remaining work = fine-tuning only.
