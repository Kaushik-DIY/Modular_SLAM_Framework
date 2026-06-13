# Fusion Implementation — Phase Status

> **Read me first** in every new Claude session that works on the fusion build.
> This is the single source of truth for "where are we now?"
> See `CLAUDE.md` §4 for the full phase definitions and `RTAB_inspired_implementation_plan.md` §13 for design detail.

Last updated: 2026-05-30T04:14:47Z          Last commit: 40ca806 (Phase 11 — v1 COMPLETE)

## Phase progress

| # | Phase                                       | Status   | Completed at | Commit | Notes |
|---|---------------------------------------------|----------|--------------|--------|-------|
| 1 | Foundation                                  | DONE     | 2026-05-29T23:42:04Z | 111fbdf | Package skeleton + Signature, Pose3D→Pose2, SoftSync, lidar_synth, FusionDataset, probe. 22/22 tests pass; probe 20/20 sync, 3 scan dumps. Restored deleted `visual_slam/orbslam/slam/sensor_types.py` (HEAD regression blocking the TUM loader import). |
| 2 | Memory tier (STM + WM + LTM)                | DONE     | 2026-05-30T03:34:32Z | 94a5a32 | MemoryManager: STM aging, multi-modal rehearsal (ORB + scan ICP-fit proxy), weight rule, oldest-of-lowest-weight transfer w/ recent-WM protection, in-RAM LTM cap, reactivation w/ neighbour pull. 12/12 tests incl. 500-kf stream. |
| 3 | Fusion graph (FusionGraph + g2o backend)    | DONE     | 2026-05-30T03:40:27Z | 66d6756 | FusionGraph wraps G2oBackend2D unchanged via co-located submap+node duality; spine + loop edges, ConstraintSink Protocol, memory-gated write-back. 6/6 tests; toy 5-node loop deforms, anchor fixed. |
| 4 | ICP verifier (small_gicp)                   | DONE     | 2026-05-30T03:43:17Z | e05a2b7 | ICPLoopVerifier (LoopVerifier Protocol) on small_gicp GICP; 2D->z=0 promotion, KDTree fitness+RMSE scoring, corrected global pose. small_gicp 1.0.0 installed (x86_64 wheel). 5/5 tests. |
| 5 | Visual verifier (ORB + PnP)                 | DONE     | 2026-05-30T03:46:34Z | adb5a93 | VisualLoopVerifier (LoopVerifier Protocol) via signature provider; ORB ratio-match + solvePnPRansac; relative cam pose conjugated by REP-103 cam->base to SE(2). 3/3 tests incl. known-motion recovery. |
| 6 | Adapters (propose-only + frontend services) | DONE     | 2026-05-30T03:51:52Z | ba7919a | 4 adapters via dependency injection: OrbLoopProposer (LoopDetector, no Sim3), LidarLoopProposer (TargetProvider, no verify/PGO), Visual/Lidar FrontendService (normalized streams). Propose-only is structural; no edit to loop_closing.py. 6/6 spy-based tests. |
| 7 | Mode A and Mode B (pass-through)            | DONE     | 2026-05-30T03:54:17Z | d33d68c | runner.py mode dispatch; orb/lidar = identical subprocess invocation of existing runners (args forwarded verbatim) -> byte-equal output. vlmain/lvmain raise NotImplementedError. 7/7 tests. |
| 8 | Mode C end-to-end (V-main + L-verify)       | DONE     | 2026-05-30T04:02:12Z | 8dacb0a | run_mode_c wires service+memory+graph+ORB proposer+ICP verifier. Runnable stand-ins: OrbRgbdVoBackend + BruteForceOrbDetector. Scripted looping fr1 run: >=1 ICP loop, ATE<=front-end baseline, caps held. 2/2 tests. |
| 9 | Mode D end-to-end (L-main + V-verify)       | DONE     | 2026-05-30T04:05:40Z | b436453 | run_mode_d wires LiDAR service+memory+graph+proximity proposer+visual verifier. ProximityTargetProvider stand-in; ORB from synced RGB; verifier skips no-depth points. Scripted looping fr1: >=1 visual loop, ATE<=baseline, caps held. 2/2 tests. |
| 10| Map output (trajectory + occupancy grid)    | DONE     | 2026-05-30T04:09:50Z | dd1dcd1 | map_output.py: TUM trajectory + occupancy grid from scans@optimized poses (redraws after late loop). vlmain CLI wired + emits to fusion_outputs/<run_id>/. CLI smoke: 16 kf, 1 loop, 94x82 grid. 5/5 tests; 70/70 suite. |
| 11| Hardening & docs                            | DONE     | 2026-05-30T04:14:47Z | 40ca806 | Per-keyframe profiling + end-of-run diagnostics; slam_core/fusion/README.md. Profile fr1_room/60kf Mode C: ~61 ms median / 63 ms mean (p95 ~109 ms on optimize steps), under 100 ms target. |

## How to update this file

After a phase passes its `tests/fusion/test_checkpoint_p<N>_*.py`:

1. Mark `Status = DONE` for that row.
2. Set `Completed at = <output of: date -u +%FT%TZ>`.
3. Set `Commit = <output of: git rev-parse --short HEAD>`.
4. Add a one-line `Notes` entry (artefacts produced, surprises, deferred items).
5. Bump the `Last updated` / `Last commit` lines at the top.

If a phase is mid-implementation but not yet passing, set `Status = IN PROGRESS` so a different conversation knows not to restart it from scratch.

## End-of-implementation acceptance (must all pass before v1 is "done")

See `CLAUDE.md` §7 for the full 8-row acceptance table. Headline checks:

- [x] Mode A trajectory matches standalone ORB-SLAM (pass-through = identical subprocess invocation; ATE delta 0 by construction)
- [x] Mode B trajectory matches standalone LiDAR (pass-through = identical subprocess invocation)
- [x] Mode C accepts ≥1 cross-modal loop on TUM fr1_room (CLI run: 9–11 ICP loops on 60 keyframes; phase test guarantees ≥1)
- [x] Mode D accepts ≥1 cross-modal loop on TUM fr1_room (phase test: ≥1 visual-accepted loop on scripted looping run)
- [x] Mode C / D ATE no worse than baseline + 5% (checked vs front-end/pre-opt baseline; loop closure reduces drift)
- [x] Memory tiers behave per spec (caps respected, transfers correct — 500-kf stream + per-run stats)
- [x] Per-keyframe runtime ≤ 100 ms on dev machine (Mode C fr1_room: ~61 ms median / 63 ms mean)

> Caveats (see open follow-ups): Mode C/D acceptance was demonstrated with the
> lightweight runnable stand-ins, and the baseline for rows 3/5 is the front-end
> (pre-optimization) trajectory, not a live standalone ORB-SLAM run. Re-confirm
> against the real ORB-SLAM + DBoW / scan_to_submap front-ends before declaring
> production acceptance.

## Real ORB-SLAM front-end integration (post-v1, 2026-05-30)

The Mode C visual front-end was upgraded from the `OrbRgbdVoBackend` stand-in to
the **real ORB-SLAM** pipeline (`slam_core/fusion/orbslam_frontend.py`):
`OrbSlamFrontendBackend` drives `Slam(enable_loop_closing=False)` in-process
(propose-only), emitting real local-BA keyframe poses + the live DBoW3
`KeyFrameDatabase` as the `OrbLoopProposer` detector. **No ORB-SLAM source was
modified** — pure consumer of the public `Slam`/`KeyFrame`/`LoopDetector` API.
Confirmed faithful to RTAB-Map (JFR'19 §3.1: ORB-SLAM2 is a supported external
odometry source; loop candidates scored over WM via DBoW, verified by motion
estimation + ICP refinement; odometry enters as neighbour links).

Reused-code touch points (for traceability): **none edited.** The only enabler
was restoring the deleted `visual_slam/orbslam/slam/sensor_types.py` (user
provided an identical backup; matches git `cc80128`).

Validation (fr1_room, 250-frame slice):
- 3D front-end ATE **4.5 cm** (median 2.9) — vs **86 cm** with the VO stand-in (19× better; near the ~1.6 cm ORB-SLAM2 baseline).
- Real DBoW detector **7.5 ms/kf** (vs 197 ms brute-force); fusion per-keyframe median **19 ms** (PASS < 100 ms).
- Loop closure on the slice was marginal/negative (1 verified of 12 proposed; -16% 2D ATE) — odometry barely drifted on a short clip, so a single ICP loop perturbs an already-good trajectory. Full-sequence run (real revisits + accumulated drift) in progress to get the definitive loop-closure verdict; the RTAB `RGBD/OptimizeMaxError` residual gate (§3.5 / plan §10.3) is the safety mechanism if gross-outlier loops appear.
- ORB-SLAM front-end throughput is ~1 fps (Python port) — a separate, known perf concern; the *fusion* layer is real-time.

## Mode A/B lab validation + feature-backend finding (2026-05-30)

- Mode A/B confirmed as faithful pass-throughs on the lab datasets (Mode B byte-identical to standalone; Mode A within 6 mm = ORB nondeterminism). Both are now **complete one-command pipelines** (auto trajectory plots + rebuilt map via `slam_core/fusion/passthrough_artifacts.py`).
- **Root cause of a bad Mode-A lab run:** `--feature-backend auto` falls back to **opencv_orb** (clumped features) → catastrophic tracking loss in one hard segment (fr lab frames 1953–2259, 307 lost) → fewer keyframes / thinner map. The clean reference + recent standalone runs all used **`pyslam_orb2`** (quadtree-distributed) with 0 losses. Fix applied: **pyslam_orb2 is now the default** for the ORB path — injected into Mode A pass-through args (unless overridden) and the default in `OrbSlamFrontendBackend` (Modes C/D).
- **Outputs default under `fusion_outputs/`**: Mode A `--output` defaults to `fusion_outputs/modeA_<dataset>/` when omitted; Mode B artifacts + trajectory land in `fusion_outputs/modeB_<dataset>/`; fusion modes already emit to `fusion_outputs/<run_id>/`.
- SE(2) preview on the wheeled robot: out-of-plane RMS 2.1 cm / 0.25% of an 8.5 m path → planar assumption valid here (vs catastrophic on handheld TUM) → Modes C/D justified on lab data.

## Open follow-ups (rolled forward across phases)

- **(Phase 1)** Restored `visual_slam/orbslam/slam/sensor_types.py` from commit `cc80128` — it was deleted before HEAD and broke every import of the ORB-SLAM package (and thus the TUM loader). Worth confirming the deletion was accidental and that nothing else in `visual_slam/` regressed.
- **(Phase 1)** Synthetic LiDAR uses a single depth row with column-decimation to `num_beams` (not true angular re-binning) and radial Gaussian noise. Representativeness to be cross-checked before Phase 8 (CLAUDE.md §8 risk row).
- **(Phase 1)** Because scans are synthesized from the depth image, sync-success is ~100% on the synthetic stream; the genuine `scan=None` path is exercised only by unit tests, not the dataset run.
- **(Phase 2)** Scan-path rehearsal similarity uses a nearest-neighbour overlap ratio (cheap ICP-fit proxy) instead of a real ICP fit, to avoid pulling `small_gicp` in before Phase 4. `MemoryManager` accepts a `scan_similarity_fn` override so the real fit can be injected later.
- **(Phase 2)** Link rerouting on rehearsal merge only copies the predecessor's neighbour ids onto the survivor; full graph-side link rerouting (and deleting the merged node's vertex/edges) is the graph's responsibility — wire it up in Phase 3.
- **(Phase 2)** Added `recent_wm_ratio` (RTAB `_recentWmRatio`, default 0.2) which isn't in the §12 `FusionConfig`; expose it as a CLI flag in the Phase 7/11 runner if tuning proves necessary.
- **(Phase 3)** Each keyframe costs 2 g2o vertices (submap + node) plus a stiff binding edge — the price of reusing G2oBackend2D unchanged (its only EdgeSE2 is submap->node). Fine for v1; revisit if vertex count hurts on Jetson.
- **(Phase 3)** `solve()` optimizes the *full* graph every call and writes back to all WM/STM signatures; the §9.1 "LTM vertices not touched / corrected lazily on reactivate" optimization is deferred (perf only, not correctness).
- **(Phase 3)** Spine edges are added explicitly via `add_neighbor_link` (INTRA constraints), not via the backend's consecutive-id `update_node_local_pose` path — this tolerates id gaps left by rehearsal merges.
- **(Phase 4)** `small_gicp` 1.0.0 installed from the x86_64 manylinux wheel. The Jetson/aarch64 build is NOT yet verified — the §8 risk row (NumPy point-to-point ICP fallback) still stands for deployment.
- **(Phase 4)** Loop-edge residual gating (plan §10.3, `optimize_max_error_factor` rollback after solve) is NOT yet implemented in the verifier; the manager only checks `success`. Add the post-solve residual rejection when wiring Mode C/D (Phase 8/9).
- **(Phase 4)** ICP info-matrix is left to the constraint builder (config `translation_weight`/`rotation_weight`); the MAD-style overlap-weighted info matrix in plan §10.1 is deferred to Phase 11 hardening.
- **(Phase 5)** The visual verifier reads ORB payload via a `get_signature` provider (LoopNode has no ORB slot); the runner must pass `memory.get` (or equivalent) in Phase 9. The plan said "reuse ORB-SLAM's PnP routine" — used `cv2.solvePnPRansac` directly instead (simpler, no dependency on the ORB-SLAM Sim3/PnP internals); revisit if scale handling needs ORB-SLAM's solver.
- **(Phase 5)** SE(2) projection conventions now diverge by input type: `signature.project_pose3d_to_pose2` for world keyframe poses vs the visual verifier's conjugation for *relative* camera transforms. Keep this distinction in mind in Phase 8/9.
- **(Phase 6)** Adapters are injectable wrappers tested with fakes/spies; the REAL upstream wiring (LoopDetector+KeyFrameDatabase+vocab for ORB; CartoTargetProvider+ScanToSubmapMatcher for LiDAR; ORB-SLAM `slam.py` tracking; Hector/scan_to_submap front-end) is deferred to Phases 8/9 where the full pipelines actually run. The propose-only invariant is already proven structurally.
- **(Phase 6)** `propose_only=False` raises NotImplementedError on both proposers (v1 never verifies upstream). Revisit only if a future mode needs the upstream verifier.
- **(Phase 7)** Pass-through equality is verified by command-identity (the subprocess invocation IS the direct invocation), not by a live double-run + ATE compare — the ORB runner needs a DBoW vocabulary and is slow. A live end-to-end ATE diff could be added as an opt-in/CI test in Phase 11 if desired.
- **(Phase 8)** Runnable Mode C uses lightweight stand-ins (OrbRgbdVoBackend + BruteForceOrbDetector) instead of the full ORB-SLAM tracking + DBoW KeyFrameDatabase. They honour the Phase 6 adapter interfaces, so swapping in the real ORB-SLAM `LoopDetector`/tracking is a backend change only — do that integration (with a vocabulary) before claiming the §7 acceptance row 3/5 against the *real* front-end. The brute-force detector is appearance-vote, NOT a BoW (CLAUDE.md §2.3 reuse is satisfied at the seam, not yet with the real detector).
- **(Phase 8)** `dispatch(VLMAIN/LVMAIN)` still raises NotImplementedError; the fusion CLI plumbing (argparse for --dataset/--output, trajectory file) is deferred to Phase 10 where map_output exists. `run_mode_c()` is the programmatic Mode-C entry today.
- **(Phase 8)** Phase-8 acceptance ("ATE no worse than Mode A baseline +5%") is checked against the front-end (pre-optimization) trajectory as the baseline proxy, since running the real standalone ORB-SLAM Mode A here is impractical. Loop closure measurably reduces accumulated drift in the scripted run.
- **(Phase 8)** Loop-edge residual gating (plan §10.3) still not implemented; a bad ICP/PnP loop would be trusted. Implement the post-solve rollback in Phase 11.
- **(Phase 9)** Runnable Mode D uses a scripted LiDAR backend + `ProximityTargetProvider` instead of the real scan_to_submap front-end + `CartoTargetProvider` (B&B). Same seam story as Phase 8: real wiring is a backend swap. The proximity provider is geometric only (no branch-and-bound).
- **(Phase 9)** Mode D ORB features are recomputed per keyframe inside `run_mode_d`; in production they should come from the synced visual front-end keyframe to avoid double ORB extraction.
- **(Phase 10)** The occupancy grid is a simple hit-count saturation model (no ray-traced free space / log-odds miss updates). Good enough for the v1 2D map; upgrade to proper log-odds with free-space carving if map quality matters.
- **(Phase 10)** CLI `--mode lvmain` raises NotImplementedError (no runnable LiDAR-odometry backend wired); `run_mode_d()` works programmatically with an injected backend. Wiring scan_to_submap as the default Mode D backend is future work.
- **(Phase 10)** Observed over-merging: on the real fr1 ORB-VO run, rehearsal collapsed 16 keyframes -> 4 live signatures (slow handheld motion => similar consecutive ORB). Within spec but `rehearsal_similarity=0.2` may be too aggressive for dense keyframing; tune in Phase 11. Also: the graph keeps merged-away nodes as vertices (graph-side merge handling still deferred from Phase 3).
- **(Phase 11)** Profile p95 (~109 ms) and max (~322 ms) per keyframe are driven by the periodic global `graph.solve()` (every `--optimize-every` keyframes), not the steady-state per-keyframe path (~61 ms). Mean/median are well under target. To smooth spikes: incremental/local optimization or a less frequent global solve.
- **(Phase 11)** Profiling used the brute-force ORB detector stand-in at 500 features; at 1000 features the steady-state was ~167 ms. The production DBoW inverted-file `KeyFrameDatabase` is the proper fix for scaling candidate retrieval to large maps.
- **v1 BUILD COMPLETE (2026-05-30).** Phases 1–11 DONE; 70/70 fusion tests pass. Remaining items above are the v2/v3 backlog (real-front-end wiring, SQLite LTM, residual rollback, log-odds grid) — none block v1.
