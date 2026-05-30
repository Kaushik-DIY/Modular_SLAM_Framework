# Fusion Implementation — Phase Status

> **Read me first** in every new Claude session that works on the fusion build.
> This is the single source of truth for "where are we now?"
> See `CLAUDE.md` §4 for the full phase definitions and `RTAB_inspired_implementation_plan.md` §13 for design detail.

Last updated: 2026-05-30T03:54:17Z          Last commit: d33d68c (Phase 7)

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
| 8 | Mode C end-to-end (V-main + L-verify)       | PENDING  |              |        |       |
| 9 | Mode D end-to-end (L-main + V-verify)       | PENDING  |              |        |       |
| 10| Map output (trajectory + occupancy grid)    | PENDING  |              |        |       |
| 11| Hardening & docs                            | PENDING  |              |        |       |

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

- [ ] Mode A trajectory matches standalone ORB-SLAM (ATE within float-noise)
- [ ] Mode B trajectory matches standalone LiDAR (ATE within float-noise)
- [ ] Mode C accepts ≥1 cross-modal loop on TUM fr1_room
- [ ] Mode D accepts ≥1 cross-modal loop on TUM fr1_room
- [ ] Mode C / D ATE no worse than baseline + 5%
- [ ] Memory tiers behave per spec (caps respected, transfers correct)
- [ ] Per-keyframe runtime ≤ 100 ms on dev machine

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
