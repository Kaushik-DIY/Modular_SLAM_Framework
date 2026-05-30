# Fusion Implementation — Phase Status

> **Read me first** in every new Claude session that works on the fusion build.
> This is the single source of truth for "where are we now?"
> See `CLAUDE.md` §4 for the full phase definitions and `RTAB_inspired_implementation_plan.md` §13 for design detail.

Last updated: 2026-05-30T03:34:32Z          Last commit: 94a5a32 (Phase 2)

## Phase progress

| # | Phase                                       | Status   | Completed at | Commit | Notes |
|---|---------------------------------------------|----------|--------------|--------|-------|
| 1 | Foundation                                  | DONE     | 2026-05-29T23:42:04Z | 111fbdf | Package skeleton + Signature, Pose3D→Pose2, SoftSync, lidar_synth, FusionDataset, probe. 22/22 tests pass; probe 20/20 sync, 3 scan dumps. Restored deleted `visual_slam/orbslam/slam/sensor_types.py` (HEAD regression blocking the TUM loader import). |
| 2 | Memory tier (STM + WM + LTM)                | DONE     | 2026-05-30T03:34:32Z | 94a5a32 | MemoryManager: STM aging, multi-modal rehearsal (ORB + scan ICP-fit proxy), weight rule, oldest-of-lowest-weight transfer w/ recent-WM protection, in-RAM LTM cap, reactivation w/ neighbour pull. 12/12 tests incl. 500-kf stream. |
| 3 | Fusion graph (FusionGraph + g2o backend)    | PENDING  |              |        |       |
| 4 | ICP verifier (small_gicp)                   | PENDING  |              |        |       |
| 5 | Visual verifier (ORB + PnP)                 | PENDING  |              |        |       |
| 6 | Adapters (propose-only + frontend services) | PENDING  |              |        |       |
| 7 | Mode A and Mode B (pass-through)            | PENDING  |              |        |       |
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
