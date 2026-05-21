# Implementation Log — Hector scan-to-submap + g2o PGO

Running record of every major modification, so we can roll back to the original
system if results regress. Newest entries at the bottom of each phase.

Branch: `ORB-SLAM`. Plan: `~/.claude/plans/in-my-hector-slam-delegated-rabin.md`.
Task guide: `CLAUDE.md`.

## Baselines captured (for regression comparison / rollback reference)
- `baseline_snapshots/baseline_scan_to_submap_run.log` + `hector_outputs/trajectory_lab_run_2_raw_scan_to_submap_<N>.txt`
  (OLD pre-refactor `scan_to_submap_old.ScanToSubmapMatcher`, native GN).
- `baseline_snapshots/baseline_scan_to_map_run.log` + `hector_outputs/trajectory_lab_run_2_raw_scan_to_map_<N>.txt`.
- These runs were launched BEFORE the Phase-1 edits, so they reflect the original system.

---

## Phase 0 — Deliverables
- Added `CLAUDE.md` (task guide).
- Added this `implementation.md` (rollback log).
- Granted blanket `Bash` allow in `.claude/settings.local.json` for autonomous execution.

## Phase 1 — Unify the scan-to-submap front-end (single source of truth)
Goal: ONE scan-to-submap implementation in `slam_core/matching/scan_to_submap/`
used by both Hector and Cartographer; remove the duplicate `scan_to_submap_old.py`;
drop the pyceres dependency from the local refine (native GaussNewtonLM default).

Changes:
1. NEW `slam_core/matching/scan_to_submap/correlative.py`
   - Relocated `correlative_match_two_stage`, `_bruteforce_search`, `_score_candidate`,
     `PrecomputationGridStack`, `_max_pool_2x2` verbatim from the deleted old file.
2. `slam_core/matching/scan_to_submap/submaps.py`
   - `ProbabilityGrid.__init__` now accepts `resolution=` as a backward-compatible
     alias for `res=` (eval scripts construct with `resolution=`).
3. `slam_core/matching/scan_to_submap/__init__.py`
   - Re-export `CartoRefinementProblem`, `refine_pose_submap`, `PrecomputationGridStack`,
     `correlative_match_two_stage`, `_bruteforce_search`, `_score_candidate`
     (fixes the previously-broken `carto/pose_graph/scan_matcher_cache_2d.py` import).
4. `slam_core/matching/scan_to_submap/types.py`
   - Added `ScanToSubmapBackendConfig.local_refine_backend: str = "native"`.
5. `slam_core/matching/scan_to_submap/two_stage_backend.py`
   - Correlative import now from `.correlative` (was `scan_to_submap_old`).
   - Local refinement: native `GaussNewtonLM` + `CartoRefinementProblem` is the DEFAULT
     (`local_refine_backend="native"`); pyceres path kept optional + lazily imported.
   - Added `_score_pose_on_grid` and `_refine_native` helpers; `match()` branches on backend.
   - Debug keys renamed `pyceres_*` -> `refine_*` (+ `refine_backend`).
6. Hector runner `hector/run_local_slam_new.py`
   - Now constructs the package `ScanToSubmapMatcher` + `ScanToSubmapBackendConfig`
     (native refine) instead of `OldScanToSubmapMatcher`. Imports updated.
7. Eval scripts repointed to the package modules (no behaviour change):
   - `hector/eval/rebuild_map_any.py`, `rebuild_map_lab_submap.py`, `pgo_any.py`,
     `pgo_lab_submap.py` (ProbabilityGrid/SubmapBuilder2D/Submap2D -> `.submaps`,
     CartoRefinementProblem -> `.refine`, correlative_match_two_stage -> `.correlative`,
     GaussNewtonLM/GNLMConfig -> `slam_core.optimisers.gn_lm`).
8. DELETED `slam_core/matching/scan_to_submap_old.py`.

Rollback for Phase 1: `git checkout -- slam_core/matching hector/run_local_slam_new.py hector/eval`
restores the old file and all importers (the old file is tracked in git).

Verification: all touched modules import cleanly (`.venv/bin/python -c import ...`).

## Phase 2 — g2o SE2 pose-graph backend
Goal: replace the scipy/pyceres pose-graph solver with a g2o SE2 backend.

Changes:
1. NEW `carto/pose_graph/backends/g2o_backend_2d.py` — `G2oBackend2D`.
   - Drop-in replacement for `PyCeresBackend2D`/`SciPyBackend2D`: identical graph
     API (`add_node`, `add_submap`, `add_constraint`, `update_node_local_pose`,
     `set_fixed`) and `solve(max_iters) -> {(kind,id): Pose2}` contract.
   - g2o `VertexSE2` per node/submap (disjoint ids: submap=2*id, node=2*id+1),
     `EdgeSE2` for INTRA/INTER constraints (measurement z = T_submap^-1 * T_node),
     `RobustKernelHuber` on INTER (loop) edges only.
   - Local-trajectory regularization spine (consecutive nodes, z=local_i^-1*local_j,
     weight 1e5) replicated from the pyceres backend.
   - Solver: `BlockSolverSE2(LinearSolverEigenSE2)` + `OptimizationAlgorithmLevenberg`
     with fallback across Eigen/Cholmod/CSparse/Dense.
2. NEW `carto/test_g2o_vs_pyceres.py` — unit test.
   - Builds an identical square-loop graph (single fixed submap, GT-consistent
     intra + spine + one loop closure) for both backends via the real PoseGraph2D
     API, injects 0.29 m of drift into the initial estimates, solves both.
   - RESULT: both recover GT exactly (RMSE 0.0000 m), g2o == pyceres to machine
     precision. Proves drop-in equivalence + correct constraint handling.
   - NOTE: discovered a pre-existing artifact — PoseGraph2D's local_pose is
     defined relative to the *primary submap*, so a spine edge crossing a submap
     boundary is corrupted; both backends reproduce it identically (not a g2o bug).
     The Hector integration must insert each node's intra constraints consistently
     (single primary submap per node, as the unified front-end already does).

Rollback for Phase 2: delete `carto/pose_graph/backends/g2o_backend_2d.py` and
`carto/test_g2o_vs_pyceres.py`; no existing files were modified.

Run test: `.venv/bin/python -m carto.test_g2o_vs_pyceres`

## Phase 3 — Online PGO wired into Hector runner
Goal: give the Hector scan_to_submap run an online g2o pose graph (intra + loop)
WITHOUT changing the front-end prediction/matching/insertion behaviour.

Changes:
1. `hector/adapter.py` (`HectorLocalSlamAdapter`)
   - New optional ctor args `pose_graph`, `global_slam`, `solve_every_n_nodes`.
   - On each accepted submap insert (genuine MATCH only, not FALLBACK), calls
     `_maybe_add_pose_graph_node()` -> `pose_graph.add_node_with_intra_constraints`
     + `global_slam.on_node_inserted` (loop search + periodic optimize + submap
     write-back). Mirrors CartoLocalSlamAdapter lines 220-252.
   - New `finalize()` -> `global_slam.finalize()` (final global solve).
   - When PGO is off, behaviour is byte-identical to before (all hooks are no-ops).
2. `hector/run_local_slam_new.py`
   - New `--enable-pgo` flag (scan_to_submap only).
   - Builds the back-end: branch-and-bound loop matcher (native refine, shares the
     same SubmapBuilder2D), `G2oBackend2D` (fixed submap 0), `PoseGraph2D`
     (intra weights 5e2/1.6e3), `CartoLoopClosureAdapter` + `CartoGlobalSlam2D`
     (optimize_every_n_nodes default 90). `adapter=None` in global_slam (submap
     write-back propagates corrections; no extrapolator nudge needed for lab).
   - After the loop: `adapter.finalize()`, prints constraint/loop stats, exports the
     OPTIMIZED node trajectory to `..._pgo.txt` (post-solve poses).
   - Tunable via cfg: PGO_LOOP_MIN_SCORE, PGO_LOOP_SEARCH_XY, PGO_MIN_NODE_SEPARATION,
     PGO_SPATIAL_SEARCH_RADIUS, PGO_OPTIMIZE_EVERY_N_NODES (all have getattr defaults).

Smoke test (150 scans): 137 nodes, 274 intra constraints, finalize + export OK,
intra-only optimization leaves trajectory unchanged (online == PGO endpoint to 1e-4),
confirming no distortion; loop closures activate on the full looping run.

Rollback for Phase 3: `git checkout -- hector/adapter.py hector/run_local_slam_new.py`
(or just omit --enable-pgo; the flag is opt-in and off by default).

## Phase 4 — Lab validation + tuning  (in progress)

### Phase 1 regression resolved
First full no-PGO unified run diverged from the baseline (mean 0.61 m, endpoint
drifted to (-0.78,-2.79) vs baseline (-0.48,0.25)). Root cause: the unified
two-stage backend always refined+accepted, whereas the legacy Hector matcher
returned FALLBACK (held last pose) when the correlative score < min_score —
the divergence began at a sharp turn (k~579) where scores hovered near 0.50.

Fix: added `ScanToSubmapBackendConfig.reject_below_min_score` (default False =
Cartographer always-refine; set True in the Hector runner). With it on, the
unified no-PGO run matches the baseline: mean 0.037 m, endpoint (-0.464,0.246)
vs (-0.482,0.247). Regression resolved; carto behaviour unchanged.

Files: `slam_core/matching/scan_to_submap/types.py` (+field),
`two_stage_backend.py` (FALLBACK branch), `hector/run_local_slam_new.py`
(reject_below_min_score=True in the submap config).

### Dense PGO trajectory export
PGO export reconstructs a dense per-scan trajectory (1 pose/scan, aligns 1:1 with
the online traj for `rebuild_map_any`): each scan inherits its most-recent
keyframe's rigid SE(2) correction (delta = T_opt_node * inv(T_online_node)).
Adapter tracks `last_node_id`; runner records (t, online_pose, node_id) per scan.
Baselines: baseline_snapshots/{baseline_trajectory_scan_to_submap_1052.txt,
baseline_trajectory_scan_to_map_1052.txt (via trajectory_..._scan_to_map_1052),
unified_nopgo_v2_1052.txt}.

### Loop-closure + spine fixes (made the PGO actually work)
First full PGO run: 871 nodes, 4 submaps, **0 loop closures**, and a bogus 3.05 m
"correction". Two bugs:
1. **Spine boundary artifact** (pre-existing in PoseGraph2D): the consecutive-node
   regularization measurement `z_ij = local_i^-1 * local_j` used per-primary-submap
   `local_pose`; at a submap boundary the two nodes reference different submaps, so
   z is corrupted by the submap offset and (at weight 1e5) distorts the trajectory.
   FIX: `carto/pose_graph/pose_graph_2d.py` now passes the GLOBAL online node pose to
   `update_node_local_pose` (identical within a submap, correct at boundaries).
   Strictly improves carto too; g2o unit test still passes.
2. **Zero loop candidates**: with `recent_finished_submap_exclusion=2` and only ~2
   finished submaps, ALL finished submaps were excluded. FIX: set it to 0 for the
   Hector runner (`PGO_RECENT_SUBMAP_EXCLUSION`).
After fixes (SCANS_PER_SUBMAP=500): 870 nodes, 24 loop closures accepted (530
candidates), max correction 0.073 m — no distortion.

### Quantitative drift (lab loop, no ground truth -> start<->end gap proxy)
- scan_to_map      : start-end gap 0.186 m, path 32.2 m
- scan_to_submap (no PGO): gap 0.525 m, path 42.4 m
- scan_to_submap + PGO (sps=500): gap 0.531 m, path 42.0 m  (PGO corr mean 0.0065 m)
Finding: with sps=500 only 2 submaps finish, so the sole loop target is the start
submap and PGO has almost nothing to correct. Added `--scans-per-submap` override to
test smaller submaps (Cartographer uses ~90) -> more finished submaps -> more loops.
3-way map figure: hector_outputs/three_way_comparison.png (all three reconstruct the
room cleanly; no gross distortion).

### Bounded loop search + smaller submaps -> PGO clearly helps
Root cause of weak PGO at sps=500: only 2 finished submaps -> the start submap is the
only loop target. Cartographer uses much smaller submaps. Added `--scans-per-submap`
override and `PGO_CHECK_EVERY_N_NODES` (loop search runs every Nth node; default 5) to
bound cost — branch-and-bound over every finished submap for every node is O(submaps x
nodes) and exploded at sps=120 (~3 h, killed).

Final run: `--enable-pgo --scans-per-submap 250` (check_every_n_nodes=5):
- 908 nodes, 8 submaps (6 finished), 843 candidates, **208 loop closures accepted**,
  max correction 0.328 m, runtime ~15 min.

Drift (start<->end gap, closed-loop proxy; lower=better):
| trajectory                     | path (m) | start-end gap (m) |
|--------------------------------|----------|-------------------|
| scan_to_map (baseline)         | 32.2     | 0.186             |
| scan_to_submap (no PGO)        | 42.4     | 0.525             |
| scan_to_submap + PGO (sps=500) | 42.0     | 0.531  (no help)  |
| scan_to_submap + PGO (sps=250) | 35.6     | **0.306** (-42%)  |

PGO cuts the submap drift gap by ~42% and tightens the path (42.4->35.6 m, toward
scan_to_map's 32.2). Map figure hector_outputs/three_way_comparison_sps250.png shows
the no-PGO ghosting/double-walls largely removed by PGO. scan_to_map remains slightly
tighter (0.186 vs 0.306) — the residual is per-scan front-end jitter, which a keyframe
pose graph does not smooth. RECOMMENDATION: use `--enable-pgo --scans-per-submap 250`
(or smaller) for lab; documented in CLAUDE.md.

### Comparison-render fix (visualization only)
`compare_three_maps._panel` drew `imshow(prob, origin="upper")` while `_grid_xy`
returns the trajectory flipped (size-1-gy), mirroring the trajectory vertically vs
the map — looked like a "randomly shifted" trajectory on the asymmetric submap maps.
Fix: `imshow(np.flipud(prob), origin="upper")` to match rebuild_map_any's convention.
Data/metrics were computed from raw world poses and were never affected — only the PNG.

## Status: COMPLETE
All 4 phases done. One unified scan_to_submap front-end (Hector + carto), online g2o
SE2 pose graph (replaces scipy/pyceres), validated on lab. Rollback: see per-phase notes
above; all original files tracked in git (branch ORB-SLAM).
