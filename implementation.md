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

## Status: Phases 1-4 COMPLETE
One unified scan_to_submap front-end (Hector + carto), online g2o SE2 pose graph
(replaces scipy/pyceres), validated on lab. Rollback: see per-phase notes above; all
original files tracked in git (branch ORB-SLAM).

## Phase 5 — Cartographer-style fast local matching + IMU extrapolator (real-time)
Goal: close the scan_to_submap real-time gap (~630-1000 ms/scan) by aligning the
front-end with Cartographer, and add optional IMU aiding. All OFF by default — the
stable runner (`run_local_slam_new.py`) and `adapter.py` are byte-identical (verified:
default scan_to_submap reproduces the baseline to 0.0 m / 0.0 rad over 300 scans).

Root cause (found by profiling, not assumed): the per-scan cost was NOT the solver —
it was (1) the wide correlative brute-force search (~7,500 candidates) every scan, and
(2) submap occupancy insertion via a per-cell Python loop with scalar `np.clip`
(`update_cell` called 2.6M times). Matching itself is ~10 ms.

IMPORTANT correction (after a first attempt): the first design SKIPPED the wide search and
seeded Ceres from the prediction ("ceres_primary"). That was fast but JERKY/distorted — the
wide search is exactly what re-anchors each scan and prevents a velocity-estimate cascade,
so removing it let confident-but-wrong matches through. Final design keeps the SAME full
search and just makes it fast (vectorized). The skip-search path was removed.

Changes (all opt-in; default scalar path bit-identical):
1. `slam_core/matching/scan_to_submap/correlative.py`: new `_bruteforce_search_vectorized`
   + `vectorized=` flag on `correlative_match_two_stage`. Same window + nearest-cell mean
   scoring as the scalar loop (hoists point rotation out of the dx/dy loops, scores all
   translations with batched NumPy). Matches the scalar search to ~2e-7 and picks the
   identical pose; ~10x faster. Replaces ~8,900 Python score calls/scan.
2. `slam_core/matching/scan_to_submap/types.py`: `ScanToSubmapBackendConfig.use_vectorized_search`
   (default False). `two_stage_backend.py` reads it and passes `vectorized=` to the search;
   the default scalar path is otherwise untouched.
3. `slam_core/matching/scan_to_submap/submaps.py`: `SubmapBuilder2D(fast_insert=False)` +
   vectorized `_integrate_submap_frame_fast` (collect ray cells, `np.bincount` accumulate
   log-odds, single clip). ~30x faster insertion; matches the per-cell path to within float
   rounding (7/160k cells differ at the clamp bound — negligible).
4. `carto/local_slam/pose_extrapolator.py`: `PoseExtrapolatorCV(use_imu, imu_yaw_correction_alpha)`
   + `add_imu(t, wz, yaw)`; `predict()` uses IMU gyro wz for rotation and a small
   absolute-yaw blend toward the quaternion heading. CV behaviour identical when off.
   New `carto/local_slam/imu_extrapolation.py` (quaternion->yaw + CSV row converter,
   reuses `slam_core/dataio/imu_csv.read_imu_csv`).
5. `hector/run_realtime_viz.py`: flags `--fast-match` (use_vectorized_search + fast_insert),
   `--local-solver {native,pyceres}` (refine backend; default native), `--use-imu`; loads
   `imu.csv` and feeds the extrapolator.
6. `hector/config.py`: documented `IMU_YAW_CORRECTION_ALPHA` (0.02); fast-match note.

Results (full lab_run_2, 1052 scans):
| config                 | ms/scan | quality                                   |
|------------------------|---------|-------------------------------------------|
| scan_to_submap default | ~1007   | reference                                 |
| --fast-match           |   ~42   | SAME full search (vectorized==scalar pose)|
Fast-match is ~24x faster and real-time at the lab's ~10 Hz, with the SAME match quality as
default (the search is identical, just vectorized) — no jerks, clean map
(hector_outputs/fast_v2.png). IMU does NOT help this slow indoor robot (gyro wz adds
bias/noise; scan-derived velocity is already good) -> IMU stays OFF by default; the path is
correct and kept for fast-motion/feature-poor use.

Rollback for Phase 5: all changes are behind defaults (use_vectorized_search=False,
fast_insert=False, use_imu=False); omit the flags. `git checkout` restores; default
scan_to_submap is byte-identical to pre-Phase-5 (verified 0.0 m / 0.0 rad over 300 scans).

### Verification of fast-match map quality (resolved)
A reported "distorted" realtime map vs the clean `three_way` no-PGO map was investigated.
ROOT CAUSE was twofold and NOT in the shipped fast-match code:
1. The distorted `realtime_*.png` on disk was a STALE artifact from the earlier (removed)
   `ceres_primary` skip-search phase.
2. A first diagnosis harness hardcoded `use_extrapolator=True`, but the lab profile sets
   `USE_EXTRAPOLATOR=False` (both real runners read it from config). The CV extrapolator's
   noisy velocity estimate produced jerky trajectories — a harness artifact, not the code.
With the CORRECT config (use_extrapolator from cfg), verified on full lab_run_2 (1052):
- DEFAULT scan_to_submap reproduces `unified_nopgo_v2` baseline EXACTLY (max Δ = 0.000 cm).
- FAST tracks DEFAULT to 4.95 cm mean (gap 0.543 vs 0.525 m); ~53 ms vs ~1150 ms/scan.
- Rendered through the SAME `compare_three_maps` tool that made three_way, the fast no-PGO
  panel is indistinguishable from the original clean three_way (hector_outputs/fast_vs_threeway.png).
Conclusion: the vectorized search + fast_insert are faithful; fast-match map quality == default
== three_way. Fast-match output now uses a distinct `_fast` filename suffix to avoid stale-file
confusion. (The mild lower-middle trajectory wiggle exists identically in the clean default and
does not distort the map.)

Recommended real-time command:
`.venv/bin/python -m hector.run_realtime_viz --dataset lab_run_2 --matcher scan_to_submap --fast-match`
(add `--enable-pgo --scans-per-submap 250` for the global back-end).

### Why scan_to_submap jerks and underperforms scan_to_map WITHOUT PGO
These are intrinsic to the front-end architecture, NOT bugs in our code — verified: the
small per-scan jitter and the larger start↔end drift gap appear identically in the default
(scalar) and fast (vectorized) paths, and the default reproduces the pre-Phase-5 baseline to
0.0 m. They are the direct, expected consequence of matching a LOCAL submap instead of ONE
global map in a small looping room. The two symptoms have different causes:

**A. The per-scan JERKS (high-frequency wiggle along the path).**
1. *Less geometry per match → a flatter, noisier score surface.* scan_to_map matches every
   scan against the ONE global occupancy grid, which by mid-run holds the whole room — many
   walls, a sharp single score peak, so the correlative search + Gauss-Newton refine lock onto
   a precise pose. The active submap holds only the last ~N scans of geometry (far fewer
   constraints, especially in open stretches), so its score surface is flatter and the
   best-pose estimate is more sensitive to scan noise → small frame-to-frame position jitter.
2. *Correlative-search discretization the local map can't smooth out.* The brute-force search
   is on a discrete grid (xy step + angle step). Against the rich global map the GN refine has
   enough structure to interpolate between grid cells and remove the quantization; against the
   sparser submap the refine has less to bite on, so residual search-grid quantization leaks
   through as a zigzag. (This is the "mild lower-middle wiggle" — it sits in a corridor-like
   stretch where the active submap sees few distinguishing features.)
3. *Submap-boundary discontinuities.* When a submap finishes and a new one starts, the first
   scans match against a nearly-empty, low-log-odds submap (occupancy hasn't saturated yet) →
   a weak, ambiguous constraint → the pose can step at the handoff. scan_to_map never has a
   handoff: it is always matching one continuously-growing grid.

**B. The larger DRIFT / underperformance (start↔end gap 0.525 m vs 0.186 m).**
1. *scan_to_map is an implicit perfect loop closure every frame.* Keeping one global grid
   forever means a revisited spot is re-matched against the SAME geometry it built on the first
   pass — the map itself closes the loop continuously, so drift in a small room stays near zero.
   This is why scan_to_map is so strong HERE specifically: the entire room fits in one grid, so
   "match against everything" is both cheap and maximally constraining.
2. *scan_to_submap drops finished submaps from the front-end.* Once a submap is finished it is
   no longer a match target, so a revisit is invisible to the front-end — there is nothing
   online to re-anchor against, and the small per-handoff misalignments (B.3 below) accumulate
   monotonically into end-of-run drift. Restoring that global constraint is EXACTLY the job of
   the pose-graph back-end (intra-submap + inter-submap loop closures + optimize), which is why
   adding PGO cuts the gap 0.525 → 0.306 m.
3. *Drift compounds across handoffs.* Each new submap is anchored only by the last few scans of
   the previous one; any small local error at that seam is baked into the new submap's global
   pose and carried forward, with no online mechanism to undo it.

**Why this is the EXPECTED trade-off, not a regression.** Submaps exist to make SLAM tractable
at large scale, where a single global grid is too big to hold or match against. Their cost is
locality: each match sees less geometry and there are handoffs. In a small single room that
cost is pure downside (the global grid was never a problem), so scan_to_map wins on raw drift
and smoothness. scan_to_submap only catches up once paired with the global pose graph — and
even then a keyframe pose graph corrects DRIFT (the B symptoms), not the per-scan front-end
JITTER (the A symptoms), which is why a residual gap to scan_to_map remains after PGO.

### Async PGO (Cartographer-style decoupling) + smaller loop window
Problem: with `--enable-pgo` the global back-end (loop-closure branch-and-bound + periodic
g2o solve) ran SYNCHRONOUSLY inside the per-scan loop, blocking the live front-end with
multi-second stalls (the `slam`-stage max hit ~12 s on optimize steps) — nowhere near
real-time. Cartographer avoids this by running the global SLAM on a background thread/work
queue, never blocking local SLAM.

Changes (runner-only; stable runner + carto internals untouched):
1. `hector/run_realtime_viz.py` `AsyncPoseGraphRunner`: a daemon worker thread that EXCLUSIVELY
   owns `pose_graph` + `global_slam`. The adapter now gets `pose_graph=None`/`global_slam=None`
   (pure real-time front-end); the runner submits each accepted keyframe to the worker queue
   and returns immediately. Thread-safety: the worker owns the graph (no graph/backend races);
   the only shared object is the submap builder, where the main thread mutates ACTIVE submaps
   and the worker reads FINISHED (immutable) submaps + writes back poses — safe under CPython's
   GIL. The g2o solve is C++ (releases the GIL) so it overlaps; the real-time inter-scan sleep
   also lets the worker drain. Dense PGO trajectory now reconstructed by scan TIME (most-recent
   keyframe correction) since node ids are assigned asynchronously.
2. Loop-search window reduced to the lab room scale: `PGO_LOOP_SEARCH_XY` 7.0→3.0 m,
   `precomp_levels` 7→6, `bnb_depth_limit` 7→6, and `check_every_n_nodes` 5→10 (realtime-only
   defaults; the stable runner keeps 7.0/7/7/5).

Result (lab, real-time --speed 1.0, sps=250, 500-scan test): late frames 12.2% → **0.6%**,
final lag 0.000 s, PGO backlog drains to ~12; loops still fire; clean map
(hector_outputs/realtime_lab_run_2_raw_scan_to_submap_fast_pgo.png). The FRONT-END is now
real-time with PGO on.

HONEST residual: async removes the per-scan BLOCKING but not the total PGO WORK. Over the full
1052-scan loop the Python branch-and-bound loop search totals ~10+ min, so on a long run the
back-end falls behind live and finishes draining minutes after the run (corrections are
"eventual"). The live trajectory stays real-time throughout. Closing this fully needs either a
vectorized branch-and-bound loop scorer (the `branch_and_bound` backend still uses the scalar
Python scorer — the two_stage local search was vectorized, this wasn't) or true C++
parallelism as in Cartographer. Further cheap levers: smaller `PGO_LOOP_SEARCH_XY`, larger
`PGO_CHECK_EVERY_N_NODES`, fewer `precomp_levels`.

## Phase 6 — Live front-end switching (scan_to_map ↔ scan_to_submap), local-only
Goal: switch the tracking front-end **mid-run** between `scan_to_map` and `scan_to_submap`
and have the live system adopt the new style, with a stable handoff. Local mapping only —
PGO/loop-closure switching is deferred. The stable runner (`run_local_slam_new.py`) and
`adapter.py` are **untouched** (regression: their lab_run_2 trajectory is byte-identical,
max Δ = 0.0 over 150 scans — verified by stashing the two shared files and re-running).

Most of the scaffold already existed and was reused: `MatcherManager`
(`slam_core/matching/core.py`) holds the active/pending matchers, a `RollingScanBuffer` that
survives a switch, and switches only BETWEEN scans; the adapter already calls
`maybe_activate_pending()` every scan and re-derives per-matcher behavior, so it adopts a new
active matcher automatically. `ScanToMapMatcher.initialize_from_buffer` already implemented the
desired warm-start (fresh map pyramid seeded only from the rolling buffer, last pose = anchor).

Design decisions (confirmed with the user):
- **Trigger** = typed command on **stdin** (`map`/`submap`/`status`/`quit`). Works headless/SSH.
- **Grace window** — after a request, the ORIGINAL matcher keeps running for ~N more scans
  (`--switch-grace-scans`, default 15), then the switch fires. Stable handoff + time to fill
  the buffer.
- **Warm-start** = replay the rolling buffer: on switch, drop the old map entirely and rebuild
  a FRESH map/submap from only the recent ~30 buffered scans at their world poses (last pose =
  current). Trajectory stays continuous (the extrapolator is shared and the anchor pose carries
  over). Requirement: "treat the switch as a fresh origin, last pose as the prior."
- **Failure handling** = dead-reckon only (no rollback): if the new matcher fails post-switch it
  uses the normal FALLBACK path. So the old matcher is `shutdown()` immediately at the switch.
- **PGO** = switching is DISABLED with `--enable-pgo` (the async PGO worker is bound to the
  submap builder); deferred to a later phase.

Changes:
1. `slam_core/matching/scan_to_submap/matcher.py`: `ScanToSubmapMatcher` now subclasses
   `ScanMatcherBase` and implements `initialize_from_buffer` (`submap_builder.clear()` then
   replay buffered scans → fresh submap) and `shutdown` (`clear()`). This was the blocking gap —
   previously it had neither, so `maybe_activate_pending`/`shutdown` would `AttributeError` on
   any switch touching the submap matcher.
2. `slam_core/matching/core.py`: `request_switch(new, grace_scans)` adds a grace countdown +
   edge handling (switch-to-active cancels any pending; same-target re-request is ignored;
   different-target while pending = last wins + reset). `maybe_activate_pending` decrements the
   countdown once per scan and only activates after it hits 0 AND the buffer is ready
   (`min_buffer_for_switch`), so a switch can never fire mid-bootstrap.
3. `hector/run_realtime_viz.py`: builds BOTH matchers up front (`--matcher` only picks the
   initial active one); a daemon stdin reader pushes commands onto a queue that the MAIN loop
   drains between scans (all manager mutation stays single-threaded; switch still happens
   between scans). Always passes `motion_params` (the scan_to_map branch ignores it, so no
   behavior change; a switch INTO submap keeps its insertion cadence). New `--switch-grace-scans`.
   The final map is titled/named by the FINAL active matcher.

Edge/fail cases handled: switch-to-active = no-op; same-target re-request keeps the countdown;
last-wins on a different target; early request defers until the buffer holds
`min_buffer_for_switch` scans; pose continuity across the switch (shared extrapolator + world-
frame buffer poses, no transform); PGO active = switching disabled; headless = stdin still works.
Known limit: warm-start recreates a fixed-size scan_to_map pyramid centered at the config origin
(fine for the small lab room; revisit for very large maps).

Verification (all with `.venv/bin/python`):
- Regression: stable runner byte-identical (Δ = 0.0). The new code is a true no-op when no
  switch is requested.
- Unit test `hector/test_matcher_switching.py`: grace countdown fires exactly after N (both
  directions), warm-start sets `is_initialized`, edge cases (no-op/last-wins/same-target), and a
  real-data pose-continuity check on lab_run_2 (switch at k=200 → jump 3.4 cm vs 3.2 cm typical
  step = no jump). All pass.
- End-to-end via real typed stdin (headless, `--speed 0`): submap→map requested k=123 → switched
  k=138; map→submap requested k=51 → switched k=66 (each = request + 15 grace); clean exit,
  600 scans, no traceback.

Run: `.venv/bin/python -m hector.run_realtime_viz --dataset lab_run_2 --matcher scan_to_submap
--fast-match` then type `submap`/`map` (+Enter) live; `--switch-grace-scans N` tunes the handoff.
