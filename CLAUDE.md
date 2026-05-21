# CLAUDE.md — Hector scan-to-submap + g2o PGO task

## Objective
Make Hector `scan_to_submap` match or beat `scan_to_map` on the lab dataset by (1) unifying a single
shared scan-to-submap front-end, (2) adding an online Cartographer-style pose-graph back-end, and
(3) using a g2o SE2 solver instead of scipy/pyceres.

## Why (root cause)
`scan_to_map` keeps ONE global occupancy grid forever, so in a small looping room every scan
re-matches against all prior geometry — an implicit "perfect loop closure" every frame → low drift.
`scan_to_submap` is a front-end ONLY: it matches one active submap, drops finished submaps, and has
no online pose graph, so drift accumulates at submap handoffs and revisits are invisible.
Cartographer's submaps only win when paired with the global pose-graph back-end (intra-submap
constraints + inter-submap loop closures + optimization fed back into submap poses).

## Architecture (after this work)
- Front-end (ONE shared module): `slam_core/matching/scan_to_submap/` — local correlative + GN
  scan-to-submap, used by BOTH Hector and Cartographer with separate config profiles. Do NOT create a
  second copy; `scan_to_submap_old.py` is removed.
- Back-end (reused from carto): `carto/pose_graph/` — `PoseGraph2D` holds nodes/submaps/constraints;
  `CartoGlobalSlam2D` runs loop-closure search + periodic optimize + write-back online.
- Solver: `carto/pose_graph/backends/g2o_backend_2d.py` (g2o `VertexSE2`/`EdgeSE2`/`BlockSolverSE2` +
  Levenberg) is the default; pyceres/scipy are fallbacks.

## Hard rules
- Exactly ONE scan-to-submap implementation; differences between Hector and Cartographer are
  parameters only, never forked logic.
- g2o is for the POSE GRAPH. Local per-scan refinement stays native `GaussNewtonLM`.
- Always export the OPTIMIZED trajectory (post-solve), never the raw online poses.
- Run with `.venv/bin/python`. g2o needs NumPy 1.26.4 (already in `.venv`).

## Validate
Run `lab_run_2` for: scan_to_map; scan_to_submap (no PGO); scan_to_submap + PGO. Rebuild maps,
compare drift/distortion and loop closure. scan_to_submap+PGO must be <= scan_to_map drift.
Unit test: square-loop pose graph, g2o vs pyceres agree < 1e-3 m / 0.1 deg.

Commands:
- `.venv/bin/python -m hector.run_local_slam_new --dataset lab_run_2 --matcher scan_to_map`
- `.venv/bin/python -m hector.run_local_slam_new --dataset lab_run_2 --matcher scan_to_submap`
- `.venv/bin/python -m hector.run_local_slam_new --dataset lab_run_2 --matcher scan_to_submap --enable-pgo`

## Recommended PGO usage (lab)
`.venv/bin/python -m hector.run_local_slam_new --dataset lab_run_2 --matcher scan_to_submap --enable-pgo --scans-per-submap 250`
Smaller submaps -> more finished submaps -> more loop closures. Loop search is bounded
by PGO_CHECK_EVERY_N_NODES (default 5). sps=500 (the dataset default) yields only ~2
finished submaps and PGO barely helps; sps=250 gives 8 submaps / 208 loop closures.

## Result (lab_run_2, start<->end drift gap, lower=better)
- scan_to_map: 0.186 m   |   scan_to_submap no-PGO: 0.525 m   |   scan_to_submap+PGO(sps250): 0.306 m
PGO removes the no-PGO ghosting (see hector_outputs/three_way_comparison_sps250.png) and
cuts drift ~42%. Residual gap to scan_to_map is front-end per-scan jitter (not fixable by
a keyframe pose graph).

## Status log
- [x] Phase 1 unify front-end + de-pyceres local refine + migrate Hector runner (scan_to_submap_old.py deleted)
- [x] Phase 2 g2o backend (carto/pose_graph/backends/g2o_backend_2d.py) + unit test (carto/test_g2o_vs_pyceres.py PASS)
- [x] Phase 3 online PGO wired into Hector runner (--enable-pgo, --scans-per-submap)
- [x] Phase 4 lab validation (3-way comparison) + tuning
