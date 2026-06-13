*General Excution Rules*

1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. refer the relevant documentations clearly understand the solution and continue implementing.
2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

# Project Implementation: RTAB-Inspired Multi-Modal SLAM (v1)

This workspace is mid-implementation on a multi-modal SLAM build that fuses the existing ORB-SLAM (RGB-D visual) and LiDAR (Hector / Cartographer) pipelines under an RTAB-Map-inspired architecture. Every Claude conversation in this repo MUST follow the sequenced phase plan below.

## 0. Autonomy & venv (strict, non-negotiable)

- **Full autonomy granted.** Do not pause to ask the user for permission before running tests, editing files, fetching web pages, creating commits, or launching subprocesses. Permissions are pre-approved in `.claude/settings.json`.
- **All Python execution MUST go through `.venv/bin/python` or `.venv/bin/pytest`.** Never use bare `python`, `python3`, `pip`, or `pytest`. The project venv lives at `/home/kaushik/slam_ws/.venv` (Python 3.11).
- **Examples (do):**
  - `.venv/bin/python -m slam_core.fusion.runner --mode vlmain ...`
  - `.venv/bin/pytest tests/fusion/test_checkpoint_p1_foundation.py -x -q`
  - `.venv/bin/pip install small_gicp`
- **Examples (do NOT):** `python -m ...`, `python3 tools/...`, `pip install ...`, `pytest ...`.

## 1. Reference documents (read these FIRST)

In any new session that touches the fusion implementation, read these files in order before producing a plan or code:

| File | Purpose | When to read |
|---|---|---|
| `FUSION_STATUS.md` | Live per-phase progress tracker | **Always, first.** Tells you which phase is next. |
| `RTAB_inspired_implementation_plan.md` | Full v1 implementation plan (the source of truth for design decisions) | Whenever planning a change, scoping a phase, or resolving ambiguity. |
| `RTAB_review.md` | Upstream RTAB-Map architectural reference | Whenever a design question maps back to "how does RTAB do this?" |
| `Hector_implementation.md` | 2D LiDAR SLAM math + pipeline reference | Whenever touching LiDAR front-end code (scan matching, pose extrapolation). |
| `implementation.md` | Historical implementation log (pre-fusion phases) | Context only; useful when reusing existing modules. |

### Secondary reference — upstream RTAB-Map sources (consult when local docs are insufficient)

Use these only if the local docs above leave a question unanswered (e.g., a subtle algorithmic detail not captured in `RTAB_review.md`, or when planning a v2 feature that needs verification against the upstream design). Quote sources inline in any new plan or design note.

| Source | URL | Use for |
|---|---|---|
| Labbé & Michaud, *RTAB-Map as an open-source lidar and visual SLAM library for large-scale and long-term online operation*, JFR 2019 | https://arxiv.org/abs/2403.06341 (HTML: https://arxiv.org/html/2403.06341v1) | Complete pipeline, sensor inputs, odometry, loop closure, ICP refinement, optimizer choices, map outputs, key parameters. The canonical paper to cite. |
| Labbé & Michaud, *Memory management for real-time appearance-based loop closure detection*, IROS 2011 | https://arxiv.org/abs/2407.15890 (HTML: https://arxiv.org/html/2407.15890v1) | The original STM / WM / LTM design, rehearsal merge, weight rule, transfer algorithm, retrieval. The definitive source for memory-tier semantics. |
| RTAB-Map official site | http://introlab.github.io/rtabmap/ | High-level architecture overview, supported sensor configurations, optimizer options, database persistence model. |
| IntRoLab project page | https://introlab.3it.usherbrooke.ca/index.php/RTAB-Map | Project background, datasets, real-time performance numbers, related publications. |
| RTAB-Map C++ source (read-only reference) | https://github.com/introlab/rtabmap | Implementation-level truth when a header signature is ambiguous. Key files: `corelib/include/rtabmap/core/{Memory,Rtabmap,Signature,Link,DBDriverSqlite3}.h`; corresponding `.cpp` for full algorithm bodies. |

Use `WebFetch` to pull any of these on demand (permissions are pre-approved in `.claude/settings.json`). When citing in a new design doc, prefer the JFR 2019 paper (`[JFR'19]`) for architecture and the IROS 2011 paper (`[IROS'11]`) for memory management, matching the citation style already used in `RTAB_review.md`.

## 2. Architectural decisions (immutable for v1)

These were locked through structured Q&A and are not to be revisited mid-implementation:

1. **Unified topology** — A new `slam_core/fusion/runner.py` is the single entry point for SLAM. Both existing front-ends (ORB-SLAM, LiDAR) become services consumed by it.
2. **Four user-selectable modes** via `--mode {orb,lidar,vlmain,lvmain}`:
   - `orb`: pass-through to the existing ORB-SLAM runner.
   - `lidar`: pass-through to the existing LiDAR runner.
   - `vlmain`: Visual-main + LiDAR-verifier. ORB-SLAM proposes loop candidates; its own verification is disabled; LiDAR ICP confirms.
   - `lvmain`: LiDAR-main + Visual-verifier. LiDAR B&B / proximity proposes; its own verification is disabled; ORB descriptors + PnP confirm.
3. **Visual loop detector** — Reuse ORB-SLAM's existing detector (`visual_slam/orbslam/slam/keyframe_database.py`) in propose-only mode. Do not build a new BoW.
4. **ICP backend** — `small_gicp` (C++ with Python bindings). Used as the cross-modal verifier only in v1. Not used inside the LiDAR front-end's per-scan path.
5. **Memory tiers** — Full STM + WM + LTM with RTAB-Map's rehearsal/weight/transfer semantics, but **in RAM only** for v1 (no SQLite persistence — that's v2).
6. **Unified graph in SE(2)** — Planar-motion assumption. ORB-SLAM's SE(3) keyframe poses are projected to `Pose2` at the fusion-layer boundary using a known camera→base TF.
7. **LiDAR PGO disabled in fusion modes** — When `--mode vlmain` or `--mode lvmain` is active, the LiDAR pipeline's g2o backend is bypassed; the unified graph owns optimization. Standalone LiDAR runs (`--mode lidar`) keep their own PGO.
8. **Signature payload** — Each fused keyframe carries: `Pose2` pose, ORB keypoints + descriptors + 3D points, the most-recent 2D LiDAR scan (within sync tolerance), and a small local occupancy grid.
9. **Module location** — All new code lives under `slam_core/fusion/`. No source files in `visual_slam/`, `hector/`, `carto/`, or `slam_core/matching/` are modified (adapters intercept instead).
10. **Outputs** — Trajectory file (TUM format) + 2D occupancy grid PNG. Dense 3D cloud and OctoMap deferred to v2.
11. **Test dataset** — TUM RGB-D (e.g., `fr1_room`, `fr1_desk`) with 2D LiDAR synthesized from a horizontal depth-image slice. Loader: extend `visual_slam/orbslam/io/tum_rgbd.py`; synthesizer: new `slam_core/fusion/lidar_synth.py`.
12. **Sensor sync** — Soft sync with ±50 ms tolerance. If no LiDAR scan falls inside the window for a given RGB-D frame, that keyframe's signature carries `scan=None` and LiDAR-side verification is skipped for it.
13. **Memory tier defaults (configurable)** — Conservative for Jetson Nano: `stm_size=30`, `wm_cap=200`, `ltm_cap=1000`, `rehearsal_similarity=0.2`. Exposed as CLI flags.
14. **Graph optimization** — Reuse `carto/pose_graph/backends/g2o_backend_2d.py` unchanged. Wrap it in a new `FusionGraph` class (keyframe-only; not submap-aware).
15. **Loop-closure abstraction** — Reuse `slam_core/loop_closure.py` Protocols (`LoopVerifier`, `TargetProvider`, `ConstraintSink`). New verifiers (`ICPLoopVerifier`, `VisualLoopVerifier`) implement these Protocols.
16. **v1 scope only** — SQLite persistence, multi-session, ROS integration, ICP-as-LiDAR-loop-verifier are all v2/v3. Do not pre-build them.

## 3. Critical code paths — REUSE, do minimal modifications if it required for functionality but clearly document it

| Existing module | Why reuse it |
|---|---|
| `carto/pose_graph/backends/g2o_backend_2d.py` | The SE(2) g2o solver. Wrap with `FusionGraph`; do not re-write a solver. |
| `slam_core/loop_closure.py` | Generic `LoopClosureManager`, `LoopVerifier`, `TargetProvider`, `ConstraintSink` Protocols. New verifiers plug in here. |
| `slam_core/matching/scan_to_submap/` | LiDAR front-end (matcher + submaps). Use as a service in fusion modes. |
| `slam_core/matching/scan_to_map.py` | Alternative LiDAR front-end (`--lidar-frontend scan_to_map`). |
| `carto/local_slam/range_to_points.py` + `pose_extrapolator.py` | LiDAR scan preprocessing + odometry prediction. Same path as standalone. |
| `visual_slam/orbslam/io/tum_rgbd.py` + `rgbd_dataset.py` | TUM RGB-D loader. Extend, don't replace. |
| `visual_slam/orbslam/slam/keyframe_database.py` | ORB-SLAM's DBoW-style loop candidate detector. Poll in propose-only mode. |
| `visual_slam/orbslam/slam/loop_closing.py` | ORB-SLAM's loop closer. Wrap with an adapter that gates verification on a `propose_only` flag. Only add a single one-line gate inside this file if the adapter cannot intercept cleanly. |
| `slam_core/common/types.py` (`Pose2`) + `se2.py` | SE(2) algebra. All fusion poses are `Pose2`. |
| `slam_core/common/types3d.py` (`Pose3D`) | ORB-SLAM keyframe pose type. Project to `Pose2` at fusion boundary. |

## 4. Implementation phases — sequenced. Do NOT skip ahead.

Each phase ends with a runnable verification step. Do not start phase N+1 until phase N's verification passes and `FUSION_STATUS.md` shows DONE.

### Phase 1 — Foundation
- **Goal:** Stand up the `slam_core/fusion/` package skeleton with `Signature`, `Pose2 ←projection← Pose3D`, `SoftSync`, synthetic LiDAR utility, and the TUM RGB-D + synthetic-LiDAR dataset loader. No fusion behaviour yet.
- **Deliverables:** `slam_core/fusion/{__init__.py, config.py, signature.py, sync.py, lidar_synth.py, dataset.py}`; `tools/probe_fusion_dataset.py`.
- **Verify:**
  - `.venv/bin/pytest tests/fusion/test_checkpoint_p1_foundation.py -x -q`
  - `.venv/bin/python tools/probe_fusion_dataset.py --dataset datasets/tum/rgbd_dataset_freiburg1_room --num-frames 20`
- **Done when:** Probe prints sync-success rate and dumps 3 synthesized scans alongside their RGB frames; unit tests for `SoftSync` window logic + `Pose3D→Pose2` projection pass.
- **Reference:** `RTAB_inspired_implementation_plan.md` §13 Phase 1; §6.3 file list.

### Phase 2 — Memory tier
- **Goal:** Implement `MemoryManager` with STM, WM, in-RAM LTM, rehearsal merge, weight rule, and transfer policy faithful to RTAB-Map §3 of `RTAB_review.md`.
- **Deliverables:** `slam_core/fusion/memory.py`.
- **Verify:** `.venv/bin/pytest tests/fusion/test_checkpoint_p2_memory.py -x -q` — includes 500-keyframe synthetic stream that must exercise: STM aging, rehearsal merge (both ORB-similarity and ICP-fit paths), weight inheritance, oldest-of-lowest-weight transfer, LTM reactivation.
- **Done when:** All memory-tier invariants hold under the synthetic stream; tier sizes stay within configured bounds.
- **Reference:** `RTAB_review.md` §3.1–§3.6; `RTAB_inspired_implementation_plan.md` §8.

### Phase 3 — Fusion graph
- **Goal:** Implement `FusionGraph` wrapping the existing `G2oBackend2D`. Keyframe-only SE(2) graph with neighbour (spine) edges and a `add_loop_constraint` API.
- **Deliverables:** `slam_core/fusion/graph.py`.
- **Verify:** `.venv/bin/pytest tests/fusion/test_checkpoint_p3_graph.py -x -q` — toy graph (5 nodes, 1 synthetic loop) deforms as expected against a known-good g2o output.
- **Done when:** Optimizer converges; first node remains fixed; node poses update; mass-poses recoverable via `get_subgraph_for_optimization()`.
- **Reference:** `RTAB_inspired_implementation_plan.md` §9; `carto/pose_graph/backends/g2o_backend_2d.py`.

### Phase 4 — ICP verifier
- **Goal:** Implement `ICPLoopVerifier` using `small_gicp`. Implements the `LoopVerifier` Protocol from `slam_core/loop_closure.py`.
- **Deliverables:** `slam_core/fusion/icp_verifier.py`; `small_gicp` installed in venv (`.venv/bin/pip install small_gicp`).
- **Verify:** `.venv/bin/pytest tests/fusion/test_checkpoint_p4_icp.py -x -q` — self-loop test (scan ↔ itself + noise) returns identity; two overlapping TUM-derived scans align within tolerance.
- **Done when:** Verifier returns `success=True` with valid transform and info matrix on positive cases; returns `success=False` with `status="matcher_failed"` or `"score_failed"` on negative cases.
- **Reference:** `RTAB_inspired_implementation_plan.md` §10.1; `slam_core/loop_closure.py` Protocol definitions.

### Phase 5 — Visual verifier
- **Goal:** Implement `VisualLoopVerifier` that runs ORB descriptor matching + PnP RANSAC. Implements the same `LoopVerifier` Protocol.
- **Deliverables:** `slam_core/fusion/visual_verifier.py`.
- **Verify:** `.venv/bin/pytest tests/fusion/test_checkpoint_p5_visual.py -x -q` — two ORB feature sets from known-overlapping TUM frames return a transform within tolerance of ground truth.
- **Done when:** Verifier emits a valid loop constraint for positive cases; rejects unrelated frames with `status="score_failed"` or `"matcher_failed"`.
- **Reference:** `RTAB_inspired_implementation_plan.md` §10.2.

### Phase 6 — Adapters
- **Goal:** Wrap existing pipelines so they can be driven by the fusion runner as services and so their own loop-verification can be disabled in fusion modes.
- **Deliverables:** `slam_core/fusion/adapters/{__init__.py, orb_loop_proposer.py, lidar_loop_proposer.py, visual_frontend_service.py, lidar_frontend_service.py}`.
- **Verify:** `.venv/bin/pytest tests/fusion/test_checkpoint_p6_adapters.py -x -q` — each adapter, called independently on a short data run, returns the expected proposal/keyframe stream and proves that the upstream pipeline's own verifier did NOT run.
- **Done when:** ORB proposer emits candidates without ORB-SLAM emitting Sim(3) constraints; LiDAR proposer emits candidates without the LiDAR PGO solving.
- **Reference:** `RTAB_inspired_implementation_plan.md` §6.2; §15 risk row on the `propose_only` gate.

### Phase 7 — Mode A and Mode B (pass-through)
- **Goal:** Stand up `runner.py` skeleton with mode dispatch. Modes `orb` and `lidar` delegate to existing runners and must produce trajectory outputs byte-equal (or float-noise equal) to direct invocation.
- **Deliverables:** `slam_core/fusion/runner.py` (mode dispatch only, no fusion paths yet).
- **Verify:** `.venv/bin/pytest tests/fusion/test_checkpoint_p7_passthrough.py -x -q` — compares ATE of `--mode orb` vs `visual_slam/orbslam/run_rgbd_slam.py` direct, and `--mode lidar` vs `hector/run_local_slam_new.py` direct on a short TUM run.
- **Done when:** ATE delta ≤ float-noise threshold for both modes.
- **Reference:** `RTAB_inspired_implementation_plan.md` §7.1, §7.2.

### Phase 8 — Mode C end-to-end (Visual-main + LiDAR verifier)
- **Goal:** Wire the visual front-end service + memory + graph + ORB loop proposer + ICP verifier into the unified runner. End-to-end runs on TUM fr1_room with synthesized LiDAR.
- **Deliverables:** Mode-C path inside `runner.py`.
- **Verify:** `.venv/bin/pytest tests/fusion/test_checkpoint_p8_mode_c.py -x -q` — full run on fr1_room. Checks: ≥1 ICP-accepted cross-modal loop; ATE no worse than Mode A baseline + 5%; memory tier counts within configured caps.
- **Done when:** All three criteria pass; logs confirm ORB-SLAM's own verification did not fire.
- **Reference:** `RTAB_inspired_implementation_plan.md` §7.3.

### Phase 9 — Mode D end-to-end (LiDAR-main + Visual verifier)
- **Goal:** Symmetric counterpart to Phase 8 — LiDAR front-end + memory + graph + LiDAR loop proposer + Visual verifier.
- **Deliverables:** Mode-D path inside `runner.py`.
- **Verify:** `.venv/bin/pytest tests/fusion/test_checkpoint_p9_mode_d.py -x -q` — same shape as Phase 8 but for Mode D.
- **Done when:** ≥1 Visual-accepted cross-modal loop; ATE comparable to Mode B baseline; memory caps respected.
- **Reference:** `RTAB_inspired_implementation_plan.md` §7.4.

### Phase 10 — Map output
- **Goal:** Trajectory TUM writer + 2D occupancy grid assembler from accumulated scans transformed by optimized poses. Must redraw correctly after a late loop closure.
- **Deliverables:** `slam_core/fusion/map_output.py`; trajectory + grid emitted by every successful runner invocation.
- **Verify:** `.venv/bin/pytest tests/fusion/test_checkpoint_p10_output.py -x -q` — checks trajectory format conformance + occupancy grid PNG is non-empty + grid changes when a synthetic late loop is injected.
- **Done when:** Outputs land in `fusion_outputs/<run_id>/` with `trajectory.tum` + `occupancy.png`.
- **Reference:** `RTAB_inspired_implementation_plan.md` §13 Phase 10.

### Phase 11 — Hardening & docs
- **Goal:** Profile end-to-end on dev machine, write `slam_core/fusion/README.md`, add diagnostics summary at end of each run.
- **Deliverables:** `slam_core/fusion/README.md`; profile numbers in `FUSION_STATUS.md` open-follow-ups section.
- **Verify:** A clean run of Mode C on fr1_room reports per-keyframe runtime ≤ 100 ms on the dev machine; README explains the four modes and config flags.
- **Done when:** README is reviewable; profile artefact attached.
- **Reference:** `RTAB_inspired_implementation_plan.md` §13 Phase 11.

## 5. Validation tests — where they live, how they run

- **Location:** `tests/fusion/test_checkpoint_p<N>_<name>.py`
- **Convention:** Mirrors the existing `tests/visual_slam/orbslam/test_checkpoint_2_X_*.py` pattern. Each test file maps 1:1 to a phase above.
- **Run a single phase test:**
  - `.venv/bin/pytest tests/fusion/test_checkpoint_p<N>_<name>.py -x -q`
- **Run all fusion tests:**
  - `.venv/bin/pytest tests/fusion/ -x -q`
- **Discovery sanity check:**
  - `.venv/bin/pytest tests/fusion/ --collect-only -q`

## 6. Phase progress tracking — `FUSION_STATUS.md`

This file at the workspace root is the **single source of truth** for "where are we now". It survives across conversations.

- **Read at session start.** Before doing anything else, read `FUSION_STATUS.md` to learn the current phase.
- **Update after each phase passes its verification.** Mark the phase row: `Status = DONE`, `Completed at = <ISO UTC>`, `Commit = <git short hash>`, `Notes = <one-line summary>`.
- **If a phase is mid-implementation, mark `Status = IN PROGRESS`** so a different conversation knows not to restart from scratch.
- **Open follow-ups section** at the bottom — anything you deferred (e.g., "ICP info-matrix using overlap heuristic, refine in Phase 11") goes here.

## 7. End-of-implementation completion criteria

The v1 build is "done" only when ALL of the following hold (verbatim from `RTAB_inspired_implementation_plan.md` §14.2):

| # | Acceptance check | How verified |
|---|---|---|
| 1 | Mode A trajectory matches standalone ORB-SLAM | ATE within float-noise of `run_rgbd_slam.py` output |
| 2 | Mode B trajectory matches standalone LiDAR | ATE within float-noise of `run_local_slam_new.py` output |
| 3 | Mode C accepts ≥1 cross-modal loop on fr1_room | Log shows ≥1 ICP-verified loop constraint emitted |
| 4 | Mode D accepts ≥1 cross-modal loop on fr1_room | Log shows ≥1 Visual-verified loop constraint emitted |
| 5 | Mode C / D ATE no worse than baseline + 5% | Reported ATE table at end of run |
| 6 | Memory tiers behave per spec | Diagnostic log shows STM size bounded, WM cap respected, LTM growth tracked |
| 7 | Live signature count ≤ STM + WM + LTM caps | Periodic memory diagnostics stay under documented total |
| 8 | Runtime per keyframe on dev machine ≤ 100 ms | Profile printed at end-of-run |

Jetson-target runtime verification is explicitly v3 and NOT part of v1 acceptance.

## 8. Risk register — consult before touching the affected subsystem

| Risk | When it matters | Mitigation |
|---|---|---|
| `small_gicp` Python bindings fail to build on Jetson Nano (aarch64) | Phase 4; Jetson deployment | Verify build on dev x86_64 in Phase 4; if Jetson build fails, v2 fallback to NumPy point-to-point ICP (acceptable because verification runs at ~1 Hz, not per scan) |
| Disabling ORB-SLAM's loop verification cleanly without editing `loop_closing.py` is harder than expected | Phase 6 | If the adapter pattern cannot intercept cleanly, add a single `propose_only` config flag in `loop_closing.py` gating the verifier call. Default `False` preserves standalone behaviour. |
| Synthesized 2D LiDAR from TUM depth slice is unrepresentative | Phase 1; Phase 8 | Tunable noise model in `lidar_synth.py`; cross-check with real LiDAR + RGB-D handheld run if available before Phase 8 |
| SE(2) projection of ORB-SLAM SE(3) loses info on uneven floors | Phase 8 ATE measurement | Document assumption; in Phase 8 measure ATE delta vs Mode A baseline; if regression > 10%, escalate to v2 SE(3) backend |
| Memory tier overhead exceeds Jetson Nano headroom | Phase 11 profiling | v1 acceptance is dev-machine only; conservative defaults; document `--stm-size`, `--wm-cap`, `--ltm-cap` flags for tuning |
| Four-mode runner becomes a maintenance burden | Across all fusion phases | Modes A and B are thin pass-throughs (≤30 lines each); Modes C and D share ~80% of code via the `LoopVerifier` Protocol abstraction |

## 9. Deferred to v2 / v3 (do NOT build now)

These are intentionally out of v1 scope. If a future request seems to require any of them, stop and ask the user — do not silently expand scope:

- SQLite persistence for LTM; multi-session resumption.
- ICP as a LiDAR-pipeline loop verifier (alongside the existing B&B).
- Compression of Signature payload (zlib + cv2.imencode).
- ROS integration on the Jetson (sensor input, TF tree).
- Dense 3D point cloud / OctoMap outputs.
- A new appearance-based BoW + Bayes filter from scratch.
- Real-time visualization in fusion modes (the existing `run_realtime_viz.py` is untouched).
