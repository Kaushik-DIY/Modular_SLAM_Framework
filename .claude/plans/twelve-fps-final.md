# Plan: Final step — push lab visual-SLAM to **12+ fps** (close the Python boundary in the hot loop)

> Status: PROPOSED (2026-06-05). This is the execution plan for milestones **M1→M5** of the
> authoritative roadmap in `nifty-zooming-starlight.md`, refined to the **12 fps** target and the
> current committed baseline (M0 reproducibility, M6 loop-closure quality, and the tracking lever
> are all DONE). All safety/dependency/discipline rules from that roadmap remain in force here
> (flag-gated `USE_CPP_CORE`, pySLAM as algorithm-only reference, no package version changes without
> the repo-wide audit + explicit confirmation, Python hot path never deleted).

---

## 1. Target & definition of done

- **12+ fps sustained** on `datasets/lab_rgbd_run_2` (4494 frames) in the **threaded** deployment
  config = **steady-state per-frame tracking ≤ 83 ms** with local mapping overlapping in its own
  thread (the rate is gated by tracking latency; LM must not stall it).
- **Reproducible**: same config → trajectory within a tight ATE band (sequential mode stays
  bit-exact for parity checks). No regression of the M6 loop-closure quality (68 mm ATE-vs-reference).
- **The post-loop GBA is measured separately** — it is a rare background event (5 firings over the
  whole lab run), not part of the steady-state per-frame budget. It must run async (GIL-released)
  so it does not block tracking; its merge-back cost is reported but excluded from the fps gate.
- **Done when:** a threaded `USE_CPP_CORE=True` full-lab run reports steady-state tracking
  ≤ 83 ms/frame (≥12 fps) on the dev machine, suite green flag-ON and flag-OFF, lab ATE within band
  vs the previous milestone, and Mode-C fusion smoke still accepts ≥1 cross-modal loop.

## 2. Current baseline (committed HEAD `2dc8978`)

| Fact | Value | Source |
|---|---|---|
| Tracking lever applied | `kNumBestCovisibilityKeyFramesTracking = 3` | config_parameters.py:87 |
| Loop-closure quality | 68 mm ATE-vs-reference (loops+GBA, θ=15) | M6 |
| Non-KF frame rate (loops-off) | **~12 fps (~83 ms)** | prior profiling |
| Mean frame rate (loops-off) | **~5.8 fps** — dragged down by KF-frame stalls | prior profiling |
| Mean fps loops+GBA (t15_GBA run) | 4.77 fps (includes 5 GBA stalls) | run_summary.json |
| **The KF-frame stall root** | LM calls the **Python** matcher `search_and_fuse` under the GIL | local_mapping_core.cpp:218,250 |
| C++ KeyFrame | built + unit-tested on 3.11.9, **dormant** (not subclassed) | keyframe.cpp |

**F0 below re-profiles this on current HEAD to produce the authoritative per-stage breakdown before
any C++ work — the numbers above are the working hypothesis, F0 makes them exact.**

## 3. Root cause: two independent ceilings (both must fall for sustained 12 fps)

1. **KF-frame stall (the dominant one).** On a keyframe frame the tracking thread blocks because
   local mapping's `fuse_map_points` / `create_new_map_points` call back into the **Python** matcher
   (`search_and_fuse`, local_mapping_core.cpp:218/250). That callback holds the GIL, so the
   "threaded" LM does **not** actually overlap tracking — the pipeline serializes (~700 ms stalls),
   pulling the mean to ~5.8 fps even though non-KF frames already hit ~12 fps. **Fixing this is what
   turns ~12 fps non-KF into ~12 fps sustained.**
2. **Tracking floor + margin.** Steady-state tracking is ~83–105 ms because the inner loop is still
   half-Python: the local map is a Python list of ~4118 `py::object` MapPoints, each cast to
   `MapPoint*` per frame, the Frame is a synced mirror, and matches cross the py boundary. To hold
   ≥12 fps **with margin** (not riding the 83 ms edge), the tracking floor needs to drop to ~65–75 ms
   by running the whole inner loop on C++ objects.

Both share one prerequisite: **wire the dormant C++ KeyFrame** so C++ MapPoints, the local-map
vector, and the LM matcher all operate on C++ objects without per-object marshalling.

## 4. Strategy

Close the Python boundary progressively, each step `USE_CPP_CORE`-gated, pySLAM-referenced, and
A/B-validated (parity + measured speedup) before the next. Order is chosen so the **highest-impact,
lowest-new-risk** lever (the LM stall, F4) lands as soon as its prerequisite (C++ KeyFrame, F1) is in.

```
F0  Re-baseline profile (no code)         → authoritative per-stage + bimodal split
F1  Wire C++ KeyFrame  (M1b→M1d)          → prerequisite for F2/F3/F4  [HIGHEST RISK]
F2  C++ local-map build (M2)              → tracking floor ↓ (search_map gets a C++ vector)
F3  C++ tracking_core   (M3)              → tracking floor ↓ (remove residual orchestration)
F4  C++ LM matcher callback, GIL-released (M4) → kills the KF-frame stall → LM overlaps
F5  Measure vs 12 fps; lean only if short (M5) → local-map size lever, last resort float64-kd
```

F4 is the single biggest fps win (removes the stall); F2+F3 buy the margin; F1 unlocks all three.

## 5. Milestones

### F0 — Re-baseline profile (no code; ~20 min)
- **Do:** threaded full-lab run on current HEAD, loops-OFF (clean steady state) **and** a second
  loops-ON+GBA run, with per-frame + per-stage timing dumped (`frame_timing.csv`, stage profiler).
- **Output:** authoritative table — non-KF ms, KF-frame ms, the LM-stall duration, per-stage split
  (`search_map`, `track_previous_frame`, `local_map_build`, pose-opt), and the GBA merge-back cost.
  This sets the exact budget every later milestone is measured against.
- **Verify:** numbers reproduce across two runs (within band). No code → no parity risk.

### F1 — Wire the C++ KeyFrame (linchpin; class already built — this is wiring + de-risking)
Make Python `KeyFrame` subclass `_CppKeyFrameBase` (exact `MapPoint` precedent), delegating
covisibility graph + spanning tree + loop edges + point management to C++; keep Python-only bits
(BoW via `keyframe_database`, image/depth, serialization). **Sub-milestones, each flag-gated + suite
+ short lab A/B — do NOT do at once:**
- **F1a** (prep DONE, commit 31bcfc8): kpsu representation guards so consumers accept C++ kps.
- **F1b**: subclass + construction (`init_feature_arrays`, avoid readonly octaves/kps_ur) + identity/
  pose/points delegation; covisibility (`add_connection`/`update_connections`/`get_*covisible*`/
  `get_weight`) → C++ (deterministic, weight-ordered — already in keyframe.cpp).
- **F1c**: spanning tree (`set_parent`/`add_child`/`get_parent`) + loop edges → C++.
- **F1d**: `get_matched_good_points` / `num_tracked_points` / `set_bad` → C++.
- **Validate every consumer** (grep counts: `is_bad` 100, `get_matched_good_points` 17,
  `update_connections` 11, `get_best_covisible_keyframes` 9, `set_bad` 6, `get_parent` 5,
  `add_loop_edge` 5, `num_tracked_points` 4) across tracking, local_mapping, loop_closing, **and the
  fusion adapters** (`slam_core/fusion/orbslam_frontend.py`).
- **Files:** `third_party/cpp_slam_core/src/{keyframe.cpp,.h,bindings.cpp}` (additive),
  `visual_slam/orbslam/slam/keyframe.py` (subclass + dispatch), `config_parameters.py`.
- **Verify:** suite green flag-ON+OFF; lab loops-off A/B within band (bit-exact sequential);
  Mode-C fusion smoke passes. **Risk:** HIGHEST (cross-KF refs, graph mutation, ~10 consumers) —
  mitigated by the incremental sub-steps + the proven MapPoint pattern + the one-line flag fallback.

### F2 — C++ local-map build (M2)
- Port `LocalMap.update` (map.py:159) → C++ `LocalCovisibilityMap` returning a **C++ MapPoint
  vector** (id/weight-ordered, deterministic). Dispatch from `map.update_local_map`. pySLAM ref:
  map.cpp `LocalCovisibilityMap::update` / `get_best_neighbors`.
- Now `search_map` receives a C++ vector — no py::list + per-point cast → the ~93 ms inner loop drops.
- **Verify:** `local_map_build` ~39 ms → target ≤ 12 ms; lab A/B within band; measured tracking drop.

### F3 — C++ tracking_core (M3)
- Port the per-frame orchestration still in Python — `track_local_map` loop,
  `count_tracked_and_non_tracked_close_points`, `propagate_map_point_matches`, pose-opt driver —
  into `cpp_slam_core` `tracking_core`, driving search_map/search_frame on C++ objects.
  pySLAM ref: tracking_core.cpp.
- **Verify:** tracking inner loop pure C++; measured steady-state tracking toward ≤ 75 ms; A/B band.

### F4 — Close the LM matcher callback, GIL-released (M4) — **the KF-stall fix**
- Make C++ `LocalMappingCore` call the **C++** matcher (`search_more_map_points_by_projection` /
  `search_and_fuse` + map-point creation) instead of the Python callback at
  local_mapping_core.cpp:218/250. Wrap the LM matcher body in `py::gil_scoped_release` (the SOC BA
  solve already does this) so the LM thread runs **truly parallel** to tracking, with proper
  map-locking around mutations.
- **Effect:** keyframe frames no longer stall — LM digests the new KF in the background; the mean
  rate converges to the non-KF rate. This is the milestone that makes ~12 fps *sustained*.
- **Verify:** F0's ~700 ms KF-stall → ≪ tracking time; mean fps approaches non-KF fps; lab A/B band;
  suite green; no map-corruption / race (sequential bit-exact check still holds).

### F5 — Measure vs 12 fps; lean only if short (M5)
- Re-profile threaded tracking. **If ≤ 83 ms steady state → 12 fps reached; stop.** If still short,
  the only remaining lever is local-map **size** (~4118 pts) — approach carefully (matcher-entangled;
  the throttle is NOT the lever, proven 2026-06-04): denser-covisibility-driven KF culling (now
  enabled by F1's deterministic C++ covis graph) or `kNumBestCovisibilityKeyFrames`/point-culling
  tuning — each validated for ATE + the 60-lost tracking baseline before keeping. Last resort:
  float64 kd-tree only if reproducibility needs it.

- **Map-density reduction (the "voxel filter" ask, 2026-06-06) lands HERE, not earlier.** The current
  map carries **3.5× the points (32,698 vs 9,306) and 1.66× the keyframes (253 vs 152)** of the clean
  reference for the same 8 m scene — real redundancy from KF over-insertion (the same M6 root cause).
  Reduce it with the **loop-safe, ORB-SLAM-native** mechanisms — tune the existing `cull_keyframes`
  (local_mapping.py:460 / local_mapping_core.cpp:310 redundancy rule) and `cull_map_points`
  (local_mapping.py:457) toward the reference regime — **NOT** a pose-voxel keyframe dedup. *A
  pose-voxel filter that drops spatially-close keyframes would delete exactly the revisit keyframes
  loop closure depends on, regressing M6 — explicitly forbidden.* Why F5 (after the port), not before:
  (1) F1–F4 are bit-parity ports validated by "ATE within band vs previous milestone" — changing map
  density mid-port moves the reference and confounds port bugs with filter effects; (2) the port is
  free (same accuracy) while density reduction spends the 60-lost / ATE budget — only worth spending
  if F1–F4 don't already reach 12 fps; (3) it must not re-disturb the just-fixed M6 PGO. Each density
  change validated for ATE-vs-reference (≤68 mm regime) + lost-frame count + loop-closure count before
  keeping. **Separable & harmless anytime:** a *cosmetic* voxel downsample of the saved `.ply` (output
  only, never the live tracking map) — out of scope for the fps gate; add on request.

## 6. Verification harness (every milestone)
- `USE_CPP_CORE` **OFF**: full `visual_slam/orbslam` suite green (500 passed, 1 skipped) — fallback intact.
- `USE_CPP_CORE` **ON**: suite green; lab loops-off A/B vs previous milestone within ATE band;
  **measured per-frame tracking recorded** (a milestone is not done without a real speedup).
- Reproducibility: two identical runs match within band (sequential = bit-exact).
- Fusion: Mode-C smoke still accepts ≥1 cross-modal loop, ATE not worse than baseline.
- Build/ABI sanity before each milestone's tests: rebuild `cpp_slam_core` against the venv (back up
  the `.so` first), confirm import + existing C++ objects still work (no ABI break).

## 7. Risk register
- **F1 (C++ KeyFrame) — HIGH.** Cross-KF references, graph mutation, ~10 consumers, fusion adapters.
  *Mitigate:* class already built+unit-tested; follow the MapPoint subclass precedent; F1b→F1d
  incrementally with suite + lab A/B after each; `USE_CPP_CORE=False` one-line fallback; Python
  KeyFrame never deleted.
- **F4 threading races.** True LM/tracking parallelism needs correct map-locking. *Mitigate:* lock
  granularity mirrors pySLAM; sequential bit-exact run is the race oracle; reproducibility is a band
  in threaded mode by design.
- **Build/ABI standing suspects:** `cpp_slam_core.so` links system OpenCV 4.5 vs venv 4.10; pybind
  3.0.4 (never copy pySLAM's 2.13.1 glue); numpy ABI embedded. Rebuild + re-verify each milestone.
- **12 fps is a stretch vs the roadmap's 10 fps.** It is reachable *only* with the full F1→F4 chain
  (F4 removes the stall, F2+F3 buy the margin) — not by tuning alone. If F5 shows the floor cannot
  reach 83 ms with margin, escalate the local-map-size lever (F5) with explicit ATE guards.

## 8. Expected fps math (hypothesis; F0 confirms, F5 measures)
- Today: non-KF ~83 ms (~12 fps), KF frames stall ~700 ms → mean ~5.8 fps.
- After **F4** (stall removed): mean → non-KF rate ≈ ~12 fps sustained, *but* riding the 83 ms edge.
- After **F2+F3** (tracking floor 83→~70 ms): steady state ~14 fps → **≥12 fps with margin.** ✓
- GBA: 5 background firings; async GIL-released so it does not gate steady-state fps (reported separately).
