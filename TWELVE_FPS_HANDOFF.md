# 12-FPS C++ Port — Implementation Handoff (for Codex / next agent)

> **Read this whole document before touching code.** It is self-contained: it assumes
> you have *no* prior conversation context. It tells you the goal, what is already done
> (with commit hashes), the exact remaining work, how to build/run/validate, the
> non-obvious gotchas, and the working discipline. Follow it carefully and systematically.

---

## 0. TL;DR — where we are and what to do next

- **Goal:** reach **≥12 fps** (really ≥10 fps is the documented gate) on the lab dataset
  `datasets/lab_rgbd_run_2` (4494 RGB-D frames) for the ORB-SLAM RGB-D pipeline, in the
  **threaded** deployment config, by porting the per-frame hot path to C++ — *without*
  regressing accuracy. Everything is behind a runtime flag so the proven system is never at risk.
- **Return-to commit (safe baseline):** `d0c774f` on branch **`orb-cpp-port`**. If anything
  you do goes wrong, you can always `git checkout d0c774f` and rebuild to get back to a
  known-good, fully-validated state.
- **What is DONE:** F0 (baseline), F1 (C++ KeyFrame wired), F2 (C++ local-map build), and the
  **F4 deadlock-class invariant** (commits `a9fb879`, `6f3f855`, `d0c774f`). Since this handoff
  was first written, native local-mapping fuse (`550b6b3`) and native triangulation epipolar
  matching (`4b29341`) were also added and validated.
- **Current full-run reference:** checkpoint
  `visual_slam/reference_audit/checkpoint_2_39_full_lab_vectorized_triang_threaded_20260608/`
  completed all 4494 lab frames at commit `25448ed` with `tracking_lost_count=60`,
  `final_state=OK`, and `avg_fps=6.18`. This improves over checkpoint 2.38's 6.03 FPS and
  reduces the monitored tracking-loss count from 76 to 60 frames. Tracking loss remains a
  `MONITOR` flag: raise it to high priority only if later full lab runs grow materially above this
  range; otherwise revisit after the remaining runtime-efficiency porting.
- **Latest full-run implementation step:** native combined tracking local-map path:
  `build_local_map -> mark_current_frame_matched_points_seen -> search_map_by_projection` in one
  C++ binding call, with the no-vote reference-keyframe fallback preserved in Python. Short
  validation: targeted tests passed, full suite passed, sequential 600-frame profiled lab run had
  0 lost / 6.04 FPS, threaded 600-frame profiled lab run had 0 lost / 7.68 FPS, and the full
  threaded lab run reached 6.03 FPS with `tracking.track_local_map` averaging 24.0 ms.
- **Latest code milestone after that:** `a0299bc` (`F4b: reduce local mapping fuse Python dispatch`)
  tightened `LocalMappingCore::fuse_map_points` around the existing native `search_and_fuse`
  kernel. C++ keyframes now use native covisibility lookup when no Python `local_map` neighbor
  provider is present, native matched-point collection, native bad-state checks, native observation
  membership checks, and native `update_info` / `update_connections` calls. Validation passed:
  focused LocalMappingCore tests, F1/F2 wiring tests, F4 invariant audit, full ORB-SLAM suite
  (`514 passed, 1 skipped`), and a 600-frame threaded lab smoke (`600/600 OK`, `0` lost,
  `final_state=OK`, `avg_fps=6.29`). This is a stability-preserving cleanup, not the main FPS
  unlock; `local_mapping.fuse_map_points` still averages about 164 ms in the 600-frame threaded
  smoke.
- **Latest helper-level milestone:** `5ecbfca` (`F4b: use native keyframe point helpers`) moved
  shared C++ `KeyFrame` helpers used by local mapping away from Python `attr()` dispatch where the
  referenced objects are native C++ objects. `update_connections`, `get_matched_good_points`,
  `get_matched_good_points_and_idxs`, and `num_tracked_points` now use direct `MapPoint` /
  `KeyFrame` calls with Python fallback preserved. Validation passed: F4 invariant audit, focused
  LocalMappingCore/KeyFrame tests (`32 passed`), full ORB-SLAM suite (`514 passed, 1 skipped`),
  and a 600-frame threaded lab smoke (`600/600 OK`, `0` lost, `final_state=OK`, `avg_fps=6.30`).
  In that smoke, `local_mapping.process_new_keyframe` averaged about 9.4 ms and
  `local_mapping.fuse_map_points` averaged about 154 ms; the larger remaining costs are still
  `local_mapping.local_BA` and `local_mapping.create_new_map_points` / fuse orchestration.
- **Latest LocalMappingCore milestone:** `d4db2ce` (`F4b: use native local mapping process and
  cull calls`) removed another layer of Python dispatch from
  `LocalMappingCore::process_new_keyframe` and `cull_map_points` for native C++ objects.
  Observation insertion, `update_info`, found-ratio checks, observation counts, first-KF lookup,
  and `set_bad` now use direct C++ calls where possible, with Python fallbacks preserved.
  Validation passed: F4 invariant audit, focused LocalMappingCore/KeyFrame tests (`32 passed`),
  full ORB-SLAM suite (`514 passed, 1 skipped`), and a 600-frame threaded lab smoke
  (`600/600 OK`, `0` lost, `final_state=OK`, `avg_fps=6.32`). In that smoke,
  `local_mapping.process_new_keyframe` averaged about 7.5 ms and
  `local_mapping.cull_map_points` about 3.2 ms. The larger remaining costs are still
  `local_mapping.local_BA` (~713 ms/keyframe) and `local_mapping.create_new_map_points`
  (~245 ms/keyframe) in this short threaded profile.
- **Latest triangulation milestone:** `bac801e` (`F4b: vectorize local mapping triangulation
  mask`) vectorized `triangulate_normalized_points` after `cv2.triangulatePoints`, replacing the
  Python per-match homogeneous normalization / finite / positive-depth loop with NumPy array
  operations while preserving output shape and zero-filled invalid points. Validation passed:
  targeted local-mapping tests (`30 passed`), full ORB-SLAM suite (`514 passed, 1 skipped`),
  explicit parity check against the old loop formula, and a 600-frame threaded lab smoke
  (`600/600 OK`, `0` lost, `final_state=OK`, `avg_fps=7.18`). In that smoke,
  `local_mapping.create_new_map_points` averaged about 194 ms, `local_mapping.fuse_map_points`
  about 140 ms, `local_mapping.local_BA` about 604 ms, and `frame.total` about 139 ms.
  This is the best short-run threaded FPS seen in this F4b cleanup series, but a full lab run is
  still needed before treating it as the new full-run reference.
- **Latest full-run validation + visualization:** checkpoint 2.39 full lab generated
  `VALIDATION_REPORT.md`, `VISUAL_COMPARISON_REPORT.md`, baseline comparison plots under
  `plots_vs_baseline/`, per-run plots under `plots/`, and sparse/semi-dense map figures under
  `map_figures/`. Full-run runtime profile: `tracking.track_local_map` mean 23.8 ms,
  `slam.track` mean 83.8 ms, `frame.total` mean 160.8 ms,
  `local_mapping.create_new_map_points` mean 110.4 ms, `local_mapping.fuse_map_points` mean
  142.5 ms, and `local_mapping.local_BA` mean 606.5 ms. Baseline-aligned trajectory RMSE vs
  `visual_slam_outputs/lab_rgbd_run_2_B_loop_gba` is about 0.236 m, improved from checkpoint 2.38's
  about 0.294 m. Because loop closing/GBA were disabled, the loop+GBA baseline remains visually
  cleaner and is still the quality target.
- **What is NOT done:** the full dataset is still well below the 10-12 FPS goal. The next high-value
  work is reducing `tracking.track_local_map` growth over the full run, most likely by continuing
  F3-style tracking-core porting (`build_local_map` -> mark seen -> projection search -> pose-opt
  orchestration) and by further reducing Python/GIL work in local mapping/BA write-back.

---

## 1. Non-negotiable operating rules

1. **Python only via the project venv.** Use `.venv/bin/python` and `.venv/bin/pytest`. Never
   bare `python`/`python3`/`pip`/`pytest`. Venv = `/home/kaushik/slam_ws/.venv` (Python 3.11.9).
2. **Flag-gate everything; default OFF = the proven system.** The C++ KeyFrame path is selected
   at import time by `Parameters.USE_CPP_KEYFRAME` (env `SLAM_USE_CPP_KEYFRAME=1`). The C++
   matcher path is `Parameters.USE_CPP_CORE` (set to `True` by `tools/run_lab_cpp.py`). With both
   off you get today's pure-Python system. Never change default behaviour.
3. **pySLAM is the algorithm reference, not to be copied verbatim.** Reference sources live at
   `third_party/pyslam_reference/pyslam/slam/cpp/{map_point,keyframe,geometry_matchers,
   tracking_core,local_mapping_core}.cpp`. Port the *logic*; rewrite the pybind glue against our
   **pybind11 3.0.4** (pySLAM pins 2.13.1 — do NOT copy its bindings). Deviate only where
   mandatory for our environment and **document every deviation in the commit message**.
4. **No package version changes** without a repo-wide audit + explicit user confirmation. Adapt
   C++ to our existing versions (pybind11 3.0.4, numpy 1.26.4, system Eigen 3.4, OpenCV 4.10 in
   the venv but the existing `.so` links system OpenCV 4.5 — a standing suspect, see §8).
5. **Per-milestone validation = PARITY first, then speed.** Each increment must (a) keep the
   flag-off suite green, (b) be behaviour-preserving flag-on (see §7 — note flag-on is *not*
   bit-identical run-to-run; use median≈0 vs the prior build, not exact equality), and (c) only
   then show a measured speedup. Do not advance on speed alone.
6. **Systematic debugging.** On any blocker: find the EXACT root cause with evidence (stacks,
   `/proc`, gdb, profiles — not guessing), compare to how pySLAM handles it, then fix cleanly.
   Do not "fix by class" on a hunch when you can capture ground truth.
7. **Commit each validated milestone** with the trailer
   `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Small, revertible commits.

---

## 2. The single most important architectural fact

**The entire flag-on object hot path currently runs GIL-HELD, so the GIL serializes it.**

- `cpp_slam_core.search_map_by_projection` and `cpp_slam_core.build_local_map` are bound
  **without** `gil_scoped_release` → they hold the GIL.
- The local-BA write-back (`visual_slam/orbslam/slam/slam_optimizer_bridge.py::unpack_local_ba`)
  is pure Python → GIL-held. (The g2o *solve* itself, in the SOC optimizer
  `third_party/slam_optimizer_core`, is already GIL-released.)
- All C++ `KeyFrame`/`MapPoint` methods are invoked **from Python** through a trampoline named
  `_w` (a Python-3.11 pybind workaround in `cpp_slam_core` bindings — it is a pure pass-through,
  NOT a lock). A thread "at `_w`" is actually executing the C++ method, GIL-held.

**Consequences you must internalize:**
- Because everything is GIL-serialized, **local mapping (LM) currently BLOCKS tracking** — they
  cannot run at the same time. That is why threaded flag-on is only ~5.3 fps even though tracking
  itself got faster (F2). LM is ~1–2 s per keyframe (the cost of driving C++ KeyFrame objects
  from Python through `_w` × thousands of calls).
- The **fps gate is per-frame *tracking* latency** (≤100 ms ⇒ ≥10 fps). In threaded mode LM is
  supposed to overlap tracking in its own thread. It can only overlap if the heavy LM work runs
  **GIL-released**. Today it doesn't ⇒ no overlap ⇒ slow.
- **The win = make the LM matcher run GIL-released** (so LM overlaps tracking). But the moment
  you release the GIL, any C++ object method that touches Python (a `py::object`/GIL op) or another
  object's mutex *while holding its own mutex* becomes a live deadlock. That is exactly why the
  **F4 invariant (§5) was done first** — it is the *prerequisite* that makes GIL-release safe.

---

## 3. Code map (the files you will touch)

| Area | File | Role |
|---|---|---|
| C++ MapPoint | `third_party/cpp_slam_core/src/map_point.cpp/.h` | observations, positions, descriptors. **Invariant-clean (done).** |
| C++ KeyFrame | `third_party/cpp_slam_core/src/keyframe.cpp/.h` | covisibility graph, spanning tree, points. **Invariant-clean (done).** |
| C++ matchers | `third_party/cpp_slam_core/src/geometry_matchers.cpp/.h` | `search_map_by_projection`, `search_frame_by_projection`, `build_local_map` (F2). |
| C++ local mapping | `third_party/cpp_slam_core/src/local_mapping_core.cpp` | **`LocalMappingCore` — 426 lines, ENTIRELY `py::object`-driven. THE target for the GIL-release port (§6).** |
| C++ bindings | `third_party/cpp_slam_core/src/bindings.cpp` | pybind glue (pybind11 3.0.4 idiom — use this as the template). |
| Build | `third_party/cpp_slam_core/CMakeLists.txt` + `build/` | see §4. |
| Py tracking | `visual_slam/orbslam/slam/tracking.py` | `update_local_map` (~L730, dispatches to C++ `build_local_map` flag-on), `track_local_map`. |
| Py local mapping | `visual_slam/orbslam/slam/local_mapping.py` | `step()`, `fuse_map_points()` (→ `LocalMappingCore.fuse_map_points`), `create_new_map_points()`. |
| Py matcher dispatch | `visual_slam/orbslam/slam/geometry_matchers.py` | routes `ProjectionMatcher.search_*` to C++ when `USE_CPP_CORE`. |
| BA bridge | `visual_slam/orbslam/slam/slam_optimizer_bridge.py` | `unpack_local_ba` (BA write-back; pure Python). |
| Flags | `visual_slam/orbslam/slam/config_parameters.py` | `USE_CPP_CORE`, `USE_CPP_KEYFRAME`. |
| Runner | `tools/run_lab_cpp.py` | sets `USE_CPP_CORE=True`, arms SIGUSR1 faulthandler dump; the canonical way to run lab. |
| pySLAM ref | `third_party/pyslam_reference/pyslam/slam/cpp/*.cpp` | algorithm reference only. |

---

## 4. How to build, install, run, and test (exact commands)

### Build the C++ extension after editing any `src/*.cpp/.h`
```bash
cd /home/kaushik/slam_ws
cmake --build third_party/cpp_slam_core/build -j4
# install the freshly built .so into the ACTIVE venv (this is the import location):
cp third_party/cpp_slam_core/build/cpp_slam_core.cpython-311-x86_64-linux-gnu.so \
   .venv/lib/python3.11/site-packages/cpp_slam_core.cpython-311-x86_64-linux-gnu.so
# sanity:
.venv/bin/python -c "import cpp_slam_core as c; print(hasattr(c,'build_local_map'), hasattr(c,'search_frame_by_projection'))"
```
> ⚠️ There is a STALE copy at `.venv_rc1_backup/lib/.../cpp_slam_core*.so` that **lacks**
> `search_frame_by_projection`. Do NOT grab it with `find ... | head -1`. The ACTIVE `.so` is the
> one under **`.venv/`** (not `.venv_rc1_backup/`).
> Back up the working `.so` before risky rebuilds: `cp <active .so> /tmp/cpp_good.so` (revert tier 3).

### Run the lab dataset (the benchmark)
```bash
# sequential (deterministic-ish), loops off, short:
SLAM_USE_CPP_KEYFRAME=1 .venv/bin/python tools/run_lab_cpp.py datasets/lab_rgbd_run_2 \
  --output /tmp/run_seq --max-frames 600 --deterministic --disable-loop-closing --print-every 100

# THREADED (the fps-relevant config — LM in its own thread):
SLAM_USE_CPP_KEYFRAME=1 .venv/bin/python tools/run_lab_cpp.py datasets/lab_rgbd_run_2 \
  --output /tmp/run_thr --max-frames 1500 --start-local-mapping-thread --disable-loop-closing \
  --deterministic --print-every 300

# flag-OFF baseline (proven system): omit SLAM_USE_CPP_KEYFRAME (USE_CPP_CORE still on via runner).
```
The run prints a `RUN SUMMARY` (frames, keyframes, lost count, **avg_fps**, output paths). The
trajectory is `trajectory*.txt` (TUM-style: `timestamp tx ty tz qx qy qz qw`).

### Tests
```bash
# F1/F2 isolation + wiring tests (insert their own sys.path, run directly):
.venv/bin/pytest tests/visual_slam/orbslam/test_f2_local_map.py \
  tests/visual_slam/orbslam/test_f1_keyframe_cpp_wired.py \
  tests/visual_slam/orbslam/test_f1_keyframe_construction.py -q

# FULL flag-off regression suite (MUST add PYTHONPATH or imports fail):
PYTHONPATH=/home/kaushik/slam_ws .venv/bin/python -m pytest tests/visual_slam/orbslam/ -q
# Expected: 510 passed, 1 skipped.
```

---

## 5. The F4 invariant (DONE — `a9fb879`, `6f3f855`, `d0c774f`) — understand it, keep it

**Invariant:** *No `MapPoint`/`KeyFrame` C++ method may hold a C++ mutex
(`_lock_features` / `_lock_pos` / `_lock_connections`) while making a Python/GIL call
(`obj.attr(...)`, a `py::object` map-key INCREF, etc.) or while calling another object's locked
method.* This is pySLAM's discipline (pySLAM does all under-lock work in pure C++ via
`KeyFramePtr`/`FramePtr` keys and `keyframe->is_stereo_observation()`); our fork keys those maps
by `py::object`, so we must move every GIL/foreign call OUT from under the lock (snapshot under
the lock → do the GIL work outside → re-acquire to apply).

Already fixed (do not regress): `MapPoint::observations/keyframes` (sort outside lock),
`MapPoint::remove_observation` (kps_ur + kf_ref recompute outside lock),
`KeyFrame::update_connections` (cross-KF `add_connection_no_lock_` + `parent.add_child` outside
lock), `KeyFrame::set_bad` (parent `Twc`/`erase_child` outside lock). `set_bad`/`replace_with`/
`update_best_descriptor`/`update_normal_and_depth` already followed the pattern.

**Audit you must re-run after ANY change to these two files** (must print CLEAN for both):
```bash
.venv/bin/python - <<'PY'
import re
for f in ["third_party/cpp_slam_core/src/map_point.cpp","third_party/cpp_slam_core/src/keyframe.cpp"]:
    src=open(f).read().splitlines(); viol=[]; i=0
    while i<len(src):
        if 'lock_guard' in src[i] or 'unique_lock' in src[i]:
            depth=0;started=False;j=i
            while j<len(src):
                depth+=src[j].count('{')-src[j].count('}')
                if '{' in src[j]:started=True
                if started and depth<=0 and j>i:break
                if j>i and re.search(r'\.attr\(|->add_observation|->remove_observation|->update_connections|->update_info|->set_bad\b|->add_child|->add_connection|->erase_child',src[j]):
                    viol.append((j+1,src[j].strip()[:60]))
                j+=1
            i=j
        else:i+=1
    print(f,'CLEAN' if not viol else viol)
PY
```

---

## 6. ★ THE REMAINING WORK — F4(2/N)b: native LocalMappingCore + GIL-release (the fps win)

This is the main task. It is a **large** milestone — treat it as several sub-commits, each
parity-validated. The goal: make the heavy local-mapping work run **GIL-released** so LM overlaps
tracking in threaded mode, dropping per-frame *tracking* latency toward ≤100 ms (≥10 fps).

### Why it is large
`LocalMappingCore` (`local_mapping_core.cpp`, 426 lines) is **entirely `py::object`-driven**:
- `fuse_map_points` (L181) imports the **Python** `ProjectionMatcher` (L186) and calls it; it does
  `p.attr(...)`, `kf.attr(...)` throughout.
- `process_new_keyframe`, `cull_map_points`, `cull_keyframes`, `_get_neighbor_keyframes` likewise.
You cannot `gil_scoped_release` around code that calls `obj.attr(...)`. So you must replace the
`py::object` operations on the hot path with **native C++ calls** on `slam::MapPoint*`/`slam::KeyFrame*`
(cast `py::object → slam::KeyFrame*` once, then call C++ methods directly — F2's `build_local_map`
in `geometry_matchers.cpp` is the template for this native-cast pattern).

### Recommended sub-steps (each its own commit, parity-validated)
1. **Profile first** to confirm the targets. Run threaded flag-on with `--profile-runtime`
   (the runner supports it) and read `runtime_profile.csv`. Expected dominant LM sections:
   `local_mapping.fuse_map_points`, `local_mapping.create_new_map_points` (these are GIL-held and
   block tracking). `local_mapping.local_BA` is already GIL-released (SOC).
2. **Native C++ `search_and_fuse`/`fuse` kernel.** Port the fuse matcher (pySLAM
   `local_mapping_core.cpp` / `geometry_matchers.cpp` `search_and_fuse`) to operate on
   `slam::MapPoint*` + `slam::KeyFrame*` (no `py::object` on the inner loop). Reuse the existing
   C++ `search_map_by_projection` machinery. Bind it; route `LocalMappingCore::fuse_map_points`
   to call the native kernel instead of the Python `ProjectionMatcher` (the `local_mapping_core.cpp:186`
   callback). Keep a Python fallback path behind the flag.
3. **Native neighbor/keyframe access** in `fuse_map_points`/`_get_neighbor_keyframes`: replace
   `kf.attr("get_best_covisible_keyframes")` etc. with native `kf_ptr->get_best_covisible_keyframes(n)`
   (the C++ KeyFrame already has these — see `keyframe.cpp`).
4. **GIL-release the native kernel:** wrap the pure-C++ inner matcher loop in
   `py::gil_scoped_release` (re-acquire only for the unavoidable `py::object` boundary writes, e.g.
   `mp->add_observation` callbacks — which the F4 invariant already made lock-safe). Validate the
   invariant audit (§5) still CLEAN and that no GIL-released code path holds a mutex while needing
   the GIL.
5. **Repeat for `create_new_map_points`** (triangulation — `EpipolarMatcher.search_frame_for_triangulation`
   in `local_mapping.py:491`; pySLAM `tracking_core`/`geometry_matchers` for the epipolar search).
6. **Validate** (see §7): flag-off suite green; flag-on behaviour-preserving; **threaded stress**
   (several full `datasets/lab_rgbd_run_2` runs — must complete, 0 catastrophic loss, no wedge);
   **measure per-frame tracking latency** (the gate) — it should drop materially once LM overlaps.

### Important: prove the win with the right metric
Measure **tracking time per frame** (not total elapsed). In threaded mode LM overlaps, so total
elapsed is dominated by whichever thread is slower; the *gate* is tracking latency. Use
`--profile-runtime` and compare `tracking.track_local_map` + `tracking.track_previous_frame`
flag-on vs flag-off, and the threaded `avg_fps`.

### Optional, lower priority after the LM win
- **F3 — C++ `tracking_core`:** port the `track_local_map` orchestration (build → search → pose-opt)
  so the local-map MapPoint vector flows from `build_local_map` straight into `search_map` **without
  the per-point `py::object` re-cast** (a combined `build_and_search`). Reference:
  `third_party/pyslam_reference/pyslam/slam/cpp/tracking_core.cpp`. Tracking is already faster
  flag-on (F2); this is a further refinement.
- **F5 — map-density reduction / loop-safe culling** (the user's "voxel" ask): once a deterministic
  C++ covisibility graph exists, tune KF/point culling to shrink the ~4k-point local map. Validate
  ATE + tracking-loss (60-lost baseline) before keeping. Do NOT use the KF-insertion throttle as the
  lever (proven not the lever). Last resort: float64 kd-tree for reproducibility.

---

## 7. Validation protocol (run this for EVERY milestone)

1. **Fallback intact:** `PYTHONPATH=/home/kaushik/slam_ws .venv/bin/python -m pytest
   tests/visual_slam/orbslam/ -q` → **510 passed, 1 skipped** (flag-off).
2. **Isolation/wiring:** F1/F2 tests green (§4).
3. **Behaviour-preserving flag-on (NOT bit-identical):** flag-on sequential is *not* bit-identical
   run-to-run — there is an inherent ~1 mm float-noise band even pre-change (M0's bit-identical
   guarantee was flag-OFF only). Validate by comparing your build's flag-on trajectory to the
   **previous commit's** flag-on trajectory on a 600-frame run: expect **median ≈ 0.000 mm** (most
   poses identical) with only a few-mm tail. A systematic offset (nonzero median) = a real regression
   to investigate. Compare with:
   ```python
   import numpy as np; a=np.loadtxt(A); b=np.loadtxt(B); n=min(len(a),len(b))
   d=np.linalg.norm(a[:n,1:4]-b[:n,1:4],axis=1)
   print('max_mm',d.max()*1000,'mean_mm',d.mean()*1000,'median_mm',np.median(d)*1000)
   ```
   To build the previous-commit `.so` as a control: `git stash` your src change → rebuild → install →
   run → `git stash pop` → rebuild → reinstall (see git history of this work for the exact dance).
4. **Threaded stress:** at least one full `datasets/lab_rgbd_run_2` threaded run completes (4494/4494,
   0 lost), plus a couple of shorter ones — **no wedge, no crash**. Watch the `RUN SUMMARY`.
5. **Speed (only after parity holds):** `--profile-runtime`; record tracking latency + threaded
   `avg_fps`. The milestone is not "done" without a real, measured improvement on the LM/tracking
   overlap.

---

## 8. Gotchas & debugging tricks (learned the hard way this session)

- **NULL vs None `py::object`:** a **default-constructed `py::object` is NULL** (`ptr()==nullptr`),
  and `.is_none()` returns **false** on NULL (NULL ≠ `Py_None`). Guarding an optional `py::object`
  with `if (!obj.is_none())` will dereference NULL and **segfault**. Use `if (obj)` (the bool
  operator = NULL check), then `is_none()` if you also need to exclude Python `None`. This exact bug
  cost a segfault in `KeyFrame::update_connections` this session.
- **gdb:** `ptrace_scope=1` and no passwordless sudo ⇒ **you cannot `gdb -p <pid>` attach.** But you
  **can launch under gdb**: `gdb --batch -ex run -ex "bt 14" --args .venv/bin/python <script>` —
  this gives the C++ backtrace of a crash (how the segfault above was pinned).
- **Live-hang inspection without ptrace:** read `/proc/<pid>/task/<tid>/stat` (field 3 = state:
  R=running, S=sleeping) and `/proc/<pid>/task/<tid>/wchan` (`futex_wait_queue` = blocked on a
  mutex/GIL). Many `futex_wait` threads are usually the TBB/g2o idle worker pool, not a deadlock.
- **SIGUSR1 thread dump:** `tools/run_lab_cpp.py` arms `faulthandler` (all threads) on **SIGUSR1**.
  `kill -USR1 <pid>` dumps every thread's Python stack to the run log. **Send it to the REAL python
  pid, not the `timeout` wrapper** — `pgrep -f "run_lab_cpp.*<output-tag>"` can match the wrapper;
  the python child is `pgrep -P <wrapper_pid>` or the multi-threaded process. A wrong-pid `ps %cpu`
  read (the wrapper sits at 0% in `sigsuspend`) caused a **false "wedge" alarm** this session — the
  real process was at 100% and completed fine. Always confirm the pid + thread count before
  concluding "deadlock."
- **"Wedge vs slow":** CPU≈100% + frame counter not advancing = a slow/CPU-bound hot spot (e.g.
  `clean_outlier_map_points` → `remove_frame_view`), NOT a mutex deadlock (which idles at CPU≈0
  with all threads in `futex_wait`). Distinguish before "fixing" a deadlock that isn't there.
- **The intermittent long-run segfault** mentioned in old notes is a Python-3.11/pybind tight-loop
  interpreter issue, separate from logic bugs; lower priority.

---

## 9. Working discipline (how the user wants this handled)

- **Think before coding; surface tradeoffs; ask if genuinely ambiguous.** Don't silently pick
  between interpretations or silently expand scope.
- **Simplicity & surgical changes:** minimum code that solves it; touch only what the task needs;
  match surrounding style; don't refactor unrelated code; only remove orphans *your* change created.
- **Goal-driven:** define the success check, loop until it passes. For this work the check is the
  §7 protocol (parity → then measured speedup).
- **Find the EXACT root cause** (evidence, not guesses), compare to pySLAM, then fix cleanly. The
  user explicitly values this over quick-but-uncertain fixes.
- **Commit each validated step** so there is always a clean rollback point.

---

## 10. Commit / rollback reference

- Branch: **`orb-cpp-port`**. Safe baseline to return to: **`d0c774f`**.
- F-series commits (most recent first):
  - `d0c774f` F4(2/N)a-fix — `KeyFrame::set_bad` parent detach outside lock (invariant complete)
  - `6f3f855` F4(2/N)a — `KeyFrame::update_connections` invariant + NULL-`py::object` segfault fix
  - `a9fb879` F4(1/N) — `MapPoint` no-GIL-op-under-mutex invariant
  - `f5e3d7d` F2(3/n) — native `build_local_map`
  - `05fa3bf` F2(2/n) — wire `build_local_map` into tracking
  - `ad88f41` F2(1/n) — C++ `build_local_map`
- Rollback tiers: (1) flip the flag off (runtime, no rebuild); (2) `git checkout d0c774f` (source) +
  rebuild; (3) restore a backed-up `.so` from `/tmp` (binary). Back up the working `.so` before any
  risky rebuild.
- The authoritative long-form roadmap (M0–M6, with history) is `.claude/plans/twelve-fps-final.md`.
  This handoff is the focused, current view; that file is the deep reference.

---

**Start here:** re-read §2 and §6, run a flag-on threaded `--profile-runtime` baseline to confirm
the LM sections dominate, then begin the native `LocalMappingCore::fuse_map_points` port as the
first sub-commit. Validate with §7 after every step. Good luck — build it properly.
