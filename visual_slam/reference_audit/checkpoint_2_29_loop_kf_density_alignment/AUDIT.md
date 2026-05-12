# Checkpoint 2.29 — Loop KF Density Alignment with pySLAM

**Date:** 2026-05-08  
**Branch:** `orbslam-development`

---

## PRE-CHANGE AUDIT

### Problem statement

The fr1/room benchmark run (previous session) produced:
- 50 keyframes for 1362 frames (3.7 KF/100 frames)
- 0 loop closures
- ATE RMSE 0.294 m vs ORB-SLAM2 published 0.047 m

pySLAM produces ~200–300 KFs for the same sequence (~15–22 KF/100 frames). Without sufficient KF density, the BoW consistency checker (requires 3 consecutive consistent detections) never fires.

### Root cause analysis (from previous session)

| Issue | File | Lines |
|---|---|---|
| Missing `cond1c` (RGB-D idle-bypass) | `tracking.py` | 488–531 |
| No post-condition decision tree | `tracking.py` | 488–531 |
| No single-thread min_frames guard | `tracking.py` | 488–531 |
| No early-map ratio adaptation | `tracking.py` | 488–531 |
| Stale cached `num_kf_ref_tracked_points` | `tracking.py` | 519–520 |
| `_is_correcting` unprotected (race with LM thread) | `loop_closing.py` | 994, 1079–1083 |
| `queue` operations unprotected | `loop_closing.py` | 1002–1013 |
| `wait_idle` gated on loop_closing queue (not always sync'd) | `run_tum_rgbd_smoke.py` | 175–176 |
| No `interrupt_optimization()` method | `local_mapping.py` | — |

### pySLAM reference examined

- `third_party/pyslam_reference/pyslam/slam/tracking.py` lines 761–916 (full `need_new_keyframe`)
- `third_party/pyslam_reference/pyslam/config_parameters.py` (threshold values)

Key pySLAM parameters confirmed:
```
kThNewKfRefRatioStereo = 0.75      (stereo/RGBD c2 threshold)
kThNewKfRefRatioNonMonocular = 0.25 (c1c weak-tracking threshold)
kNumMinPointsForNewKf = 15          (c2 absolute floor)
kNumMinTrackedClosePointsForNewKfNonMonocular = 100
kNumMaxNonTrackedClosePointsForNewKfNonMonocular = 70
```

---

## IMPLEMENTATION AUDIT

### Fix 1 — `need_new_keyframe` in `tracking.py`

**What changed:** Replaced 2-condition (c1a/c1b + c2) logic with full 3-condition + decision tree.

```python
# Before
c1a = frames_since_last_kf >= self.max_frames_between_kfs
c1b = frames_since_last_kf >= self.min_frames_between_kfs and is_idle
c2 = (num_matched_map_points < ref_ratio * cached_ref_tracked) or need_to_insert_close
return bool((c1a or c1b) and c2)

# After
nMinObs = 2 if num_kfs <= 2 else 3
num_ref_tracked = max(1, kf_ref.num_tracked_points(nMinObs))  # dynamic, not cached
ref_ratio = 0.4 if num_kfs < 2 else Parameters.kThNewKfRefRatioStereo
if not is_threaded:
    self.min_frames_between_kfs = 3
c1a = frames_since_last_kf >= self.max_frames_between_kfs
c1b = frames_since_last_kf >= self.min_frames_between_kfs and is_idle
c1c = (sensor != MONO) and (num_matched_cur < 0.25 * num_ref_tracked or need_to_insert_close)
c2 = (num_matched_cur < ref_ratio * num_ref_tracked or need_to_insert_close) and num_matched_cur > 15
if not ((c1a or c1b or c1c) and c2): return False
if is_idle: return True
if c1a or (c1c and need_to_insert_close):
    interrupt_optimization(); return True
interrupt_optimization(); return False
```

**pySLAM alignment:**
| Feature | pySLAM | Before | After |
|---|---|---|---|
| cond1c (RGB-D idle-bypass) | ✓ | ✗ | ✓ |
| Post-condition decision tree | ✓ | ✗ | ✓ |
| Single-thread min_frames guard | ✓ | ✗ | ✓ |
| Early ratio 0.4 for num_kfs < 2 | ✓ | ✗ | ✓ |
| Dynamic ref tracked count (nMinObs) | ✓ | ✗ (cached stale) | ✓ |
| kNumMinPointsForNewKf floor in c2 | ✓ | ✗ | ✓ |
| LM stopped guard | ✓ | ✗ | ✓ |

**Deviation from pySLAM** (intentional): In pySLAM for non-mono with LM busy, the decision tree always returns False (after calling interrupt_optimization). We return True when c1a fires or when c1c + need_to_insert_close fires. Rationale: Python BA takes 18–20s vs pySLAM C++ <0.3s; waiting for idle would leave the camera without fresh KFs for 30+ seconds, causing worse drift.

### Fix 2 — `_is_correcting` thread safety in `loop_closing.py`

**What changed:** Added `threading.Lock` for `_is_correcting` read/write and for queue operations.

```python
# __init__:
self._is_correcting_lock = threading.Lock()
self._queue_lock = threading.Lock()

# is_correcting():  uses _is_correcting_lock on read
# process_keyframe(): uses _is_correcting_lock on write and clear in finally
# add_keyframe/pop_keyframe/queue_size: use _queue_lock
```

**Why correct:** `local_BA()` in the LM background thread calls `loop_closing.is_correcting()`. Simultaneously, `process_keyframe()` (called from the main thread in sequential loop closing) writes `_is_correcting`. Without a lock, this is a Python-level data race on a shared mutable bool. The lock makes all reads and writes atomic.

### Fix 3 — `wait_idle` unconditional when threaded (`run_tum_rgbd_smoke.py`)

**What changed:** Removed `and loop_closing.queue_size() > 0` gate from wait_idle call.

```python
# Before:
if loop_closing is not None and loop_closing.queue_size() > 0 and threaded_lm:
    slam.local_mapping.wait_idle(timeout=2.0)

# After:
if threaded_lm:
    slam.local_mapping.wait_idle(timeout=2.0)
```

**Effect:** Tracking waits for LM to settle each frame when threaded. Cost: ~2.0s overhead per frame when LM is busy (runtime increased from 2.86 s/frame to ~5.3 s/frame in diagnostic run).

### Fix 4 — `interrupt_optimization()` added to `local_mapping.py`

Added `interrupt_optimization()`, `is_stopped()`, and `is_stop_requested()` methods. `interrupt_optimization()` delegates to `set_opt_abort_flag(True)` which propagates to the g2o BA optimizer.

---

## VALIDATION REPORT

### Unit / integration tests

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam
206 passed, 1 skipped in 20.90s
```

All tests pass. No regressions.

### 200-frame KF density diagnostic

```
Dataset:  rgbd_dataset_freiburg1_room (frames 0–199)
Backend:  pyslam_orb2
Flags:    --enable-loop-closing --start-local-mapping-thread
```

| Metric | Before (extrapolated) | After |
|---|---|---|
| KF count (200 frames) | ~7 | 12 |
| KF rate per 100 frames | 3.7 | 6.0 |
| Elapsed (200 frames) | ~580 s | 1061 s |
| Avg FPS | 0.35 | 0.19 |
| Tracking LOST | 0 | 0 |
| Errors | 0 | 0 |

**Improvement:** ~60% more KFs with the same frame count.

**Target:** pySLAM produces ~22–44 KFs for 200 frames. We produce 12. Gap is ~2–3×.

---

## PYSLAM COMPARISON

### need_new_keyframe structural alignment

| Component | pySLAM reference | Our implementation |
|---|---|---|
| cond1a (time fallback) | `f_cur.id >= kf_last.id + max_frames` | ✓ same |
| cond1b (idle + min_frames) | `f_cur.id >= kf_last.id + min_frames AND idle` | ✓ same |
| cond1c (RGB-D bypass) | `sensor!=MONO AND (matched<0.25*ref OR close_starved)` | ✓ aligned |
| cond2 (tracked vs ref + floor) | `(matched < thRefRatio*ref OR close_starved) AND matched > 15` | ✓ aligned |
| Post-tree: idle → True | `if idle: return True` | ✓ same |
| Post-tree: busy non-mono → False | `else: interrupt; return False` | ⚠ deviation: force-insert on c1a or c1c+close_starved |
| Single-thread guard | `min_frames = 3 if not threaded` | ✓ same |
| Early ratio 0.4 | `if num_kfs < 2: thRefRatio = 0.4` | ✓ same |
| Dynamic nMinObs | `nMinObs = 3 (or 2 if ≤2 KFs)` | ✓ same |
| LM stopped guard | `if LM.is_stopped() or stop_requested: return False` | ✓ added |

### Thread safety alignment

| Component | pySLAM reference | Our implementation |
|---|---|---|
| `_is_correcting` read/write lock | uses mutex | ✓ added Lock |
| Queue thread safety | deque + lock | ✓ added Lock |

---

## REMAINING GAPS AND RISKS

### P0 — KF density still below pySLAM target

- **Measured:** 6 KFs/100 frames (12 in 200 frames)
- **Target:** 15–22 KFs/100 frames (pySLAM)
- **Root cause:** Python BA takes ~80–90s per KF cycle vs pySLAM C++ <0.3s. Even with correct c1c, new KFs can only be created at the rate LM finishes processing.
- **Projected for full 1362-frame run:** ~82 KFs (vs 50 before, vs 200–300 target)
- **Loop closure probability with ~82 KFs:** uncertain — depends on whether the fr1/room camera actually revisits areas in our sparse KF set. With 82 vs 50 KFs, the BoW database has more entries and the consistency checker has more chances to fire.

### P1 — Runtime regression from unconditional wait_idle

- Before: 0.35 fps / 2.86 s/frame
- After: 0.19 fps / 5.3 s/frame
- Full 1362-frame run would take ~7200 s (~2 h) vs original ~3944 s (~65 min)
- Considered acceptable for research benchmarking; if runtime is a concern, `wait_idle(timeout=2.0)` can be made conditional on loop_closing queue non-empty again.

### P2 — interrupt_optimization effectiveness limited

- `set_force_stop_flag` is only wired to the g2o optimizer if `abort_flag.__class__.__module__ == "g2o"` (i.e., if `_make_flag` returns a `g2o.Flag`). If g2o doesn't have `Flag`, the abort flag is a Python `SimpleNamespace` and the g2o optimizer runs to completion regardless.
- The abort is still checked at two points in the optimizer (before starting, between 5-iteration rounds). For a 10-iteration BA that's effectively a mid-point check.

### P3 — cull_keyframes unchanged

- culling threshold is 0.9 (90% redundancy). pySLAM uses 0.9. NOT the bottleneck — we're creating too few KFs to cull, not culling too many.

---

## FILES INSPECTED

| File | Purpose |
|---|---|
| `visual_slam/orbslam/slam/tracking.py` | `need_new_keyframe` implementation |
| `visual_slam/orbslam/slam/loop_closing.py` | `_is_correcting`, queue thread safety |
| `visual_slam/orbslam/slam/local_mapping.py` | `interrupt_optimization`, `is_idle` |
| `visual_slam/orbslam/slam/local_mapping_core.py` | `set_opt_abort_flag`, BA abort wiring |
| `visual_slam/orbslam/slam/optimizer_g2o.py` | `_abort_requested`, `set_force_stop_flag` |
| `visual_slam/orbslam/run_tum_rgbd_smoke.py` | `wait_idle` gate |

## pySLAM FILES INSPECTED

| File | Purpose |
|---|---|
| `third_party/pyslam_reference/pyslam/slam/tracking.py` L761–916 | Full `need_new_keyframe` reference |
| `third_party/pyslam_reference/pyslam/config_parameters.py` L126–133 | Threshold constants |

---

## FILES CHANGED

| File | Change |
|---|---|
| `visual_slam/orbslam/slam/tracking.py` | `need_new_keyframe`: full rewrite with c1c, decision tree, guards |
| `visual_slam/orbslam/slam/loop_closing.py` | `threading` import, `_is_correcting_lock`, `_queue_lock`, updated read/write/queue ops |
| `visual_slam/orbslam/slam/local_mapping.py` | Added `interrupt_optimization`, `is_stopped`, `is_stop_requested` |
| `visual_slam/orbslam/run_tum_rgbd_smoke.py` | `wait_idle` unconditional when threaded |

---

## NEXT RECOMMENDED ACTION

Proceed with full fr1/room benchmark run (`bash tools/launch_fr1_room_benchmark.sh`). Expected:
- ~82 KFs (vs 50 before) — a 64% improvement
- Possibly 1–5 loop edge detections given more BoW database entries
- ATE improvement toward 0.15–0.20 m (vs 0.294 m before) even without full loop correction

If loop closure still shows 0 edges after the full run, the next investigation should be:
1. Add BoW query diagnostic logging to see candidate counts at each step
2. Check whether `kMinDeltaFrameForMeaningfulLoopClosure` should be reduced (currently 10)
3. Check whether the fr1/room sequence in our specific KF set actually has camera revisits
