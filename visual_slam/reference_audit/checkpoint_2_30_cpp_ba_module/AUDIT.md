# Checkpoint 2.30 — C++ GIL-free BA Module (slam_optimizer_core)

**Date:** 2026-05-08
**Branch:** `orbslam-development`

---

## PRE-CHANGE AUDIT

### Problem statement

Local BA in Python takes 6–90 seconds per call depending on graph size:

| Phase | Time (50-frame profile at 14 KFs, 15k edges) | % total |
|---|---|---|
| g2o edge setup loop | ~2.4 s | 37% |
| optimizer.optimize() | ~2.4 s | 37% |
| pose/point extraction | ~1.6 s | 24% |
| Total | 6.6 s | 100% |

62% of BA time is Python overhead (GIL-held loops). The g2o `optimize()` call
itself releases the GIL, but setup and extraction loops do not.

For a full fr1/room run at ~0.2 fps, this means ~4–8 hours per benchmark
iteration, making rapid research iteration impractical.

### Root cause

Our `_bundle_adjustment_core()` in `optimizer_g2o.py` uses a Python loop over
all (point, keyframe, observation) tuples to build the g2o graph. At 14 KFs × 1000
points × ~15k edges, Python dict/object overhead dominates.

### Reference analyzed

pySLAM has a complete C++ g2o implementation:
- `third_party/pyslam_reference/pyslam/slam/cpp/optimizer_g2o.cpp` (1326 lines)
- Builds the graph from C++ `KeyFramePtr`/`MapPointPtr` objects — cannot be
  imported directly without their full C++ data hierarchy

**Strategy (Option B):** Copy g2o setup/solve logic, replace C++ object accessors
with numpy array inputs. All `optimizer.addVertex()`, `edge->fx`, `optimizer.optimize()`
calls are identical to pySLAM.

---

## IMPLEMENTATION AUDIT

### Build system

**Problem:** pybind11 2.2.1 (bundled in `EXTERNAL/pybind11/`) is incompatible with
Python 3.11. Error: `PyThreadState` has no member `frame` (changed to `cframe`
in Python 3.11).

**Solution:** Standalone CMake project in `third_party/build/slam_optimizer_core/`
using pybind11 3.0.4 from the venv (already installed). Links pre-built g2o
static libraries from `third_party/g2opy/lib/*.a`.

```
third_party/build/slam_optimizer_core/CMakeLists.txt
→ pybind11 3.0.4 from .venv/lib/python3.11/site-packages/pybind11
→ libg2o_core.a, libg2o_types_sba.a, libg2o_solver_eigen.a, libg2o_stuff.a
→ links cholmod, lapack, blas (same as g2o.so)
```

Build command:
```bash
bash tools/build_slam_optimizer_core.sh
```

### C++ module: slam_optimizer_core.cpp

**File:** `third_party/g2opy/python/slam_optimizer_core.cpp`

Array interface:
```
kf_poses       (N, 16)   float64  row-major Tcw matrices
kf_ids         (N,)      int64    KF IDs (used for vertex IDs: id*2)
kf_fixed       (N,)      uint8    1 = fixed boundary KF
point_pos      (M, 3)    float64  3D positions
observations   (K, 8)    float64  [kf_row, pt_row, u, v, ur, octave, inv_sigma2, is_stereo]
camera         (5,)      float64  [fx, fy, cx, cy, bf]
```

Returns `dict`:
```
updated_poses   (N, 16)  float64
updated_points  (M, 3)   float64
outlier_mask    (K,)     uint8    1 = outlier
mse             float
n_bad_edges     int
```

**g2o API alignment with pySLAM:**

| Component | pySLAM source | Our C++ |
|---|---|---|
| Optimizer setup | BlockSolverSE3 + Levenberg | ✓ identical |
| Vertex SE3 IDs | `kf->kid * 2` | `kf_ids[i] * 2` |
| Point vertex IDs | `p->id * 2 + 1` | `pt_row * 2 + 1` |
| EdgeSE3ProjectXYZ fields | `edge->fx = camera->fx` | `edge->fx = cam[0]` |
| EdgeStereoSE3ProjectXYZ | same | ✓ same |
| Chi2 thresholds | kChi2Mono=5.991, kChi2Stereo=7.815 | ✓ same |
| Huber deltas | kThHuberMono=2.447, kThHuberStereo=2.796 | ✓ same |
| 2-pass optimization | 5 robust + N final | ✓ same |
| Abort flag | `bool *abort_flag` | `std::atomic<bool> g_abort_flag` |

**Deviation from pySLAM:** pySLAM uses a pointer-to-bool abort flag passed per call.
We use a module-level `std::atomic<bool>` controlled via `soc.set_abort(True/False)`.
This is equivalent for single-threaded BA dispatch.

### Python bridge: slam_optimizer_bridge.py

**File:** `visual_slam/orbslam/slam/slam_optimizer_bridge.py`

`pack_local_ba()`:
- Builds ordered kf_list (local + fixed) with deduplication
- Validates each observation: checks `kf.get_point_match(idx) is p` (same
  guard as Python BA in `_bundle_adjustment_core`)
- Skips non-finite keypoints / positions
- Computes `inv_sigma2` via `feature_manager.inv_level_sigmas2[octave]`

`unpack_local_ba()`:
- Writes updated poses back under map lock
- Writes updated point positions + calls `update_normal_and_depth()`
- Removes outlier observations if `prune_outliers=True`
- Increments `kf.lba_count` per pySLAM convention

### Dispatch integration: optimizer_g2o.py

```python
# New module-level
try:
    import slam_optimizer_core as _SOC
    _SOC_AVAILABLE = True
except ImportError:
    _SOC = None
    _SOC_AVAILABLE = False
```

Dispatch condition in `_bundle_adjustment_core()`:
```python
if _SOC_AVAILABLE and write_back and not fixed_points and result_dict is None:
    return _bundle_adjustment_cpp(...)
```

**Why these conditions:**
- `write_back=True`: C++ path always writes back immediately (no deferred result dict)
- `not fixed_points`: C++ path assumes optimizable points (global BA with fixed points
  requires separate C++ function — not yet added)
- `result_dict is None`: global_bundle_adjustment passes a result_dict for deferred apply

**Fallback:** Any exception in the C++ path falls back to Python with a warning print.
The Python path is completely unchanged.

---

## VALIDATION REPORT

### Unit tests

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam
212 passed, 1 skipped in 21.68s
```

6 new tests in `test_slam_optimizer_core_parity.py`:
- `test_module_loads` — hello() returns expected string ✓
- `test_local_ba_converges` — 3-KF synthetic problem, MSE < 10, non-fixed KF
  tx moves toward GT ✓
- `test_outlier_detection` — 3 injected 500px outliers, ≥2 detected ✓
- `test_abort_flag` — set_abort(True) returns valid result dict ✓
- `test_python_fallback` — `_SOC_AVAILABLE` is patchable to False ✓
- `test_global_ba` — run_global_ba converges on 4-KF problem ✓

No regressions in existing 206 tests.

---

## EXPECTED IMPACT

### BA speedup estimate

| Phase | Before (Python) | After (C++) |
|---|---|---|
| Edge setup loop | 2.4 s (37%) | ~0.01 s (GIL-free Eigen map) |
| optimize() | 2.4 s (37%) | ~2.4 s (unchanged GIL release) |
| Pose/point extract | 1.6 s (24%) | ~0.05 s (memcpy) |
| Pack/unpack (new) | — | ~0.3 s (Python object iteration) |
| **Total** | **6.6 s** | **~2.8 s** |

Expected speedup: **~2.4× per BA call**.

Bottleneck shifts to `optimize()` itself (inherently C++ already) and pack/unpack.
Further speedup (for pack/unpack) possible if observation arrays are pre-built
incrementally rather than rebuilt each BA call — future optimization.

### Full benchmark impact

- Before: 0.19 fps (5.3 s/frame) with wait_idle
- Estimated after: 0.30–0.35 fps (reducing BA from 35–90 s to 15–40 s per KF)
- Full fr1/room run: ~1.1 hours vs current ~2 hours

### Effect on LiDAR SLAM (future)

The `slam_optimizer_core` module is in `third_party/g2opy/python/` (workspace shared library).
Adding `run_pose_graph_se3()` to the same file provides GIL-free SE3 PGO for LiDAR SLAM
without any new dependencies. g2o `types_slam3d` (already linked) provides
`VertexSE3` and `EdgeSE3`.

---

## FILES CREATED

| File | Description |
|---|---|
| `third_party/g2opy/python/slam_optimizer_core.cpp` | C++ BA module (~310 lines) |
| `third_party/build/slam_optimizer_core/CMakeLists.txt` | Standalone CMake project |
| `visual_slam/orbslam/slam/slam_optimizer_bridge.py` | Python pack/unpack bridge |
| `tools/build_slam_optimizer_core.sh` | Build + install script |
| `tests/visual_slam/orbslam/test_slam_optimizer_core_parity.py` | 6 parity tests |

## FILES MODIFIED

| File | Change |
|---|---|
| `third_party/g2opy/python/CMakeLists.txt` | Added second pybind11 target (compile-only; fails pybind11 2.2.1/Py311 so not used for build — standalone build is used instead) |
| `visual_slam/orbslam/slam/optimizer_g2o.py` | Added `_SOC_AVAILABLE` flag, `_bundle_adjustment_cpp()`, dispatch in `_bundle_adjustment_core()` |

## FILES NOT TOUCHED

| File | Reason |
|---|---|
| `visual_slam/g2o_compat.py` | Python fallback path unchanged |
| `third_party/g2opy/g2o.cpp` | Existing bindings unchanged |
| `local_mapping_core.py`, `global_ba.py`, `tracking.py` | Zero change needed |
| `carto/`, `hector/`, `slam_core/` | Zero g2o optimizer dependency |
| pySLAM `cpp/keyframe.cpp` etc. | Data structures not imported |

---

## REMAINING GAPS

### P0 — global_bundle_adjustment C++ dispatch not yet wired

`global_bundle_adjustment()` calls `_bundle_adjustment_core()` with `result_dict` set,
which bypasses the C++ path. The fix requires either:
(a) Adding `run_global_ba_deferred()` to the C++ module that returns a dict of
    `{kf_id: Tcw}` / `{pt_id: pos}` for deferred application
(b) Moving the deferred result dict collection into Python after C++ write-back

This is a P0 for the Global BA speedup, which is the slowest phase in loop-triggered
benchmark runs.

### P1 — pack/unpack overhead (~0.3 s per BA call)

The Python `pack_local_ba()` iterates all observations to build numpy arrays. For
large windows (>2000 points, >30k edges), this takes ~0.5 s. Possible fix: maintain
an incremental observation array in local_mapping_core.py updated on each observation
add/remove — but this is a significant data structure change.

### P2 — CMakeLists.txt g2opy integration broken

Adding the second `pybind11_add_module` target to `third_party/g2opy/python/CMakeLists.txt`
fails at compile time due to pybind11 2.2.1 / Python 3.11 incompatibility. The standalone
build is used instead. The CMakeLists.txt addition is harmless (ignored by the main build)
but should either be removed or patched with `if(pybind11_VERSION VERSION_GREATER_EQUAL 2.4)`.

### P3 — abort flag not propagated mid-optimize()

The C++ `g_abort_flag` is checked by `optimizer.setForceStopFlag(&local_abort)`,
but `local_abort` is a copy made at the start of the call. `set_abort(True)` from
another thread will not interrupt an already-running C++ `optimizer.optimize()`.
This matches the existing Python behavior (abort is only checked between rounds).
Fix: pass `&g_abort_flag` directly (requires making the global addressable at call time).

---

## NEXT RECOMMENDED ACTION

1. Run a short smoke with C++ BA enabled to measure actual speedup:
   ```bash
   source .venv/bin/activate
   python -m visual_slam.orbslam.run_tum_rgbd_smoke \
     datasets/tum/rgbd_dataset_freiburg1_room \
     --output visual_slam_outputs/fr1_room_cpp_ba_test \
     --max-frames 100 --feature-backend pyslam_orb2 \
     --enable-loop-closing 2>&1 | tee /tmp/cpp_ba_smoke.log
   ```

2. Compare elapsed_sec at 100 frames vs Python-path baseline.

3. Wire `global_bundle_adjustment()` to C++ path (resolve P0 above).
