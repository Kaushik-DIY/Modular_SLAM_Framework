# Lab RGB-D Dataset Sanity Check Audit

**Date:** 2026-05-06
**Dataset:** `datasets/lab_rgbd_run_2`
**Source:** Intel RealSense D4xx, JetRacer robot, lab room
**Pipeline:** `visual_slam/orbslam/` — pySLAM-aligned ORB-SLAM2 RGB-D

---

## 1. Dataset Structure Verification

| Item | Expected (TUM-style) | lab_rgbd_run_2 | Status |
|---|---|---|---|
| `rgb/` folder | ✓ 16-bit PNG filenames | ✓ 4499 PNGs | OK |
| `depth/` folder | ✓ 16-bit PNG filenames | ✓ 4499 PNGs | OK |
| `rgb.txt` | `timestamp rgb/file.png` | matches | OK |
| `depth.txt` | `timestamp depth/file.png` | matches | OK |
| `associations.txt` | `ts_rgb rgb/f ts_dep depth/f` | matches, 4494 entries | OK |
| Image size | 640×480 | 640×480 uint8 RGB / uint16 depth | OK |
| `groundtruth.txt` | optional | absent (no GT, documented) | Acceptable |
| `camera.yaml` | optional | present — RealSense intrinsics | Extra (beneficial) |
| `metadata.json` | optional | present — bag export metadata | Extra (informational) |

**File cross-check:** Every RGB and depth file referenced in `associations.txt` exists on disk. Zero missing files.

### Timestamp alignment
- RGB timestamps: `1778076070.289` → `1778076226.116`
- Depth timestamps: same range (aligned_depth_to_color topic)
- All 4494 association pairs have `dt = 0.0` (perfect sync — same timestamp for RGB and depth)
- This is correct: `aligned_depth_to_color` is temporally aligned to the color frame by the RealSense SDK

### Framerate
- Nominal: 30fps (camera.yaml)
- Actual: 28.9fps avg (34.64ms mean dt, 33.33ms median)
- 3 frames with ~100ms gap (single dropped frames) in 4499 — negligible

---

## 2. Camera Parameters Verification

### Source: `camera.yaml` (auto-loaded by `rgbd_dataset.py`)

| Parameter | lab_rgbd_run_2 | TUM fr1 (old default) | Impact of using wrong |
|---|---|---|---|
| `fx` | **609.883** | 517.3 | Feature unprojection off by 18% |
| `fy` | **609.177** | 516.5 | Same |
| `cx` | **324.921** | 318.6 | Principal point shifted 6px |
| `cy` | **229.748** | 255.3 | Principal point shifted 25px |
| `DepthMapFactor` | **1000.0** | 5000.0 | All depth values × 5 too large |
| `depth_factor` | **0.001** m/raw | 0.0002 | Map metric scale catastrophically wrong |
| `depth_threshold` | **3.20 m** | 3.20 m | Same (0.08 × 40 = 3.2m both ways) |
| `width × height` | 640 × 480 | 640 × 480 | OK |
| `fps` | 30.0 | 30.0 | OK |
| `baseline` | 0.08 m | 0.08 m | Virtual stereo baseline, OK |

**How camera.yaml is loaded:**
`run_tum_rgbd_smoke.py` now calls `make_rgbd_camera(dataset)` (full path),
which checks for `camera.yaml` → calls `_load_camera_yaml()` → builds `PinholeCamera`
with correct intrinsics and `depth_map_factor=1000.0`.

### Depth factor chain (verified)
```
raw uint16 pixel (e.g. 1500) → frame.py line 448:
depth_m = raw_depth * camera.depth_factor = 1500 * 0.001 = 1.5m   ✓
```

### Depth range sanity (5 sampled frames)
| Time | Valid pixels | Min | Median | p90 | Max | % within 3.2m |
|---|---|---|---|---|---|---|
| 0s (start) | 280766 | 0.23m | 1.63m | 6.25m | 15.95m | 70.3% |
| 38s | 281708 | 0.23m | 1.28m | 2.13m | 10.82m | 100.0% |
| 77s | 291756 | 0.22m | 1.34m | 3.42m | 3.94m | 82.4% |
| 116s | 287704 | 0.22m | 1.76m | 4.74m | 22.45m | 76.0% |
| 156s (end) | 290050 | 0.23m | 1.98m | 6.97m | 14.43m | 66.5% |

- 66–100% of valid pixels are within the 3.2m close-point threshold → robust close-point map
- Values above 10m are likely RealSense specular/multipath noise — these become "far" map points and are treated accordingly
- `kMinDepth = 0.01m`: dataset min is 0.22m, so all valid depth passes the floor check

---

## 3. Pipeline Parameter Audit — Layer by Layer

### 3.1 Feature Extraction (`feature_tracker_configs.py`, `config_parameters.py`)

| Parameter | Value | Notes |
|---|---|---|
| `kNumFeatures` | 2000 | Appropriate for 640×480 |
| `kORBNumLevels` | 8 | Standard ORB-SLAM2 |
| `kORBScaleFactor` | 1.2 | Standard |
| `kSigmaLevel0` | 1.0 | Standard |
| `kFeatureMatchDefaultRatioTest` | 0.7 | Standard |
| backend | `pyslam_orb2` | Correct for benchmark |

No dataset-specific feature changes needed. Feature extraction is image-content driven, not camera-model driven.

### 3.2 Depth Handling (`frame.py`)

- `frame.py:448`: `depth_m = raw_depth * camera.depth_factor` — generic, reads from camera object
- `camera.depth_factor` = `1 / depth_map_factor` — set at construction from camera.yaml
- `kMinDepth = 0.01m` — floor filter; dataset min 0.22m safely above this
- `uR` virtual stereo coordinate: `kp.pt[0] - camera.bf / depth_m` — uses `bf = fx * baseline = 609.883 * 0.08 = 48.79`. Correct for RealSense.

No changes needed.

### 3.3 Tracking & Keyframe Generation (`tracking.py`, `tracking_core.py`)

| Parameter | Value | Lab relevance |
|---|---|---|
| `kMaxReprojectionDistanceMapRgbd` | 3 pixels | Tight but correct for RGBD |
| `kMaxReprojectionDistanceFrame` | 7 pixels | Frame-to-frame search radius |
| `far_points_threshold` | `None` | Correct: no far-point filter for RGBD |
| `max_frames_between_kfs` | `int(fps) = 30` (force-KF every 1s) | Correct at ~29fps |
| `min_frames_between_kfs` | 0 | OK |
| `kThNewKfRefRatioStereo` | 0.75 | Used for RGBD — matches ORB-SLAM2 convention |
| `kNumMinTrackedClosePointsForNewKfNonMonocular` | 100 | Close points are plentiful in lab scene |
| `kNumMaxNonTrackedClosePointsForNewKfNonMonocular` | 70 | OK |
| `kInitializerNumMinTriangulatedPointsStereo` | 100 | Smoke confirmed 791 at init — passes |
| `reproj_err_frame_map_sigma` | `kMaxReprojectionDistanceMapRgbd = 3` | RGBD branch correctly selected |

No changes needed.

### 3.4 Keyframe Culling (`local_mapping_core.py`)

| Parameter | Value | Lab relevance |
|---|---|---|
| `kKeyframeCullingRedundantObsRatio` | 0.90 | Standard — 90% obs redundancy to cull |
| `kKeyframeMaxTimeDistanceInSecForCulling` | 0.5s | Protects ~15 frames @ 30fps |
| `kKeyframeCullingMinNumPoints` | 0 | No floor, cull by redundancy only |

Culling is RGBD-aware: only **close** points (depth < 3.2m) count toward redundancy check (line 199 of `local_mapping_core.py`). Far points are correctly excluded. This is correct for RealSense depth.

No changes needed.

### 3.5 Local Bundle Adjustment (`config_parameters.py`)

| Parameter | Value | Notes |
|---|---|---|
| `kLocalBAWindowSize` | 20 KFs | Standard |
| `kLocalMappingNumNeighborKeyFramesStereo` | 10 | For RGB-D triangulation neighbor search |
| `kMinNumOfCovisiblePointsForCreatingConnection` | 15 | Fine |

No changes needed.

### 3.6 Loop Closing (`config_parameters.py`)

| Parameter | Value | Lab relevance |
|---|---|---|
| `kUseLoopClosing` | True | Enabled |
| `kMinDeltaFrameForMeaningfulLoopClosure` | 10 | Min frame gap between query and loop KF |
| `kLoopClosingMinNumMatchedMapPoints` | 60 | Inlier floor for acceptance |
| `kLoopClosingSE3RansacMaxError` | 0.25m | Appropriate for lab-scale (1–3m distances) |
| `kLoopClosingSE3RansacIterations` | 300 | Sufficient |
| `kLoopClosingGeometryCheckerMinKpsMatches` | 20 | OK |
| `kLoopClosingMaxReprojectionDistanceMapSearch` | 10 pixels | Loop fusion search |
| `kEssentialGraphLoopEdgeWeight` | 10.0 | Dominates essential graph |

No changes needed. SE3 loop geometry is scale-fixed for RGB-D, correct for lab dataset.

### 3.7 Global BA (`config_parameters.py`)

| Parameter | Value | Notes |
|---|---|---|
| `kUseGBA` | False (default) | Enable with `--enable-global-ba` flag |
| `kGlobalBAIterations` | 10 | Fine |
| `kGBAUseRobustKernel` | True | Protects against false loop damage |

No changes needed.

### 3.8 BoW Vocabulary (`bow.py`)

- Path: `third_party/local/vocabs/ORBvoc.dbow3`
- Loaded at Slam construction — same vocabulary for any ORB2 dataset
- No dataset-specific configuration needed

---

## 4. Code Changes Made

### New file: `visual_slam/orbslam/io/rgbd_dataset.py`
- `detect_dataset_type(dataset)` — returns `"custom_yaml"` for lab, `"tum_fr1/2/3"` for TUM
- `make_rgbd_camera(dataset)` — loads `camera.yaml` if present, else TUM name lookup
- `load_rgbd_associations(dataset)` — delegates to existing `load_tum_rgbd_associations`
- `tum_rgbd.py` is **not modified**

### Modified: `visual_slam/orbslam/run_tum_rgbd_smoke.py`
- Imports `make_rgbd_camera` and `detect_dataset_type` from `rgbd_dataset.py`
- Passes full dataset `Path` to `make_rgbd_camera` (triggers camera.yaml lookup)
- Prints `Dataset type:` in run header
- `make_tum_rgbd_camera` (old) no longer called by smoke runner

---

## 5. Smoke Test Results (50 frames)

```
Dataset type:  custom_yaml
Camera:        fx=609.883, fy=609.177, cx=324.921, cy=229.748
Depth factor:  0.001  (depth_map_factor=1000, RealSense mm units)
Frames loaded: 50
frames_attempted:  50
tracking_ok_count: 50
tracking_lost_count: 0
errors:          0
final_keyframes: 2
final_map_points: 1082
trajectory_poses: 50
kf_traj_consistency: n=2 max=0.0000m median=0.0000m
```

All 50 frames tracked without loss. Camera intrinsics loaded correctly from camera.yaml.

---

## 6. Remaining Risks / Notes

1. **No groundtruth** — ATE/RPE evaluation not possible. Map quality must be judged visually or via loop closure metrics.
2. **3 dropped frames** in the sequence (100ms gaps) — negligible; motion model handles single-frame gaps well.
3. **Depth noise above 10m** — specular/glass surfaces in lab produce spurious far readings. These become "far" map points, excluded from close-point logic. Acceptable.
4. **No global BA by default** — enable with `--enable-global-ba` for full benchmark run.

---

## 7. Full Run Command

```bash
source .venv/bin/activate
python3 -m visual_slam.orbslam.run_tum_rgbd_smoke \
  datasets/lab_rgbd_run_2 \
  --output visual_slam_outputs/lab_rgbd_run_2_full \
  --max-frames 0 \
  --feature-backend pyslam_orb2 \
  --enable-loop-closing \
  --enable-global-ba \
  --print-every 100
```
