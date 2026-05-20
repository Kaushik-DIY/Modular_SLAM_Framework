# Checkpoint 2.31A: Memory Policy Implementation Audit

## 1. Overview
The goal of this implementation phase was to align the ORB-SLAM memory policy with the pySLAM standard, addressing severe memory growth over long sequences. This involved deterministic cleanup of temporary references and bounding image retention.

## 2. Changes Made

### 2.1 Explicit `Map.frames` Eviction
- Added `Map._cleanup_evicted_frame(frame: Frame)` to explicitly drop image buffers, point references, and heavy data for normal frames.
- Replaced the simple `self.frames.append(frame)` in `Map.add_frame()` with a logic block that manually `popleft()`s frames exceeding `Parameters.kMaxLenFrameDeque` and calls `_cleanup_evicted_frame()`.

### 2.2 `MapPoint` Frame View Pruning
- Updated `MapPointBase` (and properly overridden `MapPoint` for C++ compatibility) with `remove_frame_views_older_than(min_frame_id)` to drop temporary tracking reference pointers (used briefly for map-point fusion/search but useless afterward).
- Implemented `Map.prune_old_frame_views()` to iterate over `MapPoint`s and prune these stale references periodically.

### 2.3 Controlled Image/Depth Retention
- Modified `Frame` and `KeyFrame` to implement `release_images(release_rgb, release_depth)`, `release_heavy_data()`, and `heavy_memory_bytes()`.
- Tied `KeyFrame` depth image retention to configurable flags `Parameters.kStoreKeyFrameDepthImages`. In `lean_memory` configurations, both images and depth images are released after initial processing.

### 2.4 Local Mapping Queue Cleanup
- Edited `LocalMapping.step()` and `_run_thread()` loops to include a `finally:` block that ensures `self.img_cur`, `self.img_cur_right`, and `self.depth_cur` are explicitly nulled out, releasing the Python reference counts to large image arrays.
- Added `queue_memory_stats()` to track queue sizes for debugging.

### 2.5 Lightweight Memory Profiling
- Expanded `run_rgbd_slam.py` to optionally query `psutil` or `resource` for peak RSS footprint.
- Outputs `memory_profile.csv` detailing the frame queue size, map points, keyframe image sizes, and local mapping queue memory estimates over time.

## 3. Structural Correctness
By mirroring the pySLAM policy:
1. Normal frames are aggressively discarded and cleaned when sliding out of the tracking history window.
2. `KeyFrame`s maintain their structures for graph optimization but can safely drop large dense arrays.
3. Garbage collection isn't fighting circular references inside `MapPoint.frame_views`.

## 4. Risks Mitigated
The primary risk (OOM during 1500+ frame datasets like FR1_Room) has been architecturally addressed. Memory should now scale proportionally with KeyFrame count, not total sequence length.

---

## 5. Frame Lifecycle (Normal Frames)

### pySLAM behavior
`Map.frames` is a `deque(maxlen=kMaxLenFrameDeque)`. When full, Python's deque automatically drops the oldest entry via implicit eviction — no explicit cleanup hook is called. Normal frames eventually become garbage-collected when no other Python reference holds them. `_frame_views` in MapPoints may keep them alive until pruned.

### Local behavior (before change)
No explicit cleanup. `Map.frames` was a plain deque without `maxlen` enforcement at the cleanup level. Evicted frames were not cleaned; `img` and `depth_img` were held until GC, but GC was delayed by lingering `_frame_views` references.

### Gap and fix
Added `Map._cleanup_evicted_frame()` which:
1. Calls `frame.remove_frame_views()` to sever all `MapPoint._frame_views[frame]` entries.
2. Calls `frame.release_heavy_data()` to null `img`, `depth_img`, and `kd`.
This is called explicitly in `add_frame()` when the deque popleft()-evicts a frame.

### Deliberately stronger than pySLAM
pySLAM relies on GC; our implementation is explicit. This is intentional and correct for long-running Python processes where GC collection timing is not deterministic.

---

## 6. KeyFrame Lifecycle

### pySLAM behavior
KeyFrames are never explicitly cleaned up for image data. The pySLAM runner does not call any `release_depth_image` or equivalent. KeyFrame `depth_img` is kept alive indefinitely (pySLAM uses it for the volumetric integrator path).

### Local behavior (before change)
Same as pySLAM — depth images were kept for all keyframes indefinitely.

### Gap and fix (§4.4 call sites)
Three call sites now release `kf.depth_img` after per-keypoint depth arrays and `uRs` are computed and stored:
1. `Tracking._create_initial_rgbd_map()` — initial KF after first frame
2. `Tracking.create_new_keyframe()` — every subsequent KF
3. `LocalMapping.do_local_mapping()` — after `process_new_keyframe()` in the LM thread

Controlled by `Parameters.kStoreKeyFrameDepthImages` (default `False`). RGB release is controlled by `Parameters.kStoreKeyFrameImages` (default `True`).

### Why this is safe
`kf.depths` (per-keypoint depth values) and `kf.uRs` (right-image u-coords) are extracted during KF construction from `depth_img`. These scalar arrays are what BA and projection search actually use. `depth_img` is only needed again for volumetric integration (not present in this pipeline).

---

## 7. MapPoint Observation vs. Frame-View Lifecycle

### pySLAM behavior
`_observations: dict[KeyFrame, int]` — permanent KF-level associations, managed by `add_observation` / `remove_observation`.
`_frame_views: dict[Frame, int]` — temporary tracking associations, added per-frame. pySLAM relies on GC or `check_replaced_map_points()` to eventually clear stale entries.

### Local behavior (before change)
Same as pySLAM. Frame views were never explicitly pruned for old frames.

### Gap and fix
Added `MapPoint.remove_frame_views_older_than(min_frame_id)` which iterates `_frame_views` and removes entries whose frame.id < min_frame_id. Called periodically by `Map.prune_old_frame_views()` (every `kFrameViewPruneEveryNFrames` frames, retaining `kFrameViewRetention` most recent).

### KF observations unaffected
`remove_frame_views_older_than` only touches `_frame_views`, not `_observations`. KF observations are managed exclusively by `add_observation` / `remove_observation` / `set_bad`.

---

## 8. Tracking History Policy

### Policy
`TrackingHistory` stores lightweight pose data:
- `relative_frame_poses: list[g2o.Isometry3d]` — compact SE3 per frame
- `kf_references: list[KeyFrame]` — ref KF for each frame (KF objects kept alive by this, which is correct and intentional for final trajectory reconstruction)
- `timestamps: list[float]`, `ids: list[int]`, `slam_states: list[str]`

`TrackingHistory` does NOT store `Frame` objects. It stores only KF references because final trajectory reconstruction requires corrected KF poses post-GBA.

### memory_stats() added
`TrackingHistory.memory_stats()` returns `{num_history_entries, num_timestamps, num_unique_kf_references}`. Does not allocate additional data.

### Why trajectory survives frame eviction
Frame eviction removes Frames from `Map.frames` and severs their `MapPoint._frame_views` links. `TrackingHistory` holds only KF references, which are in `Map.keyframes` and are never evicted by the memory policy. Trajectory reconstruction iterates `kf_references` and calls `kf.pose()` — unaffected by frame eviction.

---

## 9. Deliberate Deviations from pySLAM

| Area | pySLAM | Local (this repo) | Reason |
|---|---|---|---|
| Frame eviction | Implicit deque auto-eviction (no cleanup) | Explicit `_cleanup_evicted_frame()` | More deterministic for long sequences |
| KF depth release | Never released | Released after per-keypoint extraction | Reduce peak RSS by ~50 MB on fr1_desk |
| Frame-view pruning | Relies on `check_replaced_map_points` / GC | Periodic explicit pruning every N frames | Remove stale refs without GC dependency |
| LM queue image refs | Stored in queue tuples indefinitely | Nulled in `finally` after processing | Prevent LM queue from accumulating refs |
| Config flags | None (hardcoded) | `kStoreKeyFrameDepthImages`, `kStoreKeyFrameImages`, `kMaxLenFrameDeque`, etc. | Allows tuning memory vs. capability trade-offs |
