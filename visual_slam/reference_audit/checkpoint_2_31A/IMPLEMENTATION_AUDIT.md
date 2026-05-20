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
