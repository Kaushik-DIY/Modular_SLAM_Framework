# PRE-CHANGE pySLAM MEMORY POLICY AUDIT

## 1. pySLAM memory policy summary
The `pySLAM` repository implements a clean and robust memory policy where:
- The map only retains a short queue of recent frames (`Map.frames`) to bound memory usage from continuous normal frame processing.
- Long-term structure is preserved through `KeyFrames` and `MapPoints`.
- `MapPoint._observations` are long-term records that only store keyframe observations, used for BA and covisibility.
- `MapPoint._frame_views` are temporary records for normal frames, actively pruned as frames exit the recent-frame window.
- The final trajectory is constructed from lightweight tracking history containing relative poses and references, not by iterating over full `Frame` objects.
- Image memory is tightly controlled: normal frames release images quickly once feature matching succeeds; keyframes can release `depth_img` once point coordinates and virtual stereo parameters are set.

## 2. Local current memory policy summary
Locally, the map attempts to follow pySLAM by using a `deque(maxlen=20)` for `Map.frames`. However, memory is still leaking over long runs. Currently, `Map.frames` bounds the *reference* to frames in the map, but old frames are not explicitly cleaned up before eviction. There is no active pruning of `MapPoint` references to these evicted frames, meaning the evicted frames can be kept alive indefinitely. Images (`img` and `depth_img`) remain attached to these leaked frames. Keyframes hold onto both RGB and depth images indefinitely. The local mapping queue also may retain redundant image copies.

## 3. Which pySLAM memory behavior is already implemented locally
- The use of `deque(maxlen=20)` for `Map.frames`.
- The distinction between KeyFrames and regular Frames.
- Basic storage of `TrackingHistory`.

## 4. Which behavior is missing or incomplete locally
- Explicit cleanup logic (`_cleanup_evicted_frame`) upon deque eviction.
- Explicit pruning of old `_frame_views` from `MapPoint` objects.
- Releasing heavy components (`img`, `depth_img`, `img_right`, `kd`) from evicted normal frames.
- Releasing heavy components (`depth_img`) from `KeyFrame` objects once processed.
- Releasing queue item images from `LocalMapping`.
- Guaranteeing trajectory extraction relies only on `TrackingHistory` (although current trajectory extraction might already partially use it, tests need to verify independence from full frame objects).
- Strict severing of all frame views and keyframe observations when a `MapPoint` goes bad or is replaced.
- Active memory monitoring and limit enforcement via runner flags.

## 5. Current risks
- Old normal frames are retained in memory via leaked references (mostly `_frame_views` in `MapPoint`).
- Retained normal frames keep heavy RGB/depth arrays and KD-trees in memory, causing OOM on long sequences.
- Local mapping queue retaining images/depth.
- Debug outputs and logs retaining large diagnostic objects indefinitely.
- Potentially tying the final trajectory output to full `Frame` objects.

## 6. Exact implementation plan
1. **Map bounds:** Add `Parameters.kMaxLenFrameDeque`, explicit frame eviction inside `Map.add_frame()` that calls `_cleanup_evicted_frame(frame)`.
2. **Frame views pruning:** Implement `MapPoint.remove_frame_views_older_than()` and `Map.prune_old_frame_views()`. Call the pruning function periodically.
3. **Image retention:** Add `Frame.release_images()` and `release_heavy_data()`. Add configuration toggles `kStoreNormalFrameImages`, `kStoreKeyFrameImages`, `kStoreKeyFrameDepthImages`. Apply these to normal frames upon eviction and keyframes after setup.
4. **Local Mapping Queue:** Set `self.img_cur = None` and clear heavy references on processed queue items.
5. **Memory profiling:** Add `--profile-memory`, `--lean-memory` flags, `Map.memory_stats()`, and `memory_profile.csv` output to the runner.
6. **Tests:** Implement validation tests enforcing reference clearing on point replacement, checking trajectory reconstruction resilience against frame eviction, and confirming images are released.
