# Checkpoint 2.32A - Implementation Alignment Report

## Summary

Implemented the seven local-map/projection-search parity gaps against pySLAM commit `a95db3a0e95764b8c68b81fade544bdd6ecb912e`.

The implementation does not add a blind local-map-point cap and does not tune feature extraction, BA, loop, camera/depth, descriptor, or keyframe insertion thresholds.

## Files changed

- `visual_slam/orbslam/slam/config_parameters.py`
- `visual_slam/orbslam/slam/keyframe.py`
- `visual_slam/orbslam/slam/map_point.py`
- `visual_slam/orbslam/slam/map.py`
- `visual_slam/orbslam/slam/geometry_matchers.py`
- `visual_slam/orbslam/slam/tracking.py`
- `visual_slam/orbslam/run_rgbd_slam.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_32A_local_map_pyslam_alignment.py`

## Gap-by-gap alignment

| Gap | pySLAM reference | Local before | Local after | Score | Deviation | Test evidence |
| --- | --- | --- | --- | ---: | --- | --- |
| 1. Start local map from current-frame matched points | `pyslam/slam/map.py::LocalMapBase.get_frame_covisibles` | `Tracking.update_local_map` started from `kf_ref`/fallback reference. | `Tracking._collect_local_keyframe_votes_from_current_frame` votes keyframes from current matched good map points. | 99% | Uses `point.observations()` instead of pySLAM `point.keyframes()` because local observations expose keypoint indices too. | `test_local_keyframe_voting_uses_current_frame_matched_points`, `test_local_keyframe_voting_ignores_bad_points_and_bad_keyframes` |
| 2. Select reference before local map construction | `LocalMapBase.get_frame_covisibles` selects `most_common(1)[0][0]` before building local points. | `_elect_best_kf_ref()` ran after `Map.update_local_map(reference)`. | `Tracking.update_local_map` selects `self.kf_ref` from max vote before local keyframes/points are built. | 99% | Explicit no-vote fallback to existing valid reference, last keyframe, then last valid map keyframe per checkpoint requirement. | `test_reference_keyframe_selected_by_max_vote_before_local_points`, `test_fallback_to_existing_reference_when_no_votes` |
| 3. Build local keyframes from voted keyframes | `LocalMapBase.get_frame_covisibles` initializes `local_keyframes_list = list(viewing_keyframes.keys())`. | Initial local set was reference plus covisibles. | `Tracking._build_local_keyframes_from_votes` initializes from vote keys and preserves vote counts. | 99% | Local keyframes use local marker fields for diagnostics; pySLAM stores counts in a `Counter`. | `test_local_keyframes_include_voted_keyframes`, `test_local_keyframes_do_not_start_from_all_reference_covisibles` |
| 4. Bounded pySLAM-style expansion | `LocalMapBase.get_frame_covisibles` adds at most one best covisible, one child, and parent per local keyframe until max local keyframes. | `LocalCovisibilityMap.update` added all covisibles of the reference keyframe. | `Tracking._build_local_keyframes_from_votes` uses one best covisible, one child, and parent with duplicate/bad checks. | 97% | Child order comes from local `set` storage, as in local graph API; tests assert bounded count, not a specific child identity. | `test_parent_and_child_expansion_are_bounded`, `test_bad_keyframes_are_not_added_during_expansion` |
| 5. Honor `num_best` | `KeyFrameGraph.get_best_covisible_keyframes(N)` and `Parameters.kNumBestCovisibilityKeyFrames`. | `LocalCovisibilityMap.update(..., num_best)` ignored the argument. | Tracking expansion and fallback `Map.update_local_map(..., num_best)` use `get_best_covisible_keyframes(num_best)`. | 99% | Local fallback map update now honors `num_best`; tracking expansion still follows pySLAM's one-neighbor-per-keyframe append behavior. | `test_num_best_covisibility_keyframes_is_honored`, `test_get_best_covisible_keyframes_orders_by_weight` |
| 6. Unique local points via per-frame marker | ORB-SLAM/pySLAM tracking marker concept; local map uses unique viewed points. | Used `OrderedSetLite` only; no per-current-frame local-point marker separate from seen marker. | `MapPoint.last_track_reference_frame_id` and `Tracking._collect_local_points_from_keyframes` ensure one local-point insertion per frame. | 98% | Marker field name is local, but semantics match the checkpoint target and remain separate from `last_frame_id_seen`. | `test_local_points_are_unique_by_frame_marker`, `test_local_point_marker_is_frame_specific`, `test_bad_points_are_skipped_when_collecting_local_points` |
| 7. Mark already-matched points seen before projection | `pyslam/slam/frame.py::Frame.clean_bad_map_points`; `geometry_matchers.py::_search_map_by_projection` skips `last_frame_id_seen == f_cur.id`. | Local `clean_bad_map_points` did not mark non-bad matches seen, and projection skipped after projection/KD work. | `Tracking._mark_current_frame_matched_points_seen` runs before projection; `_search_map_by_projection` filters bad/already-seen before projection and KD query. | 99% | Local keeps this marker in `Tracking` rather than changing `Frame.clean_bad_map_points` to avoid broader frame API side effects. | `test_already_matched_points_are_marked_seen_before_projection`, `test_search_map_by_projection_skips_last_frame_seen_points_before_projection`, `test_projection_diagnostics_count_rejected_already_seen_points` |

No completed gap is below 97% alignment.

## Structural correctness

- The local-map workflow now follows pySLAM order: current-frame matches, keyframe votes, reference selection, voted keyframes, bounded expansion, local points, then projection search.
- `last_track_reference_frame_id` and `last_frame_id_seen` have separate meanings:
  - `last_track_reference_frame_id`: local-map point uniqueness for a tracking frame.
  - `last_frame_id_seen`: already-matched/seen skip before projection search.
- `num_best` is now honored by both tracking expansion and fallback covisibility map update.
- The C++ extension path was not modified.

## Test evidence

Targeted checkpoint test:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/visual_slam/orbslam/test_checkpoint_2_32A_local_map_pyslam_alignment.py
```

Result: `17 passed in 0.79s`.

Python/non-C++ visual SLAM slice:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam -k 'not cpp_slam_core'
```

Result: `278 passed, 1 skipped, 94 deselected in 27.39s`.

Full visual SLAM slice:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam
```

Result: blocked by a standalone native segmentation fault in `tests/visual_slam/orbslam/test_cpp_slam_core_phase2_frame.py::test_reset_points`.

Standalone repro:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/visual_slam/orbslam/test_cpp_slam_core_phase2_frame.py::TestFramePointMatch::test_reset_points
```

Result: segmentation fault. This checkpoint did not modify the C++ extension, and the task explicitly excludes C++ extension changes.

User waiver: this standalone C++ extension segfault was explicitly accepted as out of scope for Checkpoint 2.32A, so the required 30-frame post-change dataset comparison was run after documenting the blocker.
