# Checkpoint 2.32A - Pre-Change pySLAM Local Map Audit

## 1. pySLAM files/functions inspected

- `third_party/pyslam_reference` git commit: `a95db3a0e95764b8c68b81fade544bdd6ecb912e`
- `third_party/pyslam_reference/pyslam/slam/tracking.py`
  - `Tracking.update_local_map`
  - `Tracking.track_local_map`
- `third_party/pyslam_reference/pyslam/slam/map.py`
  - `LocalMapBase.get_frame_covisibles`
  - `LocalMapBase.update_from_keyframes`
  - `LocalCovisibilityMap.update_keyframes`
- `third_party/pyslam_reference/pyslam/slam/keyframe.py`
  - `KeyFrameGraph.get_best_covisible_keyframes`
  - `KeyFrameGraph.get_children`
  - `KeyFrameGraph.get_parent`
- `third_party/pyslam_reference/pyslam/slam/map_point.py`
  - `MapPointBase.last_frame_id_seen`
  - observation/keyframe accessors
- `third_party/pyslam_reference/pyslam/slam/geometry_matchers.py`
  - `_search_map_by_projection`
- `third_party/pyslam_reference/pyslam/slam/frame.py`
  - `Frame.clean_bad_map_points`
  - `Frame.get_matched_good_points`

## 2. Local files/functions inspected

- `visual_slam/orbslam/slam/tracking.py`
  - `Tracking.update_local_map`
  - `Tracking.track_local_map`
  - `Tracking._elect_best_kf_ref`
- `visual_slam/orbslam/slam/map.py`
  - `LocalCovisibilityMap.update`
  - `Map.update_local_map`
- `visual_slam/orbslam/slam/keyframe.py`
  - `KeyFrameGraph.get_best_covisible_keyframes`
  - spanning-tree helpers
- `visual_slam/orbslam/slam/map_point.py`
  - `MapPointBase.last_frame_id_seen`
  - observation/keyframe helpers
- `visual_slam/orbslam/slam/geometry_matchers.py`
  - `_search_map_by_projection`
  - `_prepare_visible_projection_candidates`
- `visual_slam/orbslam/slam/local_mapping.py`
- `visual_slam/orbslam/slam/local_mapping_core.py`
- `visual_slam/orbslam/slam/config_parameters.py`
- `visual_slam/orbslam/run_rgbd_slam.py`

## 3. pySLAM local map construction control flow

pySLAM `Tracking.update_local_map()` first calls `f_cur.clean_bad_map_points()`, then calls `map.local_map.get_frame_covisibles(f_cur)`.

`LocalMapBase.get_frame_covisibles(frame)` performs the tracking local-map construction:

1. Start from `frame.get_matched_good_points()`.
2. Count the keyframes observing those current-frame points.
3. Select `kf_ref` as the most common observing keyframe.
4. Initialize the local keyframe list from the voted keyframes.
5. Iterate the growing local keyframe list until `Parameters.kMaxNumOfKeyframesInLocalMap`.
6. For each local keyframe, add at most one valid best covisible keyframe from `get_best_covisible_keyframes(Parameters.kNumBestCovisibilityKeyFrames)`.
7. Add at most one valid child.
8. Add the valid parent if missing.
9. Keep the top `Parameters.kMaxNumOfKeyframesInLocalMap` keyframes by vote/count.
10. Build local points from those local keyframes.

pySLAM `_search_map_by_projection()` then projects local points and filters candidates with visibility, bad-point, and `last_frame_id_seen != f_cur.id` checks before descriptor matching.

pySLAM `Frame.clean_bad_map_points()` marks non-bad matched points as seen in the current frame and increments visibility before local projection search.

## 4. Local current local map construction control flow

Local `Tracking.update_local_map()` currently:

1. Cleans bad current-frame points.
2. Chooses a fallback reference from `self.kf_ref`, `self.kf_last`, or the last valid map keyframe.
3. Calls `Map.update_local_map(reference)`.
4. `LocalCovisibilityMap.update(reference)` starts from that reference keyframe.
5. It adds all covisible keyframes from `reference_keyframe.get_covisible_keyframes()`.
6. It collects local points from those keyframes.
7. It derives reference keyframes from local point observations.
8. Only after local points/keyframes are built, `Tracking._elect_best_kf_ref()` votes from current-frame matched points and updates `self.kf_ref`.

Local `_search_map_by_projection()` computes visible projection candidates and KD queries before applying the `last_frame_id_seen != f_cur.id` skip.

Local `Frame.clean_bad_map_points()` only removes bad points; it does not mark already-matched non-bad points as seen for the current frame.

## 5. Seven gaps confirmed/refuted

| Gap | Status | Evidence |
| --- | --- | --- |
| 1. Local map starts from old/reference keyframe | Confirmed | `Tracking.update_local_map()` calls `Map.update_local_map(reference)`, and `LocalCovisibilityMap.update()` starts from `reference_keyframe`. |
| 2. Reference selected after local map construction | Confirmed | `_elect_best_kf_ref()` is called after `self.local_points` is populated. |
| 3. Local keyframes not built from current-frame vote set | Confirmed | Initial set is `reference_keyframe` plus its covisibles, not the keyframes observing current-frame matches. |
| 4. Expansion includes too many/all covisibles | Confirmed | `LocalCovisibilityMap.update()` iterates `reference_keyframe.get_covisible_keyframes()` without a bound. |
| 5. `num_best` not honored | Confirmed | `LocalCovisibilityMap.update(reference_keyframe, num_best=...)` accepts `num_best` but does not use it. |
| 6. Local points not selected with pySLAM-style per-frame local-map marker | Confirmed | Local code uses `OrderedSetLite` uniqueness only; `MapPoint` has no local-map marker separate from `last_frame_id_seen`. |
| 7. Already-matched current-frame map points not marked seen before projection | Confirmed | Local `Frame.clean_bad_map_points()` does not mark non-bad points seen, and `Tracking.track_local_map()` does not call an equivalent marker method. |

## 6. Additional gaps found

- Projection diagnostics do not currently expose rejected bad, already-seen, visibility, KD candidate, descriptor-comparison, or projection-match counts.
- The runner has runtime and memory profiling outputs, but no `--profile-local-map` flag and no `local_map_profile.csv`.
- Local `KeyFrameGraph.get_best_covisible_keyframes(N)` already orders by weight and honors `N`, so the missing behavior is in local-map construction rather than the graph accessor.
- Parent and child expansion support exists locally, so pySLAM-style bounded parent/child expansion can be implemented without adding new graph concepts.

## 7. Exact implementation plan

1. Add local-map and projection diagnostics only:
   - Add `--profile-local-map`.
   - Add `local_map_profile.csv` with the checkpoint-required columns.
   - Add optional diagnostics dictionary support to `ProjectionMatcher.search_map_by_projection()` without changing baseline filtering order.
2. Run the required 30-frame baseline on `rgbd_dataset_freiburg1_desk` and write `BASELINE_LOCAL_MAP_DIAGNOSTIC_REPORT.md`.
3. Implement pySLAM-aligned local map construction in `Tracking`/`LocalCovisibilityMap`:
   - collect current-frame observation votes,
   - select reference before local point construction,
   - initialize local keyframes from voted keyframes,
   - bounded covisibility/child/parent expansion,
   - use `num_best`,
   - add per-frame local-point markers,
   - mark already-matched current-frame points seen before projection.
4. Add targeted tests in `tests/visual_slam/orbslam/test_checkpoint_2_32A_local_map_pyslam_alignment.py`.
5. Run targeted tests, then `tests/visual_slam/orbslam`.
6. Run only the post-change 30-frame comparison and write implementation/validation reports.
7. Print the 100-frame command for confirmation without running it.

## 8. Risks and safeguards

- Risk: Changing local-map construction can change keyframe insertion behavior via `self.kf_ref`. Safeguard: explicit tests for max-vote reference selection and no-vote fallback.
- Risk: Projection diagnostics can perturb runtime. Safeguard: diagnostics are optional and only populated when local-map profiling is enabled.
- Risk: Marking already-matched points seen could suppress valid projection matches if `last_frame_id_seen` is conflated with local-point uniqueness. Safeguard: add a separate local-map marker and tests that prove the two markers have distinct meanings.
- Risk: Parent/child expansion order can differ because local containers are Python lists/sets. Safeguard: preserve pySLAM's control-flow concepts and document any deterministic-order deviation in the alignment report.
