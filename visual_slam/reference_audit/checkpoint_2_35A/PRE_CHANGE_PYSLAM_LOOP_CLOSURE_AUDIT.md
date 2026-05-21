# Checkpoint 2.35A Pre-Change pySLAM Loop-Closure Audit

## 1. pySLAM files/functions inspected

- `third_party/pyslam_reference/pyslam/loop_closing/keyframe_database.py`
  - `KeyFrameDatabaseDBow.detect_loop_candidates`
  - `KeyFrameDatabaseDBow.detect_relocalization_candidates`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_closing.py`
  - `LoopClosing.add_keyframe`
  - `LoopClosing.run`
  - `LoopGroupConsistencyChecker.check_candidates`
  - `LoopGeometryChecker.check_candidates`
  - `LoopCorrector.correct_loop`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_base.py`
  - `LoopDetectorBase.compute_reference_similarity_score`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow2.py`
  - `LoopDetectorDBoW2.db_query`
  - `LoopDetectorDBoW2.run_task`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow3.py`
  - `LoopDetectorDBoW3.db_query`
  - `LoopDetectorDBoW3.run_task`
- `third_party/pyslam_reference/pyslam/slam/geometry_matchers.py`
  - `_search_by_sim3`
  - `_search_more_map_points_by_projection`

## 2. Local files/functions inspected

- `visual_slam/orbslam/slam/keyframe_database.py`
  - `KeyFrameDatabase.add`
  - `KeyFrameDatabase.detect_loop_candidates`
  - `KeyFrameDatabase._detect_loop_candidates_dbow3`
- `visual_slam/orbslam/slam/loop_detector.py`
  - `LoopDetector.compute_reference_similarity_score`
  - `LoopDetector.detect`
- `visual_slam/orbslam/slam/loop_closing.py`
  - `LoopClosing.process_keyframe`
  - `LoopClosing._build_loop_debug_records`
  - `LoopClosing._merge_consistency_debug`
  - `LoopClosing._merge_geometry_debug`
  - `LoopGroupConsistencyChecker.check_candidates`
  - `LoopGeometryChecker.check_candidates`
- `visual_slam/orbslam/slam/geometry_matchers.py`
  - `_search_more_map_points_by_projection`
- `visual_slam/orbslam/slam/bow_matcher.py`
  - `BoWGuidedMatcher.match`
- `visual_slam/orbslam/slam/config_parameters.py`
- `visual_slam/orbslam/slam/slam.py`
- `visual_slam/orbslam/run_rgbd_slam.py`
- `tools/evaluate_tum_trajectory.py`
- `tools/probe_loop_candidates_fr1_room.py`

## 3. pySLAM loop candidate retrieval control flow

Observed pySLAM reference behavior:

1. `LoopClosing.add_keyframe()` packages a loop-detection task with:
   - current keyframe
   - covisible keyframes
   - connected keyframes
2. `LoopDetectingProcess` runs a detector backend such as DBoW2 or DBoW3.
3. Detector computes `min_score` as the minimum BoW similarity to connected/covisible keyframes.
4. Detector returns candidate ids/scores.
5. `LoopClosing.run()` converts candidate ids back to keyframes, then performs:
   - consistency grouping
   - geometry verification
   - loop correction

Important nuance:

- `loop_detector_dbow2.py` can use a `KeyFrameDatabase` path that already implements ORB-SLAM-style common-word filtering, minScore filtering, covisibility accumulation, and 0.75-best retention.
- `loop_detector_dbow3.py` does a native database query and then lightweight filtering on:
  - temporal gap
  - score >= `min_score`
  - not connected

So the pySLAM reference itself is not fully uniform across candidate sources.

## 4. Local loop candidate retrieval control flow

Observed local behavior:

1. `LoopClosing.process_keyframe()` is called synchronously from the runner loop.
2. It immediately does `self.keyframe_database.add(keyframe)`.
3. It then calls `LoopDetector.detect(keyframe)`.
4. `LoopDetector.detect()` computes `min_score` from connected keyframes.
5. `KeyFrameDatabase.detect_loop_candidates()` is called.
6. If the native DBOW3 database exists, `_detect_loop_candidates_dbow3()` returns raw native query results after only:
   - self rejection
   - bad-keyframe rejection
   - connected-keyframe rejection
   - temporal-gap rejection
   - score >= `min_score`
7. If native DBOW3 is unavailable, only then does the code execute the inverted-file path with:
   - shared-word counting
   - max-common-word threshold
   - minScore filtering
   - covisibility accumulation
   - 0.75-best accumulated-score retention

This means the actual production path currently bypasses the more ORB-SLAM-like scoring logic whenever native DBOW3 is available.

## 5. pySLAM DBOW2/DBOW3 behavior and candidate scoring

### pySLAM DBoW2 path

`loop_detector_dbow2.py` delegates to a `KeyFrameDatabase`-style query path. That path uses:

1. shared visual words
2. connected-keyframe rejection
3. common-word threshold `> 0.8 * max_common_words`
4. BoW score computation
5. `score >= min_score`
6. covisibility accumulation over best covisible neighbors
7. retain candidates with accumulated score `> 0.75 * best_acc_score`

This is the cleanest ORB-SLAM-style reference in the pySLAM copy.

### pySLAM DBoW3 path

`loop_detector_dbow3.py`:

1. computes `min_score`
2. runs native DB query
3. filters top results by:
   - temporal separation
   - `score >= min_score`
   - not connected

It does not replicate the common-word / accumulated-score path used by `keyframe_database.py`.

## 6. Local DBOW3/native behavior and fallback behavior

### Local native DBOW3 path

`KeyFrameDatabase._detect_loop_candidates_dbow3()` currently:

1. queries native DBOW3
2. maps entry ids back to keyframes
3. filters by:
   - self
   - bad keyframe
   - connected keyframe
   - temporal gap
   - `score >= min_score`
4. returns candidates directly

Missing relative to ORB-SLAM-style retrieval:

- shared-word counting
- common-word thresholding
- covisibility score accumulation
- best accumulated score retention
- source-comparison diagnostics

### Local fallback inverted-file path

`KeyFrameDatabase.detect_loop_candidates()` fallback path already does:

1. shared-word counting
2. common-word threshold
3. BoW score computation
4. `score >= min_score`
5. covisibility accumulation
6. retain `> 0.75 * best_acc_score`

This path is structurally much closer to pySLAM/ORB-SLAM2.

## 7. pySLAM consistency group behavior

`LoopGroupConsistencyChecker.check_candidates()` in pySLAM:

1. expands each candidate into its connected-keyframe group plus itself
2. compares with previously consistent groups
3. increments consistency when any overlap exists
4. accepts only candidates that reach the configured threshold across consecutive detections

This is a temporal-grouping safeguard, not a scoring step.

## 8. Local consistency group behavior

Local `LoopGroupConsistencyChecker.check_candidates()` is structurally close to pySLAM:

1. expands candidate to connected group + self
2. compares overlap against previous groups
3. increments per-group consistency
4. exposes `passed_consistency` and debug counts

Current notable difference:

- local version stores richer per-candidate debug state, which is helpful
- no obvious structural bug was confirmed in consistency grouping itself

Current hypothesis:

- many consistency rejections may be downstream symptoms of unstable retrieval, not a consistency-group implementation bug

## 9. pySLAM geometry verification/projection expansion behavior

pySLAM geometry flow in `LoopGeometryChecker.check_candidates()`:

1. feature matching between current keyframe and loop candidates
2. orientation consistency filtering
3. Sim3/SE3 seed estimation
4. guided matching by projection / Sim3
5. optimization/refinement
6. gather loop keyframe covisible group
7. project more loop-side map points into current keyframe
8. require minimum matched map points before loop correction

This matches the classic ORB-SLAM family structure.

## 10. Local geometry verification/projection expansion behavior

Local `LoopGeometryChecker.check_candidates()` follows the same broad phases:

1. BoW-guided matching
2. RGB-D fixed-scale SE3 RANSAC seed
3. guided projection refinement
4. refined SE3 estimation
5. loop-side covisible-group map-point gathering
6. `search_more_map_points_by_projection`
7. final matched-map-point gate

Local extra safeguards currently present:

- estimated-pose distance gate before guided SE3 seeding
- estimated-pose rotation gate before guided SE3 seeding
- final matched-map-point threshold gate

These appear to be legitimate safety checks, but they are not the first thing to tune. The retrieval path must be trusted first.

## 11. Current suspected gaps confirmed/refuted

### Suspected issue A - native DBOW3 bypasses pySLAM-style scoring

Confirmed.

- `visual_slam/orbslam/slam/keyframe_database.py:162-170`
- `_detect_loop_candidates_dbow3()` returns directly when native DBOW3 is present.
- The actual production path therefore bypasses the common-word and accumulated-score pipeline.

### Suspected issue B - current keyframe may be added before query

Confirmed in local code.

- `visual_slam/orbslam/slam/loop_closing.py:1036`
- `self.keyframe_database.add(keyframe)` happens before `self.loop_detector.detect(keyframe)`.

Important note:

- The local pySLAM DBoW3 worker also appears to add before querying, but that happens inside a different async detector architecture and still relies on trivial-result filtering.
- For this repository's synchronous in-process keyframe database, query-before-add is the safer ORB-SLAM2-style control flow and matches the checkpoint's requested target.

### Suspected issue C - sparse keyframe scheduling may reduce loop geometry support

Not yet confirmed, but strongly plausible.

Evidence already available from 2.34A:

- full `fr1_room` completed with only `46` keyframes
- two late candidates reached `37/38` matched map points, below the final `60` gate

This must be diagnosed with GT-loop oracle data before making any scheduling change.

### Suspected issue D - geometric gates may be stricter than pySLAM needs

Plausible but not yet actionable.

- local code has estimated-pose distance/rotation gates and the final matched-map-point gate
- current failure pattern still points first at retrieval instability and then at sparse support

Conclusion:

- retrieval/control-flow gaps are confirmed and should be fixed before touching geometry gates

## 12. Implementation plan

Phase 0 diagnostics before behavior changes:

1. Add GT-backed loop oracle support for TUM `groundtruth.txt`
2. Add:
   - `loop_candidate_oracle.csv`
   - `loop_retrieval_profile.csv`
   - `loop_candidate_source_comparison.csv`
   - `loop_keyframe_density_profile.csv`
3. Add runner/config plumbing for `--loop-candidate-source`
4. Run baseline full `fr1_room` loop-oracle diagnostic

Phase 1 structural retrieval fixes:

1. Change local loop-closing flow to query candidates before adding the current keyframe to the DB
2. Replace the raw-return native DBOW3 path with unified pySLAM-style candidate scoring
3. Add compare-mode diagnostics:
   - DBOW3 broad-query candidates
   - inverted-file candidates
   - chosen final candidates

Phase 2 validation:

1. targeted tests
2. non-C++ visual SLAM test slice
3. post-retrieval-fix full `fr1_room` run
4. density/geometry-gate analysis reports

## 13. Risks and safeguards

Risks:

- adding diagnostics can increase loop-debug artifact volume
- compare mode can increase runtime modestly
- changing query/add order may expose assumptions in current DB bookkeeping

Safeguards:

- keep loop oracle diagnostic-only
- do not change loop thresholds or consistency thresholds in this checkpoint
- do not change tracking, LocalMapping scheduling, keyframe insertion, C++ code, or GBA
- validate with targeted tests before rerunning the full sequence
