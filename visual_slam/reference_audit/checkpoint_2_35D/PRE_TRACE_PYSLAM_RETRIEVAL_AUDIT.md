# Checkpoint 2.35D — PRE_TRACE_PYSLAM_RETRIEVAL_AUDIT

## 1. pySLAM files/functions inspected

- `third_party/pyslam_reference/pyslam/loop_closing/keyframe_database.py`
  - `KeyFrameDatabaseDBow.detect_loop_candidates()`
  - `KeyFrameDatabaseDBow.detect_relocalization_candidates()`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_closing.py`
  - `LoopGroupConsistencyChecker.check_candidates()`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow3.py`
  - `LoopDetectorDBoW3.db_query()`
  - `LoopDetectorDBoW3.run_task()`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow2.py`
  - `LoopDetectorDBoW2.db_query()`
  - `LoopDetectorDBoW2.run_task()`
- `third_party/pyslam_reference/pyslam/slam/keyframe.py`
- `third_party/pyslam_reference/pyslam/slam/map_point.py`

## 2. local files/functions inspected

- `visual_slam/orbslam/slam/keyframe_database.py`
  - `detect_loop_candidates()`
  - `_detect_loop_candidates_dbow3_raw()`
  - `_detect_loop_candidates_dbow3_scored()`
  - `_detect_loop_candidates_inverted_scored()`
  - `_score_candidate_pool()`
- `visual_slam/orbslam/slam/loop_detector.py`
  - `compute_reference_similarity_score()`
  - `detect()`
- `visual_slam/orbslam/slam/loop_closing.py`
  - `LoopGroupConsistencyChecker.check_candidates()`
  - `_build_loop_debug_records()`
  - `_populate_retrieval_diagnostics()`
  - `_finalize_candidate_diagnostics()`
  - `_build_loop_candidate_oracle_rows()`
- `visual_slam/orbslam/run_rgbd_slam.py`
  - runner flags
  - loop debug CSV emission

## 3. pySLAM retrieval stages

### 3.1 Classic inverted-file / ORB-SLAM-style path

Reference function: `third_party/pyslam_reference/pyslam/loop_closing/keyframe_database.py:KeyFrameDatabaseDBow.detect_loop_candidates()`

Observed control flow:

1. Build `sp_connected_keyframes = keyframe.get_connected_keyframes()`.
2. Iterate words in current BoW vector.
3. For each word, scan the inverted file and accumulate `num_loop_words`.
4. Exclude connected keyframes during collection.
5. Compute `max_common_words`.
6. Compute `min_common_words = int(max_common_words * 0.8)`.
7. Score only candidates with `num_loop_words > min_common_words`.
8. Apply `si >= min_score`.
9. For each surviving candidate, accumulate covisibility scores over top-10 neighbors that:
   - belong to the same query (`loop_query_id == keyframe.id`)
   - also satisfy `num_loop_words > min_common_words`
10. Track `best_acc_score`.
11. Retain candidates with `acc_score > 0.75 * best_acc_score`.
12. Deduplicate by best representative keyframe.
13. Hand retained candidates to loop consistency checking.

### 3.2 pySLAM DBoW detector path

Reference functions:

- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow3.py:LoopDetectorDBoW3.run_task()`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow2.py:LoopDetectorDBoW2.run_task()`

Observed control flow:

1. Compute `min_score` from connected keyframes.
2. Query the database for a bounded top-K result set.
3. Filter returned results by:
   - temporal gap
   - `score >= min_score` or `score > min_score`
   - connected-keyframe exclusion
4. Emit candidates directly from the bounded DB query result.

Important 2.35D note:

- pySLAM therefore exposes two distinct retrieval shapes:
  - classic inverted-file ORB-SLAM-style common-word/minScore/accumulation logic
  - separate bounded DBoW detector query logic
- the local repository currently combines these ideas: it obtains a raw DBoW3 pool, then re-scores it with ORB-SLAM-style common-word/minScore/accumulation filtering.

## 4. local retrieval stages

### 4.1 Current local hybrid path

Reference function chain:

- `LoopDetector.compute_reference_similarity_score()`
- `KeyFrameDatabase.detect_loop_candidates()`
- `_detect_loop_candidates_dbow3_raw()`
- `_detect_loop_candidates_dbow3_scored()`
- `_score_candidate_pool()`
- `LoopGroupConsistencyChecker.check_candidates()`

Observed control flow:

1. Compute `min_score` from connected keyframes.
2. Run `_detect_loop_candidates_dbow3_raw()`:
   - query native DBoW3 with `max_results = database size`
   - drop self, bad, connected, and temporally-near candidates immediately
3. Run `_detect_loop_candidates_dbow3_scored()`:
   - convert raw DBoW pool into a candidate pool
   - call `_score_candidate_pool()`
4. Run `_detect_loop_candidates_inverted_scored()`:
   - build candidate pool from inverted-file shared words
   - call `_score_candidate_pool()`
5. In `_score_candidate_pool()`:
   - deduplicate raw pool
   - apply temporal filter
   - apply connected filter
   - compute shared-word counts by BoW-set intersection
   - compute `max_common_words`
   - gate by `common_words > int(max_common_words * 0.8)`
   - compute BoW score
   - gate by `score >= min_score`
   - accumulate covisibility score across top-10 neighbors
   - retain candidates with `acc_score > 0.75 * best_acc_score`
   - rank retained candidates
6. `LoopClosing` hands retained candidates to consistency.
7. Consistency survivors go to geometry.
8. Geometry survivor, if any, becomes accepted loop.

## 5. exact diagnostic fields required for every stage

### 5.1 Raw DBOW stage

Required fields:

- current keyframe id/timestamp
- candidate id/timestamp
- raw rank
- raw score
- raw source
- raw query top-K
- raw result count
- database size before query
- self/bad/connected/temporal annotations

### 5.2 Inverted/shared-word visibility stage

Required fields:

- candidate shares at least one word with current keyframe
- shared-word count
- max common words
- common-word ratio
- common-word threshold ratio
- connected / temporal status

### 5.3 Score filter stage

Required fields:

- BoW score
- min_score
- ratio to min_score
- connected-keyframe scores used to derive `min_score`
- source connected keyframe for the minimum

### 5.4 Accumulation / retention stage

Required fields:

- candidate covisibility group ids
- candidate group member scores
- accumulated score
- best accumulated score
- retention threshold ratio
- retained yes/no
- retained rank

### 5.5 Consistency / acceptance stage

Required fields:

- retained candidate handoff
- consistency count before/after
- pass/fail consistency
- geometry pass/fail
- final support pass/fail
- accepted yes/no

### 5.6 GT-positive per-pair trace

Required fields:

- all stage-presence booleans for every GT-loop-like pair
- first failed stage
- rejection reason
- confidence / unknown explanation

## 6. where those fields can be collected in local code

- Raw DBOW query metadata:
  - `visual_slam/orbslam/slam/keyframe_database.py:_detect_loop_candidates_dbow3_raw()`
- Inverted/shared-word candidate identity:
  - `visual_slam/orbslam/slam/keyframe_database.py:_detect_loop_candidates_inverted_scored()`
  - `visual_slam/orbslam/slam/keyframe_database.py:_score_candidate_pool()`
- Common-word / minScore / accumulation details:
  - `visual_slam/orbslam/slam/keyframe_database.py:_score_candidate_pool()`
- Retained-candidate and consistency details:
  - `visual_slam/orbslam/slam/loop_closing.py`
  - `visual_slam/orbslam/slam/loop_detector.py`
- GT diagnostics only:
  - `visual_slam/orbslam/slam/loop_closing.py` using `TumLoopOracle`
- Runtime file emission:
  - `visual_slam/orbslam/run_rgbd_slam.py`
- Offline stage funnel:
  - new tool `tools/analyze_gt_loop_raw_retrieval_trace.py`

## 7. proof that planned instrumentation will not change loop decisions

Planned guardrails:

1. All new tracing is enabled only by a dedicated runner flag.
2. Trace data will be appended to side-channel data structures only.
3. No new trace data will feed back into:
   - candidate filtering
   - thresholds
   - consistency
   - geometry
   - correction
4. GT data will be read only after runtime candidate generation and used only to annotate trace rows.
5. If a higher raw DBOW query limit is added for tracing, it will be diagnostic-only and must not replace the actual candidate pool used by the existing loop decision path.
6. Existing retained candidate lists will be regression-tested against trace-enabled runs.

## 8. missing information from 2.35B / 2.35C that this checkpoint will recover

Known missing information before 2.35D:

- whether a GT-positive pair was present in raw DBOW retrieval at all
- raw DBOW rank and raw DBOW score for that pair
- whether the pair existed in the inverted/shared-word set
- whether it failed connected or temporal filtering
- whether it failed the common-word gate
- whether it failed minScore
- whether it failed accumulation retention
- whether it survived retention but failed consistency

Checkpoint 2.35D instrumentation target:

- replace broad `NOT_RETRIEVED` with an exact first-failure stage for each GT-loop-like pair whenever the runtime trace makes that stage observable
- identify the dominant first-failure stage across all GT-loop-like pairs so the next checkpoint can be a targeted correction rather than another retrieval audit
