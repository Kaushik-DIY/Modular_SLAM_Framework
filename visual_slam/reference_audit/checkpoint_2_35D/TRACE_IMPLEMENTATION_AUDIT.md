# Checkpoint 2.35D — TRACE_IMPLEMENTATION_AUDIT

## 1. Files modified

- `visual_slam/orbslam/slam/keyframe_database.py`
- `visual_slam/orbslam/slam/loop_detector.py`
- `visual_slam/orbslam/slam/loop_closing.py`
- `visual_slam/orbslam/run_rgbd_slam.py`
- `tools/analyze_gt_loop_raw_retrieval_trace.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_35D_raw_retrieval_trace.py`

## 2. Why each modification is diagnostic-only

### `visual_slam/orbslam/slam/keyframe_database.py`

- Added trace-only side-channel storage for:
  - raw DBOW query rows
  - inverted/shared-word rows
  - minScore rows
  - accumulation/retention rows
- Kept the existing chosen candidate list and scoring path intact.
- Did not change thresholds or acceptance logic.

### `visual_slam/orbslam/slam/loop_detector.py`

- Added minScore context capture so the trace can record:
  - connected keyframe count
  - min-score source keyframe id
  - connected scores used to derive minScore
- Returned the same `min_score` scalar used before.

### `visual_slam/orbslam/slam/loop_closing.py`

- Added trace-only storage for:
  - retained candidate rows
  - GT-positive per-pair rows
- Added a retrieval-trace config setter.
- Added GT-positive classification into first-failure stages.
- GT is still used only to annotate diagnostics.

### `visual_slam/orbslam/run_rgbd_slam.py`

- Added runner flags:
  - `--loop-retrieval-trace`
  - `--loop-retrieval-trace-raw-k`
- Added CSV emission for the new trace files.
- Added no new control-flow branches that affect loop acceptance.

### `tools/analyze_gt_loop_raw_retrieval_trace.py`

- Added offline funnel / summary / false-negative analysis from the emitted trace files.
- This runs after SLAM completion and cannot affect runtime decisions.

### `tests/visual_slam/orbslam/test_checkpoint_2_35D_raw_retrieval_trace.py`

- Added synthetic-only validation for:
  - required columns
  - first-failure-stage classification
  - analyzer funnel construction
  - non-behavioral trace guarantee

## 3. How the code guarantees loop decisions are unchanged

1. Existing runtime thresholds are untouched:
   - common-word threshold ratio
   - minScore threshold
   - accumulation retention ratio
   - consistency threshold
   - geometry thresholds
   - final support threshold
2. The actual candidate list still comes from the same `detect_loop_candidates()` return path.
3. The new trace rows are collected in parallel dictionaries/lists only.
4. The trace flag controls only recording and CSV emission.
5. The runner does not enable any alternate candidate source or fallback because of GT.

## 4. How GT is kept out of runtime decisions

1. GT is read only through `TumLoopOracle`.
2. GT annotations are applied after candidate retrieval.
3. GT-positive rows are written only into diagnostic CSVs.
4. No GT pair is inserted into:
   - retained candidate lists
   - consistency checking
   - geometry checking
   - loop correction
5. No GT-derived scalar feeds back into thresholds or scoring.

## 5. Trace files produced

The full diagnostic run now emits:

- `loop_raw_dbow_trace.csv`
- `loop_inverted_word_trace.csv`
- `loop_score_filter_trace.csv`
- `loop_accumulation_trace.csv`
- `loop_retained_candidate_trace.csv`
- `loop_gt_positive_trace.csv`

The offline analyzer emits:

- `gt_retrieval_stage_funnel.csv`
- `gt_retrieval_stage_summary.json`
- `gt_retrieval_false_negatives_detailed.csv`
- `gt_retrieval_stage_report.md`

## 6. Important analysis caveat

The first 2.35D runtime trace was intentionally kept as a superset and included some short-gap GT-near pairs that 2.35B did not treat as meaningful loop pairs.

For the final funnel, the analyzer therefore used:

- current 2.35D runtime GT trace as the authoritative source of per-stage evidence
- a meaningful keyframe-gap filter of `>10`
- the 2.35B `gt_loop_pairs_classified.csv` only as a historical reference count

This did not affect loop behavior. It only tightened the offline analysis population to match meaningful loop-closure scope.
