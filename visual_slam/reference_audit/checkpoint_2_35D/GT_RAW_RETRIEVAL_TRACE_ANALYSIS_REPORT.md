# Checkpoint 2.35D — GT_RAW_RETRIEVAL_TRACE_ANALYSIS_REPORT

## 1. Objective

Identify the exact first retrieval-stage loss for GT-loop-like pairs and isolate the dominant failure stage before any targeted loop-closure correction.

## 2. Exact command used

### Full diagnostic run

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
python -m visual_slam.orbslam.run_rgbd_slam \
  "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room" \
  --dataset-type tum_rgbd \
  --camera-profile auto \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35D/fr1_room_full_raw_retrieval_trace" \
  --feature-backend pyslam_orb2 \
  --enable-loop-closing \
  --disable-global-ba \
  --no-map-export \
  --profile-runtime \
  --profile-memory \
  --profile-keyframes \
  --profile-local-map \
  --loop-debug \
  --loop-candidate-source compare \
  --loop-retrieval-trace \
  --loop-retrieval-trace-raw-k 100 \
  --memory-profile-every 100 \
  --runtime-profile-every 100 \
  --memory-limit-gb 12 \
  --print-every 100
```

### Offline analyzer

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
python tools/analyze_gt_loop_raw_retrieval_trace.py \
  --trace-dir "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35D/fr1_room_full_raw_retrieval_trace" \
  --gt-loop-classified "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A/gt_loop_pairs_classified.csv" \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35D/gt_retrieval_stage_analysis"
```

## 3. Run output directory

- `visual_slam_outputs/checkpoint_2_35D/fr1_room_full_raw_retrieval_trace`

## 4. Number of keyframes

- `48`

## 5. Number of GT-loop-like pairs

- Current 2.35D trace analyzed with meaningful keyframe-gap filter `>10`: `44`
- Historical 2.35B reference count: `41`

Interpretation:

- The 2.35D runtime trace is authoritative for the current run.
- The 2.35B count is retained as a historical reference only.

## 6. Raw DBOW presence count

- `41 / 44`

## 7. Inverted/shared-word presence count

- `41 / 44`

## 8. Common-word pass count

- `15 / 44`

## 9. minScore pass count

- `15 / 44`

## 10. Accumulation pass count

- `10 / 44`

## 11. Retained candidate count

- `4 / 44`

## 12. Consistency pass count

- `2 / 44`

## 13. Accepted count

- `1 / 44`

Accepted GT-positive pair in the current run:

- `2-41`

## 14. Dominant first-failure stage

- `FAILED_COMMON_WORD_FILTER`
- Count: `21 / 44`
- Share of current GT-loop-like pairs: `47.7%`

## 15. Top 10 false-negative GT loop pairs

From `gt_retrieval_false_negatives_detailed.csv`:

1. `5-40` — `FAILED_ACCUMULATION_FILTER`
2. `13-46` — `FAILED_ACCUMULATION_FILTER`
3. `7-42` — `FAILED_ACCUMULATION_FILTER`
4. `15-47` — `FAILED_ACCUMULATION_FILTER`
5. `8-42` — `FAILED_ACCUMULATION_FILTER`
6. `1-39` — `FAILED_COMMON_WORD_FILTER`
7. `4-39` — `FAILED_COMMON_WORD_FILTER`
8. `1-40` — `FAILED_COMMON_WORD_FILTER`
9. `11-44` — `FAILED_COMMON_WORD_FILTER`
10. `2-40` — `FAILED_COMMON_WORD_FILTER`

## 16. Where the issue is concentrated

Current evidence says the dominant loss is in:

- common-word filtering first

Secondary losses exist in:

- accumulation filtering
- representative retention after accumulation
- connected-keyframe exclusion
- missing raw DBOW presence for a small minority
- consistency, but only after a much smaller retrieval survivor set

Current evidence does **not** support geometry as the dominant bottleneck for GT-positive pairs.

## 17. Whether sparse keyframe density appears causal

Sparse keyframe density does not appear to be the dominant first-failure cause in this checkpoint.

Reason:

- `41 / 44` GT-positive pairs are already present in raw DBOW.
- The largest drop is from `PASSED_CONNECTED_TEMPORAL = 36` to `PASSED_COMMON_WORD = 15`.
- That means the main loss happens after retrieval visibility but before consistency/geometry.

Density may still matter secondarily for later support, but it is not the primary blocker exposed by this funnel.

## 18. Exact next implementation recommendation

The next correction should focus on common-word filter parity with pySLAM, because `21 / 44` GT-loop-like pairs first fail at `FAILED_COMMON_WORD_FILTER`.

Recommended next checkpoint:

- audit the exact computation and ordering of:
  - candidate pool entering common-word evaluation
  - `max_common_words`
  - `min_common_words = int(max_common_words * 0.8)`
  - candidate inclusion condition `common_words > min_common_words`
- compare those steps line-by-line against:
  - `third_party/pyslam_reference/pyslam/loop_closing/keyframe_database.py:KeyFrameDatabaseDBow.detect_loop_candidates()`
- explicitly verify whether the local hybrid raw-DBOW candidate pool changes the `max_common_words` distribution relative to pySLAM’s shared-word/inverted-file candidate pool

Short conclusion:

- The next correction should target common-word filtering parity, not threshold tuning, not consistency, and not geometry, because that is the dominant first-failure stage in the final raw retrieval trace.
