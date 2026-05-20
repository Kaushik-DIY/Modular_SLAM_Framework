# GT Positive Retrieval Trace Report

## 1. Objective
- Trace every GT-loop-like pair through the currently available retrieval and verification logs.
- Split the previously broad `NOT_RETRIEVED` bucket as far as the logs actually allow.
- State clearly what still cannot be proven.

## 2. Input files used
- Full run:
  - `visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle/run_summary.json`
  - `loop_candidate_oracle.csv`
  - `loop_retrieval_profile.csv`
  - `loop_candidate_source_comparison.csv`
  - `loop_keyframe_density_profile.csv`
- 2.35B:
  - `gt_loop_pairs_all.csv`
  - `gt_loop_pairs_classified.csv`
  - `gt_loop_recall_summary.json`
  - `gt_loop_missed_pairs_top.csv`
- Dataset:
  - `datasets/tum/rgbd_dataset_freiburg1_room/groundtruth.txt`

## 3. Whether analysis required a SLAM rerun
- No.
- This checkpoint used only existing outputs and source inspection.

## 4. GT-loop-like pair count
- `41`

## 5. Recall funnel
| Stage | Count | Percent | Confidence |
| --- | --- | --- | --- |
| `GT_LOOP_LIKE_TOTAL` | `41` | `100.0%` | `high` |
| `ACTUAL_CANDIDATE_SEEN` | `5` | `12.20%` | `high` |
| `ACCEPTED` | `1` | `2.44%` | `high` |
| `FAILED_CONSISTENCY` | `2` | `4.88%` | `high` |
| `FAILED_GEOMETRY` | `1` | `2.44%` | `high` |
| `FAILED_FINAL_SUPPORT` | `1` | `2.44%` | `high` |
| `NOT_RETRIEVED_BROAD` | `36` | `87.80%` | `limited` |
| `UNKNOWN_DUE_MISSING_RAW_TRACE` | `36` | `87.80%` | `limited` |

## 6. Top missed GT-loop pairs
- Strongest missed GT-positive pairs remain:
  - `0-39`: `0.113 m`, `29.49 deg`
  - `1-39`: `0.161 m`, `20.00 deg`
  - `5-39`: `0.168 m`, `36.73 deg`
  - `6-40`: `0.169 m`, `23.15 deg`
  - `2-39`: `0.216 m`, `14.08 deg`
- For those pairs, current logs show:
  - many raw DBOW candidates existed for the current revisit keyframe
  - a wrong retained top candidate was chosen
  - the GT-positive pair itself does not appear in retained candidate identity logs

## 7. What can be proven from current logs
- The current logs prove:
  - which GT-positive pairs reached the actual retained/oracle path
  - which retained GT-positive pairs failed consistency
  - which retained GT-positive pairs failed geometry
  - which retained GT-positive pairs failed final support
- The current logs also prove per-current-keyframe counts for:
  - raw DBOW candidate totals
  - post-common-word totals
  - post-minScore totals
  - post-accumulation totals

## 8. What cannot be proven because raw candidate identities are missing
- Current logs do not prove whether a specific GT-positive missed pair:
  - was absent from raw DBOW
  - was present in raw DBOW but removed by common-word filtering
  - was removed by minScore
  - was removed by accumulated-score retention
- The exact missing field is:
  - raw candidate identity tracing before retained candidate output

## 9. Comparison with pySLAM retrieval logic
- The local pipeline is structurally close to pySLAM on:
  - reference minScore computation
  - common-word filtering
  - accumulated-score retention
  - consistency checking
  - downstream projection-based final support
- The major practical difference for this diagnostic is not stage absence.
- The major difference is that the local thesis instrumentation still lacks per-pair raw retrieval trace data, so GT-positive misses cannot be localized before the retained-candidate boundary.

## 10. Most probable loss stage
- Dominant known downstream loss stage:
  - `FAILED_CONSISTENCY`
- Dominant overall limitation:
  - raw candidate identities not logged
- Evidence from late revisit keyframes `39` to `45` suggests the broader problem is likely upstream retained-candidate selection, but current logs cannot prove whether the GT-positive pairs were already absent from raw DBOW or lost before retention.

## 11. Whether sparse keyframe density appears causal
- No.
- 2.35B density diagnostics did not flag sparse keyframe density as the dominant issue.
- The strongest GT-positive misses occur even when the revisit keyframes still have healthy tracking support and large raw DBOW candidate counts.

## 12. Recommended next implementation checkpoint
- `Checkpoint 2.35D — add raw candidate identity tracing before common-word/minScore/accumulation filters`

Recommended scope:
- log raw DBOW candidate IDs, ranks, and scores for each current keyframe
- log per-candidate shared-word counts before common-word filtering
- log per-candidate score versus `min_score`
- log per-candidate accumulated score and retained/dropped status before consistency

## Explicit answer
- Do current logs prove whether GT pairs were missing from raw DBOW?
- No. We need runtime raw candidate identity tracing in the next checkpoint.
