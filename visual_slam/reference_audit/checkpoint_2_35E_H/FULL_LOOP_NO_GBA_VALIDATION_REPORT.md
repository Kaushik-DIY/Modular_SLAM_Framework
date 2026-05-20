# Checkpoint 2.35E_H — FULL_LOOP_NO_GBA_VALIDATION_REPORT

## 1. Validation command

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
python -m visual_slam.orbslam.run_rgbd_slam \
  "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room" \
  --dataset-type tum_rgbd \
  --camera-profile auto \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35E_H/fr1_room_full_loop_no_gba_best_mode" \
  --feature-backend pyslam_orb2 \
  --enable-loop-closing \
  --disable-global-ba \
  --no-map-export \
  --profile-runtime \
  --profile-memory \
  --profile-keyframes \
  --profile-local-map \
  --loop-debug \
  --loop-candidate-source auto \
  --loop-retrieval-trace \
  --loop-retrieval-trace-raw-k 100 \
  --memory-profile-every 100 \
  --runtime-profile-every 100 \
  --memory-limit-gb 12 \
  --print-every 100
```

## 2. Run result

- frames attempted: `1362`
- tracking OK count: `1362`
- tracking lost count: `0`
- final state: `OK`
- accepted loops: `1`
- keyframes: `47`
- map points: `5342`
- elapsed seconds: `871.108`
- avg FPS: `1.56`
- peak RSS MB: `1643.797`

## 3. Generated runtime trace outputs

- `loop_raw_dbow_trace.csv`
- `loop_inverted_word_trace.csv`
- `loop_score_filter_trace.csv`
- `loop_accumulation_trace.csv`
- `loop_retained_candidate_trace.csv`
- `loop_gt_positive_trace.csv`
- `loop_consistency_progression.csv`
- `loop_geometry_trace.csv`

## 4. GT retrieval analyzer command

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
python tools/analyze_gt_loop_raw_retrieval_trace.py \
  --trace-dir "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35E_H/fr1_room_full_loop_no_gba_best_mode" \
  --gt-loop-classified "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A/gt_loop_pairs_classified.csv" \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35E_H/gt_retrieval_stage_analysis_no_gba"
```

## 5. Analyzer result

Funnel counts:

- `GT_LOOP_LIKE_TOTAL = 47`
- `RAW_DBOW_PRESENT = 44`
- `INVERTED_WORD_PRESENT = 44`
- `PASSED_CONNECTED_TEMPORAL = 44`
- `PASSED_COMMON_WORD = 14`
- `PASSED_MIN_SCORE = 13`
- `PASSED_ACCUMULATION = 12`
- `RETAINED_CANDIDATE = 8`
- `PASSED_CONSISTENCY = 5`
- `PASSED_GEOMETRY = 1`
- `ACCEPTED = 0`

Dominant first failure:

- `FAILED_COMMON_WORD_FILTER = 30`

Group-level summary:

- `GT_LOOP_LIKE_ELIGIBLE_FOR_LOOP = 44`
- `GROUP_RECALLED_EXACT = 8`
- `NOT_RETAINED_BUT_GT_EQUIVALENT_REPRESENTATIVE = 4`
- `GROUP_RECALLED_TOTAL = 12`

## 6. Accepted runtime loop safety check

The single accepted runtime loop was GT-negative:

- `current_kf_id = 42`
- `candidate_kf_id = 4`
- `gt_loop_like = False`
- `gt_translation_distance = 1.5236 m`
- `gt_rotation_angle_deg = 42.6379`

## 7. Satisfaction criteria decision

### Passed

- `tracking_ok_count == frames_attempted`
- `tracking_lost_count == 0`
- runtime remained practical
- memory remained stable inside the 12 GB limit

### Failed

- GT eligible-loop recall did not improve cleanly enough to justify correction readiness
- dominant retrieval failure was not reduced; `FAILED_COMMON_WORD_FILTER` worsened
- a false loop was accepted according to the GT oracle
- GT-loop-like accepted count from the analyzer is `0`

## 8. GBA decision

`Do not run GBA.`

Reason:

- the checkpoint’s no-GBA satisfaction criteria failed due false-loop acceptance and a worse dominant retrieval failure stage

## 9. Next recommended action

- audit the classic common-word gate with the new `classic_inverted` path
- inspect why GT-like pairs such as `KF39/41/43` cohorts still fail at common-word gating
- add a focused false-loop safety correction before any GBA validation rerun
