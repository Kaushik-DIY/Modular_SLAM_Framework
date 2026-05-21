# Checkpoint 2.35B GT Loop Recall Analysis Report

## 1. Input run directory
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle`

## 2. Dataset groundtruth path
- `/home/kaushik/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room/groundtruth.txt`

## 3. Number of keyframes and GT associations
- Keyframes analyzed: `46`
- GT-associated keyframes: `46`
- `keyframes.json` was missing in the selected run.
- Same-run reconstruction path used:
  - `frame_log_rgbd_dataset_freiburg1_room.csv`
  - `keyframe_decision_log.csv`
  - `trajectory_rgbd_dataset_freiburg1_room.txt`

## 4. GT-loop definition thresholds
- `min_time_gap_sec = 10.0`
- `min_kf_gap = 10`
- `loop_trans_threshold_m = 0.75`
- `loop_rot_threshold_deg = 45.0`
- `near_loop_trans_threshold_m = 1.5`
- `gt_association_max_dt_sec = 0.05`

## 5. Total GT-loop-like pairs
- `41`

## 6. Total GT-near-loop pairs
- `339`

## 7. Recall-by-stage table
| Stage | Count | Recall vs GT-loop-like |
| --- | --- | --- |
| `GT_LOOP_LIKE_TOTAL` | `41` | `100.0%` |
| `NOT_RETRIEVED` | `36` | `87.80%` |
| `FAILED_CONSISTENCY` | `2` | `4.88%` |
| `FAILED_GEOMETRY_MATCHES` | `1` | `2.44%` |
| `FAILED_FINAL_SUPPORT` | `1` | `2.44%` |
| `ACCEPTED` | `1` | `2.44%` |

## 8. Top missed GT-loop-like pairs
- Strongest missed true loops are extremely close in GT and still absent from the retained candidate list:
  - `0-39`: `0.113 m`, `29.49 deg`
  - `1-39`: `0.161 m`, `20.00 deg`
  - `5-39`: `0.168 m`, `36.73 deg`
  - `6-40`: `0.169 m`, `23.15 deg`
  - `2-39`: `0.216 m`, `14.08 deg`
- These are not borderline GT pairs; they are high-quality true loop opportunities.

## 9. Accepted GT-loop-like pairs
- Only one GT-loop-like pair was accepted:
  - `15-45`
  - GT distance: `0.393 m`
  - GT rotation: `10.17 deg`
  - Final matched map points: `73`

## 10. Are correct GT loop pairs present in our candidate list?
- Yes, but only rarely.
- Of `41` GT-loop-like pairs:
  - `1` was accepted
  - `4` more were present in the actual retained candidate/oracle path and then rejected downstream
  - `36` never appeared in the retained candidate oracle at all

## 11. If not, is the issue candidate retrieval or sparse keyframe density?
- Evidence points much more strongly to candidate retrieval / candidate-retention collapse than to sparse keyframe density.
- Reasons:
  - `0 / 41` GT-loop-like pairs triggered the diagnostic density concern heuristic.
  - The strongest missed GT pairs still have healthy per-keyframe support counts in the reconstructed logs, typically around `96` to `110` tracked map points on the revisit-side keyframe.
  - Late revisit keyframes `38` to `45` often have large raw DBow candidate pools:
    - `kf 38`: `37` raw DBow candidates
    - `kf 39`: `35`
    - `kf 40`: `32`
    - `kf 41`: `36`
    - `kf 42`: `40`
    - `kf 43`: `41`
    - `kf 44`: `40`
    - `kf 45`: `38`
  - Despite those raw pools, the retained candidate lists collapse to only `0` to `2` chosen candidates, and most GT-positive early keyframes never appear there.
- Conclusion:
  - The main issue is upstream retrieval / retention quality, not obviously sparse keyframe density.

## 12. If they are present, where are they rejected?
- Downstream rejection split for the `5` GT-loop-like pairs that did make it into the retained candidate/oracle path:
  - `2` failed consistency
  - `1` failed geometry matches
  - `1` failed final support after covisibility expansion
  - `1` was accepted
- Pair-level details:
  - `4-39`: `FAILED_CONSISTENCY`
  - `2-40`: `FAILED_CONSISTENCY`
  - `13-42`: `FAILED_GEOMETRY_MATCHES`
  - `8-43`: `FAILED_FINAL_SUPPORT`
  - `15-45`: `ACCEPTED`

## 13. Main suspected root cause after this analysis
- The dominant failure is that true GT loop pairs are not being surfaced in the retained candidate list for the revisit keyframes.
- This is consistent with:
  - many raw DBow candidates existing
  - very few retained candidates surviving retrieval / accumulation
  - wrong retained candidates being selected for `kf 39` to `kf 44`

## 14. Recommended next checkpoint
- `Checkpoint 2.35C — GT-positive loop candidate retrieval audit before consistency`

Recommended scope:
- For GT-positive revisit keyframes, trace:
  - raw DBow candidate identities
  - common-word ranking
  - min-score filtering
  - accumulation retention
  - why the true early keyframes lose to the retained wrong candidates
- Keep loop thresholds and downstream geometry unchanged until retrieval recall is understood.
