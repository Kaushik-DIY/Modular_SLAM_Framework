# Checkpoint 2.35I — CANDIDATE_SOURCE_MODE_COMPARISON_REPORT

## 1. Scope

- Full `fr1_room` loop-no-GBA comparison across `classic_inverted`, `dbow_detector`, and `hybrid_dbow_scored`.
- Post-run GT funnel analysis used the shared 2.35B oracle:
  - `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A/gt_loop_pairs_classified.csv`
- Runner flag support was verified in `RUNNER_HELP.txt`; all requested loop/debug/profile flags and all three modes are supported.

## 2. Run status

| Mode | frames_attempted | tracking_ok_count | tracking_lost_count | final_state | final_keyframes | final_map_points | elapsed_sec | avg_fps | peak_rss_mb |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| classic_inverted | 1362 | 1362 | 0 | OK | 47 | 4899 | 901.790 | 1.51 | 1642.555 |
| dbow_detector | 1362 | 1362 | 0 | OK | 46 | 4593 | 767.299 | 1.78 | 1625.172 |
| hybrid_dbow_scored | 1362 | 1362 | 0 | OK | 47 | 5947 | 884.069 | 1.54 | 1633.121 |

## 3. Loop result

### classic_inverted
- loop candidate events: `43`
- accepted loops: `1`
- rejected loops: `42`
- accepted loop pairs:
  - `KF16 <-> KF1 | gt_loop_like=False | gt_translation_distance=2.1753726025672018 m | gt_rotation_angle_deg=75.61940925690695`
- GT-valid accepted loops: `0 / 1`
- false accepted loops:
  - `KF16 <-> KF1 | gt_loop_like=False | gt_translation_distance=2.1753726025672018 m | gt_rotation_angle_deg=75.61940925690695`
### dbow_detector
- loop candidate events: `87`
- accepted loops: `0`
- rejected loops: `87`
- accepted loop pairs: none
- GT-valid accepted loops: `0 / 0`
- false accepted loops: none
### hybrid_dbow_scored
- loop candidate events: `38`
- accepted loops: `1`
- rejected loops: `37`
- accepted loop pairs:
  - `KF42 <-> KF2 | gt_loop_like=False | gt_translation_distance=1.5786586521474488 m | gt_rotation_angle_deg=37.265601474627445`
- GT-valid accepted loops: `0 / 1`
- false accepted loops:
  - `KF42 <-> KF2 | gt_loop_like=False | gt_translation_distance=1.5786586521474488 m | gt_rotation_angle_deg=37.265601474627445`

## 4. GT-loop recall funnel

The analyzer outputs below are reported exactly as produced by `tools/analyze_gt_loop_raw_retrieval_trace.py`. For `dbow_detector` and `hybrid_dbow_scored`, the `INVERTED_WORD_PRESENT` and downstream classic-stage fields are not semantically equivalent to the pure DBOW runtime path, but they are still useful as a diagnostic comparison baseline.

### classic_inverted
- GT_LOOP_LIKE_TOTAL: `41`
- GT_LOOP_LIKE_ELIGIBLE_FOR_LOOP: `39`
- RAW_DBOW_PRESENT: `39`
- INVERTED_WORD_PRESENT: `39`
- PASSED_CONNECTED_TEMPORAL: `39`
- PASSED_COMMON_WORD: `15`
- PASSED_MIN_SCORE: `13`
- PASSED_ACCUMULATION: `12`
- RETAINED_CANDIDATE: `7`
- PASSED_CONSISTENCY: `5`
- PASSED_GEOMETRY: `2`
- ACCEPTED: `0`
- dominant first-failure stage: `FAILED_COMMON_WORD_FILTER`
- group-level GT recall total: `12`
### dbow_detector
- GT_LOOP_LIKE_TOTAL: `37`
- GT_LOOP_LIKE_ELIGIBLE_FOR_LOOP: `30`
- RAW_DBOW_PRESENT: `35`
- INVERTED_WORD_PRESENT: `0`
- PASSED_CONNECTED_TEMPORAL: `30`
- PASSED_COMMON_WORD: `0`
- PASSED_MIN_SCORE: `7`
- PASSED_ACCUMULATION: `0`
- RETAINED_CANDIDATE: `0`
- PASSED_CONSISTENCY: `3`
- PASSED_GEOMETRY: `0`
- ACCEPTED: `0`
- dominant first-failure stage: `MISSING_FROM_INVERTED_WORD_SET`
- group-level GT recall total: `0`
### hybrid_dbow_scored
- GT_LOOP_LIKE_TOTAL: `52`
- GT_LOOP_LIKE_ELIGIBLE_FOR_LOOP: `45`
- RAW_DBOW_PRESENT: `48`
- INVERTED_WORD_PRESENT: `0`
- PASSED_CONNECTED_TEMPORAL: `45`
- PASSED_COMMON_WORD: `0`
- PASSED_MIN_SCORE: `12`
- PASSED_ACCUMULATION: `10`
- RETAINED_CANDIDATE: `4`
- PASSED_CONSISTENCY: `0`
- PASSED_GEOMETRY: `0`
- ACCEPTED: `0`
- dominant first-failure stage: `MISSING_FROM_INVERTED_WORD_SET`
- group-level GT recall total: `10`

## 5. KF42 <-> KF4 cross-run presence

- The known E–H false loop `KF42 <-> KF4` did not appear as an accepted pair in any of the new mode runs.
- `classic_inverted`: raw-visible=`1`, retained=`0`, geometry=`0`, accepted=`0`
- `dbow_detector`: raw-visible=`1`, retained=`0`, geometry=`0`, accepted=`0`
- `hybrid_dbow_scored`: raw-visible=`1`, retained=`0`, geometry=`0`, accepted=`0`

## 6. Candidate-source conclusion

- Which mode has best GT-valid recall: none produced a GT-valid accepted loop; among the three, `classic_inverted` preserved the strongest GT-like funnel survival with `GROUP_RECALLED_TOTAL=12`, `PASSED_CONSISTENCY=5`, and `PASSED_GEOMETRY=2`.
- Which mode has best false-loop safety: `dbow_detector`. It accepted `0` loops, so it introduced no false accepted loops in this comparison run.
- Which mode should be used for the next implementation checkpoint: `classic_inverted` as the primary parity path, with `dbow_detector` retained as a comparison/safety reference. `hybrid_dbow_scored` should not be the primary implementation target.
- Did `dbow_detector` reduce the common-word bottleneck: yes structurally, because the analyzer no longer shows `FAILED_COMMON_WORD_FILTER`; however, that did not improve GT-valid loop closure and yielded `0` retained GT-equivalent candidates in this run.
- Did `dbow_detector` introduce more false positives: no. It produced `0` accepted loops and therefore no accepted false loops.
- Is `classic_inverted` still blocked mainly by common-word filtering: yes. Its dominant first failure remained `FAILED_COMMON_WORD_FILTER` with `24` GT-like pairs lost there.
- Is `hybrid_dbow_scored` useful only as diagnostic/legacy: yes for now. It still mixes source concepts, produced `0` GT-valid accepted loops, and accepted another GT-negative loop (`KF42 <-> KF2`).
