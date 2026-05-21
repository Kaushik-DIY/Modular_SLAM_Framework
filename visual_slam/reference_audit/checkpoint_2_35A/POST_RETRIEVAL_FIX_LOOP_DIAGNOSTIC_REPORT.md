# Checkpoint 2.35A - Post Retrieval Fix Loop Diagnostic Report

## Scope
- Dataset: `/home/kaushik/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room`
- Output: `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle`
- Command: same full loop/no-GBA compare-mode run as baseline, with the retrieval/query-order fixes applied

## Post-fix run summary
- `1362/1362` tracking OK, `0` lost
- final state `OK`
- `46` keyframes, `4668` map points
- elapsed `895.73 s`, avg FPS `1.52`
- peak RSS `1622.77 MB`, final RSS `1622.90 MB`
- accepted loops: `1`

## Candidate retrieval and oracle summary
- oracle rows: `40` (down from `417`)
- GT-loop-like rows: `5`
- GT-near-loop rows: `18`
- GT-loop-like rows passing consistency: `3`
- accepted pair:
  - event `39`: `KF45 -> KF15`, GT `0.393 m / 10.17 deg`, final matched map points `73`

## Source-comparison result
- retrieval-profile rows: `45`
- queries with candidates after accumulation: `31`
- average raw DBOW3 broad-pool candidates/query: `19.69`
- average retained candidates/query after scoring: `0.89`
- DBOW3-scored candidates total: `40`
- inverted-file candidates total: `40`
- intersection total: `40`
- DBOW3-only total: `0`
- inverted-only total: `0`

This is the strongest success criterion of the checkpoint. Native DBOW3 retrieval no longer bypasses the pySLAM-style scoring logic, and compare mode now shows exact retained-candidate agreement between DBOW3 and inverted-file paths for this run.

## Dominant remaining rejection reasons
- `rejected_by_consistency`: `28`
- `not enough SE3 RANSAC seed inliers`: `8`
- `too few loop geometry matches`: `1`
- `estimated pose distance too large for guided SE3 loop seed`: `1`
- final covisibility-expansion gate:
  - `KF43 -> KF8`: `46 < 60`, GT `0.540 m / 10.41 deg`

## Baseline vs post-fix comparison
- accepted loops: `2 -> 1`
- elapsed time: `1058.49 s -> 895.73 s`
- peak RSS: `1656.76 MB -> 1622.77 MB`
- retained loop candidates: `417 -> 40`
- GT-loop-like candidate rows: `30 -> 5`
- source mismatches:
  - baseline DBOW3-only candidates: `376`
  - post-fix DBOW3-only candidates: `0`

## Interpretation
The retrieval fix did what it was supposed to do structurally: it removed the raw DBOW3 vs inverted-file mismatch and made retained-candidate decisions consistent across both sources. After that cleanup, the remaining failures concentrated in consistency and geometry on a much smaller set of more plausible loop pairs. Candidate retrieval is no longer the dominant uncertainty.
