# Checkpoint 2.35A - Baseline Loop Oracle Diagnostic Report

## Scope
- Task: baseline full `fr1_room` loop diagnostic before retrieval behavior changes
- Dataset: `/home/kaushik/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room`
- Output: `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_35A/baseline_fr1_room_full_loop_oracle`
- Command: compare-mode full run with loop closing enabled, Global BA disabled, map export disabled, memory limit `8 GB`

## Files inspected
- `run_summary.json`
- `loop_candidate_oracle.csv`
- `loop_retrieval_profile.csv`
- `loop_candidate_source_comparison.csv`
- `loop_keyframe_density_profile.csv`
- `loop_debug_candidates.csv`
- `runtime_profile.json`
- `memory_profile.csv`
- `keyframe_decision_log.csv`
- `local_mapping_schedule_log.csv`

## Baseline run summary
- `1362/1362` tracking OK, `0` lost
- final state `OK`
- `48` keyframes, `6459` map points
- elapsed `1058.49 s`, avg FPS `1.29`
- peak RSS `1656.76 MB`, final RSS `1656.88 MB`
- accepted loops in this baseline rerun: `2`

## Oracle summary
- oracle candidate rows: `417`
- GT-loop-like rows: `30`
- GT-near-loop rows: `261`
- GT-loop-like rows passing consistency: `25`
- accepted GT-loop-like pairs:
  - event `224`: `KF39 -> KF4`, GT `0.275 m / 9.62 deg`, final matched map points `421`
  - event `365`: `KF45 -> KF8`, GT `0.542 m / 13.32 deg`, final matched map points `65`

## Candidate-source comparison
- retrieval-profile rows: `47`
- queries with any DBOW3 raw candidates: `36`
- queries with any inverted-file retained candidates: `36`
- total DBOW3 raw candidates: `417`
- total inverted-file retained candidates: `44`
- intersection total: `41`
- DBOW3-only total: `376`
- inverted-only total: `3`

This is the key pre-fix structural finding: native DBOW3 retrieval was producing a much larger raw candidate set than the pySLAM-style inverted-file scoring path, and the two paths agreed on only `41` retained candidates.

## Dominant rejection reasons
- `rejected_by_consistency`: `191`
- `too few loop geometry matches`: `105`
- `not enough SE3 RANSAC seed inliers`: `82`
- `estimated pose distance too large for guided SE3 loop seed`: `22`
- final covisibility-expansion gate misses:
  - `KF45 -> KF9`: `59 < 60`, GT `0.363 m / 13.46 deg`
  - `KF47 -> KF13`: `21 < 60`, GT `0.234 m / 16.81 deg`

## Runtime and memory notes
- top runtime sections:
  - `slam.track`: `709.87 s`
  - `local_mapping.step`: `301.28 s`
  - `tracking.track_local_map`: `211.10 s`
  - `local_mapping.cull_map_points`: `180.56 s`
  - `local_mapping.local_BA`: `111.71 s`
- loop-closing runtime was modest relative to tracking and local mapping: `22.71 s` total
- memory remained bounded; `recent_frames=20`, `old_frame_views_total=0`

## Baseline conclusion
The pre-fix baseline did not reproduce the old `0 accepted loops` outcome, but it did confirm the structural retrieval gap: DBOW3 raw retrieval was not aligned with the pySLAM-style common-word/minScore/covisibility-accumulation path. The baseline also showed that GT-loop-like pairs existed and some were close to the final map-point gate, so the next step remained a retrieval-structure fix, not threshold tuning.
