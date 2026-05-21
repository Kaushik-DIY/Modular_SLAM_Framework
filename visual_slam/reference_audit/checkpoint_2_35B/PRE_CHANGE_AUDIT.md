# Checkpoint 2.35B Pre-Change Audit

## 1. Task / checkpoint name
- `Checkpoint 2.35B — GT loop-pair oracle recall analysis for TUM fr1_room`

## 2. Files inspected
- `AGENTS.md`
- `CODEX_CHECKPOINT_2_35B_GT_LOOP_RECALL_ANALYSIS.md`
- `visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle/run_summary.json`
- `visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle/loop_candidate_oracle.csv`
- `visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle/loop_retrieval_profile.csv`
- `visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle/loop_candidate_source_comparison.csv`
- `visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle/loop_keyframe_density_profile.csv`
- `visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle/keyframe_decision_log.csv`
- `visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle/frame_log_rgbd_dataset_freiburg1_room.csv`
- `visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle/trajectory_rgbd_dataset_freiburg1_room.txt`
- `datasets/tum/rgbd_dataset_freiburg1_room/groundtruth.txt`
- `visual_slam/orbslam/slam/loop_closing.py`
- `visual_slam/orbslam/slam/keyframe_database.py`
- `visual_slam/orbslam/run_rgbd_slam.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_35A_loop_candidate_retrieval.py`

## 3. pySLAM files inspected
- None required for this checkpoint.
- This task is diagnostic-only and does not modify tracking, LocalMapping, loop closure logic, or backend behavior.

## 4. Root cause / current hypothesis before changes
- The selected latest 2.35A full `fr1_room` loop-oracle run is missing `keyframes.json`, so the checkpoint cannot rely on the nominal export path.
- The same run does contain enough same-run artifacts to recover keyframes and timestamps without rerunning SLAM.
- Expected failure split to measure:
  - GT loop pairs never entering the retained candidate list
  - GT loop pairs entering the candidate list but failing consistency / geometry / final support

## 5. Why a diagnostic tool was needed
- Existing outputs already contain:
  - retained candidate oracle rows
  - retrieval-profile stage counts
  - source-comparison candidate lists
  - density diagnostics
- Existing tooling did not generate:
  - all GT-valid keyframe pairs for the selected run
  - unordered pair joins against actual oracle results
  - recall-by-stage summaries
  - a top-missed GT-loop report

## 6. Planned changes
- Add standalone analysis tool:
  - `tools/analyze_gt_loop_recall.py`
- Add synthetic test coverage:
  - `tests/visual_slam/orbslam/test_checkpoint_2_35B_gt_loop_oracle.py`
- Add checkpoint reports under:
  - `visual_slam/reference_audit/checkpoint_2_35B/`

## 7. Structural correctness requirement
- No algorithmic SLAM source files will be changed.
- No thresholds, scheduling, loop gates, Global BA, or C++ code will be modified.
- The tool will operate only on existing run outputs plus TUM `groundtruth.txt`.
