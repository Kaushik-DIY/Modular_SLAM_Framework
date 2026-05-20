# Checkpoint 2.35B Implementation Audit

## 1. Task / checkpoint name
- `Checkpoint 2.35B — GT loop-pair oracle recall analysis for TUM fr1_room`

## 2. Exact changes made
- Added `tools/analyze_gt_loop_recall.py`
  - Loads TUM GT poses.
  - Loads selected run summary and diagnostics.
  - Prefers `keyframes.json` when available.
  - Falls back to same-run reconstruction from:
    - `frame_log_rgbd_dataset_freiburg1_room.csv`
    - `keyframe_decision_log.csv`
    - `trajectory_rgbd_dataset_freiburg1_room.txt`
  - Generates all temporally valid keyframe pairs.
  - Associates GT poses to each keyframe.
  - Joins GT-valid pairs with actual loop candidate oracle rows by unordered keyframe pair.
  - Classifies each GT loop pair by failure stage.
  - Produces CSV/JSON summaries and an output-local markdown report.
- Added `tests/visual_slam/orbslam/test_checkpoint_2_35B_gt_loop_oracle.py`
  - GT parser coverage
  - nearest-GT association coverage
  - pair distance / rotation coverage
  - GT loop-like classification coverage
  - unordered pair-key coverage
  - oracle unordered join coverage
  - pipeline-stage classification coverage
  - summary count coverage

## 3. Files changed
- `tools/analyze_gt_loop_recall.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_35B_gt_loop_oracle.py`
- `visual_slam/reference_audit/checkpoint_2_35B/INPUT_DISCOVERY_REPORT.md`
- `visual_slam/reference_audit/checkpoint_2_35B/PRE_CHANGE_AUDIT.md`
- `visual_slam/reference_audit/checkpoint_2_35B/IMPLEMENTATION_AUDIT.md`
- `visual_slam/reference_audit/checkpoint_2_35B/VALIDATION_REPORT.md`
- `visual_slam/reference_audit/checkpoint_2_35B/GT_LOOP_RECALL_ANALYSIS_REPORT.md`

## 4. Why the changes are structurally correct
- The checkpoint remained diagnostic-only.
- All computations were performed from existing artifacts in the selected run directory plus dataset GT.
- The fallback keyframe reconstruction uses same-run logs rather than cross-run artifact mixing.
- Classification logic is tied to actual logged fields in:
  - `loop_candidate_oracle.csv`
  - `loop_retrieval_profile.csv`
  - `loop_candidate_source_comparison.csv`
  - `loop_keyframe_density_profile.csv`

## 5. Important implementation notes
- The selected run had no `keyframes.json`; recovery from same-run logs was required.
- Pair classification is strongest for:
  - retained candidate pairs present in `loop_candidate_oracle.csv`
  - alternate retained lists present in `loop_candidate_source_comparison.csv`
- Current artifacts do not expose per-pair raw pre-retention identities, so many GT pairs legitimately land in:
  - `NOT_RETRIEVED`
  rather than a finer-grained raw-only bucket.

## 6. Output artifacts generated
- `visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A/keyframe_gt_associations.csv`
- `visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A/gt_loop_pairs_all.csv`
- `visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A/gt_loop_pairs_classified.csv`
- `visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A/gt_loop_recall_by_stage.csv`
- `visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A/gt_near_loop_recall_by_stage.csv`
- `visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A/gt_loop_density_support_analysis.csv`
- `visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A/gt_loop_missed_pairs_top.csv`
- `visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A/gt_loop_recall_summary.json`
