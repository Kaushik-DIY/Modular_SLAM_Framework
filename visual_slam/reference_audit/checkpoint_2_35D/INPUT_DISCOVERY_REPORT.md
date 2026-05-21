# Checkpoint 2.35D — INPUT_DISCOVERY_REPORT

## 1. Objective

Identify the latest usable prior artifacts needed for the 2.35D raw-retrieval trace checkpoint and document why a new full diagnostic run is required.

## 2. Existing source directories inspected

- `visual_slam_outputs/checkpoint_2_35A/baseline_fr1_room_full_loop_oracle`
- `visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle`
- `visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A`
- `visual_slam_outputs/checkpoint_2_35C_diag/retrieval_trace_analysis`
- `visual_slam_outputs/checkpoint_2_34A/fr1_room_full_loop_no_gba`
- `datasets/tum/rgbd_dataset_freiburg1_room`

## 3. Latest usable prior full-run inputs

### 3.1 Preferred prior loop-debug full run

Selected directory:

- `visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle`

Relevant files present:

- `run_summary.json`
- `loop_candidate_oracle.csv`
- `loop_retrieval_profile.csv`
- `loop_candidate_source_comparison.csv`
- `loop_keyframe_density_profile.csv`
- `loop_debug_candidates.csv`
- `frame_log_rgbd_dataset_freiburg1_room.csv`
- `trajectory_rgbd_dataset_freiburg1_room.txt`
- `full_run_console.log`

Why selected:

- It is newer than the 2.34A run.
- It already contains the 2.35A loop-oracle and retrieval-profile outputs that 2.35B and 2.35C were built on.

### 3.2 Fallback earlier full run

Fallback directory:

- `visual_slam_outputs/checkpoint_2_34A/fr1_room_full_loop_no_gba`

Useful mainly for historical comparison, not for 2.35D primary tracing.

## 4. 2.35B GT-loop oracle outputs found

Directory:

- `visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A`

Required files present:

- `gt_loop_pairs_classified.csv`
- `gt_loop_pairs_all.csv`
- `gt_loop_recall_summary.json`
- `gt_loop_missed_pairs_top.csv`

Also present:

- `gt_loop_density_support_analysis.csv`
- `gt_loop_recall_by_stage.csv`
- `gt_near_loop_recall_by_stage.csv`
- `keyframe_gt_associations.csv`
- `GT_LOOP_RECALL_ANALYSIS_REPORT.md`

## 5. 2.35C diagnostic outputs found

Directory:

- `visual_slam_outputs/checkpoint_2_35C_diag/retrieval_trace_analysis`

Files present:

- `gt_loop_retrieval_trace.csv`
- `gt_loop_retrieval_funnel.csv`
- `gt_loop_top_missed_trace.csv`
- `gt_loop_retrieval_trace_summary.json`
- `GT_POSITIVE_RETRIEVAL_TRACE_REPORT.md`

## 6. Dataset GT source found

Directory:

- `datasets/tum/rgbd_dataset_freiburg1_room`

Files present:

- `groundtruth.txt`
- `associations.txt`
- `rgb.txt`
- `depth.txt`

## 7. Why a new full diagnostic run is required

Existing 2.35A / 2.35B / 2.35C artifacts are insufficient for exact first-failure attribution because they do not log:

- raw DBOW per-candidate identity before filtering
- raw DBOW rank / raw top-K presence for each GT-positive pair
- inverted/shared-word per-pair presence before common-word filtering
- per-pair minScore pass/fail and accumulation pass/fail for GT-positive candidates that were not ultimately retained

As a result:

- 2.35C can only classify many GT-positive pairs as broad not-retrieved states.
- 2.35D must rerun the full fr1_room diagnostic with deeper retrieval tracing enabled.

## 8. Planned 2.35D run output directory

Planned output directory:

- `visual_slam_outputs/checkpoint_2_35D/fr1_room_full_raw_retrieval_trace`

Planned downstream analysis directory:

- `visual_slam_outputs/checkpoint_2_35D/gt_retrieval_stage_analysis`

## 9. Inputs to be consumed by the new analyzer

Primary GT-positive classification input:

- `visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A/gt_loop_pairs_classified.csv`

Runtime trace input set to be produced by 2.35D:

- `loop_raw_dbow_trace.csv`
- `loop_inverted_word_trace.csv`
- `loop_score_filter_trace.csv`
- `loop_accumulation_trace.csv`
- `loop_retained_candidate_trace.csv`
- `loop_gt_positive_trace.csv`

## 10. Remaining risks before implementation

- The workspace is already dirty in many files, including several loop-related files. 2.35D changes must stay narrowly scoped and must not revert unrelated user work.
- Existing DBoW query behavior already filters connected/temporal candidates inside `_detect_loop_candidates_dbow3_raw()`, so 2.35D tracing needs to expose both raw returned candidates and post-filter candidates separately if we want exact first-failure attribution.
- A higher trace-only raw query limit may be necessary to determine whether GT pairs are absent from bounded native DBOW retrieval or merely outside a smaller result window.
