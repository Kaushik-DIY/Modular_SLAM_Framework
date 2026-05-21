# Checkpoint 2.35C-DIAG Input Discovery Report

## 1. Selected full-run directory
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle`

## 2. Selected 2.35B GT oracle directory
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A`

## 3. Groundtruth file path
- `/home/kaushik/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room/groundtruth.txt`

## 4. Required files found
| File | Found |
| --- | --- |
| `run_summary.json` | Yes |
| `loop_candidate_oracle.csv` | Yes |
| `loop_retrieval_profile.csv` | Yes |
| `loop_candidate_source_comparison.csv` | Yes |
| `loop_keyframe_density_profile.csv` | Yes |
| `keyframe_decision_log.csv` | Yes |
| `frame_log_rgbd_dataset_freiburg1_room.csv` | Yes |
| `gt_loop_pairs_all.csv` | Yes |
| `gt_loop_pairs_classified.csv` | Yes |
| `gt_loop_recall_summary.json` | Yes |
| `gt_loop_missed_pairs_top.csv` | Yes |
| `groundtruth.txt` | Yes |

## 5. Missing files
- `keyframes.json` is missing from the selected 2.35A run directory.

## 6. Whether analysis can proceed without rerun
- Yes.
- No SLAM rerun was needed.
- This checkpoint uses:
  - existing 2.35A loop candidate / retrieval / density outputs
  - existing 2.35B GT-pair outputs
  - offline code inspection of pySLAM and local sources

## 7. What limitations exist due to missing fields
- The selected run’s missing `keyframes.json` does not block 2.35C because 2.35B already reconstructed the GT-loop pair set.
- The real blocker is not `keyframes.json`; it is missing runtime raw-candidate identity tracing.
- Current logs contain:
  - per-current-keyframe counts for raw DBOW and later stage counts
  - retained candidate IDs after the chosen retrieval source
  - downstream oracle rows for candidates that actually reached verification
- Current logs do not contain:
  - raw DBOW candidate identity lists for each current keyframe
  - per-pair raw DBOW rank / raw DBOW score for the GT-missed pairs
  - raw shared-word candidate identity lists before common-word / minScore / accumulation filtering
- Because of that, broad `NOT_RETRIEVED` GT misses cannot be split into:
  - absent from raw DBOW
  - present in raw DBOW but removed by common-word filtering
  - removed by minScore
  - removed by accumulated-score retention
