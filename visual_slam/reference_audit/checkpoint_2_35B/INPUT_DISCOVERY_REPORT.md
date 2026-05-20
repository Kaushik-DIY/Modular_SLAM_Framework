# Checkpoint 2.35B Input Discovery Report

## 1. Selected run output directory
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle`

## 2. Dataset groundtruth path
- `/home/kaushik/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room/groundtruth.txt`

## 3. Required file existence
| File | Exists |
| --- | --- |
| `keyframes.json` | No |
| `run_summary.json` | Yes |
| `loop_candidate_oracle.csv` | Yes |
| `loop_retrieval_profile.csv` | Yes |
| `loop_candidate_source_comparison.csv` | Yes |
| `loop_keyframe_density_profile.csv` | Yes |
| `frame_timing.csv` | Yes |
| `runtime_profile.csv` | Yes |
| `memory_profile.csv` | Yes |

## 4. Keyframe count found
- `run_summary.json` reports `final_keyframes = 46`.
- The selected run omitted `keyframes.json`, but the same run contains:
  - `frame_log_rgbd_dataset_freiburg1_room.csv`
  - `keyframe_decision_log.csv`
  - `trajectory_rgbd_dataset_freiburg1_room.txt`
- Those same-run artifacts were sufficient to reconstruct all `46` keyframes with timestamps and estimated-pose associations.

## 5. Loop candidate oracle row count
- `loop_candidate_oracle.csv` rows: `40`

## 6. Rerun needed?
- No SLAM rerun was performed.
- Strictly speaking, the preferred run is missing `keyframes.json`.
- For this diagnostic-only checkpoint, rerun was still not needed because the same completed run provided enough same-run artifacts to recover:
  - keyframe IDs
  - keyframe timestamps
  - estimated trajectory associations
  - candidate/retrieval/density diagnostics
- Recovery path used: `frame_log` + `keyframe_decision_log` + `trajectory` from the selected 2.35A run.
