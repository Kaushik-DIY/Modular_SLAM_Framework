# Checkpoint 2.35C-DIAG Validation Report

## 1. Task / checkpoint name
- `Checkpoint 2.35C-DIAG — Diagnostic-only GT-positive loop retrieval tracing and pySLAM comparison`

## 2. Tests added
- `tests/visual_slam/orbslam/test_checkpoint_2_35C_diag_gt_loop_trace.py`

## 3. Test command run
```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/visual_slam/orbslam/test_checkpoint_2_35C_diag_gt_loop_trace.py
```

## 4. Test result
- `10 passed in 0.04s`

## 5. Analysis command run
```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
python tools/analyze_gt_loop_retrieval_trace.py \
  --gt-loop-classified "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A/gt_loop_pairs_classified.csv" \
  --gt-loop-all "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A/gt_loop_pairs_all.csv" \
  --loop-oracle "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle/loop_candidate_oracle.csv" \
  --retrieval-profile "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle/loop_retrieval_profile.csv" \
  --source-comparison "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle/loop_candidate_source_comparison.csv" \
  --density-profile "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle/loop_keyframe_density_profile.csv" \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35C_diag/retrieval_trace_analysis"
```

## 6. Analysis result summary
- GT-loop-like pairs: `41`
- Actual retained/oracle candidates seen: `5`
- Accepted: `1`
- Broad not-retrieved: `36`
- Dominant known loss stage: `FAILED_CONSISTENCY`
- Dominant unknown limitation: `raw candidate identities not logged`

## 7. What was proven
- The `5` GT-positive retained pairs split into:
  - `1` accepted
  - `2` failed consistency
  - `1` failed geometry
  - `1` failed final support

## 8. What remains unproven
- For the `36` GT-positive broad misses, current logs do not prove pair-level raw DBOW presence or absence.
- Therefore the exact loss stage before retained-candidate output remains unknown.

## 9. Next recommended action
- `Checkpoint 2.35D: add raw candidate identity tracing before common-word/minScore/accumulation filters.`
