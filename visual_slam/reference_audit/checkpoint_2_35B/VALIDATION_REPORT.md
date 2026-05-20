# Checkpoint 2.35B Validation Report

## 1. Task / checkpoint name
- `Checkpoint 2.35B — GT loop-pair oracle recall analysis for TUM fr1_room`

## 2. Tests added / updated
- Added `tests/visual_slam/orbslam/test_checkpoint_2_35B_gt_loop_oracle.py`

## 3. Test commands run
```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_35B_gt_loop_oracle.py
```

## 4. Test results
- Result: `11 passed in 0.21s`

## 5. Dataset validation command run
```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
python tools/analyze_gt_loop_recall.py \
  --run-dir "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle" \
  --groundtruth "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room/groundtruth.txt" \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A" \
  --min-time-gap-sec 10.0 \
  --min-kf-gap 10 \
  --loop-trans-threshold-m 0.75 \
  --loop-rot-threshold-deg 45.0 \
  --near-loop-trans-threshold-m 1.5 \
  --gt-association-max-dt-sec 0.05
```

## 6. Dataset validation results
- Selected run: `checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle`
- GT-associated keyframes: `46 / 46`
- Temporally valid keyframe pairs: `607`
- GT-loop-like pairs: `41`
- GT-near-loop pairs: `339`
- GT-loop-like stage counts:
  - `NOT_RETRIEVED = 36`
  - `FAILED_CONSISTENCY = 2`
  - `FAILED_GEOMETRY_MATCHES = 1`
  - `FAILED_FINAL_SUPPORT = 1`
  - `ACCEPTED = 1`

## 7. Interpretation
- The dominant failure is upstream of consistency / geometry:
  - `36 / 41` GT-loop-like pairs never appear in the retained candidate oracle.
- Sparse-density evidence did not dominate:
  - `0 / 41` GT-loop-like pairs were flagged with the diagnostic density concern heuristic.
- Candidate retrieval / retention is therefore the main next target.

## 8. Remaining risks
- The selected run omitted `keyframes.json`; reconstruction depended on same-run logs.
- Per-pair raw pre-retention candidate identity is not exposed by current artifacts, so `NOT_RETRIEVED` is an upper bound on true retrieval-stage loss.

## 9. Next recommended action
- `Checkpoint 2.35C: loop candidate retrieval audit focused on true GT loop coverage before consistency.`
