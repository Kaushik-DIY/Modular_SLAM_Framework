# IMPLEMENTATION_AUDIT

This checkpoint is in progress. Only parameter-value edits in `visual_slam/orbslam/slam/config_parameters.py` and assertion-only edits in `tests/visual_slam/orbslam/test_checkpoint_2_1_common.py` are permitted.

## Run `2_36W`

1. task/checkpoint name
Checkpoint `2_36W` - aggressive CW threshold reduction

2. files inspected
- `/home/kaushik/slam_ws/visual_slam/orbslam/slam/config_parameters.py`
- `/home/kaushik/slam_ws/tests/visual_slam/orbslam/test_checkpoint_2_1_common.py`
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_36W_cw_thresh_050/run_summary.json`
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_36W_cw_thresh_050/loop_gt_positive_trace.csv`
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_36W_cw_thresh_050/run_log.txt`

3. pySLAM files inspected, if relevant
- None. No visual-SLAM logic was modified.

4. root cause or current hypothesis
- The run tested whether a lower common-word ratio threshold would recover genuine loop pairs blocked by the `FAILED_COMMON_WORD_FILTER`.
- That hypothesis was partly confirmed at the funnel level, but it did not improve the primary metric (`accepted_loops`) and coincided with much worse tracking stability.

5. exact changes made
- Temporarily changed `Parameters.kLoopClosingCommonWordRatioThreshold` from `0.67` to `0.50`.
- Reverted the parameter to the baseline `0.67` after the run completed.
- No test-file assertions required changes.

6. why the changes are structurally correct
- The plan explicitly allows parameter-only tuning in `config_parameters.py`.
- KF-density guardrail parameters were left unchanged.
- No implementation files outside the allowed scope were modified.

7. tests added/updated
- None.

8. test commands run
```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_1_common.py
```

9. test results
- Passed before the run: `5 passed in 0.40s`
- Passed again after restoring the baseline value: `5 passed in 0.31s`

10. dataset validation commands and results
```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
python -u -m visual_slam.orbslam.run_rgbd_slam \
  "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room" \
  --dataset-type tum_rgbd --camera-profile auto \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_36W_cw_thresh_050" \
  --feature-backend pyslam_orb2 \
  --enable-loop-closing --disable-global-ba --no-map-export \
  --loop-debug --loop-candidate-source auto \
  --loop-retrieval-trace --loop-retrieval-trace-raw-k 100 \
  --print-every 100
```

- Final state: `OK`
- Accepted loops: `3`
- Final keyframes: `121`
- Final map points: `18545`
- Tracking lost: `26`
- Runtime: `3738.95 s`

11. remaining risks
- The stronger CW pass rate may be exposing harder candidate branches that correlate with tracking instability later in the sequence, but no causal claim is proven yet.
- Because GT-positive trace total dropped from `271` to `215`, this run is not directly comparable to `2_35V` without acknowledging the tracking regression.

12. next recommended action
- Keep `2_36W` as a documented data point, but do not treat it as the Phase 1 winner unless `2_36X` and `2_36Y` fail even harder on the primary metric and stability constraints.

## Run `2_36X`

1. task/checkpoint name
Checkpoint `2_36X` - increased retrieval width

2. files inspected
- `/home/kaushik/slam_ws/visual_slam/orbslam/slam/config_parameters.py`
- `/home/kaushik/slam_ws/tests/visual_slam/orbslam/test_checkpoint_2_1_common.py`
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_36X_topk_15/run_summary.json`
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_36X_topk_15/loop_gt_positive_trace.csv`
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_36X_topk_15/loop_geometry_trace.csv`
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_36X_topk_15/loop_debug_candidates.csv`
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_36X_topk_15/run_log.txt`

3. pySLAM files inspected, if relevant
- None. No visual-SLAM logic was modified.

4. root cause or current hypothesis
- This run tested whether widening retrieval would recover loop candidates that were missing from the raw DBOW stage.
- It preserved runtime and tracking quality well, but it did not improve genuine accepted-loop count and appears to have admitted one likely false positive.

5. exact changes made
- Temporarily changed `Parameters.kLoopDbowDetectorTopK` from `5` to `15`.
- Temporarily changed `Parameters.kMaxResultsForLoopClosure` from `5` to `15`.
- No other parameter fields were modified for this run.

6. why the changes are structurally correct
- The checkpoint plan explicitly lists these two retrieval-width parameters as the only `2_36X` changes.
- KF-density guardrail parameters remained unchanged.
- No implementation files outside the allowed scope were modified.

7. tests added/updated
- None.

8. test commands run
```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_1_common.py
```

9. test results
- Passed before the run: `5 passed in 0.30s`

10. dataset validation commands and results
```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
python -u -m visual_slam.orbslam.run_rgbd_slam \
  "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room" \
  --dataset-type tum_rgbd --camera-profile auto \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_36X_topk_15" \
  --feature-backend pyslam_orb2 \
  --enable-loop-closing --disable-global-ba --no-map-export \
  --loop-debug --loop-candidate-source auto \
  --loop-retrieval-trace --loop-retrieval-trace-raw-k 100 \
  --print-every 100
```

- Final state: `OK`
- Accepted loops: `3`
- Final keyframes: `114`
- Final map points: `16211`
- Tracking lost: `0`
- Runtime: `3671.09 s`

11. remaining risks
- `run_summary.json` reports `3` accepted loops, but only `2` GT-near accepted pairs appear in `loop_gt_positive_trace.csv`.
- The missing accepted pair is present in `loop_geometry_trace.csv` as `KF143 -> KF42` with `final_matched_map_points = 45`, but its GT distance is `1.2988 m`, which makes it a likely false positive under the checkpoint's `0.5 m` genuine-loop rule.

12. next recommended action
- Record `2_36X` as a mixed result: better runtime and tracking stability than `2_36W`, but disqualified as a clean Phase 1 winner if the likely false positive is confirmed by the checkpoint criterion.

## Run `2_36Y`

1. task/checkpoint name
Checkpoint `2_36Y` - combined moderate approach

2. files inspected
- `/home/kaushik/slam_ws/visual_slam/orbslam/slam/config_parameters.py`
- `/home/kaushik/slam_ws/tests/visual_slam/orbslam/test_checkpoint_2_1_common.py`
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_36Y_combined_moderate/run_summary.json`
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_36Y_combined_moderate/loop_gt_positive_trace.csv`
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_36Y_combined_moderate/loop_geometry_trace.csv`
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_36Y_combined_moderate/run_log.txt`

3. pySLAM files inspected, if relevant
- None. No visual-SLAM logic was modified.

4. root cause or current hypothesis
- This run tested whether a moderate CW threshold reduction paired with moderate retrieval widening could increase genuine loop recall without the tracking regression from `2_36W` or the likely false positive from `2_36X`.
- That hypothesis was supported by the result: `2_36Y` produced `4` GT-near accepted loops, `0` tracking losses, and no trace evidence of false positives.

5. exact changes made
- Temporarily changed `Parameters.kLoopClosingCommonWordRatioThreshold` from `0.67` to `0.55`.
- Temporarily changed `Parameters.kLoopDbowDetectorTopK` from `5` to `10`.
- Temporarily changed `Parameters.kMaxResultsForLoopClosure` from `5` to `10`.
- Reverted all three parameters to the baseline `2_35V` values after the run completed.

6. why the changes are structurally correct
- The checkpoint plan explicitly lists these three parameter changes as the `2_36Y` configuration.
- KF-density guardrail parameters remained unchanged.
- No implementation files outside the allowed scope were modified.

7. tests added/updated
- None.

8. test commands run
```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_1_common.py
```

9. test results
- Passed before the run: `5 passed in 0.30s`
- Passed again after restoring the baseline value: `5 passed in 0.30s`

10. dataset validation commands and results
```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
python -u -m visual_slam.orbslam.run_rgbd_slam \
  "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room" \
  --dataset-type tum_rgbd --camera-profile auto \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_36Y_combined_moderate" \
  --feature-backend pyslam_orb2 \
  --enable-loop-closing --disable-global-ba --no-map-export \
  --loop-debug --loop-candidate-source auto \
  --loop-retrieval-trace --loop-retrieval-trace-raw-k 100 \
  --print-every 100
```

- Final state: `OK`
- Accepted loops: `4`
- Final keyframes: `127`
- Final map points: `17984`
- Tracking lost: `0`
- Runtime: `4083.90 s`

11. remaining risks
- `FAILED_ACCUMULATION_FILTER` increased (`14 -> 32` vs baseline), so some additional recall pressure is still being absorbed downstream even in the winning Phase 1 configuration.
- Total GT-positive trace count (`207`) is lower than the baseline `271`, so funnel percentages across runs should still be interpreted carefully alongside whole-run behavior.

12. next recommended action
- Treat `2_36Y` as the Phase 1 winner and proceed to `2_36Z1` and `2_36Z2` as adjacent follow-up runs in the direction suggested by the tuning plan.

## Run `2_36Z1`

1. task/checkpoint name
Checkpoint `2_36Z1` - CW `0.50` plus top-K `10`

2. files inspected
- `/home/kaushik/slam_ws/visual_slam/orbslam/slam/config_parameters.py`
- `/home/kaushik/slam_ws/tests/visual_slam/orbslam/test_checkpoint_2_1_common.py`
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_36Z1_cw050_topk10/run_summary.json`
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_36Z1_cw050_topk10/loop_gt_positive_trace.csv`
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_36Z1_cw050_topk10/loop_geometry_trace.csv`
- `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_36Z1_cw050_topk10/run_log.txt`

3. pySLAM files inspected, if relevant
- None. No visual-SLAM logic was modified.

4. root cause or current hypothesis
- This run tested a more aggressive combination around the `2_36Y` winner: lower CW threshold (`0.50`) while keeping retrieval width at `10`.
- It succeeded at driving `FAILED_COMMON_WORD_FILTER` down sharply, but that improvement mostly shifted losses into `FAILED_CONNECTED_FILTER` rather than increasing accepted genuine loops.

5. exact changes made
- Temporarily changed `Parameters.kLoopClosingCommonWordRatioThreshold` from `0.67` to `0.50`.
- Temporarily changed `Parameters.kLoopDbowDetectorTopK` from `5` to `10`.
- Temporarily changed `Parameters.kMaxResultsForLoopClosure` from `5` to `10`.
- Reverted all three parameters to the baseline `2_35V` values after the run completed.

6. why the changes are structurally correct
- These were the exact user-requested `2_36Z1` tuning values.
- KF-density guardrail parameters remained unchanged.
- No implementation files outside the allowed scope were modified.

7. tests added/updated
- None.

8. test commands run
```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_1_common.py
```

9. test results
- Passed before the run: `5 passed in 0.31s`
- Passed again after restoring the baseline value: `5 passed in 0.30s`

10. dataset validation commands and results
```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
python -u -m visual_slam.orbslam.run_rgbd_slam \
  "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room" \
  --dataset-type tum_rgbd --camera-profile auto \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_36Z1_cw050_topk10" \
  --feature-backend pyslam_orb2 \
  --enable-loop-closing --disable-global-ba --no-map-export \
  --loop-debug --loop-candidate-source auto \
  --loop-retrieval-trace --loop-retrieval-trace-raw-k 100 \
  --print-every 100
```

- Final state: `OK`
- Accepted loops: `3`
- Final keyframes: `111`
- Final map points: `15573`
- Tracking lost: `0`
- Runtime: `4155.35 s`

11. remaining risks
- `FAILED_CONNECTED_FILTER` rose to `103`, which is the dominant sink in this run and suggests the extra CW-recovered candidates are mostly becoming structurally connected non-loop rejections.
- Final keyframe and map-point totals dropped below both `2_35V` and `2_36Y`, so the more aggressive setting may be reducing useful accepted structure rather than improving loop recall.

12. next recommended action
- Keep `2_36Y` as the current best run and use `2_36Z2` to test a different adjacent point rather than pushing further in the `2_36Z1` direction.
