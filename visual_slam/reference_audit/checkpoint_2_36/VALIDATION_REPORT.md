# VALIDATION_REPORT

Checkpoint `2_36` validation is in progress.

Per the tuning plan, each run will record:
- venv verification
- targeted pytest result for `tests/visual_slam/orbslam/test_checkpoint_2_1_common.py`
- `fr1_room` run command
- completion status and elapsed time
- `run_summary.json` metrics
- `loop_gt_positive_trace.csv` funnel counts
- accepted loop list

## Run `2_36W`

### Venv verification

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
python -c "import sys; print(sys.executable)"
```

Result:
- `/home/kaushik/slam_ws/.venv/bin/python`

### Unit test

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_1_common.py
```

Result:
- `5 passed in 0.40s`

### Run command

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

### Completion status
- Completed successfully
- Final state: `OK`
- Elapsed time: `3738.95 s` (`62.3 min`)

### Summary metrics
- accepted_loops: `3`
- final_keyframes: `121`
- final_map_points: `18545`
- tracking_lost_count: `26`

### GT funnel counts from `loop_gt_positive_trace.csv`
- ACCEPTED: `3`
- FAILED_CONSISTENCY: `6`
- FAILED_ACCUMULATION_FILTER: `29`
- FAILED_MIN_SCORE_FILTER: `16`
- FAILED_COMMON_WORD_FILTER: `45`
- FAILED_CONNECTED_FILTER: `48`
- MISSING_FROM_RAW_DBOW: `34`
- UNKNOWN: `34`
- TOTAL: `215`

### Accepted loop list
- `KF120 -> KF13`, GT distance `0.208 m`, final matched map points `62`
- `KF130 -> KF18`, GT distance `0.201 m`, final matched map points `65`
- `KF141 -> KF28`, GT distance `0.263 m`, final matched map points `43`

### Post-run reset validation

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_1_common.py
```

Result:
- `5 passed in 0.31s`

## Run `2_36X`

### Unit test

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_1_common.py
```

Result:
- `5 passed in 0.30s`

### Run command

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

### Completion status
- Completed successfully
- Final state: `OK`
- Elapsed time: `3671.09 s` (`61.2 min`)

### Summary metrics
- accepted_loops: `3`
- final_keyframes: `114`
- final_map_points: `16211`
- tracking_lost_count: `0`

### GT funnel counts from `loop_gt_positive_trace.csv`
- ACCEPTED: `2`
- FAILED_CONSISTENCY: `2`
- FAILED_ACCUMULATION_FILTER: `27`
- FAILED_MIN_SCORE_FILTER: `6`
- FAILED_COMMON_WORD_FILTER: `75`
- FAILED_CONNECTED_FILTER: `80`
- FAILED_SEED_GEOMETRY: `3`
- MISSING_FROM_RAW_DBOW: `46`
- UNKNOWN: `26`
- TOTAL: `267`

### Accepted loop list
- `KF129 -> KF16`, GT distance `0.165 m`, final matched map points `132`
- `KF143 -> KF42`, GT distance `1.299 m`, final matched map points `45`
- `KF153 -> KF33`, GT distance `0.145 m`, final matched map points `134`

### Interpretation note
- Only `2` accepted pairs appear in `loop_gt_positive_trace.csv` because the third accepted loop is not GT-near.
- The accepted `KF143 -> KF42` pair appears in `loop_geometry_trace.csv` with `accepted=True`, but its GT distance is `1.2988 m`, so it should be treated as a likely false positive for checkpoint comparison.

## Run `2_36Y`

### Unit test

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_1_common.py
```

Result:
- `5 passed in 0.30s`

### Run command

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

### Completion status
- Completed successfully
- Final state: `OK`
- Elapsed time: `4083.90 s` (`68.1 min`)

### Summary metrics
- accepted_loops: `4`
- final_keyframes: `127`
- final_map_points: `17984`
- tracking_lost_count: `0`

### GT funnel counts from `loop_gt_positive_trace.csv`
- ACCEPTED: `4`
- FAILED_CONSISTENCY: `2`
- FAILED_ACCUMULATION_FILTER: `32`
- FAILED_MIN_SCORE_FILTER: `7`
- FAILED_COMMON_WORD_FILTER: `36`
- FAILED_CONNECTED_FILTER: `65`
- MISSING_FROM_RAW_DBOW: `27`
- UNKNOWN: `34`
- TOTAL: `207`

### Accepted loop list
- `KF121 -> KF13`, GT distance `0.208 m`, final matched map points `234`
- `KF132 -> KF19`, GT distance `0.226 m`, final matched map points `88`
- `KF144 -> KF29`, GT distance `0.276 m`, final matched map points `73`
- `KF155 -> KF31`, GT distance `0.141 m`, final matched map points `48`

### Interpretation note
- All `4` accepted loops are GT-near in `loop_gt_positive_trace.csv`.
- The same `4` accepted pairs appear in `loop_geometry_trace.csv`, and all remain below the checkpoint's `0.5 m` genuine-loop threshold.
- This is the first Phase 1 run to improve the primary metric while preserving zero false positives and zero tracking losses.

### Post-run reset validation

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_1_common.py
```

Result:
- `5 passed in 0.30s`

## Run `2_36Z1`

### Unit test

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_1_common.py
```

Result:
- `5 passed in 0.31s`

### Run command

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

### Completion status
- Completed successfully
- Final state: `OK`
- Elapsed time: `4155.35 s` (`69.3 min`)

### Summary metrics
- accepted_loops: `3`
- final_keyframes: `111`
- final_map_points: `15573`
- tracking_lost_count: `0`

### GT funnel counts from `loop_gt_positive_trace.csv`
- ACCEPTED: `3`
- FAILED_ACCUMULATION_FILTER: `26`
- FAILED_COMMON_WORD_FILTER: `10`
- FAILED_CONNECTED_FILTER: `103`
- FAILED_MIN_SCORE_FILTER: `9`
- FAILED_SEED_GEOMETRY: `3`
- MISSING_FROM_RAW_DBOW: `33`
- UNKNOWN: `44`
- TOTAL: `231`

### Accepted loop list
- `KF119 -> KF10`, GT distance `0.316 m`, final matched map points `151`
- `KF142 -> KF28`, GT distance `0.237 m`, final matched map points `75`
- `KF152 -> KF33`, GT distance `0.279 m`, final matched map points `92`

### Interpretation note
- All `3` accepted loops are GT-near in both `loop_gt_positive_trace.csv` and `loop_geometry_trace.csv`.
- This run preserved zero false positives and zero tracking losses, but it did not improve the genuine accepted-loop count over `2_35V`, and it underperformed `2_36Y`.
- The strongest apparent gain was CW-filter recovery (`FAILED_COMMON_WORD = 10`), but the dominant bottleneck shifted to `FAILED_CONNECTED_FILTER = 103`.

### Post-run reset validation

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_1_common.py
```

Result:
- `5 passed in 0.30s`
