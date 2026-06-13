# Visual Comparison Report - Checkpoint 2.39 Vectorized Triangulation

Date: 2026-06-08

Run under review:

`visual_slam/reference_audit/checkpoint_2_39_full_lab_vectorized_triang_threaded_20260608/run`

Baseline reference:

`visual_slam_outputs/lab_rgbd_run_2_B_loop_gba`

## Purpose

Generate durable trajectory, sparse-map, and semi-dense visualization outputs for the completed full lab run, and compare them against the established loop-closure + global-BA baseline.

## Commands Used

Baseline comparison plots:

```bash
PYTHONPATH=. .venv/bin/python -m tools.plot_slam_results \
  --run-a visual_slam/reference_audit/checkpoint_2_39_full_lab_vectorized_triang_threaded_20260608/run \
  --label-a checkpoint_2_39_vectorized_triang \
  --run-b visual_slam_outputs/lab_rgbd_run_2_B_loop_gba \
  --label-b baseline_loop_gba \
  --output visual_slam/reference_audit/checkpoint_2_39_full_lab_vectorized_triang_threaded_20260608/plots_vs_baseline
```

Map/presentation figures:

```bash
PYTHONPATH=. .venv/bin/python -m tools.generate_lab_map \
  --run visual_slam/reference_audit/checkpoint_2_39_full_lab_vectorized_triang_threaded_20260608/run \
  --dataset datasets/lab_rgbd_run_2 \
  --output visual_slam/reference_audit/checkpoint_2_39_full_lab_vectorized_triang_threaded_20260608/map_figures
```

Baseline-style per-run plots:

```bash
PYTHONPATH=. .venv/bin/python -m tools.plot_rgbd_run \
  --run visual_slam/reference_audit/checkpoint_2_39_full_lab_vectorized_triang_threaded_20260608/run \
  --output visual_slam/reference_audit/checkpoint_2_39_full_lab_vectorized_triang_threaded_20260608/plots
```

## Generated Files

Comparison plots:

- `plots_vs_baseline/compare_trajectory.png`
- `plots_vs_baseline/compare_map.png`
- `plots_vs_baseline/compare_trajectory_map.png`
- `plots_vs_baseline/a/trajectory_xy.png`
- `plots_vs_baseline/a/tracking_quality.png`
- `plots_vs_baseline/a/map_xy.png`
- `plots_vs_baseline/a/metrics_panel.png`
- `plots_vs_baseline/a/metrics_table.md`
- `plots_vs_baseline/b/trajectory_xy.png`
- `plots_vs_baseline/b/tracking_quality.png`
- `plots_vs_baseline/b/map_xy.png`
- `plots_vs_baseline/b/metrics_panel.png`
- `plots_vs_baseline/b/metrics_table.md`

Map figures:

- `map_figures/eval_sparse_map.png`
- `map_figures/eval_trajectory_graph.png`
- `map_figures/eval_tracking_quality.png`
- `map_figures/pres_sparse_map.png`
- `map_figures/pres_semidense_topdown.png`
- `map_figures/pres_semidense_3d.png`
- `map_figures/pres_summary.png`

Per-run plots:

- `plots/trajectory_topdown.png`
- `plots/trajectory_3d.png`
- `plots/tracking_stats.png`
- `plots/map_topdown.png`
- `plots/map_3d.png`
- `plots/summary_panel.png`

## Metric Comparison

| Metric | Checkpoint 2.39 | Baseline loop+GBA |
|---|---:|---:|
| Total frames | 4494 | 4494 |
| Tracking OK | 4434 | 4494 |
| Tracking LOST | 60 | 0 |
| Final keyframes | 214 | 152 |
| Final map points | 37529 | 10580 |
| Filtered sparse points shown | 36027 | 9117 |
| Loop edges | 0 | 3 |
| Global BA events | 0 | 3 |
| Mean BA MSE | 1.3670 | 1.8987 |

Ground-truth ATE/RPE metrics were not available for either run in the plotting inputs, so this comparison uses trajectory alignment against the baseline as a reference only.

## Baseline-Aligned Trajectory Delta

The checkpoint trajectory was aligned to the baseline trajectory using matched timestamps.

| Metric | Value |
|---|---:|
| Matched poses | 4434 |
| Checkpoint poses | 4434 |
| Baseline poses | 4494 |
| 3D RMSE vs baseline | 0.2357 m |
| 3D mean vs baseline | 0.2210 m |
| 3D median vs baseline | 0.2411 m |
| 3D max vs baseline | 0.3575 m |
| XZ RMSE vs baseline | 0.2348 m |
| XZ mean vs baseline | 0.2200 m |
| Raw checkpoint path length | 32.3682 m |
| Baseline path length | 32.7914 m |
| Start alignment error | 0.2999 m |
| End alignment error | 0.3443 m |

## Visual Read

The checkpoint trajectory follows the same broad lab route as the baseline and is closer to the baseline than checkpoint 2.38 by the aligned trajectory metric. The sparse map remains much denser than the baseline: about 36.0k filtered displayed points versus about 9.1k in the loop+GBA baseline.

The baseline remains the cleaner visual quality target. It has zero tracking loss, three loop edges, and three global BA events. Checkpoint 2.39 was run with loop closing disabled, so its trajectory and graph are not expected to be as tightly loop-corrected as the baseline.

## Verdict

The full-run visualization outputs are generated and saved under this checkpoint folder for future comparison.

Checkpoint 2.39 improves the prior full-run reference: tracking loss is lower, full-run FPS is slightly higher, and baseline-aligned trajectory RMSE is lower. It is acceptable as the new no-loop runtime validation artifact, but it is still not visually equivalent to the loop+GBA baseline.

Note: `map_figures/pres_summary.png` uses a legacy plot title mentioning loop closure + global BA, but the numeric stats in the figure and this report correctly show `0` loop edges and `0` GBA events for this checkpoint run.
