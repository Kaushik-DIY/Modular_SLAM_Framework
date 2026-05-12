# Loop Acceptance Debug and Fix Audit

## Purpose

Repair fr1_room loop acceptance so the full RGB-D ORB2 run exercises the real chain:

`loop candidate -> accepted loop -> loop fusion -> essential graph correction -> Global BA trigger/success`.

## Dataset and Backend

- Dataset: `/home/kaushik/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room`
- Backend: `pyslam_orb2`
- Output root: `/home/kaushik/slam_ws/visual_slam_outputs/fr1_room_full_evaluation_2_26B`

## Baseline

The previous full Run C was stable but did not exercise loop-triggered GBA:

- `tracking_ok_count=1362`
- `accepted_loops=0`
- `global_ba_started=0`
- `loop_edges=0`

## Root Causes Found

- Candidate retrieval did not use native DBOW3 query parity first.
- Geometry verification rejected candidates before guided projection.
- Low-seed guided projection could accept perceptual aliases without an odometry prior.
- After loop/GBA, tracking could use a stale motion model and lose one frame.

## Structural Fixes

- Added native DBOW3 loop candidate query path.
- Added robust scale-fixed SE3 RANSAC.
- Added deep candidate diagnostics and pair JSON dumps.
- Added guided projection before final acceptance.
- Added final SE3 refinement over guided matches with the original 20-inlier gate.
- Added estimated pose distance/rotation guards for low-seed guided candidates.
- Reset tracking motion model/reference after successful loop correction/GBA.
- Added aligned map plots and robust aligned map axes.

## Diagnostic Runs

- Stop-after-loop-events diagnostics showed early rejection by consistency, then geometry seed failures.
- First guided-seed diagnostic accepted a real candidate but full run remained brittle.
- Lowered seed without pose priors accepted false aliases; those were rejected after adding estimated pose distance/rotation guards.
- Final bounded diagnostic accepted a GT-valid real loop: KF44 -> KF8, GT distance about 0.45 m, final refined inliers 24.

## Final Full Run C

- `frames_attempted=1362`
- `tracking_ok_count=1362`
- `tracking_lost_count=0`
- `errors=0`
- `accepted_loops=1`
- `essential_graph_runs=1`
- `global_ba_started=1`
- `global_ba_success=1`
- `loop_edges=1`
- `final_keyframes=53`
- `final_map_points=6464`

Accepted full-run loop:

- current KF: `46`
- candidate KF: `10`
- GT separation: about `0.13 m`, about `22.54 deg`, about `31.77 s`
- seed inliers: `26`
- guided/refined final inliers: `84`

## Metrics

- ATE RMSE SE3: `0.15981577220144136 m`
- ATE RMSE Sim3: `0.15298449765180153 m`
- RPE translation RMSE: `0.018432575438086955 m`
- RPE rotation RMSE: `0.7755095938929646 deg`

## Map and Visualization

- Estimated map: `run_C_loop_plus_gba/map_points.ply`
- Keyframes: `run_C_loop_plus_gba/keyframes.json`
- Keyframe graph: `run_C_loop_plus_gba/keyframe_graph.json`
- GT reference cloud: `reference_map/reference_cloud_gt.ply`
- Aligned side-by-side plot: `comparison/map_side_by_side_xy_aligned.png`

The aligned map comparison now uses the trajectory alignment and robust axes so the sparse estimated map and GT-reference cloud can be compared in the same coordinate frame.

## Tests Run

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_21_loop_closing.py tests/visual_slam/orbslam/test_checkpoint_2_26B_loop_acceptance_debug.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam`

Both passed after the final code changes.

## Remaining Gaps

- Full A/B ablation was deferred.
- pySLAM could not be executed directly because `torch` is absent from the project virtualenv.

## Next Recommended Action

Review generated figures and, if thesis comparison requires ablations, run full A/B/C with the repaired loop verifier.
