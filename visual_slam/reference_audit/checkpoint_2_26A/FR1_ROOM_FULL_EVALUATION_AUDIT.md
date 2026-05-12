# Checkpoint 2.26A FR1 Room Full Evaluation Audit

## 1. Purpose
Validate the existing RGB-D ORB2 pipeline on full `fr1_room` and generate benchmark logs, sparse map export, a GT-reference point cloud, and thesis-ready plots.

## 2. Dataset Used
`/home/kaushik/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk`

## 3. Backend Used
`pyslam_orb2`

## 4. Tests Run
See terminal report.

## 5. Dry-run A/B/C Summary
- Run A: `frames=100 ok=100 lost=0 state=OK loops=0 gba=0 ATE=0.019232146680487325 RPE_t=0.008227200746695557`
- Run B: `frames=100 ok=100 lost=0 state=OK loops=0 gba=0 ATE=0.024330987556595355 RPE_t=0.009272892106204885`
- Run C: `frames=100 ok=100 lost=0 state=OK loops=0 gba=0 ATE=0.024203606864303562 RPE_t=0.009465707496295269`

## 6. Full Run C Summary
`frames=596 ok=595 lost=1 state=OK loops=2 gba=2 ATE=1.0342549004414479 RPE_t=0.2278225138062677`

## 7. Loop Event Summary
- Candidates/events: `11`
- Accepted loops: `2`
- Rejected loops: `9`

## 8. Global BA Event Summary
- Started: `2`
- Success: `2`
- Failed: `0`
- Aborted: `0`

## 9. Trajectory Metrics
`{'groundtruth': '/home/kaushik/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt', 'trajectory': '/home/kaushik/slam_ws/visual_slam_outputs/fr1_desk_ablation_2_27C/run_C_loop_plus_gba/trajectory_rgbd_dataset_freiburg1_desk_smoke.txt', 'max_time_diff': 0.02, 'num_groundtruth_poses': 2335, 'num_estimated_poses': 595, 'num_associations': 595, 'ate_rmse_se3_m': 1.0342549004414479, 'ate_mean_se3_m': 0.9246959213704954, 'ate_median_se3_m': 0.8902514559090281, 'ate_max_se3_m': 1.9329538338385357, 'ate_rmse_sim3_m': 0.6160356479212605, 'sim3_scale': 0.4234499068371524, 'rpe_trans_rmse_m': 0.2278225138062677, 'rpe_rot_rmse_deg': 8.918093695628283, 'rpe_pairs': 594, 'mean_time_diff_s': 0.001726440221321683, 'max_association_time_diff_s': 0.013142108917236328}`

## 10. Map Export Summary
`{'map_points_ply': '/home/kaushik/slam_ws/visual_slam_outputs/fr1_desk_ablation_2_27C/run_C_loop_plus_gba/map_points.ply', 'keyframes_json': '/home/kaushik/slam_ws/visual_slam_outputs/fr1_desk_ablation_2_27C/run_C_loop_plus_gba/keyframes.json', 'keyframe_graph_json': '/home/kaushik/slam_ws/visual_slam_outputs/fr1_desk_ablation_2_27C/run_C_loop_plus_gba/keyframe_graph.json', 'num_exported_points': 3874, 'num_exported_keyframes': 19, 'num_covisibility_edges': 96, 'num_loop_edges': 2}`

## 11. Reference Cloud Generation Summary
`{}`

## 12. Visualization Outputs
- `/home/kaushik/slam_ws/visual_slam_outputs/fr1_desk_ablation_2_27C/comparison/trajectory_xy_comparison.png`
- `/home/kaushik/slam_ws/visual_slam_outputs/fr1_desk_ablation_2_27C/comparison/trajectory_3d_comparison.png`
- `/home/kaushik/slam_ws/visual_slam_outputs/fr1_desk_ablation_2_27C/comparison/trajectory_loop_events.png`
- `/home/kaushik/slam_ws/visual_slam_outputs/fr1_desk_ablation_2_27C/comparison/estimated_sparse_map_xy.png`
- `/home/kaushik/slam_ws/visual_slam_outputs/fr1_desk_ablation_2_27C/comparison/estimated_sparse_map_3d.png`
- `/home/kaushik/slam_ws/visual_slam_outputs/fr1_desk_ablation_2_27C/comparison/reference_cloud_xy.png`
- `/home/kaushik/slam_ws/visual_slam_outputs/fr1_desk_ablation_2_27C/comparison/map_side_by_side_xy.png`
- `/home/kaushik/slam_ws/visual_slam_outputs/fr1_desk_ablation_2_27C/comparison/keyframe_graph_xy.png`
- `/home/kaushik/slam_ws/visual_slam_outputs/fr1_desk_ablation_2_27C/comparison/metrics_table.png`
- `/home/kaushik/slam_ws/visual_slam_outputs/fr1_desk_ablation_2_27C/comparison/metrics_table.md`
- `/home/kaushik/slam_ws/visual_slam_outputs/fr1_desk_ablation_2_27C/comparison/map_side_by_side_xy_raw.png`
- `/home/kaushik/slam_ws/visual_slam_outputs/fr1_desk_ablation_2_27C/comparison/map_side_by_side_xy_aligned.png`
- `/home/kaushik/slam_ws/visual_slam_outputs/fr1_desk_ablation_2_27C/comparison/trajectory_loop_events_aligned.png`
- `/home/kaushik/slam_ws/visual_slam_outputs/fr1_desk_ablation_2_27C/comparison/estimated_sparse_map_xy_aligned.png`
- `/home/kaushik/slam_ws/visual_slam_outputs/fr1_desk_ablation_2_27C/comparison/keyframe_graph_xy_aligned.png`

## 13. Real Loop+GBA Exercised
Yes.

## 14. No Accepted Loop Diagnostic
Accepted loop closure was observed.

## 15. Remaining Gaps
None for the requested full A/B/C execution.

## 16. Next Recommended Action
Review loop-event diagnostics and thesis figures; run full A/B ablation if a complete comparison table is required.
