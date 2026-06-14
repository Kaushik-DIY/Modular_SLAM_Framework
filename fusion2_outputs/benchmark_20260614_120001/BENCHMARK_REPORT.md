# Fusion benchmark — methods x environments

Commit `90fbf8a2`  ·  generated 20260614_120001  ·  GT-free metrics (no ground truth in these datasets).

True/false loops are **conservative**: a loop is TRUE only if BOTH the geometric scan-overlap AND the graph-residual checks agree (`tools/analyze_fusion_loops.py`).


## lab_hybrid_small

### Loop closures

| combo | mode | verifier | proposals | verified | loops_accepted | true_accepted | false_accepted | disagreement | precision | accept_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| lidar_s2s_bnb | lidar | bnb | 41 | 41 | 31 | 31 | 0 | 0 | 1 | 0.756 |
| lidar_s2s_icp | lidar | icp | 41 | 41 | 30 | 26 | 4 | 4 | 0.867 | 0.732 |
| lidar_s2m_bnb | lidar | bnb | 38 | 38 | 33 | 33 | 0 | 0 | 1 | 0.868 |
| lidar_s2m_icp | lidar | icp | 38 | 38 | 36 | 36 | 0 | 0 | 1 | 0.947 |
| lidar_orb_s2s | lidar_orb | pnp | 41 | 39 | 8 | 8 | 0 | 0 | 1 | 0.195 |
| lidar_orb_s2m | lidar_orb | pnp | 38 | 36 | 8 | 8 | 0 | 0 | 1 | 0.211 |
| orb_lidar_bnb | orb_lidar | bnb | 8 | 7 | 0 | 0 | 0 | 0 | - | 0 |
| orb_lidar_icp | orb_lidar | icp | 8 | 7 | 6 | 5 | 1 | 1 | 0.833 | 0.75 |
| orb | orb | pnp | 8 | 8 | 8 | 7 | 1 | 1 | 0.875 | 1 |


### Trajectory & map quality (GT-free)

| combo | end_start_drift_m | loop_resid_rmse_m | loop_resid_rmse_deg | map_sharpness | map_occupied_cells | map_free_cells | traj_span_m |
|---|---|---|---|---|---|---|---|
| lidar_s2s_bnb | 0.494 | 0.068 | 0.33 | 0.555 | 5211 | 20294 | 8.0x4.2 |
| lidar_s2s_icp | 0.479 | 0.124 | 0.34 | 0.544 | 5279 | 20507 | 8.0x4.1 |
| lidar_s2m_bnb | 0.033 | 0.007 | 0.05 | 0.634 | 3977 | 19812 | 8.2x3.7 |
| lidar_s2m_icp | 0.018 | 0.016 | 0.1 | 0.635 | 3993 | 19837 | 8.2x3.7 |
| lidar_orb_s2s | 0.442 | 0.102 | 0.53 | 0.564 | 5054 | 20889 | 7.9x4.4 |
| lidar_orb_s2m | 0.028 | 0.016 | 0.31 | 0.636 | 3953 | 19846 | 8.2x3.7 |
| orb_lidar_bnb | 2.437 | - | - | 0.407 | 5805 | 26203 | 8.4x5.1 |
| orb_lidar_icp | 2.564 | 0.037 | 0.22 | 0.407 | 5835 | 25989 | 8.4x5.1 |
| orb | 1.224 | 0.027 | 0.45 | 0.473 | 5676 | 23641 | 8.3x3.8 |


### Cost & health

| combo | keyframes | peak_rss_gb | map_payload_mb | stm | wm | ltm | elapsed_sec | reinits | fallbacks |
|---|---|---|---|---|---|---|---|---|---|
| lidar_s2s_bnb | 159 | 0.13 | 0.6 | 30 | 122 | 0 | 21.7 | - | 1 |
| lidar_s2s_icp | 159 | 0.18 | 0.6 | 30 | 122 | 0 | 19.3 | - | 1 |
| lidar_s2m_bnb | 155 | 0.2 | 0.6 | 30 | 115 | 0 | 27.6 | - | 1 |
| lidar_s2m_icp | 155 | 0.22 | 0.6 | 30 | 115 | 0 | 26.7 | - | 1 |
| lidar_orb_s2s | 159 | 0.37 | 9 | 30 | 122 | 0 | 23.4 | - | 1 |
| lidar_orb_s2m | 155 | 0.39 | 8.3 | 30 | 115 | 0 | 30.7 | - | 1 |
| orb_lidar_bnb | 249 | 0.84 | 4.4 | 30 | 200 | 19 | 81.4 | 78 | - |
| orb_lidar_icp | 249 | 0.86 | 4.4 | 30 | 200 | 19 | 80 | 78 | - |
| orb | 249 | 0.87 | 4.4 | 30 | 200 | 19 | 80.4 | 78 | - |



## lab_hybrid

### Loop closures

| combo | mode | verifier | proposals | verified | loops_accepted | true_accepted | false_accepted | disagreement | precision | accept_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| lidar_s2s_bnb | lidar | bnb | 110 | 110 | 71 | 69 | 2 | 2 | 0.972 | 0.645 |
| lidar_s2s_icp | lidar | icp | 110 | 110 | 70 | 64 | 6 | 6 | 0.914 | 0.636 |
| lidar_s2m_bnb | lidar | bnb | 104 | 100 | 74 | 74 | 0 | 0 | 1 | 0.712 |
| lidar_s2m_icp | lidar | icp | 104 | 101 | 84 | 84 | 0 | 0 | 1 | 0.808 |
| lidar_orb_s2s | lidar_orb | pnp | 110 | 110 | 26 | 24 | 2 | 2 | 0.923 | 0.236 |
| lidar_orb_s2m | lidar_orb | pnp | 103 | 103 | 25 | 25 | 0 | 0 | 1 | 0.243 |
| orb_lidar_bnb | orb_lidar | bnb | 53 | 53 | 42 | 42 | 0 | 0 | 1 | 0.792 |
| orb_lidar_icp | orb_lidar | icp | 53 | 53 | 39 | 35 | 4 | 4 | 0.897 | 0.736 |
| orb | orb | pnp | 53 | 53 | 38 | 38 | 0 | 0 | 1 | 0.717 |


### Trajectory & map quality (GT-free)

| combo | end_start_drift_m | loop_resid_rmse_m | loop_resid_rmse_deg | map_sharpness | map_occupied_cells | map_free_cells | traj_span_m |
|---|---|---|---|---|---|---|---|
| lidar_s2s_bnb | 0.386 | 0.196 | 0.76 | 0.526 | 9797 | 40643 | 20.4x4.7 |
| lidar_s2s_icp | 1.303 | 0.192 | 0.52 | 0.532 | 9371 | 41547 | 20.2x4.7 |
| lidar_s2m_bnb | 0.113 | 0.014 | 0.07 | 0.627 | 7585 | 39381 | 20.7x3.7 |
| lidar_s2m_icp | 0.263 | 0.018 | 0.1 | 0.626 | 7745 | 39689 | 20.7x3.7 |
| lidar_orb_s2s | 0.467 | 0.107 | 0.6 | 0.536 | 8910 | 42112 | 19.3x4.5 |
| lidar_orb_s2m | 0.131 | 0.038 | 0.63 | 0.625 | 7589 | 39570 | 20.7x3.9 |
| orb_lidar_bnb | 1.156 | 0.042 | 0.54 | 0.527 | 10532 | 47745 | 19.0x7.5 |
| orb_lidar_icp | 7.948 | 0.156 | 0.75 | 0.53 | 11722 | 62822 | 20.1x8.3 |
| orb | 1.825 | 0.033 | 0.76 | 0.522 | 10690 | 48350 | 18.7x8.4 |


### Cost & health

| combo | keyframes | peak_rss_gb | map_payload_mb | stm | wm | ltm | elapsed_sec | reinits | fallbacks |
|---|---|---|---|---|---|---|---|---|---|
| lidar_s2s_bnb | 382 | 0.73 | 1.7 | 30 | 200 | 139 | 55 | - | 1 |
| lidar_s2s_icp | 382 | 0.71 | 1.7 | 30 | 200 | 139 | 51.5 | - | 1 |
| lidar_s2m_bnb | 368 | 0.72 | 1.6 | 30 | 200 | 123 | 72.7 | - | 75 |
| lidar_s2m_icp | 368 | 0.74 | 1.6 | 30 | 200 | 123 | 77.5 | - | 75 |
| lidar_orb_s2s | 382 | 0.71 | 22.9 | 30 | 200 | 139 | 65.3 | - | 1 |
| lidar_orb_s2m | 368 | 0.74 | 21.4 | 30 | 200 | 123 | 108 | - | 75 |
| orb_lidar_bnb | 645 | 0.96 | 11.3 | 30 | 200 | 415 | 225 | 117 | - |
| orb_lidar_icp | 645 | 1.01 | 11.3 | 30 | 200 | 415 | 222.9 | 117 | - |
| orb | 645 | 1.03 | 11.3 | 30 | 200 | 415 | 205.1 | 117 | - |



## Cross-map summary (precision / drift / sharpness)

| combo | lab_hybrid_small_prec | lab_hybrid_small_drift | lab_hybrid_small_sharp | lab_hybrid_prec | lab_hybrid_drift | lab_hybrid_sharp |
|---|---|---|---|---|---|---|
| lidar_s2s_bnb | 1 | 0.494 | 0.555 | 0.972 | 0.386 | 0.526 |
| lidar_s2s_icp | 0.867 | 0.479 | 0.544 | 0.914 | 1.303 | 0.532 |
| lidar_s2m_bnb | 1 | 0.033 | 0.634 | 1 | 0.113 | 0.627 |
| lidar_s2m_icp | 1 | 0.018 | 0.635 | 1 | 0.263 | 0.626 |
| lidar_orb_s2s | 1 | 0.442 | 0.564 | 0.923 | 0.467 | 0.536 |
| lidar_orb_s2m | 1 | 0.028 | 0.636 | 1 | 0.131 | 0.625 |
| orb_lidar_bnb | - | 2.437 | 0.407 | 1 | 1.156 | 0.527 |
| orb_lidar_icp | 0.833 | 2.564 | 0.407 | 0.897 | 7.948 | 0.53 |
| orb | 0.875 | 1.224 | 0.473 | 1 | 1.825 | 0.522 |


