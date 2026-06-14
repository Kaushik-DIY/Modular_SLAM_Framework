# fusion2 validation runs index

| Run dir | Mode / dataset | Result |
|---|---|---|
| `lidar_20260611_021436` | lidar / lab_hybrid (BIG) | 382 kf, **74 B&B loops**, clean 2-room map, 0.12 GB, 17 min |
| `lidar_orb_20260611_032124` | lidar_orb / lab_hybrid (BIG) | 382 kf, **26 PnP cross-modal loops** @7.5 ms, 2-room map, 0.29 GB |
| `lidar_orb_20260611_023003` | lidar_orb / small | 159 kf, 8 PnP loops @7 ms |
| `lidar_20260611_023840` | lidar --verifier icp / small | 159 kf, 30 ICP loops @10.6 ms |
| `orb_lidar_20260611_030345` | orb_lidar / small | 406 kf, 49 DBoW proposals, 1 B&B loop (gate met) |
| `orb_20260611_020300` | orb / small (pre-loop code) | 428 kf through shared map, tiers exact |

Older `lidar_2026061x_*` dirs are diagnostic iterations (0-loop score-plateau runs
kept for the verifications.csv evidence trail).

## V3 native-frontend runs (2026-06-11)

| Run dir | Mode / dataset | Result |
|---|---|---|
| `orb_native_20260611_045709` | orb native / lab_hybrid (BIG) | **13 PnP loops**, 34.5 fps, 0.71 GB, 844 kf |
| `orb_lidar_native_20260611_051046` | orb_lidar native ±180° / BIG | **22 cross-modal loops**, 25.9 fps, 0.70 GB |
| `orb_native_20260611_045338` | orb native / small | 10 PnP loops, 34 fps, 0.66 GB (A/B winner) |
| `orb_native_20260611_044709` | orb native / small (diagnostic) | REINIT-storm run that exposed the IMU-prior axis bug |
| `orb_lidar_native_20260611_050040/0537` | orb_lidar / BIG (diagnostics) | ±30°/±45° window iterations (2 loops) |

## V3.5 native-frontend + local BA runs (2026-06-11)

| Run dir | Mode / dataset | Result |
|---|---|---|
| `orb_native_20260611_162036` | orb native+BA / lab_hybrid (BIG) | **20 PnP loops, 177 reinits** (was 328), two-room corridor map, 30 fps, 0.70 GB |
| `orb_lidar_native_20260611_164725` | orb_lidar native+BA / BIG | **11 B&B loops, 260 reinits**, two room clusters, 23.9 fps, 0.71 GB |
| `orb_native_20260611_161824` | orb native+BA / small | 304 kf, 11 loops, 162 reinits (6.6%, beats reference 8.5%) |
| `../visual_slam_outputs/lab_hybrid_small_reference` | reference ORB-SLAM2 / small | 411 kf, 210 tracking-lost, 548 local-BA — baseline; also radially smeared (one-room is hard for vision) |

## V3.6 root-cause fixes (2026-06-11): frame consistency + BF recovery + scoped IMU + blind-edge handling

| Run dir | Mode / dataset | Result |
|---|---|---|
| `orb_native_20260611_210119` | orb / lab_hybrid (BIG) | **38 loops, 117 reinits, CLEAN two-room map (straight walls, end≈start)**, 32.6 fps, 0.72 GB; before/after montage inside |
| `orb_lidar_native_20260611_210543` | orb_lidar / lab_hybrid (BIG) | **43 cross-modal loops, two-room map**, 31.9 fps, 0.72 GB |

## V4 final matrix (2026-06-12) — THE thesis baseline runs

`final_matrix_20260612_154851/` — all 9 combinations on lab_hybrid (BIG), native
front-ends, IMU-assisted, fused occupancy maps. REPORT.md + maps_montage.png +
trajectories_montage.png inside; per-combo subfolders carry trajectory.tum,
occupancy.png, map.npy, verifications.csv, run_summary.json. Closures:
LiDAR-led 0.11-0.47 m; vision-led 1.16-1.81 m. All < 1 GB, all faster than
sensor rate (lidar ~1 min wall; orb-led 32 fps).
- [benchmark_20260614_120001](benchmark_20260614_120001/BENCHMARK_REPORT.md) — 9 combos x 2 maps, GT-free method comparison
