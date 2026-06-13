# Fusion v2 — final four-mode validation (lab_hybrid BIG)

Run folder: `fusion2_outputs/final_4mode_20260611_223735`

## Loop-closure accounting

| Mode | Front-end | Proposer | Verifier | Keyframes | Loops proposed | Verified | **Accepted** | Accept-rate |
|---|---|---|---|---:|---:|---:|---:|---:|
| `lidar` | LiDAR scan_to_submap | proximity (WM) | candidate-local B&B (scan) | 382 | 110 | 110 | **74** | 67% |
| `orb` | native windowed VO | DBoW appearance | ORB match + PnP | 645 | 53 | 53 | **38** | 72% |
| `orb_lidar` | native windowed VO | DBoW appearance | candidate-local B&B (LiDAR) | 645 | 53 | 53 | **43** | 81% |
| `lidar_orb` | LiDAR scan_to_submap | proximity (WM) | ORB match + PnP | 382 | 110 | 110 | **26** | 24% |

## Runtime / resources

| Mode | Keyframes | Reinits (VO) / blind | Verify ms (mean) | fps | Peak RSS | Wall (min) | Tiers S/W/L |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lidar` | 382 | - | 54.4 | - | 0.68 GB | 19.4 | 30/200/139 |
| `orb` | 645 | 117 | 2.9 | 30.45 | 0.72 GB | 3.6 | 30/200/415 |
| `orb_lidar` | 645 | 117 | 114.6 | 29.75 | 0.79 GB | 3.7 | 30/200/415 |
| `lidar_orb` | 382 | - | 9.4 | - | 0.78 GB | 19.3 | 30/200/139 |

- `orb` online IMU self-calibration: {'yaw_sign': 1.0, 'up_axis': [0.1398, -0.9901, 0.0124]}

- `orb_lidar` online IMU self-calibration: {'yaw_sign': 1.0, 'up_axis': [0.1398, -0.9901, 0.0124]}
