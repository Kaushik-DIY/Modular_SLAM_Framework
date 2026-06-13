# Fusion V4 — final 9-combination validation matrix (lab_hybrid BIG)

Run folder: `fusion2_outputs/final_matrix_20260612_154851`

## Loop-closure accounting

| # | Combo | Mode | Front-end | Verifier | KFs | Proposed | Verified | **Accepted** | Rate |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | `lidar_s2s_bnb` | lidar | scan_to_submap | B&B | 382 | 110 | 110 | **71** | 65% |
| 2 | `lidar_s2s_icp` | lidar | scan_to_submap | ICP | 382 | 110 | 110 | **34** | 31% |
| 3 | `lidar_s2m_bnb` | lidar | scan_to_map | B&B | 368 | 104 | 100 | **74** | 71% |
| 4 | `lidar_s2m_icp` | lidar | scan_to_map | ICP | 368 | 104 | 101 | **74** | 71% |
| 5 | `lidar_orb_s2s` | lidar_orb | scan_to_submap | ORB+PnP | 382 | 110 | 110 | **26** | 24% |
| 6 | `lidar_orb_s2m` | lidar_orb | scan_to_map | ORB+PnP | 368 | 103 | 103 | **25** | 24% |
| 7 | `orb_lidar_bnb` | orb_lidar | native VO | B&B | 645 | 53 | 53 | **42** | 79% |
| 8 | `orb_lidar_icp` | orb_lidar | native VO | ICP | 645 | 53 | 53 | **42** | 79% |
| 9 | `orb_pnp` | orb | native VO | ORB+PnP | 645 | 53 | 53 | **38** | 72% |

## Runtime / memory / IMU

| Combo | Wall (min) | fps / scan-ms | Verify ms | Peak RSS | Fallbacks / reinits | IMU | Tiers S/W/L |
|---|---:|---:|---:|---:|---:|---|---:|
| `lidar_s2s_bnb` | 1.0 | - | 48.7 | 0.16 GB | 1 | on | 30/200/139 |
| `lidar_s2s_icp` | 0.9 | - | 19.6 | 0.23 GB | 1 | on | 30/200/139 |
| `lidar_s2m_bnb` | 1.2 | - | 32.2 | 0.25 GB | 75 | on | 30/200/123 |
| `lidar_s2m_icp` | 1.2 | - | 18.2 | 0.25 GB | 75 | on | 30/200/123 |
| `lidar_orb_s2s` | 1.0 | - | 8.6 | 0.37 GB | 1 | on | 30/200/139 |
| `lidar_orb_s2m` | 1.3 | - | 8.4 | 0.4 GB | 75 | on | 30/200/123 |
| `orb_lidar_bnb` | 3.4 | 32.26 fps | 67.5 | 0.91 GB | 117 | on | 30/200/415 |
| `orb_lidar_icp` | 3.5 | 32.04 fps | 84.2 | 0.96 GB | 117 | on | 30/200/415 |
| `orb_pnp` | 3.4 | 32.29 fps | 3.1 | 0.99 GB | 117 | on | 30/200/415 |
