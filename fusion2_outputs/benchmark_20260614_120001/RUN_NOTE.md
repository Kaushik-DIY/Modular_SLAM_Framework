# Benchmark run 20260614_120001

- commit: `90fbf8a2`
- maps: lab_hybrid_small, lab_hybrid
- combos: lidar_s2s_bnb, lidar_s2s_icp, lidar_s2m_bnb, lidar_s2m_icp, lidar_orb_s2s, lidar_orb_s2m, orb_lidar_bnb, orb_lidar_icp, orb (9 x 2 = 18 runs)
- true/false loop labelling: conservative (geometric scan-overlap AND graph residual); thresholds d_strict=0.10 m, tau_geom=0.50, res<=0.30 m / 5 deg
- accuracy: GT-free only (no ground truth) — end-start drift, loop-residual RMSE, map sharpness
- outputs: metrics.csv, BENCHMARK_REPORT.md, montages, per-run loop_labels.csv

