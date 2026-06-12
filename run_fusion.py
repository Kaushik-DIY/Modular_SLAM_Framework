#!/usr/bin/env python
"""Fusion SLAM — single entry point (V4.3).

Choose a dataset, a mode, and (where applicable) the LiDAR local-mapping
variant and the loop verifier. Every tunable lives in ONE place:
slam_core/fusion2/config.py (FusionV2Config); LiDAR matcher profiles come from
hector/config.py per-dataset profiles.

    .venv/bin/python run_fusion.py --dataset datasets/lab_hybrid --mode lidar \
        [--lidar-frontend native_s2s|native_s2m|legacy_s2s|legacy_s2m] \
        [--verifier bnb|icp] [--output fusion2_outputs] [--max-scans N]

Modes (all through the same shared C++ map — memory tiers, SE(2) graph,
candidate-local verification):
    lidar      LiDAR front-end,  proximity proposals, scan verifier (bnb|icp)
    lidar_orb  LiDAR front-end,  proximity proposals, ORB+PnP verifier
    orb        visual front-end, DBoW proposals,      ORB+PnP verifier
    orb_lidar  visual front-end, DBoW proposals,      scan verifier (bnb|icp)

IMU is always active as the pose-prior fallback (extrapolator in LiDAR modes,
dropout dead-reckoning in visual modes).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slam_core.fusion2.runner import main

if __name__ == "__main__":
    main()
