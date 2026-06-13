#!/usr/bin/env python
"""Fusion real-time module-switching runner — entry point (V5).

Replays a dataset at real sensor cadence with a LIVE trajectory + accumulating
scan-cloud window, and lets you switch loop-closure modules and the LiDAR
front-end variant LIVE (typed on stdin) while the SAME shared C++ map keeps
mapping. The fused occupancy map is rendered at the end.

    .venv/bin/python run_fusion_realtime.py --dataset datasets/lab_hybrid \
        --mode lidar --lidar-frontend native_s2s --verifier bnb [--attach-visual]

Live commands (type + Enter):
    verifier bnb|icp|pnp   · proposer proximity|dbow   · fe s2s|s2m
    status                 · quit

Switch axes (Phase 1 = LiDAR-led front-ends):
    A1 front-end : native_s2s <-> native_s2m   (grace-buffer handoff, fresh local map)
    A2 proposer  : proximity  <-> dbow         (dbow needs descriptors: --attach-visual)
    A3 verifier  : bnb <-> icp <-> pnp         (pnp needs visual: --attach-visual)

Out of scope (rejected live, with reason): cross-sensor visual<->LiDAR front-end
(Phase 2, needs a unified multi-sensor driver); memory-tier caps / grid resolution
/ sensor calibration (startup-only — changing them mid-run destabilizes the map).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slam_core.fusion2.run_realtime import main

if __name__ == "__main__":
    main()
