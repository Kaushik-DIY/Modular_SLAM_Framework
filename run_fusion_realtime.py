#!/usr/bin/env python
"""Fusion real-time module-switching runner — entry point (V5).

Replays a dataset at real sensor cadence with a LIVE trajectory + accumulating
scan-cloud window, and lets you switch loop-closure modules and the LiDAR
front-end variant LIVE (typed on stdin) while the SAME shared C++ map keeps
mapping. The fused occupancy map is rendered at the end.

    .venv/bin/python run_fusion_realtime.py --dataset datasets/lab_hybrid \
        --mode lidar --lidar-frontend native_s2s --verifier bnb --attach-visual
    # or start visual-led:
    .venv/bin/python run_fusion_realtime.py --dataset datasets/lab_hybrid --mode orb_lidar

Live commands (type + Enter):
    verifier bnb|icp|pnp   · proposer proximity|dbow   · fe vo|lidar|s2s|s2m
    status                 · quit

Switch axes (all live; the shared C++ map persists across every switch):
    A1 front-end : visual_vo <-> native_s2s <-> native_s2m
                   (grace-buffer handoff; cross-sensor warms the new sensor's FE
                    on its own events, then flips fresh-from-origin — no teleport)
    A2 proposer  : proximity  <-> dbow         (dbow needs descriptors)
    A3 verifier  : bnb <-> icp <-> pnp         (pnp needs a visual payload)

A cross-sensor flip that strands the active verifier/proposer (e.g. onto a lean
LiDAR front-end) AUTO-FALLS-BACK to a compatible module (pnp->bnb, dbow->proximity)
and prints a notice. LiDAR keyframes carry visual only with --attach-visual (or
--mode lidar_orb); VO keyframes are always visual.

Out of scope (rejected live, with reason): memory-tier caps / grid resolution /
sensor calibration (startup-only — changing them mid-run destabilizes the map).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slam_core.fusion2.run_realtime import main

if __name__ == "__main__":
    main()
