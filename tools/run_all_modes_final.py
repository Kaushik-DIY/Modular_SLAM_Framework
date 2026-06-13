"""Final four-mode validation run on lab_hybrid (BIG).

Runs lidar / orb / orb_lidar / lidar_orb sequentially (no CPU contention so the
fps/RSS numbers are representative), each through the shared C++ map, and saves
every run's trajectory + occupancy map + verifications. Results are collected
into one folder with an all_results.json the report step reads.

Pipeline per mode (what the loop-table columns mean):
  lidar      : LiDAR scan_to_submap front-end -> proximity proposals -> candidate-local B&B (scan) confirm
  orb        : native windowed VO front-end   -> DBoW appearance proposals -> ORB+PnP confirm
  orb_lidar  : native windowed VO front-end   -> DBoW appearance proposals -> candidate-local B&B (LiDAR) confirm
  lidar_orb  : LiDAR scan_to_submap front-end -> proximity proposals -> ORB+PnP confirm
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from slam_core.fusion2.config import FusionV2Config
from slam_core.fusion2.runner import run_lidar_mode, run_orb_mode_native

DATASET = Path("datasets/lab_hybrid")
OUT = Path("fusion2_outputs") / f"final_4mode_{time.strftime('%Y%m%d_%H%M%S')}"
OUT.mkdir(parents=True, exist_ok=True)

MODES = [
    ("orb", run_orb_mode_native),
    ("orb_lidar", run_orb_mode_native),
    ("lidar", run_lidar_mode),
    ("lidar_orb", run_lidar_mode),
]

results: dict = {}
for mode, fn in MODES:
    print(f"\n{'=' * 60}\n  RUNNING MODE: {mode}\n{'=' * 60}", flush=True)
    t0 = time.perf_counter()
    cfg = FusionV2Config(mode=mode, dataset=DATASET, output_dir=OUT, print_every=3000)
    stats = fn(cfg)
    stats["wall_min"] = round((time.perf_counter() - t0) / 60.0, 1)
    results[mode] = stats
    json.dump(results, open(OUT / "all_results.json", "w"), indent=2, default=str)
    print(f"  {mode} DONE in {stats['wall_min']} min -> {stats.get('run_dir')}", flush=True)

print(f"\nALL DONE. results: {OUT}/all_results.json", flush=True)
