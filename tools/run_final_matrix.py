"""V4.5 — final 9-combination validation matrix on lab_hybrid (BIG).

All combinations run sequentially (no CPU contention) through run_fusion's
engine; every run writes trajectory.tum, fused occupancy.png (+map.npy),
verifications.csv, run_summary.json. Results collect into one folder with
all_results.json for the report builder.

| # | mode      | lidar front-end | verifier |
|---|-----------|-----------------|----------|
| 1 | lidar     | native_s2s      | bnb      |
| 2 | lidar     | native_s2s      | icp      |
| 3 | lidar     | native_s2m      | bnb      |
| 4 | lidar     | native_s2m      | icp      |
| 5 | lidar_orb | native_s2s      | pnp      |
| 6 | lidar_orb | native_s2m      | pnp      |
| 7 | orb_lidar | (native VO)     | bnb      |
| 8 | orb_lidar | (native VO)     | icp      |
| 9 | orb       | (native VO)     | pnp      |

IMU is active in every run (extrapolator for 1-6, dropout dead-reckoning 7-9).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from slam_core.fusion2.config import FusionV2Config
from slam_core.fusion2.runner import run_lidar_mode, run_orb_mode_native

DATASET = Path("datasets/lab_hybrid")
OUT = Path("fusion2_outputs") / f"final_matrix_{time.strftime('%Y%m%d_%H%M%S')}"
OUT.mkdir(parents=True, exist_ok=True)

MATRIX = [
    ("lidar_s2s_bnb",  "lidar",     dict(lidar_frontend="native_s2s", scan_verifier="bnb")),
    ("lidar_s2s_icp",  "lidar",     dict(lidar_frontend="native_s2s", scan_verifier="icp")),
    ("lidar_s2m_bnb",  "lidar",     dict(lidar_frontend="native_s2m", scan_verifier="bnb")),
    ("lidar_s2m_icp",  "lidar",     dict(lidar_frontend="native_s2m", scan_verifier="icp")),
    ("lidar_orb_s2s",  "lidar_orb", dict(lidar_frontend="native_s2s")),
    ("lidar_orb_s2m",  "lidar_orb", dict(lidar_frontend="native_s2m")),
    ("orb_lidar_bnb",  "orb_lidar", dict(scan_verifier="bnb")),
    ("orb_lidar_icp",  "orb_lidar", dict(scan_verifier="icp")),
    ("orb_pnp",        "orb",       dict()),
]

results: dict = {}
for name, mode, overrides in MATRIX:
    print(f"\n{'=' * 64}\n  [{name}] mode={mode} {overrides}\n{'=' * 64}", flush=True)
    t0 = time.perf_counter()
    cfg = FusionV2Config(mode=mode, dataset=DATASET, output_dir=OUT,
                         print_every=3000, **overrides)
    try:
        stats = (run_lidar_mode(cfg) if mode in ("lidar", "lidar_orb")
                 else run_orb_mode_native(cfg))
        stats["combo"] = name
        stats["wall_min"] = round((time.perf_counter() - t0) / 60.0, 1)
    except Exception as e:  # record the failure, keep the matrix going
        import traceback
        stats = dict(combo=name, mode=mode, error=str(e),
                     traceback=traceback.format_exc())
        print(f"  [{name}] FAILED: {e}", flush=True)
    results[name] = stats
    json.dump(results, open(OUT / "all_results.json", "w"), indent=2, default=str)
    print(f"  [{name}] done in {stats.get('wall_min', '?')} min "
          f"loops={stats.get('loops_accepted', '?')} -> {stats.get('run_dir')}",
          flush=True)

print(f"\nMATRIX COMPLETE -> {OUT}/all_results.json", flush=True)
