"""Re-render a saved fusion2 run's map as a fused log-odds occupancy grid (V4.2).

Old runs saved only trajectory.tum (+ scatter occupancy.png); scans are
re-fetched from the dataset by keyframe timestamp and integrated with the same
C++ grid the B&B verifier uses.

Usage: .venv/bin/python tools/rerender_fused_map.py <run_dir> --dataset datasets/lab_hybrid
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fusion_core as fc
from slam_core.fusion2.Dependencies.dataset import LabHybridStream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/lab_hybrid"))
    ap.add_argument("--l-occ", type=float, default=0.85)
    ap.add_argument("--l-free", type=float, default=-0.1)
    args = ap.parse_args()

    traj = np.loadtxt(args.run_dir / "trajectory.tum")
    stream = LabHybridStream(args.dataset, 0.05)

    sigs, poses = [], []
    for row in traj:
        t, x, y = row[0], row[1], row[2]
        qz, qw = row[6], row[7]
        th = 2.0 * math.atan2(qz, qw)
        scan = stream.nearest_scan(t)
        if scan is None:
            continue
        sig = fc.Signature(len(sigs), t, scan_xy=np.asarray(scan, np.float32))
        sigs.append(sig)
        poses.append(fc.Pose2(float(x), float(y), float(th)))

    gc = fc.GridConfig()
    gc.l_occ, gc.l_free = args.l_occ, args.l_free
    grid = fc.assemble_local_grid(sigs, poses, gc)
    prob = np.asarray(grid.probability())
    extent = [grid.origin_x, grid.origin_x + grid.width * grid.resolution,
              grid.origin_y, grid.origin_y + grid.height * grid.resolution]

    np.save(args.run_dir / "map.npy", prob)
    with open(args.run_dir / "map_meta.json", "w") as f:
        json.dump(dict(origin_x=grid.origin_x, origin_y=grid.origin_y,
                       resolution=grid.resolution, width=grid.width,
                       height=grid.height, extent=extent), f, indent=2)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(prob, cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower",
              extent=extent, interpolation="nearest")
    ax.plot(traj[:, 1], traj[:, 2], "-", lw=1.0, color="tab:blue", alpha=0.9)
    ax.scatter(traj[0, 1], traj[0, 2], c="g", s=50, zorder=5, label="start")
    ax.scatter(traj[-1, 1], traj[-1, 2], c="r", s=50, zorder=5, label="end")
    ax.set_title(f"fused occupancy (re-render): {args.run_dir.name} "
                 f"({len(sigs)} keyframes with scans)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.grid(alpha=0.15); ax.legend()
    fig.tight_layout()
    out = args.run_dir / "occupancy_fused.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}  ({grid.width}x{grid.height} cells @ {grid.resolution} m)")


if __name__ == "__main__":
    main()
