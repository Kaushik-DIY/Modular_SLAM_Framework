"""
Three-way occupancy-map comparison for the lab dataset:
  1. scan_to_map               (baseline reference)
  2. scan_to_submap (no PGO)   (front-end only — expected to drift)
  3. scan_to_submap + g2o PGO  (target: match/beat scan_to_map)

All three trajectories are integrated into the SAME global occupancy-grid
representation (scan_to_map-style), so the rendered differences reflect ONLY
trajectory quality, not the map data structure.

Run:
  .venv/bin/python -m hector.eval.compare_three_maps \
      --traj_map      hector_outputs/trajectory_lab_run_2_raw_scan_to_map_1052.txt \
      --traj_nopgo    baseline_snapshots/unified_nopgo_v2_1052.txt \
      --traj_pgo      hector_outputs/trajectory_lab_run_2_raw_scan_to_submap_1052_pgo.txt \
      --dataset lab_run_2 --scan-variant raw \
      --out hector_outputs/three_way_comparison.png
"""
from __future__ import annotations

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hector.config as cfg
from hector.eval._generic_eval_common import (
    configure_dataset,
    load_aligned_scan_points,
)
from hector.eval.rebuild_map_any import build_map, _grid_prob, _grid_xy


def _load_traj(path: str, min_score: float = -1e9):
    stamps, poses = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 5:
                continue
            try:
                t, x, y, th, sc = map(float, p[:5])
            except ValueError:
                continue
            if sc >= min_score:
                stamps.append(t)
                poses.append([x, y, th])
    return np.array(stamps, dtype=float), np.array(poses, dtype=float)


def _build_global(dataset, variant, path, map_res, map_size_m, ray_steps):
    stamps, poses = _load_traj(path)
    _, pts_list = load_aligned_scan_points(
        dataset_name=dataset, scan_variant=variant, stamps=stamps, voxel_filter=None
    )
    # Always integrate into a single global grid (scan_to_map style) for fairness.
    grid, traj_xy = build_map(
        matcher_type="scan_to_map",
        poses_xyt=poses,
        pts_list=pts_list,
        map_res=map_res,
        map_size_m=map_size_m,
        ray_steps=ray_steps,
        l_free=cfg.L_FREE,
        l_occ=cfg.L_OCC,
        l_min=cfg.L_MIN,
        l_max=cfg.L_MAX,
    )
    return grid, traj_xy, len(poses)


def _occupied_fraction(prob, thresh=0.65):
    return float(np.mean(prob >= thresh))


def _panel(ax, grid, traj_xy, title):
    prob = _grid_prob(grid)
    # _grid_xy returns the trajectory with a flipped row (size-1-gy), so the map
    # must be flipud'd under origin="upper" to align (same convention as
    # rebuild_map_any._render_panel). Without the flip the trajectory is mirrored
    # vertically relative to the walls.
    ax.imshow(np.flipud(prob), cmap="gray_r", origin="upper", vmin=0.0, vmax=1.0)
    gx, gy = _grid_xy(grid, traj_xy)
    if len(gx):
        ax.plot(gx, gy, "-", color="cyan", lw=0.8, alpha=0.9)
        ax.plot(gx[0], gy[0], "o", color="lime", ms=8, label="start")
        ax.plot(gx[-1], gy[-1], "X", color="red", ms=10, label="end")
    sharp = _occupied_fraction(prob)
    ax.set_title(f"{title}\nocc>=0.65 frac={sharp:.4f}", fontsize=10)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_map", required=True)
    ap.add_argument("--traj_nopgo", required=True)
    ap.add_argument("--traj_pgo", required=True)
    ap.add_argument("--dataset", default="lab_run_2")
    ap.add_argument("--scan-variant", dest="scan_variant", default="raw")
    ap.add_argument("--map_res", type=float, default=None)
    ap.add_argument("--map_size_m", type=float, default=40.0)
    ap.add_argument("--ray_steps", type=int, default=None)
    ap.add_argument("--out", default="hector_outputs/three_way_comparison.png")
    args = ap.parse_args()

    configure_dataset(args.dataset)
    map_res = args.map_res if args.map_res is not None else cfg.MAP_RESOLUTION
    ray_steps = args.ray_steps if args.ray_steps is not None else cfg.RAY_STEPS

    specs = [
        ("scan_to_map (baseline)", args.traj_map),
        ("scan_to_submap (no PGO)", args.traj_nopgo),
        ("scan_to_submap + g2o PGO", args.traj_pgo),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, (title, path) in zip(axes, specs):
        grid, traj_xy, n = _build_global(
            args.dataset, args.scan_variant, path, map_res, args.map_size_m, ray_steps
        )
        sharp = _occupied_fraction(_grid_prob(grid))
        print(f"[compare] {title:30s} poses={n:5d}  occ>=0.65 frac={sharp:.4f}  ({path})")
        _panel(ax, grid, traj_xy, title)

    fig.suptitle("Lab Room — Map Quality: scan_to_map vs scan_to_submap (no PGO) vs +g2o PGO", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[compare] Wrote {args.out}")


if __name__ == "__main__":
    main()
