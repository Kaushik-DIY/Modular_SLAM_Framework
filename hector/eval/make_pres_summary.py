#!/usr/bin/env python3
"""
make_pres_summary.py
====================
Presentation summary figure for a Hector SLAM run (lab dataset), mirroring the
visual-SLAM ``pres_summary.png`` layout (2x3 grid):

  Row 1 :  Occupancy grid map (+ trajectory) | Trajectory (top-down) | Run statistics
  Row 2 :  Match score over time            | Map growth             | Per-scan motion

Reuses the incremental map builder from ``render_map_build_video`` so the map
matches the video exactly — no need to re-run SLAM.

Usage
-----
    .venv/bin/python -m hector.eval.make_pres_summary
    .venv/bin/python -m hector.eval.make_pres_summary \\
        --traj hector_outputs/trajectory_lab_run_2_raw_scan_to_map_1052.txt \\
        --out  hector_outputs/video/pres_summary.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from slam_core.matching.scan_to_map import GridMap, _transform_points
from hector.eval.render_map_build_video import (
    _load_trajectory, _load_scan_points, _auto_traj,
)


# ---------------------------------------------------------------------------
# Map building
# ---------------------------------------------------------------------------

def _build_map(poses, pts_list, *, map_res, map_size_m, l_min, l_max,
               l_free, l_occ, ray_steps):
    """Integrate all scans; return (grid, occupied_cell_count_per_scan)."""
    grid = GridMap(res=map_res, size_m=map_size_m, l_min=l_min, l_max=l_max)
    occ_counts = np.zeros(len(poses), dtype=np.int64)
    for k in range(len(poses)):
        pts = pts_list[k]
        if pts.shape[0] > 0:
            pose_arr = poses[k]
            pts_world = _transform_points(pose_arr, pts)
            grid.integrate_scan_simple(
                pose=pose_arr, pts_world=pts_world,
                l_free=l_free, l_occ=l_occ, ray_steps=ray_steps,
            )
        occ_counts[k] = int(np.count_nonzero(grid.logodds > 0.0))
    return grid, occ_counts


def _grid_image(grid: GridMap, crop):
    """Return (val_image_lower_origin, extent_world) cropped to bbox."""
    lo = grid.logodds.astype(np.float32)
    val = np.zeros(lo.shape, dtype=np.float32)
    free = lo < 0.0
    occ = lo > 0.0
    val[free] = 45.0 + 45.0 * np.clip(lo[free] / grid.l_min, 0.0, 1.0)
    val[occ] = 150.0 + 105.0 * np.clip(lo[occ] / grid.l_max, 0.0, 1.0)

    gx_lo, gx_hi, gy_lo, gy_hi = crop
    sub = val[gy_lo:gy_hi, gx_lo:gx_hi]
    ox, oy = grid.origin
    extent = [
        (gx_lo - ox) * grid.res, (gx_hi - ox) * grid.res,
        (gy_lo - oy) * grid.res, (gy_hi - oy) * grid.res,
    ]
    return sub, extent


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _colorline(ax, xy, cmap="viridis", lw=1.6):
    pts = xy.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, linewidth=lw)
    lc.set_array(np.linspace(0, 1, len(segs)))
    ax.add_collection(lc)
    return lc


def make_summary(traj_path, out_path, *, scan_variant, map_res, map_size_m,
                 l_min, l_max, l_free, l_occ, ray_steps):
    print(f"[summary] trajectory: {traj_path}")
    stamps, poses = _load_trajectory(traj_path)
    N = len(poses)
    scores = _load_scores(traj_path)
    print(f"[summary] {N} poses; loading scans ...")
    _, pts_list = _load_scan_points(stamps, scan_variant=scan_variant)

    print("[summary] building occupancy map ...")
    grid, occ_counts = _build_map(
        poses, pts_list, map_res=map_res, map_size_m=map_size_m,
        l_min=l_min, l_max=l_max, l_free=l_free, l_occ=l_occ, ray_steps=ray_steps,
    )

    xy = poses[:, :2]
    t = stamps - stamps[0]
    # crop region (grid coords) to trajectory bbox + 3 m padding
    g = grid.world_to_grid(xy)
    pad = int(3.0 / map_res)
    gx_lo = max(0, int(g[:, 0].min()) - pad)
    gx_hi = min(grid.size, int(g[:, 0].max()) + pad + 1)
    gy_lo = max(0, int(g[:, 1].min()) - pad)
    gy_hi = min(grid.size, int(g[:, 1].max()) + pad + 1)
    img, extent = _grid_image(grid, (gx_lo, gx_hi, gy_lo, gy_hi))

    # derived stats
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path_len = float(seg.sum())
    start_end = float(np.linalg.norm(xy[-1] - xy[0]))
    x_ext = float(xy[:, 0].max() - xy[:, 0].min())
    y_ext = float(xy[:, 1].max() - xy[:, 1].min())
    dtheta = np.abs(np.diff(np.unwrap(poses[:, 2])))
    valid_score = scores[scores >= 0]
    duration = float(t[-1])
    rate = (N - 1) / duration if duration > 0 else float("nan")

    # ----- figure -----
    plt.style.use("default")
    fig = plt.figure(figsize=(15, 9.5), dpi=150)
    fig.suptitle(
        "Hector SLAM - Lab Room Run  (scan-to-map front-end, no PGO)\n"
        f"Carmen 2D LiDAR  |  {N} scans  |  occupancy grid @ {map_res*100:.0f} cm",
        fontsize=12, fontweight="bold",
    )

    # (1) occupancy map + trajectory
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(img, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=255,
               interpolation="nearest")
    ax1.plot(xy[:, 0], xy[:, 1], color="#3fa9f5", lw=1.1, alpha=0.9)
    ax1.scatter(xy[0, 0], xy[0, 1], s=55, marker="o", color="#27ae60",
                zorder=6, label="Start", edgecolors="white", linewidths=0.6)
    ax1.scatter(xy[-1, 0], xy[-1, 1], s=70, marker="X", color="#e67e22",
                zorder=6, label="End", edgecolors="white", linewidths=0.6)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("X [m]", fontsize=8)
    ax1.set_ylabel("Y [m]", fontsize=8)
    ax1.set_title("Occupancy grid map + trajectory", fontsize=9)
    ax1.legend(fontsize=7, loc="upper right")
    ax1.tick_params(labelsize=7)

    # (2) trajectory top-down (time-coloured)
    ax2 = fig.add_subplot(2, 3, 2)
    lc = _colorline(ax2, xy, cmap="viridis", lw=1.8)
    ax2.scatter(xy[0, 0], xy[0, 1], s=55, marker="o", color="#27ae60",
                zorder=6, edgecolors="white", linewidths=0.6)
    ax2.scatter(xy[-1, 0], xy[-1, 1], s=70, marker="X", color="#c0392b",
                zorder=6, edgecolors="white", linewidths=0.6)
    cbar = fig.colorbar(lc, ax=ax2, fraction=0.046, pad=0.02)
    cbar.set_label("scan progression", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax2.set_aspect("equal", adjustable="box")
    ax2.autoscale_view()
    ax2.set_xlabel("X [m]", fontsize=8)
    ax2.set_ylabel("Y [m]", fontsize=8)
    ax2.set_title("Trajectory (top-down)", fontsize=9)
    ax2.grid(True, alpha=0.25)
    ax2.tick_params(labelsize=7)

    # (3) stats table
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.axis("off")
    rows = [
        ["Total scans", str(N)],
        ["Duration [s]", f"{duration:.1f}"],
        ["Scan rate [Hz]", f"{rate:.2f}"],
        ["Trajectory length [m]", f"{path_len:.2f}"],
        ["Start->End drift [m]", f"{start_end:.3f}"],
        ["Room extent X [m]", f"{x_ext:.2f}"],
        ["Room extent Y [m]", f"{y_ext:.2f}"],
        ["Map resolution [m]", f"{map_res:.3f}"],
        ["Occupied cells (final)", str(int(occ_counts[-1]))],
        ["Mean match score", f"{valid_score.mean():.3f}" if valid_score.size else "n/a"],
        ["Min match score", f"{valid_score.min():.3f}" if valid_score.size else "n/a"],
    ]
    table = ax3.table(cellText=rows, colLabels=["Metric", "Value"],
                      loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.12, 1.5)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#34495e")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f2f4f6")
    ax3.set_title("Run statistics", fontsize=9)

    # (4) match score over time
    ax4 = fig.add_subplot(2, 3, 4)
    sc_plot = np.where(scores < 0, np.nan, scores)
    ax4.plot(t, sc_plot, lw=0.7, color="#2980b9", alpha=0.85)
    if valid_score.size:
        ax4.axhline(valid_score.mean(), color="#c0392b", ls="--", lw=1.0,
                    alpha=0.8, label=f"mean = {valid_score.mean():.3f}")
        ax4.legend(fontsize=7)
    ax4.set_xlabel("Time [s]", fontsize=8)
    ax4.set_ylabel("Match score", fontsize=8)
    ax4.set_title("Scan-match quality", fontsize=9)
    ax4.grid(True, alpha=0.25)
    ax4.tick_params(labelsize=7)

    # (5) map growth
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t, occ_counts, lw=1.0, color="#8e44ad", alpha=0.9)
    ax5.fill_between(t, occ_counts, color="#8e44ad", alpha=0.12)
    ax5.set_xlabel("Time [s]", fontsize=8)
    ax5.set_ylabel("Occupied cells", fontsize=8)
    ax5.set_title("Map growth", fontsize=9)
    ax5.grid(True, alpha=0.25)
    ax5.tick_params(labelsize=7)

    # (6) per-scan motion (translation + rotation increments)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(t[1:], seg * 100.0, lw=0.7, color="#16a085", alpha=0.85,
             label="translation [cm]")
    ax6b = ax6.twinx()
    ax6b.plot(t[1:], np.degrees(dtheta), lw=0.7, color="#e67e22", alpha=0.7,
              label="rotation [deg]")
    ax6.set_xlabel("Time [s]", fontsize=8)
    ax6.set_ylabel("Translation / scan [cm]", fontsize=8, color="#16a085")
    ax6b.set_ylabel("Rotation / scan [deg]", fontsize=8, color="#e67e22")
    ax6.tick_params(axis="y", labelcolor="#16a085", labelsize=7)
    ax6b.tick_params(axis="y", labelcolor="#e67e22", labelsize=7)
    ax6.tick_params(axis="x", labelsize=7)
    ax6.set_title("Per-scan motion", fontsize=9)
    ax6.grid(True, alpha=0.25)
    l1, lab1 = ax6.get_legend_handles_labels()
    l2, lab2 = ax6b.get_legend_handles_labels()
    ax6.legend(l1 + l2, lab1 + lab2, fontsize=7, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[summary] saved: {out_path}")


def _load_scores(path) -> np.ndarray:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    out.append(float(parts[4]))
                except ValueError:
                    out.append(-1.0)
            else:
                out.append(-1.0)
    return np.array(out, dtype=float)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Hector SLAM presentation summary figure.")
    ap.add_argument("--traj", default=None)
    ap.add_argument("--hector-out", default="hector_outputs")
    ap.add_argument("--variant", default="raw", choices=["raw", "360"])
    ap.add_argument("--out", default="hector_outputs/video/pres_summary.png")
    ap.add_argument("--map-res", type=float, default=0.05, dest="map_res")
    ap.add_argument("--map-size", type=float, default=40.0, dest="map_size")
    args = ap.parse_args()

    traj = args.traj or _auto_traj(args.hector_out, args.variant)
    if traj is None:
        print(f"[summary] ERROR: no trajectory in {args.hector_out}/", file=sys.stderr)
        return 1

    make_summary(
        traj, args.out, scan_variant=args.variant,
        map_res=args.map_res, map_size_m=args.map_size,
        l_min=-5.0, l_max=5.0, l_free=-0.1, l_occ=1.0, ray_steps=20,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
