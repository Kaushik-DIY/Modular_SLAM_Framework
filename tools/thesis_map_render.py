"""Publication-clean occupancy-map figures for the thesis (one per run).

Reads each run's map.npy + map_meta.json (extent in metres) + trajectory.tum and
renders a minimal, properly-scaled figure: crisp gray walls, equal aspect, a metric
scale bar, a clear title (<map> - <mode label>), and a thin faint trajectory. No
metric clutter in the figure — those live in the tables.

    .venv/bin/python tools/thesis_map_render.py --root thesis_outputs/thesis_<UTC>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Paper-grade defaults: embed TrueType fonts (selectable/searchable text in the
# PDF, not type-3 outlines) and keep figures tight. Visual style is unchanged.
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})

MODE_LABEL = {
    "lidar_s2s_bnb": "LiDAR · scan-to-submap · B&B",
    "lidar_s2s_icp": "LiDAR · scan-to-submap · ICP",
    "lidar_s2m_bnb": "LiDAR · scan-to-map · B&B",
    "lidar_s2m_icp": "LiDAR · scan-to-map · ICP",
    "lidar_orb_s2s": "LiDAR + visual-PnP · scan-to-submap",
    "lidar_orb_s2m": "LiDAR + visual-PnP · scan-to-map",
    "orb_lidar_bnb": "Visual VO + LiDAR-B&B verify",
    "orb_lidar_icp": "Visual VO + LiDAR-ICP verify",
    "orb":           "Visual VO + visual-PnP",
}
MAP_LABEL = {"lab_hybrid": "Two-room lab", "lab_hybrid_small": "Single-room lab"}


def _scale_bar(ax, extent, metres=5.0):
    """A clean metric scale bar at the lower-left."""
    x0, x1, y0, y1 = extent
    w, h = x1 - x0, y1 - y0
    bx = x0 + 0.06 * w
    by = y0 + 0.08 * h
    ax.add_patch(Rectangle((bx, by), metres, 0.012 * h, facecolor="black",
                           edgecolor="black", zorder=10))
    ax.text(bx + metres / 2, by + 0.03 * h, f"{metres:g} m", ha="center",
            va="bottom", fontsize=11, zorder=10)


def render(run_dir: Path, map_name: str, combo: str, out_png: Path, dpi: int = 600):
    prob = np.load(run_dir / "map.npy")
    meta = json.load(open(run_dir / "map_meta.json"))
    ext = meta["extent"]
    traj = []
    tf = run_dir / "trajectory.tum"
    if tf.exists():
        for line in open(tf):
            p = line.split()
            if len(p) >= 3:
                traj.append((float(p[1]), float(p[2])))
    traj = np.array(traj) if traj else None

    span_x, span_y = ext[1] - ext[0], ext[3] - ext[2]
    fig_w = 9.0
    fig_h = max(3.0, fig_w * span_y / max(span_x, 1e-6))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(prob, cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower",
              extent=ext, interpolation="nearest", rasterized=True)
    if traj is not None and len(traj):
        ax.plot(traj[:, 0], traj[:, 1], "-", lw=0.7, color="#1f77b4", alpha=0.55, zorder=4)
        ax.scatter(traj[0, 0], traj[0, 1], s=22, c="#15a015", zorder=6)   # start
    _scale_bar(ax, ext)
    ax.set_title(f"{MAP_LABEL.get(map_name, map_name)}  —  {MODE_LABEL.get(combo, combo)}",
                 fontsize=13, pad=8)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#999"); sp.set_linewidth(0.8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # Paper-grade output: a vector PDF (axes/title/scale-bar stay vector; the
    # occupancy grid is the only rasterized layer, embedded at `dpi`) plus a
    # high-DPI PNG for quick viewing. Style unchanged from the original figure.
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


COMBO_ORDER = ["lidar_s2s_bnb", "lidar_s2s_icp", "lidar_s2m_bnb", "lidar_s2m_icp",
               "lidar_orb_s2s", "lidar_orb_s2m", "orb_lidar_bnb", "orb_lidar_icp", "orb"]


def master(figdir: Path, map_name: str, dpi: int = 300):
    """One 3x3 master image of all 9 mode maps for a map (easy side-by-side)."""
    import matplotlib.image as mpimg
    figs = [(c, figdir / f"{map_name}__{c}.png") for c in COMBO_ORDER]
    figs = [(c, p) for c, p in figs if p.exists()]
    if not figs:
        return
    fig, axes = plt.subplots(3, 3, figsize=(22, 12))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (c, p) in zip(axes.ravel(), figs):
        ax.imshow(mpimg.imread(p), interpolation="antialiased")
        ax.set_title(MODE_LABEL.get(c, c), fontsize=12)
    fig.suptitle(f"{MAP_LABEL.get(map_name, map_name)}  —  all 9 modes "
                 f"(fused occupancy)", fontsize=17, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = figdir / f"master_{map_name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  master -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=600, help="paper-grade raster DPI")
    a = ap.parse_args()
    figdir = a.root / "figures"
    n = 0
    maps = set()
    for run_dir in sorted(a.root.glob("*/*")):
        if not (run_dir / "map.npy").exists():
            continue
        map_name, combo = run_dir.parts[-2], run_dir.parts[-1]
        render(run_dir, map_name, combo, figdir / f"{map_name}__{combo}.png", dpi=a.dpi)
        maps.add(map_name)
        n += 1
        print(f"  rendered {map_name}/{combo}")
    for map_name in sorted(maps):
        master(figdir, map_name)
    print(f"\n{n} figures + {len(maps)} master montages -> {figdir}")


if __name__ == "__main__":
    main()
