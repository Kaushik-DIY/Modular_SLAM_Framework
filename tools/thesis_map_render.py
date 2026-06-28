"""Publication-clean occupancy-map figures for the thesis (one per run).

Reads each run's map.npy + map_meta.json (extent in metres) + trajectory.tum and
renders a minimal, properly-scaled figure: crisp gray walls, equal aspect, a clear
title (<map> - <mode label>), trajectory with start/end markers, and a legend.
No scale bar, no metric clutter in the figure — those live in the tables.

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
from matplotlib.lines import Line2D

# Paper-grade defaults: embed TrueType fonts (selectable/searchable text in the
# PDF, not type-3 outlines) and keep figures tight.
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
    "s2s_frontend":  "LiDAR scan-to-submap · front-end only (no loops)",
    "s2m_frontend":  "LiDAR scan-to-map · front-end only (no loops)",
    "vo_frontend":   "Visual VO + local BA · front-end only (no loops)",
}

# Standardised config label for the master montages: front-end · proposer · verifier.
# (s2s/s2m/vo) · (proximity/dbow) · (bnb/icp/pnp) — one consistent pattern per panel.
COMBO_CONFIG = {
    "lidar_s2s_bnb": "s2s · proximity · bnb",
    "lidar_s2s_icp": "s2s · proximity · icp",
    "lidar_s2m_bnb": "s2m · proximity · bnb",
    "lidar_s2m_icp": "s2m · proximity · icp",
    "lidar_orb_s2s": "s2s · proximity · pnp",
    "lidar_orb_s2m": "s2m · proximity · pnp",
    "orb_lidar_bnb": "vo · dbow · bnb",
    "orb_lidar_icp": "vo · dbow · icp",
    "orb":           "vo · dbow · pnp",
}

# Standardised short labels used in both individual figure titles and master suptitles.
MAP_LABEL = {
    "lab_hybrid_small":  "lab_1 (single room)",
    "lab_hybrid":        "lab_2 (2 rooms)",
    "lab_hybrid_3":      "lab_3 fast (3 rooms)",
    "lab_hybrid_3_slow": "lab_3 (3 rooms)",
}

COMBO_ORDER = ["lidar_s2s_bnb", "lidar_s2s_icp", "lidar_s2m_bnb", "lidar_s2m_icp",
               "lidar_orb_s2s", "lidar_orb_s2m", "orb_lidar_bnb", "orb_lidar_icp", "orb"]

_LEGEND_HANDLES = [
    Line2D([0], [0], color="#1f77b4", lw=0.9, alpha=0.65, label="Trajectory"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#15a015", ms=6, label="Start"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="red",     ms=6, label="End"),
]


def _load_traj(tf: Path):
    """Parse trajectory.tum -> (N,2) float array, or None."""
    if not tf.exists():
        return None
    pts = []
    for line in open(tf):
        p = line.split()
        if len(p) >= 3:
            pts.append((float(p[1]), float(p[2])))
    return np.array(pts) if pts else None


def _draw_traj(ax, traj):
    """Draw trajectory line + green start + red end on ax."""
    if traj is None or len(traj) == 0:
        return
    ax.plot(traj[:, 0], traj[:, 1], "-", lw=0.7, color="#1f77b4", alpha=0.55, zorder=4)
    ax.scatter(traj[0,  0], traj[0,  1], s=22, c="#15a015", zorder=6)
    ax.scatter(traj[-1, 0], traj[-1, 1], s=22, c="red",     zorder=6)


def render(run_dir: Path, map_name: str, combo: str, out_png: Path, dpi: int = 600):
    """Render one individual occupancy-map figure (with title + legend, no scale bar)."""
    prob = np.load(run_dir / "map.npy")
    meta = json.load(open(run_dir / "map_meta.json"))
    ext  = meta["extent"]
    traj = _load_traj(run_dir / "trajectory.tum")

    span_x, span_y = ext[1] - ext[0], ext[3] - ext[2]
    fig_w = 9.0
    fig_h = max(3.0, fig_w * span_y / max(span_x, 1e-6))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(prob, cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower",
              extent=ext, interpolation="nearest", rasterized=True)
    _draw_traj(ax, traj)
    ax.legend(handles=_LEGEND_HANDLES, loc="upper right", fontsize=8, framealpha=0.7)
    ax.set_title(f"{MAP_LABEL.get(map_name, map_name)}  —  {MODE_LABEL.get(combo, combo)}",
                 fontsize=13, pad=8)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#999"); sp.set_linewidth(0.8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def master(root: Path, figdir: Path, map_name: str, dpi: int = 300):
    """3×3 master montage drawn directly from raw map.npy + trajectory.tum.

    All 9 panels share the same world-space canvas (union of all 9 extents for
    this dataset) so sizes are consistent across modes. Panel titles show only
    the mode label; the suptitle carries the standardised dataset name.
    """
    run_dirs = {c: root / map_name / c for c in COMBO_ORDER
                if (root / map_name / c / "map.npy").exists()}
    if not run_dirs:
        return

    # Compute common canvas = union of all 9 modes' map extents for this dataset.
    x0, x1, y0, y1 = math.inf, -math.inf, math.inf, -math.inf
    for rd in run_dirs.values():
        ext = json.load(open(rd / "map_meta.json"))["extent"]
        x0 = min(x0, ext[0]); x1 = max(x1, ext[1])
        y0 = min(y0, ext[2]); y1 = max(y1, ext[3])

    fig, axes = plt.subplots(3, 3, figsize=(22, 14))
    for ax in axes.ravel():
        ax.axis("off")

    for ax, c in zip(axes.ravel(), COMBO_ORDER):
        rd = run_dirs.get(c)
        if rd is None:
            continue
        prob = np.load(rd / "map.npy")
        meta = json.load(open(rd / "map_meta.json"))
        traj = _load_traj(rd / "trajectory.tum")

        ax.axis("on")
        ax.imshow(prob, cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower",
                  extent=meta["extent"], interpolation="nearest", rasterized=True)
        _draw_traj(ax, traj)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#999"); sp.set_linewidth(0.5)
        ax.set_title(COMBO_CONFIG.get(c, MODE_LABEL.get(c, c)), fontsize=11)

    label = MAP_LABEL.get(map_name, map_name)
    fig.suptitle(f"{label} — all 9 modes (final map)", fontsize=16, y=1.00)
    fig.legend(handles=_LEGEND_HANDLES, loc="lower center", ncol=3,
               fontsize=11, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])

    out = figdir / f"master_{map_name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  master -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=600, help="raster DPI for both individual figures and master montages")
    ap.add_argument("--masters-only", action="store_true",
                    help="skip individual figure renders, regenerate only the master montages")
    a = ap.parse_args()
    figdir = a.root / "figures"
    n = 0
    maps = set()
    if not a.masters_only:
        for run_dir in sorted(a.root.glob("*/*")):
            if not (run_dir / "map.npy").exists():
                continue
            map_name, combo = run_dir.parts[-2], run_dir.parts[-1]
            render(run_dir, map_name, combo, figdir / f"{map_name}__{combo}.png", dpi=a.dpi)
            maps.add(map_name)
            n += 1
            print(f"  rendered {map_name}/{combo}")
    else:
        # collect map names from existing run dirs without re-rendering
        maps = {d.parts[-2] for d in a.root.glob("*/*") if (d / "map.npy").exists()}
    for map_name in sorted(maps):
        master(a.root, figdir, map_name, dpi=a.dpi)
    print(f"\n{n} figures + {len(maps)} master montages -> {figdir}")


if __name__ == "__main__":
    main()
