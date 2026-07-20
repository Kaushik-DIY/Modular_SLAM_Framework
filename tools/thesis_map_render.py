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
})

MODE_LABEL = {
    # Consistent "front-end + proposer + verifier" naming across all panels.
    "lidar_s2s_bnb": "Scan-to-submap + Proximity + B&B",
    "lidar_s2s_icp": "Scan-to-submap + Proximity + ICP",
    "lidar_s2m_bnb": "Scan-to-map + Proximity + B&B",
    "lidar_s2m_icp": "Scan-to-map + Proximity + ICP",
    "lidar_orb_s2s": "Scan-to-submap + Proximity + PnP",
    "lidar_orb_s2m": "Scan-to-map + Proximity + PnP",
    "orb_lidar_bnb": "VO + DBoW + B&B",
    "orb_lidar_icp": "VO + DBoW + ICP",
    "orb":           "VO + DBoW + PnP",
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
    Line2D([0], [0], color="#1f77b4", lw=0.9, alpha=0.65, label="Path"),
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


# Uniform per-panel canvas so every mode renders into an IDENTICAL box (same
# pixel size + aspect) regardless of its own map extent. This is what makes the
# 2-per-row montage in the thesis look orderly. AR 2.5 (w/h) is a good fit for
# the wide indoor maps; each map is drawn aspect-equal INSIDE this box, so a map
# that does not match 2.5 is centred with the surrounding area filled by the
# same mid-grey as the "unknown" occupancy value (prob 0.5 -> gray_r 0.5), i.e.
# the letterbox blends into the map background instead of showing white bars.
_PANEL_W, _PANEL_H = 5.6, 2.24          # inches (AR 2.5); one size for ALL panels
_PANEL_AXES = (0.008, 0.012, 0.984, 0.83)   # [l,b,w,h] fig-fraction; rest = title
_UNKNOWN_GREY = (0.5, 0.5, 0.5)


def render(run_dir: Path, map_name: str, combo: str, out_png: Path, dpi: int = 600):
    """Render one individual occupancy-map figure into the shared fixed-size canvas.
    Title is the module CONFIG only (the lab is named in the figure caption), kept
    at a readable size because the canvas is not down-scaled as aggressively as the
    old 9-inch render."""
    prob = np.load(run_dir / "map.npy")
    meta = json.load(open(run_dir / "map_meta.json"))
    ext  = meta["extent"]
    traj = _load_traj(run_dir / "trajectory.tum")

    fig = plt.figure(figsize=(_PANEL_W, _PANEL_H))
    ax = fig.add_axes(_PANEL_AXES)
    ax.set_facecolor(_UNKNOWN_GREY)
    ax.imshow(prob, cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower",
              extent=ext, interpolation="nearest", rasterized=True)
    _draw_traj(ax, traj)
    ax.legend(handles=_LEGEND_HANDLES, loc="upper right", fontsize=9, framealpha=0.7)
    ax.set_title(MODE_LABEL.get(combo, combo), fontsize=18, pad=5)
    ax.set_aspect("equal")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#999"); sp.set_linewidth(0.8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)                       # full canvas (uniform), no tight crop
    fig.savefig(out_png.with_suffix(".pdf"), dpi=dpi)
    plt.close(fig)


def master(root: Path, figdir: Path, map_name: str, dpi: int = 300):
    """3×3 master montage drawn directly from raw map.npy + trajectory.tum.

    Every map is padded onto a single shared world-space canvas (the union of
    all 9 modes' extents, filled with the unknown value 0.5), so the grey
    raster rectangle is *identical in size* across all nine panels. The figure
    is sized to the union aspect ratio so the panels fill their cells uniformly.
    Panel titles show only the mode label; the suptitle carries the dataset name.
    """
    run_dirs = {c: root / map_name / c for c in COMBO_ORDER
                if (root / map_name / c / "map.npy").exists()}
    if not run_dirs:
        return

    # Common canvas = union of all 9 modes' map extents for this dataset.
    x0, x1, y0, y1 = math.inf, -math.inf, math.inf, -math.inf
    res = None
    for rd in run_dirs.values():
        meta = json.load(open(rd / "map_meta.json"))
        ext = meta["extent"]
        x0 = min(x0, ext[0]); x1 = max(x1, ext[1])
        y0 = min(y0, ext[2]); y1 = max(y1, ext[3])
        res = res or float(meta.get("resolution", 0.05))

    union_ext = [x0, x1, y0, y1]
    W = int(round((x1 - x0) / res))
    H = int(round((y1 - y0) / res))
    UNKNOWN = 0.5  # background / unknown occupancy value (renders mid-grey in gray_r)

    def _pad_to_canvas(prob, ext):
        """Paste a map's raster into the shared canvas (origin lower → row 0 = y0)."""
        canvas = np.full((H, W), UNKNOWN, dtype=float)
        h, w = prob.shape
        c0 = int(round((ext[0] - x0) / res))
        r0 = int(round((ext[2] - y0) / res))
        r1, c1 = min(H, r0 + h), min(W, c0 + w)
        if r0 >= 0 and c0 >= 0 and r1 > r0 and c1 > c0:
            canvas[r0:r1, c0:c1] = prob[:r1 - r0, :c1 - c0]
        return canvas

    # Size the figure so each cell matches the union aspect (minimises white bands).
    aspect = (x1 - x0) / (y1 - y0)
    panel_w = 6.4
    panel_h = panel_w / aspect
    title_in, legend_in, hspace = 0.5, 0.6, 0.12
    fig_w = 3 * panel_w + 0.2
    fig_h = panel_h * (3 + 2 * hspace) + title_in + legend_in

    fig, axes = plt.subplots(3, 3, figsize=(fig_w, fig_h))
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
        ax.set_facecolor(str(UNKNOWN))
        ax.imshow(_pad_to_canvas(prob, meta["extent"]), cmap="gray_r",
                  vmin=0.0, vmax=1.0, origin="lower", extent=union_ext,
                  interpolation="nearest", rasterized=True)
        _draw_traj(ax, traj)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#999"); sp.set_linewidth(0.5)
        ax.set_title(COMBO_CONFIG.get(c, MODE_LABEL.get(c, c)), fontsize=11)

    bottom = legend_in / fig_h
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - title_in / fig_h,
                        bottom=bottom, wspace=0.05, hspace=hspace)

    label = MAP_LABEL.get(map_name, map_name)
    fig.suptitle(f"{label} — all 9 modes (final map)", fontsize=16,
                 y=1 - 0.12 * title_in / fig_h)
    fig.legend(handles=_LEGEND_HANDLES, loc="center", ncol=3, fontsize=14,
               markerscale=2.2, handlelength=2.0, columnspacing=2.5,
               borderpad=0.6, framealpha=0.8, bbox_to_anchor=(0.5, bottom * 0.5))

    out = figdir / f"master_{map_name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  master -> {out}")


def montage_2x5(root: Path, figdir: Path, map_name: str, dpi: int = 200):
    """One full-page montage per dataset: nine per-mode maps in a 2-column x
    5-row grid, with the shared Trajectory/Start/End legend occupying the tenth
    (bottom-right) cell so it never overlaps a map.

    Every panel is drawn on ONE shared world-space canvas (the union of all nine
    modes' extents, padded with the unknown value 0.5). Because the axis limits,
    the aspect ratio and the cell size are identical for all panels, each map has
    the SAME outer boundary and the SAME scale/position; a mode that covers less
    area simply shows more of the mid-grey unknown background. This is the paper-
    grade replacement for nine separately-cropped images."""
    run_dirs = {c: root / map_name / c for c in COMBO_ORDER
                if (root / map_name / c / "map.npy").exists()}
    if len(run_dirs) < 9:
        print(f"  montage {map_name}: only {len(run_dirs)}/9 modes present, skipping")
        return

    # Common canvas = the real explored area shared by all nine modes, framed by
    # the union of every mode's TRAJECTORY (full extent, so the most-drifted run
    # still reaches the border and drift stays visually comparable) together with
    # each mode's WALL extent trimmed to its 2nd--98th percentile (which frames
    # the rooms while discarding the free-space scan-ray fans and the speckle a
    # drifting run sprays far outside the map). Using the raw grid union instead
    # would size the canvas to that sprawl and shrink every map into a thin strip.
    MARGIN = 1.5
    res = None
    x0 = y0 = math.inf; x1 = y1 = -math.inf
    for rd in run_dirs.values():
        meta = json.load(open(rd / "map_meta.json"))
        ext = meta["extent"]
        res = res or float(meta.get("resolution", 0.05))
        tr = _load_traj(rd / "trajectory.tum")
        if tr is not None and len(tr):
            x0 = min(x0, tr[:, 0].min()); x1 = max(x1, tr[:, 0].max())
            y0 = min(y0, tr[:, 1].min()); y1 = max(y1, tr[:, 1].max())
        occ = np.argwhere(np.load(rd / "map.npy") > 0.65)
        if len(occ) > 50:
            wx = ext[0] + occ[:, 1] * res; wy = ext[2] + occ[:, 0] * res
            x0 = min(x0, np.percentile(wx, 2)); x1 = max(x1, np.percentile(wx, 98))
            y0 = min(y0, np.percentile(wy, 2)); y1 = max(y1, np.percentile(wy, 98))
    x0 -= MARGIN; x1 += MARGIN; y0 -= MARGIN; y1 += MARGIN
    union_ext = [x0, x1, y0, y1]
    W = int(round((x1 - x0) / res)); H = int(round((y1 - y0) / res))
    UNKNOWN = 0.5

    def _pad(prob, ext):
        # Place a mode's grid on the shared canvas; grids that extend past the
        # (tighter) canvas on any side are cropped on both source and dest.
        canvas = np.full((H, W), UNKNOWN, dtype=float)
        h, w = prob.shape
        c0 = int(round((ext[0] - x0) / res)); r0 = int(round((ext[2] - y0) / res))
        dr0, dc0 = max(0, r0), max(0, c0)
        dr1, dc1 = min(H, r0 + h), min(W, c0 + w)
        if dr1 <= dr0 or dc1 <= dc0:
            return canvas
        canvas[dr0:dr1, dc0:dc1] = prob[dr0 - r0:dr1 - r0, dc0 - c0:dc1 - c0]
        return canvas

    aspect = (x1 - x0) / (y1 - y0)          # panel width / height (equal-aspect map)
    ncol, nrow = 2, 5
    panel_w = 3.5
    panel_h = panel_w / aspect
    title_in = 0.34
    fig_w = ncol * panel_w + 0.15
    fig_h = nrow * (panel_h + title_in) + 0.15

    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h))
    axes_flat = axes.ravel()               # row-major: matches COMBO_ORDER reading order
    for ax in axes_flat:
        ax.axis("off")

    for ax, c in zip(axes_flat, COMBO_ORDER):
        rd = run_dirs[c]
        prob = np.load(rd / "map.npy")
        meta = json.load(open(rd / "map_meta.json"))
        traj = _load_traj(rd / "trajectory.tum")
        ax.axis("on")
        ax.set_facecolor(str(UNKNOWN))
        ax.imshow(_pad(prob, meta["extent"]), cmap="gray_r", vmin=0.0, vmax=1.0,
                  origin="lower", extent=union_ext, interpolation="nearest", rasterized=True)
        _draw_traj(ax, traj)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#999"); sp.set_linewidth(0.6)
        ax.set_title(MODE_LABEL.get(c, c), fontsize=12, pad=4)

    # Tenth cell (bottom-right) = shared legend, on a clean white background.
    lax = axes_flat[9]
    lax.axis("off")
    lax.legend(handles=_LEGEND_HANDLES, loc="center", ncol=1, fontsize=13,
               markerscale=1.8, handlelength=2.0, borderpad=1.0,
               labelspacing=1.0, framealpha=0.9, title="Map legend",
               title_fontsize=13)

    fig.subplots_adjust(left=0.008, right=0.992, top=0.965, bottom=0.008,
                        wspace=0.04, hspace=0.36)
    out = figdir / f"montage_{map_name}.png"
    fig.savefig(out, dpi=dpi)
    fig.savefig(out.with_suffix(".pdf"), dpi=dpi)
    plt.close(fig)
    print(f"  montage -> {out}  (panel {panel_w:.2f}x{panel_h:.2f} in, aspect {aspect:.2f})")


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
