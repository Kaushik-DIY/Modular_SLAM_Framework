"""Thesis figures for the real-time module-switching capability.

Reads one switching-demo run dir (produced by run_realtime.py with
--switch-schedule) and renders, as SEPARATE files (so each is usable on its own):

  <base>_map            — final occupancy map, trajectory coloured per active-config
                          segment, numbered markers at each switch, legends.
  <base>_displacement   — keyframe-to-keyframe displacement (no-teleport check).
  <base>_tracking       — front-end tracking quality per segment.
  <base>_loops          — loop closure events: proposed vs accepted.

With --combined it also emits <base> (the stacked 4-panel composite).

    .venv/bin/python tools/thesis_switch_demo.py --run <run_dir> --out <basepath>

Inputs in the run dir: map.npy, map_meta.json, trajectory.tum, switches.csv,
timeline.csv, verifications.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

# Distinct colours for the active-config segments (cycled).
SEG_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
_FE_SHORT = {"native_s2s": "s2s", "native_s2m": "s2m", "visual_vo": "VO"}


def _read_csv(path: Path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _load_traj(tf: Path):
    """trajectory.tum -> (N,2) final corrected positions (file order = kf order)."""
    if not tf.exists():
        return None
    pts = [(float(p[1]), float(p[2])) for line in tf.open()
           if len(p := line.split()) >= 3]
    return np.array(pts) if pts else None


def _seg_label(tl_row) -> str:
    return (f"{_FE_SHORT.get(tl_row['fe'], tl_row['fe'])} · "
            f"{tl_row['proposer']} · {tl_row['verifier']}")


def _switch_label(sw) -> str:
    frm = _FE_SHORT.get(sw["frm"], sw["frm"])
    to = _FE_SHORT.get(sw["to"], sw["to"])
    base = f"{frm}→{to}"
    if str(sw.get("autofallback", "")).lower() in ("true", "1"):
        base += "  [auto-fallback]"
    return base


class _Run:
    """Parsed switching-demo run, with positions resolved on the FINAL trajectory."""

    def __init__(self, run_dir: Path):
        rd = Path(run_dir)
        self.summary = json.load(open(rd / "run_summary.json"))
        self.switches = _read_csv(rd / "switches.csv")
        self.timeline = _read_csv(rd / "timeline.csv")
        self.verifs = _read_csv(rd / "verifications.csv")
        self.prob = np.load(rd / "map.npy")
        self.ext = json.load(open(rd / "map_meta.json"))["extent"]
        if not self.timeline:
            raise SystemExit(f"{rd} has no timeline.csv — run with --switch-schedule.")

        self.kf = np.array([int(r["kf"]) for r in self.timeline])
        self.qual = np.array([float(r["quality"]) for r in self.timeline])
        self.sensor = [r["sensor"] for r in self.timeline]
        tlx = np.array([float(r["x"]) for r in self.timeline])
        tly = np.array([float(r["y"]) for r in self.timeline])

        # Prefer FINAL optimized poses (trajectory.tum, file order = kf order).
        # Fall back to live timeline poses only if the counts disagree.
        traj = _load_traj(rd / "trajectory.tum")
        if traj is not None and len(traj) == len(self.timeline):
            self.px, self.py = traj[:, 0], traj[:, 1]
        else:
            self.px, self.py = tlx, tly

        # kf -> row index, so a switch keyframe maps to its FINAL-trajectory point
        # (fixes markers floating off the line after loop correction).
        self._row = {int(k): i for i, k in enumerate(self.kf)}
        self.sw_kf = [int(s["kf"]) for s in self.switches]
        self.bounds = [int(self.kf[0])] + self.sw_kf + [int(self.kf[-1]) + 1]
        self.seg_rows = [self.timeline[min(int(np.searchsorted(self.kf, b)),
                                           len(self.timeline) - 1)]
                         for b in self.bounds[:-1]]

    def switch_xy(self, sw):
        """Final-trajectory (x, y) of a switch's keyframe."""
        i = self._row.get(int(sw["kf"]))
        if i is None:                       # nearest by kf if exact row missing
            i = int(np.argmin(np.abs(self.kf - int(sw["kf"]))))
        return self.px[i], self.py[i]

    def switch_lines(self, ax, shade=True):
        if shade:
            for si in range(len(self.bounds) - 1):
                ax.axvspan(self.bounds[si], self.bounds[si + 1],
                           color=SEG_COLORS[si % len(SEG_COLORS)], alpha=0.06, zorder=0)
        for n, k in enumerate(self.sw_kf, 1):
            ax.axvline(k, color="black", ls="--", lw=0.8, alpha=0.6)
            ax.annotate(str(n), (k, 1.0), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=8, fontweight="bold")


# ---- individual panels (each draws onto a given Axes) ----------------------

def draw_map(R: _Run, ax):
    ax.imshow(R.prob, cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower",
              extent=R.ext, interpolation="nearest", rasterized=True)
    seg_handles = []
    for si in range(len(R.bounds) - 1):
        lo, hi = R.bounds[si], R.bounds[si + 1]
        m = (R.kf >= lo) & (R.kf < hi)
        col = SEG_COLORS[si % len(SEG_COLORS)]
        ax.plot(R.px[m], R.py[m], "-", lw=1.4, color=col, alpha=0.9, zorder=4)
        seg_handles.append(Line2D([0], [0], color=col, lw=2.4,
                                  label=f"{si+1}. {_seg_label(R.seg_rows[si])}"))
    for n, s in enumerate(R.switches, 1):
        sx, sy = R.switch_xy(s)
        ax.scatter(sx, sy, s=150, marker="D", facecolor="yellow",
                   edgecolor="black", lw=1.2, zorder=8)
        ax.annotate(str(n), (sx, sy), color="black", fontsize=9, fontweight="bold",
                    ha="center", va="center", zorder=9)
    ax.scatter(R.px[0], R.py[0], s=60, c="#15a015", zorder=7)
    ax.scatter(R.px[-1], R.py[-1], s=60, c="red", zorder=7)
    sw_leg = [Line2D([0], [0], marker="D", color="w", markerfacecolor="yellow",
                     markeredgecolor="black", ms=10, label=f"{n}: {_switch_label(s)}")
              for n, s in enumerate(R.switches, 1)]
    se = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#15a015", ms=8, label="Start"),
          Line2D([0], [0], marker="o", color="w", markerfacecolor="red", ms=8, label="End")]
    leg1 = ax.legend(handles=seg_handles, loc="upper left", fontsize=8,
                     framealpha=0.85, title="Active config per segment")
    ax.add_artist(leg1)
    ax.legend(handles=sw_leg + se, loc="lower right", fontsize=8,
              framealpha=0.85, title="Switch points")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#999"); sp.set_linewidth(0.8)
    ax.set_title(f"Real-time module switching — {R.summary.get('mode','')} demo "
                 f"({len(R.timeline)} kf, {R.summary.get('loops_accepted',0)} loops, "
                 f"{len(R.switches)} switches)", fontsize=13, pad=8)


def draw_displacement(R: _Run, ax):
    step = np.hypot(np.diff(R.px), np.diff(R.py))
    ax.plot(R.kf[1:], step, "-", lw=0.9, color="#1f77b4")
    ax.set_ylabel("KF step [m]"); ax.set_xlabel("keyframe")
    ax.set_title("keyframe-to-keyframe displacement", fontsize=12, pad=16)
    ax.grid(alpha=0.2)
    R.switch_lines(ax)


def draw_tracking(R: _Run, ax):
    vo_m = np.array([s == "vo" for s in R.sensor]); li_m = ~vo_m
    axb = ax.twinx()
    axb.set_zorder(ax.get_zorder() - 1); ax.patch.set_visible(False)
    if vo_m.any():
        ax.plot(R.kf[vo_m], R.qual[vo_m], ".", ms=3, color="#9467bd")
    if li_m.any():
        axb.plot(R.kf[li_m], R.qual[li_m], ".", ms=3, color="#2ca02c")
    ax.set_ylabel("VO inliers", color="#9467bd")
    axb.set_ylabel("LiDAR match score", color="#2ca02c")
    ax.tick_params(axis="y", colors="#9467bd"); axb.tick_params(axis="y", colors="#2ca02c")
    ax.set_xlabel("keyframe")
    ax.set_title("Front-end tracking quality per segment", fontsize=12, pad=16)
    ax.grid(alpha=0.2)
    R.switch_lines(ax)


def draw_loops(R: _Run, ax):
    prop = sorted(int(v["query"]) for v in R.verifs)
    acc = sorted(int(v["query"]) for v in R.verifs
                 if str(v.get("accepted", "")).lower() in ("true", "1"))
    if prop:
        ax.step(prop, np.arange(1, len(prop) + 1), where="post",
                color="#4c72b0", lw=1.4, label=f"proposed ({len(prop)})")
    if acc:
        ax.step(acc, np.arange(1, len(acc) + 1), where="post",
                color="#d62728", lw=1.6, label=f"accepted ({len(acc)})")
    ax.set_ylabel("count (cumulative)"); ax.set_xlabel("keyframe")
    ax.set_title("loop closure events", fontsize=12, pad=16)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    R.switch_lines(ax)


def _save(fig, base: Path, dpi, tight=True):
    base.parent.mkdir(parents=True, exist_ok=True)
    bbox = "tight" if tight else None
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches=bbox)
    fig.savefig(base.with_suffix(".pdf"), dpi=dpi, bbox_inches=bbox)
    plt.close(fig)


def render(run_dir: Path, out_base: Path, dpi: int = 600, combined: bool = False):
    R = _Run(run_dir)
    out_base = Path(out_base)

    # separate files
    fig, ax = plt.subplots(figsize=(11, 7)); draw_map(R, ax)
    _save(fig, Path(f"{out_base}_map"), dpi)
    # Three stacked panels: identical figure size AND an identical explicit axes
    # rectangle so the keyframe x-axis lines up column-for-column across all three
    # (the middle panel's twin y-axis for the LiDAR score would otherwise squeeze
    # its data box under bbox="tight"). The right margin is reserved in EVERY panel
    # so the data box width is the same whether or not a panel draws a twin axis.
    PANEL_RECT = (0.075, 0.20, 0.84, 0.64)   # [left, bottom, width, height] fig-fraction
    for name, fn in [("displacement", draw_displacement),
                     ("tracking", draw_tracking),
                     ("loops", draw_loops)]:
        fig = plt.figure(figsize=(11, 3.0))
        ax = fig.add_axes(PANEL_RECT)
        fn(R, ax)
        _save(fig, Path(f"{out_base}_{name}"), dpi, tight=False)

    if combined:
        fig = plt.figure(figsize=(13, 15))
        gs = fig.add_gridspec(4, 1, height_ratios=[2.6, 1.0, 1.0, 1.0], hspace=0.30)
        draw_map(R, fig.add_subplot(gs[0]))
        draw_displacement(R, fig.add_subplot(gs[1]))
        draw_tracking(R, fig.add_subplot(gs[2]))
        draw_loops(R, fig.add_subplot(gs[3]))
        _save(fig, out_base, dpi)

    nprop = len(R.verifs)
    nacc = sum(1 for v in R.verifs if str(v.get("accepted", "")).lower() in ("true", "1"))
    print(f"  {out_base.name}: {len(R.switches)} switches, "
          f"{nprop} proposed / {nacc} accepted -> _map/_displacement/_tracking/_loops"
          f"{' (+combined)' if combined else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="switching-demo run dir")
    ap.add_argument("--out", type=Path, required=True, help="output figure basepath (no ext)")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--combined", action="store_true", help="also emit the stacked 4-panel figure")
    a = ap.parse_args()
    render(a.run, a.out, dpi=a.dpi, combined=a.combined)


if __name__ == "__main__":
    main()
