"""Thesis figure for the real-time module-switching capability.

Reads one switching-demo run dir (produced by run_realtime.py with
--switch-schedule) and renders a composite figure:

  Top    — the final occupancy map with the trajectory coloured per active-config
           segment, numbered markers at each switch, and a legend of the segments.
  Bottom — three trend panels sharing the keyframe x-axis, each with dashed lines
           at the switch keyframes:
             1. KF step-size  (no-teleport: smooth across a switch = stable handoff)
             2. Tracking quality (VO inliers on visual segments, scan score on LiDAR)
             3. Cumulative accepted loop closures (the map keeps being corrected)

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

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "savefig.bbox": "tight"})

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
    """Active-config string for the segment starting at this timeline row."""
    return (f"{_FE_SHORT.get(tl_row['fe'], tl_row['fe'])} · "
            f"{tl_row['proposer']} · {tl_row['verifier']}")


def _switch_label(sw) -> str:
    """Human transition label, e.g. 's2s→VO  [auto bnb→pnp]'."""
    frm = _FE_SHORT.get(sw["frm"], sw["frm"])
    to = _FE_SHORT.get(sw["to"], sw["to"])
    base = f"{frm}→{to}"
    if str(sw.get("autofallback", "")).lower() in ("true", "1"):
        base += "  [auto-fallback]"
    return base


def render(run_dir: Path, out_base: Path, dpi: int = 600):
    run_dir = Path(run_dir)
    summary = json.load(open(run_dir / "run_summary.json"))
    switches = _read_csv(run_dir / "switches.csv")
    timeline = _read_csv(run_dir / "timeline.csv")
    verifs = _read_csv(run_dir / "verifications.csv")
    traj = _load_traj(run_dir / "trajectory.tum")
    prob = np.load(run_dir / "map.npy")
    meta = json.load(open(run_dir / "map_meta.json"))
    ext = meta["extent"]

    if not timeline:
        raise SystemExit(f"{run_dir} has no timeline.csv — was it run with --switch-schedule?")

    kf = np.array([int(r["kf"]) for r in timeline])
    tlx = np.array([float(r["x"]) for r in timeline])
    tly = np.array([float(r["y"]) for r in timeline])
    qual = np.array([float(r["quality"]) for r in timeline])
    sensor = [r["sensor"] for r in timeline]

    # switch keyframes (effective kf where each switch took hold)
    sw_kf = [int(s["kf"]) for s in switches]
    # segment boundaries in keyframe space: [first_kf, sw1, sw2, ..., last_kf]
    bounds = [int(kf[0])] + sw_kf + [int(kf[-1]) + 1]
    # timeline row at the start of each segment -> its config label
    seg_start_rows = []
    for b in bounds[:-1]:
        idx = int(np.searchsorted(kf, b))
        seg_start_rows.append(timeline[min(idx, len(timeline) - 1)])

    # final-trajectory positions indexed to timeline kf when counts match
    # (no node removal); else fall back to the live timeline positions.
    if traj is not None and len(traj) == len(timeline):
        px, py = traj[:, 0], traj[:, 1]
    else:
        px, py = tlx, tly

    # ---- figure layout: map on top, 3 trend panels below -------------------
    fig = plt.figure(figsize=(13, 15))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.6, 1.0, 1.0, 1.0], hspace=0.28)
    axm = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3], sharex=ax2)

    # ---- top: annotated map ------------------------------------------------
    axm.imshow(prob, cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower",
               extent=ext, interpolation="nearest", rasterized=True)
    seg_handles = []
    for si in range(len(bounds) - 1):
        lo, hi = bounds[si], bounds[si + 1]
        m = (kf >= lo) & (kf < hi)
        col = SEG_COLORS[si % len(SEG_COLORS)]
        axm.plot(px[m], py[m], "-", lw=1.4, color=col, alpha=0.9, zorder=4)
        seg_handles.append(Line2D([0], [0], color=col, lw=2.4,
                                  label=f"{si+1}. {_seg_label(seg_start_rows[si])}"))
    # numbered diamonds at each switch
    for n, s in enumerate(switches, 1):
        sx, sy = float(s["x"]), float(s["y"])
        axm.scatter(sx, sy, s=150, marker="D", facecolor="yellow",
                    edgecolor="black", lw=1.2, zorder=8)
        axm.annotate(str(n), (sx, sy), color="black", fontsize=9, fontweight="bold",
                     ha="center", va="center", zorder=9)
    axm.scatter(px[0], py[0], s=60, c="#15a015", zorder=7)
    axm.scatter(px[-1], py[-1], s=60, c="red", zorder=7)
    sw_legend = [Line2D([0], [0], marker="D", color="w", markerfacecolor="yellow",
                        markeredgecolor="black", ms=10,
                        label=f"{n}: {_switch_label(s)}")
                 for n, s in enumerate(switches, 1)]
    start_end = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#15a015",
                        ms=8, label="Start"),
                 Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                        ms=8, label="End")]
    leg1 = axm.legend(handles=seg_handles, loc="upper left", fontsize=8,
                      framealpha=0.85, title="Active config per segment")
    axm.add_artist(leg1)
    axm.legend(handles=sw_legend + start_end, loc="lower right", fontsize=8,
               framealpha=0.85, title="Switch points")
    axm.set_aspect("equal"); axm.set_xticks([]); axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_edgecolor("#999"); sp.set_linewidth(0.8)
    axm.set_title(f"Real-time module switching — {summary.get('mode','')} demo "
                  f"({len(timeline)} kf, {summary.get('loops_accepted',0)} loops, "
                  f"{len(switches)} switches)", fontsize=13, pad=8)

    def _switch_lines(ax, shade=True):
        if shade:   # faint per-segment background tying panels to the map legend
            for si in range(len(bounds) - 1):
                ax.axvspan(bounds[si], bounds[si + 1],
                           color=SEG_COLORS[si % len(SEG_COLORS)], alpha=0.06, zorder=0)
        for n, k in enumerate(sw_kf, 1):
            ax.axvline(k, color="black", ls="--", lw=0.8, alpha=0.6)
            ax.annotate(str(n), (k, 1.0), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ---- panel 1: KF step-size (no-teleport) -------------------------------
    step = np.hypot(np.diff(px), np.diff(py))
    ax1.plot(kf[1:], step, "-", lw=0.9, color="#1f77b4")
    ax1.set_ylabel("KF step [m]")
    ax1.set_title("Keyframe-to-keyframe displacement — smooth across each switch = "
                  "no teleport (stable handoff)", fontsize=10)
    ax1.grid(alpha=0.2)
    _switch_lines(ax1)

    # ---- panel 2: tracking quality (twin axis: VO inliers | LiDAR score) ---
    vo_m = np.array([s == "vo" for s in sensor])
    li_m = ~vo_m
    ax2b = ax2.twinx()
    ax2b.set_zorder(ax2.get_zorder() - 1); ax2.patch.set_visible(False)
    if vo_m.any():
        ax2.plot(kf[vo_m], qual[vo_m], ".", ms=3, color="#9467bd", label="VO inliers")
    if li_m.any():
        ax2b.plot(kf[li_m], qual[li_m], ".", ms=3, color="#2ca02c", label="LiDAR score")
    ax2.set_ylabel("VO inliers", color="#9467bd")
    ax2b.set_ylabel("LiDAR match score", color="#2ca02c")
    ax2.tick_params(axis="y", colors="#9467bd")
    ax2b.tick_params(axis="y", colors="#2ca02c")
    ax2.set_title("Front-end tracking quality per segment "
                  "(visual = VO inlier count, LiDAR = scan-match score)", fontsize=10)
    ax2.grid(alpha=0.2)
    _switch_lines(ax2)

    # ---- panel 3: cumulative accepted loop closures ------------------------
    acc_kf = sorted(int(v["query"]) for v in verifs
                    if str(v.get("accepted", "")).lower() in ("true", "1"))
    if acc_kf:
        ax3.step(acc_kf, np.arange(1, len(acc_kf) + 1), where="post",
                 color="#d62728", lw=1.3)
    ax3.set_ylabel("loops accepted\n(cumulative)")
    ax3.set_xlabel("keyframe")
    ax3.set_title("Accepted loop closures (cumulative) — loops keep closing across "
                  "switches; flat stretches = route not revisiting", fontsize=10)
    ax3.grid(alpha=0.2)
    _switch_lines(ax3)

    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_base}.png / .pdf  "
          f"({len(switches)} switches, {len(acc_kf)} loops)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="switching-demo run dir")
    ap.add_argument("--out", type=Path, required=True, help="output figure basepath (no ext)")
    ap.add_argument("--dpi", type=int, default=600)
    a = ap.parse_args()
    render(a.run, a.out, dpi=a.dpi)


if __name__ == "__main__":
    main()
