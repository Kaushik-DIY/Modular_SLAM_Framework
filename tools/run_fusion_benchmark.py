"""Fusion benchmarking suite — all 9 method combos x 2 hybrid maps, for the
thesis Results & Discussion section. Everything lands in ONE dedicated folder.

Runs the matrix, labels each accepted loop true/false (GT-free, conservative:
tools/analyze_fusion_loops.py), and emits a master metrics.csv + BENCHMARK_REPORT.md
(per-map + cross-map tables) + map/trajectory montages + RUN_NOTE.md, all inside
fusion2_outputs/benchmark_<UTC>/.

    .venv/bin/python tools/run_fusion_benchmark.py
    .venv/bin/python tools/run_fusion_benchmark.py --maps lab_hybrid_small   # quick subset
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slam_core.fusion2.config import FusionV2Config
from slam_core.fusion2.runner import run_lidar_mode, run_orb_mode_native
from tools.analyze_fusion_loops import analyze_run

# combo id -> (mode, config overrides)
MATRIX = [
    ("lidar_s2s_bnb", "lidar",     dict(lidar_frontend="native_s2s", scan_verifier="bnb")),
    ("lidar_s2s_icp", "lidar",     dict(lidar_frontend="native_s2s", scan_verifier="icp")),
    ("lidar_s2m_bnb", "lidar",     dict(lidar_frontend="native_s2m", scan_verifier="bnb")),
    ("lidar_s2m_icp", "lidar",     dict(lidar_frontend="native_s2m", scan_verifier="icp")),
    ("lidar_orb_s2s", "lidar_orb", dict(lidar_frontend="native_s2s")),
    ("lidar_orb_s2m", "lidar_orb", dict(lidar_frontend="native_s2m")),
    ("orb_lidar_bnb", "orb_lidar", dict(scan_verifier="bnb")),
    ("orb_lidar_icp", "orb_lidar", dict(scan_verifier="icp")),
    ("orb",           "orb",       dict()),
]

# metric columns for metrics.csv / the report, in display order
COLS = [
    "map", "combo", "mode", "frontend", "verifier", "keyframes",
    "proposals", "verified", "loops_accepted", "true_accepted", "false_accepted",
    "disagreement", "precision", "accept_rate",
    "end_start_drift_m", "loop_resid_rmse_m", "loop_resid_rmse_deg",
    "map_sharpness", "map_occupied_cells", "map_free_cells", "traj_span_m",
    "peak_rss_gb", "map_payload_mb", "stm", "wm", "ltm",
    "elapsed_sec", "reinits", "fallbacks", "rehearsal_merges",
]


def _run_one(map_name, combo, mode, overrides, out_root):
    dataset = Path("datasets") / map_name
    run_dir = out_root / map_name / combo
    cfg = FusionV2Config(mode=mode, dataset=dataset, output_dir=run_dir.parent,
                         print_every=200, **overrides)
    t0 = time.perf_counter()
    fn = run_lidar_mode if mode in ("lidar", "lidar_orb") else run_orb_mode_native
    stats = fn(cfg)
    # the runner writes to a <mode>_<timestamp> subdir; rename to the clean combo id
    import shutil
    written = Path(stats["run_dir"])
    actual = run_dir
    if actual.exists():
        shutil.rmtree(actual)
    shutil.move(str(written), str(actual))
    stats["run_dir"] = str(actual)
    labels = analyze_run(actual, dataset)
    row = dict(map=map_name, combo=combo, mode=mode,
               frontend=stats.get("frontend"), verifier=stats.get("verifier"),
               keyframes=stats.get("keyframes"), proposals=stats.get("proposals"),
               verified=stats.get("verified"), loops_accepted=stats.get("loops_accepted"),
               peak_rss_gb=stats.get("peak_rss_gb"), map_payload_mb=stats.get("map_payload_mb"),
               stm=stats.get("stm"), wm=stats.get("wm"), ltm=stats.get("ltm"),
               elapsed_sec=stats.get("elapsed_sec"), reinits=stats.get("reinits"),
               fallbacks=stats.get("fallbacks"), rehearsal_merges=stats.get("rehearsal_merges"),
               run_dir=str(actual), wall_s=round(time.perf_counter() - t0, 1))
    row.update(labels)
    p, a = stats.get("proposals") or 0, stats.get("loops_accepted") or 0
    row["accept_rate"] = round(a / p, 3) if p else None
    # save per-run combined metrics for resumability
    (actual / "benchmark_metrics.json").write_text(json.dumps(row, indent=2, default=str))
    return row


def _fmt(v):
    return "-" if v is None else (f"{v}" if not isinstance(v, float) else f"{v:g}")


def _table(rows, cols):
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def _montage(rows, out_root, map_name, fname, kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    imgs = []
    for r in rows:
        p = Path(r["run_dir"]) / ("occupancy.png" if kind == "map" else "scan_overlay.png")
        if p.exists():
            imgs.append((r["combo"], p))
    if not imgs:
        return
    n = len(imgs)
    cols = 3
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 5, rows_n * 3.2))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (name, p) in zip(axes, imgs):
        ax.imshow(mpimg.imread(p)); ax.set_title(name, fontsize=8)
    fig.suptitle(f"{map_name} — {kind} montage", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_root / fname, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=["lab_hybrid_small", "lab_hybrid"])
    ap.add_argument("--combos", nargs="+", default=None,
                    help="subset of combo ids (default: all 9)")
    ap.add_argument("--output", type=Path, default=Path("fusion2_outputs"))
    ap.add_argument("--into", type=Path, default=None,
                    help="re-run --combos into an EXISTING benchmark folder and "
                         "rebuild the report from all runs on disk (no new folder)")
    a = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    if a.into is not None:
        out_root = a.into
    else:
        out_root = a.output / f"benchmark_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    combos = [c for c in MATRIX if a.combos is None or c[0] in a.combos]

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    print(f"benchmark -> {out_root}  (commit {commit})", flush=True)

    for map_name in a.maps:
        for combo, mode, ov in combos:
            print(f"\n{'='*64}\n  [{map_name}] {combo}\n{'='*64}", flush=True)
            try:
                row = _run_one(map_name, combo, mode, ov, out_root)
                print(f"  -> loops {row['loops_accepted']} "
                      f"(true {row.get('true_accepted')}/{row.get('false_accepted')} false), "
                      f"precision {row.get('precision')}, drift {row.get('end_start_drift_m')} m, "
                      f"sharp {row.get('map_sharpness')}, rss {row.get('peak_rss_gb')} GB", flush=True)
            except Exception as e:
                print(f"  FAILED: {e}\n{traceback.format_exc()}", flush=True)

    # gather ALL runs present on disk (re-run + pre-existing), in matrix order, so
    # an --into update rebuilds the report over the full folder.
    order = {c[0]: i for i, c in enumerate(MATRIX)}
    all_rows = []
    for map_name in a.maps:
        rows_m = []
        for d in (out_root / map_name).glob("*"):
            mj = d / "benchmark_metrics.json"
            if mj.exists():
                rows_m.append(json.loads(mj.read_text()))
        rows_m.sort(key=lambda r: order.get(r.get("combo"), 99))
        all_rows.extend(rows_m)

    # ---- metrics.csv ----
    import csv as _csv
    with open(out_root / "metrics.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    # ---- montages per map ----
    for map_name in a.maps:
        mrows = [r for r in all_rows if r.get("map") == map_name and "run_dir" in r]
        _montage(mrows, out_root, map_name, f"maps_montage_{map_name}.png", "map")
        _montage(mrows, out_root, map_name, f"trajectories_montage_{map_name}.png", "traj")

    # ---- BENCHMARK_REPORT.md ----
    rep = ["# Fusion benchmark — methods x environments\n",
           f"Commit `{commit}`  ·  generated {stamp}  ·  GT-free metrics "
           "(no ground truth in these datasets).\n",
           "True/false loops are **conservative**: a loop is TRUE only if BOTH the "
           "geometric scan-overlap AND the graph-residual checks agree "
           "(`tools/analyze_fusion_loops.py`).\n"]
    lc_cols = ["combo", "mode", "verifier", "proposals", "verified", "loops_accepted",
               "true_accepted", "false_accepted", "disagreement", "precision", "accept_rate"]
    qual_cols = ["combo", "end_start_drift_m", "loop_resid_rmse_m", "loop_resid_rmse_deg",
                 "map_sharpness", "map_occupied_cells", "map_free_cells", "traj_span_m"]
    cost_cols = ["combo", "keyframes", "peak_rss_gb", "map_payload_mb", "stm", "wm", "ltm",
                 "elapsed_sec", "reinits", "fallbacks"]
    for map_name in a.maps:
        mrows = [r for r in all_rows if r.get("map") == map_name and "run_dir" in r]
        if not mrows:
            continue
        rep += [f"\n## {map_name}\n",
                "### Loop closures\n", _table(mrows, lc_cols),
                "\n\n### Trajectory & map quality (GT-free)\n", _table(mrows, qual_cols),
                "\n\n### Cost & health\n", _table(mrows, cost_cols), "\n"]
    # cross-map: precision + drift + sharpness side by side
    rep += ["\n## Cross-map summary (precision / drift / sharpness)\n"]
    order = {c[0]: i for i, c in enumerate(MATRIX)}
    combos_seen = sorted({r["combo"] for r in all_rows if "combo" in r},
                         key=lambda c: order.get(c, 99))
    cross = []
    for cid in combos_seen:
        row = {"combo": cid}
        for map_name in a.maps:
            r = next((x for x in all_rows if x.get("combo") == cid and x.get("map") == map_name), {})
            row[f"{map_name}_prec"] = r.get("precision")
            row[f"{map_name}_drift"] = r.get("end_start_drift_m")
            row[f"{map_name}_sharp"] = r.get("map_sharpness")
        cross.append(row)
    cross_cols = ["combo"] + [f"{m}_{k}" for m in a.maps for k in ("prec", "drift", "sharp")]
    rep += [_table(cross, cross_cols), "\n"]
    (out_root / "BENCHMARK_REPORT.md").write_text("\n".join(rep) + "\n")

    # ---- RUN_NOTE.md ----
    note = [f"# Benchmark run {stamp}\n",
            f"- commit: `{commit}`",
            f"- maps: {', '.join(a.maps)}",
            f"- combos: {', '.join(combos_seen)} ({len(combos_seen)} x {len(a.maps)} = "
            f"{len(combos_seen)*len(a.maps)} runs)",
            "- true/false loop labelling: conservative (geometric scan-overlap AND graph "
            "residual); thresholds d_strict=0.10 m, tau_geom=0.50, res<=0.30 m / 5 deg",
            "- accuracy: GT-free only (no ground truth) — end-start drift, loop-residual "
            "RMSE, map sharpness",
            f"- outputs: metrics.csv, BENCHMARK_REPORT.md, montages, per-run loop_labels.csv\n"]
    (out_root / "RUN_NOTE.md").write_text("\n".join(note) + "\n")

    # ---- index pointer (fresh runs only) ----
    if a.into is None:
        idx = a.output / "INDEX.md"
        line = f"- [benchmark_{stamp}](benchmark_{stamp}/BENCHMARK_REPORT.md) — " \
               f"{len(combos_seen)} combos x {len(a.maps)} maps, GT-free method comparison\n"
        with open(idx, "a") as f:
            f.write(line)

    print(f"\nDONE -> {out_root}/BENCHMARK_REPORT.md  ({len(all_rows)} runs)")


if __name__ == "__main__":
    main()
