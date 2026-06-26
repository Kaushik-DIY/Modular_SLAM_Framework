"""Assemble per-map thesis metric tables from the realtime runs + my visual loop
labels. For each run: join its proposed candidates (by timestamp) to labels_<map>.csv
to get true-accepted / false-accepted / true-rejected (missed) -> precision/recall;
collate the run_summary metrics + map sharpness + end-start drift.

    .venv/bin/python tools/thesis_results.py --root thesis_outputs/thesis_<UTC>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import numpy as np

COMBO_ORDER = ["lidar_s2s_bnb", "lidar_s2s_icp", "lidar_s2m_bnb", "lidar_s2m_icp",
               "lidar_orb_s2s", "lidar_orb_s2m", "orb_lidar_bnb", "orb_lidar_icp", "orb"]
COLS = ["combo", "keyframes", "proposed", "accepted", "true_acc", "false_acc",
        "true_rej", "precision", "recall", "rt_lag_s", "late_pct", "kf_ms",
        "rss_gb", "payload_mb", "sharpness", "drift_m", "reinits", "fallbacks"]


def _tum(run_dir):
    poses = []
    for line in open(run_dir / "trajectory.tum"):
        p = line.split()
        if len(p) >= 8:
            poses.append((float(p[0]), float(p[1]), float(p[2])))
    return poses


def _labels(root, map_name):
    f = Path(root) / f"labels_{map_name}.csv"
    if not f.exists():
        return None
    pairs = Path(root) / f"pairs_{map_name}.csv"
    pid2key = {}
    for r in csv.DictReader(open(pairs)):
        pid2key[int(r["pair_id"])] = (round(float(r["t_query"]), 1), round(float(r["t_cand"]), 1))
    lab = {}
    for r in csv.DictReader(open(f)):
        L = r["label"].strip().upper()
        lab[pid2key[int(r["pair_id"])]] = (L == "T")   # ? / F -> False
    return lab


def _sharpness(run_dir):
    prob = np.load(run_dir / "map.npy")
    obs = np.abs(prob - 0.5) > 1e-3
    conf = np.abs(prob - 0.5) > 0.4
    return round(conf.sum() / max(1, obs.sum()), 3) if obs.sum() else None


def run_row(root, map_name, combo, labels):
    rd = Path(root) / map_name / combo
    s = json.load(open(rd / "run_summary.json"))
    poses = _tum(rd)
    ts = [p[0] for p in poses]
    ta = fa = tr = lab_prop = 0
    if labels is not None and (rd / "verifications.csv").exists():
        for r in csv.DictReader(open(rd / "verifications.csv")):
            q, c = int(r["query"]), int(r["cand"])
            if q >= len(ts) or c >= len(ts):
                continue
            key = (round(ts[q], 1), round(ts[c], 1))
            if key not in labels:
                continue
            lab_prop += 1
            true = labels[key]; acc = r.get("accepted") == "True"
            if acc and true: ta += 1
            elif acc and not true: fa += 1
            elif (not acc) and true: tr += 1
    drift = round(math.hypot(poses[-1][1] - poses[0][1], poses[-1][2] - poses[0][2]), 3) \
        if len(poses) >= 2 else None
    prec = round(ta / (ta + fa), 3) if (ta + fa) else None
    rec = round(ta / (ta + tr), 3) if (ta + tr) else None
    kf_ms = round((s.get("slam_ms_mean") or 0) + (s.get("loop_ms_mean") or 0), 2)
    return dict(combo=combo, keyframes=s.get("keyframes"), proposed=s.get("proposals"),
                accepted=s.get("loops_accepted"), true_acc=ta, false_acc=fa, true_rej=tr,
                precision=prec, recall=rec, rt_lag_s=s.get("realtime_lag_s"),
                late_pct=s.get("late_pct"), kf_ms=kf_ms, rss_gb=s.get("peak_rss_gb"),
                payload_mb=s.get("map_payload_mb"), sharpness=_sharpness(rd),
                drift_m=drift, reinits=s.get("reinits"), fallbacks=s.get("fallbacks"))


def _md_table(rows):
    head = "| " + " | ".join(COLS) + " |"
    sep = "|" + "|".join("---" for _ in COLS) + "|"
    body = ["| " + " | ".join("-" if rows_i.get(c) is None else str(rows_i.get(c))
            for c in COLS) + " |" for rows_i in rows]
    return "\n".join([head, sep] + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--maps", nargs="+", default=["lab_hybrid_small", "lab_hybrid"])
    a = ap.parse_args()
    root = Path(a.root)
    for map_name in a.maps:
        if not (root / map_name).exists():
            continue
        labels = _labels(root, map_name)
        if labels is None:
            print(f"  {map_name}: labels_{map_name}.csv MISSING — run loop labelling first "
                  "(true/false columns will be 0).")
        rows = [run_row(root, map_name, c, labels) for c in COMBO_ORDER
                if (root / map_name / c / "run_summary.json").exists()]
        with open(root / f"metrics_{map_name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        (root / f"metrics_{map_name}.md").write_text(_md_table(rows) + "\n")
        print(f"  {map_name}: {len(rows)} runs -> metrics_{map_name}.csv/.md "
              f"(labelled={'yes' if labels else 'NO'})")


if __name__ == "__main__":
    main()
