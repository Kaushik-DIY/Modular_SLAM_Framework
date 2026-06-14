"""GT-free true/false labelling of accepted loop closures (no ground truth).

For each accepted loop in a fusion run, label it TRUE only if BOTH independent
checks agree (conservative), else FALSE; also count one-sided disagreements:

  1. Geometric scan-overlap — at the OPTIMIZED poses, transform the query
     keyframe's raw LiDAR scan into the world and measure the fraction of its
     points that land within a strict distance of the candidate neighbourhood's
     scan points. A false loop forces non-matching places together, so the scans
     do not coincide -> low overlap. (Independent of the accepting verifier:
     cross-checks visual PnP loops with geometry.)
  2. Graph residual — the loop's MEASURED transform (logged rx,ry,rth) vs the
     optimized relative pose pose[cand]^-1 o pose[query]. Catches loops the
     optimizer out-voted.

Also computes GT-free trajectory + map proxies (no ATE without ground truth):
end-start return drift, loop-residual RMSE, map sharpness.

Usage:
  .venv/bin/python tools/analyze_fusion_loops.py --run <run_dir> --dataset datasets/lab_hybrid
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slam_core.fusion2.dataset import LabHybridStream


# --- thresholds (defensible defaults; all CLI-overridable) ------------------
D_STRICT = 0.10        # m, strict scan-overlap inlier distance
TAU_GEOM = 0.50        # min overlap fraction for TRUE-geom
RES_T = 0.30           # m, max translation residual for TRUE-resid
RES_R = math.radians(5.0)   # rad, max rotation residual for TRUE-resid
NBR_RADIUS = 2.0       # m, candidate-neighbourhood radius (mirrors retrieval)
MIN_SEP = 30           # exclude the query's own temporal trail


def _load_traj(run_dir: Path):
    """node_id -> (t, x, y, theta). tum line i == node i (std::map id order)."""
    poses = {}
    for i, line in enumerate(open(run_dir / "trajectory.tum")):
        p = line.split()
        if len(p) < 8:
            continue
        t, x, y = float(p[0]), float(p[1]), float(p[2])
        qz, qw = float(p[6]), float(p[7])
        poses[i] = (t, x, y, 2.0 * math.atan2(qz, qw))
    return poses


def _accepted_loops(run_dir: Path):
    out = []
    f = run_dir / "verifications.csv"
    if not f.exists():
        return out
    for r in csv.DictReader(open(f)):
        if r.get("accepted") == "True":
            def g(k):
                v = r.get(k, "")
                return float(v) if v not in ("", "None", None) else None
            out.append((int(r["query"]), int(r["cand"]), g("rx"), g("ry"), g("rth")))
    return out


def _world(scan, x, y, th):
    c, s = math.cos(th), math.sin(th)
    return scan @ np.array([[c, -s], [s, c]]).T + [x, y]


def _rel(ax, ay, ath, bx, by, bth):
    """relative pose of b in frame a (a^-1 o b)."""
    c, s = math.cos(ath), math.sin(ath)
    dx, dy = bx - ax, by - ay
    return (c * dx + s * dy, -s * dx + c * dy,
            math.atan2(math.sin(bth - ath), math.cos(bth - ath)))


def analyze_run(run_dir, dataset, d_strict=D_STRICT, tau_geom=TAU_GEOM,
                res_t=RES_T, res_r=RES_R, nbr_radius=NBR_RADIUS, min_sep=MIN_SEP):
    run_dir = Path(run_dir)
    poses = _load_traj(run_dir)
    loops = _accepted_loops(run_dir)
    stream = LabHybridStream(Path(dataset), 0.05)

    _scache = {}
    def scan_at(nid):
        if nid not in _scache:
            sc = stream.nearest_scan(poses[nid][0]) if nid in poses else None
            _scache[nid] = np.asarray(sc, float) if sc is not None and len(sc) else None
        return _scache[nid]

    ids = sorted(poses)
    pos = {i: (poses[i][1], poses[i][2]) for i in ids}

    rows = []
    residuals = []
    for q, c, rx, ry, rth in loops:
        if q not in poses or c not in poses:
            continue
        # --- 1. geometric scan-overlap at optimized poses ---
        overlap, geom_ok = 0.0, False
        qs = scan_at(q)
        if qs is not None:
            qx, qy, qth = poses[q][1], poses[q][2], poses[q][3]
            qw = _world(qs, qx, qy, qth)
            cx, cy = pos[c]
            nbr = []
            for j in ids:
                if abs(j - q) < min_sep:
                    continue
                if math.hypot(pos[j][0] - cx, pos[j][1] - cy) > nbr_radius:
                    continue
                sj = scan_at(j)
                if sj is not None:
                    nbr.append(_world(sj, poses[j][1], poses[j][2], poses[j][3]))
            if nbr:
                tree = cKDTree(np.vstack(nbr))
                d, _ = tree.query(qw, k=1)
                overlap = float((d < d_strict).mean())
                geom_ok = overlap >= tau_geom

        # --- 2. graph residual: measured rel vs optimized rel ---
        res_dxy = res_dth = None
        res_ok = False
        if rx is not None:
            ox, oy, oth = _rel(poses[c][1], poses[c][2], poses[c][3],
                               poses[q][1], poses[q][2], poses[q][3])
            res_dxy = math.hypot(rx - ox, ry - oy)
            res_dth = abs(math.atan2(math.sin(rth - oth), math.cos(rth - oth)))
            res_ok = (res_dxy <= res_t and res_dth <= res_r)
            residuals.append((res_dxy, res_dth))

        true_loop = bool(geom_ok and res_ok)
        rows.append(dict(
            query=q, cand=c, overlap=round(overlap, 3), geom_ok=geom_ok,
            res_dxy=None if res_dxy is None else round(res_dxy, 3),
            res_dth_deg=None if res_dth is None else round(math.degrees(res_dth), 2),
            res_ok=res_ok, true_loop=true_loop))

    # write per-loop labels
    if rows:
        with open(run_dir / "loop_labels.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    n = len(rows)
    n_true = sum(r["true_loop"] for r in rows)
    n_geom_only = sum(r["geom_ok"] and not r["res_ok"] for r in rows)
    n_res_only = sum(r["res_ok"] and not r["geom_ok"] for r in rows)
    res_arr = np.array(residuals) if residuals else np.zeros((0, 2))
    summary = dict(
        accepted=n, true_accepted=n_true, false_accepted=n - n_true,
        disagreement=n_geom_only + n_res_only,
        precision=round(n_true / n, 3) if n else None,
        loop_resid_rmse_m=round(float(np.sqrt(np.mean(res_arr[:, 0] ** 2))), 3) if len(res_arr) else None,
        loop_resid_rmse_deg=round(float(np.degrees(np.sqrt(np.mean(res_arr[:, 1] ** 2)))), 2) if len(res_arr) else None,
    )
    summary.update(_map_traj_metrics(run_dir, poses))
    return summary


def _map_traj_metrics(run_dir: Path, poses: dict) -> dict:
    out = {}
    # end-start return drift (GT-free trajectory-quality proxy)
    if len(poses) >= 2:
        ids = sorted(poses)
        x0, y0 = poses[ids[0]][1], poses[ids[0]][2]
        x1, y1 = poses[ids[-1]][1], poses[ids[-1]][2]
        out["end_start_drift_m"] = round(math.hypot(x1 - x0, y1 - y0), 3)
        P = np.array([[poses[i][1], poses[i][2]] for i in ids])
        span = P.max(0) - P.min(0)
        out["traj_span_m"] = f"{span[0]:.1f}x{span[1]:.1f}"
    # map sharpness from the occupancy probability grid
    mp = run_dir / "map.npy"
    if mp.exists():
        prob = np.load(mp)
        observed = np.abs(prob - 0.5) > 1e-3
        confident = np.abs(prob - 0.5) > 0.4
        nobs = int(observed.sum())
        out["map_occupied_cells"] = int((prob > 0.6).sum())
        out["map_free_cells"] = int((prob < 0.4).sum())
        out["map_sharpness"] = round(confident.sum() / nobs, 3) if nobs else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--d-strict", type=float, default=D_STRICT)
    ap.add_argument("--tau-geom", type=float, default=TAU_GEOM)
    a = ap.parse_args()
    s = analyze_run(a.run, a.dataset, d_strict=a.d_strict, tau_geom=a.tau_geom)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
