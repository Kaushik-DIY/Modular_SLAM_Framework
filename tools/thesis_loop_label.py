"""Build RGB contact sheets of every UNIQUE proposed loop candidate (across all 9
runs of a map) for manual true/false labelling, and apply labels back to runs.

Why: there is no loop ground truth, so true/false is decided by visually inspecting
the two keyframes' RGB images. A candidate's truth is a property of the two PHYSICAL
places (timestamps), so we dedup all runs' candidates by timestamp, label each unique
pair ONCE, and reuse that label for every run.

  build  : gather unique (t_query, t_cand) pairs -> pairs_<map>.csv + contact-sheet
           montages (query|candidate) under <root>/figures/loops/<map>/.
  (labelling is done by reading the montages and writing labels_<map>.csv:
   pair_id,label  where label in {T,F,?})

    .venv/bin/python tools/thesis_loop_label.py build --root thesis_outputs/thesis_<UTC>
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _tum_ts(run_dir):
    out = []
    for line in open(run_dir / "trajectory.tum"):
        p = line.split()
        if len(p) >= 8:
            out.append(float(p[0]))
    return out                          # line i -> node i timestamp


def _rgb_index(map_name):
    """sorted (timestamp, path) of RGB frames for nearest lookup."""
    rows = []
    for line in open(f"datasets/{map_name}/rgb.txt"):
        if line.startswith("#") or not line.strip():
            continue
        t, f = line.split()[:2]
        rows.append((float(t), f"datasets/{map_name}/{f}"))
    rows.sort()
    return np.array([r[0] for r in rows]), [r[1] for r in rows]


def _nearest_rgb(t, rts, rpaths):
    i = int(np.searchsorted(rts, t))
    i = max(0, min(len(rts) - 1, i))
    if i > 0 and abs(rts[i - 1] - t) < abs(rts[i] - t):
        i -= 1
    return rpaths[i]


def gather(root, map_name, bin_s=0.1):
    """unique candidate pairs across all runs: key=(round(tq),round(tc))."""
    pairs = {}   # key -> dict(t_query,t_cand,n_runs,ever_acc)
    for vc in glob.glob(f"{root}/{map_name}/*/verifications.csv"):
        rd = Path(vc).parent
        ts = _tum_ts(rd)
        for r in csv.DictReader(open(vc)):
            q, c = int(r["query"]), int(r["cand"])
            if q >= len(ts) or c >= len(ts):
                continue
            tq, tc = ts[q], ts[c]
            key = (round(tq, 1), round(tc, 1))
            d = pairs.setdefault(key, dict(t_query=tq, t_cand=tc, n_runs=0, ever_acc=0))
            d["n_runs"] += 1
            if r.get("accepted") == "True":
                d["ever_acc"] += 1
    return pairs


def build(root):
    root = Path(root)
    for map_name in ("lab_hybrid_small", "lab_hybrid"):
        runs = glob.glob(f"{root}/{map_name}/*/verifications.csv")
        if not runs:
            print(f"  {map_name}: no runs yet, skipping"); continue
        pairs = gather(root, map_name)
        items = sorted(pairs.items(), key=lambda kv: (kv[1]["t_query"], kv[1]["t_cand"]))
        # pairs.csv
        outdir = root / "figures" / "loops" / map_name
        outdir.mkdir(parents=True, exist_ok=True)
        with open(root / f"pairs_{map_name}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pair_id", "t_query", "t_cand", "n_runs", "ever_accepted"])
            for i, (k, d) in enumerate(items):
                w.writerow([i, f"{d['t_query']:.3f}", f"{d['t_cand']:.3f}", d["n_runs"], d["ever_acc"]])
        # montages
        import cv2
        rts, rpaths = _rgb_index(map_name)
        # optional auto-labels (ORB+RANSAC) to annotate sheets for visual validation
        lab = {}
        lf = root / f"labels_{map_name}.csv"
        if lf.exists():
            for r in csv.DictReader(open(lf)):
                lab[int(r["pair_id"])] = (r["label"], int(r["inliers"]),
                                          r.get("scan", "?"), r.get("evidence", ""))
        PER = 8; COLS = 2; ROWS = 4; TH = 300
        nsheet = (len(items) + PER - 1) // PER
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for s in range(nsheet):
            chunk = items[s * PER:(s + 1) * PER]
            fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 6.2, ROWS * 2.6))
            axes = np.atleast_1d(axes).ravel()
            for ax in axes:
                ax.axis("off")
            for ax, (k, d) in zip(axes, chunk):
                pid = items.index((k, d))
                qa = cv2.imread(_nearest_rgb(d["t_query"], rts, rpaths))
                ca = cv2.imread(_nearest_rgb(d["t_cand"], rts, rpaths))
                if qa is None or ca is None:
                    continue
                def thumb(im):
                    h, w = im.shape[:2]; return cv2.resize(im[:, :, ::-1], (int(w * TH / h), TH))
                qt, ct = thumb(qa), thumb(ca)
                div = np.full((TH, 4, 3), 255, np.uint8)
                combo = np.hstack([qt, div, ct])
                ax.imshow(combo)
                ann = ""
                if pid in lab:
                    L, inl, sc, ev = lab[pid]; ann = f"  [{L} inl={inl} scan={sc} {ev}]"
                ax.set_title(f"P{pid}  dt={abs(d['t_query']-d['t_cand']):.0f}s"
                             f"  acc{d['ever_acc']}/{d['n_runs']}{ann}", fontsize=9)
            fig.suptitle(f"{map_name} — loop candidates (query | candidate) — sheet {s+1}/{nsheet}",
                         fontsize=11)
            fig.tight_layout()
            fig.savefig(outdir / f"sheet_{s:02d}.png", dpi=110)
            plt.close(fig)
        print(f"  {map_name}: {len(items)} unique pairs -> {nsheet} sheets in {outdir}")


def _orb_inliers(cv2, orb, bf, im1, im2):
    """Geometrically-consistent ORB matches between two frames (RANSAC fundamental
    matrix) — the standard 'do these images see the same scene' test."""
    if im1 is None or im2 is None:
        return 0
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return 0
    good = []
    for mp in bf.knnMatch(d1, d2, k=2):
        if len(mp) == 2 and mp[0].distance < 0.75 * mp[1].distance:
            good.append(mp[0])
    if len(good) < 8:
        return 0
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 3.0, 0.99)
    return int(mask.sum()) if mask is not None else 0


def _scan_scorer(map_name):
    """Rotation-invariant 'same physical place' test on the RAW dataset LiDAR
    scans (B&B-align query scan to a grid built from the candidate scan). 360-deg
    LiDAR => a genuine revisit scores high regardless of heading, which a narrow
    camera can miss. Returns a callable (t_query, t_cand) -> coarse_score in [0,1]."""
    import fusion_core as fc
    from scipy.spatial import cKDTree
    from slam_core.fusion2.dataset import LabHybridStream
    stream = LabHybridStream(Path(f"datasets/{map_name}"), 0.15)
    gc = fc.GridConfig(); gc.resolution = 0.05; gc.l_occ = 0.4; gc.l_free = -0.1
    bnb = fc.BnbConfig(); bnb.linear_search_window = 3.0
    bnb.angular_search_window = math.pi; bnb.depth = 6

    def score(tq, tc):
        sq, sc = stream.nearest_scan(tq), stream.nearest_scan(tc)
        if sq is None or sc is None or len(sq) < 50 or len(sc) < 50:
            return 0.0
        sq = np.asarray(sq, np.float32); sc = np.asarray(sc, np.float32)
        # B&B for the coarse rotation+translation (rotation-invariant place test)
        sig = fc.Signature(0, 0.0, scan_xy=sc)
        grid = fc.assemble_local_grid([sig], [fc.Pose2(0.0, 0.0, 0.0)], gc)
        r = fc.bnb_match(grid, sq, fc.Pose2(0.0, 0.0, 0.0), bnb)
        if not r.success:
            return 0.0
        # actual overlap: fraction of aligned query points near a candidate point
        # (robust to the single-scan grid being thin).
        cth, sth = math.cos(r.pose.theta), math.sin(r.pose.theta)
        qa = sq @ np.array([[cth, sth], [-sth, cth]]) + [r.pose.x, r.pose.y]
        d, _ = cKDTree(sc).query(qa, k=1)
        return float((d < 0.15).mean())
    return score


def label(root, img_thresh=18, scan_thresh=0.55):
    """Auto-label every unique pair by PHYSICAL REVISIT = image OR scan overlap:
    TRUE if ORB+RANSAC inliers >= img_thresh OR B&B scan score >= scan_thresh.
    Writes labels_<map>.csv (pair_id,label,inliers,scan,evidence)."""
    import cv2
    root = Path(root)
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    for map_name in ("lab_hybrid_small", "lab_hybrid"):
        pf = root / f"pairs_{map_name}.csv"
        if not pf.exists():
            continue
        rts, rpaths = _rgb_index(map_name)
        sscore = _scan_scorer(map_name)
        out = []
        for r in csv.DictReader(open(pf)):
            tq, tc = float(r["t_query"]), float(r["t_cand"])
            inl = _orb_inliers(cv2, orb, bf,
                               cv2.imread(_nearest_rgb(tq, rts, rpaths)),
                               cv2.imread(_nearest_rgb(tc, rts, rpaths)))
            sc = round(sscore(tq, tc), 3)
            img_ok, scan_ok = inl >= img_thresh, sc >= scan_thresh
            ev = ("img" if img_ok else "") + ("+scan" if scan_ok else "")
            out.append(dict(pair_id=int(r["pair_id"]),
                            label="T" if (img_ok or scan_ok) else "F",
                            inliers=inl, scan=sc, evidence=ev.strip("+") or "none"))
        with open(root / f"labels_{map_name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["pair_id", "label", "inliers", "scan", "evidence"])
            w.writeheader(); w.writerows(out)
        nt = sum(d["label"] == "T" for d in out)
        byimg = sum(d["evidence"] == "img" for d in out)
        byscan = sum(d["evidence"] == "scan" for d in out)
        byboth = sum(d["evidence"] == "img+scan" for d in out)
        print(f"  {map_name}: {len(out)} pairs -> {nt} TRUE / {len(out)-nt} FALSE  "
              f"(true evidence: img-only {byimg}, scan-only {byscan}, both {byboth})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--root", required=True)
    lb = sub.add_parser("label"); lb.add_argument("--root", required=True)
    lb.add_argument("--img-thresh", type=int, default=18)
    lb.add_argument("--scan-thresh", type=float, default=0.40)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.root)
    elif a.cmd == "label":
        label(a.root, a.img_thresh, a.scan_thresh)


if __name__ == "__main__":
    main()
