"""Score each mode's semantic map against the map-level ground truth.

For every run dir given, rebuilds the semantic raster from its saved vote
tensor (class_votes.npz) in gated and/or ungated form, resamples it onto the
GT grid in WORLD coordinates (all runs share the start-anchored frame, so
pose drift shows up as genuine misalignment error -- which is the point of a
mode comparison), and reports per-class IoU / precision / recall plus macro
means. Cells the GT leaves unlabeled are excluded from scoring.

Usage:
  .venv/bin/python tools/semantic_gt_eval.py \
      --gt thesis_outputs/semantic_gt/<ref>/semantic_gt.npz \
      --runs <run_dir1> <run_dir2> ... [--variants gated ungated] \
      [--classes floor wall]           # restrict (e.g. structural-only draft GT)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slam_core.fusion2.Dependencies.semantics import (          # noqa: E402
    GROUPS, PALETTE_BGR, SemanticVoteGrid, clean_raster)

UNLABELED = -1


def load_gt(gt_path: Path):
    z = np.load(gt_path, allow_pickle=False)
    return z["labels"].astype(np.int16), json.loads(str(z["meta"]))


def run_raster(run_dir: Path, gated: bool) -> tuple:
    grid = SemanticVoteGrid.load(run_dir / "class_votes.npz")
    prob = np.load(run_dir / "map.npy") if gated else None
    raster = clean_raster(grid.argmax_raster(occ_prob=prob))
    return raster, grid.meta


def resample_to_gt(raster: np.ndarray, meta: dict, gt_meta: dict,
                   gt_shape) -> np.ndarray:
    """Nearest-neighbour lookup of the run raster at every GT cell centre."""
    gh, gw = gt_shape
    res_g, ox_g, oy_g = gt_meta["resolution"], gt_meta["origin_x"], gt_meta["origin_y"]
    res_r, ox_r, oy_r = meta["resolution"], meta["origin_x"], meta["origin_y"]
    jj, ii = np.meshgrid(np.arange(gw), np.arange(gh))
    wx = ox_g + (jj + 0.5) * res_g
    wy = oy_g + (ii + 0.5) * res_g
    cj = np.floor((wx - ox_r) / res_r).astype(np.int64)
    ci = np.floor((wy - oy_r) / res_r).astype(np.int64)
    out = np.full((gh, gw), UNLABELED, np.int16)
    ok = (ci >= 0) & (ci < raster.shape[0]) & (cj >= 0) & (cj < raster.shape[1])
    out[ok] = raster[ci[ok], cj[ok]]
    return out


def score(pred: np.ndarray, gt: np.ndarray, class_ids) -> dict:
    scored = gt != UNLABELED
    rows = {}
    for gid in class_ids:
        g = (gt == gid) & scored
        p = (pred == gid) & scored
        inter = int((g & p).sum())
        union = int((g | p).sum())
        rows[GROUPS[gid]] = dict(
            iou=round(inter / union, 3) if union else None,
            precision=round(inter / int(p.sum()), 3) if p.sum() else None,
            recall=round(inter / int(g.sum()), 3) if g.sum() else None,
            gt_cells=int(g.sum()))
    vals = [r["iou"] for r in rows.values() if r["iou"] is not None]
    acc = float((pred[scored] == gt[scored]).mean()) if scored.any() else None
    return dict(per_class=rows, miou=round(float(np.mean(vals)), 3) if vals else None,
                labeled_cell_accuracy=round(acc, 3) if acc is not None else None)


def diff_image(pred, gt, out_png):
    img = np.full((*gt.shape, 3), 255, np.uint8)
    scored = gt != UNLABELED
    ok = scored & (pred == gt)
    bad = scored & (pred != gt) & (pred != UNLABELED)
    miss = scored & (pred == UNLABELED)
    img[ok] = (120, 200, 120)      # correct  (green)
    img[bad] = (80, 80, 220)       # wrong class (red)
    img[miss] = (200, 170, 120)    # unlabeled by mode (blue-gray)
    cv2.imwrite(str(out_png), cv2.flip(cv2.resize(
        img, (gt.shape[1] * 3, gt.shape[0] * 3),
        interpolation=cv2.INTER_NEAREST), 0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["gated"],
                    choices=["gated", "ungated"])
    ap.add_argument("--classes", nargs="+", default=None,
                    help="restrict scoring to these class names")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write diff images (default: GT dir)")
    args = ap.parse_args(argv)

    gt, gt_meta = load_gt(args.gt)
    class_ids = ([GROUPS.index(c) for c in args.classes] if args.classes
                 else sorted({int(g) for g in np.unique(gt) if g >= 0}))
    out_dir = args.out_dir or args.gt.parent

    print(f"GT: {args.gt}  scored classes: {[GROUPS[g] for g in class_ids]}  "
          f"labeled cells: {int((gt != UNLABELED).sum())}")
    results = {}
    for run in args.runs:
        for variant in args.variants:
            raster, meta = run_raster(run, gated=(variant == "gated"))
            pred = resample_to_gt(raster, meta, gt_meta, gt.shape)
            s = score(pred, gt, class_ids)
            key = f"{run.name}[{variant}]"
            results[key] = s
            diff_image(pred, gt, out_dir / f"gt_diff_{run.name}_{variant}.png")
            per = "  ".join(f"{c}: IoU={r['iou']} P={r['precision']} R={r['recall']}"
                            for c, r in s["per_class"].items())
            print(f"\n{key}\n  mIoU={s['miou']}  acc={s['labeled_cell_accuracy']}\n  {per}")

    json.dump(results, open(out_dir / "gt_eval_results.json", "w"), indent=2)
    print(f"\nwrote {out_dir}/gt_eval_results.json + gt_diff_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
