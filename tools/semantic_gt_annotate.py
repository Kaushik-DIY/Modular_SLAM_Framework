"""Semantic-map ground truth: draft, paint, import.

Workflow (map-level GT for the environment, used to score every mode's
semantic map):

  1) --init   Draft a GT label grid from the reference run:
                wall  <- despeckled LiDAR-occupied cells (independent sensor,
                         non-circular w.r.t. the camera semantics under test)
                floor <- LiDAR free space within reach of the trajectory
              plus a PAINTABLE 4x PNG (semantic_gt_paint.png) and a visual
              underlay (semantic_gt_canvas.png) showing occupancy + camera-
              semantic HINTS + trajectory to guide manual annotation.

  2) MANUAL PASS (the actual ground-truthing): open semantic_gt_paint.png in
     any editor (GIMP/Krita/even MS Paint), paint furniture/equipment regions
     with the EXACT palette colours printed by --init (also saved to
     semantic_gt_palette.png), fix any wrong floor/wall. White = unlabeled
     (excluded from scoring).

  3) --import-png semantic_gt_paint.png   Convert the painted PNG back to a
     label grid -> semantic_gt.npz + semantic_gt_preview.png.

GT lives in the reference run's map frame; tools/semantic_gt_eval.py scores
any run's semantic raster against it in world coordinates.

Usage:
  .venv/bin/python tools/semantic_gt_annotate.py --run <ref_run_dir> --init
  .venv/bin/python tools/semantic_gt_annotate.py --run <ref_run_dir> \
      --import-png <gt_dir>/semantic_gt_paint.png
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
    GROUPS, PALETTE_BGR, SemanticVoteGrid, clean_raster, occupied_mask)
from slam_core.fusion2.Dependencies.semantic_pipeline import _load_traj  # noqa: E402

SCALE = 4                       # paint PNG upscale (nearest; import undoes it)
WALL_GID = GROUPS.index("wall")
FLOOR_GID = GROUPS.index("floor")
UNLABELED = -1
WHITE = (255, 255, 255)


def _labels_to_bgr(labels: np.ndarray) -> np.ndarray:
    img = np.full((*labels.shape, 3), 255, np.uint8)
    for gid, name in enumerate(GROUPS):
        img[labels == gid] = PALETTE_BGR[name]
    return img


def cmd_init(run_dir: Path, gt_dir: Path, traj_radius_m: float) -> None:
    meta = json.load(open(run_dir / "map_meta.json"))
    prob = np.load(run_dir / "map.npy")
    traj = _load_traj(run_dir)
    res, ox, oy = meta["resolution"], meta["origin_x"], meta["origin_y"]
    h, w = prob.shape

    # structural draft: wall from LiDAR structure, floor from reachable free space
    labels = np.full((h, w), UNLABELED, np.int16)
    occ = occupied_mask(prob)
    reach = np.zeros((h, w), np.uint8)
    for _, x, y, _ in traj:
        cv2.circle(reach, (int((x - ox) / res), int((y - oy) / res)),
                   int(traj_radius_m / res), 1, -1)
    labels[(prob < 0.35) & (reach > 0)] = FLOOR_GID
    labels[occ] = WALL_GID

    gt_dir.mkdir(parents=True, exist_ok=True)
    # paintable PNG (4x nearest so brush strokes are comfortable)
    paint = cv2.resize(_labels_to_bgr(labels), (w * SCALE, h * SCALE),
                       interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(gt_dir / "semantic_gt_paint.png"), cv2.flip(paint, 0))

    # visual underlay: occupancy + camera-semantic HINTS (hints only -- the
    # human paints labels, so the GT is not a copy of the system under test)
    canvas = cv2.cvtColor((255 - prob * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    votes_path = run_dir / "class_votes.npz"
    if votes_path.exists():
        grid = SemanticVoteGrid.load(votes_path)
        hint = clean_raster(grid.argmax_raster(occ_prob=prob))
        hint_bgr = _labels_to_bgr(hint)
        m = hint >= 0
        canvas[m] = (canvas[m] * 0.45 + hint_bgr[m] * 0.55).astype(np.uint8)
    for _, x, y, _ in traj:
        cv2.circle(canvas, (int((x - ox) / res), int((y - oy) / res)), 1,
                   (255, 160, 40), -1)
    canvas = cv2.resize(canvas, (w * SCALE, h * SCALE),
                        interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(gt_dir / "semantic_gt_canvas.png"), cv2.flip(canvas, 0))

    # palette card for the editor's colour picker
    sw, pad = 120, 34
    card = np.full((pad * (len(GROUPS) + 1), sw + 260, 3), 255, np.uint8)
    for gid, name in enumerate(GROUPS):
        y0 = pad * gid + 6
        cv2.rectangle(card, (8, y0), (8 + sw, y0 + pad - 12), PALETTE_BGR[name], -1)
        b, g, r = PALETTE_BGR[name]
        cv2.putText(card, f"{name}  RGB=({r},{g},{b})", (sw + 18, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(card, "white = unlabeled (not scored)",
                (8, pad * len(GROUPS) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(gt_dir / "semantic_gt_palette.png"), card)

    json.dump(dict(meta, scale=SCALE, reference_run=str(run_dir),
                   traj_radius_m=traj_radius_m),
              open(gt_dir / "semantic_gt_meta.json", "w"), indent=2)
    n_f, n_w = int((labels == FLOOR_GID).sum()), int((labels == WALL_GID).sum())
    print(f"draft GT: floor={n_f} wall={n_w} cells -> {gt_dir}/")
    print("  paint  : semantic_gt_paint.png   (edit this; palette card = "
          "semantic_gt_palette.png; guide = semantic_gt_canvas.png)")
    print("  import : tools/semantic_gt_annotate.py --run <ref_run> "
          "--import-png <gt_dir>/semantic_gt_paint.png")


def cmd_import(run_dir: Path, gt_dir: Path, png: Path, tol: int) -> None:
    meta = json.load(open(run_dir / "map_meta.json"))
    h, w = int(meta["height"]), int(meta["width"])
    img = cv2.flip(cv2.imread(str(png)), 0)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

    # nearest-palette-colour match (tolerant to editor anti-aliasing)
    cols = np.array([PALETTE_BGR[n] for n in GROUPS] + [WHITE], np.int32)
    d = np.linalg.norm(img[:, :, None, :].astype(np.int32) - cols[None, None],
                       axis=3)
    best = np.argmin(d, axis=2)
    best_d = np.take_along_axis(d, best[..., None], axis=2)[..., 0]
    labels = np.where((best < len(GROUPS)) & (best_d <= tol),
                      best, UNLABELED).astype(np.int16)

    np.savez_compressed(gt_dir / "semantic_gt.npz", labels=labels,
                        meta=json.dumps(meta), groups=np.array(GROUPS))
    prev = _labels_to_bgr(labels)
    cv2.imwrite(str(gt_dir / "semantic_gt_preview.png"),
                cv2.flip(cv2.resize(prev, (w * SCALE, h * SCALE),
                                    interpolation=cv2.INTER_NEAREST), 0))
    counts = {n: int((labels == g).sum()) for g, n in enumerate(GROUPS)
              if (labels == g).any()}
    print(f"GT imported -> {gt_dir}/semantic_gt.npz  cells per class: {counts}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True,
                    help="reference run dir (its map grid defines the GT frame)")
    ap.add_argument("--gt-dir", type=Path, default=None,
                    help="GT output dir (default thesis_outputs/semantic_gt/<dataset-ish>)")
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--import-png", type=Path, default=None)
    ap.add_argument("--traj-radius", type=float, default=3.0,
                    help="floor draft = LiDAR free space within this distance "
                         "of the trajectory (m)")
    ap.add_argument("--tol", type=int, default=60,
                    help="max colour distance when importing painted PNG")
    args = ap.parse_args(argv)

    gt_dir = args.gt_dir or Path("thesis_outputs/semantic_gt") / args.run.name
    if args.init:
        cmd_init(args.run, gt_dir, args.traj_radius)
    if args.import_png is not None:
        cmd_import(args.run, gt_dir, args.import_png, args.tol)
    if not args.init and args.import_png is None:
        ap.error("nothing to do: pass --init and/or --import-png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
