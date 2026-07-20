"""Offline semantic mapping for a finished fusion2 run (thin CLI).

All logic lives in slam_core/fusion2/Dependencies/semantic_pipeline.py; the
batch runner's --semantic flag calls the same pipeline. Produces a 4-panel
video (RGB POV | semantic view / LiDAR occupancy building | semantic map
building), a clean semantic floor-plan PNG, the vote tensor, and metrics.

Usage:
  .venv/bin/python tools/semantic_map_video.py \
      --run <run_dir> --dataset datasets/lab_hybrid [--speed 4] [--fps 10]
  .venv/bin/python tools/semantic_map_video.py --selftest --dataset datasets/lab_hybrid
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slam_core.fusion2.Dependencies.semantic_pipeline import run_semantic_pipeline  # noqa: E402


def selftest(dataset: Path, model_dir: Path) -> int:
    import cv2
    from slam_core.fusion2.Dependencies.dataset import LabHybridStream
    from slam_core.fusion2.Dependencies.semantics import GROUPS, SegmenterBackend
    stream = LabHybridStream(dataset)
    entries = stream.rgbd_entries()
    t, rgb_p, _ = entries[len(entries) // 2]
    rgb = cv2.imread(str(rgb_p))
    backend = SegmenterBackend(model_dir)
    backend.segment(rgb)                       # warm-up (session init)
    t0 = time.perf_counter()
    gmap = backend.segment(rgb)
    ms = (time.perf_counter() - t0) * 1e3
    print(f"selftest frame t={t:.3f} ({rgb_p.name}), seg {ms:.0f} ms")
    n = gmap.size
    for gid, name in enumerate(GROUPS):
        frac = float((gmap == gid).sum()) / n
        if frac > 0.001:
            print(f"  {name:8s} {frac * 100:5.1f}%")
    print(f"  ignore   {float((gmap == 255).sum()) / n * 100:5.1f}%")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, help="finished run dir")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="output mp4 (default <run>/semantic_map.mp4)")
    ap.add_argument("--model", type=Path,
                    default=Path("third_party/models/segformer_b2_ade20k"),
                    help="segmentation model dir (b2 = quality default; "
                         "third_party/models/segformer_b0_ade20k = ~10x faster)")
    ap.add_argument("--speed", type=float, default=4.0,
                    help="playback vs real time: 1=actual run speed, N=N times faster")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--seg-interval", type=float, default=0.4,
                    help="segment one RGB frame every N dataset seconds")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--min-votes", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max-depth", type=float, default=4.0)
    ap.add_argument("--obstacle-max-depth", type=float, default=3.0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-lidar-gate", action="store_true",
                    help="camera-only semantics: skip LiDAR gating in the "
                         "raster and LiDAR walls in the PNG (cross-modal "
                         "ablation; outputs get a _nogate suffix)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-run segmentation even if a cache exists")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest(args.dataset, args.model)
    if args.run is None:
        ap.error("--run is required (or use --selftest)")

    run_semantic_pipeline(
        run_dir=args.run, dataset=args.dataset, model_dir=args.model,
        speed=args.speed, fps=args.fps, seg_interval=args.seg_interval,
        stride=args.stride, min_votes=args.min_votes,
        max_depth=args.max_depth, obstacle_max_depth=args.obstacle_max_depth,
        alpha=args.alpha, video=not args.no_video, refresh=args.refresh,
        lidar_gate=not args.no_lidar_gate, out_mp4=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
