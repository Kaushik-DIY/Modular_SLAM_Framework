"""
Phase 1 verification probe.

Plays a TUM RGB-D sequence through FusionDataset + SoftSync, reports the
sync-success rate, and dumps a few synthesized scans alongside their RGB
frames so the synthetic LiDAR can be eyeballed against the scene.

Usage:
    .venv/bin/python tools/probe_fusion_dataset.py \
        --dataset datasets/tum/rgbd_dataset_freiburg1_room --num-frames 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slam_core.fusion.dataset import FusionDataset
from slam_core.fusion.sync import SoftSync


def _dump_scan_pair(out_dir: Path, idx: int, rgb, scan) -> None:
    fig, (ax_rgb, ax_scan) = plt.subplots(1, 2, figsize=(10, 4))

    ax_rgb.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    ax_rgb.set_title(f"RGB frame {idx}")
    ax_rgb.axis("off")

    if scan is not None and len(scan) > 0:
        ax_scan.scatter(scan[:, 1], scan[:, 0], s=4)  # x=left, y=forward
        ax_scan.set_title(f"Synth 2D LiDAR ({len(scan)} pts)")
    else:
        ax_scan.set_title("Synth 2D LiDAR (no returns)")
    ax_scan.set_xlabel("left [m]")
    ax_scan.set_ylabel("forward [m]")
    ax_scan.set_aspect("equal", adjustable="datalim")
    ax_scan.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"scan_pair_{idx:03d}.png", dpi=90)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 fusion dataset probe")
    parser.add_argument("--dataset", required=True, help="TUM RGB-D sequence dir")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--sync-tolerance", type=float, default=0.050)
    parser.add_argument("--num-beams", type=int, default=360)
    parser.add_argument("--noise-sigma", type=float, default=0.02)
    parser.add_argument("--dump-count", type=int, default=3,
                        help="number of RGB+scan pairs to dump")
    parser.add_argument("--out-dir", default="fusion_outputs/probe_p1")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = FusionDataset(
        args.dataset, num_beams=args.num_beams, noise_sigma=args.noise_sigma
    )
    sync = SoftSync(tolerance_s=args.sync_tolerance)

    total = 0          # finalized RGB-D frames
    matched = 0        # frames that got a scan inside the window
    dumped = 0
    pt_counts = []

    frames = list(dataset.iter_frames(max_frames=args.num_frames))
    for fr in frames:
        sync.push_rgbd(fr.rgb_t, fr.rgb, fr.depth)
        sync.push_scan(fr.scan_t, fr.scan)
        sample = sync.try_pop()
        while sample is not None:
            total += 1
            if sample.scan is not None:
                matched += 1
                pt_counts.append(len(sample.scan))
                if dumped < args.dump_count:
                    _dump_scan_pair(out_dir, total, sample.rgb, sample.scan)
                    dumped += 1
            sample = sync.try_pop()

    # Drain remaining buffered frames.
    sample = sync.try_pop(flush=True)
    while sample is not None:
        total += 1
        if sample.scan is not None:
            matched += 1
            pt_counts.append(len(sample.scan))
            if dumped < args.dump_count:
                _dump_scan_pair(out_dir, total, sample.rgb, sample.scan)
                dumped += 1
        sample = sync.try_pop(flush=True)

    rate = (matched / total * 100.0) if total else 0.0
    avg_pts = (sum(pt_counts) / len(pt_counts)) if pt_counts else 0.0

    print(f"dataset           : {args.dataset}")
    print(f"frames played     : {len(frames)}")
    print(f"frames finalized  : {total}")
    print(f"sync tolerance    : {args.sync_tolerance * 1000:.0f} ms")
    print(f"sync-success rate : {matched}/{total} ({rate:.1f}%)")
    print(f"avg scan points   : {avg_pts:.1f}")
    print(f"dumped pairs      : {dumped} -> {out_dir}")


if __name__ == "__main__":
    main()
