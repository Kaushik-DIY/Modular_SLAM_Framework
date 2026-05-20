#!/usr/bin/env python3
"""
Legacy TUM RGB-D smoke runner wrapper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from visual_slam.orbslam.run_rgbd_slam import run_rgbd_slam


def run_tum_rgbd_smoke(
    dataset: Path,
    output_dir: Path,
    max_frames: int = 30,
    start_index: int = 0,
    print_every: int = 1,
    feature_backend: str | None = None,
    enable_loop_closing: bool = False,
    enable_global_ba: bool = False,
    global_ba_after_loop: bool | None = None,
    global_ba_iterations: int = 10,
    loop_debug: bool = False,
    stop_after_loop_events: int = 0,
    stop_after_accepted_loops: int = 0,
    dump_loop_candidate_reports: bool = False,
    start_local_mapping_thread: bool = False,
    lm_wait_timeout: float = 0.5,
):
    print("run_tum_rgbd_smoke.py is legacy; forwarding to run_rgbd_slam.py")
    return run_rgbd_slam(
        dataset=dataset,
        output_dir=output_dir,
        dataset_type="tum_rgbd",
        camera_profile="auto",
        max_frames=max_frames,
        start_index=start_index,
        print_every=print_every,
        feature_backend=feature_backend or "auto",
        enable_loop_closing=enable_loop_closing,
        enable_global_ba=enable_global_ba,
        global_ba_after_loop=bool(enable_global_ba if global_ba_after_loop is None else global_ba_after_loop),
        global_ba_iterations=global_ba_iterations,
        loop_debug=loop_debug,
        stop_after_loop_events=stop_after_loop_events,
        stop_after_accepted_loops=stop_after_accepted_loops,
        dump_loop_candidate_reports=dump_loop_candidate_reports,
        start_local_mapping_thread=start_local_mapping_thread,
        lm_wait_timeout=lm_wait_timeout,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the legacy ORB RGB-D smoke wrapper on a TUM RGB-D sequence."
    )
    parser.add_argument("dataset", type=Path, help="Path to TUM RGB-D sequence folder")
    parser.add_argument("--output", type=Path, default=Path("visual_slam_outputs/orbslam_tum_smoke"))
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument(
        "--feature-backend",
        choices=("opencv_orb", "pyslam_orb2", "auto"),
        default=None,
        help="Optional extractor backend override for this smoke run.",
    )
    loop_group = parser.add_mutually_exclusive_group()
    loop_group.add_argument("--enable-loop-closing", action="store_true", help="Enable RGB-D loop closing.")
    loop_group.add_argument("--disable-loop-closing", action="store_true", help="Disable loop closing.")
    gba_group = parser.add_mutually_exclusive_group()
    gba_group.add_argument("--enable-global-ba", action="store_true", help="Enable loop-triggered Global BA.")
    gba_group.add_argument("--disable-global-ba", action="store_true", help="Disable loop-triggered Global BA.")
    parser.add_argument("--global-ba-after-loop", action="store_true", help="Run Global BA after accepted loop closures.")
    parser.add_argument("--global-ba-iterations", type=int, default=10)
    parser.add_argument("--loop-debug", action="store_true")
    parser.add_argument("--stop-after-loop-events", type=int, default=0)
    parser.add_argument("--stop-after-accepted-loops", type=int, default=0)
    parser.add_argument("--dump-loop-candidate-reports", action="store_true")
    parser.add_argument(
        "--start-local-mapping-thread",
        action="store_true",
        help="Run local mapping on a background thread.",
    )
    parser.add_argument(
        "--lm-wait-timeout",
        type=float,
        default=0.5,
        help="Seconds the tracker waits for LM idle in threaded mode.",
    )

    args = parser.parse_args(argv)
    run_tum_rgbd_smoke(
        dataset=args.dataset,
        output_dir=args.output,
        max_frames=args.max_frames,
        start_index=args.start_index,
        print_every=args.print_every,
        feature_backend=args.feature_backend,
        enable_loop_closing=bool(args.enable_loop_closing and not args.disable_loop_closing),
        enable_global_ba=bool(args.enable_global_ba and not args.disable_global_ba),
        global_ba_after_loop=bool(args.global_ba_after_loop or args.enable_global_ba),
        global_ba_iterations=int(args.global_ba_iterations),
        loop_debug=bool(args.loop_debug),
        stop_after_loop_events=int(args.stop_after_loop_events),
        stop_after_accepted_loops=int(args.stop_after_accepted_loops),
        dump_loop_candidate_reports=bool(args.dump_loop_candidate_reports),
        start_local_mapping_thread=bool(args.start_local_mapping_thread),
        lm_wait_timeout=float(args.lm_wait_timeout),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
