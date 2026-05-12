#!/usr/bin/env python3
"""
Short TUM RGB-D smoke runner.
This script validates end-to-end frame ingestion and basic SLAM execution on a short sequence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from tools.export_orbslam_map import export_orbslam_map
from tools.run_fr1_room_full_evaluation import (
    LOOP_DEBUG_COLUMNS,
    _append_loop_debug_records,
    _write_candidate_pair_reports,
    write_csv,
)
from visual_slam.orbslam.io import (
    load_tum_rgbd_associations,
    save_tum_trajectory,
)
from visual_slam.orbslam.io.rgbd_dataset import detect_dataset_type, make_rgbd_camera
from visual_slam.orbslam.slam import Slam, SensorType, SlamState


def _load_rgb(path: Path):
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise FileNotFoundError(f"Could not load RGB image: {path}")

    # Keep RGB frames in OpenCV BGR layout for the feature front-end.
    return img_bgr


def _load_depth(path: Path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise FileNotFoundError(f"Could not load depth image: {path}")

    return depth


def _state_name(state):
    try:
        return state.name
    except Exception:
        return str(state)


def _finite_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


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
    dataset = Path(dataset).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_tum_rgbd_associations(dataset)

    if len(frames) == 0:
        raise RuntimeError(f"No RGB-D associations found in {dataset}")

    frames = frames[start_index:]

    if max_frames > 0:
        frames = frames[:max_frames]

    dataset_type = detect_dataset_type(dataset)
    camera = make_rgbd_camera(dataset)
    feature_tracker_config = None
    if feature_backend:
        feature_tracker_config = {"extractor_backend": feature_backend}

    slam = Slam(
        camera=camera,
        sensor_type=SensorType.RGBD,
        headless=True,
        start_local_mapping_thread=start_local_mapping_thread,
        feature_tracker_config=feature_tracker_config,
        enable_loop_closing=enable_loop_closing,
        enable_global_ba=enable_global_ba,
        global_ba_after_loop=global_ba_after_loop,
        global_ba_iterations=global_ba_iterations,
    )
    threaded_lm = slam.start_local_mapping_thread

    print("=" * 80)
    print("ORB-SLAM RGB-D smoke run")
    print("=" * 80)
    print(f"Dataset:       {dataset}")
    print(f"Dataset type:  {dataset_type}")
    print(f"Output:        {output_dir}")
    print(f"Frames loaded: {len(frames)}")
    print(f"Feature backend: {feature_backend or 'default'}")
    print(f"Loop closing:  {'enabled' if enable_loop_closing else 'disabled'}")
    print(f"Global BA:     {'enabled' if enable_global_ba else 'disabled'}")
    print(f"LM threading:  {'enabled (wait='+str(lm_wait_timeout)+'s)' if threaded_lm else 'disabled (sequential)'}")
    print(f"Camera:        fx={camera.fx:.3f}, fy={camera.fy:.3f}, cx={camera.cx:.3f}, cy={camera.cy:.3f}")
    print(f"Depth factor:  {camera.depth_factor}  (1/depth_map_factor, units: m/raw)")
    print("=" * 80)

    start_t = time.time()

    num_ok = 0
    num_lost = 0
    num_errors = 0

    per_frame_log = []
    loop_debug_rows = []
    accepted_loop_count = 0
    stop_requested = False
    pair_report_dir = output_dir / "loop_candidate_pair_reports"

    for i, entry in enumerate(frames):
        frame_idx = start_index + i

        try:
            rgb = _load_rgb(entry.rgb_path)
            depth = _load_depth(entry.depth_path)

            ok = slam.track(
                img=rgb,
                img_right=None,
                depth=depth,
                img_id=frame_idx,
                timestamp=entry.timestamp,
            )

            # Sequential mode advances local mapping explicitly after each frame.
            if not threaded_lm:
                while slam.local_mapping.queue_size() > 0:
                    slam.local_mapping.step()
            loop_closing = getattr(slam, "loop_closing", None)
            if threaded_lm:
                slam.local_mapping.wait_idle(timeout=lm_wait_timeout)
            while loop_closing is not None and loop_closing.queue_size() > 0:
                event_start = len(loop_debug_rows) + 1
                accepted_loop = bool(loop_closing.step())
                loop_diag_current = loop_closing.last_diagnostics
                if loop_debug:
                    _append_loop_debug_records(loop_debug_rows, loop_diag_current)
                if dump_loop_candidate_reports:
                    _write_candidate_pair_reports(
                        pair_report_dir,
                        loop_diag_current,
                        event_start=event_start,
                        dump_all=True,
                    )
                if accepted_loop:
                    accepted_loop_count += 1
                if stop_after_loop_events > 0 and len(loop_debug_rows) >= int(stop_after_loop_events):
                    stop_requested = True
                    break
                if stop_after_accepted_loops > 0 and accepted_loop_count >= int(stop_after_accepted_loops):
                    stop_requested = True
                    break

            state = slam.get_tracking_state()

            if ok and state == SlamState.OK:
                num_ok += 1
            elif state == SlamState.LOST:
                num_lost += 1

            n_kf = slam.map.num_keyframes()
            n_mp = slam.map.num_points()
            n_frames = slam.map.num_frames()

            n_pose = len(slam.tracking.poses)
            n_hist = len(slam.tracking.tracking_history.timestamps)

            mean_pose_opt_chi2_error = _finite_or_none(slam.tracking.mean_pose_opt_chi2_error)
            loop_diag = getattr(getattr(slam, "loop_closing", None), "last_diagnostics", None)

            row = {
                "i": frame_idx,
                "timestamp": entry.timestamp,
                "ok": bool(ok),
                "state": _state_name(state),
                "keyframes": n_kf,
                "points": n_mp,
                "frames": n_frames,
                "poses": n_pose,
                "history": n_hist,
                "last_tracked": slam.tracking.num_matched_map_points,
                "last_ba_mse": mean_pose_opt_chi2_error,
                "lm_last_fused": slam.local_mapping.last_num_fused_points,
                "lm_last_triangulated": slam.local_mapping.last_num_triangulated_points,
                "loop_global_ba_started": bool(getattr(loop_diag, "global_ba_started", False)),
                "loop_global_ba_success": bool(getattr(loop_diag, "global_ba_success", False)),
                "loop_global_ba_reason": getattr(loop_diag, "global_ba_reason", ""),
                "loop_global_ba_edges": int(getattr(loop_diag, "global_ba_num_edges", 0)),
                "loop_global_ba_inliers": int(getattr(loop_diag, "global_ba_num_inliers", 0)),
                "loop_global_ba_mse_after": _finite_or_none(getattr(loop_diag, "global_ba_mean_error_after", None)),
            }
            per_frame_log.append(row)

            if print_every > 0 and (i % print_every == 0 or i == len(frames) - 1):
                print(
                    f"[{i+1:04d}/{len(frames):04d}] "
                    f"idx={frame_idx:05d} "
                    f"state={row['state']} ok={row['ok']} "
                    f"kf={n_kf} mp={n_mp} "
                    f"tracked={row['last_tracked']} "
                    f"ba_mse={row['last_ba_mse'] if row['last_ba_mse'] is not None else 'NA'}"
                )

        except Exception as exc:
            num_errors += 1
            print(f"[ERROR] frame_idx={frame_idx} timestamp={entry.timestamp:.6f}: {type(exc).__name__}: {exc}")

            # Fail fast here so short validation runs expose runtime issues clearly.
            raise

        if stop_requested:
            print("[STOP] loop diagnostic stop condition reached")
            break

    elapsed = time.time() - start_t

    trajectory = slam.get_final_trajectory()
    ok_pairs = [
        (pose, ts)
        for pose, ts, state in zip(
            trajectory["poses"],
            trajectory["timestamps"],
            trajectory["slam_states"],
        )
        if state == SlamState.OK
    ]
    poses = [p for p, _ in ok_pairs]
    timestamps = [t for _, t in ok_pairs]

    traj_file = output_dir / f"trajectory_{dataset.name}_smoke.txt"
    if len(poses) > 0:
        save_tum_trajectory(poses, timestamps, traj_file)

    log_file = output_dir / f"frame_log_{dataset.name}_smoke.csv"
    with open(log_file, "w") as f:
        f.write(
            "i,timestamp,ok,state,keyframes,points,frames,poses,history,"
            "last_tracked,last_ba_mse,lm_last_fused,lm_last_triangulated,"
            "loop_global_ba_started,loop_global_ba_success,loop_global_ba_reason,"
            "loop_global_ba_edges,loop_global_ba_inliers,loop_global_ba_mse_after\n"
        )

        for row in per_frame_log:
            f.write(
                f"{row['i']},{row['timestamp']:.6f},{int(row['ok'])},{row['state']},"
                f"{row['keyframes']},{row['points']},{row['frames']},"
                f"{row['poses']},{row['history']},"
                f"{row['last_tracked']},{row['last_ba_mse'] if row['last_ba_mse'] is not None else ''},"
                f"{row['lm_last_fused']},{row['lm_last_triangulated']},"
                f"{int(row['loop_global_ba_started'])},{int(row['loop_global_ba_success'])},"
                f"{row['loop_global_ba_reason']},{row['loop_global_ba_edges']},"
                f"{row['loop_global_ba_inliers']},"
                f"{row['loop_global_ba_mse_after'] if row['loop_global_ba_mse_after'] is not None else ''}\n"
            )
    if loop_debug:
        write_csv(output_dir / "loop_debug_candidates.csv", loop_debug_rows, LOOP_DEBUG_COLUMNS)

    map_export = export_orbslam_map(slam, output_dir)

    print("=" * 80)
    print("SMOKE SUMMARY")
    print("=" * 80)
    print(f"frames_attempted:     {len(frames)}")
    print(f"tracking_ok_count:    {num_ok}")
    print(f"tracking_lost_count:  {num_lost}")
    print(f"errors:               {num_errors}")
    print(f"final_state:          {_state_name(slam.get_tracking_state())}")
    print(f"final_keyframes:      {slam.map.num_keyframes()}")
    print(f"final_map_points:     {slam.map.num_points()}")
    print(f"final_frames:         {slam.map.num_frames()}")
    print(f"trajectory_poses:     {len(poses)}")
    print(f"elapsed_sec:          {elapsed:.3f}")
    print(f"avg_fps:              {len(frames) / max(elapsed, 1e-9):.2f}")
    print(f"trajectory_file:      {traj_file}")
    print(f"frame_log_file:       {log_file}")
    print(f"map_points_ply:       {map_export['map_points_ply']}")
    print(f"keyframes_json:       {map_export['keyframes_json']}")
    print(f"keyframe_graph_json:  {map_export['keyframe_graph_json']}")
    print(f"exported_map_points:  {map_export['num_exported_points']}")
    print(f"exported_keyframes:   {map_export['num_exported_keyframes']}")
    print(f"loop_edges:           {map_export['num_loop_edges']}")
    kf_consistency = slam.compute_kf_trajectory_consistency()
    if kf_consistency["n_checked"] > 0:
        print(f"kf_traj_consistency:  n={kf_consistency['n_checked']} "
              f"max={kf_consistency['max_diff_m']:.4f}m "
              f"median={kf_consistency['median_diff_m']:.4f}m")
    if loop_debug:
        print(f"loop_debug_file:      {output_dir / 'loop_debug_candidates.csv'}")
        print(f"loop_debug_events:    {len(loop_debug_rows)}")
        print(f"accepted_loops:       {accepted_loop_count}")
    print("=" * 80)

    slam.shutdown()

    # Keep the smoke-run acceptance rules strict and easy to inspect.
    if slam.map.num_keyframes() < 1:
        raise RuntimeError("Smoke failed: no keyframe was created.")

    if slam.map.num_points() < 100:
        raise RuntimeError("Smoke failed: fewer than 100 map points were created.")

    if len(poses) < 1:
        raise RuntimeError("Smoke failed: no trajectory pose was stored.")

    return slam


def main():
    parser = argparse.ArgumentParser(
        description="Run the ORB RGB-D smoke test on a TUM RGB-D sequence."
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
        help="Run local mapping on a background thread (faster on multi-core).",
    )
    parser.add_argument(
        "--lm-wait-timeout",
        type=float,
        default=0.5,
        help="Seconds the tracker waits for LM idle in threaded mode (default 0.5).",
    )

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
