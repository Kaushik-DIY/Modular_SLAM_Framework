#!/usr/bin/env python3
"""
Main dataset-agnostic RGB-D SLAM runner.
This script loads a dataset, executes the pipeline, and writes run artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from tools.export_orbslam_map import export_orbslam_map
from tools.run_fr1_room_full_evaluation import (
    LOOP_DEBUG_COLUMNS,
    _append_loop_debug_records,
    _write_candidate_pair_reports,
    write_csv,
)
from visual_slam.orbslam.io import save_tum_trajectory
from visual_slam.orbslam.io.rgbd_dataset import (
    DATASET_TYPE_AUTO,
    DATASET_TYPE_LAB,
    DATASET_TYPE_TUM,
    detect_dataset_type,
    load_rgbd_associations,
    make_rgbd_camera,
    resolve_camera_metadata,
)
from visual_slam.orbslam.slam import Slam, SlamState, SensorType


FRAME_LOG_COLUMNS = [
    "i",
    "timestamp",
    "ok",
    "state",
    "keyframes",
    "points",
    "frames",
    "poses",
    "history",
    "last_tracked",
    "last_ba_mse",
    "lm_last_fused",
    "lm_last_triangulated",
    "loop_global_ba_started",
    "loop_global_ba_success",
    "loop_global_ba_reason",
    "loop_global_ba_edges",
    "loop_global_ba_inliers",
    "loop_global_ba_mse_after",
]


def _load_rgb(path: Path):
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load RGB image: {path}")
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


def _json_dump(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("_.-")
    return text or "unnamed"


def build_completed_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def build_standardized_output_stem(dataset_type: str, dataset_name: str, completed_timestamp: str) -> str:
    return f"{_slugify(dataset_type)}__{_slugify(dataset_name)}__completed_{completed_timestamp}"


def build_standardized_artifact_paths(output_dir: Path, dataset_type: str, dataset_name: str, completed_timestamp: str) -> dict:
    stem = build_standardized_output_stem(dataset_type, dataset_name, completed_timestamp)
    return {
        "stem": stem,
        "trajectory_file": output_dir / f"trajectory__{stem}.txt",
        "frame_log_file": output_dir / f"frame_log__{stem}.csv",
        "map_points_ply": output_dir / f"map_points__{stem}.ply",
        "keyframes_json": output_dir / f"keyframes__{stem}.json",
        "keyframe_graph_json": output_dir / f"keyframe_graph__{stem}.json",
        "effective_run_config_json": output_dir / f"effective_run_config__{stem}.json",
        "run_summary_json": output_dir / f"run_summary__{stem}.json",
        "loop_debug_file": output_dir / f"loop_debug_candidates__{stem}.csv",
    }


def _copy_if_exists(src: Path | None, dst: Path | None) -> str | None:
    if src is None or dst is None or not Path(src).exists():
        return None
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def build_effective_run_config(
    *,
    dataset: Path,
    dataset_name: str,
    dataset_type: str,
    output_dir: Path,
    camera_profile: str,
    camera_config: Path | None,
    associations: Path | None,
    camera_metadata: dict,
    feature_backend: str,
    enable_loop_closing: bool,
    enable_global_ba: bool,
    global_ba_after_loop: bool,
    global_ba_iterations: int,
    max_frames: int,
    start_index: int,
    print_every: int,
    loop_debug: bool,
    stop_after_loop_events: int,
    stop_after_accepted_loops: int,
    dump_loop_candidate_reports: bool,
    start_local_mapping_thread: bool,
    lm_wait_timeout: float,
    completed_timestamp: str | None = None,
    standardized_output_stem: str | None = None,
) -> dict:
    return {
        "dataset_path": str(dataset),
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "output_dir": str(output_dir),
        "completed_timestamp": completed_timestamp,
        "standardized_output_stem": standardized_output_stem,
        "camera_profile": camera_profile,
        "camera_config": str(camera_config) if camera_config is not None else None,
        "associations": str(associations) if associations is not None else None,
        "camera": {
            "source": camera_metadata.get("camera_source"),
            "sensor_type": camera_metadata.get("sensor_type"),
            "width": camera_metadata.get("width"),
            "height": camera_metadata.get("height"),
            "fps": camera_metadata.get("fps"),
            "fx": camera_metadata.get("fx"),
            "fy": camera_metadata.get("fy"),
            "cx": camera_metadata.get("cx"),
            "cy": camera_metadata.get("cy"),
            "distortion": camera_metadata.get("distortion"),
            "depth_map_factor": camera_metadata.get("depth_map_factor"),
            "depth_factor": camera_metadata.get("depth_factor"),
            "depth_threshold": camera_metadata.get("depth_threshold"),
            "depth_threshold_source": camera_metadata.get("depth_threshold_source"),
            "baseline_m": camera_metadata.get("baseline_m"),
            "baseline_source": camera_metadata.get("baseline_source"),
            "bf": camera_metadata.get("bf"),
        },
        "feature_backend": feature_backend,
        "enable_loop_closing": bool(enable_loop_closing),
        "enable_global_ba": bool(enable_global_ba),
        "global_ba_after_loop": bool(global_ba_after_loop),
        "global_ba_iterations": int(global_ba_iterations),
        "max_frames": int(max_frames),
        "start_index": int(start_index),
        "print_every": int(print_every),
        "loop_debug": bool(loop_debug),
        "stop_after_loop_events": int(stop_after_loop_events),
        "stop_after_accepted_loops": int(stop_after_accepted_loops),
        "dump_loop_candidate_reports": bool(dump_loop_candidate_reports),
        "start_local_mapping_thread": bool(start_local_mapping_thread),
        "lm_wait_timeout": float(lm_wait_timeout),
    }


def write_effective_run_config(output_dir: Path, config: dict) -> Path:
    return _json_dump(output_dir / "effective_run_config.json", config)


def build_run_summary(
    *,
    dataset_name: str,
    dataset_type: str,
    frames_attempted: int,
    tracking_ok_count: int,
    tracking_lost_count: int,
    errors: int,
    final_state: str,
    keyframes: int,
    map_points: int,
    trajectory_poses: int,
    elapsed_sec: float,
    avg_fps: float,
    feature_backend: str,
    enable_loop_closing: bool,
    enable_global_ba: bool,
    global_ba_after_loop: bool,
    loop_debug_events: int,
    accepted_loops: int,
    output_files: dict,
    completed_timestamp: str,
    standardized_output_stem: str,
) -> dict:
    return {
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "completed_timestamp": completed_timestamp,
        "standardized_output_stem": standardized_output_stem,
        "frames_attempted": int(frames_attempted),
        "tracking_ok_count": int(tracking_ok_count),
        "tracking_lost_count": int(tracking_lost_count),
        "errors": int(errors),
        "final_state": final_state,
        "keyframes": int(keyframes),
        "map_points": int(map_points),
        "trajectory_poses": int(trajectory_poses),
        "elapsed_sec": float(elapsed_sec),
        "avg_fps": float(avg_fps),
        "feature_backend": feature_backend,
        "loop_closing_enabled": bool(enable_loop_closing),
        "global_ba_enabled": bool(enable_global_ba),
        "global_ba_after_loop": bool(global_ba_after_loop),
        "loop_debug_events": int(loop_debug_events),
        "accepted_loops": int(accepted_loops),
        "final_keyframes": int(keyframes),
        "final_map_points": int(map_points),
        "trajectory_file": output_files.get("trajectory_file"),
        "frame_log_file": output_files.get("frame_log_file"),
        "map_points_ply": output_files.get("map_points_ply"),
        "keyframes_json": output_files.get("keyframes_json"),
        "keyframe_graph_json": output_files.get("keyframe_graph_json"),
        "effective_run_config_json": output_files.get("effective_run_config_json"),
        "loop_debug_file": output_files.get("loop_debug_file"),
        "standardized_trajectory_file": output_files.get("standardized_trajectory_file"),
        "standardized_frame_log_file": output_files.get("standardized_frame_log_file"),
        "standardized_map_points_ply": output_files.get("standardized_map_points_ply"),
        "standardized_keyframes_json": output_files.get("standardized_keyframes_json"),
        "standardized_keyframe_graph_json": output_files.get("standardized_keyframe_graph_json"),
        "standardized_effective_run_config_json": output_files.get("standardized_effective_run_config_json"),
        "standardized_loop_debug_file": output_files.get("standardized_loop_debug_file"),
    }


def write_run_summary(output_dir: Path, summary: dict) -> Path:
    return _json_dump(output_dir / "run_summary.json", summary)


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the final RGB-D ORB-SLAM pipeline on TUM or lab datasets.")
    parser.add_argument("dataset", type=Path, help="Path to the RGB-D dataset root.")
    parser.add_argument(
        "--dataset-type",
        choices=(DATASET_TYPE_AUTO, DATASET_TYPE_TUM, DATASET_TYPE_LAB),
        default=DATASET_TYPE_AUTO,
        help="Dataset layout selector. Use auto unless detection is ambiguous.",
    )
    parser.add_argument(
        "--camera-profile",
        default="auto",
        help="Camera profile for TUM datasets. Use auto to preserve the current Freiburg detection logic.",
    )
    parser.add_argument("--camera-config", type=Path, default=None, help="Optional camera.yaml for lab_rgbd datasets.")
    parser.add_argument("--associations", type=Path, default=None, help="Optional associations.txt override.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for trajectory, logs, and map export.")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process. Use 0 or -1 for full sequence.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument(
        "--feature-backend",
        choices=("opencv_orb", "pyslam_orb2", "auto"),
        default="auto",
        help="Feature extractor backend override.",
    )
    loop_group = parser.add_mutually_exclusive_group()
    loop_group.add_argument("--enable-loop-closing", action="store_true", help="Enable loop closing.")
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
    parser.add_argument("--start-local-mapping-thread", action="store_true")
    parser.add_argument("--lm-wait-timeout", type=float, default=0.5)
    return parser


def run_rgbd_slam(
    dataset: Path,
    output_dir: Path,
    *,
    dataset_type: str = DATASET_TYPE_AUTO,
    camera_profile: str = "auto",
    camera_config: Path | None = None,
    associations: Path | None = None,
    max_frames: int = 0,
    start_index: int = 0,
    print_every: int = 1,
    feature_backend: str = "auto",
    enable_loop_closing: bool = False,
    enable_global_ba: bool = False,
    global_ba_after_loop: bool = False,
    global_ba_iterations: int = 10,
    loop_debug: bool = False,
    stop_after_loop_events: int = 0,
    stop_after_accepted_loops: int = 0,
    dump_loop_candidate_reports: bool = False,
    start_local_mapping_thread: bool = False,
    lm_wait_timeout: float = 0.5,
) -> dict:
    dataset = Path(dataset).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    global_ba_after_loop = bool(global_ba_after_loop or enable_global_ba)

    effective_dataset_type = dataset_type
    if effective_dataset_type == DATASET_TYPE_AUTO:
        effective_dataset_type = detect_dataset_type(dataset)

    camera = make_rgbd_camera(
        dataset,
        dataset_type=effective_dataset_type,
        camera_profile=camera_profile,
        camera_config=camera_config,
    )
    camera_metadata = resolve_camera_metadata(
        dataset,
        dataset_type=effective_dataset_type,
        camera_profile=camera_profile,
        camera_config=camera_config,
    )
    dataset_name = str(camera_metadata.get("dataset_name") or dataset.name)

    frames = load_rgbd_associations(dataset, associations=associations)
    if len(frames) == 0:
        raise RuntimeError(f"No RGB-D associations found in {dataset}")

    if start_index > 0:
        frames = frames[start_index:]
    if max_frames > 0:
        frames = frames[:max_frames]

    selected_backend = None if feature_backend in {None, "auto"} else feature_backend
    feature_tracker_config = None if selected_backend is None else {"extractor_backend": selected_backend}

    run_config = build_effective_run_config(
        dataset=dataset,
        dataset_name=dataset_name,
        dataset_type=effective_dataset_type,
        output_dir=output_dir,
        camera_profile=camera_profile,
        camera_config=Path(camera_metadata["camera_source"]).resolve() if effective_dataset_type == DATASET_TYPE_LAB else camera_config,
        associations=associations,
        camera_metadata=camera_metadata,
        feature_backend=feature_backend,
        enable_loop_closing=enable_loop_closing,
        enable_global_ba=enable_global_ba,
        global_ba_after_loop=global_ba_after_loop,
        global_ba_iterations=global_ba_iterations,
        max_frames=max_frames,
        start_index=start_index,
        print_every=print_every,
        loop_debug=loop_debug,
        stop_after_loop_events=stop_after_loop_events,
        stop_after_accepted_loops=stop_after_accepted_loops,
        dump_loop_candidate_reports=dump_loop_candidate_reports,
        start_local_mapping_thread=start_local_mapping_thread,
        lm_wait_timeout=lm_wait_timeout,
    )
    effective_run_config_path = write_effective_run_config(output_dir, run_config)

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
    print("Final RGB-D ORB-SLAM run")
    print("=" * 80)
    print(f"Dataset:             {dataset}")
    print(f"Dataset type:        {effective_dataset_type}")
    print(f"Output:              {output_dir}")
    print(f"Frames loaded:       {len(frames)}")
    print(f"Camera source:       {camera_metadata.get('camera_source')}")
    print(
        f"Camera intrinsics:   fx={camera.fx:.3f}, fy={camera.fy:.3f}, "
        f"cx={camera.cx:.3f}, cy={camera.cy:.3f}"
    )
    print(f"Image size:          {camera.width}x{camera.height}")
    print(f"Depth map factor:    {camera_metadata.get('depth_map_factor')}")
    print(f"Depth factor:        {camera.depth_factor}")
    print(
        f"Baseline (m):        {camera_metadata.get('baseline_m')} "
        f"[{camera_metadata.get('baseline_source', 'unspecified')}]"
    )
    print(
        f"Depth threshold:     {camera_metadata.get('depth_threshold')} "
        f"[{camera_metadata.get('depth_threshold_source', 'unspecified')}]"
    )
    print(f"bf:                  {camera_metadata.get('bf')}")
    print(f"Feature backend:     {feature_backend}")
    print(f"Loop closing:        {'enabled' if enable_loop_closing else 'disabled'}")
    print(f"Global BA:           {'enabled' if enable_global_ba else 'disabled'}")
    print(f"LM threading:        {'enabled (wait='+str(lm_wait_timeout)+'s)' if threaded_lm else 'disabled (sequential)'}")
    print("=" * 80)

    start_t = time.time()
    num_ok = 0
    num_lost = 0
    num_errors = 0
    accepted_loop_count = 0
    stop_requested = False

    per_frame_log: list[dict] = []
    loop_debug_rows: list[dict] = []
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
    poses = [pose for pose, _ in ok_pairs]
    timestamps = [stamp for _, stamp in ok_pairs]

    traj_file = output_dir / f"trajectory_{dataset_name}.txt"
    save_tum_trajectory(poses, timestamps, traj_file)

    frame_log_file = output_dir / f"frame_log_{dataset_name}.csv"
    write_csv(frame_log_file, per_frame_log, FRAME_LOG_COLUMNS)

    loop_debug_file = None
    if loop_debug:
        loop_debug_file = output_dir / "loop_debug_candidates.csv"
        write_csv(loop_debug_file, loop_debug_rows, LOOP_DEBUG_COLUMNS)

    map_export = export_orbslam_map(slam, output_dir)
    completed_timestamp = build_completed_timestamp()
    standardized_paths = build_standardized_artifact_paths(
        output_dir,
        effective_dataset_type,
        dataset_name,
        completed_timestamp,
    )
    standardized_output_stem = standardized_paths["stem"]

    kf_consistency = {"n_checked": 0}
    if hasattr(slam, "compute_kf_trajectory_consistency"):
        kf_consistency = slam.compute_kf_trajectory_consistency()

    run_config["completed_timestamp"] = completed_timestamp
    run_config["standardized_output_stem"] = standardized_output_stem
    effective_run_config_path = write_effective_run_config(output_dir, run_config)
    standardized_effective_config = _copy_if_exists(
        effective_run_config_path,
        standardized_paths["effective_run_config_json"],
    )

    standardized_output_files = {
        "trajectory_file": _copy_if_exists(traj_file, standardized_paths["trajectory_file"]),
        "frame_log_file": _copy_if_exists(frame_log_file, standardized_paths["frame_log_file"]),
        "map_points_ply": _copy_if_exists(Path(map_export["map_points_ply"]), standardized_paths["map_points_ply"]),
        "keyframes_json": _copy_if_exists(Path(map_export["keyframes_json"]), standardized_paths["keyframes_json"]),
        "keyframe_graph_json": _copy_if_exists(Path(map_export["keyframe_graph_json"]), standardized_paths["keyframe_graph_json"]),
        "effective_run_config_json": standardized_effective_config,
        "loop_debug_file": _copy_if_exists(loop_debug_file, standardized_paths["loop_debug_file"]) if loop_debug_file is not None else None,
    }
    output_files = {
        "trajectory_file": str(traj_file),
        "frame_log_file": str(frame_log_file),
        "map_points_ply": map_export["map_points_ply"],
        "keyframes_json": map_export["keyframes_json"],
        "keyframe_graph_json": map_export["keyframe_graph_json"],
        "effective_run_config_json": str(effective_run_config_path),
        "loop_debug_file": str(loop_debug_file) if loop_debug_file is not None else None,
        "standardized_trajectory_file": standardized_output_files["trajectory_file"],
        "standardized_frame_log_file": standardized_output_files["frame_log_file"],
        "standardized_map_points_ply": standardized_output_files["map_points_ply"],
        "standardized_keyframes_json": standardized_output_files["keyframes_json"],
        "standardized_keyframe_graph_json": standardized_output_files["keyframe_graph_json"],
        "standardized_effective_run_config_json": standardized_output_files["effective_run_config_json"],
        "standardized_loop_debug_file": standardized_output_files["loop_debug_file"],
    }
    summary = build_run_summary(
        dataset_name=dataset_name,
        dataset_type=effective_dataset_type,
        frames_attempted=len(frames),
        tracking_ok_count=num_ok,
        tracking_lost_count=num_lost,
        errors=num_errors,
        final_state=_state_name(slam.get_tracking_state()),
        keyframes=slam.map.num_keyframes(),
        map_points=slam.map.num_points(),
        trajectory_poses=len(poses),
        elapsed_sec=elapsed,
        avg_fps=len(frames) / max(elapsed, 1e-9),
        feature_backend=feature_backend,
        enable_loop_closing=enable_loop_closing,
        enable_global_ba=enable_global_ba,
        global_ba_after_loop=global_ba_after_loop,
        loop_debug_events=len(loop_debug_rows),
        accepted_loops=accepted_loop_count,
        output_files=output_files,
        completed_timestamp=completed_timestamp,
        standardized_output_stem=standardized_output_stem,
    )
    summary_path = write_run_summary(output_dir, summary)
    standardized_summary_path = _copy_if_exists(summary_path, standardized_paths["run_summary_json"])
    summary["standardized_run_summary_json"] = standardized_summary_path
    summary_path = write_run_summary(output_dir, summary)
    standardized_summary_path = _copy_if_exists(summary_path, standardized_paths["run_summary_json"])

    print("=" * 80)
    print("RUN SUMMARY")
    print("=" * 80)
    print(f"frames_attempted:     {summary['frames_attempted']}")
    print(f"tracking_ok_count:    {summary['tracking_ok_count']}")
    print(f"tracking_lost_count:  {summary['tracking_lost_count']}")
    print(f"errors:               {summary['errors']}")
    print(f"final_state:          {summary['final_state']}")
    print(f"keyframes:            {summary['keyframes']}")
    print(f"map_points:           {summary['map_points']}")
    print(f"trajectory_poses:     {summary['trajectory_poses']}")
    print(f"elapsed_sec:          {summary['elapsed_sec']:.3f}")
    print(f"avg_fps:              {summary['avg_fps']:.2f}")
    print(f"trajectory_file:      {traj_file}")
    print(f"frame_log_file:       {frame_log_file}")
    print(f"map_points_ply:       {map_export['map_points_ply']}")
    print(f"keyframes_json:       {map_export['keyframes_json']}")
    print(f"keyframe_graph_json:  {map_export['keyframe_graph_json']}")
    print(f"effective_config:     {effective_run_config_path}")
    print(f"completed_timestamp:  {completed_timestamp}")
    print(f"standardized_stem:    {standardized_output_stem}")
    print(f"standardized_summary: {standardized_summary_path}")
    if loop_debug_file is not None:
        print(f"loop_debug_file:      {loop_debug_file}")
        print(f"loop_debug_events:    {len(loop_debug_rows)}")
        print(f"accepted_loops:       {accepted_loop_count}")
    if kf_consistency.get("n_checked", 0) > 0:
        print(
            f"kf_traj_consistency:  n={kf_consistency['n_checked']} "
            f"max={kf_consistency['max_diff_m']:.4f}m "
            f"median={kf_consistency['median_diff_m']:.4f}m"
        )
    print("=" * 80)

    slam.shutdown()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = create_arg_parser()
    args = parser.parse_args(argv)

    enable_loop_closing = bool(args.enable_loop_closing and not args.disable_loop_closing)
    enable_global_ba = bool(args.enable_global_ba and not args.disable_global_ba)

    run_rgbd_slam(
        dataset=args.dataset,
        output_dir=args.output,
        dataset_type=args.dataset_type,
        camera_profile=args.camera_profile,
        camera_config=args.camera_config,
        associations=args.associations,
        max_frames=args.max_frames,
        start_index=args.start_index,
        print_every=args.print_every,
        feature_backend=args.feature_backend,
        enable_loop_closing=enable_loop_closing,
        enable_global_ba=enable_global_ba,
        global_ba_after_loop=bool(args.global_ba_after_loop),
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
