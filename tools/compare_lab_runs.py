#!/usr/bin/env python3
"""
tools/compare_lab_runs.py

Extracts key metrics from two completed lab SLAM runs (Run A: baseline,
Run B: loop+GBA) and writes a structured JSON summary + a side-by-side
comparison table printed to stdout.

Usage:
    python3 -m tools.compare_lab_runs \
        --run-a visual_slam_outputs/lab_rgbd_run_2_A_baseline \
        --run-b visual_slam_outputs/lab_rgbd_run_2_B_loop_gba \
        --output visual_slam_outputs/lab_comparison/comparison_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def _load_frame_log(run_dir: Path) -> list[dict]:
    candidates = list(run_dir.glob("frame_log_*.csv"))
    if not candidates:
        return []
    return list(csv.DictReader(open(candidates[0])))


def _load_trajectory(run_dir: Path) -> np.ndarray | None:
    candidates = list(run_dir.glob("trajectory_*.txt"))
    if not candidates:
        return None
    poses = []
    with open(candidates[0]) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                poses.append([float(p) for p in parts[1:8]])  # tx ty tz qx qy qz qw
    return np.array(poses) if poses else None


def _traj_stats(traj: np.ndarray | None) -> dict:
    if traj is None or len(traj) == 0:
        return {"num_poses": 0, "path_length_m": None, "x_range_m": None, "z_range_m": None, "y_range_m": None}
    txyz = traj[:, :3]
    # path length: sum of incremental distances
    diffs = np.diff(txyz, axis=0)
    path_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    x_range = float(txyz[:, 0].max() - txyz[:, 0].min())
    z_range = float(txyz[:, 2].max() - txyz[:, 2].min())
    y_range = float(txyz[:, 1].max() - txyz[:, 1].min())
    return {
        "num_poses": len(traj),
        "path_length_m": round(path_len, 3),
        "x_range_m": round(x_range, 3),
        "z_range_m": round(z_range, 3),
        "y_range_m": round(y_range, 3),
    }


def _log_stats(log: list[dict]) -> dict:
    if not log:
        return {}
    num_ok = sum(1 for r in log if r.get("ok", "0") == "1")
    num_lost = sum(1 for r in log if r.get("state", "") == "LOST")
    kf_counts = [int(r["keyframes"]) for r in log if r.get("keyframes")]
    mp_counts = [int(r["points"]) for r in log if r.get("points")]
    tracked = [int(r["last_tracked"]) for r in log if r.get("last_tracked")]
    ba_mse = [float(r["last_ba_mse"]) for r in log if r.get("last_ba_mse") and r["last_ba_mse"] != ""]

    # GBA events: detect 0->1 transitions
    gba_events = []
    prev = "0"
    for r in log:
        cur = r.get("loop_global_ba_started", "0")
        if cur == "1" and prev == "0":
            gba_events.append({
                "frame": int(r.get("i", 0)),
                "timestamp": float(r.get("timestamp", 0)),
                "keyframes": int(r.get("keyframes", 0)),
                "map_points": int(r.get("points", 0)),
                "success": r.get("loop_global_ba_success", "0") == "1",
                "edges": int(r.get("loop_global_ba_edges", 0) or 0),
                "inliers": int(r.get("loop_global_ba_inliers", 0) or 0),
                "mse_after": float(r["loop_global_ba_mse_after"]) if r.get("loop_global_ba_mse_after") else None,
                "reason": r.get("loop_global_ba_reason", ""),
            })
        prev = cur

    # KF growth: first frame where max KF count is first reached
    final_kf = max(kf_counts) if kf_counts else 0
    kf_stabilize_frame = next((i for i, r in enumerate(log) if int(r.get("keyframes", 0)) == final_kf), -1)

    return {
        "total_frames": len(log),
        "frames_ok": num_ok,
        "frames_lost": num_lost,
        "tracking_success_rate": round(num_ok / len(log) * 100, 2) if log else 0,
        "final_keyframes": max(kf_counts) if kf_counts else 0,
        "final_map_points": max(mp_counts) if mp_counts else 0,
        "kf_stabilize_frame": kf_stabilize_frame,
        "mean_tracked_pts": round(float(np.mean(tracked)), 1) if tracked else None,
        "median_tracked_pts": round(float(np.median(tracked)), 1) if tracked else None,
        "min_tracked_pts": int(min(tracked)) if tracked else None,
        "mean_ba_mse": round(float(np.mean(ba_mse)), 4) if ba_mse else None,
        "median_ba_mse": round(float(np.median(ba_mse)), 4) if ba_mse else None,
        "gba_events": gba_events,
        "num_gba_events": len(gba_events),
        "num_gba_successful": sum(1 for e in gba_events if e["success"]),
    }


def _find_run_meta(run_dir: Path) -> dict:
    """Try to extract elapsed time etc. from a log file in the same parent dir."""
    log_files = list(run_dir.parent.glob(f"{run_dir.name}*.log"))
    meta = {}
    if log_files:
        with open(log_files[0]) as f:
            for line in f:
                line = line.strip()
                if "elapsed_sec" in line:
                    try:
                        meta["elapsed_sec"] = float(line.split(":")[1].strip())
                    except Exception:
                        pass
                if "avg_fps" in line:
                    try:
                        meta["avg_fps"] = float(line.split(":")[1].strip())
                    except Exception:
                        pass
    return meta


def extract_run_summary(run_dir: Path, label: str) -> dict:
    log = _load_frame_log(run_dir)
    traj = _load_trajectory(run_dir)
    meta = _find_run_meta(run_dir)

    ls = _log_stats(log)
    ts = _traj_stats(traj)

    # Check for map export artifacts
    has_ply = bool(list(run_dir.glob("map_points*.ply")))
    has_kf_json = bool(list(run_dir.glob("keyframes*.json")))
    has_graph_json = bool(list(run_dir.glob("keyframe_graph*.json")))

    return {
        "label": label,
        "run_dir": str(run_dir),
        "log_stats": ls,
        "traj_stats": ts,
        "meta": meta,
        "artifacts": {
            "has_map_points_ply": has_ply,
            "has_keyframes_json": has_kf_json,
            "has_graph_json": has_graph_json,
        },
    }


def print_comparison(a: dict, b: dict) -> None:
    la = a["log_stats"]
    lb = b["log_stats"]
    ta = a["traj_stats"]
    tb = b["traj_stats"]

    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    W = 30
    print()
    print("=" * 80)
    print("LAB RGB-D SLAM RUN COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<40} {'Run A (Baseline)':<18} {'Run B (Loop+GBA)':<18}")
    print("-" * 80)

    rows = [
        ("Total frames",         la.get("total_frames"),        lb.get("total_frames")),
        ("Frames OK",             la.get("frames_ok"),           lb.get("frames_ok")),
        ("Frames LOST",           la.get("frames_lost"),         lb.get("frames_lost")),
        ("Tracking success (%)",  la.get("tracking_success_rate"), lb.get("tracking_success_rate")),
        ("Final keyframes",       la.get("final_keyframes"),     lb.get("final_keyframes")),
        ("Final map points",      la.get("final_map_points"),    lb.get("final_map_points")),
        ("KF stabilize frame",    la.get("kf_stabilize_frame"),  lb.get("kf_stabilize_frame")),
        ("Mean tracked pts",      la.get("mean_tracked_pts"),    lb.get("mean_tracked_pts")),
        ("Min tracked pts",       la.get("min_tracked_pts"),     lb.get("min_tracked_pts")),
        ("Mean BA MSE (px²)",     la.get("mean_ba_mse"),         lb.get("mean_ba_mse")),
        ("Trajectory poses",      ta.get("num_poses"),           tb.get("num_poses")),
        ("Path length (m)",       ta.get("path_length_m"),       tb.get("path_length_m")),
        ("X span (m)",            ta.get("x_range_m"),           tb.get("x_range_m")),
        ("Z span (m)",            ta.get("z_range_m"),           tb.get("z_range_m")),
        ("Y span / height (m)",   ta.get("y_range_m"),           tb.get("y_range_m")),
        ("GBA events",            la.get("num_gba_events"),      lb.get("num_gba_events")),
        ("GBA successful",        la.get("num_gba_successful"),  lb.get("num_gba_successful")),
    ]
    for name, va, vb in rows:
        print(f"  {name:<38} {fmt(va):<18} {fmt(vb):<18}")

    print("-" * 80)
    print()
    print("  GBA Events (Run B):")
    for i, ev in enumerate(lb.get("gba_events", [])):
        inlier_rate = f"{ev['inliers']/max(ev['edges'],1)*100:.1f}%" if ev["edges"] else "N/A"
        print(f"    [{i+1}] frame={ev['frame']}  kf={ev['keyframes']}  mp={ev['map_points']}  "
              f"edges={ev['edges']}  inliers={ev['inliers']} ({inlier_rate})  "
              f"mse_after={fmt(ev['mse_after'])}  success={ev['success']}")
    print("=" * 80)
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True, type=Path, help="Baseline run output dir")
    parser.add_argument("--run-b", required=True, type=Path, help="Loop+GBA run output dir")
    parser.add_argument("--output", required=True, type=Path, help="JSON output path")
    args = parser.parse_args(argv)

    print(f"Extracting stats from Run A: {args.run_a}")
    a = extract_run_summary(args.run_a, "baseline_no_loop")
    print(f"Extracting stats from Run B: {args.run_b}")
    b = extract_run_summary(args.run_b, "loop_gba")

    print_comparison(a, b)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"run_a": a, "run_b": b}, f, indent=2, default=str)
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
