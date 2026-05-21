#!/usr/bin/env python3
"""
tools/validate_orbslam_pyslam_port.py

Validation harness for the pySLAM-aligned ORB/RGB-D port.

Run from repository root:

    python tools/validate_orbslam_pyslam_port.py \
        --dataset "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" \
        --output "$HOME/slam_ws/visual_slam_outputs/codex_validation"

This script intentionally checks behavior, not only unit-test success.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class ReprojectionStats:
    label: str
    matched_slots: int
    usable: int
    bad_depth: int
    nonfinite: int
    bad_points: int
    chi2_median: float
    chi2_p90: float
    chi2_p99: float
    chi2_max: float
    inlier_count: int
    z_median: float
    z_min: float
    z_max: float


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    print("=" * 100)
    print("$", " ".join(cmd))
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    result = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=merged_env)
    print(result.stdout)
    return result


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("inf")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def reprojection_stats(frame_or_kf, label: str) -> ReprojectionStats:
    from visual_slam.orbslam.slam.config_parameters import Parameters

    Tcw = np.asarray(frame_or_kf.Tcw(), dtype=np.float64)
    Rcw = Tcw[:3, :3]
    tcw = Tcw[:3, 3]

    fx = float(frame_or_kf.camera.fx)
    fy = float(frame_or_kf.camera.fy)
    cx = float(frame_or_kf.camera.cx)
    cy = float(frame_or_kf.camera.cy)

    points = getattr(frame_or_kf, "points", [])
    kps = getattr(frame_or_kf, "kpsu", None)
    if kps is None or len(kps) == 0:
        kps = getattr(frame_or_kf, "kps", [])

    chi2_values: list[float] = []
    z_values: list[float] = []
    bad_depth = 0
    nonfinite = 0
    bad_points = 0

    for idx, p in enumerate(points):
        if p is None:
            continue
        if hasattr(p, "is_bad") and p.is_bad():
            bad_points += 1
            continue
        if idx >= len(kps):
            continue

        try:
            pw = np.asarray(p.get_position(), dtype=np.float64).reshape(3)
        except Exception:
            continue

        if not np.all(np.isfinite(pw)):
            nonfinite += 1
            continue

        pc = Rcw @ pw + tcw
        if not np.all(np.isfinite(pc)):
            nonfinite += 1
            continue

        z = float(pc[2])
        if z <= Parameters.kMinDepth:
            bad_depth += 1
            continue

        u = fx * pc[0] / z + cx
        v = fy * pc[1] / z + cy
        if not np.isfinite(u) or not np.isfinite(v):
            nonfinite += 1
            continue

        obs_u, obs_v = kps[idx].pt
        chi2 = float((u - obs_u) ** 2 + (v - obs_v) ** 2)
        chi2_values.append(chi2)
        z_values.append(z)

    matched_slots = sum(1 for p in points if p is not None)

    return ReprojectionStats(
        label=label,
        matched_slots=matched_slots,
        usable=len(chi2_values),
        bad_depth=bad_depth,
        nonfinite=nonfinite,
        bad_points=bad_points,
        chi2_median=median(chi2_values) if chi2_values else float("inf"),
        chi2_p90=percentile(chi2_values, 90),
        chi2_p99=percentile(chi2_values, 99),
        chi2_max=max(chi2_values) if chi2_values else float("inf"),
        inlier_count=sum(1 for v in chi2_values if v <= 5.991),
        z_median=median(z_values) if z_values else float("inf"),
        z_min=min(z_values) if z_values else float("inf"),
        z_max=max(z_values) if z_values else float("inf"),
    )


def print_stats(stats: ReprojectionStats) -> None:
    print(
        f"[{stats.label}] matched={stats.matched_slots} usable={stats.usable} "
        f"bad_depth={stats.bad_depth} nonfinite={stats.nonfinite} bad_points={stats.bad_points}"
    )
    print(
        f"[{stats.label}] chi2 median={stats.chi2_median:.3f} "
        f"p90={stats.chi2_p90:.3f} p99={stats.chi2_p99:.3f} max={stats.chi2_max:.3f} "
        f"inliers={stats.inlier_count}/{stats.usable} "
        f"z median={stats.z_median:.3f} min={stats.z_min:.6f} max={stats.z_max:.3f}"
    )


def check_local_ba_consistency(dataset: Path) -> bool:
    from visual_slam.orbslam.io import load_tum_rgbd_associations, make_tum_rgbd_camera
    from visual_slam.orbslam.slam import Slam, SensorType

    print("=" * 100)
    print("LOCAL BA CONSISTENCY CHECK")

    frames = load_tum_rgbd_associations(dataset)[:2]
    camera = make_tum_rgbd_camera(dataset.name)

    slam = Slam(
        camera=camera,
        sensor_type=SensorType.RGBD,
        headless=True,
        start_local_mapping_thread=False,
    )

    for i, entry in enumerate(frames):
        rgb = cv2.imread(str(entry.rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(entry.depth_path), cv2.IMREAD_UNCHANGED)
        ok = slam.track(img=rgb, img_right=None, depth=depth, img_id=i, timestamp=entry.timestamp)
        print(
            f"[TRACK] frame={i} ok={ok} state={slam.get_tracking_state().name} "
            f"kfs={slam.map.num_keyframes()} points={slam.map.num_points()} "
            f"queue={slam.local_mapping.queue_size()} tracked={slam.tracking.num_matched_map_points}"
        )
        if not ok:
            print("FAIL: first two frames must track before local BA consistency can be evaluated.")
            return False

    before_frame = reprojection_stats(slam.map.get_frame(-1), "FRAME1_BEFORE_LOCAL_MAPPING")
    before_kf = reprojection_stats(slam.map.get_last_keyframe(), "KF1_BEFORE_LOCAL_MAPPING")
    print_stats(before_frame)
    print_stats(before_kf)

    while slam.local_mapping.queue_size() > 0:
        slam.local_mapping.step()

    after_frame = reprojection_stats(slam.map.get_frame(-1), "FRAME1_AFTER_LOCAL_MAPPING")
    after_kf = reprojection_stats(slam.map.get_last_keyframe(), "KF1_AFTER_LOCAL_MAPPING")
    print_stats(after_frame)
    print_stats(after_kf)

    ok = True

    # Behavioral gates based on the known failure mode.
    if after_kf.chi2_p90 > max(100.0, 20.0 * before_kf.chi2_p90):
        print("FAIL: local mapping/local BA caused p90 reprojection error explosion.")
        ok = False
    if after_kf.z_min < 0.05:
        print("FAIL: local mapping/local BA produced near-zero depth map points.")
        ok = False
    if after_kf.z_max > 100.0:
        print("FAIL: local mapping/local BA produced extreme depth map points.")
        ok = False
    if after_kf.nonfinite > 0:
        print("FAIL: non-finite map geometry after local mapping.")
        ok = False

    return ok


def run_tum_smoke(repo: Path, dataset: Path, output: Path, n: int) -> tuple[bool, dict[str, str], str]:
    out_dir = output / f"tum_smoke_{n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"

    cmd = [
        sys.executable,
        "-m",
        "visual_slam.orbslam.run_tum_rgbd_smoke",
        str(dataset),
        "--output",
        str(out_dir),
        "--max-frames",
        str(n),
        "--print-every",
        "1",
    ]
    result = run_cmd(cmd, cwd=repo)
    log_file.write_text(result.stdout)

    summary: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in {
                "frames_attempted",
                "tracking_ok_count",
                "tracking_lost_count",
                "errors",
                "final_state",
                "final_keyframes",
                "final_map_points",
                "trajectory_poses",
                "avg_fps",
            }:
                summary[key] = value

    ok = result.returncode == 0 and int(summary.get("errors", "999")) == 0
    return ok, summary, result.stdout


def evaluate_smoke(summary: dict[str, str], n: int) -> bool:
    ok_count = int(summary.get("tracking_ok_count", "0"))
    lost_count = int(summary.get("tracking_lost_count", "999"))
    final_state = summary.get("final_state", "UNKNOWN")

    print(f"SMOKE {n} summary:", summary)

    if n == 3:
        return ok_count == 3 and lost_count == 0 and final_state == "OK"
    if n == 10:
        return ok_count >= 8 and final_state == "OK"
    if n == 30:
        return ok_count >= 24 and lost_count <= 6
    return ok_count > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-smoke30", action="store_true")
    args = parser.parse_args()

    repo = Path.cwd()
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not dataset.exists():
        print(f"Dataset not found: {dataset}")
        return 2

    all_ok = True

    if not args.skip_pytest:
        pytest_result = run_cmd(
            [sys.executable, "-m", "pytest", "-q", "tests/visual_slam/orbslam"],
            cwd=repo,
            env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        )
        if pytest_result.returncode != 0:
            print("FAIL: pytest failed.")
            all_ok = False

    if not check_local_ba_consistency(dataset):
        all_ok = False

    for n in ([3, 10] if args.skip_smoke30 else [3, 10, 30]):
        ok, summary, log_text = run_tum_smoke(repo, dataset, output, n)
        if not ok or not evaluate_smoke(summary, n):
            print(f"FAIL: TUM smoke {n} did not satisfy acceptance criteria.")
            all_ok = False

        lowered = log_text.lower()
        forbidden = ["traceback", "runtimewarning", "nan", "overflow", "0 vertices to optimize"]
        hits = [w for w in forbidden if w in lowered]
        if hits:
            print(f"FAIL: smoke {n} log contains forbidden warnings/errors: {hits}")
            all_ok = False

    if all_ok:
        print("=" * 100)
        print("VALIDATION PASSED")
        return 0

    print("=" * 100)
    print("VALIDATION FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
