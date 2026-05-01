#!/usr/bin/env python3
"""Run ORB extractor backend durability checks on TUM RGB-D sequences."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class DurabilityResult:
    backend: str
    frame_count: str
    output_dir: Path
    returncode: int
    frames_attempted: int = 0
    tracking_ok_count: int = 0
    tracking_lost_count: int = 0
    errors: int = 0
    final_state: str = "UNKNOWN"
    final_keyframes: int = 0
    final_map_points: int = 0
    avg_fps: float = 0.0
    trajectory_file: str = ""
    frame_log_file: str = ""
    command_log: str = ""
    error_message: str = ""
    trajectory_eval_status: str = "not_run"
    ate_rmse_se3_m: float | None = None
    ate_rmse_sim3_m: float | None = None
    rpe_trans_rmse_m: float | None = None
    rpe_rot_rmse_deg: float | None = None
    num_associations: int | None = None

    @property
    def ok_ratio(self) -> float:
        if self.frames_attempted <= 0:
            return 0.0
        return self.tracking_ok_count / float(self.frames_attempted)


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ORB backend durability checks.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-counts", nargs="+", default=["100", "300", "full"])
    parser.add_argument("--backends", nargs="+", default=["opencv_orb", "pyslam_orb2"])
    parser.add_argument("--groundtruth", type=Path, default=None)
    parser.add_argument("--max-time-diff", type=float, default=0.02)
    parser.add_argument("--print-every", type=int, default=25)
    return parser


def frame_count_to_max_frames(frame_count: str) -> int:
    normalized = str(frame_count).strip().lower()
    if normalized == "full":
        return 0
    value = int(normalized)
    if value <= 0:
        raise ValueError(f"Frame count must be positive or 'full': {frame_count}")
    return value


def parse_smoke_stdout(stdout: str) -> dict[str, str]:
    keys = {
        "frames_attempted",
        "tracking_ok_count",
        "tracking_lost_count",
        "errors",
        "final_state",
        "final_keyframes",
        "final_map_points",
        "avg_fps",
        "trajectory_file",
        "frame_log_file",
    }
    summary: dict[str, str] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in keys:
            summary[key] = value.strip()
    return summary


def _int_value(summary: dict[str, str], key: str) -> int:
    try:
        return int(summary.get(key, "0"))
    except ValueError:
        return 0


def _float_value(summary: dict[str, str], key: str) -> float:
    try:
        return float(summary.get(key, "0"))
    except ValueError:
        return 0.0


def run_backend_smoke(
    dataset: Path,
    output: Path,
    backend: str,
    frame_count: str,
    print_every: int = 25,
) -> DurabilityResult:
    run_dir = output / backend / str(frame_count)
    run_dir.mkdir(parents=True, exist_ok=True)
    command_log = run_dir / "command.log"
    max_frames = frame_count_to_max_frames(frame_count)

    cmd = [
        sys.executable,
        "-m",
        "visual_slam.orbslam.run_tum_rgbd_smoke",
        str(dataset),
        "--output",
        str(run_dir),
        "--max-frames",
        str(max_frames),
        "--print-every",
        str(print_every),
        "--feature-backend",
        backend,
    ]

    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    command_log.write_text("$ " + " ".join(cmd) + "\n\n" + completed.stdout)

    summary = parse_smoke_stdout(completed.stdout)
    return DurabilityResult(
        backend=backend,
        frame_count=str(frame_count),
        output_dir=run_dir,
        returncode=completed.returncode,
        frames_attempted=_int_value(summary, "frames_attempted"),
        tracking_ok_count=_int_value(summary, "tracking_ok_count"),
        tracking_lost_count=_int_value(summary, "tracking_lost_count"),
        errors=_int_value(summary, "errors"),
        final_state=summary.get("final_state", "UNKNOWN"),
        final_keyframes=_int_value(summary, "final_keyframes"),
        final_map_points=_int_value(summary, "final_map_points"),
        avg_fps=_float_value(summary, "avg_fps"),
        trajectory_file=summary.get("trajectory_file", ""),
        frame_log_file=summary.get("frame_log_file", ""),
        command_log=str(command_log),
        error_message="" if completed.returncode == 0 else f"run_tum_rgbd_smoke exited {completed.returncode}",
    )


def evaluate_result_trajectory(
    result: DurabilityResult,
    groundtruth: Path | None,
    max_time_diff: float = 0.02,
) -> DurabilityResult:
    if groundtruth is None or not groundtruth.exists():
        result.trajectory_eval_status = "groundtruth_missing"
        return result

    if result.returncode != 0:
        result.trajectory_eval_status = "smoke_failed"
        return result

    if not result.trajectory_file or not Path(result.trajectory_file).exists():
        result.trajectory_eval_status = "trajectory_missing"
        return result

    from tools.evaluate_tum_trajectory import evaluate_trajectories

    eval_dir = result.output_dir / "trajectory_eval"
    try:
        metrics = evaluate_trajectories(
            groundtruth_path=groundtruth,
            trajectory_path=Path(result.trajectory_file),
            output_dir=eval_dir,
            max_time_diff=max_time_diff,
        )
    except Exception as exc:
        result.trajectory_eval_status = "failed"
        result.error_message = (result.error_message + "; " if result.error_message else "") + (
            f"trajectory evaluation failed: {type(exc).__name__}: {exc}"
        )
        return result

    result.trajectory_eval_status = "ok"
    result.ate_rmse_se3_m = float(metrics["ate_rmse_se3_m"])
    result.ate_rmse_sim3_m = float(metrics["ate_rmse_sim3_m"])
    result.rpe_trans_rmse_m = float(metrics["rpe_trans_rmse_m"])
    result.rpe_rot_rmse_deg = float(metrics["rpe_rot_rmse_deg"])
    result.num_associations = int(metrics["num_associations"])
    return result


def run_durability(
    dataset: Path,
    output: Path,
    frame_counts: list[str],
    backends: list[str],
    groundtruth: Path | None = None,
    max_time_diff: float = 0.02,
    print_every: int = 25,
) -> list[DurabilityResult]:
    dataset = Path(dataset).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if groundtruth is None:
        groundtruth = dataset / "groundtruth.txt"
    else:
        groundtruth = Path(groundtruth).expanduser().resolve()

    results: list[DurabilityResult] = []
    for backend in backends:
        for frame_count in frame_counts:
            print(f"starting durability run: backend={backend} frame_count={frame_count}", flush=True)
            result = run_backend_smoke(
                dataset=dataset,
                output=output,
                backend=backend,
                frame_count=str(frame_count),
                print_every=print_every,
            )
            result = evaluate_result_trajectory(result, groundtruth, max_time_diff=max_time_diff)
            results.append(result)
            print(
                f"finished durability run: backend={backend} frame_count={frame_count} "
                f"return={result.returncode} ok={result.tracking_ok_count}/{result.frames_attempted} "
                f"lost={result.tracking_lost_count} eval={result.trajectory_eval_status}",
                flush=True,
            )
            write_metrics_csv(output / "backend_durability_metrics.csv", results)
            write_summary_markdown(output / "backend_durability_summary.md", results, dataset, groundtruth)

    write_metrics_csv(output / "backend_durability_metrics.csv", results)
    write_summary_markdown(output / "backend_durability_summary.md", results, dataset, groundtruth)
    return results


def _metric_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)


def write_metrics_csv(path: Path, results: list[DurabilityResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "backend",
        "frame_count",
        "returncode",
        "frames_attempted",
        "tracking_ok_count",
        "tracking_lost_count",
        "ok_ratio",
        "errors",
        "final_state",
        "final_keyframes",
        "final_map_points",
        "avg_fps",
        "trajectory_eval_status",
        "num_associations",
        "ate_rmse_se3_m",
        "ate_rmse_sim3_m",
        "rpe_trans_rmse_m",
        "rpe_rot_rmse_deg",
        "trajectory_file",
        "frame_log_file",
        "command_log",
        "error_message",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "backend": result.backend,
                    "frame_count": result.frame_count,
                    "returncode": result.returncode,
                    "frames_attempted": result.frames_attempted,
                    "tracking_ok_count": result.tracking_ok_count,
                    "tracking_lost_count": result.tracking_lost_count,
                    "ok_ratio": f"{result.ok_ratio:.6f}",
                    "errors": result.errors,
                    "final_state": result.final_state,
                    "final_keyframes": result.final_keyframes,
                    "final_map_points": result.final_map_points,
                    "avg_fps": f"{result.avg_fps:.6f}",
                    "trajectory_eval_status": result.trajectory_eval_status,
                    "num_associations": _metric_value(result.num_associations),
                    "ate_rmse_se3_m": _metric_value(result.ate_rmse_se3_m),
                    "ate_rmse_sim3_m": _metric_value(result.ate_rmse_sim3_m),
                    "rpe_trans_rmse_m": _metric_value(result.rpe_trans_rmse_m),
                    "rpe_rot_rmse_deg": _metric_value(result.rpe_rot_rmse_deg),
                    "trajectory_file": result.trajectory_file,
                    "frame_log_file": result.frame_log_file,
                    "command_log": result.command_log,
                    "error_message": result.error_message,
                }
            )


def write_summary_markdown(
    path: Path,
    results: list[DurabilityResult],
    dataset: Path,
    groundtruth: Path | None,
) -> None:
    gt_text = "missing"
    if groundtruth is not None and groundtruth.exists():
        gt_text = str(groundtruth)

    lines = [
        "# ORB Backend Durability Summary",
        "",
        f"Dataset: `{dataset}`",
        f"Ground truth: `{gt_text}`",
        "",
        "| Backend | Frames | Return | OK/Lost | OK ratio | Final | KF | MP | FPS | ATE SE3 | ATE Sim3 | RPE trans | RPE rot deg | Eval |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            f"| {result.backend} | {result.frame_count} | {result.returncode} | "
            f"{result.tracking_ok_count}/{result.tracking_lost_count} | "
            f"{result.ok_ratio:.3f} | {result.final_state} | {result.final_keyframes} | "
            f"{result.final_map_points} | {result.avg_fps:.3f} | "
            f"{_metric_value(result.ate_rmse_se3_m)} | {_metric_value(result.ate_rmse_sim3_m)} | "
            f"{_metric_value(result.rpe_trans_rmse_m)} | {_metric_value(result.rpe_rot_rmse_deg)} | "
            f"{result.trajectory_eval_status} |"
        )

    lines.extend(
        [
            "",
            "## Backend Recommendation",
            "",
            "Keep `opencv_orb` as the default backend unless the durability and trajectory metrics "
            "show that `pyslam_orb2` is clearly better and the user approves a default switch.",
            "",
            "## Raw Results",
            "",
            "Machine-readable metrics are in `backend_durability_metrics.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_results_json(path: Path, results: list[DurabilityResult]) -> None:
    serializable = []
    for result in results:
        row = dict(result.__dict__)
        row["output_dir"] = str(row["output_dir"])
        serializable.append(row)
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")


def acceptance_failed(results: list[DurabilityResult]) -> bool:
    failed = False
    for result in results:
        if result.frame_count == "full":
            continue
        if result.returncode != 0 or result.errors != 0:
            failed = True
        if result.frames_attempted > 0 and result.ok_ratio < 0.95:
            failed = True
    return failed


def main(argv: list[str] | None = None) -> int:
    parser = create_arg_parser()
    args = parser.parse_args(argv)

    results = run_durability(
        dataset=args.dataset,
        output=args.output,
        frame_counts=[str(v) for v in args.frame_counts],
        backends=[str(v) for v in args.backends],
        groundtruth=args.groundtruth,
        max_time_diff=args.max_time_diff,
        print_every=args.print_every,
    )
    write_results_json(Path(args.output).expanduser() / "backend_durability_metrics.json", results)

    for result in results:
        print(
            f"{result.backend} {result.frame_count}: "
            f"return={result.returncode} ok={result.tracking_ok_count}/{result.frames_attempted} "
            f"lost={result.tracking_lost_count} state={result.final_state} "
            f"ate={_metric_value(result.ate_rmse_se3_m)} eval={result.trajectory_eval_status}"
        )

    print(f"summary: {Path(args.output).expanduser() / 'backend_durability_summary.md'}")
    return 1 if acceptance_failed(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
