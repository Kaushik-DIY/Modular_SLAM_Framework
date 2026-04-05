import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CARTO_DIR = REPO_ROOT / "carto_outputs"
DEFAULT_HECTOR_DIR = REPO_ROOT / "hector_outputs"
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "fr079"
DEFAULT_DATASET_LOG = DEFAULT_DATASET_DIR / "fr079.clf"
DEFAULT_RELATIONS_PATH = DEFAULT_DATASET_DIR / "fr079.relations"


@dataclass(frozen=True)
class MatcherFiles:
    trajectory: Path
    label: str


@dataclass(frozen=True)
class PoseSequence:
    times: np.ndarray
    poses: np.ndarray


@dataclass(frozen=True)
class DatasetReference:
    laser: PoseSequence
    odom: PoseSequence


@dataclass(frozen=True)
class RelationSummary:
    count: int
    time_min: float
    time_max: float
    overlap_count: int


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def se2_inverse(poses: np.ndarray) -> np.ndarray:
    c = np.cos(poses[:, 2])
    s = np.sin(poses[:, 2])
    x = -(c * poses[:, 0] + s * poses[:, 1])
    y = s * poses[:, 0] - c * poses[:, 1]
    th = wrap_angle(-poses[:, 2])
    return np.column_stack((x, y, th))


def se2_compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c = np.cos(a[:, 2])
    s = np.sin(a[:, 2])
    x = a[:, 0] + c * b[:, 0] - s * b[:, 1]
    y = a[:, 1] + s * b[:, 0] + c * b[:, 1]
    th = wrap_angle(a[:, 2] + b[:, 2])
    return np.column_stack((x, y, th))


def se2_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return se2_compose(se2_inverse(a), b)


def load_estimated_trajectory(path: Path) -> PoseSequence:
    data = np.loadtxt(path, dtype=float, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 5:
        raise ValueError(f"{path} must have 5 columns [t x y theta score], got {data.shape[1]}")
    return PoseSequence(times=data[:, 0], poses=data[:, 1:4])


def load_reference_trajectory(path: Path, timestamps_path: Path | None) -> PoseSequence:
    data = np.loadtxt(path, dtype=float, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] == 3:
        if timestamps_path is not None:
            times = np.loadtxt(timestamps_path, dtype=float, comments="#")
            if times.ndim != 1:
                times = np.asarray(times).reshape(-1)
            if len(times) != len(data):
                raise ValueError(
                    f"{timestamps_path} length {len(times)} does not match {path} length {len(data)}"
                )
        else:
            times = np.arange(len(data), dtype=float)
        poses = data[:, :3]
    elif data.shape[1] >= 4:
        times = data[:, 0]
        poses = data[:, 1:4]
    else:
        raise ValueError(f"{path} must have either 3 columns [x y theta] or 4+ columns [t x y theta ...]")

    return PoseSequence(times=np.asarray(times, dtype=float), poses=np.asarray(poses, dtype=float))


def load_dataset_reference(log_path: Path) -> DatasetReference:
    times = []
    laser_poses = []
    odom_poses = []

    with log_path.open("r", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts or parts[0] != "FLASER":
                continue

            try:
                beam_count = int(parts[1])
            except (ValueError, IndexError):
                continue

            base = 2 + beam_count
            if len(parts) < base + 7:
                continue

            try:
                laser_x = float(parts[base])
                laser_y = float(parts[base + 1])
                laser_th = float(parts[base + 2])
                odom_x = float(parts[base + 3])
                odom_y = float(parts[base + 4])
                odom_th = float(parts[base + 5])
                t = float(parts[-3])
            except ValueError:
                continue

            times.append(t)
            laser_poses.append((laser_x, laser_y, laser_th))
            odom_poses.append((odom_x, odom_y, odom_th))

    if not times:
        raise ValueError(f"No FLASER scans were parsed from {log_path}")

    times_arr = np.asarray(times, dtype=float)
    laser_arr = np.asarray(laser_poses, dtype=float)
    odom_arr = np.asarray(odom_poses, dtype=float)
    return DatasetReference(
        laser=PoseSequence(times=times_arr, poses=laser_arr),
        odom=PoseSequence(times=times_arr, poses=odom_arr),
    )


def summarize_relations(path: Path, reference_times: np.ndarray) -> RelationSummary:
    timestamps = []
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                a = float(parts[0])
                b = float(parts[1])
            except ValueError:
                continue
            timestamps.extend((a, b))

    if not timestamps:
        return RelationSummary(count=0, time_min=float("nan"), time_max=float("nan"), overlap_count=0)

    time_arr = np.asarray(timestamps, dtype=float)
    ref_min = float(reference_times.min())
    ref_max = float(reference_times.max())
    overlap_mask = (time_arr >= ref_min) & (time_arr <= ref_max)
    return RelationSummary(
        count=len(time_arr) // 2,
        time_min=float(time_arr.min()),
        time_max=float(time_arr.max()),
        overlap_count=int(np.sum(overlap_mask) // 2),
    )


def nearest_indices(reference_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(reference_times, query_times)
    idx = np.clip(idx, 0, len(reference_times) - 1)
    left = np.clip(idx - 1, 0, len(reference_times) - 1)
    choose_left = np.abs(reference_times[left] - query_times) <= np.abs(reference_times[idx] - query_times)
    return np.where(choose_left, left, idx)


def align_reference_to_estimate(
    estimate: PoseSequence,
    reference: PoseSequence,
    max_time_diff: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(reference.times) == len(estimate.times) and np.allclose(reference.times, estimate.times):
        return estimate.poses, reference.poses

    if len(reference.times) == len(estimate.times) and np.allclose(reference.times, np.arange(len(reference.times))):
        return estimate.poses, reference.poses

    indices = nearest_indices(reference.times, estimate.times)
    matched_reference = reference.poses[indices]
    time_error = np.abs(reference.times[indices] - estimate.times)

    if max_time_diff is not None:
        valid = time_error <= max_time_diff
    else:
        valid = np.ones(len(indices), dtype=bool)

    if not np.any(valid):
        raise ValueError("No overlapping estimate/reference samples after timestamp alignment")

    return estimate.poses[valid], matched_reference[valid]


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values ** 2)))


def compute_ate_metrics(estimate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    error = se2_between(reference, estimate)
    trans_error = np.hypot(error[:, 0], error[:, 1])
    return {
        "samples": int(len(trans_error)),
        "rmse_m": rmse(trans_error),
        "mean_m": float(np.mean(trans_error)),
        "max_m": float(np.max(trans_error)),
    }


def compute_rpe_metrics(estimate: np.ndarray, reference: np.ndarray, delta: int) -> dict[str, float]:
    if delta < 1:
        raise ValueError("RPE delta must be >= 1")
    if len(estimate) <= delta or len(reference) <= delta:
        raise ValueError("Not enough samples to compute RPE for the requested delta")

    est_rel = se2_between(estimate[:-delta], estimate[delta:])
    ref_rel = se2_between(reference[:-delta], reference[delta:])
    rel_error = se2_between(ref_rel, est_rel)

    trans_error = np.hypot(rel_error[:, 0], rel_error[:, 1])
    rot_error = np.abs(rel_error[:, 2])
    return {
        "samples": int(len(trans_error)),
        "trans_rmse_m": rmse(trans_error),
        "rot_rmse_deg": rmse(np.rad2deg(rot_error)),
    }


def compute_motion_stability_metrics(estimate: np.ndarray) -> dict[str, float]:
    if len(estimate) < 3:
        raise ValueError("Need at least 3 poses to compute motion stability")

    increments = se2_between(estimate[:-1], estimate[1:])
    delta_increments = increments[1:] - increments[:-1]
    delta_increments[:, 2] = wrap_angle(delta_increments[:, 2])

    translational_change = np.hypot(delta_increments[:, 0], delta_increments[:, 1])
    rotational_change = np.abs(delta_increments[:, 2])
    combined = np.sqrt(
        delta_increments[:, 0] ** 2
        + delta_increments[:, 1] ** 2
        + delta_increments[:, 2] ** 2
    )

    return {
        "samples": int(len(combined)),
        "mean_instability": float(np.mean(combined)),
        "rmse_instability": rmse(combined),
        "trans_rmse_m": rmse(translational_change),
        "rot_rmse_deg": rmse(np.rad2deg(rotational_change)),
    }


def candidate_stems(matcher: str, scan_count: int | None) -> Iterable[str]:
    base = f"trajectory_scan_to_{matcher}"
    if scan_count is not None:
        yield f"{base}_{scan_count}"
    yield base


def resolve_files(output_dir: Path, matcher: str, scan_count: int | None) -> MatcherFiles:
    for stem in candidate_stems(matcher, scan_count):
        traj = output_dir / f"{stem}.txt"
        if traj.exists():
            return MatcherFiles(trajectory=traj, label=stem)

    matches = sorted(output_dir.glob(f"trajectory_scan_to_{matcher}_*.txt"))
    if scan_count is None and matches:
        matches = [p for p in matches if not p.name.endswith("_debug.txt")]

        def sort_key(path: Path) -> tuple[int, str]:
            suffix = path.stem.rsplit("_", 1)[-1]
            return (int(suffix) if suffix.isdigit() else -1, path.name)

        traj = max(matches, key=sort_key)
        return MatcherFiles(trajectory=traj, label=traj.stem)

    raise FileNotFoundError(f"Could not find trajectory file for matcher '{matcher}' in {output_dir}")


def load_reference_from_args(args: argparse.Namespace) -> tuple[PoseSequence, str]:
    if args.reference is not None:
        return load_reference_trajectory(args.reference, args.reference_stamps), "external"

    dataset_ref = load_dataset_reference(args.dataset_log)
    if args.reference_source == "dataset_laser":
        return dataset_ref.laser, "dataset_laser"
    return dataset_ref.odom, "dataset_odom"


def evaluate_matcher(
    trajectory_path: Path,
    reference: PoseSequence,
    rpe_delta: int,
    max_time_diff: float | None,
) -> dict[str, dict[str, float]]:
    estimate = load_estimated_trajectory(trajectory_path)
    estimate_poses, reference_poses = align_reference_to_estimate(estimate, reference, max_time_diff)
    return {
        "ate": compute_ate_metrics(estimate_poses, reference_poses),
        "rpe": compute_rpe_metrics(estimate_poses, reference_poses, rpe_delta),
        "stability": compute_motion_stability_metrics(estimate_poses),
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def print_metric_block(name: str, metrics: dict[str, dict[str, float]]) -> None:
    ate = metrics["ate"]
    rpe = metrics["rpe"]
    stability = metrics["stability"]

    print(name)
    print(
        f"  ATE: rmse={fmt(ate['rmse_m'])} m, mean={fmt(ate['mean_m'])} m, "
        f"max={fmt(ate['max_m'])} m, samples={ate['samples']}"
    )
    print(
        f"  RPE: trans_rmse={fmt(rpe['trans_rmse_m'])} m, "
        f"rot_rmse={fmt(rpe['rot_rmse_deg'])} deg, samples={rpe['samples']}"
    )
    print(
        f"  Motion stability: mean_instability={fmt(stability['mean_instability'])}, "
        f"rmse_instability={fmt(stability['rmse_instability'])}, "
        f"trans_rmse={fmt(stability['trans_rmse_m'])} m, "
        f"rot_rmse={fmt(stability['rot_rmse_deg'])} deg"
    )


def print_winner(label_a: str, metrics_a: dict[str, dict[str, float]], label_b: str, metrics_b: dict[str, dict[str, float]]) -> None:
    comparisons = [
        ("ATE RMSE", metrics_a["ate"]["rmse_m"], metrics_b["ate"]["rmse_m"], "m"),
        ("RPE translation RMSE", metrics_a["rpe"]["trans_rmse_m"], metrics_b["rpe"]["trans_rmse_m"], "m"),
        ("RPE rotation RMSE", metrics_a["rpe"]["rot_rmse_deg"], metrics_b["rpe"]["rot_rmse_deg"], "deg"),
        ("Motion stability RMSE", metrics_a["stability"]["rmse_instability"], metrics_b["stability"]["rmse_instability"], ""),
    ]
    for metric_name, value_a, value_b, unit in comparisons:
        if np.isclose(value_a, value_b):
            winner = "tie"
            detail = f"{fmt(value_a)} {unit}".strip()
        elif value_a < value_b:
            winner = label_a
            detail = f"{fmt(value_a)} {unit} vs {fmt(value_b)} {unit}".strip()
        else:
            winner = label_b
            detail = f"{fmt(value_b)} {unit} vs {fmt(value_a)} {unit}".strip()
        print(f"  {metric_name}: {winner} ({detail})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare scan_to_map and scan_to_submap trajectory quality for Cartographer and Hector."
    )
    parser.add_argument("--carto-dir", type=Path, default=DEFAULT_CARTO_DIR)
    parser.add_argument("--hector-dir", type=Path, default=DEFAULT_HECTOR_DIR)
    parser.add_argument(
        "--dataset-log",
        type=Path,
        default=DEFAULT_DATASET_LOG,
        help="Dataset CARMEN log used to derive scan timestamps and odometry from FLASER entries.",
    )
    parser.add_argument(
        "--relations-path",
        type=Path,
        default=DEFAULT_RELATIONS_PATH,
        help="Dataset relations file linked to the same scan timestamps.",
    )
    parser.add_argument(
        "--reference-source",
        choices=["dataset_odom", "dataset_laser"],
        default="dataset_odom",
        help="Default reference source to derive from the raw dataset when --reference is not provided.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional external reference trajectory path. If set, it overrides dataset-derived references.",
    )
    parser.add_argument(
        "--reference-stamps",
        type=Path,
        default=None,
        help="Optional timestamps for an external 3-column reference file [x y theta].",
    )
    parser.add_argument(
        "--scan-count",
        type=int,
        default=None,
        help="Optional numeric suffix to target files like trajectory_scan_to_map_<count>.txt.",
    )
    parser.add_argument(
        "--rpe-delta",
        type=int,
        default=1,
        help="Pose interval used for Relative Pose Error. Use 1 for scan-to-scan evaluation.",
    )
    parser.add_argument(
        "--max-time-diff",
        type=float,
        default=None,
        help="Optional max allowed timestamp mismatch when aligning estimate to reference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference, reference_source = load_reference_from_args(args)
    relation_summary = summarize_relations(args.relations_path, reference.times) if args.relations_path.exists() else None

    carto_sub = resolve_files(args.carto_dir, "submap", args.scan_count)
    carto_map = resolve_files(args.carto_dir, "map", args.scan_count)
    hector_sub = resolve_files(args.hector_dir, "submap", args.scan_count)
    hector_map = resolve_files(args.hector_dir, "map", args.scan_count)

    carto_sub_metrics = evaluate_matcher(carto_sub.trajectory, reference, args.rpe_delta, args.max_time_diff)
    carto_map_metrics = evaluate_matcher(carto_map.trajectory, reference, args.rpe_delta, args.max_time_diff)
    hector_sub_metrics = evaluate_matcher(hector_sub.trajectory, reference, args.rpe_delta, args.max_time_diff)
    hector_map_metrics = evaluate_matcher(hector_map.trajectory, reference, args.rpe_delta, args.max_time_diff)

    print("Matcher Trajectory Evaluation")
    if args.reference is None:
        print(f"Dataset log     : {args.dataset_log}")
        print(f"Reference       : {reference_source} from dataset FLASER entries")
    else:
        print(f"Reference       : {args.reference} (external)")
        if args.reference_stamps is not None:
            print(f"Reference times : {args.reference_stamps}")
    if args.relations_path.exists() and relation_summary is not None:
        print(f"Relations file  : {args.relations_path}")
        print(
            f"Relations span  : count={relation_summary.count}, "
            f"t_min={relation_summary.time_min:.3f}, t_max={relation_summary.time_max:.3f}, "
            f"overlap_with_reference={relation_summary.overlap_count}"
        )
    print(f"Carto dir       : {args.carto_dir}")
    print(f"Hector dir      : {args.hector_dir}")
    print(f"RPE delta       : {args.rpe_delta}")
    if args.max_time_diff is not None:
        print(f"Max time diff   : {args.max_time_diff:.4f} s")
    print()

    print("Cartographer")
    print_metric_block("  scan_to_submap", carto_sub_metrics)
    print_metric_block("  scan_to_map", carto_map_metrics)
    print("  Better matcher by metric")
    print_winner("scan_to_submap", carto_sub_metrics, "scan_to_map", carto_map_metrics)
    print()

    print("Hector")
    print_metric_block("  scan_to_submap", hector_sub_metrics)
    print_metric_block("  scan_to_map", hector_map_metrics)
    print("  Better matcher by metric")
    print_winner("scan_to_submap", hector_sub_metrics, "scan_to_map", hector_map_metrics)


if __name__ == "__main__":
    main()
