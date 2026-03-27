import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CARTO_DIR = REPO_ROOT / "carto_outputs"
DEFAULT_HECTOR_DIR = REPO_ROOT / "hector_outputs"


@dataclass(frozen=True)
class MatcherFiles:
    trajectory: Path
    debug: Path
    label: str


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def load_trajectory(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 5:
        raise ValueError(f"{path} must have 5 columns, got {data.shape[1]}")
    return data


def load_debug(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=float, comments="#", skiprows=2)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 12:
        raise ValueError(f"{path} must have at least 12 columns, got {data.shape[1]}")
    return data


def candidate_stems(matcher: str, scan_count: int | None) -> Iterable[str]:
    base = f"trajectory_scan_to_{matcher}"
    if scan_count is not None:
        yield f"{base}_{scan_count}"
    yield base


def resolve_files(output_dir: Path, matcher: str, scan_count: int | None) -> MatcherFiles:
    for stem in candidate_stems(matcher, scan_count):
        traj = output_dir / f"{stem}.txt"
        dbg = output_dir / f"{stem}_debug.txt"
        if traj.exists() and dbg.exists():
            return MatcherFiles(trajectory=traj, debug=dbg, label=stem)

    matches = sorted(output_dir.glob(f"trajectory_scan_to_{matcher}_*.txt"))
    matches = [p for p in matches if not p.name.endswith("_debug.txt")]
    if scan_count is None and matches:
        def sort_key(path: Path) -> tuple[int, str]:
            suffix = path.stem.rsplit("_", 1)[-1]
            return (int(suffix) if suffix.isdigit() else -1, path.name)

        traj = max(matches, key=sort_key)
        dbg = output_dir / f"{traj.stem}_debug.txt"
        if dbg.exists():
            return MatcherFiles(trajectory=traj, debug=dbg, label=traj.stem)

    raise FileNotFoundError(
        f"Could not find trajectory/debug files for matcher '{matcher}' in {output_dir}"
    )


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones_like(arrays[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def summarize_scores(score: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(score)),
        "std": float(np.std(score)),
        "min": float(np.min(score)),
        "max": float(np.max(score)),
        "neg_count": int(np.sum(score < 0.0)),
    }


def summarize_debug(debug: np.ndarray) -> dict[str, float]:
    score = debug[:, 5]
    inliers = debug[:, 6]
    dx = debug[:, 7]
    dy = debug[:, 8]
    dth_deg = np.rad2deg(wrap_angle(debug[:, 9]))

    valid = finite_mask(score, inliers, dx, dy, dth_deg) & (score >= 0.0)
    if not np.any(valid):
        raise ValueError("No valid debug rows available for summary")

    trans = np.hypot(dx[valid], dy[valid])
    abs_rot = np.abs(dth_deg[valid])
    suspicious = np.sum((inliers[valid] >= 85.0) & (trans > 0.2))

    return {
        "score_mean": float(np.mean(score[valid])),
        "inliers_mean": float(np.mean(inliers[valid])),
        "inliers_min": float(np.min(inliers[valid])),
        "trans_p95": float(np.percentile(trans, 95)),
        "trans_p99": float(np.percentile(trans, 99)),
        "trans_max": float(np.max(trans)),
        "rot_p95": float(np.percentile(abs_rot, 95)),
        "rot_p99": float(np.percentile(abs_rot, 99)),
        "rot_max": float(np.max(abs_rot)),
        "suspicious_count": int(suspicious),
    }


def compare_trajectories(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    n = min(len(a), len(b))
    pos_diff = np.hypot(a[:n, 1] - b[:n, 1], a[:n, 2] - b[:n, 2])
    th_diff = np.abs(np.rad2deg(wrap_angle(a[:n, 3] - b[:n, 3])))
    return {
        "samples": n,
        "pos_mean": float(np.mean(pos_diff)),
        "pos_max": float(np.max(pos_diff)),
        "th_mean": float(np.mean(th_diff)),
        "th_max": float(np.max(th_diff)),
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def print_matcher_block(name: str, traj_stats: dict[str, float], dbg_stats: dict[str, float]) -> None:
    print(name)
    print(
        f"  trajectory score: mean={fmt(traj_stats['mean'])}, std={fmt(traj_stats['std'])}, "
        f"min={fmt(traj_stats['min'])}, max={fmt(traj_stats['max'])}, neg={traj_stats['neg_count']}"
    )
    print(
        f"  debug score/inliers: mean_score={fmt(dbg_stats['score_mean'])}, "
        f"mean_inliers={fmt(dbg_stats['inliers_mean'])}, min_inliers={fmt(dbg_stats['inliers_min'])}"
    )
    print(
        f"  translation delta [m]: p95={fmt(dbg_stats['trans_p95'])}, "
        f"p99={fmt(dbg_stats['trans_p99'])}, max={fmt(dbg_stats['trans_max'])}"
    )
    print(
        f"  rotation delta [deg]: p95={fmt(dbg_stats['rot_p95'])}, "
        f"p99={fmt(dbg_stats['rot_p99'])}, max={fmt(dbg_stats['rot_max'])}"
    )
    print(
        f"  suspicious high-inlier large-motion events: {dbg_stats['suspicious_count']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantitatively compare Cartographer and Hector local SLAM outputs."
    )
    parser.add_argument("--carto-dir", type=Path, default=DEFAULT_CARTO_DIR)
    parser.add_argument("--hector-dir", type=Path, default=DEFAULT_HECTOR_DIR)
    parser.add_argument(
        "--scan-count",
        type=int,
        default=None,
        help="Optional numeric suffix to target files like trajectory_scan_to_map_<count>.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    carto_sub = resolve_files(args.carto_dir, "submap", args.scan_count)
    carto_map = resolve_files(args.carto_dir, "map", args.scan_count)
    hector_sub = resolve_files(args.hector_dir, "submap", args.scan_count)
    hector_map = resolve_files(args.hector_dir, "map", args.scan_count)

    carto_sub_traj = load_trajectory(carto_sub.trajectory)
    carto_map_traj = load_trajectory(carto_map.trajectory)
    hector_sub_traj = load_trajectory(hector_sub.trajectory)
    hector_map_traj = load_trajectory(hector_map.trajectory)

    carto_sub_dbg = load_debug(carto_sub.debug)
    carto_map_dbg = load_debug(carto_map.debug)
    hector_sub_dbg = load_debug(hector_sub.debug)
    hector_map_dbg = load_debug(hector_map.debug)

    print("Local SLAM Quantitative Comparison")
    print(f"Carto dir : {args.carto_dir}")
    print(f"Hector dir: {args.hector_dir}")
    print(f"Carto files : {carto_sub.label}, {carto_map.label}")
    print(f"Hector files: {hector_sub.label}, {hector_map.label}")
    print()

    print("Cartographer")
    print_matcher_block(
        "  scan_to_submap",
        summarize_scores(carto_sub_traj[:, 4]),
        summarize_debug(carto_sub_dbg),
    )
    print_matcher_block(
        "  scan_to_map",
        summarize_scores(carto_map_traj[:, 4]),
        summarize_debug(carto_map_dbg),
    )
    print()

    print("Hector")
    print_matcher_block(
        "  scan_to_submap",
        summarize_scores(hector_sub_traj[:, 4]),
        summarize_debug(hector_sub_dbg),
    )
    print_matcher_block(
        "  scan_to_map",
        summarize_scores(hector_map_traj[:, 4]),
        summarize_debug(hector_map_dbg),
    )
    print()

    print("Within-Algorithm Divergence")
    carto_within = compare_trajectories(carto_sub_traj, carto_map_traj)
    hector_within = compare_trajectories(hector_sub_traj, hector_map_traj)
    print(
        "  Cartographer submap vs map: "
        f"pos_mean={fmt(carto_within['pos_mean'])} m, pos_max={fmt(carto_within['pos_max'])} m, "
        f"heading_mean={fmt(carto_within['th_mean'])} deg, heading_max={fmt(carto_within['th_max'])} deg"
    )
    print(
        "  Hector submap vs map: "
        f"pos_mean={fmt(hector_within['pos_mean'])} m, pos_max={fmt(hector_within['pos_max'])} m, "
        f"heading_mean={fmt(hector_within['th_mean'])} deg, heading_max={fmt(hector_within['th_max'])} deg"
    )
    print()

    print("Cross-Algorithm Agreement")
    sub_cross = compare_trajectories(carto_sub_traj, hector_sub_traj)
    map_cross = compare_trajectories(carto_map_traj, hector_map_traj)
    print(
        "  Carto vs Hector (scan_to_submap): "
        f"pos_mean={fmt(sub_cross['pos_mean'])} m, pos_max={fmt(sub_cross['pos_max'])} m, "
        f"heading_mean={fmt(sub_cross['th_mean'])} deg, heading_max={fmt(sub_cross['th_max'])} deg"
    )
    print(
        "  Carto vs Hector (scan_to_map): "
        f"pos_mean={fmt(map_cross['pos_mean'])} m, pos_max={fmt(map_cross['pos_max'])} m, "
        f"heading_mean={fmt(map_cross['th_mean'])} deg, heading_max={fmt(map_cross['th_max'])} deg"
    )


if __name__ == "__main__":
    main()
