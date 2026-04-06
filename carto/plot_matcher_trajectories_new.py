import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


OUTPUT_DIR = Path("carto_outputs")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot trajectory metrics for a single matcher report."
    )
    parser.add_argument(
        "trajectory",
        nargs="?",
        help="Path to a trajectory .txt file. Defaults to the most recent trajectory report.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively after saving them.",
    )
    return parser.parse_args()


def wrap_angle(angle_rad):
    angle_rad = np.asarray(angle_rad, dtype=float)
    wrapped = np.full_like(angle_rad, np.nan, dtype=float)
    valid = np.isfinite(angle_rad)
    wrapped[valid] = (angle_rad[valid] + np.pi) % (2.0 * np.pi) - np.pi
    return wrapped


def find_latest_trajectory():
    candidates = []
    for path in OUTPUT_DIR.glob("trajectory_*.txt"):
        name = path.name
        if any(
            token in name
            for token in (
                "_debug.txt",
                "_optimized_nodes.txt",
                "_optimized_submaps.txt",
                "_inserted_local_nodes.txt",
                "_aligned.txt",
            )
        ):
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError("No trajectory reports found in carto_outputs.")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_paths(trajectory_arg):
    traj_path = Path(trajectory_arg) if trajectory_arg else find_latest_trajectory()
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    stem = traj_path.stem
    debug_path = traj_path.with_name(f"{stem}_debug.txt")
    if not debug_path.exists():
        raise FileNotFoundError(f"Debug file not found: {debug_path}")

    return traj_path, debug_path


def load_traj(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return {
        "t": data[:, 0],
        "x": data[:, 1],
        "y": data[:, 2],
        "theta": data[:, 3],
        "score": data[:, 4],
    }


def load_debug(path):
    comment_lines = 0
    metadata = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            comment_lines += 1
            text = line[1:].strip()
            if "=" in text:
                key, value = text.split("=", 1)
                metadata[key.strip()] = value.strip()

    data = np.genfromtxt(
        path,
        names=True,
        dtype=float,
        skip_header=comment_lines,
        encoding="utf-8",
    )

    if getattr(data, "ndim", 0) == 0:
        data = np.array([data], dtype=data.dtype)

    columns = {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}
    return columns, metadata


def sanitize_metric(values, invalid_value=-1.0):
    arr = np.asarray(values, dtype=float).copy()
    arr[arr == invalid_value] = np.nan
    return arr


def save_figure(path, show):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()


def plot_xy(base_path, traj, label, show):
    plt.figure(figsize=(8, 6))
    plt.plot(traj["x"], traj["y"], label=label, color="tab:blue", linewidth=1.8)
    plt.scatter(traj["x"][0], traj["y"][0], color="tab:green", s=45, label="start")
    plt.scatter(traj["x"][-1], traj["y"][-1], color="tab:red", s=45, label="end")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"{label}: XY Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    save_figure(base_path.with_name(f"{base_path.stem}_xy.png"), show)


def plot_score(base_path, traj, label, show):
    plt.figure(figsize=(9, 4))
    plt.plot(traj["score"], color="tab:blue", linewidth=1.4)
    plt.xlabel("scan index")
    plt.ylabel("score")
    plt.title(f"{label}: Matcher Score")
    plt.grid(True)
    save_figure(base_path.with_name(f"{base_path.stem}_score.png"), show)


def plot_inliers(base_path, debug, label, show):
    if "inliers" not in debug:
        return

    inliers = sanitize_metric(debug["inliers"])
    if np.all(np.isnan(inliers)):
        return

    plt.figure(figsize=(9, 4))
    plt.plot(inliers, color="tab:orange", linewidth=1.4)
    plt.xlabel("scan index")
    plt.ylabel("inliers")
    plt.title(f"{label}: Inliers")
    plt.grid(True)
    save_figure(base_path.with_name(f"{base_path.stem}_inliers.png"), show)


def plot_delta(base_path, debug, label, show):
    required = ("dx", "dy", "dtheta")
    if any(name not in debug for name in required):
        return

    dx = np.asarray(debug["dx"], dtype=float)
    dy = np.asarray(debug["dy"], dtype=float)
    dtheta_deg = np.rad2deg(wrap_angle(np.asarray(debug["dtheta"], dtype=float)))

    plt.figure(figsize=(10, 6))
    plt.plot(dx, label="dx [m]", linewidth=1.2)
    plt.plot(dy, label="dy [m]", linewidth=1.2)
    plt.plot(dtheta_deg, label="dtheta [deg]", linewidth=1.2)
    plt.xlabel("scan index")
    plt.ylabel("delta")
    plt.title(f"{label}: Match Delta")
    plt.grid(True)
    plt.legend()
    save_figure(base_path.with_name(f"{base_path.stem}_delta.png"), show)


def plot_heading(base_path, traj, label, show):
    plt.figure(figsize=(9, 4))
    plt.plot(np.rad2deg(traj["theta"]), color="tab:purple", linewidth=1.4)
    plt.xlabel("scan index")
    plt.ylabel("heading [deg]")
    plt.title(f"{label}: Heading")
    plt.grid(True)
    save_figure(base_path.with_name(f"{base_path.stem}_heading.png"), show)


def plot_graph_growth(base_path, debug, label, show):
    optional = [name for name in ("constraints_total", "nodes", "submaps") if name in debug]
    if not optional:
        return

    plt.figure(figsize=(9, 4))
    for name in optional:
        plt.plot(debug[name], label=name.replace("_", " "), linewidth=1.4)
    plt.xlabel("scan index")
    plt.ylabel("count")
    plt.title(f"{label}: Graph Growth")
    plt.grid(True)
    plt.legend()
    save_figure(base_path.with_name(f"{base_path.stem}_graph_growth.png"), show)


def main():
    args = parse_args()
    traj_path, debug_path = resolve_paths(args.trajectory)

    traj = load_traj(traj_path)
    debug, metadata = load_debug(debug_path)

    label_parts = [metadata.get("matcher_type", traj_path.stem.replace("trajectory_", ""))]
    backend = metadata.get("submap_backend_type")
    if backend:
        label_parts.append(backend)
    label = " / ".join(label_parts)

    print(f"Using trajectory: {traj_path}")
    print(f"Using debug: {debug_path}")

    plot_xy(traj_path, traj, label, args.show)
    plot_score(traj_path, traj, label, args.show)
    plot_inliers(traj_path, debug, label, args.show)
    plot_delta(traj_path, debug, label, args.show)
    plot_heading(traj_path, traj, label, args.show)
    plot_graph_growth(traj_path, debug, label, args.show)

    print(f"Wrote: {traj_path.with_name(f'{traj_path.stem}_xy.png')}")
    print(f"Wrote: {traj_path.with_name(f'{traj_path.stem}_score.png')}")
    if "inliers" in debug and not np.all(np.isnan(sanitize_metric(debug["inliers"]))):
        print(f"Wrote: {traj_path.with_name(f'{traj_path.stem}_inliers.png')}")
    if all(name in debug for name in ("dx", "dy", "dtheta")):
        print(f"Wrote: {traj_path.with_name(f'{traj_path.stem}_delta.png')}")
    print(f"Wrote: {traj_path.with_name(f'{traj_path.stem}_heading.png')}")
    if any(name in debug for name in ("constraints_total", "nodes", "submaps")):
        print(f"Wrote: {traj_path.with_name(f'{traj_path.stem}_graph_growth.png')}")


if __name__ == "__main__":
    main()
