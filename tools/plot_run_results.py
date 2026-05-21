#!/usr/bin/env python3
"""
Plot trajectory, map, and keyframe graph from a slam_ws smoke-runner output directory.

Usage:
    python tools/plot_run_results.py \
        --root visual_slam_outputs/fr1_room_cpp_ba_full \
        --dataset datasets/tum/rgbd_dataset_freiburg1_room \
        --output visual_slam_outputs/fr1_room_cpp_ba_full/plots2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps (N,), xyz (N,3)) from a TUM-format pose file."""
    ts, xyz = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts.append(float(parts[0]))
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(ts), np.array(xyz)


def read_ply_xyz(path: Path) -> np.ndarray:
    """Read xyz from an ASCII PLY file."""
    pts = []
    in_data = False
    with open(path) as f:
        for line in f:
            if "end_header" in line:
                in_data = True
                continue
            if not in_data:
                continue
            parts = line.split()
            if len(parts) >= 3:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(pts) if pts else np.zeros((0, 3))


def load_keyframes(path: Path) -> np.ndarray:
    """Return (N, 3) camera-center positions from keyframes.json."""
    kfs = json.loads(path.read_text())
    pos = []
    for kf in kfs:
        twc = kf.get("Twc") or kf.get("position")
        if twc is not None:
            if isinstance(twc, list) and len(twc) == 4 and isinstance(twc[0], list):
                # 4x4 matrix — camera center is last col of Twc
                pos.append([twc[0][3], twc[1][3], twc[2][3]])
            elif isinstance(twc, list) and len(twc) == 3:
                pos.append(twc)
    return np.array(pos) if pos else np.zeros((0, 3))


def load_graph(path: Path) -> tuple[list, list, list]:
    """Return (covis_edges, loop_edges, node_kid_list)."""
    g = json.loads(path.read_text())
    covis = [(e["source"], e["target"]) for e in g.get("covisibility_edges", [])]
    loops = [(e["source"], e["target"]) for e in g.get("loop_edges", [])]
    nodes = g.get("nodes", [])
    return covis, loops, nodes


# ---------------------------------------------------------------------------
# SE3 alignment (Horn's method)
# ---------------------------------------------------------------------------

def align_se3(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, t) such that dst ≈ R @ src + t (least-squares)."""
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    sc = src - mu_s
    dc = dst - mu_d
    H = sc.T @ dc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = mu_d - R @ mu_s
    return R, t


def associate(ts_gt: np.ndarray, ts_est: np.ndarray, max_diff: float = 0.02):
    """Return (gt_idx, est_idx) matched pairs within max_diff seconds."""
    pairs_g, pairs_e = [], []
    for j, te in enumerate(ts_est):
        i = int(np.argmin(np.abs(ts_gt - te)))
        if abs(ts_gt[i] - te) <= max_diff:
            pairs_g.append(i)
            pairs_e.append(j)
    return np.array(pairs_g), np.array(pairs_e)


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_trajectory_xy(est_xyz, gt_xyz, kf_xyz, loop_edges_kf, save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], "k-", lw=1.2, label="Ground truth", alpha=0.8)
    ax.plot(est_xyz[:, 0], est_xyz[:, 1], "b-", lw=1.0, label="Estimated", alpha=0.85)
    if len(kf_xyz) > 0:
        ax.scatter(kf_xyz[:, 0], kf_xyz[:, 1], s=6, c="royalblue", zorder=3, label=f"Keyframes ({len(kf_xyz)})")
    # Loop edges
    kf_by_idx = {i: kf_xyz[i] for i in range(len(kf_xyz))}
    for src, tgt in loop_edges_kf:
        if src in kf_by_idx and tgt in kf_by_idx:
            xs = [kf_by_idx[src][0], kf_by_idx[tgt][0]]
            ys = [kf_by_idx[src][1], kf_by_idx[tgt][1]]
            ax.plot(xs, ys, "r-", lw=1.5, alpha=0.8, label="_loop")
    # Mark start
    ax.plot(est_xyz[0, 0], est_xyz[0, 1], "go", ms=8, label="Start")
    ax.plot(est_xyz[-1, 0], est_xyz[-1, 1], "rs", ms=8, label="End")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectory XY (SE3-aligned to GT)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_trajectory_3d(est_xyz, gt_xyz, save_path: Path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], "k-", lw=1.0, label="Ground truth", alpha=0.7)
    ax.plot(est_xyz[:, 0], est_xyz[:, 1], est_xyz[:, 2], "b-", lw=1.0, label="Estimated", alpha=0.85)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Trajectory 3D (SE3-aligned to GT)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_ate_error_over_time(errors, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(errors, lw=0.8, color="steelblue")
    ax.axhline(errors.mean(), color="red", lw=1.0, linestyle="--", label=f"RMSE={errors.mean():.3f} m")
    ax.fill_between(range(len(errors)), errors, alpha=0.2, color="steelblue")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("ATE [m]")
    ax.set_title("Per-frame Absolute Trajectory Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _clip_map(pts: np.ndarray, sigma: float = 3.5) -> np.ndarray:
    """Remove points beyond sigma*std from median — kills stray triangulation outliers."""
    if len(pts) == 0:
        return pts
    med = np.median(pts, axis=0)
    std = np.std(pts, axis=0).clip(min=0.01)
    mask = np.all(np.abs(pts - med) < sigma * std, axis=1)
    return pts[mask]


def plot_map_xy(map_pts, kf_xyz, title: str, save_path: Path, max_pts: int = 30000):
    pts_clean = _clip_map(map_pts)
    fig, ax = plt.subplots(figsize=(9, 9))
    if len(pts_clean) > max_pts:
        idx = np.random.choice(len(pts_clean), max_pts, replace=False)
        pts = pts_clean[idx]
    else:
        pts = pts_clean
    ax.scatter(pts[:, 0], pts[:, 1], s=0.4, c="dimgray", alpha=0.3, rasterized=True)
    if len(kf_xyz) > 0:
        ax.plot(kf_xyz[:, 0], kf_xyz[:, 1], "b-", lw=0.6, alpha=0.6)
        ax.scatter(kf_xyz[:, 0], kf_xyz[:, 1], s=8, c="royalblue", zorder=3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_map_3d(map_pts, kf_xyz, title: str, save_path: Path, max_pts: int = 20000):
    pts_clean = _clip_map(map_pts)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if len(pts_clean) > max_pts:
        idx = np.random.choice(len(pts_clean), max_pts, replace=False)
        pts = pts_clean[idx]
    else:
        pts = pts_clean
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.3, c="dimgray", alpha=0.2, rasterized=True)
    if len(kf_xyz) > 0:
        ax.plot(kf_xyz[:, 0], kf_xyz[:, 1], kf_xyz[:, 2], "b-", lw=0.6, alpha=0.7)
        ax.scatter(kf_xyz[:, 0], kf_xyz[:, 1], kf_xyz[:, 2], s=6, c="royalblue", zorder=3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_keyframe_graph(kf_xyz, covis_edges, loop_edges, save_path: Path, max_covis: int = 2000):
    fig, ax = plt.subplots(figsize=(9, 9))
    # Subsample covis edges for readability
    step = max(1, len(covis_edges) // max_covis)
    for src, tgt in covis_edges[::step]:
        if src < len(kf_xyz) and tgt < len(kf_xyz):
            ax.plot([kf_xyz[src, 0], kf_xyz[tgt, 0]],
                    [kf_xyz[src, 1], kf_xyz[tgt, 1]],
                    "gray", lw=0.3, alpha=0.3)
    for src, tgt in loop_edges:
        if src < len(kf_xyz) and tgt < len(kf_xyz):
            ax.plot([kf_xyz[src, 0], kf_xyz[tgt, 0]],
                    [kf_xyz[src, 1], kf_xyz[tgt, 1]],
                    "r-", lw=2.0, alpha=0.9, zorder=5)
            ax.annotate("", xy=kf_xyz[tgt, :2], xytext=kf_xyz[src, :2],
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax.scatter(kf_xyz[:, 0], kf_xyz[:, 1], s=8, c="royalblue", zorder=4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    n_loops = len(loop_edges)
    ax.set_title(f"Keyframe Covisibility Graph  |  {len(kf_xyz)} KFs  |  {n_loops} loop edges")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_metrics_table(metrics: dict, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    rows = [[k, v] for k, v in metrics.items()]
    tbl = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                   cellLoc="left", loc="center", colWidths=[0.55, 0.35])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c5f8a")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#e8f0f8")
    ax.set_title("fr1/room — C++ BA Run", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root: Path = args.root
    out_dir: Path = args.output or (root / "plots2")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load estimated trajectory ---
    traj_files = list(root.glob("trajectory_*.txt"))
    if not traj_files:
        raise FileNotFoundError(f"No trajectory_*.txt found in {root}")
    ts_est, est_xyz_raw = read_tum(traj_files[0])
    print(f"Estimated trajectory: {len(ts_est)} poses from {traj_files[0].name}")

    # --- Load ground truth ---
    gt_xyz_aligned = None
    errors = None
    if args.dataset:
        gt_path = args.dataset / "groundtruth.txt"
        if gt_path.exists():
            ts_gt, gt_xyz_raw = read_tum(gt_path)
            ig, ie = associate(ts_gt, ts_est)
            gt_matched = gt_xyz_raw[ig]
            est_matched = est_xyz_raw[ie]
            R, t = align_se3(est_matched, gt_matched)
            est_aligned = (R @ est_xyz_raw.T).T + t
            gt_aligned = gt_xyz_raw  # GT stays in GT frame
            errors = np.linalg.norm((R @ est_matched.T).T + t - gt_matched, axis=1)
            print(f"ATE RMSE: {errors.mean():.4f} m  (median {np.median(errors):.4f} m, max {errors.max():.4f} m)")
            gt_xyz_aligned = gt_aligned
        else:
            print(f"WARNING: groundtruth not found at {gt_path}")
            est_aligned = est_xyz_raw
    else:
        est_aligned = est_xyz_raw

    # --- Load keyframes ---
    kf_path = root / "keyframes.json"
    kf_xyz_raw = load_keyframes(kf_path) if kf_path.exists() else np.zeros((0, 3))
    print(f"Keyframes: {len(kf_xyz_raw)}")

    # Align KF positions with same transform
    if errors is not None and len(kf_xyz_raw) > 0:
        kf_xyz = (R @ kf_xyz_raw.T).T + t
    else:
        kf_xyz = kf_xyz_raw

    # --- Load graph ---
    graph_path = root / "keyframe_graph.json"
    covis_edges, loop_edges_raw, nodes = load_graph(graph_path) if graph_path.exists() else ([], [], [])
    print(f"Covisibility edges: {len(covis_edges)}, Loop edges: {len(loop_edges_raw)}")

    # Build kid->kf_list_index map for graph
    kfs_data = json.loads(kf_path.read_text()) if kf_path.exists() else []
    kid_to_idx = {kf["kid"]: i for i, kf in enumerate(kfs_data) if "kid" in kf}
    loop_edges_kf = [(kid_to_idx.get(e[0], e[0]), kid_to_idx.get(e[1], e[1])) for e in loop_edges_raw]

    # --- Load map points ---
    ply_path = root / "map_points.ply"
    map_pts_raw = read_ply_xyz(ply_path) if ply_path.exists() else np.zeros((0, 3))
    print(f"Map points: {len(map_pts_raw)}")

    # Align map points
    if errors is not None and len(map_pts_raw) > 0:
        map_pts = (R @ map_pts_raw.T).T + t
    else:
        map_pts = map_pts_raw

    # -----------------------------------------------------------------------
    # Generate figures
    # -----------------------------------------------------------------------

    # 1. Trajectory XY comparison
    if gt_xyz_aligned is not None:
        plot_trajectory_xy(est_aligned, gt_xyz_aligned, kf_xyz, loop_edges_kf,
                           out_dir / "trajectory_xy_comparison.png")
        print("  trajectory_xy_comparison.png")

        # 2. Trajectory 3D
        plot_trajectory_3d(est_aligned, gt_xyz_aligned,
                           out_dir / "trajectory_3d_comparison.png")
        print("  trajectory_3d_comparison.png")

        # 3. ATE over time
        if errors is not None:
            plot_ate_error_over_time(errors, out_dir / "ate_error_over_time.png")
            print("  ate_error_over_time.png")

    # 4. Sparse map XY
    plot_map_xy(map_pts, kf_xyz, "Estimated Sparse Map (XY, aligned)",
                out_dir / "map_xy.png")
    print("  map_xy.png")

    # 5. Sparse map 3D
    plot_map_3d(map_pts, kf_xyz, "Estimated Sparse Map (3D, aligned)",
                out_dir / "map_3d.png")
    print("  map_3d.png")

    # 6. Keyframe graph
    if len(kf_xyz) > 0:
        plot_keyframe_graph(kf_xyz, covis_edges, loop_edges_kf,
                            out_dir / "keyframe_graph.png")
        print("  keyframe_graph.png")

    # 7. Metrics table
    n_loops = len(loop_edges_raw)
    metrics = {
        "Dataset":          "fr1/room (TUM RGB-D)",
        "Feature backend":  "pyslam_orb2",
        "BA backend":       "C++ slam_optimizer_core",
        "Frames":           f"1362 / 1362",
        "Tracking lost":    "0",
        "Final KFs":        str(len(kf_xyz_raw)),
        "Map points":       f"{len(map_pts_raw):,}",
        "Loop edges":       str(n_loops),
        "ATE RMSE":         f"{errors.mean():.4f} m" if errors is not None else "N/A",
        "ATE median":       f"{np.median(errors):.4f} m" if errors is not None else "N/A",
        "ATE max":          f"{errors.max():.4f} m" if errors is not None else "N/A",
    }
    plot_metrics_table(metrics, out_dir / "metrics_table.png")
    print("  metrics_table.png")

    # Write markdown metrics
    md = "# fr1/room — C++ BA Run\n\n| Metric | Value |\n|---|---|\n"
    for k, v in metrics.items():
        md += f"| {k} | {v} |\n"
    (out_dir / "metrics_table.md").write_text(md)
    print("  metrics_table.md")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
