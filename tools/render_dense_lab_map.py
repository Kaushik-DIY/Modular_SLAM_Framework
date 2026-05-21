#!/usr/bin/env python3
"""
Render high-density presentation maps from a completed RGB-D SLAM run.

Strategy
--------
1. Project EVERY pixel-depth observation through each keyframe's optimised Twc
   into world coordinates (stride=2 instead of the default 5).
2. Voxel-grid downsample the resulting cloud at a fine resolution (default 1 cm)
   to merge multi-view observations into clean, defined surfaces. Per voxel:
     - position = mean of contained points
     - color    = mean RGB of contained points
     - count    = how many observations fell into this voxel
3. Render four presentation views:
     - dense_rgb_topdown.png       Top-down with RGB color
     - dense_height_topdown.png    Top-down with height (Y) as color (clean shape)
     - dense_rgb_3d.png            3D perspective with RGB color
     - dense_occupancy_topdown.png Top-down with observation density as color (heatmap)

This does not implement TSDF fusion / mesh extraction — those need Open3D —
but voxel-binning + per-voxel color averaging gives substantially crisper
results than the current "every-5th-pixel + 3σ filter" approach.

Usage
-----
python -m tools.render_dense_lab_map \\
    --run visual_slam_outputs/checkpoint_2_36Z3_lab_phase3_tightened \\
    --dataset datasets/lab_rgbd_run_2 \\
    --output visual_slam_outputs/checkpoint_2_36Z3_lab_phase3_tightened/dense_map
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_camera_yaml(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("%YAML"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            v = v.strip()
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v
    return out


def read_associations(path: Path) -> list[tuple[float, str, str]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            out.append((float(parts[0]), parts[1], parts[3]))
    return out


def build_dense_cloud(
    keyframes_json: Path,
    dataset_path: Path,
    *,
    stride: int = 2,
    min_depth_m: float = 0.20,
    max_depth_m: float = 4.0,
    use_every_n_kf: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project pixel-depth observations from all keyframes into the world frame.

    Returns (points Nx3, colors Nx3 uint8, kf_origins Mx3)
    """
    cam = read_camera_yaml(dataset_path / "camera.yaml")
    fx = float(cam["Camera.fx"])
    fy = float(cam["Camera.fy"])
    cx = float(cam["Camera.cx"])
    cy = float(cam["Camera.cy"])
    depth_factor = 1.0 / float(cam["DepthMapFactor"])

    kf_data = json.loads(keyframes_json.read_text())
    assoc = read_associations(dataset_path / "associations.txt")
    assoc_ts = np.asarray([a[0] for a in assoc], dtype=np.float64)

    uu = np.arange(0, 640, stride, dtype=np.float32)
    vv = np.arange(0, 480, stride, dtype=np.float32)
    U, V = np.meshgrid(uu, vv)
    xc_base = (U - cx) / fx
    yc_base = (V - cy) / fy

    pts_all = []
    col_all = []
    kf_origins = []

    n_used = 0
    for kf in kf_data[::use_every_n_kf]:
        try:
            Twc = np.asarray(kf["Twc"], dtype=np.float64).reshape(4, 4)
            ts = float(kf.get("timestamp") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue

        idx = int(np.argmin(np.abs(assoc_ts - ts)))
        _, rgb_rel, dep_rel = assoc[idx]

        rgb_img = cv2.imread(str(dataset_path / rgb_rel))
        dep_img = cv2.imread(str(dataset_path / dep_rel), cv2.IMREAD_UNCHANGED)
        if rgb_img is None or dep_img is None:
            continue

        dep_sub = dep_img[::stride, ::stride].astype(np.float32) * depth_factor
        rgb_sub = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)[::stride, ::stride]

        H, W = dep_sub.shape
        xc_b = xc_base[:H, :W]
        yc_b = yc_base[:H, :W]

        valid = (dep_sub > min_depth_m) & (dep_sub < max_depth_m)
        if not np.any(valid):
            continue

        z = dep_sub[valid]
        xc = xc_b[valid] * z
        yc = yc_b[valid] * z

        pts_cam = np.stack([xc, yc, z], axis=1)
        pts_world = (Twc[:3, :3] @ pts_cam.T).T + Twc[:3, 3]

        colors = rgb_sub[valid]
        pts_all.append(pts_world.astype(np.float32))
        col_all.append(colors)
        kf_origins.append(Twc[:3, 3])
        n_used += 1

    if not pts_all:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), np.empty((0, 3))

    pts = np.vstack(pts_all)
    col = np.vstack(col_all)
    kf_origins = np.asarray(kf_origins, dtype=np.float64)
    print(f"  raw cloud: {len(pts):,} pts from {n_used} keyframes (stride={stride})")
    return pts, col, kf_origins


def voxel_downsample(
    pts: np.ndarray,
    col: np.ndarray,
    voxel_size: float,
    min_obs_per_voxel: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin points into a voxel grid, average position and color per voxel.

    min_obs_per_voxel: require at least N observations per voxel (cleans up
    single-view noise / depth jitter).

    Returns (pts_voxel Mx3, col_voxel Mx3, obs_per_voxel Mx1)
    """
    if len(pts) == 0:
        return pts, col, np.array([])

    # Compute voxel indices
    origin = pts.min(axis=0)
    idx = np.floor((pts - origin) / voxel_size).astype(np.int64)

    # Pack 3D index into 1D key
    idx_min = idx.min(axis=0)
    idx -= idx_min
    extent = idx.max(axis=0) + 1
    # Use a stable encoding (safe for ~20m scenes at 1cm = ~2000^3)
    key = idx[:, 0].astype(np.int64) * (extent[1] * extent[2]) + idx[:, 1] * extent[2] + idx[:, 2]

    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    pts_sorted = pts[order]
    col_sorted = col[order].astype(np.float32)

    # Group by unique key
    unique_keys, starts, counts = np.unique(key_sorted, return_index=True, return_counts=True)

    # Filter by minimum observations
    keep = counts >= min_obs_per_voxel
    starts = starts[keep]
    counts = counts[keep]

    n = len(starts)
    pts_vox = np.empty((n, 3), dtype=np.float32)
    col_vox = np.empty((n, 3), dtype=np.uint8)
    for i, (s, c) in enumerate(zip(starts, counts)):
        seg_pts = pts_sorted[s:s + c]
        seg_col = col_sorted[s:s + c]
        pts_vox[i] = seg_pts.mean(axis=0)
        col_vox[i] = np.clip(seg_col.mean(axis=0), 0, 255).astype(np.uint8)

    print(f"  voxel-downsampled to {len(pts_vox):,} voxels at {voxel_size*1000:.0f} mm "
          f"(min_obs={min_obs_per_voxel}; from {len(pts):,} raw pts)")
    return pts_vox, col_vox, counts


def filter_to_scene(
    pts: np.ndarray,
    col: np.ndarray,
    obs: np.ndarray,
    *,
    x_lim: tuple = (-4.0, 4.0),
    y_lim: tuple = (-2.0, 2.5),
    z_lim: tuple = (-3.0, 12.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (
        (pts[:, 0] > x_lim[0]) & (pts[:, 0] < x_lim[1])
        & (pts[:, 1] > y_lim[0]) & (pts[:, 1] < y_lim[1])
        & (pts[:, 2] > z_lim[0]) & (pts[:, 2] < z_lim[1])
    )
    return pts[mask], col[mask], obs[mask] if len(obs) else obs


def plot_dense_topdown_rgb(pts, col, kf_origins, trajectory, out_path, title):
    fig, ax = plt.subplots(figsize=(11, 14))
    # Sort by Y (height) so floor (low Y) renders behind, walls/ceiling on top
    order = np.argsort(pts[:, 1])[::-1]
    ax.scatter(pts[order, 0], pts[order, 2],
               c=col[order] / 255.0, s=0.5, marker=".", linewidths=0)
    if trajectory is not None and len(trajectory) > 0:
        ax.plot(trajectory[:, 0], trajectory[:, 2], "-", color="#27ae60",
                lw=1.6, alpha=0.85, label="Trajectory")
        ax.scatter([trajectory[0, 0]], [trajectory[0, 2]], c="#2ecc71",
                   s=120, zorder=6, label="Start", edgecolors="black", linewidths=1)
        ax.scatter([trajectory[-1, 0]], [trajectory[-1, 2]], c="#e74c3c",
                   s=120, marker="X", zorder=6, label="End", edgecolors="black", linewidths=1)
    ax.set_xlabel("X (m, right)")
    ax.set_ylabel("Z (m, forward)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_facecolor("#0a0a0a")
    ax.legend(loc="upper right", facecolor="#222", labelcolor="white", edgecolor="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    fig.patch.set_facecolor("#101418")
    ax.grid(alpha=0.15, color="white")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path.name}")


def plot_dense_topdown_height(pts, kf_origins, trajectory, out_path, title):
    fig, ax = plt.subplots(figsize=(11, 14))
    # Color by -Y (height; lower Y = higher above floor on RGB-D where Y is down)
    height = -pts[:, 1]
    order = np.argsort(pts[:, 1])[::-1]
    sc = ax.scatter(pts[order, 0], pts[order, 2],
                    c=height[order], s=0.6, marker=".", linewidths=0,
                    cmap="viridis", vmin=np.percentile(height, 2), vmax=np.percentile(height, 98))
    if trajectory is not None and len(trajectory) > 0:
        ax.plot(trajectory[:, 0], trajectory[:, 2], "-", color="#ff5252", lw=1.6, alpha=0.85, label="Trajectory")
        ax.scatter([trajectory[0, 0]], [trajectory[0, 2]], c="#2ecc71",
                   s=120, zorder=6, label="Start", edgecolors="black", linewidths=1)
        ax.scatter([trajectory[-1, 0]], [trajectory[-1, 2]], c="#e74c3c",
                   s=120, marker="X", zorder=6, label="End", edgecolors="black", linewidths=1)
    ax.set_xlabel("X (m, right)")
    ax.set_ylabel("Z (m, forward)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, label="Height above floor (m)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path.name}")


def plot_dense_topdown_occupancy(pts, obs, trajectory, out_path, title):
    """Color by observation count — shows where multi-view fusion was strongest."""
    fig, ax = plt.subplots(figsize=(11, 14))
    # Log-scale: most voxels have 2-5 obs, structural surfaces have 20+
    log_obs = np.log10(obs.astype(float).clip(min=1))
    order = np.argsort(log_obs)
    sc = ax.scatter(pts[order, 0], pts[order, 2],
                    c=log_obs[order], s=0.6, marker=".", linewidths=0,
                    cmap="plasma", vmin=0, vmax=np.percentile(log_obs, 99))
    if trajectory is not None and len(trajectory) > 0:
        ax.plot(trajectory[:, 0], trajectory[:, 2], "-", color="#00ffcc", lw=1.6, alpha=0.85, label="Trajectory")
        ax.scatter([trajectory[0, 0]], [trajectory[0, 2]], c="#2ecc71",
                   s=120, zorder=6, label="Start", edgecolors="black", linewidths=1)
        ax.scatter([trajectory[-1, 0]], [trajectory[-1, 2]], c="#e74c3c",
                   s=120, marker="X", zorder=6, label="End", edgecolors="black", linewidths=1)
    ax.set_xlabel("X (m, right)")
    ax.set_ylabel("Z (m, forward)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label("log₁₀(observations per voxel)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path.name}")


def plot_dense_3d(pts, col, trajectory, out_path, title):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection="3d")
    # Subsample for 3D rendering speed
    n = len(pts)
    if n > 200_000:
        idx = np.random.default_rng(0).choice(n, 200_000, replace=False)
        pts = pts[idx]; col = col[idx]
    ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1],
               c=col / 255.0, s=0.5, marker=".", linewidths=0)
    if trajectory is not None and len(trajectory) > 0:
        ax.plot(trajectory[:, 0], trajectory[:, 2], -trajectory[:, 1],
                "-", color="#ff5252", lw=2.0, alpha=0.9, label="Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_zlabel("-Y (m, up)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.view_init(elev=25, azim=-50)
    ax.set_box_aspect((1, 1.4, 0.35))
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path.name}")


def load_trajectory(run_dir: Path) -> np.ndarray:
    cands = sorted(run_dir.glob("trajectory_*.txt"))
    cands = [c for c in cands if "completed_" not in c.name] or cands
    if not cands:
        return np.empty((0, 3))
    poses = []
    with cands[0].open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            poses.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(poses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=2,
                        help="Pixel stride (1=full density, 2=half, 5=existing)")
    parser.add_argument("--voxel-size", type=float, default=0.015,
                        help="Voxel grid size in metres (default 1.5 cm)")
    parser.add_argument("--max-depth", type=float, default=4.0,
                        help="Discard depths beyond this (m)")
    parser.add_argument("--min-obs", type=int, default=3,
                        help="Minimum observations per voxel to keep")
    parser.add_argument("--label", default=None, help="Title label for plots")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    label = args.label or args.run.name

    print(f"[1/3] Building dense cloud from {args.run.name} ...")
    kf_json = args.run / "keyframes.json"
    pts, col, kf_origins = build_dense_cloud(
        kf_json, args.dataset,
        stride=args.stride,
        max_depth_m=args.max_depth,
    )
    if len(pts) == 0:
        print("ERROR: no dense points produced — check dataset and run directory")
        return

    print(f"[2/3] Voxel-downsampling at {args.voxel_size*1000:.0f} mm "
          f"(min_obs={args.min_obs}) ...")
    pts_v, col_v, obs_v = voxel_downsample(pts, col, args.voxel_size,
                                            min_obs_per_voxel=args.min_obs)
    pts_v, col_v, obs_v = filter_to_scene(pts_v, col_v, obs_v)
    print(f"  scene-filtered to {len(pts_v):,} voxels")

    traj = load_trajectory(args.run)
    print(f"  trajectory: {len(traj)} poses")

    print("[3/3] Rendering presentation plots ...")
    plot_dense_topdown_rgb(
        pts_v, col_v, kf_origins, traj,
        args.output / "dense_rgb_topdown.png",
        f"Dense RGB map — {label}\n{len(pts_v):,} voxels @ {args.voxel_size*1000:.0f} mm")
    plot_dense_topdown_height(
        pts_v, kf_origins, traj,
        args.output / "dense_height_topdown.png",
        f"Dense map (height-colored) — {label}\n{len(pts_v):,} voxels @ {args.voxel_size*1000:.0f} mm")
    plot_dense_topdown_occupancy(
        pts_v, obs_v, traj,
        args.output / "dense_occupancy_topdown.png",
        f"Multi-view fusion density — {label}\nBright = surfaces observed by many KFs")
    plot_dense_3d(
        pts_v, col_v, traj,
        args.output / "dense_rgb_3d.png",
        f"Dense 3D reconstruction — {label}\n{len(pts_v):,} voxels (subsampled for 3D)")

    # Also save the cloud as PLY for downstream tools
    ply_path = args.output / "dense_cloud.ply"
    with ply_path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts_v)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts_v, col_v):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
    print(f"  saved: dense_cloud.ply ({len(pts_v):,} colored points)")

    print(f"\nDone. Outputs in {args.output}")


if __name__ == "__main__":
    main()
