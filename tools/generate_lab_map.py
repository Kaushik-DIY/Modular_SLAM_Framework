#!/usr/bin/env python3
"""
=============================================================================
tools/generate_lab_map.py

Comprehensive map visualiser for lab RGB-D ORB-SLAM runs.
Generates evaluation figures (pipeline assessment) and presentation
figures (supervisor-ready).

Requirements
------------
  After run:  map_points.ply  keyframes.json  keyframe_graph.json
  Dataset:    rgb/  depth/  associations.txt  camera.yaml  (for semi-dense)

Coordinate convention (ORB-SLAM2 world frame, Twc)
---------------------------------------------------
  X = right      (camera frame at first keyframe)
  Y = down       (height — stays near-constant for a floor robot)
  Z = forward
  Floor plan:    X (horizontal) vs Z (vertical, forward)
  Height axis:   −Y  (negated so "up" reads naturally on plots)

Figures generated
-----------------
  EVALUATION (pipeline quality assessment)
    eval_sparse_map.png           Sparse ORB feature map, top-down X–Z
    eval_trajectory_graph.png     Path + covisibility / spanning / loop edges
    eval_tracking_quality.png     Tracked pts, map growth, BA MSE over time

  PRESENTATION (supervisor-ready)
    pres_sparse_map.png           Polished sparse map, styled like TUM reference
    pres_semidense_topdown.png    Semi-dense RGB-D top-down projection
    pres_semidense_3d.png         Semi-dense RGB-D 3-D perspective
    pres_summary.png              One-page overview panel

Usage
-----
  python3 -m tools.generate_lab_map \\
      --run     visual_slam_outputs/lab_rgbd_run_2_map \\
      --dataset datasets/lab_rgbd_run_2 \\
      --output  visual_slam_outputs/lab_rgbd_run_2_map/map_figures
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _find_file(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def read_trajectory(path: Path) -> np.ndarray:
    """TUM format → Nx4 [ts, tx, ty, tz] (Twc camera centre in world)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 4:
                continue
            rows.append([float(p[0]), float(p[1]), float(p[2]), float(p[3])])
    return np.asarray(rows, dtype=np.float64)


def read_frame_log(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_ply(path: Path, max_pts: int = 500_000) -> np.ndarray:
    """ASCII PLY → Nx3 float64.  Uniform subsamples if larger than max_pts."""
    if not path.exists():
        return np.empty((0, 3), dtype=np.float64)
    pts = []
    in_header = True
    with open(path) as f:
        for line in f:
            if in_header:
                if line.strip() == "end_header":
                    in_header = False
                continue
            p = line.split()
            if len(p) >= 3:
                try:
                    pts.append([float(p[0]), float(p[1]), float(p[2])])
                except ValueError:
                    continue
    arr = np.asarray(pts, dtype=np.float64)
    if len(arr) > max_pts:
        arr = arr[np.linspace(0, len(arr) - 1, max_pts, dtype=np.int64)]
    return arr


def filter_map_points_to_scene(
    pts: np.ndarray,
    ref_pts: np.ndarray,
    padding_m: float = 3.0,
) -> np.ndarray:
    """Remove outlier map points outside the trajectory bounding box + padding.

    Triangulation failures on near-parallel rays produce points at hundreds of
    metres, collapsing the plot axes.  Clipping to the trajectory bbox + padding
    discards these while keeping all room-scale points.
    """
    if len(pts) == 0 or len(ref_pts) == 0:
        return pts
    lo = ref_pts.min(axis=0) - padding_m
    hi = ref_pts.max(axis=0) + padding_m
    mask = np.all((pts >= lo) & (pts <= hi), axis=1)
    n_removed = len(pts) - int(mask.sum())
    if n_removed > 0:
        print(f"  [map filter] removed {n_removed} outlier map points "
              f"(kept {int(mask.sum())}/{len(pts)})")
    return pts[mask]


def read_keyframes(path: Path) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, float]]:
    """
    Returns:
        positions   Nx3 world positions
        lookup      {kid: position_xyz}
        ts_lookup   {kid: timestamp}
    """
    if not path.exists():
        return np.empty((0, 3)), {}, {}
    data = json.loads(path.read_text())
    positions, lookup, ts_lookup = [], {}, {}
    for kf in data:
        try:
            pos = np.asarray(kf["position"], dtype=np.float64)
            kid = int(kf["kid"])
            positions.append(pos)
            lookup[kid] = pos
            ts_lookup[kid] = float(kf.get("timestamp") or 0.0)
        except (KeyError, TypeError):
            continue
    return np.asarray(positions, dtype=np.float64), lookup, ts_lookup


def read_graph(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_camera_yaml(path: Path) -> dict:
    params: dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            m = re.match(r"^([A-Za-z0-9_.]+)\s*:\s*(.+)$", line)
            if m:
                try:
                    params[m.group(1)] = float(m.group(2).strip())
                except ValueError:
                    params[m.group(1)] = m.group(2).strip()
    return params


def read_associations(path: Path) -> list[tuple[float, str, str]]:
    """Returns list of (timestamp, rgb_relpath, depth_relpath)."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                entries.append((float(parts[0]), parts[1], parts[3]))
    return entries


# ---------------------------------------------------------------------------
# Semi-dense RGB-D reconstruction
# ---------------------------------------------------------------------------

def build_semidense_cloud(
    keyframes_json: Path,
    dataset_path: Path,
    stride: int = 5,
    max_depth_m: float = 3.0,
    max_total_pts: int = 1_500_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project depth+RGB images into world frame for each keyframe.

    Uses Twc from keyframes.json and actual sensor images from dataset.
    Returns:
        points  Nx3 float64  world XYZ
        colors  Nx3 uint8    RGB
    """
    import cv2

    if not keyframes_json.exists():
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)

    cam_params = read_camera_yaml(dataset_path / "camera.yaml")
    fx = float(cam_params["Camera.fx"])
    fy = float(cam_params["Camera.fy"])
    cx = float(cam_params["Camera.cx"])
    cy = float(cam_params["Camera.cy"])
    depth_factor = 1.0 / float(cam_params["DepthMapFactor"])  # raw → metres

    kf_data = json.loads(keyframes_json.read_text())
    assoc = read_associations(dataset_path / "associations.txt")
    assoc_ts = np.asarray([a[0] for a in assoc], dtype=np.float64)

    # Pre-build pixel grid for vectorised projection
    uu = np.arange(0, 640, stride, dtype=np.float32)
    vv = np.arange(0, 480, stride, dtype=np.float32)
    U, V = np.meshgrid(uu, vv)
    xc_base = (U - cx) / fx   # shape (H/s, W/s)
    yc_base = (V - cy) / fy

    all_pts, all_col = [], []

    # Use every 2nd keyframe to keep cloud manageable
    kf_step = max(1, len(kf_data) // 100)

    for kf in kf_data[::kf_step]:
        try:
            Twc = np.asarray(kf["Twc"], dtype=np.float64).reshape(4, 4)
            ts = float(kf.get("timestamp") or 0.0)
        except (KeyError, TypeError):
            continue

        # Find nearest association
        idx = int(np.argmin(np.abs(assoc_ts - ts)))
        _, rgb_rel, dep_rel = assoc[idx]

        rgb_img = cv2.imread(str(dataset_path / rgb_rel))
        dep_img = cv2.imread(str(dataset_path / dep_rel), cv2.IMREAD_UNCHANGED)
        if rgb_img is None or dep_img is None:
            continue

        # Subsample depth and RGB
        dep_sub = dep_img[::stride, ::stride].astype(np.float32) * depth_factor
        rgb_sub = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)[::stride, ::stride]

        H, W = dep_sub.shape
        xc_b = xc_base[:H, :W]
        yc_b = yc_base[:H, :W]

        valid = (dep_sub > 0.1) & (dep_sub < max_depth_m)
        if not np.any(valid):
            continue

        z = dep_sub[valid]
        xc = xc_b[valid] * z
        yc = yc_b[valid] * z

        # Camera frame point cloud: [xc, yc, z] in (X_right, Y_down, Z_forward)
        pts_cam = np.stack([xc, yc, z], axis=1)          # Nx3
        pts_world = (Twc[:3, :3] @ pts_cam.T).T + Twc[:3, 3]

        colors = rgb_sub[valid]  # Nx3

        all_pts.append(pts_world.astype(np.float32))
        all_col.append(colors)

        if sum(len(p) for p in all_pts) > max_total_pts:
            break

    if not all_pts:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)

    pts = np.vstack(all_pts).astype(np.float64)
    col = np.vstack(all_col)

    # Remove statistical outliers: keep points within 3σ of centroid
    if len(pts) > 100:
        dists = np.linalg.norm(pts - np.median(pts, axis=0), axis=1)
        sigma = np.std(dists)
        mask = dists < np.median(dists) + 3.5 * sigma
        pts, col = pts[mask], col[mask]

    print(f"  semi-dense cloud: {len(pts):,} points  "
          f"(from {len(kf_data[::kf_step])} keyframes, stride={stride})")
    return pts, col


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

def _set_equal_xz(ax, arrays: list[np.ndarray], pad: float = 0.08) -> None:
    """Equal-aspect square bounds for X–Z (columns 0 and 2 of Nx3 arrays)."""
    pts_list = []
    for a in arrays:
        if len(a) == 0:
            continue
        if a.ndim == 2 and a.shape[1] >= 3:
            pts_list.append(a[:, [0, 2]])
    if not pts_list:
        return
    pts = np.vstack(pts_list)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return
    # Robust bounds: clip 0.5% outliers
    lo = np.percentile(pts, 0.5, axis=0)
    hi = np.percentile(pts, 99.5, axis=0)
    center = (lo + hi) * 0.5
    span = max(float(np.max(hi - lo)), 0.5)
    half = span / 2 + span * pad
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal", adjustable="box")


def _draw_graph_edges(ax, graph: dict, kf_lookup: dict[int, np.ndarray]) -> None:
    """Draw covisibility, spanning-tree, and loop edges on an X–Z plot."""
    for edge in graph.get("covisibility_edges", []):
        a = kf_lookup.get(int(edge.get("source", -1)))
        b = kf_lookup.get(int(edge.get("target", -1)))
        if a is not None and b is not None:
            ax.plot([a[0], b[0]], [a[2], b[2]], color="#aab7b8", lw=0.3, alpha=0.35, zorder=1)
    for edge in graph.get("spanning_tree_edges", []):
        a = kf_lookup.get(int(edge.get("source", -1)))
        b = kf_lookup.get(int(edge.get("target", -1)))
        if a is not None and b is not None:
            ax.plot([a[0], b[0]], [a[2], b[2]], color="#626567", lw=0.7, alpha=0.55, zorder=2)
    for edge in graph.get("loop_edges", []):
        a = kf_lookup.get(int(edge.get("source", -1)))
        b = kf_lookup.get(int(edge.get("target", -1)))
        if a is not None and b is not None:
            ax.plot([a[0], b[0]], [a[2], b[2]], color="#e74c3c", lw=1.8, alpha=0.9, zorder=3)


def _find_gba_timestamps(frame_log: list[dict]) -> list[float]:
    ts = []
    for i in range(1, len(frame_log)):
        if (frame_log[i]["loop_global_ba_started"] == "1"
                and frame_log[i - 1]["loop_global_ba_started"] == "0"):
            ts.append(float(frame_log[i]["timestamp"]))
    return ts


# ---------------------------------------------------------------------------
# EVALUATION figures
# ---------------------------------------------------------------------------

def plot_eval_sparse_map(
    run_dir: Path,
    map_pts: np.ndarray,
    traj: np.ndarray,
    kf_pos: np.ndarray,
    output_path: Path,
) -> None:
    """
    Sparse ORB feature map — top-down X–Z.
    Styled like the TUM reference (dark points, white background).
    """
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.set_facecolor("white")

    if len(map_pts) > 0:
        ax.scatter(map_pts[:, 0], map_pts[:, 2],
                   s=0.6, color="#273746", alpha=0.55, linewidths=0)

    _set_equal_xz(ax, [map_pts, kf_pos] if len(kf_pos) > 0 else [map_pts])

    ax.set_xlabel("x  [m]", fontsize=10)
    ax.set_ylabel("z  [m]", fontsize=10)
    ax.set_title(f"Estimated sparse map  ({run_dir.name})\n"
                 f"{len(map_pts):,} ORB feature points  |  ORB-SLAM2 world frame  X–Z",
                 fontsize=9)
    ax.grid(True, alpha=0.2, lw=0.4, color="#cccccc")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  saved: {output_path.name}")


def plot_eval_trajectory_graph(
    run_dir: Path,
    map_pts: np.ndarray,
    traj: np.ndarray,
    kf_pos: np.ndarray,
    kf_lookup: dict,
    graph: dict,
    output_path: Path,
) -> None:
    """
    Trajectory + keyframe covisibility / spanning-tree / loop graph
    overlaid on sparse map.  Primary evaluation figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.set_facecolor("white")

    # Sparse map background
    if len(map_pts) > 0:
        ax.scatter(map_pts[:, 0], map_pts[:, 2],
                   s=0.5, color="#aab7b8", alpha=0.35, linewidths=0, zorder=1)

    # Keyframe graph edges
    _draw_graph_edges(ax, graph, kf_lookup)

    # Trajectory
    if len(traj) > 0:
        ax.plot(traj[:, 1], traj[:, 3], color="#2980b9", lw=0.9,
                alpha=0.7, zorder=4, label="Trajectory")

    # Keyframe nodes
    if len(kf_pos) > 0:
        ax.scatter(kf_pos[:, 0], kf_pos[:, 2], s=18, color="#e67e22",
                   zorder=5, linewidths=0.4, edgecolors="#ca6f1e",
                   label=f"Keyframes ({len(kf_pos)})")

    # Start / end
    ax.scatter(traj[0, 1], traj[0, 3], s=90, marker="o", color="#27ae60",
               zorder=7, label="Start")
    ax.scatter(traj[-1, 1], traj[-1, 3], s=90, marker="X", color="#c0392b",
               zorder=7, label="End")

    n_loop = len(graph.get("loop_edges", []))
    n_covis = len(graph.get("covisibility_edges", []))

    # Legend patches for edge types
    legend_extras = [
        mpatches.Patch(color="#aab7b8", label=f"Covisibility ({n_covis})"),
        mpatches.Patch(color="#626567", label="Spanning tree"),
        mpatches.Patch(color="#e74c3c", label=f"Loop edges ({n_loop})"),
    ]

    _set_equal_xz(ax, [a for a in [map_pts, kf_pos] if len(a) > 0])
    ax.set_xlabel("x  [m]", fontsize=10)
    ax.set_ylabel("z  [m]  (forward)", fontsize=10)
    ax.set_title(
        f"Trajectory + keyframe graph  ({run_dir.name})\n"
        f"KFs: {len(kf_pos)}   Loop edges: {n_loop}   Map pts: {len(map_pts):,}",
        fontsize=9,
    )
    ax.grid(True, alpha=0.15, lw=0.4, color="#cccccc")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + legend_extras, labels + [p.get_label() for p in legend_extras],
              fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  saved: {output_path.name}")


def plot_eval_tracking_quality(
    run_dir: Path,
    frame_log: list[dict],
    gba_ts: list[float],
    output_path: Path,
) -> None:
    """Tracking stats panel: tracked pts / map pts / KF count / BA MSE."""
    t0 = float(frame_log[0]["timestamp"])
    t_sec = [float(r["timestamp"]) - t0 for r in frame_log]
    tracked = [int(r["last_tracked"]) if r["last_tracked"] else 0 for r in frame_log]
    map_pts = [int(r["points"]) for r in frame_log]
    kf_cnts = [int(r["keyframes"]) for r in frame_log]
    ba_mse_raw = []
    for r in frame_log:
        try:
            ba_mse_raw.append(float(r["last_ba_mse"]))
        except (ValueError, TypeError):
            ba_mse_raw.append(float("nan"))
    gba_sec = [ts - t0 for ts in gba_ts]

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), dpi=150, sharex=True)
    fig.suptitle(f"Pipeline tracking quality  —  {run_dir.name}", fontsize=10)

    styles = dict(lw=0.8, alpha=0.85)
    for ts in gba_sec:
        for ax in axes:
            ax.axvline(ts, color="#1abc9c", lw=1.5, ls="--", alpha=0.75)

    axes[0].plot(t_sec, tracked, color="#2980b9", **styles)
    axes[0].set_ylabel("Tracked\nmap pts", fontsize=8)
    axes[0].grid(True, alpha=0.25)
    if gba_sec:
        axes[0].axvline(gba_sec[0], color="#1abc9c", lw=1.5, ls="--", alpha=0.75,
                        label=f"Global BA ({len(gba_sec)}×)")
        axes[0].legend(fontsize=7)

    axes[1].plot(t_sec, map_pts, color="#8e44ad", **styles)
    axes[1].set_ylabel("Total\nmap pts", fontsize=8)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(t_sec, kf_cnts, color="#e67e22", **styles)
    axes[2].set_ylabel("Keyframe\ncount", fontsize=8)
    axes[2].grid(True, alpha=0.25)

    valid_mse = [v for v in ba_mse_raw if np.isfinite(v)]
    p99 = np.percentile(valid_mse, 99) if valid_mse else 10.0
    axes[3].plot(t_sec, ba_mse_raw, color="#e74c3c", **styles)
    axes[3].set_ylim(0, p99 * 1.15)
    axes[3].set_ylabel("BA MSE\n(pose opt)", fontsize=8)
    axes[3].set_xlabel("Elapsed time  [s]", fontsize=9)
    axes[3].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {output_path.name}")


# ---------------------------------------------------------------------------
# PRESENTATION figures
# ---------------------------------------------------------------------------

def plot_pres_sparse_map(
    run_dir: Path,
    map_pts: np.ndarray,
    traj: np.ndarray,
    kf_pos: np.ndarray,
    graph: dict,
    kf_lookup: dict,
    output_path: Path,
) -> None:
    """
    Polished two-panel figure mirroring the TUM reference style.
    Left:  sparse feature map (clean dark dots)
    Right: trajectory + keyframe graph (showing loop closure)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=150)
    for ax in axes:
        ax.set_facecolor("white")

    # Left: sparse map
    ax = axes[0]
    if len(map_pts) > 0:
        ax.scatter(map_pts[:, 0], map_pts[:, 2],
                   s=0.7, color="#273746", alpha=0.6, linewidths=0)
    _set_equal_xz(ax, [map_pts])
    ax.set_xlabel("x  [m]", fontsize=10)
    ax.set_ylabel("z  [m]", fontsize=10)
    ax.set_title("Estimated sparse map", fontsize=11)
    ax.grid(True, alpha=0.18, lw=0.4, color="#cccccc")

    # Right: trajectory + graph
    ax = axes[1]
    if len(map_pts) > 0:
        ax.scatter(map_pts[:, 0], map_pts[:, 2],
                   s=0.4, color="#aab7b8", alpha=0.25, linewidths=0, zorder=1)
    _draw_graph_edges(ax, graph, kf_lookup)
    if len(traj) > 0:
        ax.plot(traj[:, 1], traj[:, 3], color="#2980b9",
                lw=1.0, alpha=0.8, zorder=4, label="Trajectory")
    if len(kf_pos) > 0:
        ax.scatter(kf_pos[:, 0], kf_pos[:, 2], s=16, color="#e67e22",
                   zorder=5, linewidths=0, label=f"Keyframes ({len(kf_pos)})")
    ax.scatter(traj[0, 1], traj[0, 3], s=80, marker="o",
               color="#27ae60", zorder=7, label="Start")
    ax.scatter(traj[-1, 1], traj[-1, 3], s=80, marker="X",
               color="#c0392b", zorder=7, label="End")
    n_loop = len(graph.get("loop_edges", []))
    loop_patch = mpatches.Patch(color="#e74c3c",
                                label=f"Loop closure edges ({n_loop})")
    _set_equal_xz(ax, [a for a in [map_pts, kf_pos] if len(a) > 0])
    ax.set_xlabel("x  [m]", fontsize=10)
    ax.set_ylabel("z  [m]", fontsize=10)
    ax.set_title("Trajectory + keyframe graph", fontsize=11)
    ax.grid(True, alpha=0.18, lw=0.4, color="#cccccc")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [loop_patch], labels + [loop_patch.get_label()],
              fontsize=8, loc="best")

    fig.suptitle(
        f"ORB-SLAM2 RGB-D  —  {run_dir.name}\n"
        f"Map pts: {len(map_pts):,}   KFs: {len(kf_pos)}   "
        f"Loop edges: {n_loop}   (ORB2 features, Intel RealSense)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  saved: {output_path.name}")


def plot_pres_semidense_topdown(
    run_dir: Path,
    pts: np.ndarray,
    colors: np.ndarray,
    traj: np.ndarray,
    output_path: Path,
) -> None:
    """
    Semi-dense RGB-D map — top-down X–Z, coloured with actual camera RGB.
    Shows recognisable room structure (walls, floor, furniture outlines).
    """
    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
    ax.set_facecolor("#1a1a1a")

    if len(pts) > 0:
        # Normalise colours to [0,1]
        rgba = colors.astype(np.float32) / 255.0
        ax.scatter(pts[:, 0], pts[:, 2],
                   c=rgba, s=0.5, alpha=0.65, linewidths=0)

    # Trajectory overlay
    if len(traj) > 0:
        ax.plot(traj[:, 1], traj[:, 3], color="white", lw=1.0,
                alpha=0.7, zorder=5, label="Trajectory")
    ax.scatter(traj[0, 1], traj[0, 3], s=70, marker="o",
               color="#2ecc71", zorder=6, label="Start")
    ax.scatter(traj[-1, 1], traj[-1, 3], s=70, marker="X",
               color="#e74c3c", zorder=6, label="End")

    _set_equal_xz(ax, [pts])

    ax.set_xlabel("x  [m]", fontsize=10, color="white")
    ax.set_ylabel("z  [m]  (forward)", fontsize=10, color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#555555")
    ax.set_title(
        f"Semi-dense RGB-D map — top-down  ({run_dir.name})\n"
        f"{len(pts):,} points  |  Intel RealSense D4xx  |  ORB-SLAM2 poses",
        fontsize=9, color="white",
    )
    ax.grid(True, alpha=0.15, lw=0.4, color="#555555")
    ax.legend(fontsize=8, facecolor="#333333", labelcolor="white")
    fig.set_facecolor("#1a1a1a")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="#1a1a1a")
    plt.close(fig)
    print(f"  saved: {output_path.name}")


def plot_pres_semidense_3d(
    run_dir: Path,
    pts: np.ndarray,
    colors: np.ndarray,
    traj: np.ndarray,
    output_path: Path,
) -> None:
    """
    Semi-dense RGB-D map — 3-D perspective.
    Y is negated so the plot reads naturally (up = positive).
    """
    fig = plt.figure(figsize=(11, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    fig.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    if len(pts) > 0:
        # Subsample for 3-D rendering
        step = max(1, len(pts) // 80_000)
        p = pts[::step]
        c = colors[::step].astype(np.float32) / 255.0
        ax.scatter(p[:, 0], p[:, 2], -p[:, 1],
                   c=c, s=0.5, alpha=0.5, linewidths=0)

    if len(traj) > 0:
        ax.plot(traj[:, 1], traj[:, 3], -traj[:, 2],
                color="white", lw=0.9, alpha=0.7, label="Trajectory")
    ax.scatter(traj[0, 1], traj[0, 3], -traj[0, 2],
               s=60, marker="o", color="#2ecc71", zorder=6, label="Start")
    ax.scatter(traj[-1, 1], traj[-1, 3], -traj[-1, 2],
               s=60, marker="X", color="#e74c3c", zorder=6, label="End")

    ax.set_xlabel("x [m]", fontsize=7, color="white", labelpad=3)
    ax.set_ylabel("z [m]  (fwd)", fontsize=7, color="white", labelpad=3)
    ax.set_zlabel("up [m]  (−y)", fontsize=7, color="white", labelpad=3)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor("#333333")
        axis._axinfo["grid"]["color"] = "#333333"
    ax.tick_params(colors="white", labelsize=6)
    ax.set_title(
        f"Semi-dense RGB-D map — 3D  ({run_dir.name})\n"
        f"Colour = actual camera RGB  |  Y negated for natural up-view",
        fontsize=8, color="white",
    )
    ax.legend(fontsize=7, facecolor="#222222", labelcolor="white")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  saved: {output_path.name}")


def plot_pres_summary(
    run_dir: Path,
    map_pts: np.ndarray,
    traj: np.ndarray,
    kf_pos: np.ndarray,
    kf_lookup: dict,
    graph: dict,
    frame_log: list[dict],
    gba_ts: list[float],
    output_path: Path,
) -> None:
    """One-page supervisor summary: sparse map, trajectory graph, stats."""
    fig = plt.figure(figsize=(16, 10), dpi=150)

    ok_count = sum(1 for r in frame_log if r["state"] == "OK")
    lost_count = sum(1 for r in frame_log if r["state"] == "LOST")
    ba_vals = [float(r["last_ba_mse"]) for r in frame_log
               if r["last_ba_mse"] and r["last_ba_mse"] not in ("", "None")]
    x_ext = traj[:, 1].ptp()
    z_ext = traj[:, 3].ptp()
    y_ext = traj[:, 2].ptp()
    start_end = float(np.linalg.norm(traj[-1, 1:4] - traj[0, 1:4]))
    n_loop = len(graph.get("loop_edges", []))
    duration = float(frame_log[-1]["timestamp"]) - float(frame_log[0]["timestamp"])

    fig.suptitle(
        f"ORB-SLAM2 RGB-D Pipeline — Lab Room Run  ({run_dir.name})\n"
        f"Intel RealSense D4xx  ·  ORB2 features  ·  Loop closure + Global BA",
        fontsize=11, fontweight="bold",
    )

    # ---- sparse map (top-left) ----
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor("white")
    if len(map_pts) > 0:
        ax1.scatter(map_pts[:, 0], map_pts[:, 2],
                    s=0.6, color="#273746", alpha=0.55, linewidths=0)
    _set_equal_xz(ax1, [map_pts])
    ax1.set_xlabel("x [m]", fontsize=8)
    ax1.set_ylabel("z [m]", fontsize=8)
    ax1.set_title("Sparse feature map", fontsize=9, fontweight="bold")
    ax1.grid(True, alpha=0.2, lw=0.4)
    ax1.tick_params(labelsize=7)

    # ---- trajectory + graph (top-middle) ----
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor("white")
    if len(map_pts) > 0:
        ax2.scatter(map_pts[:, 0], map_pts[:, 2],
                    s=0.4, color="#aab7b8", alpha=0.2, linewidths=0, zorder=1)
    _draw_graph_edges(ax2, graph, kf_lookup)
    if len(traj) > 0:
        ax2.plot(traj[:, 1], traj[:, 3], color="#2980b9", lw=0.9, alpha=0.8, zorder=4)
    if len(kf_pos) > 0:
        ax2.scatter(kf_pos[:, 0], kf_pos[:, 2], s=12, color="#e67e22", zorder=5, linewidths=0)
    ax2.scatter(traj[0, 1], traj[0, 3], s=60, marker="o", color="#27ae60", zorder=7)
    ax2.scatter(traj[-1, 1], traj[-1, 3], s=60, marker="X", color="#c0392b", zorder=7)
    _set_equal_xz(ax2, [a for a in [map_pts, kf_pos] if len(a) > 0])
    ax2.set_xlabel("x [m]", fontsize=8)
    ax2.set_ylabel("z [m]", fontsize=8)
    ax2.set_title(f"Trajectory + KF graph  (loop edges: {n_loop})", fontsize=9, fontweight="bold")
    ax2.grid(True, alpha=0.2, lw=0.4)
    ax2.tick_params(labelsize=7)

    # ---- stats table (top-right) ----
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.axis("off")
    ax3.set_title("Run statistics", fontsize=9, fontweight="bold")
    rows_data = [
        ["Total frames",       f"{len(frame_log)}"],
        ["Tracking OK / LOST", f"{ok_count} / {lost_count}"],
        ["Final keyframes",    f"{max(int(r['keyframes']) for r in frame_log)}"],
        ["Final map points",   f"{max(int(r['points']) for r in frame_log):,}"],
        ["Loop closure edges", f"{n_loop}"],
        ["Global BA events",   f"{len(gba_ts)}"],
        ["Run duration",       f"{duration:.1f} s"],
        ["Room extent X × Z",  f"{x_ext:.2f} m × {z_ext:.2f} m"],
        ["Height variation Y", f"{y_ext:.3f} m"],
        ["Start → End dist",   f"{start_end:.3f} m"],
        ["Mean BA MSE",        f"{np.mean(ba_vals):.4f}" if ba_vals else "—"],
    ]
    tbl = ax3.table(cellText=rows_data,
                    colLabels=["Metric", "Value"],
                    loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.5)

    # ---- tracked pts (bottom-left) ----
    t0 = float(frame_log[0]["timestamp"])
    t_sec = [float(r["timestamp"]) - t0 for r in frame_log]
    tracked = [int(r["last_tracked"]) if r["last_tracked"] else 0 for r in frame_log]
    gba_sec = [ts - t0 for ts in gba_ts]

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(t_sec, tracked, lw=0.7, color="#2980b9", alpha=0.8)
    for ts in gba_sec:
        ax4.axvline(ts, color="#1abc9c", lw=1.3, ls="--", alpha=0.8, label="Global BA")
    if gba_sec:
        ax4.legend(fontsize=7)
    ax4.set_xlabel("Time [s]", fontsize=8)
    ax4.set_ylabel("Tracked map pts", fontsize=8)
    ax4.set_title("Tracking quality", fontsize=9, fontweight="bold")
    ax4.grid(True, alpha=0.25)
    ax4.tick_params(labelsize=7)

    # ---- map growth (bottom-middle) ----
    map_pt_counts = [int(r["points"]) for r in frame_log]
    kf_counts = [int(r["keyframes"]) for r in frame_log]
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t_sec, map_pt_counts, lw=0.8, color="#8e44ad", alpha=0.9, label="Map pts")
    ax5b = ax5.twinx()
    ax5b.plot(t_sec, kf_counts, lw=0.8, color="#e67e22", alpha=0.7, ls="--", label="KFs")
    ax5b.set_ylabel("Keyframes", fontsize=7, color="#e67e22")
    ax5b.tick_params(axis="y", labelcolor="#e67e22", labelsize=6)
    for ts in gba_sec:
        ax5.axvline(ts, color="#1abc9c", lw=1.3, ls="--", alpha=0.8)
    ax5.set_xlabel("Time [s]", fontsize=8)
    ax5.set_ylabel("Map points", fontsize=8, color="#8e44ad")
    ax5.tick_params(axis="y", labelcolor="#8e44ad", labelsize=7)
    ax5.set_title("Map growth", fontsize=9, fontweight="bold")
    ax5.grid(True, alpha=0.25)
    lines1, lbl1 = ax5.get_legend_handles_labels()
    lines2, lbl2 = ax5b.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, lbl1 + lbl2, fontsize=7)

    # ---- BA MSE (bottom-right) ----
    ba_mse = []
    for r in frame_log:
        try:
            ba_mse.append(float(r["last_ba_mse"]))
        except (ValueError, TypeError):
            ba_mse.append(float("nan"))
    p99 = np.percentile([v for v in ba_mse if np.isfinite(v)], 99) if ba_vals else 5.0
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(t_sec, ba_mse, lw=0.6, color="#e74c3c", alpha=0.8)
    ax6.set_ylim(0, p99 * 1.15)
    for ts in gba_sec:
        ax6.axvline(ts, color="#1abc9c", lw=1.3, ls="--", alpha=0.8)
    ax6.set_xlabel("Time [s]", fontsize=8)
    ax6.set_ylabel("Pose opt. BA MSE", fontsize=8)
    ax6.set_title("Bundle adjustment MSE", fontsize=9, fontweight="bold")
    ax6.grid(True, alpha=0.25)
    ax6.tick_params(labelsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all(run_dir: Path, dataset_path: Path | None, output_dir: Path) -> list[Path]:
    run_dir = run_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    traj_path = _find_file(run_dir, "trajectory_*.txt")
    log_path = _find_file(run_dir, "frame_log_*.csv")
    ply_path = run_dir / "map_points.ply"
    kf_json = run_dir / "keyframes.json"
    graph_json = run_dir / "keyframe_graph.json"

    if traj_path is None:
        raise FileNotFoundError(f"No trajectory_*.txt in {run_dir}")
    if log_path is None:
        raise FileNotFoundError(f"No frame_log_*.csv in {run_dir}")

    print(f"Run dir:   {run_dir}")
    print(f"Trajectory:{traj_path.name}")
    print(f"Map PLY:   {'found' if ply_path.exists() else 'NOT FOUND — run with map export enabled'}")
    print(f"KF JSON:   {'found' if kf_json.exists() else 'NOT FOUND'}")
    print(f"Dataset:   {dataset_path or '(not provided — semi-dense skipped)'}")
    print(f"Output:    {output_dir}")
    print()

    traj = read_trajectory(traj_path)
    frame_log = read_frame_log(log_path)
    map_pts = read_ply(ply_path)
    kf_positions, kf_lookup, kf_ts = read_keyframes(kf_json)
    graph = read_graph(graph_json)
    gba_ts = _find_gba_timestamps(frame_log)

    # Filter outlier triangulations using trajectory as scene reference
    ref = traj[:, 1:4] if len(traj) > 0 else kf_positions
    map_pts = filter_map_points_to_scene(map_pts, ref, padding_m=3.0)

    print(f"Poses:       {len(traj)}")
    print(f"Map points:  {len(map_pts):,}")
    print(f"Keyframes:   {len(kf_positions)}")
    print(f"Loop edges:  {len(graph.get('loop_edges', []))}")
    print(f"GBA events:  {len(gba_ts)}")
    print()

    generated = []

    print("Generating evaluation figures...")
    p = output_dir / "eval_sparse_map.png"
    plot_eval_sparse_map(run_dir, map_pts, traj, kf_positions, p)
    generated.append(p)

    p = output_dir / "eval_trajectory_graph.png"
    plot_eval_trajectory_graph(run_dir, map_pts, traj, kf_positions, kf_lookup, graph, p)
    generated.append(p)

    p = output_dir / "eval_tracking_quality.png"
    plot_eval_tracking_quality(run_dir, frame_log, gba_ts, p)
    generated.append(p)

    print("\nGenerating presentation figures...")
    p = output_dir / "pres_sparse_map.png"
    plot_pres_sparse_map(run_dir, map_pts, traj, kf_positions, graph, kf_lookup, p)
    generated.append(p)

    if dataset_path is not None and kf_json.exists():
        print("  Building semi-dense RGB-D cloud (this takes ~30s)...")
        sd_pts, sd_col = build_semidense_cloud(kf_json, dataset_path)
        if len(sd_pts) > 0:
            p = output_dir / "pres_semidense_topdown.png"
            plot_pres_semidense_topdown(run_dir, sd_pts, sd_col, traj, p)
            generated.append(p)

            p = output_dir / "pres_semidense_3d.png"
            plot_pres_semidense_3d(run_dir, sd_pts, sd_col, traj, p)
            generated.append(p)
        else:
            print("  (semi-dense cloud empty — skipping)")
    else:
        print("  (semi-dense skipped: --dataset not provided or keyframes.json missing)")

    p = output_dir / "pres_summary.png"
    plot_pres_summary(run_dir, map_pts, traj, kf_positions, kf_lookup,
                      graph, frame_log, gba_ts, p)
    generated.append(p)

    return generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path,
                        help="SLAM run output directory (map_points.ply + keyframes.json required)")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Dataset root (rgb/ depth/ camera.yaml) for semi-dense reconstruction")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for figures (default: <run>/map_figures/)")
    args = parser.parse_args(argv)
    output = args.output or (args.run / "map_figures")
    paths = generate_all(args.run, args.dataset, output)
    print(f"\nGenerated {len(paths)} figures in {output}")
    for p in paths:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
