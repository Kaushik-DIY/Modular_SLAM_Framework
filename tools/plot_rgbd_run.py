#!/usr/bin/env python3
"""
=============================================================================
tools/plot_rgbd_run.py

Trajectory and tracking-stats visualiser for a single RGB-D ORB-SLAM run.
No ground truth required.

Coordinate convention (ORB-SLAM2 / pySLAM world frame, Twc):
  X = right  (camera frame at initialisation)
  Y = down   (height — small range for a floor robot)
  Z = forward
  Top-down floor-plan view  →  X (horizontal) vs Z (vertical)

Outputs
-------
  trajectory_topdown.png     Bird's-eye (X–Z) path, KF positions, GBA markers
  trajectory_3d.png          Full 3-D trajectory
  tracking_stats.png         Tracked pts / map pts / BA MSE over time
  map_topdown.png            Sparse map points + KF graph top-down (X–Z)
  map_3d.png                 Sparse map points 3-D
  summary_panel.png          4-panel overview figure

Usage
-----
  python3 -m tools.plot_rgbd_run \\
      --run  visual_slam_outputs/lab_rgbd_run_2_full \\
      --output visual_slam_outputs/lab_rgbd_run_2_full/plots
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _find_file(run_dir: Path, pattern: str) -> Path | None:
    matches = sorted(run_dir.glob(pattern))
    return matches[0] if matches else None


def read_trajectory(path: Path) -> np.ndarray:
    """
    Read TUM-format trajectory file.
    Returns Nx4 array: [timestamp, tx, ty, tz]  (Twc camera centre in world).
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(rows, dtype=np.float64)


def read_frame_log(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_kf_timestamps(frame_log: list[dict]) -> list[float]:
    """Return timestamp of each frame where a new keyframe was first registered."""
    timestamps = []
    prev_kf = 0
    for row in frame_log:
        kf = int(row["keyframes"])
        if kf > prev_kf:
            timestamps.append(float(row["timestamp"]))
            prev_kf = kf
    return timestamps


def find_gba_timestamps(frame_log: list[dict]) -> list[float]:
    """Return timestamps of unique Global BA trigger events (0→1 transitions)."""
    timestamps = []
    for i in range(1, len(frame_log)):
        if (frame_log[i]["loop_global_ba_started"] == "1"
                and frame_log[i - 1]["loop_global_ba_started"] == "0"):
            timestamps.append(float(frame_log[i]["timestamp"]))
    return timestamps


def lookup_positions_by_timestamp(
    trajectory: np.ndarray,
    query_timestamps: list[float],
) -> np.ndarray:
    """
    For each query timestamp, find the nearest pose in the trajectory.
    Returns Nx3 array of [tx, ty, tz] world positions.
    """
    if len(query_timestamps) == 0 or len(trajectory) == 0:
        return np.empty((0, 3), dtype=np.float64)
    traj_ts = trajectory[:, 0]
    positions = []
    for ts in query_timestamps:
        idx = int(np.argmin(np.abs(traj_ts - ts)))
        positions.append(trajectory[idx, 1:4])
    return np.asarray(positions, dtype=np.float64)


# ---------------------------------------------------------------------------
# Map IO helpers
# ---------------------------------------------------------------------------

def read_ply_points(path: Path, max_points: int = 300_000) -> np.ndarray:
    """Read ASCII PLY file, return Nx3 float64 array. Subsamples if too large."""
    if not path.exists():
        return np.empty((0, 3), dtype=np.float64)
    points = []
    in_header = True
    with open(path) as f:
        for line in f:
            if in_header:
                if line.strip() == "end_header":
                    in_header = False
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points, dtype=np.int64)
        pts = pts[idx]
    return pts


def filter_map_points_to_scene(
    pts: np.ndarray,
    ref_pts: np.ndarray,
    padding_m: float = 3.0,
) -> np.ndarray:
    """Remove outlier map points that lie far outside the trajectory bounding box.

    Triangulation failures (near-parallel rays, textureless surfaces) produce
    map points at hundreds or thousands of metres. These destroy axis scaling
    when plotting.  Clipping to the trajectory bbox + padding keeps all
    legitimate room-scale points while discarding numerical outliers.
    """
    if len(pts) == 0 or len(ref_pts) == 0:
        return pts
    lo = ref_pts.min(axis=0) - padding_m
    hi = ref_pts.max(axis=0) + padding_m
    mask = np.all((pts >= lo) & (pts <= hi), axis=1)
    n_removed = len(pts) - mask.sum()
    if n_removed > 0:
        print(f"  [map filter] removed {n_removed} outlier map points outside "
              f"scene bbox (kept {mask.sum()}/{len(pts)})")
    return pts[mask]


def read_keyframes_json(path: Path) -> np.ndarray:
    """Read keyframes.json, return Nx3 world positions (from Twc[:3,3])."""
    if not path.exists():
        return np.empty((0, 3), dtype=np.float64)
    import json
    data = json.loads(path.read_text())
    positions = []
    for kf in data:
        try:
            positions.append(list(kf["position"]))
        except (KeyError, TypeError):
            continue
    return np.asarray(positions, dtype=np.float64)


def read_graph_edges(path: Path) -> dict:
    """Read keyframe_graph.json, return dict of edge lists keyed by type."""
    if not path.exists():
        return {}
    import json
    return json.loads(path.read_text())


def _kf_position_lookup(keyframes_json_path: Path) -> dict[int, np.ndarray]:
    """Return {kid: position_xyz} dict from keyframes.json."""
    if not keyframes_json_path.exists():
        return {}
    import json
    data = json.loads(keyframes_json_path.read_text())
    out = {}
    for kf in data:
        try:
            out[int(kf["kid"])] = np.asarray(kf["position"], dtype=np.float64)
        except (KeyError, TypeError):
            continue
    return out


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _set_equal_axes_xz(ax, arrays: list[np.ndarray], pad_frac: float = 0.08) -> None:
    """Equal aspect, square view for X–Z plots."""
    pts = np.vstack([a[:, [0, 2]] for a in arrays if len(a) > 0])
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    center = (mins + maxs) * 0.5
    span = max(float(np.max(maxs - mins)), 1e-3)
    pad = span * pad_frac
    ax.set_xlim(center[0] - span / 2 - pad, center[0] + span / 2 + pad)
    ax.set_ylim(center[1] - span / 2 - pad, center[1] + span / 2 + pad)
    ax.set_aspect("equal", adjustable="box")


def _colorline_xz(ax, traj: np.ndarray, lw: float = 1.4) -> None:
    """Draw X–Z trajectory coloured from blue (start) to red (end) by time."""
    x, z = traj[:, 1], traj[:, 3]
    n = len(x)
    if n < 2:
        return
    colors = cm.plasma(np.linspace(0.1, 0.9, n - 1))
    for i in range(n - 1):
        ax.plot(x[i:i+2], z[i:i+2], color=colors[i], lw=lw, solid_capstyle="round")


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------

def plot_topdown(
    run_dir: Path,
    trajectory: np.ndarray,
    kf_positions: np.ndarray,
    gba_positions: np.ndarray,
    output_path: Path,
) -> None:
    """
    Bird's-eye (X–Z) floor-plan view.

    Colour gradient (blue→yellow→red) encodes temporal progress.
    X = right,  Z = forward  (ORB-SLAM2 world frame).
    """
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    _colorline_xz(ax, trajectory, lw=1.4)

    # Colourbar proxy
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Time (start → end)", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["start", "mid", "end"])

    # Keyframe positions
    if len(kf_positions) > 0:
        ax.scatter(
            kf_positions[:, 0], kf_positions[:, 2],
            s=12, color="#e67e22", zorder=4, label=f"Keyframes ({len(kf_positions)})",
            linewidths=0, alpha=0.75,
        )

    # Global BA trigger positions
    if len(gba_positions) > 0:
        ax.scatter(
            gba_positions[:, 0], gba_positions[:, 2],
            s=90, marker="*", color="#1abc9c", zorder=5,
            label=f"Global BA ({len(gba_positions)})", linewidths=0.5,
            edgecolors="#0e6655",
        )

    # Start and end markers
    ax.scatter(
        trajectory[0, 1], trajectory[0, 3],
        s=80, marker="o", color="#27ae60", zorder=6, label="Start",
    )
    ax.scatter(
        trajectory[-1, 1], trajectory[-1, 3],
        s=80, marker="X", color="#c0392b", zorder=6, label="End",
    )

    _set_equal_axes_xz(ax, [trajectory[:, [1, 2, 3]]])  # uses X and Z

    ax.set_xlabel("X  [m]  (right)", fontsize=10)
    ax.set_ylabel("Z  [m]  (forward)", fontsize=10)
    ax.set_title(
        f"Trajectory — top-down view  ({run_dir.name})\n"
        f"ORB-SLAM2 world frame: X=right, Z=forward",
        fontsize=9,
    )
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {output_path.name}")


def plot_3d(
    run_dir: Path,
    trajectory: np.ndarray,
    kf_positions: np.ndarray,
    gba_positions: np.ndarray,
    output_path: Path,
) -> None:
    """Full 3-D trajectory plot."""
    fig = plt.figure(figsize=(9, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = trajectory[:, 1], trajectory[:, 2], trajectory[:, 3]

    # Colour-coded path
    n = len(x)
    colors = cm.plasma(np.linspace(0.1, 0.9, max(n - 1, 1)))
    for i in range(n - 1):
        ax.plot(x[i:i+2], z[i:i+2], y[i:i+2],   # swap Y and Z for natural 3-D view
                color=colors[i], lw=1.0, alpha=0.85)

    if len(kf_positions) > 0:
        ax.scatter(
            kf_positions[:, 0], kf_positions[:, 2], kf_positions[:, 1],
            s=10, color="#e67e22", alpha=0.7, label=f"KFs ({len(kf_positions)})",
        )

    if len(gba_positions) > 0:
        ax.scatter(
            gba_positions[:, 0], gba_positions[:, 2], gba_positions[:, 1],
            s=80, marker="*", color="#1abc9c", zorder=5,
            label=f"Global BA ({len(gba_positions)})",
        )

    ax.scatter(x[0], z[0], y[0], s=80, marker="o", color="#27ae60", label="Start", zorder=6)
    ax.scatter(x[-1], z[-1], y[-1], s=80, marker="X", color="#c0392b", label="End", zorder=6)

    ax.set_xlabel("X [m]  (right)", fontsize=8, labelpad=4)
    ax.set_ylabel("Z [m]  (forward)", fontsize=8, labelpad=4)
    ax.set_zlabel("Y [m]  (up, negated)", fontsize=8, labelpad=4)
    ax.set_title(
        f"Trajectory 3-D  ({run_dir.name})\n"
        f"colour: time (blue→red),  Y axis negated for natural up-view",
        fontsize=8,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {output_path.name}")


def plot_tracking_stats(
    run_dir: Path,
    frame_log: list[dict],
    gba_timestamps: list[float],
    output_path: Path,
) -> None:
    """3-panel tracking quality figure: tracked pts, map pts, BA MSE."""
    frame_ids = [int(r["i"]) for r in frame_log]
    tracked = [int(r["last_tracked"]) if r["last_tracked"] else 0 for r in frame_log]
    map_pts = [int(r["points"]) for r in frame_log]
    ba_mse_raw = []
    for r in frame_log:
        try:
            ba_mse_raw.append(float(r["last_ba_mse"]))
        except (ValueError, TypeError):
            ba_mse_raw.append(float("nan"))

    timestamps = [float(r["timestamp"]) for r in frame_log]
    t0 = timestamps[0]
    t_sec = [ts - t0 for ts in timestamps]

    # Convert GBA timestamps to elapsed seconds
    gba_sec = [ts - t0 for ts in gba_timestamps]

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), dpi=150, sharex=True)

    # Panel 1: tracked map points
    axes[0].plot(t_sec, tracked, lw=0.8, color="#2980b9", alpha=0.85)
    axes[0].set_ylabel("Tracked map pts", fontsize=9)
    axes[0].set_title(
        f"Tracking quality over time  ({run_dir.name})", fontsize=9
    )
    axes[0].grid(True, alpha=0.25)
    for ts in gba_sec:
        axes[0].axvline(ts, color="#1abc9c", lw=1.4, ls="--", alpha=0.8, label="Global BA")
    if gba_sec:
        axes[0].legend(fontsize=8)

    # Panel 2: total map points
    axes[1].plot(t_sec, map_pts, lw=0.9, color="#8e44ad", alpha=0.85)
    axes[1].set_ylabel("Map points (total)", fontsize=9)
    axes[1].grid(True, alpha=0.25)
    for ts in gba_sec:
        axes[1].axvline(ts, color="#1abc9c", lw=1.4, ls="--", alpha=0.8)

    # Panel 3: BA MSE
    valid_mse = [v for v in ba_mse_raw if np.isfinite(v)]
    p99 = np.percentile(valid_mse, 99) if valid_mse else 10.0
    axes[2].plot(t_sec, ba_mse_raw, lw=0.7, color="#e74c3c", alpha=0.8)
    axes[2].set_ylim(0, p99 * 1.1)
    axes[2].set_ylabel("Pose opt. BA MSE", fontsize=9)
    axes[2].set_xlabel("Elapsed time [s]", fontsize=9)
    axes[2].grid(True, alpha=0.25)
    for ts in gba_sec:
        axes[2].axvline(ts, color="#1abc9c", lw=1.4, ls="--", alpha=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {output_path.name}")


def plot_map_topdown(
    run_dir: Path,
    map_points: np.ndarray,
    kf_positions: np.ndarray,
    trajectory: np.ndarray,
    graph: dict,
    kf_lookup: dict,
    output_path: Path,
) -> None:
    """
    Top-down (X–Z) sparse map view.
    Map points as dense scatter, keyframe graph overlaid, trajectory in blue.
    """
    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)

    # Map points — colour by height (Y value) to give depth cue
    if len(map_points) > 0:
        y_vals = map_points[:, 1]
        y_norm = (y_vals - y_vals.min()) / max(y_vals.ptp(), 1e-6)
        ax.scatter(
            map_points[:, 0], map_points[:, 2],
            c=y_norm, cmap="viridis", s=0.8, alpha=0.5, linewidths=0,
            label=f"Map points ({len(map_points)})",
        )
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Height (Y, low→high)", fontsize=8)

    # Covisibility edges (light, thin)
    for edge in graph.get("covisibility_edges", []):
        a = kf_lookup.get(int(edge.get("source", -1)))
        b = kf_lookup.get(int(edge.get("target", -1)))
        if a is not None and b is not None:
            ax.plot([a[0], b[0]], [a[2], b[2]], color="#95a5a6", lw=0.35, alpha=0.4)

    # Spanning tree edges
    for edge in graph.get("spanning_tree_edges", []):
        a = kf_lookup.get(int(edge.get("source", -1)))
        b = kf_lookup.get(int(edge.get("target", -1)))
        if a is not None and b is not None:
            ax.plot([a[0], b[0]], [a[2], b[2]], color="#7f8c8d", lw=0.7, alpha=0.6)

    # Loop closure edges (bold red)
    loop_plotted = False
    for edge in graph.get("loop_edges", []):
        a = kf_lookup.get(int(edge.get("source", -1)))
        b = kf_lookup.get(int(edge.get("target", -1)))
        if a is not None and b is not None:
            ax.plot([a[0], b[0]], [a[2], b[2]], color="#e74c3c", lw=1.6, alpha=0.8,
                    label="Loop edges" if not loop_plotted else "")
            loop_plotted = True

    # Keyframe positions
    if len(kf_positions) > 0:
        ax.scatter(
            kf_positions[:, 0], kf_positions[:, 2],
            s=16, color="#e67e22", zorder=4, linewidths=0,
            label=f"Keyframes ({len(kf_positions)})",
        )

    # Trajectory overlay (thin blue)
    if len(trajectory) > 0:
        ax.plot(trajectory[:, 1], trajectory[:, 3], color="#2980b9",
                lw=0.6, alpha=0.5, label="Trajectory", zorder=3)

    # Start / end
    ax.scatter(trajectory[0, 1], trajectory[0, 3], s=80, marker="o",
               color="#27ae60", zorder=6, label="Start")
    ax.scatter(trajectory[-1, 1], trajectory[-1, 3], s=80, marker="X",
               color="#c0392b", zorder=6, label="End")

    all_pts = [p for p in [map_points, kf_positions] if len(p) > 0]
    _set_equal_axes_xz(ax, all_pts if all_pts else [trajectory[:, [1, 2, 3]]])

    loop_count = len(graph.get("loop_edges", []))
    ax.set_xlabel("X  [m]  (right)", fontsize=10)
    ax.set_ylabel("Z  [m]  (forward)", fontsize=10)
    ax.set_title(
        f"Sparse map — top-down (X–Z)  |  {run_dir.name}\n"
        f"Map pts: {len(map_points)}   KFs: {len(kf_positions)}   "
        f"Loop edges: {loop_count}",
        fontsize=9,
    )
    ax.grid(True, alpha=0.2, lw=0.4)
    ax.legend(fontsize=8, loc="best", markerscale=1.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {output_path.name}")


def plot_map_3d(
    run_dir: Path,
    map_points: np.ndarray,
    kf_positions: np.ndarray,
    trajectory: np.ndarray,
    output_path: Path,
) -> None:
    """3-D sparse map with trajectory and keyframes."""
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    if len(map_points) > 0:
        # Subsample for 3D rendering speed
        step = max(1, len(map_points) // 50_000)
        mp = map_points[::step]
        y_norm = (mp[:, 1] - mp[:, 1].min()) / max(mp[:, 1].ptp(), 1e-6)
        ax.scatter(mp[:, 0], mp[:, 2], -mp[:, 1],   # negate Y so up = positive
                   c=y_norm, cmap="viridis", s=0.6, alpha=0.35, linewidths=0)

    if len(trajectory) > 0:
        ax.plot(trajectory[:, 1], trajectory[:, 3], -trajectory[:, 2],
                color="#2980b9", lw=0.8, alpha=0.7, label="Trajectory")

    if len(kf_positions) > 0:
        ax.scatter(kf_positions[:, 0], kf_positions[:, 2], -kf_positions[:, 1],
                   s=12, color="#e67e22", alpha=0.8, label=f"KFs ({len(kf_positions)})")

    ax.scatter(trajectory[0, 1], trajectory[0, 3], -trajectory[0, 2],
               s=80, marker="o", color="#27ae60", label="Start", zorder=6)
    ax.scatter(trajectory[-1, 1], trajectory[-1, 3], -trajectory[-1, 2],
               s=80, marker="X", color="#c0392b", label="End", zorder=6)

    ax.set_xlabel("X [m]  (right)", fontsize=8)
    ax.set_ylabel("Z [m]  (forward)", fontsize=8)
    ax.set_zlabel("up [m]  (−Y)", fontsize=8)
    ax.set_title(
        f"Sparse map — 3D  |  {run_dir.name}\n"
        f"colour = height,  Y negated so up is positive",
        fontsize=8,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {output_path.name}")


def plot_summary_panel(
    run_dir: Path,
    trajectory: np.ndarray,
    kf_positions: np.ndarray,
    gba_positions: np.ndarray,
    frame_log: list[dict],
    gba_timestamps: list[float],
    output_path: Path,
) -> None:
    """4-panel summary: top-down path, tracking, map pts, stats table."""
    fig = plt.figure(figsize=(14, 10), dpi=150)
    fig.suptitle(
        f"ORB-SLAM2 RGB-D run: {run_dir.name}\n"
        f"Frames: {len(frame_log)}   KFs: {len(kf_positions)}   "
        f"Map pts: {max(int(r['points']) for r in frame_log)}   "
        f"Global BA events: {len(gba_timestamps)}",
        fontsize=10,
    )

    # ---- top-left: top-down trajectory ----
    ax1 = fig.add_subplot(2, 2, 1)
    _colorline_xz(ax1, trajectory, lw=1.2)
    if len(kf_positions) > 0:
        ax1.scatter(kf_positions[:, 0], kf_positions[:, 2], s=8,
                    color="#e67e22", zorder=4, alpha=0.7)
    if len(gba_positions) > 0:
        ax1.scatter(gba_positions[:, 0], gba_positions[:, 2], s=70,
                    marker="*", color="#1abc9c", zorder=5)
    ax1.scatter(trajectory[0, 1], trajectory[0, 3], s=60, marker="o",
                color="#27ae60", zorder=6, label="Start")
    ax1.scatter(trajectory[-1, 1], trajectory[-1, 3], s=60, marker="X",
                color="#c0392b", zorder=6, label="End")
    _set_equal_axes_xz(ax1, [trajectory[:, [1, 2, 3]]])
    ax1.set_xlabel("X [m]  (right)", fontsize=8)
    ax1.set_ylabel("Z [m]  (forward)", fontsize=8)
    ax1.set_title("Top-down (X–Z)", fontsize=9)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=7)

    # ---- top-right: tracked map points ----
    t0 = float(frame_log[0]["timestamp"])
    t_sec = [float(r["timestamp"]) - t0 for r in frame_log]
    tracked = [int(r["last_tracked"]) if r["last_tracked"] else 0 for r in frame_log]
    gba_sec = [ts - t0 for ts in gba_timestamps]

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t_sec, tracked, lw=0.7, color="#2980b9", alpha=0.85)
    for ts in gba_sec:
        ax2.axvline(ts, color="#1abc9c", lw=1.3, ls="--", alpha=0.85, label="GBA")
    if gba_sec:
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles[:1], labels[:1], fontsize=7)
    ax2.set_xlabel("Time [s]", fontsize=8)
    ax2.set_ylabel("Tracked map pts", fontsize=8)
    ax2.set_title("Tracking", fontsize=9)
    ax2.grid(True, alpha=0.25)

    # ---- bottom-left: map point growth ----
    map_pts = [int(r["points"]) for r in frame_log]
    kf_counts = [int(r["keyframes"]) for r in frame_log]

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t_sec, map_pts, lw=0.8, color="#8e44ad", alpha=0.9, label="Map points")
    ax3b = ax3.twinx()
    ax3b.plot(t_sec, kf_counts, lw=0.8, color="#e67e22", alpha=0.7, ls="--", label="Keyframes")
    ax3b.set_ylabel("Keyframes", fontsize=8, color="#e67e22")
    ax3b.tick_params(axis="y", labelcolor="#e67e22", labelsize=7)
    for ts in gba_sec:
        ax3.axvline(ts, color="#1abc9c", lw=1.3, ls="--", alpha=0.85)
    ax3.set_xlabel("Time [s]", fontsize=8)
    ax3.set_ylabel("Map points", fontsize=8, color="#8e44ad")
    ax3.tick_params(axis="y", labelcolor="#8e44ad", labelsize=7)
    ax3.set_title("Map growth", fontsize=9)
    ax3.grid(True, alpha=0.25)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    # ---- bottom-right: stats table ----
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")

    x_range = trajectory[:, 1].max() - trajectory[:, 1].min()
    y_range = trajectory[:, 2].max() - trajectory[:, 2].min()
    z_range = trajectory[:, 3].max() - trajectory[:, 3].min()
    start_to_end = float(np.linalg.norm(trajectory[-1, 1:4] - trajectory[0, 1:4]))
    ok_count = sum(1 for r in frame_log if r["state"] == "OK")
    lost_count = sum(1 for r in frame_log if r["state"] == "LOST")
    ba_mse_vals = [float(r["last_ba_mse"]) for r in frame_log
                   if r["last_ba_mse"] and r["last_ba_mse"] not in ("", "None")]
    ba_mse_mean = float(np.mean(ba_mse_vals)) if ba_mse_vals else float("nan")

    rows_data = [
        ["Total frames", str(len(frame_log))],
        ["Tracking OK", str(ok_count)],
        ["Tracking LOST", str(lost_count)],
        ["Final keyframes", str(max(kf_counts))],
        ["Final map points", str(max(map_pts))],
        ["Global BA events", str(len(gba_timestamps))],
        ["X extent [m]", f"{x_range:.3f}"],
        ["Z extent [m]", f"{z_range:.3f}"],
        ["Y extent [m]", f"{y_range:.3f}"],
        ["Start→End dist [m]", f"{start_to_end:.3f}"],
        ["Mean BA MSE", f"{ba_mse_mean:.4f}"],
    ]

    table = ax4.table(
        cellText=rows_data,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.15, 1.4)
    ax4.set_title("Run statistics", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_plots(run_dir: Path, output_dir: Path) -> list[Path]:
    run_dir = run_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    traj_path = _find_file(run_dir, "trajectory_*.txt")
    log_path = _find_file(run_dir, "frame_log_*.csv")
    ply_path = run_dir / "map_points.ply"
    kf_json_path = run_dir / "keyframes.json"
    graph_json_path = run_dir / "keyframe_graph.json"

    if traj_path is None:
        raise FileNotFoundError(f"No trajectory_*.txt found in {run_dir}")
    if log_path is None:
        raise FileNotFoundError(f"No frame_log_*.csv found in {run_dir}")

    has_map = ply_path.exists() and kf_json_path.exists()

    print(f"Run:        {run_dir}")
    print(f"Trajectory: {traj_path.name}")
    print(f"Frame log:  {log_path.name}")
    print(f"Map PLY:    {'found' if ply_path.exists() else 'not found — skipping map plots'}")
    print(f"Output:     {output_dir}")
    print()

    trajectory = read_trajectory(traj_path)
    frame_log = read_frame_log(log_path)

    kf_timestamps = find_kf_timestamps(frame_log)
    gba_timestamps = find_gba_timestamps(frame_log)
    kf_positions = lookup_positions_by_timestamp(trajectory, kf_timestamps)
    gba_positions = lookup_positions_by_timestamp(trajectory, gba_timestamps)

    print(f"Poses:           {len(trajectory)}")
    print(f"Keyframes:       {len(kf_timestamps)}")
    print(f"GBA events:      {len(gba_timestamps)}")
    print(f"X range: {trajectory[:,1].min():.3f} → {trajectory[:,1].max():.3f} m")
    print(f"Y range: {trajectory[:,2].min():.3f} → {trajectory[:,2].max():.3f} m")
    print(f"Z range: {trajectory[:,3].min():.3f} → {trajectory[:,3].max():.3f} m")
    print()

    generated = []

    p = output_dir / "trajectory_topdown.png"
    plot_topdown(run_dir, trajectory, kf_positions, gba_positions, p)
    generated.append(p)

    p = output_dir / "trajectory_3d.png"
    plot_3d(run_dir, trajectory, kf_positions, gba_positions, p)
    generated.append(p)

    p = output_dir / "tracking_stats.png"
    plot_tracking_stats(run_dir, frame_log, gba_timestamps, p)
    generated.append(p)

    if has_map:
        map_points = read_ply_points(ply_path)
        kf_pos_from_json = read_keyframes_json(kf_json_path)
        # Filter outlier triangulations before any axis scaling
        ref = trajectory[:, 1:4] if len(trajectory) > 0 else kf_pos_from_json
        map_points = filter_map_points_to_scene(map_points, ref, padding_m=3.0)
        graph = read_graph_edges(graph_json_path)
        kf_lookup = _kf_position_lookup(kf_json_path)

        p = output_dir / "map_topdown.png"
        plot_map_topdown(run_dir, map_points, kf_pos_from_json, trajectory, graph, kf_lookup, p)
        generated.append(p)

        p = output_dir / "map_3d.png"
        plot_map_3d(run_dir, map_points, kf_pos_from_json, trajectory, p)
        generated.append(p)
    else:
        print("  (map_topdown.png and map_3d.png skipped — map_points.ply not found)")

    p = output_dir / "summary_panel.png"
    plot_summary_panel(
        run_dir, trajectory, kf_positions, gba_positions,
        frame_log, gba_timestamps, p,
    )
    generated.append(p)

    return generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", required=True, type=Path,
        help="Path to the SLAM run output directory (contains trajectory_*.txt and frame_log_*.csv)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Where to write plot PNGs (default: <run>/plots/)",
    )
    args = parser.parse_args(argv)
    output_dir = args.output if args.output else args.run / "plots"
    paths = generate_plots(args.run, output_dir)
    print(f"\nGenerated {len(paths)} figures in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
