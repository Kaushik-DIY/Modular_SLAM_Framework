"""Output writers shared by the batch and real-time runners: TUM trajectory,
fused log-odds occupancy grid, scan-overlay debug image, and run summary.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

import fusion_core as fc

from slam_core.fusion2.config import FusionV2Config
from slam_core.fusion2.backend import SharedMap


def _anchor_poses(poses: np.ndarray) -> np.ndarray:
    """Express every node pose in the robot-START frame so EVERY mode renders
    in the same standard convention: the first keyframe sits at the origin
    facing +x. Without this the VO front-end starts at heading 90° (the
    BASE_T_CAM ∘ CAMERA_GROUND_TRANSFORM projection of identity), rotating the
    orb maps 90° vs the LiDAR maps. A pure rigid re-frame — map/trajectory
    geometry is unchanged, only the global orientation is normalized.
    `poses` columns: [nid, x, y, theta], assumed nid-sorted (gauge = row 0)."""
    if len(poses) == 0:
        return poses
    ax, ay, ath = float(poses[0, 1]), float(poses[0, 2]), float(poses[0, 3])
    c, s = math.cos(ath), math.sin(ath)
    out = poses.copy()
    dx = poses[:, 1] - ax
    dy = poses[:, 2] - ay
    out[:, 1] = c * dx + s * dy          # R(-ath) @ (p - anchor)
    out[:, 2] = -s * dx + c * dy
    out[:, 3] = np.arctan2(np.sin(poses[:, 3] - ath), np.cos(poses[:, 3] - ath))
    return out


def render_fused_occupancy(render_sigs, render_poses, traj_xyt, grid_cfg, out_png,
                           title, npy_path=None, meta_path=None):
    """Fuse the given signatures' scans (at their optimized poses) into ONE
    log-odds occupancy grid (C++ assemble_local_grid) and render it grayscale
    with the trajectory overlay. Shared by the batch runner (write_outputs) and
    the real-time runner (V5) so the map convention is identical. `traj_xyt` is
    an (N,3) x/y/theta array for the blue path + start/end markers. Returns
    (prob, extent) or (None, None) if there is nothing with scans to render."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not render_sigs:
        return None, None
    # Integrate local scans at optimized poses into one log-odds grid.
    grid = fc.assemble_local_grid(render_sigs, render_poses, grid_cfg)
    prob = np.asarray(grid.probability())
    extent = [grid.origin_x, grid.origin_x + grid.width * grid.resolution,
              grid.origin_y, grid.origin_y + grid.height * grid.resolution]
    if npy_path is not None:
        np.save(npy_path, prob)
    if meta_path is not None:
        # Store georeferencing metadata needed to re-render the saved grid.
        with open(meta_path, "w") as f:
            json.dump(dict(origin_x=grid.origin_x, origin_y=grid.origin_y,
                           resolution=grid.resolution, width=grid.width,
                           height=grid.height, extent=extent), f, indent=2)
    traj = np.asarray(traj_xyt, dtype=float)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(prob, cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower",
              extent=extent, interpolation="nearest")
    if len(traj):
        ax.plot(traj[:, 0], traj[:, 1], "-", lw=1.0, color="tab:blue", alpha=0.9)
        ax.scatter(traj[0, 0], traj[0, 1], c="g", s=50, zorder=5, label="start")
        ax.scatter(traj[-1, 0], traj[-1, 1], c="r", s=50, zorder=5, label="end")
    ax.set_title(title)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.grid(alpha=0.15); ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return prob, extent


def write_outputs(shared: SharedMap, cfg: FusionV2Config, run_dir: Path,
                  kf_stamps: dict, stats: dict,
                  skip_scan_ids: Optional[set] = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    # Normalize the global gauge before writing trajectory and map products.
    poses = _anchor_poses(np.asarray(shared.graph.poses()))

    with open(run_dir / "trajectory.tum", "w") as f:
        for nid, x, y, th in poses:
            # TUM format expects xyz plus quaternion; z/roll/pitch are planar zeros.
            t = kf_stamps.get(int(nid), float(nid))
            qz, qw = math.sin(th / 2.0), math.cos(th / 2.0)
            f.write(f"{t:.6f} {x:.6f} {y:.6f} 0.0 0.0 0.0 {qz:.9f} {qw:.9f}\n")

    # Collect renderable signatures (scans at optimized poses).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    render_sigs, render_poses, pts_all = [], [], []
    for nid, x, y, th in poses:
        if skip_scan_ids and int(nid) in skip_scan_ids:
            continue   # blind-pose keyframes (REINIT) must not paint the map
        sig = shared.memory.get(int(nid))
        if sig is None or not sig.has_scan:
            continue
        render_sigs.append(sig)
        render_poses.append(fc.Pose2(float(x), float(y), float(th)))
        sc = np.asarray(sig.scan_xy, dtype=np.float64)
        c, s = math.cos(th), math.sin(th)
        # Keep a raw point overlay for debugging map sharpness against the grid.
        pts_all.append(sc @ np.array([[c, -s], [s, c]]).T + [x, y])

    title = (f"fusion2 --mode {cfg.mode}: {len(poses)} keyframes, "
             f"{stats.get('loops_accepted', 0)} loops "
             f"(STM {shared.memory.stm_count()} / WM {shared.memory.wm_count()} / "
             f"LTM {shared.memory.ltm_count()})")

    # Primary output: FUSED log-odds occupancy grid (V4.2, thesis-grade).
    # Reuses the same C++ integration the B&B verifier trusts; overlapping
    # observations reinforce walls instead of smearing as a scatter band.
    render_fused_occupancy(
        render_sigs, render_poses, poses[:, 1:4] if len(poses) else np.zeros((0, 3)),
        shared.grid_cfg, run_dir / "occupancy.png", title,
        npy_path=run_dir / "map.npy", meta_path=run_dir / "map_meta.json")

    # Secondary debug output: raw scan scatter (the pre-V4.2 rendering).
    fig, ax = plt.subplots(figsize=(12, 7))
    if pts_all:
        P = np.vstack(pts_all)
        ax.scatter(P[:, 0], P[:, 1], s=0.2, c="k", alpha=0.25, linewidths=0)
    ax.plot(poses[:, 1], poses[:, 2], "-", lw=0.8, color="tab:blue", alpha=0.9)
    ax.scatter(poses[0, 1], poses[0, 2], c="g", s=50, zorder=5, label="start")
    ax.scatter(poses[-1, 1], poses[-1, 2], c="r", s=50, zorder=5, label="end")
    ax.set_title(title)
    ax.axis("equal"); ax.grid(alpha=0.2); ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "scan_overlay.png", dpi=110)
    plt.close(fig)

    with open(run_dir / "run_summary.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)


def _rss_gb() -> float:
    """Resident memory in GiB from procfs; returns -1 outside Linux procfs."""
    import os
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024.0 / 1024.0
    return -1.0
