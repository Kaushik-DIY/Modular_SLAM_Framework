#!/usr/bin/env python3
"""
render_map_build_video.py
=========================
Split-screen map-building video for Hector SLAM on the lab dataset.

  Left  : Trajectory lines growing over time (bird's-eye world view)
  Right : Occupancy map being built scan-by-scan + current laser scan overlay

Uses existing trajectory + scan files — no need to re-run SLAM.

Usage
-----
    # Auto-discover latest scan_to_map trajectory for lab_run_2:
    .venv/bin/python -m hector.eval.render_map_build_video

    # Explicit paths:
    .venv/bin/python -m hector.eval.render_map_build_video \\
        --traj  hector_outputs/trajectory_lab_run_2_raw_scan_to_map_1052.txt \\
        --out   hector_outputs/video/hector_map_build_lab.mp4 \\
        --fps   30 --frame-step 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from slam_core.matching.scan_to_map import GridMap, _transform_points
from carto.local_slam.range_to_points import ranges_to_points


# ─────────────────────────────────────────────────────────────────────────────
# Visual style constants
# ─────────────────────────────────────────────────────────────────────────────

BG_DARK = (12, 12, 16)              # panel background (near-black)
MAP_BG = (0, 0, 0)                  # occupancy map background (true black, unknown space)
GRID_COLOR = (34, 34, 44)
DIVIDER_COLOR = (70, 70, 84)
HEADER_RULE = (70, 80, 96)

TRAJ_CMAP_START = np.array([90, 230, 110], dtype=np.float32)   # green (earliest)
TRAJ_CMAP_MID   = np.array([70, 190, 255], dtype=np.float32)   # sky-blue (mid)
TRAJ_CMAP_END   = np.array([60, 110, 255], dtype=np.float32)   # vivid blue (current)

ROBOT_COLOR = (60, 150, 255)        # bright orange (BGR): current robot dot
START_COLOR = (90, 230, 110)        # green: start marker
SCAN_PT_COLOR = (70, 180, 255)      # warm orange: current scan endpoints
RAY_COLOR = (70, 70, 90)            # faint laser rays

TITLE_COLOR = (235, 235, 245)
SUBTITLE_COLOR = (165, 200, 225)
LABEL_COLOR = (205, 215, 235)


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_trajectory(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (stamps (N,), poses (N,3)) from 'ts x y theta score' file."""
    stamps, poses = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                t, x, y, th = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                stamps.append(t)
                poses.append([x, y, th])
            except ValueError:
                continue
    return np.array(stamps, dtype=float), np.array(poses, dtype=float)


def _load_scan_points(
    stamps: np.ndarray,
    scan_variant: str = "raw",
    voxel_filter: bool = True,
) -> tuple[object, list[np.ndarray]]:
    """
    Load lab_run_2 scans in order (1:1 with trajectory rows).
    Returns (profile, pts_list) where pts_list[k] is (M,2) sensor-frame points.
    """
    from slam_core.dataio.dataset_catalog import load_dataset_scans
    from slam_core.matching.preprocessing import PointCloudProcessor, PointCloudProcessorConfig

    profile, all_scans = load_dataset_scans("lab_run_2", scan_variant=scan_variant)
    scan_stamps = np.array([s["t"] for s in all_scans], dtype=float)
    N = len(stamps)

    proc = PointCloudProcessor(PointCloudProcessorConfig(
        fixed_voxel_size=0.03,
        adaptive_voxel_max_size=0.10,
        adaptive_min_num_points=200,
        adaptive_num_iterations=6,
        enabled=voxel_filter,
    ))

    pts_list: list[np.ndarray] = []
    for t in stamps[:N]:
        idx = int(np.argmin(np.abs(scan_stamps - t)))
        s = all_scans[idx]
        raw = ranges_to_points(
            s["ranges"], profile.angle_min, profile.angle_inc,
            profile.range_min, profile.range_max,
        )
        pts, _ = proc.process(raw)
        pts_list.append(pts)

    return profile, pts_list


def _auto_traj(out_dir: str, variant: str) -> Optional[str]:
    p = Path(out_dir)
    candidates = sorted(
        p.glob(f"trajectory_lab_run_2_{variant}_scan_to_map_*.txt"),
        key=lambda f: f.stat().st_mtime, reverse=True,
    )
    for c in candidates:
        if "_debug" not in c.name and "_pgo" not in c.name:
            return str(c)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _traj_color(frac: float) -> tuple[int, int, int]:
    """Return BGR for a fractional position in the trajectory color gradient."""
    if frac < 0.5:
        t = frac * 2.0
        c = (1 - t) * TRAJ_CMAP_START + t * TRAJ_CMAP_MID
    else:
        t = (frac - 0.5) * 2.0
        c = (1 - t) * TRAJ_CMAP_MID + t * TRAJ_CMAP_END
    return (int(c[0]), int(c[1]), int(c[2]))


def _put_text(
    img: np.ndarray,
    text: str,
    x: int, y: int,
    color: tuple[int, int, int] = (220, 220, 230),
    scale: float = 0.65,
    thickness: int = 1,
) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


class _Projector:
    """Equal-scale world->pixel mapping (preserves aspect, centers content)."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray,
                 x0: int, y0: int, w: int, h: int):
        span = np.maximum(hi - lo, 1e-6)
        self.scale = min(w / span[0], h / span[1])
        # Center the content within the [x0, x0+w] x [y0, y0+h] box.
        used_w = span[0] * self.scale
        used_h = span[1] * self.scale
        self.ox = x0 + (w - used_w) * 0.5
        self.oy = y0 + (h - used_h) * 0.5
        self.lo = lo
        self.hi = hi
        self.m_per_px = 1.0 / self.scale

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        if len(xy) == 0:
            return np.empty((0, 2), dtype=np.int32)
        px = self.ox + (xy[:, 0] - self.lo[0]) * self.scale
        py = self.oy + (self.hi[1] - xy[:, 1]) * self.scale   # y flipped
        return np.stack([px, py], axis=1).astype(np.int32)


def _draw_heading_arrow(
    canvas: np.ndarray,
    cx: int, cy: int, theta: float,
    length: int = 18, color: tuple = (255, 200, 80),
) -> None:
    ex = int(cx + length * np.cos(theta))
    ey = int(cy - length * np.sin(theta))   # y flipped in image
    cv2.arrowedLine(canvas, (cx, cy), (ex, ey), color, 2, cv2.LINE_AA, tipLength=0.4)


def _draw_scale_bar(
    canvas: np.ndarray,
    x0: int, y0: int,
    scale_m: float, m_per_px: float,
    color: tuple = (210, 210, 220),
) -> None:
    bar_px = max(2, int(scale_m / max(m_per_px, 1e-9)))
    cv2.line(canvas, (x0, y0), (x0 + bar_px, y0), color, 2, cv2.LINE_AA)
    cv2.line(canvas, (x0, y0 - 5), (x0, y0 + 5), color, 2, cv2.LINE_AA)
    cv2.line(canvas, (x0 + bar_px, y0 - 5), (x0 + bar_px, y0 + 5), color, 2, cv2.LINE_AA)
    _put_text(canvas, f"{scale_m:.0f} m", x0 + bar_px // 2 - 14, y0 - 9, color, 0.52)


def _logodds_to_bgr(grid: GridMap) -> np.ndarray:
    """
    Render the occupancy grid on a BLACK background (thesis style):
      unknown (logodds==0) -> black,  free (logodds<0) -> dark gray,
      occupied (logodds>0) -> white.  Intensity scales with confidence.
    """
    lo = grid.logodds.astype(np.float32)
    val = np.zeros(lo.shape, dtype=np.float32)

    free = lo < 0.0
    occ = lo > 0.0
    # free space: subtle dark-gray corridor, brighter with confidence
    val[free] = 45.0 + 45.0 * np.clip(lo[free] / grid.l_min, 0.0, 1.0)
    # occupied: bright walls, near-white at high confidence
    val[occ] = 150.0 + 105.0 * np.clip(lo[occ] / grid.l_max, 0.0, 1.0)

    gray = np.flipud(np.clip(val, 0, 255).astype(np.uint8))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# Panel builders
# ─────────────────────────────────────────────────────────────────────────────

def _draw_header(panel: np.ndarray, title: str, subtitle: str,
                 right_text: str, header_h: int) -> None:
    """Clean header band with a title, subtitle and right-aligned counter."""
    w = panel.shape[1]
    _put_text(panel, title, 24, 40, TITLE_COLOR, 0.88, 2)
    _put_text(panel, subtitle, 24, 64, SUBTITLE_COLOR, 0.55, 1)
    if right_text:
        (tw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.66, 2)
        _put_text(panel, right_text, w - tw - 24, 40, (210, 225, 255), 0.66, 2)
    cv2.line(panel, (24, header_h - 8), (w - 24, header_h - 8), HEADER_RULE, 1, cv2.LINE_AA)


def _build_left_panel(
    poses: np.ndarray,
    current_k: int,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    panel_w: int, panel_h: int,
    header_h: int = 80, footer_h: int = 70,
) -> np.ndarray:
    """Trajectory view: path grown up to current_k with gradient colour."""
    panel = np.full((panel_h, panel_w, 3), BG_DARK, dtype=np.uint8)

    view_x0, view_y0 = 24, header_h
    view_w = panel_w - 48
    view_h = panel_h - header_h - footer_h
    proj = _Projector(bounds_lo, bounds_hi, view_x0, view_y0, view_w, view_h)

    total = max(len(poses) - 1, 1)

    # Trajectory polyline (gradient by time)
    if current_k >= 1:
        pxs = proj(poses[:current_k + 1, :2])
        for i in range(len(pxs) - 1):
            col = _traj_color(i / total)
            cv2.line(panel, tuple(pxs[i].tolist()), tuple(pxs[i + 1].tolist()),
                     col, 2, cv2.LINE_AA)

    # Start marker
    sp = proj(poses[:1, :2])[0]
    cv2.circle(panel, tuple(sp.tolist()), 7, START_COLOR, -1, cv2.LINE_AA)
    cv2.circle(panel, tuple(sp.tolist()), 7, (255, 255, 255), 1, cv2.LINE_AA)

    # Current robot position + heading
    if current_k < len(poses):
        cp = proj(poses[current_k:current_k + 1, :2])[0]
        cx, cy = int(cp[0]), int(cp[1])
        theta = float(poses[current_k, 2])
        cv2.circle(panel, (cx, cy), 9, ROBOT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(panel, (cx, cy), 9, (255, 255, 255), 1, cv2.LINE_AA)
        _draw_heading_arrow(panel, cx, cy, theta, length=22, color=(90, 235, 255))

    # Header + scale bar
    _draw_header(panel, "Trajectory",
                 "Hector SLAM  (scan-to-map)  |  lab_run_2",
                 f"scan {current_k + 1:>4d} / {len(poses)}", header_h)
    _draw_scale_bar(panel, view_x0, panel_h - 30, 2.0, proj.m_per_px)

    # Legend (bottom-right)
    leg_x, leg_y = panel_w - 220, panel_h - 34
    cv2.circle(panel, (leg_x, leg_y - 4), 6, START_COLOR, -1, cv2.LINE_AA)
    _put_text(panel, "Start", leg_x + 12, leg_y, LABEL_COLOR, 0.52)
    cv2.circle(panel, (leg_x + 86, leg_y - 4), 6, ROBOT_COLOR, -1, cv2.LINE_AA)
    _put_text(panel, "Robot", leg_x + 98, leg_y, LABEL_COLOR, 0.52)

    return panel


def _build_right_panel(
    grid: GridMap,
    poses: np.ndarray,
    current_k: int,
    pts_world_curr: np.ndarray,
    crop_ix: tuple[int, int],
    crop_iy: tuple[int, int],
    panel_w: int, panel_h: int,
    header_h: int = 80, footer_h: int = 70,
) -> np.ndarray:
    """Map panel: occupancy grid (black bg) + current scan overlay."""
    panel = np.full((panel_h, panel_w, 3), BG_DARK, dtype=np.uint8)

    area_x0, area_y0 = 24, header_h
    area_w = panel_w - 48
    area_h = panel_h - header_h - footer_h

    # Build map image (black background) and crop to trajectory bbox.
    map_bgr_full = _logodds_to_bgr(grid)
    ix_lo, ix_hi = crop_ix
    iy_lo, iy_hi = crop_iy
    map_crop = map_bgr_full[iy_lo:iy_hi, ix_lo:ix_hi]
    if map_crop.shape[0] < 2 or map_crop.shape[1] < 2:
        map_crop = map_bgr_full
        ix_lo, iy_lo = 0, 0

    crop_h, crop_w = map_crop.shape[:2]
    # Aspect-preserving fit (letterbox) into the map area.
    scale = min(area_w / crop_w, area_h / crop_h)
    dw, dh = int(crop_w * scale), int(crop_h * scale)
    map_scaled = cv2.resize(map_crop, (dw, dh), interpolation=cv2.INTER_NEAREST)
    off_x = area_x0 + (area_w - dw) // 2
    off_y = area_y0 + (area_h - dh) // 2
    panel[off_y:off_y + dh, off_x:off_x + dw] = map_scaled

    def grid_to_panel(gx: float, gy: float) -> tuple[int, int]:
        # After flipud: image y = (size-1) - gy
        ix = gx
        iy = (grid.size - 1) - gy
        px = int((ix - ix_lo) * scale) + off_x
        py = int((iy - iy_lo) * scale) + off_y
        return px, py

    # Faint laser rays robot -> scan endpoints
    if current_k < len(poses) and pts_world_curr.shape[0] > 0:
        rg = grid.world_to_grid(poses[current_k:current_k + 1, :2])[0]
        r_px, r_py = grid_to_panel(float(rg[0]), float(rg[1]))
        ray_overlay = panel.copy()
        g_pts = grid.world_to_grid(pts_world_curr)
        for gp in g_pts[grid.in_bounds(g_pts)]:
            ep = grid_to_panel(float(gp[0]), float(gp[1]))
            cv2.line(ray_overlay, (r_px, r_py), ep, RAY_COLOR, 1, cv2.LINE_AA)
        cv2.addWeighted(ray_overlay, 0.35, panel, 0.65, 0, panel)

    # Current scan endpoints (bright)
    if pts_world_curr.shape[0] > 0:
        g_pts = grid.world_to_grid(pts_world_curr)
        for gp in g_pts[grid.in_bounds(g_pts)]:
            px, py = grid_to_panel(float(gp[0]), float(gp[1]))
            cv2.circle(panel, (px, py), 2, SCAN_PT_COLOR, -1, cv2.LINE_AA)

    # Robot marker + heading
    if current_k < len(poses):
        theta = float(poses[current_k, 2])
        rg = grid.world_to_grid(poses[current_k:current_k + 1, :2])[0]
        r_px, r_py = grid_to_panel(float(rg[0]), float(rg[1]))
        cv2.circle(panel, (r_px, r_py), 7, ROBOT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(panel, (r_px, r_py), 7, (255, 255, 255), 1, cv2.LINE_AA)
        _draw_heading_arrow(panel, r_px, r_py, theta, length=18, color=(90, 235, 255))

    # Header + scale bar (use real map scale: panel px per metre)
    _draw_header(panel, "Occupancy Grid Map",
                 "incremental scan-to-map integration",
                 f"scans integrated: {current_k + 1}", header_h)
    m_per_px = grid.res / scale
    _draw_scale_bar(panel, area_x0, panel_h - 30, 2.0, m_per_px)

    # Legend (bottom-right)
    leg_x, leg_y = panel_w - 290, panel_h - 34
    cv2.circle(panel, (leg_x, leg_y - 4), 5, SCAN_PT_COLOR, -1, cv2.LINE_AA)
    _put_text(panel, "current scan", leg_x + 12, leg_y, LABEL_COLOR, 0.52)
    cv2.circle(panel, (leg_x + 150, leg_y - 4), 5, ROBOT_COLOR, -1, cv2.LINE_AA)
    _put_text(panel, "robot", leg_x + 162, leg_y, LABEL_COLOR, 0.52)

    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────────────────────────────────────

def render_video(
    traj_path: str,
    out_path: str,
    scan_variant: str,
    frame_step: int,
    fps: float,
    max_scans: int,
    out_w: int,
    out_h: int,
    map_res: float,
    map_size_m: float,
    l_occ: float,
    l_free: float,
    l_min: float,
    l_max: float,
    ray_steps: int,
    traj_margin: int,
) -> None:
    print(f"[render] Loading trajectory: {traj_path}")
    stamps, poses = _load_trajectory(traj_path)
    if max_scans > 0:
        stamps = stamps[:max_scans]
        poses = poses[:max_scans]
    N = len(poses)
    print(f"[render] Loaded {N} poses")

    print(f"[render] Loading scan data (variant={scan_variant}) ...")
    profile, pts_list = _load_scan_points(stamps, scan_variant=scan_variant)
    print(f"[render] {len(pts_list)} scans loaded and pre-processed")

    # Viewport bounds for left (trajectory) panel
    traj_xy = poses[:, :2]
    span = traj_xy.max(0) - traj_xy.min(0)
    pad_m = max(float(span.max()) * 0.12, 1.5)
    bounds_lo = traj_xy.min(0) - pad_m
    bounds_hi = traj_xy.max(0) + pad_m
    # Square viewport
    center = (bounds_lo + bounds_hi) * 0.5
    half = float(np.max(bounds_hi - bounds_lo)) * 0.5 * 1.05
    bounds_lo = center - half
    bounds_hi = center + half

    # Compute map crop region (fixed, based on full trajectory)
    grid = GridMap(res=map_res, size_m=map_size_m, l_min=l_min, l_max=l_max)
    all_gxy = grid.world_to_grid(traj_xy)
    pad_px = int(3.0 / map_res)  # 3 m padding in pixels
    ix_lo = max(0, int(all_gxy[:, 0].min()) - pad_px)
    ix_hi = min(grid.size, int(all_gxy[:, 0].max()) + pad_px + 1)
    gy_lo = int(all_gxy[:, 1].min())
    gy_hi = int(all_gxy[:, 1].max())
    # After flipud: iy = size-1 - gy; top of crop = size-1 - gy_hi
    iy_lo = max(0, (grid.size - 1) - gy_hi - pad_px)
    iy_hi = min(grid.size, (grid.size - 1) - gy_lo + pad_px + 1)
    crop_ix = (ix_lo, ix_hi)
    crop_iy = (iy_lo, iy_hi)

    # Video writer
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {out_path}")

    panel_w = out_w // 2
    panel_h = out_h

    total_frames = (N + frame_step - 1) // frame_step
    print(f"[render] Rendering {total_frames} frames  (frame_step={frame_step}, fps={fps})")

    prev_k = -1
    for frame_idx in range(total_frames):
        current_k = min(frame_idx * frame_step, N - 1)

        # Integrate new scans into map (incremental — only scans since last frame)
        for k in range(prev_k + 1, current_k + 1):
            pts = pts_list[k]
            if pts.shape[0] == 0:
                continue
            pose_arr = poses[k]
            pts_world = _transform_points(pose_arr, pts)
            grid.integrate_scan_simple(
                pose=pose_arr,
                pts_world=pts_world,
                l_free=l_free,
                l_occ=l_occ,
                ray_steps=ray_steps,
            )
        prev_k = current_k

        # Current scan points in world frame (for overlay)
        pts_curr = pts_list[current_k]
        pts_world_curr = _transform_points(poses[current_k], pts_curr) if pts_curr.shape[0] > 0 else np.empty((0, 2))

        # Render panels
        left = _build_left_panel(poses, current_k, bounds_lo, bounds_hi, panel_w, panel_h)
        right = _build_right_panel(
            grid, poses, current_k, pts_world_curr,
            crop_ix, crop_iy, panel_w, panel_h,
        )

        # Divider line
        frame = np.hstack([left, right])
        cv2.line(frame, (panel_w - 1, 0), (panel_w - 1, out_h), DIVIDER_COLOR, 2)

        # Progress bar
        progress = (frame_idx + 1) / total_frames
        bar_y = out_h - 6
        bar_x0, bar_x1 = 8, out_w - 8
        cv2.rectangle(frame, (bar_x0, bar_y - 4), (bar_x1, bar_y), (40, 40, 50), -1)
        cv2.rectangle(frame, (bar_x0, bar_y - 4), (bar_x0 + int((bar_x1 - bar_x0) * progress), bar_y), (80, 160, 255), -1)

        writer.write(frame)

        if (frame_idx + 1) % 50 == 0 or frame_idx == total_frames - 1:
            print(f"[render]  frame {frame_idx + 1}/{total_frames}  scan {current_k + 1}/{N}")

    writer.release()
    print(f"[render] Video saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Render Hector SLAM map-building split-screen video for lab_run_2."
    )
    ap.add_argument("--traj", default=None,
                    help="Trajectory file (auto-discovered from hector_outputs/ if omitted).")
    ap.add_argument("--hector-out", default="hector_outputs",
                    help="Directory to search for auto-discovery (default: hector_outputs).")
    ap.add_argument("--variant", default="raw", choices=["raw", "360"],
                    help="Scan variant (default: raw).")
    ap.add_argument("--out", default="hector_outputs/video/hector_map_build_lab.mp4",
                    help="Output MP4 path.")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="Output video fps (default: 30).")
    ap.add_argument("--frame-step", type=int, default=3, dest="frame_step",
                    help="Scans per video frame (default: 3, ~9x real-time at 10 Hz scan rate).")
    ap.add_argument("--max-scans", type=int, default=0, dest="max_scans",
                    help="Cap number of scans (0=all).")
    ap.add_argument("--width", type=int, default=1920,
                    help="Output video width (default: 1920).")
    ap.add_argument("--height", type=int, default=1080,
                    help="Output video height (default: 1080).")
    ap.add_argument("--map-res", type=float, default=0.05, dest="map_res",
                    help="Map resolution in m (default: 0.05).")
    ap.add_argument("--map-size", type=float, default=40.0, dest="map_size",
                    help="Map side length in m (default: 40.0).")
    ap.add_argument("--no-filter", action="store_true", dest="no_filter",
                    help="Disable voxel pre-filtering of scan points.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    traj = args.traj
    if traj is None:
        traj = _auto_traj(args.hector_out, args.variant)
        if traj is None:
            print(f"[render] ERROR: no trajectory file found in {args.hector_out}/", file=sys.stderr)
            return 1
        print(f"[render] Auto-discovered trajectory: {traj}")

    render_video(
        traj_path=traj,
        out_path=args.out,
        scan_variant=args.variant,
        frame_step=max(1, args.frame_step),
        fps=args.fps,
        max_scans=args.max_scans,
        out_w=args.width,
        out_h=args.height,
        map_res=args.map_res,
        map_size_m=args.map_size,
        l_occ=1.0,
        l_free=-0.1,
        l_min=-5.0,
        l_max=5.0,
        ray_steps=20,
        traj_margin=60,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
