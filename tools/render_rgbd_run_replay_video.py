#!/usr/bin/env python3
"""
Render replay videos for a completed RGB-D SLAM run.

Layouts
-------
`side_by_side`:
  Left pane:
    Original RGB frames replayed from the dataset.

  Right pane:
    Incremental top-down map reconstruction using the run's keyframe poses and
    the recorded RGB-D images.

`tri_view`:
  Upper-left:
    Original RGB frames.

  Lower-left:
    Estimated sparse-map growth panel based on the exported sparse point cloud
    plus run trajectory/keyframes.

  Right:
    Incremental top-down RGB-D map build.

The implementation is optimized for the completed lab run artifacts used in
this workspace, but it stays generic enough for TUM-like RGB-D datasets with:

  rgb/  depth/  associations.txt  camera.yaml

Example
-------
python -m tools.render_rgbd_run_replay_video \
  --run visual_slam_outputs/lab_rgbd_run_2_B_loop_gba \
  --dataset datasets/lab_rgbd_run_2 \
  --output visual_slam_outputs/lab_rgbd_run_2_B_loop_gba/video/replay_map_build.mp4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class AssocEntry:
    timestamp: float
    rgb_rel: str
    depth_rel: str


@dataclass
class CameraParams:
    width: int
    height: int
    fps: float
    fx: float
    fy: float
    cx: float
    cy: float
    depth_factor: float


@dataclass
class KeyframeProjection:
    kid: int
    img_id: int
    timestamp: float
    points_xz: np.ndarray
    colors_bgr: np.ndarray
    position_xz: np.ndarray


def read_ply_points(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 3), dtype=np.float32)
    points = []
    in_header = True
    with open(path, encoding="utf-8") as f:
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
    return np.asarray(points, dtype=np.float32)


def filter_map_points_to_scene(
    points_xyz: np.ndarray,
    trajectory_xyz: np.ndarray,
    *,
    padding_m: float = 3.0,
) -> np.ndarray:
    """Clip sparse-map outliers to the trajectory bbox plus padding.

    The exported sparse map can contain far-away triangulation outliers. Those
    points collapse the visualization bounds and make the map appear misaligned
    relative to the trajectory. We reuse the same style of bbox filtering used
    in the plotting tools.
    """
    if len(points_xyz) == 0 or len(trajectory_xyz) == 0:
        return points_xyz
    lo = trajectory_xyz.min(axis=0) - padding_m
    hi = trajectory_xyz.max(axis=0) + padding_m
    mask = np.all((points_xyz >= lo) & (points_xyz <= hi), axis=1)
    return points_xyz[mask]


def _find_file(run_dir: Path, pattern: str) -> Path | None:
    matches = sorted(run_dir.glob(pattern))
    return matches[0] if matches else None


def read_trajectory(path: Path) -> np.ndarray:
    rows = []
    with open(path, encoding="utf-8") as f:
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
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_associations(path: Path) -> list[AssocEntry]:
    entries: list[AssocEntry] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                entries.append(AssocEntry(float(parts[0]), parts[1], parts[3]))
    return entries


def read_camera_yaml(path: Path) -> CameraParams:
    params: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            m = re.match(r"^([A-Za-z0-9_.]+)\s*:\s*(.+)$", line)
            if m:
                params[m.group(1)] = m.group(2).strip()

    def _get_float(*keys: str, default: float | None = None) -> float:
        for key in keys:
            if key in params:
                return float(params[key])
        if default is None:
            raise KeyError(keys[0])
        return float(default)

    def _get_int(*keys: str, default: int | None = None) -> int:
        return int(round(_get_float(*keys, default=float(default) if default is not None else None)))

    fps = _get_float("Camera.fps", "image.fps", default=30.0)
    depth_map_factor = _get_float("DepthMapFactor", "depth.depth_map_factor", default=1000.0)
    return CameraParams(
        width=_get_int("Camera.width", "image.width", default=640),
        height=_get_int("Camera.height", "image.height", default=480),
        fps=fps,
        fx=_get_float("Camera.fx", "camera.fx"),
        fy=_get_float("Camera.fy", "camera.fy"),
        cx=_get_float("Camera.cx", "camera.cx"),
        cy=_get_float("Camera.cy", "camera.cy"),
        depth_factor=1.0 / depth_map_factor,
    )


def read_keyframes(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.sort(key=lambda item: int(item.get("img_id", item.get("frame_id", 0))))
    return data


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to load RGB frame: {path}")
    return img


def _read_depth(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to load depth frame: {path}")
    return img


def _safe_float(value: str | None) -> float | None:
    if value in (None, "", "NA", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _state_color(row: dict) -> tuple[int, int, int]:
    if str(row.get("state", "")).upper() == "OK":
        return (80, 210, 110)
    return (70, 120, 235)


def _robust_bounds(points_xz: list[np.ndarray], trajectory_xz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arrays = [trajectory_xz]
    arrays.extend(arr for arr in points_xz if len(arr) > 0)
    pts = np.vstack(arrays)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    lo = np.percentile(pts, 0.5, axis=0)
    hi = np.percentile(pts, 99.5, axis=0)
    center = (lo + hi) * 0.5
    span = max(float(np.max(hi - lo)), 0.5)
    half = span * 0.58
    return center - half, center + half


def _world_to_px(points_xz: np.ndarray, lo: np.ndarray, hi: np.ndarray, width: int, height: int, margin: int) -> np.ndarray:
    if len(points_xz) == 0:
        return np.empty((0, 2), dtype=np.int32)
    inner_w = max(1, width - 2 * margin)
    inner_h = max(1, height - 2 * margin)
    norm_x = (points_xz[:, 0] - lo[0]) / max(hi[0] - lo[0], 1e-9)
    norm_z = (points_xz[:, 1] - lo[1]) / max(hi[1] - lo[1], 1e-9)
    px = margin + np.clip(norm_x, 0.0, 1.0) * inner_w
    py = height - margin - np.clip(norm_z, 0.0, 1.0) * inner_h
    return np.round(np.stack([px, py], axis=1)).astype(np.int32)


def build_keyframe_projections(
    dataset_dir: Path,
    associations: list[AssocEntry],
    camera: CameraParams,
    keyframes: list[dict],
    *,
    depth_stride: int,
    max_depth_m: float,
    min_depth_m: float,
) -> list[KeyframeProjection]:
    uu = np.arange(0, camera.width, depth_stride, dtype=np.float32)
    vv = np.arange(0, camera.height, depth_stride, dtype=np.float32)
    U, V = np.meshgrid(uu, vv)
    xc_base = (U - camera.cx) / camera.fx
    yc_base = (V - camera.cy) / camera.fy

    projections: list[KeyframeProjection] = []
    for kf in keyframes:
        img_id = int(kf.get("img_id", kf.get("frame_id", -1)))
        if img_id < 0 or img_id >= len(associations):
            continue
        assoc = associations[img_id]
        rgb = _read_rgb(dataset_dir / assoc.rgb_rel)
        depth = _read_depth(dataset_dir / assoc.depth_rel)
        depth_sub = depth[::depth_stride, ::depth_stride].astype(np.float32) * camera.depth_factor
        rgb_sub = rgb[::depth_stride, ::depth_stride]

        h, w = depth_sub.shape
        xc = xc_base[:h, :w]
        yc = yc_base[:h, :w]

        valid = (depth_sub >= min_depth_m) & (depth_sub <= max_depth_m)
        if not np.any(valid):
            points_xz = np.empty((0, 2), dtype=np.float32)
            colors = np.empty((0, 3), dtype=np.uint8)
        else:
            z = depth_sub[valid]
            x = xc[valid] * z
            y = yc[valid] * z
            pts_cam = np.stack([x, y, z], axis=1)
            Twc = np.asarray(kf["Twc"], dtype=np.float64).reshape(4, 4)
            pts_world = (Twc[:3, :3] @ pts_cam.T).T + Twc[:3, 3]
            points_xz = pts_world[:, [0, 2]].astype(np.float32)
            colors = rgb_sub[valid].astype(np.uint8)

        projections.append(
            KeyframeProjection(
                kid=int(kf["kid"]),
                img_id=img_id,
                timestamp=float(kf.get("timestamp", assoc.timestamp)),
                points_xz=points_xz,
                colors_bgr=colors,
                position_xz=np.asarray(kf["position"], dtype=np.float32)[[0, 2]],
            )
        )
    return projections


def _draw_points(canvas: np.ndarray, px: np.ndarray, colors: np.ndarray) -> None:
    if len(px) == 0:
        return
    h, w = canvas.shape[:2]
    mask = (
        (px[:, 0] >= 0) & (px[:, 0] < w - 1) &
        (px[:, 1] >= 0) & (px[:, 1] < h - 1)
    )
    px = px[mask]
    colors = colors[mask]
    if len(px) == 0:
        return
    x = px[:, 0]
    y = px[:, 1]
    canvas[y, x] = colors
    canvas[y, x + 1] = colors
    canvas[y + 1, x] = colors


def _draw_plain_points(canvas: np.ndarray, px: np.ndarray, color: tuple[int, int, int], radius: int = 1) -> None:
    if len(px) == 0:
        return
    h, w = canvas.shape[:2]
    mask = (
        (px[:, 0] >= 0) & (px[:, 0] < w) &
        (px[:, 1] >= 0) & (px[:, 1] < h)
    )
    for p in px[mask]:
        cv2.circle(canvas, tuple(int(v) for v in p), radius, color, -1, cv2.LINE_AA)


def _put_line(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.65, thickness: int = 1) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _fit_with_letterbox(img: np.ndarray, width: int, height: int, fill: tuple[int, int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), fill, dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _build_sparse_growth_panel(
    sparse_points_xz: np.ndarray,
    trajectory_xz: np.ndarray,
    keyframe_positions_xz: np.ndarray,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    current_frame_idx: int,
    current_map_points: int,
    final_map_points_stat: int,
    current_keyframes: int,
    panel_width: int,
    panel_height: int,
) -> np.ndarray:
    panel = np.full((panel_height, panel_width, 3), (14, 14, 16), dtype=np.uint8)
    inner_margin = 20
    map_size = min(panel_width - 2 * inner_margin, panel_height - 64)
    map_size = max(map_size, 240)
    x0 = (panel_width - map_size) // 2
    y0 = 46

    map_canvas = np.full((map_size, map_size, 3), (18, 18, 20), dtype=np.uint8)
    for frac in np.linspace(0.0, 1.0, 6):
        x = int(round(18 + frac * (map_size - 36)))
        y = int(round(18 + frac * (map_size - 36)))
        cv2.line(map_canvas, (x, 18), (x, map_size - 18), (36, 36, 40), 1, cv2.LINE_AA)
        cv2.line(map_canvas, (18, y), (map_size - 18, y), (36, 36, 40), 1, cv2.LINE_AA)

    sparse_points_px = _world_to_px(sparse_points_xz, bounds_lo, bounds_hi, map_size, map_size, 18)
    traj_px = _world_to_px(trajectory_xz, bounds_lo, bounds_hi, map_size, map_size, 18)
    keyframe_positions_px = _world_to_px(keyframe_positions_xz, bounds_lo, bounds_hi, map_size, map_size, 18)

    if final_map_points_stat > 0 and len(sparse_points_px) > 0:
        reveal_ratio = np.clip(current_map_points / max(final_map_points_stat, 1), 0.0, 1.0)
        reveal_count = int(round(reveal_ratio * len(sparse_points_px)))
        if reveal_count > 0:
            _draw_plain_points(map_canvas, sparse_points_px[:reveal_count], (210, 210, 220), radius=1)

    if current_frame_idx > 0:
        cv2.polylines(
            map_canvas,
            [traj_px[: current_frame_idx + 1].astype(np.int32)],
            False,
            (80, 170, 255),
            2,
            cv2.LINE_AA,
        )
    if current_keyframes > 0 and len(keyframe_positions_px) > 0:
        _draw_plain_points(map_canvas, keyframe_positions_px[:current_keyframes], (0, 170, 255), radius=2)
    if current_frame_idx < len(traj_px):
        cv2.circle(map_canvas, tuple(int(v) for v in traj_px[current_frame_idx]), 5, (60, 60, 255), -1, cv2.LINE_AA)
    cv2.circle(map_canvas, tuple(int(v) for v in traj_px[0]), 5, (70, 210, 100), -1, cv2.LINE_AA)

    panel[y0:y0 + map_size, x0:x0 + map_size] = map_canvas
    _put_line(panel, "Estimated Sparse Map Growth", 18, 24, (235, 235, 235), 0.72, 2)
    _put_line(panel, "progressively revealed exported sparse point cloud", 18, 42, (180, 200, 210), 0.5, 1)
    _put_line(panel, f"visible sparse pts ~ {min(len(sparse_points_px), int(round(max(current_map_points, 0) / max(final_map_points_stat, 1) * len(sparse_points_px)))):,}", 18, panel_height - 18, (190, 190, 200), 0.55, 1)
    return panel


def render_video(
    run_dir: Path,
    dataset_dir: Path,
    output_path: Path,
    *,
    max_frames: int,
    frame_step: int,
    depth_stride: int,
    max_depth_m: float,
    min_depth_m: float,
    output_width: int,
    output_height: int,
    fps_override: float | None,
    layout: str,
) -> Path:
    trajectory_path = _find_file(run_dir, "trajectory_*.txt")
    frame_log_path = _find_file(run_dir, "frame_log_*.csv")
    keyframes_path = run_dir / "keyframes.json"
    if trajectory_path is None or frame_log_path is None or not keyframes_path.exists():
        raise FileNotFoundError("Run directory is missing trajectory, frame log, or keyframes artifacts")

    associations = read_associations(dataset_dir / "associations.txt")
    camera = read_camera_yaml(dataset_dir / "camera.yaml")
    trajectory = read_trajectory(trajectory_path)
    frame_log = read_frame_log(frame_log_path)
    keyframes = read_keyframes(keyframes_path)
    sparse_map_points = read_ply_points(run_dir / "map_points.ply")

    if len(frame_log) != len(associations):
        raise RuntimeError(f"Frame log length ({len(frame_log)}) does not match associations length ({len(associations)})")
    if len(trajectory) != len(frame_log):
        raise RuntimeError(f"Trajectory length ({len(trajectory)}) does not match frame log length ({len(frame_log)})")

    keyframe_projections = build_keyframe_projections(
        dataset_dir,
        associations,
        camera,
        keyframes,
        depth_stride=depth_stride,
        max_depth_m=max_depth_m,
        min_depth_m=min_depth_m,
    )

    trajectory_xyz = trajectory[:, 1:4].astype(np.float32)
    sparse_map_points = filter_map_points_to_scene(sparse_map_points, trajectory_xyz, padding_m=3.0)
    trajectory_xz = trajectory[:, [1, 3]].astype(np.float32)
    sparse_map_xz = sparse_map_points[:, [0, 2]].astype(np.float32) if len(sparse_map_points) > 0 else np.empty((0, 2), dtype=np.float32)
    bounds_lo, bounds_hi = _robust_bounds([kf.points_xz for kf in keyframe_projections] + ([sparse_map_xz] if len(sparse_map_xz) > 0 else []), trajectory_xz)

    if layout == "tri_view":
        right_w = int(round(output_width * 0.52))
        left_w = output_width - right_w
    else:
        right_w = output_width // 2
        left_w = output_width - right_w
    map_size = min(right_w - 40, output_height - 100)
    map_size = max(map_size, 480)
    map_margin = 32

    traj_px = _world_to_px(trajectory_xz, bounds_lo, bounds_hi, map_size, map_size, map_margin)
    sparse_map_px = _world_to_px(sparse_map_xz, bounds_lo, bounds_hi, map_size, map_size, map_margin) if len(sparse_map_xz) > 0 else np.empty((0, 2), dtype=np.int32)
    for kf in keyframe_projections:
        kf.points_px = _world_to_px(kf.points_xz, bounds_lo, bounds_hi, map_size, map_size, map_margin)
        kf.pos_px = _world_to_px(kf.position_xz[None, :], bounds_lo, bounds_hi, map_size, map_size, map_margin)[0]
    keyframe_positions_xz = np.asarray([kf.position_xz for kf in keyframe_projections], dtype=np.float32) if keyframe_projections else np.empty((0, 2), dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = fps_override if fps_override is not None else max(camera.fps / max(frame_step, 1), 1.0)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for {output_path}")

    map_layer = np.full((map_size, map_size, 3), (18, 18, 20), dtype=np.uint8)
    path_layer = np.zeros_like(map_layer)
    grid_layer = np.zeros_like(map_layer)
    for frac in np.linspace(0.0, 1.0, 6):
        x = int(round(map_margin + frac * (map_size - 2 * map_margin)))
        y = int(round(map_margin + frac * (map_size - 2 * map_margin)))
        cv2.line(grid_layer, (x, map_margin), (x, map_size - map_margin), (38, 38, 42), 1, cv2.LINE_AA)
        cv2.line(grid_layer, (map_margin, y), (map_size - map_margin, y), (38, 38, 42), 1, cv2.LINE_AA)

    next_kf = 0
    prev_pose_px = traj_px[0]
    prev_frame_idx = 0
    last_loop_flash_until = -1
    final_map_points_stat = max(int(row["points"]) for row in frame_log)

    end_frame = len(frame_log) if max_frames <= 0 else min(len(frame_log), max_frames)

    for frame_idx in range(0, end_frame, frame_step):
        while next_kf < len(keyframe_projections) and keyframe_projections[next_kf].img_id <= frame_idx:
            kf = keyframe_projections[next_kf]
            _draw_points(map_layer, kf.points_px, kf.colors_bgr)
            cv2.circle(map_layer, tuple(int(v) for v in kf.pos_px), 3, (0, 170, 255), -1, cv2.LINE_AA)
            next_kf += 1

        for step_idx in range(prev_frame_idx + 1, frame_idx + 1):
            cur_px = traj_px[step_idx]
            cv2.line(path_layer, tuple(int(v) for v in prev_pose_px), tuple(int(v) for v in cur_px), (240, 220, 70), 2, cv2.LINE_AA)
            prev_pose_px = cur_px
        prev_frame_idx = frame_idx

        row = frame_log[frame_idx]
        assoc = associations[frame_idx]
        rgb = _read_rgb(dataset_dir / assoc.rgb_rel)

        if frame_idx > 0 and row.get("loop_global_ba_started") == "1" and frame_log[frame_idx - 1].get("loop_global_ba_started") == "0":
            last_loop_flash_until = frame_idx + 45

        left_panel = np.full((output_height, left_w, 3), (12, 12, 12), dtype=np.uint8)
        right_panel = np.full((output_height, right_w, 3), (14, 14, 16), dtype=np.uint8)
        if layout == "tri_view":
            top_h = output_height // 2
            bottom_h = output_height - top_h
            fitted_rgb = _fit_with_letterbox(rgb, left_w - 24, top_h - 92, (24, 24, 26))
            rgb_y0 = 58
            left_panel[rgb_y0:rgb_y0 + fitted_rgb.shape[0], 12:12 + fitted_rgb.shape[1]] = fitted_rgb
            _put_line(left_panel, "Recorded RGB Replay", 18, 28, (235, 235, 235), 0.82, 2)
            _put_line(left_panel, f"frame {frame_idx + 1}/{end_frame}", 18, 50, (160, 220, 255), 0.64, 2)
            sparse_panel = _build_sparse_growth_panel(
                sparse_map_xz,
                trajectory_xz,
                keyframe_positions_xz,
                bounds_lo,
                bounds_hi,
                frame_idx,
                int(row["points"]),
                final_map_points_stat,
                int(row["keyframes"]),
                left_w,
                bottom_h,
            )
            left_panel[top_h:, :] = sparse_panel
            _put_line(left_panel, f"timestamp {assoc.timestamp:.6f}", 18, top_h - 16, (205, 205, 205), 0.52, 1)
        else:
            fitted_rgb = _fit_with_letterbox(rgb, left_w - 24, output_height - 120, (24, 24, 26))
            left_panel[70:70 + fitted_rgb.shape[0], 12:12 + fitted_rgb.shape[1]] = fitted_rgb
            _put_line(left_panel, "Dataset RGB Replay", 18, 34, (235, 235, 235), 0.9, 2)
            _put_line(left_panel, f"frame {frame_idx + 1}/{end_frame}", 18, 58, (160, 220, 255), 0.7, 2)
            _put_line(left_panel, f"timestamp {assoc.timestamp:.6f}", 18, output_height - 64, (205, 205, 205), 0.6, 1)
        _put_line(left_panel, f"state {row['state']}   tracked {row['last_tracked']}", 18, output_height - 34, _state_color(row), 0.65, 2)

        map_img = cv2.addWeighted(map_layer, 1.0, grid_layer, 0.35, 0.0)
        map_img = cv2.addWeighted(map_img, 1.0, path_layer, 1.0, 0.0)
        current_px = traj_px[frame_idx]
        cv2.circle(map_img, tuple(int(v) for v in current_px), 7, (60, 60, 255), -1, cv2.LINE_AA)
        cv2.circle(map_img, tuple(int(v) for v in traj_px[0]), 6, (70, 210, 100), -1, cv2.LINE_AA)
        if frame_idx >= end_frame - frame_step:
            cv2.circle(map_img, tuple(int(v) for v in traj_px[-1]), 7, (255, 70, 70), 2, cv2.LINE_AA)
        if last_loop_flash_until >= frame_idx:
            cv2.rectangle(map_img, (6, 6), (map_size - 7, map_size - 7), (40, 220, 220), 3, cv2.LINE_AA)

        map_y0 = 56
        map_x0 = (right_w - map_size) // 2
        right_panel[map_y0:map_y0 + map_size, map_x0:map_x0 + map_size] = map_img
        _put_line(right_panel, "Incremental Top-Down Map Build", 20, 30, (235, 235, 235), 0.9, 2)
        _put_line(right_panel, "SLAM keyframe poses + RGB-D accumulation", 20, 52, (180, 210, 210), 0.65, 1)
        _put_line(right_panel, f"keyframes {row['keyframes']}   map points {row['points']}", 20, output_height - 66, (255, 205, 120), 0.7, 2)
        _put_line(right_panel, f"tracked {row['last_tracked']}", 20, output_height - 38, (120, 220, 255), 0.7, 2)

        progress = (frame_idx + 1) / end_frame
        bar_w = output_width - 40
        bar_x0 = 20
        bar_y = output_height - 14
        full_frame = np.hstack([left_panel, right_panel])
        cv2.rectangle(full_frame, (bar_x0, bar_y - 6), (bar_x0 + bar_w, bar_y), (55, 55, 60), -1)
        cv2.rectangle(full_frame, (bar_x0, bar_y - 6), (bar_x0 + int(round(bar_w * progress)), bar_y), (70, 180, 255), -1)

        writer.write(full_frame)

    writer.release()
    return output_path


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render replay videos for a completed RGB-D SLAM run.")
    parser.add_argument("--run", type=Path, required=True, help="Completed run directory containing trajectory/frame log/keyframes.")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset root with rgb/depth/associations.txt/camera.yaml.")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument("--frame-step", type=int, default=3, help="Replay every Nth frame. Default: 3.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap for preview renders. Default: full run.")
    parser.add_argument("--depth-stride", type=int, default=12, help="Pixel stride used for map accumulation. Default: 12.")
    parser.add_argument("--max-depth-m", type=float, default=3.0, help="Maximum depth used for map accumulation.")
    parser.add_argument("--min-depth-m", type=float, default=0.15, help="Minimum depth used for map accumulation.")
    parser.add_argument("--width", type=int, default=1920, help="Output video width.")
    parser.add_argument("--height", type=int, default=1080, help="Output video height.")
    parser.add_argument("--fps", type=float, default=0.0, help="Override output fps. Default uses dataset_fps/frame_step.")
    parser.add_argument("--layout", choices=("side_by_side", "tri_view"), default="side_by_side", help="Video layout preset.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_arg_parser()
    args = parser.parse_args(argv)
    output = render_video(
        args.run.expanduser().resolve(),
        args.dataset.expanduser().resolve(),
        args.output.expanduser().resolve(),
        max_frames=int(args.max_frames),
        frame_step=max(1, int(args.frame_step)),
        depth_stride=max(1, int(args.depth_stride)),
        max_depth_m=float(args.max_depth_m),
        min_depth_m=float(args.min_depth_m),
        output_width=int(args.width),
        output_height=int(args.height),
        fps_override=(float(args.fps) if float(args.fps) > 0 else None),
        layout=str(args.layout),
    )
    print(f"saved video: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
