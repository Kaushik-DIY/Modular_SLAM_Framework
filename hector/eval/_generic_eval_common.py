from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np

import hector.config as cfg

from carto.local_slam.range_to_points import ranges_to_points
from slam_core.dataio.dataset_catalog import DATASET_NAMES, load_dataset_scans
from slam_core.matching.preprocessing import PointCloudProcessor, PointCloudProcessorConfig


def dataset_tag(dataset_name: str, scan_variant: Optional[str]) -> str:
    if dataset_name == "lab_run_2":
        return f"{dataset_name}_{scan_variant or '360'}"
    return dataset_name


def parse_trajectory_context(traj_path: str) -> dict:
    path = Path(traj_path)
    stem = path.stem

    matcher_type = None
    if "_scan_to_map_" in stem:
        matcher_type = "scan_to_map"
        tag = stem.split("_scan_to_map_")[0].removeprefix("trajectory_")
    elif "_scan_to_submap_" in stem:
        matcher_type = "scan_to_submap"
        tag = stem.split("_scan_to_submap_")[0].removeprefix("trajectory_")
    else:
        raise ValueError(f"Could not infer matcher type from trajectory name: {path.name}")

    dataset_name = None
    scan_variant = None
    if tag.startswith("lab_run_2_"):
        dataset_name = "lab_run_2"
        scan_variant = tag[len("lab_run_2_"):]
    elif tag in DATASET_NAMES:
        dataset_name = tag
    else:
        raise ValueError(f"Could not infer dataset tag from trajectory name: {path.name}")

    debug_path = path.with_name(f"{stem}_debug.txt")
    if debug_path.exists():
        meta = _read_debug_metadata(debug_path)
        dataset_name = meta.get("dataset_name", dataset_name)
        scan_variant = meta.get("dataset_scan_variant", scan_variant)
        matcher_type = meta.get("matcher_type", matcher_type)

    return {
        "dataset_name": dataset_name,
        "scan_variant": scan_variant,
        "matcher_type": matcher_type,
        "dataset_tag": dataset_tag(dataset_name, scan_variant),
        "traj_path": str(path),
        "debug_path": str(debug_path) if debug_path.exists() else None,
    }


def _read_debug_metadata(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                continue
            body = line[1:].strip()
            if "=" not in body:
                continue
            key, value = body.split("=", 1)
            meta[key.strip()] = value.strip()
    return meta


def resolve_latest_local_traj(
    out_dir: str,
    dataset_name: Optional[str] = None,
    scan_variant: Optional[str] = None,
    matcher_type: Optional[str] = None,
) -> Optional[str]:
    out = Path(out_dir)
    candidates = []
    for path in out.glob("trajectory_*.txt"):
        name = path.name
        if "_debug" in name or "_pgo" in name:
            continue
        try:
            ctx = parse_trajectory_context(str(path))
        except ValueError:
            continue
        if dataset_name is not None and ctx["dataset_name"] != dataset_name:
            continue
        if dataset_name == "lab_run_2" and scan_variant is not None and ctx["scan_variant"] != scan_variant:
            continue
        if matcher_type is not None and ctx["matcher_type"] != matcher_type:
            continue
        candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def resolve_pgo_traj(local_traj_path: str, explicit_pgo: Optional[str], out_dir: str) -> Optional[str]:
    if explicit_pgo:
        return explicit_pgo

    local_path = Path(local_traj_path)
    direct = local_path.with_name(f"{local_path.stem}_pgo.txt")
    if direct.exists():
        return str(direct)

    try:
        ctx = parse_trajectory_context(str(local_path))
    except ValueError:
        return None

    out = Path(out_dir)
    candidates = []
    pattern = re.compile(r"^trajectory_.*_pgo\.txt$")
    for path in out.glob("trajectory_*.txt"):
        if not pattern.match(path.name):
            continue
        try:
            pgo_ctx = parse_pgo_trajectory_context(str(path))
        except ValueError:
            continue
        if pgo_ctx["dataset_name"] != ctx["dataset_name"]:
            continue
        if pgo_ctx["matcher_type"] != ctx["matcher_type"]:
            continue
        if ctx["dataset_name"] == "lab_run_2" and pgo_ctx["scan_variant"] != ctx["scan_variant"]:
            continue
        candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def parse_pgo_trajectory_context(traj_path: str) -> dict:
    path = Path(traj_path)
    stem = path.stem
    if not stem.endswith("_pgo"):
        raise ValueError(f"Not a PGO trajectory: {path.name}")
    base = path.with_name(f"{stem[:-4]}.txt")
    return parse_trajectory_context(str(base))


def load_local_traj(path: str, min_score: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    stamps, poses = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                t, x, y, th, sc = map(float, parts[:5])
            except ValueError:
                continue
            if sc >= min_score:
                stamps.append(t)
                poses.append([x, y, th])
    if not stamps:
        raise RuntimeError(f"No accepted poses found in {path} with min_score={min_score}")
    return np.array(stamps, dtype=float), np.array(poses, dtype=float)


def load_pgo_traj(path: str) -> np.ndarray:
    poses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                if len(parts) >= 4:
                    x, y, th = float(parts[1]), float(parts[2]), float(parts[3])
                elif len(parts) == 3:
                    x, y, th = float(parts[0]), float(parts[1]), float(parts[2])
                else:
                    continue
            except ValueError:
                continue
            poses.append([x, y, th])
    if not poses:
        raise RuntimeError(f"No poses found in {path}")
    return np.array(poses, dtype=float)


def configure_dataset(dataset_name: str) -> None:
    cfg.DATASET_NAME = dataset_name
    cfg._apply_profile(dataset_name)


def load_aligned_scan_points(
    dataset_name: str,
    scan_variant: Optional[str],
    stamps: np.ndarray,
    voxel_filter: Optional[bool] = None,
) -> tuple[object, list[np.ndarray]]:
    configure_dataset(dataset_name)
    profile, all_scans = load_dataset_scans(dataset_name, scan_variant=scan_variant)
    scan_stamps = np.array([s["t"] for s in all_scans], dtype=float)

    proc = PointCloudProcessor(
        PointCloudProcessorConfig(
            fixed_voxel_size=cfg.VOXEL_FIXED_SIZE,
            adaptive_voxel_max_size=cfg.VOXEL_ADAPTIVE_MAX_SIZE,
            adaptive_min_num_points=cfg.VOXEL_ADAPTIVE_MIN_POINTS,
            adaptive_num_iterations=cfg.VOXEL_ADAPTIVE_ITERS,
            enabled=cfg.VOXEL_FILTER_ENABLED if voxel_filter is None else bool(voxel_filter),
        )
    )

    pts_list: list[np.ndarray] = []
    for t in stamps:
        idx = int(np.argmin(np.abs(scan_stamps - t)))
        scan = all_scans[idx]
        pts_raw = ranges_to_points(
            scan["ranges"],
            profile.angle_min,
            profile.angle_inc,
            max(cfg.LIDAR_MIN_RANGE, profile.range_min),
            profile.range_max,
            stride=cfg.BEAM_STRIDE,
        )
        pts, _ = proc.process(pts_raw)
        pts_list.append(pts)

    return profile, pts_list


def default_map_size_m(dataset_name: str, matcher_type: str) -> float:
    configure_dataset(dataset_name)
    return cfg.MAP_SIZE_METERS


def pgo_output_stem(traj_path: str) -> str:
    return Path(traj_path).stem


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
