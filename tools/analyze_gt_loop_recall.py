#!/usr/bin/env python3
"""Analyze GT-valid loop-pair recall against loop candidate diagnostics."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PoseSample:
    timestamp: float
    translation: np.ndarray
    rotation: np.ndarray
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class KeyframeRecord:
    kf_id: int
    timestamp: float
    num_map_points: int | None
    frame_id: int | None = None
    estimated_pose: PoseSample | None = None
    source: str = "unknown"
    map_point_source: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--groundtruth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-time-gap-sec", type=float, default=10.0)
    parser.add_argument("--min-kf-gap", type=int, default=10)
    parser.add_argument("--loop-trans-threshold-m", type=float, default=0.75)
    parser.add_argument("--loop-rot-threshold-deg", type=float, default=45.0)
    parser.add_argument("--near-loop-trans-threshold-m", type=float, default=1.5)
    parser.add_argument("--gt-association-max-dt-sec", type=float, default=0.05)
    return parser.parse_args()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def _int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_list_cell(value: Any) -> list[int]:
    if value in {None, ""}:
        return []
    if isinstance(value, list):
        return [int(v) for v in value]
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    result: list[int] = []
    for item in parsed:
        item_int = _int(item)
        if item_int is not None:
            result.append(item_int)
    return result


def _pair_key(kf_a: int, kf_b: int) -> str:
    return f"{min(int(kf_a), int(kf_b))}-{max(int(kf_a), int(kf_b))}"


def _rotation_from_quaternion_xyzw(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError("Quaternion norm is zero")
    x, y, z, w = quat / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_angle_degrees(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    rel = np.asarray(rotation_a, dtype=np.float64).reshape(3, 3).T @ np.asarray(rotation_b, dtype=np.float64).reshape(3, 3)
    trace_value = max(-1.0, min(1.0, (float(np.trace(rel)) - 1.0) * 0.5))
    return float(math.degrees(math.acos(trace_value)))


def load_tum_groundtruth(path: Path) -> list[PoseSample]:
    poses: list[PoseSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(f"Invalid TUM pose line {line_number} in {path}: expected 8 columns, got {len(parts)}")
            timestamp, tx, ty, tz, qx, qy, qz, qw = [float(part) for part in parts]
            rotation = _rotation_from_quaternion_xyzw(qx, qy, qz, qw)
            translation = np.asarray([tx, ty, tz], dtype=np.float64)
            poses.append(
                PoseSample(
                    timestamp=timestamp,
                    translation=translation,
                    rotation=rotation,
                    quaternion_xyzw=(qx, qy, qz, qw),
                )
            )
    return poses


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _find_run_file(run_dir: Path, plain_name: str, summary_key: str | None = None, summary: dict[str, Any] | None = None) -> Path | None:
    direct = run_dir / plain_name
    if direct.exists():
        return direct
    if summary_key and summary:
        candidate = summary.get(summary_key)
        if candidate:
            candidate_path = Path(candidate)
            if candidate_path.exists():
                return candidate_path
    if summary_key and summary:
        standardized_key = f"standardized_{summary_key}"
        candidate = summary.get(standardized_key)
        if candidate:
            candidate_path = Path(candidate)
            if candidate_path.exists():
                return candidate_path
    matches = sorted(run_dir.glob(f"{Path(plain_name).stem}__*{Path(plain_name).suffix}"))
    return matches[-1] if matches else None


def _matrix_to_pose(matrix: Any, *, invert: bool = False) -> PoseSample | None:
    try:
        arr = np.asarray(matrix, dtype=np.float64)
    except Exception:
        return None
    if arr.shape != (4, 4):
        return None
    if invert:
        arr = np.linalg.inv(arr)
    rotation = arr[:3, :3]
    translation = arr[:3, 3]
    return PoseSample(
        timestamp=math.nan,
        translation=translation,
        rotation=rotation,
        quaternion_xyzw=(math.nan, math.nan, math.nan, math.nan),
    )


def load_keyframes_from_json(path: Path) -> list[KeyframeRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "keyframes" in payload and isinstance(payload["keyframes"], list):
            entries = payload["keyframes"]
        elif "data" in payload and isinstance(payload["data"], list):
            entries = payload["data"]
        else:
            entries = list(payload.values()) if all(isinstance(v, dict) for v in payload.values()) else []
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = []

    records: list[KeyframeRecord] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        kf_id = _int(entry.get("kid"))
        if kf_id is None:
            kf_id = _int(entry.get("kf_id"))
        if kf_id is None:
            kf_id = _int(entry.get("id"))
        if kf_id is None:
            continue
        timestamp = _float(entry.get("timestamp"))
        if timestamp is None:
            continue
        num_map_points = _int(entry.get("num_points"))
        if num_map_points is None:
            num_map_points = _int(entry.get("num_map_points"))
        pose = _matrix_to_pose(entry.get("Twc"))
        if pose is None:
            pose = _matrix_to_pose(entry.get("Tcw"), invert=True)
        if pose is not None:
            pose = PoseSample(
                timestamp=timestamp,
                translation=pose.translation,
                rotation=pose.rotation,
                quaternion_xyzw=pose.quaternion_xyzw,
            )
        records.append(
            KeyframeRecord(
                kf_id=int(kf_id),
                timestamp=float(timestamp),
                num_map_points=num_map_points,
                frame_id=_int(entry.get("frame_id")),
                estimated_pose=pose,
                source="keyframes_json",
                map_point_source="num_points" if num_map_points is not None else "",
            )
        )
    return sorted(records, key=lambda record: (record.kf_id, record.timestamp))


def reconstruct_keyframes_from_logs(run_dir: Path, summary: dict[str, Any]) -> list[KeyframeRecord]:
    frame_log_path = _find_run_file(run_dir, "frame_log_rgbd_dataset_freiburg1_room.csv", "frame_log_file", summary)
    keyframe_log_path = _find_run_file(run_dir, "keyframe_decision_log.csv", "keyframe_decision_log_file", summary)
    if frame_log_path is None or keyframe_log_path is None:
        missing = [str(path) for path in [frame_log_path, keyframe_log_path] if path is None]
        raise FileNotFoundError(f"Cannot reconstruct keyframes without frame log and keyframe decision log: {missing}")

    frame_rows = _csv_rows(frame_log_path)
    keyframe_rows = _csv_rows(keyframe_log_path)
    if not frame_rows:
        raise ValueError(f"Frame log is empty: {frame_log_path}")

    frame_by_id: dict[int, dict[str, str]] = {}
    for row in frame_rows:
        frame_id = _int(row.get("i"))
        if frame_id is None:
            frame_id = _int(row.get("frame_id"))
        if frame_id is None:
            continue
        frame_by_id[int(frame_id)] = row

    inserted_rows = [row for row in keyframe_rows if _bool(row.get("inserted"))]
    inserted_rows.sort(key=lambda row: (_int(row.get("frame_id")) or -1, _float(row.get("timestamp")) or -1.0))

    initial_row = frame_rows[0]
    initial_frame_id = _int(initial_row.get("i"))
    initial_timestamp = _float(initial_row.get("timestamp"))
    if initial_timestamp is None:
        raise ValueError(f"Initial frame log row has no timestamp: {frame_log_path}")

    records: list[KeyframeRecord] = [
        KeyframeRecord(
            kf_id=0,
            timestamp=float(initial_timestamp),
            num_map_points=_int(initial_row.get("last_tracked")) or _int(initial_row.get("points")),
            frame_id=initial_frame_id,
            estimated_pose=None,
            source="reconstructed_from_logs",
            map_point_source="frame_log.last_tracked_or_points",
        )
    ]

    seen_kf_ids = {0}
    for row in inserted_rows:
        last_keyframe_id = _int(row.get("last_keyframe_id"))
        num_keyframes_before_insert = _int(row.get("num_keyframes"))
        kf_id = None
        if last_keyframe_id is not None:
            kf_id = last_keyframe_id + 1
        elif num_keyframes_before_insert is not None:
            kf_id = num_keyframes_before_insert
        if kf_id is None or kf_id in seen_kf_ids:
            continue
        frame_id = _int(row.get("frame_id"))
        frame_row = frame_by_id.get(int(frame_id)) if frame_id is not None else None
        timestamp = _float(row.get("timestamp"))
        if timestamp is None and frame_row is not None:
            timestamp = _float(frame_row.get("timestamp"))
        if timestamp is None:
            raise ValueError(f"Inserted keyframe row has no timestamp: {row}")
        num_map_points = _int(row.get("num_matched_cur"))
        map_point_source = "keyframe_decision_log.num_matched_cur"
        if num_map_points is None and frame_row is not None:
            num_map_points = _int(frame_row.get("last_tracked")) or _int(frame_row.get("points"))
            map_point_source = "frame_log.last_tracked_or_points"
        records.append(
            KeyframeRecord(
                kf_id=int(kf_id),
                timestamp=float(timestamp),
                num_map_points=num_map_points,
                frame_id=frame_id,
                estimated_pose=None,
                source="reconstructed_from_logs",
                map_point_source=map_point_source,
            )
        )
        seen_kf_ids.add(int(kf_id))

    expected_keyframes = _int(summary.get("final_keyframes")) or _int(summary.get("keyframes"))
    if expected_keyframes is not None and len(records) != expected_keyframes:
        raise ValueError(
            f"Reconstructed keyframe count mismatch: recovered {len(records)} from logs, expected {expected_keyframes}"
        )
    return sorted(records, key=lambda record: (record.kf_id, record.timestamp))


def load_run_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        matches = sorted(run_dir.glob("run_summary__*.json"))
        if matches:
            summary_path = matches[-1]
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary in {run_dir}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_estimated_trajectory(run_dir: Path, summary: dict[str, Any]) -> list[PoseSample]:
    trajectory_path = _find_run_file(run_dir, "trajectory_rgbd_dataset_freiburg1_room.txt", "trajectory_file", summary)
    if trajectory_path is None or not trajectory_path.exists():
        return []
    return load_tum_groundtruth(trajectory_path)


def nearest_pose(poses: list[PoseSample], timestamp: float, max_dt: float) -> tuple[PoseSample | None, float | None]:
    if not poses:
        return None, None
    timestamps = np.asarray([pose.timestamp for pose in poses], dtype=np.float64)
    idx = int(np.argmin(np.abs(timestamps - float(timestamp))))
    pose = poses[idx]
    dt = abs(float(pose.timestamp) - float(timestamp))
    if dt > max_dt:
        return None, dt
    return pose, dt


def attach_estimated_poses(
    keyframes: list[KeyframeRecord],
    estimated_trajectory: list[PoseSample],
    max_dt: float,
) -> list[KeyframeRecord]:
    attached: list[KeyframeRecord] = []
    for record in keyframes:
        pose, _ = nearest_pose(estimated_trajectory, record.timestamp, max_dt=max_dt)
        attached.append(
            KeyframeRecord(
                kf_id=record.kf_id,
                timestamp=record.timestamp,
                num_map_points=record.num_map_points,
                frame_id=record.frame_id,
                estimated_pose=pose,
                source=record.source,
                map_point_source=record.map_point_source,
            )
        )
    return attached


def load_keyframes(run_dir: Path, summary: dict[str, Any], estimated_trajectory: list[PoseSample], max_dt: float) -> tuple[list[KeyframeRecord], dict[str, Any]]:
    keyframes_path = _find_run_file(run_dir, "keyframes.json", "keyframes_json", summary)
    metadata = {"keyframe_input_source": "", "keyframes_json_path": str(keyframes_path) if keyframes_path else None}
    if keyframes_path and keyframes_path.exists():
        keyframes = load_keyframes_from_json(keyframes_path)
        metadata["keyframe_input_source"] = "keyframes_json"
    else:
        keyframes = reconstruct_keyframes_from_logs(run_dir, summary)
        metadata["keyframe_input_source"] = "reconstructed_from_logs"
    keyframes = attach_estimated_poses(keyframes, estimated_trajectory, max_dt=max_dt)
    return keyframes, metadata


def associate_keyframes_to_gt(
    keyframes: list[KeyframeRecord],
    gt_poses: list[PoseSample],
    *,
    max_dt: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in keyframes:
        gt_pose, dt = nearest_pose(gt_poses, record.timestamp, max_dt=max_dt)
        row: dict[str, Any] = {
            "kf_id": record.kf_id,
            "kf_timestamp": record.timestamp,
            "gt_timestamp": gt_pose.timestamp if gt_pose is not None else "",
            "dt_sec": dt if dt is not None else "",
            "gt_available": gt_pose is not None,
            "gt_tx": gt_pose.translation[0] if gt_pose is not None else "",
            "gt_ty": gt_pose.translation[1] if gt_pose is not None else "",
            "gt_tz": gt_pose.translation[2] if gt_pose is not None else "",
            "gt_qx": gt_pose.quaternion_xyzw[0] if gt_pose is not None else "",
            "gt_qy": gt_pose.quaternion_xyzw[1] if gt_pose is not None else "",
            "gt_qz": gt_pose.quaternion_xyzw[2] if gt_pose is not None else "",
            "gt_qw": gt_pose.quaternion_xyzw[3] if gt_pose is not None else "",
        }
        rows.append(row)
    return rows


def _pair_metrics(record_a: KeyframeRecord, record_b: KeyframeRecord, pose_a: PoseSample, pose_b: PoseSample) -> tuple[float, float]:
    translation = float(np.linalg.norm(pose_b.translation - pose_a.translation))
    rotation_deg = rotation_angle_degrees(pose_a.rotation, pose_b.rotation)
    return translation, rotation_deg


def generate_gt_pairs(
    keyframes: list[KeyframeRecord],
    gt_associations: list[dict[str, Any]],
    *,
    min_time_gap_sec: float,
    min_kf_gap: int,
    loop_trans_threshold_m: float,
    loop_rot_threshold_deg: float,
    near_loop_trans_threshold_m: float,
) -> list[dict[str, Any]]:
    gt_by_kf_id: dict[int, PoseSample] = {}
    for association in gt_associations:
        if _bool(association.get("gt_available")):
            qx = float(association["gt_qx"])
            qy = float(association["gt_qy"])
            qz = float(association["gt_qz"])
            qw = float(association["gt_qw"])
            gt_by_kf_id[int(association["kf_id"])] = PoseSample(
                timestamp=float(association["gt_timestamp"]),
                translation=np.asarray(
                    [float(association["gt_tx"]), float(association["gt_ty"]), float(association["gt_tz"])],
                    dtype=np.float64,
                ),
                rotation=_rotation_from_quaternion_xyzw(qx, qy, qz, qw),
                quaternion_xyzw=(qx, qy, qz, qw),
            )

    pairs: list[dict[str, Any]] = []
    ordered = sorted(keyframes, key=lambda record: (record.timestamp, record.kf_id))
    for idx_i, record_i in enumerate(ordered):
        for record_j in ordered[idx_i + 1 :]:
            time_gap_sec = abs(float(record_j.timestamp) - float(record_i.timestamp))
            kf_gap = abs(int(record_j.kf_id) - int(record_i.kf_id))
            if time_gap_sec < min_time_gap_sec or kf_gap < min_kf_gap:
                continue
            gt_pose_i = gt_by_kf_id.get(record_i.kf_id)
            gt_pose_j = gt_by_kf_id.get(record_j.kf_id)
            gt_translation_distance = ""
            gt_rotation_angle_deg = ""
            gt_loop_like = False
            gt_near_loop = False
            if gt_pose_i is not None and gt_pose_j is not None:
                gt_translation_distance, gt_rotation_angle_deg = _pair_metrics(record_i, record_j, gt_pose_i, gt_pose_j)
                gt_loop_like = bool(
                    gt_translation_distance <= loop_trans_threshold_m and gt_rotation_angle_deg <= loop_rot_threshold_deg
                )
                gt_near_loop = bool(gt_translation_distance <= near_loop_trans_threshold_m)

            estimated_translation_distance = ""
            estimated_rotation_angle_deg = ""
            if record_i.estimated_pose is not None and record_j.estimated_pose is not None:
                estimated_translation_distance, estimated_rotation_angle_deg = _pair_metrics(
                    record_i,
                    record_j,
                    record_i.estimated_pose,
                    record_j.estimated_pose,
                )

            pairs.append(
                {
                    "pair_key": _pair_key(record_i.kf_id, record_j.kf_id),
                    "kf_i": record_i.kf_id,
                    "kf_j": record_j.kf_id,
                    "timestamp_i": record_i.timestamp,
                    "timestamp_j": record_j.timestamp,
                    "time_gap_sec": time_gap_sec,
                    "kf_id_gap": kf_gap,
                    "gt_available_i": gt_pose_i is not None,
                    "gt_available_j": gt_pose_j is not None,
                    "gt_translation_distance": gt_translation_distance,
                    "gt_rotation_angle_deg": gt_rotation_angle_deg,
                    "estimated_translation_distance": estimated_translation_distance,
                    "estimated_rotation_angle_deg": estimated_rotation_angle_deg,
                    "gt_loop_like": gt_loop_like,
                    "gt_near_loop": gt_near_loop,
                    "num_map_points_i": record_i.num_map_points if record_i.num_map_points is not None else "",
                    "num_map_points_j": record_j.num_map_points if record_j.num_map_points is not None else "",
                }
            )
    return pairs


def load_loop_candidate_oracle(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "loop_candidate_oracle.csv"
    if not path.exists():
        matches = sorted(run_dir.glob("loop_candidate_oracle__*.csv"))
        if matches:
            path = matches[-1]
    if not path.exists():
        return {}
    rows = _csv_rows(path)
    by_pair: dict[str, dict[str, Any]] = {}
    for row in rows:
        current_kf_id = _int(row.get("current_kf_id"))
        candidate_kf_id = _int(row.get("candidate_kf_id"))
        if current_kf_id is None or candidate_kf_id is None:
            continue
        key = _pair_key(current_kf_id, candidate_kf_id)
        existing = by_pair.get(key)
        payload = dict(row)
        payload["pair_key"] = key
        if existing is None:
            by_pair[key] = payload
            continue
        accepted_score = int(_bool(payload.get("accepted")))
        existing_score = int(_bool(existing.get("accepted")))
        payload_final = _int(payload.get("final_matched_map_points")) or 0
        existing_final = _int(existing.get("final_matched_map_points")) or 0
        payload_rank = _int(payload.get("candidate_rank")) or 999999
        existing_rank = _int(existing.get("candidate_rank")) or 999999
        if (accepted_score, payload_final, -payload_rank) > (existing_score, existing_final, -existing_rank):
            by_pair[key] = payload
    return by_pair


def load_source_comparison(run_dir: Path) -> dict[int, dict[str, Any]]:
    path = run_dir / "loop_candidate_source_comparison.csv"
    if not path.exists():
        matches = sorted(run_dir.glob("loop_candidate_source_comparison__*.csv"))
        if matches:
            path = matches[-1]
    if not path.exists():
        return {}
    result: dict[int, dict[str, Any]] = {}
    for row in _csv_rows(path):
        kf_id = _int(row.get("kf_id"))
        if kf_id is None:
            continue
        result[int(kf_id)] = {
            "kf_id": int(kf_id),
            "timestamp": _float(row.get("timestamp")),
            "candidate_source": row.get("candidate_source", ""),
            "dbow3_candidates": _parse_list_cell(row.get("dbow3_candidates")),
            "inverted_file_candidates": _parse_list_cell(row.get("inverted_file_candidates")),
            "intersection_candidates": _parse_list_cell(row.get("intersection_candidates")),
            "dbow3_only_candidates": _parse_list_cell(row.get("dbow3_only_candidates")),
            "inverted_only_candidates": _parse_list_cell(row.get("inverted_only_candidates")),
            "chosen_candidates": _parse_list_cell(row.get("chosen_candidates")),
        }
    return result


def load_loop_retrieval_profile(run_dir: Path) -> dict[int, dict[str, Any]]:
    path = run_dir / "loop_retrieval_profile.csv"
    if not path.exists():
        matches = sorted(run_dir.glob("loop_retrieval_profile__*.csv"))
        if matches:
            path = matches[-1]
    if not path.exists():
        return {}
    result: dict[int, dict[str, Any]] = {}
    for row in _csv_rows(path):
        kf_id = _int(row.get("kf_id"))
        if kf_id is None:
            continue
        result[int(kf_id)] = dict(row)
    return result


def load_density_profile(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "loop_keyframe_density_profile.csv"
    if not path.exists():
        matches = sorted(run_dir.glob("loop_keyframe_density_profile__*.csv"))
        if matches:
            path = matches[-1]
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in _csv_rows(path):
        current_kf_id = _int(row.get("current_kf_id"))
        candidate_kf_id = _int(row.get("candidate_kf_id"))
        if current_kf_id is None or candidate_kf_id is None:
            continue
        result[_pair_key(current_kf_id, candidate_kf_id)] = dict(row)
    return result


def classify_pipeline_stage(
    oracle_row: dict[str, Any] | None,
    source_row: dict[str, Any] | None,
    pair: dict[str, Any],
) -> str:
    if oracle_row is not None:
        if _bool(oracle_row.get("accepted")):
            return "ACCEPTED"
        rejection_reason = str(oracle_row.get("rejection_reason", "") or "").lower()
        rejection_stage = str(oracle_row.get("rejection_stage", "") or "").lower()
        if (
            "final support" in rejection_reason
            or "final matched map points" in rejection_reason
            or "matched map points after covisibility expansion" in rejection_reason
            or "covisibility expansion" in rejection_reason
        ):
            return "FAILED_FINAL_SUPPORT"
        if "refined inliers" in rejection_reason or "guided final matches" in rejection_reason:
            return "FAILED_REFINED_GEOMETRY"
        if "pose distance" in rejection_reason or "estimated pose distance" in rejection_reason:
            return "FAILED_POSE_DISTANCE_GATE"
        if "seed inliers" in rejection_reason or "ransac seed" in rejection_reason:
            return "FAILED_SEED_GEOMETRY"
        if "too few loop geometry matches" in rejection_reason or "geometry matches" in rejection_reason:
            return "FAILED_GEOMETRY_MATCHES"
        if "consistency" in rejection_reason or rejection_stage == "consistency":
            return "FAILED_CONSISTENCY"
        if rejection_stage == "geometry":
            seed_inliers = _int(oracle_row.get("seed_inliers")) or 0
            refined_inliers = _int(oracle_row.get("refined_inliers")) or 0
            guided_matches = _int(oracle_row.get("guided_projection_matches")) or 0
            final_matches = _int(oracle_row.get("final_matched_map_points")) or 0
            if final_matches > 0 and final_matches < 20:
                return "FAILED_FINAL_SUPPORT"
            if guided_matches > 0:
                return "FAILED_REFINED_GEOMETRY"
            if refined_inliers > seed_inliers and seed_inliers > 0:
                return "FAILED_REFINED_GEOMETRY"
            if seed_inliers > 0:
                return "FAILED_SEED_GEOMETRY"
            return "FAILED_GEOMETRY_MATCHES"
        return "UNKNOWN_STAGE"

    current_kf_id = int(pair["kf_j"])
    candidate_kf_id = int(pair["kf_i"])
    for source_candidate_owner in (current_kf_id, candidate_kf_id):
        source = source_row if source_row and int(source_row.get("kf_id", -1)) == source_candidate_owner else None
        if source is None:
            continue
        candidate_other = candidate_kf_id if source_candidate_owner == current_kf_id else current_kf_id
        union_candidates = set(source.get("dbow3_candidates", [])) | set(source.get("inverted_file_candidates", []))
        if candidate_other in union_candidates:
            chosen_candidates = set(source.get("chosen_candidates", []))
            if candidate_other in chosen_candidates:
                return "RETRIEVED_RAW_ONLY"
            return "FAILED_SCORE_OR_COMMON_WORD_FILTER"
    return "NOT_RETRIEVED"


def classify_gt_pairs(
    gt_pairs: list[dict[str, Any]],
    oracle_by_pair: dict[str, dict[str, Any]],
    source_by_kf: dict[int, dict[str, Any]],
    density_by_pair: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in gt_pairs:
        pair_key = str(pair["pair_key"])
        oracle_row = oracle_by_pair.get(pair_key)
        source_row = source_by_kf.get(int(pair["kf_j"])) or source_by_kf.get(int(pair["kf_i"]))
        density_row = density_by_pair.get(pair_key, {})
        stage = classify_pipeline_stage(oracle_row, source_row, pair)
        merged = dict(pair)
        merged.update(
            {
                "pipeline_stage": stage,
                "actual_candidate_seen": oracle_row is not None,
                "candidate_source": (oracle_row or {}).get("candidate_source", (source_row or {}).get("candidate_source", "")),
                "rejection_stage": (oracle_row or {}).get("rejection_stage", ""),
                "rejection_reason": (oracle_row or {}).get("rejection_reason", ""),
                "accepted": _bool((oracle_row or {}).get("accepted")),
                "bow_score": (oracle_row or {}).get("bow_score", ""),
                "common_words": (oracle_row or {}).get("common_words", ""),
                "accumulated_score": (oracle_row or {}).get("accumulated_score", ""),
                "consistency_score": (oracle_row or {}).get("consistency_score", ""),
                "raw_bow_matches": (oracle_row or {}).get("raw_bow_matches", ""),
                "valid_bow_map_point_matches": (oracle_row or {}).get("valid_bow_map_point_matches", ""),
                "seed_inliers": (oracle_row or {}).get("seed_inliers", ""),
                "refined_inliers": (oracle_row or {}).get("refined_inliers", ""),
                "guided_projection_matches": (oracle_row or {}).get("guided_projection_matches", ""),
                "final_matched_map_points": (oracle_row or {}).get("final_matched_map_points", density_row.get("final_matched_map_points", "")),
                "candidate_group_size": density_row.get("candidate_group_size", ""),
                "current_neighbor_count": density_row.get("current_neighbor_count", ""),
                "candidate_neighbor_count": density_row.get("candidate_neighbor_count", ""),
                "current_local_map_points": density_row.get("current_local_map_points", ""),
                "candidate_local_map_points": density_row.get("candidate_local_map_points", ""),
            }
        )
        rows.append(merged)
    return rows


def build_recall_by_stage(rows: list[dict[str, Any]], *, predicate_key: str) -> list[dict[str, Any]]:
    stage_order = [
        "GT_LOOP_LIKE_TOTAL" if predicate_key == "gt_loop_like" else "GT_NEAR_LOOP_TOTAL",
        "NOT_RETRIEVED",
        "RETRIEVED_RAW_ONLY",
        "FAILED_SCORE_OR_COMMON_WORD_FILTER",
        "FAILED_CONSISTENCY",
        "FAILED_SEED_GEOMETRY",
        "FAILED_GEOMETRY_MATCHES",
        "FAILED_POSE_DISTANCE_GATE",
        "FAILED_REFINED_GEOMETRY",
        "FAILED_FINAL_SUPPORT",
        "ACCEPTED",
        "UNKNOWN_STAGE",
    ]
    selected = [row for row in rows if _bool(row.get(predicate_key))]
    total = len(selected)
    counts = {stage: 0 for stage in stage_order}
    counts[stage_order[0]] = total
    for row in selected:
        counts[row["pipeline_stage"]] = counts.get(row["pipeline_stage"], 0) + 1
    output: list[dict[str, Any]] = []
    for stage in stage_order:
        count = counts.get(stage, 0)
        recall = 100.0 if stage == stage_order[0] and total > 0 else 0.0
        if stage != stage_order[0]:
            recall = (100.0 * count / total) if total > 0 else 0.0
        output.append({"stage": stage, "count": count, "recall_vs_gt_loop_like_percent": recall})
    return output


def dominant_loss_stage(rows: list[dict[str, Any]], *, predicate_key: str) -> str:
    selected = [row for row in rows if _bool(row.get(predicate_key))]
    counts: dict[str, int] = {}
    for row in selected:
        stage = row["pipeline_stage"]
        if stage == "ACCEPTED":
            continue
        counts[stage] = counts.get(stage, 0) + 1
    if not counts:
        return ""
    return max(sorted(counts.items()), key=lambda item: item[1])[0]


def suggest_next_focus(
    rows: list[dict[str, Any]],
    *,
    predicate_key: str,
) -> str:
    dominant = dominant_loss_stage(rows, predicate_key=predicate_key)
    density_concerns = sum(1 for row in rows if _bool(row.get(predicate_key)) and _bool(row.get("density_concern")))
    total = sum(1 for row in rows if _bool(row.get(predicate_key)))
    density_ratio = (density_concerns / total) if total > 0 else 0.0
    if dominant in {"NOT_RETRIEVED", "FAILED_SCORE_OR_COMMON_WORD_FILTER", "RETRIEVED_RAW_ONLY"}:
        if density_ratio >= 0.35:
            return "Checkpoint 2.35C: keyframe density and retrieval coverage audit for true fr1_room loops."
        return "Checkpoint 2.35C: loop candidate retrieval audit focused on true GT loop coverage before consistency."
    if dominant == "FAILED_CONSISTENCY":
        return "Checkpoint 2.35C: loop consistency-gate recall audit on true GT loop pairs."
    if dominant in {"FAILED_SEED_GEOMETRY", "FAILED_GEOMETRY_MATCHES", "FAILED_POSE_DISTANCE_GATE", "FAILED_REFINED_GEOMETRY"}:
        return "Checkpoint 2.35C: SE3 loop geometry verification audit on GT-positive candidate pairs."
    if dominant == "FAILED_FINAL_SUPPORT":
        return "Checkpoint 2.35C: projection-expansion and final support audit for GT-positive loop pairs."
    if density_ratio >= 0.35:
        return "Checkpoint 2.35C: sparse keyframe density audit for GT-positive fr1_room loop opportunities."
    return "Checkpoint 2.35C: targeted true-loop failure audit using the dominant rejection stage from this report."


def build_density_support_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        min_pair_map_points_candidates = [
            value
            for value in [_int(row.get("num_map_points_i")), _int(row.get("num_map_points_j"))]
            if value is not None
        ]
        min_pair_map_points = min(min_pair_map_points_candidates) if min_pair_map_points_candidates else None
        candidate_group_size = _int(row.get("candidate_group_size"))
        final_matched_map_points = _int(row.get("final_matched_map_points"))
        density_concern = bool(
            (min_pair_map_points is not None and min_pair_map_points < 50)
            or (candidate_group_size is not None and candidate_group_size <= 2)
            or (final_matched_map_points is not None and 0 < final_matched_map_points < 20)
        )
        output.append(
            {
                "pair_key": row["pair_key"],
                "kf_i": row["kf_i"],
                "kf_j": row["kf_j"],
                "gt_translation_distance": row["gt_translation_distance"],
                "gt_rotation_angle_deg": row["gt_rotation_angle_deg"],
                "pipeline_stage": row["pipeline_stage"],
                "num_map_points_i": row["num_map_points_i"],
                "num_map_points_j": row["num_map_points_j"],
                "min_pair_map_points": min_pair_map_points if min_pair_map_points is not None else "",
                "time_gap_sec": row["time_gap_sec"],
                "kf_id_gap": row["kf_id_gap"],
                "candidate_group_size": row.get("candidate_group_size", ""),
                "final_matched_map_points": row.get("final_matched_map_points", ""),
                "density_concern": density_concern,
            }
        )
    return output


def annotate_density(rows: list[dict[str, Any]], density_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    density_by_pair = {str(row["pair_key"]): row for row in density_rows}
    annotated: list[dict[str, Any]] = []
    for row in rows:
        density = density_by_pair.get(str(row["pair_key"]), {})
        merged = dict(row)
        merged["density_concern"] = density.get("density_concern", False)
        annotated.append(merged)
    return annotated


def _suggested_investigation(stage: str) -> str:
    mapping = {
        "NOT_RETRIEVED": "Check whether the run ever inserted enough spatially separated keyframes near this loop region.",
        "RETRIEVED_RAW_ONLY": "Inspect source-comparison logs to confirm whether an alternate retrieval source surfaced this pair.",
        "FAILED_SCORE_OR_COMMON_WORD_FILTER": "Audit common-word, min-score, and accumulation retention for this current keyframe.",
        "FAILED_CONSISTENCY": "Inspect consistency-group overlap and temporal persistence for this true loop candidate.",
        "FAILED_SEED_GEOMETRY": "Audit seed correspondences and SE3 RANSAC inlier support for this true loop.",
        "FAILED_GEOMETRY_MATCHES": "Check whether BoW-guided correspondences are too sparse before SE3 verification.",
        "FAILED_POSE_DISTANCE_GATE": "Audit the estimated pose-distance gate against GT-positive candidates.",
        "FAILED_REFINED_GEOMETRY": "Inspect refined inliers and guided projection expansion after the seed stage.",
        "FAILED_FINAL_SUPPORT": "Inspect final matched map point support relative to the acceptance threshold.",
        "UNKNOWN_STAGE": "Inspect the raw oracle record and loop debug candidate report for this pair.",
    }
    return mapping.get(stage, "")


def build_top_missed_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missed = [row for row in rows if _bool(row.get("gt_loop_like")) and row["pipeline_stage"] != "ACCEPTED"]
    missed.sort(
        key=lambda row: (
            float(row["gt_translation_distance"]) if row["gt_translation_distance"] not in {"", None} else math.inf,
            float(row["gt_rotation_angle_deg"]) if row["gt_rotation_angle_deg"] not in {"", None} else math.inf,
            int(row["kf_i"]),
            int(row["kf_j"]),
        )
    )
    output: list[dict[str, Any]] = []
    for row in missed:
        output.append(
            {
                "pair_key": row["pair_key"],
                "kf_i": row["kf_i"],
                "kf_j": row["kf_j"],
                "gt_translation_distance": row["gt_translation_distance"],
                "gt_rotation_angle_deg": row["gt_rotation_angle_deg"],
                "pipeline_stage": row["pipeline_stage"],
                "rejection_reason": row["rejection_reason"],
                "bow_score": row["bow_score"],
                "common_words": row["common_words"],
                "accumulated_score": row["accumulated_score"],
                "consistency_score": row["consistency_score"],
                "seed_inliers": row["seed_inliers"],
                "final_matched_map_points": row["final_matched_map_points"],
                "num_map_points_i": row["num_map_points_i"],
                "num_map_points_j": row["num_map_points_j"],
                "suggested_investigation": _suggested_investigation(str(row["pipeline_stage"])),
            }
        )
    return output


def build_summary_json(
    *,
    keyframes: list[KeyframeRecord],
    gt_associations: list[dict[str, Any]],
    gt_pairs: list[dict[str, Any]],
    classified_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    gt_loop_like_rows = [row for row in classified_rows if _bool(row.get("gt_loop_like"))]
    gt_near_loop_rows = [row for row in classified_rows if _bool(row.get("gt_near_loop"))]
    gt_loop_like_stage_counts: dict[str, int] = {}
    gt_near_loop_stage_counts: dict[str, int] = {}
    for row in gt_loop_like_rows:
        gt_loop_like_stage_counts[row["pipeline_stage"]] = gt_loop_like_stage_counts.get(row["pipeline_stage"], 0) + 1
    for row in gt_near_loop_rows:
        gt_near_loop_stage_counts[row["pipeline_stage"]] = gt_near_loop_stage_counts.get(row["pipeline_stage"], 0) + 1
    return {
        "num_keyframes": len(keyframes),
        "num_gt_associated_keyframes": sum(1 for row in gt_associations if _bool(row.get("gt_available"))),
        "num_all_temporally_valid_pairs": len(gt_pairs),
        "num_gt_loop_like_pairs": len(gt_loop_like_rows),
        "num_gt_near_loop_pairs": len(gt_near_loop_rows),
        "num_accepted_gt_loop_like_pairs": sum(1 for row in gt_loop_like_rows if row["pipeline_stage"] == "ACCEPTED"),
        "gt_loop_like_stage_counts": gt_loop_like_stage_counts,
        "gt_near_loop_stage_counts": gt_near_loop_stage_counts,
        "dominant_gt_loop_loss_stage": dominant_loss_stage(classified_rows, predicate_key="gt_loop_like"),
        "suggested_next_focus": suggest_next_focus(classified_rows, predicate_key="gt_loop_like"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body]) if body else "\n".join([header, sep])


def write_report(
    *,
    report_path: Path,
    run_dir: Path,
    groundtruth: Path,
    keyframes: list[KeyframeRecord],
    gt_associations: list[dict[str, Any]],
    summary: dict[str, Any],
    recall_rows: list[dict[str, Any]],
    near_recall_rows: list[dict[str, Any]],
    top_missed_rows: list[dict[str, Any]],
    accepted_rows: list[dict[str, Any]],
    classified_rows: list[dict[str, Any]],
    config: dict[str, Any],
    keyframe_metadata: dict[str, Any],
) -> None:
    gt_loop_total = summary["num_gt_loop_like_pairs"]
    near_loop_total = summary["num_gt_near_loop_pairs"]
    dominant_stage = summary["dominant_gt_loop_loss_stage"] or "none"
    density_concern_count = sum(1 for row in classified_rows if _bool(row.get("gt_loop_like")) and _bool(row.get("density_concern")))
    report_lines = [
        "# GT Loop Recall Analysis Report",
        "",
        "## 1. Input run directory",
        f"- `{run_dir}`",
        "",
        "## 2. Dataset groundtruth path",
        f"- `{groundtruth}`",
        "",
        "## 3. Number of keyframes and GT associations",
        f"- Keyframes: `{len(keyframes)}`",
        f"- GT-associated keyframes: `{summary['num_gt_associated_keyframes']}`",
        f"- Keyframe source: `{keyframe_metadata.get('keyframe_input_source', '')}`",
        f"- `keyframes.json` path: `{keyframe_metadata.get('keyframes_json_path')}`",
        "",
        "## 4. GT-loop definition thresholds",
        f"- `min_time_gap_sec = {config['min_time_gap_sec']}`",
        f"- `min_kf_gap = {config['min_kf_gap']}`",
        f"- `loop_trans_threshold_m = {config['loop_trans_threshold_m']}`",
        f"- `loop_rot_threshold_deg = {config['loop_rot_threshold_deg']}`",
        f"- `near_loop_trans_threshold_m = {config['near_loop_trans_threshold_m']}`",
        f"- `gt_association_max_dt_sec = {config['gt_association_max_dt_sec']}`",
        "",
        "## 5. Total GT-loop-like pairs",
        f"- `{gt_loop_total}`",
        "",
        "## 6. Total GT-near-loop pairs",
        f"- `{near_loop_total}`",
        "",
        "## 7. Recall-by-stage table",
        markdown_table(recall_rows, ["stage", "count", "recall_vs_gt_loop_like_percent"]),
        "",
        "### GT-near-loop recall",
        markdown_table(near_recall_rows, ["stage", "count", "recall_vs_gt_loop_like_percent"]),
        "",
        "## 8. Top missed GT-loop-like pairs",
        markdown_table(top_missed_rows[:10], ["pair_key", "kf_i", "kf_j", "gt_translation_distance", "gt_rotation_angle_deg", "pipeline_stage", "rejection_reason", "suggested_investigation"]),
        "",
        "## 9. Accepted GT-loop-like pairs",
        markdown_table(
            accepted_rows[:10],
            ["pair_key", "kf_i", "kf_j", "gt_translation_distance", "gt_rotation_angle_deg", "final_matched_map_points"],
        ),
        "",
        "## 10. Where most GT loops are lost",
        f"- Dominant GT-loop loss stage: `{dominant_stage}`",
        "",
        "## 11. Sparse-keyframe density interpretation",
        f"- GT-loop-like pairs flagged with density concern: `{density_concern_count}` / `{gt_loop_total}`",
        "",
        "## 12. Main suspected root cause after this analysis",
        f"- `{summary['suggested_next_focus']}`",
        "",
        "## 13. Recommended next checkpoint",
        f"- `{summary['suggested_next_focus']}`",
        "",
        "## Direct answers",
        f"- Are correct GT loop pairs present in the candidate list? `{summary['gt_loop_like_stage_counts'].get('ACCEPTED', 0) > 0 or summary['gt_loop_like_stage_counts'].get('FAILED_CONSISTENCY', 0) > 0 or summary['gt_loop_like_stage_counts'].get('FAILED_SEED_GEOMETRY', 0) > 0 or summary['gt_loop_like_stage_counts'].get('FAILED_GEOMETRY_MATCHES', 0) > 0 or summary['gt_loop_like_stage_counts'].get('FAILED_POSE_DISTANCE_GATE', 0) > 0 or summary['gt_loop_like_stage_counts'].get('FAILED_REFINED_GEOMETRY', 0) > 0 or summary['gt_loop_like_stage_counts'].get('FAILED_FINAL_SUPPORT', 0) > 0}`",
        f"- If not, is the issue likely candidate retrieval or sparse keyframe density? `dominant={dominant_stage}`, `density_concern_pairs={density_concern_count}`",
        f"- If yes, where are they getting rejected? `stage_counts={json.dumps(summary['gt_loop_like_stage_counts'], sort_keys=True)}`",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    groundtruth = args.groundtruth.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_run_summary(run_dir)
    gt_poses = load_tum_groundtruth(groundtruth)
    estimated_trajectory = load_estimated_trajectory(run_dir, summary)
    keyframes, keyframe_metadata = load_keyframes(
        run_dir,
        summary,
        estimated_trajectory,
        max_dt=max(0.05, float(args.gt_association_max_dt_sec)),
    )

    gt_associations = associate_keyframes_to_gt(
        keyframes,
        gt_poses,
        max_dt=float(args.gt_association_max_dt_sec),
    )
    gt_pairs = generate_gt_pairs(
        keyframes,
        gt_associations,
        min_time_gap_sec=float(args.min_time_gap_sec),
        min_kf_gap=int(args.min_kf_gap),
        loop_trans_threshold_m=float(args.loop_trans_threshold_m),
        loop_rot_threshold_deg=float(args.loop_rot_threshold_deg),
        near_loop_trans_threshold_m=float(args.near_loop_trans_threshold_m),
    )

    oracle_by_pair = load_loop_candidate_oracle(run_dir)
    source_by_kf = load_source_comparison(run_dir)
    retrieval_by_kf = load_loop_retrieval_profile(run_dir)
    density_by_pair = load_density_profile(run_dir)
    classified_rows = classify_gt_pairs(gt_pairs, oracle_by_pair, source_by_kf, density_by_pair)
    density_rows = build_density_support_analysis(classified_rows)
    classified_rows = annotate_density(classified_rows, density_rows)

    recall_rows = build_recall_by_stage(classified_rows, predicate_key="gt_loop_like")
    near_recall_rows = build_recall_by_stage(classified_rows, predicate_key="gt_near_loop")
    top_missed_rows = build_top_missed_pairs(classified_rows)
    accepted_rows = [
        row
        for row in classified_rows
        if _bool(row.get("gt_loop_like")) and row["pipeline_stage"] == "ACCEPTED"
    ]
    accepted_rows.sort(
        key=lambda row: (
            float(row["gt_translation_distance"]) if row["gt_translation_distance"] not in {"", None} else math.inf,
            float(row["gt_rotation_angle_deg"]) if row["gt_rotation_angle_deg"] not in {"", None} else math.inf,
        )
    )
    summary_json = build_summary_json(
        keyframes=keyframes,
        gt_associations=gt_associations,
        gt_pairs=gt_pairs,
        classified_rows=classified_rows,
    )

    write_csv(
        output_dir / "keyframe_gt_associations.csv",
        gt_associations,
        [
            "kf_id",
            "kf_timestamp",
            "gt_timestamp",
            "dt_sec",
            "gt_available",
            "gt_tx",
            "gt_ty",
            "gt_tz",
            "gt_qx",
            "gt_qy",
            "gt_qz",
            "gt_qw",
        ],
    )
    write_csv(
        output_dir / "gt_loop_pairs_all.csv",
        gt_pairs,
        [
            "pair_key",
            "kf_i",
            "kf_j",
            "timestamp_i",
            "timestamp_j",
            "time_gap_sec",
            "kf_id_gap",
            "gt_available_i",
            "gt_available_j",
            "gt_translation_distance",
            "gt_rotation_angle_deg",
            "estimated_translation_distance",
            "estimated_rotation_angle_deg",
            "gt_loop_like",
            "gt_near_loop",
            "num_map_points_i",
            "num_map_points_j",
        ],
    )
    write_csv(
        output_dir / "gt_loop_pairs_classified.csv",
        classified_rows,
        [
            "pair_key",
            "kf_i",
            "kf_j",
            "timestamp_i",
            "timestamp_j",
            "time_gap_sec",
            "kf_id_gap",
            "gt_available_i",
            "gt_available_j",
            "gt_translation_distance",
            "gt_rotation_angle_deg",
            "estimated_translation_distance",
            "estimated_rotation_angle_deg",
            "gt_loop_like",
            "gt_near_loop",
            "num_map_points_i",
            "num_map_points_j",
            "pipeline_stage",
            "actual_candidate_seen",
            "candidate_source",
            "rejection_stage",
            "rejection_reason",
            "accepted",
            "bow_score",
            "common_words",
            "accumulated_score",
            "consistency_score",
            "raw_bow_matches",
            "valid_bow_map_point_matches",
            "seed_inliers",
            "refined_inliers",
            "guided_projection_matches",
            "final_matched_map_points",
            "density_concern",
        ],
    )
    write_csv(
        output_dir / "gt_loop_recall_by_stage.csv",
        recall_rows,
        ["stage", "count", "recall_vs_gt_loop_like_percent"],
    )
    write_csv(
        output_dir / "gt_near_loop_recall_by_stage.csv",
        near_recall_rows,
        ["stage", "count", "recall_vs_gt_loop_like_percent"],
    )
    write_csv(
        output_dir / "gt_loop_density_support_analysis.csv",
        density_rows,
        [
            "pair_key",
            "kf_i",
            "kf_j",
            "gt_translation_distance",
            "gt_rotation_angle_deg",
            "pipeline_stage",
            "num_map_points_i",
            "num_map_points_j",
            "min_pair_map_points",
            "time_gap_sec",
            "kf_id_gap",
            "candidate_group_size",
            "final_matched_map_points",
            "density_concern",
        ],
    )
    write_csv(
        output_dir / "gt_loop_missed_pairs_top.csv",
        top_missed_rows,
        [
            "pair_key",
            "kf_i",
            "kf_j",
            "gt_translation_distance",
            "gt_rotation_angle_deg",
            "pipeline_stage",
            "rejection_reason",
            "bow_score",
            "common_words",
            "accumulated_score",
            "consistency_score",
            "seed_inliers",
            "final_matched_map_points",
            "num_map_points_i",
            "num_map_points_j",
            "suggested_investigation",
        ],
    )
    (output_dir / "gt_loop_recall_summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    config = {
        "min_time_gap_sec": float(args.min_time_gap_sec),
        "min_kf_gap": int(args.min_kf_gap),
        "loop_trans_threshold_m": float(args.loop_trans_threshold_m),
        "loop_rot_threshold_deg": float(args.loop_rot_threshold_deg),
        "near_loop_trans_threshold_m": float(args.near_loop_trans_threshold_m),
        "gt_association_max_dt_sec": float(args.gt_association_max_dt_sec),
    }
    write_report(
        report_path=output_dir / "GT_LOOP_RECALL_ANALYSIS_REPORT.md",
        run_dir=run_dir,
        groundtruth=groundtruth,
        keyframes=keyframes,
        gt_associations=gt_associations,
        summary=summary_json,
        recall_rows=recall_rows,
        near_recall_rows=near_recall_rows,
        top_missed_rows=top_missed_rows,
        accepted_rows=accepted_rows,
        classified_rows=classified_rows,
        config=config,
        keyframe_metadata=keyframe_metadata,
    )

    console_summary = {
        "run_dir": str(run_dir),
        "groundtruth": str(groundtruth),
        "output": str(output_dir),
        "num_keyframes": len(keyframes),
        "num_gt_associated_keyframes": summary_json["num_gt_associated_keyframes"],
        "num_gt_loop_like_pairs": summary_json["num_gt_loop_like_pairs"],
        "gt_loop_like_stage_counts": summary_json["gt_loop_like_stage_counts"],
        "suggested_next_focus": summary_json["suggested_next_focus"],
        "retrieval_profile_rows": len(retrieval_by_kf),
    }
    print(json.dumps(console_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
