#!/usr/bin/env python3
"""Probe likely fr1_room loop pairs from ground truth against local loop verification."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_tum_reference_cloud import read_tum_groundtruth
from visual_slam.orbslam.io import load_tum_rgbd_associations, make_tum_rgbd_camera
from visual_slam.orbslam.slam import SensorType, Slam
from visual_slam.orbslam.slam.loop_closing import LoopGeometryChecker


@dataclass(frozen=True)
class OraclePair:
    current_kf_id: int
    candidate_kf_id: int
    current_timestamp: float
    candidate_timestamp: float
    translation_m: float
    rotation_deg: float


def rotation_angle_degrees(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R = np.asarray(R_a, dtype=np.float64).reshape(3, 3).T @ np.asarray(R_b, dtype=np.float64).reshape(3, 3)
    value = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) * 0.5))
    return float(math.degrees(math.acos(value)))


def nearest_gt_pose(groundtruth, timestamp: float):
    if not groundtruth:
        return None
    times = np.asarray([pose.timestamp for pose in groundtruth], dtype=np.float64)
    idx = int(np.argmin(np.abs(times - float(timestamp))))
    return groundtruth[idx]


def find_oracle_pairs(
    keyframes: list,
    groundtruth,
    *,
    min_time_separation_sec: float = 20.0,
    max_translation_m: float = 0.4,
    max_rotation_deg: float = 35.0,
    max_pairs: int = 20,
) -> list[OraclePair]:
    keyframes = sorted(keyframes, key=lambda kf: float(getattr(kf, "timestamp", getattr(kf, "id", 0.0)) or 0.0))
    gt_by_kid = {}
    for keyframe in keyframes:
        pose = nearest_gt_pose(groundtruth, float(getattr(keyframe, "timestamp", 0.0) or 0.0))
        if pose is not None:
            gt_by_kid[int(getattr(keyframe, "kid", getattr(keyframe, "id", -1)))] = pose

    pairs: list[OraclePair] = []
    for i, candidate in enumerate(keyframes):
        candidate_kid = int(getattr(candidate, "kid", getattr(candidate, "id", -1)))
        gt_candidate = gt_by_kid.get(candidate_kid)
        if gt_candidate is None:
            continue
        for current in keyframes[i + 1 :]:
            current_kid = int(getattr(current, "kid", getattr(current, "id", -1)))
            gt_current = gt_by_kid.get(current_kid)
            if gt_current is None:
                continue
            dt = abs(float(getattr(current, "timestamp", 0.0)) - float(getattr(candidate, "timestamp", 0.0)))
            if dt < min_time_separation_sec:
                continue
            trans = float(np.linalg.norm(gt_current.Twc[:3, 3] - gt_candidate.Twc[:3, 3]))
            if trans > max_translation_m:
                continue
            rot = rotation_angle_degrees(gt_current.Twc[:3, :3], gt_candidate.Twc[:3, :3])
            if rot > max_rotation_deg:
                continue
            pairs.append(
                OraclePair(
                    current_kf_id=current_kid,
                    candidate_kf_id=candidate_kid,
                    current_timestamp=float(getattr(current, "timestamp", 0.0) or 0.0),
                    candidate_timestamp=float(getattr(candidate, "timestamp", 0.0) or 0.0),
                    translation_m=trans,
                    rotation_deg=rot,
                )
            )
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _load_rgb(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _load_depth(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return image


def build_slam_keyframes(dataset: Path, backend: str, max_frames: int = 0):
    frames = load_tum_rgbd_associations(dataset)
    if max_frames > 0:
        frames = frames[: int(max_frames)]
    slam = Slam(
        camera=make_tum_rgbd_camera(dataset.name),
        sensor_type=SensorType.RGBD,
        headless=True,
        start_local_mapping_thread=False,
        feature_tracker_config={"extractor_backend": backend},
        enable_loop_closing=False,
    )
    for frame_idx, entry in enumerate(frames):
        slam.track(
            _load_rgb(entry.rgb_path),
            img_right=None,
            depth=_load_depth(entry.depth_path),
            img_id=frame_idx,
            timestamp=entry.timestamp,
        )
        while slam.local_mapping.queue_size() > 0:
            slam.local_mapping.step()
    return slam, slam.map.get_keyframes()


def probe_pairs(dataset: Path, output: Path, backend: str, max_pairs: int = 20, max_frames: int = 0) -> dict:
    dataset = Path(dataset).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    pair_dir = output / "pair_reports"
    pair_dir.mkdir(parents=True, exist_ok=True)

    slam, keyframes = build_slam_keyframes(dataset, backend, max_frames=max_frames)
    groundtruth = read_tum_groundtruth(dataset / "groundtruth.txt")
    pairs = find_oracle_pairs(keyframes, groundtruth, max_pairs=max_pairs)
    keyframe_by_kid = {int(getattr(kf, "kid", getattr(kf, "id", -1))): kf for kf in keyframes}
    checker = LoopGeometryChecker(keyframe_database=slam.keyframe_database)

    rows = []
    for event_id, pair in enumerate(pairs, start=1):
        current = keyframe_by_kid[pair.current_kf_id]
        candidate = keyframe_by_kid[pair.candidate_kf_id]
        accepted = checker.check_candidates(current, [candidate])
        report = dict(checker.last_candidate_reports.get(pair.candidate_kf_id, {}))
        report.update(
            {
                "oracle_pair": pair.__dict__,
                "accepted": bool(accepted),
                "top_rejection_reason": getattr(checker, "last_error", "") if not accepted else "",
            }
        )
        (pair_dir / f"oracle_pair_{event_id}_kf_{pair.current_kf_id}_{pair.candidate_kf_id}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
        )
        rows.append(
            {
                **pair.__dict__,
                "event_id": event_id,
                "accepted": int(bool(accepted)),
                "bow_matches": int(report.get("bow_matches_after_orientation", 0) or 0),
                "geometry_inliers": int(report.get("geometry_ransac_inliers", 0) or 0),
                "guided_projection_matches": int(report.get("guided_projection_matches", 0) or 0),
                "final_inliers": int(report.get("final_inliers", 0) or 0),
                "rejection_reason": report.get("top_rejection_reason", ""),
            }
        )

    csv_path = output / "oracle_pairs.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "event_id",
            "current_kf_id",
            "candidate_kf_id",
            "current_timestamp",
            "candidate_timestamp",
            "translation_m",
            "rotation_deg",
            "accepted",
            "bow_matches",
            "geometry_inliers",
            "guided_projection_matches",
            "final_inliers",
            "rejection_reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "dataset": str(dataset),
        "backend": backend,
        "keyframes": len(keyframes),
        "oracle_pairs": len(pairs),
        "accepted_pairs": int(sum(row["accepted"] for row in rows)),
        "oracle_pairs_csv": str(csv_path),
        "pair_reports": str(pair_dir),
    }
    (output / "oracle_probe_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", default="pyslam_orb2", choices=["pyslam_orb2", "opencv_orb", "auto"])
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(probe_pairs(args.dataset, args.output, args.backend, args.max_pairs, args.max_frames), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
