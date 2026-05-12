#!/usr/bin/env python3
"""
Compare OpenCV ORB and optional pySLAM ORB2 extractor backends.

The tool intentionally keeps OpenCV ORB as the stable baseline.  If the
``orbslam2_features`` module is absent from the project venv, pySLAM ORB2 is
reported as unavailable and the command still exits successfully.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BACKENDS = ("opencv_orb", "pyslam_orb2")


@dataclass
class BackendReport:
    backend: str
    available: bool
    unavailable_reason: str = ""
    avg_features: float = 0.0
    avg_fps: float = 0.0
    avg_grid_coverage: float = 0.0
    avg_match_count: float = 0.0
    descriptor_dtype: str = "n/a"
    descriptor_shape: str = "n/a"
    octave_histogram: dict[int, int] | None = None
    smoke: dict[int, dict[str, str]] | None = None


def make_synthetic_tum_like_images(num_images: int = 3) -> list[np.ndarray]:
    images = []
    for i in range(num_images):
        rng = np.random.default_rng(100 + i)
        image = rng.integers(0, 80, size=(480, 640), dtype=np.uint8)
        dx = 2 * i
        for x in range(55, 610, 70):
            cv2.circle(image, (x + dx, 170), 18, 255, 2)
            cv2.rectangle(image, (x - 16 + dx, 300), (x + 16 + dx, 334), 190, 2)
        for y in range(65, 435, 70):
            cv2.line(image, (75 + dx, y), (565 + dx, y), 220, 2)
        images.append(image)
    return images


def load_dataset_images(dataset: Path, max_frames: int) -> list[np.ndarray]:
    from visual_slam.orbslam.io import load_tum_rgbd_associations

    frames = load_tum_rgbd_associations(dataset)[:max_frames]
    images: list[np.ndarray] = []
    for entry in frames:
        image = cv2.imread(str(entry.rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load RGB image: {entry.rgb_path}")
        images.append(image)
    return images


def grid_coverage(keypoints, image_shape, rows: int = 4, cols: int = 4) -> float:
    if len(keypoints) == 0:
        return 0.0
    height, width = image_shape[:2]
    occupied = set()
    for kp in keypoints:
        x, y = kp.pt
        col = min(cols - 1, max(0, int(cols * x / max(width, 1))))
        row = min(rows - 1, max(0, int(rows * y / max(height, 1))))
        occupied.add((row, col))
    return len(occupied) / float(rows * cols)


def descriptor_match_count(des1, des2, ratio_test: float = 0.75) -> int:
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(des1, des2, k=2)
    good = 0
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_test * n.distance:
            good += 1
    return good


def collect_backend_metrics(backend_name: str, images: list[np.ndarray]) -> tuple[BackendReport, list[dict]]:
    from visual_slam.orbslam.local_features import BackendUnavailableError, create_extractor_backend

    try:
        backend = create_extractor_backend(backend_name)
    except (BackendUnavailableError, ImportError) as exc:
        return BackendReport(backend=backend_name, available=False, unavailable_reason=str(exc)), []

    rows = []
    octave_hist: dict[int, int] = {}
    feature_counts = []
    fps_values = []
    coverage_values = []
    match_counts = []
    prev_des = None
    last_shape = "n/a"
    last_dtype = "n/a"

    for idx, image in enumerate(images):
        t0 = time.perf_counter()
        result = backend.extract(image)
        elapsed = max(time.perf_counter() - t0, 1e-9)

        count = len(result.keypoints)
        feature_counts.append(count)
        fps_values.append(1.0 / elapsed)
        coverage = grid_coverage(result.keypoints, image.shape)
        coverage_values.append(coverage)
        match_count = descriptor_match_count(prev_des, result.descriptors)
        match_counts.append(match_count)
        prev_des = result.descriptors

        if result.descriptors is not None:
            last_shape = "x".join(str(v) for v in result.descriptors.shape)
            last_dtype = str(result.descriptors.dtype)

        frame_hist: dict[int, int] = {}
        for octave in np.asarray(result.octaves, dtype=np.int32).tolist():
            octave_hist[octave] = octave_hist.get(octave, 0) + 1
            frame_hist[octave] = frame_hist.get(octave, 0) + 1

        rows.append(
            {
                "backend": backend_name,
                "frame_index": idx,
                "feature_count": count,
                "descriptor_shape": last_shape,
                "descriptor_dtype": last_dtype,
                "grid_coverage": f"{coverage:.6f}",
                "match_count_prev": match_count,
                "fps": f"{1.0 / elapsed:.3f}",
                "octave_histogram": format_histogram(frame_hist),
            }
        )

    report = BackendReport(
        backend=backend_name,
        available=True,
        avg_features=mean(feature_counts) if feature_counts else 0.0,
        avg_fps=mean(fps_values) if fps_values else 0.0,
        avg_grid_coverage=mean(coverage_values) if coverage_values else 0.0,
        avg_match_count=mean(match_counts) if match_counts else 0.0,
        descriptor_dtype=last_dtype,
        descriptor_shape=last_shape,
        octave_histogram=octave_hist,
    )
    return report, rows


def run_backend_smoke(dataset: Path, output_dir: Path, backend_name: str, nframes: int) -> dict[str, str]:
    from visual_slam.orbslam.io import load_tum_rgbd_associations, make_tum_rgbd_camera
    from visual_slam.orbslam.slam import Slam, SensorType, SlamState

    frames = load_tum_rgbd_associations(dataset)[:nframes]
    camera = make_tum_rgbd_camera(dataset.name)
    slam = Slam(
        camera=camera,
        sensor_type=SensorType.RGBD,
        headless=True,
        start_local_mapping_thread=False,
        feature_tracker_config={"extractor_backend": backend_name},
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ok_count = 0
    lost_count = 0
    errors = 0

    for idx, entry in enumerate(frames):
        rgb = cv2.imread(str(entry.rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(entry.depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise FileNotFoundError(f"Could not load TUM RGB-D frame {idx}")
        ok = slam.track(img=rgb, img_right=None, depth=depth, img_id=idx, timestamp=entry.timestamp)
        while slam.local_mapping.queue_size() > 0:
            slam.local_mapping.step()
        state = slam.get_tracking_state()
        if ok and state == SlamState.OK:
            ok_count += 1
        elif state == SlamState.LOST:
            lost_count += 1

    elapsed = max(time.perf_counter() - started, 1e-9)
    summary = {
        "frames_attempted": str(len(frames)),
        "tracking_ok_count": str(ok_count),
        "tracking_lost_count": str(lost_count),
        "errors": str(errors),
        "final_state": slam.get_tracking_state().name,
        "final_keyframes": str(slam.map.num_keyframes()),
        "final_map_points": str(slam.map.num_points()),
        "avg_fps": f"{len(frames) / elapsed:.2f}",
    }

    with open(output_dir / f"smoke_{nframes}_summary.txt", "w") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    return summary


def write_frame_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "backend",
        "frame_index",
        "feature_count",
        "descriptor_shape",
        "descriptor_dtype",
        "grid_coverage",
        "match_count_prev",
        "fps",
        "octave_histogram",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, reports: list[BackendReport], dataset: Path, smoke_enabled: bool) -> None:
    lines = [
        "# ORB Extractor Comparison",
        "",
        f"Dataset: `{dataset}`",
        f"Smoke validation run: `{smoke_enabled}`",
        "",
        "| Backend | Available | Avg features | Descriptor | Grid coverage | Match count | Avg FPS | Smoke 3/10/30 |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for report in reports:
        if not report.available:
            lines.append(
                f"| {report.backend} | no | n/a | n/a | n/a | n/a | n/a | unavailable: {report.unavailable_reason} |"
            )
            continue
        smoke_text = "not run"
        if report.smoke:
            parts = []
            for n in (3, 10, 30):
                if n in report.smoke:
                    summary = report.smoke[n]
                    parts.append(
                        f"{n}: {summary.get('tracking_ok_count', '0')}/{summary.get('frames_attempted', '0')} "
                        f"lost={summary.get('tracking_lost_count', '0')} "
                        f"{summary.get('final_state', 'UNKNOWN')} "
                        f"kf={summary.get('final_keyframes', 'n/a')} "
                        f"mp={summary.get('final_map_points', 'n/a')}"
                    )
            smoke_text = "; ".join(parts)
        lines.append(
            f"| {report.backend} | yes | {report.avg_features:.1f} | "
            f"{report.descriptor_dtype} {report.descriptor_shape} | "
            f"{report.avg_grid_coverage:.3f} | {report.avg_match_count:.1f} | "
            f"{report.avg_fps:.2f} | {smoke_text} |"
        )

    lines.extend(
        [
            "",
            "## Default Backend Decision",
            "",
            "Keep `opencv_orb` as the default backend. pySLAM ORB2 may be selected only when "
            "`orbslam2_features` is available inside the project virtual environment and passes "
            "the same smoke gates.",
            "",
            "Projection-match and pose-optimization inlier metrics are not emitted by the current "
            "extractor-only comparison path; the TUM smoke summaries record downstream tracking "
            "OK/lost counts plus final keyframe and map-point counts instead.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def compare_extractors(dataset: Path, output: Path, max_frames: int = 30) -> list[BackendReport]:
    dataset = Path(dataset).expanduser()
    output = Path(output).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    smoke_enabled = dataset.exists()
    images = load_dataset_images(dataset, max_frames) if smoke_enabled else make_synthetic_tum_like_images(max_frames)

    all_rows = []
    reports = []
    for backend_name in BACKENDS:
        report, rows = collect_backend_metrics(backend_name, images)
        all_rows.extend(rows)

        if report.available and smoke_enabled:
            smoke_dir = output / f"{backend_name}_smoke"
            report.smoke = {}
            for n in (3, 10, 30):
                if n <= len(images):
                    report.smoke[n] = run_backend_smoke(dataset, smoke_dir, backend_name, n)

        reports.append(report)

    write_frame_metrics_csv(output / "extractor_frame_metrics.csv", all_rows)
    write_summary(output / "extractor_comparison_summary.md", reports, dataset, smoke_enabled)
    return reports


def format_histogram(histogram: dict[int, int]) -> str:
    return ";".join(f"{k}:{histogram[k]}" for k in sorted(histogram))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ORB extractor backends.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=30)
    args = parser.parse_args()

    reports = compare_extractors(args.dataset, args.output, max_frames=args.max_frames)
    for report in reports:
        if report.available:
            print(
                f"{report.backend}: avg_features={report.avg_features:.1f} "
                f"avg_fps={report.avg_fps:.2f} descriptor={report.descriptor_dtype} {report.descriptor_shape}"
            )
        else:
            print(f"{report.backend}: unavailable ({report.unavailable_reason})")
    print(f"summary: {Path(args.output).expanduser() / 'extractor_comparison_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
