from pathlib import Path

import numpy as np

from tools.compare_orb_extractors import (
    collect_backend_metrics,
    compare_extractors,
    make_synthetic_tum_like_images,
)
from visual_slam.orbslam.local_features import PySLAMORB2Backend


def test_comparison_tool_imports_and_synthetic_images():
    images = make_synthetic_tum_like_images(2)

    assert len(images) == 2
    assert images[0].shape == (480, 640)
    assert images[0].dtype == np.uint8


def test_opencv_orb_metrics_generated_for_synthetic_images():
    images = make_synthetic_tum_like_images(3)
    report, rows = collect_backend_metrics("opencv_orb", images)

    assert report.available is True
    assert report.avg_features > 100
    assert report.descriptor_dtype == "uint8"
    assert report.descriptor_shape.endswith("x32")
    assert report.avg_grid_coverage > 0.5
    assert len(rows) == 3


def test_compare_extractors_writes_report_when_dataset_absent(tmp_path):
    output = tmp_path / "comparison"
    reports = compare_extractors(
        dataset=Path("/definitely/not/a/tum/dataset"),
        output=output,
        max_frames=3,
    )

    summary = output / "extractor_comparison_summary.md"
    metrics = output / "extractor_frame_metrics.csv"

    assert summary.exists()
    assert metrics.exists()
    text = summary.read_text()
    assert "opencv_orb" in text
    assert "pyslam_orb2" in text
    assert "Default Backend Decision" in text
    assert any(report.backend == "opencv_orb" and report.available for report in reports)

    if not PySLAMORB2Backend.is_available():
        assert "unavailable" in text.lower()


def test_pyslam_orb2_unavailable_report_exits_successfully():
    images = make_synthetic_tum_like_images(1)
    report, rows = collect_backend_metrics("pyslam_orb2", images)

    if PySLAMORB2Backend.is_available():
        assert report.available is True
        assert rows
    else:
        assert report.available is False
        assert "orbslam2_features" in report.unavailable_reason
        assert rows == []
