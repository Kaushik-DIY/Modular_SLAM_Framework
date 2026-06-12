"""C6 — fusion2 lidar-mode pipeline on a short lab_hybrid slice.

Validates the full orchestration loop: front-end -> Signature -> memory tiers ->
graph spine -> (bounded) proposals -> candidate-local B&B -> outputs. Uses 600
scans so it runs in a few minutes; the full-run gate is exercised by the C6
validation run recorded in FUSION2_STATUS.md.
"""
import json
from pathlib import Path

import numpy as np
import pytest

fusion_core = pytest.importorskip("fusion_core")

DATASET = Path("datasets/lab_hybrid_small")

pytestmark = pytest.mark.skipif(not DATASET.exists(),
                                reason="lab_hybrid dataset not present")


@pytest.fixture(scope="module")
def short_run(tmp_path_factory):
    from slam_core.fusion2.config import FusionV2Config
    from slam_core.fusion2.runner import run_lidar_mode

    out = tmp_path_factory.mktemp("fusion2_c6")
    cfg = FusionV2Config(mode="lidar", dataset=DATASET, output_dir=out,
                         max_scans=600, print_every=10_000)
    stats = run_lidar_mode(cfg)
    return cfg, stats


def test_pipeline_completes_and_tracks(short_run):
    cfg, stats = short_run
    assert stats["scans"] == 600
    assert stats["keyframes"] >= 20
    # memory bounded
    assert stats["stm"] <= cfg.stm_size
    assert stats["wm"] <= cfg.wm_cap


def test_rehearsal_only_when_stationary(short_run):
    _, stats = short_run
    # moving stream: merges must be a small minority of keyframes (the lab robot
    # pauses occasionally), never the majority as with ungated auto-similarity.
    assert stats["rehearsal_merges"] < stats["keyframes"] * 0.4


def test_memory_footprint_tiny(short_run):
    _, stats = short_run
    assert stats["peak_rss_gb"] < 1.0  # vs ~10 GB Python-object equivalents
    assert stats["map_payload_mb"] < 50


def test_outputs_written(short_run):
    _, stats = short_run
    run_dir = Path(stats["run_dir"])
    traj = run_dir / "trajectory.tum"
    assert traj.exists()
    rows = [l.split() for l in open(traj) if l.strip()]
    assert len(rows) == stats["keyframes"]
    assert all(len(r) == 8 for r in rows)  # TUM format
    assert (run_dir / "occupancy.png").stat().st_size > 10_000
    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["mode"] == "lidar"
