"""
Phase 10 checkpoint — Map output.

Per CLAUDE.md §4 Phase 10: TUM trajectory format conformance; occupancy grid
PNG is non-empty; the grid changes when a synthetic late loop closure deforms
the trajectory; outputs land in ``<output_dir>/<run_id>/`` as ``trajectory.tum``
+ ``occupancy.png``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from slam_core.common.types import Pose2
from slam_core.common.se2 import transform_points, pose_compose
from slam_core.fusion.map_output import (
    write_tum_trajectory,
    assemble_occupancy_grid,
    save_occupancy_png,
    emit_run_outputs,
)
from slam_core.fusion.runner import FusionRunResult


def _ring_scan(n=200, r=1.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def test_tum_trajectory_format(tmp_path):
    traj = [(0.0, Pose2(0, 0, 0)), (1.5, Pose2(1.0, 2.0, math.pi / 2))]
    path = write_tum_trajectory(traj, tmp_path / "trajectory.tum")
    lines = [l for l in path.read_text().splitlines() if l and not l.startswith("#")]
    assert len(lines) == 2
    parts = lines[1].split()
    assert len(parts) == 8                       # t tx ty tz qx qy qz qw
    t, tx, ty, tz, qx, qy, qz, qw = map(float, parts)
    assert (tx, ty, tz) == pytest.approx((1.0, 2.0, 0.0))
    # yaw = pi/2 -> qz = sin(pi/4), qw = cos(pi/4); unit quaternion
    assert qz == pytest.approx(math.sin(math.pi / 4))
    assert math.isclose(qx * qx + qy * qy + qz * qz + qw * qw, 1.0, abs_tol=1e-9)


def test_occupancy_grid_non_empty():
    scan = _ring_scan()
    grid = assemble_occupancy_grid([(Pose2(0, 0, 0), scan), (Pose2(3, 0, 0), scan)],
                                   resolution=0.1)
    assert not grid.is_empty
    assert grid.prob.shape[0] > 1 and grid.prob.shape[1] > 1
    assert np.any(grid.prob > 0)


def test_occupancy_grid_changes_after_late_loop():
    scan = _ring_scan()
    ids = list(range(5))
    scans = {i: scan for i in ids}
    # drifted (pre-loop) poses vs corrected (post-loop) poses
    drifted = {i: Pose2(i * 1.0, 0.3 * i, 0.0) for i in ids}
    corrected = {i: Pose2(i * 1.0, 0.0, 0.0) for i in ids}

    g_before = assemble_occupancy_grid([(drifted[i], scans[i]) for i in ids], resolution=0.1)
    g_after = assemble_occupancy_grid([(corrected[i], scans[i]) for i in ids], resolution=0.1)
    # the grid genuinely redraws: different shape or different content
    assert g_before.prob.shape != g_after.prob.shape or not np.array_equal(
        g_before.prob, g_after.prob)


def test_emit_run_outputs_writes_files(tmp_path):
    ids = list(range(4))
    scan = _ring_scan()
    result = FusionRunResult(
        keyframe_ids=ids,
        trajectory=[(float(i), Pose2(i * 0.5, 0.0, 0.0)) for i in ids],
        optimized_poses={i: Pose2(i * 0.5, 0.0, 0.0) for i in ids},
        frontend_poses={i: Pose2(i * 0.5, 0.0, 0.0) for i in ids},
        keyframe_scans={i: scan for i in ids},
    )
    paths = emit_run_outputs(result, tmp_path, run_id="run_test")
    assert paths["trajectory"].exists()
    assert paths["occupancy"].exists()
    assert paths["run_dir"] == tmp_path / "run_test"
    assert paths["occupancy"].stat().st_size > 0
    # trajectory has one row per keyframe
    rows = [l for l in paths["trajectory"].read_text().splitlines()
            if l and not l.startswith("#")]
    assert len(rows) == len(ids)


def test_empty_grid_saves_placeholder(tmp_path):
    grid = assemble_occupancy_grid([], resolution=0.1)
    assert grid.is_empty
    p = save_occupancy_png(grid, tmp_path / "empty.png")
    assert p.exists() and p.stat().st_size > 0
