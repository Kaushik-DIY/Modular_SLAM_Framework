"""
Map outputs — TUM trajectory writer + 2D occupancy grid assembler.

RTAB_inspired_implementation_plan.md §13 Phase 10. Two artefacts per run, landing
in ``<output_dir>/<run_id>/``:

- ``trajectory.tum`` — TUM-format planar trajectory (z=0, yaw-only quaternion).
- ``occupancy.png`` — a 2D occupancy grid assembled from every keyframe's scan
  transformed by its *optimized* pose. Because it is built from optimized poses,
  it redraws correctly after a late loop closure deforms the graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from slam_core.common.types import Pose2
from slam_core.common.se2 import transform_points


# --------------------------------------------------------------------------
# Trajectory (TUM format)
# --------------------------------------------------------------------------

def write_tum_trajectory(trajectory: Sequence[Tuple[float, Pose2]], path) -> Path:
    """Write ``[(timestamp, Pose2), ...]`` as TUM ``t tx ty tz qx qy qz qw``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# timestamp tx ty tz qx qy qz qw"]
    for t, p in trajectory:
        qz = math.sin(p.theta / 2.0)
        qw = math.cos(p.theta / 2.0)
        lines.append(
            f"{float(t):.6f} {p.x:.9f} {p.y:.9f} 0.000000000 "
            f"0.000000000 0.000000000 {qz:.9f} {qw:.9f}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------
# Occupancy grid
# --------------------------------------------------------------------------

@dataclass
class OccupancyGrid:
    prob: np.ndarray              # (H, W) occupancy probability in [0, 1]
    resolution: float             # metres per cell
    origin: Tuple[float, float]   # world (x, y) of cell (row=0, col=0)

    @property
    def is_empty(self) -> bool:
        return self.prob.size == 0 or not np.any(self.prob > 0.0)


def assemble_occupancy_grid(
    pose_scan_pairs: Sequence[Tuple[Pose2, np.ndarray]],
    resolution: float = 0.05,
    padding: float = 1.0,
    hit_saturation: int = 3,
) -> OccupancyGrid:
    """Accumulate scans transformed by their poses into an occupancy grid.

    Each scan point (sensor frame, (M,2)) is transformed to world by its
    keyframe pose; cell hit-counts are turned into an occupancy probability
    that saturates after ``hit_saturation`` hits.
    """
    world_pts: List[np.ndarray] = []
    for pose, scan in pose_scan_pairs:
        if scan is None or len(scan) == 0:
            continue
        world_pts.append(transform_points(pose, np.asarray(scan, float)[:, :2]))

    if not world_pts:
        return OccupancyGrid(np.zeros((0, 0), float), resolution, (0.0, 0.0))

    P = np.vstack(world_pts)
    min_x, min_y = P.min(axis=0) - padding
    max_x, max_y = P.max(axis=0) + padding
    W = max(1, int(math.ceil((max_x - min_x) / resolution)))
    H = max(1, int(math.ceil((max_y - min_y) / resolution)))

    counts = np.zeros((H, W), dtype=np.float64)
    ix = ((P[:, 0] - min_x) / resolution).astype(int)
    iy = ((P[:, 1] - min_y) / resolution).astype(int)
    inb = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    np.add.at(counts, (iy[inb], ix[inb]), 1.0)

    prob = np.clip(counts / float(max(1, hit_saturation)), 0.0, 1.0)
    return OccupancyGrid(prob, resolution, (float(min_x), float(min_y)))


def save_occupancy_png(grid: OccupancyGrid, path) -> Path:
    """Save the grid as a PNG (occupied = dark, north = up)."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if grid.prob.size == 0:
        img = np.full((1, 1), 255, dtype=np.uint8)
    else:
        img = (255.0 * (1.0 - grid.prob)).astype(np.uint8)
        img = np.flipud(img)  # world +y up
    cv2.imwrite(str(path), img)
    return path


# --------------------------------------------------------------------------
# Top-level emission for a runner result
# --------------------------------------------------------------------------

def emit_run_outputs(result, output_dir, run_id: Optional[str] = None,
                     resolution: float = 0.05) -> Dict[str, Path]:
    """Write ``trajectory.tum`` + ``occupancy.png`` for a FusionRunResult."""
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_path = write_tum_trajectory(result.trajectory, out_dir / "trajectory.tum")

    pairs = [(result.optimized_poses.get(i) or result.frontend_poses.get(i),
              result.keyframe_scans.get(i))
             for i in result.keyframe_ids]
    pairs = [(p, s) for p, s in pairs if p is not None]
    grid = assemble_occupancy_grid(pairs, resolution=resolution)
    grid_path = save_occupancy_png(grid, out_dir / "occupancy.png")

    return {"run_dir": out_dir, "trajectory": traj_path, "occupancy": grid_path}
