"""
Post-run artefact generation for the pass-through modes (A = ORB, B = LiDAR).

Makes Modes A/B *complete* one-command pipelines: after the fusion runner
delegates to the existing SLAM runner, it generates the same trajectory plots +
final rebuilt map that the standalone runners' eval/plot tools produce — so the
user never has to run the original runner or its plotting tools separately.

- Mode A (visual): reuses ``tools/plot_rgbd_run.py`` (trajectory_topdown/3d,
  tracking_stats, map_topdown/3d) and ``tools/generate_lab_map.py`` (sparse +
  semi-dense maps, summary), run as subprocesses on the ORB output directory.
- Mode B (LiDAR): builds the final occupancy map from the trajectory + scans
  (reusing the dataset scan loader + ``assemble_occupancy_grid``) and renders
  the trajectory plots — no SLAM re-run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Mode A (visual) — chain the existing ORB plot/eval tools
# --------------------------------------------------------------------------

def generate_visual_artifacts(run_dir, dataset: Optional[str], out_dir) -> Dict[str, Path]:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT) + (os.pathsep + env.get("PYTHONPATH", ""))

    produced: Dict[str, Path] = {}

    def _run(script: str, args: List[str], label: str):
        cmd = [sys.executable, str(_REPO_ROOT / "tools" / script), *args]
        r = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env)
        if r.returncode == 0:
            produced[label] = out_dir

    _run("plot_rgbd_run.py", ["--run", str(run_dir), "--output", str(out_dir / "plots")],
         "trajectory_plots")
    lab_args = ["--run", str(run_dir), "--output", str(out_dir / "map_figures")]
    if dataset:
        lab_args += ["--dataset", str(dataset)]
    _run("generate_lab_map.py", lab_args, "rebuilt_map")
    return produced


# --------------------------------------------------------------------------
# Mode B (LiDAR) — build occupancy map + trajectory plots from the run output
# --------------------------------------------------------------------------

def _load_lidar_trajectory(path) -> np.ndarray:
    """Read the Hector trajectory (t x y theta score)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 4:
            rows.append([float(p[0]), float(p[1]), float(p[2]), float(p[3])])
    return np.array(rows)


def generate_lidar_artifacts(traj_path, dataset_name: str, out_dir,
                             scan_variant: Optional[str] = None,
                             resolution: float = 0.05) -> Dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from slam_core.common.types import Pose2
    from slam_core.dataio.dataset_catalog import load_dataset_scans
    from carto.local_slam.range_to_points import ranges_to_points
    from slam_core.fusion.map_output import assemble_occupancy_grid, save_occupancy_png

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: Dict[str, Path] = {}

    T = _load_lidar_trajectory(traj_path)
    if len(T) == 0:
        return produced

    # --- final rebuilt occupancy map (scans transformed by trajectory poses) ---
    try:
        profile, scans = load_dataset_scans(dataset_name, scan_variant=scan_variant)
        rmin = float(getattr(profile, "range_min", 0.1))
        rmax = float(getattr(profile, "range_max", 30.0))
        stride = int(getattr(profile, "beam_stride", 1) or 1)
        n = min(len(T), len(scans))
        pairs = []
        for i in range(n):
            pose = Pose2(float(T[i, 1]), float(T[i, 2]), float(T[i, 3]))
            ranges = np.asarray(scans[i]["ranges"], dtype=float)
            pts = ranges_to_points(ranges, profile.angle_min, profile.angle_inc,
                                   rmin, rmax, stride=stride)
            if len(pts):
                pairs.append((pose, pts))
        grid = assemble_occupancy_grid(pairs, resolution=resolution)
        map_path = save_occupancy_png(grid, out_dir / "final_map.png")
        produced["rebuilt_map"] = map_path
    except Exception as e:  # never block on map rendering
        print(f"[artifacts] LiDAR map build skipped: {e}")

    # --- trajectory plots ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(T[:, 1], T[:, 2], "b-", lw=1)
    ax.scatter(T[0, 1], T[0, 2], c="g", s=40, label="start")
    ax.scatter(T[-1, 1], T[-1, 2], c="r", s=40, marker="X", label="end")
    ax.set_title("LiDAR trajectory (top-down)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); p = out_dir / "trajectory_xy.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    produced["trajectory_xy"] = p

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(np.degrees(T[:, 3]), "m-", lw=1)
    ax.set_title("Heading (theta) vs scan"); ax.set_xlabel("scan"); ax.set_ylabel("deg")
    ax.grid(alpha=0.3)
    fig.tight_layout(); p = out_dir / "trajectory_theta.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    produced["trajectory_theta"] = p

    return produced
