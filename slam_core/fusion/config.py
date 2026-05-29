"""
Fusion runner configuration.

Defines the four user-selectable modes and the single FusionConfig dataclass
that carries every tunable knob. Defaults are the Jetson-Nano-conservative
values locked in CLAUDE.md §2.13 and RTAB_inspired_implementation_plan.md §12.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class Mode(str, Enum):
    """The four user-selectable SLAM modes (CLAUDE.md §2.2)."""

    ORB = "orb"        # pass-through to existing ORB-SLAM runner
    LIDAR = "lidar"    # pass-through to existing LiDAR runner
    VLMAIN = "vlmain"  # visual-main proposes, LiDAR ICP verifies
    LVMAIN = "lvmain"  # LiDAR-main proposes, visual PnP verifies


@dataclass
class FusionConfig:
    """All tunable parameters for a fusion run.

    Required (no default): mode, dataset_path, output_dir. Everything else
    matches the §12 defaults; the runner exposes one CLI flag per knob.
    """

    # --- Required ---
    mode: Mode
    dataset_path: str
    output_dir: str

    # --- LiDAR front-end choice (Modes B / C / D) ---
    lidar_frontend: Literal["scan_to_submap", "scan_to_map"] = "scan_to_submap"

    # --- Sensor sync ---
    sync_tolerance_s: float = 0.050

    # --- Memory tiers ---
    stm_size: int = 30
    wm_cap: int = 200
    ltm_cap: int = 1000
    rehearsal_similarity: float = 0.2

    # --- ICP verifier ---
    icp_max_corr: float = 0.5
    icp_fitness: float = 0.6
    icp_inlier_rmse: float = 0.10

    # --- Visual verifier ---
    visual_nndr: float = 0.7
    visual_min_inliers: int = 15

    # --- Optimization ---
    optimize_every_n_keyframes: int = 30
    optimize_max_error_factor: float = 1.0
