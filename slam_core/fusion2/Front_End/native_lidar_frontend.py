"""Native C++ LiDAR front-end wrapper."""
from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np

import fusion_core as fc
import hector.config as hcfg
from carto.local_slam.imu_extrapolation import imu_rows_to_samples
from slam_core.common.types import Pose2
from slam_core.dataio.imu_csv import read_imu_csv


def _config_from_hector_profile(dataset_name: str, matcher_kind: str,
                                kf_min_dist_m: float, kf_min_angle_rad: float,
                                kf_min_dt_s: float) -> "fc.LidarFrontendConfig":
    # Copy the active hector profile into the C++ front-end config.
    hcfg._apply_profile(dataset_name)
    c = hcfg
    cfg = fc.LidarFrontendConfig()
    cfg.matcher = (fc.LidarMatcherKind.SCAN_TO_SUBMAP
                   if matcher_kind == "scan_to_submap"
                   else fc.LidarMatcherKind.SCAN_TO_MAP)

    cfg.voxel.enabled = bool(c.VOXEL_FILTER_ENABLED)
    cfg.voxel.fixed_size = float(c.VOXEL_FIXED_SIZE)
    cfg.voxel.adaptive_max_size = float(c.VOXEL_ADAPTIVE_MAX_SIZE)
    cfg.voxel.adaptive_min_points = int(c.VOXEL_ADAPTIVE_MIN_POINTS)
    cfg.voxel.adaptive_iters = int(c.VOXEL_ADAPTIVE_ITERS)

    cfg.extrap.max_dt = float(c.EXTRAP_MAX_DT)
    cfg.extrap.init_vxy = float(c.EXTRAP_INIT_VXY)
    cfg.extrap.init_wz = float(c.EXTRAP_INIT_WZ)
    cfg.extrap.use_imu = True
    cfg.extrap.imu_yaw_correction_alpha = float(
        getattr(c, "IMU_YAW_CORRECTION_ALPHA", 0.02))

    cfg.submap_builder.submap_size_m = float(c.SUBMAP_SIZE_METERS)
    cfg.submap_builder.resolution = float(c.SUBMAP_RESOLUTION)
    cfg.submap_builder.scans_per_submap = int(c.SCANS_PER_SUBMAP)
    cfg.submap_builder.l0 = float(c.L0)
    cfg.submap_builder.l_occ = float(c.L_OCC)
    cfg.submap_builder.l_free = float(c.L_FREE)
    cfg.submap_builder.l_min = float(c.L_MIN)
    cfg.submap_builder.l_max = float(c.L_MAX)

    sm = cfg.submap_matcher
    # Scan-to-submap search and refinement parameters.
    sm.min_score = float(c.SUBMAP_MIN_SCORE)
    sm.min_valid = int(c.SUBMAP_MIN_VALID)
    sm.precomp_levels = int(c.SUBMAP_PRECOMP_LEVELS)
    sm.max_match_points = int(c.SUBMAP_MAX_MATCH_POINTS)
    sm.max_refine_points = int(c.SUBMAP_MAX_REFINE_POINTS)
    sm.refine_min_points = int(c.SUBMAP_REFINE_MIN_POINTS)
    sm.refine_w_trans = float(c.SUBMAP_REFINE_W_TRANS)
    sm.refine_w_rot = float(c.SUBMAP_REFINE_W_ROT)
    sm.refine_iters = int(c.SUBMAP_REFINE_ITERS)
    sm.refine_damping = float(c.SUBMAP_REFINE_DAMPING)
    sm.refine_step_clip_xy = float(c.SUBMAP_REFINE_STEP_CLIP_XY)
    sm.refine_step_clip_th = float(np.deg2rad(c.SUBMAP_REFINE_STEP_CLIP_TH_DEG))
    sm.coarse.xy_window = float(c.SUBMAP_COARSE_XY_WINDOW)
    sm.coarse.theta_window = float(c.SUBMAP_COARSE_TH_WINDOW)
    sm.coarse.xy_step = float(c.SUBMAP_COARSE_XY_STEP)
    sm.coarse.theta_step = float(c.SUBMAP_COARSE_TH_STEP)
    sm.coarse.level = int(c.SUBMAP_COARSE_LEVEL)
    sm.fine.xy_window = float(c.SUBMAP_FINE_XY_WINDOW)
    sm.fine.theta_window = float(c.SUBMAP_FINE_TH_WINDOW)
    sm.fine.xy_step = float(c.SUBMAP_FINE_XY_STEP)
    sm.fine.theta_step = float(c.SUBMAP_FINE_TH_STEP)
    sm.fine.level = int(c.SUBMAP_FINE_LEVEL)

    mm = cfg.map_matcher
    # Scan-to-map occupancy pyramid and correspondence parameters.
    mm.base_res = float(c.MAP_RESOLUTION)
    mm.size_m = float(c.MAP_SIZE_METERS)
    mm.num_levels = int(c.PYRAMID_LEVELS)
    mm.l0 = float(c.L0)
    mm.l_occ = float(c.L_OCC)
    mm.l_free = float(c.L_FREE)
    mm.l_min = float(c.L_MIN)
    mm.l_max = float(c.L_MAX)
    mm.ray_steps = int(c.RAY_STEPS)
    mm.bootstrap_scans = max(1, int(getattr(c, "N_BOOTSTRAP_SCANS", 1)))
    mm.gn_iters_per_level = [int(v) for v in c.GN_ITERS_PER_LEVEL]
    mm.gn_damping = float(c.GN_DAMPING)
    mm.min_points = int(c.CORR_MAP_MIN_POINTS)
    mm.min_score = float(c.CORR_MAP_MIN_SCORE)
    mm.step_clip_xy = float(c.CORR_MAP_STEP_CLIP_XY)
    mm.step_clip_th = float(np.deg2rad(c.GN_STEP_CLIP_TH_DEG))
    mm.map_update_every = max(1, int(getattr(c, "MAP_UPDATE_EVERY", 1)))

    cfg.keyframe.min_dist_m = float(kf_min_dist_m)
    cfg.keyframe.min_angle_rad = float(kf_min_angle_rad)
    cfg.keyframe.min_dt_s = float(kf_min_dt_s)
    return cfg


class NativeLidarFrontend:
    """Thin Python adapter around fusion_core.NativeLidarFrontend."""

    def __init__(self, kind: str = "native_s2s", dataset_name: str = "lab_hybrid",
                 imu_path: Optional[str] = None, kf_min_dist_m: float = 0.25,
                 kf_min_angle_rad: float = math.radians(12.0),
                 kf_min_dt_s: float = 2.0, imu_samples=None):
        matcher_kind = "scan_to_submap" if kind == "native_s2s" else "scan_to_map"
        self.matcher_kind = matcher_kind
        cfg = _config_from_hector_profile(dataset_name, matcher_kind,
                                          kf_min_dist_m, kf_min_angle_rad,
                                          kf_min_dt_s)
        self._fe = fc.NativeLidarFrontend(cfg)

        # live (growing) IMU list injected for online/ROS takes precedence over csv
        self._imu_idx = 0
        if imu_samples is not None:
            self._imu = imu_samples
        elif imu_path and os.path.exists(imu_path):
            self._imu = imu_rows_to_samples(read_imu_csv(imu_path))
        else:
            self._imu = []

        self.fallback_count = 0
        self.last_score = float("nan")    # coarse match score of the most recent scan
        self.process_ms: List[float] = []

    def process(self, t: float, scan_xy: np.ndarray) -> Tuple[Pose2, np.ndarray, bool]:
        """Track one scan and expose the same return contract as LidarFrontend."""
        batch = []
        while self._imu_idx < len(self._imu) and self._imu[self._imu_idx][0] <= t:
            batch.append(self._imu[self._imu_idx])
            self._imu_idx += 1
        imu = np.asarray(batch, dtype=np.float64) if batch else None

        r = self._fe.process(np.asarray(scan_xy, dtype=np.float64), float(t),
                             imu_samples=imu)
        if r.fallback:
            self.fallback_count += 1
        self.last_score = float(r.score)
        self.process_ms.append(r.process_ms)
        # Return filtered points for diagnostics; raw scans are stored by callers.
        pts = np.ascontiguousarray(self._fe.last_filtered_points(), dtype=np.float32)
        return Pose2(r.pose.x, r.pose.y, r.pose.theta), pts, bool(r.is_keyframe)
