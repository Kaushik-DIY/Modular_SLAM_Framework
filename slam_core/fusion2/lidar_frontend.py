"""LiDAR front-end service for fusion v2.

Reuses the proven hector scan_to_submap stack (matcher + extrapolator + IMU +
voxel preprocessing) exactly as `hector/run_local_slam_new.py` wires it, but
with the pose-graph backend DISABLED — graph ownership belongs to fusion_core
(CLAUDE.md §2.7). Emits per-scan odometry poses and keyframe decisions.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np

import hector.config as hcfg
from carto.local_slam.imu_extrapolation import imu_rows_to_samples
from carto.local_slam.pose_extrapolator import PoseExtrapolatorCV
from slam_core.matching.preprocessing import (
    PointCloudProcessor,
    PointCloudProcessorConfig,
)
from hector.adapter import HectorLocalSlamAdapter
from slam_core.common.types import Pose2
from slam_core.dataio.imu_csv import read_imu_csv
from slam_core.matching.core import MatcherManager
from slam_core.matching.scan_to_submap import ScanToSubmapMatcher, SubmapBuilder2D
from slam_core.matching.scan_to_submap.types import (
    ScanToSubmapBackendConfig,
    SubmapSearchWindow,
)


class LidarFrontend:
    """Per-scan LiDAR odometry (scan_to_submap OR scan_to_map) with
    motion-filter keyframing. `matcher_kind` selects the local-mapping
    variant; loop verification is front-end-agnostic (candidate-local grids
    in fusion_core), so any verifier composes with either kind."""

    def __init__(self, dataset_name: str = "lab_hybrid", imu_path: Optional[str] = None,
                 kf_min_dist_m: float = 0.25, kf_min_angle_rad: float = math.radians(12.0),
                 kf_min_dt_s: float = 2.0, use_vectorized_search: bool = True,
                 matcher_kind: str = "scan_to_submap"):
        hcfg._apply_profile(dataset_name)
        cfg = hcfg
        if matcher_kind not in ("scan_to_submap", "scan_to_map"):
            raise ValueError(f"unknown matcher_kind {matcher_kind!r}")
        self.matcher_kind = matcher_kind

        self.point_processor = PointCloudProcessor(
            PointCloudProcessorConfig(
                fixed_voxel_size=cfg.VOXEL_FIXED_SIZE,
                adaptive_voxel_max_size=cfg.VOXEL_ADAPTIVE_MAX_SIZE,
                adaptive_min_num_points=cfg.VOXEL_ADAPTIVE_MIN_POINTS,
                adaptive_num_iterations=cfg.VOXEL_ADAPTIVE_ITERS,
                enabled=cfg.VOXEL_FILTER_ENABLED,
            )
        )
        if matcher_kind == "scan_to_submap":
            matcher = self._build_s2s_matcher(cfg, use_vectorized_search)
        else:
            matcher = self._build_s2m_matcher(cfg)
        manager = MatcherManager(
            active_matcher=matcher,
            rolling_buffer_size=cfg.ROLLING_BUFFER_SIZE,
            min_buffer_for_switch=cfg.MIN_BUFFER_FOR_SWITCH,
        )
        self.extrap = PoseExtrapolatorCV(
            max_dt=cfg.EXTRAP_MAX_DT,
            init_vxy=cfg.EXTRAP_INIT_VXY,
            init_wz=cfg.EXTRAP_INIT_WZ,
            use_imu=True,
            imu_yaw_correction_alpha=float(getattr(cfg, "IMU_YAW_CORRECTION_ALPHA", 0.02)),
        )
        # No pose_graph / global_slam: fusion_core owns global consistency.
        self.adapter = HectorLocalSlamAdapter(
            matcher_manager=manager,
            extrapolator=self.extrap,
            use_extrapolator=True,
            pose_graph=None,
            global_slam=None,
        )

        self._imu: List[Tuple[float, float, float]] = []
        self._imu_idx = 0
        if imu_path and os.path.exists(imu_path):
            self._imu = imu_rows_to_samples(read_imu_csv(imu_path))

        self.kf_min_dist = float(kf_min_dist_m)
        self.kf_min_angle = float(kf_min_angle_rad)
        self.kf_min_dt = float(kf_min_dt_s)
        self._last_kf_pose: Optional[Pose2] = None
        self._last_kf_t: Optional[float] = None
        self._k = -1
        self.fallback_count = 0   # scans where matching failed -> IMU/CV prediction

    def _build_s2s_matcher(self, cfg, use_vectorized_search: bool):
        self.submaps = SubmapBuilder2D(
            submap_size_m=cfg.SUBMAP_SIZE_METERS,
            resolution=cfg.SUBMAP_RESOLUTION,
            scans_per_submap=cfg.SCANS_PER_SUBMAP,
            ray_steps=cfg.RAY_STEPS,
            l0=cfg.L0, l_occ=cfg.L_OCC, l_free=cfg.L_FREE,
            l_min=cfg.L_MIN, l_max=cfg.L_MAX,
        )
        backend_cfg = ScanToSubmapBackendConfig(
            min_score=cfg.SUBMAP_MIN_SCORE,
            min_valid=cfg.SUBMAP_MIN_VALID,
            max_match_points=cfg.SUBMAP_MAX_MATCH_POINTS,
            max_refine_points=cfg.SUBMAP_MAX_REFINE_POINTS,
            refine_min_points=cfg.SUBMAP_REFINE_MIN_POINTS,
            refine_w_trans=cfg.SUBMAP_REFINE_W_TRANS,
            refine_w_rot=cfg.SUBMAP_REFINE_W_ROT,
            refine_iters=cfg.SUBMAP_REFINE_ITERS,
            refine_damping=cfg.SUBMAP_REFINE_DAMPING,
            refine_step_clip_xy=cfg.SUBMAP_REFINE_STEP_CLIP_XY,
            refine_step_clip_th=float(np.deg2rad(cfg.SUBMAP_REFINE_STEP_CLIP_TH_DEG)),
            use_vectorized_search=bool(use_vectorized_search),
            coarse=SubmapSearchWindow(
                xy_window=cfg.SUBMAP_COARSE_XY_WINDOW,
                theta_window=cfg.SUBMAP_COARSE_TH_WINDOW,
                xy_step=cfg.SUBMAP_COARSE_XY_STEP,
                theta_step=cfg.SUBMAP_COARSE_TH_STEP,
                level=cfg.SUBMAP_COARSE_LEVEL,
            ),
            fine=SubmapSearchWindow(
                xy_window=cfg.SUBMAP_FINE_XY_WINDOW,
                theta_window=cfg.SUBMAP_FINE_TH_WINDOW,
                xy_step=cfg.SUBMAP_FINE_XY_STEP,
                theta_step=cfg.SUBMAP_FINE_TH_STEP,
                level=cfg.SUBMAP_FINE_LEVEL,
            ),
        )
        return ScanToSubmapMatcher(submap_builder=self.submaps,
                                   backend_config=backend_cfg)

    def _build_s2m_matcher(self, cfg):
        # Mirrors hector/run_local_slam_new.py's scan_to_map wiring.
        from slam_core.matching.scan_to_map import ScanToMapMatcher
        self.submaps = None
        map_params = dict(
            base_res=cfg.MAP_RESOLUTION,
            size_m=cfg.MAP_SIZE_METERS,
            num_levels=cfg.PYRAMID_LEVELS,
            l0=cfg.L0, l_min=cfg.L_MIN, l_max=cfg.L_MAX,
            l_free=cfg.L_FREE, l_occ=cfg.L_OCC,
            ray_steps=cfg.RAY_STEPS,
        )
        corr_params = dict(
            gn_iters_per_level=cfg.GN_ITERS_PER_LEVEL,
            gn_damping=cfg.GN_DAMPING,
            min_points=cfg.CORR_MAP_MIN_POINTS,
            min_inliers_accept=cfg.CORR_MAP_MIN_INLIERS,
            min_score=cfg.CORR_MAP_MIN_SCORE,
            step_clip_xy=cfg.CORR_MAP_STEP_CLIP_XY,
            step_clip_th=float(np.deg2rad(cfg.GN_STEP_CLIP_TH_DEG)),
        )
        return ScanToMapMatcher(map_params=map_params, corr_params=corr_params)

    def process(self, t: float, scan_xy: np.ndarray) -> Tuple[Pose2, np.ndarray, bool]:
        """Track one scan. Returns (front-end pose, voxel-filtered points,
        is_keyframe)."""
        self._k += 1
        while self._imu_idx < len(self._imu) and self._imu[self._imu_idx][0] <= t:
            ts_i, wz_i, yaw_i = self._imu[self._imu_idx]
            self.extrap.add_imu(ts_i, wz_i, yaw_i)
            self._imu_idx += 1

        pts, _ = self.point_processor.process(np.asarray(scan_xy, dtype=float))
        pose, _result, _do_insert, _did = self.adapter.process_scan(
            k=self._k, t=t, scan_points_local=pts,
            odom_pose_world=None, odom_alpha=0.0,
        )
        if _result is not None and not getattr(_result, "success", True):
            # matcher fell back to the (IMU-informed) extrapolator prediction
            self.fallback_count += 1

        is_kf = False
        if self._last_kf_pose is None:
            is_kf = True
        else:
            dx = pose.x - self._last_kf_pose.x
            dy = pose.y - self._last_kf_pose.y
            dth = abs(math.atan2(math.sin(pose.theta - self._last_kf_pose.theta),
                                 math.cos(pose.theta - self._last_kf_pose.theta)))
            if (math.hypot(dx, dy) >= self.kf_min_dist or dth >= self.kf_min_angle
                    or (t - self._last_kf_t) >= self.kf_min_dt):
                is_kf = True
        if is_kf:
            self._last_kf_pose = Pose2(pose.x, pose.y, pose.theta)
            self._last_kf_t = t
        return pose, np.ascontiguousarray(pts, dtype=np.float32), is_kf


def make_lidar_frontend(kind: str, dataset_name: str = "lab_hybrid",
                        imu_path: Optional[str] = None,
                        kf_min_dist_m: float = 0.25,
                        kf_min_angle_rad: float = math.radians(12.0),
                        kf_min_dt_s: float = 2.0):
    """Front-end factory for the fusion runner (V4.3).

    kind: native_s2s | native_s2m (C++, V4.4) | legacy_s2s | legacy_s2m (Python).
    All variants share the `process(t, scan_xy) -> (Pose2, pts_f32, is_kf)`
    contract the runner consumes.
    """
    kf = dict(kf_min_dist_m=kf_min_dist_m, kf_min_angle_rad=kf_min_angle_rad,
              kf_min_dt_s=kf_min_dt_s)
    if kind in ("legacy_s2s", "legacy_s2m"):
        matcher_kind = "scan_to_submap" if kind == "legacy_s2s" else "scan_to_map"
        return LidarFrontend(dataset_name=dataset_name, imu_path=imu_path,
                             matcher_kind=matcher_kind, **kf)
    if kind in ("native_s2s", "native_s2m"):
        from slam_core.fusion2.native_lidar_frontend import NativeLidarFrontend
        return NativeLidarFrontend(kind=kind, dataset_name=dataset_name,
                                   imu_path=imu_path, **kf)
    raise ValueError(f"unknown lidar front-end kind {kind!r} "
                     "(expected native_s2s|native_s2m|legacy_s2s|legacy_s2m)")
