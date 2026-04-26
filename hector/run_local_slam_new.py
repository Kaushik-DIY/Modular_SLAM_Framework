import argparse
import os
import numpy as np

from slam_core.common.types import Pose2
from slam_core.dataio.dataset_catalog import load_dataset_scans

import hector.config as cfg

from carto.local_slam.range_to_points import ranges_to_points
from carto.local_slam.pose_extrapolator import PoseExtrapolatorCV

from slam_core.matching.core import MatcherManager
from slam_core.matching.preprocessing import PointCloudProcessor, PointCloudProcessorConfig
from slam_core.matching.scan_to_submap import (
    SubmapBuilder2D,
    ScanToSubmapMatcher as _NewScanToSubmapMatcher,  # PyCeres-backed (not used here)
)
# The old pure-numpy matcher accepts a corr_params dict and has no PyCeres dependency.
# It is the correct backend for the Hector runner's scan_to_submap mode.
from slam_core.matching.scan_to_submap_old import ScanToSubmapMatcher as OldScanToSubmapMatcher
from slam_core.matching.scan_to_map import ScanToMapMatcher

from hector.adapter import (
    HectorLocalSlamAdapter,
    make_motion_filter_from_expected_velocity,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hector SLAM — local SLAM runner (all datasets)"
    )
    p.add_argument(
        "--dataset",
        choices=list(cfg._PROFILES.keys()),
        default=None,
        help="Dataset to process (overrides DATASET_NAME in config.py)",
    )
    p.add_argument(
        "--scan-variant",
        choices=["raw", "360"],
        default=None,
        dest="scan_variant",
        help="lab_run_2 scan variant: 'raw' (909 beams) or '360' (default: from config)",
    )
    p.add_argument(
        "--max-scans",
        type=int,
        default=None,
        dest="max_scans",
        help="Cap number of scans processed (overrides MAX_SCANS in config.py)",
    )
    p.add_argument(
        "--matcher",
        choices=["scan_to_submap", "scan_to_map"],
        default=None,
        help="Matcher type (overrides MATCHER_TYPE in config.py)",
    )
    return p.parse_args()


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    """Push CLI arguments into the config module so all cfg.XXX reads are consistent."""
    if args.dataset is not None:
        cfg.DATASET_NAME = args.dataset
        cfg._apply_profile(args.dataset)
    if args.scan_variant is not None:
        cfg.DATASET_SCAN_VARIANT = args.scan_variant
    if args.max_scans is not None:
        cfg.MAX_SCANS = args.max_scans
    if args.matcher is not None:
        cfg.MATCHER_TYPE = args.matcher


def _format_delta(delta) -> str:
    if delta is None:
        return "d=None"
    arr = np.asarray(delta, dtype=float).reshape(-1)
    if arr.shape[0] < 3:
        return "d=None"
    return f"d=({arr[0]:.3f},{arr[1]:.3f},{arr[2]:.3f})"


def _format_inliers(inliers) -> str:
    if inliers is None:
        return "inl=None"
    return f"inl={int(inliers)}"


def _resolve_initial_pose(profile, first_scan) -> Pose2:
    if profile.has_odom and first_scan.get("odom") is not None:
        return Pose2(*first_scan["odom"])
    return Pose2(
        cfg.INITIAL_POSE_X,
        cfg.INITIAL_POSE_Y,
        cfg.INITIAL_POSE_THETA,
    )


def main():
    args = _parse_args()
    _apply_cli_overrides(args)

    profile, scans = load_dataset_scans(
        cfg.DATASET_NAME,
        scan_variant=cfg.DATASET_SCAN_VARIANT,
    )

    if not scans:
        raise RuntimeError(f"No scans loaded from {profile.scan_path}")

    use_extrapolator = getattr(cfg, "USE_EXTRAPOLATOR", True)

    print("=" * 60)
    print(f"Dataset      : {cfg.DATASET_NAME}")
    print(f"Scan variant : {cfg.DATASET_SCAN_VARIANT}")
    print(f"Scan file    : {profile.scan_path}")
    print(
        f"Geometry     : beams={profile.num_beams}"
        f"  angle=[{np.rad2deg(profile.angle_min):.1f}°, {np.rad2deg(profile.angle_max):.1f}°]"
        f"  range=[{profile.range_min:.2f}, {profile.range_max:.2f}] m"
        f"  has_odom={profile.has_odom}"
    )
    if profile.imu_path is not None:
        print(f"IMU file     : {profile.imu_path}")
    print(f"Total scans  : {len(scans)}")
    print(f"Matcher      : {cfg.MATCHER_TYPE}")
    print(f"Extrapolator : {'ON' if use_extrapolator else 'OFF (raw odom / last pose)'}")
    print(f"Odom alpha   : {cfg.ODOM_ALPHA}")
    print(f"Voxel filter : {cfg.VOXEL_FILTER_ENABLED}")
    print("=" * 60)

    matcher_type = cfg.MATCHER_TYPE
    max_scans = cfg.MAX_SCANS
    verbose_every = cfg.VERBOSE_EVERY

    if max_scans is not None:
        scans = scans[:int(max_scans)]
        print(f"Using first {len(scans)} scans (capped by MAX_SCANS={max_scans})")

    # -------------------------------------------------------
    # Shared point-cloud preprocessing
    # -------------------------------------------------------
    # Voxel filtering is enabled per-dataset via VOXEL_FILTER_ENABLED:
    #   lab_run_2 raw: 909 beams → filter down to ~200 useful points
    #   fr079 / intel: 360 / 180 beams → no filtering needed
    point_processor = PointCloudProcessor(
        PointCloudProcessorConfig(
            fixed_voxel_size=cfg.VOXEL_FIXED_SIZE,
            adaptive_voxel_max_size=cfg.VOXEL_ADAPTIVE_MAX_SIZE,
            adaptive_min_num_points=cfg.VOXEL_ADAPTIVE_MIN_POINTS,
            adaptive_num_iterations=cfg.VOXEL_ADAPTIVE_ITERS,
            enabled=cfg.VOXEL_FILTER_ENABLED,
        )
    )

    # -------------------------------------------------------
    # Shared matcher dependencies
    # -------------------------------------------------------
    submaps = SubmapBuilder2D(
        submap_size_m=cfg.SUBMAP_SIZE_METERS,
        resolution=cfg.SUBMAP_RESOLUTION,
        scans_per_submap=cfg.SCANS_PER_SUBMAP,
        ray_steps=cfg.RAY_STEPS,
        l0=cfg.L0,
        l_occ=cfg.L_OCC,
        l_free=cfg.L_FREE,
        l_min=cfg.L_MIN,
        l_max=cfg.L_MAX,
    )

    # scan_to_map GN params — jump gates removed (not in original Hector SLAM)
    corr_params_map = dict(
        gn_iters_per_level=cfg.GN_ITERS_PER_LEVEL,
        gn_damping=cfg.GN_DAMPING,
        min_points=cfg.CORR_MAP_MIN_POINTS,
        min_inliers_accept=cfg.CORR_MAP_MIN_INLIERS,
        min_score=cfg.CORR_MAP_MIN_SCORE,
        step_clip_xy=cfg.CORR_MAP_STEP_CLIP_XY,
        step_clip_th=np.deg2rad(1.0),
    )

    if matcher_type == "scan_to_submap":
        corr_params_submap = dict(
            min_score=cfg.SUBMAP_MIN_SCORE,
            max_match_points=cfg.SUBMAP_MAX_MATCH_POINTS,
            max_refine_points=cfg.SUBMAP_MAX_REFINE_POINTS,
            min_valid=cfg.SUBMAP_MIN_VALID,
            precomp_levels=3,
            coarse_level=2,
            coarse_xy_window=cfg.SUBMAP_COARSE_XY_WINDOW,
            coarse_th_window=cfg.SUBMAP_COARSE_TH_WINDOW,
            coarse_xy_step=cfg.SUBMAP_COARSE_XY_STEP,
            coarse_th_step=cfg.SUBMAP_COARSE_TH_STEP,
            fine_level=0,
            fine_xy_window=cfg.SUBMAP_FINE_XY_WINDOW,
            fine_th_window=cfg.SUBMAP_FINE_TH_WINDOW,
            fine_xy_step=cfg.SUBMAP_FINE_XY_STEP,
            fine_th_step=cfg.SUBMAP_FINE_TH_STEP,
            do_refine=True,
            refine_min_points=cfg.SUBMAP_REFINE_MIN_POINTS,
            refine_w_trans=cfg.SUBMAP_REFINE_W_TRANS,
            refine_w_rot=cfg.SUBMAP_REFINE_W_ROT,
            refine_iters=12,
            refine_damping=1e-3,
            refine_step_clip_xy=0.10,
            refine_step_clip_th=float(np.deg2rad(5.0)),
        )
        matcher = OldScanToSubmapMatcher(
            submap_builder=submaps,
            corr_params=corr_params_submap,
        )

    elif matcher_type == "scan_to_map":
        map_params = dict(
            base_res=cfg.MAP_RESOLUTION,
            size_m=cfg.MAP_SIZE_METERS,
            num_levels=cfg.PYRAMID_LEVELS,
            l0=cfg.L0,
            l_min=cfg.L_MIN,
            l_max=cfg.L_MAX,
            l_free=cfg.L_FREE,
            l_occ=cfg.L_OCC,
            ray_steps=cfg.RAY_STEPS,
        )

        matcher = ScanToMapMatcher(
            map_params=map_params,
            corr_params=corr_params_map,
        )

    else:
        raise ValueError(f"Unsupported MATCHER_TYPE: {matcher_type!r}")

    matcher_manager = MatcherManager(
        active_matcher=matcher,
        rolling_buffer_size=30,
        min_buffer_for_switch=20,
    )

    # -------------------------------------------------------
    # Extrapolator
    # -------------------------------------------------------
    extrap = PoseExtrapolatorCV(
        max_dt=cfg.EXTRAP_MAX_DT,
        init_vxy=cfg.EXTRAP_INIT_VXY,
        init_wz=cfg.EXTRAP_INIT_WZ,
    )

    # -------------------------------------------------------
    # Motion filter
    # -------------------------------------------------------
    motion_params = make_motion_filter_from_expected_velocity(
        target_insert_period_s=cfg.TARGET_INSERT_PERIOD_S,
        v_expected_mps=cfg.V_EXPECTED_MPS,
        w_expected_rps=cfg.W_EXPECTED_RPS,
    )
    print(
        "MotionFilter thresholds:"
        f"  time={motion_params.max_time_seconds:.3f}s"
        f"  dist={motion_params.max_distance_meters:.3f}m"
        f"  angle={np.rad2deg(motion_params.max_angle_radians):.2f}deg"
    )

    # -------------------------------------------------------
    # Adapter
    # -------------------------------------------------------
    adapter = HectorLocalSlamAdapter(
        matcher_manager=matcher_manager,
        extrapolator=extrap,
        motion_params=(None if matcher_type == "scan_to_map" else motion_params),
        use_extrapolator=use_extrapolator,
    )

    # -------------------------------------------------------
    # Initialize extrapolator from first scan
    # -------------------------------------------------------
    first = scans[0]
    pose0 = _resolve_initial_pose(profile, first)
    adapter.initialize_extrapolator(float(first["t"]), pose0)

    # -------------------------------------------------------
    # Output paths
    # -------------------------------------------------------
    os.makedirs("hector_outputs", exist_ok=True)

    dataset_tag = cfg.DATASET_NAME
    if cfg.DATASET_NAME == "lab_run_2":
        dataset_tag = f"lab_run_2_{cfg.DATASET_SCAN_VARIANT}"

    traj_path = f"hector_outputs/trajectory_{dataset_tag}_{matcher_type}_{len(scans)}.txt"
    meta_path = f"hector_outputs/trajectory_{dataset_tag}_{matcher_type}_{len(scans)}_debug.txt"

    # -------------------------------------------------------
    # Main scan loop
    # -------------------------------------------------------
    _prev_submap_count   = 0
    _prev_finished_count = 0

    with open(traj_path, "w") as f_traj, open(meta_path, "w") as f_meta:
        f_meta.write(f"# dataset_name={cfg.DATASET_NAME}\n")
        f_meta.write(f"# dataset_scan_variant={cfg.DATASET_SCAN_VARIANT}\n")
        f_meta.write(f"# dataset_scan_file={profile.scan_path}\n")
        f_meta.write(f"# matcher_type={matcher_type}\n")
        f_meta.write(
            "k t x y theta score inliers dx dy dtheta do_insert did_insert\n"
        )

        for k, s in enumerate(scans):
            t = float(s["t"])
            odom_raw = s.get("odom")
            odom = Pose2(*odom_raw) if odom_raw is not None else None

            pts_raw = ranges_to_points(
                s["ranges"],
                profile.angle_min,
                profile.angle_inc,
                max(cfg.LIDAR_MIN_RANGE, profile.range_min),
                profile.range_max,
                stride=cfg.BEAM_STRIDE,
            )
            pts, proc_debug = point_processor.process(pts_raw)

            pose, result, do_insert, did_insert = adapter.process_scan(
                k=k,
                t=t,
                scan_points_local=pts,
                odom_pose_world=odom,
                odom_alpha=(cfg.ODOM_ALPHA if odom is not None else 0.0),
            )

            # Preserve the actual matcher score even on failure so diagnostics
            # show what the score really was (not a misleading -1.0).
            score = float(result.score)
            mode  = "MATCH" if result.success else "FALLBACK"

            # Print a one-line diagnosis for every non-bootstrap FALLBACK.
            # Only score and inlier gates remain (jump gates removed per original).
            if not result.success and matcher_type == "scan_to_map":
                dbg = result.debug or {}
                reason = dbg.get("reason", "")
                if reason not in ("bootstrap_seeding", "map_not_initialized"):
                    _sc   = dbg.get("score", score)
                    _msc  = dbg.get("min_score", cfg.CORR_MAP_MIN_SCORE)
                    _inl  = dbg.get("valid_points_finest", -1)
                    _minl = dbg.get("min_inliers_accept", cfg.CORR_MAP_MIN_INLIERS)
                    _tj   = dbg.get("trans_jump", 0.0)
                    _rj   = dbg.get("rot_jump_deg", 0.0)
                    _flags = []
                    if _sc  < _msc:  _flags.append(f"score={_sc:.3f}<{_msc}")
                    if _inl < _minl: _flags.append(f"inliers={_inl}<{_minl}")
                    print(
                        f"  !! FALLBACK k={k:5d}  score={_sc:.3f}"
                        + (f"  inl={_inl}" if _inl >= 0 else "")
                        + (f"  tj={_tj:.3f}m" if _tj > 0 else "")
                        + (f"  rj={_rj:.1f}°" if _rj > 0 else "")
                        + ("  [" + "  ".join(_flags) + "]" if _flags else "")
                    )

            # Submap lifecycle events
            if matcher_type == "scan_to_submap":
                sb = getattr(matcher_manager.active_matcher, "submap_builder", None)
                if sb is not None:
                    n_active   = len(sb.get_active_submaps())
                    n_finished = len(sb.get_finished_submaps())
                    n_total    = n_active + n_finished
                    if n_total > _prev_submap_count:
                        for sid in range(_prev_submap_count, n_total):
                            print(
                                f"  ┌─── SUBMAP #{sid} CREATED at scan k={k}  "
                                f"(pose=({pose.x:.2f},{pose.y:.2f},{pose.theta:.2f}))  "
                                f"active={n_active}  finished={n_finished}"
                            )
                        _prev_submap_count = n_total
                    if n_finished > _prev_finished_count:
                        for fid in sb.consume_newly_finished_ids():
                            sm = sb.get_submap_by_id(fid)
                            print(
                                f"  └─── SUBMAP #{fid} FINISHED at scan k={k}  "
                                f"({sm.num_inserted} scans inserted)  "
                                f"active={n_active}  finished={n_finished}"
                            )
                        _prev_finished_count = n_finished

            delta   = getattr(result, "refine_delta", None)
            inliers = getattr(result, "inliers", None)

            if delta is None:
                dx = dy = dtheta = np.nan
            else:
                arr = np.asarray(delta, dtype=float).reshape(-1)
                if arr.shape[0] >= 3:
                    dx, dy, dtheta = float(arr[0]), float(arr[1]), float(arr[2])
                else:
                    dx = dy = dtheta = np.nan

            f_traj.write(
                f"{t:.6f} {pose.x:.6f} {pose.y:.6f} {pose.theta:.6f} {score:.6f}\n"
            )
            f_meta.write(
                f"{k} {t:.6f} {pose.x:.6f} {pose.y:.6f} {pose.theta:.6f} "
                f"{score:.6f} "
                f"{-1 if inliers is None else int(inliers)} "
                f"{dx:.6f} {dy:.6f} {dtheta:.6f} "
                f"{int(do_insert)} {int(did_insert)}\n"
            )

            if (k % verbose_every) == 0:
                motion_debug = adapter.last_motion_debug() or {}
                print(
                    f"k={k:5d} {mode:<8s}"
                    f"  pose=({pose.x:.2f},{pose.y:.2f},{np.rad2deg(pose.theta):.1f}°)"
                    f"  score={score:.3f}"
                    f"  {_format_delta(delta)}"
                    f"  {_format_inliers(inliers)}"
                    f"  raw={proc_debug['n_input']}"
                    f"  pts={proc_debug['n_after_adaptive']}"
                    f"  di={int(did_insert)}"
                )

    print(f"\nWrote trajectory : {traj_path}")
    print(f"Wrote debug log  : {meta_path}")

    # Final submap summary
    if matcher_type == "scan_to_submap":
        sb = getattr(matcher_manager.active_matcher, "submap_builder", None)
        if sb is not None:
            n_active   = len(sb.get_active_submaps())
            n_finished = len(sb.get_finished_submaps())
            print(f"\n{'=' * 60}")
            print(
                f"  Submap summary: {n_active + n_finished} total  "
                f"({n_finished} finished, {n_active} still active)"
            )
            for sm in sb.get_finished_submaps() + sb.get_active_submaps():
                label = "FINISHED" if sm.finished else "ACTIVE  "
                print(
                    f"    submap #{sm.id:2d}  {label}"
                    f"  scans={sm.num_inserted:4d}"
                    f"  origin=({sm.pose_world.x:.2f}, {sm.pose_world.y:.2f})"
                )
            print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
