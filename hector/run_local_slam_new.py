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
    profile, scans = load_dataset_scans(
        cfg.DATASET_NAME,
        scan_variant=cfg.DATASET_SCAN_VARIANT,
    )

    if not scans:
        raise RuntimeError(f"No scans loaded from {profile.scan_path}")

    print("Loaded scans:", len(scans))
    print("Dataset:", cfg.DATASET_NAME)
    print("Scan variant:", cfg.DATASET_SCAN_VARIANT)
    print("Scan file:", profile.scan_path)
    print(
        "Dataset geometry:",
        f"beams={profile.num_beams}",
        f"angle_min={profile.angle_min:.6f}",
        f"angle_max={profile.angle_max:.6f}",
        f"angle_inc={profile.angle_inc:.6f}",
        f"range=[{profile.range_min:.3f}, {profile.range_max:.3f}]",
        f"has_odom={profile.has_odom}",
    )
    if profile.imu_path is not None:
        print("IMU file:", profile.imu_path)

    matcher_type = cfg.MATCHER_TYPE
    max_scans = cfg.MAX_SCANS
    verbose_every = cfg.VERBOSE_EVERY

    print("Runner:", "Hector")
    print("Matcher_type", matcher_type)

    if max_scans is not None:
        scans = scans[:int(max_scans)]
        print("Using scans:", len(scans))

    # ------------------------------------------------
    # Shared point-cloud preprocessing
    # ------------------------------------------------
    # Enable voxel filtering for the lab dataset: 909 raw beams per scan is
    # more than the GN solver benefits from, and filtering removes motion blur
    # artefacts from beam clustering near close obstacles.
    point_processor = PointCloudProcessor(
        PointCloudProcessorConfig(
            fixed_voxel_size=0.03,
            adaptive_voxel_max_size=0.10,
            adaptive_min_num_points=200,
            adaptive_num_iterations=6,
            enabled=(cfg.DATASET_NAME == "lab_run_2"),
        )
    )

    # ------------------------------------------------
    # Shared matcher dependencies
    # ------------------------------------------------
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




    # scan_to_map correlation params tuned for lab_run_2:
    # - step_clip_xy raised to 0.15m: allows GN to correct larger misalignments
    #   (original 0.03m was too tight for a 10Hz lab scan with real motion)
    # - max_translation_jump raised to 0.5m: accommodates ~1 m/s walking speed
    # - min_inliers_accept raised to 60: with 909 beams this is a mild threshold
    # - gn_iters and damping now read from config for consistency
    corr_params_map = dict(
        gn_iters_per_level=cfg.GN_ITERS_PER_LEVEL,
        gn_damping=cfg.GN_DAMPING,
        min_points=60,
        min_inliers_accept=60,
        min_score=0.50,
        step_clip_xy=0.15,
        step_clip_th=np.deg2rad(5.0),
        max_translation_jump=0.50,
        max_rotation_jump=np.deg2rad(20.0),
    )

    if matcher_type == "scan_to_submap":
        # The old pure-numpy ScanToSubmapMatcher accepts a corr_params dict.
        # Parameters are tuned for lab_run_2:
        #   max_match_points raised from 60 -> 200: with voxel filtering we have
        #     ~204 pts/scan; 60 was far too sparse for the correlative stage.
        #   coarse_xy_step halved to 0.10 m (2 cells at 5cm res) for better init.
        #   min_valid raised to 30: a meaningful occupancy support floor.
        #   refine_w_trans = 0.1: loose translation prior so GN can move freely.
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
            base_res=cfg.MAP_RESOLUTION,          # use MAP_RESOLUTION, not SUBMAP_RESOLUTION
            size_m=cfg.MAP_SIZE_METERS,
            num_levels=cfg.PYRAMID_LEVELS,        # was hard-coded 4; now matches config (3)
            l0=cfg.L0,
            l_min=cfg.L_MIN,
            l_max=cfg.L_MAX,
            l_free=cfg.L_FREE,
            l_occ=cfg.L_OCC,
            ray_steps=cfg.RAY_STEPS,
            bootstrap_scans=cfg.N_BOOTSTRAP_SCANS,  # multi-scan map seeding
        )

        matcher = ScanToMapMatcher(
            map_params=map_params,
            corr_params=corr_params_map,
        )

    else:
        raise ValueError(f"Unsupported MATCHER_TYPE: {matcher_type}")

    matcher_manager = MatcherManager(
        active_matcher=matcher,
        rolling_buffer_size=30,
        min_buffer_for_switch=20,
    )

    # ------------------------------------------------
    # Extrapolator
    # ------------------------------------------------
    extrap = PoseExtrapolatorCV(
        max_dt=cfg.EXTRAP_MAX_DT,
        init_vxy=cfg.EXTRAP_INIT_VXY,
        init_wz=cfg.EXTRAP_INIT_WZ,
    )

    # ------------------------------------------------
    # Motion filter
    # ------------------------------------------------
    motion_params = make_motion_filter_from_expected_velocity(
        target_insert_period_s=cfg.TARGET_INSERT_PERIOD_S,
        v_expected_mps=cfg.V_EXPECTED_MPS,
        w_expected_rps=cfg.W_EXPECTED_RPS,
    )

    print(
        "MotionFilter thresholds:",
        f"time={motion_params.max_time_seconds:.3f}s",
        f"dist={motion_params.max_distance_meters:.3f}m",
        f"angle={np.rad2deg(motion_params.max_angle_radians):.2f}deg",
    )

    # ------------------------------------------------
    # Adapter
    # ------------------------------------------------
    adapter = HectorLocalSlamAdapter(
        matcher_manager=matcher_manager,
        extrapolator=extrap,
        motion_params=motion_params,
    )

    # ------------------------------------------------
    # Initialize extrapolator
    # ------------------------------------------------
    first = scans[0]
    pose0 = _resolve_initial_pose(profile, first)
    adapter.initialize_extrapolator(float(first["t"]), pose0)

    # ------------------------------------------------
    # Output
    # ------------------------------------------------
    os.makedirs("hector_outputs", exist_ok=True)

    dataset_tag = cfg.DATASET_NAME
    if cfg.DATASET_NAME == "lab_run_2":
        dataset_tag = f"{cfg.DATASET_NAME}_{cfg.DATASET_SCAN_VARIANT}"

    traj_path = f"hector_outputs/trajectory_{dataset_tag}_{matcher_type}_{len(scans)}.txt"
    meta_path = f"hector_outputs/trajectory_{dataset_tag}_{matcher_type}_{len(scans)}_debug.txt"

    with open(traj_path, "w") as f_traj, open(meta_path, "w") as f_meta:
        f_meta.write(f"# dataset_name={cfg.DATASET_NAME}\n")
        f_meta.write(f"# dataset_scan_variant={cfg.DATASET_SCAN_VARIANT}\n")
        f_meta.write(f"# dataset_scan_file={profile.scan_path}\n")
        f_meta.write(f"# matcher_type={matcher_type}\n")
        f_meta.write(
            "k t x y theta score inliers dx dy dtheta do_insert did_insert\n"
        )

        # Track submap lifecycle events (only relevant for scan_to_submap)
        _prev_submap_count = 0  # tracks total submaps created so far
        _prev_finished_count = 0

        for k, s in enumerate(scans):
            t = float(s["t"])
            odom_raw = s.get("odom")
            odom = Pose2(*odom_raw) if odom_raw is not None else None

            pts_raw = ranges_to_points(
                s["ranges"],
                profile.angle_min,
                profile.angle_inc,
                max(cfg.LIDAR_MIN_RANGE, profile.range_min),
                profile.range_max,   # use dataset sensor limit directly (16m for lab_run_2)
                stride=cfg.BEAM_STRIDE,
            )
            pts, proc_debug = point_processor.process(pts_raw)

            pose, result, do_insert, did_insert = adapter.process_scan(
                t=t,
                scan_points_local=pts,
                odom_pose_world=odom,
                odom_alpha=(cfg.ODOM_ALPHA if odom is not None else 0.0),
            )

            score = float(result.score) if result.success else -1.0
            mode = "MATCH" if result.success else "FALLBACK"

            # ── Submap lifecycle logging ──────────────────────────────────
            if matcher_type == "scan_to_submap":
                active_matcher = matcher_manager.active_matcher
                sb = getattr(active_matcher, 'submap_builder', None)
                if sb is not None:
                    n_active = len(sb.get_active_submaps())
                    n_finished = len(sb.get_finished_submaps())
                    n_total = n_active + n_finished
                    # Detect newly created submaps
                    if n_total > _prev_submap_count:
                        for sid in range(_prev_submap_count, n_total):
                            print(f"  ┌─── SUBMAP #{sid} CREATED at scan k={k}  "
                                  f"(pose=({pose.x:.2f},{pose.y:.2f},{pose.theta:.2f}))  "
                                  f"active={n_active}  finished={n_finished}")
                        _prev_submap_count = n_total
                    # Detect newly finished submaps
                    if n_finished > _prev_finished_count:
                        for fid in sb.consume_newly_finished_ids():
                            sm = sb.get_submap_by_id(fid)
                            print(f"  └─── SUBMAP #{fid} FINISHED at scan k={k}  "
                                  f"({sm.num_inserted} scans inserted)  "
                                  f"active={n_active}  finished={n_finished}")
                        _prev_finished_count = n_finished

            delta = getattr(result, "refine_delta", None)
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

                dmsg = _format_delta(delta)
                imsg = _format_inliers(inliers)

                print(
                    "motion:",
                    f"matcher={motion_debug.get('matcher_name', matcher_type)}",
                    f"dtrans={motion_debug.get('dtrans', 0.0):.3f}",
                    f"drot_deg={motion_debug.get('drot_deg', 0.0):.3f}",
                    f"dtime={motion_debug.get('dtime', 0.0):.3f}",
                    f"do_insert={motion_debug.get('do_insert', False)}",
                    f"did_insert={motion_debug.get('did_insert', False)}",
                )

                print(
                    "preprocess:",
                    f"raw={proc_debug['n_input']}",
                    f"fixed={proc_debug['n_after_fixed']}",
                    f"adaptive={proc_debug['n_after_adaptive']}",
                )

                print(
                    f"k={k} {mode} pose=({pose.x:.2f},{pose.y:.2f},{pose.theta:.2f}) "
                    f"score={score:.3f} {dmsg} {imsg}"
                )

    print("Wrote:", traj_path)
    print("Wrote:", meta_path)

    # Final submap summary
    if matcher_type == "scan_to_submap":
        sb = getattr(matcher_manager.active_matcher, 'submap_builder', None)
        if sb is not None:
            n_active = len(sb.get_active_submaps())
            n_finished = len(sb.get_finished_submaps())
            print(f"\n{'='*60}")
            print(f"  Submap summary: {n_active + n_finished} total  "
                  f"({n_finished} finished, {n_active} still active)")
            for sm in sb.get_finished_submaps() + sb.get_active_submaps():
                label = "FINISHED" if sm.finished else "ACTIVE  "
                print(f"    submap #{sm.id:2d}  {label}  "
                      f"scans={sm.num_inserted:4d}  "
                      f"origin=({sm.pose_world.x:.2f}, {sm.pose_world.y:.2f})")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()