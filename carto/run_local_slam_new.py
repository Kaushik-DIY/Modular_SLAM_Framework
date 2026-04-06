import os
import numpy as np

from slam_core.dataio.carmen import read_carmen_log
from slam_core.common.types import Pose2

from carto.config import (
    ANGLE_MIN,
    ANGLE_INC,
    RANGE_MIN,
    RANGE_MAX,
    BEAM_STRIDE,
    SUBMAP_SIZE_METERS,
    SUBMAP_RESOLUTION,
    SCANS_PER_SUBMAP,
    RAY_STEPS,
    L0,
    L_FREE,
    L_OCC,
    L_MIN,
    L_MAX,
    EXTRAP_MAX_DT,
    EXTRAP_INIT_VXY,
    EXTRAP_INIT_WZ,
)

from carto.local_slam.range_to_points import ranges_to_points
from carto.local_slam.pose_extrapolator import PoseExtrapolatorCV

from carto.pose_graph.pose_graph_2d import PoseGraph2D
from carto.pose_graph.backends.scipy_backend_2d import SciPyBackend2D

from slam_core.matching.preprocessing import PointCloudProcessor, PointCloudProcessorConfig
from slam_core.matching.core import MatcherManager
from slam_core.matching.scan_to_submap import (
    SubmapBuilder2D,
    ScanToSubmapMatcher,
    ScanToSubmapBackendConfig,
    SubmapSearchWindow,
)
from slam_core.matching.scan_to_map import ScanToMapMatcher

from carto.adapter import (
    CartoLocalSlamAdapter,
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


def main():
    clf_path = "datasets/fr079/fr079.clf"
    scans = read_carmen_log(clf_path)

    print("Loaded scans:", len(scans))

    # ------------------------------------------------
    # Experiment config
    # ------------------------------------------------
    MATCHER_TYPE = "scan_to_submap"          # "scan_to_submap" or "scan_to_map"
    SUBMAP_BACKEND_TYPE = "two_stage_bruteforce"   # "two_stage_bruteforce" or "branch_and_bound"
    MAX_SCANS = 700                          # set None for full dataset
    VERBOSE_EVERY = 10

    print("runner:", "carto_local_only")
    print("matcher_type:", MATCHER_TYPE)
    if MATCHER_TYPE == "scan_to_submap":
        print("submap_backend_type:", SUBMAP_BACKEND_TYPE)

    if MAX_SCANS is not None:
        scans = scans[:int(MAX_SCANS)]
        print("Using scans:", len(scans))

    # ------------------------------------------------
    # Shared point-cloud preprocessing
    # ------------------------------------------------
    point_processor = PointCloudProcessor(
        PointCloudProcessorConfig(
            fixed_voxel_size=0.05,
            adaptive_voxel_max_size=0.0,
            adaptive_min_num_points=90,
            adaptive_num_iterations=8,
            enabled=True,
        )
    )

    # ------------------------------------------------
    # Submap builder (only used by scan_to_submap)
    # ------------------------------------------------
    submaps = SubmapBuilder2D(
        submap_size_m=SUBMAP_SIZE_METERS,
        resolution=SUBMAP_RESOLUTION,
        scans_per_submap=SCANS_PER_SUBMAP,
        ray_steps=RAY_STEPS,
        l0=L0,
        l_occ=L_OCC,
        l_free=L_FREE,
        l_min=L_MIN,
        l_max=L_MAX,
    )

    # ------------------------------------------------
    # Matcher configs
    # ------------------------------------------------
    local_submap_backend_config = ScanToSubmapBackendConfig(
        backend_type=SUBMAP_BACKEND_TYPE,
        min_score=0.52 if SUBMAP_BACKEND_TYPE == "two_stage_bruteforce" else 0.66,
        min_valid=20,
        precomp_levels=8 if SUBMAP_BACKEND_TYPE == "branch_and_bound" else 3,
        do_refine=True,
        max_match_points=60 if SUBMAP_BACKEND_TYPE == "two_stage_bruteforce" else 40,
        max_refine_points=180 if SUBMAP_BACKEND_TYPE == "two_stage_bruteforce" else 120,
        refine_min_points=20 if SUBMAP_BACKEND_TYPE == "two_stage_bruteforce" else 25,
        refine_w_trans=1.0,
        refine_w_rot=1.0,
        refine_iters=8,
        refine_damping=1e-3,
        refine_eps_stop=1e-6,
        refine_step_clip_xy=0.10,
        refine_step_clip_th=np.deg2rad(5.0),
        refine_verbose=False,
        coarse=SubmapSearchWindow(
            xy_window=0.8,
            theta_window=0.3,
            xy_step=0.20 if SUBMAP_BACKEND_TYPE == "two_stage_bruteforce" else 0.05,
            theta_step=0.08 if SUBMAP_BACKEND_TYPE == "two_stage_bruteforce" else 0.02,
            level=2 if SUBMAP_BACKEND_TYPE == "two_stage_bruteforce" else 0,
        ),
        fine=(
            SubmapSearchWindow(
                xy_window=0.25,
                theta_window=0.12,
                xy_step=0.05,
                theta_step=0.02,
                level=0,
            )
            if SUBMAP_BACKEND_TYPE == "two_stage_bruteforce"
            else None
        ),
        bnb_depth_limit=8,
        bnb_min_rotational_step=0.02,
        bnb_branching=4,
    )

    corr_params_map = dict(
        gn_iters_per_level=[15, 12, 10, 8],
        gn_damping=1e-3,
        min_points=20,
        min_inliers_accept=25,
        min_score=0.45,
        step_clip_xy=0.02,
        step_clip_th=np.deg2rad(0.7),
    )

    # ------------------------------------------------
    # Matcher selection + creation
    # ------------------------------------------------
    if MATCHER_TYPE == "scan_to_submap":
        matcher = ScanToSubmapMatcher(
            submap_builder=submaps,
            backend_config=local_submap_backend_config,
        )

    elif MATCHER_TYPE == "scan_to_map":
        map_params = dict(
            base_res=SUBMAP_RESOLUTION,
            size_m=80.0,
            num_levels=4,
            l0=L0,
            l_min=L_MIN,
            l_max=L_MAX,
            l_free=L_FREE,
            l_occ=L_OCC,
            ray_steps=RAY_STEPS,
        )

        matcher = ScanToMapMatcher(
            map_params=map_params,
            corr_params=corr_params_map,
        )

    else:
        raise ValueError(f"Unsupported MATCHER_TYPE: {MATCHER_TYPE}")

    matcher_manager = MatcherManager(
        active_matcher=matcher,
        rolling_buffer_size=30,
        min_buffer_for_switch=20,
    )

    # ------------------------------------------------
    # Extrapolator
    # ------------------------------------------------
    extrap = PoseExtrapolatorCV(
        max_dt=EXTRAP_MAX_DT,
        init_vxy=EXTRAP_INIT_VXY,
        init_wz=EXTRAP_INIT_WZ,
    )

    # ------------------------------------------------
    # Motion filter
    # ------------------------------------------------
    TARGET_INSERT_PERIOD_S = 0.5
    V_EXPECTED_MPS = 0.40
    W_EXPECTED_RPS = np.deg2rad(30.0)

    motion_params = make_motion_filter_from_expected_velocity(
        target_insert_period_s=TARGET_INSERT_PERIOD_S,
        v_expected_mps=V_EXPECTED_MPS,
        w_expected_rps=W_EXPECTED_RPS,
    )

    print(
        "MotionFilter thresholds:",
        f"time={motion_params.max_time_seconds:.3f}s",
        f"dist={motion_params.max_distance_meters:.3f}m",
        f"angle={np.rad2deg(motion_params.max_angle_radians):.2f}deg",
    )

    # ------------------------------------------------
    # Pose graph
    # ------------------------------------------------
    backend = SciPyBackend2D(huber_scale=1.0)
    backend.set_fixed("submap", 0)

    pg = PoseGraph2D(
        backend=backend,
        sig_xy=0.05,
        sig_theta=np.deg2rad(1.0),
    )

    # ------------------------------------------------
    # Adapter
    # ------------------------------------------------
    adapter = CartoLocalSlamAdapter(
        matcher_manager=matcher_manager,
        extrapolator=extrap,
        pose_graph=pg,
        motion_params=motion_params,
        solve_every_n_nodes=30,
        global_slam=None,
    )

    # ------------------------------------------------
    # Initialize extrapolator
    # ------------------------------------------------
    first = scans[0]
    adapter.initialize_extrapolator(
        float(first["t"]),
        Pose2(*first["odom"]),
    )

    # ------------------------------------------------
    # Output
    # ------------------------------------------------
    os.makedirs("carto_outputs", exist_ok=True)

    suffix = MATCHER_TYPE
    if MATCHER_TYPE == "scan_to_submap":
        suffix = f"{MATCHER_TYPE}_{SUBMAP_BACKEND_TYPE}"

    traj_path = f"carto_outputs/trajectory_{suffix}_{len(scans)}.txt"
    meta_path = f"carto_outputs/trajectory_{suffix}_{len(scans)}_debug.txt"

    with open(traj_path, "w") as f_traj, open(meta_path, "w") as f_meta:
        f_meta.write(f"# matcher_type={MATCHER_TYPE}\n")
        if MATCHER_TYPE == "scan_to_submap":
            f_meta.write(f"# submap_backend_type={SUBMAP_BACKEND_TYPE}\n")
        f_meta.write(
            "k t x y theta score inliers dx dy dtheta do_insert did_insert "
            "constraints_total constraints_intra constraints_loop nodes submaps\n"
        )

        # ------------------------------------------------
        # Main loop
        # ------------------------------------------------
        for k, s in enumerate(scans):
            t = float(s["t"])
            odom = Pose2(*s["odom"])

            pts_raw = ranges_to_points(
                s["ranges"],
                ANGLE_MIN,
                ANGLE_INC,
                RANGE_MIN,
                RANGE_MAX,
                stride=BEAM_STRIDE,
            )
            pts, proc_debug = point_processor.process(pts_raw)

            pose, result, do_insert, did_insert = adapter.process_scan(
                t=t,
                scan_points_local=pts,
                odom_pose_world=odom,
                odom_alpha=0.2,
            )

            score = float(result.score) if result.success else -1.0
            mode = "MATCH" if result.success else "FALLBACK"

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

            constraint_counts = pg.get_constraint_counts()
            n_constraints_total = constraint_counts["total"]
            n_constraints_intra = constraint_counts["intra"]
            n_constraints_loop = constraint_counts["loop"]

            n_nodes = len(pg.backend.nodes)
            n_submaps = len(pg.backend.submaps)

            f_traj.write(
                f"{t:.6f} {pose.x:.6f} {pose.y:.6f} {pose.theta:.6f} {score:.6f}\n"
            )

            f_meta.write(
                f"{k} {t:.6f} {pose.x:.6f} {pose.y:.6f} {pose.theta:.6f} "
                f"{score:.6f} "
                f"{-1 if inliers is None else int(inliers)} "
                f"{dx:.6f} {dy:.6f} {dtheta:.6f} "
                f"{int(do_insert)} {int(did_insert)} "
                f"{n_constraints_total} {n_constraints_intra} {n_constraints_loop} "
                f"{n_nodes} {n_submaps}\n"
            )

            if (k % VERBOSE_EVERY) == 0:
                motion_debug = adapter.last_motion_debug() or {}

                dmsg = _format_delta(delta)
                imsg = _format_inliers(inliers)

                print(
                    "motion:",
                    f"matcher={motion_debug.get('matcher_name', MATCHER_TYPE)}",
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

                print(
                    "constraints:",
                    f"total={n_constraints_total}",
                    f"intra={n_constraints_intra}",
                    f"loop={n_constraints_loop}",
                )
                print("nodes:", n_nodes)
                print("submaps:", n_submaps)

    adapter.finalize()

    print("Wrote:", traj_path)
    print("Wrote:", meta_path)


if __name__ == "__main__":
    main()