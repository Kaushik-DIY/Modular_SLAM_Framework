import os
import numpy as np

from slam_core.dataio.carmen import read_carmen_log
from slam_core.common.types import Pose2

from carto.map_reconstruction_adapter import CartoMapReconstructionAdapter
from slam_core.map_reconstruction import ReconstructionConfig

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
)
from slam_core.matching.scan_to_map import ScanToMapMatcher

from carto.adapter import (
    CartoLocalSlamAdapter,
    make_motion_filter_from_expected_velocity,
)
from carto.pose_graph.global_slam_2d import CartoGlobalSlam2D
from carto.pose_graph.constraint_builder_2d import ConstraintBuilder2DConfig


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

def _write_optimized_outputs(pose_graph, out_prefix: str) -> None:
    optimized = pose_graph.get_optimized_poses()
    if not optimized:
        print("No optimized poses available.")
        return

    node_items = []
    submap_items = []

    for key, pose in optimized.items():
        kind, idx = key
        if kind == "node":
            node_items.append((int(idx), pose))
        elif kind == "submap":
            submap_items.append((int(idx), pose))

    node_items.sort(key=lambda x: x[0])
    submap_items.sort(key=lambda x: x[0])

    node_path = f"{out_prefix}_optimized_nodes.txt"
    submap_path = f"{out_prefix}_optimized_submaps.txt"

    with open(node_path, "w") as f:
        f.write("# node_id x y theta\n")
        for node_id, pose in node_items:
            f.write(f"{node_id} {pose.x:.6f} {pose.y:.6f} {pose.theta:.6f}\n")

    with open(submap_path, "w") as f:
        f.write("# submap_id x y theta\n")
        for submap_id, pose in submap_items:
            f.write(f"{submap_id} {pose.x:.6f} {pose.y:.6f} {pose.theta:.6f}\n")

    print("Wrote:", node_path)
    print("Wrote:", submap_path)


def main():
    clf_path = "datasets/fr079/fr079.clf"
    scans = read_carmen_log(clf_path)

    print("Loaded scans:", len(scans))

    # ------------------------------------------------
    # Experiment config
    # ------------------------------------------------
    MATCHER_TYPE = "scan_to_submap"   # loop closure is active only in this mode
    MAX_SCANS = 700                   # set None for full dataset
    VERBOSE_EVERY = 10

    print("runner:", "carto_loop_closure")
    print("matcher_type:", MATCHER_TYPE)

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
    # Submap builder
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
    corr_params_submap = dict(
        min_score=0.58,
        odom_alpha=0.2,
        max_match_points=60,
        min_valid=20,
        precomp_levels=3,
        coarse_level=2,
        coarse_xy_window=0.8,
        coarse_th_window=0.3,
        coarse_xy_step=0.20,
        coarse_th_step=0.08,
        fine_level=0,
        fine_xy_window=0.25,
        fine_th_window=0.12,
        fine_xy_step=0.05,
        fine_th_step=0.02,
        do_refine=True,
        max_refine_points=220,
        refine_min_points=35,
        refine_w_trans=1.0,
        refine_w_rot=1.0,
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
    # Matcher selection
    # ------------------------------------------------
    if MATCHER_TYPE == "scan_to_submap":
        matcher = ScanToSubmapMatcher(
            submap_builder=submaps,
            corr_params=corr_params_submap,
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
    # Loop closure adapter
    # ------------------------------------------------
    global_slam = None
    if MATCHER_TYPE == "scan_to_submap":
        global_slam = CartoGlobalSlam2D(
            matcher=matcher,
            pose_graph=pg,
            config=ConstraintBuilder2DConfig(
                max_constraint_distance=4.5,
                sampling_ratio=0.60,
                min_score=0.66,
                optimize_every_n_nodes=100,
                min_node_index_separation=40,
                recent_finished_submap_exclusion=3,
                min_valid=20,
                coarse_level=2,
                coarse_xy_window=0.8,
                coarse_th_window=0.3,
                coarse_xy_step=0.20,
                coarse_th_step=0.08,
                do_refine=True,
                max_match_points=40,
                max_refine_points=120,
                refine_min_points=25,
                consistency_max_translation_delta=1.0,
                consistency_max_rotation_delta=np.deg2rad(20.0),
                loop_translation_weight=10.0,
                loop_rotation_weight=50.0,
                min_score_for_weight=0.66,
                max_score_for_weight=0.90,
                min_weight_scale=0.50,
                max_weight_scale=1.50,
                sampler_seed=7,
            ),
        )

    # ------------------------------------------------
    # SLAM adapter
    # ------------------------------------------------
    adapter = CartoLocalSlamAdapter(
        matcher_manager=matcher_manager,
        extrapolator=extrap,
        pose_graph=pg,
        motion_params=motion_params,
        solve_every_n_nodes=30,
        global_slam=global_slam,
    )

    # ------------------------------------------------
    # Initialize
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

    out_prefix = f"carto_outputs/trajectory_{MATCHER_TYPE}_loop_{len(scans)}"
    traj_path = f"{out_prefix}.txt"
    meta_path = f"{out_prefix}_debug.txt"

    with open(traj_path, "w") as f_traj, open(meta_path, "w") as f_meta:
        f_meta.write(f"# matcher_type={MATCHER_TYPE}\n")
        f_meta.write("# loop_closure_enabled=1\n")
        """ f_meta.write(
            "k t x y theta score inliers dx dy dtheta do_insert did_insert constraints nodes submaps\n"
        ) """
        f_meta.write(
            "k t x y theta score inliers dx dy dtheta do_insert did_insert "
            "constraints_total constraints_intra constraints_loop "
            "loop_candidates loop_accepted loop_rejected loop_duplicates "
            "nodes submaps\n"
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

            if global_slam is not None:
                lc_stats = global_slam.get_stats()
                loop_candidates = lc_stats["candidate_pairs"]
                loop_accepted = lc_stats["accepted_pairs"]
                loop_rejected = lc_stats["rejected_pairs"]
                loop_duplicates = lc_stats["duplicate_pairs"]
            else:
                loop_candidates = 0
                loop_accepted = 0
                loop_rejected = 0
                loop_duplicates = 0

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
                f"{loop_candidates} {loop_accepted} {loop_rejected} {loop_duplicates} "
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
                print(
                    "loop_stats:",
                    f"candidates={loop_candidates}",
                    f"accepted={loop_accepted}",
                    f"rejected={loop_rejected}",
                    f"duplicates={loop_duplicates}",
                )
                print("nodes:", n_nodes)
                print("submaps:", n_submaps)

                if global_slam is not None:
                    recent_events = global_slam.get_recent_events(3)
                    for ev in recent_events:
                        score_msg = "NA" if not np.isfinite(ev.score) else f"{ev.score:.3f}"
                        print(
                            "loop_event:",
                            f"node={ev.node_id}",
                            f"target={ev.target_id}",
                            f"score={score_msg}",
                            f"accepted={ev.accepted}",
                            f"status={ev.status}",
                        )

    adapter.finalize()
    _write_optimized_outputs(pg, out_prefix)
    if MATCHER_TYPE == "scan_to_submap":
        reconstructor = CartoMapReconstructionAdapter(
            matcher=matcher,
            pose_graph=pg,
            config=ReconstructionConfig(
                global_resolution=SUBMAP_RESOLUTION,
                informative_evidence_threshold=0.05,
                evidence_clip_min=-10.0,
                evidence_clip_max=10.0,
                tile_cell_stride=1,
                map_margin_m=1.0,
            ),
        )
        reconstructor.save_before_after_plot(out_prefix)

    if global_slam is not None:
        final_stats = global_slam.get_stats()
        final_counts = pg.get_constraint_counts()
        print(
            "FINAL_LOOP_SUMMARY:",
            f"candidates={final_stats['candidate_pairs']}",
            f"accepted={final_stats['accepted_pairs']}",
            f"rejected={final_stats['rejected_pairs']}",
            f"duplicates={final_stats['duplicate_pairs']}",
            f"intra_constraints={final_counts['intra']}",
            f"loop_constraints={final_counts['loop']}",
            f"total_constraints={final_counts['total']}",
        )

    print("Wrote:", traj_path)
    print("Wrote:", meta_path)


if __name__ == "__main__":
    main()