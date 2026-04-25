from __future__ import annotations

import os
import numpy as np

from slam_core.dataio.intel_carmen import read_intel_carmen_log
from slam_core.common.types import Pose2

from carto.map_reconstruction_adapter import CartoMapReconstructionAdapter
from slam_core.map_reconstruction import ReconstructionConfig

from carto.config import (
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
from carto.pose_graph.backends.pyceres_backend_2d import PyCeresBackend2D

from slam_core.matching.preprocessing import PointCloudProcessor, PointCloudProcessorConfig
from slam_core.matching.core import MatcherManager
from slam_core.matching.scan_to_submap import (
    SubmapBuilder2D,
    ScanToSubmapMatcher,
    ScanToSubmapBackendConfig,
    SubmapSearchWindow,
)

from carto.adapter import (
    CartoLocalSlamAdapter,
    make_motion_filter_from_expected_velocity,
)
from carto.pose_graph.global_slam_2d import CartoGlobalSlam2D
from slam_core.loop_closure import LoopClosureConfig
from carto.loop_closure_adapter import CartoLoopClosureAdapter


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


def _build_run_summary_lines(
    pose_graph,
    global_slam,
    note: str,
    last_k: int,
    last_t: float,
    last_pose: Pose2 | None,
) -> list[str]:
    lines: list[str] = []

    lines.append(f"RUN_NOTE: {note}")
    lines.append(f"LAST_STEP: k={last_k}")
    lines.append(f"LAST_TIME: t={last_t:.6f}" if np.isfinite(last_t) else "LAST_TIME: t=nan")

    if last_pose is not None:
        lines.append(
            "LAST_POSE: "
            f"x={last_pose.x:.6f} y={last_pose.y:.6f} theta={last_pose.theta:.6f}"
        )

    counts = pose_graph.get_constraint_counts()
    lines.append(
        "CONSTRAINT_COUNTS: "
        f"intra={counts['intra']} "
        f"loop={counts['loop']} "
        f"total={counts['total']}"
    )

    if global_slam is not None:
        stats = global_slam.get_stats()
        lines.append(
            "FINAL_LOOP_SUMMARY: "
            f"candidates={stats['candidate_pairs']} "
            f"accepted={stats['accepted_pairs']} "
            f"rejected={stats['rejected_pairs']} "
            f"duplicates={stats['duplicate_pairs']} "
            f"intra_constraints={counts['intra']} "
            f"loop_constraints={counts['loop']} "
            f"total_constraints={counts['total']}"
        )

        if hasattr(global_slam, "get_diagnostics_summary"):
            diag = global_slam.get_diagnostics_summary()
            lines.append(
                "FINAL_LOOP_DIAGNOSTICS: "
                f"accepted_from_new_node_search={diag.get('accepted_from_new_node_search', 0)} "
                f"accepted_from_finished_submap_search={diag.get('accepted_from_finished_submap_search', 0)} "
                f"rejected_sampled_out={diag.get('rejected_sampled_out', 0)} "
                f"rejected_matcher_failed={diag.get('rejected_matcher_failed', 0)} "
                f"rejected_score_failed={diag.get('rejected_score_failed', 0)} "
                f"rejected_geometry_failed={diag.get('rejected_geometry_failed', 0)} "
                f"rejected_duplicate={diag.get('rejected_duplicate', 0)}"
            )

            if diag.get("accepted_score_count", 0) > 0:
                lines.append(
                    "FINAL_LOOP_ACCEPTED_SCORES: "
                    f"count={diag['accepted_score_count']} "
                    f"min={diag['accepted_score_min']:.3f} "
                    f"mean={diag['accepted_score_mean']:.3f} "
                    f"median={diag['accepted_score_median']:.3f} "
                    f"max={diag['accepted_score_max']:.3f} "
                    f"near_min_score_count={diag['accepted_near_min_score_count']}"
                )

            if diag.get("score_failed_count", 0) > 0:
                lines.append(
                    "FINAL_LOOP_SCORE_FAILED: "
                    f"count={diag['score_failed_count']} "
                    f"min={diag['score_failed_min']:.3f} "
                    f"mean={diag['score_failed_mean']:.3f} "
                    f"median={diag['score_failed_median']:.3f} "
                    f"max={diag['score_failed_max']:.3f}"
                )

    if hasattr(pose_graph.backend, "get_last_summary"):
        summary = pose_graph.backend.get_last_summary()
        if summary is not None:
            try:
                lines.append(f"CERES_SUMMARY: {summary.BriefReport()}")
            except Exception:
                lines.append(f"CERES_SUMMARY: {summary}")

    return lines


def _save_run_summary(
    path: str,
    pose_graph,
    global_slam,
    note: str,
    last_k: int,
    last_t: float,
    last_pose: Pose2 | None,
) -> None:
    lines = _build_run_summary_lines(
        pose_graph=pose_graph,
        global_slam=global_slam,
        note=note,
        last_k=last_k,
        last_t=last_t,
        last_pose=last_pose,
    )

    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    for line in lines:
        print(line)


def _save_checkpoint(
    pose_graph,
    global_slam,
    out_prefix: str,
    tag: str,
    last_k: int,
    last_t: float,
    last_pose: Pose2 | None,
) -> None:
    checkpoint_prefix = f"{out_prefix}_{tag}"

    try:
        optimized = pose_graph.get_optimized_poses()
        if not optimized:
            try:
                pose_graph.solve(max_iters=20)
            except Exception as e:
                print(f"checkpoint_solve_warning: {e}")

        _write_optimized_outputs(pose_graph, checkpoint_prefix)
    except Exception as e:
        print(f"checkpoint_write_warning: {e}")

    try:
        _save_run_summary(
            path=f"{checkpoint_prefix}_summary.txt",
            pose_graph=pose_graph,
            global_slam=global_slam,
            note=tag,
            last_k=last_k,
            last_t=last_t,
            last_pose=last_pose,
        )
    except Exception as e:
        print(f"checkpoint_summary_warning: {e}")


def main():
    clf_path = "datasets/intel/intel.clf"
    scans = read_intel_carmen_log(clf_path)

    print("Loaded scans:", len(scans))

    MATCHER_TYPE = "scan_to_submap"
    MAX_SCANS = 7000
    VERBOSE_EVERY = 10

    CHECKPOINT_EVERY_NODES = 1000
    FLUSH_EVERY_STEPS = 25

    print("runner:", "carto_loop_closure_intel")
    print("matcher_type:", MATCHER_TYPE)

    if MAX_SCANS is not None:
        scans = scans[:int(MAX_SCANS)]
        print("Using scans:", len(scans))

    # Intel scan geometry
    INTEL_ANGLE_MIN = -np.pi / 2.0
    INTEL_ANGLE_INC = np.pi / 180.0
    INTEL_RANGE_MIN = 0.001
    INTEL_RANGE_MAX = 50.0

    point_processor = PointCloudProcessor(
        PointCloudProcessorConfig(
            fixed_voxel_size=0.05,
            adaptive_voxel_max_size=0.0,
            adaptive_min_num_points=90,
            adaptive_num_iterations=8,
            enabled=True,
        )
    )

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

    # Local matcher: reuse the working Intel local settings.
    local_submap_backend_config = ScanToSubmapBackendConfig(
        backend_type="two_stage_bruteforce",
        min_score=0.70,
        min_valid=20,
        precomp_levels=3,
        do_refine=True,
        max_match_points=60,
        max_refine_points=180,
        refine_min_points=20,
        refine_w_trans=4.0,
        refine_w_rot=2.0,
        refine_iters=8,
        refine_damping=1e-3,
        refine_eps_stop=1e-6,
        refine_step_clip_xy=0.10,
        refine_step_clip_th=np.deg2rad(5.0),
        refine_verbose=False,
        coarse=SubmapSearchWindow(
            xy_window=0.5,
            theta_window=0.18,
            xy_step=0.15,
            theta_step=0.05,
            level=2,
        ),
        fine=SubmapSearchWindow(
            xy_window=0.25,
            theta_window=0.12,
            xy_step=0.05,
            theta_step=0.02,
            level=0,
        ),
    )

    # Loop matcher: broader than local, but still bounded.
    loop_constrained_window = SubmapSearchWindow(
        xy_window=1.5,
        theta_window=np.deg2rad(12.0),
        xy_step=0.05,
        theta_step=0.02,
        level=0,
    )

    loop_submap_backend_config = ScanToSubmapBackendConfig(
        backend_type="branch_and_bound",
        min_score=0.70,
        global_localization_min_score=0.70,
        min_valid=20,
        precomp_levels=7,
        do_refine=True,
        max_match_points=80,
        max_refine_points=180,
        refine_min_points=20,
        refine_w_trans=10.0,
        refine_w_rot=40.0,
        coarse=loop_constrained_window,
        fine=None,
        bnb_depth_limit=7,
        bnb_min_rotational_step=0.02,
        bnb_branching=4,
    )

    local_matcher = ScanToSubmapMatcher(
        submap_builder=submaps,
        backend_config=local_submap_backend_config,
    )
    loop_matcher = ScanToSubmapMatcher(
        submap_builder=submaps,
        backend_config=loop_submap_backend_config,
    )
    matcher = local_matcher

    matcher_manager = MatcherManager(
        active_matcher=matcher,
        rolling_buffer_size=30,
        min_buffer_for_switch=20,
    )

    extrap = PoseExtrapolatorCV(
        max_dt=EXTRAP_MAX_DT,
        init_vxy=EXTRAP_INIT_VXY,
        init_wz=EXTRAP_INIT_WZ,
        pose_queue_duration_s=1.5,
        odom_queue_duration_s=1.5,
        odom_trust=0.35,
    )

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

    backend = PyCeresBackend2D(
        huber_scale=1e1,
        linear_solver_type="SPARSE_NORMAL_CHOLESKY",
        num_threads=1,
        minimizer_progress_to_stdout=False,
        local_slam_pose_translation_weight=1e5,
        local_slam_pose_rotation_weight=1e5,
    )
    backend.set_fixed("submap", 0)
    print("Using backend:", type(backend).__name__)

    pg = PoseGraph2D(
        backend=backend,
        submap_builder=submaps,
        intra_translation_weight=5e2,
        intra_rotation_weight=1.6e3,
    )

    loop_config = LoopClosureConfig(
        min_score=float(loop_submap_backend_config.min_score),
        global_localization_min_score=float(loop_submap_backend_config.global_localization_min_score),
        translation_weight=3.0e3,
        rotation_weight=2.0e4,
        optimize_every_n_nodes=120,
        min_node_index_separation=140,
        spatial_search_radius=6.0,
        max_candidate_targets_per_new_node=1,
        historical_node_stride=5,
        max_candidate_nodes_per_finished_target=0,
        recent_finished_submap_exclusion=3,
        finished_submap_verification_budget_per_tick=12,
        force_full_submap_for_finished_submap_search=False,
        finished_submap_full_search_failure_threshold=3,
        max_loop_translation_residual_m=1.0,
        max_loop_rotation_residual_rad=0.20944,
    )

    global_slam = CartoGlobalSlam2D(
        loop_closure_adapter=CartoLoopClosureAdapter(
            matcher=loop_matcher,
            pose_graph=pg,
            config=loop_config,
        ),
        pose_graph=pg,
        optimize_every_n_nodes=120,
        adapter=None,
        correction_alpha=0.5,
    )

    adapter = CartoLocalSlamAdapter(
        matcher_manager=matcher_manager,
        extrapolator=extrap,
        pose_graph=pg,
        motion_params=motion_params,
        solve_every_n_nodes=30,
        global_slam=global_slam,
    )
    global_slam.set_adapter(adapter)

    first = scans[0]
    adapter.initialize_extrapolator(
        float(first["t"]),
        Pose2(*first["odom"]),
    )

    os.makedirs("carto_outputs", exist_ok=True)

    out_prefix = f"carto_outputs/trajectory_scan_to_submap_loop_intel_{len(scans)}"
    traj_path = f"{out_prefix}.txt"
    meta_path = f"{out_prefix}_debug.txt"

    last_k = -1
    last_t = float("nan")
    last_pose: Pose2 | None = None
    last_checkpoint_node_count = -1

    with open(traj_path, "w") as f_traj, open(meta_path, "w") as f_meta:
        try:
            f_meta.write("# matcher_type=scan_to_submap\n")
            f_meta.write("# loop_closure_enabled=1\n")
            f_meta.write(
                "k t x y theta score inliers dx dy dtheta do_insert did_insert "
                "constraints_total constraints_intra constraints_loop "
                "loop_candidates loop_accepted loop_rejected loop_duplicates "
                "nodes submaps\n"
            )

            for k, s in enumerate(scans):
                t = float(s["t"])
                odom = Pose2(*s["odom"])

                pts_raw = ranges_to_points(
                    s["ranges"],
                    INTEL_ANGLE_MIN,
                    INTEL_ANGLE_INC,
                    INTEL_RANGE_MIN,
                    INTEL_RANGE_MAX,
                    stride=BEAM_STRIDE,
                )
                pts, proc_debug = point_processor.process(pts_raw)

                pose, result, do_insert, did_insert = adapter.process_scan(
                    t=t,
                    scan_points_local=pts,
                    odom_pose_world=odom,
                    odom_alpha=0.0,
                )

                last_k = int(k)
                last_t = float(t)
                last_pose = pose

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

                lc_stats = global_slam.get_stats()
                loop_candidates = lc_stats["candidate_pairs"]
                loop_accepted = lc_stats["accepted_pairs"]
                loop_rejected = lc_stats["rejected_pairs"]
                loop_duplicates = lc_stats["duplicate_pairs"]

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

                if (k % FLUSH_EVERY_STEPS) == 0:
                    f_traj.flush()
                    f_meta.flush()

                if (
                    n_nodes > 0
                    and (n_nodes % CHECKPOINT_EVERY_NODES) == 0
                    and n_nodes != last_checkpoint_node_count
                ):
                    f_traj.flush()
                    f_meta.flush()

                    _save_checkpoint(
                        pose_graph=pg,
                        global_slam=global_slam,
                        out_prefix=out_prefix,
                        tag=f"checkpoint_nodes_{n_nodes}",
                        last_k=last_k,
                        last_t=last_t,
                        last_pose=last_pose,
                    )
                    last_checkpoint_node_count = n_nodes

                if (k % VERBOSE_EVERY) == 0:
                    motion_debug = adapter.last_motion_debug() or {}

                    dmsg = _format_delta(delta)
                    imsg = _format_inliers(inliers)

                    debug_obj = getattr(result, "debug", None)
                    extra = getattr(debug_obj, "extra", {}) if debug_obj is not None else {}
                    init_mode = "pred" if extra.get("used_pred_initializer", False) else "coarse"
                    pyceres_reason = extra.get("pyceres_reason", "na")
                    coarse_valid = extra.get("coarse_trusted", extra.get("coarse_valid", None))
                    pyceres_summary = extra.get("pyceres_summary", None)

                    print(
                        "motion:",
                        f"matcher={motion_debug.get('matcher_name', 'scan_to_submap')}",
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
                        f"score={score:.3f} init={init_mode} coarse_valid={coarse_valid} "
                        f"pyceres={pyceres_reason} {dmsg} {imsg}"
                    )

                    if (not result.success) and (pyceres_summary is not None):
                        print("pyceres_summary:", pyceres_summary)

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

                    recent_events = global_slam.get_recent_events(3)
                    for ev in recent_events:
                        score_msg = "NA" if not np.isfinite(ev.score) else f"{ev.score:.3f}"

                        trans_msg = getattr(ev, "translation_residual_m", None)
                        rot_msg = getattr(ev, "rotation_residual_rad", None)

                        print(
                            "loop_event:",
                            f"node={ev.node_id}",
                            f"target={ev.target_id}",
                            f"score={score_msg}",
                            f"accepted={ev.accepted}",
                            f"status={ev.status}",
                            f"source={ev.source}",
                            f"full_submap={getattr(ev, 'used_full_submap', False)}",
                            f"trans_res={trans_msg if trans_msg is not None else 'NA'}",
                            f"rot_res_deg={np.rad2deg(rot_msg) if rot_msg is not None else 'NA'}",
                        )

        except KeyboardInterrupt:
            print("\nRun interrupted by user. Saving partial results...")
            f_traj.flush()
            f_meta.flush()

            _save_checkpoint(
                pose_graph=pg,
                global_slam=global_slam,
                out_prefix=out_prefix,
                tag=f"interrupted_k_{last_k}",
                last_k=last_k,
                last_t=last_t,
                last_pose=last_pose,
            )
            return

        except Exception as e:
            print(f"\nRun failed with exception: {e}")
            f_traj.flush()
            f_meta.flush()

            _save_checkpoint(
                pose_graph=pg,
                global_slam=global_slam,
                out_prefix=out_prefix,
                tag=f"failed_k_{last_k}",
                last_k=last_k,
                last_t=last_t,
                last_pose=last_pose,
            )
            raise

        finally:
            f_traj.flush()
            f_meta.flush()

    adapter.finalize()
    _write_optimized_outputs(pg, out_prefix)

    if hasattr(pg.backend, "get_last_summary"):
        summary = pg.backend.get_last_summary()
        if summary is not None:
            try:
                print("CERES_SUMMARY:", summary.BriefReport())
            except Exception:
                print("CERES_SUMMARY:", summary)

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

    if hasattr(global_slam, "get_diagnostics_summary"):
        diag = global_slam.get_diagnostics_summary()
        print(
            "FINAL_LOOP_DIAGNOSTICS:",
            f"accepted_from_new_node_search={diag.get('accepted_from_new_node_search', 0)}",
            f"accepted_from_finished_submap_search={diag.get('accepted_from_finished_submap_search', 0)}",
            f"rejected_sampled_out={diag.get('rejected_sampled_out', 0)}",
            f"rejected_matcher_failed={diag.get('rejected_matcher_failed', 0)}",
            f"rejected_score_failed={diag.get('rejected_score_failed', 0)}",
            f"rejected_geometry_failed={diag.get('rejected_geometry_failed', 0)}",
            f"rejected_duplicate={diag.get('rejected_duplicate', 0)}",
        )

        if diag.get("accepted_score_count", 0) > 0:
            print(
                "FINAL_LOOP_ACCEPTED_SCORES:",
                f"count={diag['accepted_score_count']}",
                f"min={diag['accepted_score_min']:.3f}",
                f"mean={diag['accepted_score_mean']:.3f}",
                f"median={diag['accepted_score_median']:.3f}",
                f"max={diag['accepted_score_max']:.3f}",
                f"near_min_score_count={diag['accepted_near_min_score_count']}",
            )

        if diag.get("score_failed_count", 0) > 0:
            print(
                "FINAL_LOOP_SCORE_FAILED:",
                f"count={diag['score_failed_count']}",
                f"min={diag['score_failed_min']:.3f}",
                f"mean={diag['score_failed_mean']:.3f}",
                f"median={diag['score_failed_median']:.3f}",
                f"max={diag['score_failed_max']:.3f}",
            )

    _save_run_summary(
        path=f"{out_prefix}_summary.txt",
        pose_graph=pg,
        global_slam=global_slam,
        note="completed",
        last_k=last_k,
        last_t=last_t,
        last_pose=last_pose,
    )

    print("Wrote:", traj_path)
    print("Wrote:", meta_path)

    print("first odom:", scans[0]["odom"])
    print("first laser_pose:", scans[0].get("laser_pose", None))
    print("last odom:", scans[-1]["odom"])
    print("last laser_pose:", scans[-1].get("laser_pose", None))
    print("duration_s:", scans[-1]["t"] - scans[0]["t"])
    print("num_beams_first_scan:", len(scans[0]["ranges"]))


if __name__ == "__main__":
    main()