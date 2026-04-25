from __future__ import annotations

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
from carto.pose_graph.backends.pyceres_backend_2d import PyCeresBackend2D

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
from carto.pose_graph.global_slam_2d import CartoGlobalSlam2D
from slam_core.loop_closure import LoopClosureConfig
from carto.loop_closure_adapter import CartoLoopClosureAdapter


def _format_delta(delta) -> str:
    """Format pose-refinement delta for concise logging."""
    if delta is None:
        return "d=None"
    arr = np.asarray(delta, dtype=float).reshape(-1)
    if arr.shape[0] < 3:
        return "d=None"
    return f"d=({arr[0]:.3f},{arr[1]:.3f},{arr[2]:.3f})"


def _format_inliers(inliers) -> str:
    """Format inlier count for concise logging."""
    if inliers is None:
        return "inl=None"
    return f"inl={int(inliers)}"


def _write_optimized_outputs(pose_graph, out_prefix: str) -> None:
    """
    Write optimized node and submap trajectories to text files.
    """
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
    """
    Build a compact text summary of the current run state.
    """
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
    """
    Save a compact human-readable run summary.
    """
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
    """
    Save partial optimized outputs and a compact summary.
    """
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
    clf_path = "datasets/fr079/fr079.clf"
    scans = read_carmen_log(clf_path)

    print("Loaded scans:", len(scans))

    # ------------------------------------------------------------------
    # Experiment configuration
    # ------------------------------------------------------------------
    MATCHER_TYPE = "scan_to_submap"   # loop closure is active only in this mode
    MAX_SCANS = 700                  # distinct save length
    VERBOSE_EVERY = 10

    # Long-run safety and checkpoint settings.
    CHECKPOINT_EVERY_NODES = 1000
    FLUSH_EVERY_STEPS = 25

    print("runner:", "carto_loop_closure")
    print("matcher_type:", MATCHER_TYPE)

    if MAX_SCANS is not None:
        scans = scans[:int(MAX_SCANS)]
        print("Using scans:", len(scans))

    # ------------------------------------------------------------------
    # Shared point-cloud preprocessing
    # ------------------------------------------------------------------
    point_processor = PointCloudProcessor(
        PointCloudProcessorConfig(
            fixed_voxel_size=0.05,
            adaptive_voxel_max_size=0.0,
            adaptive_min_num_points=90,
            adaptive_num_iterations=8,
            enabled=True,
        )
    )

    # ------------------------------------------------------------------
    # Submap builder
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Matcher configuration
    # ------------------------------------------------------------------
    local_submap_backend_config = ScanToSubmapBackendConfig(
        backend_type="two_stage_bruteforce",
        # min_score: Cartographer's canonical threshold after the submap lifecycle
        # fix — the matching submap always has >=N/2 scans at match time, so
        # scores are reliably 0.60-0.90 in well-covered corridor geometry.
        # 0.45 is a small safety margin below the expected 0.60+ range to avoid
        # FALLBACK on slight geometry ambiguities, while rejecting true mismatches.
        min_score=0.55,
        min_valid=20,
        precomp_levels=3,
        do_refine=True,
        max_match_points=80,
        max_refine_points=200,
        refine_min_points=20,
        refine_w_trans=10.0,     # Cartographer ceres: translation_weight=10.0
        refine_w_rot=40.0,       # Cartographer ceres: rotation_weight=40.0
        coarse=SubmapSearchWindow(
            # Search window narrowed from 2.5m to 0.8m now that the extrapolator
            # is pure finite-difference (alpha=0.0) — no more 10-scan velocity lag.
            # 0.8m covers ±4 scan periods of motion at 0.4m/s, which is generous.
            xy_window=0.2,
            theta_window=np.deg2rad(5.0),     # ~23 degrees
            xy_step=0.05,
            theta_step=0.03,
            level=2,
        ),
        fine=SubmapSearchWindow(
            xy_window=0.2,
            theta_window=np.deg2rad(5.0),
            xy_step=0.02,
            theta_step=0.01,
            level=0,
        ),
    )

    loop_constrained_window = SubmapSearchWindow(
        # Runner-owned constrained loop-search window. Start from the
        # Cartographer paper/reference example (7m / 30deg) and tune here
        # per dataset instead of hardcoding inside matcher logic.
        xy_window=7.0,
        theta_window=np.deg2rad(30.0),
        xy_step=0.05,
        theta_step=0.02,
        level=0,
    )

    loop_submap_backend_config = ScanToSubmapBackendConfig(
        backend_type="branch_and_bound",
        # Cartographer pose_graph.lua: min_score=0.55, global_min=0.60
        min_score=0.55,
        global_localization_min_score=0.60,
        min_valid=20,
        # branch_and_bound_depth = 7 in Cartographer
        precomp_levels=7,
        do_refine=True,
        max_match_points=80,
        max_refine_points=180,
        refine_min_points=20,
        # Cartographer ceres_scan_matcher defaults
        refine_w_trans=10.0,
        refine_w_rot=40.0,
        coarse=loop_constrained_window,
        fine=None,
        bnb_depth_limit=7,         # Cartographer: branch_and_bound_depth=7
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

    # ------------------------------------------------------------------
    # Matcher selection
    # ------------------------------------------------------------------
    if MATCHER_TYPE == "scan_to_submap":
        local_matcher = ScanToSubmapMatcher(
            submap_builder=submaps,
            backend_config=local_submap_backend_config,
        )
        loop_matcher = ScanToSubmapMatcher(
            submap_builder=submaps,
            backend_config=loop_submap_backend_config,
        )
        matcher = local_matcher

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
        loop_matcher = None

    else:
        raise ValueError(f"Unsupported MATCHER_TYPE: {MATCHER_TYPE}")

    matcher_manager = MatcherManager(
        active_matcher=matcher,
        rolling_buffer_size=30,
        min_buffer_for_switch=20,
    )

    # ------------------------------------------------------------------
    # Extrapolator
    # ------------------------------------------------------------------
    extrap = PoseExtrapolatorCV(
        max_dt=EXTRAP_MAX_DT,
        init_vxy=EXTRAP_INIT_VXY,
        init_wz=EXTRAP_INIT_WZ,
    )

    # ------------------------------------------------------------------
    # Motion filter — Cartographer-faithful thresholds (motion_filter.lua):
    #   max_distance = 0.1m,  max_angle = ~0.5°,  max_time = 5.0s
    # Our previous values (0.20m, 15°, 0.5s) were 2-65x too large:
    #   - 40-60% of scans were skipped for slow indoor motion
    #   - Submaps had half as many points → lower match scores
    # With these tighter thresholds, almost every scan reaches the submap.
    TARGET_INSERT_PERIOD_S = 5.0           # Cartographer: max_time_seconds=5.0
    V_EXPECTED_MPS = 0.02                  # → dist = 0.02×5 = 0.10m ≈ Carto 0.1m
    W_EXPECTED_RPS = np.deg2rad(0.1)       # → angle = very small, floor at 0.5°

    motion_params = make_motion_filter_from_expected_velocity(
        target_insert_period_s=TARGET_INSERT_PERIOD_S,
        v_expected_mps=V_EXPECTED_MPS,
        w_expected_rps=W_EXPECTED_RPS,
        min_dist=0.05,
        min_ang=np.deg2rad(0.3),   # floor ~0.005 rad, close to Cartographer 0.004
        max_dist=0.15,             # cap at 0.15m (previously 0.50m)
        max_ang=np.deg2rad(2.0),   # cap at 2° (previously 10°)
    )

    print(
        "MotionFilter thresholds:",
        f"time={motion_params.max_time_seconds:.3f}s",
        f"dist={motion_params.max_distance_meters:.3f}m",
        f"angle={np.rad2deg(motion_params.max_angle_radians):.2f}deg",
    )

    # ------------------------------------------------------------------
    # Pose graph backend — Cartographer canonical configuration
    # ------------------------------------------------------------------
    backend = PyCeresBackend2D(
        # Cartographer pose_graph.lua: optimization_problem.huber_scale = 1e1
        huber_scale=1e1,
        # AUTO: selects SPARSE_NORMAL_CHOLESKY for large graphs, DENSE_QR for small
        linear_solver_type="AUTO",
        num_threads=1,
        minimizer_progress_to_stdout=False,
        # Cartographer pose_graph.lua: local_slam_pose_*_weight = 1e5
        local_slam_pose_translation_weight=1e5,
        local_slam_pose_rotation_weight=1e5,
    )
    backend.set_fixed("submap", 0)
    print("Using backend:", type(backend).__name__)

    pg = PoseGraph2D(
        backend=backend,
        # Pass the live submap builder so optimized poses are written back after solve
        submap_builder=submaps,
        # Cartographer pose_graph.lua: matcher_translation/rotation_weight
        intra_translation_weight=5e2,
        intra_rotation_weight=1.6e3,
    )

    # ------------------------------------------------------------------
    # Global SLAM / loop closure
    # ------------------------------------------------------------------
    global_slam = None
    if MATCHER_TYPE == "scan_to_submap":
        loop_config = LoopClosureConfig(
            min_score=float(loop_submap_backend_config.min_score),
            global_localization_min_score=float(loop_submap_backend_config.global_localization_min_score),
            translation_weight=1.1e4,
            rotation_weight=1e5,
            optimize_every_n_nodes=90,
            min_node_index_separation=90,
            spatial_search_radius=15.0,
            max_candidate_targets_per_new_node=3,  # new_node_max_targets=3
            historical_node_stride=3,              # finished_submap_node_stride=3
            max_candidate_nodes_per_finished_target=0,
            recent_finished_submap_exclusion=3,
            finished_submap_verification_budget_per_tick=24,
            force_full_submap_for_finished_submap_search=False,
            finished_submap_full_search_failure_threshold=3,
        )
        global_slam = CartoGlobalSlam2D(
            loop_closure_adapter=CartoLoopClosureAdapter(
                matcher=loop_matcher,
                pose_graph=pg,
                config=loop_config,
            ),
            pose_graph=pg,
            # Cartographer: optimize_every_n_nodes = 90
            optimize_every_n_nodes=90,
            # Will be wired to `adapter` below after it is constructed
            adapter=None,
            correction_alpha=0.5,
        )

    # ------------------------------------------------------------------
    # SLAM adapter
    # ------------------------------------------------------------------
    adapter = CartoLocalSlamAdapter(
        matcher_manager=matcher_manager,
        extrapolator=extrap,
        pose_graph=pg,
        motion_params=motion_params,
        solve_every_n_nodes=30,
        global_slam=global_slam,
    )

    # Wire the adapter reference into global_slam so that post-solve
    # extrapolator corrections can be applied (Work Item 8 & 9).
    if global_slam is not None:
        global_slam.set_adapter(adapter)

    # ------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------
    first = scans[0]
    adapter.initialize_extrapolator(
        float(first["t"]),
        Pose2(*first["odom"]),
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    os.makedirs("carto_outputs", exist_ok=True)

    out_prefix = f"carto_outputs/trajectory_{MATCHER_TYPE}_loop_{len(scans)}"
    traj_path = f"{out_prefix}.txt"
    meta_path = f"{out_prefix}_debug.txt"

    last_k = -1
    last_t = float("nan")
    last_pose: Pose2 | None = None
    last_checkpoint_node_count = -1

    with open(traj_path, "w") as f_traj, open(meta_path, "w") as f_meta:
        try:
            f_meta.write(f"# matcher_type={MATCHER_TYPE}\n")
            f_meta.write("# loop_closure_enabled=1\n")
            f_meta.write(
                "k t x y theta score inliers dx dy dtheta do_insert did_insert "
                "constraints_total constraints_intra constraints_loop "
                "loop_candidates loop_accepted loop_rejected loop_duplicates "
                "nodes submaps\n"
            )

            # ----------------------------------------------------------
            # Main loop
            # ----------------------------------------------------------
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
                                f"status={getattr(ev, 'status', 'unknown')}",
                                f"source={getattr(ev, 'source', 'unknown')}",
                                f"full_submap={getattr(ev, 'used_full_submap', False)}",
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

    # ------------------------------------------------------------------
    # Finalize and save final outputs
    # ------------------------------------------------------------------
    adapter.finalize()
    _write_optimized_outputs(pg, out_prefix)

    if hasattr(pg.backend, "get_last_summary"):
        summary = pg.backend.get_last_summary()
        if summary is not None:
            try:
                print("CERES_SUMMARY:", summary.BriefReport())
            except Exception:
                print("CERES_SUMMARY:", summary)

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

        if hasattr(global_slam, "get_diagnostics_summary"):
            diag = global_slam.get_diagnostics_summary()
            print(
                "FINAL_LOOP_DIAGNOSTICS:",
                f"accepted_from_new_node_search={diag.get('accepted_from_new_node_search', 0)}",
                f"accepted_from_finished_submap_search={diag.get('accepted_from_finished_submap_search', 0)}",
                f"rejected_sampled_out={diag.get('rejected_sampled_out', 0)}",
                f"rejected_matcher_failed={diag.get('rejected_matcher_failed', 0)}",
                f"rejected_score_failed={diag.get('rejected_score_failed', 0)}",
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


if __name__ == "__main__":
    main()
