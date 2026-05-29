"""
Real-time Hector SLAM visualization runner (standalone).

Replays a dataset at its real sensor cadence (sleep to scan timestamps) and draws
the trajectory LIVE in an on-screen window as each scan is processed — mirroring how
mapping behaves in a real run. The occupancy map is built and shown only at the END,
after the full run completes.

This is intentionally separate from the stable batch runner (hector/run_local_slam_new.py),
which is left untouched. It reuses the same matcher / adapter / PGO machinery.

Examples:
  .venv/bin/python -m hector.run_realtime_viz --dataset lab_run_2 --matcher scan_to_map
  .venv/bin/python -m hector.run_realtime_viz --dataset lab_run_2 --matcher scan_to_submap
  .venv/bin/python -m hector.run_realtime_viz --dataset lab_run_2 --matcher scan_to_submap \
        --enable-pgo --scans-per-submap 250

Per-stage timing is printed live and summarized at the end so that, if real-time is not
achieved, the dominant stage (preprocess / slam / draw) is immediately visible.
"""
from __future__ import annotations

import argparse
import bisect
import os
import queue
import threading
import time

import numpy as np

from slam_core.common.types import Pose2
from slam_core.common.se2 import pose_compose, inverse_pose
from slam_core.dataio.dataset_catalog import load_dataset_scans

import hector.config as cfg

from carto.local_slam.range_to_points import ranges_to_points
from carto.local_slam.pose_extrapolator import PoseExtrapolatorCV

from slam_core.matching.core import MatcherManager
from slam_core.matching.preprocessing import PointCloudProcessor, PointCloudProcessorConfig
from slam_core.matching.scan_to_submap import (
    SubmapBuilder2D,
    ScanToSubmapMatcher,
    ScanToSubmapBackendConfig,
    SubmapSearchWindow,
)
from slam_core.matching.scan_to_map import ScanToMapMatcher, _transform_points

from hector.adapter import (
    HectorLocalSlamAdapter,
    MotionFilterParams,
    make_motion_filter_from_expected_velocity,
)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hector SLAM — real-time visualization runner (live trajectory + timing)"
    )
    p.add_argument("--dataset", choices=list(cfg._PROFILES.keys()), default=None)
    p.add_argument("--scan-variant", choices=["raw", "360"], default=None, dest="scan_variant")
    p.add_argument("--max-scans", type=int, default=None, dest="max_scans")
    p.add_argument("--matcher", choices=["scan_to_submap", "scan_to_map"], default=None)
    p.add_argument(
        "--enable-pgo", action="store_true", dest="enable_pgo",
        help="Enable online g2o pose-graph optimization (scan_to_submap only).",
    )
    p.add_argument("--scans-per-submap", type=int, default=None, dest="scans_per_submap")
    p.add_argument(
        "--speed", type=float, default=cfg.PLAYBACK_SPEED,
        help="Playback speed multiplier. 1.0=real-time (sleep to timestamps), "
             "2.0=2x, 0=as-fast-as-possible (no sleeping). Default from cfg.PLAYBACK_SPEED.",
    )
    p.add_argument("--draw-every", type=int, default=cfg.DRAW_EVERY, dest="draw_every",
                   help="Redraw the live plot every N scans (lower = smoother, costlier). "
                        "Default from cfg.DRAW_EVERY.")
    p.add_argument("--verbose-every", type=int, default=cfg.REALTIME_VERBOSE_EVERY, dest="verbose_every",
                   help="Print a live timing line every N scans. Default from cfg.REALTIME_VERBOSE_EVERY.")
    p.add_argument("--no-map", action="store_true", dest="no_map",
                   help="Skip building/showing the final occupancy map.")
    p.add_argument("--live-points", action="store_true", dest="live_points",
                   help="Overlay the accumulating world-frame laser point cloud on the live "
                        "trajectory window (and save a <prefix>_points.png at the end).")
    p.add_argument("--points-max", type=int, default=cfg.POINTS_MAX, dest="points_max",
                   help="Cap displayed point-cloud points (stride-subsampled above this). "
                        "Default from cfg.POINTS_MAX.")
    p.add_argument("--save-prefix", default=None, dest="save_prefix",
                   help="Output filename prefix (default: realtime_<dataset>_<matcher>).")
    p.add_argument(
        "--fast-match", action="store_true", dest="fast_match",
        help="Fast local matching (scan_to_submap only): same full correlative search + "
             "refine as default, but vectorized + vectorized submap insertion. ~10x faster "
             "per scan with the same map quality.",
    )
    p.add_argument(
        "--local-solver", choices=["pyceres", "native"], default="native", dest="local_solver",
        help="Continuous local-refine solver used by --fast-match (default: native "
             "GaussNewtonLM). 'pyceres' = Cartographer CeresScanMatcher2D (slower cost fn).",
    )
    p.add_argument(
        "--use-imu", action="store_true", dest="use_imu",
        help="Feed the IMU (imu.csv: gyro yaw-rate + quaternion yaw) into the extrapolator "
             "for a better motion prediction. Forces the extrapolator ON.",
    )
    p.add_argument(
        "--use-motion-filter", action="store_true", dest="use_motion_filter",
        help="Cartographer-style motion-filter keyframing for scan_to_map: skip GN matching "
             "on sub-threshold scans (dead-reckon them via the extrapolator). Visualizes the "
             "skip live (SKIP markers). Forces the extrapolator ON; pair with --use-imu.",
    )
    p.add_argument(
        "--switch-grace-scans", type=int, default=cfg.SWITCH_GRACE_SCANS, dest="switch_grace_scans",
        help="When you switch the matcher live (type 'map'/'submap' on stdin), the original "
             "matcher keeps running for this many more scans before the switch takes effect "
             "(stable handoff). Default 15.",
    )
    return p.parse_args()


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    if args.dataset is not None:
        cfg.DATASET_NAME = args.dataset
        cfg._apply_profile(args.dataset)
    if args.scan_variant is not None:
        cfg.DATASET_SCAN_VARIANT = args.scan_variant
    if args.max_scans is not None:
        cfg.MAX_SCANS = args.max_scans
    if args.matcher is not None:
        cfg.MATCHER_TYPE = args.matcher
    if getattr(args, "scans_per_submap", None) is not None:
        cfg.SCANS_PER_SUBMAP = int(args.scans_per_submap)


def _resolve_initial_pose(profile, first_scan) -> Pose2:
    if profile.has_odom and first_scan.get("odom") is not None:
        return Pose2(*first_scan["odom"])
    return Pose2(cfg.INITIAL_POSE_X, cfg.INITIAL_POSE_Y, cfg.INITIAL_POSE_THETA)


# ----------------------------------------------------------------------------
# Matcher / PGO construction (mirrors run_local_slam_new.py; stable file untouched)
# ----------------------------------------------------------------------------
def _build_submap_matcher(submaps: SubmapBuilder2D, fast_match: bool = False,
                          local_solver: str = "native") -> ScanToSubmapMatcher:
    # Fast mode keeps the SAME full correlative search + refine as the default (so map
    # quality is unchanged), but runs the search vectorized (batched NumPy instead of a
    # per-candidate Python loop) — combined with vectorized submap insertion (fast_insert)
    # this removes both per-scan bottlenecks. local_solver picks the refine backend.
    fast = bool(fast_match)
    config = ScanToSubmapBackendConfig(
        backend_type="two_stage_bruteforce",
        use_vectorized_search=fast,
        local_refine_backend=(str(local_solver) if fast else "native"),
        reject_below_min_score=True,
        min_score=cfg.SUBMAP_MIN_SCORE,
        max_match_points=cfg.SUBMAP_MAX_MATCH_POINTS,
        max_refine_points=cfg.SUBMAP_MAX_REFINE_POINTS,
        min_valid=cfg.SUBMAP_MIN_VALID,
        precomp_levels=cfg.SUBMAP_PRECOMP_LEVELS,
        do_refine=True,
        refine_min_points=cfg.SUBMAP_REFINE_MIN_POINTS,
        refine_w_trans=cfg.SUBMAP_REFINE_W_TRANS,
        refine_w_rot=cfg.SUBMAP_REFINE_W_ROT,
        refine_iters=cfg.SUBMAP_REFINE_ITERS,
        refine_damping=cfg.SUBMAP_REFINE_DAMPING,
        refine_step_clip_xy=cfg.SUBMAP_REFINE_STEP_CLIP_XY,
        refine_step_clip_th=float(np.deg2rad(cfg.SUBMAP_REFINE_STEP_CLIP_TH_DEG)),
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
    return ScanToSubmapMatcher(submap_builder=submaps, backend_config=config)


def _build_map_matcher() -> ScanToMapMatcher:
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
    corr_params_map = dict(
        gn_iters_per_level=cfg.GN_ITERS_PER_LEVEL,
        gn_damping=cfg.GN_DAMPING,
        min_points=cfg.CORR_MAP_MIN_POINTS,
        min_inliers_accept=cfg.CORR_MAP_MIN_INLIERS,
        min_score=cfg.CORR_MAP_MIN_SCORE,
        step_clip_xy=cfg.CORR_MAP_STEP_CLIP_XY,
        step_clip_th=np.deg2rad(cfg.GN_STEP_CLIP_TH_DEG),
    )
    return ScanToMapMatcher(map_params=map_params, corr_params=corr_params_map)


def _build_pgo_stack(submaps: SubmapBuilder2D):
    """Returns (pose_graph, global_slam). Construction matches the stable runner."""
    from carto.pose_graph.pose_graph_2d import PoseGraph2D
    from carto.pose_graph.backends.g2o_backend_2d import G2oBackend2D
    from carto.pose_graph.global_slam_2d import CartoGlobalSlam2D
    from carto.loop_closure_adapter import CartoLoopClosureAdapter
    from slam_core.loop_closure import LoopClosureConfig

    loop_backend_config = ScanToSubmapBackendConfig(
        backend_type="branch_and_bound",
        local_refine_backend="native",
        min_score=float(cfg.PGO_LOOP_MIN_SCORE),
        global_localization_min_score=float(cfg.PGO_LOOP_MIN_SCORE),
        min_valid=cfg.SUBMAP_MIN_VALID,
        precomp_levels=int(cfg.PGO_LOOP_PRECOMP_LEVELS),
        do_refine=True,
        max_match_points=cfg.SUBMAP_MAX_MATCH_POINTS,
        max_refine_points=cfg.SUBMAP_MAX_REFINE_POINTS,
        refine_min_points=cfg.SUBMAP_REFINE_MIN_POINTS,
        refine_w_trans=cfg.SUBMAP_REFINE_W_TRANS,
        refine_w_rot=cfg.SUBMAP_REFINE_W_ROT,
        coarse=SubmapSearchWindow(
            xy_window=float(cfg.PGO_LOOP_SEARCH_XY),
            theta_window=float(np.deg2rad(cfg.PGO_LOOP_SEARCH_TH_DEG)),
            xy_step=0.05,
            theta_step=0.02,
            level=0,
        ),
        fine=None,
        bnb_depth_limit=int(cfg.PGO_LOOP_BNB_DEPTH),
        bnb_min_rotational_step=float(cfg.PGO_LOOP_BNB_MIN_ROT_STEP),
        bnb_branching=int(cfg.PGO_LOOP_BNB_BRANCHING),
    )
    loop_matcher = ScanToSubmapMatcher(submap_builder=submaps, backend_config=loop_backend_config)

    g2o_backend = G2oBackend2D(
        huber_scale=float(cfg.PGO_HUBER_SCALE),
        max_num_iterations=int(cfg.PGO_MAX_ITERATIONS),
        local_slam_pose_translation_weight=float(cfg.PGO_LOCAL_TRANS_WEIGHT),
        local_slam_pose_rotation_weight=float(cfg.PGO_LOCAL_ROT_WEIGHT),
    )
    g2o_backend.set_fixed("submap", 0)
    print("Using PGO backend:", type(g2o_backend).__name__)

    pose_graph = PoseGraph2D(
        backend=g2o_backend,
        submap_builder=submaps,
        intra_translation_weight=float(cfg.PGO_INTRA_TRANS_WEIGHT),
        intra_rotation_weight=float(cfg.PGO_INTRA_ROT_WEIGHT),
    )

    loop_config = LoopClosureConfig(
        min_score=float(cfg.PGO_LOOP_MIN_SCORE),
        translation_weight=float(cfg.PGO_LOOP_TRANS_WEIGHT),
        rotation_weight=float(cfg.PGO_LOOP_ROT_WEIGHT),
        min_node_index_separation=int(cfg.PGO_MIN_NODE_SEPARATION),
        spatial_search_radius=float(cfg.PGO_SPATIAL_SEARCH_RADIUS),
        max_candidate_targets_per_new_node=int(cfg.PGO_MAX_CANDIDATE_TARGETS),
        historical_node_stride=int(cfg.PGO_HISTORICAL_NODE_STRIDE),
        check_every_n_nodes=int(cfg.PGO_CHECK_EVERY_N_NODES),
        recent_finished_submap_exclusion=int(cfg.PGO_RECENT_SUBMAP_EXCLUSION),
    )
    global_slam = CartoGlobalSlam2D(
        loop_closure_adapter=CartoLoopClosureAdapter(
            matcher=loop_matcher, pose_graph=pose_graph, config=loop_config,
        ),
        pose_graph=pose_graph,
        optimize_every_n_nodes=int(cfg.PGO_OPTIMIZE_EVERY_N_NODES),
        adapter=None,
        correction_alpha=float(cfg.PGO_CORRECTION_ALPHA),
    )
    return pose_graph, global_slam


# ----------------------------------------------------------------------------
# Live trajectory plot
# ----------------------------------------------------------------------------
class LiveTrajectoryPlot:
    """A single reusable matplotlib window showing the growing robot path.

    Falls back gracefully to a no-op (headless) mode if no interactive backend
    is available; the run still completes and the final map PNG is saved.
    """

    def __init__(self, title: str):
        self.ok = False
        self.fig = None
        self.ax = None
        self._line = None
        self._cur = None
        self._start = None
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            print("[viz] No DISPLAY/WAYLAND_DISPLAY; running headless (final map PNG only).")
            return

        import matplotlib
        # matplotlib.use() does NOT import the GUI binding — that happens lazily
        # when the first figure is created. So we must actually build the figure
        # inside the try to detect a backend whose binding is missing (e.g. Qt5Agg
        # with no PyQt5 in the venv) and fall through to the next candidate.
        last_err = None
        for backend in ("Qt5Agg", "TkAgg", "GTK3Agg"):
            try:
                matplotlib.use(backend, force=True)
                import matplotlib.pyplot as plt
                plt.ion()
                fig, ax = plt.subplots(figsize=(8, 8))
                fig.canvas.draw()           # forces the GUI binding to load
                plt.show(block=False)
            except Exception as e:
                last_err = e
                try:
                    import matplotlib.pyplot as _plt
                    _plt.close("all")
                except Exception:
                    pass
                continue
            # backend works — finish setup
            self._plt = plt
            self.fig, self.ax = fig, ax
            ax.set_title(title)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.3)
            # World-frame laser point cloud (drawn UNDER the trajectory). Hidden
            # unless point data is supplied, so runs without --live-points are unchanged.
            (self._pts,) = ax.plot([], [], ".", color="#555555", ms=1.0, alpha=0.5,
                                   zorder=1, label="point cloud")
            (self._line,) = ax.plot([], [], "-", color="#1f77b4", lw=1.2, zorder=3,
                                    label="trajectory")
            # Keyframe markers: poses where GN actually ran (motion-filter mode).
            # Hidden unless keyframe data is supplied, so non-MF runs look unchanged.
            (self._kf,) = ax.plot([], [], ".", color="#ffaa00", ms=4, zorder=4, label="keyframe (GN)")
            (self._cur,) = ax.plot([], [], "o", color="#ff4466", ms=7, zorder=5, label="current")
            (self._start,) = ax.plot([], [], "o", color="#00aa55", ms=8, zorder=5, label="start")
            ax.legend(loc="upper right", fontsize=8)
            fig.canvas.draw()
            plt.show(block=False)
            self.ok = True
            print(f"[viz] Live window backend: {backend}")
            return

        print(f"[viz] No usable interactive backend ({last_err}); running headless (final map PNG only).")

    def update(self, xs, ys, kf_xs=None, kf_ys=None, pts_x=None, pts_y=None) -> None:
        if not self.ok or len(xs) == 0:
            return
        if pts_x is not None and pts_y is not None:
            self._pts.set_data(pts_x, pts_y)
        self._line.set_data(xs, ys)
        if kf_xs is not None and kf_ys is not None:
            self._kf.set_data(kf_xs, kf_ys)
        self._cur.set_data([xs[-1]], [ys[-1]])
        self._start.set_data([xs[0]], [ys[0]])
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def keep_open(self) -> None:
        if self.ok:
            print("[viz] Run complete — close the plot window to exit.")
            self._plt.ioff()
            self._plt.show(block=True)


# ----------------------------------------------------------------------------
# Timing accumulator
# ----------------------------------------------------------------------------
class StageTimer:
    STAGES = ("preprocess", "slam", "draw", "sleep")

    def __init__(self):
        self.total = {s: 0.0 for s in self.STAGES}
        self.max = {s: 0.0 for s in self.STAGES}
        self.count = 0
        self.late_frames = 0  # compute (excl. sleep) exceeded the inter-scan period

    def add(self, stage: str, dt: float) -> None:
        self.total[stage] += dt
        if dt > self.max[stage]:
            self.max[stage] = dt

    def compute_total(self) -> float:
        """Sum of all stages except sleep (the actual work per run)."""
        return sum(self.total[s] for s in self.STAGES if s != "sleep")

    def print_summary(self, total_scans: int, final_lag: float) -> None:
        comp = self.compute_total()
        print("\n" + "=" * 64)
        print("  PER-STAGE TIMING SUMMARY")
        print("=" * 64)
        print(f"  {'stage':<12} {'total[s]':>10} {'mean[ms]':>10} {'max[ms]':>10} {'%comp':>8}")
        print("  " + "-" * 58)
        for s in self.STAGES:
            tot = self.total[s]
            mean_ms = (tot / total_scans * 1e3) if total_scans else 0.0
            max_ms = self.max[s] * 1e3
            pct = (tot / comp * 100.0) if (comp > 0 and s != "sleep") else 0.0
            pct_str = f"{pct:6.1f}%" if s != "sleep" else "     -"
            print(f"  {s:<12} {tot:>10.3f} {mean_ms:>10.2f} {max_ms:>10.2f} {pct_str:>8}")
        print("  " + "-" * 58)
        mean_comp_ms = (comp / total_scans * 1e3) if total_scans else 0.0
        print(f"  {'COMPUTE':<12} {comp:>10.3f} {mean_comp_ms:>10.2f}")
        print("=" * 64)
        # bottleneck = the work stage with the largest total
        work = {s: self.total[s] for s in self.STAGES if s != "sleep"}
        bottleneck = max(work, key=work.get) if work else "n/a"
        print(f"  Scans            : {total_scans}")
        print(f"  Late frames      : {self.late_frames} "
              f"({(100.0 * self.late_frames / total_scans) if total_scans else 0.0:.1f}% over sensor period)")
        print(f"  Bottleneck stage : {bottleneck}")
        if final_lag <= 0.05:
            print(f"  Verdict          : REAL-TIME OK (final lag {final_lag:.3f}s)")
        else:
            print(f"  Verdict          : BEHIND real-time by {final_lag:.3f}s — bottleneck = {bottleneck}")
        print("=" * 64)


# ----------------------------------------------------------------------------
# PGO dense-trajectory reconstruction (by scan time -> most-recent keyframe correction)
# ----------------------------------------------------------------------------
def _reconstruct_pgo_poses_by_time(pose_graph, scan_records) -> list[Pose2]:
    """Map each (t, online_pose) scan record to its corrected pose using the rigid SE(2)
    correction of the most-recent keyframe NODE at/just-before that scan time. Time-based
    (not node-id based) so it works with the asynchronous back-end where node ids are
    assigned later by the worker thread."""
    deltas = []   # (node_time, correction Pose2)
    for nd in pose_graph.nodes:
        online = pose_graph.drifted_nodes.get(int(nd.id))
        if online is None:
            continue
        deltas.append((float(nd.time), pose_compose(nd.pose, inverse_pose(online))))
    deltas.sort(key=lambda x: x[0])
    times = [d[0] for d in deltas]
    identity = Pose2(0.0, 0.0, 0.0)
    out = []
    for (t_s, pose_s) in scan_records:
        i = bisect.bisect_right(times, float(t_s)) - 1
        delta = deltas[i][1] if i >= 0 else identity
        out.append(pose_compose(delta, pose_s))
    return out


# ----------------------------------------------------------------------------
# Asynchronous (background-thread) global SLAM — Cartographer-style decoupling
# ----------------------------------------------------------------------------
class AsyncPoseGraphRunner:
    """Runs loop-closure search + pose-graph optimization on a background thread so the
    real-time front-end never blocks on the (heavy) global back-end.

    Thread-safety model:
      - The worker thread is the SOLE owner of pose_graph + global_slam; the main thread
        never calls into them during the run (it only submits keyframe jobs). So there are
        no graph/backend data races.
      - The one shared object is the submap_builder: the main thread inserts into ACTIVE
        submaps while the worker reads FINISHED submaps (whose grids are immutable once
        finished) for loop matching and writes optimized submap poses back. Under CPython's
        GIL these are safe (list ops / Pose2 reference assignment are atomic; finished grids
        never change).
    The g2o solve is C++ and releases the GIL, so it genuinely overlaps the front-end; the
    front-end's real-time sleep between scans also lets the worker drain its queue.
    """

    def __init__(self, pose_graph, global_slam):
        self.pose_graph = pose_graph
        self.global_slam = global_slam
        self._q: "queue.Queue" = queue.Queue()
        self.node_count = 0
        self._busy = False
        self._thread = threading.Thread(target=self._worker, name="pgo-worker", daemon=True)
        self._thread.start()

    def submit(self, t, pose, scan_points, insertion_submaps) -> None:
        """Enqueue an accepted keyframe for asynchronous graph insertion + loop search."""
        self._q.put((float(t), pose, scan_points, list(insertion_submaps)))

    def backlog(self) -> int:
        return self._q.qsize() + (1 if self._busy else 0)

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                break
            t, pose, scan_points, submaps = item
            self._busy = True
            try:
                node_id = self.pose_graph.add_node_with_intra_constraints(
                    t=t, node_pose_world=pose, active_submaps=submaps)
                self.node_count += 1
                self.global_slam.on_node_inserted(
                    node_id=node_id, timestamp=t, scan_points=scan_points,
                    pose_global=pose, insertion_submaps=submaps)
            except Exception as e:  # never let a back-end hiccup kill the worker
                print(f"[pgo-worker] error processing node: {e}")
            finally:
                self._busy = False
                self._q.task_done()

    def finalize(self) -> None:
        """Wait for the queue to drain, stop the worker, then run the final global solve
        (single-threaded again, so this is safe)."""
        self._q.join()
        self._q.put(None)
        self._thread.join(timeout=120)
        if self.global_slam is not None:
            self.global_slam.finalize()


# ----------------------------------------------------------------------------
# Live matcher-switch trigger (stdin reader thread)
# ----------------------------------------------------------------------------
# Map typed commands -> canonical matcher name. The reader thread only pushes a
# requested NAME onto a queue; the main loop drains it between scans and calls
# matcher_manager.request_switch, so all manager mutation stays single-threaded.
_SWITCH_ALIASES = {
    "map": "scan_to_map", "m": "scan_to_map", "1": "scan_to_map",
    "scan_to_map": "scan_to_map",
    "submap": "scan_to_submap", "s": "scan_to_submap", "2": "scan_to_submap",
    "scan_to_submap": "scan_to_submap",
}


def _start_switch_reader(request_q: "queue.Queue") -> threading.Thread:
    """Background daemon thread reading stdin lines for live switch commands."""
    def _reader() -> None:
        import sys
        for line in sys.stdin:                       # blocks; ends on EOF
            cmd = line.strip().lower()
            if not cmd:
                continue
            if cmd in ("status", "?"):
                request_q.put(("status", None))
            elif cmd in ("quit", "exit", "q"):
                request_q.put(("quit", None))
            elif cmd in _SWITCH_ALIASES:
                request_q.put(("switch", _SWITCH_ALIASES[cmd]))
            else:
                print(f"[switch] unknown command {cmd!r}. Type: map | submap | status | quit")
    th = threading.Thread(target=_reader, name="switch-reader", daemon=True)
    th.start()
    return th


def _apply_switch_requests(request_q, manager, matchers_by_name, grace, k) -> bool:
    """Drain queued stdin commands on the MAIN thread (so the manager is only ever
    mutated single-threaded). Returns True if a quit was requested."""
    quit_requested = False
    while True:
        try:
            kind, payload = request_q.get_nowait()
        except queue.Empty:
            break
        if kind == "quit":
            quit_requested = True
        elif kind == "status":
            pend = manager._target_matcher_name if manager.switch_requested else None
            print(f"[switch] active={manager.active_matcher.name} pending={pend} "
                  f"grace_remaining={manager.grace_remaining} (k={k})")
        elif kind == "switch":
            target = matchers_by_name.get(payload)
            if target is None:
                continue
            if target is manager.active_matcher:
                print(f"[switch] already running {payload}; ignored.")
                continue
            manager.request_switch(target, grace_scans=grace)
            # The actual switch also waits until the rolling buffer holds
            # >= min_buffer_for_switch scans, so an early request just defers (it
            # never switches mid-bootstrap).
            extra = ("" if len(manager.buffer) >= manager.min_buffer_for_switch
                     else f" (and once >= {manager.min_buffer_for_switch} scans of history exist)")
            print(f"[switch] requested {payload}; effective in ~{grace} scans (after k={k}){extra}.")
    return quit_requested


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    _apply_cli_overrides(args)

    profile, scans = load_dataset_scans(cfg.DATASET_NAME, scan_variant=cfg.DATASET_SCAN_VARIANT)
    if not scans:
        raise RuntimeError(f"No scans loaded from {profile.scan_path}")

    matcher_type = cfg.MATCHER_TYPE
    if cfg.MAX_SCANS is not None:
        scans = scans[: int(cfg.MAX_SCANS)]

    enable_pgo = bool(args.enable_pgo) and matcher_type == "scan_to_submap"
    if bool(args.enable_pgo) and matcher_type != "scan_to_submap":
        print("[warn] --enable-pgo ignored: only valid with --matcher scan_to_submap")

    # Opt-in IMU aiding + motion-filter keyframing. Either forces the
    # extrapolator on (both depend on a real motion prediction, not last-pose).
    use_motion_filter = bool(getattr(args, "use_motion_filter", False)) \
        or bool(getattr(cfg, "USE_MOTION_FILTER", False))
    use_imu = bool(getattr(args, "use_imu", False)) \
        or bool(getattr(cfg, "USE_IMU", False))
    use_extrapolator = (
        bool(getattr(cfg, "USE_EXTRAPOLATOR", True)) or use_imu or use_motion_filter
    )

    print("=" * 60)
    print(f"Dataset      : {cfg.DATASET_NAME}  ({cfg.DATASET_SCAN_VARIANT})")
    print(f"Total scans  : {len(scans)}")
    print(f"Matcher      : {matcher_type}")
    print(f"Extrapolator : {'ON' if use_extrapolator else 'OFF (last pose)'}")
    print(f"Motion filter: {'ON (keyframe skip)' if use_motion_filter else 'OFF (match every scan)'}")
    print(f"Online PGO   : {'ON' if enable_pgo else 'OFF'}")
    print(f"Playback     : {'real-time (sleep to timestamps)' if args.speed > 0 else 'as-fast-as-possible'}"
          f"  speed={args.speed}x")
    print("=" * 60)

    # --- preprocessing + matcher + (optional) PGO stack ---
    point_processor = PointCloudProcessor(
        PointCloudProcessorConfig(
            fixed_voxel_size=cfg.VOXEL_FIXED_SIZE,
            adaptive_voxel_max_size=cfg.VOXEL_ADAPTIVE_MAX_SIZE,
            adaptive_min_num_points=cfg.VOXEL_ADAPTIVE_MIN_POINTS,
            adaptive_num_iterations=cfg.VOXEL_ADAPTIVE_ITERS,
            enabled=cfg.VOXEL_FILTER_ENABLED,
        )
    )

    submaps = SubmapBuilder2D(
        submap_size_m=cfg.SUBMAP_SIZE_METERS,
        resolution=cfg.SUBMAP_RESOLUTION,
        scans_per_submap=cfg.SCANS_PER_SUBMAP,
        ray_steps=cfg.RAY_STEPS,
        l0=cfg.L0, l_occ=cfg.L_OCC, l_free=cfg.L_FREE, l_min=cfg.L_MIN, l_max=cfg.L_MAX,
    )

    # The realtime runner ALWAYS uses the vectorized (fast) path for both search and
    # insertion, regardless of --fast-match.  The slow Python paths cost:
    #   - correlative search: ~700ms/scan (18k candidates × 2 submaps, pure Python loop)
    #   - Bresenham insertion: ~200ms/scan (200pts × 2 submaps, pure Python loop)
    # Both dwarf the 104ms scan period of lab_run_2, causing seconds of accumulated lag.
    # The vectorized paths produce bit-identical scores / map quality; there is no reason
    # to use the slow paths in a real-time context.
    # --fast-match is kept as a no-op flag for backwards-compatibility.
    fast_match = True
    submaps.fast_insert = True

    # Build BOTH matchers up front so the live front-end can be switched between them.
    # --matcher only selects which one is INITIALLY active; the other is the switch target.
    submap_matcher = _build_submap_matcher(
        submaps, fast_match=fast_match,
        local_solver=str(getattr(args, "local_solver", "native")),
    )
    map_matcher = _build_map_matcher()
    matchers_by_name = {"scan_to_submap": submap_matcher, "scan_to_map": map_matcher}
    if matcher_type not in matchers_by_name:
        raise ValueError(f"Unsupported matcher: {matcher_type!r}")
    matcher = matchers_by_name[matcher_type]
    if matcher_type == "scan_to_submap":
        print(f"Local match  : vectorized search + insert ({str(getattr(args, 'local_solver', 'native'))} refine)")

    # Live matcher switching (typed on stdin) — local mapping only. Disabled with PGO
    # because the async PGO worker is bound to the submap builder (deferred to a later phase).
    switching_enabled = not enable_pgo
    switch_grace = int(getattr(args, "switch_grace_scans", 15))

    matcher_manager = MatcherManager(
        active_matcher=matcher, rolling_buffer_size=cfg.ROLLING_BUFFER_SIZE,
        min_buffer_for_switch=cfg.MIN_BUFFER_FOR_SWITCH,
    )

    # --- optional IMU stream for the extrapolator (use_imu resolved above) ---
    imu_samples = []
    if use_imu:
        if getattr(profile, "imu_path", None) is not None and os.path.exists(str(profile.imu_path)):
            from slam_core.dataio.imu_csv import read_imu_csv
            from carto.local_slam.imu_extrapolation import imu_rows_to_samples
            imu_samples = imu_rows_to_samples(read_imu_csv(str(profile.imu_path)))
            print(f"IMU          : ON ({len(imu_samples)} samples from {profile.imu_path})")
        else:
            use_imu = False
            print("[warn] --use-imu ignored: no imu_path for this dataset")

    if use_motion_filter and not use_imu and getattr(profile, "imu_path", None) is not None:
        print("[warn] Motion filter is ON but IMU is OFF. Skipped scans are dead-reckoned by "
              "constant-velocity only, which drifts badly through turns. Add --use-imu.")

    extrap = PoseExtrapolatorCV(
        max_dt=cfg.EXTRAP_MAX_DT, init_vxy=cfg.EXTRAP_INIT_VXY, init_wz=cfg.EXTRAP_INIT_WZ,
        use_imu=use_imu,
        imu_yaw_correction_alpha=float(getattr(cfg, "IMU_YAW_CORRECTION_ALPHA", 0.02)),
    )

    motion_params = make_motion_filter_from_expected_velocity(
        target_insert_period_s=cfg.TARGET_INSERT_PERIOD_S,
        v_expected_mps=cfg.V_EXPECTED_MPS,
        w_expected_rps=cfg.W_EXPECTED_RPS,
    )

    # Explicit keyframe thresholds for the scan_to_map motion-filter-skip decision
    # (kept separate from motion_params, which drives submap insertion cadence).
    mf_keyframe_params = MotionFilterParams(
        max_time_seconds=float(getattr(cfg, "MF_MAX_TIME_S", 0.5)),
        max_distance_meters=float(getattr(cfg, "MF_MAX_DIST_M", 0.10)),
        max_angle_radians=float(np.deg2rad(float(getattr(cfg, "MF_MAX_ANGLE_DEG", 2.0)))),
        min_distance_meters=0.0,
        min_angle_radians=0.0,
        max_distance_cap_meters=10.0,
        max_angle_cap_radians=np.deg2rad(180.0),
    )
    if use_motion_filter:
        print(
            "MotionFilter keyframe thresholds (scan_to_map skip):"
            f"  time={mf_keyframe_params.max_time_seconds:.3f}s"
            f"  dist={mf_keyframe_params.max_distance_meters:.3f}m"
            f"  angle={np.rad2deg(mf_keyframe_params.max_angle_radians):.2f}deg"
        )

    pose_graph = None
    global_slam = None
    apgo = None
    if enable_pgo:
        pose_graph, global_slam = _build_pgo_stack(submaps)
        # Run the global back-end (loop closure + optimization) on a background thread so
        # the real-time front-end never blocks. The adapter therefore gets NO pose_graph/
        # global_slam (pure front-end); the runner submits keyframes to the async worker.
        apgo = AsyncPoseGraphRunner(pose_graph, global_slam)
        print("Online PGO   : ON (async background thread; loop window "
              f"{float(getattr(cfg, 'PGO_LOOP_SEARCH_XY', 3.0))} m)")

    adapter = HectorLocalSlamAdapter(
        matcher_manager=matcher_manager,
        extrapolator=extrap,
        # Always provide the motion filter so a live switch INTO scan_to_submap keeps its
        # insertion cadence. The adapter's scan_to_map branch ignores it (uses
        # MAP_UPDATE_EVERY), so scan_to_map behavior is unchanged.
        motion_params=motion_params,
        use_extrapolator=use_extrapolator,
        pose_graph=None,
        global_slam=None,
        solve_every_n_nodes=int(getattr(cfg, "PGO_OPTIMIZE_EVERY_N_NODES", 90)),
        # Motion-filter keyframing applies to the scan_to_map branch only; the
        # submap path keeps motion_params for its insertion cadence.
        motion_filter_skip=use_motion_filter,
        mf_keyframe_params=mf_keyframe_params,
    )

    first = scans[0]
    adapter.initialize_extrapolator(float(first["t"]), _resolve_initial_pose(profile, first))

    # --- live matcher-switch trigger (stdin) ---
    switch_q: "queue.Queue" = queue.Queue()
    if switching_enabled:
        _start_switch_reader(switch_q)
        print(f"Live switch  : type 'map' / 'submap' (+Enter) to switch front-end "
              f"(grace {switch_grace} scans); 'status' / 'quit'.")
    else:
        print("Live switch  : DISABLED (--enable-pgo). Front-end switching is local-only.")

    # --- live plot + timing ---
    win_title = f"Hector real-time — {cfg.DATASET_NAME} / {matcher_type}" + (" +PGO" if enable_pgo else "")
    live = LiveTrajectoryPlot(win_title)
    timer = StageTimer()

    xs: list[float] = []
    ys: list[float] = []
    kf_xs: list[float] = []   # keyframe positions (GN actually ran) for live overlay
    kf_ys: list[float] = []
    poses_xyt: list[list[float]] = []
    pts_list: list[np.ndarray] = []
    pgo_scan_records: list = []

    # Live world-frame point cloud (one (N,2) chunk per scan), accumulated only
    # when --live-points is set so the default run pays no cost.
    live_points = bool(getattr(args, "live_points", False))
    points_max = int(getattr(args, "points_max", 80000))
    cloud_chunks: list[np.ndarray] = []
    if live_points:
        print(f"Live points  : ON (world-frame cloud overlay; display cap {points_max})")

    rmin = max(cfg.LIDAR_MIN_RANGE, profile.range_min)
    t0_data = float(scans[0]["t"])
    t0_wall = time.perf_counter()
    final_lag = 0.0
    imu_idx = 0  # cursor into the time-sorted IMU stream

    quit_requested = False
    for k, s in enumerate(scans):
        t = float(s["t"])
        odom_raw = s.get("odom")
        odom = Pose2(*odom_raw) if odom_raw is not None else None

        # --- apply any live switch commands (main-thread only) ---
        if switching_enabled:
            if _apply_switch_requests(switch_q, matcher_manager, matchers_by_name,
                                      switch_grace, k):
                quit_requested = True
                print(f"[switch] quit requested at k={k}; finishing up.")
                break

        # --- feed all IMU samples up to this scan time into the extrapolator ---
        if use_imu:
            while imu_idx < len(imu_samples) and imu_samples[imu_idx][0] <= t:
                _it, _iwz, _iyaw = imu_samples[imu_idx]
                extrap.add_imu(_it, _iwz, _iyaw)
                imu_idx += 1

        # --- preprocess ---
        t_a = time.perf_counter()
        pts_raw = ranges_to_points(
            s["ranges"], profile.angle_min, profile.angle_inc, rmin, profile.range_max,
            stride=cfg.BEAM_STRIDE,
        )
        pts, _proc_debug = point_processor.process(pts_raw)
        timer.add("preprocess", time.perf_counter() - t_a)

        # --- slam (front-end only: predict + match + insert). PGO is OFF this path. ---
        # process_scan calls matcher_manager.maybe_activate_pending() internally, so a
        # pending switch can take effect here; detect it by the active-matcher name change.
        _name_before = matcher_manager.active_matcher.name
        t_b = time.perf_counter()
        pose, result, _do_insert, did_insert = adapter.process_scan(
            k=k, t=t, scan_points_local=pts, odom_pose_world=odom,
            odom_alpha=(cfg.ODOM_ALPHA if odom is not None else 0.0),
        )
        dt_slam = time.perf_counter() - t_b
        timer.add("slam", dt_slam)
        if matcher_manager.active_matcher.name != _name_before:
            print(f"[switch] >>> switched to {matcher_manager.active_matcher.name} at k={k} "
                  f"(was {_name_before}); fresh origin, last pose as prior.")

        # --- submit accepted keyframes to the async PGO worker (non-blocking) ---
        # Only GENUINELY MATCHED keyframes become pose-graph nodes. With the
        # submap motion filter, a sub-threshold scan is dead-reckoned and still
        # inserted (did_insert=True, success=True) to keep the submap dense — but
        # it is NOT a keyframe, so it must not enter the graph. _do_insert (the
        # keyframe decision) is False for skipped scans, so gate on that.
        is_keyframe = _do_insert and getattr(result, "method", "") != "motion_filter_skip"
        if enable_pgo and is_keyframe and did_insert and result.success:
            insertion_submaps = matcher_manager.active_matcher.get_last_inserted_submaps()
            if insertion_submaps:
                apgo.submit(t, pose, pts, insertion_submaps)

        xs.append(pose.x)
        ys.append(pose.y)
        # A keyframe is a scan where GN actually ran (i.e. NOT dead-reckoned by
        # the motion filter). Overlaid on the live plot so the skip is visible.
        is_skip = (getattr(result, "method", "") == "motion_filter_skip")
        if not is_skip:
            kf_xs.append(pose.x)
            kf_ys.append(pose.y)
        poses_xyt.append([pose.x, pose.y, pose.theta])
        pts_list.append(pts)
        if live_points and pts.shape[0] > 0:
            cloud_chunks.append(
                _transform_points(np.array([pose.x, pose.y, pose.theta], dtype=float), pts)
            )
        if enable_pgo:
            pgo_scan_records.append((t, pose))  # (time, online pose); corrected at the end

        # --- draw (throttled) — live shows the ONLINE trajectory (real-time);
        #     PGO corrections are folded in for the final map after the worker drains. ---
        t_c = time.perf_counter()
        if live.ok and (k % max(1, args.draw_every) == 0):
            px = py = None
            if live_points and cloud_chunks:
                cloud = np.vstack(cloud_chunks)
                # Stride-subsample so a large cloud stays cheap to redraw.
                if cloud.shape[0] > points_max:
                    cloud = cloud[:: (cloud.shape[0] // points_max + 1)]
                px, py = cloud[:, 0], cloud[:, 1]
            kfx = kf_xs if use_motion_filter else None
            kfy = kf_ys if use_motion_filter else None
            live.update(xs, ys, kfx, kfy, px, py)
        timer.add("draw", time.perf_counter() - t_c)

        # --- real-time pacing ---
        # A scan is "late" if its compute (everything since t_a, i.e. preprocess+slam+draw)
        # exceeded the time until the next scan — i.e. we could not keep the sensor cadence.
        period = (float(scans[k + 1]["t"]) - t) if (k + 1) < len(scans) else 0.0
        elapsed_compute = time.perf_counter() - t_a
        if period > 0 and elapsed_compute > period and args.speed > 0:
            timer.late_frames += 1

        if args.speed > 0:
            target_wall = t0_wall + (t - t0_data) / args.speed
            now = time.perf_counter()
            sleep_s = target_wall - now
            t_d = time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
                final_lag = 0.0
            else:
                final_lag = -sleep_s  # behind schedule by this much
            timer.add("sleep", time.perf_counter() - t_d)

        timer.count += 1

        if (k % max(1, args.verbose_every)) == 0:
            if getattr(result, "method", "") == "motion_filter_skip":
                mode = "SKIP"
            elif result.success:
                mode = "MATCH"
            else:
                mode = "FALLBACK"
            active_m = matcher_manager.active_matcher
            submap_info = ""
            if active_m.name == "scan_to_submap":
                sb = active_m.submap_builder
                submap_info = f" | submaps act={len(sb.active)} fin={len(sb.finished_submaps)}"
            print(
                f"k={k:5d} {mode:<8s} pose=({pose.x:6.2f},{pose.y:6.2f},{np.rad2deg(pose.theta):6.1f}°)"
                f" score={float(result.score):.3f}"
                + submap_info
                + f" | slam={dt_slam * 1e3:6.1f}ms"
                + (f" pgo_backlog={apgo.backlog():3d}" if enable_pgo else "")
                + f" lag={final_lag:6.3f}s"
            )

    # --- drain the async PGO worker + final optimization + corrected trajectory ---
    poses_final = np.array(poses_xyt, dtype=float)
    if enable_pgo and apgo is not None:
        print(f"\nDraining async PGO backlog ({apgo.backlog()} keyframes) + final optimization...")
        apgo.finalize()  # waits for the worker queue to drain, then runs the final solve
        counts = pose_graph.get_constraint_counts()
        print(f"  Pose graph: {len(pose_graph.nodes)} nodes, {len(pose_graph.submaps)} submaps, "
              f"constraints total={counts['total']} intra={counts['intra']} loop={counts['loop']}")
        lc = global_slam.get_stats()
        print(f"  Loop closure: candidates={lc['candidate_pairs']} "
              f"accepted={lc['accepted_pairs']} rejected={lc['rejected_pairs']}")
        corr = _reconstruct_pgo_poses_by_time(pose_graph, pgo_scan_records)
        poses_final = np.array([[p.x, p.y, p.theta] for p in corr], dtype=float)
        if live.ok:
            live.update(list(poses_final[:, 0]), list(poses_final[:, 1]))

    # --- timing summary ---
    processed_scans = len(pts_list)  # may be < len(scans) if the user typed 'quit'
    timer.print_summary(total_scans=processed_scans, final_lag=final_lag)

    if use_motion_filter:
        total_mf = adapter.matched_count + adapter.skipped_count
        print(
            f"  Motion filter    : {adapter.matched_count} keyframes GN-matched, "
            f"{adapter.skipped_count} dead-reckoned (skipped) of {total_mf} "
            f"→ {100.0 * adapter.skipped_count / max(1, total_mf):.1f}% of GN solves avoided"
        )

    # --- final occupancy map ---
    # Name/title by the FINAL active matcher (it may differ from --matcher after a live switch).
    final_matcher_type = matcher_manager.active_matcher.name
    if not args.no_map:
        _build_and_show_map(final_matcher_type, poses_final, pts_list, enable_pgo, args, live)
    if live_points:
        _save_point_cloud_png(final_matcher_type, poses_final, pts_list, enable_pgo, args, live, points_max)

    live.keep_open()


def _grid_world_extent(grid) -> list[float]:
    """World-coordinate extent [xmin, xmax, ymin, ymax] of an occupancy grid,
    so it can be drawn with imshow(origin='lower', extent=...) in the SAME
    world frame (y-up, meters) as the live trajectory window."""
    res = float(grid.res)
    if hasattr(grid, "origin_world"):          # ProbabilityGrid (scan_to_submap stitch)
        ox, oy = float(grid.origin_world[0]), float(grid.origin_world[1])
        w, h = int(grid.w), int(grid.h)
        return [ox, ox + w * res, oy, oy + h * res]
    # GridMap (scan_to_map): origin is the world-origin cell, size is square
    ox_cell, oy_cell = float(grid.origin[0]), float(grid.origin[1])
    size = int(grid.size)
    return [(0 - ox_cell) * res, (size - ox_cell) * res,
            (0 - oy_cell) * res, (size - oy_cell) * res]


def _output_prefix(matcher_type, enable_pgo, args) -> str:
    dataset_tag = cfg.DATASET_NAME
    if cfg.DATASET_NAME == "lab_run_2":
        dataset_tag = f"lab_run_2_{cfg.DATASET_SCAN_VARIANT}"
    suffix = ("_fast" if bool(getattr(args, "fast_match", False)) and matcher_type == "scan_to_submap" else "")
    suffix += ("_pgo" if enable_pgo else "")
    return args.save_prefix or f"realtime_{dataset_tag}_{matcher_type}{suffix}"


def _save_point_cloud_png(matcher_type, poses_final, pts_list, enable_pgo, args, live, points_max) -> None:
    """Render the accumulated world-frame laser cloud (built from the FINAL poses, so
    PGO corrections are reflected) + trajectory to a PNG. Works headless (Agg)."""
    if len(pts_list) == 0:
        return
    chunks = [
        _transform_points(poses_final[i], pts_list[i])
        for i in range(min(len(poses_final), len(pts_list)))
        if pts_list[i].shape[0] > 0
    ]
    if not chunks:
        return
    cloud = np.vstack(chunks)
    if cloud.shape[0] > points_max:
        cloud = cloud[:: (cloud.shape[0] // points_max + 1)]

    prefix = _output_prefix(matcher_type, enable_pgo, args)
    os.makedirs("hector_outputs", exist_ok=True)
    out_png = os.path.join("hector_outputs", f"{prefix}_points.png")

    import matplotlib
    if not live.ok:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10), dpi=130)
    ax.scatter(cloud[:, 0], cloud[:, 1], s=0.5, c="#222222", alpha=0.5, linewidths=0)
    if len(poses_final) > 0:
        ax.plot(poses_final[:, 0], poses_final[:, 1], "-", color="#1f77b4", lw=1.0, alpha=0.9)
        ax.plot(poses_final[0, 0], poses_final[0, 1], "o", color="#00aa55", ms=8, label="start")
        ax.plot(poses_final[-1, 0], poses_final[-1, 1], "X", color="#ff4466", ms=10, label="end")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    dataset_tag = cfg.DATASET_NAME if cfg.DATASET_NAME != "lab_run_2" else f"lab_run_2_{cfg.DATASET_SCAN_VARIANT}"
    ax.set_title(f"Point cloud — {dataset_tag} / {matcher_type} ({cloud.shape[0]} pts shown)"
                 + (" +PGO" if enable_pgo else ""))
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print(f"Wrote point cloud: {out_png}")
    if live.ok:
        plt.show(block=False)
        plt.pause(0.5)


def _build_and_show_map(matcher_type, poses_final, pts_list, enable_pgo, args, live) -> None:
    from hector.eval.rebuild_map_any import build_map, _grid_prob
    from hector.eval._generic_eval_common import default_map_size_m

    map_res = cfg.MAP_RESOLUTION
    # Always render the final occupancy map by integrating every scan into ONE global grid
    # (scan_to_map-style), regardless of the tracking matcher. Stitching per-submap grids
    # (the scan_to_submap build path) thickens/doubles walls where submaps overlap and is
    # purely a rendering artifact; the single global grid gives crisp walls and reflects the
    # trajectory quality only — the same convention hector/eval/compare_three_maps uses.
    map_size_m = default_map_size_m(cfg.DATASET_NAME, "scan_to_map")
    print(f"\nBuilding final occupancy map ({len(pts_list)} scans, global grid, "
          f"res={map_res}m, size={map_size_m}m)...")
    grid, traj_xy = build_map(
        matcher_type="scan_to_map", poses_xyt=poses_final, pts_list=pts_list,
        map_res=map_res, map_size_m=map_size_m, ray_steps=cfg.RAY_STEPS,
        l_free=cfg.L_FREE, l_occ=cfg.L_OCC, l_min=cfg.L_MIN, l_max=cfg.L_MAX, label="final",
    )
    prob = _grid_prob(grid)
    extent = _grid_world_extent(grid)

    dataset_tag = cfg.DATASET_NAME
    if cfg.DATASET_NAME == "lab_run_2":
        dataset_tag = f"lab_run_2_{cfg.DATASET_SCAN_VARIANT}"
    # Distinct suffix per mode so a fast-match map never overwrites (or is mistaken for) a
    # default-mode map of the same name.
    suffix = ("_fast" if bool(getattr(args, "fast_match", False)) and matcher_type == "scan_to_submap" else "")
    suffix += ("_pgo" if enable_pgo else "")
    prefix = args.save_prefix or f"realtime_{dataset_tag}_{matcher_type}{suffix}"
    os.makedirs("hector_outputs", exist_ok=True)
    out_png = os.path.join("hector_outputs", f"{prefix}.png")

    import matplotlib
    if not live.ok:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10), dpi=130)
    # World-coordinate frame, identical to the live window: x right, y UP, meters.
    # origin='lower' + extent maps grid row 0 (low y) to the bottom — no flipud,
    # no pixel-index axis — so the occupancy map and trajectory share the live frame.
    ax.imshow(prob, cmap="binary_r", vmin=0.2, vmax=0.8, interpolation="nearest",
              origin="lower", extent=extent)
    if len(traj_xy) > 0:
        ax.plot(traj_xy[:, 0], traj_xy[:, 1], "-", color="#1f77b4", lw=1.0, alpha=0.9)
        ax.plot(traj_xy[0, 0], traj_xy[0, 1], "o", color="#00aa55", ms=8, label="start")
        ax.plot(traj_xy[-1, 0], traj_xy[-1, 1], "X", color="#ff4466", ms=10, label="end")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Final map — {dataset_tag} / {matcher_type}" + (" +PGO" if enable_pgo else ""))
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print(f"Wrote final map: {out_png}")
    if live.ok:
        plt.show(block=False)
        plt.pause(0.5)


if __name__ == "__main__":
    main()
