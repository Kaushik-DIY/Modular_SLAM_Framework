"""Fusion real-time module-switching runner (V5).

A separate, experimental runner beside the stable batch `runner.py` (which is
left untouched) — mirroring how `hector/run_realtime_viz.py` sits beside
`hector/run_local_slam_new.py`. It:

  * replays a dataset at real sensor cadence (sleep to scan timestamps),
  * draws the trajectory + accumulating LiDAR point cloud LIVE,
  * lets the user switch loop-closure modules (and, V5.3, the LiDAR front-end
    variant) LIVE on stdin while the SAME shared C++ map keeps mapping,
  * renders the fused occupancy map at the end.

Phase 1 (this file): LiDAR-led front-ends (modes lidar / lidar_orb). The three
switch axes:
  - A3 verifier : bnb | icp | pnp       (pnp needs visual payload; see --attach-visual)
  - A2 proposer : proximity | dbow      (dbow needs descriptors; see --attach-visual)
  - A1 frontend : native_s2s | native_s2m   (live s2s<->s2m, V5.3)

Cross-sensor (visual VO <-> LiDAR) switching is Phase 2 (needs a unified
multi-sensor driver) and is rejected live with a clear reason.

Live switch commands (type + Enter):  verifier bnb|icp|pnp · proposer proximity|dbow
  · fe s2s|s2m · status · quit

Example:
  .venv/bin/python -m slam_core.fusion2.run_realtime --dataset datasets/lab_hybrid \
      --lidar-frontend native_s2s --verifier bnb --attach-visual
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

import fusion_core as fc
from slam_core.fusion2.config import FusionV2Config
from slam_core.fusion2.dataset import LabHybridStream
from slam_core.fusion2.lidar_frontend import make_lidar_frontend
from slam_core.fusion2.runner import (_rel, _rel_sane, build_shared_map,
                                      propose_candidates,
                                      verify_candidate_bnb, verify_candidate_icp)


# ---------------------------------------------------------------------------
# Per-keyframe ingestion into the shared map (replicates run_lidar_mode exactly,
# but with the proposer + verifier as live-switchable fields).
# ---------------------------------------------------------------------------
class IngestEngine:
    def __init__(self, shared, cfg: FusionV2Config, K=None,
                 verifier: str = "bnb", proposer: str = "proximity"):
        self.shared = shared
        self.cfg = cfg
        self.K = K
        self.verifier = verifier          # bnb | icp | pnp
        self.proposer = proposer          # proximity | dbow
        self._appearance = None           # lazily built AppearanceIndex (dbow)
        self.kf_id = -1
        self.last_fe_pose = None
        self.last_graph_pose = None
        self.kf_stamps: dict = {}
        self.stats = dict(keyframes=0, proposals=0, verified=0, loops_accepted=0,
                          rehearsal_merges=0, optimize_calls=0)

    # -- appearance index (only needed for the dbow proposer) --------------
    def ensure_appearance(self):
        if self._appearance is None:
            from slam_core.fusion2.appearance_index import AppearanceIndex
            self._appearance = AppearanceIndex(
                min_score=self.cfg.dbow_min_score,
                min_separation=self.cfg.min_kf_separation,
                max_candidates=self.cfg.max_candidates_per_query)
        return self._appearance

    def index_descriptors(self, kf_id: int, des):
        """Keep the DBoW index populated whenever descriptors exist, so the
        proposer can be switched to 'dbow' at any time (only sees indexed KFs)."""
        if des is not None and len(des) and self._appearance is not None:
            self._appearance.add(kf_id, np.asarray(des, np.uint8))

    # -- one keyframe ------------------------------------------------------
    def ingest(self, t, fe_pose, raw_scan, visual=None):
        """Insert a keyframe (scan + optional visual payload) into the shared
        map; returns (kf_id, node_pose). Mirrors run_lidar_mode lines 415-462."""
        cfg, shared = self.cfg, self.shared
        self.kf_id += 1
        kf_id = self.kf_id
        self.kf_stamps[kf_id] = t
        if self.last_fe_pose is None:
            node_pose = fe_pose
        else:
            node_pose = self.last_graph_pose.compose(_rel(self.last_fe_pose, fe_pose))

        kpts = des = pts3d = None
        if visual is not None:
            kpts, des, pts3d = visual
        if kpts is not None:
            sig = fc.Signature(kf_id, t, kpts=kpts, des=des, pts3d=pts3d, scan_xy=raw_scan)
        else:
            sig = fc.Signature(kf_id, t, scan_xy=raw_scan)
        sig.pose = node_pose

        if self.last_fe_pose is not None:
            d = _rel(self.last_fe_pose, fe_pose)
            stationary = math.hypot(d.x, d.y) < 0.05 and abs(d.theta) < math.radians(2.0)
        else:
            stationary = False
        res = shared.memory.insert(sig, similarity=-1.0 if stationary else 0.0)
        if res.rehearsal_merged:
            self.stats["rehearsal_merges"] += 1
        shared.graph.add_node(kf_id, node_pose)
        if self.last_fe_pose is not None:
            rel = _rel(self.last_fe_pose, fe_pose)
            sig.add_link(kf_id - 1, fc.LinkType.NEIGHBOR, rel,
                         cfg.spine_trans_weight, cfg.spine_rot_weight)
            shared.graph.add_spine_edge(kf_id - 1, kf_id, rel)
        self.last_fe_pose = fe_pose
        self.last_graph_pose = node_pose
        self.stats["keyframes"] += 1
        # always index descriptors when present (keeps dbow switchable)
        self.index_descriptors(kf_id, des)
        return kf_id, node_pose, sig

    # -- proposer dispatch -------------------------------------------------
    def _propose(self, kf_id, node_pose, sig):
        if self.proposer == "dbow":
            idx = self.ensure_appearance()
            if sig.has_visual:
                return [c for c, _ in idx.query(kf_id, np.asarray(sig.des, np.uint8))]
            return []
        return propose_candidates(self.shared, self.cfg, kf_id, node_pose)

    # -- verifier dispatch (gates replicate run_lidar_mode) ----------------
    def _verify(self, kf_id, sig, node_pose, cand, raw_scan):
        """Returns (accepted, rel_pose|None)."""
        cfg, shared = self.cfg, self.shared
        if self.verifier == "pnp":
            from slam_core.fusion2.visual_features import (cam_rel_to_base_se2,
                                                           pnp_verify)
            cand_sig = shared.memory.get(cand)
            if cand_sig is None or not cand_sig.has_visual or not sig.has_visual:
                return False, None
            self.stats["verified"] += 1
            ok, T_tq, n_in, ratio = pnp_verify(
                np.asarray(sig.kpts), np.asarray(sig.des),
                np.asarray(cand_sig.des), np.asarray(cand_sig.pts3d), self.K,
                nndr=cfg.pnp_nndr, min_inliers=cfg.pnp_min_inliers)
            if not ok:
                return False, None
            rx, ry, rth = cam_rel_to_base_se2(T_tq)
            rel = fc.Pose2(rx, ry, rth)
            pred = _rel(shared.graph.get_pose(cand), node_pose)
            if _rel_sane(rel, pred, cfg.pnp_rel_sanity_m, cfg.pnp_rel_sanity_rad) \
                    or n_in >= cfg.pnp_strong_inliers:
                return True, rel
            return False, None

        # scan verifiers (bnb / icp)
        if self.verifier == "icp":
            r = verify_candidate_icp(shared, cfg, kf_id, raw_scan, node_pose, cand,
                                     seed_from_bnb=False)
        else:
            r = verify_candidate_bnb(shared, cfg, kf_id, raw_scan, node_pose, cand)
        if r is None:
            return False, None
        self.stats["verified"] += 1
        if self.verifier == "icp":
            accepted = (r.success and r.coarse_score >= cfg.icp_accept_fitness
                        and r.refined_score >= cfg.accept_refined_min)
        else:
            accepted = (r.success and r.coarse_score >= cfg.accept_coarse_min
                        and r.refined_score >= cfg.accept_refined_min)
        if not accepted:
            return False, None
        cand_pose = shared.graph.get_pose(cand)
        rel = _rel(cand_pose, r.pose)
        pred = _rel(cand_pose, node_pose)
        if not _rel_sane(rel, pred, cfg.scan_rel_sanity_m, cfg.scan_rel_sanity_rad):
            return False, None
        return True, rel

    # -- close loops + optimize for one keyframe ---------------------------
    def close_loops(self, kf_id, node_pose, sig, raw_scan):
        cfg, shared = self.cfg, self.shared
        any_loop = False
        if kf_id % cfg.propose_every_n_kf == 0 and kf_id > cfg.min_kf_separation:
            for cand in self._propose(kf_id, node_pose, sig):
                self.stats["proposals"] += 1
                accepted, rel = self._verify(kf_id, sig, node_pose, cand, raw_scan)
                if accepted and rel is not None:
                    shared.graph.add_loop_edge(cand, kf_id, rel,
                                               cfg.loop_trans_weight, cfg.loop_rot_weight)
                    sig.add_link(cand, fc.LinkType.LOOP, rel,
                                 cfg.loop_trans_weight, cfg.loop_rot_weight)
                    shared.memory.on_loop_confirmed(kf_id, cand)
                    self.stats["loops_accepted"] += 1
                    any_loop = True
        # Online SLAM: optimize IMMEDIATELY on an accepted loop (so the robot's
        # corrected localization is available at the moment of closure), as well
        # as on the periodic cadence. The expensive fused-map render stays at the
        # end; only the cheap pose-graph solve runs live. Returns True if the
        # graph was re-optimized this keyframe (-> caller refreshes the display).
        did_opt = False
        if (any_loop and kf_id > 0) or (kf_id > 0 and kf_id % cfg.optimize_every_n_kf == 0):
            shared.graph.optimize()
            self.stats["optimize_calls"] += 1
            self.last_graph_pose = shared.graph.get_pose(kf_id)
            did_opt = True
        return did_opt


# ---------------------------------------------------------------------------
# LiDAR front-end manager with grace-buffer handoff (V5.3, s2s<->s2m).
#
# On a switch request the ORIGINAL front-end keeps driving the graph while a
# fresh target front-end warms up on the SAME incoming scans for `grace_scans`
# scans, then takes over. The new front-end starts from its own origin (no state
# transfer); continuity is preserved by resetting the engine's relative-motion
# baseline to the new front-end's pose at the flip (the user's design). Mirrors
# MatcherManager.request_switch from hector/run_realtime_viz.py.
# ---------------------------------------------------------------------------
class FrontEndManager:
    def __init__(self, make_fe, initial_kind: str, grace_scans: int = 15):
        self._make = make_fe
        self.active_kind = initial_kind
        self.active = make_fe(initial_kind)
        self.grace_scans = int(grace_scans)
        self._pending = None
        self._pending_kind = None
        self._grace = 0

    def request_switch(self, target_kind: str) -> str:
        if target_kind == self.active_kind:
            return f"already {target_kind}; ignored"
        if self._pending is not None:
            return f"switch to {self._pending_kind} already in progress; ignored"
        self._pending = self._make(target_kind)
        self._pending_kind = target_kind
        self._grace = self.grace_scans
        return f"requested {target_kind}; effective in {self.grace_scans} scans (warming up)"

    def process(self, t, scan):
        """Returns (pose, pts, is_kf, flip_pose_or_None). flip_pose is the new
        front-end's current pose when a handoff completes THIS scan (the engine
        resets its relative-motion baseline to it)."""
        out = self.active.process(t, scan)
        flip = None
        if self._pending is not None:
            warm = self._pending.process(t, scan)   # warm up (output ignored)
            self._grace -= 1
            if self._grace <= 0:
                self.active = self._pending
                self.active_kind = self._pending_kind
                self._pending = self._pending_kind = None
                wp = warm[0]
                flip = fc.Pose2(float(wp.x), float(wp.y), float(wp.theta))
        return (*out, flip)

    @property
    def pending(self):
        return self._pending_kind


# ---------------------------------------------------------------------------
# Live trajectory + accumulating scan-cloud window (ported/trimmed from
# hector/run_realtime_viz.py::LiveTrajectoryPlot). Headless-safe.
# ---------------------------------------------------------------------------
class LiveView:
    def __init__(self, title: str):
        self.ok = False
        self.fig = self.ax = None
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            print("[viz] headless (no DISPLAY) — final PNG only.")
            return
        import matplotlib
        for backend in ("Qt5Agg", "TkAgg", "GTK3Agg"):
            try:
                matplotlib.use(backend, force=True)
                import matplotlib.pyplot as plt
                plt.ion()
                fig, ax = plt.subplots(figsize=(8, 8))
                fig.canvas.draw()
                plt.show(block=False)
            except Exception:
                try:
                    import matplotlib.pyplot as _plt
                    _plt.close("all")
                except Exception:
                    pass
                continue
            self._plt, self.fig, self.ax = plt, fig, ax
            ax.set_title(title); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
            ax.set_aspect("equal", adjustable="datalim"); ax.grid(True, alpha=0.3)
            (self._pts,) = ax.plot([], [], ".", color="#555", ms=1.0, alpha=0.5, zorder=1)
            (self._line,) = ax.plot([], [], "-", color="#1f77b4", lw=1.2, zorder=3)
            (self._cur,) = ax.plot([], [], "o", color="#ff4466", ms=7, zorder=5)
            (self._start,) = ax.plot([], [], "o", color="#00aa55", ms=8, zorder=5)
            fig.canvas.draw(); plt.show(block=False)
            self.ok = True
            print(f"[viz] live backend: {backend}")
            return
        print("[viz] no interactive backend — final PNG only.")

    def update(self, xs, ys, px=None, py=None, title=None):
        if not self.ok or not xs:
            return
        if px is not None:
            self._pts.set_data(px, py)
        self._line.set_data(xs, ys)
        self._cur.set_data([xs[-1]], [ys[-1]])
        self._start.set_data([xs[0]], [ys[0]])
        if title:
            self.ax.set_title(title)
        self.ax.relim(); self.ax.autoscale_view()
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()

    def keep_open(self):
        if self.ok:
            print("[viz] done — close the window to exit.")
            self._plt.ioff(); self._plt.show(block=True)


class StageTimer:
    STAGES = ("preprocess", "slam", "loop", "draw", "sleep")

    def __init__(self):
        self.total = {s: 0.0 for s in self.STAGES}
        self.late = 0
        self.n = 0

    def add(self, s, dt):
        self.total[s] += dt

    def summary(self, lag):
        comp = sum(self.total[s] for s in self.STAGES if s != "sleep")
        print("\n" + "=" * 56 + "\n  REAL-TIME TIMING\n" + "=" * 56)
        for s in self.STAGES:
            mean_ms = self.total[s] / self.n * 1e3 if self.n else 0.0
            print(f"  {s:<11} {self.total[s]:8.3f}s  {mean_ms:7.2f} ms/scan")
        work = {s: self.total[s] for s in self.STAGES if s != "sleep"}
        bn = max(work, key=work.get) if work else "n/a"
        print(f"  scans={self.n}  late={self.late} "
              f"({100.0*self.late/max(1,self.n):.1f}%)  bottleneck={bn}")
        print(f"  verdict: {'REAL-TIME OK' if lag <= 0.05 else f'BEHIND by {lag:.3f}s'}")
        print("=" * 56)


# ---------------------------------------------------------------------------
# CLI + main loop
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Fusion real-time module-switching runner (V5)")
    p.add_argument("--dataset", type=Path, default=Path("datasets/lab_hybrid"))
    p.add_argument("--mode", choices=("lidar", "lidar_orb"), default="lidar",
                   help="LiDAR-led front-end (visual-led is Phase 2).")
    p.add_argument("--lidar-frontend", choices=("native_s2s", "native_s2m"),
                   default="native_s2s")
    p.add_argument("--verifier", choices=("bnb", "icp", "pnp"), default="bnb")
    p.add_argument("--proposer", choices=("proximity", "dbow"), default="proximity")
    p.add_argument("--attach-visual", action="store_true",
                   help="Extract+store ORB on every LiDAR keyframe so pnp/dbow are "
                        "available live (~78 KB/keyframe).")
    p.add_argument("--max-scans", type=int, default=0)
    p.add_argument("--speed", type=float, default=1.0,
                   help="1.0=real-time, 0=as-fast-as-possible.")
    p.add_argument("--draw-every", type=int, default=5)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--output", type=Path, default=Path("fusion2_outputs"))
    p.add_argument("--no-map", action="store_true")
    return p.parse_args()


def main(argv=None):
    args = _parse_args()
    cfg = FusionV2Config(mode=args.mode, dataset=args.dataset, output_dir=args.output,
                         lidar_frontend=args.lidar_frontend, scan_verifier=args.verifier,
                         max_scans=args.max_scans)

    # visual payload availability (lean default; --attach-visual or lidar_orb opts in)
    attach_visual = bool(args.attach_visual) or args.mode == "lidar_orb"
    if args.verifier == "pnp" and not attach_visual:
        raise SystemExit("--verifier pnp needs visual payload: add --attach-visual "
                         "(or --mode lidar_orb).")
    if args.proposer == "dbow" and not attach_visual:
        raise SystemExit("--proposer dbow needs descriptors: add --attach-visual.")

    stream = LabHybridStream(cfg.dataset, cfg.sync_tolerance_s)
    imu_path = str(Path(cfg.dataset) / "imu.csv")

    def _make_fe(kind):
        return make_lidar_frontend(kind, dataset_name="lab_hybrid", imu_path=imu_path,
                                   kf_min_dist_m=cfg.kf_min_dist_m,
                                   kf_min_angle_rad=cfg.kf_min_angle_rad,
                                   kf_min_dt_s=cfg.kf_min_dt_s)

    fem = FrontEndManager(_make_fe, cfg.lidar_frontend, grace_scans=15)
    shared = build_shared_map(cfg)

    K = None
    if attach_visual:
        import yaml
        sc = yaml.safe_load(open(Path(cfg.dataset) / "sensor_config.yaml"))["camera"]
        K = np.array([[sc["fx"], 0, sc["cx"]], [0, sc["fy"], sc["cy"]], [0, 0, 1]])

    eng = IngestEngine(shared, cfg, K=K, verifier=args.verifier, proposer=args.proposer)
    if args.proposer == "dbow":
        eng.ensure_appearance()

    print("=" * 60)
    print(f"Dataset   : {cfg.dataset}")
    print(f"Front-end : {cfg.lidar_frontend}  (mode {cfg.mode})")
    print(f"Verifier  : {args.verifier}   Proposer: {args.proposer}")
    print(f"Visual    : {'attached' if attach_visual else 'lean (scan only)'}")
    print(f"Playback  : {'real-time' if args.speed > 0 else 'max'} (speed={args.speed}x)")
    print("Live cmds : verifier bnb|icp|pnp · proposer proximity|dbow · "
          "fe s2s|s2m · status · quit")
    print("=" * 60)

    # live switch stdin reader (V5.2 wires dispatch; V5.1 accepts status/quit)
    switch_q: "queue.Queue" = queue.Queue()
    _start_switch_reader(switch_q)

    live = LiveView(f"fusion realtime — {cfg.mode}/{cfg.lidar_frontend}")
    timer = StageTimer()

    xs, ys, cloud_chunks = [], [], []
    kf_ids, kf_scans = [], []     # for live re-projection on loop correction
    t0_data = t0_wall = None
    lag = 0.0
    quit_req = False

    scans = list(stream.lidar_stream(cfg.max_scans))
    for k, (t, scan) in enumerate(scans):
        if t0_data is None:
            t0_data, t0_wall = t, time.perf_counter()

        # drain live commands (main thread only)
        quit_req = _apply_switches(switch_q, eng, fem, attach_visual, k)
        if quit_req:
            scans = scans[:k]
            break

        t_a = time.perf_counter()
        fe_pose_py, pts, is_kf, flip = fem.process(t, scan)
        timer.add("slam", time.perf_counter() - t_a)
        fe_pose = fc.Pose2(float(fe_pose_py.x), float(fe_pose_py.y), float(fe_pose_py.theta))
        if flip is not None:
            # handoff complete: reset the relative-motion baseline to the new
            # front-end's frame so chaining continues seamlessly from the last
            # graph pose (no jump). cfg.lidar_frontend tracks the active kind.
            eng.last_fe_pose = flip
            cfg.lidar_frontend = fem.active_kind
            print(f"[switch] >>> front-end now {fem.active_kind} at k={k} "
                  f"(graph continues from last pose).")

        if is_kf:
            visual = None
            if attach_visual:
                visual = _extract_visual(stream, t, K)
            t_b = time.perf_counter()
            kf_id, node_pose, sig = eng.ingest(t, fe_pose, scan, visual=visual)
            did_opt = eng.close_loops(kf_id, node_pose, sig, scan)
            timer.add("loop", time.perf_counter() - t_b)
            kf_ids.append(kf_id)
            kf_scans.append(np.asarray(scan, np.float64))
            if did_opt:
                # LOOP CORRECTION applied live: the graph just re-optimized, so
                # every past pose may have shifted. Rebuild the WHOLE displayed
                # trajectory + cloud from the corrected graph (the robot's
                # localization snaps to the corrected estimate immediately).
                xs, ys, cloud_chunks = _rebuild_display(shared, kf_ids, kf_scans)
            else:
                gp = shared.graph.get_pose(kf_id)
                xs.append(gp.x); ys.append(gp.y)
                c, s = math.cos(gp.theta), math.sin(gp.theta)
                cloud_chunks.append(np.asarray(scan, np.float64)
                                    @ np.array([[c, -s], [s, c]]).T + [gp.x, gp.y])

        # live draw (throttled)
        t_c = time.perf_counter()
        if live.ok and xs and k % max(1, args.draw_every) == 0:
            cloud = np.vstack(cloud_chunks) if cloud_chunks else None
            if cloud is not None and len(cloud) > 60000:
                cloud = cloud[:: len(cloud) // 60000 + 1]
            live.update(xs, ys,
                        cloud[:, 0] if cloud is not None else None,
                        cloud[:, 1] if cloud is not None else None,
                        title=f"{cfg.mode}/{cfg.lidar_frontend} v={eng.verifier} "
                              f"p={eng.proposer} kf={eng.stats['keyframes']} "
                              f"loops={eng.stats['loops_accepted']}")
        timer.add("draw", time.perf_counter() - t_c)

        # real-time pacing
        period = (scans[k + 1][0] - t) if (k + 1) < len(scans) else 0.0
        if period > 0 and (time.perf_counter() - t_a) > period and args.speed > 0:
            timer.late += 1
        if args.speed > 0:
            target = t0_wall + (t - t0_data) / args.speed
            sl = target - time.perf_counter()
            t_d = time.perf_counter()
            if sl > 0:
                time.sleep(sl); lag = 0.0
            else:
                lag = -sl
            timer.add("sleep", time.perf_counter() - t_d)
        timer.n += 1

        if k % max(1, args.print_every) == 0:
            print(f"k={k:5d} kf={eng.stats['keyframes']:4d} "
                  f"loops={eng.stats['loops_accepted']:3d} v={eng.verifier} "
                  f"p={eng.proposer} fe={cfg.lidar_frontend} lag={lag:6.3f}s")

    # final optimize + outputs (+ auto-display the corrected fused map)
    shared.graph.optimize()
    eng.stats["optimize_calls"] += 1
    timer.summary(lag)
    _finalize_and_show(shared, cfg, eng, args, live)


# ---- helpers --------------------------------------------------------------
def _extract_visual(stream, t, K):
    import cv2
    from slam_core.fusion2.visual_features import extract_orb_rgbd
    pair = stream.nearest_rgbd(t)
    if pair is None:
        return None
    rgb = cv2.imread(str(pair[0]), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(pair[1]), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        return None
    kpts, des, pts3d = extract_orb_rgbd(rgb, depth, K)
    return None if kpts is None else (kpts, des, pts3d)


_SWITCH_CMDS = {"verifier", "proposer", "fe", "status", "quit", "exit", "q", "?"}


def _start_switch_reader(q):
    def _reader():
        import sys
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd:
                q.put(cmd)
    threading.Thread(target=_reader, name="switch-reader", daemon=True).start()


def _apply_switches(q, eng, fem, attach_visual, k) -> bool:
    """Drain stdin commands on the MAIN thread. Returns True on quit.
    Verifier/proposer (V5.2) + LiDAR front-end s2s<->s2m (V5.3) switching."""
    while True:
        try:
            cmd = q.get_nowait()
        except queue.Empty:
            return False
        parts = cmd.split()
        if not parts:
            continue
        head = parts[0]
        if head in ("quit", "exit", "q"):
            print(f"[switch] quit at k={k}.")
            return True
        if head in ("status", "?"):
            pend = f" (switching->{fem.pending})" if fem.pending else ""
            print(f"[status] k={k} fe={fem.active_kind}{pend} verifier={eng.verifier} "
                  f"proposer={eng.proposer} kf={eng.stats['keyframes']} "
                  f"loops={eng.stats['loops_accepted']} "
                  f"visual={'on' if attach_visual else 'off'}")
            continue
        if head == "verifier" and len(parts) == 2:
            v = parts[1]
            if v not in ("bnb", "icp", "pnp"):
                print(f"[switch] unknown verifier {v!r}")
            elif v == "pnp" and not attach_visual:
                print("[switch] pnp unavailable: run with --attach-visual (no visual "
                      "payload on keyframes).")
            else:
                eng.verifier = v
                print(f"[switch] verifier -> {v} (effective next proposal).")
            continue
        if head == "proposer" and len(parts) == 2:
            pr = parts[1]
            if pr not in ("proximity", "dbow"):
                print(f"[switch] unknown proposer {pr!r}")
            elif pr == "dbow" and not attach_visual:
                print("[switch] dbow unavailable: run with --attach-visual (no "
                      "descriptors indexed).")
            else:
                if pr == "dbow":
                    eng.ensure_appearance()
                eng.proposer = pr
                print(f"[switch] proposer -> {pr} (dbow only sees indexed keyframes).")
            continue
        if head == "fe" and len(parts) == 2:
            target = parts[1]
            alias = {"s2s": "native_s2s", "s2m": "native_s2m",
                     "native_s2s": "native_s2s", "native_s2m": "native_s2m"}
            if target in ("orb", "vo", "visual", "lidar"):
                print("[switch] cross-sensor front-end switching (visual<->LiDAR) is "
                      "Phase 2: the two sensors run on different stream rates with no "
                      "merged timeline. Not available live.")
            elif target not in alias:
                print(f"[switch] unknown front-end {target!r}. Try: fe s2s | fe s2m")
            else:
                print(f"[switch] {fem.request_switch(alias[target])}")
            continue
        print(f"[switch] unknown command {cmd!r}. "
              "Try: verifier bnb|icp|pnp · proposer proximity|dbow · status · quit")


def _rebuild_display(shared, kf_ids, kf_scans):
    """Re-read every keyframe pose from the (just-optimized) graph and re-project
    its scan, so the live trajectory + cloud reflect the loop correction at once.
    Cheap (poses + a numpy transform); runs only on optimize events."""
    xs, ys, chunks = [], [], []
    for kfid, scan in zip(kf_ids, kf_scans):
        gp = shared.graph.get_pose(kfid)
        xs.append(gp.x); ys.append(gp.y)
        c, s = math.cos(gp.theta), math.sin(gp.theta)
        chunks.append(scan @ np.array([[c, -s], [s, c]]).T + [gp.x, gp.y])
    return xs, ys, chunks


def _finalize_and_show(shared, cfg, eng, args, live):
    """Write outputs and DISPLAY the final corrected fused map directly (no
    second command). Uses the live interactive backend when available, else Agg."""
    import json

    from slam_core.fusion2.runner import _anchor_poses
    run_dir = Path(args.output) / f"realtime_{cfg.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    poses = _anchor_poses(np.asarray(shared.graph.poses()))
    with open(run_dir / "trajectory.tum", "w") as f:
        for nid, x, y, th in poses:
            t = eng.kf_stamps.get(int(nid), float(nid))
            qz, qw = math.sin(th / 2.0), math.cos(th / 2.0)
            f.write(f"{t:.6f} {x:.6f} {y:.6f} 0.0 0.0 0.0 {qz:.9f} {qw:.9f}\n")
    with open(run_dir / "run_summary.json", "w") as f:
        json.dump(dict(mode=cfg.mode, frontend=cfg.lidar_frontend,
                       final_verifier=eng.verifier, final_proposer=eng.proposer,
                       **eng.stats), f, indent=2, default=str)

    if args.no_map or not len(poses):
        print(f"\nWrote: {run_dir}")
        if live.ok:
            live.keep_open()
        return

    # fuse all scans at the FINAL corrected poses into one occupancy grid
    rs, rp = [], []
    for nid, x, y, th in poses:
        sig = shared.memory.get(int(nid))
        if sig is not None and sig.has_scan:
            rs.append(sig); rp.append(fc.Pose2(float(x), float(y), float(th)))
    if not rs:
        print(f"\nWrote: {run_dir}")
        return
    grid = fc.assemble_local_grid(rs, rp, shared.grid_cfg)
    prob = np.asarray(grid.probability())
    extent = [grid.origin_x, grid.origin_x + grid.width * grid.resolution,
              grid.origin_y, grid.origin_y + grid.height * grid.resolution]
    np.save(run_dir / "map.npy", prob)
    with open(run_dir / "map_meta.json", "w") as f:
        json.dump(dict(origin_x=grid.origin_x, origin_y=grid.origin_y,
                       resolution=grid.resolution, width=grid.width,
                       height=grid.height, extent=extent), f, indent=2)

    # render on the live interactive backend if present (so it can be SHOWN),
    # else Agg (headless: save only).
    import matplotlib
    if not live.ok:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    if live.ok and live.fig is not None:
        plt.close(live.fig)                 # replace the live trajectory window
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(prob, cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower",
              extent=extent, interpolation="nearest")
    ax.plot(poses[:, 1], poses[:, 2], "-", lw=1.0, color="tab:blue", alpha=0.9)
    ax.scatter(poses[0, 1], poses[0, 2], c="g", s=50, zorder=5, label="start")
    ax.scatter(poses[-1, 1], poses[-1, 2], c="r", s=50, zorder=5, label="end")
    ax.set_title(f"realtime {cfg.mode}/{cfg.lidar_frontend} — final corrected map: "
                 f"{len(poses)} kf, {eng.stats['loops_accepted']} loops")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.grid(alpha=0.15); ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "occupancy.png", dpi=200)
    print(f"\nWrote: {run_dir}")
    if live.ok:
        print("[viz] showing final corrected map — close the window to exit.")
        live._plt.ioff()
        live._plt.show(block=True)
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
