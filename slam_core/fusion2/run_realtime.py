"""Fusion real-time module-switching runner (V5).

A separate, experimental runner beside the stable batch `runner.py` (which is
left untouched) — mirroring how `hector/run_realtime_viz.py` sits beside
`hector/run_local_slam_new.py`. It:

  * replays a dataset at real sensor cadence (sleep to event timestamps) over a
    UNIFIED multi-sensor timeline (LiDAR ~10 Hz + RGB-D ~30 Hz merged on one
    clock, V5.6),
  * draws the trajectory + accumulating LiDAR point cloud LIVE,
  * lets the user switch loop-closure modules, the LiDAR front-end variant, and
    (V5.7) the tracking SENSOR itself LIVE on stdin while the SAME shared C++ map
    keeps mapping,
  * renders the fused occupancy map at the end.

The shared map (memory tiers + SE(2) graph + signatures) is sensor-agnostic, and
both front-ends emit REP-103 base poses, so the per-keyframe relative-motion
chaining (node_pose = last_graph_pose ∘ rel) is continuous across any switch.

Front-end axis (all live-switchable, V5.7): visual_vo | lidar_s2s | lidar_s2m.
Loop axes: verifier bnb|icp|pnp · proposer proximity|dbow.

Live switch commands (type + Enter):
    verifier bnb|icp|pnp · proposer proximity|dbow · fe vo|s2s|s2m · status · quit

Example:
  .venv/bin/python -m slam_core.fusion2.run_realtime --dataset datasets/lab_hybrid \
      --mode lidar --lidar-frontend native_s2s --verifier bnb --attach-visual
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import fusion_core as fc
from slam_core.fusion.signature import (CAMERA_GROUND_TRANSFORM,
                                        project_pose3d_to_pose2)
from slam_core.fusion2.config import FusionV2Config
from slam_core.fusion2.dataset import LabHybridStream
from slam_core.fusion2.lidar_frontend import make_lidar_frontend
from slam_core.fusion2.runner import (_rel, _rel_sane, build_shared_map,
                                      propose_candidates,
                                      verify_candidate_bnb, verify_candidate_icp)
from slam_core.fusion2.visual_features import BASE_T_CAM

VO_MODES = ("orb", "orb_lidar")
SENSOR_OF_MODE = {"lidar": "lidar", "lidar_orb": "lidar", "orb": "vo", "orb_lidar": "vo"}


# SE(2) node poses MUST be REP-103 (x-forward) — the same frame the PnP loop
# edges (cam_rel_to_base_se2) and LiDAR scans live in (the v3.6 frame fix).
def _cam_to_se2(Twc) -> fc.Pose2:
    p = project_pose3d_to_pose2(Twc, base_T_cam=BASE_T_CAM,
                                world_transform=CAMERA_GROUND_TRANSFORM)
    return fc.Pose2(p.x, p.y, p.theta)


# ---------------------------------------------------------------------------
# Uniform per-keyframe contract produced by either front-end adapter. The
# adapter pre-computes everything sensor-specific (the relative motion, the
# stationarity/insert-similarity, the blind flag, the visual payload) so the
# IngestEngine can apply it uniformly regardless of which sensor produced it.
# ---------------------------------------------------------------------------
@dataclass
class KeyframeData:
    t: float
    fe_pose: fc.Pose2                 # this KF's front-end pose (first-KF anchor)
    rel: Optional[fc.Pose2]           # rel motion from prev graph node; None=first/post-switch
    raw_scan: np.ndarray              # (N,2) float; may be empty
    visual: Optional[tuple]           # (kpts, des, pts3d) or None
    blind: bool                       # REINIT KF: soft spine + excluded from map
    similarity: float                 # memory.insert similarity (stationary gate baked in)
    sensor: str                       # "lidar" | "vo"


# ---------------------------------------------------------------------------
# Per-keyframe ingestion into the shared map. Generalized over the sensor: the
# LiDAR path (run_lidar_mode) and the VO path (run_orb_mode_native) differ only
# in the KeyframeData their adapters produce; this applies them identically.
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
        self.last_graph_pose = None
        self.blind_ids: set = set()
        self.kf_stamps: dict = {}
        self.stats = dict(keyframes=0, proposals=0, verified=0, loops_accepted=0,
                          rehearsal_merges=0, optimize_calls=0, reinits=0)

    # -- modality configuration -------------------------------------------
    def set_active_sensor(self, sensor: str):
        """Widen the B&B angular search window for visual-led scan verification
        (orb_lidar): VO heading drift needs the ±45° window vs the LiDAR-led
        ±30°. (ICP is standalone/prediction-seeded and needs no per-sensor
        config -- see verify_candidate_icp.)"""
        self.shared.bnb_cfg.angular_search_window = (
            self.cfg.orb_bnb_window_th if sensor == "vo" else self.cfg.bnb_window_th)

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
    def ingest(self, kfd: KeyframeData):
        """Insert a keyframe into the shared map; returns (kf_id, node_pose, sig).
        Unifies run_lidar_mode (415-462) and run_orb_mode_native (833-866)."""
        cfg, shared = self.cfg, self.shared
        self.kf_id += 1
        kf_id = self.kf_id
        self.kf_stamps[kf_id] = kfd.t
        if kfd.rel is None:
            # very first KF -> front-end pose; first KF after a switch -> continue
            # from the last graph pose (no spine edge across the gap).
            node_pose = self.last_graph_pose if self.last_graph_pose is not None else kfd.fe_pose
        else:
            node_pose = self.last_graph_pose.compose(kfd.rel)

        scan = kfd.raw_scan if kfd.raw_scan is not None else np.zeros((0, 2), np.float32)
        kpts = des = pts3d = None
        if kfd.visual is not None:
            kpts, des, pts3d = kfd.visual
        if kpts is not None:
            sig = fc.Signature(kf_id, kfd.t, kpts=kpts, des=des, pts3d=pts3d, scan_xy=scan)
        else:
            sig = fc.Signature(kf_id, kfd.t, scan_xy=scan)
        sig.pose = node_pose

        res = shared.memory.insert(sig, similarity=kfd.similarity)
        if res.rehearsal_merged:
            self.stats["rehearsal_merges"] += 1
        shared.graph.add_node(kf_id, node_pose)
        if kfd.blind:
            self.blind_ids.add(kf_id)
        if kfd.rel is not None:
            tw = cfg.blind_spine_weight if kfd.blind else cfg.spine_trans_weight
            rw = cfg.blind_spine_weight if kfd.blind else cfg.spine_rot_weight
            sig.add_link(kf_id - 1, fc.LinkType.NEIGHBOR, kfd.rel, tw, rw)
            shared.graph.add_spine_edge(kf_id - 1, kf_id, kfd.rel, tw, rw)
        self.last_graph_pose = node_pose
        self.stats["keyframes"] += 1
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

    # -- verifier dispatch (gates replicate the batch runner) --------------
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

        # scan verifiers (bnb / icp) — ICP is standalone, prediction-seeded
        if self.verifier == "icp":
            r = verify_candidate_icp(shared, cfg, kf_id, raw_scan, node_pose, cand)
        else:
            r = verify_candidate_bnb(shared, cfg, kf_id, raw_scan, node_pose, cand)
        if r is None:
            return False, None
        self.stats["verified"] += 1
        if self.verifier == "icp":
            accepted = (r.success and r.coarse_score >= cfg.icp_accept_fitness
                        and r.refined_score <= cfg.icp_accept_rmse)
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
        # dbow queries every keyframe (its index enforces separation); proximity
        # is gated to every Nth keyframe past the separation horizon.
        propose = (self.proposer == "dbow") or \
                  (kf_id % cfg.propose_every_n_kf == 0 and kf_id > cfg.min_kf_separation)
        if propose:
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
# transfer); continuity is preserved by resetting the relative-motion baseline to
# the new front-end's pose at the flip. Mirrors MatcherManager.request_switch
# from hector/run_realtime_viz.py.
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
# Per-sensor front-end adapters. Each consumes ITS sensor's timeline events and
# produces a uniform KeyframeData (or None on a non-keyframe event), so the
# driver and IngestEngine are sensor-agnostic.
# ---------------------------------------------------------------------------
class LidarFEAdapter:
    sensor = "lidar"

    def __init__(self, cfg, stream, K, attach_visual, initial_kind, grace_scans=15):
        self.cfg = cfg
        self.stream = stream
        self.K = K
        self.attach_visual = attach_visual
        imu_path = str(Path(cfg.dataset) / "imu.csv")
        make_fe = lambda kind: make_lidar_frontend(
            kind, dataset_name="lab_hybrid", imu_path=imu_path,
            kf_min_dist_m=cfg.kf_min_dist_m, kf_min_angle_rad=cfg.kf_min_angle_rad,
            kf_min_dt_s=cfg.kf_min_dt_s)
        self.fem = FrontEndManager(make_fe, initial_kind, grace_scans)
        self.last_fe_pose: Optional[fc.Pose2] = None
        self._last_pose: Optional[fc.Pose2] = None    # latest FE estimate (any scan)

    @property
    def variant(self) -> str:
        return self.fem.active_kind

    def current_pose(self) -> Optional[fc.Pose2]:
        """The latest front-end pose estimate (for a cross-sensor flip baseline)."""
        return self._last_pose

    def reset_baseline(self, pose: fc.Pose2):
        """On a cross-sensor handoff INTO this adapter, anchor relative-motion
        chaining at `pose` so the next keyframe continues from the last graph
        pose (V5.3 fresh-from-handoff mechanism, sensor-independent)."""
        self.last_fe_pose = pose

    def feed(self, ev) -> Tuple[Optional[KeyframeData], Optional[fc.Pose2]]:
        _, t, scan = ev
        fe_pose_py, pts, is_kf, flip = self.fem.process(t, scan)
        self._last_pose = fc.Pose2(float(fe_pose_py.x), float(fe_pose_py.y),
                                   float(fe_pose_py.theta))
        if flip is not None:
            # handoff complete: reset the relative-motion baseline to the new
            # front-end's frame so chaining continues seamlessly (no jump).
            self.last_fe_pose = flip
        if not is_kf:
            return None, flip
        fe_pose = fc.Pose2(float(fe_pose_py.x), float(fe_pose_py.y), float(fe_pose_py.theta))
        if self.last_fe_pose is None:
            rel = None
            stationary = False
        else:
            rel = _rel(self.last_fe_pose, fe_pose)
            stationary = math.hypot(rel.x, rel.y) < 0.05 and abs(rel.theta) < math.radians(2.0)
        visual = _extract_visual(self.stream, t, self.K) if self.attach_visual else None
        self.last_fe_pose = fe_pose
        kfd = KeyframeData(t=t, fe_pose=fe_pose, rel=rel,
                           raw_scan=np.asarray(scan, np.float64), visual=visual,
                           blind=False, similarity=(-1.0 if stationary else 0.0),
                           sensor="lidar")
        return kfd, flip


class VoFEAdapter:
    sensor = "vo"

    def __init__(self, cfg, stream):
        from slam_core.fusion2.vo_orb_frontend import NativeOrbFrontend
        self.cfg = cfg
        self.stream = stream
        self.fe = NativeOrbFrontend(cfg.dataset,
                                    imu_path=str(Path(cfg.dataset) / "imu.csv"),
                                    depth_max=cfg.vo_depth_max,
                                    imu_dropout=cfg.vo_imu_dropout,
                                    vo_overrides=cfg.vo_overrides)
        self.K = self.fe.K
        self.reinits = 0
        self.variant = "visual_vo"
        self._last_pose: Optional[fc.Pose2] = None    # latest tracked pose (any frame)
        self._baseline_override: Optional[fc.Pose2] = None

    def current_pose(self) -> Optional[fc.Pose2]:
        return self._last_pose

    def reset_baseline(self, pose: fc.Pose2):
        """On a cross-sensor handoff INTO VO, anchor the NEXT keyframe's relative
        motion at `pose` (the VO estimate at the flip instant) so it chains from
        the last graph pose — instead of from the discarded grace keyframe."""
        self._baseline_override = pose

    def feed(self, ev) -> Tuple[Optional[KeyframeData], Optional[fc.Pose2]]:
        import cv2
        _, t, rgb_p, depth_p = ev
        rgb = cv2.imread(str(rgb_p), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_p), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            return None, None
        Twc, state, nkf = self.fe.track(rgb, depth, t)
        self._last_pose = _cam_to_se2(Twc)
        if state == fc.VoState.REINIT:
            self.reinits += 1
        if nkf is None:
            return None, None
        fe_pose = _cam_to_se2(nkf.Twc)
        blind = nkf.state == fc.VoState.REINIT
        if self._baseline_override is not None:
            # first keyframe after a cross-sensor handoff: chain from the flip
            # baseline, then resume the BA-refined prev_Twc spine.
            rel = _rel(self._baseline_override, fe_pose)
            self._baseline_override = None
        elif nkf.prev_Twc is None:
            rel = None
        else:
            rel = _rel(_cam_to_se2(nkf.prev_Twc), fe_pose)
        scan = self.stream.nearest_scan(t)
        scan = scan if scan is not None else np.zeros((0, 2), np.float32)
        visual = (nkf.kpts, nkf.des, nkf.pts3d_cam)
        kfd = KeyframeData(t=t, fe_pose=fe_pose, rel=rel,
                           raw_scan=np.asarray(scan, np.float64), visual=visual,
                           blind=blind, similarity=0.0, sensor="vo")
        return kfd, None


def make_adapter(sensor_kind: str, cfg, stream, K, attach_visual, grace_scans=15):
    """Build the adapter for a front-end kind:
    visual_vo | native_s2s | native_s2m (the last two share the LiDAR adapter)."""
    if sensor_kind == "visual_vo":
        return VoFEAdapter(cfg, stream)
    return LidarFEAdapter(cfg, stream, K, attach_visual, sensor_kind, grace_scans)


SENSOR_OF_KIND = {"visual_vo": "vo", "native_s2s": "lidar", "native_s2m": "lidar"}
SENSOR_OF_EVENT = {"lidar": "lidar", "rgbd": "vo"}    # merged-timeline tag -> sensor


# ---------------------------------------------------------------------------
# SensorManager — orchestrates CROSS-sensor (visual VO <-> LiDAR) front-end
# switches on top of the per-sensor adapters (V5.7). Same-sensor LiDAR variant
# switches (s2s<->s2m) delegate to the LiDAR adapter's own FrontEndManager.
#
# On a cross-sensor request a fresh target adapter is built and warmed up on ITS
# sensor's events for `grace_scans` events while the active adapter keeps driving
# the graph. At the flip the target becomes active and its relative-motion
# baseline is reset to its own pose at that instant, so the first post-flip
# keyframe continues from the last graph pose (valid because both adapters emit
# REP-103 base poses).
# ---------------------------------------------------------------------------
class SensorManager:
    def __init__(self, make_adapter_fn, initial_adapter, grace_scans: int = 20):
        self._make = make_adapter_fn          # (kind) -> adapter
        self.active = initial_adapter
        self.grace_scans = int(grace_scans)
        self.pending = None
        self.pending_kind = None
        self._grace = 0

    @property
    def active_sensor(self) -> str:
        return self.active.sensor

    @property
    def active_variant(self) -> str:
        return self.active.variant

    def request_switch(self, target_kind: str) -> str:
        target_sensor = SENSOR_OF_KIND[target_kind]
        if target_sensor == self.active.sensor:
            # same-sensor: a LiDAR variant switch handled by the LiDAR adapter.
            if isinstance(self.active, LidarFEAdapter):
                return self.active.fem.request_switch(target_kind)
            return f"already on {target_kind}; ignored"
        if self.pending is not None:
            return f"cross-sensor switch to {self.pending_kind} in progress; ignored"
        self.pending = self._make(target_kind)
        self.pending_kind = target_kind
        self._grace = self.grace_scans
        return (f"requested cross-sensor -> {target_kind}; effective in "
                f"{self.grace_scans} {target_sensor} events (warming up)")

    def feed(self, ev):
        """Route an event. The pending adapter (if any) warms up on its sensor's
        events; the active adapter drives the graph on its sensor's events.
        Returns (kfd, fe_flip, sensor_flip):
          kfd          — KeyframeData for the graph, or None
          fe_flip      — same-sensor (s2s<->s2m) variant handoff pose, or None
          sensor_flip  — dict(kind, sensor) when a cross-sensor flip completes."""
        sensor = SENSOR_OF_EVENT[ev[0]]
        sensor_flip = None
        just_flipped = False
        if self.pending is not None and sensor == self.pending.sensor:
            self.pending.feed(ev)                  # warm up (output discarded)
            self._grace -= 1
            if self._grace <= 0:
                pk = self.pending_kind
                self.active = self.pending
                self.pending = self.pending_kind = None
                base = self.active.current_pose()
                if base is not None:
                    self.active.reset_baseline(base)
                sensor_flip = dict(kind=pk, sensor=self.active.sensor)
                just_flipped = True
        kfd = fe_flip = None
        if not just_flipped and sensor == self.active.sensor:
            kfd, fe_flip = self.active.feed(ev)
        return kfd, fe_flip, sensor_flip


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
        print("\n" + "=" * 56 + "\n  REAL-TIME TIMING\n" + "=" * 56)
        for s in self.STAGES:
            mean_ms = self.total[s] / self.n * 1e3 if self.n else 0.0
            print(f"  {s:<11} {self.total[s]:8.3f}s  {mean_ms:7.2f} ms/event")
        work = {s: self.total[s] for s in self.STAGES if s != "sleep"}
        bn = max(work, key=work.get) if work else "n/a"
        print(f"  events={self.n}  late={self.late} "
              f"({100.0*self.late/max(1,self.n):.1f}%)  bottleneck={bn}")
        print(f"  verdict: {'REAL-TIME OK' if lag <= 0.05 else f'BEHIND by {lag:.3f}s'}")
        print("=" * 56)


# ---------------------------------------------------------------------------
# Unified multi-sensor timeline
# ---------------------------------------------------------------------------
def merged_events(stream: LabHybridStream):
    """Two-pointer merge of the LiDAR (~10 Hz) and RGB-D (~30 Hz) timelines into
    one time-ordered event stream. Yields:
        ("lidar", t, scan_points)         and
        ("rgbd",  t, rgb_path, depth_path)
    RGB-D events carry only paths (cheap); images are decoded lazily by the VO
    adapter, so merging both streams costs nothing for a LiDAR-led run."""
    lid = stream.lidar_stream(0)
    rgb = iter(stream.rgbd_entries())
    nl = next(lid, None)
    nr = next(rgb, None)
    while nl is not None or nr is not None:
        if nr is None or (nl is not None and nl[0] <= nr[0]):
            yield ("lidar", float(nl[0]), nl[1])
            nl = next(lid, None)
        else:
            yield ("rgbd", float(nr[0]), nr[1], nr[2])
            nr = next(rgb, None)


# ---------------------------------------------------------------------------
# CLI + main loop
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Fusion real-time module-switching runner (V5)")
    p.add_argument("--dataset", type=Path, default=Path("datasets/lab_hybrid"))
    p.add_argument("--mode", choices=("lidar", "lidar_orb", "orb", "orb_lidar"),
                   default="lidar",
                   help="Starting front-end: lidar* (LiDAR-led) or orb* (visual-led).")
    p.add_argument("--lidar-frontend", choices=("native_s2s", "native_s2m"),
                   default="native_s2s")
    p.add_argument("--verifier", choices=("bnb", "icp", "pnp"), default="bnb",
                   help="Scan verifier (bnb|icp) for scan modes; pnp is auto for "
                        "orb/lidar_orb.")
    p.add_argument("--proposer", choices=("proximity", "dbow"), default=None,
                   help="Override; defaults to dbow for orb* and proximity for lidar*.")
    p.add_argument("--attach-visual", action="store_true",
                   help="Extract+store ORB on every LiDAR keyframe so pnp/dbow are "
                        "available live (~78 KB/keyframe).")
    p.add_argument("--max-scans", type=int, default=0,
                   help="Cap on active-sensor keyframe-driving events (0 = all).")
    p.add_argument("--grace-scans", type=int, default=20,
                   help="Warm-up events for a front-end handoff (variant + cross-sensor).")
    p.add_argument("--speed", type=float, default=1.0,
                   help="1.0=real-time, 0=as-fast-as-possible.")
    p.add_argument("--draw-every", type=int, default=5)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--output", type=Path, default=Path("fusion2_outputs"))
    p.add_argument("--no-map", action="store_true")
    return p.parse_args()


def _mode_defaults(mode: str, scan_verifier: str) -> Tuple[str, str]:
    """(proposer, verifier) for a starting mode, matching the batch runner."""
    if mode == "orb":
        return "dbow", "pnp"
    if mode == "orb_lidar":
        return "dbow", scan_verifier          # bnb | icp (B&B-seeded)
    if mode == "lidar_orb":
        return "proximity", "pnp"
    return "proximity", scan_verifier          # lidar


def main(argv=None):
    args = _parse_args()
    cfg = FusionV2Config(mode=args.mode, dataset=args.dataset, output_dir=args.output,
                         lidar_frontend=args.lidar_frontend, scan_verifier=args.verifier,
                         max_scans=args.max_scans)
    start_sensor = SENSOR_OF_MODE[args.mode]
    proposer, verifier = _mode_defaults(args.mode, args.verifier)
    if args.proposer is not None:
        proposer = args.proposer

    # LiDAR adapters attach the visual payload (so pnp/dbow survive a switch TO
    # LiDAR) iff the user opted in (--attach-visual) or started in lidar_orb.
    # VO adapters are always visual. This holds for the initial AND any pending
    # (cross-sensor) adapter, so capability is consistent across switches.
    lidar_attach = bool(args.attach_visual) or args.mode == "lidar_orb"
    start_visual = True if start_sensor == "vo" else lidar_attach
    if verifier == "pnp" and not start_visual:
        raise SystemExit("pnp verification needs visual payload: add --attach-visual "
                         "(or --mode lidar_orb).")
    if proposer == "dbow" and not start_visual:
        raise SystemExit("--proposer dbow needs descriptors: add --attach-visual.")

    stream = LabHybridStream(cfg.dataset, cfg.sync_tolerance_s)

    # camera intrinsics — needed by any visual-attaching adapter (PnP / dbow).
    import yaml
    sc = yaml.safe_load(open(Path(cfg.dataset) / "sensor_config.yaml"))["camera"]
    K = np.array([[sc["fx"], 0, sc["cx"]], [0, sc["fy"], sc["cy"]], [0, 0, 1]])

    def _build(kind):
        return make_adapter(kind, cfg, stream, K, lidar_attach,
                            grace_scans=args.grace_scans)

    initial_kind = "visual_vo" if start_sensor == "vo" else cfg.lidar_frontend
    sm = SensorManager(_build, _build(initial_kind), grace_scans=args.grace_scans)
    shared = build_shared_map(cfg)
    eng = IngestEngine(shared, cfg, K=K, verifier=verifier, proposer=proposer)
    eng.set_active_sensor(start_sensor)
    if proposer == "dbow":
        eng.ensure_appearance()
    visual_available = _visual_available(sm.active)

    print("=" * 60)
    print(f"Dataset   : {cfg.dataset}")
    print(f"Front-end : {sm.active_variant}  (mode {cfg.mode}, sensor {start_sensor})")
    print(f"Verifier  : {verifier}   Proposer: {proposer}")
    print(f"Visual    : {'available' if visual_available else 'lean (scan only)'}"
          f"{'' if start_sensor == 'vo' else ' (LiDAR attach=%s)' % lidar_attach}")
    print(f"Playback  : {'real-time' if args.speed > 0 else 'max'} (speed={args.speed}x)")
    print("Live cmds : verifier bnb|icp|pnp · proposer proximity|dbow · "
          "fe vo|lidar|s2s|s2m · status · quit")
    print("=" * 60)

    switch_q: "queue.Queue" = queue.Queue()
    _start_switch_reader(switch_q)

    live = LiveView(f"fusion realtime — {cfg.mode}/{sm.active_variant}")
    timer = StageTimer()

    xs, ys, cloud_chunks = [], [], []
    kf_ids, kf_scans = [], []     # for live re-projection on loop correction
    t0_data = t0_wall = None
    lag = 0.0
    active_events = 0

    events = list(merged_events(stream))
    n_events = len(events)
    for k, ev in enumerate(events):
        sensor, t = ev[0], ev[1]
        if t0_data is None:
            t0_data, t0_wall = t, time.perf_counter()
        ev_start = time.perf_counter()

        # drain live commands (main thread only)
        if _apply_switches(switch_q, eng, sm, k):
            break

        # route the event: pending FE warms up on its sensor, active FE drives.
        t_a = time.perf_counter()
        kfd, fe_flip, sensor_flip = sm.feed(ev)
        timer.add("slam", time.perf_counter() - t_a)
        if sensor_flip is not None:
            # cross-sensor flip: re-tune scan verification for the new modality
            # and auto-fall-back any module the new front-end cannot feed.
            eng.set_active_sensor(sensor_flip["sensor"])
            fb = _autofallback(eng, sm.active)
            visual_available = _visual_available(sm.active)
            cfg.lidar_frontend = sm.active_variant
            extra = (" [auto-fallback: " + ", ".join(fb) + "]") if fb else ""
            print(f"[switch] >>> SENSOR now {sensor_flip['sensor']} "
                  f"({sm.active_variant}) at k={k} (graph continues from last "
                  f"pose).{extra}")
        if fe_flip is not None:
            cfg.lidar_frontend = sm.active_variant
            print(f"[switch] >>> front-end now {sm.active_variant} at k={k} "
                  f"(graph continues from last pose).")
        if kfd is not None:
            active_events += 1
            scan = np.asarray(kfd.raw_scan, np.float64)
            t_b = time.perf_counter()
            kf_id, node_pose, sig = eng.ingest(kfd)
            did_opt = eng.close_loops(kf_id, node_pose, sig, scan)
            timer.add("loop", time.perf_counter() - t_b)
            kf_ids.append(kf_id)
            kf_scans.append(scan)
            if did_opt:
                # LOOP CORRECTION applied live: rebuild the WHOLE displayed
                # trajectory + cloud from the corrected graph at once.
                xs, ys, cloud_chunks = _rebuild_display(shared, kf_ids, kf_scans)
            else:
                gp = shared.graph.get_pose(kf_id)
                xs.append(gp.x); ys.append(gp.y)
                if len(scan):
                    c, s = math.cos(gp.theta), math.sin(gp.theta)
                    cloud_chunks.append(scan @ np.array([[c, -s], [s, c]]).T + [gp.x, gp.y])

        # live draw (throttled)
        t_c = time.perf_counter()
        if live.ok and xs and k % max(1, args.draw_every) == 0:
            cloud = np.vstack(cloud_chunks) if cloud_chunks else None
            if cloud is not None and len(cloud) > 60000:
                cloud = cloud[:: len(cloud) // 60000 + 1]
            live.update(xs, ys,
                        cloud[:, 0] if cloud is not None else None,
                        cloud[:, 1] if cloud is not None else None,
                        title=f"{cfg.mode}/{sm.active_variant} v={eng.verifier} "
                              f"p={eng.proposer} kf={eng.stats['keyframes']} "
                              f"loops={eng.stats['loops_accepted']}")
        timer.add("draw", time.perf_counter() - t_c)

        # real-time pacing
        period = (events[k + 1][1] - t) if (k + 1) < n_events else 0.0
        if period > 0 and (time.perf_counter() - ev_start) > period and args.speed > 0:
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
                  f"p={eng.proposer} fe={sm.active_variant} lag={lag:6.3f}s")

        if args.max_scans and active_events >= args.max_scans:
            break

    # final optimize + outputs (+ auto-display the corrected fused map)
    shared.graph.optimize()
    eng.stats["optimize_calls"] += 1
    if isinstance(sm.active, VoFEAdapter):
        eng.stats["reinits"] = sm.active.reinits
    timer.summary(lag)
    _finalize_and_show(shared, cfg, eng, sm.active, args, live)


def _visual_available(adapter) -> bool:
    """Whether the active adapter's keyframes carry a visual payload (pnp/dbow)."""
    if isinstance(adapter, VoFEAdapter):
        return True
    return bool(getattr(adapter, "attach_visual", False))


def _autofallback(eng, new_adapter) -> list:
    """When a cross-sensor flip strands the active verifier/proposer (the new
    front-end can't feed it), fall back to a compatible module and report it."""
    msgs = []
    if not _visual_available(new_adapter):
        if eng.verifier == "pnp":
            eng.verifier = "bnb"
            msgs.append("verifier pnp->bnb")
        if eng.proposer == "dbow":
            eng.proposer = "proximity"
            msgs.append("proposer dbow->proximity")
    return msgs


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


def _start_switch_reader(q):
    def _reader():
        import sys
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd:
                q.put(cmd)
    threading.Thread(target=_reader, name="switch-reader", daemon=True).start()


_FE_ALIAS = {"s2s": "native_s2s", "s2m": "native_s2m",
             "native_s2s": "native_s2s", "native_s2m": "native_s2m",
             "vo": "visual_vo", "orb": "visual_vo", "visual": "visual_vo",
             "visual_vo": "visual_vo", "lidar": "native_s2s"}


def _apply_switches(q, eng, sm, k) -> bool:
    """Drain stdin commands on the MAIN thread. Returns True on quit. Switches:
    verifier (bnb|icp|pnp), proposer (proximity|dbow), and the front-end —
    LiDAR variant s2s<->s2m AND cross-sensor visual<->LiDAR (V5.7)."""
    visual_available = _visual_available(sm.active)
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
            pend = f" (switching->{sm.pending_kind})" if sm.pending_kind else ""
            if isinstance(sm.active, LidarFEAdapter) and sm.active.fem.pending:
                pend = f" (variant->{sm.active.fem.pending})"
            print(f"[status] k={k} sensor={sm.active_sensor} fe={sm.active_variant}{pend} "
                  f"verifier={eng.verifier} proposer={eng.proposer} "
                  f"kf={eng.stats['keyframes']} loops={eng.stats['loops_accepted']} "
                  f"visual={'on' if visual_available else 'off'}")
            continue
        if head == "verifier" and len(parts) == 2:
            v = parts[1]
            if v not in ("bnb", "icp", "pnp"):
                print(f"[switch] unknown verifier {v!r}")
            elif v == "pnp" and not visual_available:
                print("[switch] pnp unavailable on the active front-end (no visual "
                      "payload). Switch to a visual front-end or run --attach-visual.")
            else:
                eng.verifier = v
                print(f"[switch] verifier -> {v} (effective next proposal).")
            continue
        if head == "proposer" and len(parts) == 2:
            pr = parts[1]
            if pr not in ("proximity", "dbow"):
                print(f"[switch] unknown proposer {pr!r}")
            elif pr == "dbow" and not visual_available:
                print("[switch] dbow unavailable on the active front-end (no "
                      "descriptors). Switch to a visual front-end or run --attach-visual.")
            else:
                if pr == "dbow":
                    eng.ensure_appearance()
                eng.proposer = pr
                print(f"[switch] proposer -> {pr} (dbow only sees indexed keyframes).")
            continue
        if head == "fe" and len(parts) == 2:
            target = parts[1]
            if target not in _FE_ALIAS:
                print(f"[switch] unknown front-end {target!r}. "
                      "Try: fe vo | fe lidar | fe s2s | fe s2m")
            else:
                print(f"[switch] {sm.request_switch(_FE_ALIAS[target])}")
            continue
        print(f"[switch] unknown command {cmd!r}. "
              "Try: verifier bnb|icp|pnp · proposer proximity|dbow · "
              "fe vo|lidar|s2s|s2m · status · quit")


def _rebuild_display(shared, kf_ids, kf_scans):
    """Re-read every keyframe pose from the (just-optimized) graph and re-project
    its scan, so the live trajectory + cloud reflect the loop correction at once.
    Cheap (poses + a numpy transform); runs only on optimize events."""
    xs, ys, chunks = [], [], []
    for kfid, scan in zip(kf_ids, kf_scans):
        gp = shared.graph.get_pose(kfid)
        xs.append(gp.x); ys.append(gp.y)
        if len(scan):
            c, s = math.cos(gp.theta), math.sin(gp.theta)
            chunks.append(scan @ np.array([[c, -s], [s, c]]).T + [gp.x, gp.y])
    return xs, ys, chunks


def _finalize_and_show(shared, cfg, eng, adapter, args, live):
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
        json.dump(dict(mode=cfg.mode, frontend=adapter.variant,
                       final_verifier=eng.verifier, final_proposer=eng.proposer,
                       blind_kfs=len(eng.blind_ids), **eng.stats), f, indent=2, default=str)

    if args.no_map or not len(poses):
        print(f"\nWrote: {run_dir}")
        if live.ok:
            live.keep_open()
        return

    # fuse all scans at the FINAL corrected poses into one occupancy grid;
    # REINIT (blind) keyframes' scans are dead-reckoned and excluded.
    rs, rp = [], []
    for nid, x, y, th in poses:
        if int(nid) in eng.blind_ids:
            continue
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
    ax.set_title(f"realtime {cfg.mode}/{adapter.variant} — final corrected map: "
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
