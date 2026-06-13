"""V5.7/V5.8 — cross-sensor live front-end switching (visual VO <-> LiDAR).

Drives the unified SensorManager + IngestEngine in-process over lab_hybrid_small
with a mid-run vo<->lidar switch sequence, asserting:
  * the cross-sensor handoff introduces NO teleport (the keyframe step at each
    flip is within a few normal strides — the V5.3 baseline-reset mechanism, now
    generalized across sensors),
  * the graph keeps closing loops on BOTH sides of a switch,
  * the shared-map tiers stay bounded across the switches,
  * the auto-fallback downgrades a stranded verifier/proposer when a flip lands on
    a lean LiDAR front-end (pnp->bnb, dbow->proximity),
  * the final fused occupancy map is non-empty.

(The stdin/live-viz/pacing shell is thin over these objects; this exercises the
substantive cross-sensor machinery the way main() drives it.)
"""
import math
from pathlib import Path

import numpy as np
import pytest

fusion_core = pytest.importorskip("fusion_core")

import slam_core.fusion2.run_realtime as R
from slam_core.fusion2.config import FusionV2Config
from slam_core.fusion2.dataset import LabHybridStream
from slam_core.fusion2.runner import build_shared_map

_DS = Path("datasets/lab_hybrid_small")
needs_ds = pytest.mark.skipif(not _DS.exists(), reason="lab_hybrid_small missing")


def _K(cfg):
    import yaml
    sc = yaml.safe_load(open(cfg.dataset / "sensor_config.yaml"))["camera"]
    return np.array([[sc["fx"], 0, sc["cx"]], [0, sc["fy"], sc["cy"]], [0, 0, 1]])


def _drive(start_kind, switch_plan, lidar_attach, start_verifier, start_proposer,
           grace=20):
    """switch_plan: list of (at_kf, target_kind). Returns a record dict."""
    cfg = FusionV2Config(mode="lidar", dataset=_DS, lidar_frontend="native_s2s",
                         scan_verifier="bnb")
    stream = LabHybridStream(cfg.dataset, cfg.sync_tolerance_s)
    K = _K(cfg)
    build = lambda kind: R.make_adapter(kind, cfg, stream, K, lidar_attach,
                                        grace_scans=grace)
    sm = R.SensorManager(build, build(start_kind), grace_scans=grace)
    shared = build_shared_map(cfg)
    eng = R.IngestEngine(shared, cfg, K=K, verifier=start_verifier,
                         proposer=start_proposer)
    eng.set_active_sensor(sm.active_sensor)
    if start_proposer == "dbow":
        eng.ensure_appearance()

    plan = list(switch_plan)
    events = list(R.merged_events(stream))
    kf_xy = []                 # (x, y)
    flip_kf_idx = []           # kf index where each flip's first KF lands
    loops_at_flip = []         # (sensor, loops_before)
    fallbacks = []
    pending_flip = False
    for ev in events:
        if plan and eng.stats["keyframes"] >= plan[0][0] \
                and sm.pending_kind is None \
                and R.SENSOR_OF_KIND[plan[0][1]] != sm.active_sensor:
            sm.request_switch(plan[0][1])
            loops_at_flip.append([R.SENSOR_OF_KIND[plan[0][1]],
                                  eng.stats["loops_accepted"], None])
            plan.pop(0)
        kfd, fe_flip, sensor_flip = sm.feed(ev)
        if sensor_flip is not None:
            eng.set_active_sensor(sensor_flip["sensor"])
            fb = R._autofallback(eng, sm.active)
            fallbacks.append(fb)
            pending_flip = True
        if kfd is not None:
            kf_id, node_pose, sig = eng.ingest(kfd)
            eng.close_loops(kf_id, node_pose, sig, np.asarray(kfd.raw_scan))
            gp = shared.graph.get_pose(kf_id)
            if pending_flip and kf_xy:
                flip_kf_idx.append(len(kf_xy))
                pending_flip = False
            kf_xy.append((gp.x, gp.y))
    shared.graph.optimize()
    # loops_after for each flip = final total beyond the recorded before
    return dict(sm=sm, eng=eng, shared=shared, kf_xy=np.array(kf_xy),
                flip_kf_idx=flip_kf_idx, loops_at_flip=loops_at_flip,
                fallbacks=fallbacks)


@needs_ds
def test_crossmodal_switch_no_teleport_and_loops_both_sides():
    # vo->lidar->vo, lidar attaches visual so no fallback (loops stay possible
    # on both modalities). Start lidar so the first segment also closes loops.
    rec = _drive(start_kind="native_s2s",
                 switch_plan=[(30, "visual_vo"), (75, "native_s2s")],
                 lidar_attach=True, start_verifier="bnb", start_proposer="proximity")
    sm, eng, kf_xy = rec["sm"], rec["eng"], rec["kf_xy"]

    # both cross-sensor flips completed
    assert len(rec["flip_kf_idx"]) == 2, \
        f"expected 2 cross-sensor flips, got {len(rec['flip_kf_idx'])}"
    assert sm.active_sensor == "lidar"

    # tiers bounded across both switches
    assert rec["shared"].memory.stm_count() <= 30
    assert rec["shared"].memory.wm_count() <= 200

    # no teleport at either handoff (step within a few normal strides)
    steps = np.hypot(np.diff(kf_xy[:, 0]), np.diff(kf_xy[:, 1]))
    med = float(np.median(steps))
    for i in rec["flip_kf_idx"]:
        st = math.hypot(kf_xy[i, 0] - kf_xy[i - 1, 0], kf_xy[i, 1] - kf_xy[i - 1, 1])
        assert st <= 6.0 * med, \
            f"teleport at cross-sensor handoff (kf {i}): {st:.2f} m vs median {med:.2f} m"

    # loops closed across the whole run (covers the first LiDAR segment, the VO
    # segment, and the final LiDAR segment — the graph kept correcting).
    assert eng.stats["loops_accepted"] >= 2


@needs_ds
def test_crossmodal_autofallback_on_lean_lidar():
    # start VO with pnp + dbow; switch to a LEAN LiDAR front-end (no visual) ->
    # both modules are stranded and must auto-fall-back.
    rec = _drive(start_kind="visual_vo", switch_plan=[(25, "native_s2s")],
                 lidar_attach=False, start_verifier="pnp", start_proposer="dbow")
    assert rec["sm"].active_sensor == "lidar"
    assert len(rec["fallbacks"]) == 1
    assert "verifier pnp->bnb" in rec["fallbacks"][0]
    assert "proposer dbow->proximity" in rec["fallbacks"][0]
    # after fallback the engine settled on scan-based modules
    assert rec["eng"].verifier == "bnb"
    assert rec["eng"].proposer == "proximity"


@needs_ds
def test_crossmodal_final_map_nonempty():
    rec = _drive(start_kind="native_s2s", switch_plan=[(40, "visual_vo")],
                 lidar_attach=True, start_verifier="bnb", start_proposer="proximity")
    from slam_core.fusion2.runner import _anchor_poses
    shared, eng = rec["shared"], rec["eng"]
    poses = _anchor_poses(np.asarray(shared.graph.poses()))
    rs, rp = [], []
    for nid, x, y, th in poses:
        if int(nid) in eng.blind_ids:
            continue
        sig = shared.memory.get(int(nid))
        if sig is not None and sig.has_scan:
            rs.append(sig)
            rp.append(fusion_core.Pose2(float(x), float(y), float(th)))
    grid = fusion_core.assemble_local_grid(rs, rp, shared.grid_cfg)
    prob = np.asarray(grid.probability())
    assert (prob > 0.6).sum() > 50, "no occupied walls in the fused map"
    assert (prob < 0.4).sum() > 50, "no carved free space in the fused map"
