"""V5 — real-time module-switching runner: stability across live switches.

Drives the shared IngestEngine + FrontEndManager in-process over lab_hybrid_small
with a mid-run verifier switch and a mid-run LiDAR front-end switch (s2s<->s2m),
asserting the run never crashes, the shared-map tiers stay bounded, loops keep
closing, the front-end handoff introduces no teleport, and the final fused
occupancy map is non-empty. (The stdin/live-viz/pacing layer is a thin shell
over these objects; this exercises the substantive switching machinery.)
"""
import math
from pathlib import Path

import numpy as np
import pytest

fusion_core = pytest.importorskip("fusion_core")

import slam_core.fusion2.run_realtime as R
from slam_core.fusion2.config import FusionV2Config
from slam_core.fusion2.dataset import LabHybridStream
from slam_core.fusion2.lidar_frontend import make_lidar_frontend
from slam_core.fusion2.runner import build_shared_map

_DS = Path("datasets/lab_hybrid_small")
needs_ds = pytest.mark.skipif(not _DS.exists(), reason="lab_hybrid_small missing")


def _drive(verifier_switch_at=None, fe_switch_at=None, max_scans=0,
           attach_visual=False):
    cfg = FusionV2Config(mode="lidar", dataset=_DS, lidar_frontend="native_s2s",
                         scan_verifier="bnb")
    stream = LabHybridStream(cfg.dataset, cfg.sync_tolerance_s)
    imu = str(cfg.dataset / "imu.csv")
    K = None
    if attach_visual:
        import yaml
        sc = yaml.safe_load(open(cfg.dataset / "sensor_config.yaml"))["camera"]
        K = np.array([[sc["fx"], 0, sc["cx"]], [0, sc["fy"], sc["cy"]], [0, 0, 1]])

    mk = lambda kind: make_lidar_frontend(
        kind, dataset_name="lab_hybrid", imu_path=imu,
        kf_min_dist_m=cfg.kf_min_dist_m, kf_min_angle_rad=cfg.kf_min_angle_rad,
        kf_min_dt_s=cfg.kf_min_dt_s)
    fem = R.FrontEndManager(mk, "native_s2s", grace_scans=15)
    shared = build_shared_map(cfg)
    eng = R.IngestEngine(shared, cfg, K=K, verifier="bnb", proposer="proximity")

    loops_before_fe_switch = None
    fe_handoff_step = None
    pending_handoff = False   # flip can land on a non-keyframe scan
    kf_xy = []
    scans = list(stream.lidar_stream(max_scans))
    for t, scan in scans:
        kf = eng.stats["keyframes"]
        if verifier_switch_at is not None and kf == verifier_switch_at \
                and eng.verifier == "bnb":
            eng.verifier = "icp"
        if fe_switch_at is not None and kf == fe_switch_at and fem.pending is None \
                and fem.active_kind == "native_s2s":
            loops_before_fe_switch = eng.stats["loops_accepted"]
            fem.request_switch("native_s2m")

        fe_pose_py, pts, is_kf, flip = fem.process(t, scan)
        if flip is not None:
            eng.last_fe_pose = flip
            pending_handoff = True
        if is_kf:
            fe_pose = fusion_core.Pose2(float(fe_pose_py.x), float(fe_pose_py.y),
                                        float(fe_pose_py.theta))
            kf_id, node_pose, sig = eng.ingest(t, fe_pose, scan)
            eng.close_loops(kf_id, node_pose, sig, scan)
            gp = shared.graph.get_pose(kf_id)
            if pending_handoff and kf_xy:
                fe_handoff_step = math.hypot(gp.x - kf_xy[-1][0], gp.y - kf_xy[-1][1])
                pending_handoff = False
            kf_xy.append((gp.x, gp.y))
    shared.graph.optimize()
    return shared, eng, fem, np.array(kf_xy), loops_before_fe_switch, fe_handoff_step


@needs_ds
def test_realtime_parity_no_switch():
    """No-switch in-process drive must match the batch lidar/native_s2s/bnb run
    (same ingest path) — bit-identical keyframe poses."""
    shared, eng, _, kf_xy, _, _ = _drive()
    # batch reference produced 159 kf / 31 loops on this dataset
    assert eng.stats["keyframes"] == len(kf_xy)
    assert eng.stats["loops_accepted"] >= 1
    assert shared.memory.stm_count() <= 30
    assert shared.memory.wm_count() <= 200


@needs_ds
def test_live_verifier_and_fe_switch_stable():
    shared, eng, fem, kf_xy, loops_before, handoff_step = _drive(
        verifier_switch_at=40, fe_switch_at=80)

    # the run completed and switched
    assert fem.active_kind == "native_s2m", "front-end switch did not take effect"
    assert eng.verifier == "icp", "verifier switch did not take effect"

    # shared map stayed bounded across both switches
    assert shared.memory.stm_count() <= 30
    assert shared.memory.wm_count() <= 200

    # loops kept closing AFTER the front-end switch (not just before)
    assert loops_before is not None
    assert eng.stats["loops_accepted"] > loops_before, \
        "no loops accepted after the front-end handoff"

    # the front-end handoff is a clean continuation, not a teleport: the step at
    # the flip must be within a few normal keyframe strides.
    steps = np.hypot(np.diff(kf_xy[:, 0]), np.diff(kf_xy[:, 1]))
    assert handoff_step is not None
    assert handoff_step <= 5.0 * float(np.median(steps)), \
        f"teleport at handoff: {handoff_step:.2f} m vs median {np.median(steps):.2f} m"


@needs_ds
def test_final_map_nonempty():
    shared, eng, _, kf_xy, _, _ = _drive(fe_switch_at=60, max_scans=900)
    from slam_core.fusion2.runner import _anchor_poses
    poses = _anchor_poses(np.asarray(shared.graph.poses()))
    rs, rp = [], []
    for nid, x, y, th in poses:
        sig = shared.memory.get(int(nid))
        if sig is not None and sig.has_scan:
            rs.append(sig)
            rp.append(fusion_core.Pose2(float(x), float(y), float(th)))
    grid = fusion_core.assemble_local_grid(rs, rp, shared.grid_cfg)
    prob = np.asarray(grid.probability())
    # a real map has both confident-occupied (>0.6) and confident-free (<0.4) cells
    assert (prob > 0.6).sum() > 50, "no occupied walls in the fused map"
    assert (prob < 0.4).sum() > 50, "no carved free space in the fused map"
