"""V6 — online/ROS live event source: drive the fusion engine from a LIVE
in-memory stream (no disk replay, no pacing), exactly as `run_realtime
--source ros` does on the Jetson, but fed by the dev replay feeder instead of
the ROS bridge.

Covers the three substantive online changes: the blocking live-event generator,
in-memory (decoded) RGB/depth images in the adapters, and the injected live IMU
buffer replacing imu.csv. Plus a Unix-socket wire round-trip (the bridge format).
"""
import math
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np
import pytest

fusion_core = pytest.importorskip("fusion_core")

import slam_core.fusion2.run_realtime as R
import slam_core.fusion2.ros_source as S
from slam_core.fusion2.config import FusionV2Config
from slam_core.fusion2.runner import build_shared_map, _anchor_poses

_DS = Path("datasets/lab_hybrid_small")
needs_ds = pytest.mark.skipif(not _DS.exists(), reason="lab_hybrid_small missing")


# --- unit: live IMU buffer ---------------------------------------------------
def test_live_imu_buffer():
    buf = S.LiveImuBuffer()
    for i in range(5):
        buf.add(1.0 + 0.1 * i, 0.05, 0.10 * i)
    assert buf.num_samples == 5
    # interpolates between samples; unwrapped/continuous
    y = buf.yaw_at(1.05)
    assert 0.0 < y < 0.10
    # samples list is what the LiDAR extrapolator drains
    assert buf.samples[0][0] == 1.0 and len(buf.samples[-1]) == 3
    assert buf.yaw_at(99.0) is not None     # clamps past the end
    assert buf.yaw_at(-1.0) is None          # before the start


# --- unit: Unix-socket wire round-trip (the bridge format) -------------------
@needs_ds
def test_wire_roundtrip():
    import cv2
    sock_path = f"/tmp/test_fusion_ros_{int(time.time()*1e6)}.sock"
    src = S.RosEventSource(sock_path, sync_tol=0.05).start()
    time.sleep(0.2)
    cli = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    cli.connect(sock_path)

    # imu, lidar, rgbd events through the wire
    cli.sendall(S.encode_event(S.TYPE_IMU, 1.0, struct.pack(">dd", 0.05, 0.3)))
    scan = np.random.rand(50, 2).astype(np.float32)
    cli.sendall(S.encode_event(S.TYPE_LIDAR, 1.1, scan.tobytes()))
    rgb = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    depth = (np.random.rand(8, 8) * 1000).astype(np.uint16)
    rb = cv2.imencode(".png", rgb)[1].tobytes()
    db = cv2.imencode(".png", depth)[1].tobytes()
    payload = struct.pack(">I", len(rb)) + rb + struct.pack(">I", len(db)) + db
    cli.sendall(S.encode_event(S.TYPE_RGBD, 1.2, payload))
    time.sleep(0.3)
    cli.close()

    got = []
    for ev in src.live_events(tick_timeout=0.2):
        if ev[0] in ("lidar", "rgbd"):
            got.append(ev[0])
        if len(got) >= 2:
            break
    assert "lidar" in got and "rgbd" in got
    assert src.imu.num_samples >= 1
    assert src.nearest_scan(1.1) is not None and src.nearest_scan(1.1).shape == (50, 2)


# --- the key test: drive the engine from the LIVE source ---------------------
def _drive_live(mode, max_events):
    """Mirror run_realtime.main() minus viz/pacing, fed by the live source."""
    import yaml
    cfg = FusionV2Config(mode=mode, dataset=_DS)
    src = S.LiveEventSource(sync_tol=cfg.sync_tolerance_s)
    R_ = R
    sc = yaml.safe_load(open(_DS / "sensor_config.yaml"))["camera"]
    K = np.array([[sc["fx"], 0, sc["cx"]], [0, sc["fy"], sc["cy"]], [0, 0, 1]])
    start_sensor = R_.SENSOR_OF_MODE[mode]
    proposer, verifier = R_._mode_defaults(mode, cfg.scan_verifier)
    lidar_attach = mode == "lidar_orb"
    build = lambda kind: R_.make_adapter(kind, cfg, src, K, lidar_attach, imu=src.imu)
    initial = "visual_vo" if start_sensor == "vo" else cfg.lidar_frontend
    sm = R_.SensorManager(build, build(initial))
    shared = build_shared_map(cfg)
    eng = R_.IngestEngine(shared, cfg, K=K, verifier=verifier, proposer=proposer)
    eng.set_active_sensor(start_sensor)
    if proposer == "dbow":
        eng.ensure_appearance()

    # start the feeder AFTER the consumer is set up
    S.replay_dataset(src, _DS, sync_tol=cfg.sync_tolerance_s, max_events=max_events)

    saw_tick = False
    for k, ev in enumerate(src.live_events(tick_timeout=1.0)):
        if ev[0] == "__tick__":
            saw_tick = True
            continue
        kfd, fe_flip, sensor_flip = sm.feed(ev)
        if sensor_flip is not None:
            eng.set_active_sensor(sensor_flip["sensor"])
            R_._autofallback(eng, sm.active)
        if kfd is not None:
            kf_id, node_pose, sig = eng.ingest(kfd)
            eng.close_loops(kf_id, node_pose, sig, np.asarray(kfd.raw_scan))
    shared.graph.optimize()
    return shared, eng


@needs_ds
def test_live_drive_lidar():
    shared, eng = _drive_live("lidar", max_events=1200)
    assert eng.stats["keyframes"] > 10
    assert shared.memory.stm_count() <= 30 and shared.memory.wm_count() <= 200
    # fused map is coherent (built from live LiDAR + injected IMU)
    poses = _anchor_poses(np.asarray(shared.graph.poses()))
    rs, rp = [], []
    for nid, x, y, th in poses:
        sig = shared.memory.get(int(nid))
        if sig is not None and sig.has_scan:
            rs.append(sig); rp.append(fusion_core.Pose2(float(x), float(y), float(th)))
    grid = fusion_core.assemble_local_grid(rs, rp, shared.grid_cfg)
    prob = np.asarray(grid.probability())
    assert (prob > 0.6).sum() > 50 and (prob < 0.4).sum() > 50


@needs_ds
def test_live_drive_orb_lidar_in_memory_images():
    # orb_lidar = VO front-end: exercises in-memory rgb/depth arrays (not paths)
    # + nearest_scan from the live rolling buffer + injected IMU.
    shared, eng = _drive_live("orb_lidar", max_events=0)
    assert eng.stats["keyframes"] > 10
    poses = _anchor_poses(np.asarray(shared.graph.poses()))
    assert len(poses) == eng.stats["keyframes"]
