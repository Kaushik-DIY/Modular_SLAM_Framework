"""Live event source for online SLAM (ROS-fed) — Jetson deployment (V6).

Replaces the offline disk dataset (`LabHybridStream` + `merged_events`) with a
LIVE, in-memory event stream so `run_realtime.py --source ros` consumes sensor
data as it arrives and never saves/replays. The actual ROS subscriptions live in
a Python-2 node (`tools/ros/fusion_ros_bridge.py`, ROS Melodic) which forwards
sensor bundles over a Unix-domain socket; this module (py3.8 runtime) decodes
them and exposes the exact interface the fusion engine already consumes:

  * live_events()      — blocking generator of ("lidar", t, scan) /
                         ("rgbd", t, rgb_ndarray, depth_ndarray) in arrival order
                         (drop-in for the materialised merged_events(stream) list)
  * nearest_scan(t) / nearest_rgbd(t) — rolling-buffer soft sync (cross-modal payload)
  * imu (LiveImuBuffer) — file-IMU-compatible (yaw_at + growing .samples), injected
                          into the front-ends in place of imu.csv

The wire format is deliberately pickle-free (py2<->py3 safe): a 1-byte type, an
8-byte double timestamp, a 4-byte length, then raw bytes (JPEG rgb / PNG16 depth /
float32 scan / float64 imu). The same framing is produced by the bridge.
"""
from __future__ import annotations

import math
import socket
import struct
import threading
import queue
from bisect import bisect_right
from collections import deque
from typing import Optional, Tuple

import numpy as np

# ---- wire protocol (shared with tools/ros/fusion_ros_bridge.py) -------------
_HDR = struct.Struct(">cdI")   # type(1) | timestamp(double) | payload_len(uint32)
TYPE_LIDAR = b"L"
TYPE_RGBD = b"R"
TYPE_IMU = b"I"


def encode_event(etype: bytes, t: float, payload: bytes) -> bytes:
    """Pack one bridge message with the fixed binary header."""
    return _HDR.pack(etype, float(t), len(payload)) + payload


def _recv_exact(sock, n: int) -> Optional[bytes]:
    """Read exactly n bytes or return None on EOF."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


# ---------------------------------------------------------------------------
class LiveImuBuffer:
    """IMU buffer fed live, exposing both APIs the front-ends need:
      - `.samples` : growing list of (t, wz, yaw) the LiDAR extrapolator drains
                     (its loop is `while idx < len(samples) and samples[idx][0] <= t`);
      - `.yaw_at(t)`: interpolated UNWRAPPED yaw the VO dropout-prior uses
                     (mirrors ImuFallbackExtrapolator.yaw_at)."""

    def __init__(self, maxlen: int = 200000):
        self.samples = []                      # (t, wz, yaw_raw) — shared with LiDAR FE
        self._t = deque(maxlen=maxlen)
        self._yaw_unwrapped = deque(maxlen=maxlen)
        self._last_raw = None
        self._last_unw = 0.0
        self._lock = threading.Lock()

    def add(self, t: float, wz: float, yaw: float) -> None:
        with self._lock:
            self.samples.append((float(t), float(wz), float(yaw)))
            if self._last_raw is None:
                unw = float(yaw)
            else:
                # Unwrap yaw so interpolation crosses +/-pi continuously.
                unw = self._last_unw + math.atan2(math.sin(yaw - self._last_raw),
                                                  math.cos(yaw - self._last_raw))
            self._last_raw, self._last_unw = float(yaw), unw
            self._t.append(float(t)); self._yaw_unwrapped.append(unw)

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def yaw_at(self, t: float) -> Optional[float]:
        with self._lock:
            ts, ys = list(self._t), list(self._yaw_unwrapped)
        if not ts or t < ts[0] or t > ts[-1]:
            return float(ys[-1]) if ys and t >= ts[-1] else None
        i = bisect_right(ts, t)
        if i <= 0:
            return float(ys[0])
        if i >= len(ts):
            return float(ys[-1])
        t0, t1 = ts[i - 1], ts[i]
        if abs(t1 - t0) < 1e-12:
            return float(ys[i])
        a = (t - t0) / (t1 - t0)
        # Linear interpolation is valid on the unwrapped yaw track.
        return float((1.0 - a) * ys[i - 1] + a * ys[i])


# ---------------------------------------------------------------------------
class LiveEventSource:
    """Core live source (transport-agnostic). The ROS subclass feeds it via the
    socket; the dev replay feeder feeds it from a recorded dataset. Either way the
    fusion runner sees the same interface as the disk path."""

    def __init__(self, sync_tol: float = 0.05, scan_buf: int = 200,
                 rgbd_buf: int = 200, queue_max: int = 2000):
        self.sync_tol = float(sync_tol)
        self.imu = LiveImuBuffer()
        self._scan_buf = deque(maxlen=scan_buf)   # (t, scan)
        self._rgbd_buf = deque(maxlen=rgbd_buf)    # (t, rgb, depth)
        self._eventq: "queue.Queue" = queue.Queue(maxsize=queue_max)
        self._lock = threading.Lock()
        self._closed = False

    # -- producers (called by the transport / feeder) ----------------------
    def push_lidar(self, t: float, scan: np.ndarray) -> None:
        scan = np.asarray(scan, np.float64)
        with self._lock:
            self._scan_buf.append((float(t), scan))
        # Queue the event after buffering so soft sync can find it immediately.
        self._put(("lidar", float(t), scan))

    def push_rgbd(self, t: float, rgb: np.ndarray, depth: np.ndarray) -> None:
        with self._lock:
            self._rgbd_buf.append((float(t), rgb, depth))
        # Store decoded arrays in live mode; disk mode stores file paths instead.
        self._put(("rgbd", float(t), rgb, depth))

    def push_imu(self, t: float, wz: float, yaw: float) -> None:
        self.imu.add(t, wz, yaw)

    def close(self) -> None:
        self._closed = True
        try:
            # Sentinel wakes the consumer even if it is blocked on the queue.
            self._eventq.put_nowait(("__stop__", 0.0))
        except queue.Full:
            pass

    def _put(self, ev) -> None:
        try:
            self._eventq.put_nowait(ev)
        except queue.Full:
            # backpressure: drop the OLDEST event to stay real-time (online SLAM
            # must keep up; a stale frame is worse than a dropped one).
            try:
                self._eventq.get_nowait()
                self._eventq.put_nowait(ev)
            except queue.Empty:
                pass

    # -- consumer interface (matches the disk path) ------------------------
    def live_events(self, tick_timeout: float = 0.5):
        """Blocking generator of sensor events. Drains the queue until the
        `__stop__` sentinel (so no tail events are lost on close). On a stall it
        yields a ("__tick__", t) heartbeat so the runner can still process stdin
        switches / quit even when no sensor data is flowing."""
        while True:
            try:
                ev = self._eventq.get(timeout=tick_timeout)
            except queue.Empty:
                if self._closed:
                    return
                yield ("__tick__", 0.0)
                continue
            if ev[0] == "__stop__":
                return
            yield ev

    def nearest_scan(self, t: float) -> Optional[np.ndarray]:
        with self._lock:
            buf = list(self._scan_buf)
        best, best_dt = None, self.sync_tol
        for ts, scan in reversed(buf):           # recent first
            dt = abs(ts - t)
            if dt <= best_dt:
                best, best_dt = scan, dt
            elif ts < t - self.sync_tol:
                # Older buffered scans cannot beat the tolerance once ordered.
                break
        return best

    def nearest_rgbd(self, t: float):
        with self._lock:
            buf = list(self._rgbd_buf)
        best, best_dt = None, self.sync_tol
        for ts, rgb, depth in reversed(buf):
            dt = abs(ts - t)
            if dt <= best_dt:
                best, best_dt = (rgb, depth), dt
            elif ts < t - self.sync_tol:
                break
        return best


# ---------------------------------------------------------------------------
class RosEventSource(LiveEventSource):
    """LiveEventSource fed by the ROS py2 bridge over a Unix-domain socket."""

    def __init__(self, socket_path: str, **kw):
        super().__init__(**kw)
        self.socket_path = socket_path
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        import os
        try:
            os.unlink(socket_path)
        except OSError:
            pass
        self._srv.bind(socket_path)
        self._srv.listen(1)
        self._thread = threading.Thread(target=self._serve, name="ros-bridge",
                                        daemon=True)

    def start(self) -> "RosEventSource":
        self._thread.start()
        return self

    def _serve(self):
        import cv2
        conn, _ = self._srv.accept()
        try:
            while not self._closed:
                hdr = _recv_exact(conn, _HDR.size)
                if hdr is None:
                    break
                etype, t, n = _HDR.unpack(hdr)
                payload = _recv_exact(conn, n) if n else b""
                if payload is None:
                    break
                if etype == TYPE_LIDAR:
                    # Payload is packed float32 xy pairs in scanner frame.
                    self.push_lidar(t, np.frombuffer(payload, np.float32).reshape(-1, 2))
                elif etype == TYPE_RGBD:
                    # RGB-D payload stores [rgb_len][rgb_jpeg][depth_len][depth_png].
                    rlen = struct.unpack(">I", payload[:4])[0]
                    rgb_b = payload[4:4 + rlen]
                    dep_b = payload[8 + rlen:]   # skip the 4-byte depth-len marker
                    rgb = cv2.imdecode(np.frombuffer(rgb_b, np.uint8), cv2.IMREAD_COLOR)
                    depth = cv2.imdecode(np.frombuffer(dep_b, np.uint8), cv2.IMREAD_UNCHANGED)
                    if rgb is not None and depth is not None:
                        self.push_rgbd(t, rgb, depth)
                elif etype == TYPE_IMU:
                    # IMU payload stores yaw rate and absolute yaw.
                    wz, yaw = struct.unpack(">dd", payload)
                    self.push_imu(t, wz, yaw)
        finally:
            conn.close()
            self.close()


# ---------------------------------------------------------------------------
def replay_dataset(source: LiveEventSource, dataset, sync_tol: float = 0.05,
                   max_events: int = 0, realtime: bool = False):
    """Dev/test feeder: push a recorded LabHybridStream into a LiveEventSource as
    if it were arriving live (no ROS, no socket). Exercises the exact runner live
    path on x86 before the Jetson. Runs in a thread; calls source.close() at end."""
    import time

    from slam_core.dataio.imu_csv import read_imu_csv
    from carto.local_slam.imu_extrapolation import imu_rows_to_samples
    from slam_core.fusion2.Dependencies.dataset import LabHybridStream
    from pathlib import Path

    stream = LabHybridStream(Path(dataset), sync_tol)
    # pre-load the whole IMU track (a real run feeds it incrementally; for replay
    # we push it up front so yaw_at/drain always have history).
    for ts, wz, yaw in imu_rows_to_samples(read_imu_csv(str(Path(dataset) / "imu.csv"))):
        source.push_imu(ts, wz, yaw)

    # merge lidar + rgbd by timestamp (same order the live driver would see)
    from slam_core.fusion2.run_realtime import merged_events  # late import (avoids cycle)
    import cv2
    events = merged_events(stream) if max_events <= 0 else \
        (e for i, e in enumerate(merged_events(stream)) if i < max_events)

    def _run():
        t0 = None; w0 = None
        for ev in events:
            if realtime:
                # Match dataset timestamps to wall time for timing tests.
                if t0 is None:
                    t0, w0 = ev[1], time.perf_counter()
                sl = (ev[1] - t0) - (time.perf_counter() - w0)
                if sl > 0:
                    time.sleep(sl)
            if ev[0] == "lidar":
                source.push_lidar(ev[1], ev[2])
            else:
                rgb = cv2.imread(str(ev[2]), cv2.IMREAD_COLOR)
                depth = cv2.imread(str(ev[3]), cv2.IMREAD_UNCHANGED)
                if rgb is not None and depth is not None:
                    source.push_rgbd(ev[1], rgb, depth)
        source.close()

    th = threading.Thread(target=_run, name="replay-feeder", daemon=True)
    th.start()
    return th
