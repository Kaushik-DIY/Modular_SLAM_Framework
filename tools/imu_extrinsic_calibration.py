"""Camera/IMU + LiDAR/IMU extrinsic calibration probe (offline, no new recording).

Recovers the IMU mount tilt and the camera<->IMU rotation from data we already
have, and quantifies the yaw error each pipeline inherits from the mount tilt.

Three independent estimators of "gravity-up in the IMU frame":
  1. accelerometer  — mean specific force in low-motion windows (UNUSABLE on
     lab_hybrid: the IMU reports gravity-compensated linear accel, |a|~0).
  2. AHRS quaternion — world +Z expressed in the IMU frame (the gravity reference
     here); gives the IMU body-Z tilt directly.
  3. hand-eye AX=XB — camera<->IMU rotation from matched VO and IMU rotation
     increments; the principled replacement for the VO's PCA up-axis proxy.

Run:
  .venv/bin/python tools/imu_extrinsic_calibration.py --dataset datasets/lab_hybrid
  .venv/bin/python tools/imu_extrinsic_calibration.py --dataset datasets/lab_hybrid_small --max-frames 0
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carto.local_slam.imu_extrapolation import quaternion_to_yaw
from slam_core.dataio.imu_csv import read_imu_csv


def quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def log_so3(R):
    """Rotation matrix -> rotation vector (axis * angle)."""
    c = (np.trace(R) - 1.0) * 0.5
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-8:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2.0 * math.sin(th)) * th


def procrustes(M):
    """Nearest rotation to the linear map M (solves min sum|b_i - X a_i| with
    M = sum b_i a_i^T)."""
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1, 1, d]) @ Vt


# ---------------------------------------------------------------------------
def load_imu(path):
    rows = read_imu_csv(path)
    t = np.array([float(r["timestamp"]) for r in rows])
    q = np.array([[float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])]
                  for r in rows])
    w = np.array([[float(r["wx"]), float(r["wy"]), float(r["wz"])] for r in rows])
    a = np.array([[float(r["ax"]), float(r["ay"]), float(r["az"])] for r in rows])
    return t, q, w, a


def gravity_up_in_imu(q):
    """world +Z expressed in the IMU frame, averaged (AHRS gravity reference)."""
    ups = np.array([quat_to_R(*qq).T @ np.array([0, 0, 1.0]) for qq in q])
    u = ups.mean(0)
    return u / np.linalg.norm(u), ups.std(0)


def analyse_tilt(t, q, w, a):
    print("=" * 64)
    print("  IMU MOUNT TILT  (gravity reference)")
    print("=" * 64)
    amag = np.linalg.norm(a, axis=1)
    n_grav = int((np.abs(amag - 9.80665) < 0.5).sum())
    print(f"  accel: {n_grav}/{len(a)} samples near |a|=g  -> "
          f"{'usable' if n_grav > 50 else 'UNUSABLE (gravity-compensated accel)'}")

    up, ups_std = gravity_up_in_imu(q)
    tilt = math.degrees(math.acos(min(1.0, abs(up @ np.array([0, 0, 1.0])))))
    print(f"  [AHRS] gravity-up in IMU frame = {up.round(4).tolist()}  "
          f"(stability std {np.linalg.norm(ups_std):.4f})")
    print(f"  [AHRS] IMU body +Z tilt from gravity = {tilt:.2f} deg")

    # LiDAR channel 1: gyro_z as yaw-rate vs true (gyro . up)
    yr_true = w @ up
    m = np.abs(yr_true) > 0.05
    scale = float(np.polyfit(yr_true[m], w[m, 2], 1)[0]) if m.sum() > 10 else float("nan")
    print(f"\n  [LiDAR wz]   gyro_z = {scale:.5f} x true_yaw_rate  "
          f"({abs(1 - scale) * 100:.2f}% -> {abs(1 - scale) * 90:.3f} deg / 90 deg turn)")

    # LiDAR channel 2: quaternion_to_yaw (body-Z) vs gravity-aligned yaw, deltas
    yaw_b = np.unwrap(np.array([quaternion_to_yaw(*qq) for qq in q]))

    def grav_yaw(qq):
        f = quat_to_R(*qq)[:, 0]
        fh = f - (f @ np.array([0, 0, 1.0])) * np.array([0, 0, 1.0])
        return math.atan2(fh[1], fh[0])
    yaw_g = np.unwrap(np.array([grav_yaw(qq) for qq in q]))
    derr = np.degrees(np.diff(yaw_b) - np.diff(yaw_g))
    cum = math.degrees((yaw_b[-1] - yaw_b[0]) - (yaw_g[-1] - yaw_g[0]))
    print(f"  [LiDAR yaw]  body-Z vs gravity yaw: per-step std {derr.std():.4f} deg, "
          f"cumulative over run {cum:+.3f} deg")
    print("  => LiDAR IMU pose-prior is already gravity-synced (planar motion; "
          "no roll/pitch to cross-couple).")
    return up


# ---------------------------------------------------------------------------
def run_vo(dataset, max_frames):
    """Collect (t, world_R_cam, ok) per RGB-D frame from the native VO."""
    import cv2

    from slam_core.fusion2.dataset import LabHybridStream
    from slam_core.fusion2.vo_orb_frontend import NativeOrbFrontend
    import fusion_core as fc

    stream = LabHybridStream(dataset, 0.05)
    fe = NativeOrbFrontend(dataset, imu_path=str(Path(dataset) / "imu.csv"),
                           depth_max=4.0, imu_dropout=True)
    out = []
    n = 0
    for t, rgb_p, depth_p, _ in stream.rgbd_stream(max_frames):
        rgb = cv2.imread(str(rgb_p), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_p), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            continue
        Twc, state, _ = fe.track(rgb, depth, t)
        ok = state == fc.VoState.OK
        out.append((float(t), np.asarray(Twc)[:3, :3].copy(), ok))
        n += 1
    return out


def hand_eye(vo, t_imu, q_imu, up_imu, min_deg=0.4):
    """Solve X = imu_R_cam from matched camera & IMU rotation increments.

    world_R_cam = world_R_imu . X  =>  A_i = X^T B_i X  =>  beta_i = X alpha_i
    (alpha = log camera increment, beta = log IMU increment)."""
    # IMU orientation interpolation by nearest sample (30 Hz, dense enough)
    def Rimu_at(t):
        i = int(np.searchsorted(t_imu, t))
        i = max(0, min(len(t_imu) - 1, i))
        return quat_to_R(*q_imu[i])

    A, B = [], []
    used = 0
    for (t0, Rc0, ok0), (t1, Rc1, ok1) in zip(vo[:-1], vo[1:]):
        if not (ok0 and ok1):
            continue
        if not (0.0 < t1 - t0 < 0.3):
            continue
        Ai = Rc0.T @ Rc1
        Bi = Rimu_at(t0).T @ Rimu_at(t1)
        a, b = log_so3(Ai), log_so3(Bi)
        # keep pairs with real rotation on BOTH sensors (well-conditioned)
        if np.linalg.norm(a) < math.radians(min_deg) or np.linalg.norm(b) < math.radians(min_deg):
            continue
        A.append(a); B.append(b); used += 1
    if used < 30:
        print(f"\n  hand-eye: only {used} usable rotation pairs -- too few.")
        return None
    A = np.array(A); B = np.array(B)
    X = procrustes(B.T @ A)                     # beta = X alpha
    resid = np.degrees(np.linalg.norm(B - A @ X.T, axis=1))
    print("\n" + "=" * 64)
    print("  HAND-EYE  AX=XB  (camera <-> IMU rotation)")
    print("=" * 64)
    print(f"  usable rotation pairs: {used}")
    print(f"  X = imu_R_cam =\n{np.array2string(X, precision=4, prefix='      ')}")
    print(f"  residual |beta - X alpha|: median {np.median(resid):.3f} deg, "
          f"mean {resid.mean():.3f} deg")
    rv = log_so3(X)
    print(f"  camera<->IMU rotation: angle {math.degrees(np.linalg.norm(rv)):.2f} deg "
          f"about axis {(rv / (np.linalg.norm(rv) + 1e-12)).round(3).tolist()}")
    return X


def compare_to_pca(X, vo, t_imu, q_imu, up_imu):
    """Predict the VO's PCA up-axis (gravity in the camera-WORLD frame) from the
    hand-eye X + IMU, and compare to the current proxy [0.140,-0.990,0.012]."""
    # world_R_cam(t0) = world_R_imu(t0) . X  ->  realworld_up in cam-world frame
    # = world_R_cam(t0)^T @ [0,0,1] = X^T @ (world_R_imu(t0)^T @ [0,0,1]) = X^T @ up_imu
    pred_up = X.T @ up_imu
    pred_up = pred_up / np.linalg.norm(pred_up)
    # sign: the VO up proxy points "down-image" (~ -Y); align signs for comparison
    if pred_up[1] > 0:
        pred_up = -pred_up
    pca = np.array([0.140, -0.990, 0.012])
    ang = math.degrees(math.acos(min(1.0, abs(pred_up @ pca))))
    print("\n" + "=" * 64)
    print("  VO up-axis: hand-eye prediction vs current PCA proxy")
    print("=" * 64)
    print(f"  hand-eye predicted gravity-up in cam-world : {pred_up.round(4).tolist()}")
    print(f"  current PCA proxy (vo_orb_frontend._up)     : {pca.round(4).tolist()}")
    print(f"  angle between them: {ang:.2f} deg")
    naive = math.degrees(math.acos(min(1.0, abs(pred_up @ np.array([0, -1.0, 0])))))
    print(f"  (camera mount tilt from naive [0,-1,0]: {naive:.2f} deg)")


def time_sync(vo, t_imu, w_imu, up_imu, X, max_lag_s=0.5):
    """Estimate the camera<->IMU time offset by cross-correlating the two true
    yaw-rate signals (rotation about gravity-up). ~0 offset => the IMU shares the
    dataset clock (so the LiDAR, on the same clock, is time-synced too)."""
    up_cam = X.T @ up_imu                     # gravity-up in the camera body frame
    up_cam /= np.linalg.norm(up_cam)
    tv, yv = [], []
    for (t0, Rc0, ok0), (t1, Rc1, ok1) in zip(vo[:-1], vo[1:]):
        if ok0 and ok1 and 0.0 < t1 - t0 < 0.2:
            yaw = log_so3(Rc0.T @ Rc1) @ up_cam    # signed yaw about gravity-up
            tv.append(0.5 * (t0 + t1))
            yv.append(yaw / (t1 - t0))
    tv, yv = np.array(tv), np.array(yv)
    if len(tv) < 50:
        print("\n  time-sync: too few VO frames.")
        return
    dt = 0.05
    t0 = max(tv[0], t_imu[0]); t1 = min(tv[-1], t_imu[-1])
    grid = np.arange(t0, t1, dt)
    yv_g = np.interp(grid, tv, yv); yv_g -= yv_g.mean()
    yi_g = np.interp(grid, t_imu, w_imu @ up_imu); yi_g -= yi_g.mean()
    lags = np.arange(-int(max_lag_s / dt), int(max_lag_s / dt) + 1)
    cc = np.array([np.corrcoef(np.roll(yv_g, int(L)), yi_g)[0, 1] for L in lags])
    best = float(lags[int(np.argmax(cc))] * dt)
    synced = abs(best) <= 0.05 and cc.max() > 0.5
    print("\n" + "=" * 64)
    print("  TIME SYNC  (camera/IMU yaw-rate cross-correlation)")
    print("=" * 64)
    print(f"  peak correlation {cc.max():.3f} at offset {best*1000:+.0f} ms")
    print(f"  => {'SYNCED' if synced else 'CHECK'} (|offset| {abs(best)*1000:.0f} ms; "
          f"threshold 50 ms). LiDAR is on the same dataset clock as the IMU.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("datasets/lab_hybrid"))
    ap.add_argument("--max-frames", type=int, default=0,
                    help="VO frame cap for the hand-eye (0 = all).")
    ap.add_argument("--no-vo", action="store_true",
                    help="Skip the VO hand-eye (tilt analysis only, fast).")
    args = ap.parse_args()

    t_imu, q_imu, w_imu, a_imu = load_imu(str(args.dataset / "imu.csv"))
    print(f"IMU: {len(t_imu)} rows, {t_imu[-1]-t_imu[0]:.1f}s @ "
          f"{len(t_imu)/(t_imu[-1]-t_imu[0]):.1f} Hz\n")
    up_imu = analyse_tilt(t_imu, q_imu, w_imu, a_imu)

    if args.no_vo:
        return
    print("\n(running VO to collect camera rotations for the hand-eye...)")
    vo = run_vo(args.dataset, args.max_frames)
    n_ok = sum(1 for _, _, ok in vo)
    print(f"VO frames: {len(vo)}  ({n_ok} OK / {len(vo)-n_ok} non-OK)")
    X = hand_eye(vo, t_imu, q_imu, up_imu)
    if X is not None:
        compare_to_pca(X, vo, t_imu, q_imu, up_imu)
        time_sync(vo, t_imu, w_imu, up_imu, X)


if __name__ == "__main__":
    main()
