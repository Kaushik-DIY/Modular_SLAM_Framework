"""
Mode C full-run evaluation harness.

Runs the visual-main + LiDAR-ICP-verifier pipeline end-to-end on a TUM sequence
and reports: loop proposal/verification counts, ATE vs ground truth (3D
front-end VO + 2D before/after optimization), per-module timing for real-time
assessment, and memory-tier behaviour. Emits trajectory.tum, the occupancy grid
(final rebuilt map), and comparison plots.

Usage:
    .venv/bin/python tools/eval_mode_c.py \
        --dataset datasets/tum/rgbd_dataset_freiburg1_room --max-frames 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slam_core.fusion.config import FusionConfig, Mode
from slam_core.fusion.dataset import FusionDataset
from slam_core.fusion.frontends import OrbRgbdVoBackend, BruteForceOrbDetector
from slam_core.fusion.lidar_synth import synthesize_2d_scan
from slam_core.fusion.signature import CAMERA_GROUND_TRANSFORM
from slam_core.fusion.runner import run_mode_c
from slam_core.fusion.map_output import emit_run_outputs, assemble_occupancy_grid, save_occupancy_png


class _RawFrame:
    """Frame carrying RAW uint16 depth (for ORB-SLAM) + a metres-derived scan."""
    def __init__(self, rgb, depth_raw, rgb_t, scan):
        self.rgb, self.depth, self.rgb_t, self.scan = rgb, depth_raw, rgb_t, scan


def build_real_frames(dataset: FusionDataset, max_frames):
    n = len(dataset.frames) if max_frames is None else min(max_frames, len(dataset.frames))
    out = []
    for i in range(n):
        fr = dataset.frames[i]
        rgb = cv2.imread(str(fr.rgb_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(fr.depth_path), cv2.IMREAD_UNCHANGED)
        depth_m = depth_raw.astype(np.float64) * dataset.depth_factor
        scan = synthesize_2d_scan(depth_m, dataset.K, num_beams=360, noise_sigma=0.0)
        out.append(_RawFrame(rgb, depth_raw, float(fr.timestamp), scan))
    return out


# --------------------------------------------------------------------------
# Alignment + metrics
# --------------------------------------------------------------------------

def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity transform aligning src onto dst."""
    n, dim = src.shape
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    s = 1.0
    if with_scale:
        var = (sc ** 2).sum() / n
        s = float((D * np.diag(S)).sum() / var) if var > 0 else 1.0
    t = mu_d - s * R @ mu_s
    aligned = (s * (R @ src.T).T) + t
    return aligned, R, t, s


def ate(aligned: np.ndarray, dst: np.ndarray) -> dict:
    e = np.linalg.norm(aligned - dst, axis=1)
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mean": float(e.mean()),
            "median": float(np.median(e)), "max": float(e.max())}


def pca_to_2d(pts3d: np.ndarray) -> np.ndarray:
    c = pts3d - pts3d.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    return c @ Vt[:2].T


def load_gt(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        rows.append([float(p[0]), float(p[1]), float(p[2]), float(p[3])])
    return np.array(rows)  # (N, 4): t, x, y, z


def associate(kf_times, gt, tol=0.02):
    gt_t = gt[:, 0]
    idx, keep = [], []
    for i, t in enumerate(kf_times):
        j = int(np.argmin(np.abs(gt_t - t)))
        if abs(gt_t[j] - t) <= tol:
            idx.append(j)
            keep.append(i)
    return np.array(keep, dtype=int), np.array(idx, dtype=int)


# --------------------------------------------------------------------------
# Run + report
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_room")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--optimize-every", type=int, default=20)
    ap.add_argument("--keyframe-every", type=int, default=5)
    ap.add_argument("--n-features", type=int, default=500)
    ap.add_argument("--real", action="store_true",
                    help="use the real ORB-SLAM front-end + DBoW detector")
    ap.add_argument("--out", default="fusion_outputs/eval_mode_c")
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    dataset = FusionDataset(ds_path, num_beams=360, noise_sigma=0.0)
    max_frames = None if args.max_frames in (0, -1) else args.max_frames

    cfg = FusionConfig(mode=Mode.VLMAIN, dataset_path=str(ds_path), output_dir=args.out,
                       optimize_every_n_keyframes=args.optimize_every)

    if args.real:
        from slam_core.fusion.orbslam_frontend import OrbSlamFrontendBackend
        print("[eval] building real ORB-SLAM front-end (this is slow, ~1 fps)...")
        frames = build_real_frames(dataset, max_frames)
        backend = OrbSlamFrontendBackend(dataset.camera)
        detector = backend.make_loop_detector()
        t_wall = time.perf_counter()
        result = run_mode_c(cfg, frames, backend, world_transform=CAMERA_GROUND_TRANSFORM,
                            loop_detector=detector, min_index_separation=0)
        wall = time.perf_counter() - t_wall
        front_label = "ORB-SLAM (real, local-BA)"
    else:
        frames = list(dataset.iter_frames(max_frames=max_frames))
        backend = OrbRgbdVoBackend(dataset.K, n_features=args.n_features,
                                   keyframe_every=args.keyframe_every)
        detector = BruteForceOrbDetector(min_votes=80, top_k=3)
        t_wall = time.perf_counter()
        result = run_mode_c(cfg, frames, backend, world_transform=CAMERA_GROUND_TRANSFORM,
                            loop_detector=detector, min_index_separation=10)
        wall = time.perf_counter() - t_wall
        front_label = "ORB-VO stand-in"

    ids = result.keyframe_ids
    kf_times = np.array([result.trajectory[i][0] for i in range(len(ids))])

    # --- GT comparison ---
    gt = load_gt(ds_path / "groundtruth.txt")
    keep, gidx = associate(kf_times, gt)
    est3d = np.array([result.frontend_poses_3d[ids[i]][:3, 3] for i in keep])
    gt3d = gt[gidx, 1:4]
    a3d, *_ = umeyama(est3d, gt3d, with_scale=True)
    ate3d = ate(a3d, gt3d)

    # 2D (planar) ATE before vs after optimization, GT projected via PCA
    gt2d = pca_to_2d(gt3d)
    fe2d = np.array([[result.frontend_poses[ids[i]].x, result.frontend_poses[ids[i]].y]
                     for i in keep])
    op2d = np.array([[result.optimized_poses[ids[i]].x, result.optimized_poses[ids[i]].y]
                     for i in keep])
    afe, *_ = umeyama(fe2d, gt2d, with_scale=True)
    aop, *_ = umeyama(op2d, gt2d, with_scale=True)
    ate2d_fe, ate2d_op = ate(afe, gt2d), ate(aop, gt2d)

    # --- emit artefacts ---
    paths = emit_run_outputs(result, args.out, run_id="run")
    out_dir = Path(paths["run_dir"])

    # before/after occupancy maps (proves rebuild on loop closure)
    def grid_png(poses, name):
        pairs = [(poses[i], result.keyframe_scans.get(i)) for i in ids if poses.get(i) is not None]
        save_occupancy_png(assemble_occupancy_grid(pairs, resolution=0.05), out_dir / name)

    grid_png(result.frontend_poses, "map_before_loops.png")
    grid_png(result.optimized_poses, "map_after_loops.png")

    # trajectory comparison plot
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].plot(gt2d[:, 0], gt2d[:, 1], "k-", lw=2, label="GT (PCA-2D)")
    ax[0].plot(afe[:, 0], afe[:, 1], "r--", lw=1, label="front-end (pre-opt)")
    ax[0].plot(aop[:, 0], aop[:, 1], "b-", lw=1, label="optimized (post-loop)")
    ax[0].set_title("Trajectory vs GT (aligned, top-down)")
    ax[0].axis("equal"); ax[0].legend(); ax[0].grid(alpha=0.3)
    og = assemble_occupancy_grid(
        [(result.optimized_poses[i], result.keyframe_scans.get(i)) for i in ids
         if result.optimized_poses.get(i) is not None], resolution=0.05)
    ax[1].imshow(np.flipud(1 - og.prob), cmap="gray", origin="upper")
    ax[1].set_title("Final rebuilt occupancy map")
    fig.tight_layout()
    fig.savefig(out_dir / "evaluation.png", dpi=110)
    plt.close(fig)

    # --- tables ---
    mt = result.module_times_ms
    rt = result.runtime_summary()
    nkf = len(ids)
    print("\n" + "=" * 64)
    print(f"  MODE C EVALUATION — {ds_path.name}")
    print("=" * 64)
    print(f"  frames processed         : {len(frames)}")
    print(f"  keyframes                : {nkf}")
    print(f"  GT-associated keyframes  : {len(keep)}")
    print(f"  total wall time          : {wall:.1f} s  ({nkf / wall:.1f} kf/s)")

    print("\n  --- LOOP CLOSURE ---")
    ls = result.loop_stats
    print(f"  candidates proposed (ORB): {ls['proposed']}")
    print(f"  verified+confirmed (ICP) : {ls['verified']}")
    print(f"  rejected by ICP          : {ls['rejected']}")
    if result.loop_reject_reasons:
        for k, v in result.loop_reject_reasons.items():
            print(f"      reject[{k:14s}]  : {v}")
    vr = 100.0 * ls['verified'] / ls['proposed'] if ls['proposed'] else 0.0
    print(f"  ICP verification rate    : {vr:.1f}%")

    print("\n  --- ACCURACY vs GROUND TRUTH ---")
    print(f"  3D ATE front-end VO  rmse: {ate3d['rmse']*100:.1f} cm "
          f"(mean {ate3d['mean']*100:.1f}, med {ate3d['median']*100:.1f}, max {ate3d['max']*100:.1f})")
    print(f"  2D ATE pre-optimization  : {ate2d_fe['rmse']*100:.1f} cm")
    print(f"  2D ATE post-optimization : {ate2d_op['rmse']*100:.1f} cm")
    impr = 100.0 * (1 - ate2d_op['rmse'] / ate2d_fe['rmse']) if ate2d_fe['rmse'] else 0.0
    print(f"  loop-closure improvement : {impr:+.1f}%")

    print("\n  --- PER-MODULE TIME (total ms / % / per-kf ms) ---")
    tot = sum(mt.values()) or 1.0
    for k in ("frontend", "signature", "memory", "graph", "propose", "verify", "solve"):
        print(f"  {k:10s} : {mt[k]:8.1f} ms  {100*mt[k]/tot:5.1f}%   {mt[k]/max(1,nkf):6.2f} ms/kf")
    print(f"  {'TOTAL':10s} : {tot:8.1f} ms")
    print(f"  per-keyframe wall: mean {rt['mean_ms']:.1f} ms | median {rt['median_ms']:.1f} "
          f"ms | p95 {rt['p95_ms']:.1f} ms  -> realtime(<=100ms): "
          f"{'PASS' if rt['median_ms'] <= 100 else 'OVER'}")

    print("\n  --- MEMORY TIERS ---")
    print(f"  {result.memory_stats}  (caps: stm<={cfg.stm_size} wm<={cfg.wm_cap} ltm<={cfg.ltm_cap})")

    print("\n  --- ARTEFACTS ---")
    print(f"  {out_dir}/trajectory.tum")
    print(f"  {out_dir}/occupancy.png  (final rebuilt map)")
    print(f"  {out_dir}/map_before_loops.png  /  map_after_loops.png")
    print(f"  {out_dir}/evaluation.png  (trajectory + map)")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
