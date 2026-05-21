#!/usr/bin/env python3
"""
tools/plot_slam_results.py

Unified plotting for ORB-SLAM RGB-D run outputs from run_rgbd_slam.py.

Per-run figures (written to --output/<label>/):
  trajectory_xy.png       Top-down trajectory vs GT (SE3-aligned when GT given,
                          otherwise SLAM world frame X–Z floor plan)
  ate_rpe_over_time.png   ATE + RPE time series (GT required)
  tracking_quality.png    Tracked pts / keyframes / map pts / BA MSE
  map_xy.png              Sparse map + KF graph + trajectory
  metrics_panel.png       ATE/RPE/tracking metric table

Cross-run comparison (--output/ root, both --run-a and --run-b required):
  compare_trajectory.png  Side-by-side trajectory comparison
  compare_map.png         Side-by-side sparse map comparison
  compare_trajectory_map.png
                         2x2 comparison panel with trajectory + sparse map

GT map comparison (when --gt + --dataset provided for a run):
  sparse_vs_gt_map.png    SLAM sparse map vs GT depth-projected cloud

Convenience
-----------
If --dataset-a / --dataset-b are omitted, the script tries to recover the
dataset root from each run's effective_run_config.json.

If --gt-a / --gt-b are omitted and the resolved dataset root contains
groundtruth.txt, it is used automatically (useful for TUM RGB-D runs).

Usage
-----
  python3 -m tools.plot_slam_results \\
      --run-a  visual_slam_outputs/fr1_room_cpp_ba_full \\
      --gt-a   datasets/tum/rgbd_dataset_freiburg1_room/groundtruth.txt \\
      --dataset-a datasets/tum/rgbd_dataset_freiburg1_room \\
      --run-b  visual_slam_outputs/lab_rgbd_run_2_B_loop_gba \\
      --label-a "fr1-room" --label-b "lab-run-2" \\
      --output  visual_slam_outputs/combined_plots

Single-run mode (omit --run-b):
  python3 -m tools.plot_slam_results \\
      --run-a  visual_slam_outputs/fr1_room_cpp_ba_full \\
      --gt-a   datasets/tum/rgbd_dataset_freiburg1_room/groundtruth.txt \\
      --dataset-a datasets/tum/rgbd_dataset_freiburg1_room \\
      --output  visual_slam_outputs/fr1_room_cpp_ba_full/plots
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _find_file(d: Path, pattern: str) -> Path | None:
    hits = sorted(d.glob(pattern))
    return hits[0] if hits else None


def _load_json(p: Path, default=None):
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return default


def _resolve_dataset_from_run(run_dir: Path) -> Path | None:
    cfg = _load_json(run_dir / "effective_run_config.json", {})
    dataset_path = cfg.get("dataset_path")
    if not dataset_path:
        return None
    path = Path(dataset_path).expanduser().resolve()
    return path if path.exists() else None


def _resolve_gt_from_dataset(dataset_path: Path | None) -> Path | None:
    if dataset_path is None:
        return None
    candidate = dataset_path / "groundtruth.txt"
    return candidate if candidate.exists() else None


def _load_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(v, default=float("nan")) -> float:
    try:
        val = float(v)
        return val if np.isfinite(val) else default
    except (TypeError, ValueError):
        return default


def read_trajectory(p: Path) -> np.ndarray:
    """TUM trajectory → (N, 4) array [ts, tx, ty, tz] (camera centre in world)."""
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                rows.append([float(parts[0]), float(parts[1]),
                              float(parts[2]), float(parts[3])])
    return np.asarray(rows, dtype=np.float64) if rows else np.empty((0, 4))


def read_ply_points(p: Path, max_pts: int = 300_000) -> np.ndarray:
    if not p.exists():
        return np.empty((0, 3), dtype=np.float64)
    pts = []
    in_hdr = True
    with open(p) as f:
        for line in f:
            if in_hdr:
                if line.strip() == "end_header":
                    in_hdr = False
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if len(arr) > max_pts:
        arr = arr[np.linspace(0, len(arr) - 1, max_pts, dtype=np.int64)]
    return arr


def read_keyframes_json(p: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    if not p.exists():
        return np.empty((0, 3)), {}
    data = _load_json(p, [])
    positions, lookup = [], {}
    for kf in data:
        try:
            pos = np.asarray(kf["position"], dtype=np.float64)
            kid = int(kf["kid"])
            positions.append(pos)
            lookup[kid] = pos
        except (KeyError, TypeError):
            continue
    return np.asarray(positions, dtype=np.float64) if positions else np.empty((0, 3)), lookup


def read_graph_json(p: Path) -> dict:
    return _load_json(p, {})


def filter_map_points(pts: np.ndarray, ref: np.ndarray, padding: float = 3.0) -> np.ndarray:
    if len(pts) == 0 or len(ref) == 0:
        return pts
    lo = ref.min(axis=0) - padding
    hi = ref.max(axis=0) + padding
    mask = np.all((pts >= lo) & (pts <= hi), axis=1)
    removed = len(pts) - int(mask.sum())
    if removed > 0:
        print(f"  [map filter] removed {removed} outlier pts (kept {int(mask.sum())}/{len(pts)})")
    return pts[mask]


def gba_timestamps_from_log(frame_log: list[dict]) -> list[float]:
    ts = []
    for i in range(1, len(frame_log)):
        prev = frame_log[i - 1].get("loop_global_ba_started", "0")
        curr = frame_log[i].get("loop_global_ba_started", "0")
        if curr in ("1", "True", "true") and prev not in ("1", "True", "true"):
            ts.append(float(frame_log[i]["timestamp"]))
    return ts


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _associate_timestamps(
    est_ts: np.ndarray,
    gt_ts: np.ndarray,
    max_diff: float = 0.02,
) -> list[tuple[int, int]]:
    pairs = []
    for i, et in enumerate(est_ts):
        j = int(np.argmin(np.abs(gt_ts - et)))
        if abs(gt_ts[j] - et) <= max_diff:
            pairs.append((i, j))
    return pairs


def _align_se3(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Umeyama SE(3) alignment (scale=1) from src → dst."""
    from tools.evaluate_tum_trajectory import align_se3
    R, t, _ = align_se3(src, dst)
    return R, t


def _transform(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts
    return (R @ pts.T).T + t


# ---------------------------------------------------------------------------
# GT cloud reconstruction from depth images
# ---------------------------------------------------------------------------

def _read_associations(dataset_path: Path) -> list[tuple[float, str, str]]:
    """Read associations.txt → [(ts, rgb_relpath, depth_relpath)]."""
    assoc_file = dataset_path / "associations.txt"
    if not assoc_file.exists():
        return []
    entries = []
    with open(assoc_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                entries.append((float(parts[0]), parts[1], parts[3]))
    return entries


def build_gt_cloud(
    gt_poses: list,          # list of TumPose (with .matrix = Twc 4x4)
    dataset_path: Path,
    camera_params: dict,     # from effective_run_config.json camera section
    stride: int = 8,
    max_depth_m: float = 4.0,
    max_pts: int = 800_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project depth images into world frame using GT poses.
    Returns (points Nx3, colors Nx3 uint8).
    """
    try:
        import cv2
    except ImportError:
        print("  [warn] cv2 not available — GT cloud skipped")
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

    fx = float(camera_params.get("fx", 0))
    fy = float(camera_params.get("fy", 0))
    cx = float(camera_params.get("cx", 0))
    cy = float(camera_params.get("cy", 0))
    depth_factor = float(camera_params.get("depth_factor") or
                         1.0 / float(camera_params.get("depth_map_factor", 5000)))
    if fx == 0 or fy == 0:
        print("  [warn] invalid camera params — GT cloud skipped")
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

    assoc = _read_associations(dataset_path)
    if not assoc:
        print("  [warn] no associations.txt found — GT cloud skipped")
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

    assoc_ts = np.asarray([a[0] for a in assoc], dtype=np.float64)
    gt_ts = np.asarray([p.timestamp for p in gt_poses], dtype=np.float64)

    # Subsample GT poses evenly
    step = max(1, len(gt_poses) // 200)
    selected_gt = gt_poses[::step]

    # Pixel grids (recomputed per image size, cached for common sizes)
    all_pts, all_col = [], []

    for gt_pose in selected_gt:
        Twc = gt_pose.matrix  # 4x4 world←camera
        ts = gt_pose.timestamp
        ai = int(np.argmin(np.abs(assoc_ts - ts)))
        if abs(assoc_ts[ai] - ts) > 0.1:
            continue
        _, rgb_rel, dep_rel = assoc[ai]

        dep_img = cv2.imread(str(dataset_path / dep_rel), cv2.IMREAD_UNCHANGED)
        rgb_img = cv2.imread(str(dataset_path / rgb_rel))
        if dep_img is None or rgb_img is None:
            continue

        dep_sub = dep_img[::stride, ::stride].astype(np.float32) * depth_factor
        rgb_sub = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)[::stride, ::stride]

        H, W = dep_sub.shape
        U, V = np.meshgrid(np.arange(W, dtype=np.float32),
                           np.arange(H, dtype=np.float32))
        xc_base = (U * stride - cx) / fx
        yc_base = (V * stride - cy) / fy

        valid = (dep_sub > 0.05) & (dep_sub < max_depth_m)
        if not np.any(valid):
            continue

        z = dep_sub[valid]
        pts_cam = np.stack([xc_base[valid] * z, yc_base[valid] * z, z], axis=1)
        pts_world = (Twc[:3, :3] @ pts_cam.T).T + Twc[:3, 3]

        all_pts.append(pts_world.astype(np.float32))
        all_col.append(rgb_sub[valid])

        if sum(len(p) for p in all_pts) > max_pts:
            break

    if not all_pts:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

    pts = np.vstack(all_pts).astype(np.float64)
    col = np.vstack(all_col)

    # Statistical outlier removal
    if len(pts) > 200:
        dists = np.linalg.norm(pts - np.median(pts, axis=0), axis=1)
        thr = np.median(dists) + 3.5 * np.std(dists)
        pts, col = pts[dists < thr], col[dists < thr]

    print(f"  GT cloud: {len(pts):,} pts from {len(selected_gt)} GT poses (stride={stride})")
    return pts, col


# ---------------------------------------------------------------------------
# Axis / style helpers
# ---------------------------------------------------------------------------

def _set_equal_xy(ax, arrays: list[np.ndarray], *, robust: bool = False) -> None:
    valid = [a[:, :2] for a in arrays if a is not None and len(a) >= 2]
    if not valid:
        return
    pts = np.vstack(valid)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return
    if robust and len(pts) >= 20:
        lo = np.percentile(pts, 1, axis=0)
        hi = np.percentile(pts, 99, axis=0)
    else:
        lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) * 0.5
    span = max(float(np.max(hi - lo)), 0.5)
    pad = span * 0.1
    ax.set_xlim(center[0] - span / 2 - pad, center[0] + span / 2 + pad)
    ax.set_ylim(center[1] - span / 2 - pad, center[1] + span / 2 + pad)
    ax.set_aspect("equal", adjustable="box")


def _set_equal_xz(ax, arrays: list[np.ndarray], pad_frac: float = 0.08) -> None:
    """Equal aspect for X–Z from Nx3 arrays (columns 0 and 2)."""
    pts_list = [a[:, [0, 2]] for a in arrays if a is not None and len(a) >= 2 and a.ndim == 2 and a.shape[1] >= 3]
    if not pts_list:
        return
    pts = np.vstack(pts_list)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return
    lo = np.percentile(pts, 0.5, axis=0)
    hi = np.percentile(pts, 99.5, axis=0)
    center = (lo + hi) * 0.5
    span = max(float(np.max(hi - lo)), 0.5)
    half = span / 2 + span * pad_frac
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal", adjustable="box")


def _colorline_2d(ax, x: np.ndarray, y: np.ndarray, lw: float = 1.4) -> None:
    """Draw 2-D path coloured from blue→red by sequence position."""
    n = len(x)
    if n < 2:
        return
    colors = cm.plasma(np.linspace(0.1, 0.9, n - 1))
    for i in range(n - 1):
        ax.plot(x[i:i+2], y[i:i+2], color=colors[i], lw=lw, solid_capstyle="round")


def _draw_graph_edges(
    ax, graph: dict, kf_lookup: dict[int, np.ndarray],
    mode: str = "xz",
) -> None:
    """Draw covisibility / spanning / loop edges. mode='xz' or 'xy'."""
    def _pt(pos):
        if pos is None:
            return None
        return (pos[0], pos[2]) if mode == "xz" else (pos[0], pos[1])

    for edge, color, lw, alpha in [
        ("covisibility_edges", "#aab7b8", 0.35, 0.35),
        ("spanning_tree_edges", "#626567", 0.65, 0.55),
        ("loop_edges",          "#e74c3c", 1.8,  0.9),
    ]:
        for e in graph.get(edge, []):
            a = _pt(kf_lookup.get(int(e.get("source", -1))))
            b = _pt(kf_lookup.get(int(e.get("target", -1))))
            if a and b:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, alpha=alpha, zorder=2)


def _finalize_legend(ax, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(**kwargs)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run_data(
    run_dir: Path,
    gt_path: Path | None,
    dataset_path: Path | None,
    label: str,
) -> dict:
    from tools.evaluate_tum_trajectory import read_tum_poses, associate_poses

    run_dir = run_dir.expanduser().resolve()
    if dataset_path is None:
        dataset_path = _resolve_dataset_from_run(run_dir)
    if gt_path is None:
        gt_path = _resolve_gt_from_dataset(dataset_path)
    d: dict = {
        "run_dir": run_dir,
        "label": label,
        "dataset_path": dataset_path,
        "gt_path": gt_path,
    }

    # Trajectory
    traj_file = _find_file(run_dir, "trajectory_*.txt")
    d["trajectory"] = read_trajectory(traj_file) if traj_file else np.empty((0, 4))

    # Frame log
    log_file = _find_file(run_dir, "frame_log_*.csv")
    d["frame_log"] = _load_csv(log_file) if log_file else []
    d["gba_timestamps"] = gba_timestamps_from_log(d["frame_log"])

    # Sparse map
    d["map_points"] = read_ply_points(run_dir / "map_points.ply")
    kf_pos, kf_lookup = read_keyframes_json(run_dir / "keyframes.json")
    d["kf_positions"] = kf_pos
    d["kf_lookup"] = kf_lookup
    d["graph"] = read_graph_json(run_dir / "keyframe_graph.json")

    # Run summary and config
    d["summary"] = _load_json(run_dir / "run_summary.json", {})
    d["config"] = _load_json(run_dir / "effective_run_config.json", {})
    d["dataset_type"] = d.get("summary", {}).get("dataset_type") or d.get("config", {}).get("dataset_type")

    # Pre-computed trajectory eval
    eval_dir = run_dir / "trajectory_eval"
    d["assoc_rows"] = _load_csv(eval_dir / "associated_poses.csv")
    d["metrics"] = _load_json(eval_dir / "trajectory_metrics.json", {})

    # GT alignment
    d["has_gt"] = False
    d["gt_positions"] = np.empty((0, 3))
    d["gt_poses"] = []
    d["est_aligned"] = np.empty((0, 3))
    d["gt_assoc"] = np.empty((0, 3))
    d["assoc_timestamps"] = np.empty(0)
    d["ate_errors"] = np.empty(0)
    d["alignment"] = None

    if gt_path and gt_path.exists():
        d["has_gt"] = True
        gt_poses = read_tum_poses(gt_path)
        d["gt_poses"] = gt_poses
        d["gt_positions"] = np.asarray([p.translation for p in gt_poses], dtype=np.float64)

        # Use pre-computed associations if available
        if d["assoc_rows"]:
            _f = lambda k: np.array([_safe_float(r[k]) for r in d["assoc_rows"]])
            gt_pos   = np.stack([_f("gt_tx"), _f("gt_ty"), _f("gt_tz")], axis=1)
            est_raw  = np.stack([_f("est_tx"), _f("est_ty"), _f("est_tz")], axis=1)
            est_se3  = np.stack([_f("est_se3_tx"), _f("est_se3_ty"), _f("est_se3_tz")], axis=1)
            d["gt_assoc"]          = gt_pos
            d["est_aligned"]       = est_se3
            d["assoc_timestamps"]  = np.array([_safe_float(r["timestamp_est"]) for r in d["assoc_rows"]])
            d["ate_errors"]        = np.linalg.norm(est_se3 - gt_pos, axis=1)
            if len(est_raw) >= 3:
                try:
                    R, t = _align_se3(est_raw, gt_pos)
                    d["alignment"] = (R, t)
                except Exception:
                    pass
        else:
            # On-the-fly alignment from trajectory
            traj = d["trajectory"]
            if len(traj) >= 3 and len(gt_poses) >= 3:
                try:
                    est_poses = read_tum_poses(traj_file) if traj_file else []
                    if est_poses:
                        assocs = associate_poses(gt_poses, est_poses, max_time_diff=0.05)
                        if len(assocs) >= 5:
                            gt_pts  = np.asarray([a.gt.translation       for a in assocs], dtype=np.float64)
                            est_pts = np.asarray([a.estimate.translation  for a in assocs], dtype=np.float64)
                            est_ts  = np.asarray([a.estimate.timestamp    for a in assocs], dtype=np.float64)
                            R, t = _align_se3(est_pts, gt_pts)
                            est_aligned = _transform(est_pts, R, t)
                            d["gt_assoc"]         = gt_pts
                            d["est_aligned"]      = est_aligned
                            d["assoc_timestamps"] = est_ts
                            d["ate_errors"]       = np.linalg.norm(est_aligned - gt_pts, axis=1)
                            d["alignment"]        = (R, t)
                except Exception as e:
                    print(f"  [warn] on-the-fly alignment failed: {e}")

    # Filter map points to scene bounds
    ref = d["est_aligned"] if len(d["est_aligned"]) > 0 else d["trajectory"][:, 1:4]
    if len(ref) > 0 and len(d["map_points"]) > 0:
        d["map_points"] = filter_map_points(d["map_points"], ref, padding=3.0)

    return d


# ---------------------------------------------------------------------------
# Per-run: trajectory_xy
# ---------------------------------------------------------------------------

def plot_trajectory_xy(data: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    ax.set_facecolor("white")

    traj = data["trajectory"]
    gt = data["gt_positions"]
    est = data["est_aligned"]
    label = data["label"]

    if data["has_gt"] and len(gt):
        ax.plot(gt[:, 0], gt[:, 1], color="black", lw=2.0, zorder=3, label="Ground truth")

    if len(est):
        _colorline_2d(ax, est[:, 0], est[:, 1], lw=1.5)
        ax.scatter(est[0, 0], est[0, 1], s=70, color="#27ae60", zorder=6, label="Start")
        ax.scatter(est[-1, 0], est[-1, 1], s=70, color="#c0392b", marker="X", zorder=6, label="End")
        arr_for_scale = [arr for arr in [gt, est] if len(arr)]
        _set_equal_xy(ax, arr_for_scale)
        xlabel, ylabel = "x [m]", "y [m]"
        coord_note = "SE3-aligned to GT" if data["has_gt"] else "estimated"
    elif len(traj):
        # Fall back: SLAM world frame X–Z
        _colorline_2d(ax, traj[:, 1], traj[:, 3], lw=1.5)
        ax.scatter(traj[0, 1], traj[0, 3], s=70, color="#27ae60", zorder=6, label="Start")
        ax.scatter(traj[-1, 1], traj[-1, 3], s=70, color="#c0392b", marker="X", zorder=6, label="End")
        _set_equal_xy(ax, [traj[:, [1, 3]]])
        xlabel, ylabel = "X [m]  (right)", "Z [m]  (forward)"
        coord_note = "SLAM world frame X–Z"
    else:
        ax.set_title(f"Trajectory ({label}) — no data")
        fig.tight_layout()
        out = out_dir / "trajectory_xy.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    # GBA markers
    gba_ts = data["gba_timestamps"]
    if gba_ts and len(est) > 0:
        ts_ref = data["assoc_timestamps"] if len(data["assoc_timestamps"]) else None
        pts_ref = est if len(est) else None
        if ts_ref is not None and pts_ref is not None:
            for gts in gba_ts:
                idx = int(np.argmin(np.abs(ts_ref - gts)))
                ax.scatter(pts_ref[idx, 0], pts_ref[idx, 1], s=100, marker="*",
                           color="#1abc9c", zorder=7)

    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Time (start→end)", fontsize=8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["start", "mid", "end"])

    ate_txt = ""
    m = data.get("metrics", {})
    if m.get("ate_rmse_se3_m"):
        ate_txt = f"  |  ATE RMSE = {m['ate_rmse_se3_m']:.4f} m"

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"Trajectory — {label}{ate_txt}\n({coord_note})", fontsize=9)
    ax.grid(True, alpha=0.22)
    _finalize_legend(ax, fontsize=8, loc="best")
    fig.tight_layout()
    out = out_dir / "trajectory_xy.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  saved: {out.name}")
    return out


# ---------------------------------------------------------------------------
# Per-run: ATE + RPE over time
# ---------------------------------------------------------------------------

def _compute_rpe(assoc_rows: list[dict], alignment) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps, trans_errors) for consecutive associated poses."""
    if len(assoc_rows) < 2 or alignment is None:
        return np.empty(0), np.empty(0)
    R, t = alignment
    timestamps, trans_errs = [], []
    for i in range(len(assoc_rows) - 1):
        r0, r1 = assoc_rows[i], assoc_rows[i + 1]
        try:
            gt0 = np.array([float(r0["gt_tx"]), float(r0["gt_ty"]), float(r0["gt_tz"])])
            gt1 = np.array([float(r1["gt_tx"]), float(r1["gt_ty"]), float(r1["gt_tz"])])
            e0  = np.array([float(r0["est_tx"]), float(r0["est_ty"]), float(r0["est_tz"])])
            e1  = np.array([float(r1["est_tx"]), float(r1["est_ty"]), float(r1["est_tz"])])
        except (KeyError, ValueError):
            continue
        e0_a = R @ e0 + t
        e1_a = R @ e1 + t
        trans_errs.append(float(np.linalg.norm((gt1 - gt0) - (e1_a - e0_a))))
        timestamps.append(float(r0["timestamp_est"]))
    return np.asarray(timestamps), np.asarray(trans_errs)


def plot_ate_rpe_over_time(data: dict, out_dir: Path) -> Path | None:
    ate_ts  = data["assoc_timestamps"]
    ate_err = data["ate_errors"]
    out = out_dir / "ate_rpe_over_time.png"

    if not data["has_gt"] or len(ate_ts) == 0:
        fig, ax = plt.subplots(figsize=(9, 3), dpi=150)
        ax.text(0.5, 0.5, "No ground truth available — ATE/RPE not computed",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title(f"ATE / RPE — {data['label']}")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  saved: {out.name}  (no GT)")
        return out

    rpe_ts, rpe_err = _compute_rpe(data["assoc_rows"], data["alignment"])

    m = data.get("metrics", {})
    ate_rmse = m.get("ate_rmse_se3_m", float(np.sqrt(np.mean(ate_err ** 2))) if len(ate_err) else float("nan"))
    rpe_rmse = m.get("rpe_trans_rmse_m", float(np.sqrt(np.nanmean(rpe_err ** 2))) if len(rpe_err) else float("nan"))

    n_rows = 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 6), dpi=150, sharex=True)
    fig.suptitle(f"ATE + RPE over Time — {data['label']}", fontsize=10)

    t0 = ate_ts[0]

    # ATE
    t_rel = ate_ts - t0
    axes[0].plot(t_rel, ate_err, color="#2468b2", lw=1.1, alpha=0.85, label="ATE per pose")
    axes[0].axhline(y=float(np.mean(ate_err)), color="#e74c3c", lw=1.4, ls="--",
                    label=f"Mean {np.mean(ate_err):.4f} m")
    axes[0].fill_between(t_rel, 0, ate_err, alpha=0.12, color="#2468b2")
    axes[0].set_ylabel("ATE [m]", fontsize=9)
    axes[0].set_title(f"ATE  (RMSE = {ate_rmse:.4f} m)", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    # GBA markers
    for gts in data["gba_timestamps"]:
        axes[0].axvline(gts - t0, color="#1abc9c", lw=1.3, ls=":", alpha=0.8)

    # RPE
    if len(rpe_ts) > 0:
        t_rpe = rpe_ts - t0
        axes[1].plot(t_rpe, rpe_err, color="#8e44ad", lw=1.0, alpha=0.85, label="RPE translation")
        axes[1].axhline(y=float(np.nanmean(rpe_err)), color="#e74c3c", lw=1.4, ls="--",
                        label=f"Mean {np.nanmean(rpe_err):.4f} m")
        axes[1].set_title(f"RPE translation  (RMSE = {rpe_rmse:.4f} m)", fontsize=9)
        axes[1].legend(fontsize=8)
        for gts in data["gba_timestamps"]:
            axes[1].axvline(gts - t0, color="#1abc9c", lw=1.3, ls=":", alpha=0.8)
    else:
        axes[1].text(0.5, 0.5, "RPE requires pre-computed associations",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("RPE translation", fontsize=9)

    axes[1].set_xlabel("Elapsed time [s]", fontsize=9)
    axes[1].set_ylabel("RPE trans [m]", fontsize=9)
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out.name}")
    return out


# ---------------------------------------------------------------------------
# Per-run: tracking quality
# ---------------------------------------------------------------------------

def plot_tracking_quality(data: dict, out_dir: Path) -> Path:
    rows = data["frame_log"]
    out = out_dir / "tracking_quality.png"

    if not rows:
        fig, ax = plt.subplots(figsize=(9, 3), dpi=150)
        ax.set_title(f"Tracking Quality — {data['label']}  (no frame log)")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    indices  = [_safe_float(r.get("i", 0)) for r in rows]
    tracked  = [_safe_float(r.get("last_tracked", 0), 0) for r in rows]
    kf_cnt   = [_safe_float(r.get("keyframes", 0), 0) for r in rows]
    map_pts  = [_safe_float(r.get("points", 0), 0) for r in rows]
    states   = [r.get("state", "OK") for r in rows]
    ba_mse   = [_safe_float(r.get("last_ba_mse", float("nan"))) for r in rows]

    t0 = float(rows[0]["timestamp"])
    t_sec = [float(r["timestamp"]) - t0 for r in rows]
    gba_sec = [ts - t0 for ts in data["gba_timestamps"]]

    lost_idx     = [t for t, s in zip(t_sec, states) if s == "LOST"]
    lost_tracked = [v for v, s in zip(tracked, states) if s == "LOST"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), dpi=150, sharex=True)
    fig.suptitle(f"Tracking Quality over Run — {data['label']}", fontsize=10)

    kws = dict(lw=0.85, alpha=0.88)
    gba_kw = dict(color="#1abc9c", lw=1.3, ls="--", alpha=0.8)

    axes[0].plot(t_sec, tracked, color="#2468b2", **kws, label="Tracked map pts")
    if lost_idx:
        axes[0].scatter(lost_idx, lost_tracked, color="#e74c3c", s=7, zorder=4, label="LOST")
    for ts in gba_sec:
        axes[0].axvline(ts, **gba_kw)
    if gba_sec:
        axes[0].axvline(gba_sec[0], label=f"Global BA ({len(gba_sec)}×)", **gba_kw)
    axes[0].set_ylabel("Tracked pts", fontsize=8)
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.22)

    axes[1].plot(t_sec, kf_cnt, color="#1f9d55", **kws)
    for ts in gba_sec:
        axes[1].axvline(ts, **gba_kw)
    axes[1].set_ylabel("Keyframes", fontsize=8)
    axes[1].grid(True, alpha=0.22)

    axes[2].plot(t_sec, map_pts, color="#8e44ad", **kws)
    for ts in gba_sec:
        axes[2].axvline(ts, **gba_kw)
    axes[2].set_ylabel("Map points", fontsize=8)
    axes[2].grid(True, alpha=0.22)

    valid_mse = [v for v in ba_mse if np.isfinite(v)]
    p99 = np.percentile(valid_mse, 99) if valid_mse else 10.0
    axes[3].plot(t_sec, ba_mse, color="#e74c3c", lw=0.7, alpha=0.85)
    axes[3].set_ylim(0, p99 * 1.15)
    for ts in gba_sec:
        axes[3].axvline(ts, **gba_kw)
    axes[3].set_ylabel("BA MSE\n(pose opt)", fontsize=8)
    axes[3].set_xlabel("Elapsed time [s]", fontsize=9)
    axes[3].grid(True, alpha=0.22)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out.name}")
    return out


# ---------------------------------------------------------------------------
# Per-run: map_xy
# ---------------------------------------------------------------------------

def plot_map_xy(data: dict, out_dir: Path) -> Path:
    out = out_dir / "map_xy.png"
    map_pts  = data["map_points"]
    kf_pos   = data["kf_positions"]
    kf_look  = data["kf_lookup"]
    graph    = data["graph"]
    traj     = data["trajectory"]
    label    = data["label"]
    alignment = data["alignment"]
    est      = data["est_aligned"]
    gt       = data["gt_positions"]

    use_gt_frame = data["has_gt"] and len(est) > 0 and alignment is not None

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.set_facecolor("white")

    if use_gt_frame:
        # Everything in GT world frame (XY)
        R, t = alignment

        def _xy(pts):
            if pts is None or len(pts) == 0:
                return np.empty((0, 2))
            a = _transform(pts, R, t)
            return a[:, :2]

        map_aligned = _xy(map_pts)
        kf_aligned  = {k: _transform(v.reshape(1, 3), R, t).reshape(3)
                       for k, v in kf_look.items()}
        kf_pos_aligned = np.asarray(list(kf_aligned.values()), dtype=np.float64) if kf_aligned else np.empty((0, 3))

        if len(map_aligned):
            y_h = map_aligned[:, 1]
            ax.scatter(map_aligned[:, 0], map_aligned[:, 1],
                       c=(y_h - y_h.min()) / max(y_h.ptp(), 1e-6),
                       cmap="viridis", s=0.8, alpha=0.45, linewidths=0)

        if len(gt):
            ax.plot(gt[:, 0], gt[:, 1], color="black", lw=1.6, alpha=0.8, label="GT", zorder=3)
        if len(est):
            ax.plot(est[:, 0], est[:, 1], color="#2980b9", lw=0.9, alpha=0.65, label="Estimated", zorder=4)

        # Edges
        for e_type, color, lw, alpha in [
            ("covisibility_edges", "#aab7b8", 0.3, 0.3),
            ("spanning_tree_edges", "#7f8c8d", 0.6, 0.5),
            ("loop_edges",          "#e74c3c", 1.7, 0.9),
        ]:
            for e in graph.get(e_type, []):
                a = kf_aligned.get(int(e.get("source", -1)))
                b = kf_aligned.get(int(e.get("target", -1)))
                if a is not None and b is not None:
                    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, alpha=alpha, zorder=2)

        if len(kf_pos_aligned):
            ax.scatter(kf_pos_aligned[:, 0], kf_pos_aligned[:, 1],
                       s=15, color="#e67e22", zorder=5, linewidths=0,
                       label=f"KFs ({len(kf_pos_aligned)})")

        if len(est):
            ax.scatter(est[0, 0], est[0, 1], s=80, marker="o", color="#27ae60", zorder=7)
            ax.scatter(est[-1, 0], est[-1, 1], s=80, marker="X", color="#c0392b", zorder=7)

        refs = [a for a in [map_aligned, kf_pos_aligned[:, :2] if len(kf_pos_aligned) else None, gt[:, :2], est[:, :2]] if a is not None and len(a) > 0]
        _set_equal_xy(ax, refs, robust=True)
        xlabel, ylabel = "x [m]", "y [m]"
        coord_note = "GT world frame (SE3-aligned)"
    else:
        # SLAM world frame X–Z
        if len(map_pts):
            y_h = map_pts[:, 1]
            ax.scatter(map_pts[:, 0], map_pts[:, 2],
                       c=(y_h - y_h.min()) / max(y_h.ptp(), 1e-6),
                       cmap="viridis", s=0.7, alpha=0.45, linewidths=0)

        _draw_graph_edges(ax, graph, kf_look, mode="xz")

        if len(traj):
            ax.plot(traj[:, 1], traj[:, 3], color="#2980b9", lw=0.7, alpha=0.6, label="Trajectory", zorder=3)

        if len(kf_pos):
            ax.scatter(kf_pos[:, 0], kf_pos[:, 2], s=14, color="#e67e22",
                       zorder=5, linewidths=0, label=f"KFs ({len(kf_pos)})")

        if len(traj):
            ax.scatter(traj[0, 1], traj[0, 3], s=80, marker="o", color="#27ae60", zorder=7)
            ax.scatter(traj[-1, 1], traj[-1, 3], s=80, marker="X", color="#c0392b", zorder=7)

        refs = [a for a in [map_pts, kf_pos] if len(a) > 0]
        _set_equal_xz(ax, refs if refs else ([traj[:, [1, 2, 3]]] if len(traj) else []))
        xlabel, ylabel = "X [m]  (right)", "Z [m]  (forward)"
        coord_note = "SLAM world frame X–Z"

    n_loop = len(graph.get("loop_edges", []))
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(
        f"Sparse map + KF graph — {label}  ({coord_note})\n"
        f"Map pts: {len(map_pts):,}   KFs: {len(kf_pos)}   Loop edges: {n_loop}",
        fontsize=9,
    )
    ax.grid(True, alpha=0.2, lw=0.4)
    _finalize_legend(ax, fontsize=8, loc="best", markerscale=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  saved: {out.name}")
    return out


# ---------------------------------------------------------------------------
# Per-run: metrics panel
# ---------------------------------------------------------------------------

def plot_metrics_panel(data: dict, out_dir: Path) -> Path:
    m = data.get("metrics", {})
    fl = data["frame_log"]
    s  = data.get("summary", {})

    ok_cnt   = sum(1 for r in fl if r.get("state") == "OK")
    lost_cnt = sum(1 for r in fl if r.get("state") == "LOST")
    max_kf   = max((int(r["keyframes"]) for r in fl), default=s.get("keyframes", "?"))
    max_mp   = max((int(r["points"])    for r in fl), default=s.get("map_points", "?"))
    ba_vals  = [_safe_float(r.get("last_ba_mse")) for r in fl if r.get("last_ba_mse")]
    ba_vals  = [v for v in ba_vals if np.isfinite(v)]

    rows = [
        ("Dataset",              data["label"]),
        ("Total frames",         s.get("frames_attempted", len(fl))),
        ("Tracking OK",          ok_cnt),
        ("Tracking LOST",        lost_cnt),
        ("Final keyframes",      max_kf),
        ("Final map points",     max_mp),
        ("Global BA events",     len(data["gba_timestamps"])),
        ("Loop edges",           len(data["graph"].get("loop_edges", []))),
        ("Mean BA MSE",          f"{np.mean(ba_vals):.4f}" if ba_vals else "N/A"),
        ("", ""),
        ("─── Ground truth metrics ───", ""),
        ("ATE RMSE SE(3) [m]",   f"{m['ate_rmse_se3_m']:.6f}" if "ate_rmse_se3_m" in m else "N/A"),
        ("ATE mean SE(3) [m]",   f"{m['ate_mean_se3_m']:.6f}" if "ate_mean_se3_m" in m else "N/A"),
        ("ATE median SE(3) [m]", f"{m['ate_median_se3_m']:.6f}" if "ate_median_se3_m" in m else "N/A"),
        ("ATE max SE(3) [m]",    f"{m['ate_max_se3_m']:.6f}" if "ate_max_se3_m" in m else "N/A"),
        ("RPE trans RMSE [m]",   f"{m['rpe_trans_rmse_m']:.6f}" if "rpe_trans_rmse_m" in m else "N/A"),
        ("RPE rot RMSE [deg]",   f"{m['rpe_rot_rmse_deg']:.6f}" if "rpe_rot_rmse_deg" in m else "N/A"),
        ("Assoc. poses",         m.get("num_associations", "N/A")),
    ]

    fig_h = max(4.5, 0.38 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(8, fig_h), dpi=150)
    ax.axis("off")
    tbl = ax.table(
        cellText=[[k, str(v)] for k, v in rows],
        colLabels=["Metric", "Value"],
        loc="center", cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.3, 1.45)
    ax.set_title(f"Metrics — {data['label']}", fontsize=11, pad=12)
    fig.tight_layout()
    out = out_dir / "metrics_panel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out.name}")

    # Also write markdown
    md = ["# Metrics — " + data["label"], "", "| Metric | Value |", "|---|---:|"]
    for k, v in rows:
        md.append(f"| {k} | {v} |")
    (out_dir / "metrics_table.md").write_text("\n".join(md) + "\n")

    return out


# ---------------------------------------------------------------------------
# Per-run: sparse vs GT map
# ---------------------------------------------------------------------------

def plot_sparse_vs_gt_map(data: dict, out_dir: Path) -> Path | None:
    out = out_dir / "sparse_vs_gt_map.png"
    if not data["has_gt"]:
        print(f"  skipped: sparse_vs_gt_map.png (no GT)")
        return None
    if data["dataset_path"] is None:
        print(f"  skipped: sparse_vs_gt_map.png (--dataset not provided)")
        return None

    alignment = data["alignment"]
    map_pts   = data["map_points"]
    est       = data["est_aligned"]
    gt_pos    = data["gt_positions"]
    label     = data["label"]

    # Build GT cloud from depth images using GT poses
    cam_params = data.get("config", {}).get("camera", {})
    print(f"  Building GT cloud from {data['dataset_path'].name} ...")
    gt_cloud_pts, gt_cloud_col = build_gt_cloud(
        data["gt_poses"], data["dataset_path"], cam_params
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=150)
    fig.suptitle(f"SLAM sparse map vs GT depth map — {label}", fontsize=11)

    # Left: SLAM sparse map aligned to GT frame
    ax = axes[0]
    ax.set_facecolor("white")
    if alignment and len(map_pts):
        R, t = alignment
        map_aligned = _transform(map_pts, R, t)
        ax.scatter(map_aligned[:, 0], map_aligned[:, 1],
                   s=0.8, color="#273746", alpha=0.5, linewidths=0,
                   label=f"SLAM map pts ({len(map_pts):,})")
    if len(gt_pos):
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], color="black", lw=1.4, alpha=0.7,
                label="GT trajectory", zorder=3)
    if len(est):
        ax.plot(est[:, 0], est[:, 1], color="#e74c3c", lw=1.0, alpha=0.8,
                label="Estimated (aligned)", zorder=4)

    refs = [a for a in [map_aligned if (alignment and len(map_pts)) else None, gt_pos] if a is not None and len(a) > 0]
    if refs:
        _set_equal_xy(ax, refs, robust=True)
    ax.set_xlabel("x [m]", fontsize=9)
    ax.set_ylabel("y [m]", fontsize=9)
    ax.set_title("SLAM sparse map (SE3-aligned to GT)", fontsize=9)
    ax.grid(True, alpha=0.2)
    _finalize_legend(ax, fontsize=8, markerscale=3)

    # Right: GT depth-projected cloud
    ax = axes[1]
    ax.set_facecolor("#1a1a1a")
    if len(gt_cloud_pts):
        rgba = gt_cloud_col.astype(np.float32) / 255.0
        ax.scatter(gt_cloud_pts[:, 0], gt_cloud_pts[:, 1],
                   c=rgba, s=0.5, alpha=0.6, linewidths=0)
        _set_equal_xy(ax, [gt_cloud_pts[:, :2]], robust=True)
    if len(gt_pos):
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], color="white", lw=1.2, alpha=0.8,
                label="GT trajectory")
    ax.set_xlabel("x [m]", fontsize=9, color="white")
    ax.set_ylabel("y [m]", fontsize=9, color="white")
    ax.tick_params(colors="white")
    ax.set_title(f"GT depth cloud ({len(gt_cloud_pts):,} pts)", fontsize=9, color="white")
    ax.legend(fontsize=8, facecolor="#333333", labelcolor="white")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out.name}")
    return out


# ---------------------------------------------------------------------------
# Comparison: side-by-side trajectory
# ---------------------------------------------------------------------------

def _traj_panel(ax, data: dict, title_prefix: str = "") -> None:
    """Draw a single trajectory panel (GT-aligned XY or SLAM XZ)."""
    est  = data["est_aligned"]
    gt   = data["gt_positions"]
    traj = data["trajectory"]
    gba_ts = data["gba_timestamps"]

    if data["has_gt"] and len(est):
        if len(gt):
            ax.plot(gt[:, 0], gt[:, 1], "k-", lw=1.8, alpha=0.8, label="GT")
        _colorline_2d(ax, est[:, 0], est[:, 1], lw=1.4)
        if len(est):
            ax.scatter(est[0, 0], est[0, 1], s=60, color="#27ae60", zorder=6)
            ax.scatter(est[-1, 0], est[-1, 1], s=60, color="#c0392b", marker="X", zorder=6)
        refs = [a for a in [gt, est] if len(a)]
        _set_equal_xy(ax, refs)
        xlabel, ylabel = "x [m]", "y [m]"
    elif len(traj):
        _colorline_2d(ax, traj[:, 1], traj[:, 3], lw=1.4)
        ax.scatter(traj[0, 1], traj[0, 3], s=60, color="#27ae60", zorder=6)
        ax.scatter(traj[-1, 1], traj[-1, 3], s=60, color="#c0392b", marker="X", zorder=6)
        _set_equal_xy(ax, [traj[:, [1, 3]]])
        xlabel, ylabel = "X [m]", "Z [m]"
    else:
        ax.text(0.5, 0.5, "No trajectory", ha="center", va="center", transform=ax.transAxes)
        return

    m = data.get("metrics", {})
    rmse_txt = f"  ATE={m['ate_rmse_se3_m']:.4f} m" if "ate_rmse_se3_m" in m else ""
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(f"{title_prefix}{data['label']}{rmse_txt}", fontsize=9)
    ax.grid(True, alpha=0.22)
    ax.tick_params(labelsize=7)


def plot_compare_trajectory(data_a: dict, data_b: dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=150, constrained_layout=True)
    fig.suptitle("Trajectory Comparison", fontsize=11, fontweight="bold")
    _traj_panel(axes[0], data_a)
    _traj_panel(axes[1], data_b)

    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.015, pad=0.01)
    cbar.set_label("Time (start→end)", fontsize=8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["start", "mid", "end"])

    out = out_dir / "compare_trajectory.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out.name}")
    return out


# ---------------------------------------------------------------------------
# Comparison: side-by-side sparse map
# ---------------------------------------------------------------------------

def _map_panel(ax, data: dict) -> None:
    map_pts = data["map_points"]
    kf_pos  = data["kf_positions"]
    kf_look = data["kf_lookup"]
    graph   = data["graph"]
    traj    = data["trajectory"]
    est     = data["est_aligned"]
    label   = data["label"]
    alignment = data["alignment"]

    ax.set_facecolor("white")

    use_gt = data["has_gt"] and len(est) > 0 and alignment is not None

    if use_gt:
        R, t = alignment
        if len(map_pts):
            ma = _transform(map_pts, R, t)
            ax.scatter(ma[:, 0], ma[:, 1], s=0.7, color="#273746", alpha=0.45, linewidths=0)
        if len(data["gt_positions"]):
            ax.plot(data["gt_positions"][:, 0], data["gt_positions"][:, 1],
                    "k-", lw=1.3, alpha=0.7, label="GT")
        if len(est):
            ax.plot(est[:, 0], est[:, 1], color="#2980b9", lw=0.8, alpha=0.65, label="Est.")
            ax.scatter(est[0, 0], est[0, 1], s=60, marker="o", color="#27ae60", zorder=6)
            ax.scatter(est[-1, 0], est[-1, 1], s=60, marker="X", color="#c0392b", zorder=6)
        refs = [a for a in [ma if len(map_pts) else None, data["gt_positions"]] if a is not None and len(a) > 0]
        if refs:
            _set_equal_xy(ax, refs, robust=True)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
    else:
        if len(map_pts):
            ax.scatter(map_pts[:, 0], map_pts[:, 2], s=0.7, color="#273746", alpha=0.45, linewidths=0)
        _draw_graph_edges(ax, graph, kf_look, mode="xz")
        if len(traj):
            ax.plot(traj[:, 1], traj[:, 3], color="#2980b9", lw=0.7, alpha=0.5)
            ax.scatter(traj[0, 1], traj[0, 3], s=60, marker="o", color="#27ae60", zorder=6)
            ax.scatter(traj[-1, 1], traj[-1, 3], s=60, marker="X", color="#c0392b", zorder=6)
        refs = [a for a in [map_pts, kf_pos] if len(a) > 0]
        _set_equal_xz(ax, refs if refs else ([traj[:, [1, 2, 3]]] if len(traj) else []))
        ax.set_xlabel("X [m]", fontsize=8)
        ax.set_ylabel("Z [m]", fontsize=8)

    n_loop = len(graph.get("loop_edges", []))
    ax.set_title(f"{label}\n{len(map_pts):,} pts  {len(kf_pos)} KFs  {n_loop} loop edges", fontsize=9)
    ax.grid(True, alpha=0.2, lw=0.4)
    ax.tick_params(labelsize=7)
    _finalize_legend(ax, fontsize=7)


def plot_compare_map(data_a: dict, data_b: dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=150)
    fig.suptitle("Sparse Map Comparison", fontsize=11, fontweight="bold")
    _map_panel(axes[0], data_a)
    _map_panel(axes[1], data_b)
    fig.tight_layout()
    out = out_dir / "compare_map.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  saved: {out.name}")
    return out


def plot_compare_trajectory_map(data_a: dict, data_b: dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150, constrained_layout=True)
    fig.suptitle("Trajectory and Sparse Map Comparison", fontsize=12, fontweight="bold")

    _traj_panel(axes[0, 0], data_a, title_prefix="Trajectory — ")
    _traj_panel(axes[0, 1], data_b, title_prefix="Trajectory — ")
    _map_panel(axes[1, 0], data_a)
    _map_panel(axes[1, 1], data_b)

    out = out_dir / "compare_trajectory_map.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  saved: {out.name}")
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_run_plots(data: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for name, fn in [
        ("trajectory_xy.png",       lambda: plot_trajectory_xy(data, out_dir)),
        ("ate_rpe_over_time.png",    lambda: plot_ate_rpe_over_time(data, out_dir)),
        ("tracking_quality.png",     lambda: plot_tracking_quality(data, out_dir)),
        ("map_xy.png",               lambda: plot_map_xy(data, out_dir)),
        ("metrics_panel.png",        lambda: plot_metrics_panel(data, out_dir)),
        ("sparse_vs_gt_map.png",     lambda: plot_sparse_vs_gt_map(data, out_dir)),
    ]:
        try:
            p = fn()
            if p is not None:
                generated.append(p)
        except Exception as exc:
            print(f"  [err] {name}: {exc}")
    return generated


def generate_all(
    run_a: Path,
    gt_a: Path | None,
    dataset_a: Path | None,
    label_a: str,
    run_b: Path | None,
    gt_b: Path | None,
    dataset_b: Path | None,
    label_b: str,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)

    single = run_b is None

    # Per-run A
    print(f"\n{'='*60}")
    print(f"Loading run A: {run_a.name}  [{label_a}]")
    print(f"{'='*60}")
    data_a = load_run_data(run_a, gt_a, dataset_a, label_a)
    out_a = output if single else output / "a"
    print(f"\nGenerating per-run plots → {out_a}")
    gen_a = generate_run_plots(data_a, out_a)
    print(f"Generated {len(gen_a)} figures in {out_a}")

    if single:
        return

    # Per-run B
    print(f"\n{'='*60}")
    print(f"Loading run B: {run_b.name}  [{label_b}]")
    print(f"{'='*60}")
    data_b = load_run_data(run_b, gt_b, dataset_b, label_b)
    out_b = output / "b"
    print(f"\nGenerating per-run plots → {out_b}")
    gen_b = generate_run_plots(data_b, out_b)
    print(f"Generated {len(gen_b)} figures in {out_b}")

    # Comparison
    print(f"\n{'='*60}")
    print(f"Generating comparison plots → {output}")
    print(f"{'='*60}")
    comp = []
    for name, fn in [
        ("compare_trajectory.png", lambda: plot_compare_trajectory(data_a, data_b, output)),
        ("compare_map.png",        lambda: plot_compare_map(data_a, data_b, output)),
        ("compare_trajectory_map.png", lambda: plot_compare_trajectory_map(data_a, data_b, output)),
    ]:
        try:
            comp.append(fn())
        except Exception as exc:
            print(f"  [err] {name}: {exc}")
    print(f"Generated {len(comp)} comparison figures in {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--run-a", required=True, type=Path,
                        help="Run A output directory (from run_rgbd_slam.py)")
    parser.add_argument("--gt-a",  type=Path, default=None,
                        help="Ground-truth trajectory for run A (TUM format)")
    parser.add_argument("--dataset-a", type=Path, default=None,
                        help="Raw dataset root for run A (needed for GT map)")
    parser.add_argument("--label-a", default=None,
                        help="Short label for run A (default: directory name)")

    parser.add_argument("--run-b", type=Path, default=None,
                        help="Run B output directory (optional — enables comparison plots)")
    parser.add_argument("--gt-b",  type=Path, default=None,
                        help="Ground-truth trajectory for run B (TUM format)")
    parser.add_argument("--dataset-b", type=Path, default=None,
                        help="Raw dataset root for run B (needed for GT map)")
    parser.add_argument("--label-b", default=None,
                        help="Short label for run B (default: directory name)")

    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory for all figures")

    args = parser.parse_args(argv)

    run_a    = args.run_a.expanduser().resolve()
    run_b    = args.run_b.expanduser().resolve() if args.run_b else None
    gt_a     = args.gt_a.expanduser().resolve()  if args.gt_a  else None
    gt_b     = args.gt_b.expanduser().resolve()  if args.gt_b  else None
    ds_a     = args.dataset_a.expanduser().resolve() if args.dataset_a else None
    ds_b     = args.dataset_b.expanduser().resolve() if args.dataset_b else None
    label_a  = args.label_a or run_a.name
    label_b  = args.label_b or (run_b.name if run_b else "")
    output   = args.output.expanduser().resolve()

    generate_all(run_a, gt_a, ds_a, label_a, run_b, gt_b, ds_b, label_b, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
