#!/usr/bin/env python3
"""Generate ATE/RPE evaluation plots for a single TUM RGB-D SLAM run.

Inputs (all read from --run-dir):
  trajectory_*.txt          estimated trajectory (TUM format)
  trajectory_eval/          output of evaluate_tum_trajectory.py
    associated_poses.csv
    trajectory_metrics.json
  frame_log_*.csv           per-frame tracking log
  map_points.ply            sparse map (optional)
  keyframes.json            keyframe poses (optional)
  keyframe_graph.json       covisibility/loop graph (optional)

Outputs (written to --output or <run-dir>/plots/):
  trajectory_xy.png
  trajectory_3d.png
  ate_over_time.png
  rpe_over_time.png
  tracking_quality.png
  map_xy.png
  keyframe_graph_xy.png
  metrics_panel.png
  metrics_table.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tools.evaluate_tum_trajectory import (
    align_se3,
    read_tum_poses,
    transform_positions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path, default=None):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return default


def _load_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _find_file(run_dir: Path, pattern: str) -> Path | None:
    candidates = sorted(run_dir.glob(pattern))
    return candidates[0] if candidates else None


def _safe_float(value, default=float("nan")) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _set_equal_xy(ax, arrays: list[np.ndarray], *, robust: bool = False) -> None:
    finite = [arr[:, :2] for arr in arrays if arr is not None and len(arr) >= 2]
    if not finite:
        return
    pts = np.vstack(finite)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return
    if robust and len(pts) >= 20:
        mins = np.percentile(pts, 1.0, axis=0)
        maxs = np.percentile(pts, 99.0, axis=0)
    else:
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
    center = (mins + maxs) * 0.5
    span = max(float(np.max(maxs - mins)), 1e-6)
    pad = span * 0.1
    ax.set_xlim(center[0] - span / 2 - pad, center[0] + span / 2 + pad)
    ax.set_ylim(center[1] - span / 2 - pad, center[1] + span / 2 + pad)
    ax.set_aspect("equal", adjustable="box")


def read_ply_points(path: Path, max_points: int = 200000) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 3), dtype=np.float64)
    points = []
    in_header = True
    with open(path, "r") as f:
        for line in f:
            if in_header:
                if line.strip() == "end_header":
                    in_header = False
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if max_points and len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points, dtype=np.int64)
        pts = pts[idx]
    return pts


def filter_map_points(pts: np.ndarray, ref: np.ndarray, padding: float = 3.0) -> np.ndarray:
    if len(pts) == 0 or len(ref) == 0:
        return pts
    lo = ref.min(axis=0) - padding
    hi = ref.max(axis=0) + padding
    mask = np.all((pts >= lo) & (pts <= hi), axis=1)
    removed = len(pts) - mask.sum()
    if removed > 0:
        print(f"  [map filter] removed {removed} outlier map points (kept {mask.sum()}/{len(pts)})")
    return pts[mask]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run_data(run_dir: Path, gt_path: Path | None) -> dict:
    data = {}

    # Ground truth
    data["gt_poses"] = []
    data["gt_positions"] = np.empty((0, 3), dtype=np.float64)
    if gt_path and gt_path.exists():
        try:
            data["gt_poses"] = read_tum_poses(gt_path)
            data["gt_positions"] = np.asarray(
                [p.translation for p in data["gt_poses"]], dtype=np.float64
            )
        except Exception as e:
            print(f"  [warn] could not read GT: {e}")

    # Estimated trajectory
    traj_file = _find_file(run_dir, "trajectory_*.txt")
    data["est_poses"] = []
    data["est_positions"] = np.empty((0, 3), dtype=np.float64)
    if traj_file:
        try:
            data["est_poses"] = read_tum_poses(traj_file)
            data["est_positions"] = np.asarray(
                [p.translation for p in data["est_poses"]], dtype=np.float64
            )
        except Exception as e:
            print(f"  [warn] could not read estimated trajectory: {e}")

    # Associated poses from evaluation
    eval_dir = run_dir / "trajectory_eval"
    assoc_csv = eval_dir / "associated_poses.csv"
    data["assoc_rows"] = _load_csv_rows(assoc_csv)

    if data["assoc_rows"]:
        data["assoc_gt"] = np.asarray(
            [[_safe_float(r["gt_tx"]), _safe_float(r["gt_ty"]), _safe_float(r["gt_tz"])]
             for r in data["assoc_rows"]], dtype=np.float64
        )
        data["assoc_est_raw"] = np.asarray(
            [[_safe_float(r["est_tx"]), _safe_float(r["est_ty"]), _safe_float(r["est_tz"])]
             for r in data["assoc_rows"]], dtype=np.float64
        )
        data["assoc_est_se3"] = np.asarray(
            [[_safe_float(r["est_se3_tx"]), _safe_float(r["est_se3_ty"]), _safe_float(r["est_se3_tz"])]
             for r in data["assoc_rows"]], dtype=np.float64
        )
        data["assoc_timestamps"] = np.asarray(
            [_safe_float(r["timestamp_est"]) for r in data["assoc_rows"]], dtype=np.float64
        )
        # ATE per-pose
        data["ate_errors"] = np.linalg.norm(
            data["assoc_est_se3"] - data["assoc_gt"], axis=1
        )
    else:
        data["assoc_gt"] = np.empty((0, 3), dtype=np.float64)
        data["assoc_est_raw"] = np.empty((0, 3), dtype=np.float64)
        data["assoc_est_se3"] = np.empty((0, 3), dtype=np.float64)
        data["assoc_timestamps"] = np.empty(0, dtype=np.float64)
        data["ate_errors"] = np.empty(0, dtype=np.float64)

    # Alignment transform (from associated poses raw → se3)
    data["alignment"] = None
    if len(data["assoc_est_raw"]) >= 3 and len(data["assoc_gt"]) >= 3:
        try:
            R, t, _ = align_se3(data["assoc_est_raw"], data["assoc_gt"])
            data["alignment"] = (R, t)
        except Exception:
            pass

    # Trajectory metrics JSON
    metrics_json = _load_json(eval_dir / "trajectory_metrics.json", {})
    data["metrics"] = metrics_json or {}

    # Frame log
    frame_log_file = _find_file(run_dir, "frame_log_*.csv")
    data["frame_log"] = _load_csv_rows(frame_log_file) if frame_log_file else []

    # Map points
    data["map_points"] = read_ply_points(run_dir / "map_points.ply")

    # Keyframes
    kf_data = _load_json(run_dir / "keyframes.json", [])
    data["keyframes"] = {}
    if kf_data:
        for row in kf_data:
            try:
                data["keyframes"][int(row["kid"])] = np.asarray(
                    row["position"], dtype=np.float64
                ).reshape(3)
            except Exception:
                continue

    # Keyframe graph
    data["graph"] = _load_json(run_dir / "keyframe_graph.json", {})

    return data


def _apply_alignment(pts: np.ndarray, alignment) -> np.ndarray:
    if alignment is None or len(pts) == 0:
        return pts
    R, t = alignment
    return transform_positions(pts, R, t, scale=1.0)


# ---------------------------------------------------------------------------
# RPE computation from associated poses
# ---------------------------------------------------------------------------

def compute_rpe_series(assoc_rows: list[dict], alignment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (timestamps, trans_errors, rot_errors) for consecutive pose pairs."""
    if len(assoc_rows) < 2 or alignment is None:
        return np.empty(0), np.empty(0), np.empty(0)

    R_align, t_align = alignment
    timestamps = []
    trans_errors = []
    rot_errors = []

    for i in range(len(assoc_rows) - 1):
        r0, r1 = assoc_rows[i], assoc_rows[i + 1]
        try:
            gt0 = np.array([float(r0["gt_tx"]), float(r0["gt_ty"]), float(r0["gt_tz"])])
            gt1 = np.array([float(r1["gt_tx"]), float(r1["gt_ty"]), float(r1["gt_tz"])])
            e0 = np.array([float(r0["est_tx"]), float(r0["est_ty"]), float(r0["est_tz"])])
            e1 = np.array([float(r1["est_tx"]), float(r1["est_ty"]), float(r1["est_tz"])])
        except (KeyError, ValueError):
            continue

        # align to GT frame
        e0_a = R_align @ e0 + t_align
        e1_a = R_align @ e1 + t_align

        gt_rel_t = gt1 - gt0
        est_rel_t = e1_a - e0_a
        trans_errors.append(float(np.linalg.norm(gt_rel_t - est_rel_t)))
        timestamps.append(float(r0["timestamp_est"]))
        rot_errors.append(float("nan"))  # rotation not available without full SO3 matrices

    return (
        np.asarray(timestamps, dtype=np.float64),
        np.asarray(trans_errors, dtype=np.float64),
        np.asarray(rot_errors, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Plot generators
# ---------------------------------------------------------------------------

def plot_trajectory_xy(data: dict, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    gt = data["gt_positions"]
    est_aligned = data["assoc_est_se3"]
    est_raw = data["est_positions"]

    if len(gt):
        ax.plot(gt[:, 0], gt[:, 1], color="black", lw=2.0, label="Ground truth", zorder=3)

    display_est = est_aligned if len(est_aligned) else est_raw
    if len(display_est):
        ax.plot(display_est[:, 0], display_est[:, 1],
                color="#d04a3a", lw=1.6, label="Estimated (SE3-aligned)", zorder=2, alpha=0.9)

        # start / end markers
        ax.scatter([display_est[0, 0]], [display_est[0, 1]],
                   s=60, color="#2ecc71", zorder=5, label="Start")
        ax.scatter([display_est[-1, 0]], [display_est[-1, 1]],
                   s=60, color="#e74c3c", zorder=5, label="End")

    _set_equal_xy(ax, [arr for arr in [gt, display_est] if len(arr)])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectory Comparison (top-down XY)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = output_dir / "trajectory_xy.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_trajectory_3d(data: dict, output_dir: Path) -> Path:
    fig = plt.figure(figsize=(8, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    gt = data["gt_positions"]
    est_aligned = data["assoc_est_se3"]
    est_raw = data["est_positions"]

    if len(gt):
        ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="black", lw=1.8, label="Ground truth")

    display_est = est_aligned if len(est_aligned) else est_raw
    if len(display_est):
        ax.plot(display_est[:, 0], display_est[:, 1], display_est[:, 2],
                color="#d04a3a", lw=1.4, label="Estimated (SE3-aligned)", alpha=0.9)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Trajectory Comparison 3D")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = output_dir / "trajectory_3d.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_ate_over_time(data: dict, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)

    ts = data["assoc_timestamps"]
    ate = data["ate_errors"]

    if len(ts) and len(ate):
        t_rel = ts - ts[0]
        ax.plot(t_rel, ate, color="#2468b2", lw=1.2, alpha=0.8, label="ATE per pose")
        ax.axhline(y=float(np.mean(ate)), color="#e74c3c", lw=1.5,
                   linestyle="--", label=f"Mean {np.mean(ate):.4f} m")
        ax.fill_between(t_rel, 0, ate, alpha=0.15, color="#2468b2")

        metrics = data["metrics"]
        rmse_val = metrics.get("ate_rmse_se3_m")
        if rmse_val is not None:
            title = f"ATE over Time  (RMSE={rmse_val:.4f} m)"
        else:
            title = "ATE over Time"
        ax.set_title(title)
    else:
        ax.set_title("ATE over Time (no data)")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ATE [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = output_dir / "ate_over_time.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_rpe_over_time(data: dict, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)

    ts, trans_err, _ = compute_rpe_series(data["assoc_rows"], data["alignment"])

    if len(ts):
        t_rel = ts - ts[0]
        ax.plot(t_rel, trans_err, color="#8e44ad", lw=1.2, alpha=0.8, label="RPE translation")
        ax.axhline(y=float(np.nanmean(trans_err)), color="#e74c3c", lw=1.5,
                   linestyle="--", label=f"Mean {np.nanmean(trans_err):.4f} m")

        metrics = data["metrics"]
        rpe_rmse = metrics.get("rpe_trans_rmse_m")
        if rpe_rmse is not None:
            title = f"RPE Translation over Time  (RMSE={rpe_rmse:.4f} m)"
        else:
            title = "RPE Translation over Time"
        ax.set_title(title)
    else:
        ax.set_title("RPE Translation over Time (no data)")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("RPE translation [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = output_dir / "rpe_over_time.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_tracking_quality(data: dict, output_dir: Path) -> Path:
    rows = data["frame_log"]
    if not rows:
        fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
        ax.set_title("Tracking Quality (no frame log)")
        fig.savefig(output_dir / "tracking_quality.png")
        plt.close(fig)
        return output_dir / "tracking_quality.png"

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=150, sharex=True)

    indices = [_safe_float(r.get("i", r.get("frame_index", 0))) for r in rows]
    tracked = [_safe_float(r.get("last_tracked", r.get("tracked_map_points", 0)), 0) for r in rows]
    keyframes = [_safe_float(r.get("keyframes", 0), 0) for r in rows]
    map_points = [_safe_float(r.get("points", r.get("map_points", 0)), 0) for r in rows]
    states = [r.get("state", "OK") for r in rows]

    ok_mask = [s == "OK" for s in states]
    lost_mask = [s == "LOST" for s in states]

    ax0 = axes[0]
    ax0.plot(indices, tracked, color="#2468b2", lw=1.0, alpha=0.8, label="Tracked map pts")
    lost_idx = [i for i, m in zip(indices, lost_mask) if m]
    lost_tracked = [t for t, m in zip(tracked, lost_mask) if m]
    if lost_idx:
        ax0.scatter(lost_idx, lost_tracked, color="#e74c3c", s=8, zorder=4, label="LOST")
    ax0.set_ylabel("Tracked points")
    ax0.legend(fontsize=7)
    ax0.grid(True, alpha=0.2)

    ax1 = axes[1]
    ax1.plot(indices, keyframes, color="#1f9d55", lw=1.2, label="Keyframes")
    ax1.set_ylabel("Keyframes")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.2)

    ax2 = axes[2]
    ax2.plot(indices, map_points, color="#8e44ad", lw=1.0, label="Map points")
    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("Map points")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.2)

    fig.suptitle("Tracking Quality over Run", fontsize=11)
    fig.tight_layout()
    out = output_dir / "tracking_quality.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_map_xy(data: dict, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    map_pts = data["map_points"]
    alignment = data["alignment"]
    gt = data["gt_positions"]

    if len(map_pts):
        map_aligned = _apply_alignment(map_pts, alignment)
        # filter outliers
        ref = data["assoc_est_se3"] if len(data["assoc_est_se3"]) else gt
        if len(ref):
            map_aligned = filter_map_points(map_aligned, ref, padding=3.0)
        ax.scatter(map_aligned[:, 0], map_aligned[:, 1],
                   s=0.8, color="#273746", alpha=0.5, label="Map points", linewidths=0)

    if len(gt):
        ax.plot(gt[:, 0], gt[:, 1], color="black", lw=1.5, label="GT trajectory", zorder=3)

    est_aligned = data["assoc_est_se3"]
    if len(est_aligned):
        ax.plot(est_aligned[:, 0], est_aligned[:, 1],
                color="#d04a3a", lw=1.2, label="Estimated", zorder=2, alpha=0.85)

    arrays = [arr for arr in [map_pts, gt, est_aligned] if len(arr)]
    if arrays:
        _set_equal_xy(ax, arrays, robust=True)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Sparse Map + Trajectory (XY, SE3-aligned)")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, markerscale=4)
    fig.tight_layout()
    out = output_dir / "map_xy.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_keyframe_graph_xy(data: dict, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    kfs = data["keyframes"]
    graph = data["graph"]
    alignment = data["alignment"]

    if kfs:
        kf_aligned = {
            kid: _apply_alignment(pos.reshape(1, 3), alignment).reshape(3)
            for kid, pos in kfs.items()
        }
        kpts = np.asarray(list(kf_aligned.values()), dtype=np.float64)
        ax.scatter(kpts[:, 0], kpts[:, 1], s=18, color="#1b4f72", zorder=4, label="Keyframes")

        for edge_group, color, lw in [
            ("spanning_tree_edges", "#7f8c8d", 0.7),
            ("covisibility_edges", "#b7950b", 0.4),
            ("loop_edges", "#c0392b", 1.6),
        ]:
            drawn_loop = False
            for edge in graph.get(edge_group, []):
                a = kf_aligned.get(int(edge.get("source", -1)))
                b = kf_aligned.get(int(edge.get("target", -1)))
                if a is not None and b is not None:
                    lbl = edge_group.replace("_", " ") if not drawn_loop and edge_group == "loop_edges" else None
                    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, alpha=0.65,
                            label=lbl)
                    if edge_group == "loop_edges":
                        drawn_loop = True

        _set_equal_xy(ax, [kpts])
    else:
        ax.set_title("Keyframe Graph (no data)")

    gt = data["gt_positions"]
    if len(gt):
        ax.plot(gt[:, 0], gt[:, 1], color="black", lw=1.2, label="GT trajectory", alpha=0.6, zorder=2)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Keyframe Graph (SE3-aligned)")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = output_dir / "keyframe_graph_xy.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_metrics_panel(data: dict, output_dir: Path) -> Path:
    m = data["metrics"]

    rows = [
        ("Estimated poses", m.get("num_estimated_poses", "N/A")),
        ("GT poses", m.get("num_groundtruth_poses", "N/A")),
        ("Associated poses", m.get("num_associations", "N/A")),
        ("ATE RMSE SE(3) [m]", f"{m['ate_rmse_se3_m']:.6f}" if "ate_rmse_se3_m" in m else "N/A"),
        ("ATE mean SE(3) [m]", f"{m['ate_mean_se3_m']:.6f}" if "ate_mean_se3_m" in m else "N/A"),
        ("ATE median SE(3) [m]", f"{m['ate_median_se3_m']:.6f}" if "ate_median_se3_m" in m else "N/A"),
        ("ATE max SE(3) [m]", f"{m['ate_max_se3_m']:.6f}" if "ate_max_se3_m" in m else "N/A"),
        ("ATE RMSE Sim(3) [m]", f"{m['ate_rmse_sim3_m']:.6f}" if "ate_rmse_sim3_m" in m else "N/A"),
        ("Sim(3) scale", f"{m['sim3_scale']:.6f}" if "sim3_scale" in m else "N/A"),
        ("RPE trans RMSE [m]", f"{m['rpe_trans_rmse_m']:.6f}" if "rpe_trans_rmse_m" in m else "N/A"),
        ("RPE rot RMSE [deg]", f"{m['rpe_rot_rmse_deg']:.6f}" if "rpe_rot_rmse_deg" in m else "N/A"),
        ("RPE pairs", m.get("rpe_pairs", "N/A")),
        ("Mean time diff [s]", f"{m['mean_time_diff_s']:.6f}" if "mean_time_diff_s" in m else "N/A"),
    ]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(rows) + 1.5)), dpi=150)
    ax.axis("off")
    table = ax.table(
        cellText=[[k, str(v)] for k, v in rows],
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.4, 1.5)
    ax.set_title("Trajectory Evaluation Metrics", pad=12, fontsize=11)
    fig.tight_layout()
    out = output_dir / "metrics_panel.png"
    fig.savefig(out)
    plt.close(fig)

    # also write markdown
    md_lines = [
        "# Trajectory Evaluation Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for k, v in rows:
        md_lines.append(f"| {k} | {v} |")
    (output_dir / "metrics_table.md").write_text("\n".join(md_lines) + "\n")

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_PLOT_NAMES = [
    "trajectory_xy.png",
    "trajectory_3d.png",
    "ate_over_time.png",
    "rpe_over_time.png",
    "tracking_quality.png",
    "map_xy.png",
    "keyframe_graph_xy.png",
    "metrics_panel.png",
    "metrics_table.md",
]


def generate_all_plots(
    run_dir: Path,
    gt_path: Path | None,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading run data from: {run_dir}")
    data = load_run_data(run_dir, gt_path)

    m = data["metrics"]
    if m:
        print(f"  Metrics loaded: ATE RMSE SE3={m.get('ate_rmse_se3_m', 'N/A'):.6f} m  "
              f"RPE trans={m.get('rpe_trans_rmse_m', 'N/A'):.6f} m")
    print(f"  GT positions:   {len(data['gt_positions'])}")
    print(f"  Est positions:  {len(data['est_positions'])}")
    print(f"  Associations:   {len(data['assoc_rows'])}")
    print(f"  Map points:     {len(data['map_points'])}")
    print(f"  Keyframes:      {len(data['keyframes'])}")

    generated = []
    for fn, gen in [
        ("trajectory_xy.png", lambda: plot_trajectory_xy(data, output_dir)),
        ("trajectory_3d.png", lambda: plot_trajectory_3d(data, output_dir)),
        ("ate_over_time.png", lambda: plot_ate_over_time(data, output_dir)),
        ("rpe_over_time.png", lambda: plot_rpe_over_time(data, output_dir)),
        ("tracking_quality.png", lambda: plot_tracking_quality(data, output_dir)),
        ("map_xy.png", lambda: plot_map_xy(data, output_dir)),
        ("keyframe_graph_xy.png", lambda: plot_keyframe_graph_xy(data, output_dir)),
        ("metrics_panel.png", lambda: plot_metrics_panel(data, output_dir)),
    ]:
        try:
            path = gen()
            generated.append(path)
            print(f"  [ok] {path.name}")
        except Exception as exc:
            print(f"  [err] {fn}: {exc}")

    return generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Run output directory (contains trajectory_*.txt, map_points.ply, etc.)")
    parser.add_argument("--groundtruth", type=Path, default=None,
                        help="Path to groundtruth.txt in TUM format")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output plot directory (default: <run-dir>/plots)")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2

    gt_path = args.groundtruth.expanduser().resolve() if args.groundtruth else None
    output_dir = args.output.expanduser().resolve() if args.output else run_dir / "plots"

    generated = generate_all_plots(run_dir, gt_path, output_dir)
    print(f"\nGenerated {len(generated)} figures in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
