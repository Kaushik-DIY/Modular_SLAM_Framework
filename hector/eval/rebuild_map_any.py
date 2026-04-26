"""
Dataset-agnostic occupancy-map rebuild for Hector SLAM trajectories.

Supports:
  - datasets: lab_run_2, fr079, intel
  - matchers: scan_to_map, scan_to_submap
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hector.config as cfg

from slam_core.common.types import Pose2
from slam_core.matching.scan_to_map import GridMap, _transform_points
from slam_core.matching.scan_to_submap_old import ProbabilityGrid, SubmapBuilder2D

from hector.eval._generic_eval_common import (
    configure_dataset,
    dataset_tag,
    default_map_size_m,
    ensure_dir,
    load_aligned_scan_points,
    load_local_traj,
    load_pgo_traj,
    parse_trajectory_context,
    resolve_latest_local_traj,
    resolve_pgo_traj,
)


def build_map(
    matcher_type: str,
    poses_xyt: np.ndarray,
    pts_list: list[np.ndarray],
    map_res: float,
    map_size_m: float,
    ray_steps: int,
    l_free: float,
    l_occ: float,
    l_min: float,
    l_max: float,
    label: str = "",
):
    N = min(len(poses_xyt), len(pts_list))
    traj_xy = []
    prefix = f"[rebuild_map_any] [{label}]" if label else "[rebuild_map_any]"

    if matcher_type == "scan_to_map":
        grid = GridMap(res=map_res, size_m=map_size_m, l_min=l_min, l_max=l_max)
        for pose_row, pts in zip(poses_xyt[:N], pts_list[:N]):
            pose_arr = np.array([float(pose_row[0]), float(pose_row[1]), float(pose_row[2])], dtype=float)
            traj_xy.append([pose_arr[0], pose_arr[1]])
            if pts.shape[0] == 0:
                continue
            pts_world = _transform_points(pose_arr, pts)
            grid.integrate_scan_simple(
                pose=pose_arr,
                pts_world=pts_world,
                l_free=l_free,
                l_occ=l_occ,
                ray_steps=ray_steps,
            )
        print(f"{prefix} Map built: {N} scans integrated")
        return grid, np.array(traj_xy, dtype=float)

    builder = SubmapBuilder2D(
        submap_size_m=cfg.SUBMAP_SIZE_METERS,
        resolution=map_res,
        scans_per_submap=cfg.SCANS_PER_SUBMAP,
        ray_steps=ray_steps,
        l0=cfg.L0,
        l_occ=l_occ,
        l_free=l_free,
        l_min=l_min,
        l_max=l_max,
    )
    for pose_row, pts in zip(poses_xyt[:N], pts_list[:N]):
        pose = Pose2(float(pose_row[0]), float(pose_row[1]), float(pose_row[2]))
        traj_xy.append([pose.x, pose.y])
        if pts.shape[0] == 0:
            continue
        builder.insert_scan(pose, pts)
    print(f"{prefix} Built submaps: {len(builder.finished_submaps)} finished, {len(builder.active)} active")

    global_grid = ProbabilityGrid(
        size_m=map_size_m,
        resolution=map_res,
        l0=cfg.L0,
        l_occ=l_occ,
        l_free=l_free,
        l_min=l_min,
        l_max=l_max,
    )
    for sm in builder.finished_submaps + builder.active:
        sm_grid = sm.grid
        gx, gy = global_grid.world_to_grid(*sm_grid.origin_world)
        gx0, gy0 = max(0, gx), max(0, gy)
        gx1 = min(global_grid.w, gx + sm_grid.w)
        gy1 = min(global_grid.h, gy + sm_grid.h)
        sx0, sy0 = gx0 - gx, gy0 - gy
        sx1 = sx0 + (gx1 - gx0)
        sy1 = sy0 + (gy1 - gy0)
        if gx1 > gx0 and gy1 > gy0:
            global_patch = global_grid.L[gy0:gy1, gx0:gx1]
            sm_patch = sm_grid.L[sy0:sy1, sx0:sx1]
            global_grid.L[gy0:gy1, gx0:gx1] = np.clip(global_patch + sm_patch, global_grid.l_min, global_grid.l_max)
    return global_grid, np.array(traj_xy, dtype=float)


def _grid_prob(grid) -> np.ndarray:
    if hasattr(grid, "prob"):
        return grid.prob().astype(np.float32)
    return grid.probability().astype(np.float32)


def _grid_shape(grid) -> int:
    return grid.size if hasattr(grid, "size") else grid.w


def _grid_xy(grid, traj_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if traj_xy.shape[0] == 0:
        return np.array([]), np.array([])
    if hasattr(grid, "size"):
        gxy = grid.world_to_grid(traj_xy)
        return gxy[:, 0], grid.size - 1 - gxy[:, 1]
    gxy = np.array([grid.world_to_grid(row[0], row[1]) for row in traj_xy], dtype=int)
    return gxy[:, 0], grid.h - 1 - gxy[:, 1]


def _render_panel(ax, prob: np.ndarray, traj_xy: np.ndarray, grid, title: str, traj_color: str = "cool") -> None:
    import matplotlib
    import matplotlib.cm as cm

    ax.set_facecolor("#1a1a2e")
    ax.imshow(np.flipud(prob), cmap="binary_r", vmin=0.2, vmax=0.8, interpolation="nearest", origin="upper")
    px, py = _grid_xy(grid, traj_xy)
    if len(px) > 0:
        cmap = matplotlib.colormaps[traj_color] if hasattr(matplotlib, "colormaps") else cm.get_cmap(traj_color)
        colours = cmap(np.linspace(0, 1, len(px)))
        ax.scatter(px, py, c=colours, s=3, linewidths=0, zorder=3, alpha=0.85)
        ax.plot(px[0], py[0], "o", color="#00ff88", markersize=8, zorder=4, label="Start")
        ax.plot(px[-1], py[-1], "X", color="#ff4466", markersize=10, zorder=4, label="End")
    scale_m = 2.0
    scale_px = scale_m / grid.res
    x0, y0 = 20, prob.shape[0] - 30
    ax.annotate("", xy=(x0 + scale_px, y0), xytext=(x0, y0), arrowprops=dict(arrowstyle="<->", color="white", lw=1.5))
    ax.text(x0 + scale_px / 2, y0 - 12, f"{scale_m:.0f} m", color="white", ha="center", va="bottom", fontsize=8)
    ax.set_title(title, color="white", fontsize=11, pad=6)
    ax.tick_params(colors="#888899", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.legend(loc="upper right", fontsize=7, facecolor="#1a1a2e", labelcolor="white", framealpha=0.75)


def _render_diff_panel(ax, traj_local: np.ndarray, traj_pgo: np.ndarray, grid_ref, prob_ref: np.ndarray) -> None:
    ax.set_facecolor("#1a1a2e")
    ax.imshow(np.flipud(prob_ref), cmap="binary_r", vmin=0.2, vmax=0.8, interpolation="nearest", origin="upper")

    def _overlay(traj_xy, color, label, marker):
        px, py = _grid_xy(grid_ref, traj_xy)
        if len(px) == 0:
            return
        ax.plot(px, py, "-", color=color, lw=1.2, alpha=0.85, label=label, zorder=3)
        ax.plot(px[0], py[0], "o", color="#00ff88", markersize=8, zorder=5)
        ax.plot(px[-1], py[-1], marker, color=color, markersize=9, zorder=5)

    _overlay(traj_local, "#ff9933", "Local SLAM", "s")
    _overlay(traj_pgo, "#33ccff", "After PGO", "X")

    N = min(len(traj_local), len(traj_pgo))
    stride = max(1, N // 40) if N > 0 else 1
    for i in range(0, N, stride):
        g_loc = _grid_xy(grid_ref, traj_local[i : i + 1])
        g_pgo = _grid_xy(grid_ref, traj_pgo[i : i + 1])
        if len(g_loc[0]) == 0 or len(g_pgo[0]) == 0:
            continue
        dx = float(g_pgo[0][0] - g_loc[0][0])
        dy = float(g_pgo[1][0] - g_loc[1][0])
        if np.hypot(dx, dy) < 0.5:
            continue
        ax.annotate("", xy=(float(g_pgo[0][0]), float(g_pgo[1][0])), xytext=(float(g_loc[0][0]), float(g_loc[1][0])),
                    arrowprops=dict(arrowstyle="->", color="#ffdd44", lw=0.8, alpha=0.7), zorder=4)

    ax.set_title("Trajectory Comparison", color="white", fontsize=11, pad=6)
    ax.tick_params(colors="#888899", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.legend(loc="upper right", fontsize=7, facecolor="#1a1a2e", labelcolor="white", framealpha=0.75)


def save_individual_png(prob: np.ndarray, traj_xy: np.ndarray, grid, out_path: str, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[rebuild_map_any] matplotlib not available, skipping PNG")
        return
    fig, ax = plt.subplots(figsize=(10, 10), dpi=130)
    fig.patch.set_facecolor("#1a1a2e")
    _render_panel(ax, prob, traj_xy, grid, title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_comparison_png(
    prob_local: np.ndarray,
    traj_local: np.ndarray,
    grid_local,
    prob_pgo: np.ndarray,
    traj_pgo: np.ndarray,
    grid_pgo,
    out_path: str,
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[rebuild_map_any] matplotlib not available, skipping comparison PNG")
        return
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=140)
    fig.patch.set_facecolor("#1a1a2e")
    _render_panel(axes[0], prob_local, traj_local, grid_local, "Local SLAM")
    _render_panel(axes[1], prob_pgo, traj_pgo, grid_pgo, "PGO Corrected")
    _render_diff_panel(axes[2], traj_local, traj_pgo, grid_pgo, prob_pgo)
    plt.suptitle(title, color="white", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _print_stats(prob: np.ndarray, res: float, size: int, label: str = "") -> None:
    occ = int((prob > 0.65).sum())
    free = int((prob < 0.35).sum())
    unk = int(prob.size) - occ - free
    tag = f"[{label}] " if label else ""
    print(f"[rebuild_map_any] {tag}Stats: {occ} occupied  {free} free  {unk} unknown  ({res}m res, {size}x{size})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild Hector occupancy maps across datasets and matchers.")
    ap.add_argument("--traj", default=None, help="Local SLAM trajectory. Auto-discovered if omitted.")
    ap.add_argument("--traj_pgo", default=None, help="PGO trajectory. Auto-discovered if omitted.")
    ap.add_argument("--dataset", default=None, choices=["lab_run_2", "fr079", "intel"])
    ap.add_argument("--scan-variant", dest="scan_variant", default=None, choices=["raw", "360"])
    ap.add_argument("--matcher", default=None, choices=["scan_to_map", "scan_to_submap"])
    ap.add_argument("--hector_out", default="hector_outputs")
    ap.add_argument("--out_dir", default="hector_outputs")
    ap.add_argument("--min_score", type=float, default=0.0)
    ap.add_argument("--map_res", type=float, default=None)
    ap.add_argument("--map_size_m", type=float, default=None)
    ap.add_argument("--ray_steps", type=int, default=None)
    ap.add_argument("--no_pgo", action="store_true")
    ap.add_argument("--no_filter", action="store_true")
    args = ap.parse_args()

    local_path = args.traj
    if local_path is None:
        local_path = resolve_latest_local_traj(
            out_dir=args.hector_out,
            dataset_name=args.dataset,
            scan_variant=args.scan_variant,
            matcher_type=args.matcher,
        )
        if local_path is None:
            ap.error("No matching local trajectory found. Pass --traj explicitly.")

    ctx = parse_trajectory_context(local_path)
    dataset_name = args.dataset or ctx["dataset_name"]
    scan_variant = args.scan_variant or ctx["scan_variant"]
    matcher_type = args.matcher or ctx["matcher_type"]

    configure_dataset(dataset_name)
    map_res = args.map_res if args.map_res is not None else cfg.MAP_RESOLUTION
    map_size_m = args.map_size_m if args.map_size_m is not None else default_map_size_m(dataset_name, matcher_type)
    ray_steps = args.ray_steps if args.ray_steps is not None else cfg.RAY_STEPS

    pgo_path: Optional[str] = None
    if not args.no_pgo:
        pgo_path = resolve_pgo_traj(local_path, args.traj_pgo, args.hector_out)

    print(f"[rebuild_map_any] Local trajectory: {local_path}")
    print(f"[rebuild_map_any] Dataset={dataset_name}  variant={scan_variant}  matcher={matcher_type}")
    if pgo_path:
        print(f"[rebuild_map_any] PGO trajectory:   {pgo_path}")
    elif not args.no_pgo:
        print("[rebuild_map_any] No PGO trajectory found; generating local-only outputs.")

    stamps, poses_local = load_local_traj(local_path, min_score=args.min_score)
    poses_pgo: Optional[np.ndarray] = None
    if pgo_path:
        poses_pgo = load_pgo_traj(pgo_path)
        N = min(len(poses_local), len(poses_pgo))
        poses_local_aligned = poses_local[:N]
        poses_pgo = poses_pgo[:N]
        stamps_aligned = stamps[:N]
        delta = np.linalg.norm(poses_pgo[:, :2] - poses_local_aligned[:, :2], axis=1)
        print(f"[rebuild_map_any] PGO correction: max={delta.max():.4f}m  mean={delta.mean():.4f}m")
    else:
        poses_local_aligned = poses_local
        stamps_aligned = stamps

    print("[rebuild_map_any] Loading and preprocessing scans ...")
    _, pts_list = load_aligned_scan_points(
        dataset_name=dataset_name,
        scan_variant=scan_variant,
        stamps=stamps_aligned,
        voxel_filter=(False if args.no_filter else None),
    )
    print(f"[rebuild_map_any] Scans ready: {len(pts_list)}  mean_pts={np.mean([p.shape[0] for p in pts_list]):.0f}")

    ensure_dir(args.out_dir)
    stem = Path(local_path).stem
    kw = dict(
        matcher_type=matcher_type,
        map_res=map_res,
        map_size_m=map_size_m,
        ray_steps=ray_steps,
        l_free=cfg.L_FREE,
        l_occ=cfg.L_OCC,
        l_min=cfg.L_MIN,
        l_max=cfg.L_MAX,
    )

    grid_local, traj_local = build_map(poses_xyt=poses_local_aligned, pts_list=pts_list, label="local", **kw)
    prob_local = _grid_prob(grid_local)
    npy_local = os.path.join(args.out_dir, f"map_{stem}_local.npy")
    png_local = os.path.join(args.out_dir, f"map_{stem}_local.png")
    np.save(npy_local, prob_local)
    save_individual_png(prob_local, traj_local, grid_local, png_local, title=f"Hector Local Map — {dataset_tag(dataset_name, scan_variant)}")
    _print_stats(prob_local, map_res, _grid_shape(grid_local), label="Local")

    if poses_pgo is not None:
        grid_pgo, traj_pgo = build_map(poses_xyt=poses_pgo, pts_list=pts_list, label="pgo", **kw)
        prob_pgo = _grid_prob(grid_pgo)
        npy_pgo = os.path.join(args.out_dir, f"map_{stem}_pgo.npy")
        png_pgo = os.path.join(args.out_dir, f"map_{stem}_pgo.png")
        png_cmp = os.path.join(args.out_dir, f"map_{stem}_comparison.png")
        np.save(npy_pgo, prob_pgo)
        save_individual_png(prob_pgo, traj_pgo, grid_pgo, png_pgo, title=f"Hector PGO Map — {dataset_tag(dataset_name, scan_variant)}")
        save_comparison_png(
            prob_local=prob_local,
            traj_local=traj_local,
            grid_local=grid_local,
            prob_pgo=prob_pgo,
            traj_pgo=traj_pgo,
            grid_pgo=grid_pgo,
            out_path=png_cmp,
            title=f"Hector Map Comparison — {dataset_tag(dataset_name, scan_variant)} — {matcher_type}",
        )
        _print_stats(prob_pgo, map_res, _grid_shape(grid_pgo), label="PGO")

    print("[rebuild_map_any] Done.")


if __name__ == "__main__":
    main()
