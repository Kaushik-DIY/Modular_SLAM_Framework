"""
rebuild_map_lab_submap.py
=========================
Post-run occupancy-map reconstruction for the lab_run_2 dataset (scan_to_submap).

Generates three outputs in one run:
  1. **Local SLAM map** — built from the raw scan-to-map trajectory
  2. **PGO map** — built from the PGO-corrected trajectory (if available)
  3. **Comparison PNG** — side-by-side (or triple) panel showing both maps and a
     trajectory overlay diff panel

Usage
-----
    # Auto-discover latest trajectories and produce all three maps:
    python -m hector.eval.rebuild_map_lab

    # Explicit paths:
    python -m hector.eval.rebuild_map_lab \\
        --traj  hector_outputs/trajectory_lab_run_2_raw_scan_to_map_1087.txt \\
        --traj_pgo hector_outputs/trajectory_lab_run_2_raw_scan_to_map_1087_pgo.txt \\
        --variant raw \\
        --out_dir hector_outputs

    # Skip PGO map (local-only mode):
    python -m hector.eval.rebuild_map_lab --no_pgo
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

from slam_core.matching.scan_to_submap_old import ProbabilityGrid, SubmapBuilder2D
from slam_core.common.types import Pose2
from carto.local_slam.range_to_points import ranges_to_points


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_local_traj(path: str, min_score: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the local SLAM trajectory file (t x y theta score ...).
    Returns:
      stamps  (N,)  — timestamps of accepted poses
      poses   (N,3) — [x, y, theta]
    Only rows with score >= min_score are included.
    """
    stamps, poses = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                t, x, y, th, sc = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                if sc >= min_score:
                    stamps.append(t)
                    poses.append([x, y, th])
            except ValueError:
                continue
    if not stamps:
        raise RuntimeError(f"No accepted poses found in {path} with min_score={min_score}")
    return np.array(stamps, dtype=float), np.array(poses, dtype=float)


def load_pgo_traj(path: str) -> np.ndarray:
    """
    Load a PGO trajectory file (t x y theta  OR  x y theta).
    Returns poses (N,3) — [x, y, theta].
    """
    poses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                if len(parts) >= 4:
                    # timestamp x y theta
                    x, y, th = float(parts[1]), float(parts[2]), float(parts[3])
                elif len(parts) == 3:
                    x, y, th = float(parts[0]), float(parts[1]), float(parts[2])
                else:
                    continue
                poses.append([x, y, th])
            except ValueError:
                continue
    if not poses:
        raise RuntimeError(f"No poses found in {path}")
    return np.array(poses, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Scan loading + preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _load_aligned_scan_points(
    stamps: np.ndarray,
    scan_variant: str,
    voxel_filter: bool = True,
) -> tuple[object, list[np.ndarray]]:
    """
    Load lab_run_2 scans, align them to `stamps` by nearest timestamp, and
    optionally voxel-filter each scan.

    Returns (profile, pts_list) where pts_list[i] is (M,2) in sensor frame.
    """
    from slam_core.dataio.dataset_catalog import load_dataset_scans
    from slam_core.matching.preprocessing import PointCloudProcessor, PointCloudProcessorConfig

    profile, all_scans = load_dataset_scans("lab_run_2", scan_variant=scan_variant)
    scan_stamps = np.array([s["t"] for s in all_scans], dtype=float)

    proc = PointCloudProcessor(PointCloudProcessorConfig(
        fixed_voxel_size=0.03,
        adaptive_voxel_max_size=0.10,
        adaptive_min_num_points=200,
        adaptive_num_iterations=6,
        enabled=voxel_filter,
    ))

    pts_list: list[np.ndarray] = []
    for t in stamps:
        idx = int(np.argmin(np.abs(scan_stamps - t)))
        s = all_scans[idx]
        pts_raw = ranges_to_points(
            s["ranges"],
            profile.angle_min,
            profile.angle_inc,
            profile.range_min,
            profile.range_max,
        )
        pts, _ = proc.process(pts_raw)
        pts_list.append(pts)

    return profile, pts_list


# ─────────────────────────────────────────────────────────────────────────────
# Map builder
# ─────────────────────────────────────────────────────────────────────────────

def build_map(
    poses_xyt: np.ndarray,
    pts_list: list[np.ndarray],
    map_res: float = 0.05,
    map_size_m: float = 40.0,
    ray_steps: int = 20,
    l_free: float = -0.1,
    l_occ: float = 1.0,
    l_min: float = -5.0,
    l_max: float = 5.0,
    label: str = "",
) -> tuple[ProbabilityGrid, np.ndarray]:
    """
    Simulate the scan_to_submap backend directly inside the map builder.
    """
    N = min(len(poses_xyt), len(pts_list))
    
    # 1. Build submaps
    builder = SubmapBuilder2D(
        submap_size_m=20.0,
        resolution=map_res,
        scans_per_submap=500,
        ray_steps=ray_steps,
        l0=0.0,
        l_occ=l_occ,
        l_free=l_free,
        l_min=l_min,
        l_max=l_max
    )
    
    traj_xy = []
    prefix = f"[rebuild_map_lab] [{label}]" if label else "[rebuild_map_lab]"
    
    for i, (pose_row, pts) in enumerate(zip(poses_xyt[:N], pts_list[:N])):
        x, y, th = float(pose_row[0]), float(pose_row[1]), float(pose_row[2])
        pose_arr = Pose2(x, y, th)
        traj_xy.append([x, y])
        
        if pts.shape[0] == 0:
            continue
            
        builder.insert_scan(pose_arr, pts)
        
    print(f"{prefix} Built submaps: {len(builder.finished_submaps)} finished, {len(builder.active)} active")

    # 2. Render submaps into a global map
    global_grid = ProbabilityGrid(
        size_m=map_size_m,
        resolution=map_res,
        l0=0.0,
        l_occ=l_occ,
        l_free=l_free,
        l_min=-5.0, 
        l_max=5.0
    )
    
    all_submaps = builder.finished_submaps + builder.active
    
    for sm in all_submaps:
        sm_grid = sm.grid
        gx, gy = global_grid.world_to_grid(*sm_grid.origin_world)
        
        gx0 = max(0, gx)
        gy0 = max(0, gy)
        gx1 = min(global_grid.w, gx + sm_grid.w)
        gy1 = min(global_grid.h, gy + sm_grid.h)
        
        sx0 = gx0 - gx
        sy0 = gy0 - gy
        sx1 = sx0 + (gx1 - gx0)
        sy1 = sy0 + (gy1 - gy0)
        
        if gx1 > gx0 and gy1 > gy0:
            global_patch = global_grid.L[gy0:gy1, gx0:gx1]
            sm_patch = sm_grid.L[sy0:sy1, sx0:sx1]
            global_grid.L[gy0:gy1, gx0:gx1] = np.clip(global_patch + sm_patch, global_grid.l_min, global_grid.l_max)

    return global_grid, np.array(traj_xy, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_panel(
    ax,
    prob: np.ndarray,
    traj_xy: np.ndarray,
    grid: ProbabilityGrid,
    title: str,
    traj_label: str = "Trajectory",
    traj_color: str = "cool",
) -> None:
    """Render a single occupancy-map panel onto a matplotlib Axes."""
    import matplotlib
    import matplotlib.cm as cm

    ax.set_facecolor("#1a1a2e")
    display = np.flipud(prob)
    ax.imshow(
        display,
        cmap="binary_r",
        vmin=0.2,
        vmax=0.8,
        interpolation="nearest",
        origin="upper",
    )

    if traj_xy.shape[0] > 0:
        gxy = np.array([grid.world_to_grid(row[0], row[1]) for row in traj_xy], dtype=int)
        px = gxy[:, 0]
        py = grid.h - 1 - gxy[:, 1]
        n = len(px)
        cmap = matplotlib.colormaps[traj_color]
        colours = cmap(np.linspace(0, 1, n))
        ax.scatter(px, py, c=colours, s=3, linewidths=0, zorder=3, alpha=0.85)
        ax.plot(px[0], py[0], "o", color="#00ff88", markersize=8, zorder=4, label="Start")
        ax.plot(px[-1], py[-1], "X", color="#ff4466", markersize=10, zorder=4, label="End")

    # Scale bar (2 m)
    scale_m = 2.0
    scale_px = scale_m / grid.res
    x0, y0 = 20, display.shape[0] - 30
    ax.annotate(
        "",
        xy=(x0 + scale_px, y0),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="<->", color="white", lw=1.5),
    )
    ax.text(x0 + scale_px / 2, y0 - 12, f"{scale_m:.0f} m",
            color="white", ha="center", va="bottom", fontsize=8)

    ax.set_title(title, color="white", fontsize=11, pad=6)
    ax.tick_params(colors="#888899", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.legend(loc="upper right", fontsize=7, facecolor="#1a1a2e",
              labelcolor="white", framealpha=0.75)


def _render_diff_panel(
    ax,
    traj_local: np.ndarray,
    traj_pgo: np.ndarray,
    grid_ref: ProbabilityGrid,
    prob_ref: np.ndarray,
) -> None:
    """
    Trajectory-diff panel: render the PGO map as background and overlay
    both trajectories (local = warm orange, PGO = cool cyan) so the
    difference is visible.
    """
    ax.set_facecolor("#1a1a2e")
    ax.imshow(np.flipud(prob_ref), cmap="binary_r", vmin=0.2, vmax=0.8,
              interpolation="nearest", origin="upper")

    def _overlay(traj_xy, color, label, marker):
        if traj_xy.shape[0] == 0:
            return
        gxy = np.array([grid_ref.world_to_grid(row[0], row[1]) for row in traj_xy], dtype=int)
        px, py = gxy[:, 0], grid_ref.h - 1 - gxy[:, 1]
        ax.plot(px, py, "-", color=color, lw=1.2, alpha=0.85, label=label, zorder=3)
        ax.plot(px[0], py[0], "o", color="#00ff88", markersize=8, zorder=5)
        ax.plot(px[-1], py[-1], marker, color=color, markersize=9, zorder=5)

    _overlay(traj_local, "#ff9933", "Local SLAM", "s")
    _overlay(traj_pgo,   "#33ccff", "After PGO",  "X")

    # Correction-magnitude colorbar: draw displacement arrows at every 50th node
    N = min(len(traj_local), len(traj_pgo))
    stride = max(1, N // 40)
    for i in range(0, N, stride):
        g_loc = grid_ref.world_to_grid(traj_local[i:i+1][0][0], traj_local[i:i+1][0][1])
        g_pgo = grid_ref.world_to_grid(traj_pgo[i:i+1][0][0], traj_pgo[i:i+1][0][1])
        dx = float(g_pgo[0] - g_loc[0])
        dy = float((grid_ref.h - 1 - g_pgo[1]) - (grid_ref.h - 1 - g_loc[1]))
        mag = float(np.hypot(dx, dy))
        if mag < 0.5:
            continue
        ax.annotate(
            "",
            xy=(float(g_pgo[0]), float(grid_ref.h - 1 - g_pgo[1])),
            xytext=(float(g_loc[0]), float(grid_ref.h - 1 - g_loc[1])),
            arrowprops=dict(arrowstyle="->", color="#ffdd44", lw=0.8, alpha=0.7),
            zorder=4,
        )

    ax.set_title("Trajectory Comparison  (local → PGO shift)", color="white",
                 fontsize=11, pad=6)
    ax.tick_params(colors="#888899", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.legend(loc="upper right", fontsize=7, facecolor="#1a1a2e",
              labelcolor="white", framealpha=0.75)


def save_individual_png(
    prob: np.ndarray,
    traj_xy: np.ndarray,
    grid: ProbabilityGrid,
    out_path: str,
    title: str,
) -> None:
    """Save a single-panel occupancy map PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[rebuild_map_lab] matplotlib not available — skipping PNG")
        return

    fig, ax = plt.subplots(figsize=(10, 10), dpi=130)
    fig.patch.set_facecolor("#1a1a2e")
    _render_panel(ax, prob, traj_xy, grid, title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[rebuild_map_lab] Saved: {out_path}")


def save_comparison_png(
    prob_local: np.ndarray,
    traj_local: np.ndarray,
    grid_local: ProbabilityGrid,
    prob_pgo: np.ndarray,
    traj_pgo: np.ndarray,
    grid_pgo: ProbabilityGrid,
    out_path: str,
    n_local: int,
    n_pgo: int,
    n_lc: Optional[int] = None,
) -> None:
    """
    Save a 3-panel comparison PNG:
      [Local SLAM map]  |  [PGO map]  |  [Trajectory diff]
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[rebuild_map_lab] matplotlib not available — skipping comparison PNG")
        return

    fig, axes = plt.subplots(1, 3, figsize=(27, 9), dpi=120)
    fig.patch.set_facecolor("#0d0d1f")

    lc_str = f"  ·  {n_lc} loop closures" if n_lc is not None else ""
    fig.suptitle(
        f"Lab Room — Submap SLAM  |  Local SLAM vs PGO{lc_str}",
        color="white", fontsize=14, y=1.01,
    )

    _render_panel(
        axes[0], prob_local, traj_local, grid_local,
        title=f"Local SLAM only  ({n_local} poses)",
        traj_color="autumn",
    )
    _render_panel(
        axes[1], prob_pgo, traj_pgo, grid_pgo,
        title=f"After PGO  ({n_pgo} poses)",
        traj_color="cool",
    )
    _render_diff_panel(
        axes[2], traj_local, traj_pgo, grid_pgo, prob_pgo,
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[rebuild_map_lab] Comparison PNG saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Auto-discovery helpers
# ─────────────────────────────────────────────────────────────────────────────

def _latest_local_traj(out_dir: str, variant: str) -> Optional[str]:
    p = Path(out_dir)
    candidates = sorted(
        p.glob(f"trajectory_lab_run_2_{variant}_scan_to_submap_*.txt"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        if "_debug" not in c.name and "_pgo" not in c.name:
            return str(c)
    return None


def _latest_pgo_traj(out_dir: str, variant: str) -> Optional[str]:
    p = Path(out_dir)
    candidates = sorted(
        p.glob(f"trajectory_lab_run_2_{variant}_scan_to_submap_*_pgo.txt"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


def _count_lc_from_pgo_log(pgo_traj_path: str) -> Optional[int]:
    """Try to infer the loop-closure count from the PGO trajectory stem."""
    # The pgo_lab.py script prints 'LC accepted' lines — we can't recover that
    # from the file alone, so return None (caller will omit from title).
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Rebuild lab_run_2 occupancy maps:\n"
            "  • Local SLAM map  (from scan-to-map trajectory)\n"
            "  • PGO map         (from PGO-corrected trajectory, if available)\n"
            "  • Comparison PNG  (3-panel side-by-side)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--traj", default=None,
                    help="Local SLAM trajectory (auto-discovered if omitted).")
    ap.add_argument("--traj_pgo", default=None,
                    help="PGO trajectory (auto-discovered if omitted).")
    ap.add_argument("--variant", default="raw", choices=["raw", "360"],
                    help="Scan variant to replay (default: raw).")
    ap.add_argument("--hector_out", default="hector_outputs",
                    help="Directory searched for auto-discovery (default: hector_outputs).")
    ap.add_argument("--out_dir", default="hector_outputs",
                    help="Output directory for maps and PNGs (default: hector_outputs).")
    ap.add_argument("--min_score", type=float, default=0.0,
                    help="Min score to include a local-SLAM pose (default: 0.0 = all).")
    ap.add_argument("--map_res", type=float, default=0.05,
                    help="Map resolution in metres (default: 0.05).")
    ap.add_argument("--map_size_m", type=float, default=40.0,
                    help="Map side length in metres (default: 40.0).")
    ap.add_argument("--ray_steps", type=int, default=20,
                    help="Bresenham ray steps for map integration (default: 20).")
    ap.add_argument("--no_pgo", action="store_true",
                    help="Skip PGO map even if a PGO trajectory exists.")
    ap.add_argument("--no_filter", action="store_true",
                    help="Disable voxel pre-filtering of scan points.")
    args = ap.parse_args()

    # ── 1. Trajectory resolution ──────────────────────────────────────────────
    local_path = args.traj
    if local_path is None:
        local_path = _latest_local_traj(args.hector_out, args.variant)
        if local_path is None:
            ap.error(f"No local SLAM trajectory found in '{args.hector_out}'. "
                     "Run the SLAM pipeline first or pass --traj.")
        print(f"[rebuild_map_lab] Auto-selected local trajectory: {local_path}")

    pgo_path: Optional[str] = None
    if not args.no_pgo:
        pgo_path = args.traj_pgo or _latest_pgo_traj(args.hector_out, args.variant)
        if pgo_path:
            print(f"[rebuild_map_lab] Auto-selected PGO trajectory:   {pgo_path}")
        else:
            print("[rebuild_map_lab] No PGO trajectory found — "
                  "run `python -m hector.eval.pgo_lab` to generate one.")

    # ── 2. Load trajectories ──────────────────────────────────────────────────
    stamps, poses_local = load_local_traj(local_path, min_score=args.min_score)
    print(f"[rebuild_map_lab] Local SLAM: {len(poses_local)} accepted poses")

    poses_pgo: Optional[np.ndarray] = None
    if pgo_path:
        poses_pgo = load_pgo_traj(pgo_path)
        # Trim to same length as local if needed
        N = min(len(poses_local), len(poses_pgo))
        poses_local_aligned = poses_local[:N]
        poses_pgo = poses_pgo[:N]
        stamps_aligned = stamps[:N]
        print(f"[rebuild_map_lab]      PGO: {len(poses_pgo)} poses")
        # Report correction magnitude
        delta = np.linalg.norm(poses_pgo[:, :2] - poses_local_aligned[:, :2], axis=1)
        print(f"[rebuild_map_lab] PGO correction: "
              f"max={delta.max():.4f}m  mean={delta.mean():.4f}m")
    else:
        poses_local_aligned = poses_local
        stamps_aligned = stamps

    # ── 3. Load scans (shared between both maps) ───────────────────────────────
    print("[rebuild_map_lab] Loading and preprocessing scans ...")
    profile, pts_list = _load_aligned_scan_points(
        stamps=stamps_aligned,
        scan_variant=args.variant,
        voxel_filter=not args.no_filter,
    )
    print(f"[rebuild_map_lab] Scans ready: {len(pts_list)} "
          f"(mean {np.mean([p.shape[0] for p in pts_list]):.0f} pts/scan)")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = Path(local_path).stem

    kw = dict(
        map_res=args.map_res,
        map_size_m=args.map_size_m,
        ray_steps=args.ray_steps,
        l_free=-0.1,
        l_occ=1.0,
        l_min=-5.0,
        l_max=5.0,
    )

    # ── 4. Build local SLAM map ────────────────────────────────────────────────
    print("[rebuild_map_lab] Building local SLAM map ...")
    grid_local, traj_local = build_map(
        poses_xyt=poses_local_aligned,
        pts_list=pts_list,
        label="local",
        **kw,
    )
    prob_local = grid_local.probability().astype(np.float32)

    # Save local map files
    npy_local = os.path.join(args.out_dir, f"map_{stem}_local.npy")
    png_local = os.path.join(args.out_dir, f"map_{stem}_local.png")
    np.save(npy_local, prob_local)
    print(f"[rebuild_map_lab] Local map .npy: {npy_local}")
    save_individual_png(
        prob_local, traj_local, grid_local, png_local,
        title=f"Lab Room — Local SLAM  ({args.variant}, {len(traj_local)} poses)",
    )
    _print_stats(prob_local, args.map_res, grid_local.w, label="Local")

    # ── 5. Build PGO map (if available) ───────────────────────────────────────
    if poses_pgo is not None:
        print("[rebuild_map_lab] Building PGO map ...")
        grid_pgo, traj_pgo = build_map(
            poses_xyt=poses_pgo,
            pts_list=pts_list,
            label="pgo",
            **kw,
        )
        prob_pgo = grid_pgo.probability().astype(np.float32)

        npy_pgo = os.path.join(args.out_dir, f"map_{stem}_pgo.npy")
        png_pgo = os.path.join(args.out_dir, f"map_{stem}_pgo.png")
        np.save(npy_pgo, prob_pgo)
        print(f"[rebuild_map_lab] PGO map .npy: {npy_pgo}")
        save_individual_png(
            prob_pgo, traj_pgo, grid_pgo, png_pgo,
            title=f"Lab Room — PGO Corrected  ({args.variant}, {len(traj_pgo)} poses)",
        )
        _print_stats(prob_pgo, args.map_res, grid_pgo.w, label="PGO")

        # ── 6. Comparison PNG ──────────────────────────────────────────────────
        print("[rebuild_map_lab] Rendering comparison PNG ...")
        png_cmp = os.path.join(args.out_dir, f"map_{stem}_comparison.png")
        save_comparison_png(
            prob_local=prob_local,
            traj_local=traj_local,
            grid_local=grid_local,
            prob_pgo=prob_pgo,
            traj_pgo=traj_pgo,
            grid_pgo=grid_pgo,
            out_path=png_cmp,
            n_local=len(traj_local),
            n_pgo=len(traj_pgo),
            n_lc=None,   # unknown here; shown in pgo_lab.py's own output
        )
    else:
        print("[rebuild_map_lab] PGO trajectory not available — "
              "only local SLAM map generated.")
        print("[rebuild_map_lab] Run `python -m hector.eval.pgo_lab` then re-run this script.")

    print("[rebuild_map_lab] Done.")


def _print_stats(prob: np.ndarray, res: float, size: int, label: str = "") -> None:
    occ  = int((prob > 0.65).sum())
    free = int((prob < 0.35).sum())
    unk  = int(prob.size) - occ - free
    tag  = f"[{label}] " if label else ""
    print(f"[rebuild_map_lab] {tag}Stats: "
          f"{occ} occupied  {free} free  {unk} unknown  "
          f"({res}m res, {size}×{size})")


if __name__ == "__main__":
    main()
