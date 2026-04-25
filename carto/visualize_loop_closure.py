#!/usr/bin/env python3
"""
visualize_loop_closure.py
=========================
Standalone visualization tool for Cartographer-style loop closure results.

Reads output files produced by run_loop_closure_slam.py from the
carto_outputs/ directory and produces:

    Plot 1 — Trajectory: raw odometry, before/after optimization, submap centers
    Plot 2 — Loop constraint edges overlaid on the trajectory
    Plot 3 — BnB score histogram (accepted vs. rejected)
    Plot 4 — Intra vs. inter constraint count timeline

Usage
-----
    cd /home/kaushik/Antigravity/slam_ws_AG
    source .venv/bin/activate
    python carto/visualize_loop_closure.py [--prefix carto_outputs/trajectory_...]

No dependency on the SLAM framework itself — reads text files only.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_poses(path: str) -> Optional[np.ndarray]:
    """Load a pose text file: columns = [id, x, y, theta]."""
    if not os.path.isfile(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                rows.append([float(p) for p in parts[:4]])
            except ValueError:
                continue
    if not rows:
        return None
    return np.array(rows, dtype=np.float64)


def _load_trajectory(traj_path: str) -> Optional[np.ndarray]:
    """Load the raw trajectory file: columns = [k, t, x, y, theta, ...]."""
    if not os.path.isfile(traj_path):
        return None
    rows = []
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                rows.append([float(p) for p in parts[:5]])
            except ValueError:
                continue
    if not rows:
        return None
    return np.array(rows, dtype=np.float64)  # [k, t, x, y, theta]


def _load_debug_file(debug_path: str) -> Tuple[List[dict], dict]:
    """
    Parse the meta/debug text file from run_loop_closure_slam.py.
    Returns (per_step_rows, summary_dict).
    """
    rows = []
    summary = {}
    if not os.path.isfile(debug_path):
        return rows, summary

    header = None
    with open(debug_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("# matcher"):
                continue
            if line.startswith("# loop_closure"):
                continue
            if line.startswith("#") and header is None:
                header = line.lstrip("#").split()
                continue
            # Summary lines
            if ":" in line and not line[0].isdigit():
                key, _, val = line.partition(":")
                summary[key.strip()] = val.strip()
                continue
            # Data rows
            if header and line[0].isdigit():
                parts = line.split()
                if len(parts) >= len(header):
                    try:
                        row = {h: float(v) for h, v in zip(header, parts)}
                        rows.append(row)
                    except ValueError:
                        continue
    return rows, summary


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _draw_trajectory(ax, xy: np.ndarray, label: str, color: str, lw=1.2, alpha=1.0, zorder=2):
    if xy is None or len(xy) < 2:
        return
    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=lw, alpha=alpha,
            label=label, zorder=zorder)


def _draw_scattered_poses(ax, poses: np.ndarray, label: str, color: str,
                           marker="o", s=25, zorder=3, alpha=0.8):
    if poses is None or len(poses) == 0:
        return
    ax.scatter(poses[:, 0], poses[:, 1], c=color, s=s, marker=marker,
               label=label, zorder=zorder, alpha=alpha)


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------

def visualize(
    prefix: str,
    traj_path: Optional[str] = None,
    out_dir: str = "carto_outputs",
) -> None:
    """
    Generate all visualization plots for one run.

    Parameters
    ----------
    prefix : str
        Path prefix used by run_loop_closure_slam.py, e.g.
        ``carto_outputs/trajectory_scan_to_submap_loop_700``
    traj_path : str, optional
        Path to the raw trajectory file. Defaults to ``{prefix}.txt``.
    out_dir : str
        Directory to write PNG output files.
    """
    traj_path = traj_path or f"{prefix}.txt"
    node_path = f"{prefix}_optimized_nodes.txt"
    submap_path = f"{prefix}_optimized_submaps.txt"
    debug_path = f"{prefix}_debug.txt"

    print(f"Loading: {traj_path}")
    traj = _load_trajectory(traj_path)

    print(f"Loading: {node_path}")
    nodes = _load_poses(node_path)

    print(f"Loading: {submap_path}")
    submaps = _load_poses(submap_path)

    print(f"Loading: {debug_path}")
    debug_rows, summary = _load_debug_file(debug_path)

    # ------------------------------------------------------------------ #
    # Plot 1: Trajectory comparison (before/after optimization)
    # ------------------------------------------------------------------ #
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    ax1.set_facecolor("#1a1a2e")
    fig1.patch.set_facecolor("#0f0f1a")

    if traj is not None:
        _draw_trajectory(ax1, traj[:, 2:4], "Dead reckoning (raw odom)", "#888888",
                         lw=1.0, alpha=0.5, zorder=1)

    if nodes is not None and len(nodes) > 0:
        # The debug file rows contain per-step x,y which are the pre-optimization
        # positions. nodes file contains post-optimization positions.
        if debug_rows:
            raw_xy = np.array([[r["x"], r["y"]] for r in debug_rows
                                if "x" in r and "y" in r])
            if len(raw_xy) > 0:
                _draw_trajectory(ax1, raw_xy, "Before optimization", "#e63946",
                                 lw=1.5, alpha=0.85, zorder=2)

        _draw_trajectory(ax1, nodes[:, 1:3], "After optimization (nodes)",
                         "#06d6a0", lw=1.8, alpha=0.95, zorder=3)

    if submaps is not None and len(submaps) > 0:
        _draw_scattered_poses(ax1, submaps[:, 1:3], "Submap centers",
                              "#ffd166", marker="D", s=60, zorder=4)

    ax1.set_title("Trajectory: Before vs. After Loop Closure",
                  color="white", fontsize=14, pad=12)
    ax1.set_xlabel("x [m]", color="lightgray")
    ax1.set_ylabel("y [m]", color="lightgray")
    ax1.tick_params(colors="lightgray")
    ax1.set_aspect("equal")
    ax1.grid(True, color="#333355", linewidth=0.5)
    legend = ax1.legend(facecolor="#22223b", edgecolor="#555577", labelcolor="white",
                         fontsize=10)
    plt.tight_layout()

    out1 = os.path.join(out_dir, "viz_trajectory_comparison.png")
    fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor=fig1.get_facecolor())
    print(f"Wrote: {out1}")
    plt.close(fig1)

    # ------------------------------------------------------------------ #
    # Plot 2: Loop closure edges
    # ------------------------------------------------------------------ #
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    ax2.set_facecolor("#1a1a2e")
    fig2.patch.set_facecolor("#0f0f1a")

    if nodes is not None and len(nodes) > 0:
        _draw_trajectory(ax2, nodes[:, 1:3], "Optimized trajectory",
                         "#06d6a0", lw=1.5, alpha=0.85, zorder=2)
        node_xy = {int(r[0]): (r[1], r[2]) for r in nodes}

    if submaps is not None and len(submaps) > 0:
        _draw_scattered_poses(ax2, submaps[:, 1:3], "Submap centers",
                              "#ffd166", marker="D", s=60, zorder=4)
        submap_xy = {int(r[0]): (r[1], r[2]) for r in submaps}

    # Draw loop edges from debug rows that have constraint info
    loop_edge_count = 0
    if debug_rows and nodes is not None and submaps is not None:
        node_xy_map = {int(r[0]): (r[1], r[2]) for r in nodes}
        submap_xy_map = {int(r[0]): (r[1], r[2]) for r in submaps}

        for row in debug_rows:
            if row.get("constraints_loop", 0) <= 0:
                continue
            # If we have enough info to draw an edge, do so
            # The debug file tracks per-step loop count; we draw edges between
            # the current node and nearest submap as a proxy.
            nid = int(row.get("k", -1))
            n_loop = int(row.get("constraints_loop", 0))
            if nid in node_xy_map and n_loop > 0:
                nx, ny = node_xy_map[nid]
                # Find the submap closest to this node (proxy for actual edge)
                best_dist = float("inf")
                best_sm = None
                for sid, (sx, sy) in submap_xy_map.items():
                    d = np.hypot(nx - sx, ny - sy)
                    if d < best_dist:
                        best_dist = d
                        best_sm = (sx, sy)
                if best_sm is not None and best_dist < 20.0:
                    ax2.plot([nx, best_sm[0]], [ny, best_sm[1]],
                             color="#118ab2", linewidth=0.7, alpha=0.6, zorder=3)
                    loop_edge_count += 1

    ax2.set_title(f"Loop Closure Constraint Edges (~{loop_edge_count} drawn)",
                  color="white", fontsize=14, pad=12)
    ax2.set_xlabel("x [m]", color="lightgray")
    ax2.set_ylabel("y [m]", color="lightgray")
    ax2.tick_params(colors="lightgray")
    ax2.set_aspect("equal")
    ax2.grid(True, color="#333355", linewidth=0.5)
    ax2.legend(facecolor="#22223b", edgecolor="#555577", labelcolor="white", fontsize=10)
    plt.tight_layout()

    out2 = os.path.join(out_dir, "viz_loop_edges.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    print(f"Wrote: {out2}")
    plt.close(fig2)

    # ------------------------------------------------------------------ #
    # Plot 3: Score histogram
    # ------------------------------------------------------------------ #
    if debug_rows:
        scores = [r["score"] for r in debug_rows if "score" in r and r["score"] > 0]
        if scores:
            fig3, ax3 = plt.subplots(figsize=(9, 5))
            ax3.set_facecolor("#1a1a2e")
            fig3.patch.set_facecolor("#0f0f1a")

            bins = np.linspace(0.0, 1.0, 40)
            ax3.hist(scores, bins=bins, color="#06d6a0", edgecolor="#0f0f1a",
                     alpha=0.85, label="Scan match scores")
            ax3.axvline(0.55, color="#e63946", linewidth=2.0, linestyle="--",
                        label="min_score=0.55 (Cartographer default)")
            ax3.axvline(0.60, color="#ffd166", linewidth=1.5, linestyle=":",
                        label="global_min=0.60")

            ax3.set_title("Scan Match Score Distribution", color="white", fontsize=13)
            ax3.set_xlabel("Score", color="lightgray")
            ax3.set_ylabel("Count", color="lightgray")
            ax3.tick_params(colors="lightgray")
            ax3.legend(facecolor="#22223b", edgecolor="#555577", labelcolor="white")
            ax3.grid(True, color="#333355", linewidth=0.5, axis="y")
            fig3.patch.set_facecolor("#0f0f1a")
            plt.tight_layout()

            out3 = os.path.join(out_dir, "viz_score_histogram.png")
            fig3.savefig(out3, dpi=150, bbox_inches="tight",
                         facecolor=fig3.get_facecolor())
            print(f"Wrote: {out3}")
            plt.close(fig3)

    # ------------------------------------------------------------------ #
    # Plot 4: Constraint count timeline
    # ------------------------------------------------------------------ #
    if debug_rows:
        ks = [r["k"] for r in debug_rows if "k" in r]
        intra = [r.get("constraints_intra", 0) for r in debug_rows if "k" in r]
        loop = [r.get("constraints_loop", 0) for r in debug_rows if "k" in r]

        if ks and (max(intra) > 0 or max(loop) > 0):
            fig4, ax4 = plt.subplots(figsize=(11, 5))
            ax4.set_facecolor("#1a1a2e")
            fig4.patch.set_facecolor("#0f0f1a")

            ax4.plot(ks, intra, color="#06d6a0", linewidth=1.8,
                     label="Intra-submap constraints")
            ax4.plot(ks, loop, color="#e63946", linewidth=1.8,
                     label="Loop closure constraints")
            ax4.fill_between(ks, loop, alpha=0.25, color="#e63946")

            ax4.set_title("Constraint Count Over Time", color="white", fontsize=13)
            ax4.set_xlabel("Scan index k", color="lightgray")
            ax4.set_ylabel("Constraint count", color="lightgray")
            ax4.tick_params(colors="lightgray")
            ax4.legend(facecolor="#22223b", edgecolor="#555577", labelcolor="white")
            ax4.grid(True, color="#333355", linewidth=0.5)
            plt.tight_layout()

            out4 = os.path.join(out_dir, "viz_constraint_timeline.png")
            fig4.savefig(out4, dpi=150, bbox_inches="tight",
                         facecolor=fig4.get_facecolor())
            print(f"Wrote: {out4}")
            plt.close(fig4)

    # ------------------------------------------------------------------ #
    # Print summary
    # ------------------------------------------------------------------ #
    if summary:
        print("\n--- Run Summary ---")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    print("\nVisualization complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _find_latest_prefix(out_dir: str) -> Optional[str]:
    """Find the most recently modified trajectory prefix in out_dir."""
    candidates = glob.glob(os.path.join(out_dir, "trajectory_*_optimized_nodes.txt"))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0].replace("_optimized_nodes.txt", "")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Cartographer loop closure results from carto_outputs/"
    )
    parser.add_argument(
        "--prefix", "-p",
        default=None,
        help="Path prefix for output files (without extension). "
             "Defaults to the most recently written trajectory in carto_outputs/.",
    )
    parser.add_argument(
        "--out-dir", "-o",
        default="carto_outputs",
        help="Directory to write PNG visualizations (default: carto_outputs/)",
    )
    args = parser.parse_args()

    prefix = args.prefix
    if prefix is None:
        prefix = _find_latest_prefix(args.out_dir)
        if prefix is None:
            print(
                "ERROR: No trajectory files found in carto_outputs/. "
                "Run run_loop_closure_slam.py first.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Auto-detected prefix: {prefix}")

    os.makedirs(args.out_dir, exist_ok=True)
    visualize(prefix=prefix, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
