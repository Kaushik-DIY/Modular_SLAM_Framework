# eval/pose_graph_lc.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from slam.pose_graph import PoseGraph2D, se2_between, wrap_angle


@dataclass
class Relation2D:
    a: float
    b: float
    dx: float
    dy: float
    dtheta: float


def load_traj(traj_path: str) -> np.ndarray:
    # supports header lines starting with '#'
    traj = np.loadtxt(traj_path, comments="#")
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
    assert traj.shape[1] >= 3, "trajectory must have at least 3 columns: x y theta"
    return traj[:, :3].astype(float)


def load_stamps(stamps_path: str) -> np.ndarray:
    ts = np.loadtxt(stamps_path, comments="#")
    ts = np.array(ts, dtype=float).reshape(-1)
    return ts


def load_relations(rel_path: str) -> list[Relation2D]:
    rels: list[Relation2D] = []
    with open(rel_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Freiburg relations typically have 8 columns:
            # t_i t_j dx dy 0 0 0 dtheta
            if len(parts) < 5:
                continue
            a = float(parts[0])
            b = float(parts[1])
            dx = float(parts[2])
            dy = float(parts[3])
            dtheta = float(parts[-1])  # last is yaw/heading increment
            rels.append(Relation2D(a=a, b=b, dx=dx, dy=dy, dtheta=dtheta))
    return rels


def nearest_index(stamps: np.ndarray, t: float) -> int:
    # stamps assumed sorted
    idx = int(np.searchsorted(stamps, t))
    if idx <= 0:
        return 0
    if idx >= len(stamps):
        return len(stamps) - 1
    # choose closer of idx-1 and idx
    if abs(stamps[idx] - t) < abs(stamps[idx - 1] - t):
        return idx
    return idx - 1


def build_and_optimize(
    traj_init: np.ndarray,
    stamps: np.ndarray,
    rels: list[Relation2D],
    max_dt: float,
    odom_sig_xy: float,
    odom_sig_th: float,
    lc_sig_xy: float,
    lc_sig_th: float,
    pgo_iters: int,
    pgo_damping: float,
    debug: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    traj_init: Nx3 (x,y,theta) initial nodes
    stamps  : Nx1 time stamps aligned to traj rows
    rels    : Freiburg relative constraints (time-based)
    Returns:
      traj_opt: Nx3 optimized poses
      stats   : dict
    """

    N = traj_init.shape[0]
    assert len(stamps) == N, f"stamps ({len(stamps)}) != traj ({N})"

    # Build pose graph
    g = PoseGraph2D()
    for k in range(N):
        g.add_node(traj_init[k].copy())

    Omega_odom = np.diag([
        1.0 / (odom_sig_xy**2),
        1.0 / (odom_sig_xy**2),
        1.0 / (odom_sig_th**2),
    ])

    Omega_lc = np.diag([
        1.0 / (lc_sig_xy**2),
        1.0 / (lc_sig_xy**2),
        1.0 / (lc_sig_th**2),
    ])

    # Sequential edges (keep local structure)
    seq_edges = 0
    for k in range(N - 1):
        z = se2_between(traj_init[k], traj_init[k + 1])  # relative from initial
        g.add_edge(k, k + 1, z, Omega_odom)
        seq_edges += 1

    # Loop edges from relations
    used = 0
    skipped_time = 0
    skipped_dt = 0
    debug_printed = 0
    skipped_yaw = 0

    # yaw_gate_rad = np.deg2rad(15.0)

    for r in rels:
        if r.a < stamps[0] or r.a > stamps[-1] or r.b < stamps[0] or r.b > stamps[-1]:
            skipped_time += 1
            continue

        i = nearest_index(stamps, r.a)
        j = nearest_index(stamps, r.b)

        # enforce ordering i<j for stability (still ok for graph)
        if j == i:
            skipped_dt += 1
            continue

        dt_i = abs(stamps[i] - r.a)
        dt_j = abs(stamps[j] - r.b)
        if dt_i > max_dt or dt_j > max_dt:
            skipped_dt += 1
            continue

        # ---------- OPTION B: YAW OUTLIER GATE (NEW) ----------
        # xi = traj_init[i]
        # xj = traj_init[j]
        # z_hat = se2_between(xi, xj)  # predicted relative from current trajectory

        # yaw_err = wrap_angle(z_hat[2] - z[2])
        # if abs(yaw_err) > yaw_gate_rad:
        #     skipped_yaw += 1
        #     continue
        # ------------------------------------------------------

        z = np.array([r.dx, r.dy, r.dtheta], dtype=float)
        z[2] = wrap_angle(z[2])

        g.add_edge(i, j, z, Omega_lc)
        used += 1

        if debug and debug_printed < 10:
            xi = traj_init[i]
            xj = traj_init[j]
            z_hat = se2_between(xi, xj)
            print(f"\nDEBUG REL {debug_printed}")
            print("t_i,t_j:", r.a, r.b)
            print("matched stamps:", stamps[i], stamps[j])
            print("i,j:", i, j)
            print("xi:", xi)
            print("xj:", xj)
            print("z_gt:", z)
            print("z_hat:", z_hat)
            print("z_hat(deg):", np.rad2deg(z_hat[2]), " z_gt(deg):", np.rad2deg(z[2]))
            debug_printed += 1

    if debug:
        print("\nDEBUG: rel time range:", min(rr.a for rr in rels), max(rr.b for rr in rels))
        print("DEBUG: pose time range:", float(stamps.min()), float(stamps.max()))

    # Optimize
    g.optimize(iters=pgo_iters, damping=pgo_damping, fix_first=True)

    traj_opt = np.array(g.nodes, dtype=float)
    traj_opt[:, 2] = np.array([wrap_angle(th) for th in traj_opt[:, 2]])

    stats = {
        "N": N,
        "seq_edges": seq_edges,
        "rels_total": len(rels),
        "rels_used": used,
        "rels_skipped_time": skipped_time,
        "rels_skipped_dt": skipped_dt,
        "max_dt": max_dt,
        # "rels_skipped_yaw": skipped_yaw,   # NEW
        # "yaw_gate_deg": np.rad2deg(yaw_gate_rad),  # NEW
    }
    return traj_opt, stats


def save_overlay_plot(traj_a: np.ndarray, traj_b: np.ndarray, out_path: str, label_a="no_lc", label_b="with_lc"):
    plt.figure()
    plt.plot(traj_a[:, 0], traj_a[:, 1], linewidth=1.0, label=label_a)
    plt.plot(traj_b[:, 0], traj_b[:, 1], linewidth=1.0, label=label_b)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("Trajectory overlay (baseline vs loop-closed)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="baseline trajectory (x y theta)")
    ap.add_argument("--stamps", required=True, help="timestamps aligned to trajectory rows")
    ap.add_argument("--rels", required=True, help="Freiburg .relations file")
    ap.add_argument("--out_traj", default="outputs/trajectory_fr079_with_lc.txt")
    ap.add_argument("--out_overlay", default="outputs/overlay_no_lc_vs_lc.png")

    ap.add_argument("--max_dt", type=float, default=2.0, help="max time mismatch (s) for matching relations to stamps")

    ap.add_argument("--odom_sig_xy", type=float, default=0.10)
    ap.add_argument("--odom_sig_th_deg", type=float, default=5.0)

    ap.add_argument("--lc_sig_xy", type=float, default=0.02)
    ap.add_argument("--lc_sig_th_deg", type=float, default=1.0)

    ap.add_argument("--pgo_iters", type=int, default=15)
    ap.add_argument("--pgo_damping", type=float, default=1e-6)

    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    traj0 = load_traj(args.traj)
    stamps = load_stamps(args.stamps)
    rels = load_relations(args.rels)

    traj_lc, stats = build_and_optimize(
        traj_init=traj0,
        stamps=stamps,
        rels=rels,
        max_dt=args.max_dt,
        odom_sig_xy=args.odom_sig_xy,
        odom_sig_th=np.deg2rad(args.odom_sig_th_deg),
        lc_sig_xy=args.lc_sig_xy,
        lc_sig_th=np.deg2rad(args.lc_sig_th_deg),
        pgo_iters=args.pgo_iters,
        pgo_damping=args.pgo_damping,
        debug=args.debug,
    )

    np.savetxt(args.out_traj, traj_lc, fmt="%.6f", header="x y theta")
    save_overlay_plot(traj0, traj_lc, args.out_overlay, "no_lc", "with_lc")

    print("PGO done.")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("Saved:", args.out_traj)
    print("Saved:", args.out_overlay)


if __name__ == "__main__":
    main()
