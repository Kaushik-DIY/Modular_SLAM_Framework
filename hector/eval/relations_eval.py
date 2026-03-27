from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Relation2D:
    a: float  # timestamp (seconds)
    b: float  # timestamp (seconds)
    dx: float
    dy: float
    dtheta: float


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def se2_inv(x: np.ndarray) -> np.ndarray:
    """Inverse of SE(2) pose x=[x,y,theta]."""
    c, s = np.cos(x[2]), np.sin(x[2])
    R_T = np.array([[c, s], [-s, c]], dtype=float)
    t_inv = -R_T @ x[:2]
    return np.array([t_inv[0], t_inv[1], wrap_angle(-x[2])], dtype=float)


def se2_compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose SE(2) poses a ⊕ b."""
    ca, sa = np.cos(a[2]), np.sin(a[2])
    R = np.array([[ca, -sa], [sa, ca]], dtype=float)
    t = a[:2] + R @ b[:2]
    return np.array([t[0], t[1], wrap_angle(a[2] + b[2])], dtype=float)


def se2_between(xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
    """Relative transform xi^{-1} ⊕ xj."""
    return se2_compose(se2_inv(xi), xj)


def parse_relations(path: str) -> List[Relation2D]:
    """
    Freiburg fr079.relations (your file) looks like 8 columns:
      t_i  t_j  dx  dy  ?  ?  ?  dtheta
    Example:
      1216.630413 1217.490267 0.407950 -0.016330 0.000000 0.000000 0.000000 -0.069620

    We parse:
      a=t_i, b=t_j, dx=col2, dy=col3, dtheta=last_col (if 8 cols)
    """
    rels: List[Relation2D] = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                a = float(parts[0])
                b = float(parts[1])
                dx = float(parts[2])
                dy = float(parts[3])

                # Key: for your file (8 cols), dtheta is the last column.
                if len(parts) == 8:
                    dth = float(parts[7])
                else:
                    # fallback: assume 5th token is dtheta
                    dth = float(parts[4])
            except ValueError:
                continue

            rels.append(Relation2D(a=a, b=b, dx=dx, dy=dy, dtheta=dth))

    return rels


def nearest_index(timestamps: np.ndarray, t: float) -> int:
    """Return index of closest timestamp to t."""
    idx = int(np.searchsorted(timestamps, t))
    if idx <= 0:
        return 0
    if idx >= len(timestamps):
        return len(timestamps) - 1
    if abs(timestamps[idx] - t) < abs(timestamps[idx - 1] - t):
        return idx
    return idx - 1


def evaluate_time(
    traj: np.ndarray,
    stamps: np.ndarray,
    rels: List[Relation2D],
    convention: str = "ij",
    filter_out_of_range: bool = True,
    max_dt: float | None = None,
    debug: bool = False,
) -> dict:
    """
    convention:
      - "ij":  z_hat = between(x_i, x_j), z_gt = [dx,dy,dtheta]
      - "ji":  z_hat = between(x_j, x_i), z_gt = [dx,dy,dtheta]
      - "ij_inv_meas": z_hat = between(x_i, x_j), z_gt = inverse([dx,dy,dtheta])

    filter_out_of_range:
      If True, skip relations whose timestamps are outside [stamps[0], stamps[-1]].
      This is IMPORTANT for your case because your rels go to ~2258s while traj ends ~1426s.
    """
    tmin, tmax = float(stamps[0]), float(stamps[-1])

    trans_err = []
    rot_err = []

    used = 0
    skipped_range = 0
    skipped_dt = 0
    dt_list = []

    for r in rels:
        dt = abs(r.b - r.a)
        if max_dt is not None and dt > max_dt:
            skipped_dt += 1
            continue
        if filter_out_of_range:
            if r.a < tmin or r.a > tmax or r.b < tmin or r.b > tmax:
                skipped_range += 1
                continue

        i = nearest_index(stamps, r.a)
        j = nearest_index(stamps, r.b)

        # time association error diagnostics
        dt_list.append(abs(stamps[i] - r.a))
        dt_list.append(abs(stamps[j] - r.b))

        xi = traj[i]
        xj = traj[j]

        z_gt = np.array([r.dx, r.dy, r.dtheta], dtype=float)

        if convention == "ij":
            z_hat = se2_between(xi, xj)
        elif convention == "ji":
            z_hat = se2_between(xj, xi)
        elif convention == "ij_inv_meas":
            z_hat = se2_between(xi, xj)
            z_gt = se2_inv(z_gt)
        else:
            raise ValueError(f"Unknown convention: {convention}")

        if debug and used < 10:
            print("\nDEBUG REL", used)
            print("t_i,t_j:", r.a, r.b)
            print("matched stamps:", float(stamps[i]), float(stamps[j]))
            print("xi:", xi)
            print("xj:", xj)
            print("z_gt:", z_gt)
            print("z_hat:", z_hat)
            print("z_hat(deg):", z_hat[2] * 180/np.pi, " z_gt(deg):", z_gt[2] * 180/np.pi)

        # error = z_gt^{-1} ⊕ z_hat
        e = se2_between(z_gt, z_hat)
        e[2] = wrap_angle(e[2])

        trans_err.append(np.hypot(e[0], e[1]))
        rot_err.append(abs(e[2]))
        used += 1

    trans_err = np.array(trans_err, dtype=float)
    rot_err = np.array(rot_err, dtype=float)
    dt_arr = np.array(dt_list, dtype=float) if len(dt_list) else np.array([], dtype=float)

    out = {
        "convention": convention,
        "relations_total": len(rels),
        "relations_used": used,
        "relations_skipped_out_of_range": skipped_range,
        "rmse_trans_m": float(np.sqrt(np.mean(trans_err**2))) if used else float("nan"),
        "mean_trans_m": float(np.mean(trans_err)) if used else float("nan"),
        "rmse_rot_deg": float(np.sqrt(np.mean((rot_err * 180.0 / np.pi) ** 2))) if used else float("nan"),
        "mean_rot_deg": float(np.mean(rot_err) * 180.0 / np.pi) if used else float("nan"),
        "nearest_dt_mean_s": float(dt_arr.mean()) if dt_arr.size else float("nan"),
        "nearest_dt_max_s": float(dt_arr.max()) if dt_arr.size else float("nan"),
        "traj_time_min": float(tmin),
        "traj_time_max": float(tmax),
        "relations_skipped_dt": skipped_dt,
    }
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="outputs/trajectory_fr079.txt (x y theta)")
    ap.add_argument("--rels", required=True, help="datasets/freiburg/fr079.relations")
    ap.add_argument("--mode", choices=["time"], default="time")
    ap.add_argument("--stamps", required=True, help="outputs/timestamps_fr079.txt")
    ap.add_argument(
        "--convention",
        choices=["ij", "ji", "ij_inv_meas"],
        default="ij",
        help="Test transform direction / measurement convention",
    )
    ap.add_argument(
        "--no_filter",
        action="store_true",
        help="Do NOT filter relations outside the trajectory timestamp range (not recommended).",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Print extra debug info (first relations + time ranges).",
    )
    ap.add_argument(
    "--max_dt",
    type=float,
    default=None,
    help="Use only relations with |t_j - t_i| <= max_dt (seconds)"
    )
    args = ap.parse_args()

    traj = np.loadtxt(args.traj, dtype=float, comments="#")
    stamps = np.loadtxt(args.stamps, dtype=float, comments="#")
    rels = parse_relations(args.rels)

    if args.debug:
        print("DEBUG: first relations parsed")
        print(rels[0] if len(rels) > 0 else "No relations parsed!")
        print(rels[1] if len(rels) > 1 else "(only one relation)")
        rel_ts = np.array([r.a for r in rels] + [r.b for r in rels], dtype=float)
        print("DEBUG: rel time range:", float(rel_ts.min()), float(rel_ts.max()))
        print("DEBUG: pose time range:", float(stamps.min()), float(stamps.max()))

    res = evaluate_time(
        traj=traj,
        stamps=stamps,
        rels=rels,
        convention=args.convention,
        filter_out_of_range=(not args.no_filter),
        max_dt=args.max_dt,
        debug=args.debug,
    )

    # Pretty print
    for k in [
        "convention",
        "relations_total",
        "relations_used",
        "relations_skipped_out_of_range",
        "rmse_trans_m",
        "rmse_rot_deg",
        "mean_trans_m",
        "mean_rot_deg",
        "nearest_dt_mean_s",
        "nearest_dt_max_s",
        "traj_time_min",
        "traj_time_max",
    ]:
        print(f"{k}: {res[k]}")


if __name__ == "__main__":
    main()

