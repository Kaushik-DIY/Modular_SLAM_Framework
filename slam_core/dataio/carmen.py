from __future__ import annotations

import numpy as np


def read_carmen_log(path: str):
    """
    Minimal parser for Freiburg CARMEN logs.

    Reads only FLASER lines.

    Returns a list of dicts:
      {
        "ranges": np.ndarray,
        "odom": (odom_x, odom_y, odom_theta),
        "laser_pose": (x, y, theta),
        "t": timestamp
      }

    Notes
    -----
    Standard FLASER layout is:

        FLASER n r0 ... r(n-1) x y theta odom_x odom_y odom_theta
        tv rv forward_safety_dist side_safety_dist turn_axis timestamp host

    The first pose triplet is preserved as `laser_pose` for diagnostics.
    The second pose triplet is the actual odometry pose and is exposed as `odom`.
    """
    scans = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "FLASER":
                continue

            try:
                n = int(parts[1])
            except (ValueError, IndexError):
                continue

            # Need at least:
            # FLASER n [n ranges] x y theta odom_x odom_y odom_theta
            min_len = 2 + n + 6
            if len(parts) < min_len:
                continue

            try:
                ranges = np.asarray(parts[2:2 + n], dtype=float)

                base = 2 + n

                laser_x = float(parts[base + 0])
                laser_y = float(parts[base + 1])
                laser_th = float(parts[base + 2])

                odom_x = float(parts[base + 3])
                odom_y = float(parts[base + 4])
                odom_th = float(parts[base + 5])

                # Freiburg CARMEN logs place timestamp near the end.
                # In the current dataset this matches the paper duration well.
                t = float(parts[-3])
            except Exception:
                continue

            scans.append(
                {
                    "ranges": ranges,
                    "odom": (odom_x, odom_y, odom_th),
                    "laser_pose": (laser_x, laser_y, laser_th),
                    "t": t,
                }
            )

    return scans