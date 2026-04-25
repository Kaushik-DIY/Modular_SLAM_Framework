from __future__ import annotations

import numpy as np


def read_intel_carmen_log(path: str):
    """
    Parser for the classic Intel Research Lab CARMEN logfile.

    Expected FLASER layout used by common intel.clf tooling:
        FLASER n r0 ... r(n-1) odom_x odom_y odom_theta ... timestamp ...

    Notes
    -----
    This parser is intentionally separate from the Freiburg parser because
    the Intel logfile conventions differ from the FR079 parser currently
    used in this workspace.
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

            # Conservative lower bound based on the commonly used intel.clf format.
            if len(parts) < (n + 9):
                continue

            try:
                ranges = np.asarray(parts[2:2 + n], dtype=float)

                base = 2 + n
                odom_x = float(parts[base + 0])
                odom_y = float(parts[base + 1])
                odom_th = float(parts[base + 2])

                # Common intel.clf converters use tokens[n + 8] as the timestamp.
                t = float(parts[n + 8])
            except Exception:
                continue

            scans.append(
                {
                    "ranges": ranges,
                    "odom": (odom_x, odom_y, odom_th),
                    # Intel parser currently exposes only odometry pose.
                    "laser_pose": (odom_x, odom_y, odom_th),
                    "t": t,
                }
            )

    return scans