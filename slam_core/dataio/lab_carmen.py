from __future__ import annotations

import numpy as np


def read_lab_carmen_log(path: str):
    """
    Parser for the custom minimal lab dataset export.

    Expected line format
    --------------------
        FLASER_MIN <timestamp> <num_readings> <r1> ... <rn>

    Returned scan dictionaries follow the same broad structure used by the
    existing Freiburg/Intel readers so downstream runners can stay simple.
    Because this export contains no odometry or laser pose, those fields are
    intentionally set to ``None``.
    """
    scans = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "FLASER_MIN":
                continue

            if len(parts) < 3:
                continue

            try:
                t = float(parts[1])
                n = int(parts[2])
            except (ValueError, IndexError):
                continue

            min_len = 3 + n
            if len(parts) < min_len:
                continue

            try:
                ranges = np.asarray(parts[3:3 + n], dtype=float)
            except Exception:
                continue

            scans.append(
                {
                    "ranges": ranges,
                    "odom": None,
                    "laser_pose": None,
                    "t": t,
                }
            )

    return scans