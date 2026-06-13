from __future__ import annotations

import csv
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


def read_lab_hybrid_lidar_csv(path: str):
    """
    Parser for the hybrid lab LiDAR CSV export.

    Expected columns
    ----------------
        timestamp, frame_id, angle_min, angle_max, angle_increment,
        range_min, range_max, num_ranges, ranges

    The ranges column contains space-separated range values.
    """
    scans = []

    with open(path, "r", newline="", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n = int(row.get("num_ranges", "0"))
                ranges = np.asarray(str(row["ranges"]).split(), dtype=float)
                if n > 0 and len(ranges) != n:
                    continue
                scans.append(
                    {
                        "ranges": ranges,
                        "odom": None,
                        "laser_pose": None,
                        "t": float(row["timestamp"]),
                    }
                )
            except Exception:
                continue

    return scans
