# dataio/carmen.py
from __future__ import annotations
import numpy as np


def read_carmen_log(path: str):
    """
    Robust minimal parser for Freiburg CARMEN logs.
    Reads ONLY FLASER lines.

    Returns a list of dicts:
      {
        "ranges": np.ndarray,
        "odom": (x, y, theta),
        "t": timestamp
      }
    """
    scans = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line:
                continue

            parts = line.strip().split()
            if not parts:
                continue

            # Only parse FLASER lines
            if parts[0] != "FLASER":
                continue

            # Number of beams
            try:
                n = int(parts[1])
            except (ValueError, IndexError):
                continue

            # Sanity check length
            # FLASER n r0..r(n-1) x y theta ... timestamp
            if len(parts) < 2 + n + 3:
                continue

            try:
                ranges = np.array(parts[2:2 + n], dtype=float)

                base = 2 + n
                x = float(parts[base])
                y = float(parts[base + 1])
                th = float(parts[base + 2])

                # Timestamp is usually last entry
                t = float(parts[-3])
            except Exception:
                t = float(parts[-1])
                continue

            scans.append({
                "ranges": ranges,
                "odom": (x, y, th),
                "t": t
            })

    return scans
