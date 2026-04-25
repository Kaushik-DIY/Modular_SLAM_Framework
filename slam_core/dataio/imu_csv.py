from __future__ import annotations

import csv
from typing import List, Dict, Any


def read_imu_csv(path: str) -> List[Dict[str, Any]]:
    """
    Read the exported IMU CSV file produced from the lab ROS bag.

    Columns
    -------
    timestamp, frame_id, qx, qy, qz, qw, wx, wy, wz, ax, ay, az
    """
    rows: List[Dict[str, Any]] = []

    with open(path, "r", newline="", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    {
                        "timestamp": float(row["timestamp"]),
                        "frame_id": row["frame_id"],
                        "qx": float(row["qx"]),
                        "qy": float(row["qy"]),
                        "qz": float(row["qz"]),
                        "qw": float(row["qw"]),
                        "wx": float(row["wx"]),
                        "wy": float(row["wy"]),
                        "wz": float(row["wz"]),
                        "ax": float(row["ax"]),
                        "ay": float(row["ay"]),
                        "az": float(row["az"]),
                    }
                )
            except Exception:
                continue

    return rows