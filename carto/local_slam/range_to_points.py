"""Conversion utilities from polar laser ranges to Cartesian scan points."""

import numpy as np


def ranges_to_points(
    ranges: np.ndarray,
    angle_min: float,
    angle_inc: float,
    rmin: float,
    rmax: float,
    stride: int = 1,
) -> np.ndarray:
    """Convert a range scan into 2D Cartesian points in the laser frame."""
    idx = np.arange(len(ranges), dtype=int)
    if stride > 1:
        # Subsampling is applied to both indices and ranges so angle assignment stays consistent.
        idx = idx[::stride]
        ranges = ranges[::stride]

    angles = angle_min + idx * angle_inc
    mask = np.isfinite(ranges) & (ranges >= rmin) & (ranges <= rmax)

    # Filter invalid and out-of-range beams before projection so the matcher only sees usable geometry.
    ranges = ranges[mask]
    angles = angles[mask]

    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return np.stack([x, y], axis=1)
