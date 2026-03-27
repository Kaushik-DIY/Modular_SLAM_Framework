import numpy as np

def ranges_to_points(ranges: np.ndarray,
                     angle_min: float,
                     angle_inc: float,
                     rmin: float,
                     rmax: float,
                     stride: int = 1) -> np.ndarray:
    idx = np.arange(len(ranges), dtype=int)
    if stride > 1:
        idx = idx[::stride]
        ranges = ranges[::stride]

    angles = angle_min + idx * angle_inc
    mask = np.isfinite(ranges) & (ranges >= rmin) & (ranges <= rmax)

    ranges = ranges[mask]
    angles = angles[mask]

    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return np.stack([x, y], axis=1)