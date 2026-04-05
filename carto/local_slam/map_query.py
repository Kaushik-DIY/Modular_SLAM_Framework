"""Map-query utilities used by the local scan matcher."""

import numpy as np


def _bilinear(img: np.ndarray, x: float, y: float):
    """Return the bilinearly interpolated cell value at floating-point pixel coordinates."""
    h, w = img.shape
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    # Bilinear interpolation needs the four surrounding cells to exist.
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return None

    dx = x - x0
    dy = y - y0
    v00 = img[y0, x0]
    v10 = img[y0, x1]
    v01 = img[y1, x0]
    v11 = img[y1, x1]
    return v00 * (1 - dx) * (1 - dy) + v10 * dx * (1 - dy) + v01 * (1 - dx) * dy + v11 * dx * dy


def score_pose(prob_img: np.ndarray, pts_g: np.ndarray, min_valid: int = 20) -> float:
    """Score a candidate pose by averaging interpolated map probabilities at scan-hit locations."""
    s = 0.0
    n = 0
    for x, y in pts_g:
        v = _bilinear(prob_img, x, y)
        if v is None:
            continue
        s += float(v)
        n += 1

    # Reject poses that barely overlap the map; otherwise a tiny in-bounds subset can look deceptively good.
    if n < min_valid:
        return -1e9
    return s / n
