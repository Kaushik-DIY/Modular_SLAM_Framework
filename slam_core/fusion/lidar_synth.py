"""
Synthetic 2D LiDAR from a TUM depth image.

RTAB_inspired_implementation_plan.md §11.2. Pick a horizontal row of the depth
image (default: the principal-point row, closest to the optical axis), back-
project each valid column into the camera frame, and treat the result as a 2D
scan on the ground plane. Range noise is added to mimic a real planar LiDAR.

Output convention: points are returned in a 2D sensor frame as (forward, left),
where forward = camera +Z (optical axis) and left = camera -X. This matches the
"x forward, y left" planar convention used by the LiDAR front-ends.

The noise model is tunable per the Phase 1 risk row (CLAUDE.md §8): synthesized
LiDAR may be unrepresentative, so ``noise_sigma`` is exposed for calibration.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def synthesize_2d_scan(
    depth: np.ndarray,
    K: np.ndarray,
    row_index: Optional[int] = None,
    num_beams: int = 360,
    range_min: float = 0.3,
    range_max: float = 10.0,
    noise_sigma: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Synthesize a 2D LiDAR scan from a metric depth image.

    Parameters
    ----------
    depth : (H, W) array
        Depth in metres. Zero / non-finite / out-of-range pixels are dropped.
    K : (3, 3) array
        Camera intrinsics ``[[fx,0,cx],[0,fy,cy],[0,0,1]]``.
    row_index : int, optional
        Image row to slice. Defaults to ``round(cy)``.
    num_beams : int
        Upper bound on returned points; denser rows are uniformly decimated.
    range_min, range_max : float
        Valid range gate in metres.
    noise_sigma : float
        Std-dev of additive radial (range) Gaussian noise in metres. Set 0 to
        disable.
    rng : np.random.Generator, optional
        Source of randomness; a default generator is used if omitted.

    Returns
    -------
    (M, 2) float array
        Scan points in the sensor frame as (forward, left). Empty (0, 2) array
        if the row has no valid returns.
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D (H, W), got shape {depth.shape}")
    H, W = depth.shape

    K = np.asarray(K, dtype=np.float64)
    fx, cx = K[0, 0], K[0, 2]

    if row_index is None:
        row_index = int(round(K[1, 2]))  # cy
    row_index = int(np.clip(row_index, 0, H - 1))

    z = depth[row_index, :]
    u = np.arange(W, dtype=np.float64)

    valid = np.isfinite(z) & (z > 0.0) & (z >= range_min) & (z <= range_max)
    z = z[valid]
    u = u[valid]
    if z.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    # Back-project the row: X is camera-right, Z is forward (optical axis).
    X = (u - cx) * z / fx
    forward = z
    left = -X
    pts = np.stack([forward, left], axis=1)

    if noise_sigma and noise_sigma > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        r = np.linalg.norm(pts, axis=1)
        bearing = np.arctan2(pts[:, 1], pts[:, 0])
        r = r + rng.normal(0.0, noise_sigma, size=r.shape)
        pts = np.stack([r * np.cos(bearing), r * np.sin(bearing)], axis=1)

    if num_beams and pts.shape[0] > num_beams:
        idx = np.linspace(0, pts.shape[0] - 1, num_beams).astype(int)
        pts = pts[idx]

    return pts
