from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import List
import numpy as np


def _sliding_max_forward_1d(arr: np.ndarray, width: int) -> np.ndarray:
    """
    Compute:
        out[i] = max(arr[i : i + width])
    with clipping at the end of the array.

    This matches the forward-looking window behavior needed by Cartographer-style
    precomputation grids for branch-and-bound upper bounds.
    """
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    n = int(a.shape[0])
    if n == 0:
        return a.copy()
    if width <= 1:
        return a.copy()

    padded = np.concatenate(
        [a, np.full((width - 1,), -np.inf, dtype=np.float32)],
        axis=0,
    )
    out = np.empty((n,), dtype=np.float32)
    dq = deque()

    for j in range(padded.shape[0]):
        while dq and padded[dq[-1]] <= padded[j]:
            dq.pop()
        dq.append(j)

        start = j - width + 1
        if start >= 0:
            while dq and dq[0] < start:
                dq.popleft()
            if start < n:
                out[start] = padded[dq[0]]

    return out


def _forward_max_filter_2d(grid: np.ndarray, width: int) -> np.ndarray:
    """
    Compute a 2D forward-looking max filter:
        out[y, x] = max(grid[y:y+width, x:x+width])

    This is the precomputation structure described in the Cartographer paper
    for branch-and-bound upper-bound scoring.
    """
    g = np.asarray(grid, dtype=np.float32)
    if g.ndim != 2:
        raise ValueError(f"Expected 2D grid, got shape {g.shape}")
    if width <= 1:
        return g.copy()

    # First pass: horizontal forward maxima.
    row_max = np.empty_like(g, dtype=np.float32)
    for y in range(g.shape[0]):
        row_max[y, :] = _sliding_max_forward_1d(g[y, :], width)

    # Second pass: vertical forward maxima.
    out = np.empty_like(g, dtype=np.float32)
    for x in range(g.shape[1]):
        out[:, x] = _sliding_max_forward_1d(row_max[:, x], width)

    return out


@dataclass(frozen=True)
class PrecomputationGrid2D:
    """
    Cartographer-style precomputation grid for a fixed branch width.

    Each cell stores the maximum probability in the forward-looking width x width
    box beginning at that cell.
    """
    width: int
    values: np.ndarray

    def get_value(self, gy: int, gx: int) -> float:
        if gy < 0 or gx < 0 or gy >= self.values.shape[0] or gx >= self.values.shape[1]:
            return 0.0
        return float(self.values[gy, gx])


class PrecomputationGridStack2D:
    """
    Stack of precomputation grids for branch-and-bound search.

    Level i corresponds to width = 2**i, matching the Cartographer paper and
    implementation.
    """

    def __init__(self, prob_grid: np.ndarray, branch_and_bound_depth: int) -> None:
        depth = int(branch_and_bound_depth)
        if depth < 1:
            raise ValueError("branch_and_bound_depth must be >= 1")

        base = np.asarray(prob_grid, dtype=np.float32)
        if base.ndim != 2:
            raise ValueError(f"Expected 2D probability grid, got shape {base.shape}")

        self._grids: List[PrecomputationGrid2D] = []
        for i in range(depth):
            width = 1 << i
            values = _forward_max_filter_2d(base, width)
            self._grids.append(PrecomputationGrid2D(width=width, values=values))

    def max_depth(self) -> int:
        return len(self._grids) - 1

    def get(self, depth: int) -> PrecomputationGrid2D:
        depth = int(np.clip(depth, 0, len(self._grids) - 1))
        return self._grids[depth]