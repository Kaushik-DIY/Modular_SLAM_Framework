from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class Pose2:
    x: float
    y: float
    theta: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta], dtype=float)
    
    @dataclass
    class RangeData2D:
        t: float
        ranges: np.ndarray          # shape (N,)
        angle_min: float            # rad
        angle_inc: float            # rad
        odom: Optional[Pose2] = None