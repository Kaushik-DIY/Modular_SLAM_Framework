"""SE(2) helpers on fusion_core.Pose2 shared by the batch and real-time runners."""
from __future__ import annotations

import math

import fusion_core as fc


def _rel(a: fc.Pose2, b: fc.Pose2) -> fc.Pose2:
    """T_a^{-1} ∘ T_b."""
    return a.inverse().compose(b)


def _fc_pose(p) -> fc.Pose2:
    """Convert a Python Pose2-like object into the fusion_core Pose2 type."""
    return fc.Pose2(float(p.x), float(p.y), float(p.theta))


def _rel_sane(rel: fc.Pose2, pred: fc.Pose2, max_m: float, max_rad: float,
              abs_max_m: float = 1e9) -> bool:
    """Loop-edge sanity: verified rel pose must roughly agree with the graph
    prediction. Rejects rotational-ambiguity false positives that score well
    on the matcher but disagree wildly with the (drift-bounded) odometry.

    ``abs_max_m`` additionally caps the absolute magnitude of the loop
    transform: a genuine same-place revisit overlaps at a small relative pose,
    so a several-metre transform is a corridor slide-lock even when it agrees
    with a drifted prediction (drift-independent gate; off by default)."""
    return (math.hypot(rel.x - pred.x, rel.y - pred.y) <= max_m
            and abs(math.atan2(math.sin(rel.theta - pred.theta),
                               math.cos(rel.theta - pred.theta))) <= max_rad
            and math.hypot(rel.x, rel.y) <= abs_max_m)
