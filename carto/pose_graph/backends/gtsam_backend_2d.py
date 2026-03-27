# carto/pose_graph/backends/gtsam_backend_2d.py

from __future__ import annotations
import numpy as np
import gtsam

from carto.common.types import Pose2


def _key(kind: str, idx: int) -> int:
    """Safe GTSAM key creation (avoids segfault in some python bindings)."""
    idx = int(idx)
    if kind == "node":
        return int(gtsam.Symbol('x', idx).key())   
    if kind == "submap":
        return int(gtsam.Symbol('s', idx).key())
    raise ValueError(f"Unknown key kind: {kind}")


def _assert_finite_pose(p: Pose2, name: str):
    arr = np.array([p.x, p.y, p.theta], dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} has non-finite values: {arr}")


class GTSAMBackend2D:
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self._added_keys: set[int] = set()

        # results
        self.result: gtsam.Values | None = None

    def add_node(self, node_id: int, pose_world: Pose2):
        _assert_finite_pose(pose_world, "node_pose_world")

        k = _key("node", node_id)
        if k in self._added_keys:
            return

        # Create Pose2 using explicit floats
        p = gtsam.Pose2(float(pose_world.x), float(pose_world.y), float(pose_world.theta))

        # Insert initial estimate

        import math

        if not (math.isfinite(pose_world.x) and
                math.isfinite(pose_world.y) and
                math.isfinite(pose_world.theta)):
            print("INVALID NODE POSE:", pose_world)
            raise ValueError("Non-finite pose detected")

        self.initial.insert(k, p)
        self._added_keys.add(k)

    def add_submap(self, submap_id: int, pose_world: Pose2):
        _assert_finite_pose(pose_world, "submap_pose_world")

        k = _key("submap", submap_id)
        if k in self._added_keys:
            return

        p = gtsam.Pose2(float(pose_world.x), float(pose_world.y), float(pose_world.theta))

        import math

        if not (math.isfinite(pose_world.x) and
                math.isfinite(pose_world.y) and
                math.isfinite(pose_world.theta)):
            print("INVALID NODE POSE:", pose_world)
            raise ValueError("Non-finite pose detected")

        self.initial.insert(k, p)
        self._added_keys.add(k)

    def add_between(self, kind_a: str, id_a: int, kind_b: str, id_b: int,
                    relative_pose: Pose2,
                    sig_xy: float, sig_theta: float):

        _assert_finite_pose(relative_pose, "relative_pose")

        ka = _key(kind_a, id_a)
        kb = _key(kind_b, id_b)

        # diagonal noise model
        # IMPORTANT: some GTSAM python wheels segfault with numpy arrays here.
        sig_xy = float(sig_xy)
        sig_theta = float(sig_theta)

        if not (sig_xy > 0.0 and sig_theta > 0.0):
            raise ValueError(f"Invalid sigmas: sig_xy={sig_xy}, sig_theta={sig_theta}")

        sigmas = gtsam.Vector3(sig_xy, sig_xy, sig_theta)
        model = gtsam.noiseModel.Diagonal.Sigmas(sigmas)

        meas = gtsam.Pose2(
            float(relative_pose.x),
            float(relative_pose.y),
            float(relative_pose.theta)
        )

        # Extra sanity (cheap but prevents undefined behavior)
        if int(ka) == int(kb):
            raise ValueError(f"BetweenFactorPose2 got identical keys: {ka} == {kb}")

        self.graph.add(gtsam.BetweenFactorPose2(int(ka), int(kb), meas, model))

    def solve(self, max_iters: int = 50):
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(int(max_iters))
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, params)
        self.result = optimizer.optimize()
        return self.result