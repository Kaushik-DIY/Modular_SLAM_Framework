from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import least_squares

from carto.common.types import Pose2
from carto.common.se2 import wrap_angle
from carto.pose_graph.constraint import (
    PoseGraphNode,
    PoseGraphSubmap,
    PoseGraphConstraint,
    INTRA_SUBMAP,
    INTER_SUBMAP,
)


def _pose_to_vec(p: Pose2) -> np.ndarray:
    return np.array([float(p.x), float(p.y), float(p.theta)], dtype=float)


def _vec_to_pose(v: np.ndarray) -> Pose2:
    return Pose2(float(v[0]), float(v[1]), float(v[2]))


def _se2_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, ath = float(a[0]), float(a[1]), float(a[2])
    bx, by, bth = float(b[0]), float(b[1]), float(b[2])

    ca, sa = np.cos(ath), np.sin(ath)
    dx = bx - ax
    dy = by - ay

    x_rel = ca * dx + sa * dy
    y_rel = -sa * dx + ca * dy
    th_rel = wrap_angle(bth - ath)
    return np.array([x_rel, y_rel, th_rel], dtype=float)


def _apply_huber_block(residual_block: np.ndarray, delta: float) -> np.ndarray:
    """
    Block-wise Huber weighting, applied only to loop-closure constraints.

    This is the closest practical analogue of Cartographer's per-residual-block
    Huber loss when using SciPy instead of Ceres.
    """
    if delta <= 0.0:
        return residual_block

    norm = float(np.linalg.norm(residual_block))
    if norm <= delta or norm <= 1e-12:
        return residual_block

    weight = np.sqrt(delta / norm)
    return weight * residual_block


class SciPyBackend2D:
    def __init__(self, huber_scale: float = 1.0):
        self.nodes: Dict[int, Pose2] = {}
        self.submaps: Dict[int, Pose2] = {}
        self.constraints: List[PoseGraphConstraint] = []

        self._fixed: Optional[Tuple[str, int]] = ("submap", 0)
        self._optimized: Dict[Tuple[str, int], Pose2] = {}
        self._last_result = None

        self.huber_scale = float(huber_scale)

    def set_fixed(self, kind: str, idx: int):
        self._fixed = (str(kind), int(idx))

    def add_node(self, *args):
        if len(args) == 1 and isinstance(args[0], PoseGraphNode):
            n: PoseGraphNode = args[0]
            self.nodes[int(n.id)] = n.pose
            return
        if len(args) == 2 and isinstance(args[0], (int, np.integer)) and isinstance(args[1], Pose2):
            node_id, pose = int(args[0]), args[1]
            self.nodes[node_id] = pose
            return
        raise TypeError("add_node expects (PoseGraphNode) OR (node_id:int, pose:Pose2)")

    def add_submap(self, *args):
        if len(args) == 1 and isinstance(args[0], PoseGraphSubmap):
            sm: PoseGraphSubmap = args[0]
            self.submaps[int(sm.id)] = sm.pose
            return
        if len(args) == 2 and isinstance(args[0], (int, np.integer)) and isinstance(args[1], Pose2):
            sid, pose = int(args[0]), args[1]
            self.submaps[sid] = pose
            return
        raise TypeError("add_submap expects (PoseGraphSubmap) OR (submap_id:int, pose:Pose2)")

    def add_constraint(self, constraint: PoseGraphConstraint):
        self.constraints.append(constraint)

    def _pack_state(self) -> Tuple[np.ndarray, Dict[Tuple[str, int], slice]]:
        keys: List[Tuple[str, int]] = []
        keys += [("submap", sid) for sid in sorted(self.submaps.keys())]
        keys += [("node", nid) for nid in sorted(self.nodes.keys())]

        fixed = self._fixed
        slices: Dict[Tuple[str, int], slice] = {}

        x_list = []
        cursor = 0
        for k in keys:
            if fixed is not None and k == fixed:
                continue

            slices[k] = slice(cursor, cursor + 3)
            cursor += 3

            if k[0] == "submap":
                x_list.append(_pose_to_vec(self.submaps[k[1]]))
            else:
                x_list.append(_pose_to_vec(self.nodes[k[1]]))

        if len(x_list) == 0:
            return np.zeros((0,), dtype=float), slices

        return np.concatenate(x_list, axis=0), slices

    def _pose_for_key(self, key: Tuple[str, int], x: np.ndarray, sl: Dict[Tuple[str, int], slice]) -> np.ndarray:
        if self._fixed is not None and key == self._fixed:
            kind, idx = key
            if kind == "submap":
                return _pose_to_vec(self.submaps[idx])
            return _pose_to_vec(self.nodes[idx])

        if key not in sl:
            raise KeyError(
                f"Constraint references missing variable {key}. "
                f"Did you call add_node/add_submap before adding the constraint?"
            )
        return x[sl[key]]

    def solve(self, max_iters: int = 50):
        if self._fixed is not None:
            k, idx = self._fixed
            if (k == "submap" and idx not in self.submaps) or (k == "node" and idx not in self.nodes):
                self._optimized = {("submap", sid): p for sid, p in self.submaps.items()}
                self._optimized.update({("node", nid): p for nid, p in self.nodes.items()})
                return self._optimized

        if len(self.constraints) == 0:
            self._optimized = {("submap", sid): p for sid, p in self.submaps.items()}
            self._optimized.update({("node", nid): p for nid, p in self.nodes.items()})
            return self._optimized

        x0, sl = self._pack_state()

        def residuals(x: np.ndarray) -> np.ndarray:
            res_all = []

            for c in self.constraints:
                submap_key = ("submap", int(c.submap_id))
                node_key = ("node", int(c.node_id))

                T_submap = self._pose_for_key(submap_key, x, sl)
                T_node = self._pose_for_key(node_key, x, sl)

                pred = _se2_between(T_submap, T_node)
                meas = _pose_to_vec(c.pose.relative_pose)

                r = pred - meas
                r[2] = wrap_angle(r[2])

                weighted = np.array(
                    [
                        np.sqrt(float(c.pose.translation_weight)) * r[0],
                        np.sqrt(float(c.pose.translation_weight)) * r[1],
                        np.sqrt(float(c.pose.rotation_weight)) * r[2],
                    ],
                    dtype=float,
                )

                if c.tag == INTER_SUBMAP:
                    weighted = _apply_huber_block(weighted, self.huber_scale)

                res_all.append(weighted)

            return np.concatenate(res_all, axis=0)

        out = least_squares(
            residuals,
            x0,
            method="trf",
            max_nfev=int(max_iters),
            loss="linear",
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        self._last_result = out
        xopt = out.x

        self._optimized = {}

        if self._fixed is not None:
            kind, idx = self._fixed
            p = self.submaps[idx] if kind == "submap" else self.nodes[idx]
            self._optimized[(kind, idx)] = p

        for sid in self.submaps.keys():
            k = ("submap", int(sid))
            if self._fixed is not None and k == self._fixed:
                continue
            self._optimized[k] = _vec_to_pose(self._pose_for_key(k, xopt, sl))

        for nid in self.nodes.keys():
            k = ("node", int(nid))
            if self._fixed is not None and k == self._fixed:
                continue
            self._optimized[k] = _vec_to_pose(self._pose_for_key(k, xopt, sl))

        return self._optimized

    def get_optimized_poses(self):
        return dict(self._optimized)