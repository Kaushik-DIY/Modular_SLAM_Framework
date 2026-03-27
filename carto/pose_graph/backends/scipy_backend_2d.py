from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import least_squares

from carto.common.types import Pose2
from carto.common.se2 import wrap_angle
from carto.pose_graph.constraint import PoseGraphNode, PoseGraphSubmap, PoseGraphConstraint


def _pose_to_vec(p: Pose2) -> np.ndarray:
    return np.array([float(p.x), float(p.y), float(p.theta)], dtype=float)

def _vec_to_pose(v: np.ndarray) -> Pose2:
    return Pose2(float(v[0]), float(v[1]), float(v[2]))

def _se2_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """returns: a^{-1} ⊕ b  for a,b in world, each [x,y,theta]."""
    ax, ay, ath = float(a[0]), float(a[1]), float(a[2])
    bx, by, bth = float(b[0]), float(b[1]), float(b[2])

    ca, sa = np.cos(ath), np.sin(ath)
    dx = bx - ax
    dy = by - ay
    x_rel =  ca * dx + sa * dy
    y_rel = -sa * dx + ca * dy
    th_rel = wrap_angle(bth - ath)
    return np.array([x_rel, y_rel, th_rel], dtype=float)


class SciPyBackend2D:
    """
    Pure-Python (SciPy) backend:
      - variables: nodes + submaps (SE2)
      - constraints: PoseGraphConstraint with 3x3 information
      - solve: nonlinear least squares (trf default, optional robust loss)
    """

    def __init__(self):
        self.nodes: Dict[int, Pose2] = {}
        self.submaps: Dict[int, Pose2] = {}
        self.constraints: List[PoseGraphConstraint] = []

        self._fixed: Optional[Tuple[str, int]] = ("submap", 0)  # gauge anchor
        self._optimized: Dict[Tuple[str, int], Pose2] = {}
        self._last_result = None

    def set_fixed(self, kind: str, idx: int):
        self._fixed = (str(kind), int(idx))

    # --- supports BOTH calling styles ---
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

    def add_between(self, kind_a: str, id_a: int, kind_b: str, id_b: int,
                    relative_pose: Pose2, sig_xy: float, sig_theta: float):
        info = np.diag([
            1.0 / (float(sig_xy) ** 2),
            1.0 / (float(sig_xy) ** 2),
            1.0 / (float(sig_theta) ** 2),
        ])
        self.add_constraint(PoseGraphConstraint(
            type_from=str(kind_a),
            id_from=int(id_a),
            type_to=str(kind_b),
            id_to=int(id_b),
            relative_pose=relative_pose,
            information=info
        ))

    def _pack_state(self) -> Tuple[np.ndarray, Dict[Tuple[str, int], slice], List[Tuple[str, int]]]:
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
            return np.zeros((0,), dtype=float), slices, keys

        return np.concatenate(x_list, axis=0), slices, keys

    def _pose_for_key(self, key: Tuple[str, int], x: np.ndarray, sl: Dict[Tuple[str, int], slice]) -> np.ndarray:
        # fixed anchor
        if self._fixed is not None and key == self._fixed:
            kind, idx = key
            if kind == "submap":
                if idx not in self.submaps:
                    raise KeyError(f"Fixed key {key} not in submaps yet.")
                return _pose_to_vec(self.submaps[idx])
            if kind == "node":
                if idx not in self.nodes:
                    raise KeyError(f"Fixed key {key} not in nodes yet.")
                return _pose_to_vec(self.nodes[idx])
            raise ValueError(f"Unknown kind in fixed key: {key}")

        # free variable
        if key not in sl:
            # This is the most common “gotcha”: constraint references variable never added.
            raise KeyError(
                f"Constraint references missing variable {key}. "
                f"Did you call add_node/add_submap before adding the constraint?"
            )
        return x[sl[key]]

    def solve(
        self,
        max_iters: int = 50,
        method: str = "trf",
        loss: str = "linear",        # "linear", "huber", "soft_l1", "cauchy", "arctan"
        f_scale: float = 1.0,        # only used for robust losses
    ):
        # Need an anchor variable present (otherwise gauge freedom).
        if self._fixed is not None:
            k, idx = self._fixed
            if (k == "submap" and idx not in self.submaps) or (k == "node" and idx not in self.nodes):
                # If anchor doesn't exist yet, don't optimize; just return current.
                self._optimized = {("submap", sid): p for sid, p in self.submaps.items()}
                self._optimized.update({("node", nid): p for nid, p in self.nodes.items()})
                return self._optimized

        if len(self.constraints) == 0:
            self._optimized = {("submap", sid): p for sid, p in self.submaps.items()}
            self._optimized.update({("node", nid): p for nid, p in self.nodes.items()})
            return self._optimized

        x0, sl, _ = self._pack_state()

        # sqrt information via Cholesky
        chol_list = []
        for c in self.constraints:
            info = np.asarray(c.information, dtype=float)
            info = 0.5 * (info + info.T)
            info = info + 1e-12 * np.eye(3)
            L = np.linalg.cholesky(info)  # info = L L^T
            chol_list.append(L)

        def residuals(x: np.ndarray) -> np.ndarray:
            res_all = []
            for c, L in zip(self.constraints, chol_list):
                akey = (str(c.type_from), int(c.id_from))
                bkey = (str(c.type_to), int(c.id_to))

                Ta = self._pose_for_key(akey, x, sl)
                Tb = self._pose_for_key(bkey, x, sl)

                pred = _se2_between(Ta, Tb)
                meas = _pose_to_vec(c.relative_pose)

                r = pred - meas
                r[2] = wrap_angle(r[2])

                # Whitening: r^T info r == || L^T r ||^2
                res_all.append(L.T @ r)

            return np.concatenate(res_all, axis=0)

        # Auto method selection: "lm" only valid if m >= n
        # Here m = 3 * num_constraints, n = 3 * num_variables_free
        if method == "lm":
            m = 3 * len(self.constraints)
            n = int(x0.size)
            if m < n or n == 0:
                method = "trf"

        out = least_squares(
            residuals,
            x0,
            method=method,
            max_nfev=int(max_iters),
            loss=str(loss),
            f_scale=float(f_scale),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        self._last_result = out
        xopt = out.x

        # write optimized poses
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