from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import g2o  # type: ignore

from carto.common.types import Pose2
from carto.common.se2 import wrap_angle
from carto.pose_graph.constraint import (
    PoseGraphNode,
    PoseGraphSubmap,
    PoseGraphConstraint,
    INTRA_SUBMAP,
    INTER_SUBMAP,
)


# ---------------------------------------------------------------------------
# Vertex id mapping
# ---------------------------------------------------------------------------
# Submap and node ids are independent integer sequences, so map them onto a
# single disjoint g2o vertex-id space: submaps -> even, nodes -> odd.
def _submap_vid(submap_id: int) -> int:
    return 2 * int(submap_id)


def _node_vid(node_id: int) -> int:
    return 2 * int(node_id) + 1


def _se2(p: Pose2) -> "g2o.SE2":
    return g2o.SE2(float(p.x), float(p.y), float(p.theta))


def _vec_to_pose(v: np.ndarray) -> Pose2:
    return Pose2(float(v[0]), float(v[1]), float(wrap_angle(float(v[2]))))


class G2oBackend2D:
    """
    g2o SE2 pose-graph backend.

    Drop-in replacement for PyCeresBackend2D / SciPyBackend2D: same graph-mutation
    API and the same {(kind, id): Pose2} solve() contract, so it plugs straight
    into PoseGraph2D. Replaces the Python-based Ceres/SciPy solvers with the g2o
    C++ optimizer (VertexSE2 / EdgeSE2 / BlockSolverSE2 + Levenberg).

    Mirrors Cartographer's OptimizationProblem2D::Solve():
    1. INTRA_SUBMAP constraints (scan-to-submap, trusted, no robust kernel)
    2. INTER_SUBMAP constraints (loop closure, Huber robust kernel)
    3. LocalSlamPose consecutive-node regularization (the trajectory 'spine')

    Default weights from pose_graph.lua:
        local_slam_pose_translation_weight = 1e5
        local_slam_pose_rotation_weight    = 1e5
        huber_scale                        = 1e1
    """

    def __init__(
        self,
        huber_scale: float = 1e1,
        max_num_iterations: int = 50,
        local_slam_pose_translation_weight: float = 1e5,
        local_slam_pose_rotation_weight: float = 1e5,
        verbose: bool = False,
    ) -> None:
        # Graph state
        self.nodes: Dict[int, Pose2] = {}
        self.submaps: Dict[int, Pose2] = {}
        self.constraints: List[PoseGraphConstraint] = []

        # Local trajectory regularization data
        self.node_local_poses: Dict[int, Pose2] = {}
        self._node_insertion_order: List[int] = []

        self._fixed: Optional[Tuple[str, int]] = ("submap", 0)
        self._optimized: Dict[Tuple[str, int], Pose2] = {}
        self._last_report: Optional[str] = None

        self.huber_scale = float(huber_scale)
        self.max_num_iterations = int(max_num_iterations)
        self.local_slam_pose_translation_weight = float(local_slam_pose_translation_weight)
        self.local_slam_pose_rotation_weight = float(local_slam_pose_rotation_weight)
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------
    # Graph mutation API (matches PyCeresBackend2D)
    # ------------------------------------------------------------------

    def set_fixed(self, kind: str, idx: int):
        self._fixed = (str(kind), int(idx))

    def update_node_local_pose(self, node_id: int, local_pose: Pose2) -> None:
        nid = int(node_id)
        self.node_local_poses[nid] = local_pose
        self._node_insertion_order.append(nid)

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

    # ------------------------------------------------------------------
    # Solver helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_optimizer() -> "g2o.SparseOptimizer":
        optimizer = g2o.SparseOptimizer()
        linear_solver = None
        for name in (
            "LinearSolverEigenSE2",
            "LinearSolverCholmodSE2",
            "LinearSolverCSparseSE2",
            "LinearSolverDenseSE2",
        ):
            cls = getattr(g2o, name, None)
            if cls is None:
                continue
            try:
                linear_solver = cls()
                break
            except Exception:
                continue
        if linear_solver is None:
            raise RuntimeError("No usable g2o SE2 linear solver found.")
        block_solver = g2o.BlockSolverSE2(linear_solver)
        algorithm = g2o.OptimizationAlgorithmLevenberg(block_solver)
        optimizer.set_algorithm(algorithm)
        return optimizer

    @staticmethod
    def _information(t_weight: float, r_weight: float) -> np.ndarray:
        return np.diag(
            [
                max(float(t_weight), 1e-12),
                max(float(t_weight), 1e-12),
                max(float(r_weight), 1e-12),
            ]
        ).astype(np.float64)

    def _build_current_state(self) -> Dict[Tuple[str, int], Pose2]:
        out = {("submap", sid): p for sid, p in self.submaps.items()}
        out.update({("node", nid): p for nid, p in self.nodes.items()})
        return out

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(self, max_iters: int = 50):
        # Trivial cases: anchor missing or no constraints -> no optimization.
        if self._fixed is not None:
            kind, idx = self._fixed
            if (kind == "submap" and idx not in self.submaps) or \
               (kind == "node" and idx not in self.nodes):
                self._optimized = self._build_current_state()
                return self._optimized

        if len(self.constraints) == 0:
            self._optimized = self._build_current_state()
            return self._optimized

        optimizer = self._make_optimizer()

        # --- Vertices -------------------------------------------------
        for sid, pose in self.submaps.items():
            v = g2o.VertexSE2()
            v.set_id(_submap_vid(sid))
            v.set_estimate(_se2(pose))
            optimizer.add_vertex(v)

        for nid, pose in self.nodes.items():
            v = g2o.VertexSE2()
            v.set_id(_node_vid(nid))
            v.set_estimate(_se2(pose))
            optimizer.add_vertex(v)

        # Fix the anchor pose (first submap by default).
        if self._fixed is not None:
            fk, fidx = self._fixed
            fvid = _submap_vid(fidx) if fk == "submap" else _node_vid(fidx)
            anchor = optimizer.vertex(fvid)
            if anchor is not None:
                anchor.set_fixed(True)

        # --- 1+2. INTRA + INTER submap constraints --------------------
        for c in self.constraints:
            svid = _submap_vid(int(c.submap_id))
            nvid = _node_vid(int(c.node_id))
            vs = optimizer.vertex(svid)
            vn = optimizer.vertex(nvid)
            if vs is None or vn is None:
                raise KeyError(
                    f"Constraint references missing vertex pair submap={c.submap_id}, "
                    f"node={c.node_id}. Add the submap and node before the constraint."
                )

            e = g2o.EdgeSE2()
            e.set_vertex(0, vs)  # submap
            e.set_vertex(1, vn)  # node ; measurement z = T_submap^{-1} * T_node
            e.set_measurement(_se2(c.pose.relative_pose))
            e.set_information(
                self._information(c.pose.translation_weight, c.pose.rotation_weight)
            )
            if c.tag == INTER_SUBMAP and self.huber_scale > 0.0:
                rk = g2o.RobustKernelHuber()
                rk.set_delta(float(self.huber_scale))
                e.set_robust_kernel(rk)
            optimizer.add_edge(e)

        # --- 3. Local trajectory regularization (consecutive nodes) ---
        self._add_local_trajectory_regularization(optimizer)

        # --- Optimize -------------------------------------------------
        optimizer.initialize_optimization()
        optimizer.set_verbose(bool(self.verbose))
        iters = int(max_iters) if max_iters is not None else self.max_num_iterations
        optimizer.optimize(int(iters))
        self._last_report = f"g2o optimize: {len(self.constraints)} constraints, {iters} iters"

        # --- Extract results -----------------------------------------
        self._optimized = {}

        for sid in self.submaps.keys():
            key = ("submap", int(sid))
            if self._fixed is not None and key == self._fixed:
                self._optimized[key] = self.submaps[sid]
                continue
            v = optimizer.vertex(_submap_vid(sid))
            pose = _vec_to_pose(v.estimate().to_vector())
            self._optimized[key] = pose
            self.submaps[sid] = pose

        for nid in self.nodes.keys():
            key = ("node", int(nid))
            if self._fixed is not None and key == self._fixed:
                self._optimized[key] = self.nodes[nid]
                continue
            v = optimizer.vertex(_node_vid(nid))
            pose = _vec_to_pose(v.estimate().to_vector())
            self._optimized[key] = pose
            self.nodes[nid] = pose

        return self._optimized

    def _add_local_trajectory_regularization(self, optimizer: "g2o.SparseOptimizer") -> int:
        """Consecutive-node SE2 constraints (z_ij = local_i^{-1} * local_j).

        Mirrors optimization_problem_2d.cc lines 289-333: trusted local-SLAM
        measurements, high weight, no robust kernel. Preserves trajectory shape
        while allowing globally-consistent loop-closure deformation.
        """
        n_added = 0
        info = self._information(
            self.local_slam_pose_translation_weight,
            self.local_slam_pose_rotation_weight,
        )
        sorted_ids = sorted(self._node_insertion_order)
        for k in range(len(sorted_ids) - 1):
            nid_a = sorted_ids[k]
            nid_b = sorted_ids[k + 1]
            if nid_b != nid_a + 1:
                continue

            local_a = self.node_local_poses.get(nid_a)
            local_b = self.node_local_poses.get(nid_b)
            if local_a is None or local_b is None:
                continue

            va = optimizer.vertex(_node_vid(nid_a))
            vb = optimizer.vertex(_node_vid(nid_b))
            if va is None or vb is None:
                continue

            # z_ij = local_a^{-1} * local_b   (computed in SE2 algebra)
            z = _se2(local_a).inverse() * _se2(local_b)

            e = g2o.EdgeSE2()
            e.set_vertex(0, va)
            e.set_vertex(1, vb)
            e.set_measurement(z)
            e.set_information(info)
            optimizer.add_edge(e)
            n_added += 1

        return n_added

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_optimized_poses(self):
        return dict(self._optimized)

    def get_solver_report(self) -> Optional[str]:
        return self._last_report
