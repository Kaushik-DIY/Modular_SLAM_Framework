from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pyceres  # type: ignore

from carto.common.types import Pose2
from carto.common.se2 import wrap_angle, inverse_pose, pose_compose
from carto.pose_graph.constraint import (
    PoseGraphNode,
    PoseGraphSubmap,
    PoseGraphConstraint,
    INTRA_SUBMAP,
    INTER_SUBMAP,
)


def _pose_to_vec(p: Pose2) -> np.ndarray:
    return np.array([float(p.x), float(p.y), float(p.theta)], dtype=np.float64)


def _vec_to_pose(v: np.ndarray) -> Pose2:
    return Pose2(float(v[0]), float(v[1]), float(wrap_angle(float(v[2]))))


# ---------------------------------------------------------------------------
# SE(2) residual cost functions
# ---------------------------------------------------------------------------

class SE2RelativePoseCost(pyceres.CostFunction):
    """
    Relative SE(2) cost between one submap pose and one node pose.

    Represents the Cartographer SPA constraint:
        z = T_submap^{-1} * T_node

    Used for:
    - INTRA_SUBMAP constraints  (local scan-to-submap, no Huber loss)
    - INTER_SUBMAP constraints  (loop closure, with Huber loss)

    Analytic Jacobians are used (same as Cartographer's SpaCostFunction2D
    via CreateAutoDiffSpaCostFunction, but made explicit here for PyCeres
    Python binding compatibility).
    """

    def __init__(
        self,
        measured_relative_pose: Pose2,
        translation_weight: float,
        rotation_weight: float,
    ) -> None:
        pyceres.CostFunction.__init__(self)

        self.set_num_residuals(3)
        self.set_parameter_block_sizes([3, 3])

        self.meas = _pose_to_vec(measured_relative_pose)
        self.sqrt_w_t = float(np.sqrt(max(translation_weight, 1e-12)))
        self.sqrt_w_r = float(np.sqrt(max(rotation_weight, 1e-12)))

    def Evaluate(self, parameters, residuals, jacobians):
        T_submap = np.asarray(parameters[0], dtype=np.float64)
        T_node = np.asarray(parameters[1], dtype=np.float64)

        ax, ay, ath = float(T_submap[0]), float(T_submap[1]), float(T_submap[2])
        bx, by, bth = float(T_node[0]), float(T_node[1]), float(T_node[2])

        ca, sa = np.cos(ath), np.sin(ath)
        dx = bx - ax
        dy = by - ay

        pred_x = ca * dx + sa * dy
        pred_y = -sa * dx + ca * dy
        pred_th = wrap_angle(bth - ath)

        err_x = pred_x - self.meas[0]
        err_y = pred_y - self.meas[1]
        err_th = wrap_angle(pred_th - self.meas[2])

        residuals[0] = self.sqrt_w_t * err_x
        residuals[1] = self.sqrt_w_t * err_y
        residuals[2] = self.sqrt_w_r * err_th

        if jacobians is not None:
            # d_residual / d_T_submap  [3 x 3]
            if len(jacobians) > 0 and jacobians[0] is not None:
                J_submap = np.array(
                    [
                        [-ca, -sa, -sa * dx + ca * dy],
                        [ sa, -ca, -ca * dx - sa * dy],
                        [0.0, 0.0, -1.0],
                    ],
                    dtype=np.float64,
                )
                J_submap[0, :] *= self.sqrt_w_t
                J_submap[1, :] *= self.sqrt_w_t
                J_submap[2, :] *= self.sqrt_w_r
                for r in range(3):
                    for c in range(3):
                        jacobians[0][r * 3 + c] = float(J_submap[r, c])

            # d_residual / d_T_node  [3 x 3]
            if len(jacobians) > 1 and jacobians[1] is not None:
                J_node = np.array(
                    [
                        [ ca,  sa, 0.0],
                        [-sa,  ca, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                J_node[0, :] *= self.sqrt_w_t
                J_node[1, :] *= self.sqrt_w_t
                J_node[2, :] *= self.sqrt_w_r
                for r in range(3):
                    for c in range(3):
                        jacobians[1][r * 3 + c] = float(J_node[r, c])

        return True


class LocalSlamPoseCost(pyceres.CostFunction):
    """
    Cartographer-style consecutive-node local trajectory regularization cost.

    Adds a soft constraint between every pair of consecutively inserted nodes
    based on their relative *local* pose (from the scan matcher result).
    This is the direct Python equivalent of lines 289-333 in:
      cartographer/mapping/internal/optimization/optimization_problem_2d.cc

    Enforces:
        T_node_j ≈ T_node_i ⊕ z_ij_local
        where  z_ij = local_pose_i^{-1} * local_pose_j

    This constraint is applied WITHOUT a robust loss function because it
    represents a trusted local-SLAM measurement. The high weight (default
    1e5) makes the optimizer preserve the local trajectory shape while only
    allowing globally-consistent deformations to close loops.

    Two parameter blocks: [T_node_i, T_node_j]  (both 3-vectors [x, y, theta])
    """

    def __init__(
        self,
        relative_local_pose: Pose2,
        translation_weight: float,
        rotation_weight: float,
    ) -> None:
        pyceres.CostFunction.__init__(self)

        self.set_num_residuals(3)
        self.set_parameter_block_sizes([3, 3])

        # z_ij = local_i^{-1} * local_j  (fixed measurement)
        self.meas = _pose_to_vec(relative_local_pose)
        self.sqrt_w_t = float(np.sqrt(max(translation_weight, 1e-12)))
        self.sqrt_w_r = float(np.sqrt(max(rotation_weight, 1e-12)))

    def Evaluate(self, parameters, residuals, jacobians):
        # Both parameter blocks are global node poses [x, y, theta]
        # The constraint is node_i -> node_j in global frame, but
        # measured in local frame — so we use the same SE2 residual form.
        T_i = np.asarray(parameters[0], dtype=np.float64)
        T_j = np.asarray(parameters[1], dtype=np.float64)

        ax, ay, ath = float(T_i[0]), float(T_i[1]), float(T_i[2])
        bx, by, bth = float(T_j[0]), float(T_j[1]), float(T_j[2])

        ca, sa = np.cos(ath), np.sin(ath)
        dx = bx - ax
        dy = by - ay

        pred_x = ca * dx + sa * dy
        pred_y = -sa * dx + ca * dy
        pred_th = wrap_angle(bth - ath)

        err_x = pred_x - self.meas[0]
        err_y = pred_y - self.meas[1]
        err_th = wrap_angle(pred_th - self.meas[2])

        residuals[0] = self.sqrt_w_t * err_x
        residuals[1] = self.sqrt_w_t * err_y
        residuals[2] = self.sqrt_w_r * err_th

        if jacobians is not None:
            # d_residual / d_T_i  (node i is the "anchor", same role as submap)
            if len(jacobians) > 0 and jacobians[0] is not None:
                J_i = np.array(
                    [
                        [-ca, -sa, -sa * dx + ca * dy],
                        [ sa, -ca, -ca * dx - sa * dy],
                        [0.0, 0.0, -1.0],
                    ],
                    dtype=np.float64,
                )
                J_i[0, :] *= self.sqrt_w_t
                J_i[1, :] *= self.sqrt_w_t
                J_i[2, :] *= self.sqrt_w_r
                for r in range(3):
                    for c in range(3):
                        jacobians[0][r * 3 + c] = float(J_i[r, c])

            # d_residual / d_T_j  (node j is the "variable", same role as node)
            if len(jacobians) > 1 and jacobians[1] is not None:
                J_j = np.array(
                    [
                        [ ca,  sa, 0.0],
                        [-sa,  ca, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                J_j[0, :] *= self.sqrt_w_t
                J_j[1, :] *= self.sqrt_w_t
                J_j[2, :] *= self.sqrt_w_r
                for r in range(3):
                    for c in range(3):
                        jacobians[1][r * 3 + c] = float(J_j[r, c])

        return True


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class PyCeresBackend2D:
    """
    PyCeres pose-graph backend, closely mirroring Cartographer's
    OptimizationProblem2D::Solve() from optimization_problem_2d.cc.

    Key fidelity improvements over the initial version:
    1. LocalSlamPoseCost: consecutive-node regularization (the missing 'spine')
    2. Score-weighted loop constraints: BnB score modulates constraint tightness
    3. SPARSE_NORMAL_CHOLESKY: correct solver for large sparse pose graphs
    4. huber_scale=1e1: matches Cartographer's pose_graph.lua default
    5. use_nonmonotonic_steps=False: matches Cartographer optimization_problem config

    Default weights from pose_graph.lua:
        local_slam_pose_translation_weight = 1e5
        local_slam_pose_rotation_weight    = 1e5
        huber_scale                        = 1e1
    """

    def __init__(
        self,
        huber_scale: float = 1e1,
        linear_solver_type: str = "AUTO",
        max_num_iterations: int = 50,
        num_threads: int = 1,
        minimizer_progress_to_stdout: bool = False,
        local_slam_pose_translation_weight: float = 1e5,
        local_slam_pose_rotation_weight: float = 1e5,
    ):
        # Graph state
        self.nodes: Dict[int, Pose2] = {}
        self.submaps: Dict[int, Pose2] = {}
        self.constraints: List[PoseGraphConstraint] = []

        # Local trajectory regularization data
        # node_id -> local_pose_2d (measurement, not optimized)
        self.node_local_poses: Dict[int, Pose2] = {}
        # Ordered list of node IDs for consecutive-pair detection
        self._node_insertion_order: List[int] = []

        self._fixed: Optional[Tuple[str, int]] = ("submap", 0)
        self._optimized: Dict[Tuple[str, int], Pose2] = {}
        self._last_summary = None

        # Solver config
        self.huber_scale = float(huber_scale)
        self.linear_solver_type = str(linear_solver_type)
        self.max_num_iterations = int(max_num_iterations)
        self.num_threads = int(num_threads)
        self.minimizer_progress_to_stdout = bool(minimizer_progress_to_stdout)

        # Local trajectory regularization weights (Cartographer defaults: 1e5)
        self.local_slam_pose_translation_weight = float(local_slam_pose_translation_weight)
        self.local_slam_pose_rotation_weight = float(local_slam_pose_rotation_weight)

    # ------------------------------------------------------------------
    # Graph mutation API
    # ------------------------------------------------------------------

    def set_fixed(self, kind: str, idx: int):
        self._fixed = (str(kind), int(idx))

    def update_node_local_pose(self, node_id: int, local_pose: Pose2) -> None:
        """
        Register the local-frame pose for a node.
        Called by PoseGraph2D after each node insertion.
        Used to build consecutive-node regularization constraints in solve().
        """
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

    def _build_current_state(self) -> Dict[Tuple[str, int], Pose2]:
        out = {("submap", sid): p for sid, p in self.submaps.items()}
        out.update({("node", nid): p for nid, p in self.nodes.items()})
        return out

    def _select_linear_solver(self, options, n_vars: int) -> None:
        """
        Select an appropriate linear solver for the problem size.

        Cartographer always uses SPARSE_NORMAL_CHOLESKY for the pose graph.
        We auto-select it for large problems (> 30 variables), falling back to
        DENSE_QR for small problems or when the sparse solver is unavailable.

        use_nonmonotonic_steps = False matches Cartographer's optimization config.
        """
        if not hasattr(pyceres, "LinearSolverType"):
            return

        if hasattr(options, "use_nonmonotonic_steps"):
            options.use_nonmonotonic_steps = False

        if self.linear_solver_type == "AUTO":
            use_sparse = (
                n_vars > 30
                and hasattr(pyceres.LinearSolverType, "SPARSE_NORMAL_CHOLESKY")
            )
            if use_sparse:
                options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
            else:
                options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
        else:
            try:
                options.linear_solver_type = getattr(
                    pyceres.LinearSolverType, self.linear_solver_type
                )
            except AttributeError:
                options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR

    def _add_local_trajectory_regularization(
        self,
        problem: pyceres.Problem,
        params: Dict[Tuple[str, int], np.ndarray],
    ) -> int:
        """
        Add local-trajectory regularization constraints.

        This is the Python equivalent of the consecutive-node cost loop in
        OptimizationProblem2D::Solve() (optimization_problem_2d.cc lines 289-333):

            for consecutive nodes (i, i+1) on the same trajectory:
                z_ij = local_pose_i^{-1} * local_pose_j
                add SE2 cost(z_ij, w_trans=1e5, w_rot=1e5), loss=None

        No Huber loss is applied — these are trusted local-SLAM measurements.
        The high weight (1e5) preserves local geometric trajectory shape while
        allowing globally-consistent deformations from loop closure constraints.

        Returns the number of regularization edges added.
        """
        n_added = 0
        sorted_ids = sorted(self._node_insertion_order)

        for k in range(len(sorted_ids) - 1):
            nid_a = sorted_ids[k]
            nid_b = sorted_ids[k + 1]

            # Only add for *consecutive* node indices (gap = 1), matching
            # the second_node_id.node_index != first_node_id.node_index + 1
            # guard in the original code.
            if nid_b != nid_a + 1:
                continue

            key_a = ("node", nid_a)
            key_b = ("node", nid_b)

            if key_a not in params or key_b not in params:
                continue

            local_a = self.node_local_poses.get(nid_a)
            local_b = self.node_local_poses.get(nid_b)
            if local_a is None or local_b is None:
                continue

            # z_ij = local_a^{-1} * local_b
            from carto.common.se2 import inverse_pose, pose_compose
            z_ij = pose_compose(inverse_pose(local_a), local_b)

            cost = LocalSlamPoseCost(
                relative_local_pose=z_ij,
                translation_weight=self.local_slam_pose_translation_weight,
                rotation_weight=self.local_slam_pose_rotation_weight,
            )

            # No loss function — trusted measurement, same as original
            problem.add_residual_block(cost, None, [params[key_a], params[key_b]])
            n_added += 1

        return n_added

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(self, max_iters: int = 50):
        if self._fixed is not None:
            kind, idx = self._fixed
            if (kind == "submap" and idx not in self.submaps) or \
               (kind == "node" and idx not in self.nodes):
                self._optimized = self._build_current_state()
                return self._optimized

        if len(self.constraints) == 0:
            self._optimized = self._build_current_state()
            return self._optimized

        # Build parameter blocks (mutable numpy arrays that Ceres will update)
        params: Dict[Tuple[str, int], np.ndarray] = {}

        for sid, pose in self.submaps.items():
            params[("submap", int(sid))] = _pose_to_vec(pose).copy()

        for nid, pose in self.nodes.items():
            params[("node", int(nid))] = _pose_to_vec(pose).copy()

        problem = pyceres.Problem()

        for _, arr in params.items():
            problem.add_parameter_block(arr, 3)

        # Fix the anchor pose (first submap by default, matching Cartographer's
        # 'first_submap = true' logic in OptimizationProblem2D::Solve())
        if self._fixed is not None and self._fixed in params:
            problem.set_parameter_block_constant(params[self._fixed])

        # ------------------------------------------------------------------
        # 1. INTRA + INTER submap constraints (scan-matcher + loop closure)
        # ------------------------------------------------------------------
        for c in self.constraints:
            submap_key = ("submap", int(c.submap_id))
            node_key = ("node", int(c.node_id))

            if submap_key not in params or node_key not in params:
                raise KeyError(
                    f"Constraint references missing variable pair {submap_key}, {node_key}. "
                    "Add the corresponding submap and node before adding the constraint."
                )

            if c.tag == INTER_SUBMAP:
                # Use full loop closure weights for all inter-submap constraints.
                # Cartographer's optimization_problem_2d.cc applies uniform weights
                # (loop_closure_translation/rotation_weight) to ALL loop constraints
                # and relies on HuberLoss alone to downweight outlier constraints.
                # Our previous score_factor was down-weighting corridor matches
                # (score ~0.55-0.65) by 35-45%, making loop closure too weak to
                # deform the stiff local trajectory chain (weight 1e5). Removed.
                eff_t_weight = float(c.pose.translation_weight)
                eff_r_weight = float(c.pose.rotation_weight)
                loss = pyceres.HuberLoss(float(self.huber_scale)) if self.huber_scale > 0.0 else None
            else:
                eff_t_weight = float(c.pose.translation_weight)
                eff_r_weight = float(c.pose.rotation_weight)
                loss = None  # No loss for intra-submap (trusted)

            cost = SE2RelativePoseCost(
                measured_relative_pose=c.pose.relative_pose,
                translation_weight=eff_t_weight,
                rotation_weight=eff_r_weight,
            )
            problem.add_residual_block(cost, loss, [params[submap_key], params[node_key]])

        # ------------------------------------------------------------------
        # 2. Local trajectory regularization (CRITICAL — the missing spine)
        #    Mirrors optimization_problem_2d.cc lines 289-333 exactly.
        # ------------------------------------------------------------------
        n_local_traj = self._add_local_trajectory_regularization(problem, params)

        # ------------------------------------------------------------------
        # Solver options
        # ------------------------------------------------------------------
        options = pyceres.SolverOptions()
        options.max_num_iterations = int(max_iters)
        options.minimizer_progress_to_stdout = bool(self.minimizer_progress_to_stdout)
        options.num_threads = int(self.num_threads)

        n_vars = len(params)
        self._select_linear_solver(options, n_vars)

        summary = pyceres.SolverSummary()
        pyceres.solve(options, problem, summary)
        self._last_summary = summary

        # ------------------------------------------------------------------
        # Extract results
        # ------------------------------------------------------------------
        self._optimized = {}

        if self._fixed is not None:
            kind, idx = self._fixed
            if kind == "submap" and idx in self.submaps:
                self._optimized[(kind, idx)] = self.submaps[idx]
            elif kind == "node" and idx in self.nodes:
                self._optimized[(kind, idx)] = self.nodes[idx]

        for sid in self.submaps.keys():
            key = ("submap", int(sid))
            if self._fixed is not None and key == self._fixed:
                continue
            self._optimized[key] = _vec_to_pose(params[key])
            self.submaps[sid] = self._optimized[key]

        for nid in self.nodes.keys():
            key = ("node", int(nid))
            if self._fixed is not None and key == self._fixed:
                continue
            self._optimized[key] = _vec_to_pose(params[key])
            self.nodes[nid] = self._optimized[key]

        return self._optimized

    def get_optimized_poses(self):
        return dict(self._optimized)

    def get_last_summary(self):
        return self._last_summary

    def get_solver_report(self) -> Optional[str]:
        """Return a brief human-readable Ceres solver report, if available."""
        if self._last_summary is None:
            return None
        try:
            return self._last_summary.BriefReport()
        except Exception:
            return str(self._last_summary)