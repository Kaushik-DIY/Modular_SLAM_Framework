from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyceres  # type: ignore

from slam_core.common.types import Pose2
from slam_core.common.se2 import wrap_angle


def _pose_to_vec(p: Pose2) -> np.ndarray:
    return np.array([float(p.x), float(p.y), float(p.theta)], dtype=np.float64)


def _vec_to_pose(v: np.ndarray) -> Pose2:
    return Pose2(float(v[0]), float(v[1]), float(wrap_angle(float(v[2]))))


class LocalScanMatchCost2D(pyceres.CostFunction):
    """
    Cartographer-like 2D local scan-matching cost.

    One parameter block:
        [x, y, theta] in submap coordinates

    Residual structure:
    - occupied-space residual for each scan point
    - translation prior residuals (x, y)
    - rotation prior residual  (theta)

    Notes
    -----
    This mirrors the mathematical intent of Cartographer's local Ceres scan
    matcher:

    1. Occupied-space alignment against the active submap
    2. Translation prior around the extrapolated/predicted pose
    3. Rotation prior around the extrapolated/predicted pose

    It is intentionally restricted to the local 2D scan-to-submap problem.
    """

    def __init__(
        self,
        grid,
        points_local: np.ndarray,
        pred_pose_sub: Pose2,
        translation_weight: float,
        rotation_weight: float,
        invalid_point_residual: float = 1.0,
    ) -> None:
        pyceres.CostFunction.__init__(self)

        self.grid = grid
        self.points_local = np.asarray(points_local, dtype=np.float64)
        self.pred = _pose_to_vec(pred_pose_sub)

        self.translation_weight = float(translation_weight)
        self.rotation_weight = float(rotation_weight)
        self.invalid_point_residual = float(invalid_point_residual)

        n_points = int(self.points_local.shape[0])
        self.set_num_residuals(n_points + 3)
        self.set_parameter_block_sizes([3])

    def _interp_prob_and_grad(
        self,
        qx: np.ndarray,
        qy: np.ndarray,
    ):
        """
        Bilinear interpolation of occupancy probability and grid gradients.
        """
        gx = (qx - float(self.grid.origin_world[0])) / float(self.grid.res)
        gy = (qy - float(self.grid.origin_world[1])) / float(self.grid.res)

        ix = np.floor(gx).astype(int)
        iy = np.floor(gy).astype(int)
        fx = gx - ix
        fy = gy - iy

        valid = (
            (ix >= 0)
            & (ix + 1 < int(self.grid.w))
            & (iy >= 0)
            & (iy + 1 < int(self.grid.h))
        )

        p = np.zeros_like(qx, dtype=np.float64)
        dpx = np.zeros_like(qx, dtype=np.float64)
        dpy = np.zeros_like(qy, dtype=np.float64)

        if not np.any(valid):
            return p, dpx, dpy, valid

        prob = self.grid.probability().astype(np.float64)

        i0 = ix[valid]
        j0 = iy[valid]
        wx = fx[valid]
        wy = fy[valid]

        p00 = prob[j0, i0]
        p10 = prob[j0, i0 + 1]
        p01 = prob[j0 + 1, i0]
        p11 = prob[j0 + 1, i0 + 1]

        pv = (
            (1.0 - wx) * (1.0 - wy) * p00
            + wx * (1.0 - wy) * p10
            + (1.0 - wx) * wy * p01
            + wx * wy * p11
        )

        dpxv = (
            ((1.0 - wy) * (p10 - p00) + wy * (p11 - p01))
            / float(self.grid.res)
        )
        dpyv = (
            ((1.0 - wx) * (p01 - p00) + wx * (p11 - p10))
            / float(self.grid.res)
        )

        p[valid] = pv
        dpx[valid] = dpxv
        dpy[valid] = dpyv

        return p, dpx, dpy, valid

    def Evaluate(self, parameters, residuals, jacobians):
        pose = np.asarray(parameters[0], dtype=np.float64)
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])

        c = np.cos(th)
        s = np.sin(th)

        px = self.points_local[:, 0]
        py = self.points_local[:, 1]

        qx = c * px - s * py + x
        qy = s * px + c * py + y

        p, dpx, dpy, valid = self._interp_prob_and_grad(qx, qy)

        n_points = int(self.points_local.shape[0])

        # Occupied-space residuals.
        for i in range(n_points):
            if valid[i]:
                residuals[i] = float(1.0 - p[i])
            else:
                residuals[i] = float(self.invalid_point_residual)

        # Translation and rotation priors around the predicted pose.
        residuals[n_points + 0] = float(self.translation_weight * (x - self.pred[0]))
        residuals[n_points + 1] = float(self.translation_weight * (y - self.pred[1]))
        residuals[n_points + 2] = float(self.rotation_weight * wrap_angle(th - self.pred[2]))

        if jacobians is not None and len(jacobians) > 0 and jacobians[0] is not None:
            J = np.zeros((n_points + 3, 3), dtype=np.float64)

            if np.any(valid):
                pxv = px[valid]
                pyv = py[valid]

                dqx_dth = -s * pxv - c * pyv
                dqy_dth = c * pxv - s * pyv

                # Residual is r = 1 - p(q), so dr/dq = -dp/dq
                J_occ = np.stack(
                    [
                        -dpx[valid],
                        -dpy[valid],
                        -(dpx[valid] * dqx_dth + dpy[valid] * dqy_dth),
                    ],
                    axis=1,
                )

                valid_idx = np.flatnonzero(valid)
                J[valid_idx, :] = J_occ

            # Prior Jacobian.
            J[n_points + 0, 0] = float(self.translation_weight)
            J[n_points + 1, 1] = float(self.translation_weight)
            J[n_points + 2, 2] = float(self.rotation_weight)

            for r in range(n_points + 3):
                for cidx in range(3):
                    jacobians[0][r * 3 + cidx] = float(J[r, cidx])

        return True


@dataclass
class LocalPyCeresMatchResult:
    """
    Result of local PyCeres refinement in submap coordinates.
    """
    success: bool
    pose_sub: Pose2
    valid_points: int
    refined_score: float
    summary: Optional[object]
    reason: str = ""


class PyCeresLocalScanMatcher2D:
    """
    PyCeres-backed local 2D scan matcher.

    This class is designed to sit exactly where Cartographer uses the local
    Ceres scan matcher:
    - pred_pose_sub: prior center from the extrapolator
    - initial_pose_sub: optimization initializer, optionally improved by the
      correlative matcher
    - output pose_sub: refined local estimate in submap coordinates
    """

    def __init__(
        self,
        *,
        translation_weight: float,
        rotation_weight: float,
        max_num_iterations: int = 20,
        num_threads: int = 1,
        minimizer_progress_to_stdout: bool = False,
        function_tolerance: float = 1e-6,
        gradient_tolerance: float = 1e-10,
        parameter_tolerance: float = 1e-8,
        linear_solver_type: str = "DENSE_QR",
        invalid_point_residual: float = 1.0,
    ) -> None:
        self.translation_weight = float(translation_weight)
        self.rotation_weight = float(rotation_weight)
        self.max_num_iterations = int(max_num_iterations)
        self.num_threads = int(num_threads)
        self.minimizer_progress_to_stdout = bool(minimizer_progress_to_stdout)
        self.function_tolerance = float(function_tolerance)
        self.gradient_tolerance = float(gradient_tolerance)
        self.parameter_tolerance = float(parameter_tolerance)
        self.linear_solver_type = str(linear_solver_type)
        self.invalid_point_residual = float(invalid_point_residual)

        self._last_summary: Optional[object] = None

    def _score_pose_on_probability_grid(
        self,
        grid,
        points_local: np.ndarray,
        pose_sub: Pose2,
    ) -> tuple[float, int]:
        """
        Diagnostic occupied-space score and valid support count at a pose.
        """
        if points_local.size == 0:
            return -1.0, 0

        prob_grid = grid.probability().astype(np.float64)

        c = np.cos(float(pose_sub.theta))
        s = np.sin(float(pose_sub.theta))

        vals = []
        valid_points = 0

        for px, py in np.asarray(points_local, dtype=np.float64):
            x = c * float(px) - s * float(py) + float(pose_sub.x)
            y = s * float(px) + c * float(py) + float(pose_sub.y)

            gx, gy = grid.world_to_grid(x, y)
            if grid.in_bounds(gx, gy):
                vals.append(float(prob_grid[gy, gx]))
                valid_points += 1

        if not vals:
            return -1.0, 0

        return float(np.mean(vals)), int(valid_points)

    def match(
        self,
        *,
        grid,
        points_local: np.ndarray,
        pred_pose_sub: Pose2,
        initial_pose_sub: Pose2,
        min_valid_points: int,
    ) -> LocalPyCeresMatchResult:
        """
        Run local PyCeres refinement.

        Parameters
        ----------
        grid:
            Occupancy grid of the target submap.
        points_local:
            Scan points in the scanner/local frame.
        pred_pose_sub:
            Prior center from the extrapolator prediction expressed in submap frame.
        initial_pose_sub:
            Optimization initializer, usually:
              - coarse correlative pose if available, else
              - pred_pose_sub
        min_valid_points:
            Minimum in-bounds support required to accept the refined result.
        """
        pts = np.asarray(points_local, dtype=np.float64)
        if pts.shape[0] == 0:
            return LocalPyCeresMatchResult(
                success=False,
                pose_sub=pred_pose_sub,
                valid_points=0,
                refined_score=-1.0,
                summary=None,
                reason="empty_scan",
            )

        x = _pose_to_vec(initial_pose_sub).copy()

        problem = pyceres.Problem()
        cost = LocalScanMatchCost2D(
            grid=grid,
            points_local=pts,
            pred_pose_sub=pred_pose_sub,
            translation_weight=self.translation_weight,
            rotation_weight=self.rotation_weight,
            invalid_point_residual=self.invalid_point_residual,
        )
        problem.add_residual_block(cost, None, [x])

        options = pyceres.SolverOptions()
        options.max_num_iterations = int(self.max_num_iterations)
        options.minimizer_progress_to_stdout = bool(self.minimizer_progress_to_stdout)
        options.num_threads = int(self.num_threads)
        options.function_tolerance = float(self.function_tolerance)
        options.gradient_tolerance = float(self.gradient_tolerance)
        options.parameter_tolerance = float(self.parameter_tolerance)

        if hasattr(pyceres, self.linear_solver_type):
            options.linear_solver_type = getattr(pyceres, self.linear_solver_type)

        summary = pyceres.SolverSummary()
        pyceres.solve(options, problem, summary)
        self._last_summary = summary

        x[2] = float(wrap_angle(float(x[2])))
        pose_sub = _vec_to_pose(x)

        finite = np.all(np.isfinite(x))
        refined_score, valid_points = self._score_pose_on_probability_grid(
            grid=grid,
            points_local=pts,
            pose_sub=pose_sub,
        )

        if not finite:
            return LocalPyCeresMatchResult(
                success=False,
                pose_sub=pred_pose_sub,
                valid_points=int(valid_points),
                refined_score=float(refined_score),
                summary=summary,
                reason="nonfinite_solution",
            )

        if int(valid_points) < int(min_valid_points):
            return LocalPyCeresMatchResult(
                success=False,
                pose_sub=pred_pose_sub,
                valid_points=int(valid_points),
                refined_score=float(refined_score),
                summary=summary,
                reason="insufficient_valid_support",
            )

        return LocalPyCeresMatchResult(
            success=True,
            pose_sub=pose_sub,
            valid_points=int(valid_points),
            refined_score=float(refined_score),
            summary=summary,
            reason="ok",
        )

    def get_last_summary(self):
        return self._last_summary