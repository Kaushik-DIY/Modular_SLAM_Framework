from __future__ import annotations

import numpy as np

from slam_core.common.types import Pose2
from slam_core.common.se2 import wrap_angle


class CartoRefinementProblem:
    def __init__(self, grid, pts_local, pred_pose_sub, min_points=20, w_trans=1.0, w_rot=1.0):
        self.grid = grid
        self.pts_local = np.asarray(pts_local, dtype=float)
        self.pred_pose_sub = np.asarray(pred_pose_sub, dtype=float).reshape(3)
        self.min_points = int(min_points)
        self.w_trans = float(w_trans)
        self.w_rot = float(w_rot)
        self.last_num_valid = 0

    def _interp_prob_and_grad(self, x, y):
        gx = (x - self.grid.origin_world[0]) / self.grid.res
        gy = (y - self.grid.origin_world[1]) / self.grid.res

        ix = np.floor(gx).astype(int)
        iy = np.floor(gy).astype(int)
        fx = gx - ix
        fy = gy - iy

        valid = (
            (ix >= 0) & (ix + 1 < self.grid.w) &
            (iy >= 0) & (iy + 1 < self.grid.h)
        )

        p = np.zeros_like(x, dtype=float)
        dpx = np.zeros_like(x, dtype=float)
        dpy = np.zeros_like(y, dtype=float)

        if not np.any(valid):
            return p, dpx, dpy, valid

        prob = self.grid.probability()

        i0 = ix[valid]
        j0 = iy[valid]
        wx = fx[valid]
        wy = fy[valid]

        p00 = prob[j0, i0]
        p10 = prob[j0, i0 + 1]
        p01 = prob[j0 + 1, i0]
        p11 = prob[j0 + 1, i0 + 1]

        pv = (
            (1 - wx) * (1 - wy) * p00
            + wx * (1 - wy) * p10
            + (1 - wx) * wy * p01
            + wx * wy * p11
        )

        dpxv = (((1 - wy) * (p10 - p00) + wy * (p11 - p01)) / self.grid.res)
        dpyv = (((1 - wx) * (p01 - p00) + wx * (p11 - p10)) / self.grid.res)

        p[valid] = pv
        dpx[valid] = dpxv
        dpy[valid] = dpyv

        return p, dpx, dpy, valid

    def compute_r_J(self, xvec):
        x, y, th = map(float, np.asarray(xvec).reshape(3))
        c, s = np.cos(th), np.sin(th)

        qx = c * self.pts_local[:, 0] - s * self.pts_local[:, 1] + x
        qy = s * self.pts_local[:, 0] + c * self.pts_local[:, 1] + y

        p, dpx, dpy, valid = self._interp_prob_and_grad(qx, qy)

        self.last_num_valid = int(np.count_nonzero(valid))

        if self.last_num_valid < self.min_points:
            r = np.array([10.0], dtype=float)
            J = np.zeros((1, 3), dtype=float)
            return r, J

        r_occ = 1.0 - p[valid]

        px = self.pts_local[valid, 0]
        py = self.pts_local[valid, 1]
        dqx_dth = -s * px - c * py
        dqy_dth = c * px - s * py

        J_occ = np.stack(
            [
                -dpx[valid],
                -dpy[valid],
                -(dpx[valid] * dqx_dth + dpy[valid] * dqy_dth),
            ],
            axis=1,
        )

        pred = self.pred_pose_sub
        r_prior = np.array(
            [
                self.w_trans * (x - pred[0]),
                self.w_trans * (y - pred[1]),
                self.w_rot * wrap_angle(th - pred[2]),
            ],
            dtype=float,
        )

        J_prior = np.array(
            [
                [self.w_trans, 0.0, 0.0],
                [0.0, self.w_trans, 0.0],
                [0.0, 0.0, self.w_rot],
            ],
            dtype=float,
        )

        r = np.concatenate([r_occ, r_prior], axis=0)
        J = np.vstack([J_occ, J_prior])
        return r, J


def refine_pose_submap(refine_solver, problem: CartoRefinementProblem, initial_pose_sub: Pose2) -> Pose2:
    x0 = np.array([initial_pose_sub.x, initial_pose_sub.y, initial_pose_sub.theta], dtype=float)
    x_opt = refine_solver.solve(x0, problem.compute_r_J).reshape(3)
    x_opt[2] = wrap_angle(x_opt[2])
    return Pose2(float(x_opt[0]), float(x_opt[1]), float(x_opt[2]))
