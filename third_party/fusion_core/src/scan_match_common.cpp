#include "scan_match_common.h"

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

namespace fusion {

Pose2 refine_pose_lm(const float* prob, int W, int H, double ox, double oy,
                     double res, const MatX2d& pts, const Pose2& initial,
                     const Pose2& prior, const RefineParams& rp) {
    const Eigen::Vector3d pred(prior.x, prior.y, prior.theta);
    Eigen::Vector3d x(initial.x, initial.y, initial.theta);
    const double lam = rp.damping;
    const int n_pts = static_cast<int>(pts.rows());

    for (int it = 0; it < rp.iters; ++it) {
        const double c = std::cos(x[2]), s = std::sin(x[2]);
        Eigen::Matrix3d H_mat = Eigen::Matrix3d::Zero();
        Eigen::Vector3d g = Eigen::Vector3d::Zero();
        int n_valid = 0;

        for (int i = 0; i < n_pts; ++i) {
            const double px = pts(i, 0), py = pts(i, 1);
            const double qx = c * px - s * py + x[0];
            const double qy = s * px + c * py + x[1];
            const double gx = (qx - ox) / res, gy = (qy - oy) / res;
            const int ix = static_cast<int>(std::floor(gx));
            const int iy = static_cast<int>(std::floor(gy));
            if (ix < 0 || ix + 1 >= W || iy < 0 || iy + 1 >= H) continue;
            const double wx = gx - ix, wy = gy - iy;
            const double p00 = prob[static_cast<size_t>(iy) * W + ix];
            const double p10 = prob[static_cast<size_t>(iy) * W + ix + 1];
            const double p01 = prob[static_cast<size_t>(iy + 1) * W + ix];
            const double p11 = prob[static_cast<size_t>(iy + 1) * W + ix + 1];
            const double pv = (1 - wx) * (1 - wy) * p00 + wx * (1 - wy) * p10 +
                              (1 - wx) * wy * p01 + wx * wy * p11;
            const double dpx = ((1 - wy) * (p10 - p00) + wy * (p11 - p01)) / res;
            const double dpy = ((1 - wx) * (p01 - p00) + wx * (p11 - p10)) / res;
            const double r = 1.0 - pv;
            const double dqx_dth = -s * px - c * py;
            const double dqy_dth = c * px - s * py;
            Eigen::Vector3d J(-dpx, -dpy, -(dpx * dqx_dth + dpy * dqy_dth));
            H_mat += J * J.transpose();
            g += J * r;
            ++n_valid;
        }
        if (n_valid < rp.min_points) break;

        // prior residuals (anchor to `prior`)
        const double r0 = rp.w_trans * (x[0] - pred[0]);
        const double r1 = rp.w_trans * (x[1] - pred[1]);
        const double r2 = rp.w_rot * normalize_angle(x[2] - pred[2]);
        H_mat(0, 0) += rp.w_trans * rp.w_trans;
        H_mat(1, 1) += rp.w_trans * rp.w_trans;
        H_mat(2, 2) += rp.w_rot * rp.w_rot;
        g[0] += rp.w_trans * r0;
        g[1] += rp.w_trans * r1;
        g[2] += rp.w_rot * r2;

        H_mat += lam * Eigen::Matrix3d::Identity();
        Eigen::Vector3d dx = -H_mat.ldlt().solve(g);
        dx[0] = std::clamp(dx[0], -rp.step_clip_xy, rp.step_clip_xy);
        dx[1] = std::clamp(dx[1], -rp.step_clip_xy, rp.step_clip_xy);
        dx[2] = std::clamp(dx[2], -rp.step_clip_th, rp.step_clip_th);
        x += dx;
        if (dx.norm() < rp.eps_stop) break;
    }
    return Pose2(x[0], x[1], x[2]);
}

std::pair<double, int> score_pose_on_grid_raw(const float* prob, int W, int H,
                                              double ox, double oy, double res,
                                              const MatX2d& pts, const Pose2& pose) {
    if (pts.rows() == 0) return {-1.0, 0};
    const double c = std::cos(pose.theta), s = std::sin(pose.theta);
    double sum = 0.0;
    int n = 0;
    for (int i = 0; i < pts.rows(); ++i) {
        const double qx = c * pts(i, 0) - s * pts(i, 1) + pose.x;
        const double qy = s * pts(i, 0) + c * pts(i, 1) + pose.y;
        const int gx = static_cast<int>(std::floor((qx - ox) / res));
        const int gy = static_cast<int>(std::floor((qy - oy) / res));
        if (gx < 0 || gx >= W || gy < 0 || gy >= H) continue;
        sum += prob[static_cast<size_t>(gy) * W + gx];
        ++n;
    }
    return {n == 0 ? -1.0 : sum / n, n};
}

MatX2f stride_subsample(const MatX2f& pts, int max_points) {
    if (max_points <= 0 || pts.rows() <= max_points) return pts;
    const int stride = std::max<int>(1, pts.rows() / max_points);
    MatX2f sub((pts.rows() + stride - 1) / stride, 2);
    int k = 0;
    for (int i = 0; i < pts.rows(); i += stride) sub.row(k++) = pts.row(i);
    return sub.topRows(k);
}

}  // namespace fusion
