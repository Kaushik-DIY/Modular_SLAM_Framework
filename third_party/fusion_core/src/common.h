#pragma once
// fusion_core common types. Mirrors slam_core/common/types.py::Pose2 semantics
// (x, y, theta in world frame; theta normalized to (-pi, pi]).

#include <Eigen/Core>
#include <cmath>
#include <cstdint>

namespace fusion {

inline double normalize_angle(double a) {
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a <= -M_PI) a += 2.0 * M_PI;
    return a;
}

struct Pose2 {
    double x = 0.0;
    double y = 0.0;
    double theta = 0.0;

    Pose2() = default;
    Pose2(double x_, double y_, double th_) : x(x_), y(y_), theta(normalize_angle(th_)) {}

    // this ∘ other (compose: apply `other` in this frame)
    Pose2 compose(const Pose2& o) const {
        const double c = std::cos(theta), s = std::sin(theta);
        return Pose2(x + c * o.x - s * o.y,
                     y + s * o.x + c * o.y,
                     theta + o.theta);
    }

    Pose2 inverse() const {
        const double c = std::cos(theta), s = std::sin(theta);
        return Pose2(-(c * x + s * y), -(-s * x + c * y), -theta);
    }

    // Transform a local-frame point into the world frame.
    Eigen::Vector2d transform(const Eigen::Vector2d& p) const {
        const double c = std::cos(theta), s = std::sin(theta);
        return {x + c * p.x() - s * p.y(), y + s * p.x() + c * p.y()};
    }
};

// Row-major packed arrays (zero-copy numpy views are created over these).
using MatX2f = Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>;
using MatX3f = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatX2d = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;

}  // namespace fusion
