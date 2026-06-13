#pragma once
// PoseExtrapolatorCV — verbatim port of carto/local_slam/pose_extrapolator.py
// (V4.4-P1). Cartographer-like 2D constant-velocity extrapolator with optional
// IMU aiding: gyro wz overrides the pose-derived yaw rate, and the absolute
// quaternion yaw (offset-latched at first use) gently corrects the predicted
// heading (alpha nudge). The FALLBACK pose on matcher failure is exactly this
// IMU-informed prediction — the always-on IMU pose-prior requirement.

#include <algorithm>
#include <cmath>
#include <deque>
#include <optional>
#include <utility>

#include "common.h"

namespace fusion {

struct ExtrapolatorConfig {
    double max_dt = 0.5;
    double init_vxy = 0.0;
    double init_wz = 0.0;
    double pose_queue_duration_s = 1.5;
    double odom_queue_duration_s = 1.5;
    double odom_trust = 0.35;
    double max_linear_speed_mps = 2.0;
    double max_angular_speed_rps = 2.0;
    bool use_imu = true;
    double imu_yaw_correction_alpha = 0.02;
};

class PoseExtrapolatorCV {
public:
    explicit PoseExtrapolatorCV(ExtrapolatorConfig cfg = {})
        : _cfg(cfg), _vx(cfg.init_vxy), _vy(cfg.init_vxy), _wz(cfg.init_wz) {
        _cfg.odom_trust = std::clamp(_cfg.odom_trust, 0.0, 1.0);
        _cfg.imu_yaw_correction_alpha =
            std::clamp(_cfg.imu_yaw_correction_alpha, 0.0, 1.0);
    }

    bool has_state() const { return !_pose_q.empty(); }

    void add_pose(double t, const Pose2& pose) {
        append_state(_pose_q, t, pose, _cfg.pose_queue_duration_s);
    }
    void add_odometry(double t, const Pose2& pose) {
        append_state(_odom_q, t, pose, _cfg.odom_queue_duration_s);
    }
    void add_imu(double t, double wz, bool has_yaw, double yaw) {
        _imu_t = t;
        _imu_wz = wz;
        _has_imu_wz = true;
        if (has_yaw) {
            _imu_yaw = normalize_angle(yaw);
            _has_imu_yaw = true;
        }
    }

    Pose2 predict(double t) {
        if (!has_state()) return Pose2(0.0, 0.0, 0.0);
        const auto& [last_t, last_pose] = _pose_q.back();

        double dt = t - last_t;
        dt = std::max(0.0, std::min(dt, _cfg.max_dt));

        auto [vx, vy, wz] = estimate_blended_velocity();
        if (_cfg.use_imu && _has_imu_wz)
            wz = std::clamp(_imu_wz, -_cfg.max_angular_speed_rps,
                            _cfg.max_angular_speed_rps);
        _vx = vx; _vy = vy; _wz = wz;

        double pred_theta = normalize_angle(last_pose.theta + wz * dt);
        if (_cfg.use_imu && _has_imu_yaw && _cfg.imu_yaw_correction_alpha > 0.0) {
            if (!_has_yaw_offset) {
                _imu_yaw_offset = normalize_angle(last_pose.theta - _imu_yaw);
                _has_yaw_offset = true;
            }
            const double imu_heading = normalize_angle(_imu_yaw + _imu_yaw_offset);
            pred_theta = normalize_angle(
                pred_theta + _cfg.imu_yaw_correction_alpha *
                                 normalize_angle(imu_heading - pred_theta));
        }
        return Pose2(last_pose.x + vx * dt, last_pose.y + vy * dt, pred_theta);
    }

    double vx() const { return _vx; }
    double vy() const { return _vy; }
    double wz() const { return _wz; }

private:
    using Q = std::deque<std::pair<double, Pose2>>;

    void append_state(Q& q, double t, const Pose2& pose, double duration_s) {
        if (!q.empty() && std::abs(q.back().first - t) <= 1e-9)
            q.back() = {t, pose};
        else
            q.emplace_back(t, pose);
        const double newest = q.back().first;
        while (q.size() > 1 && (newest - q.front().first) > duration_s)
            q.pop_front();
    }

    std::optional<std::array<double, 3>> estimate_velocity(const Q& q) const {
        if (q.size() < 2) return std::nullopt;
        const auto& [t0, p0] = q.front();
        const auto& [t1, p1] = q.back();
        const double dt = t1 - t0;
        if (dt <= 1e-6) return std::nullopt;
        const double L = _cfg.max_linear_speed_mps, A = _cfg.max_angular_speed_rps;
        return std::array<double, 3>{
            std::clamp((p1.x - p0.x) / dt, -L, L),
            std::clamp((p1.y - p0.y) / dt, -L, L),
            std::clamp(normalize_angle(p1.theta - p0.theta) / dt, -A, A)};
    }

    std::array<double, 3> estimate_blended_velocity() const {
        const auto pv = estimate_velocity(_pose_q);
        const auto ov = estimate_velocity(_odom_q);
        if (pv && ov) {
            const double a = _cfg.odom_trust;
            return {(1 - a) * (*pv)[0] + a * (*ov)[0],
                    (1 - a) * (*pv)[1] + a * (*ov)[1],
                    (1 - a) * (*pv)[2] + a * (*ov)[2]};
        }
        if (pv) return *pv;
        if (ov) return *ov;
        return {_vx, _vy, _wz};
    }

    ExtrapolatorConfig _cfg;
    double _vx, _vy, _wz;
    Q _pose_q, _odom_q;
    double _imu_t = 0.0, _imu_wz = 0.0, _imu_yaw = 0.0, _imu_yaw_offset = 0.0;
    bool _has_imu_wz = false, _has_imu_yaw = false, _has_yaw_offset = false;
};

}  // namespace fusion
