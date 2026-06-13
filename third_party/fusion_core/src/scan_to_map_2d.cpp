#include "scan_to_map_2d.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

#include <Eigen/Dense>

namespace fusion {

namespace {
inline double rint_he(double v) { return std::nearbyint(v); }   // np.round parity
}

// ---------------------------------------------------------------------------
// HectorGridMap
// ---------------------------------------------------------------------------

HectorGridMap::HectorGridMap(double res, double size_m, double l0, double l_min,
                             double l_max)
    : _res(res), _l0(l0), _l_min(l_min), _l_max(l_max) {
    _size = static_cast<int>(std::ceil(size_m / res));
    if (_size % 2 == 1) _size += 1;
    _origin = _size / 2.0;
    _L.assign(static_cast<size_t>(_size) * _size, static_cast<float>(l0));
}

const std::vector<float>& HectorGridMap::prob() const {
    if (_dirty || _prob.empty()) {
        _prob.resize(_L.size());
        for (size_t i = 0; i < _L.size(); ++i)
            _prob[i] = static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(_L[i]))));
    }
    return _prob;
}

void HectorGridMap::gradients(const std::vector<float>*& gx,
                              const std::vector<float>*& gy) const {
    if (_dirty || _gx.empty()) {
        const auto& p = prob();
        const int n = _size;
        _gx.assign(p.size(), 0.0f);
        _gy.assign(p.size(), 0.0f);
        // gx = 0.5*(p[:,2:] - p[:,:-2]) with edge padding
        for (int y = 0; y < n; ++y) {
            const size_t row = static_cast<size_t>(y) * n;
            for (int x = 1; x + 1 < n; ++x)
                _gx[row + x] = 0.5f * (p[row + x + 1] - p[row + x - 1]);
            _gx[row + 0] = _gx[row + 1];                  // edge pad
            _gx[row + n - 1] = _gx[row + n - 2];
        }
        for (int y = 1; y + 1 < n; ++y) {
            const size_t row = static_cast<size_t>(y) * n;
            for (int x = 0; x < n; ++x)
                _gy[row + x] = 0.5f * (p[row + n + x] - p[row - n + x]);
        }
        for (int x = 0; x < n; ++x) {                     // edge pad rows
            _gy[x] = _gy[static_cast<size_t>(n) + x];
            _gy[static_cast<size_t>(n - 1) * n + x] = _gy[static_cast<size_t>(n - 2) * n + x];
        }
        _dirty = false;
    }
    gx = &_gx;
    gy = &_gy;
}

void HectorGridMap::add_logodds_batch(const std::vector<std::pair<int, int>>& cells,
                                      double delta) {
    // NumPy fancy-index semantics: each UNIQUE in-bounds cell gets the delta
    // exactly once per call.
    std::unordered_set<int64_t> seen;
    seen.reserve(cells.size() * 2);
    for (const auto& [x, y] : cells) {
        if (x < 0 || y < 0 || x >= _size || y >= _size) continue;
        const int64_t key = static_cast<int64_t>(y) * _size + x;
        if (!seen.insert(key).second) continue;
        float& v = _L[static_cast<size_t>(key)];
        v = static_cast<float>(std::clamp(static_cast<double>(v) + delta, _l_min, _l_max));
    }
    _dirty = true;
}

void HectorGridMap::integrate_scan_simple(double pose_x, double pose_y,
                                          const MatX2dM& pts_world, double l_free,
                                          double l_occ, int ray_steps) {
    double x0, y0;
    world_to_grid(pose_x, pose_y, x0, y0);

    std::vector<std::pair<int, int>> ray_cells, end_cells;
    end_cells.reserve(pts_world.rows());

    for (int i = 0; i < pts_world.rows(); ++i) {
        double x1, y1;
        world_to_grid(pts_world(i, 0), pts_world(i, 1), x1, y1);
        if (!in_bounds_f(x1, y1)) continue;   // endpoint mask, origin unchecked

        // linspace(x0, x1, ray_steps)[:-1] rounded -> one free batch PER RAY
        ray_cells.clear();
        for (int k = 0; k + 1 < ray_steps; ++k) {
            const double tt = ray_steps == 1 ? 0.0
                              : static_cast<double>(k) / (ray_steps - 1);
            const double xs = x0 + (x1 - x0) * tt;
            const double ys = y0 + (y1 - y0) * tt;
            ray_cells.emplace_back(static_cast<int>(rint_he(xs)),
                                   static_cast<int>(rint_he(ys)));
        }
        add_logodds_batch(ray_cells, l_free);

        end_cells.emplace_back(static_cast<int>(rint_he(x1)),
                               static_cast<int>(rint_he(y1)));
    }
    if (!end_cells.empty()) add_logodds_batch(end_cells, l_occ);
}

// ---------------------------------------------------------------------------
// GN alignment
// ---------------------------------------------------------------------------

Pose2 align_pose_gauss_newton_2d(const HectorGridMap& grid, const Pose2& init,
                                 const MatX2dM& pts, int iters, double damping,
                                 int min_points, double step_clip_xy,
                                 double step_clip_th) {
    Eigen::Vector3d pose(init.x, init.y, normalize_angle(init.theta));
    const auto& prob = grid.prob();
    const std::vector<float>*gxg = nullptr, *gyg = nullptr;
    grid.gradients(gxg, gyg);
    const int n = static_cast<int>(pts.rows());

    for (int it = 0; it < iters; ++it) {
        const double c = std::cos(pose[2]), s = std::sin(pose[2]);
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        Eigen::Vector3d g = Eigen::Vector3d::Zero();
        int valid = 0;

        for (int i = 0; i < n; ++i) {
            const double px = pts(i, 0), py = pts(i, 1);
            const double wx = c * px - s * py + pose[0];
            const double wy = s * px + c * py + pose[1];
            double gx, gy;
            grid.world_to_grid(wx, wy, gx, gy);
            if (!grid.in_bounds_f(gx, gy)) continue;
            ++valid;

            const double m = grid.bilinear(prob, gx, gy);
            const double r = 1.0 - m;
            const double dmx = grid.bilinear(*gxg, gx, gy) / grid.res();
            const double dmy = grid.bilinear(*gyg, gx, gy) / grid.res();
            // d(world)/d(theta): dR = [[-s,-c],[c,-s]]
            const double dpx_dth = -s * px - c * py;
            const double dpy_dth = c * px - s * py;
            Eigen::Vector3d J(-dmx, -dmy, -(dmx * dpx_dth + dmy * dpy_dth));
            H += J * J.transpose();
            g += J * r;
        }
        if (valid < min_points) break;

        H += damping * Eigen::Matrix3d::Identity();
        Eigen::Vector3d delta = -H.ldlt().solve(g);
        delta[0] = std::clamp(delta[0], -step_clip_xy, step_clip_xy);
        delta[1] = std::clamp(delta[1], -step_clip_xy, step_clip_xy);
        delta[2] = std::clamp(delta[2], -step_clip_th, step_clip_th);
        pose[0] += delta[0];
        pose[1] += delta[1];
        pose[2] = normalize_angle(pose[2] + delta[2]);
        if (delta.norm() < 1e-6) break;
    }
    return Pose2(pose[0], pose[1], normalize_angle(pose[2]));
}

// ---------------------------------------------------------------------------
// MapMatcher
// ---------------------------------------------------------------------------

MapMatcher::MapMatcher(MapMatcherConfig cfg) : _cfg(std::move(cfg)) {
    // levels coarse -> fine: res = base * 2^(L-1-i)
    for (int i = 0; i < _cfg.num_levels; ++i) {
        const double res = _cfg.base_res * std::pow(2.0, _cfg.num_levels - 1 - i);
        _levels.push_back(std::make_unique<HectorGridMap>(
            res, _cfg.size_m, _cfg.l0, _cfg.l_min, _cfg.l_max));
    }
}

void MapMatcher::insert(const Pose2& pose_world, const MatX2dM& scan_local) {
    const double c = std::cos(pose_world.theta), s = std::sin(pose_world.theta);
    MatX2dM pts_world(scan_local.rows(), 2);
    for (int i = 0; i < scan_local.rows(); ++i) {
        pts_world(i, 0) = c * scan_local(i, 0) - s * scan_local(i, 1) + pose_world.x;
        pts_world(i, 1) = s * scan_local(i, 0) + c * scan_local(i, 1) + pose_world.y;
    }
    for (auto& g : _levels)
        g->integrate_scan_simple(pose_world.x, pose_world.y, pts_world,
                                 _cfg.l_free, _cfg.l_occ, _cfg.ray_steps);
    _initialized = true;
}

MapMatchResult MapMatcher::match(const MatX2dM& scan_local, const Pose2& predicted) {
    MapMatchResult res;
    res.pose_world = predicted;

    if (_bootstrap_count < _cfg.bootstrap_scans) {
        insert(predicted, scan_local);
        ++_bootstrap_count;
        res.bootstrap = true;
        return res;                       // success=false, score=-1
    }
    if (!_initialized) return res;

    // coarse -> fine (levels already sorted res descending)
    Pose2 pose_est = predicted;
    for (size_t lvl = 0; lvl < _levels.size(); ++lvl) {
        const int iters = _cfg.gn_iters_per_level[
            std::min(lvl, _cfg.gn_iters_per_level.size() - 1)];
        pose_est = align_pose_gauss_newton_2d(
            *_levels[lvl], pose_est, scan_local, iters, _cfg.gn_damping,
            _cfg.min_points, _cfg.step_clip_xy, _cfg.step_clip_th);
    }

    // score: mean bilinear prob at the finest level
    const HectorGridMap& fg = *_levels.back();
    const auto& prob = fg.prob();
    const double c = std::cos(pose_est.theta), s = std::sin(pose_est.theta);
    double sum = 0.0;
    int inliers = 0;
    for (int i = 0; i < scan_local.rows(); ++i) {
        const double wx = c * scan_local(i, 0) - s * scan_local(i, 1) + pose_est.x;
        const double wy = s * scan_local(i, 0) + c * scan_local(i, 1) + pose_est.y;
        double gx, gy;
        fg.world_to_grid(wx, wy, gx, gy);
        if (!fg.in_bounds_f(gx, gy)) continue;
        sum += fg.bilinear(prob, gx, gy);
        ++inliers;
    }
    const double score = inliers >= _cfg.min_points ? sum / inliers : -1.0;

    res.score = score;
    res.inliers = inliers;
    res.success = score >= _cfg.min_score;
    res.pose_world = res.success ? pose_est : predicted;
    return res;
}

}  // namespace fusion
