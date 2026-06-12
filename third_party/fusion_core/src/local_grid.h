#pragma once
// LocalGrid — log-odds occupancy grid assembled from a set of Signatures' scans
// at their current graph poses. This is the candidate-local verification target
// for the B&B matcher: scans "grouped around the loop candidate in the map" are
// rasterized into one small grid, and the query scan is matched against it.
//
// Update model (matches slam_core ProbabilityGrid semantics):
//   * hit cell:  L += l_occ
//   * free cells along the sensor->hit ray (integer Bresenham): L += l_free
//   * L clipped to [l_min, l_max];  p = 1 / (1 + exp(-L))

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "common.h"
#include "signature.h"

namespace fusion {

struct GridConfig {
    double resolution = 0.05;
    double l_occ = 0.85;
    double l_free = -0.4;
    double l_min = -5.0;
    double l_max = 5.0;
    double margin = 1.0;   // metres of padding around the scan extent
};

class LocalGrid {
public:
    LocalGrid(double origin_x, double origin_y, int width, int height, GridConfig cfg)
        : _ox(origin_x), _oy(origin_y), _w(width), _h(height), _cfg(cfg),
          _L(static_cast<size_t>(width) * height, 0.0f) {}

    int width() const { return _w; }
    int height() const { return _h; }
    double origin_x() const { return _ox; }
    double origin_y() const { return _oy; }
    double resolution() const { return _cfg.resolution; }

    bool in_bounds(int gx, int gy) const {
        return gx >= 0 && gx < _w && gy >= 0 && gy < _h;
    }
    void world_to_grid(double x, double y, int& gx, int& gy) const {
        gx = static_cast<int>(std::floor((x - _ox) / _cfg.resolution));
        gy = static_cast<int>(std::floor((y - _oy) / _cfg.resolution));
    }

    float& at(int gx, int gy) { return _L[static_cast<size_t>(gy) * _w + gx]; }
    float at(int gx, int gy) const { return _L[static_cast<size_t>(gy) * _w + gx]; }
    const float* data() const { return _L.data(); }
    float* data() { return _L.data(); }

    void add_log_odds(int gx, int gy, double dl) {
        if (!in_bounds(gx, gy)) return;
        float& v = at(gx, gy);
        v = static_cast<float>(std::clamp(static_cast<double>(v) + dl, _cfg.l_min, _cfg.l_max));
    }

    // Integrate one scan taken from `pose` (sensor at robot origin).
    void integrate_scan(const MatX2f& scan_xy, const Pose2& pose) {
        int sgx, sgy;
        world_to_grid(pose.x, pose.y, sgx, sgy);
        for (int i = 0; i < scan_xy.rows(); ++i) {
            const Eigen::Vector2d w =
                pose.transform({static_cast<double>(scan_xy(i, 0)),
                                static_cast<double>(scan_xy(i, 1))});
            int gx, gy;
            world_to_grid(w.x(), w.y(), gx, gy);
            bresenham_free(sgx, sgy, gx, gy);
            add_log_odds(gx, gy, _cfg.l_occ);
        }
    }

    // Probability grid p = sigmoid(L) as a flat row-major array.
    std::vector<float> probability() const {
        std::vector<float> p(_L.size());
        for (size_t i = 0; i < _L.size(); ++i)
            p[i] = 1.0f / (1.0f + std::exp(-_L[i]));
        return p;
    }

private:
    // Mark free space along the ray, EXCLUDING the hit cell.
    void bresenham_free(int x0, int y0, int x1, int y1) {
        int dx = std::abs(x1 - x0), dy = -std::abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;
        int x = x0, y = y0;
        while (x != x1 || y != y1) {
            add_log_odds(x, y, _cfg.l_free);
            const int e2 = 2 * err;
            if (e2 >= dy) { err += dy; x += sx; }
            if (e2 <= dx) { err += dx; y += sy; }
        }
    }

    double _ox, _oy;
    int _w, _h;
    GridConfig _cfg;
    std::vector<float> _L;
};

using LocalGridPtr = std::shared_ptr<LocalGrid>;

// Assemble a grid that covers all the given signatures' scans (at the supplied
// poses) plus margin, then integrate every scan.
inline LocalGridPtr assemble_local_grid(const std::vector<SignaturePtr>& sigs,
                                        const std::vector<Pose2>& poses,
                                        GridConfig cfg = {}) {
    double min_x = 1e18, min_y = 1e18, max_x = -1e18, max_y = -1e18;
    bool any = false;
    for (size_t k = 0; k < sigs.size(); ++k) {
        const auto& s = sigs[k];
        if (!s || !s->has_scan()) continue;
        const Pose2& p = poses[k];
        min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
        for (int i = 0; i < s->scan_xy.rows(); ++i) {
            const Eigen::Vector2d w = p.transform({static_cast<double>(s->scan_xy(i, 0)),
                                                   static_cast<double>(s->scan_xy(i, 1))});
            min_x = std::min(min_x, w.x()); max_x = std::max(max_x, w.x());
            min_y = std::min(min_y, w.y()); max_y = std::max(max_y, w.y());
            any = true;
        }
    }
    if (!any) return std::make_shared<LocalGrid>(0.0, 0.0, 1, 1, cfg);

    const double ox = min_x - cfg.margin, oy = min_y - cfg.margin;
    const int w = static_cast<int>(std::ceil((max_x - min_x + 2 * cfg.margin) / cfg.resolution)) + 1;
    const int h = static_cast<int>(std::ceil((max_y - min_y + 2 * cfg.margin) / cfg.resolution)) + 1;
    auto grid = std::make_shared<LocalGrid>(ox, oy, w, h, cfg);
    for (size_t k = 0; k < sigs.size(); ++k) {
        if (!sigs[k] || !sigs[k]->has_scan()) continue;
        grid->integrate_scan(sigs[k]->scan_xy, poses[k]);
    }
    return grid;
}

}  // namespace fusion
