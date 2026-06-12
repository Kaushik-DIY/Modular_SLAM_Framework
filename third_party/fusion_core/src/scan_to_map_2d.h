#pragma once
// scan_to_map (V4.4-P3): Hector-style single growing map pyramid +
// coarse-to-fine Gauss-Newton. Verbatim port of slam_core/matching/scan_to_map.py.
// Parity-critical details:
//   * GridMap size = ceil(size_m/res), bumped to even; world (0,0) at the grid
//     CENTER (origin = size/2 in cell units); float grid coords.
//   * in_bounds margin: x,y in [1, size-2) (bilinear safety).
//   * integrate_scan_simple: linspace rays (ray_steps samples), np.round
//     (half-even) cells; NumPy fancy-index semantics — duplicates within ONE
//     add_logodds_at call receive the delta ONCE (per-ray free batch; one
//     batched endpoint call per scan). Endpoint mask (in_bounds) applied
//     BEFORE rays; the origin is never checked.
//   * GN: residual 1 - bilinear(prob); gradients are CACHED central-difference
//     grids sampled bilinearly, /res at use; H = JtJ + damping I; step clip;
//     stop |delta| < 1e-6; min_points is only an early-exit.
//   * match(): bootstrap scans insert at the dead-reckoned pose and return
//     success=false; levels processed coarse->fine; score = mean bilinear prob
//     at the finest level; success = score >= min_score (NO inlier gate).

#include <memory>
#include <vector>

#include "common.h"

namespace fusion {

using MatX2dM = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;

class HectorGridMap {
public:
    HectorGridMap(double res, double size_m, double l0, double l_min, double l_max);

    double res() const { return _res; }
    int size() const { return _size; }

    // float grid coords (cell units), world (0,0) at center
    inline void world_to_grid(double x, double y, double& gx, double& gy) const {
        gx = x / _res + _origin;
        gy = y / _res + _origin;
    }
    inline bool in_bounds_f(double gx, double gy) const {
        return gx >= 1.0 && gy >= 1.0 && gx < _size - 2.0 && gy < _size - 2.0;
    }

    const std::vector<float>& prob() const;                  // sigmoid cache
    void gradients(const std::vector<float>*& gx,            // central-diff caches
                   const std::vector<float>*& gy) const;

    inline double bilinear(const std::vector<float>& g, double x, double y) const {
        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));
        const double dx = x - x0, dy = y - y0;
        const double v00 = g[static_cast<size_t>(y0) * _size + x0];
        const double v10 = g[static_cast<size_t>(y0) * _size + x0 + 1];
        const double v01 = g[static_cast<size_t>(y0 + 1) * _size + x0];
        const double v11 = g[static_cast<size_t>(y0 + 1) * _size + x0 + 1];
        const double v0 = v00 * (1.0 - dx) + v10 * dx;
        const double v1 = v01 * (1.0 - dx) + v11 * dx;
        return v0 * (1.0 - dy) + v1 * dy;
    }

    // one fancy-index batch: duplicates get the delta ONCE; clip per cell
    void add_logodds_batch(const std::vector<std::pair<int, int>>& cells, double delta);
    void integrate_scan_simple(double pose_x, double pose_y,
                               const MatX2dM& pts_world, double l_free,
                               double l_occ, int ray_steps);

    const std::vector<float>& log_odds() const { return _L; }

private:
    double _res, _origin;
    int _size;
    double _l0, _l_min, _l_max;
    std::vector<float> _L;
    mutable bool _dirty = true;
    mutable std::vector<float> _prob, _gx, _gy;
};

struct MapMatcherConfig {
    double base_res = 0.05;            // MAP_RESOLUTION
    double size_m = 40.0;              // MAP_SIZE_METERS
    int num_levels = 3;                // PYRAMID_LEVELS
    double l0 = 0.0, l_occ = 1.0, l_free = -0.1, l_min = -5.0, l_max = 5.0;
    int ray_steps = 20;
    int bootstrap_scans = 1;           // N_BOOTSTRAP_SCANS
    std::vector<int> gn_iters_per_level{20, 15, 10};
    double gn_damping = 1e-4;          // GN_DAMPING (lab)
    int min_points = 60;               // CORR_MAP_MIN_POINTS
    double min_score = 0.10;           // CORR_MAP_MIN_SCORE
    double step_clip_xy = 0.03;        // CORR_MAP_STEP_CLIP_XY
    double step_clip_th = 0.0174532925;  // 1 deg
    int map_update_every = 1;          // MAP_UPDATE_EVERY
};

struct MapMatchResult {
    bool success = false;
    bool bootstrap = false;
    double score = -1.0;
    int inliers = 0;
    Pose2 pose_world;                  // matched, or predicted on failure
};

class MapMatcher {
public:
    explicit MapMatcher(MapMatcherConfig cfg);

    MapMatchResult match(const MatX2dM& scan_local, const Pose2& predicted);
    // integrate at pose into all levels (update_target)
    void insert(const Pose2& pose_world, const MatX2dM& scan_local);
    bool initialized() const { return _initialized; }
    const HectorGridMap& finest() const { return *_levels.back(); }

private:
    MapMatcherConfig _cfg;
    std::vector<std::unique_ptr<HectorGridMap>> _levels;   // coarse -> fine
    int _bootstrap_count = 0;
    bool _initialized = false;
};

// GN alignment on one level; returns refined pose (align_pose_gauss_newton).
Pose2 align_pose_gauss_newton_2d(const HectorGridMap& grid, const Pose2& init,
                                 const MatX2dM& pts, int iters, double damping,
                                 int min_points, double step_clip_xy,
                                 double step_clip_th);

}  // namespace fusion
