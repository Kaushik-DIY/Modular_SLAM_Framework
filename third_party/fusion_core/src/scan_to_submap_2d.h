#pragma once
// scan_to_submap (V4.4-P2): Cartographer-style rotating submaps + two-stage
// correlative search + GN/LM refine. Verbatim port of
// slam_core/matching/scan_to_submap/{submaps,correlative,two_stage_backend}.py
// — parity-critical details (do NOT "improve"):
//   * Bresenham uses strict >/< error stepping (Python _bresenham), free on
//     cells[:-1] INCLUDING the start cell, occ on the endpoint, per-cell clip;
//     a ray is skipped entirely if origin OR endpoint is out of bounds.
//   * grid size = nearbyint(size_m/res) (np.round half-even).
//   * Pyramid = 2x2 max-pool (crop to even dims), NOT the bnb forward-max.
//   * Correlative scoring: nearbyint nearest-cell on the LEVEL image, count
//     < min_valid => -1e9; iteration theta(outer) -> dx -> dy, strict `>`.
//   * Offsets generated as -w + k*step while <= w + 1e-9 (np.arange parity).
//   * Refine initializes from the coarse pose but anchors its prior to the
//     PREDICTION (scan_match_common::refine_pose_lm with initial != prior).

#include <memory>
#include <vector>

#include "common.h"
#include "scan_match_common.h"

namespace fusion {

using MatX2dRef = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;

// ---------------------------------------------------------------------------

struct ProbGrid2D {
    int w = 0, h = 0;
    double res = 0.05;
    double ox = 0.0, oy = 0.0;       // origin_world = (-size/2, -size/2)
    double l0 = 0.0, l_occ = 1.0, l_free = -0.1, l_min = -5.0, l_max = 5.0;
    std::vector<float> L;            // row-major (h, w) log-odds

    ProbGrid2D() = default;
    ProbGrid2D(double size_m, double res_, double l0_, double l_occ_,
               double l_free_, double l_min_, double l_max_);

    void world_to_grid(double x, double y, int& gx, int& gy) const {
        gx = static_cast<int>(std::floor((x - ox) / res));
        gy = static_cast<int>(std::floor((y - oy) / res));
    }
    bool in_bounds(int gx, int gy) const {
        return gx >= 0 && gx < w && gy >= 0 && gy < h;
    }
    void update_cell(int gx, int gy, double dl) {
        if (!in_bounds(gx, gy)) return;
        float& v = L[static_cast<size_t>(gy) * w + gx];
        v = static_cast<float>(std::clamp(static_cast<double>(v) + dl, l_min, l_max));
        _dirty = true;
    }
    // sigmoid(L), cached until the next update.
    const std::vector<float>& probability() const;

private:
    mutable bool _dirty = true;
    mutable std::vector<float> _prob;
};

struct Submap2D {
    int id = -1;
    ProbGrid2D grid;
    Pose2 pose_world;
    int num_inserted = 0;
    bool finished = false;
};
using Submap2DPtr = std::shared_ptr<Submap2D>;

// ---------------------------------------------------------------------------

struct SubmapBuilderConfig {
    double submap_size_m = 20.0;
    double resolution = 0.05;
    int scans_per_submap = 250;
    double l0 = 0.0, l_occ = 1.0, l_free = -0.1, l_min = -5.0, l_max = 5.0;
};

class SubmapBuilder2D {
public:
    explicit SubmapBuilder2D(SubmapBuilderConfig cfg) : _cfg(cfg) {}

    bool insert_scan(const Pose2& pose_world, const MatX2dRef& scan_local);
    const std::vector<Submap2DPtr>& active() const { return _active; }
    const std::vector<Submap2DPtr>& finished() const { return _finished; }

private:
    Submap2DPtr new_submap(const Pose2& pose_world);
    void ensure_two_active(const Pose2& pose_world);
    void maybe_finish_oldest(const Pose2& pose_world);
    void integrate(Submap2D& sm, double origin_x_sub, double origin_y_sub,
                   const MatX2dRef& endpoints_sub);

    SubmapBuilderConfig _cfg;
    std::vector<Submap2DPtr> _active, _finished;
    int _next_id = 0;
    bool _initialized = false;
};

// ---------------------------------------------------------------------------

// 2x2 max-pool stack over a probability image (correlative.py).
class MaxPoolStack {
public:
    MaxPoolStack(const std::vector<float>& prob, int h, int w, int num_levels);
    const std::vector<float>& level(int i) const { return _levels[i]; }
    int level_h(int i) const { return _hs[i]; }
    int level_w(int i) const { return _ws[i]; }

private:
    std::vector<std::vector<float>> _levels;
    std::vector<int> _hs, _ws;
};

struct SearchWindow {
    double xy_window = 1.0;
    double theta_window = 0.4;
    double xy_step = 0.10;
    double theta_step = 0.05;
    int level = 2;
};

// single-implementation brute-force search (scalar/vectorized-equivalent
// tie-breaking: theta outer, dx, dy, strict `>`). Returns best (pose, score).
std::pair<Pose2, double> bruteforce_search(
    const MaxPoolStack& stack, int level, double grid_ox, double grid_oy,
    double res, const MatX2dRef& pts_local, const Pose2& center,
    const SearchWindow& win, int min_valid);

// ---------------------------------------------------------------------------

struct SubmapMatcherConfig {
    double min_score = 0.5;            // coarse-trust gate
    bool reject_below_min_score = false;
    int min_valid = 30;
    int precomp_levels = 3;
    int max_match_points = 200;
    int max_refine_points = 200;
    int refine_min_points = 30;
    double refine_w_trans = 0.1;
    double refine_w_rot = 1.0;
    int refine_iters = 12;
    double refine_damping = 1e-3;
    double refine_eps_stop = 1e-6;
    double refine_step_clip_xy = 0.10;
    double refine_step_clip_th = 0.0872664626;   // 5 deg
    SearchWindow coarse{1.0, 0.40, 0.10, 0.05, 2};
    SearchWindow fine{0.25, 0.12, 0.05, 0.02, 0};
};

struct SubmapMatchResult {
    bool success = false;
    double score = -1.0;          // coarse correlative confidence (response.score)
    double refined_score = -1.0;  // diagnostic
    Pose2 pose_world;             // final (matched) or predicted (FALLBACK)
};

// Port of TwoStageBruteForceSubmapBackend.match + matcher.py target selection.
SubmapMatchResult submap_match(const SubmapBuilder2D& builder,
                               const MatX2dRef& scan_local,
                               const Pose2& predicted_world,
                               const SubmapMatcherConfig& cfg);

}  // namespace fusion
