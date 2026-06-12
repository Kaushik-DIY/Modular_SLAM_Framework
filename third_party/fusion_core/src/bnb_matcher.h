#pragma once
// Correlative branch-and-bound scan matcher + Gauss-Newton/LM refine.
//
// Faithful C++ port of the algorithms in
//   slam_core/matching/scan_to_submap/branch_and_bound_backend.py
//   slam_core/matching/scan_to_submap/precomputation_grid_2d.py
//   slam_core/matching/scan_to_submap/refine.py  (+ optimisers/gn_lm.py)
// operating on the candidate-local LocalGrid from retrieval (C4) instead of a
// submap. Same parameter semantics so Python-vs-C++ parity is testable on
// identical inputs.

#include <cstdint>
#include <memory>
#include <vector>

#include "common.h"
#include "local_grid.h"

namespace fusion {

struct BnbConfig {
    double linear_search_window = 3.5;   // metres (half-extent)
    double angular_search_window = 0.2618;  // rad (half-extent, 15 deg)
    int depth = 6;                       // precomputation/branch depth
    double min_rotational_step = 0.02;   // rad
    int max_match_points = 200;          // coarse-search point cap (stride-subsampled)
    // refine (CartoRefinementProblem + GaussNewtonLM defaults)
    bool do_refine = true;
    int max_refine_points = 180;
    int refine_min_points = 20;
    double refine_w_trans = 1.0;
    double refine_w_rot = 1.0;
    int refine_iters = 8;
    double refine_damping = 1e-3;
    double refine_eps_stop = 1e-6;
    double refine_step_clip_xy = 0.10;
    double refine_step_clip_th = 0.0872664626;  // 5 deg
};

struct BnbResult {
    bool success = false;
    double coarse_score = -1.0;   // best correlative score in [0,1]
    double refined_score = -1.0;  // mean occupancy at final pose
    Pose2 pose;                   // final pose (grid/world frame of the LocalGrid)
    bool refined = false;
    int num_rotations = 0;
    int num_points_match = 0;
};

// Multi-resolution forward-looking max stack over the grid's probability map.
class PrecompStack {
public:
    PrecompStack(const LocalGrid& grid, int depth);
    int max_depth() const { return static_cast<int>(_levels.size()) - 1; }
    const std::vector<float>& level(int d) const;
    int width() const { return _w; }
    int height() const { return _h; }

private:
    int _w, _h;
    std::vector<std::vector<float>> _levels;  // level i: width 2^i window max
};

// Match `scan_local` (sensor-frame points) against `grid`, searching around
// `predicted_pose` (pose of the sensor in the grid's world frame).
BnbResult bnb_match(const LocalGrid& grid, const MatX2f& scan_local,
                    const Pose2& predicted_pose, const BnbConfig& cfg);

}  // namespace fusion
