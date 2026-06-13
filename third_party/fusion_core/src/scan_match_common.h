#pragma once
// Shared scan-matching primitives (V4.4-P0), extracted from bnb_matcher.cpp so
// the native LiDAR front-end (scan_to_submap_2d / scan_to_map_2d) reuses the
// exact same GN/LM occupancy refinement and grid scoring as the B&B verifier.
//
// Semantics are ports of slam_core/matching/scan_to_submap/refine.py
// (CartoRefinementProblem) + optimisers/gn_lm.py (GaussNewtonLM). The ONE
// generalization vs the original bnb refine_lm: `initial` (state seed) and
// `prior` (anchor of the prior residuals) are separate arguments — the
// two-stage backend initializes from the coarse correlative pose while
// anchoring the prior to the extrapolator prediction. bnb_match passes
// initial == prior == coarse, which is bit-identical to the pre-V4.4 code.

#include <utility>
#include <vector>

#include "common.h"

namespace fusion {

struct RefineParams {
    int iters = 8;
    double damping = 1e-3;
    double eps_stop = 1e-6;
    double step_clip_xy = 0.10;
    double step_clip_th = 0.0872664626;   // 5 deg
    double w_trans = 1.0;
    double w_rot = 1.0;
    int min_points = 20;
};

// GN/LM refine of an SE(2) pose against a probability image (row-major HxW,
// origin (ox,oy), cell size `res`). Residuals: occupancy mismatch (1 - p) per
// valid point (bilinear value + gradient) plus weighted pose-prior rows.
Pose2 refine_pose_lm(const float* prob, int W, int H, double ox, double oy,
                     double res, const MatX2d& pts, const Pose2& initial,
                     const Pose2& prior, const RefineParams& rp);

// Mean probability at floor-rounded cells under `pose` -> {score, n_valid};
// score = -1 when no point lands in bounds.
std::pair<double, int> score_pose_on_grid_raw(const float* prob, int W, int H,
                                              double ox, double oy, double res,
                                              const MatX2d& pts, const Pose2& pose);

// Every-stride-th row subsample down to <= max_points (Python [::stride] parity).
MatX2f stride_subsample(const MatX2f& pts, int max_points);

}  // namespace fusion
