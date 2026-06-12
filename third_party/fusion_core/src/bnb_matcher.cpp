#include "bnb_matcher.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>

#include <Eigen/Dense>

#include "scan_match_common.h"

namespace fusion {

namespace {

// ---- precomputation_grid_2d.py ports --------------------------------------

// out[i] = max(arr[i : i+width]) with end clipping (deque sliding max).
void sliding_max_forward_1d(const float* a, int n, int width, float* out) {
    if (width <= 1) {
        std::copy(a, a + n, out);
        return;
    }
    std::deque<int> dq;
    const int padded = n + width - 1;
    for (int j = 0; j < padded; ++j) {
        const float vj = j < n ? a[j] : -std::numeric_limits<float>::infinity();
        while (!dq.empty()) {
            const float vb = dq.back() < n ? a[dq.back()]
                                           : -std::numeric_limits<float>::infinity();
            if (vb <= vj) dq.pop_back(); else break;
        }
        dq.push_back(j);
        const int start = j - width + 1;
        if (start >= 0) {
            while (!dq.empty() && dq.front() < start) dq.pop_front();
            if (start < n)
                out[start] = dq.front() < n ? a[dq.front()]
                                            : -std::numeric_limits<float>::infinity();
        }
    }
}

// out[y,x] = max over the width x width box starting at (y,x).
std::vector<float> forward_max_filter_2d(const std::vector<float>& g, int h, int w,
                                         int width) {
    if (width <= 1) return g;
    std::vector<float> row_max(static_cast<size_t>(h) * w);
    for (int y = 0; y < h; ++y)
        sliding_max_forward_1d(g.data() + static_cast<size_t>(y) * w, w, width,
                               row_max.data() + static_cast<size_t>(y) * w);
    std::vector<float> out(static_cast<size_t>(h) * w);
    std::vector<float> col(h), colo(h);
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) col[y] = row_max[static_cast<size_t>(y) * w + x];
        sliding_max_forward_1d(col.data(), h, width, colo.data());
        for (int y = 0; y < h; ++y) out[static_cast<size_t>(y) * w + x] = colo[y];
    }
    return out;
}

// ---- branch_and_bound_backend.py ports -------------------------------------

struct LinearBounds {
    int min_x, max_x, min_y, max_y;
};

struct Candidate {
    int scan_index;
    int x_off, y_off;
    double orientation;
    double score = -1.0;
};

double compute_angular_step(const MatX2f& pts, double resolution, double min_step) {
    if (pts.rows() == 0) return min_step;
    double dmax = 0.0;
    for (int i = 0; i < pts.rows(); ++i)
        dmax = std::max(dmax, static_cast<double>(pts.row(i).norm()));
    if (dmax <= 1e-9) return min_step;
    double term = 1.0 - (resolution * resolution) / (2.0 * dmax * dmax);
    term = std::clamp(term, -1.0, 1.0);
    double derived = std::acos(term);
    if (!std::isfinite(derived) || derived <= 1e-9) derived = min_step;
    return std::max(min_step, derived);
}

void score_candidates(const std::vector<float>& level, int h, int w,
                      const std::vector<std::vector<std::pair<int, int>>>& scans,
                      std::vector<Candidate>& cands) {
    for (auto& c : cands) {
        const auto& scan = scans[c.scan_index];
        if (scan.empty()) { c.score = -1.0; continue; }
        double sum = 0.0;
        int valid = 0;
        for (const auto& [gx0, gy0] : scan) {
            const int gx = gx0 + c.x_off, gy = gy0 + c.y_off;
            if (gx < 0 || gx >= w || gy < 0 || gy >= h) continue;
            sum += level[static_cast<size_t>(gy) * w + gx];
            ++valid;
        }
        c.score = valid == 0 ? -1.0 : sum / valid;
    }
    std::sort(cands.begin(), cands.end(),
              [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
}

Candidate branch_and_bound(const PrecompStack& stack,
                           const std::vector<std::vector<std::pair<int, int>>>& scans,
                           const std::vector<LinearBounds>& bounds,
                           const std::vector<Candidate>& cands, int depth,
                           double min_score) {
    if (cands.empty()) return Candidate{0, 0, 0, 0.0, min_score};
    if (depth == 0) return cands.front();

    Candidate best{0, 0, 0, 0.0, min_score};
    const int half = 1 << (depth - 1);
    const int h = stack.height(), w = stack.width();

    for (const auto& c : cands) {
        if (c.score <= best.score) break;
        std::vector<Candidate> children;
        const LinearBounds& b = bounds[c.scan_index];
        for (int dx : {0, half}) {
            const int xn = c.x_off + dx;
            if (xn > b.max_x) continue;
            for (int dy : {0, half}) {
                const int yn = c.y_off + dy;
                if (yn > b.max_y) continue;
                children.push_back(Candidate{c.scan_index, xn, yn, c.orientation, -1.0});
            }
        }
        if (children.empty()) continue;
        score_candidates(stack.level(depth - 1), h, w, scans, children);
        Candidate child = branch_and_bound(stack, scans, bounds, children, depth - 1,
                                           best.score);
        if (child.score > best.score) best = child;
    }
    return best;
}

}  // namespace

PrecompStack::PrecompStack(const LocalGrid& grid, int depth)
    : _w(grid.width()), _h(grid.height()) {
    const auto prob = grid.probability();
    _levels.reserve(depth);
    for (int i = 0; i < depth; ++i)
        _levels.push_back(forward_max_filter_2d(prob, _h, _w, 1 << i));
}

const std::vector<float>& PrecompStack::level(int d) const {
    return _levels[static_cast<size_t>(std::clamp(d, 0, max_depth()))];
}

BnbResult bnb_match(const LocalGrid& grid, const MatX2f& scan_local,
                    const Pose2& predicted_pose, const BnbConfig& cfg) {
    BnbResult res;
    res.pose = predicted_pose;
    if (scan_local.rows() == 0 || grid.width() <= 1) return res;

    // Coarse-stage point cap (stride subsample, same as Python).
    MatX2f pts_match = stride_subsample(scan_local, cfg.max_match_points);
    res.num_points_match = static_cast<int>(pts_match.rows());

    const double resol = grid.resolution();
    const int h = grid.height(), w = grid.width();

    // Search parameters.
    const double astep = compute_angular_step(pts_match, resol, cfg.min_rotational_step);
    const int n_ang = static_cast<int>(std::ceil(cfg.angular_search_window / astep));
    std::vector<double> angles;
    for (int k = -n_ang; k <= n_ang; ++k) angles.push_back(k * astep);
    res.num_rotations = static_cast<int>(angles.size());

    const int linear_bound = static_cast<int>(std::ceil(cfg.linear_search_window / resol));
    const int gx0 = static_cast<int>(std::floor((predicted_pose.x - grid.origin_x()) / resol));
    const int gy0 = static_cast<int>(std::floor((predicted_pose.y - grid.origin_y()) / resol));

    // Discretize rotated scans at the predicted translation; shrink bounds to fit.
    std::vector<std::vector<std::pair<int, int>>> scans(angles.size());
    std::vector<LinearBounds> bounds(angles.size());
    for (size_t a = 0; a < angles.size(); ++a) {
        const double th = predicted_pose.theta + angles[a];
        const double c = std::cos(th), s = std::sin(th);
        auto& scan = scans[a];
        scan.reserve(pts_match.rows());
        int smin_x = 0, smax_x = w - 1, smin_y = 0, smax_y = h - 1;  // allowed offsets
        LinearBounds b{std::max(-linear_bound, -gx0), std::min(linear_bound, w - 1 - gx0),
                       std::max(-linear_bound, -gy0), std::min(linear_bound, h - 1 - gy0)};
        int pmin_x = INT32_MAX, pmax_x = INT32_MIN, pmin_y = INT32_MAX, pmax_y = INT32_MIN;
        for (int i = 0; i < pts_match.rows(); ++i) {
            const double qx = c * pts_match(i, 0) - s * pts_match(i, 1) + predicted_pose.x;
            const double qy = s * pts_match(i, 0) + c * pts_match(i, 1) + predicted_pose.y;
            const int gx = static_cast<int>(std::floor((qx - grid.origin_x()) / resol));
            const int gy = static_cast<int>(std::floor((qy - grid.origin_y()) / resol));
            scan.emplace_back(gx, gy);
            pmin_x = std::min(pmin_x, gx); pmax_x = std::max(pmax_x, gx);
            pmin_y = std::min(pmin_y, gy); pmax_y = std::max(pmax_y, gy);
        }
        // shrink-to-fit: keep all shifted points inside the grid
        b.min_x = std::max(b.min_x, -pmin_x);
        b.max_x = std::min(b.max_x, w - 1 - pmax_x);
        b.min_y = std::max(b.min_y, -pmin_y);
        b.max_y = std::min(b.max_y, h - 1 - pmax_y);
        bounds[a] = b;
        (void)smin_x; (void)smax_x; (void)smin_y; (void)smax_y;
    }

    PrecompStack stack(grid, cfg.depth);

    // Lowest-resolution candidates.
    const int step = 1 << stack.max_depth();
    std::vector<Candidate> cands;
    for (size_t a = 0; a < angles.size(); ++a) {
        const LinearBounds& b = bounds[a];
        for (int xo = b.min_x; xo <= b.max_x; xo += step)
            for (int yo = b.min_y; yo <= b.max_y; yo += step)
                cands.push_back(Candidate{static_cast<int>(a), xo, yo, angles[a], -1.0});
    }
    if (cands.empty()) return res;

    score_candidates(stack.level(stack.max_depth()), h, w, scans, cands);
    Candidate best = branch_and_bound(stack, scans, bounds, cands, stack.max_depth(), 0.0);

    res.coarse_score = best.score;
    res.success = std::isfinite(best.score) && best.score >= 0.0;

    Pose2 coarse(predicted_pose.x + best.x_off * resol,
                 predicted_pose.y + best.y_off * resol,
                 predicted_pose.theta + best.orientation);
    Pose2 final_pose = coarse;

    const auto prob = grid.probability();
    if (cfg.do_refine && res.success) {
        // float->double is exact, so the double-point refine helper is
        // bit-identical to the pre-V4.4 float-point version here.
        const MatX2d pts_ref =
            stride_subsample(scan_local, cfg.max_refine_points).cast<double>();
        RefineParams rp;
        rp.iters = cfg.refine_iters;
        rp.damping = cfg.refine_damping;
        rp.eps_stop = cfg.refine_eps_stop;
        rp.step_clip_xy = cfg.refine_step_clip_xy;
        rp.step_clip_th = cfg.refine_step_clip_th;
        rp.w_trans = cfg.refine_w_trans;
        rp.w_rot = cfg.refine_w_rot;
        rp.min_points = cfg.refine_min_points;
        // initial == prior == coarse: identical to the pre-V4.4 refine_lm.
        final_pose = refine_pose_lm(prob.data(), grid.width(), grid.height(),
                                    grid.origin_x(), grid.origin_y(),
                                    grid.resolution(), pts_ref, coarse, coarse, rp);
        res.refined = true;
    }
    res.refined_score = score_pose_on_grid_raw(prob.data(), grid.width(), grid.height(),
                                               grid.origin_x(), grid.origin_y(),
                                               grid.resolution(),
                                               scan_local.cast<double>(),
                                               final_pose).first;
    res.pose = final_pose;
    return res;
}

}  // namespace fusion
