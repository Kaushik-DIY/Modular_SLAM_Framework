#include "scan_to_submap_2d.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>

namespace fusion {

namespace {

// np.rint parity (round-half-even). std::nearbyint honours FE_TONEAREST,
// the default rounding mode == banker's rounding.
inline double rint_he(double v) { return std::nearbyint(v); }

// Python _bresenham: strict > / < comparisons on the error term.
inline void bresenham_cells(int gx0, int gy0, int gx1, int gy1,
                            std::vector<std::pair<int, int>>& out) {
    out.clear();
    const int dx = std::abs(gx1 - gx0), dy = std::abs(gy1 - gy0);
    const int sx = gx0 < gx1 ? 1 : -1, sy = gy0 < gy1 ? 1 : -1;
    int err = dx - dy;
    int x = gx0, y = gy0;
    while (true) {
        out.emplace_back(x, y);
        if (x == gx1 && y == gy1) break;
        const int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x += sx; }
        if (e2 < dx)  { err += dx; y += sy; }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// ProbGrid2D
// ---------------------------------------------------------------------------

ProbGrid2D::ProbGrid2D(double size_m, double res_, double l0_, double l_occ_,
                       double l_free_, double l_min_, double l_max_) {
    res = res_;
    w = static_cast<int>(rint_he(size_m / res_));
    h = static_cast<int>(rint_he(size_m / res_));
    ox = -size_m / 2.0;
    oy = -size_m / 2.0;
    l0 = l0_; l_occ = l_occ_; l_free = l_free_; l_min = l_min_; l_max = l_max_;
    L.assign(static_cast<size_t>(w) * h, static_cast<float>(l0_));
}

const std::vector<float>& ProbGrid2D::probability() const {
    if (_dirty) {
        _prob.resize(L.size());
        for (size_t i = 0; i < L.size(); ++i)
            _prob[i] = static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(L[i]))));
        _dirty = false;
    }
    return _prob;
}

// ---------------------------------------------------------------------------
// SubmapBuilder2D
// ---------------------------------------------------------------------------

Submap2DPtr SubmapBuilder2D::new_submap(const Pose2& pose_world) {
    auto sm = std::make_shared<Submap2D>();
    sm->id = _next_id++;
    sm->grid = ProbGrid2D(_cfg.submap_size_m, _cfg.resolution, _cfg.l0,
                          _cfg.l_occ, _cfg.l_free, _cfg.l_min, _cfg.l_max);
    sm->pose_world = pose_world;
    return sm;
}

void SubmapBuilder2D::ensure_two_active(const Pose2& pose_world) {
    while (_active.size() < 2) _active.push_back(new_submap(pose_world));
}

void SubmapBuilder2D::maybe_finish_oldest(const Pose2& pose_world) {
    // Step 1: mark the oldest active submap finished when full (do NOT remove).
    if (!_active.empty() && !_active[0]->finished &&
        _active[0]->num_inserted >= _cfg.scans_per_submap) {
        _active[0]->finished = true;
        _finished.push_back(_active[0]);
    }
    // Step 2: rotate only when the second submap is mature (>= N/2 scans).
    if (_active.size() >= 2 && _active[0]->finished &&
        _active[1]->num_inserted >= _cfg.scans_per_submap / 2) {
        _active.erase(_active.begin());
        _active.push_back(new_submap(pose_world));
        // startup simultaneous-fill edge case
        if (!_active.empty() && !_active[0]->finished &&
            _active[0]->num_inserted >= _cfg.scans_per_submap) {
            _active[0]->finished = true;
            _finished.push_back(_active[0]);
        }
    }
}

void SubmapBuilder2D::integrate(Submap2D& sm, double origin_x_sub,
                                double origin_y_sub, const MatX2dRef& endpoints_sub) {
    ProbGrid2D& grid = sm.grid;
    int gx0, gy0;
    grid.world_to_grid(origin_x_sub, origin_y_sub, gx0, gy0);

    std::vector<std::pair<int, int>> cells;
    for (int i = 0; i < endpoints_sub.rows(); ++i) {
        int gx1, gy1;
        grid.world_to_grid(endpoints_sub(i, 0), endpoints_sub(i, 1), gx1, gy1);
        if (!grid.in_bounds(gx0, gy0)) continue;   // ray skipped entirely
        if (!grid.in_bounds(gx1, gy1)) continue;
        bresenham_cells(gx0, gy0, gx1, gy1, cells);
        for (size_t k = 0; k + 1 < cells.size(); ++k)
            grid.update_cell(cells[k].first, cells[k].second, grid.l_free);
        grid.update_cell(gx1, gy1, grid.l_occ);
    }
}

bool SubmapBuilder2D::insert_scan(const Pose2& pose_world, const MatX2dRef& scan_local) {
    if (!_initialized) {
        ensure_two_active(pose_world);
        _initialized = true;
    }

    // endpoints in world frame
    const double c = std::cos(pose_world.theta), s = std::sin(pose_world.theta);
    MatX2dRef endpoints_world(scan_local.rows(), 2);
    for (int i = 0; i < scan_local.rows(); ++i) {
        endpoints_world(i, 0) = c * scan_local(i, 0) - s * scan_local(i, 1) + pose_world.x;
        endpoints_world(i, 1) = s * scan_local(i, 0) + c * scan_local(i, 1) + pose_world.y;
    }

    for (auto& sm : _active) {
        if (sm->finished) continue;
        const Pose2 inv = sm->pose_world.inverse();
        const double ci = std::cos(inv.theta), si = std::sin(inv.theta);
        const double ox_sub = ci * pose_world.x - si * pose_world.y + inv.x;
        const double oy_sub = si * pose_world.x + ci * pose_world.y + inv.y;
        MatX2dRef endpoints_sub(endpoints_world.rows(), 2);
        for (int i = 0; i < endpoints_world.rows(); ++i) {
            endpoints_sub(i, 0) = ci * endpoints_world(i, 0) - si * endpoints_world(i, 1) + inv.x;
            endpoints_sub(i, 1) = si * endpoints_world(i, 0) + ci * endpoints_world(i, 1) + inv.y;
        }
        integrate(*sm, ox_sub, oy_sub, endpoints_sub);
        sm->num_inserted += 1;
    }

    maybe_finish_oldest(pose_world);
    ensure_two_active(pose_world);
    return true;
}

// ---------------------------------------------------------------------------
// MaxPoolStack + brute-force search
// ---------------------------------------------------------------------------

MaxPoolStack::MaxPoolStack(const std::vector<float>& prob, int h, int w,
                           int num_levels) {
    _levels.push_back(prob);
    _hs.push_back(h);
    _ws.push_back(w);
    for (int l = 1; l < num_levels; ++l) {
        const auto& prev = _levels.back();
        const int ph = _hs.back(), pw = _ws.back();
        const int h2 = ph / 2, w2 = pw / 2;
        std::vector<float> out(static_cast<size_t>(h2) * w2);
        for (int y = 0; y < h2; ++y)
            for (int x = 0; x < w2; ++x) {
                const float a = prev[static_cast<size_t>(2 * y) * pw + 2 * x];
                const float b = prev[static_cast<size_t>(2 * y) * pw + 2 * x + 1];
                const float c = prev[static_cast<size_t>(2 * y + 1) * pw + 2 * x];
                const float d = prev[static_cast<size_t>(2 * y + 1) * pw + 2 * x + 1];
                out[static_cast<size_t>(y) * w2 + x] = std::max(std::max(a, b), std::max(c, d));
            }
        _levels.push_back(std::move(out));
        _hs.push_back(h2);
        _ws.push_back(w2);
    }
}

std::pair<Pose2, double> bruteforce_search(
    const MaxPoolStack& stack, int level, double grid_ox, double grid_oy,
    double res, const MatX2dRef& pts_local, const Pose2& center,
    const SearchWindow& win, int min_valid) {
    const auto& img = stack.level(level);
    const int h = stack.level_h(level), w = stack.level_w(level);
    const double scale = static_cast<double>(1 << level);
    const int n = static_cast<int>(pts_local.rows());

    auto offsets = [](double window, double step) {
        std::vector<double> v;
        for (int k = 0;; ++k) {
            const double o = -window + k * step;
            if (o > window + 1e-9) break;
            v.push_back(o);
        }
        return v;
    };
    const auto xs = offsets(win.xy_window, win.xy_step);
    const auto ys = offsets(win.xy_window, win.xy_step);
    const auto ths = offsets(win.theta_window, win.theta_step);

    Pose2 best_pose = center;
    double best_score = -1e9;

    std::vector<double> bx(n), by(n);
    for (double dth : ths) {
        const double th = normalize_angle(center.theta + dth);
        const double c = std::cos(th), s = std::sin(th);
        for (int i = 0; i < n; ++i) {
            // base grid coords at the CENTER translation, level scale applied;
            // dx/dy shift in grid units added per candidate (vectorized parity)
            const double qx = c * pts_local(i, 0) - s * pts_local(i, 1) + center.x;
            const double qy = s * pts_local(i, 0) + c * pts_local(i, 1) + center.y;
            bx[i] = (qx - grid_ox) / (res * scale);
            by[i] = (qy - grid_oy) / (res * scale);
        }
        for (double dx : xs) {
            const double offx = dx / (res * scale);
            for (double dy : ys) {
                const double offy = dy / (res * scale);
                double sum = 0.0;
                int valid = 0;
                for (int i = 0; i < n; ++i) {
                    const int gx = static_cast<int>(rint_he(bx[i] + offx));
                    const int gy = static_cast<int>(rint_he(by[i] + offy));
                    if (gx < 0 || gx >= w || gy < 0 || gy >= h) continue;
                    sum += img[static_cast<size_t>(gy) * w + gx];
                    ++valid;
                }
                const double score = valid < min_valid ? -1e9 : sum / valid;
                if (score > best_score) {
                    best_score = score;
                    best_pose = Pose2(center.x + dx, center.y + dy, th);
                }
            }
        }
    }
    return {best_pose, best_score};
}

// ---------------------------------------------------------------------------
// submap_match (TwoStageBruteForceSubmapBackend.match port)
// ---------------------------------------------------------------------------

SubmapMatchResult submap_match(const SubmapBuilder2D& builder,
                               const MatX2dRef& scan_local,
                               const Pose2& predicted_world,
                               const SubmapMatcherConfig& cfg) {
    SubmapMatchResult res;
    res.pose_world = predicted_world;
    if (builder.active().empty()) return res;   // success=false, score=-1

    const Submap2D& target = *builder.active()[0];
    const Pose2 inv = target.pose_world.inverse();
    const Pose2 pred_sub = inv.compose(predicted_world);

    // coarse-stage stride subsample (match diet)
    MatX2dRef pts_match = scan_local;
    if (pts_match.rows() > cfg.max_match_points) {
        const int stride = std::max<int>(1, pts_match.rows() / cfg.max_match_points);
        MatX2dRef sub((pts_match.rows() + stride - 1) / stride, 2);
        int k = 0;
        for (int i = 0; i < pts_match.rows(); i += stride) sub.row(k++) = pts_match.row(i);
        pts_match = sub.topRows(k);
    }

    const auto& prob = target.grid.probability();
    MaxPoolStack stack(prob, target.grid.h, target.grid.w, cfg.precomp_levels);

    auto [coarse_pose, coarse_raw] = bruteforce_search(
        stack, cfg.coarse.level, target.grid.ox, target.grid.oy, target.grid.res,
        pts_match, pred_sub, cfg.coarse, cfg.min_valid);
    auto [fine_pose, fine_raw] = bruteforce_search(
        stack, cfg.fine.level, target.grid.ox, target.grid.oy, target.grid.res,
        pts_match, coarse_pose, cfg.fine, cfg.min_valid);

    const double raw = std::isfinite(fine_raw) ? fine_raw : -1.0;
    const double coarse_score = raw >= 0.0 ? raw : -1.0;
    const bool coarse_available =
        coarse_score >= 0.0 && std::isfinite(fine_pose.x) &&
        std::isfinite(fine_pose.y) && std::isfinite(fine_pose.theta);
    const bool coarse_trusted = coarse_available && coarse_score >= cfg.min_score;

    if (!coarse_trusted && cfg.reject_below_min_score) {
        res.success = false;
        res.score = coarse_score;
        res.pose_world = predicted_world;
        return res;
    }

    const Pose2 initial = coarse_trusted ? fine_pose : pred_sub;

    // refine-stage stride subsample of the FULL scan
    MatX2dRef pts_ref = scan_local;
    if (pts_ref.rows() > cfg.max_refine_points) {
        const int stride = std::max<int>(1, pts_ref.rows() / cfg.max_refine_points);
        MatX2dRef sub((pts_ref.rows() + stride - 1) / stride, 2);
        int k = 0;
        for (int i = 0; i < pts_ref.rows(); i += stride) sub.row(k++) = pts_ref.row(i);
        pts_ref = sub.topRows(k);
    }
    RefineParams rp;
    rp.iters = cfg.refine_iters;
    rp.damping = cfg.refine_damping;
    rp.eps_stop = cfg.refine_eps_stop;
    rp.step_clip_xy = cfg.refine_step_clip_xy;
    rp.step_clip_th = cfg.refine_step_clip_th;
    rp.w_trans = cfg.refine_w_trans;
    rp.w_rot = cfg.refine_w_rot;
    rp.min_points = cfg.refine_min_points;
    const Pose2 final_sub = refine_pose_lm(
        prob.data(), target.grid.w, target.grid.h, target.grid.ox, target.grid.oy,
        target.grid.res, pts_ref, initial, pred_sub, rp);

    const auto [refined_score, refine_valid] = score_pose_on_grid_raw(
        prob.data(), target.grid.w, target.grid.h, target.grid.ox, target.grid.oy,
        target.grid.res, pts_ref, final_sub);

    const bool finite = std::isfinite(final_sub.x) && std::isfinite(final_sub.y) &&
                        std::isfinite(final_sub.theta);
    if (!finite || refine_valid < cfg.refine_min_points) {
        res.success = false;
        res.score = -1.0;
        res.refined_score = refined_score;
        res.pose_world = predicted_world;
        return res;
    }

    res.success = true;
    res.score = coarse_score;
    res.refined_score = refined_score;
    res.pose_world = target.pose_world.compose(final_sub);
    return res;
}

}  // namespace fusion
