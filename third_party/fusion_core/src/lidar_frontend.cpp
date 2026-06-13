#include "lidar_frontend.h"
#include <chrono>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace fusion {

MatX2d fixed_voxel_filter(const MatX2d& pts, double voxel_size) {
    if (voxel_size <= 0.0 || pts.rows() == 0) return pts;
    const int n = static_cast<int>(pts.rows());

    struct Key { int64_t ix, iy; int idx; };
    std::vector<Key> keys(n);
    for (int i = 0; i < n; ++i)
        keys[i] = {static_cast<int64_t>(std::floor(pts(i, 0) / voxel_size)),
                   static_cast<int64_t>(std::floor(pts(i, 1) / voxel_size)), i};
    // np.unique(axis=0) emits voxels in lexicographic (ix, iy) order — the
    // output row order is part of the contract (stride subsampling downstream).
    // stable_sort keeps original index order within a voxel, so the centroid
    // summation order matches np.add.at exactly.
    std::stable_sort(keys.begin(), keys.end(), [](const Key& a, const Key& b) {
        return a.ix != b.ix ? a.ix < b.ix : a.iy < b.iy;
    });

    MatX2d out(n, 2);
    int m = 0;
    int i = 0;
    while (i < n) {
        int j = i;
        double sx = 0.0, sy = 0.0;
        while (j < n && keys[j].ix == keys[i].ix && keys[j].iy == keys[i].iy) {
            sx += pts(keys[j].idx, 0);
            sy += pts(keys[j].idx, 1);
            ++j;
        }
        const double c = static_cast<double>(j - i);
        out(m, 0) = sx / c;
        out(m, 1) = sy / c;
        ++m;
        i = j;
    }
    return out.topRows(m);
}

MatX2d adaptive_voxel_filter(const MatX2d& pts, double max_voxel_size,
                             int min_num_points, int num_iterations) {
    if (pts.rows() == 0) return pts;
    if (max_voxel_size <= 0.0) return pts;
    if (pts.rows() <= min_num_points) return pts;

    double low = 0.0, high = max_voxel_size;
    MatX2d best = pts;
    for (int it = 0; it < std::max(1, num_iterations); ++it) {
        const double mid = 0.5 * (low + high);
        if (mid <= 1e-9) break;
        MatX2d filtered = fixed_voxel_filter(pts, mid);
        if (filtered.rows() >= min_num_points) {
            best = filtered;
            low = mid;
        } else {
            high = mid;
        }
    }
    return best;
}

MatX2d voxel_preprocess(const MatX2d& pts, const VoxelFilterConfig& cfg) {
    if (!cfg.enabled || pts.rows() == 0) return pts;
    const MatX2d fixed = fixed_voxel_filter(pts, cfg.fixed_size);
    return adaptive_voxel_filter(fixed, cfg.adaptive_max_size,
                                 cfg.adaptive_min_points, cfg.adaptive_iters);
}

// ---------------------------------------------------------------------------
// LidarFrontend orchestrator (hector/adapter.py process_scan port, in the
// fusion2 wiring: motion filter dormant -> match + insert every scan; no odom)
// ---------------------------------------------------------------------------

LidarFrontend::LidarFrontend(LidarFrontendConfig cfg)
    : _cfg(std::move(cfg)), _extrap(_cfg.extrap) {
    if (_cfg.matcher == LidarMatcherKind::SCAN_TO_SUBMAP)
        _submaps = std::make_unique<SubmapBuilder2D>(_cfg.submap_builder);
    else
        _s2m = std::make_unique<MapMatcher>(_cfg.map_matcher);
}

bool LidarFrontend::keyframe_decision(const Pose2& pose, double t) {
    if (!_has_kf) {
        _last_kf_pose = pose;
        _last_kf_t = t;
        _has_kf = true;
        return true;
    }
    const double dx = pose.x - _last_kf_pose.x;
    const double dy = pose.y - _last_kf_pose.y;
    const double dth = std::abs(normalize_angle(pose.theta - _last_kf_pose.theta));
    if (std::hypot(dx, dy) >= _cfg.keyframe.min_dist_m ||
        dth >= _cfg.keyframe.min_angle_rad ||
        (t - _last_kf_t) >= _cfg.keyframe.min_dt_s) {
        _last_kf_pose = pose;
        _last_kf_t = t;
        return true;
    }
    return false;
}

LidarScanResult LidarFrontend::process(const MatX2d& scan_xy, double t) {
    const auto t0 = std::chrono::steady_clock::now();
    ++_k;
    _last_pts = voxel_preprocess(scan_xy, _cfg.voxel);

    const Pose2 pred = _extrap.has_state() ? _extrap.predict(t) : Pose2(0.0, 0.0, 0.0);

    // insert filter (dormant by default: do_insert = true)
    bool motion_insert = true;
    if (_cfg.insert_filter.enabled && _has_insert) {
        const double dtr = std::hypot(pred.x - _last_insert_pose.x,
                                      pred.y - _last_insert_pose.y);
        const double dro = std::abs(normalize_angle(pred.theta - _last_insert_pose.theta));
        const double dti = t - _last_insert_t;
        motion_insert = dtr > _cfg.insert_filter.max_dist_m ||
                        dro > _cfg.insert_filter.max_angle_rad ||
                        dti > _cfg.insert_filter.max_time_s;
    }

    LidarScanResult out = (_cfg.matcher == LidarMatcherKind::SCAN_TO_SUBMAP)
                              ? process_s2s(_last_pts, t, pred, motion_insert)
                              : process_s2m(_last_pts, t, pred, motion_insert);
    out.num_points = static_cast<int>(_last_pts.rows());
    out.is_keyframe = keyframe_decision(out.pose, t);
    out.process_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0).count();
    return out;
}

LidarScanResult LidarFrontend::process_s2s(const MatX2d& pts, double t,
                                           const Pose2& pred, bool do_insert) {
    LidarScanResult res;

    // motion-filter skip path (opt-in; dead-reckon, but still insert)
    if (_cfg.insert_filter.enabled && _cfg.insert_filter.skip_below &&
        _has_insert && !do_insert) {
        res.pose = pred;
        res.skipped = true;
        _submaps->insert_scan(pred, pts);   // keep submaps dense (anchor NOT updated)
        res.inserted = true;
        return res;
    }

    const SubmapMatchResult m = submap_match(*_submaps, pts, pred, _cfg.submap_matcher);
    res.matched = m.success;
    res.score = m.score;
    res.refined_score = m.refined_score;
    res.pose = m.pose_world;             // == pred on failure (FALLBACK)
    res.fallback = !m.success;

    if (m.success) _extrap.add_pose(t, m.pose_world);

    if (do_insert) {
        _submaps->insert_scan(res.pose, pts);
        res.inserted = true;
        _last_insert_pose = res.pose;
        _last_insert_t = t;
        _has_insert = true;
    }
    return res;
}

LidarScanResult LidarFrontend::process_s2m(const MatX2d& pts, double t,
                                           const Pose2& pred, bool do_insert_motion) {
    LidarScanResult res;

    // skip path (opt-in): dead-reckon, NO insert (keeps the global grid crisp)
    if (_cfg.insert_filter.enabled && _cfg.insert_filter.skip_below &&
        _s2m->initialized() && _has_insert && !do_insert_motion) {
        res.pose = pred;
        res.skipped = true;
        return res;
    }

    // insert cadence: motion filter when enabled, else every map_update_every-th
    bool do_insert;
    if (_cfg.insert_filter.enabled && _cfg.insert_filter.skip_below)
        do_insert = do_insert_motion;
    else
        do_insert = (_k % std::max(1, _cfg.map_matcher.map_update_every)) == 0;

    const MapMatchResult m = _s2m->match(pts, pred);
    res.matched = m.success;
    res.score = m.score;
    res.pose = m.pose_world;             // == pred on failure / bootstrap
    res.fallback = !m.success;

    if (m.bootstrap) return res;          // inserted inside match(); no extrap update

    if (m.success) _extrap.add_pose(t, m.pose_world);

    if (do_insert) {
        _s2m->insert(res.pose, pts);      // dead-reckoned insert on failure (by design)
        res.inserted = true;
        _last_insert_pose = res.pose;
        _last_insert_t = t;
        _has_insert = true;
        if (!m.success) _extrap.add_pose(t, pred);   // dead-reckon feedback (adapter)
    }
    return res;
}

}  // namespace fusion
