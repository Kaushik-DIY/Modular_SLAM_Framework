#include "vo_frontend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <unordered_map>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>

namespace fusion {

namespace {

// Hamming distance between two 32-byte ORB descriptors.
inline int hamming32(const uint8_t* a, const uint8_t* b) {
    int d = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t x;
        uint64_t y;
        std::memcpy(&x, a + 8 * i, 8);
        std::memcpy(&y, b + 8 * i, 8);
        d += static_cast<int>(__builtin_popcountll(x ^ y));
    }
    return d;
}

inline Eigen::Matrix4d exp_se3(const Eigen::Matrix<double, 6, 1>& xi) {
    // xi = [omega, v]
    const Eigen::Vector3d w = xi.head<3>(), v = xi.tail<3>();
    const double th = w.norm();
    Eigen::Matrix3d R, V;
    Eigen::Matrix3d W = (Eigen::Matrix3d() << 0, -w.z(), w.y(),
                         w.z(), 0, -w.x(), -w.y(), w.x(), 0).finished();
    if (th < 1e-10) {
        R = Eigen::Matrix3d::Identity() + W;
        V = Eigen::Matrix3d::Identity();
    } else {
        const double a = std::sin(th) / th;
        const double b = (1 - std::cos(th)) / (th * th);
        const double c = (1 - a) / (th * th);
        R = Eigen::Matrix3d::Identity() + a * W + b * W * W;
        V = Eigen::Matrix3d::Identity() + b * W + c * W * W;
    }
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = V * v;
    return T;
}

}  // namespace

VoFrontend::VoFrontend(VoConfig cfg) : _cfg(cfg) {
    _orb = cv::ORB::create(cfg.n_features * 2,  // over-extract, bucket down
                           cfg.scale_factor, cfg.n_levels);
    _level_sigma2.resize(cfg.n_levels);
    float s2 = 1.0f;
    const float f2 = cfg.scale_factor * cfg.scale_factor;
    for (int i = 0; i < cfg.n_levels; ++i) {
        _level_sigma2[i] = s2;
        s2 *= f2;
    }
}

void VoFrontend::extract(const cv::Mat& gray, std::vector<cv::KeyPoint>& kps,
                         cv::Mat& des) const {
    std::vector<cv::KeyPoint> raw;
    cv::Mat raw_des;
    _orb->detectAndCompute(gray, cv::noArray(), raw, raw_des);
    if (raw.empty()) { kps.clear(); des = cv::Mat(); return; }

    // Grid bucketing: top max_per_cell by response per cell, then global cap.
    const int cw = (gray.cols + _cfg.grid_cell_px - 1) / _cfg.grid_cell_px;
    const int ch = (gray.rows + _cfg.grid_cell_px - 1) / _cfg.grid_cell_px;
    std::vector<std::vector<int>> cells(static_cast<size_t>(cw) * ch);
    for (int i = 0; i < static_cast<int>(raw.size()); ++i) {
        const int gx = static_cast<int>(raw[i].pt.x) / _cfg.grid_cell_px;
        const int gy = static_cast<int>(raw[i].pt.y) / _cfg.grid_cell_px;
        cells[static_cast<size_t>(gy) * cw + gx].push_back(i);
    }
    std::vector<int> keep;
    keep.reserve(raw.size());
    for (auto& cell : cells) {
        if (static_cast<int>(cell.size()) > _cfg.max_per_cell) {
            std::partial_sort(cell.begin(), cell.begin() + _cfg.max_per_cell, cell.end(),
                              [&](int a, int b) { return raw[a].response > raw[b].response; });
            cell.resize(_cfg.max_per_cell);
        }
        keep.insert(keep.end(), cell.begin(), cell.end());
    }
    if (static_cast<int>(keep.size()) > _cfg.n_features) {
        std::partial_sort(keep.begin(), keep.begin() + _cfg.n_features, keep.end(),
                          [&](int a, int b) { return raw[a].response > raw[b].response; });
        keep.resize(_cfg.n_features);
    }
    kps.clear();
    kps.reserve(keep.size());
    des.create(static_cast<int>(keep.size()), 32, CV_8U);
    for (int j = 0; j < static_cast<int>(keep.size()); ++j) {
        kps.push_back(raw[keep[j]]);
        raw_des.row(keep[j]).copyTo(des.row(j));
    }
}

VoKeyframePtr VoFrontend::make_keyframe(double stamp, const Eigen::Matrix4d& Twc,
                                        const std::vector<cv::KeyPoint>& kps,
                                        const cv::Mat& des, const cv::Mat& depth,
                                        const std::vector<std::pair<int, int>>& matches,
                                        const std::vector<bool>& inlier_mask) {
    auto kf = std::make_shared<VoKeyframe>();
    kf->id = _next_kf_id++;
    kf->stamp = stamp;
    kf->Twc = Twc;
    const int n = static_cast<int>(kps.size());
    kf->kpts.resize(n, 2);
    kf->pts3d_cam.resize(n, 3);
    kf->octave.resize(n);
    kf->des = des.clone();

    // Attach observations from this frame to existing matched window points.
    // matches.first indexes the gathered arrays, which mirror _points 1:1.
    std::vector<char> matched_kp(n, 0);
    for (size_t k = 0; k < matches.size(); ++k) {
        if (k < inlier_mask.size() && !inlier_mask[k]) continue;
        const int p = matches[k].first, i = matches[k].second;
        if (p >= 0 && p < static_cast<int>(_points.size()) && i >= 0 && i < n) {
            _points[p].obs.push_back({kf->id, i});
            matched_kp[i] = 1;
        }
    }

    // Payload (all keypoints) + fresh-point candidates (valid depth, unmatched).
    struct Cand { float z; int idx; };
    std::vector<Cand> fresh;
    fresh.reserve(n);
    const Eigen::Matrix3d R = Twc.block<3, 3>(0, 0);
    const Eigen::Vector3d t = Twc.block<3, 1>(0, 3);
    for (int i = 0; i < n; ++i) {
        const float u = kps[i].pt.x, v = kps[i].pt.y;
        kf->kpts(i, 0) = u;
        kf->kpts(i, 1) = v;
        kf->octave[i] = std::clamp(kps[i].octave, 0, _cfg.n_levels - 1);
        float z = 0.0f;
        const int ui = static_cast<int>(u), vi = static_cast<int>(v);
        if (ui >= 0 && ui < depth.cols && vi >= 0 && vi < depth.rows)
            z = static_cast<float>(depth.at<uint16_t>(vi, ui)) /
                static_cast<float>(_cfg.depth_factor);
        if (z > _cfg.depth_min && z < _cfg.depth_max) {
            kf->pts3d_cam(i, 0) = static_cast<float>((u - _cfg.cx) / _cfg.fx * z);
            kf->pts3d_cam(i, 1) = static_cast<float>((v - _cfg.cy) / _cfg.fy * z);
            kf->pts3d_cam(i, 2) = z;
            // Every keyframe seeds fresh, well-localized depth points (as the
            // original lean VO did) to keep the tracking pool rich; matched
            // keypoints ALSO attach an observation to their existing point
            // above (that is what gives local BA its multi-view constraints).
            fresh.push_back({z, i});
        } else {
            kf->pts3d_cam.row(i).setConstant(std::numeric_limits<float>::quiet_NaN());
        }
    }

    // New map points: <=max_points_per_kf shallowest unmatched valid-depth kps.
    std::sort(fresh.begin(), fresh.end(),
              [](const Cand& a, const Cand& b) { return a.z < b.z; });
    const int m = std::min<int>(_cfg.max_points_per_kf, static_cast<int>(fresh.size()));
    for (int j = 0; j < m; ++j) {
        const int i = fresh[j].idx;
        VoMapPoint mp;
        const Eigen::Vector3d pc = kf->pts3d_cam.row(i).cast<double>().transpose();
        mp.pw = R * pc + t;
        std::memcpy(mp.des.data(), kf->des.ptr<uint8_t>(i), 32);
        mp.octave = kf->octave[i];
        mp.obs.push_back({kf->id, i});
        _points.push_back(std::move(mp));
    }

    _window.push_back(kf);
    // Slide the window; prune observations of evicted keyframes and drop any
    // map point left with no observers.
    while (static_cast<int>(_window.size()) > _cfg.window_size) {
        const int dead = _window.front()->id;
        _window.pop_front();
        size_t w = 0;
        for (size_t r = 0; r < _points.size(); ++r) {
            auto& mp = _points[r];
            mp.obs.erase(std::remove_if(mp.obs.begin(), mp.obs.end(),
                          [dead](const VoObs& o) { return o.kf_id == dead; }),
                         mp.obs.end());
            if (!mp.obs.empty()) {
                if (w != r) _points[w] = std::move(_points[r]);
                ++w;
            }
        }
        _points.resize(w);
    }
    rebuild_gathered();
    return kf;
}

void VoFrontend::rebuild_gathered() {
    const int N = static_cast<int>(_points.size());
    _g_world.resize(N, 3);
    _g_des.create(N, 32, CV_8U);
    _g_octave.resize(N);
    for (int p = 0; p < N; ++p) {
        _g_world.row(p) = _points[p].pw.transpose().cast<float>();
        std::memcpy(_g_des.ptr<uint8_t>(p), _points[p].des.data(), 32);
        _g_octave[p] = _points[p].octave;
    }
}

int VoFrontend::match_window(const std::vector<cv::KeyPoint>& kps, const cv::Mat& des,
                             const Eigen::Matrix4d& Twc_pred, double radius_px,
                             std::vector<std::pair<int, int>>& matches) const {
    matches.clear();
    if (_g_world.rows() == 0 || kps.empty()) return 0;

    // grid-bin current keypoints for radius lookup
    const int cell = 16;
    int max_x = 0, max_y = 0;
    for (const auto& kp : kps) {
        max_x = std::max(max_x, static_cast<int>(kp.pt.x));
        max_y = std::max(max_y, static_cast<int>(kp.pt.y));
    }
    const int cw = max_x / cell + 2, ch = max_y / cell + 2;
    std::vector<std::vector<int>> bins(static_cast<size_t>(cw) * ch);
    for (int i = 0; i < static_cast<int>(kps.size()); ++i)
        bins[static_cast<size_t>(kps[i].pt.y / cell) * cw +
             static_cast<int>(kps[i].pt.x / cell)].push_back(i);

    const Eigen::Matrix4d Tcw = Twc_pred.inverse();
    const Eigen::Matrix3d R = Tcw.block<3, 3>(0, 0);
    const Eigen::Vector3d t = Tcw.block<3, 1>(0, 3);
    std::vector<int> best_for_kp(kps.size(), -1);
    std::vector<int> best_dist_kp(kps.size(), INT32_MAX);

    for (int p = 0; p < _g_world.rows(); ++p) {
        const Eigen::Vector3d pc = R * _g_world.row(p).cast<double>().transpose() + t;
        if (pc.z() <= 0.05) continue;
        const double u = _cfg.fx * pc.x() / pc.z() + _cfg.cx;
        const double v = _cfg.fy * pc.y() / pc.z() + _cfg.cy;
        const int gx0 = std::max(0, static_cast<int>((u - radius_px) / cell));
        const int gx1 = std::min(cw - 1, static_cast<int>((u + radius_px) / cell));
        const int gy0 = std::max(0, static_cast<int>((v - radius_px) / cell));
        const int gy1 = std::min(ch - 1, static_cast<int>((v + radius_px) / cell));
        if (gx1 < 0 || gy1 < 0 || gx0 >= cw || gy0 >= ch) continue;

        int best = _cfg.max_hamming + 1, second = INT32_MAX, best_i = -1;
        const uint8_t* pd = _g_des.ptr<uint8_t>(p);
        for (int gy = gy0; gy <= gy1; ++gy)
            for (int gx = gx0; gx <= gx1; ++gx)
                for (int i : bins[static_cast<size_t>(gy) * cw + gx]) {
                    const double du = kps[i].pt.x - u, dv = kps[i].pt.y - v;
                    if (du * du + dv * dv > radius_px * radius_px) continue;
                    const int d = hamming32(pd, des.ptr<uint8_t>(i));
                    if (d < best) { second = best; best = d; best_i = i; }
                    else if (d < second) second = d;
                }
        if (best_i < 0) continue;
        if (second <= _cfg.max_hamming &&
            static_cast<double>(best) > _cfg.ratio_test * second) continue;
        // keep best point per keypoint
        if (best < best_dist_kp[best_i]) {
            best_dist_kp[best_i] = best;
            best_for_kp[best_i] = p;
        }
    }
    for (int i = 0; i < static_cast<int>(kps.size()); ++i)
        if (best_for_kp[i] >= 0) matches.emplace_back(best_for_kp[i], i);
    return static_cast<int>(matches.size());
}

int VoFrontend::match_bruteforce(const std::vector<cv::KeyPoint>& kps, const cv::Mat& des,
                                 std::vector<std::pair<int, int>>& matches) const {
    matches.clear();
    const int P = _g_des.rows, N = des.rows;
    if (P == 0 || N == 0) return 0;
    // Best window point per current keypoint, strict hamming + ratio (the
    // pose-free path must be high-precision: a wrong 3D-2D match feeds PnP).
    for (int i = 0; i < N; ++i) {
        const uint8_t* di = des.ptr<uint8_t>(i);
        int best = 256, second = 256, bp = -1;
        for (int p = 0; p < P; ++p) {
            const int d = hamming32(di, _g_des.ptr<uint8_t>(p));
            if (d < best) { second = best; best = d; bp = p; }
            else if (d < second) second = d;
        }
        if (bp >= 0 && best <= _cfg.bf_max_hamming &&
            static_cast<double>(best) < _cfg.bf_ratio_test * second)
            matches.emplace_back(bp, i);
    }
    return static_cast<int>(matches.size());
}

bool VoFrontend::recover_pose(const std::vector<cv::KeyPoint>& kps,
                              const std::vector<std::pair<int, int>>& matches,
                              Eigen::Matrix4d& Tcw) const {
    if (static_cast<int>(matches.size()) < _cfg.min_inliers_reinit) return false;
    std::vector<cv::Point3f> obj;
    std::vector<cv::Point2f> img;
    obj.reserve(matches.size());
    img.reserve(matches.size());
    for (const auto& [p, i] : matches) {
        obj.emplace_back(_g_world(p, 0), _g_world(p, 1), _g_world(p, 2));
        img.emplace_back(kps[i].pt);
    }
    const cv::Mat K = (cv::Mat_<double>(3, 3) << _cfg.fx, 0, _cfg.cx,
                                                 0, _cfg.fy, _cfg.cy, 0, 0, 1);
    cv::Mat rvec, tvec;
    std::vector<int> inl;
    if (!cv::solvePnPRansac(obj, img, K, cv::noArray(), rvec, tvec, false, 100,
                            static_cast<float>(_cfg.bf_pnp_reproj_px), 0.98, inl,
                            cv::SOLVEPNP_EPNP))
        return false;
    if (static_cast<int>(inl.size()) < _cfg.min_inliers_reinit) return false;
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    Tcw = Eigen::Matrix4d::Identity();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) Tcw(r, c) = R.at<double>(r, c);
        Tcw(r, 3) = tvec.at<double>(r);
    }
    return true;
}

int VoFrontend::optimize_pose(const std::vector<cv::KeyPoint>& kps,
                              const std::vector<std::pair<int, int>>& matches,
                              Eigen::Matrix4d& Tcw, std::vector<bool>& inlier_mask) const {
    const int n = static_cast<int>(matches.size());
    inlier_mask.assign(n, true);
    if (n < 6) return 0;

    int inliers = n;
    for (int round = 0; round < _cfg.opt_rounds; ++round) {
        for (int it = 0; it < _cfg.opt_iters_per_round; ++it) {
            Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
            const Eigen::Matrix3d R = Tcw.block<3, 3>(0, 0);
            const Eigen::Vector3d t = Tcw.block<3, 1>(0, 3);
            for (int k = 0; k < n; ++k) {
                if (!inlier_mask[k]) continue;
                const auto& [p, i] = matches[k];
                const Eigen::Vector3d pc = R * _g_world.row(p).cast<double>().transpose() + t;
                if (pc.z() <= 1e-6) { continue; }
                const double iz = 1.0 / pc.z();
                const double u = _cfg.fx * pc.x() * iz + _cfg.cx;
                const double v = _cfg.fy * pc.y() * iz + _cfg.cy;
                const double s2 = _level_sigma2[std::clamp(kps[i].octave, 0, _cfg.n_levels - 1)];
                const double w_info = 1.0 / s2;
                Eigen::Vector2d r(u - kps[i].pt.x, v - kps[i].pt.y);
                // Huber weight
                const double rn = r.norm() / std::sqrt(s2);
                double w_h = 1.0;
                if (rn > _cfg.huber_delta_px) w_h = _cfg.huber_delta_px / rn;
                // d(u,v)/d(pc)
                Eigen::Matrix<double, 2, 3> Jp;
                Jp << _cfg.fx * iz, 0, -_cfg.fx * pc.x() * iz * iz,
                      0, _cfg.fy * iz, -_cfg.fy * pc.y() * iz * iz;
                // d(pc)/d(xi) with left-mult update: [-[pc]x | I]
                Eigen::Matrix<double, 3, 6> Jx;
                Jx << 0, pc.z(), -pc.y(), 1, 0, 0,
                      -pc.z(), 0, pc.x(), 0, 1, 0,
                      pc.y(), -pc.x(), 0, 0, 0, 1;
                const Eigen::Matrix<double, 2, 6> J = Jp * Jx;
                H += w_h * w_info * J.transpose() * J;
                g += w_h * w_info * J.transpose() * r;
            }
            const Eigen::Matrix<double, 6, 1> dx =
                (H + 1e-6 * Eigen::Matrix<double, 6, 6>::Identity()).ldlt().solve(-g);
            Tcw = exp_se3(dx) * Tcw;
            if (dx.norm() < 1e-8) break;
        }
        // outlier rejection
        const Eigen::Matrix3d R = Tcw.block<3, 3>(0, 0);
        const Eigen::Vector3d t = Tcw.block<3, 1>(0, 3);
        inliers = 0;
        for (int k = 0; k < n; ++k) {
            const auto& [p, i] = matches[k];
            const Eigen::Vector3d pc = R * _g_world.row(p).cast<double>().transpose() + t;
            bool ok = pc.z() > 1e-6;
            if (ok) {
                const double u = _cfg.fx * pc.x() / pc.z() + _cfg.cx;
                const double v = _cfg.fy * pc.y() / pc.z() + _cfg.cy;
                const double du = u - kps[i].pt.x, dv = v - kps[i].pt.y;
                const double s2 = _level_sigma2[std::clamp(kps[i].octave, 0, _cfg.n_levels - 1)];
                ok = (du * du + dv * dv) / s2 <= _cfg.chi2_outlier;
            }
            inlier_mask[k] = ok;
            inliers += ok;
        }
        if (inliers < 6) break;
    }
    return inliers;
}

void VoFrontend::local_ba() {
    const int C = static_cast<int>(_window.size());
    if (C < 2) return;

    // Camera-from-world poses; oldest window KF (pos 0) fixed as gauge.
    std::unordered_map<int, int> kf_pos;
    std::vector<Eigen::Matrix4d> Tcw(C);
    for (int c = 0; c < C; ++c) {
        kf_pos[_window[c]->id] = c;
        Tcw[c] = _window[c]->Twc.inverse();
    }
    const int NV = C - 1;                       // non-fixed cameras
    auto cam_var = [](int pos) { return pos - 1; };  // pos 0 -> -1 (fixed)

    // Only points co-observed (>= min_obs_for_ba) constrain relative poses.
    std::vector<int> active;
    active.reserve(_points.size());
    for (int p = 0; p < static_cast<int>(_points.size()); ++p)
        if (static_cast<int>(_points[p].obs.size()) >= _cfg.min_obs_for_ba)
            active.push_back(p);
    if (active.empty()) return;
    // Bound BA cost: keep the most-observed (most-constraining) points.
    if (static_cast<int>(active.size()) > _cfg.local_ba_max_points) {
        std::partial_sort(active.begin(), active.begin() + _cfg.local_ba_max_points,
                          active.end(), [&](int a, int b) {
                              return _points[a].obs.size() > _points[b].obs.size();
                          });
        active.resize(_cfg.local_ba_max_points);
    }
    std::vector<Eigen::Vector3d> Pw(active.size());
    for (size_t a = 0; a < active.size(); ++a) Pw[a] = _points[active[a]].pw;

    using M66 = Eigen::Matrix<double, 6, 6>;
    using V6 = Eigen::Matrix<double, 6, 1>;
    using M36 = Eigen::Matrix<double, 3, 6>;
    const double hd = _cfg.huber_delta_px;

    // Robust (Huber) reprojection cost over all active observations — used as
    // the LM accept/reject criterion so a badly-conditioned camera can never be
    // dragged metres by a single Gauss-Newton step.
    auto cost = [&](const std::vector<Eigen::Matrix4d>& T,
                    const std::vector<Eigen::Vector3d>& P) {
        double c = 0.0;
        for (size_t a = 0; a < active.size(); ++a) {
            const VoMapPoint& mp = _points[active[a]];
            const double s2 = _level_sigma2[std::clamp(mp.octave, 0, _cfg.n_levels - 1)];
            for (const VoObs& o : mp.obs) {
                auto it = kf_pos.find(o.kf_id);
                if (it == kf_pos.end()) continue;
                const Eigen::Vector3d pc =
                    T[it->second].block<3, 3>(0, 0) * P[a] + T[it->second].block<3, 1>(0, 3);
                if (pc.z() <= 1e-6) continue;
                const double u = _cfg.fx * pc.x() / pc.z() + _cfg.cx;
                const double v = _cfg.fy * pc.y() / pc.z() + _cfg.cy;
                const VoKeyframe* kf = _window[it->second].get();
                const double du = u - kf->kpts(o.kp_idx, 0), dv = v - kf->kpts(o.kp_idx, 1);
                const double rn = std::sqrt((du * du + dv * dv) / s2);
                c += rn <= hd ? 0.5 * rn * rn : hd * (rn - 0.5 * hd);
            }
        }
        return c;
    };

    struct PB { Eigen::Matrix3d Hpp; Eigen::Vector3d bp;
                std::vector<std::pair<int, M36>> Hpc; };
    double lambda = 1e-3;
    double prev_cost = cost(Tcw, Pw);

    for (int iter = 0; iter < _cfg.local_ba_iters; ++iter) {
        // Linearize once at the current estimate.
        std::vector<M66> Hcc(NV, M66::Zero());
        std::vector<V6> bc(NV, V6::Zero());
        std::vector<PB> pbs(active.size());
        for (size_t a = 0; a < active.size(); ++a) {
            const VoMapPoint& mp = _points[active[a]];
            const Eigen::Vector3d& pw = Pw[a];
            const double s2 = _level_sigma2[std::clamp(mp.octave, 0, _cfg.n_levels - 1)];
            const double w_info = 1.0 / s2;
            Eigen::Matrix3d Hpp = Eigen::Matrix3d::Zero();
            Eigen::Vector3d bp = Eigen::Vector3d::Zero();
            std::vector<std::pair<int, M36>> Hpc;
            for (const VoObs& o : mp.obs) {
                auto it = kf_pos.find(o.kf_id);
                if (it == kf_pos.end()) continue;
                const int pos = it->second;
                const Eigen::Matrix3d Rc = Tcw[pos].block<3, 3>(0, 0);
                const Eigen::Vector3d tc = Tcw[pos].block<3, 1>(0, 3);
                const Eigen::Vector3d pc = Rc * pw + tc;
                if (pc.z() <= 1e-6) continue;
                const double iz = 1.0 / pc.z();
                const double u = _cfg.fx * pc.x() * iz + _cfg.cx;
                const double v = _cfg.fy * pc.y() * iz + _cfg.cy;
                const VoKeyframe* kf = _window[pos].get();
                Eigen::Vector2d r(u - kf->kpts(o.kp_idx, 0), v - kf->kpts(o.kp_idx, 1));
                const double rn = r.norm() / std::sqrt(s2);
                const double w_h = rn > hd ? hd / rn : 1.0;
                const double W = w_h * w_info;
                Eigen::Matrix<double, 2, 3> Jp;
                Jp << _cfg.fx * iz, 0, -_cfg.fx * pc.x() * iz * iz,
                      0, _cfg.fy * iz, -_cfg.fy * pc.y() * iz * iz;
                const Eigen::Matrix<double, 2, 3> Jpoint = Jp * Rc;
                Hpp += W * Jpoint.transpose() * Jpoint;
                bp += W * Jpoint.transpose() * r;
                const int cv = cam_var(pos);
                if (cv >= 0) {
                    Eigen::Matrix<double, 3, 6> Jx;
                    Jx << 0, pc.z(), -pc.y(), 1, 0, 0,
                          -pc.z(), 0, pc.x(), 0, 1, 0,
                          pc.y(), -pc.x(), 0, 0, 0, 1;
                    const Eigen::Matrix<double, 2, 6> Jcam = Jp * Jx;
                    Hcc[cv] += W * Jcam.transpose() * Jcam;
                    bc[cv] += W * Jcam.transpose() * r;
                    Hpc.emplace_back(cv, (W * Jpoint.transpose() * Jcam).eval());
                }
            }
            pbs[a].Hpp = Hpp;
            pbs[a].bp = bp;
            pbs[a].Hpc = std::move(Hpc);
        }

        // LM trust-region: damp the diagonal, solve, accept only if cost drops.
        bool accepted = false;
        for (int trial = 0; trial < 6 && !accepted; ++trial) {
            Eigen::MatrixXd S = Eigen::MatrixXd::Zero(6 * NV, 6 * NV);
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(6 * NV);
            std::vector<Eigen::Matrix3d> Hpp_inv(active.size());
            for (size_t a = 0; a < active.size(); ++a) {
                Eigen::Matrix3d Hd = pbs[a].Hpp;
                Hd.diagonal() *= (1.0 + lambda);
                Hd += 1e-9 * Eigen::Matrix3d::Identity();
                Hpp_inv[a] = Hd.inverse();
                const Eigen::Vector3d& bp = pbs[a].bp;
                for (size_t x = 0; x < pbs[a].Hpc.size(); ++x) {
                    const int ci = pbs[a].Hpc[x].first;
                    rhs.segment<6>(6 * ci) -= pbs[a].Hpc[x].second.transpose() * Hpp_inv[a] * bp;
                    for (size_t y = 0; y < pbs[a].Hpc.size(); ++y)
                        S.block<6, 6>(6 * ci, 6 * pbs[a].Hpc[y].first) -=
                            pbs[a].Hpc[x].second.transpose() * Hpp_inv[a] * pbs[a].Hpc[y].second;
                }
            }
            for (int c = 0; c < NV; ++c) {
                M66 Hd = Hcc[c];
                Hd.diagonal() *= (1.0 + lambda);
                S.block<6, 6>(6 * c, 6 * c) += Hd + 1e-9 * M66::Identity();
                rhs.segment<6>(6 * c) += bc[c];
            }
            const Eigen::VectorXd dxc = S.ldlt().solve(-rhs);
            if (!dxc.allFinite()) { lambda = std::min(lambda * 8.0, 1e8); continue; }
            std::vector<Eigen::Matrix4d> Tt = Tcw;
            std::vector<Eigen::Vector3d> Pt = Pw;
            for (int c = 0; c < NV; ++c)
                Tt[c + 1] = exp_se3(dxc.segment<6>(6 * c)) * Tcw[c + 1];
            for (size_t a = 0; a < active.size(); ++a) {
                Eigen::Vector3d acc = pbs[a].bp;
                for (const auto& [ci, Hi] : pbs[a].Hpc) acc += Hi * dxc.segment<6>(6 * ci);
                Pt[a] -= Hpp_inv[a] * acc;
            }
            const double new_cost = cost(Tt, Pt);
            if (new_cost < prev_cost) {
                const double rel_impr = (prev_cost - new_cost) / std::max(prev_cost, 1e-9);
                Tcw.swap(Tt); Pw.swap(Pt);
                prev_cost = new_cost;
                lambda = std::max(lambda * 0.3, 1e-7);
                accepted = true;
                if (rel_impr < 1e-3) { iter = _cfg.local_ba_iters; }  // converged
            } else {
                lambda = std::min(lambda * 8.0, 1e8);
            }
        }
        if (!accepted) break;   // converged or stuck
    }

    // Write back refined poses (oldest fixed) and point positions.
    static const bool dbg = std::getenv("FUSION_BA_DEBUG") != nullptr;
    if (dbg) {
        double maxd = 0.0;
        for (int c = 1; c < C; ++c)
            maxd = std::max(maxd,
                (Tcw[c].inverse().block<3, 1>(0, 3) - _window[c]->Twc.block<3, 1>(0, 3)).norm());
        std::fprintf(stderr, "[BA] C=%d active=%zu max_cam_dt=%.4f m\n",
                     C, active.size(), maxd);
    }
    for (int c = 1; c < C; ++c) _window[c]->Twc = Tcw[c].inverse();
    for (size_t a = 0; a < active.size(); ++a) _points[active[a]].pw = Pw[a];
    rebuild_gathered();
}

VoResult VoFrontend::track(const cv::Mat& gray, const cv::Mat& depth, double stamp,
                           const Eigen::Matrix4d& prior_Twc, bool has_prior) {
    const auto t0 = std::chrono::steady_clock::now();
    VoResult res;
    std::vector<cv::KeyPoint> kps;
    cv::Mat des;
    extract(gray, kps, des);
    ++_frames_since_kf;
    const std::vector<std::pair<int, int>> no_matches;
    const std::vector<bool> no_mask;

    auto finish = [&](VoResult& r) {
        r.track_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0).count();
        return r;
    };
    // Record prev-emitted pose info on a new keyframe (refined, same BA epoch
    // when the previous KF is still in the window; cached across reinit).
    auto set_prev = [&](VoResult& r) {
        if (static_cast<int>(_window.size()) >= 2) {
            r.prev_Twc = _window[_window.size() - 2]->Twc;
            r.has_prev = true;
        } else if (_have_prev_emit) {
            r.prev_Twc = _prev_emit_Twc;
            r.has_prev = true;
        } else {
            r.has_prev = false;
        }
        _prev_emit_Twc = _window.back()->Twc;
        _have_prev_emit = true;
    };

    // bootstrap / dead-window: seed a keyframe at the prior (or identity)
    if (_window.empty()) {
        _points.clear();
        const Eigen::Matrix4d Twc = has_prior ? prior_Twc : Eigen::Matrix4d::Identity();
        auto kf = make_keyframe(stamp, Twc, kps, des, depth, no_matches, no_mask);
        _last_Twc = Twc;
        _has_pose = true;
        _frames_since_kf = 0;
        _ref_matched = static_cast<int>(_points.size());
        res.Twc = kf->Twc;
        res.state = VoState::INIT;
        res.new_keyframe = true;
        res.kf_id = kf->id;
        set_prev(res);
        return finish(res);
    }

    Eigen::Matrix4d Twc_pred = has_prior ? prior_Twc : _last_Twc * _velocity;

    std::vector<std::pair<int, int>> matches;
    int nm = match_window(kps, des, Twc_pred, _cfg.match_radius_px, matches);
    if (nm < _cfg.min_inliers_reinit)
        nm = match_window(kps, des, Twc_pred, _cfg.match_radius_fallback_px, matches);

    Eigen::Matrix4d Tcw = Twc_pred.inverse();
    std::vector<bool> mask;
    int inliers = nm >= 6 ? optimize_pose(kps, matches, Tcw, mask) : 0;

    if (inliers < _cfg.min_inliers_reinit) {
        // Projection matching failed — usually a wrong motion model during a
        // fast turn, NOT missing features (probe: median 912 kps on failure
        // frames). Recover pose-free: strict brute-force descriptor matching +
        // PnP RANSAC needs no prediction at all (the reference ORB-SLAM2
        // survives the same turns via its equivalent track_reference_keyframe).
        std::vector<std::pair<int, int>> bf;
        if (match_bruteforce(kps, des, bf) >= _cfg.min_inliers_reinit) {
            Eigen::Matrix4d Tcw_r;
            if (recover_pose(kps, bf, Tcw_r)) {
                std::vector<bool> mask_r;
                const int in_r = optimize_pose(kps, bf, Tcw_r, mask_r);
                if (in_r >= _cfg.min_inliers_reinit) {
                    Tcw = Tcw_r;
                    matches = std::move(bf);
                    mask = std::move(mask_r);
                    nm = static_cast<int>(matches.size());
                    inliers = in_r;
                }
            }
        }
    }
    res.n_matches = nm;
    res.n_inliers = inliers;

    if (inliers < _cfg.min_inliers_reinit) {
        ++_consec_fail;
        if (_consec_fail < _cfg.reinit_patience) {
            // brief failure: coast on the prediction, keep the window alive so
            // tracking can recover next frame without nuking the map. No
            // keyframe, not counted as a reinit.
            _last_Twc = Twc_pred;
            res.Twc = Twc_pred;
            res.state = VoState::WEAK;
            return finish(res);
        }
        // persistent failure: re-seed the window from depth at the prior pose
        // (the proven depth re-init safety net). Stale points from before the
        // dropout poison projection matching (they tested worse), so the window
        // and local map are reset; the caller's prior keeps the path continuous.
        _consec_fail = 0;
        _window.clear();
        _points.clear();
        auto kf = make_keyframe(stamp, Twc_pred, kps, des, depth, no_matches, no_mask);
        _last_Twc = Twc_pred;
        // velocity intentionally kept: carry-through keeps turning through
        // featureless stretches instead of freezing the motion model.
        _frames_since_kf = 0;
        _ref_matched = static_cast<int>(_points.size());
        res.Twc = kf->Twc;
        res.state = VoState::REINIT;
        res.new_keyframe = true;
        res.kf_id = kf->id;
        set_prev(res);
    } else {
        _consec_fail = 0;
        const Eigen::Matrix4d Twc = Tcw.inverse();
        _velocity = _last_Twc.inverse() * Twc;   // per-frame velocity (unrefined)
        _last_Twc = Twc;
        res.Twc = Twc;
        res.state = inliers >= _cfg.min_inliers_ok ? VoState::OK : VoState::WEAK;

        const bool starving = inliers < _cfg.kf_min_close_points;
        const bool weak_vs_ref = inliers < _cfg.kf_ref_ratio * _ref_matched;
        if ((_frames_since_kf >= _cfg.kf_min_frames && (weak_vs_ref || starving)) ||
            _frames_since_kf >= _cfg.kf_max_frames) {
            auto kf = make_keyframe(stamp, Twc, kps, des, depth, matches, mask);
            if (_cfg.enable_local_ba) local_ba();
            _frames_since_kf = 0;
            _ref_matched = std::max(inliers, 1);
            // adopt the BA-refined pose for the new keyframe so the absolute
            // frame stays consistent with the refined window points.
            res.Twc = kf->Twc;
            _last_Twc = kf->Twc;
            res.new_keyframe = true;
            res.kf_id = kf->id;
            set_prev(res);
        }
    }
    return finish(res);
}

const VoKeyframe* VoFrontend::keyframe(int id) const {
    for (const auto& kf : _window)
        if (kf->id == id) return kf.get();
    return nullptr;
}

size_t VoFrontend::window_bytes() const {
    size_t b = 0;
    for (const auto& kf : _window)
        b += kf->kpts.size() * 4 + kf->des.total() + kf->pts3d_cam.size() * 4 +
             kf->octave.size() * 4;
    for (const auto& mp : _points)
        b += sizeof(VoMapPoint) + mp.obs.size() * sizeof(VoObs);
    return b;
}

}  // namespace fusion
