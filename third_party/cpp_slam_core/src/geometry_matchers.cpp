#include "geometry_matchers.h"

#include "frame.h"
#include "keyframe.h"
#include "map_point.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cppcore {

using slam::Frame;
using slam::KeyFrame;
using slam::MapPoint;

// Faithful port of visual_slam/orbslam/slam/geometry_matchers.py
// _search_map_by_projection (the post-vectorization version). Semantics matched
// 1:1 against the Python matcher and validated bit-for-bit by the per-point
// parity harness (predicted level, projection, radius, kd candidates, occupancy,
// octave window, descriptor distance, best/second-best, ratio test all agree):
//   visibility prep (project + in-image + depth + viewing-cos + min/max dist)
//   -> per-point kd radius query -> octave/occupancy filter (LIVE occupancy)
//   -> streaming best/second-best + ratio test -> add_frame_view.
// Side effects match pySLAM: increase_visible() per visible point and
// add_frame_view() per match; increase_found() is intentionally NOT called here.
std::pair<int, std::vector<int>> search_map_by_projection(
    const py::list &points, py::object frame_obj,
    py::array_t<float, py::array::c_style | py::array::forcecast> scale_factors,
    const MatchParams &p) {

    Frame &f = frame_obj.cast<Frame &>();
    const int frame_id = f.id;
    const Eigen::Vector3d Ow = f.Ow();

    const float *sf = scale_factors.data();
    const int n_levels = static_cast<int>(scale_factors.size());

    const int n_feat = static_cast<int>(f.octaves.size());

    int found_count = 0;
    std::vector<int> found_fidxs;

    for (auto handle : points) {
        py::object obj = py::reinterpret_borrow<py::object>(handle);
        if (obj.is_none()) continue;
        MapPoint *mp = nullptr;
        try { mp = obj.cast<MapPoint *>(); } catch (...) { continue; }
        if (mp == nullptr || mp->is_bad()) continue;
        if (mp->last_frame_id_seen == frame_id) continue;  // already seen this frame

        // ---- visibility prep ----
        const Eigen::Vector3d pw = mp->get_position();
        if (!pw.allFinite() || pw.norm() > 1.0e9) continue;

        const Eigen::Vector3d uvz = f.project_world(pw);  // (u, v, depth)
        const double u = uvz(0), v = uvz(1), z = uvz(2);
        if (!(z > p.min_depth)) continue;
        if (!f.is_in_image(u, v, z)) continue;
        if (z > p.far_points_threshold) continue;

        const Eigen::Vector3d PO = pw - Ow;
        const double dist = PO.norm();

        // Distance window — matches Python _prepare_visible_projection_candidates
        // (via MapPoint.get_all_pos_info): keep iff 0.8*min < dist < 1.2*max.
        // Our min_distance()/max_distance() return raw values; tolerance here.
        const double dmin = 0.8 * static_cast<double>(mp->min_distance());
        const double dmax = 1.2 * static_cast<double>(mp->max_distance());
        if (dist <= dmin || dist >= dmax) continue;

        // Viewing-cos — apply iff the normal is valid (matches Python: keep iff
        // cos_view > kViewingCosLimitForPoint; points with no normal are kept).
        const Eigen::Vector3d &normal = mp->normal;
        if (normal.allFinite() && normal.norm() > 1e-12) {
            const double cos_view = normal.dot(PO) / std::max(dist, 1e-12);
            if (!(cos_view > p.viewing_cos_limit)) continue;
        }

        mp->increase_visible();

        // ---- predicted scale level (DOUBLE precision, matching Python
        // MapPoint.predict_scale exactly — not the float predict_detection_level
        // which truncates dist to float + hardcodes log(1.2f); that float/double
        // mismatch flips the ceil() level near boundaries -> divergent matches). ----
        int predicted_level = 0;
        {
            const double max_d = static_cast<double>(mp->max_distance());
            if (max_d > 0.0 && std::isfinite(max_d)) {
                const double ratio = max_d / std::max(dist, 1e-12);
                int n_scale = static_cast<int>(std::ceil(std::log(ratio) / p.log_scale_factor));
                if (n_scale < 0) n_scale = 0;
                else if (n_scale >= p.num_levels) n_scale = p.num_levels - 1;
                predicted_level = n_scale;
            }
        }

        // ---- kd radius query at the predicted scale level ----
        float kp_scale = (predicted_level >= 0 && predicted_level < n_levels) ? sf[predicted_level] : 1.0f;
        const float radius = p.max_reproj_distance * kp_scale;

        const std::vector<int> cand = f.kd_query_ball(static_cast<float>(u), static_cast<float>(v), radius);
        if (cand.empty()) continue;

        // ---- streaming best / second-best over valid candidates ----
        float best_dist = std::numeric_limits<float>::infinity();
        float best_dist2 = std::numeric_limits<float>::infinity();
        int best_level = -1, best_level2 = -1, best_k_idx = -1;

        for (int idx : cand) {
            if (idx < 0 || idx >= n_feat) continue;
            // occupancy (LIVE, matching pySLAM: f_cur->points[idx] is read at
            // evaluation time, so a feature claimed by an earlier map point in
            // THIS call is excluded for later ones — first-come wins).
            if (idx < static_cast<int>(f.points.size())) {
                const py::object &pc = f.points[idx];
                if (!pc.is_none()) {
                    MapPoint *mpc = nullptr;
                    try { mpc = pc.cast<MapPoint *>(); } catch (...) { mpc = nullptr; }
                    if (mpc != nullptr && mpc->num_observations() > 0) continue;
                }
            }
            const int kp_level = f.octaves(idx);
            if (kp_level < predicted_level - 1 || kp_level > predicted_level) continue;

            const float d = mp->min_des_distance(f.des.row(idx));
            if (d < best_dist) {
                best_dist2 = best_dist; best_level2 = best_level;
                best_dist = d; best_level = kp_level; best_k_idx = idx;
            } else if (d < best_dist2) {
                best_dist2 = d; best_level2 = kp_level;
            }
        }

        if (best_k_idx > -1 && best_dist < p.max_descriptor_distance) {
            // ratio test: reject ambiguous matches at the same octave level
            if (best_level == best_level2 && best_dist > best_dist2 * p.ratio_test) continue;
            // NOTE: pySLAM's search_map_by_projection does NOT call increase_found()
            // here (found-ratio is maintained elsewhere); we match that.
            if (mp->add_frame_view(frame_obj, best_k_idx)) {
                ++found_count;
                found_fidxs.push_back(best_k_idx);
            }
        }
    }

    return {found_count, found_fidxs};
}

// Faithful port of _search_frame_by_projection (see header). The visibility prep
// (project + in-image + depth + 0.8*min/1.2*max distance window + viewing-cos) is
// identical to search_map (both Python functions share
// _prepare_visible_projection_candidates). The differences are: REF-octave radius
// + octave window, argmin + threshold with NO ratio test, an optional stereo
// right-u check, and that add_frame_view + the rotation-histogram filter are left
// to the Python dispatcher (so this returns the pre-filter (idxs_ref, idxs_cur)).
std::pair<std::vector<int>, std::vector<int>> search_frame_by_projection(
    const py::list &ref_points, const std::vector<int> &ref_idxs,
    const std::vector<int> &ref_octaves, py::object f_cur_obj,
    py::array_t<float, py::array::c_style | py::array::forcecast> scale_factors,
    float max_reproj_distance, float max_descriptor_distance,
    float viewing_cos_limit, float min_depth, bool do_stereo_check) {

    Frame &f = f_cur_obj.cast<Frame &>();
    const Eigen::Vector3d Ow = f.Ow();

    const float *sf = scale_factors.data();
    const int n_levels = static_cast<int>(scale_factors.size());
    const int n_feat = static_cast<int>(f.octaves.size());
    const int n_kps_ur = static_cast<int>(f.kps_ur.size());

    std::vector<int> idxs_ref_out, idxs_cur_out;
    idxs_ref_out.reserve(ref_idxs.size());
    idxs_cur_out.reserve(ref_idxs.size());

    int pi = -1;
    for (auto handle : ref_points) {
        ++pi;  // index into ref_idxs / ref_octaves (parallel to ref_points)
        const int ref_idx = ref_idxs[pi];
        const int ref_octave = ref_octaves[pi];

        py::object obj = py::reinterpret_borrow<py::object>(handle);
        if (obj.is_none()) continue;
        MapPoint *mp = nullptr;
        try { mp = obj.cast<MapPoint *>(); } catch (...) { continue; }
        if (mp == nullptr || mp->is_bad()) continue;

        // ---- visibility prep (matches _prepare_visible_projection_candidates) ----
        const Eigen::Vector3d pw = mp->get_position();
        if (!pw.allFinite() || pw.norm() > 1.0e9) continue;

        const Eigen::Vector3d uvz = f.project_world(pw);  // (u, v, depth)
        const double u = uvz(0), v = uvz(1), z = uvz(2);
        if (!(z > min_depth)) continue;
        if (!f.is_in_image(u, v, z)) continue;

        const Eigen::Vector3d PO = pw - Ow;
        const double dist = PO.norm();
        const double dmin = 0.8 * static_cast<double>(mp->min_distance());
        const double dmax = 1.2 * static_cast<double>(mp->max_distance());
        if (dist <= dmin || dist >= dmax) continue;

        const Eigen::Vector3d &normal = mp->normal;
        if (normal.allFinite() && normal.norm() > 1e-12) {
            const double cos_view = normal.dot(PO) / std::max(dist, 1e-12);
            if (!(cos_view > viewing_cos_limit)) continue;
        }

        // ---- kd radius query at the REF octave scale ----
        const float kp_ref_scale =
            (ref_octave >= 0 && ref_octave < n_levels) ? sf[ref_octave] : 1.0f;
        const float radius = max_reproj_distance * kp_ref_scale;
        const std::vector<int> cand =
            f.kd_query_ball(static_cast<float>(u), static_cast<float>(v), radius);
        if (cand.empty()) continue;

        // projected right-u coordinate (only needed for the stereo check)
        const double proj_ur = do_stereo_check ? f.stereo_ur(u, z) : 0.0;

        // ---- argmin descriptor over valid candidates (NO ratio test) ----
        float best_dist = std::numeric_limits<float>::infinity();
        int best_k_idx = -1;

        for (int idx : cand) {
            if (idx < 0 || idx >= n_feat) continue;
            // occupancy: a candidate whose map point already has a keyframe
            // observation (num_observations() > 0) is unavailable. Matches the
            // Python precompute (num_observations is unchanged by add_frame_view).
            if (idx < static_cast<int>(f.points.size())) {
                const py::object &pc = f.points[idx];
                if (!pc.is_none()) {
                    MapPoint *mpc = nullptr;
                    try { mpc = pc.cast<MapPoint *>(); } catch (...) { mpc = nullptr; }
                    if (mpc != nullptr && mpc->num_observations() > 0) continue;
                }
            }
            const int kp_level = f.octaves(idx);
            // octave window: within +/-1 of the REF octave
            if (kp_level < ref_octave - 1 || kp_level > ref_octave + 1) continue;
            // stereo right-u consistency: reject iff measured ur >= 0 and the
            // reprojection error exceeds max_reproj_distance * scale[cur_octave].
            if (do_stereo_check && idx < n_kps_ur) {
                const float kp_ur = f.kps_ur(idx);
                if (kp_ur >= 0.0f) {
                    const double err_ur = std::fabs(proj_ur - static_cast<double>(kp_ur));
                    const float cur_scale =
                        (kp_level >= 0 && kp_level < n_levels) ? sf[kp_level] : 1.0f;
                    if (err_ur >= static_cast<double>(max_reproj_distance * cur_scale)) continue;
                }
            }

            const float d = mp->min_des_distance(f.des.row(idx));
            if (d < best_dist) {
                best_dist = d;
                best_k_idx = idx;
            }
        }

        if (best_k_idx >= 0 && best_dist < max_descriptor_distance) {
            idxs_ref_out.push_back(ref_idx);
            idxs_cur_out.push_back(best_k_idx);
        }
    }

    return {idxs_ref_out, idxs_cur_out};
}

int search_and_fuse(
    const py::list &points, py::object keyframe_obj,
    py::array_t<float, py::array::c_style | py::array::forcecast> scale_factors,
    py::array_t<float, py::array::c_style | py::array::forcecast> inv_level_sigmas2,
    float max_reproj_distance, float max_descriptor_distance,
    double log_scale_factor, int num_levels, float min_depth, float chi2_mono) {

    KeyFrame *kf = nullptr;
    try { kf = keyframe_obj.cast<KeyFrame *>(); } catch (...) { return 0; }
    if (kf == nullptr || kf->is_bad()) return 0;

    const float descriptor_gate = 0.5f * max_descriptor_distance;
    if (!(descriptor_gate > 0.0f) || points.size() == 0) return 0;

    struct Candidate {
        py::object obj;
        MapPoint *mp;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(points.size());
    for (auto handle : points) {
        py::object obj = py::reinterpret_borrow<py::object>(handle);
        if (obj.is_none()) continue;
        MapPoint *mp = nullptr;
        try { mp = obj.cast<MapPoint *>(); } catch (...) { continue; }
        if (mp == nullptr || mp->is_bad()) continue;
        if (mp->is_in_keyframe(keyframe_obj)) continue;
        candidates.push_back({obj, mp});
    }
    if (candidates.empty()) return 0;

    // Cache Python camera intrinsics while the GIL is still held; projection is
    // pure C++ after this point.
    (void)kf->is_in_image(0.0, 0.0, 1.0);

    struct FuseMatch {
        int candidate_idx;
        int keypoint_idx;
    };
    std::vector<FuseMatch> matches;
    matches.reserve(candidates.size());

    const float *sf = scale_factors.data();
    const int n_scale = static_cast<int>(scale_factors.size());
    const float *inv_sigmas = inv_level_sigmas2.data();
    const int n_sigmas = static_cast<int>(inv_level_sigmas2.size());
    const int n_feat = static_cast<int>(kf->octaves.size());
    const Eigen::Vector3d Ow = kf->Ow();

    {
        py::gil_scoped_release release;

        for (int ci = 0; ci < static_cast<int>(candidates.size()); ++ci) {
            MapPoint *mp = candidates[ci].mp;
            if (mp == nullptr) continue;

            const Eigen::Vector3d pw = mp->get_position();
            if (!pw.allFinite() || pw.norm() > 1.0e9) continue;

            const Eigen::Vector3d uvz = kf->project_world(pw);
            const double u = uvz(0), v = uvz(1), z = uvz(2);
            if (!(z > min_depth)) continue;
            if (!kf->is_in_image(u, v, z)) continue;

            const Eigen::Vector3d PO = pw - Ow;
            const double dist = PO.norm();

            int predicted_level = 0;
            const double max_d = static_cast<double>(mp->max_distance());
            if (max_d > 0.0 && std::isfinite(max_d) && log_scale_factor > 0.0) {
                const double ratio = max_d / std::max(dist, 1e-12);
                int level = static_cast<int>(std::ceil(std::log(ratio) / log_scale_factor));
                if (level < 0) level = 0;
                else if (level >= num_levels) level = num_levels - 1;
                predicted_level = level;
            }

            const float kp_scale =
                (predicted_level >= 0 && predicted_level < n_scale) ? sf[predicted_level] : 1.0f;
            const float radius = max_reproj_distance * kp_scale;
            const std::vector<int> kd_idxs =
                kf->kd_query_ball(static_cast<float>(u), static_cast<float>(v), radius);
            if (kd_idxs.empty()) continue;

            float best_dist = std::numeric_limits<float>::infinity();
            int best_kd_idx = -1;

            for (int kd_idx : kd_idxs) {
                if (kd_idx < 0 || kd_idx >= n_feat) continue;
                const int kp_level = kf->octaves(kd_idx);
                if (kp_level < predicted_level - 1 || kp_level > predicted_level) continue;

                const float inv_sigma2 =
                    (kp_level >= 0 && kp_level < n_sigmas) ? inv_sigmas[kp_level] : 1.0f;
                const double dx = u - static_cast<double>(kf->kpsu(kd_idx, 0));
                const double dy = v - static_cast<double>(kf->kpsu(kd_idx, 1));
                const double chi2 = (dx * dx + dy * dy) * static_cast<double>(inv_sigma2);
                if (chi2 > static_cast<double>(chi2_mono)) continue;

                const float descriptor_dist = mp->min_des_distance(kf->des.row(kd_idx));
                if (descriptor_dist < best_dist) {
                    best_dist = descriptor_dist;
                    best_kd_idx = kd_idx;
                }
            }

            if (best_kd_idx > -1 && best_dist < descriptor_gate) {
                matches.push_back({ci, best_kd_idx});
            }
        }
    }

    int fused_pts_count = 0;
    for (const auto &m : matches) {
        Candidate &cand = candidates[m.candidate_idx];
        MapPoint *point = cand.mp;
        if (point == nullptr) continue;

        py::object existing = kf->get_point_match(m.keypoint_idx);
        if (!existing.is_none()) {
            MapPoint *existing_mp = nullptr;
            try { existing_mp = existing.cast<MapPoint *>(); } catch (...) { existing_mp = nullptr; }
            if (existing_mp != nullptr) {
                if (existing_mp->num_observations() > point->num_observations()) {
                    point->replace_with(existing_mp->shared_from_this());
                } else {
                    existing_mp->replace_with(point->shared_from_this());
                    point->add_observation(keyframe_obj, m.keypoint_idx);
                }
            }
        } else {
            point->add_observation(keyframe_obj, m.keypoint_idx);
        }

        point->update_info();
        ++fused_pts_count;
    }

    return fused_pts_count;
}

// ---- F2: expanding tracking local-map build --------------------------------
// Port of tracking._build_local_keyframes_from_votes +
// _collect_local_points_from_keyframes. Parity notes vs OUR Python:
//  - KFs deduped by KID (C++ KeyFrame __hash__/__eq__ are kid-based; the Python
//    keyframe_counts dict keys on kf whose __hash__==kid).
//  - TRANSITIVE expansion: the index loop reads the GROWING vector, mirroring
//    Python `for kf in local_keyframes_list` (which sees appends), capped at
//    max_kfs via the size check at the loop top.
//  - sort by vote count desc, STABLE (ties keep insertion order, == Python sorted).
//  - local_points deduped per-call by MapPoint identity == the Python
//    last_track_reference_frame_id==frame_id check (read nowhere else), so the
//    per-point attr write is intentionally omitted. The diagnostics side-effects
//    (last_tracking_frame_id / tracking_vote_count) are also omitted (profiling-only).
std::pair<py::list, py::list> build_local_map(
    py::object f_cur, int num_best, int max_kfs, int frame_id) {
    (void)frame_id;  // dedup is per-call (see note above)

    // NATIVE C++ object access (F2 3/n): the KFs/MapPoints are Python subclasses of
    // the C++ base, so cast py::object -> slam::KeyFrame*/MapPoint* and call the C++
    // methods DIRECTLY (no obj.attr() -> _w trampoline -> C++ round-trip). Only f_cur
    // is a Python Frame, so its single entry call goes via attr().
    auto as_kf = [](const py::object &o) -> slam::KeyFrame * {
        if (o.is_none()) return nullptr;
        try { return o.cast<slam::KeyFrame *>(); } catch (...) { return nullptr; }
    };
    auto as_mp = [](const py::object &o) -> slam::MapPoint * {
        if (o.is_none()) return nullptr;
        try { return o.cast<slam::MapPoint *>(); } catch (...) { return nullptr; }
    };

    // 1. Votes: current frame's matched good points -> observing KFs (dedup by kid).
    std::unordered_map<int, int> vote_count;
    std::vector<py::object> vote_kf;  // first-seen kf py::object per kid, in order
    {
        py::object cur_pts = f_cur.attr("get_matched_good_points")();  // Python Frame entry
        for (auto item : cur_pts) {
            py::object pobj = py::reinterpret_borrow<py::object>(item);
            slam::MapPoint *mp = as_mp(pobj);
            if (!mp || mp->is_bad()) continue;
            for (auto &pr : mp->observations()) {          // native
                slam::KeyFrame *kf = as_kf(pr.first);
                if (!kf || kf->is_bad()) continue;
                int kid = kf->kid;
                auto it = vote_count.find(kid);
                if (it == vote_count.end()) { vote_count[kid] = 1; vote_kf.push_back(pr.first); }
                else { it->second++; }
            }
        }
    }

    // 2. Expanding local keyframes (transitive, capped at max_kfs).
    std::vector<py::object> local_obj;
    std::vector<slam::KeyFrame *> local_ptr;
    std::vector<int> local_counts;
    std::unordered_set<int> in_local;
    for (auto &kfobj : vote_kf) {
        slam::KeyFrame *kf = as_kf(kfobj);
        local_obj.push_back(kfobj);
        local_ptr.push_back(kf);
        local_counts.push_back(vote_count[kf->kid]);
        in_local.insert(kf->kid);
    }
    auto try_add = [&](const py::object &kfobj) -> bool {
        slam::KeyFrame *kf = as_kf(kfobj);
        if (!kf || kf->is_bad() || in_local.count(kf->kid)) return false;
        local_obj.push_back(kfobj);
        local_ptr.push_back(kf);
        local_counts.push_back(1);
        in_local.insert(kf->kid);
        return true;
    };
    for (std::size_t i = 0; i < local_obj.size(); ++i) {
        if (static_cast<int>(local_obj.size()) >= max_kfs) break;
        slam::KeyFrame *kf = local_ptr[i];
        if (!kf) continue;
        for (auto &n : kf->get_best_covisible_keyframes(num_best))  // native
            if (try_add(n)) break;
        for (auto &c : kf->get_children())                          // native
            if (try_add(c)) break;
        try_add(kf->get_parent());                                  // native
    }

    // sort by count desc (stable), keep top max_kfs
    std::vector<std::size_t> order(local_obj.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return local_counts[a] > local_counts[b]; });
    int n_keep = std::min<int>(max_kfs, static_cast<int>(order.size()));

    py::list local_keyframes_out;
    std::vector<slam::KeyFrame *> kept_ptr;
    kept_ptr.reserve(n_keep);
    for (int i = 0; i < n_keep; ++i) {
        local_keyframes_out.append(local_obj[order[i]]);
        kept_ptr.push_back(local_ptr[order[i]]);
    }

    // 3. Collect local points (per-call dedup by MapPoint identity). get_points() is
    // the C++ base method (full list incl None); the is_good filter makes it == the
    // Python _collect's get_matched_points.
    py::list local_points_out;
    std::unordered_set<slam::MapPoint *> seen_pts;
    for (auto *kf : kept_ptr) {
        if (!kf) continue;
        for (auto &pobj : kf->get_points()) {                       // native
            slam::MapPoint *mp = as_mp(pobj);
            if (!mp || mp->is_bad()) continue;
            if (mp->get_replacement()) continue;                    // replacement set -> skip
            if (!seen_pts.insert(mp).second) continue;
            local_points_out.append(pobj);
        }
    }

    return {local_keyframes_out, local_points_out};
}

}  // namespace cppcore
