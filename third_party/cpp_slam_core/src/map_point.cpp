#include "map_point.h"

#include <algorithm>
#include <sstream>

namespace slam {

// ---- Static members --------------------------------------------------------
std::atomic<int> MapPoint::_next_id{0};
std::mutex MapPoint::_id_lock;

int MapPoint::next_id() {
    std::lock_guard<std::mutex> lk(_id_lock);
    return _next_id.load();
}

void MapPoint::set_id(int id) {
    _next_id.store(id);
}

// ---- Construction ----------------------------------------------------------
MapPoint::MapPoint(int given_id)
    : id(given_id < 0 ? _next_id.fetch_add(1) : given_id),
      _pos(Eigen::Vector3d::Zero()),
      normal(Eigen::Vector3d(0, 0, 1)),
      _min_distance(0.0f),
      _max_distance(std::numeric_limits<float>::infinity()),
      _num_observations(0),
      num_times_visible(1),
      num_times_found(1),
      last_frame_id_seen(-1),
      lba_count(0),
      corrected_by_kf(0),
      corrected_reference(0),
      pt_GBA(Eigen::Vector3d::Zero()),
      is_pt_GBA_valid(false),
      GBA_kf_id(0) {}

// ---- Position access -------------------------------------------------------
Eigen::Vector3d MapPoint::get_position() const {
    std::lock_guard<std::mutex> lk(_lock_pos);
    return _pos;
}

void MapPoint::update_position(const Eigen::Vector3d &pos) {
    std::lock_guard<std::mutex> lk(_lock_pos);
    _pos = pos;
}

float MapPoint::min_distance() const {
    std::lock_guard<std::mutex> lk(_lock_pos);
    return _min_distance;
}

float MapPoint::max_distance() const {
    std::lock_guard<std::mutex> lk(_lock_pos);
    return _max_distance;
}

// ---- Descriptor ------------------------------------------------------------
cv::Mat MapPoint::get_descriptor() const {
    std::lock_guard<std::mutex> lk(_lock_features);
    return des.clone();
}

float MapPoint::_hamming_distance(const cv::Mat &a, const cv::Mat &b) {
    return static_cast<float>(cv::norm(a, b, cv::NORM_HAMMING));
}

float MapPoint::min_des_distance(const cv::Mat &query_des) const {
    std::lock_guard<std::mutex> lk(_lock_features);
    if (des.empty()) return 256.0f;
    return _hamming_distance(des, query_des);
}

// ---- Observation management ------------------------------------------------
bool MapPoint::add_observation(py::object kf, int idx) {
    // GIL must be held by caller (py::object operations)
    std::lock_guard<std::mutex> lk(_lock_features);
    if (_observations.count(kf)) return false;

    _observations[kf] = idx;

    // Count stereo observations (kf.kps_ur[idx] >= 0) as weight 2
    int weight = 1;
    try {
        auto kps_ur = kf.attr("kps_ur");
        if (!kps_ur.is_none()) {
            auto ur_val = kps_ur[py::int_(idx)];
            if (!ur_val.is_none() && ur_val.cast<float>() >= 0.0f) weight = 2;
        }
    } catch (...) {}
    _num_observations += weight;

    // Register match in keyframe
    try {
        kf.attr("set_point_match")(shared_from_this(), idx);
    } catch (...) {}

    return true;
}

void MapPoint::remove_observation(py::object kf, int idx, bool map_no_lock) {
    // GIL must be held by caller
    bool do_remove_match = false;
    bool do_set_bad = false;
    int obs_idx = idx;

    {
        std::lock_guard<std::mutex> lk(_lock_features);
        auto it = _observations.find(kf);
        if (it == _observations.end()) return;

        obs_idx = (idx >= 0) ? idx : it->second;
        do_remove_match = (idx >= 0);

        _observations.erase(it);

        // Re-count weight for removed observation
        int weight = 1;
        try {
            auto kps_ur = kf.attr("kps_ur");
            if (!kps_ur.is_none()) {
                auto ur_val = kps_ur[py::int_(obs_idx)];
                if (!ur_val.is_none() && ur_val.cast<float>() >= 0.0f) weight = 2;
            }
        } catch (...) {}
        _num_observations = std::max(0, _num_observations - weight);

        do_set_bad = (_num_observations <= 2);

        // Update kf_ref to next available observation
        if (!kf_ref.is_none() && kf_ref.ptr() == kf.ptr() && !_observations.empty()) {
            kf_ref = _observations.begin()->first;
        }
    }

    if (do_remove_match && obs_idx >= 0) {
        try { kf.attr("remove_point_match")(obs_idx); } catch (...) {}
    } else if (!do_remove_match) {
        try { kf.attr("remove_point")(shared_from_this()); } catch (...) {}
    }

    if (do_set_bad) {
        set_bad(map_no_lock);
    }
}

std::vector<std::pair<py::object, int>> MapPoint::observations() const {
    std::lock_guard<std::mutex> lk(_lock_features);
    return std::vector<std::pair<py::object, int>>(
        _observations.begin(), _observations.end());
}

std::vector<py::object> MapPoint::keyframes() const {
    std::lock_guard<std::mutex> lk(_lock_features);
    std::vector<py::object> kfs;
    kfs.reserve(_observations.size());
    for (const auto &kv : _observations) kfs.push_back(kv.first);
    return kfs;
}

bool MapPoint::is_in_keyframe(py::object kf) const {
    std::lock_guard<std::mutex> lk(_lock_features);
    return _observations.count(kf) > 0;
}

int MapPoint::get_observation_idx(py::object kf) const {
    std::lock_guard<std::mutex> lk(_lock_features);
    auto it = _observations.find(kf);
    return (it != _observations.end()) ? it->second : -1;
}

int MapPoint::num_observations() const {
    return _num_observations;
}

// ---- Frame views -----------------------------------------------------------
bool MapPoint::add_frame_view(py::object frame, int idx) {
    std::lock_guard<std::mutex> lk(_lock_features);
    if (_frame_views.count(frame)) return false;
    _frame_views[frame] = idx;
    try { frame.attr("set_point_match")(shared_from_this(), idx); } catch (...) {}
    return true;
}

void MapPoint::remove_frame_view(py::object frame, int idx) {
    std::lock_guard<std::mutex> lk(_lock_features);
    auto it = _frame_views.find(frame);
    if (it == _frame_views.end()) return;
    int rm_idx = (idx >= 0) ? idx : it->second;
    _frame_views.erase(it);
    try { frame.attr("remove_point_match")(rm_idx); } catch (...) {}
}

bool MapPoint::is_in_frame(py::object frame) const {
    std::lock_guard<std::mutex> lk(_lock_features);
    return _frame_views.count(frame) > 0;
}

int MapPoint::get_frame_view_idx(py::object frame) const {
    std::lock_guard<std::mutex> lk(_lock_features);
    auto it = _frame_views.find(frame);
    return (it != _frame_views.end()) ? it->second : -1;
}

std::vector<std::pair<py::object, int>> MapPoint::frame_views() const {
    std::lock_guard<std::mutex> lk(_lock_features);
    return std::vector<std::pair<py::object, int>>(_frame_views.begin(), _frame_views.end());
}

std::vector<py::object> MapPoint::frames() const {
    std::lock_guard<std::mutex> lk(_lock_features);
    std::vector<py::object> result;
    result.reserve(_frame_views.size());
    for (const auto &kv : _frame_views) result.push_back(kv.first);
    return result;
}

bool MapPoint::is_bad_or_is_in_keyframe(py::object kf) const {
    std::lock_guard<std::mutex> lk(_lock_features);
    return _is_bad.load() || (_observations.count(kf) > 0);
}

bool MapPoint::is_good_with_min_obs(int min_obs) const {
    std::lock_guard<std::mutex> lk(_lock_features);
    return !_is_bad.load() && (_num_observations >= min_obs);
}

void MapPoint::set_des(const py::array_t<uint8_t> &arr) {
    if (arr.size() < 32) return;
    auto r = arr.unchecked<1>();
    std::lock_guard<std::mutex> lk(_lock_features);
    des = cv::Mat(1, 32, CV_8U);
    for (int i = 0; i < 32; ++i) des.at<uint8_t>(0, i) = r(i);
}

// ---- Status / replacement --------------------------------------------------
void MapPoint::set_bad(bool map_no_lock) {
    if (_is_bad.exchange(true)) return;  // already bad

    // Snapshot observations before clearing (GIL must be held)
    std::vector<std::pair<py::object, int>> obs_copy;
    {
        std::lock_guard<std::mutex> lk(_lock_features);
        obs_copy.assign(_observations.begin(), _observations.end());
        _observations.clear();
        _frame_views.clear();
        _num_observations = 0;
    }

    // Remove match from all observing KFs
    for (auto &[kf, idx] : obs_copy) {
        try { kf.attr("remove_point_match")(idx); } catch (...) {}
    }

    // Remove from map
    if (!map.is_none()) {
        try {
            if (map_no_lock) {
                map.attr("remove_map_point_no_lock_")(shared_from_this());
            } else {
                map.attr("remove_map_point")(shared_from_this());
            }
        } catch (...) {}
    }
}

void MapPoint::replace_with(MapPointPtr p) {
    if (!p || p.get() == this || _is_bad.load()) return;

    // Snapshot observations and stats
    std::vector<std::pair<py::object, int>> obs_copy;
    int snap_found, snap_visible;
    {
        std::lock_guard<std::mutex> lk(_lock_features);
        std::lock_guard<std::mutex> lk2(_lock_replacement);
        obs_copy.assign(_observations.begin(), _observations.end());
        _observations.clear();
        _frame_views.clear();
        _replacement_cpp = p;
        snap_found   = num_times_found;
        snap_visible = num_times_visible;
    }
    _is_bad.store(true);

    // Transfer observations to replacement
    for (auto &[kf, idx] : obs_copy) {
        if (p->is_in_keyframe(kf)) {
            try { kf.attr("remove_point_match")(idx); } catch (...) {}
        } else {
            p->add_observation(kf, idx);
        }
    }

    // Transfer found/visible statistics
    p->increase_found(snap_found);
    p->increase_visible(snap_visible);

    // Remove from map
    if (!map.is_none()) {
        try { map.attr("remove_map_point")(shared_from_this()); } catch (...) {}
    }

    p->update_info();
}

MapPointPtr MapPoint::get_replacement() const {
    std::lock_guard<std::mutex> lk(_lock_replacement);
    return _replacement_cpp;
}

// ---- Info update -----------------------------------------------------------
void MapPoint::update_info() {
    update_best_descriptor();
    update_normal_and_depth();
}

void MapPoint::update_best_descriptor(bool /*force*/) {
    // Collect descriptors from all observing KFs (GIL held)
    std::vector<cv::Mat> descriptors;
    {
        std::lock_guard<std::mutex> lk(_lock_features);
        descriptors.reserve(_observations.size());
        for (const auto &[kf, idx] : _observations) {
            try {
                auto des_arr = kf.attr("des");
                if (!des_arr.is_none()) {
                    // des_arr is np.ndarray (N, 32) uint8
                    auto np_des = des_arr.template cast<py::array_t<uint8_t>>();
                    auto r = np_des.template unchecked<2>();
                    if (idx < static_cast<int>(r.shape(0))) {
                        cv::Mat d(1, 32, CV_8U);
                        std::memcpy(d.data, &r(idx, 0), 32);
                        descriptors.push_back(d);
                    }
                }
            } catch (...) {}
        }
    }

    if (descriptors.empty()) return;
    if (descriptors.size() == 1) {
        std::lock_guard<std::mutex> lk(_lock_features);
        des = descriptors[0].clone();
        return;
    }

    // Find medoid: descriptor with minimum max distance to all others
    int N = static_cast<int>(descriptors.size());
    std::vector<std::vector<float>> D(N, std::vector<float>(N, 0.0f));
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) {
            float d = _hamming_distance(descriptors[i], descriptors[j]);
            D[i][j] = D[j][i] = d;
        }

    int best = 0;
    float best_max = std::numeric_limits<float>::max();
    for (int i = 0; i < N; ++i) {
        float mx = *std::max_element(D[i].begin(), D[i].end());
        if (mx < best_max) { best_max = mx; best = i; }
    }

    std::lock_guard<std::mutex> lk(_lock_features);
    des = descriptors[best].clone();
}

void MapPoint::update_normal_and_depth(bool /*force*/) {
    // GIL held. Compute mean normal and update min/max distance.
    std::vector<std::pair<py::object, int>> obs_copy;
    {
        std::lock_guard<std::mutex> lk(_lock_features);
        obs_copy.assign(_observations.begin(), _observations.end());
    }
    if (obs_copy.empty()) return;

    Eigen::Vector3d pos = get_position();
    Eigen::Vector3d normal_sum = Eigen::Vector3d::Zero();
    float min_dist = std::numeric_limits<float>::max();
    float max_dist = 0.0f;

    for (const auto &[kf, idx] : obs_copy) {
        try {
            // Camera center: kf.Ow may be a property (array) or method (callable)
            Eigen::Vector3d Ow;
            py::object Ow_val = kf.attr("Ow");
            if (!py::isinstance<py::array>(Ow_val)) {
                // callable method — call it
                try { Ow_val = Ow_val(); } catch (...) { continue; }
            }
            if (py::isinstance<py::array>(Ow_val)) {
                auto a = Ow_val.cast<py::array_t<double>>();
                auto r = a.unchecked<1>();
                Ow = Eigen::Vector3d(r(0), r(1), r(2));
            } else {
                continue;
            }

            Eigen::Vector3d n = pos - Ow;
            float dist = static_cast<float>(n.norm());
            if (dist > 1e-10f) {
                normal_sum += n / dist;
                if (dist < min_dist) min_dist = dist;
                if (dist > max_dist) max_dist = dist;
            }

            // Scale distance by octave level
            int octave = 0;
            try {
                auto octaves_attr = kf.attr("octaves");
                if (!octaves_attr.is_none()) {
                    octave = octaves_attr[py::int_(idx)].cast<int>();
                }
            } catch (...) {}

            // Scale factor adjustment (ORB: each level multiplies by 1.2)
            float scale = 1.0f;
            try {
                auto fm = kf.attr("_feature_manager");
                if (!fm.is_none()) {
                    auto sf = fm.attr("scale_factors");
                    if (!sf.is_none()) {
                        scale = sf[py::int_(octave)].cast<float>();
                    }
                }
            } catch (...) { scale = std::pow(1.2f, octave); }

            min_dist = std::min(min_dist, dist / scale);
            max_dist = std::max(max_dist, dist * scale);
        } catch (...) {}
    }

    if (normal_sum.norm() > 1e-10) {
        std::lock_guard<std::mutex> lk(_lock_pos);
        normal = normal_sum.normalized();
        // Store raw values without invariance margins so Python predict_scale()
        // and get_{min,max}_distance_invariance() work correctly (Python applies
        // 0.8x / 1.2x at call time, not at storage time).
        _min_distance = min_dist;
        _max_distance = max_dist;
    }
}

// ---- Scale prediction ------------------------------------------------------
int MapPoint::predict_detection_level(float dist) const {
    if (_max_distance <= 0.0f || !std::isfinite(_max_distance)) return 0;
    // Match Python predict_scale(): ratio = max_distance / dist
    float ratio = _max_distance / std::max(dist, 1e-9f);
    int level = static_cast<int>(std::ceil(std::log(ratio) / std::log(1.2f)));
    return std::max(0, std::min(level, 7));  // ORB has max 8 levels
}

// ---- Statistics ------------------------------------------------------------
void MapPoint::increase_visible(int n) { num_times_visible += n; }
void MapPoint::increase_found(int n)   { num_times_found += n; }
float MapPoint::get_found_ratio() const {
    if (num_times_visible <= 0) return 0.0f;
    return static_cast<float>(num_times_found) / static_cast<float>(num_times_visible);
}

// ---- Repr ------------------------------------------------------------------
std::string MapPoint::__repr__() const {
    std::ostringstream ss;
    ss << "MapPoint(id=" << id << ", n_obs=" << _num_observations
       << ", bad=" << _is_bad.load() << ")";
    return ss.str();
}

} // namespace slam
