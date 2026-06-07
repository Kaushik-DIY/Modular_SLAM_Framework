#include "keyframe.h"

#include <algorithm>
#include <cstdint>
#include <sstream>

namespace slam {

// Stable sort key for a keyframe py::object: kid (then id, then pointer address as
// last resort). Covisibility is stored in an unordered_map keyed by pointer
// (PyObjHash) -> run-to-run varying order; ordering output by this stable key keeps
// get_best/connected covisibles deterministic (M0/M1 reproducibility parity with
// the Python KeyFrameGraph fix). GIL is held by all callers (py::object attr access).
static long _kf_sort_key(const py::object &kf) {
    try { return kf.attr("kid").cast<long>(); } catch (...) {}
    try { return kf.attr("id").cast<long>(); } catch (...) {}
    return static_cast<long>(reinterpret_cast<std::uintptr_t>(kf.ptr()));
}

// ---- Construction ----------------------------------------------------------
KeyFrame::KeyFrame(int given_kid, int given_frame_id)
    : Frame(given_frame_id),
      kid(given_kid >= 0 ? given_kid : (given_frame_id >= 0 ? given_frame_id : id)),
      map(py::none()),
      Tcw_GBA(py::none()),
      Tcw_before_GBA(py::none()),
      _parent(py::none()) {}

// ---- Status ----------------------------------------------------------------
void KeyFrame::set_not_erase() {
    std::lock_guard<std::mutex> lk(_lock_connections);
    not_to_erase = true;
}

void KeyFrame::set_erase() {
    bool should_set_bad = false;
    {
        std::lock_guard<std::mutex> lk(_lock_connections);
        if (_loop_edges.empty()) {
            not_to_erase = false;
        }
        if (to_be_erased_kf) {
            should_set_bad = true;
        }
    }
    if (should_set_bad) set_bad();
}

void KeyFrame::set_bad() {
    if (kid == 0) return;

    std::vector<py::object> connected_copy;
    {
        std::lock_guard<std::mutex> lk(_lock_connections);
        if (not_to_erase) {
            to_be_erased_kf = true;
            return;
        }
        for (const auto &[kf, _w] : _covis_weights) {
            connected_copy.push_back(kf);
        }
    }

    // Remove from all covisible KFs' graphs
    for (auto &kf_obj : connected_copy) {
        try {
            kf_obj.attr("erase_connection")(py::cast(shared_from_this()));
        } catch (...) {}
    }

    // Remove observations from map points
    std::vector<std::pair<int, py::object>> pts_copy;
    {
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            if (!points[i].is_none()) {
                pts_copy.push_back({i, points[i]});
            }
        }
    }
    for (auto &[idx, mp_obj] : pts_copy) {
        try {
            mp_obj.attr("remove_observation")(py::cast(shared_from_this()), idx);
        } catch (...) {}
        points[idx] = py::none();
    }

    py::object parent_snapshot;
    {
        std::lock_guard<std::mutex> lk(_lock_connections);
        reset_covisibility();
        parent_snapshot = _parent;   // snapshot; the parent callouts run outside the lock
        _children.clear();
        _kf_is_bad.store(true);
    }

    // Detach from parent OUTSIDE _lock_connections: erase_child acquires the PARENT's
    // _lock_connections -> a KF<->KF nesting if done under our own (F4 invariant).
    // bool(py::object) guards against a default-constructed (NULL) _parent.
    if (parent_snapshot && !parent_snapshot.is_none()) {
        try {
            // Tcp = Tcw * parent.Twc() (stored as pose attribute — simplified for now)
            auto parent_Twc = parent_snapshot.attr("Twc")();
            (void)parent_Twc;
        } catch (...) {}
        try {
            parent_snapshot.attr("erase_child")(py::cast(shared_from_this()));
        } catch (...) {}
    }

    // Remove from map
    if (!map.is_none()) {
        try { map.attr("remove_keyframe")(py::cast(shared_from_this())); } catch (...) {}
    }
}

// ---- Covisibility graph ----------------------------------------------------
void KeyFrame::_rebuild_ordered_covis_no_lock_() {
    _ordered_covis.assign(_covis_weights.begin(), _covis_weights.end());
    // weight DESC, then stable keyframe id ASC (std::sort is not stable and
    // _covis_weights iterates in pointer order, so the id tie-break is required
    // for run-to-run deterministic covisibility — M1 parity with the Python fix).
    std::sort(_ordered_covis.begin(), _ordered_covis.end(),
              [](const auto &a, const auto &b) {
                  if (a.second != b.second) return a.second > b.second;
                  return _kf_sort_key(a.first) < _kf_sort_key(b.first);
              });
}

void KeyFrame::add_connection_no_lock_(py::object other_kf, int weight) {
    if (other_kf.is_none() || other_kf.ptr() == py::cast(shared_from_this()).ptr()) return;
    _covis_weights[other_kf] = weight;
    _rebuild_ordered_covis_no_lock_();
}

void KeyFrame::add_connection(py::object other_kf, int weight) {
    std::lock_guard<std::mutex> lk(_lock_connections);
    add_connection_no_lock_(other_kf, weight);
}

void KeyFrame::erase_connection_no_lock_(py::object other_kf) {
    auto it = _covis_weights.find(other_kf);
    if (it != _covis_weights.end()) {
        _covis_weights.erase(it);
        _rebuild_ordered_covis_no_lock_();
    }
}

void KeyFrame::erase_connection(py::object other_kf) {
    std::lock_guard<std::mutex> lk(_lock_connections);
    erase_connection_no_lock_(other_kf);
}

std::vector<py::object> KeyFrame::get_connected_keyframes() const {
    std::lock_guard<std::mutex> lk(_lock_connections);
    std::vector<py::object> result;
    result.reserve(_covis_weights.size());
    for (const auto &[kf, _w] : _covis_weights) result.push_back(kf);
    // Deterministic order (the unordered_map iterates in pointer order).
    std::sort(result.begin(), result.end(),
              [](const py::object &a, const py::object &b) { return _kf_sort_key(a) < _kf_sort_key(b); });
    return result;
}

std::vector<py::object> KeyFrame::get_covisible_keyframes() const {
    std::lock_guard<std::mutex> lk(_lock_connections);
    std::vector<py::object> result;
    result.reserve(_ordered_covis.size());
    for (const auto &[kf, _w] : _ordered_covis) result.push_back(kf);
    return result;
}

std::vector<py::object> KeyFrame::get_best_covisible_keyframes(int N) const {
    std::lock_guard<std::mutex> lk(_lock_connections);
    std::vector<py::object> result;
    int take = std::min(N, static_cast<int>(_ordered_covis.size()));
    result.reserve(take);
    for (int i = 0; i < take; ++i) result.push_back(_ordered_covis[i].first);
    return result;
}

std::vector<py::object> KeyFrame::get_covisible_by_weight(int min_weight) const {
    std::lock_guard<std::mutex> lk(_lock_connections);
    std::vector<py::object> result;
    for (const auto &[kf, w] : _ordered_covis) {
        if (w <= min_weight) break;
        result.push_back(kf);
    }
    return result;
}

int KeyFrame::get_weight(py::object other_kf) const {
    std::lock_guard<std::mutex> lk(_lock_connections);
    auto it = _covis_weights.find(other_kf);
    return (it != _covis_weights.end()) ? it->second : 0;
}

void KeyFrame::reset_covisibility() {
    _covis_weights.clear();
    _ordered_covis.clear();
}

void KeyFrame::update_connections() {
    // Build counter of co-visibility from shared map points.
    // Points can be C++ MapPoint or Python MapPoint — both respond to keyframes().
    std::vector<py::object> pts_copy;
    {
        for (const auto &p : points) {
            if (!p.is_none()) pts_copy.push_back(p);
        }
    }

    if (pts_copy.empty()) return;

    // Count co-visible KFs (GIL held — accessing Python objects)
    std::unordered_map<py::object, int, PyObjHash, PyObjEqual> counter;
    py::object self_obj = py::cast(shared_from_this());

    for (const auto &mp_obj : pts_copy) {
        // Skip bad points
        try {
            if (mp_obj.attr("is_bad")().cast<bool>()) continue;
        } catch (...) {}

        // Get observing KFs for this map point
        std::vector<py::object> obs_kfs;
        try {
            auto kfs_list = mp_obj.attr("keyframes")();
            for (auto kf_item : kfs_list.cast<py::list>()) {
                obs_kfs.push_back(py::reinterpret_borrow<py::object>(kf_item));
            }
        } catch (...) { continue; }

        for (const auto &kf_obj : obs_kfs) {
            if (kf_obj.ptr() == self_obj.ptr()) continue;

            // Skip KFs with same kid (self by another wrapper)
            try {
                int other_kid = kf_obj.attr("kid").cast<int>();
                if (other_kid == kid) continue;
            } catch (...) {}

            // Skip bad KFs
            try {
                if (kf_obj.attr("is_bad")().cast<bool>()) continue;
            } catch (...) {}

            counter[kf_obj]++;
        }
    }

    if (counter.empty()) return;

    // Find max weight
    int w_max = 0;
    py::object kf_max;
    for (const auto &[kf, w] : counter) {
        if (w > w_max) { w_max = w; kf_max = kf; }
    }

    // Get min threshold from Parameters
    int min_covis = 15;  // default pySLAM threshold
    try {
        py::object params = py::module_::import(
            "visual_slam.orbslam.slam.config_parameters").attr("Parameters");
        min_covis = params.attr("kMinNumOfCovisiblePointsForCreatingConnection").cast<int>();
    } catch (...) {}

    // Sort counter by weight descending
    std::vector<std::pair<py::object, int>> sorted_counter(counter.begin(), counter.end());
    std::sort(sorted_counter.begin(), sorted_counter.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    // F4 invariant: write only pure-C++ state under _lock_connections; collect the
    // cross-KF notifications (each calls into ANOTHER keyframe's locked methods via
    // Python — a GIL op) and perform them OUTSIDE the lock. Holding our own
    // _lock_connections while calling kf.add_connection_no_lock_/add_child would be a
    // KF<->KF lock + GIL<->mutex inversion once the LM matcher runs GIL-released.
    std::vector<std::pair<py::object, int>> notify;   // (kf, w) to add_connection_no_lock_
    py::object parent_to_set;                          // kf_max if first connection set here

    // The parent candidate's is_bad gate is a callout -> evaluate outside the lock.
    bool kf_max_ok = false;
    if (_is_first_connection && kid != 0 && !kf_max.is_none()) {
        try { kf_max_ok = !kf_max.attr("is_bad")().cast<bool>(); } catch (...) { kf_max_ok = false; }
    }

    {
        std::lock_guard<std::mutex> lk(_lock_connections);
        _covis_weights = counter;

        _ordered_covis.clear();
        if (w_max >= min_covis) {
            for (const auto &[kf, w] : sorted_counter) {
                if (w >= min_covis) {
                    notify.push_back({kf, w});
                    _ordered_covis.push_back({kf, w});
                } else {
                    break;
                }
            }
        } else {
            notify.push_back({kf_max, w_max});
            _ordered_covis.push_back({kf_max, w_max});
        }

        // Spanning-tree parent on first connection: pure-C++ state write here; the
        // parent.add_child(self) notification is performed outside the lock below
        // (mirrors set_parent_no_lock_, whose only callout is add_child).
        if (_is_first_connection && kid != 0 && !kf_max.is_none() && kf_max_ok) {
            _parent = kf_max;
            _init_parent = true;
            _is_first_connection = false;
            parent_to_set = kf_max;
        }
    }

    // Cross-KF notifications OUTSIDE the lock.
    for (auto &[kf, w] : notify) {
        try { kf.attr("add_connection_no_lock_")(self_obj, py::int_(w)); } catch (...) {}
    }
    if (parent_to_set) {   // bool(py::object) is a NULL-ptr check (a default-constructed
                           // py::object is NULL, not None — is_none() would be false on it)
        try { parent_to_set.attr("add_child")(self_obj); } catch (...) {}
    }
}

// ---- Spanning tree ---------------------------------------------------------
void KeyFrame::set_parent_no_lock_(py::object kf) {
    if (kf.is_none() || kf.ptr() == py::cast(shared_from_this()).ptr()) return;
    _parent = kf;
    _init_parent = true;
    try { kf.attr("add_child")(py::cast(shared_from_this())); } catch (...) {}
}

void KeyFrame::set_parent(py::object kf) {
    std::lock_guard<std::mutex> lk(_lock_connections);
    set_parent_no_lock_(kf);
}

py::object KeyFrame::get_parent() const {
    std::lock_guard<std::mutex> lk(_lock_connections);
    return _parent;
}

void KeyFrame::add_child_no_lock_(py::object kf) {
    if (!kf.is_none()) _children.push_back(kf);
}

void KeyFrame::add_child(py::object kf) {
    std::lock_guard<std::mutex> lk(_lock_connections);
    add_child_no_lock_(kf);
}

void KeyFrame::erase_child_no_lock_(py::object kf) {
    auto it = std::find_if(_children.begin(), _children.end(),
                           [&kf](const py::object &c) { return c.ptr() == kf.ptr(); });
    if (it != _children.end()) _children.erase(it);
}

void KeyFrame::erase_child(py::object kf) {
    std::lock_guard<std::mutex> lk(_lock_connections);
    erase_child_no_lock_(kf);
}

std::vector<py::object> KeyFrame::get_children() const {
    std::lock_guard<std::mutex> lk(_lock_connections);
    return _children;
}

bool KeyFrame::has_child(py::object kf) const {
    std::lock_guard<std::mutex> lk(_lock_connections);
    return std::any_of(_children.begin(), _children.end(),
                       [&kf](const py::object &c) { return c.ptr() == kf.ptr(); });
}

// ---- Loop edges ------------------------------------------------------------
void KeyFrame::add_loop_edge(py::object kf) {
    std::lock_guard<std::mutex> lk(_lock_connections);
    not_to_erase = true;
    if (!kf.is_none() && kf.ptr() != py::cast(shared_from_this()).ptr()) {
        _loop_edges.push_back(kf);
    }
}

std::vector<py::object> KeyFrame::get_loop_edges() const {
    std::lock_guard<std::mutex> lk(_lock_connections);
    return _loop_edges;
}

// ---- Helpers ---------------------------------------------------------------
std::vector<py::object> KeyFrame::get_matched_good_points() const {
    std::vector<py::object> result;
    for (const auto &p : points) {
        if (p.is_none()) continue;
        try {
            if (!p.attr("is_bad")().cast<bool>()) result.push_back(p);
        } catch (...) {}
    }
    return result;
}

std::vector<std::pair<py::object, int>> KeyFrame::get_matched_good_points_and_idxs() const {
    std::vector<std::pair<py::object, int>> result;
    for (int i = 0; i < (int)points.size(); i++) {
        const auto &p = points[i];
        if (p.is_none()) continue;
        try {
            if (!p.attr("is_bad")().cast<bool>()) result.emplace_back(p, i);
        } catch (...) {}
    }
    return result;
}

std::vector<py::object> KeyFrame::get_points() const {
    return points;
}

int KeyFrame::num_tracked_points(int min_obs) const {
    int count = 0;
    for (const auto &p : points) {
        if (p.is_none()) continue;
        try {
            if (p.attr("is_bad")().cast<bool>()) continue;
            if (min_obs > 0) {
                int n = p.attr("num_observations")().cast<int>();
                if (n < min_obs) continue;
            }
            ++count;
        } catch (...) {}
    }
    return count;
}

// ---- Repr ------------------------------------------------------------------
std::string KeyFrame::__repr__() const {
    std::ostringstream ss;
    ss << "KeyFrame(kid=" << kid << ", frame_id=" << id
       << ", n_kps=" << num_kps() << ", bad=" << _kf_is_bad.load() << ")";
    return ss.str();
}

} // namespace slam
