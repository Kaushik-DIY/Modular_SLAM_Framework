#pragma once
// FusionGraph2D — keyframe-only SE(2) pose graph on native g2o (types_slam2d).
//
// Conventions copied from carto/pose_graph/backends/g2o_backend_2d.py so results
// are interchangeable:
//   * information = diag(tw, tw, rw)
//   * spine (consecutive-keyframe) edges: trusted, NO robust kernel
//   * loop edges: Huber kernel (delta = huber_scale)
//   * one anchor node held fixed; Levenberg-Marquardt + BlockSolverSE2(Eigen)
// The optimizer is rebuilt from stored state on each optimize() call (same
// semantics as the Python backend; cheap in C++ even at 10k nodes).

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "common.h"

namespace fusion {

struct GraphConfig {
    double huber_scale = 10.0;
    double spine_trans_weight = 1e5;
    double spine_rot_weight = 1e5;
    int max_iterations = 50;
};

struct GraphEdge {
    int32_t from_id;
    int32_t to_id;
    Pose2 rel_pose;        // z = T_from^{-1} * T_to
    double trans_weight;
    double rot_weight;
    bool is_loop;          // loop edges get the Huber kernel
};

class FusionGraph2D {
public:
    explicit FusionGraph2D(GraphConfig cfg = {}) : _cfg(cfg) {}

    void add_node(int32_t id, const Pose2& pose) {
        std::lock_guard<std::mutex> lk(_m);
        const bool first = _nodes.empty();
        _nodes[id] = pose;
        if (first && _fixed_id < 0) _fixed_id = id;
    }

    void set_node_pose(int32_t id, const Pose2& pose) {
        std::lock_guard<std::mutex> lk(_m);
        _nodes.at(id) = pose;
    }

    void set_fixed(int32_t id) {
        std::lock_guard<std::mutex> lk(_m);
        _fixed_id = id;
    }

    void add_spine_edge(int32_t from, int32_t to, const Pose2& rel,
                        double tw = -1.0, double rw = -1.0) {
        std::lock_guard<std::mutex> lk(_m);
        _edges.push_back(GraphEdge{from, to, rel,
                                   tw > 0 ? tw : _cfg.spine_trans_weight,
                                   rw > 0 ? rw : _cfg.spine_rot_weight, false});
    }

    void add_loop_edge(int32_t from, int32_t to, const Pose2& rel,
                       double tw, double rw) {
        std::lock_guard<std::mutex> lk(_m);
        _edges.push_back(GraphEdge{from, to, rel, tw, rw, true});
    }

    // Runs g2o; returns chi2 after optimization. Thread-safe; call GIL-released.
    double optimize(int iterations = -1);

    Pose2 get_pose(int32_t id) const {
        std::lock_guard<std::mutex> lk(_m);
        return _nodes.at(id);
    }

    bool has_node(int32_t id) const {
        std::lock_guard<std::mutex> lk(_m);
        return _nodes.count(id) > 0;
    }

    // Sorted-by-id snapshot: rows of (id, x, y, theta).
    std::vector<std::array<double, 4>> poses_snapshot() const {
        std::lock_guard<std::mutex> lk(_m);
        std::vector<std::array<double, 4>> out;
        out.reserve(_nodes.size());
        for (const auto& kv : _nodes)
            out.push_back({static_cast<double>(kv.first), kv.second.x, kv.second.y,
                           kv.second.theta});
        return out;
    }

    size_t num_nodes() const { std::lock_guard<std::mutex> lk(_m); return _nodes.size(); }
    size_t num_edges() const { std::lock_guard<std::mutex> lk(_m); return _edges.size(); }
    size_t num_loop_edges() const {
        std::lock_guard<std::mutex> lk(_m);
        size_t n = 0;
        for (const auto& e : _edges) n += e.is_loop;
        return n;
    }
    int32_t fixed_id() const { std::lock_guard<std::mutex> lk(_m); return _fixed_id; }

private:
    GraphConfig _cfg;
    mutable std::mutex _m;
    std::map<int32_t, Pose2> _nodes;  // ordered: deterministic vertex order
    std::vector<GraphEdge> _edges;
    int32_t _fixed_id = -1;
};

}  // namespace fusion
