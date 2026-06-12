#pragma once
// Neighborhood retrieval — "take the relevant scans around the identified loop
// candidate based on how it is grouped in the map":
//   1. BFS over signature links from the candidate, up to `graph_depth` hops
//      (reactivating LTM members into WM as they are touched — RTAB retrieval).
//   2. Plus any WM/STM signature whose graph pose lies within `metric_radius`
//      of the candidate's pose (captures spatially-adjacent nodes that are far
//      in the graph).
// Returns the gathered signatures and their CURRENT graph poses, ready for
// assemble_local_grid().

#include <queue>
#include <unordered_set>
#include <vector>

#include "fusion_graph.h"
#include "memory_manager.h"
#include "signature.h"

namespace fusion {

struct Neighborhood {
    std::vector<SignaturePtr> signatures;
    std::vector<Pose2> poses;            // graph poses, parallel to signatures
    std::vector<int32_t> reactivated;    // ids pulled back from LTM
};

inline Neighborhood retrieve_neighborhood(MemoryManager& mem, FusionGraph2D& graph,
                                          int32_t candidate_id, int graph_depth = 2,
                                          double metric_radius = 5.0,
                                          bool scans_only = true) {
    Neighborhood out;
    std::unordered_set<int32_t> picked;

    // RTAB retrieval: pull the candidate (+ link neighbors) out of LTM first.
    out.reactivated = mem.reactivate(candidate_id, graph_depth);

    // 1) BFS over links.
    std::queue<std::pair<int32_t, int>> q;
    q.push({candidate_id, 0});
    while (!q.empty()) {
        auto [id, depth] = q.front();
        q.pop();
        if (picked.count(id)) continue;
        SignaturePtr s = mem.get(id);
        if (!s) continue;
        picked.insert(id);
        if (depth < graph_depth) {
            for (const Link& l : s->links) q.push({l.to_id, depth + 1});
        }
    }

    // 2) Metric neighbors among resident (STM+WM) signatures.
    if (metric_radius > 0.0 && graph.has_node(candidate_id)) {
        const Pose2 c = graph.get_pose(candidate_id);
        auto consider = [&](int32_t id) {
            if (picked.count(id) || !graph.has_node(id)) return;
            const Pose2 p = graph.get_pose(id);
            const double dx = p.x - c.x, dy = p.y - c.y;
            if (dx * dx + dy * dy <= metric_radius * metric_radius) picked.insert(id);
        };
        for (int32_t id : mem.wm_ids()) consider(id);
        for (int32_t id : mem.stm_ids()) consider(id);
    }

    for (int32_t id : picked) {
        SignaturePtr s = mem.get(id);
        if (!s) continue;
        if (scans_only && !s->has_scan()) continue;
        out.signatures.push_back(s);
        out.poses.push_back(graph.has_node(id) ? graph.get_pose(id) : s->pose);
    }
    return out;
}

}  // namespace fusion
