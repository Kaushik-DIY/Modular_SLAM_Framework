#pragma once
// MemoryManager — RTAB-Map-faithful STM / WM / LTM tiers (RTAB_review.md §3).
//
//  * STM: bounded FIFO of the newest signatures; hidden from loop closure
//    (callers must draw candidates from WM only). Oldest ages out to WM.
//  * WM: loop-closure working set, capped; over cap the OLDEST of the
//    LOWEST-WEIGHT members transfers to LTM (§3.4 transfer rule).
//  * LTM: behind the LtmStore interface (in-RAM now, SQLite later).
//  * Rehearsal (§3.4): if the new signature is similar enough to its STM
//    predecessor, they merge — the new one survives, inherits weight
//    (w_t += w_c + 1) and the predecessor's links; predecessor is deleted.
//  * Loop weight bump (§3.4): w_t += w_i + 1 on confirmed closure.
//
// All methods are thread-safe (single internal mutex) and pure C++ (callers
// release the GIL).

#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ltm_store.h"
#include "signature.h"

namespace fusion {

struct MemoryConfig {
    int stm_size = 30;
    int wm_cap = 200;
    double rehearsal_similarity = 0.20;
    bool rehearsal_enabled = true;
    int reactivation_neighbor_depth = 1;
};

enum class Tier : uint8_t { STM = 0, WM = 1, LTM = 2, NONE = 3 };

struct InsertResult {
    int32_t id = -1;
    double similarity = -1.0;            // similarity used for rehearsal test
    bool rehearsal_merged = false;
    int32_t merged_predecessor_id = -1;  // deleted signature (if merged)
    std::vector<int32_t> moved_to_wm;    // STM -> WM agings this insert
    std::vector<int32_t> transferred_to_ltm;  // WM -> LTM transfers this insert
};

class MemoryManager {
public:
    MemoryManager(MemoryConfig cfg, std::shared_ptr<LtmStore> ltm)
        : _cfg(cfg), _ltm(std::move(ltm)) {}

    // similarity < 0 => auto: scan-overlap vs STM predecessor when both carry
    // scans, else 0 (no rehearsal).
    InsertResult insert(const SignaturePtr& sig, double similarity = -1.0) {
        std::lock_guard<std::mutex> lk(_m);
        InsertResult res;
        res.id = sig->id;

        SignaturePtr pred = _stm.empty() ? nullptr : _stm.back();

        double sim = similarity;
        if (sim < 0.0) {
            sim = (pred && pred->has_scan() && sig->has_scan())
                      ? scan_overlap_similarity(*pred, *sig)
                      : 0.0;
        }
        res.similarity = sim;

        if (_cfg.rehearsal_enabled && pred && sim > _cfg.rehearsal_similarity) {
            // Rehearsal merge: new signature survives, predecessor deleted.
            sig->weight += pred->weight + 1;
            for (const Link& l : pred->links) {
                if (l.to_id != sig->id) sig->links.push_back(l);
            }
            sig->links.push_back(Link{pred->id, LinkType::MERGED, Pose2(), 0.0, 0.0});
            _stm.pop_back();
            _index.erase(pred->id);
            res.rehearsal_merged = true;
            res.merged_predecessor_id = pred->id;
        }

        _stm.push_back(sig);
        _index[sig->id] = Tier::STM;

        // STM aging: oldest moves to WM.
        while (static_cast<int>(_stm.size()) > _cfg.stm_size) {
            SignaturePtr oldest = _stm.front();
            _stm.pop_front();
            _wm[oldest->id] = oldest;
            _access[oldest->id] = ++_access_seq;
            _index[oldest->id] = Tier::WM;
            res.moved_to_wm.push_back(oldest->id);
        }

        // WM transfer: oldest of the lowest-weight members -> LTM.
        while (static_cast<int>(_wm.size()) > _cfg.wm_cap) {
            int32_t victim = pick_transfer_victim();
            SignaturePtr s = _wm[victim];
            _ltm->store(s);
            _wm.erase(victim);
            _access.erase(victim);
            _index[victim] = Tier::LTM;
            res.transferred_to_ltm.push_back(victim);
        }
        return res;
    }

    // Confirmed loop closure: query's weight absorbs matched weight + 1 (§3.4),
    // and both ends are "touched" (recently useful -> last to transfer).
    void on_loop_confirmed(int32_t query_id, int32_t matched_id) {
        std::lock_guard<std::mutex> lk(_m);
        SignaturePtr q = get_unlocked(query_id);
        SignaturePtr t = get_unlocked(matched_id);
        if (!q || !t) return;
        q->weight += t->weight + 1;
        touch_unlocked(query_id);
        touch_unlocked(matched_id);
    }

    // LTM -> WM reactivation of `id` plus graph neighbors within depth.
    // Returns the ids actually moved back into WM.
    std::vector<int32_t> reactivate(int32_t id, int neighbor_depth = -1) {
        std::lock_guard<std::mutex> lk(_m);
        if (neighbor_depth < 0) neighbor_depth = _cfg.reactivation_neighbor_depth;
        std::vector<int32_t> moved;
        std::unordered_set<int32_t> visited;
        std::vector<std::pair<int32_t, int>> frontier{{id, 0}};
        while (!frontier.empty()) {
            auto [cur, depth] = frontier.back();
            frontier.pop_back();
            if (!visited.insert(cur).second) continue;
            SignaturePtr s = nullptr;
            if (_ltm->contains(cur)) {
                s = _ltm->load(cur);
                if (s) {
                    _ltm->erase(cur);
                    _wm[cur] = s;
                    _access[cur] = ++_access_seq;
                    _index[cur] = Tier::WM;
                    moved.push_back(cur);
                }
            } else {
                s = get_unlocked(cur);
            }
            if (s && depth < neighbor_depth) {
                for (const Link& l : s->links)
                    frontier.push_back({l.to_id, depth + 1});
            }
        }
        return moved;
    }

    void touch(int32_t id) {
        std::lock_guard<std::mutex> lk(_m);
        touch_unlocked(id);
    }

    Tier tier(int32_t id) const {
        std::lock_guard<std::mutex> lk(_m);
        auto it = _index.find(id);
        return it == _index.end() ? Tier::NONE : it->second;
    }

    SignaturePtr get(int32_t id) {
        std::lock_guard<std::mutex> lk(_m);
        return get_unlocked(id);
    }

    std::vector<int32_t> stm_ids() const {
        std::lock_guard<std::mutex> lk(_m);
        std::vector<int32_t> out;
        out.reserve(_stm.size());
        for (const auto& s : _stm) out.push_back(s->id);
        return out;
    }

    std::vector<int32_t> wm_ids() const {
        std::lock_guard<std::mutex> lk(_m);
        std::vector<int32_t> out;
        out.reserve(_wm.size());
        for (const auto& kv : _wm) out.push_back(kv.first);
        return out;
    }

    size_t stm_count() const { std::lock_guard<std::mutex> lk(_m); return _stm.size(); }
    size_t wm_count() const { std::lock_guard<std::mutex> lk(_m); return _wm.size(); }
    size_t ltm_count() const { return _ltm->size(); }

    size_t payload_bytes() const {
        std::lock_guard<std::mutex> lk(_m);
        size_t total = _ltm->payload_bytes();
        for (const auto& s : _stm) total += s->payload_bytes();
        for (const auto& kv : _wm) total += kv.second->payload_bytes();
        return total;
    }

    // Coarse scan-overlap similarity: fraction of b's points landing in cells
    // occupied by a's points (0.1 m grid). Cheap O(M) auto-rehearsal signal for
    // lidar-only streams.
    static double scan_overlap_similarity(const Signature& a, const Signature& b) {
        if (a.scan_xy.rows() == 0 || b.scan_xy.rows() == 0) return 0.0;
        constexpr double res = 0.1;
        std::unordered_set<int64_t> cells;
        cells.reserve(static_cast<size_t>(a.scan_xy.rows()) * 2);
        for (int i = 0; i < a.scan_xy.rows(); ++i) {
            const int64_t cx = static_cast<int64_t>(std::floor(a.scan_xy(i, 0) / res));
            const int64_t cy = static_cast<int64_t>(std::floor(a.scan_xy(i, 1) / res));
            cells.insert((cx << 32) ^ (cy & 0xffffffff));
        }
        int hits = 0;
        for (int i = 0; i < b.scan_xy.rows(); ++i) {
            const int64_t cx = static_cast<int64_t>(std::floor(b.scan_xy(i, 0) / res));
            const int64_t cy = static_cast<int64_t>(std::floor(b.scan_xy(i, 1) / res));
            if (cells.count((cx << 32) ^ (cy & 0xffffffff))) ++hits;
        }
        return static_cast<double>(hits) / static_cast<double>(b.scan_xy.rows());
    }

private:
    SignaturePtr get_unlocked(int32_t id) {
        for (const auto& s : _stm)
            if (s->id == id) return s;
        auto it = _wm.find(id);
        if (it != _wm.end()) return it->second;
        if (_ltm->contains(id)) return _ltm->load(id);
        return nullptr;
    }

    void touch_unlocked(int32_t id) {
        if (_wm.count(id)) _access[id] = ++_access_seq;
    }

    // §3.4: lowest weight first; among equals, the oldest (smallest access seq).
    int32_t pick_transfer_victim() const {
        int32_t victim = -1;
        int32_t best_w = INT32_MAX;
        uint64_t best_age = UINT64_MAX;
        for (const auto& kv : _wm) {
            const int32_t w = kv.second->weight;
            const uint64_t age = _access.at(kv.first);
            if (w < best_w || (w == best_w && age < best_age)) {
                victim = kv.first;
                best_w = w;
                best_age = age;
            }
        }
        return victim;
    }

    MemoryConfig _cfg;
    std::shared_ptr<LtmStore> _ltm;
    mutable std::mutex _m;
    std::deque<SignaturePtr> _stm;
    std::unordered_map<int32_t, SignaturePtr> _wm;
    std::unordered_map<int32_t, uint64_t> _access;  // WM last-access sequence
    std::unordered_map<int32_t, Tier> _index;
    uint64_t _access_seq = 0;
};

}  // namespace fusion
