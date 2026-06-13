#pragma once
// LtmStore — the persistence seam for Long-Term Memory.
//
// The MemoryManager talks ONLY to this interface, so the in-RAM v2 store can be
// swapped for a SQLite-backed implementation (v3: bounded RAM regardless of run
// length, on-disk map surviving restarts) without touching tier logic.

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "signature.h"

namespace fusion {

class LtmStore {
public:
    virtual ~LtmStore() = default;
    virtual void store(const SignaturePtr& sig) = 0;
    virtual SignaturePtr load(int32_t id) = 0;       // nullptr when absent
    virtual bool contains(int32_t id) const = 0;
    virtual void erase(int32_t id) = 0;
    virtual size_t size() const = 0;
    virtual std::vector<int32_t> ids() const = 0;
    virtual size_t payload_bytes() const = 0;        // diagnostics
};

class InRamLtmStore : public LtmStore {
public:
    void store(const SignaturePtr& sig) override {
        std::lock_guard<std::mutex> lk(_m);
        _store[sig->id] = sig;
    }
    SignaturePtr load(int32_t id) override {
        std::lock_guard<std::mutex> lk(_m);
        auto it = _store.find(id);
        return it == _store.end() ? nullptr : it->second;
    }
    bool contains(int32_t id) const override {
        std::lock_guard<std::mutex> lk(_m);
        return _store.count(id) > 0;
    }
    void erase(int32_t id) override {
        std::lock_guard<std::mutex> lk(_m);
        _store.erase(id);
    }
    size_t size() const override {
        std::lock_guard<std::mutex> lk(_m);
        return _store.size();
    }
    std::vector<int32_t> ids() const override {
        std::lock_guard<std::mutex> lk(_m);
        std::vector<int32_t> out;
        out.reserve(_store.size());
        for (const auto& kv : _store) out.push_back(kv.first);
        return out;
    }
    size_t payload_bytes() const override {
        std::lock_guard<std::mutex> lk(_m);
        size_t total = 0;
        for (const auto& kv : _store) total += kv.second->payload_bytes();
        return total;
    }

private:
    mutable std::mutex _m;
    std::unordered_map<int32_t, SignaturePtr> _store;
};

}  // namespace fusion
