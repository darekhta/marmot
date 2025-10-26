#include "kernel_query.hpp"

#include <array>
#include <mutex>

#include "core/dispatch/kernel_query.h"

namespace marmot::graph {

namespace {

struct backend_caps_cache_t {
    std::array<std::once_flag, MARMOT_BACKEND_COUNT> once;
    std::array<marmot_device_caps_t, MARMOT_BACKEND_COUNT> caps;
    std::array<bool, MARMOT_BACKEND_COUNT> valid;
};

backend_caps_cache_t &backend_caps_cache(void) {
    static backend_caps_cache_t cache = {};
    return cache;
}

bool backend_caps_cached(marmot_backend_type_t backend, marmot_device_caps_t *caps_out) {
    const int backend_index = (int)backend;
    if (caps_out == nullptr || backend_index < 0 || backend_index >= (int)MARMOT_BACKEND_COUNT) {
        return false;
    }

    const size_t index = (size_t)backend_index;
    backend_caps_cache_t &cache = backend_caps_cache();
    std::call_once(cache.once[index], [backend, index]() {
        marmot_device_caps_t caps = {};
        const bool ok = marmot_backend_detect_default_caps(backend, &caps);
        backend_caps_cache_t &cache_ref = backend_caps_cache();
        cache_ref.valid[index] = ok;
        if (ok) {
            cache_ref.caps[index] = caps;
        }
    });

    if (!cache.valid[index]) {
        return false;
    }

    *caps_out = cache.caps[index];
    return true;
}

marmot_kernel_selection_t kernel_selection_unsupported(const char *reason) {
    return {
        .supported = false,
        .kernel_id = MARMOT_KERNEL_INVALID,
        .op_index = MARMOT_KERNEL_OP_INDEX_INVALID,
        .estimated_us = 0.0,
        .est_comm_us = 0.0,
        .est_workspace_mb = 0.0,
        .confidence = 0.0f,
        .fallback_reason = reason,
        .shardable_axes = 0,
        .device_affinity = 0,
    };
}

} // namespace

marmot_kernel_selection_t
query_backend_for_node(marmot_backend_type_t backend, const marmot_op_signature_t *signature) {
    marmot_device_caps_t caps = {};
    if (!backend_caps_cached(backend, &caps)) {
        return kernel_selection_unsupported("Backend not supported");
    }
    return marmot_backend_query_kernel(backend, signature, &caps);
}

} // namespace marmot::graph
