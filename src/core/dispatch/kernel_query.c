#include "kernel_query.h"

#include "marmot/error.h"

#include <string.h>

#include "backends/cpu/cpu_caps.h"
#include "core/dispatch/fusion_flags.h"

extern marmot_kernel_selection_t
marmot_cpu_query_kernel(const marmot_op_signature_t *sig, const marmot_device_caps_t *caps);
extern bool marmot_cpu_default_preferences(const marmot_device_caps_t *caps, marmot_backend_preferences_t *out);

#if MARMOT_ENABLE_METAL
#include "backends/metal/internal/metal_kernel_query.gen.h"
extern marmot_device_caps_t marmot_metal_detect_default_caps(void);
extern bool marmot_metal_default_preferences(const marmot_device_caps_t *caps, marmot_backend_preferences_t *out);
#endif

static marmot_kernel_selection_t marmot_kernel_selection_unsupported(const char *reason) {
    return (marmot_kernel_selection_t){
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

static marmot_kernel_selection_t marmot_backend_query_kernel_impl(
    marmot_backend_type_t backend_type, const marmot_op_signature_t *sig, const marmot_device_caps_t *caps
) {
    switch (backend_type) {
    case MARMOT_BACKEND_CPU:
        return marmot_cpu_query_kernel(sig, caps);
    case MARMOT_BACKEND_METAL:
#if MARMOT_ENABLE_METAL
        return marmot_metal_query_kernel(sig, caps);
#else
        (void)sig;
        (void)caps;
        return marmot_kernel_selection_unsupported("Metal backend not available");
#endif
    default:
        return marmot_kernel_selection_unsupported("Backend query not available");
    }
}

marmot_kernel_selection_t marmot_backend_query_kernel_with_fallback(
    marmot_backend_type_t backend_type, const marmot_op_signature_t *sig, const marmot_device_caps_t *caps,
    marmot_op_signature_t *resolved_sig_out
) {
    if (sig == nullptr || caps == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null signature or caps for backend kernel query");
        return marmot_kernel_selection_unsupported("Invalid kernel query arguments");
    }

    marmot_kernel_selection_t sel = marmot_backend_query_kernel_impl(backend_type, sig, caps);
    if (!sel.supported && sig->variant_flags != MARMOT_FUSION_NONE) {
        marmot_op_signature_t unfused_sig = *sig;
        unfused_sig.variant_flags = MARMOT_FUSION_NONE;
        marmot_kernel_selection_t unfused_sel = marmot_backend_query_kernel_impl(backend_type, &unfused_sig, caps);
        if (unfused_sel.supported) {
            if (resolved_sig_out != nullptr) {
                *resolved_sig_out = unfused_sig;
            }
            return unfused_sel;
        }
    }

    if (resolved_sig_out != nullptr) {
        *resolved_sig_out = *sig;
    }
    return sel;
}

marmot_kernel_selection_t marmot_backend_query_kernel(
    marmot_backend_type_t backend_type, const marmot_op_signature_t *sig, const marmot_device_caps_t *caps
) {
    return marmot_backend_query_kernel_with_fallback(backend_type, sig, caps, nullptr);
}

bool marmot_backend_detect_default_caps(marmot_backend_type_t backend_type, marmot_device_caps_t *caps_out) {
    if (caps_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null output buffer for backend caps");
        return false;
    }

    memset(caps_out, 0, sizeof(*caps_out));
    switch (backend_type) {
    case MARMOT_BACKEND_CPU:
        *caps_out = marmot_cpu_detect_capabilities();
        return true;
    case MARMOT_BACKEND_METAL:
#if MARMOT_ENABLE_METAL
        *caps_out = marmot_metal_detect_default_caps();
        return true;
#else
        return false;
#endif
    default:
        return false;
    }
}

bool marmot_backend_get_default_preferences(
    marmot_backend_type_t backend_type, const marmot_device_caps_t *caps, marmot_backend_preferences_t *prefs_out
) {
    if (prefs_out == nullptr || caps == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null arguments for backend preferences");
        return false;
    }

    memset(prefs_out, 0, sizeof(*prefs_out));
    switch (backend_type) {
    case MARMOT_BACKEND_CPU:
        return marmot_cpu_default_preferences(caps, prefs_out);
    case MARMOT_BACKEND_METAL:
#if MARMOT_ENABLE_METAL
        return marmot_metal_default_preferences(caps, prefs_out);
#else
        return false;
#endif
    default:
        return false;
    }
}
