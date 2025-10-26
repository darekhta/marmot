#ifndef MARMOT_CORE_DISPATCH_KERNEL_QUERY_H
#define MARMOT_CORE_DISPATCH_KERNEL_QUERY_H

#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/graph/kernel_selection.h"
#include "marmot/graph/op_signature.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    marmot_context_policy_t policy;
    marmot_routing_policy_t routing_policy;
} marmot_backend_preferences_t;

marmot_kernel_selection_t marmot_backend_query_kernel_with_fallback(
    marmot_backend_type_t backend_type, const marmot_op_signature_t *sig, const marmot_device_caps_t *caps,
    marmot_op_signature_t *resolved_sig_out
);

marmot_kernel_selection_t marmot_backend_query_kernel(
    marmot_backend_type_t backend_type, const marmot_op_signature_t *sig, const marmot_device_caps_t *caps
);

bool marmot_backend_detect_default_caps(marmot_backend_type_t backend_type, marmot_device_caps_t *caps_out);

bool marmot_backend_get_default_preferences(
    marmot_backend_type_t backend_type, const marmot_device_caps_t *caps, marmot_backend_preferences_t *prefs_out
);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_CORE_DISPATCH_KERNEL_QUERY_H
