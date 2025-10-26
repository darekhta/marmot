#ifndef MARMOT_DEVICE_H
#define MARMOT_DEVICE_H

#include "allocator.h"
#include "config.h"
#include "device_caps.h"
#include "ops_types.h"
#include "tensor.h"
#include "traits_ids.gen.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Routing policy
//------------------------------------------------------------------------------

typedef enum {
    MARMOT_QUANT_ACTIVATION_AUTO = 0,
    MARMOT_QUANT_ACTIVATION_FORCE_DIRECT = 1,
} marmot_quant_activation_mode_t;

//------------------------------------------------------------------------------
// Context policy (backend-dependent defaults)
//------------------------------------------------------------------------------

typedef struct {
    marmot_dtype_t embedding_quant_output_dtype;
    marmot_quant_activation_mode_t quant_activation_mode;
    uint32_t variant_flags_mask;
    bool embedding_prefer_gpu_private;
    bool embedding_allow_quant_decode_on_the_fly;
    bool matmul_requires_temp_tensors;
    bool matmul_prefer_packed_weights;
} marmot_context_policy_t;

//------------------------------------------------------------------------------
// Device operations table (public backend contract)
//------------------------------------------------------------------------------

struct marmot_device_ops {
    // Lifecycle
    marmot_error_t (*init)(void **device_ctx);
    void (*destroy)(const void *device_ctx);
    marmot_error_t (*configure)(const void *device_ctx, const marmot_context_t *ctx);

    // Memory management
    marmot_error_t (*alloc)(const void *device_ctx, size_t size, void **ptr);
    void (*free)(const void *device_ctx, void *ptr);
    marmot_error_t (*memcpy_to_device)(const void *device_ctx, void *dst, const void *src, size_t size);
    marmot_error_t (*memcpy_from_device)(const void *device_ctx, void *dst, const void *src, size_t size);
    marmot_error_t (*synchronize)(const void *device_ctx);
    marmot_error_t (*allocator_usage)(const void *device_ctx, marmot_allocator_usage_t *usage);

    // Graph execution hooks (optional - nullptr means no-op)
    // Used by graph executor to batch multiple operations into a single submission
    marmot_error_t (*graph_batch_begin)(void *device_ctx);
    marmot_error_t (*graph_batch_end)(void *device_ctx, bool commit);

    // Resource cleanup hooks (optional - nullptr means no-op)
    // Called when host memory is about to be freed, allows backend cache invalidation
    void (*on_host_ptr_freed)(void *device_ctx, const void *ptr);
    void (*on_host_range_freed)(void *device_ctx, const void *start, size_t length);
};

//------------------------------------------------------------------------------
// Runtime context
//------------------------------------------------------------------------------

struct marmot_context {
    marmot_backend_type_t backend_type;
    const marmot_device_ops_t *ops;
    void *device_ctx;
    marmot_routing_policy_t routing_policy;
    marmot_context_policy_t policy;
    marmot_profile_id_t best_profile;
    marmot_device_caps_t device_caps;
};

//------------------------------------------------------------------------------
// Context management helpers
//------------------------------------------------------------------------------
MARMOT_NODISCARD marmot_context_t *marmot_init(marmot_backend_type_t backend);
void marmot_destroy(marmot_context_t *ctx);

void marmot_context_set_routing_policy(marmot_context_t *ctx, marmot_routing_policy_t policy);
marmot_routing_policy_t marmot_context_get_routing_policy(const marmot_context_t *ctx);
void marmot_context_set_quant_activation_mode(marmot_context_t *ctx, marmot_quant_activation_mode_t activation_mode);
marmot_quant_activation_mode_t marmot_context_get_quant_activation_mode(const marmot_context_t *ctx);
void marmot_context_set_policy(marmot_context_t *ctx, const marmot_context_policy_t *policy);
MARMOT_NODISCARD const marmot_context_policy_t *marmot_context_get_policy(const marmot_context_t *ctx);
MARMOT_NODISCARD marmot_backend_type_t marmot_context_get_backend(const marmot_context_t *ctx);
MARMOT_NODISCARD const char *marmot_context_get_backend_name(const marmot_context_t *ctx);
MARMOT_NODISCARD bool marmot_context_supports_dtype(const marmot_context_t *ctx, marmot_dtype_t dtype);

//------------------------------------------------------------------------------
// Backend synchronization helpers
//------------------------------------------------------------------------------

MARMOT_NODISCARD marmot_error_t marmot_device_synchronize(const marmot_context_t *ctx);

//------------------------------------------------------------------------------
// Graph execution batching (used by graph executor, safe to call on any backend)
//------------------------------------------------------------------------------

MARMOT_NODISCARD marmot_error_t marmot_graph_batch_begin(const marmot_context_t *ctx);
MARMOT_NODISCARD marmot_error_t marmot_graph_batch_end(const marmot_context_t *ctx, bool commit);
MARMOT_NODISCARD bool marmot_graph_batch_is_active(const marmot_context_t *ctx);

MARMOT_NODISCARD const marmot_device_ops_t *marmot_get_cpu_ops(void);

#ifdef __APPLE__
MARMOT_NODISCARD const marmot_device_ops_t *marmot_get_metal_ops(void);
#endif

#ifdef __cplusplus
}
#endif

#endif
