#include "marmot/allocator.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>

#include "backends/cpu/cpu_backend_internal.h"
#include "backends/cpu/cpu_caps.h"
#include "context_internal.h"
#include "core/dispatch/kernel_query.h"

static const char *marmot_backend_name_from_type(marmot_backend_type_t backend) {
    switch (backend) {
    case MARMOT_BACKEND_CPU:
        return "cpu";
    case MARMOT_BACKEND_METAL:
        return "metal";
    case MARMOT_BACKEND_CUDA:
        return "cuda";
    default:
        return "unknown";
    }
}

typedef struct {
    const marmot_context_t *ctx;
    size_t depth;
} marmot_graph_batch_state_entry_t;

static thread_local marmot_graph_batch_state_entry_t g_graph_batch_state[8];
static thread_local size_t g_graph_batch_state_count = 0;

static marmot_graph_batch_state_entry_t *marmot_graph_batch_state_find(const marmot_context_t *ctx) {
    if (ctx == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; i < g_graph_batch_state_count; ++i) {
        if (g_graph_batch_state[i].ctx == ctx) {
            return &g_graph_batch_state[i];
        }
    }
    return nullptr;
}

static void marmot_graph_batch_state_increment(const marmot_context_t *ctx) {
    marmot_graph_batch_state_entry_t *entry = marmot_graph_batch_state_find(ctx);
    if (entry != nullptr) {
        entry->depth++;
        return;
    }
    if (g_graph_batch_state_count >= (sizeof(g_graph_batch_state) / sizeof(g_graph_batch_state[0]))) {
        return;
    }
    g_graph_batch_state[g_graph_batch_state_count++] = (marmot_graph_batch_state_entry_t){
        .ctx = ctx,
        .depth = 1,
    };
}

static void marmot_graph_batch_state_decrement(const marmot_context_t *ctx) {
    marmot_graph_batch_state_entry_t *entry = marmot_graph_batch_state_find(ctx);
    if (entry == nullptr) {
        return;
    }
    if (entry->depth > 1) {
        entry->depth--;
        return;
    }

    const size_t idx = (size_t)(entry - g_graph_batch_state);
    if (idx >= g_graph_batch_state_count) {
        return;
    }
    g_graph_batch_state_count--;
    if (idx != g_graph_batch_state_count) {
        g_graph_batch_state[idx] = g_graph_batch_state[g_graph_batch_state_count];
    }
    g_graph_batch_state[g_graph_batch_state_count] = (marmot_graph_batch_state_entry_t){0};
}

marmot_context_t *marmot_init(marmot_backend_type_t backend) {
    const marmot_device_ops_t *ops = nullptr;

    switch (backend) {
    case MARMOT_BACKEND_CPU:
        ops = marmot_get_cpu_ops();
        break;

#if defined(__APPLE__) && MARMOT_ENABLE_METAL
    case MARMOT_BACKEND_METAL:
        ops = marmot_get_metal_ops();
        break;
#elif defined(__APPLE__)
    case MARMOT_BACKEND_METAL:
        marmot_set_error(MARMOT_ERROR_DEVICE_NOT_AVAILABLE, "Metal backend not built in this configuration");
        return nullptr;
#endif

    case MARMOT_BACKEND_CUDA:
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "CUDA backend not yet implemented");
        return nullptr;

    default:
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unknown backend type");
        return nullptr;
    }

    if (ops == nullptr) {
        marmot_set_error(MARMOT_ERROR_DEVICE_NOT_AVAILABLE, "Backend ops not available");
        return nullptr;
    }

    marmot_context_t *ctx = calloc(1, sizeof(marmot_context_t));
    if (ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate context");
        return nullptr;
    }

    ctx->backend_type = backend;
    ctx->ops = ops;

    if (!marmot_backend_detect_default_caps(backend, &ctx->device_caps)) {
        if (backend == MARMOT_BACKEND_CPU) {
            ctx->device_caps = marmot_cpu_detect_capabilities();
        } else {
            ctx->device_caps = (marmot_device_caps_t){
                .peak_flops_tflops_fp32 = 1.0f,
                .peak_flops_tflops_fp16 = 1.0f,
                .mem_bw_gbps = 50.0f,
                .launch_overhead_us = 5.0f,
                .edge_penalty_alpha = 0.0f,
            };
        }
    }

    if (backend == MARMOT_BACKEND_CPU) {
        ctx->best_profile = marmot_cpu_detect_best_profile();
    } else {
        ctx->best_profile = MARMOT_PROFILE_INVALID;
    }

    marmot_context_apply_default_policy(ctx);

    if (ops->init != nullptr) {
        marmot_error_t err = ops->init(&ctx->device_ctx);
        if (err != MARMOT_SUCCESS) {
            free(ctx);
            marmot_set_error(err, "Backend initialization failed");
            return nullptr;
        }
    }

    if (ops->configure != nullptr) {
        marmot_error_t cfg_err = ops->configure(ctx->device_ctx, ctx);
        if (cfg_err != MARMOT_SUCCESS) {
            if (ops->destroy != nullptr && ctx->device_ctx != nullptr) {
                ops->destroy(ctx->device_ctx);
            }
            free(ctx);
            marmot_set_error(cfg_err, "Backend configuration failed");
            return nullptr;
        }
    }

    return ctx;
}

void marmot_destroy(marmot_context_t *ctx) {
    if (ctx == nullptr) {
        return;
    }

    const char *debug_alloc = getenv("MARMOT_DEBUG_ALLOCATOR");
    if (debug_alloc != nullptr && ctx->ops != nullptr && ctx->ops->allocator_usage != nullptr &&
        ctx->device_ctx != nullptr) {
        marmot_allocator_usage_t usage = {0};
        if (ctx->ops->allocator_usage(ctx->device_ctx, &usage) == MARMOT_SUCCESS) {
            fprintf(
                stderr, "[marmot] allocator[%s]: current=%zub peak=%zub\n",
                marmot_backend_name_from_type(ctx->backend_type), usage.current_bytes, usage.peak_bytes
            );
        }
    }

    if (ctx->ops != nullptr && ctx->ops->destroy != nullptr && ctx->device_ctx != nullptr) {
        ctx->ops->destroy(ctx->device_ctx);
    }

    free(ctx);
}

static marmot_error_t
marmot_context_set_thread_count_impl(marmot_context_t *ctx, size_t num_threads, bool explicit_override) {
    if (ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null context passed to set thread count");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (num_threads == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Thread count must be greater than zero");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (ctx->backend_type != MARMOT_BACKEND_CPU || ctx->device_ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Runtime thread count control is only supported on CPU");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    marmot_error_t err = cpu_context_set_num_threads(ctx->device_ctx, num_threads, explicit_override);
    if (err != MARMOT_SUCCESS) {
        marmot_set_error(err, "Failed to update CPU thread count");
    }
    return err;
}

marmot_error_t marmot_context_set_thread_count(marmot_context_t *ctx, size_t num_threads) {
    return marmot_context_set_thread_count_impl(ctx, num_threads, true);
}

marmot_error_t marmot_context_set_thread_count_auto(marmot_context_t *ctx, size_t num_threads) {
    return marmot_context_set_thread_count_impl(ctx, num_threads, false);
}

size_t marmot_context_get_thread_count(const marmot_context_t *ctx) {
    if (ctx == nullptr || ctx->backend_type != MARMOT_BACKEND_CPU || ctx->device_ctx == nullptr) {
        return 0;
    }
    return cpu_context_get_num_threads(ctx->device_ctx);
}

bool marmot_context_thread_count_is_explicit(const marmot_context_t *ctx) {
    if (ctx == nullptr || ctx->backend_type != MARMOT_BACKEND_CPU || ctx->device_ctx == nullptr) {
        return false;
    }
    return cpu_context_thread_count_is_explicit(ctx->device_ctx);
}

marmot_error_t marmot_device_synchronize(const marmot_context_t *ctx) {
    if (ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null context passed to device synchronize");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (ctx->ops == nullptr) {
        marmot_set_error(MARMOT_ERROR_DEVICE_NOT_AVAILABLE, "Context missing device operations");
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
    }

    if (ctx->ops->synchronize == nullptr) {
        return MARMOT_SUCCESS;
    }

    marmot_error_t err = ctx->ops->synchronize(ctx->device_ctx);
    if (err != MARMOT_SUCCESS) {
        marmot_set_error(err, "Backend synchronization failed");
    }
    return err;
}

marmot_error_t marmot_graph_batch_begin(const marmot_context_t *ctx) {
    if (ctx == nullptr || ctx->ops == nullptr) {
        return MARMOT_SUCCESS; // No-op for null context
    }
    if (ctx->ops->graph_batch_begin == nullptr) {
        return MARMOT_SUCCESS; // Backend doesn't support batching
    }
    marmot_error_t err = ctx->ops->graph_batch_begin(ctx->device_ctx);
    if (err == MARMOT_SUCCESS) {
        marmot_graph_batch_state_increment(ctx);
    }
    return err;
}

marmot_error_t marmot_graph_batch_end(const marmot_context_t *ctx, bool commit) {
    if (ctx == nullptr || ctx->ops == nullptr) {
        return MARMOT_SUCCESS;
    }
    if (ctx->ops->graph_batch_end == nullptr) {
        return MARMOT_SUCCESS;
    }
    marmot_error_t err = ctx->ops->graph_batch_end(ctx->device_ctx, commit);
    if (err == MARMOT_SUCCESS) {
        marmot_graph_batch_state_decrement(ctx);
    }
    return err;
}

bool marmot_graph_batch_is_active(const marmot_context_t *ctx) {
    marmot_graph_batch_state_entry_t *entry = marmot_graph_batch_state_find(ctx);
    return entry != nullptr && entry->depth > 0;
}

void marmot_context_set_routing_policy(marmot_context_t *ctx, marmot_routing_policy_t policy) {
    if (ctx == nullptr) {
        return;
    }
    ctx->routing_policy = policy;
    if (ctx->ops != nullptr && ctx->ops->configure != nullptr && ctx->device_ctx != nullptr) {
        ctx->ops->configure(ctx->device_ctx, ctx);
    }
}

marmot_routing_policy_t marmot_context_get_routing_policy(const marmot_context_t *ctx) {
    if (ctx == nullptr) {
        return MARMOT_ROUTING_AUTO;
    }
    return ctx->routing_policy;
}

void marmot_context_set_quant_activation_mode(marmot_context_t *ctx, marmot_quant_activation_mode_t activation_mode) {
    if (ctx == nullptr) {
        return;
    }
    ctx->policy.quant_activation_mode = activation_mode;
    if (ctx->ops != nullptr && ctx->ops->configure != nullptr && ctx->device_ctx != nullptr) {
        ctx->ops->configure(ctx->device_ctx, ctx);
    }
}

marmot_quant_activation_mode_t marmot_context_get_quant_activation_mode(const marmot_context_t *ctx) {
    if (ctx == nullptr) {
        return MARMOT_QUANT_ACTIVATION_AUTO;
    }
    return ctx->policy.quant_activation_mode;
}

void marmot_context_set_policy(marmot_context_t *ctx, const marmot_context_policy_t *policy) {
    if (ctx == nullptr || policy == nullptr) {
        return;
    }
    ctx->policy = *policy;
    if (ctx->ops != nullptr && ctx->ops->configure != nullptr && ctx->device_ctx != nullptr) {
        ctx->ops->configure(ctx->device_ctx, ctx);
    }
}

const marmot_context_policy_t *marmot_context_get_policy(const marmot_context_t *ctx) {
    if (ctx == nullptr) {
        return nullptr;
    }
    return &ctx->policy;
}

marmot_backend_type_t marmot_context_get_backend(const marmot_context_t *ctx) {
    if (ctx == nullptr) {
        return MARMOT_BACKEND_CPU;
    }
    return ctx->backend_type;
}

const char *marmot_context_get_backend_name(const marmot_context_t *ctx) {
    if (ctx == nullptr) {
        return "unknown";
    }
    return marmot_backend_name_from_type(ctx->backend_type);
}

bool marmot_context_supports_dtype(const marmot_context_t *ctx, marmot_dtype_t dtype) {
    if (ctx == nullptr) {
        return false;
    }
    return marmot_dtype_is_supported_on_backend(ctx->backend_type, dtype);
}
