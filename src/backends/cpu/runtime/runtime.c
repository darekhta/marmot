#include <unistd.h>

#include "core/dispatch/kernel_query.h"
#include "cpu_backend_internal.h"
#include "cpu_caps.h"

size_t cpu_default_thread_count(void) {
    // Cache the result since CPU topology detection can be expensive
    // (especially Intel hybrid detection which requires thread affinity)
    static size_t cached_count = 0;
    if (cached_count == 0) {
        marmot_device_caps_t caps = marmot_cpu_detect_capabilities();
        cached_count = marmot_cpu_optimal_thread_count(&caps);
    }
    return cached_count;
}

static cpu_capabilities_t cpu_capabilities_clamp(const cpu_capabilities_t *requested) {
    const cpu_capabilities_t *compiled = cpu_compiled_capabilities();
    cpu_capabilities_t effective = *compiled;
    if (requested != nullptr) {
        effective.has_neon = requested->has_neon && compiled->has_neon;
        effective.has_avx2 = requested->has_avx2 && compiled->has_avx2;
        effective.has_f16c = requested->has_f16c && compiled->has_f16c;
        effective.has_accelerate = requested->has_accelerate && compiled->has_accelerate;
    }
    return effective;
}

static void cpu_context_apply_capabilities(cpu_context_t *ctx, const cpu_capabilities_t *requested) {
    if (ctx == nullptr) {
        return;
    }
    cpu_capabilities_t effective = cpu_capabilities_clamp(requested);
    ctx->runtime_caps = effective;
}

void cpu_context_use_compiled_capabilities(const void *device_ctx) {
    cpu_context_apply_capabilities((cpu_context_t *)device_ctx, cpu_compiled_capabilities());
}

void cpu_context_force_scalar(const void *device_ctx) {
    cpu_capabilities_t scalar_caps = {
        .has_neon = false,
        .has_avx2 = false,
        .has_f16c = false,
        .has_accelerate = false,
    };
    cpu_context_apply_capabilities((cpu_context_t *)device_ctx, &scalar_caps);
}

static bool cpu_quant_env_force_q8(void) {
    const char *env = getenv("MARMOT_QUANT_CPU_ACT");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }
    return env[0] == 'q' || env[0] == 'Q';
}

bool marmot_cpu_default_preferences(const marmot_device_caps_t *caps, marmot_backend_preferences_t *out) {
    if (out == nullptr) {
        return false;
    }
    (void)caps;
    *out = (marmot_backend_preferences_t){
        .policy =
            {
                .embedding_quant_output_dtype = MARMOT_DTYPE_FLOAT32,
                .quant_activation_mode = MARMOT_QUANT_ACTIVATION_AUTO,
                .variant_flags_mask = UINT32_MAX,
                .embedding_prefer_gpu_private = false,
                .embedding_allow_quant_decode_on_the_fly = true,
                .matmul_requires_temp_tensors = false,
                .matmul_prefer_packed_weights = false,
            },
        .routing_policy = MARMOT_ROUTING_ALWAYS_CPU,
    };
    return true;
}

marmot_error_t cpu_init(void **device_ctx) {
    cpu_context_t *ctx = calloc(1, sizeof(cpu_context_t));
    if (ctx == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    ctx->num_threads = cpu_default_thread_count();
    ctx->quant_activation_mode = MARMOT_QUANT_ACTIVATION_AUTO;
    ctx->force_q8_activations = cpu_quant_env_force_q8();
    marmot_rope_freq_cache_init(&ctx->rope_cache);
    cpu_rope_sincos_cache_init(&ctx->rope_sincos_cache);

    cpu_context_use_compiled_capabilities(ctx);
    ctx->allocator_ops = marmot_get_allocator(MARMOT_BACKEND_CPU);
    pthread_mutex_init(&ctx->allocator_tracker.mutex, nullptr);
    ctx->allocator_tracker.head = nullptr;

    // Initialize GEMM scratch buffer pool
    marmot_neon_scratch_pool_init(&ctx->neon_scratch_pool, ctx->num_threads);

    void *probe = nullptr;
    if (cpu_alloc(ctx, 1, &probe) == MARMOT_SUCCESS) {
        cpu_free(ctx, probe);
    }

    ctx->initialized = 1;
    *device_ctx = ctx;
    return MARMOT_SUCCESS;
}

void cpu_destroy(const void *device_ctx) {
    if (device_ctx == nullptr) {
        return;
    }

    cpu_context_t *ctx = (cpu_context_t *)device_ctx;
    if (ctx->allocator_ops == nullptr) {
        ctx->allocator_ops = marmot_get_allocator(MARMOT_BACKEND_CPU);
    }

    pthread_mutex_lock(&ctx->allocator_tracker.mutex);
    cpu_allocation_entry_t *entry = ctx->allocator_tracker.head;
    while (entry != nullptr) {
        cpu_allocation_entry_t *next = entry->next;
        if (ctx->allocator_ops != nullptr) {
            ctx->allocator_ops->free(ctx, &entry->info);
        } else {
            free(entry->info.ptr);
        }
        free(entry);
        entry = next;
    }
    pthread_mutex_unlock(&ctx->allocator_tracker.mutex);
    pthread_mutex_destroy(&ctx->allocator_tracker.mutex);

    marmot_rope_freq_cache_destroy(&ctx->rope_cache);
    cpu_rope_sincos_cache_destroy(&ctx->rope_sincos_cache);

    // Free RoPE scratch buffer
    free(ctx->rope_positions_scratch);
    ctx->rope_positions_scratch = nullptr;
    ctx->rope_positions_capacity = 0;

    // Free GEMM scratch buffer pool
    marmot_neon_scratch_pool_destroy(&ctx->neon_scratch_pool);

    free(ctx);
}

marmot_error_t cpu_configure(const void *device_ctx, const marmot_context_t *ctx) {
    if (device_ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    cpu_context_t *cpu_ctx = (cpu_context_t *)device_ctx;
    cpu_ctx->quant_activation_mode = ctx != nullptr ? ctx->policy.quant_activation_mode : MARMOT_QUANT_ACTIVATION_AUTO;
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_synchronize(const void *device_ctx) {
    (void)device_ctx;
    return MARMOT_SUCCESS;
}

int32_t *cpu_get_rope_positions_scratch(cpu_context_t *ctx, size_t seq_len) {
    if (ctx == nullptr || seq_len == 0) {
        return nullptr;
    }

    // Grow buffer if needed
    if (ctx->rope_positions_capacity < seq_len) {
        free(ctx->rope_positions_scratch);
        ctx->rope_positions_scratch = malloc(seq_len * sizeof(int32_t));
        if (ctx->rope_positions_scratch == nullptr) {
            ctx->rope_positions_capacity = 0;
            return nullptr;
        }
        ctx->rope_positions_capacity = seq_len;
    }

    return ctx->rope_positions_scratch;
}
