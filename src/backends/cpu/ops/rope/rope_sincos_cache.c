#include <math.h>
#include <string.h>

#include "cpu_backend_internal.h"

static void cpu_rope_sincos_cache_set_params(
    cpu_rope_sincos_cache_t *cache, const marmot_rope_freq_cache_t *freq_cache, size_t dim, size_t pair_count
) {
    cache->dim = dim;
    cache->pair_count = pair_count;
    cache->theta = freq_cache->theta;
    cache->freq_scale = freq_cache->freq_scale;
    cache->ext_factor = freq_cache->ext_factor;
    cache->attn_factor = freq_cache->attn_factor;
    cache->beta_fast = freq_cache->beta_fast;
    cache->beta_slow = freq_cache->beta_slow;
    cache->orig_ctx_len = freq_cache->orig_ctx_len;
    cache->scaling_type = freq_cache->scaling_type;
    cache->attn_scale = freq_cache->attn_scale;
}

static bool cpu_rope_sincos_cache_matches(
    const cpu_rope_sincos_cache_t *cache, const marmot_rope_freq_cache_t *freq_cache, size_t dim, size_t pair_count
) {
    return cache->sincos != nullptr && cache->dim == dim && cache->pair_count == pair_count &&
        cache->theta == freq_cache->theta && cache->freq_scale == freq_cache->freq_scale &&
        cache->ext_factor == freq_cache->ext_factor && cache->attn_factor == freq_cache->attn_factor &&
        cache->beta_fast == freq_cache->beta_fast && cache->beta_slow == freq_cache->beta_slow &&
        cache->orig_ctx_len == freq_cache->orig_ctx_len && cache->scaling_type == freq_cache->scaling_type &&
        cache->attn_scale == freq_cache->attn_scale;
}

static bool cpu_rope_positions_max(const marmot_tensor_t *positions, size_t count, size_t *max_pos_out) {
    if (positions == nullptr || max_pos_out == nullptr) {
        return false;
    }

    int64_t max_pos = 0;
    if (positions->dtype == MARMOT_DTYPE_INT32) {
        const int32_t *pos_data = (const int32_t *)positions->data;
        for (size_t i = 0; i < count; ++i) {
            const int32_t pos = pos_data[i];
            if (pos < 0) {
                return false;
            }
            if (pos > max_pos) {
                max_pos = pos;
            }
        }
    } else if (positions->dtype == MARMOT_DTYPE_INT64) {
        const int64_t *pos_data = (const int64_t *)positions->data;
        for (size_t i = 0; i < count; ++i) {
            const int64_t pos = pos_data[i];
            if (pos < 0) {
                return false;
            }
            if (pos > max_pos) {
                max_pos = pos;
            }
        }
    } else {
        return false;
    }

    if ((uint64_t)max_pos > (uint64_t)SIZE_MAX - 1u) {
        return false;
    }

    *max_pos_out = (size_t)max_pos;
    return true;
}

void cpu_rope_sincos_cache_init(cpu_rope_sincos_cache_t *cache) {
    if (cache == nullptr) {
        return;
    }
    memset(cache, 0, sizeof(*cache));
    cache->theta = 0.0f;
    cache->freq_scale = 1.0f;
    cache->ext_factor = 0.0f;
    cache->attn_factor = 1.0f;
    cache->beta_fast = 0.0f;
    cache->beta_slow = 0.0f;
    cache->orig_ctx_len = 0;
    cache->scaling_type = MARMOT_ROPE_SCALING_NONE;
    cache->attn_scale = 1.0f;
}

void cpu_rope_sincos_cache_reset(cpu_rope_sincos_cache_t *cache) {
    if (cache == nullptr) {
        return;
    }
    if (cache->owns_storage && cache->sincos != nullptr) {
        free(cache->sincos);
    }
    cache->sincos = nullptr;
    cache->capacity_positions = 0;
    cache->cached_positions = 0;
    cache->pair_count = 0;
    cache->dim = 0;
    cache->theta = 0.0f;
    cache->freq_scale = 1.0f;
    cache->ext_factor = 0.0f;
    cache->attn_factor = 1.0f;
    cache->beta_fast = 0.0f;
    cache->beta_slow = 0.0f;
    cache->orig_ctx_len = 0;
    cache->scaling_type = MARMOT_ROPE_SCALING_NONE;
    cache->attn_scale = 1.0f;
    cache->owns_storage = false;
}

void cpu_rope_sincos_cache_destroy(cpu_rope_sincos_cache_t *cache) {
    cpu_rope_sincos_cache_reset(cache);
}

marmot_error_t cpu_rope_sincos_cache_ensure(
    cpu_context_t *ctx, const marmot_rope_freq_span_t *span, const marmot_tensor_t *positions, size_t count,
    bool *out_use_cache
) {
    if (out_use_cache != nullptr) {
        *out_use_cache = false;
    }
    if (ctx == nullptr || span == nullptr || positions == nullptr || span->freqs == nullptr || span->dim == 0 ||
        count == 0) {
        return MARMOT_SUCCESS;
    }

    size_t max_pos = 0;
    if (!cpu_rope_positions_max(positions, count, &max_pos)) {
        return MARMOT_SUCCESS;
    }

    const size_t dim = span->dim;
    if ((dim & 1u) != 0u) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE sincos cache requires even dimensions");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const size_t pair_count = dim / 2;
    if (pair_count == 0) {
        return MARMOT_SUCCESS;
    }

    const marmot_rope_freq_cache_t *freq_cache = &ctx->rope_cache;
    cpu_rope_sincos_cache_t *cache = &ctx->rope_sincos_cache;

    if (!cpu_rope_sincos_cache_matches(cache, freq_cache, dim, pair_count)) {
        cpu_rope_sincos_cache_reset(cache);
        cpu_rope_sincos_cache_set_params(cache, freq_cache, dim, pair_count);
    }

    const size_t required_positions = max_pos + 1u;
    const size_t stride = pair_count * 2u;

    if (cache->sincos == nullptr || cache->capacity_positions < required_positions) {
        if (stride == 0 || required_positions > SIZE_MAX / stride / sizeof(float)) {
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "RoPE sincos cache size overflow");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        const size_t new_size = required_positions * stride;
        float *replacement = (float *)realloc(cache->sincos, new_size * sizeof(float));
        if (replacement == nullptr) {
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to grow RoPE sincos cache");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        cache->sincos = replacement;
        cache->capacity_positions = required_positions;
        cache->owns_storage = true;
    }

    if (cache->cached_positions < required_positions) {
        for (size_t pos = cache->cached_positions; pos < required_positions; ++pos) {
            float *dst = cache->sincos + pos * stride;
            const float pos_f = (float)pos;
            for (size_t i = 0; i < pair_count; ++i) {
                const float angle = pos_f * span->freqs[i];
                float sin_theta = 0.0f;
                float cos_theta = 0.0f;
                cpu_sincosf(angle, &sin_theta, &cos_theta);
                dst[2 * i] = cos_theta * span->attn_scale;
                dst[2 * i + 1] = sin_theta * span->attn_scale;
            }
        }
        cache->cached_positions = required_positions;
    }

    if (out_use_cache != nullptr) {
        *out_use_cache = true;
    }

    return MARMOT_SUCCESS;
}
