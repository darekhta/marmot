#include "rope.h"

#include "marmot/error.h"
#include "marmot/ops_types.h"

#include <stdint.h>
#include <stdlib.h>

#include <math.h>

typedef struct {
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float attn_scale;
    float corr_low;
    float corr_high;
} rope_scaling_state_t;

static constexpr marmot_rope_freq_cache_t kDefaultRopeCache = {
    .freq_scale = 1.0f,
    .attn_factor = 1.0f,
    .scaling_type = MARMOT_ROPE_SCALING_NONE,
    .attn_scale = 1.0f,
};

static constexpr float kMarmotPi = 3.14159265358979323846f;

static float rope_yarn_ramp(float low, float high, float pair_index) {
    float denom = high - low;
    if (denom < 0.001f) {
        denom = 0.001f;
    }
    const float y = (pair_index - low) / denom;
    const float clamped = fminf(1.0f, fmaxf(0.0f, y));
    return 1.0f - clamped;
}

static float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    const float denom = 2.0f * logf(base);
    if (denom == 0.0f || n_rot <= 0.0f || n_ctx_orig <= 0) {
        return 0.0f;
    }
    return (float)n_dims * logf((float)n_ctx_orig / (n_rot * 2.0f * kMarmotPi)) / denom;
}

static void rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float *low, float *high
) {
    float low_raw = rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base);
    float high_raw = rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base);
    if (low != nullptr) {
        *low = fmaxf(0.0f, floorf(low_raw));
    }
    if (high != nullptr) {
        *high = fminf((float)(n_dims - 1), ceilf(high_raw));
    }
}

static void rope_scaling_prepare(const marmot_rope_params_t *params, size_t dim, rope_scaling_state_t *state) {
    float freq_scale = params != nullptr ? params->freq_scale : 1.0f;
    float ext_factor = params != nullptr ? params->ext_factor : 0.0f;
    float attn_factor = params != nullptr ? params->attn_factor : 1.0f;
    marmot_rope_scaling_type_t scaling_type = params != nullptr ? params->scaling_type : MARMOT_ROPE_SCALING_NONE;

    if (freq_scale <= 0.0f) {
        freq_scale = 1.0f;
    }
    if (attn_factor <= 0.0f) {
        attn_factor = 1.0f;
    }

    if (scaling_type == MARMOT_ROPE_SCALING_NONE || scaling_type == MARMOT_ROPE_SCALING_UNSPECIFIED) {
        freq_scale = 1.0f;
        ext_factor = 0.0f;
    } else if (scaling_type == MARMOT_ROPE_SCALING_LINEAR || scaling_type == MARMOT_ROPE_SCALING_LONGROPE) {
        ext_factor = 0.0f;
    }

    float corr_low = 0.0f;
    float corr_high = 0.0f;
    if (ext_factor != 0.0f && params != nullptr && params->orig_ctx_len > 0 && params->theta > 0.0f &&
        params->beta_fast > 0.0f && params->beta_slow > 0.0f) {
        rope_yarn_corr_dims(
            (int)dim, (int)params->orig_ctx_len, params->theta, params->beta_fast, params->beta_slow, &corr_low,
            &corr_high
        );
    } else {
        ext_factor = 0.0f;
    }

    float attn_scale = attn_factor;
    if (ext_factor != 0.0f) {
        const float inv_scale = 1.0f / freq_scale;
        attn_scale *= 1.0f + 0.1f * logf(inv_scale);
    }

    if (state != nullptr) {
        state->freq_scale = freq_scale;
        state->ext_factor = ext_factor;
        state->attn_factor = attn_factor;
        state->attn_scale = attn_scale;
        state->corr_low = corr_low;
        state->corr_high = corr_high;
    }
}

static void
compute_rope_frequencies(float *buffer, size_t half_dim, size_t dim, float theta, const rope_scaling_state_t *scaling) {
    for (size_t i = 0; i < half_dim; ++i) {
        float freq = powf(theta, -((float)(2 * i) / (float)dim));
        float scale = scaling->freq_scale;
        if (scaling->ext_factor != 0.0f) {
            const float ramp = rope_yarn_ramp(scaling->corr_low, scaling->corr_high, (float)i);
            const float mix = ramp * scaling->ext_factor;
            scale = scaling->freq_scale + mix * (1.0f - scaling->freq_scale);
        }
        buffer[i] = freq * scale;
    }
}

void marmot_rope_freq_cache_init(marmot_rope_freq_cache_t *cache) {
    if (cache == nullptr) {
        return;
    }
    *cache = kDefaultRopeCache;
}

void marmot_rope_freq_cache_reset(marmot_rope_freq_cache_t *cache) {
    if (cache == nullptr) {
        return;
    }
    if (cache->owns_storage && cache->freqs != nullptr) {
        free(cache->freqs);
    }
    *cache = kDefaultRopeCache;
}

void marmot_rope_freq_cache_destroy(marmot_rope_freq_cache_t *cache) {
    marmot_rope_freq_cache_reset(cache);
}

marmot_error_t marmot_rope_freq_cache_ensure(
    marmot_rope_freq_cache_t *cache, size_t dim, const marmot_rope_params_t *params, marmot_rope_freq_span_t *out_span
) {
    if (out_span == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE frequency span output is null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    out_span->freqs = nullptr;
    out_span->dim = 0;
    out_span->attn_scale = 1.0f;
    out_span->owns_buffer = false;
    if (params == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE parameters are null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if ((dim & 1u) != 0u) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE frequency cache requires even dimensions");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t half_dim = dim / 2;
    if (half_dim == 0) {
        out_span->freqs = nullptr;
        out_span->dim = 0;
        out_span->owns_buffer = false;
        return MARMOT_SUCCESS;
    }

    if (half_dim > SIZE_MAX / sizeof(float)) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "RoPE frequency buffer overflow");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    rope_scaling_state_t scaling = {0};
    rope_scaling_prepare(params, dim, &scaling);

    const float theta = params->theta;

    if (cache == nullptr) {
        float *tmp = (float *)malloc(half_dim * sizeof(float));
        if (tmp == nullptr) {
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate RoPE frequency buffer");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        compute_rope_frequencies(tmp, half_dim, dim, theta, &scaling);
        out_span->freqs = tmp;
        out_span->dim = dim;
        out_span->attn_scale = scaling.attn_scale;
        out_span->owns_buffer = true;
        return MARMOT_SUCCESS;
    }

    if (cache->freqs != nullptr && cache->dim == dim && cache->theta == theta && cache->capacity_pairs >= half_dim &&
        cache->freq_scale == scaling.freq_scale && cache->ext_factor == scaling.ext_factor &&
        cache->attn_factor == scaling.attn_factor && cache->beta_fast == params->beta_fast &&
        cache->beta_slow == params->beta_slow && cache->orig_ctx_len == params->orig_ctx_len &&
        cache->scaling_type == params->scaling_type) {
        out_span->freqs = cache->freqs;
        out_span->dim = dim;
        out_span->attn_scale = cache->attn_scale;
        out_span->owns_buffer = false;
        return MARMOT_SUCCESS;
    }

    if (!cache->owns_storage || cache->capacity_pairs < half_dim) {
        float *replacement = (float *)realloc(cache->freqs, half_dim * sizeof(float));
        if (replacement == nullptr) {
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to grow RoPE frequency cache");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        cache->freqs = replacement;
        cache->capacity_pairs = half_dim;
        cache->owns_storage = true;
    }

    float *buffer = cache->freqs;
    compute_rope_frequencies(buffer, half_dim, dim, theta, &scaling);

    cache->dim = dim;
    cache->theta = theta;
    cache->freq_scale = scaling.freq_scale;
    cache->ext_factor = scaling.ext_factor;
    cache->attn_factor = scaling.attn_factor;
    cache->beta_fast = params->beta_fast;
    cache->beta_slow = params->beta_slow;
    cache->orig_ctx_len = params->orig_ctx_len;
    cache->scaling_type = params->scaling_type;
    cache->attn_scale = scaling.attn_scale;
    out_span->freqs = cache->freqs;
    out_span->dim = dim;
    out_span->attn_scale = scaling.attn_scale;
    out_span->owns_buffer = false;
    return MARMOT_SUCCESS;
}
