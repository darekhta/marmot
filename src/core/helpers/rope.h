#ifndef MARMOT_CORE_HELPERS_ROPE_H
#define MARMOT_CORE_HELPERS_ROPE_H

#include "marmot/tensor.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct marmot_rope_freq_cache {
    float *freqs;
    size_t capacity_pairs;
    size_t dim;
    float theta;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    uint32_t orig_ctx_len;
    marmot_rope_scaling_type_t scaling_type;
    float attn_scale;
    bool owns_storage;
} marmot_rope_freq_cache_t;

void marmot_rope_freq_cache_init(marmot_rope_freq_cache_t *cache);
void marmot_rope_freq_cache_reset(marmot_rope_freq_cache_t *cache);
void marmot_rope_freq_cache_destroy(marmot_rope_freq_cache_t *cache);

typedef struct marmot_rope_freq_span {
    const float *freqs;
    size_t dim;
    float attn_scale;
    bool owns_buffer;
} marmot_rope_freq_span_t;

marmot_error_t marmot_rope_freq_cache_ensure(
    marmot_rope_freq_cache_t *cache, size_t dim, const marmot_rope_params_t *params, marmot_rope_freq_span_t *out_span
);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_CORE_HELPERS_ROPE_H
