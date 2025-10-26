#ifndef MARMOT_OPS_NEURAL_H
#define MARMOT_OPS_NEURAL_H

#include "../ops_types.h"
#include "../tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Layer normalization descriptor helpers and operation.
static inline marmot_layernorm_desc_t marmot_layernorm_desc_default(void) {
    marmot_layernorm_desc_t desc = {
        .x = nullptr,
        .residual = nullptr,
        .weight = nullptr,
        .bias = nullptr,
        .out = nullptr,
        .eps = 0.0f,
    };
    return desc;
}

// Layer normalization: out = (x - mean) / sqrt(variance + eps) * weight + bias.
MARMOT_NODISCARD marmot_error_t marmot_layernorm(const marmot_context_t *ctx, const marmot_layernorm_desc_t *desc);

static inline marmot_rmsnorm_desc_t marmot_rmsnorm_desc_default(void) {
    marmot_rmsnorm_desc_t desc = {
        .x = nullptr,
        .residual = nullptr,
        .weight = nullptr,
        .out = nullptr,
        .eps = 0.0f,
    };
    return desc;
}

// RMS normalization: out = (x + residual) / sqrt(mean(...) + eps) * weight.
MARMOT_NODISCARD marmot_error_t marmot_rmsnorm(const marmot_context_t *ctx, const marmot_rmsnorm_desc_t *desc);

// Gemma RMS normalization: out = (x + residual) / sqrt(mean(...) + eps) * (weight + 1).
MARMOT_NODISCARD marmot_error_t marmot_rmsnorm_gemma(const marmot_context_t *ctx, const marmot_rmsnorm_desc_t *desc);

static inline marmot_softmax_desc_t marmot_softmax_desc_default(void) {
    marmot_softmax_desc_t desc = {
        .x = nullptr,
        .out = nullptr,
        .axis = -1,
    };
    return desc;
}

// Softmax: out = exp(x - max(x)) / sum(exp(x - max(x))) along specified axis.
MARMOT_NODISCARD marmot_error_t marmot_softmax(const marmot_context_t *ctx, const marmot_softmax_desc_t *desc);

// Descriptor validation helpers
static inline bool marmot_layernorm_desc_is_valid(const marmot_layernorm_desc_t *desc) {
    return desc != nullptr && desc->x != nullptr && desc->out != nullptr;
}

static inline bool marmot_rmsnorm_desc_is_valid(const marmot_rmsnorm_desc_t *desc) {
    return desc != nullptr && desc->x != nullptr && desc->out != nullptr;
}

static inline bool marmot_softmax_desc_is_valid(const marmot_softmax_desc_t *desc) {
    return desc != nullptr && desc->x != nullptr && desc->out != nullptr;
}

// Embedding operations
MARMOT_NODISCARD marmot_error_t
marmot_embedding_lookup(const marmot_context_t *ctx, const marmot_embedding_desc_t *desc);

MARMOT_NODISCARD marmot_error_t
marmot_embedding_gather(const marmot_context_t *ctx, const marmot_embedding_gather_desc_t *desc);

// Descriptor helpers
static inline marmot_embedding_desc_t marmot_embedding_desc_default(void) {
    marmot_embedding_desc_t desc = {
        .weights = nullptr,
        .token_ids = nullptr,
        .out = nullptr,
        .dtype_out = (marmot_dtype_t)MARMOT_DTYPE_COUNT,
        .scale = 1.0f,
        .padding_id = -1,
        .bounds_check = true,
        .ragged = false,
        .row_offsets = nullptr,
        .num_row_offsets = 0,
        .prefer_gpu_private = MARMOT_PREFERENCE_DEFAULT,
        .allow_quant_decode_on_the_fly = MARMOT_PREFERENCE_DEFAULT,
    };
    return desc;
}

static inline marmot_embedding_gather_desc_t marmot_embedding_gather_desc_default(void) {
    marmot_embedding_gather_desc_t desc = {
        .weights = nullptr,
        .token_ids = nullptr,
        .out = nullptr,
        .dtype_out = (marmot_dtype_t)MARMOT_DTYPE_COUNT,
        .scale = 1.0f,
        .padding_id = -1,
        .bounds_check = true,
        .prefer_gpu_private = MARMOT_PREFERENCE_DEFAULT,
        .allow_quant_decode_on_the_fly = MARMOT_PREFERENCE_DEFAULT,
    };
    return desc;
}

// Embedding descriptor validation helpers
static inline bool marmot_embedding_desc_is_valid(const marmot_embedding_desc_t *desc) {
    return desc != nullptr && desc->weights != nullptr && desc->token_ids != nullptr && desc->out != nullptr;
}

static inline bool marmot_embedding_gather_desc_is_valid(const marmot_embedding_gather_desc_t *desc) {
    return desc != nullptr && desc->weights != nullptr && desc->token_ids != nullptr && desc->out != nullptr;
}

#ifdef __cplusplus
}
#endif

#endif // MARMOT_OPS_NEURAL_H
