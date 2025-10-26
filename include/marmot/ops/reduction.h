#ifndef MARMOT_OPS_REDUCTION_H
#define MARMOT_OPS_REDUCTION_H

#include "../ops_types.h"
#include "../tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Reduction descriptor helpers
static inline marmot_reduction_desc_t marmot_reduction_desc_default(void) {
    marmot_reduction_desc_t desc = {
        .input = nullptr,
        .out = nullptr,
        .indices_out = nullptr,
        .axes = nullptr,
        .num_axes = 0,
        .keepdims = false,
        .unbiased = false,
        .epsilon = 0.0f,
    };
    return desc;
}

static inline bool marmot_reduction_desc_is_valid(const marmot_reduction_desc_t *desc) {
    return desc != nullptr && desc->input != nullptr && desc->out != nullptr;
}

// Reduction operations
MARMOT_NODISCARD marmot_error_t marmot_reduce_sum(const marmot_context_t *ctx, const marmot_reduction_desc_t *desc);
MARMOT_NODISCARD marmot_error_t marmot_reduce_mean(const marmot_context_t *ctx, const marmot_reduction_desc_t *desc);
MARMOT_NODISCARD marmot_error_t marmot_reduce_prod(const marmot_context_t *ctx, const marmot_reduction_desc_t *desc);
MARMOT_NODISCARD marmot_error_t marmot_reduce_max(const marmot_context_t *ctx, const marmot_reduction_desc_t *desc);
MARMOT_NODISCARD marmot_error_t marmot_reduce_min(const marmot_context_t *ctx, const marmot_reduction_desc_t *desc);
MARMOT_NODISCARD marmot_error_t marmot_reduce_argmax(const marmot_context_t *ctx, const marmot_reduction_desc_t *desc);
MARMOT_NODISCARD marmot_error_t marmot_reduce_argmin(const marmot_context_t *ctx, const marmot_reduction_desc_t *desc);
MARMOT_NODISCARD marmot_error_t marmot_reduce_any(const marmot_context_t *ctx, const marmot_reduction_desc_t *desc);
MARMOT_NODISCARD marmot_error_t marmot_reduce_all(const marmot_context_t *ctx, const marmot_reduction_desc_t *desc);
MARMOT_NODISCARD marmot_error_t
marmot_reduce_variance(const marmot_context_t *ctx, const marmot_reduction_desc_t *desc);
MARMOT_NODISCARD marmot_error_t marmot_reduce_std(const marmot_context_t *ctx, const marmot_reduction_desc_t *desc);
MARMOT_NODISCARD marmot_error_t marmot_reduce_norm_l1(const marmot_context_t *ctx, const marmot_reduction_desc_t *desc);
MARMOT_NODISCARD marmot_error_t marmot_reduce_norm_l2(const marmot_context_t *ctx, const marmot_reduction_desc_t *desc);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_OPS_REDUCTION_H
