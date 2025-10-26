#ifndef MARMOT_OPS_UTILS_H
#define MARMOT_OPS_UTILS_H

#include "ops/matmul.h"
#include "ops_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t norm_size;
    size_t outer_size;
} marmot_norm_shape_t;

typedef struct {
    int axis;
    size_t axis_size;
    size_t inner_stride;
    size_t outer_size;
    size_t row_count;
} marmot_softmax_shape_t;

typedef struct {
    bool allow_residual;
    bool allow_weight;
    bool require_weight;
    bool allow_bias;
} marmot_norm_validation_opts_t;

MARMOT_NODISCARD marmot_error_t marmot_norm_validate(
    const marmot_tensor_t *x, const marmot_tensor_t *residual, const marmot_tensor_t *weight,
    const marmot_tensor_t *bias, marmot_tensor_t *out, const marmot_norm_validation_opts_t *opts,
    marmot_norm_shape_t *shape
);

MARMOT_NODISCARD marmot_error_t
marmot_softmax_prepare(const marmot_tensor_t *x, const marmot_tensor_t *out, int axis, marmot_softmax_shape_t *shape);

MARMOT_NODISCARD marmot_error_t marmot_infer_unary_output_shape(const marmot_shape_t *input, marmot_shape_t *out);
MARMOT_NODISCARD marmot_error_t
marmot_infer_binary_output_shape(const marmot_shape_t *lhs, const marmot_shape_t *rhs, marmot_shape_t *out);
MARMOT_NODISCARD marmot_error_t marmot_infer_ternary_output_shape(
    const marmot_shape_t *a, const marmot_shape_t *b, const marmot_shape_t *c, marmot_shape_t *out
);

MARMOT_NODISCARD marmot_error_t
marmot_infer_linear_output_shape(const marmot_shape_t *input, const marmot_shape_t *weight, marmot_shape_t *out);
MARMOT_NODISCARD marmot_error_t
marmot_infer_matmul_output_shape(const marmot_shape_t *a, const marmot_shape_t *b, marmot_shape_t *out);
MARMOT_NODISCARD marmot_error_t marmot_infer_matmul_qkv_output_shape(
    const marmot_shape_t *input, marmot_matmul_qkv_layout_t layout, const marmot_shape_t *fused_weight,
    const marmot_shape_t *wq, const marmot_shape_t *wk, const marmot_shape_t *wv, marmot_shape_t *out_q,
    marmot_shape_t *out_k, marmot_shape_t *out_v
);

MARMOT_NODISCARD marmot_error_t marmot_infer_embedding_output_shape(
    const marmot_shape_t *weights, const marmot_shape_t *token_ids, marmot_shape_t *out
);

MARMOT_NODISCARD marmot_error_t marmot_infer_reduction_output_shape(
    const marmot_shape_t *input, const int32_t *axes, size_t num_axes, bool keepdims, marmot_shape_t *out
);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_OPS_UTILS_H
