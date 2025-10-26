#ifndef MARMOT_OPS_UNARY_H
#define MARMOT_OPS_UNARY_H

#include "../ops_types.h"
#include "../tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Unary element-wise math
MARMOT_NODISCARD marmot_error_t marmot_abs(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out);
MARMOT_NODISCARD marmot_error_t marmot_neg(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out);
MARMOT_NODISCARD marmot_error_t
marmot_sign(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out);
MARMOT_NODISCARD marmot_error_t
marmot_sqrt(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out);
MARMOT_NODISCARD marmot_error_t marmot_exp(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out);
MARMOT_NODISCARD marmot_error_t marmot_log(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out);
MARMOT_NODISCARD marmot_error_t
marmot_bitwise_not(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out);

// Activation functions
MARMOT_NODISCARD marmot_error_t
marmot_relu(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out);
MARMOT_NODISCARD marmot_error_t
marmot_gelu(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out);
MARMOT_NODISCARD marmot_error_t
marmot_gelu_tanh(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out);
MARMOT_NODISCARD marmot_error_t
marmot_silu(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out);
MARMOT_NODISCARD marmot_error_t
marmot_swish(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out);
MARMOT_NODISCARD marmot_error_t
marmot_sigmoid(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out);
MARMOT_NODISCARD marmot_error_t
marmot_tanh(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out);
MARMOT_NODISCARD marmot_error_t
marmot_mish(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out);
MARMOT_NODISCARD marmot_error_t
marmot_elu(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out, float alpha);
MARMOT_NODISCARD marmot_error_t
marmot_selu(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out, float alpha, float lambda);
MARMOT_NODISCARD marmot_error_t
marmot_leaky_relu(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out, float negative_slope);
MARMOT_NODISCARD marmot_error_t
marmot_prelu(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out, float slope);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_OPS_UNARY_H
