#ifndef MARMOT_OPS_ELEMENTWISE_H
#define MARMOT_OPS_ELEMENTWISE_H

#include "../ops_types.h"
#include "../tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Element-wise addition: out = a + b
MARMOT_NODISCARD marmot_error_t
marmot_add(const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out);

// Element-wise addition with activation: out = activation(a + b)
MARMOT_NODISCARD marmot_error_t
marmot_add_relu(const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out);
MARMOT_NODISCARD marmot_error_t
marmot_add_gelu(const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out);
MARMOT_NODISCARD marmot_error_t
marmot_add_silu(const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out);

// Element-wise subtraction: out = a - b
MARMOT_NODISCARD marmot_error_t
marmot_sub(const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out);

// Element-wise multiplication: out = a * b
MARMOT_NODISCARD marmot_error_t
marmot_mul(const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out);

// SwiGLU: out = silu(a) * b
MARMOT_NODISCARD marmot_error_t
marmot_swiglu(const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out);

// GeGLU: out = gelu(a) * b
MARMOT_NODISCARD marmot_error_t
marmot_geglu(const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out);

// Element-wise division: out = a / b (integer division truncates toward zero)
MARMOT_NODISCARD marmot_error_t
marmot_div(const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out);

// Element-wise minimum/maximum
MARMOT_NODISCARD marmot_error_t
marmot_min(const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out);
MARMOT_NODISCARD marmot_error_t
marmot_max(const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out);

// Power and modulo
MARMOT_NODISCARD marmot_error_t
marmot_pow(const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out);
MARMOT_NODISCARD marmot_error_t
marmot_mod(const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out);

// Bitwise operations on integer tensors
MARMOT_NODISCARD marmot_error_t marmot_bitwise_and(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
);
MARMOT_NODISCARD marmot_error_t marmot_bitwise_or(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
);
MARMOT_NODISCARD marmot_error_t marmot_bitwise_xor(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
);
MARMOT_NODISCARD marmot_error_t marmot_bitwise_shift_left(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
);
MARMOT_NODISCARD marmot_error_t marmot_bitwise_shift_right(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
);
MARMOT_NODISCARD marmot_error_t marmot_bitwise_shift_right_logical(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
);

// Comparison helpers produce UINT8 tensors (0 or 1)
MARMOT_NODISCARD marmot_error_t marmot_compare_eq(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
);
MARMOT_NODISCARD marmot_error_t marmot_compare_ne(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
);
MARMOT_NODISCARD marmot_error_t marmot_compare_lt(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
);
MARMOT_NODISCARD marmot_error_t marmot_compare_le(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
);
MARMOT_NODISCARD marmot_error_t marmot_compare_gt(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
);
MARMOT_NODISCARD marmot_error_t marmot_compare_ge(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
);

// Ternary fused multiply-add: out = a * b + c (float-family tensors)
MARMOT_NODISCARD marmot_error_t marmot_fma(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, const marmot_tensor_t *c,
    marmot_tensor_t *out
);

// Element-wise select: out = mask ? true_value : false_value
MARMOT_NODISCARD marmot_error_t marmot_where(
    const marmot_context_t *ctx, const marmot_tensor_t *mask, const marmot_tensor_t *true_value,
    const marmot_tensor_t *false_value, marmot_tensor_t *out
);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_OPS_ELEMENTWISE_H
