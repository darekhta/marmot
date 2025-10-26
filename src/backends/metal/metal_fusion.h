#pragma once

#include "marmot/device.h"
#include "marmot/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

marmot_error_t metal_add_relu_fused(
    const marmot_context_t *ctx, const marmot_tensor_t *input_a, const marmot_tensor_t *input_b, marmot_tensor_t *output
);

marmot_error_t metal_add_gelu_fused(
    const marmot_context_t *ctx, const marmot_tensor_t *input_a, const marmot_tensor_t *input_b, marmot_tensor_t *output
);

marmot_error_t metal_add_silu_fused(
    const marmot_context_t *ctx, const marmot_tensor_t *input_a, const marmot_tensor_t *input_b, marmot_tensor_t *output
);

marmot_error_t metal_mul_add_fused(
    const marmot_context_t *ctx, const marmot_tensor_t *input_a, const marmot_tensor_t *input_b,
    const marmot_tensor_t *input_c, marmot_tensor_t *output
);

#ifdef __cplusplus
}
#endif
