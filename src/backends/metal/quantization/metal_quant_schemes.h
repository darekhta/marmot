#pragma once

#ifdef __APPLE__

#include "metal_backend_internal.h"

marmot_error_t metal_quantize_q4_0_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_dequantize_q4_0_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_quantize_q4_1_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_dequantize_q4_1_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_quantize_q5_0_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_dequantize_q5_0_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_quantize_q5_1_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_dequantize_q5_1_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_quantize_q8_0_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_dequantize_q8_0_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_quantize_q8_1_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_dequantize_q8_1_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_quantize_q2_k_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_dequantize_q2_k_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_quantize_q3_k_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_dequantize_q3_k_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_quantize_q4_k_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_dequantize_q4_k_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_quantize_q5_k_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_dequantize_q5_k_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_quantize_q6_k_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_dequantize_q6_k_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_quantize_q8_k_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t metal_dequantize_q8_k_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);

#endif // __APPLE__
