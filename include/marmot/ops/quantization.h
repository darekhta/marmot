#ifndef MARMOT_OPS_QUANTIZATION_H
#define MARMOT_OPS_QUANTIZATION_H

#include "../ops_types.h"
#include "../tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Quantization parameter helpers
MARMOT_NODISCARD marmot_error_t marmot_compute_quant_params(
    const marmot_context_t *ctx, const marmot_tensor_t *tensor, marmot_dtype_t target_dtype, size_t block_size,
    marmot_quant_params_t *out_params
);

// Quantize/dequantize tensors
MARMOT_NODISCARD marmot_error_t marmot_quantize(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_quant_params_t *quant_params,
    marmot_tensor_t *output
);
MARMOT_NODISCARD marmot_error_t
marmot_dequantize(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);

// INT4/INT5/INT8 block quantization
MARMOT_NODISCARD marmot_error_t
marmot_quantize_q4_0(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_dequantize_q4_0(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_quantize_q4_1(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_dequantize_q4_1(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_quantize_q5_0(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_dequantize_q5_0(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_quantize_q5_1(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_dequantize_q5_1(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_quantize_q8_0(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_dequantize_q8_0(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_quantize_q8_1(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_dequantize_q8_1(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);

// K-Quant formats
MARMOT_NODISCARD marmot_error_t
marmot_quantize_q2_k(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_dequantize_q2_k(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_quantize_q3_k(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_dequantize_q3_k(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_quantize_q4_k(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_dequantize_q4_k(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_quantize_q5_k(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_dequantize_q5_k(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_quantize_q6_k(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_dequantize_q6_k(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_quantize_q8_k(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
MARMOT_NODISCARD marmot_error_t
marmot_dequantize_q8_k(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output);

// Quantized vector dot products
MARMOT_NODISCARD marmot_error_t
marmot_vec_dot(const marmot_context_t *ctx, const marmot_vec_dot_descriptor_t *desc, float *result);
MARMOT_NODISCARD marmot_error_t marmot_vec_dot_descriptor_from_tensors(
    const marmot_tensor_t *input, const marmot_tensor_t *weight, marmot_vec_dot_descriptor_t *out
);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_OPS_QUANTIZATION_H
