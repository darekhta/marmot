#include "ops/matmul/quantized/internal/vec_dot.h"

#include "ops/matmul/quantized/matmul_quant_kernels.h"
#include "quantization/format_metadata.h"

marmot_error_t cpu_vec_dot(const void *device_ctx, const marmot_vec_dot_descriptor_t *desc, float *result) {
    if (desc == nullptr || result == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "cpu_vec_dot: null descriptor or result");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (desc->weights == nullptr || desc->activations == nullptr || desc->num_blocks == 0) {
        *result = 0.0f;
        return MARMOT_SUCCESS;
    }
    if (device_ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "cpu_vec_dot: null device context");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (desc->weight_kind >= MARMOT_QUANT_KIND_COUNT || desc->activation_kind >= MARMOT_QUANT_KIND_COUNT) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "cpu_vec_dot: unsupported quantization kinds");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const cpu_quant_format_info_t *format = cpu_quant_format_info(desc->weight_kind);
    if (format == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "cpu_vec_dot: unsupported quantization kind");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (format->layout != desc->layout) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "cpu_vec_dot: quantization layout mismatch");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const marmot_quant_kind_t expected_activation = cpu_quant_format_activation_kind(desc->weight_kind);
    if (expected_activation != desc->activation_kind) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "cpu_vec_dot: activation kind mismatch");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const cpu_matmul_quant_kernel_t *kernel = cpu_matmul_quant_select_kernel(device_ctx, desc->weight_kind);
    if (kernel == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "cpu_vec_dot: kernel not available");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    switch (expected_activation) {
    case MARMOT_QUANT_KIND_Q8_0:
        if (kernel->ops.dot_q8_0 == nullptr) {
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "cpu_vec_dot: activation kind not supported");
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        *result = kernel->ops.dot_q8_0(desc->weights, (const marmot_q8_0_block_t *)desc->activations, desc->num_blocks);
        return MARMOT_SUCCESS;
    case MARMOT_QUANT_KIND_Q8_K:
        if (kernel->ops.dot_q8_k == nullptr) {
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "cpu_vec_dot: activation kind not supported");
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        *result = kernel->ops.dot_q8_k(desc->weights, (const marmot_q8_k_block_t *)desc->activations, desc->num_blocks);
        return MARMOT_SUCCESS;
    default:
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "cpu_vec_dot: activation kind not supported");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
}
