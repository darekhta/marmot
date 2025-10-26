#include "marmot/error.h"
#include "marmot/macros.h"
#include "marmot/quant_traits.h"
#include "marmot/tensor.h"

#include <float.h>
#include <math.h>
#include <string.h>

#include "core/helpers/quant.h"
#include "cpu_backend_internal.h"
#include "quantization/format_metadata.h"

marmot_error_t cpu_compute_quant_params(
    [[maybe_unused]] const void *device_ctx, const marmot_tensor_t *tensor, marmot_dtype_t target_dtype,
    size_t block_size, marmot_quant_params_t *out_params
) {
    if (unlikely(tensor == nullptr || out_params == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (unlikely(tensor->dtype != MARMOT_DTYPE_FLOAT32)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Compute quant params only supports FLOAT32 input");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    if (unlikely(block_size != 0)) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Per-block quantization not yet implemented");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const float *data = (const float *)tensor->data;
    size_t num_elements = marmot_tensor_num_elements(tensor);

    float min_val = data[0];
    float max_val = data[0];
    for (size_t i = 1; i < num_elements; i++) {
        if (data[i] < min_val)
            min_val = data[i];
        if (data[i] > max_val)
            max_val = data[i];
    }

    float qmin, qmax;
    switch (target_dtype) {
    case MARMOT_DTYPE_INT8:
        qmin = -128.0f;
        qmax = 127.0f;
        break;
    case MARMOT_DTYPE_UINT8:
        qmin = 0.0f;
        qmax = 255.0f;
        break;
    default:
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Quantization only supports INT8 and UINT8 currently");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    float scale = (max_val - min_val) / (qmax - qmin);
    if (scale < 1e-8f) {
        scale = 1.0f;
    }

    float zero_point = -min_val / scale + qmin;
    zero_point = fmaxf(qmin, fminf(qmax, zero_point));

    out_params->scale = scale;
    out_params->zero_point = zero_point;
    out_params->block_size = block_size;

    return MARMOT_SUCCESS;
}

marmot_error_t cpu_quantize(
    [[maybe_unused]] const void *device_ctx, const marmot_tensor_t *input, const marmot_quant_params_t *quant_params,
    marmot_tensor_t *output
) {
    if (unlikely(input == nullptr || quant_params == nullptr || output == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (unlikely(input->dtype != MARMOT_DTYPE_FLOAT32)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Quantize only supports FLOAT32 input");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    if (unlikely(input->shape.ndim != output->shape.ndim)) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Input and output shapes must match");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    for (size_t i = 0; i < input->shape.ndim; i++) {
        if (unlikely(input->shape.shape[i] != output->shape.shape[i])) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Input and output shapes must match");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }

    const float *in_data = (const float *)input->data;
    size_t num_elements = marmot_tensor_num_elements(input);
    float scale = quant_params->scale;
    float zero_point = quant_params->zero_point;

    switch (output->dtype) {
    case MARMOT_DTYPE_INT8: {
        marmot_int8_t *out_data = (marmot_int8_t *)output->data;
        for (size_t i = 0; i < num_elements; i++) {
            float scaled = in_data[i] / scale + zero_point;
            int32_t quantized = (int32_t)roundf(scaled);
            quantized = (quantized < -128) ? -128 : (quantized > 127 ? 127 : quantized);
            out_data[i].value = (int8_t)quantized;
        }
        break;
    }
    case MARMOT_DTYPE_UINT8: {
        marmot_uint8_t *out_data = (marmot_uint8_t *)output->data;
        for (size_t i = 0; i < num_elements; i++) {
            float scaled = in_data[i] / scale + zero_point;
            int32_t quantized = (int32_t)roundf(scaled);
            quantized = (quantized < 0) ? 0 : (quantized > 255 ? 255 : quantized);
            out_data[i].value = (uint8_t)quantized;
        }
        break;
    }
    default:
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Quantize only supports INT8 and UINT8 output currently");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    if (output->quant_params == nullptr) {
        output->quant_params = malloc(sizeof(marmot_quant_params_t));
        if (output->quant_params == nullptr) {
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate quant params");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }
    memcpy(output->quant_params, quant_params, sizeof(marmot_quant_params_t));

    return MARMOT_SUCCESS;
}

marmot_error_t cpu_dequantize(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    if (unlikely(input == nullptr || output == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (unlikely(output->dtype != MARMOT_DTYPE_FLOAT32)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Dequantize requires FLOAT32 output");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    if (unlikely(input->quant_params == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Input tensor has no quantization parameters");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (unlikely(input->shape.ndim != output->shape.ndim)) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Input and output shapes must match");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    for (size_t i = 0; i < input->shape.ndim; i++) {
        if (unlikely(input->shape.shape[i] != output->shape.shape[i])) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Input and output shapes must match");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }

    size_t num_elements = marmot_tensor_num_elements(input);
    float *out_data = (float *)output->data;
    float scale = input->quant_params->scale;
    float zero_point = input->quant_params->zero_point;

    switch (input->dtype) {
    case MARMOT_DTYPE_INT8: {
        const marmot_int8_t *in_data = (const marmot_int8_t *)input->data;
        for (size_t i = 0; i < num_elements; ++i) {
            out_data[i] = ((float)in_data[i].value - zero_point) * scale;
        }
        break;
    }
    case MARMOT_DTYPE_UINT8: {
        const marmot_uint8_t *in_data = (const marmot_uint8_t *)input->data;
        for (size_t i = 0; i < num_elements; ++i) {
            out_data[i] = ((float)in_data[i].value - zero_point) * scale;
        }
        break;
    }
    default:
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Dequantize only supports INT8 and UINT8 input currently");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_quantize_blockwise(const marmot_quant_traits_t *traits, const marmot_tensor_t *input, marmot_tensor_t *output) {
    if (traits == nullptr || traits->quantize_block == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Quantization traits unavailable");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (unlikely(input == nullptr || output == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensor pointer");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (unlikely(input->dtype != MARMOT_DTYPE_FLOAT32)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Block quantization requires FLOAT32 input");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (unlikely(output->dtype != MARMOT_DTYPE_UINT8 && output->dtype != MARMOT_DTYPE_INT8)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Block quantization requires UINT8 or INT8 output buffer");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    const float *f32_data = (const float *)input->data;
    uint8_t *dst = (uint8_t *)output->data;
    size_t block_size = traits->block_size;
    size_t block_bytes = traits->block_bytes;
    const cpu_quant_format_info_t *format = cpu_quant_format_info(traits->kind);
    if (format != nullptr) {
        block_size = format->block_values;
        block_bytes = format->block_bytes;
    }
    if (block_size == 0 || block_bytes == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Quantization format metadata is incomplete");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_quant_row_config_t row_cfg;
    if (!marmot_quant_compute_row_config(input, block_size, &row_cfg)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid tensor layout for quantization");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t required_bytes = row_cfg.num_blocks * block_bytes;
    if (unlikely(required_bytes > marmot_tensor_size_bytes(output))) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Quantized output buffer too small");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t global_block_idx = 0;
    for (size_t row_idx = 0; row_idx < row_cfg.num_rows; ++row_idx) {
        const float *row_data = f32_data + row_idx * row_cfg.row_size;
        for (size_t block_idx = 0; block_idx < row_cfg.blocks_per_row; ++block_idx, ++global_block_idx) {
            size_t block_start = block_idx * block_size;
            size_t block_end = block_start + block_size;
            if (block_end > row_cfg.row_size) {
                block_end = row_cfg.row_size;
            }
            const size_t block_len = block_end - block_start;
            void *block_out = dst + global_block_idx * block_bytes;
            marmot_error_t block_err =
                traits->quantize_block(row_data + block_start, (uint32_t)block_len, block_out, nullptr);
            if (block_err != MARMOT_SUCCESS) {
                return block_err;
            }
        }
    }

    output->quant_kind = traits->kind;
    output->quant_layout = traits->layout;
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_dequantize_blockwise(const marmot_quant_traits_t *traits, const marmot_tensor_t *input, marmot_tensor_t *output) {
    if (traits == nullptr || traits->dequantize_block == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Dequantization traits unavailable");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (unlikely(input == nullptr || output == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensor pointer");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (unlikely(input->dtype != MARMOT_DTYPE_UINT8 && input->dtype != MARMOT_DTYPE_INT8)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Block dequantization expects UINT8 or INT8 input buffer");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (unlikely(output->dtype != MARMOT_DTYPE_FLOAT32)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Block dequantization requires FLOAT32 output");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    const uint8_t *src = (const uint8_t *)input->data;
    float *dst = (float *)output->data;
    size_t block_size = traits->block_size;
    size_t block_bytes = traits->block_bytes;
    const cpu_quant_format_info_t *format = cpu_quant_format_info(traits->kind);
    if (format != nullptr) {
        block_size = format->block_values;
        block_bytes = format->block_bytes;
    }
    if (block_size == 0 || block_bytes == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Quantization format metadata is incomplete");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_quant_row_config_t row_cfg;
    if (!marmot_quant_compute_row_config(output, block_size, &row_cfg)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid tensor layout for dequantization");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t required_bytes = row_cfg.num_blocks * block_bytes;
    if (unlikely(required_bytes > marmot_tensor_size_bytes(input))) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Quantized input buffer too small");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t global_block_idx = 0;
    for (size_t row_idx = 0; row_idx < row_cfg.num_rows; ++row_idx) {
        float *row_out = dst + row_idx * row_cfg.row_size;
        for (size_t block_idx = 0; block_idx < row_cfg.blocks_per_row; ++block_idx, ++global_block_idx) {
            size_t block_start = block_idx * block_size;
            size_t block_end = block_start + block_size;
            if (block_end > row_cfg.row_size) {
                block_end = row_cfg.row_size;
            }
            const size_t block_len = block_end - block_start;
            const void *block_in = src + global_block_idx * block_bytes;
            marmot_error_t block_err =
                traits->dequantize_block(block_in, (uint32_t)block_len, row_out + block_start, nullptr);
            if (block_err != MARMOT_SUCCESS) {
                return block_err;
            }
        }
    }

    return MARMOT_SUCCESS;
}
