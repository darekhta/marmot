#include "matmul_epilogue.h"

#include "cpu_backend_internal.h"

static bool cpu_matmul_bias_dtype_supported(marmot_dtype_t out_dtype, marmot_dtype_t bias_dtype) {
    if (out_dtype == bias_dtype) {
        return true;
    }
    if (bias_dtype == MARMOT_DTYPE_FLOAT32) {
        if (out_dtype == MARMOT_DTYPE_FLOAT16 || out_dtype == MARMOT_DTYPE_BFLOAT16) {
            return true;
        }
    }
    return false;
}

static bool cpu_matmul_validate_bias(
    const marmot_tensor_t *out, const marmot_tensor_t *bias, size_t *feature_dim, bool *is_scalar
) {
    if (out == nullptr || bias == nullptr || feature_dim == nullptr || is_scalar == nullptr) {
        return false;
    }
    if (!cpu_matmul_bias_dtype_supported(out->dtype, bias->dtype)) {
        return false;
    }
    size_t last_dim = out->shape.ndim > 0 ? out->shape.shape[out->shape.ndim - 1] : 1;
    size_t bias_elems = marmot_tensor_num_elements(bias);
    if (bias_elems == 1) {
        *feature_dim = last_dim;
        *is_scalar = true;
        return true;
    }
    if (bias->shape.ndim != 1 || bias->shape.shape[0] != last_dim) {
        return false;
    }
    *feature_dim = last_dim;
    *is_scalar = false;
    return true;
}

typedef marmot_error_t (*cpu_matmul_bias_fn)(
    const marmot_tensor_t *out, const marmot_tensor_t *bias, size_t rows, size_t cols, bool scalar_bias
);

static marmot_error_t cpu_matmul_bias_f64(
    const marmot_tensor_t *out, const marmot_tensor_t *bias, size_t rows, size_t cols, bool scalar_bias
) {
    double *out_data = (double *)out->data;
    const double *bias_data = (const double *)bias->data;
    for (size_t r = 0; r < rows; r++) {
        double *row = out_data + r * cols;
        if (scalar_bias) {
            double b = bias_data[0];
            for (size_t c = 0; c < cols; c++) {
                row[c] += b;
            }
        } else {
            for (size_t c = 0; c < cols; c++) {
                row[c] += bias_data[c];
            }
        }
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_matmul_bias_f32(
    const marmot_tensor_t *out, const marmot_tensor_t *bias, size_t rows, size_t cols, bool scalar_bias
) {
    if (bias->dtype != MARMOT_DTYPE_FLOAT32) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    float *out_data = (float *)out->data;
    const float *bias_data = (const float *)bias->data;
    for (size_t r = 0; r < rows; r++) {
        float *row = out_data + r * cols;
        if (scalar_bias) {
            float b = bias_data[0];
            for (size_t c = 0; c < cols; c++) {
                row[c] += b;
            }
        } else {
            for (size_t c = 0; c < cols; c++) {
                row[c] += bias_data[c];
            }
        }
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_matmul_bias_f16(
    const marmot_tensor_t *out, const marmot_tensor_t *bias, size_t rows, size_t cols, bool scalar_bias
) {
    marmot_float16_t *out_data = (marmot_float16_t *)out->data;
    if (bias->dtype == MARMOT_DTYPE_FLOAT16) {
        const marmot_float16_t *bias_data = (const marmot_float16_t *)bias->data;
        for (size_t r = 0; r < rows; r++) {
            marmot_float16_t *row = out_data + r * cols;
            if (scalar_bias) {
                _Float16 b = marmot_float16_to_native(bias_data[0]);
                for (size_t c = 0; c < cols; c++) {
                    _Float16 value = marmot_float16_to_native(row[c]);
                    row[c] = marmot_native_to_float16((_Float16)(value + b));
                }
            } else {
                for (size_t c = 0; c < cols; c++) {
                    _Float16 value = marmot_float16_to_native(row[c]);
                    _Float16 b = marmot_float16_to_native(bias_data[c]);
                    row[c] = marmot_native_to_float16((_Float16)(value + b));
                }
            }
        }
        return MARMOT_SUCCESS;
    }
    if (bias->dtype == MARMOT_DTYPE_FLOAT32) {
        const float *bias_data = (const float *)bias->data;
        for (size_t r = 0; r < rows; r++) {
            marmot_float16_t *row = out_data + r * cols;
            if (scalar_bias) {
                float b = bias_data[0];
                for (size_t c = 0; c < cols; c++) {
                    float value = (float)marmot_float16_to_native(row[c]);
                    row[c] = marmot_native_to_float16((_Float16)(value + b));
                }
            } else {
                for (size_t c = 0; c < cols; c++) {
                    float value = (float)marmot_float16_to_native(row[c]);
                    float b = bias_data[c];
                    row[c] = marmot_native_to_float16((_Float16)(value + b));
                }
            }
        }
        return MARMOT_SUCCESS;
    }
    return MARMOT_ERROR_UNSUPPORTED_DTYPE;
}

static marmot_error_t cpu_matmul_bias_bf16(
    const marmot_tensor_t *out, const marmot_tensor_t *bias, size_t rows, size_t cols, bool scalar_bias
) {
    marmot_bfloat16_t *out_data = (marmot_bfloat16_t *)out->data;
    if (bias->dtype == MARMOT_DTYPE_BFLOAT16) {
        const marmot_bfloat16_t *bias_data = (const marmot_bfloat16_t *)bias->data;
        for (size_t r = 0; r < rows; r++) {
            marmot_bfloat16_t *row = out_data + r * cols;
            if (scalar_bias) {
                float b = marmot_bf16_to_f32_ref(bias_data[0]);
                for (size_t c = 0; c < cols; c++) {
                    float value = marmot_bf16_to_f32_ref(row[c]);
                    row[c] = marmot_f32_to_bf16_ref(value + b);
                }
            } else {
                for (size_t c = 0; c < cols; c++) {
                    float value = marmot_bf16_to_f32_ref(row[c]);
                    float b = marmot_bf16_to_f32_ref(bias_data[c]);
                    row[c] = marmot_f32_to_bf16_ref(value + b);
                }
            }
        }
        return MARMOT_SUCCESS;
    }
    if (bias->dtype == MARMOT_DTYPE_FLOAT32) {
        const float *bias_data = (const float *)bias->data;
        for (size_t r = 0; r < rows; r++) {
            marmot_bfloat16_t *row = out_data + r * cols;
            if (scalar_bias) {
                float b = bias_data[0];
                for (size_t c = 0; c < cols; c++) {
                    float value = marmot_bf16_to_f32_ref(row[c]);
                    row[c] = marmot_f32_to_bf16_ref(value + b);
                }
            } else {
                for (size_t c = 0; c < cols; c++) {
                    float value = marmot_bf16_to_f32_ref(row[c]);
                    float b = bias_data[c];
                    row[c] = marmot_f32_to_bf16_ref(value + b);
                }
            }
        }
        return MARMOT_SUCCESS;
    }
    return MARMOT_ERROR_UNSUPPORTED_DTYPE;
}

static const cpu_matmul_bias_fn k_cpu_matmul_bias_handlers[MARMOT_DTYPE_COUNT] = {
    [MARMOT_DTYPE_FLOAT64] = cpu_matmul_bias_f64,
    [MARMOT_DTYPE_FLOAT32] = cpu_matmul_bias_f32,
    [MARMOT_DTYPE_FLOAT16] = cpu_matmul_bias_f16,
    [MARMOT_DTYPE_BFLOAT16] = cpu_matmul_bias_bf16,
};

static marmot_error_t cpu_matmul_apply_bias(
    const marmot_tensor_t *out, const marmot_tensor_t *bias, size_t rows, size_t cols, bool scalar_bias
) {
    if (bias == nullptr) {
        return MARMOT_SUCCESS;
    }
    if (out->dtype >= MARMOT_DTYPE_COUNT) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Matmul bias dtype unsupported");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    cpu_matmul_bias_fn handler = k_cpu_matmul_bias_handlers[out->dtype];
    if (handler == nullptr) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Matmul bias dtype unsupported");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    return handler(out, bias, rows, cols, scalar_bias);
}

marmot_error_t
cpu_matmul_apply_epilogue(const void *device_ctx, marmot_tensor_t *out, const marmot_matmul_epilogue_t *epilogue) {
    if (epilogue == nullptr) {
        return MARMOT_SUCCESS;
    }
    (void)device_ctx;
    const marmot_tensor_t *bias = epilogue->bias;
    size_t feature_dim = 0;
    bool scalar_bias = false;
    if (bias != nullptr) {
        if (!cpu_matmul_validate_bias(out, bias, &feature_dim, &scalar_bias)) {
            marmot_set_error(
                MARMOT_ERROR_DIMENSION_MISMATCH, "Matmul fused bias must be scalar or match the last dimension"
            );
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        size_t rows = marmot_tensor_num_elements(out) / feature_dim;
        marmot_error_t bias_err = cpu_matmul_apply_bias(out, bias, rows, feature_dim, scalar_bias);
        if (bias_err != MARMOT_SUCCESS) {
            return bias_err;
        }
    }
    return MARMOT_SUCCESS;
}
