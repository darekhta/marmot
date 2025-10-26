#include "ops/normalization/normalization_internal.h"

// ==================================================================
// CPU Backend Normalization Operations (Scalar)
// ==================================================================

// ===================================================================
// LayerNorm Helper Functions
// ===================================================================

// Scalar fallback implementations
static inline void
compute_mean_variance_scalar(const float *data, const float *residual, size_t n, float *mean_out, float *var_out) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float v = data[i];
        if (residual != nullptr) {
            v += residual[i];
        }
        sum += v;
        sum_sq += v * v;
    }
    float inv_n = 1.0f / (float)n;
    float mean = sum * inv_n;
    float variance = sum_sq * inv_n - mean * mean;
    if (variance < 0.0f) {
        variance = 0.0f;
    }
    *mean_out = mean;
    *var_out = variance;
}

static inline void normalize_scalar(
    const float *x, const float *residual, float *out, size_t n, float mean, float inv_std, const float *weight,
    const float *bias
) {
    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        if (residual != nullptr) {
            v += residual[i];
        }
        float normalized = (v - mean) * inv_std;
        if (weight != nullptr) {
            normalized *= weight[i];
        }
        if (bias != nullptr) {
            normalized += bias[i];
        }
        out[i] = normalized;
    }
}

static inline void compute_mean_variance_scalar_f64(
    const double *data, const double *residual, size_t n, double *mean_out, double *var_out
) {
    double sum = 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double v = data[i];
        if (residual != nullptr) {
            v += residual[i];
        }
        sum += v;
        sum_sq += v * v;
    }
    double inv_n = 1.0 / (double)n;
    double mean = sum * inv_n;
    double variance = sum_sq * inv_n - mean * mean;
    if (variance < 0.0) {
        variance = 0.0;
    }
    *mean_out = mean;
    *var_out = variance;
}

static inline void normalize_scalar_f64(
    const double *x, const double *residual, double *out, size_t n, double mean, double inv_std, const double *weight,
    const double *bias
) {
    for (size_t i = 0; i < n; ++i) {
        double v = x[i];
        if (residual != nullptr) {
            v += residual[i];
        }
        double normalized = (v - mean) * inv_std;
        if (weight != nullptr) {
            normalized *= weight[i];
        }
        if (bias != nullptr) {
            normalized += bias[i];
        }
        out[i] = normalized;
    }
}

marmot_error_t cpu_layernorm_mixed_vector_f32(
    marmot_dtype_t dtype, const void *input_data, const void *residual_data, const float *weight, const float *bias,
    void *out_data, size_t outer_size, size_t norm_size, float eps
) {
    if (dtype == MARMOT_DTYPE_FLOAT16) {
        const marmot_float16_t *x = (const marmot_float16_t *)input_data;
        const marmot_float16_t *residual = (const marmot_float16_t *)residual_data;
        marmot_float16_t *out = (marmot_float16_t *)out_data;
        for (size_t row = 0; row < outer_size; ++row) {
            const marmot_float16_t *x_row = x + row * norm_size;
            const marmot_float16_t *res_row = residual != nullptr ? residual + row * norm_size : nullptr;
            marmot_float16_t *out_row = out + row * norm_size;

            float sum = 0.0f;
            float sum_sq = 0.0f;
            for (size_t i = 0; i < norm_size; ++i) {
                float v = (float)marmot_float16_to_native(x_row[i]);
                if (res_row != nullptr) {
                    v += (float)marmot_float16_to_native(res_row[i]);
                }
                sum += v;
                sum_sq += v * v;
            }

            float mean = sum / (float)norm_size;
            float var = (sum_sq / (float)norm_size) - (mean * mean);
            if (var < 0.0f) {
                var = 0.0f;
            }
            float inv_std = cpu_norm_inv_sqrt_f32(var + eps);

            for (size_t i = 0; i < norm_size; ++i) {
                float v = (float)marmot_float16_to_native(x_row[i]);
                if (res_row != nullptr) {
                    v += (float)marmot_float16_to_native(res_row[i]);
                }
                float normalized = (v - mean) * inv_std;
                if (weight != nullptr) {
                    normalized *= weight[i];
                }
                if (bias != nullptr) {
                    normalized += bias[i];
                }
                out_row[i] = marmot_f32_to_f16_ref(normalized);
            }
        }
        return MARMOT_SUCCESS;
    }

    if (dtype == MARMOT_DTYPE_BFLOAT16) {
        const marmot_bfloat16_t *x = (const marmot_bfloat16_t *)input_data;
        const marmot_bfloat16_t *residual = (const marmot_bfloat16_t *)residual_data;
        marmot_bfloat16_t *out = (marmot_bfloat16_t *)out_data;
        for (size_t row = 0; row < outer_size; ++row) {
            const marmot_bfloat16_t *x_row = x + row * norm_size;
            const marmot_bfloat16_t *res_row = residual != nullptr ? residual + row * norm_size : nullptr;
            marmot_bfloat16_t *out_row = out + row * norm_size;

            float sum = 0.0f;
            float sum_sq = 0.0f;
            for (size_t i = 0; i < norm_size; ++i) {
                float v = marmot_bf16_to_f32_ref(x_row[i]);
                if (res_row != nullptr) {
                    v += marmot_bf16_to_f32_ref(res_row[i]);
                }
                sum += v;
                sum_sq += v * v;
            }

            float mean = sum / (float)norm_size;
            float var = (sum_sq / (float)norm_size) - (mean * mean);
            if (var < 0.0f) {
                var = 0.0f;
            }
            float inv_std = cpu_norm_inv_sqrt_f32(var + eps);

            for (size_t i = 0; i < norm_size; ++i) {
                float v = marmot_bf16_to_f32_ref(x_row[i]);
                if (res_row != nullptr) {
                    v += marmot_bf16_to_f32_ref(res_row[i]);
                }
                float normalized = (v - mean) * inv_std;
                if (weight != nullptr) {
                    normalized *= weight[i];
                }
                if (bias != nullptr) {
                    normalized += bias[i];
                }
                out_row[i] = marmot_f32_to_bf16_ref(normalized);
            }
        }
        return MARMOT_SUCCESS;
    }

    marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Mixed-dtype layernorm supported only for f16/bf16 inputs");
    return MARMOT_ERROR_UNSUPPORTED_DTYPE;
}

marmot_error_t cpu_rmsnorm_mixed_vector_f32(
    marmot_dtype_t dtype, const void *input_data, const void *residual_data, const float *weight, void *out_data,
    size_t outer_size, size_t norm_size, float eps, float weight_offset
) {
    if (dtype == MARMOT_DTYPE_FLOAT16) {
        const marmot_float16_t *x = (const marmot_float16_t *)input_data;
        const marmot_float16_t *residual = (const marmot_float16_t *)residual_data;
        marmot_float16_t *out = (marmot_float16_t *)out_data;
        for (size_t row = 0; row < outer_size; ++row) {
            const marmot_float16_t *x_row = x + row * norm_size;
            const marmot_float16_t *res_row = residual != nullptr ? residual + row * norm_size : nullptr;
            marmot_float16_t *out_row = out + row * norm_size;

            float sum_sq = 0.0f;
            for (size_t i = 0; i < norm_size; ++i) {
                float v = (float)marmot_float16_to_native(x_row[i]);
                if (res_row != nullptr) {
                    v += (float)marmot_float16_to_native(res_row[i]);
                }
                sum_sq += v * v;
            }
            float mean_sq = sum_sq / (float)norm_size;
            float inv_rms = cpu_norm_inv_sqrt_f32(mean_sq + eps);

            for (size_t i = 0; i < norm_size; ++i) {
                float v = (float)marmot_float16_to_native(x_row[i]);
                if (res_row != nullptr) {
                    v += (float)marmot_float16_to_native(res_row[i]);
                }
                float normalized = v * inv_rms;
                if (weight != nullptr) {
                    normalized *= weight[i] + weight_offset;
                }
                out_row[i] = marmot_f32_to_f16_ref(normalized);
            }
        }
        return MARMOT_SUCCESS;
    }

    if (dtype == MARMOT_DTYPE_BFLOAT16) {
        const marmot_bfloat16_t *x = (const marmot_bfloat16_t *)input_data;
        const marmot_bfloat16_t *residual = (const marmot_bfloat16_t *)residual_data;
        marmot_bfloat16_t *out = (marmot_bfloat16_t *)out_data;
        for (size_t row = 0; row < outer_size; ++row) {
            const marmot_bfloat16_t *x_row = x + row * norm_size;
            const marmot_bfloat16_t *res_row = residual != nullptr ? residual + row * norm_size : nullptr;
            marmot_bfloat16_t *out_row = out + row * norm_size;

            float sum_sq = 0.0f;
            for (size_t i = 0; i < norm_size; ++i) {
                float v = marmot_bf16_to_f32_ref(x_row[i]);
                if (res_row != nullptr) {
                    v += marmot_bf16_to_f32_ref(res_row[i]);
                }
                sum_sq += v * v;
            }
            float mean_sq = sum_sq / (float)norm_size;
            float inv_rms = cpu_norm_inv_sqrt_f32(mean_sq + eps);

            for (size_t i = 0; i < norm_size; ++i) {
                float v = marmot_bf16_to_f32_ref(x_row[i]);
                if (res_row != nullptr) {
                    v += marmot_bf16_to_f32_ref(res_row[i]);
                }
                float normalized = v * inv_rms;
                if (weight != nullptr) {
                    normalized *= weight[i] + weight_offset;
                }
                out_row[i] = marmot_f32_to_bf16_ref(normalized);
            }
        }
        return MARMOT_SUCCESS;
    }

    marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Mixed-dtype rmsnorm supported only for f16/bf16 inputs");
    return MARMOT_ERROR_UNSUPPORTED_DTYPE;
}

// ===================================================================
// LayerNorm Helpers - FLOAT16
// ===================================================================

static void layernorm_row_f16_native(
    const marmot_float16_t *x_row, const marmot_float16_t *residual_row, const marmot_float16_t *weight,
    const marmot_float16_t *bias, size_t norm_size, float eps, marmot_float16_t *out_row
) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (size_t i = 0; i < norm_size; ++i) {
        float v = (float)marmot_float16_to_native(x_row[i]);
        if (residual_row != nullptr) {
            v += (float)marmot_float16_to_native(residual_row[i]);
        }
        sum += v;
        sum_sq += v * v;
    }

    float mean = sum / (float)norm_size;
    float var = (sum_sq / (float)norm_size) - (mean * mean);
    if (var < 0.0f) {
        var = 0.0f;
    }
    float inv_std = cpu_norm_inv_sqrt_f32(var + eps);

    for (size_t i = 0; i < norm_size; ++i) {
        float v = (float)marmot_float16_to_native(x_row[i]);
        if (residual_row != nullptr) {
            v += (float)marmot_float16_to_native(residual_row[i]);
        }
        float normalized = (v - mean) * inv_std;
        if (weight != nullptr) {
            normalized *= (float)marmot_float16_to_native(weight[i]);
        }
        if (bias != nullptr) {
            normalized += (float)marmot_float16_to_native(bias[i]);
        }
        out_row[i] = marmot_f32_to_f16_ref(normalized);
    }
}

// ===================================================================
// LayerNorm Helpers - BFLOAT16
// ===================================================================

static void layernorm_row_bf16_scalar(
    const marmot_bfloat16_t *x_row, const marmot_bfloat16_t *residual_row, const marmot_bfloat16_t *weight,
    const marmot_bfloat16_t *bias, size_t norm_size, float eps, marmot_bfloat16_t *out_row
) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (size_t i = 0; i < norm_size; i++) {
        float v = marmot_bf16_to_f32_ref(x_row[i]);
        if (residual_row != nullptr) {
            v += marmot_bf16_to_f32_ref(residual_row[i]);
        }
        sum += v;
        sum_sq += v * v;
    }
    float mean = sum / (float)norm_size;
    float variance = (sum_sq / (float)norm_size) - (mean * mean);
    if (variance < 0.0f) {
        variance = 0.0f;
    }

    float inv_std = cpu_norm_inv_sqrt_f32(variance + eps);

    for (size_t i = 0; i < norm_size; i++) {
        float value = marmot_bf16_to_f32_ref(x_row[i]);
        if (residual_row != nullptr) {
            value += marmot_bf16_to_f32_ref(residual_row[i]);
        }
        float normalized = (value - mean) * inv_std;
        if (weight != nullptr) {
            normalized *= marmot_bf16_to_f32_ref(weight[i]);
        }
        if (bias != nullptr) {
            normalized += marmot_bf16_to_f32_ref(bias[i]);
        }
        out_row[i] = marmot_f32_to_bf16_ref(normalized);
    }
}

#if MARMOT_ENABLE_FP8
static void layernorm_fp8_e4m3(
    cpu_context_t *ctx, const marmot_float8_e4m3_t *x, const marmot_float8_e4m3_t *residual,
    const marmot_float8_e4m3_t *weight, const marmot_float8_e4m3_t *bias, size_t outer_size, size_t norm_size,
    float eps, marmot_float8_e4m3_t *out
) {
    (void)ctx;
    for (size_t row = 0; row < outer_size; row++) {
        const marmot_float8_e4m3_t *x_row = x + row * norm_size;
        const marmot_float8_e4m3_t *res_row = residual != nullptr ? residual + row * norm_size : nullptr;
        marmot_float8_e4m3_t *out_row = out + row * norm_size;

        _Float16 sum = (_Float16)0.0f;
        for (size_t i = 0; i < norm_size; i++) {
            _Float16 value = marmot_fp8_e4m3_to_native(x_row[i]);
            if (res_row != nullptr) {
                value = (_Float16)(value + marmot_fp8_e4m3_to_native(res_row[i]));
            }
            sum = (_Float16)(sum + value);
        }
        _Float16 denom = (_Float16)norm_size;
        _Float16 mean = (_Float16)(sum / denom);

        _Float16 variance = (_Float16)0.0f;
        for (size_t i = 0; i < norm_size; i++) {
            _Float16 value = marmot_fp8_e4m3_to_native(x_row[i]);
            if (res_row != nullptr) {
                value = (_Float16)(value + marmot_fp8_e4m3_to_native(res_row[i]));
            }
            _Float16 diff = (_Float16)(value - mean);
            variance = (_Float16)(variance + diff * diff);
        }
        variance = (_Float16)(variance / denom);
        _Float16 eps_half = (_Float16)eps;
        float inv_std_f = cpu_norm_inv_sqrt_f32((float)(variance + eps_half));
        _Float16 inv_std = (_Float16)inv_std_f;

        for (size_t i = 0; i < norm_size; i++) {
            _Float16 value = marmot_fp8_e4m3_to_native(x_row[i]);
            if (res_row != nullptr) {
                value = (_Float16)(value + marmot_fp8_e4m3_to_native(res_row[i]));
            }
            _Float16 normalized = (_Float16)((value - mean) * inv_std);
            if (weight != nullptr) {
                normalized = (_Float16)(normalized * marmot_fp8_e4m3_to_native(weight[i]));
            }
            if (bias != nullptr) {
                normalized = (_Float16)(normalized + marmot_fp8_e4m3_to_native(bias[i]));
            }
            out_row[i] = marmot_native_to_fp8_e4m3(normalized);
        }
    }
}

static void layernorm_fp8_e5m2(
    cpu_context_t *ctx, const marmot_float8_e5m2_t *x, const marmot_float8_e5m2_t *residual,
    const marmot_float8_e5m2_t *weight, const marmot_float8_e5m2_t *bias, size_t outer_size, size_t norm_size,
    float eps, marmot_float8_e5m2_t *out
) {
    (void)ctx;
    for (size_t row = 0; row < outer_size; row++) {
        const marmot_float8_e5m2_t *x_row = x + row * norm_size;
        const marmot_float8_e5m2_t *res_row = residual != nullptr ? residual + row * norm_size : nullptr;
        marmot_float8_e5m2_t *out_row = out + row * norm_size;

        _Float16 sum = (_Float16)0.0f;
        for (size_t i = 0; i < norm_size; i++) {
            _Float16 value = marmot_fp8_e5m2_to_native(x_row[i]);
            if (res_row != nullptr) {
                value = (_Float16)(value + marmot_fp8_e5m2_to_native(res_row[i]));
            }
            sum = (_Float16)(sum + value);
        }
        _Float16 denom = (_Float16)norm_size;
        _Float16 mean = (_Float16)(sum / denom);

        _Float16 variance = (_Float16)0.0f;
        for (size_t i = 0; i < norm_size; i++) {
            _Float16 value = marmot_fp8_e5m2_to_native(x_row[i]);
            if (res_row != nullptr) {
                value = (_Float16)(value + marmot_fp8_e5m2_to_native(res_row[i]));
            }
            _Float16 diff = (_Float16)(value - mean);
            variance = (_Float16)(variance + diff * diff);
        }
        variance = (_Float16)(variance / denom);
        _Float16 eps_half = (_Float16)eps;
        float inv_std_f = cpu_norm_inv_sqrt_f32((float)(variance + eps_half));
        _Float16 inv_std = (_Float16)inv_std_f;

        for (size_t i = 0; i < norm_size; i++) {
            _Float16 value = marmot_fp8_e5m2_to_native(x_row[i]);
            if (res_row != nullptr) {
                value = (_Float16)(value + marmot_fp8_e5m2_to_native(res_row[i]));
            }
            _Float16 normalized = (_Float16)((value - mean) * inv_std);
            if (weight != nullptr) {
                normalized = (_Float16)(normalized * marmot_fp8_e5m2_to_native(weight[i]));
            }
            if (bias != nullptr) {
                normalized = (_Float16)(normalized + marmot_fp8_e5m2_to_native(bias[i]));
            }
            out_row[i] = marmot_native_to_fp8_e5m2(normalized);
        }
    }
}
#endif

// ===================================================================
// RMSNorm Helper Functions
// ===================================================================

static float compute_mean_square_scalar(const float *x, const float *residual, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        if (residual != nullptr) {
            v += residual[i];
        }
        sum += v * v;
    }
    return sum / (float)n;
}

static void normalize_rms_scalar(
    const float *x, const float *residual, float *out, size_t n, float inv_rms, const float *weight, float weight_offset
) {
    if (residual != nullptr) {
        if (weight != nullptr) {
            for (size_t i = 0; i < n; i++) {
                float v = x[i] + residual[i];
                out[i] = v * inv_rms * (weight[i] + weight_offset);
            }
        } else {
            for (size_t i = 0; i < n; i++) {
                float v = x[i] + residual[i];
                out[i] = v * inv_rms;
            }
        }
        return;
    }

    if (weight != nullptr) {
        for (size_t i = 0; i < n; i++) {
            out[i] = x[i] * inv_rms * (weight[i] + weight_offset);
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            out[i] = x[i] * inv_rms;
        }
    }
}

static double compute_mean_square_scalar_f64(const double *x, const double *residual, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double v = x[i];
        if (residual != nullptr) {
            v += residual[i];
        }
        sum += v * v;
    }
    return sum / (double)n;
}

static void normalize_rms_scalar_f64(
    const double *x, const double *residual, double *out, size_t n, double inv_rms, const double *weight,
    double weight_offset
) {
    if (residual != nullptr) {
        if (weight != nullptr) {
            for (size_t i = 0; i < n; ++i) {
                double v = x[i] + residual[i];
                out[i] = v * inv_rms * (weight[i] + weight_offset);
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                double v = x[i] + residual[i];
                out[i] = v * inv_rms;
            }
        }
        return;
    }

    if (weight != nullptr) {
        for (size_t i = 0; i < n; ++i) {
            out[i] = x[i] * inv_rms * (weight[i] + weight_offset);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            out[i] = x[i] * inv_rms;
        }
    }
}

// ===================================================================
// RMSNorm Helpers - FLOAT16
// ===================================================================

static void rmsnorm_row_f16_native(
    const marmot_float16_t *x_row, const marmot_float16_t *residual_row, const marmot_float16_t *weight,
    size_t norm_size, float eps, float weight_offset, marmot_float16_t *out_row
) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < norm_size; ++i) {
        float v = (float)marmot_float16_to_native(x_row[i]);
        if (residual_row != nullptr) {
            v += (float)marmot_float16_to_native(residual_row[i]);
        }
        sum_sq += v * v;
    }

    float mean_sq = sum_sq / (float)norm_size;
    float inv_rms = cpu_norm_inv_sqrt_f32(mean_sq + eps);
    for (size_t i = 0; i < norm_size; ++i) {
        float v = (float)marmot_float16_to_native(x_row[i]);
        if (residual_row != nullptr) {
            v += (float)marmot_float16_to_native(residual_row[i]);
        }
        float normalized = v * inv_rms;
        if (weight != nullptr) {
            normalized *= (float)marmot_float16_to_native(weight[i]) + weight_offset;
        }
        out_row[i] = marmot_f32_to_f16_ref(normalized);
    }
}

// ===================================================================
// RMSNorm Helpers - BFLOAT16
// ===================================================================

static void rmsnorm_row_bf16_scalar(
    const marmot_bfloat16_t *x_row, const marmot_bfloat16_t *residual_row, const marmot_bfloat16_t *weight,
    size_t norm_size, float eps, float weight_offset, marmot_bfloat16_t *out_row
) {
    float sum = 0.0f;
    for (size_t i = 0; i < norm_size; i++) {
        float value = marmot_bf16_to_f32_ref(x_row[i]);
        if (residual_row != nullptr) {
            value += marmot_bf16_to_f32_ref(residual_row[i]);
        }
        sum += value * value;
    }
    float mean_sq = sum / (float)norm_size;
    float inv_rms = cpu_norm_inv_sqrt_f32(mean_sq + eps);

    for (size_t i = 0; i < norm_size; i++) {
        float value = marmot_bf16_to_f32_ref(x_row[i]);
        if (residual_row != nullptr) {
            value += marmot_bf16_to_f32_ref(residual_row[i]);
        }
        value *= inv_rms;
        if (weight != nullptr) {
            value *= marmot_bf16_to_f32_ref(weight[i]) + weight_offset;
        }
        out_row[i] = marmot_f32_to_bf16_ref(value);
    }
}

#if MARMOT_ENABLE_FP8
static void rmsnorm_fp8_e4m3(
    cpu_context_t *ctx, const marmot_float8_e4m3_t *x, const marmot_float8_e4m3_t *residual,
    const marmot_float8_e4m3_t *weight, size_t outer_size, size_t norm_size, float eps, float weight_offset,
    marmot_float8_e4m3_t *out
) {
    (void)ctx;
    for (size_t row = 0; row < outer_size; row++) {
        const marmot_float8_e4m3_t *x_row = x + row * norm_size;
        const marmot_float8_e4m3_t *res_row = residual != nullptr ? residual + row * norm_size : nullptr;
        marmot_float8_e4m3_t *out_row = out + row * norm_size;

        _Float16 sum = (_Float16)0.0f;
        for (size_t i = 0; i < norm_size; i++) {
            _Float16 value = marmot_fp8_e4m3_to_native(x_row[i]);
            if (res_row != nullptr) {
                value = (_Float16)(value + marmot_fp8_e4m3_to_native(res_row[i]));
            }
            sum = (_Float16)(sum + value * value);
        }
        _Float16 denom = (_Float16)norm_size;
        _Float16 mean_square = (_Float16)(sum / denom);
        float inv_rms_f = cpu_norm_inv_sqrt_f32((float)mean_square + eps);
        _Float16 inv_rms = (_Float16)inv_rms_f;

        for (size_t i = 0; i < norm_size; i++) {
            _Float16 value = marmot_fp8_e4m3_to_native(x_row[i]);
            if (res_row != nullptr) {
                value = (_Float16)(value + marmot_fp8_e4m3_to_native(res_row[i]));
            }
            _Float16 normalized = (_Float16)(value * inv_rms);
            if (weight != nullptr) {
                normalized = (_Float16)(normalized * (marmot_fp8_e4m3_to_native(weight[i]) + (_Float16)weight_offset));
            }
            out_row[i] = marmot_native_to_fp8_e4m3(normalized);
        }
    }
}

static void rmsnorm_fp8_e5m2(
    cpu_context_t *ctx, const marmot_float8_e5m2_t *x, const marmot_float8_e5m2_t *residual,
    const marmot_float8_e5m2_t *weight, size_t outer_size, size_t norm_size, float eps, float weight_offset,
    marmot_float8_e5m2_t *out
) {
    (void)ctx;
    for (size_t row = 0; row < outer_size; row++) {
        const marmot_float8_e5m2_t *x_row = x + row * norm_size;
        const marmot_float8_e5m2_t *res_row = residual != nullptr ? residual + row * norm_size : nullptr;
        marmot_float8_e5m2_t *out_row = out + row * norm_size;

        _Float16 sum = (_Float16)0.0f;
        for (size_t i = 0; i < norm_size; i++) {
            _Float16 value = marmot_fp8_e5m2_to_native(x_row[i]);
            if (res_row != nullptr) {
                value = (_Float16)(value + marmot_fp8_e5m2_to_native(res_row[i]));
            }
            sum = (_Float16)(sum + value * value);
        }
        _Float16 denom = (_Float16)norm_size;
        _Float16 mean_square = (_Float16)(sum / denom);
        float inv_rms_f = cpu_norm_inv_sqrt_f32((float)mean_square + eps);
        _Float16 inv_rms = (_Float16)inv_rms_f;

        for (size_t i = 0; i < norm_size; i++) {
            _Float16 value = marmot_fp8_e5m2_to_native(x_row[i]);
            if (res_row != nullptr) {
                value = (_Float16)(value + marmot_fp8_e5m2_to_native(res_row[i]));
            }
            _Float16 normalized = (_Float16)(value * inv_rms);
            if (weight != nullptr) {
                normalized = (_Float16)(normalized * (marmot_fp8_e5m2_to_native(weight[i]) + (_Float16)weight_offset));
            }
            out_row[i] = marmot_native_to_fp8_e5m2(normalized);
        }
    }
}
#endif

// ===================================================================
// LayerNorm/RMSNorm Kernel Implementations
// ===================================================================

static void layernorm_f64_scalar_range(void *context, size_t start, size_t end) {
    layernorm_f64_context_t *ctx = (layernorm_f64_context_t *)context;
    for (size_t row = start; row < end; ++row) {
        const double *x_row = ctx->x + row * ctx->norm_size;
        const double *res_row = ctx->residual != nullptr ? ctx->residual + row * ctx->norm_size : nullptr;
        double *out_row = ctx->out + row * ctx->norm_size;
        double mean = 0.0;
        double variance = 0.0;
        compute_mean_variance_scalar_f64(x_row, res_row, ctx->norm_size, &mean, &variance);
        double inv_std = 1.0 / sqrt(variance + ctx->eps);
        normalize_scalar_f64(x_row, res_row, out_row, ctx->norm_size, mean, inv_std, ctx->weight, ctx->bias);
    }
}

static marmot_error_t cpu_layernorm_f64_scalar(const void *device_ctx, const cpu_layernorm_params_t *params) {
    (void)device_ctx;
    if (params == nullptr || params->x == nullptr || params->out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (cpu_norm_no_work(params->norm_size, params->outer_size)) {
        return MARMOT_SUCCESS;
    }
    layernorm_f64_context_t ctx = {
        .x = (const double *)params->x,
        .residual = (const double *)params->residual,
        .out = (double *)params->out,
        .weight = (const double *)params->weight,
        .bias = (const double *)params->bias,
        .norm_size = params->norm_size,
        .eps = (double)params->eps,
    };
    cpu_norm_dispatch_rows(params->outer_size, params->norm_size, &ctx, layernorm_f64_scalar_range);
    return MARMOT_SUCCESS;
}

static void rmsnorm_f64_scalar_range(void *context, size_t start, size_t end) {
    rmsnorm_f64_context_t *ctx = (rmsnorm_f64_context_t *)context;
    for (size_t row = start; row < end; ++row) {
        const double *x_row = ctx->x + row * ctx->norm_size;
        const double *res_row = ctx->residual != nullptr ? ctx->residual + row * ctx->norm_size : nullptr;
        double *out_row = ctx->out + row * ctx->norm_size;
        double mean_sq = compute_mean_square_scalar_f64(x_row, res_row, ctx->norm_size);
        double inv_rms = 1.0 / sqrt(mean_sq + ctx->eps);
        normalize_rms_scalar_f64(x_row, res_row, out_row, ctx->norm_size, inv_rms, ctx->weight, ctx->weight_offset);
    }
}

static marmot_error_t cpu_rmsnorm_f64_scalar(const void *device_ctx, const cpu_rmsnorm_params_t *params) {
    (void)device_ctx;
    if (params == nullptr || params->x == nullptr || params->out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (cpu_norm_no_work(params->norm_size, params->outer_size)) {
        return MARMOT_SUCCESS;
    }
    rmsnorm_f64_context_t ctx = {
        .x = (const double *)params->x,
        .residual = (const double *)params->residual,
        .out = (double *)params->out,
        .weight = (const double *)params->weight,
        .norm_size = params->norm_size,
        .eps = (double)params->eps,
        .weight_offset = (double)params->weight_offset,
    };
    cpu_norm_dispatch_rows(params->outer_size, params->norm_size, &ctx, rmsnorm_f64_scalar_range);
    return MARMOT_SUCCESS;
}

static void layernorm_f32_scalar_range(void *context, size_t start, size_t end) {
    layernorm_f32_context_t *ctx = (layernorm_f32_context_t *)context;
    for (size_t row = start; row < end; ++row) {
        const float *x_row = ctx->x + row * ctx->norm_size;
        const float *res_row = ctx->residual != nullptr ? ctx->residual + row * ctx->norm_size : nullptr;
        float *out_row = ctx->out + row * ctx->norm_size;
        float mean = 0.0f;
        float variance = 0.0f;
        compute_mean_variance_scalar(x_row, res_row, ctx->norm_size, &mean, &variance);
        float inv_std = cpu_norm_inv_sqrt_f32(variance + ctx->eps);
        normalize_scalar(x_row, res_row, out_row, ctx->norm_size, mean, inv_std, ctx->weight, ctx->bias);
    }
}

static void rmsnorm_f32_scalar_range(void *context, size_t start, size_t end) {
    rmsnorm_f32_context_t *ctx = (rmsnorm_f32_context_t *)context;
    for (size_t row = start; row < end; ++row) {
        const float *x_row = ctx->x + row * ctx->norm_size;
        const float *res_row = ctx->residual != nullptr ? ctx->residual + row * ctx->norm_size : nullptr;
        float *out_row = ctx->out + row * ctx->norm_size;
        float mean_sq = compute_mean_square_scalar(x_row, res_row, ctx->norm_size);
        float inv_rms = cpu_norm_inv_sqrt_f32(mean_sq + ctx->eps);
        normalize_rms_scalar(x_row, res_row, out_row, ctx->norm_size, inv_rms, ctx->weight, ctx->weight_offset);
    }
}

static marmot_error_t cpu_layernorm_f32_scalar(const void *device_ctx, const cpu_layernorm_params_t *params) {
    (void)device_ctx;
    if (params == nullptr || params->x == nullptr || params->out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (cpu_norm_no_work(params->norm_size, params->outer_size)) {
        return MARMOT_SUCCESS;
    }
    layernorm_f32_context_t ctx = {
        .x = (const float *)params->x,
        .residual = (const float *)params->residual,
        .out = (float *)params->out,
        .weight = (const float *)params->weight,
        .bias = (const float *)params->bias,
        .norm_size = params->norm_size,
        .eps = params->eps,
    };

    cpu_norm_dispatch_rows(params->outer_size, params->norm_size, &ctx, layernorm_f32_scalar_range);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_rmsnorm_f32_scalar(const void *device_ctx, const cpu_rmsnorm_params_t *params) {
    (void)device_ctx;
    if (params == nullptr || params->x == nullptr || params->out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (cpu_norm_no_work(params->norm_size, params->outer_size)) {
        return MARMOT_SUCCESS;
    }
    rmsnorm_f32_context_t ctx = {
        .x = (const float *)params->x,
        .residual = (const float *)params->residual,
        .out = (float *)params->out,
        .weight = (const float *)params->weight,
        .norm_size = params->norm_size,
        .eps = params->eps,
        .weight_offset = params->weight_offset,
    };

    cpu_norm_dispatch_rows(params->outer_size, params->norm_size, &ctx, rmsnorm_f32_scalar_range);
    return MARMOT_SUCCESS;
}

static void layernorm_f16_scalar_range(void *context, size_t start, size_t end) {
    layernorm_f16_context_t *ctx = (layernorm_f16_context_t *)context;
    for (size_t row = start; row < end; ++row) {
        const marmot_float16_t *x_row = ctx->x + row * ctx->norm_size;
        const marmot_float16_t *res_row = ctx->residual != nullptr ? ctx->residual + row * ctx->norm_size : nullptr;
        marmot_float16_t *out_row = ctx->out + row * ctx->norm_size;
        layernorm_row_f16_native(x_row, res_row, ctx->weight, ctx->bias, ctx->norm_size, ctx->eps, out_row);
    }
}

static void rmsnorm_f16_scalar_range(void *context, size_t start, size_t end) {
    rmsnorm_f16_context_t *ctx = (rmsnorm_f16_context_t *)context;
    for (size_t row = start; row < end; ++row) {
        const marmot_float16_t *x_row = ctx->x + row * ctx->norm_size;
        const marmot_float16_t *res_row = ctx->residual != nullptr ? ctx->residual + row * ctx->norm_size : nullptr;
        marmot_float16_t *out_row = ctx->out + row * ctx->norm_size;
        rmsnorm_row_f16_native(x_row, res_row, ctx->weight, ctx->norm_size, ctx->eps, ctx->weight_offset, out_row);
    }
}

static void layernorm_bf16_scalar_range(void *context, size_t start, size_t end) {
    layernorm_bf16_context_t *ctx = (layernorm_bf16_context_t *)context;
    for (size_t row = start; row < end; ++row) {
        const marmot_bfloat16_t *x_row = ctx->x + row * ctx->norm_size;
        const marmot_bfloat16_t *res_row = ctx->residual != nullptr ? ctx->residual + row * ctx->norm_size : nullptr;
        marmot_bfloat16_t *out_row = ctx->out + row * ctx->norm_size;
        layernorm_row_bf16_scalar(x_row, res_row, ctx->weight, ctx->bias, ctx->norm_size, ctx->eps, out_row);
    }
}

static void rmsnorm_bf16_scalar_range(void *context, size_t start, size_t end) {
    rmsnorm_bf16_context_t *ctx = (rmsnorm_bf16_context_t *)context;
    for (size_t row = start; row < end; ++row) {
        const marmot_bfloat16_t *x_row = ctx->x + row * ctx->norm_size;
        const marmot_bfloat16_t *res_row = ctx->residual != nullptr ? ctx->residual + row * ctx->norm_size : nullptr;
        marmot_bfloat16_t *out_row = ctx->out + row * ctx->norm_size;
        rmsnorm_row_bf16_scalar(x_row, res_row, ctx->weight, ctx->norm_size, ctx->eps, ctx->weight_offset, out_row);
    }
}

static marmot_error_t cpu_layernorm_f16_scalar(const void *device_ctx, const cpu_layernorm_params_t *params) {
    (void)device_ctx;
    if (params == nullptr || params->x == nullptr || params->out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (cpu_norm_no_work(params->norm_size, params->outer_size)) {
        return MARMOT_SUCCESS;
    }
    layernorm_f16_context_t ctx = {
        .x = (const marmot_float16_t *)params->x,
        .residual = (const marmot_float16_t *)params->residual,
        .out = (marmot_float16_t *)params->out,
        .weight = (const marmot_float16_t *)params->weight,
        .bias = (const marmot_float16_t *)params->bias,
        .norm_size = params->norm_size,
        .eps = params->eps,
    };
    cpu_norm_dispatch_rows(params->outer_size, params->norm_size, &ctx, layernorm_f16_scalar_range);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_rmsnorm_f16_scalar(const void *device_ctx, const cpu_rmsnorm_params_t *params) {
    (void)device_ctx;
    if (params == nullptr || params->x == nullptr || params->out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (cpu_norm_no_work(params->norm_size, params->outer_size)) {
        return MARMOT_SUCCESS;
    }
    rmsnorm_f16_context_t ctx = {
        .x = (const marmot_float16_t *)params->x,
        .residual = (const marmot_float16_t *)params->residual,
        .out = (marmot_float16_t *)params->out,
        .weight = (const marmot_float16_t *)params->weight,
        .norm_size = params->norm_size,
        .eps = params->eps,
        .weight_offset = params->weight_offset,
    };
    cpu_norm_dispatch_rows(params->outer_size, params->norm_size, &ctx, rmsnorm_f16_scalar_range);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_layernorm_bf16_scalar(const void *device_ctx, const cpu_layernorm_params_t *params) {
    (void)device_ctx;
    if (params == nullptr || params->x == nullptr || params->out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (cpu_norm_no_work(params->norm_size, params->outer_size)) {
        return MARMOT_SUCCESS;
    }
    layernorm_bf16_context_t ctx = {
        .x = (const marmot_bfloat16_t *)params->x,
        .residual = (const marmot_bfloat16_t *)params->residual,
        .out = (marmot_bfloat16_t *)params->out,
        .weight = (const marmot_bfloat16_t *)params->weight,
        .bias = (const marmot_bfloat16_t *)params->bias,
        .norm_size = params->norm_size,
        .eps = params->eps,
    };
    cpu_norm_dispatch_rows(params->outer_size, params->norm_size, &ctx, layernorm_bf16_scalar_range);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_rmsnorm_bf16_scalar(const void *device_ctx, const cpu_rmsnorm_params_t *params) {
    (void)device_ctx;
    if (params == nullptr || params->x == nullptr || params->out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (cpu_norm_no_work(params->norm_size, params->outer_size)) {
        return MARMOT_SUCCESS;
    }
    rmsnorm_bf16_context_t ctx = {
        .x = (const marmot_bfloat16_t *)params->x,
        .residual = (const marmot_bfloat16_t *)params->residual,
        .out = (marmot_bfloat16_t *)params->out,
        .weight = (const marmot_bfloat16_t *)params->weight,
        .norm_size = params->norm_size,
        .eps = params->eps,
        .weight_offset = params->weight_offset,
    };
    cpu_norm_dispatch_rows(params->outer_size, params->norm_size, &ctx, rmsnorm_bf16_scalar_range);
    return MARMOT_SUCCESS;
}

#if MARMOT_ENABLE_FP8
static marmot_error_t cpu_layernorm_fp8_e4m3(const void *device_ctx, const cpu_layernorm_params_t *params) {
    (void)device_ctx;
    if (params == nullptr || params->x == nullptr || params->out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (cpu_norm_no_work(params->norm_size, params->outer_size)) {
        return MARMOT_SUCCESS;
    }
    const marmot_float8_e4m3_t *x_data = (const marmot_float8_e4m3_t *)params->x;
    const marmot_float8_e4m3_t *residual = (const marmot_float8_e4m3_t *)params->residual;
    marmot_float8_e4m3_t *out_data = (marmot_float8_e4m3_t *)params->out;
    const marmot_float8_e4m3_t *weight = (const marmot_float8_e4m3_t *)params->weight;
    const marmot_float8_e4m3_t *bias = (const marmot_float8_e4m3_t *)params->bias;
    const size_t norm = params->norm_size;
    const size_t outer = params->outer_size;
    const float eps = params->eps;
    layernorm_fp8_e4m3(nullptr, x_data, residual, weight, bias, outer, norm, eps, out_data);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_layernorm_fp8_e5m2(const void *device_ctx, const cpu_layernorm_params_t *params) {
    (void)device_ctx;
    if (params == nullptr || params->x == nullptr || params->out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (cpu_norm_no_work(params->norm_size, params->outer_size)) {
        return MARMOT_SUCCESS;
    }
    const marmot_float8_e5m2_t *x_data = (const marmot_float8_e5m2_t *)params->x;
    const marmot_float8_e5m2_t *residual = (const marmot_float8_e5m2_t *)params->residual;
    marmot_float8_e5m2_t *out_data = (marmot_float8_e5m2_t *)params->out;
    const marmot_float8_e5m2_t *weight = (const marmot_float8_e5m2_t *)params->weight;
    const marmot_float8_e5m2_t *bias = (const marmot_float8_e5m2_t *)params->bias;
    const size_t norm = params->norm_size;
    const size_t outer = params->outer_size;
    const float eps = params->eps;
    layernorm_fp8_e5m2(nullptr, x_data, residual, weight, bias, outer, norm, eps, out_data);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_rmsnorm_fp8_e4m3(const void *device_ctx, const cpu_rmsnorm_params_t *params) {
    (void)device_ctx;
    if (params == nullptr || params->x == nullptr || params->out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (cpu_norm_no_work(params->norm_size, params->outer_size)) {
        return MARMOT_SUCCESS;
    }
    const marmot_float8_e4m3_t *x_data = (const marmot_float8_e4m3_t *)params->x;
    const marmot_float8_e4m3_t *residual = (const marmot_float8_e4m3_t *)params->residual;
    marmot_float8_e4m3_t *out_data = (marmot_float8_e4m3_t *)params->out;
    const marmot_float8_e4m3_t *weight = (const marmot_float8_e4m3_t *)params->weight;
    const size_t norm = params->norm_size;
    const size_t outer = params->outer_size;
    const float eps = params->eps;
    rmsnorm_fp8_e4m3(nullptr, x_data, residual, weight, outer, norm, eps, params->weight_offset, out_data);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_rmsnorm_fp8_e5m2(const void *device_ctx, const cpu_rmsnorm_params_t *params) {
    (void)device_ctx;
    if (params == nullptr || params->x == nullptr || params->out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (cpu_norm_no_work(params->norm_size, params->outer_size)) {
        return MARMOT_SUCCESS;
    }
    const marmot_float8_e5m2_t *x_data = (const marmot_float8_e5m2_t *)params->x;
    const marmot_float8_e5m2_t *residual = (const marmot_float8_e5m2_t *)params->residual;
    marmot_float8_e5m2_t *out_data = (marmot_float8_e5m2_t *)params->out;
    const marmot_float8_e5m2_t *weight = (const marmot_float8_e5m2_t *)params->weight;
    const size_t norm = params->norm_size;
    const size_t outer = params->outer_size;
    const float eps = params->eps;
    rmsnorm_fp8_e5m2(nullptr, x_data, residual, weight, outer, norm, eps, params->weight_offset, out_data);
    return MARMOT_SUCCESS;
}
#endif

// ===================================================================
// Trait Registration
// ===================================================================

const cpu_norm_traits_t cpu_norm_f64_scalar_traits = {
    .dtype = MARMOT_DTYPE_FLOAT64,
    .impl_kind = CPU_NORM_IMPL_SCALAR,
    .ops = {
        .layernorm = cpu_layernorm_f64_scalar,
        .rmsnorm = cpu_rmsnorm_f64_scalar,
        .impl_name = "f64_scalar",
    },
};
CPU_NORM_REGISTER_TRAITS(cpu_norm_f64_scalar_traits)

const cpu_norm_traits_t cpu_norm_f32_scalar_traits = {
    .dtype = MARMOT_DTYPE_FLOAT32,
    .impl_kind = CPU_NORM_IMPL_SCALAR,
    .ops = {
        .layernorm = cpu_layernorm_f32_scalar,
        .rmsnorm = cpu_rmsnorm_f32_scalar,
        .impl_name = "f32_scalar",
    },
};
CPU_NORM_REGISTER_TRAITS(cpu_norm_f32_scalar_traits)

const cpu_norm_traits_t cpu_norm_f16_scalar_traits = {
    .dtype = MARMOT_DTYPE_FLOAT16,
    .impl_kind = CPU_NORM_IMPL_SCALAR,
    .ops = {
        .layernorm = cpu_layernorm_f16_scalar,
        .rmsnorm = cpu_rmsnorm_f16_scalar,
        .impl_name = "f16_scalar",
    },
};
CPU_NORM_REGISTER_TRAITS(cpu_norm_f16_scalar_traits)

const cpu_norm_traits_t cpu_norm_bf16_scalar_traits = {
    .dtype = MARMOT_DTYPE_BFLOAT16,
    .impl_kind = CPU_NORM_IMPL_SCALAR,
    .ops = {
        .layernorm = cpu_layernorm_bf16_scalar,
        .rmsnorm = cpu_rmsnorm_bf16_scalar,
        .impl_name = "bf16_scalar",
    },
};
CPU_NORM_REGISTER_TRAITS(cpu_norm_bf16_scalar_traits)

#if MARMOT_ENABLE_FP8
const cpu_norm_traits_t cpu_norm_fp8_e4m3_traits = {
    .dtype = MARMOT_DTYPE_FLOAT8_E4M3,
    .impl_kind = CPU_NORM_IMPL_SCALAR,
    .ops = {
        .layernorm = cpu_layernorm_fp8_e4m3,
        .rmsnorm = cpu_rmsnorm_fp8_e4m3,
        .impl_name = "fp8_e4m3",
    },
};
CPU_NORM_REGISTER_TRAITS(cpu_norm_fp8_e4m3_traits)

const cpu_norm_traits_t cpu_norm_fp8_e5m2_traits = {
    .dtype = MARMOT_DTYPE_FLOAT8_E5M2,
    .impl_kind = CPU_NORM_IMPL_SCALAR,
    .ops = {
        .layernorm = cpu_layernorm_fp8_e5m2,
        .rmsnorm = cpu_rmsnorm_fp8_e5m2,
        .impl_name = "fp8_e5m2",
    },
};
CPU_NORM_REGISTER_TRAITS(cpu_norm_fp8_e5m2_traits)
#endif
