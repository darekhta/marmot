#include "cpu_backend_internal.h"

#if MARMOT_ENABLE_NEON

static marmot_error_t cpu_reduce_f16_neon_require_numeric(const void *base, double *out_value, size_t length) {
    if (base == nullptr || out_value == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in F16 NEON reduction kernel");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (length == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "Reduction over zero elements is undefined");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_f16_neon_require_arg(const void *base, double *out_value, uint64_t *out_index, size_t length) {
    marmot_error_t status = cpu_reduce_f16_neon_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    if (out_index == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Arg reduction requires index output");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return MARMOT_SUCCESS;
}

static inline float cpu_reduce_f16_neon_horizontal_sum_f32(float32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float32x2_t pair = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    float32x2_t sum2 = vpadd_f32(pair, pair);
    return vget_lane_f32(sum2, 0);
#endif
}

static inline float cpu_reduce_f16_neon_horizontal_max_f32(float32x4_t v) {
#if defined(__aarch64__)
    return vmaxvq_f32(v);
#else
    float tmp[4];
    vst1q_f32(tmp, v);
    float best = tmp[0];
    if (tmp[1] > best) {
        best = tmp[1];
    }
    if (tmp[2] > best) {
        best = tmp[2];
    }
    if (tmp[3] > best) {
        best = tmp[3];
    }
    return best;
#endif
}

static inline float cpu_reduce_f16_neon_horizontal_min_f32(float32x4_t v) {
#if defined(__aarch64__)
    return vminvq_f32(v);
#else
    float tmp[4];
    vst1q_f32(tmp, v);
    float best = tmp[0];
    if (tmp[1] < best) {
        best = tmp[1];
    }
    if (tmp[2] < best) {
        best = tmp[2];
    }
    if (tmp[3] < best) {
        best = tmp[3];
    }
    return best;
#endif
}

static marmot_error_t cpu_reduce_f16_neon_sum_impl(const marmot_float16_t *data, size_t length, double *out_value) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 8 <= length; i += 8) {
        uint16x8_t raw = vld1q_u16((const uint16_t *)(data + i));
        float16x8_t v16 = vreinterpretq_f16_u16(raw);
        float32x4_t lo = vcvt_f32_f16(vget_low_f16(v16));
        float32x4_t hi = vcvt_f32_f16(vget_high_f16(v16));
        acc = vaddq_f32(acc, lo);
        acc = vaddq_f32(acc, hi);
    }
    float sum = cpu_reduce_f16_neon_horizontal_sum_f32(acc);
    for (; i < length; ++i) {
        sum += (float)marmot_float16_to_native(data[i]);
    }
    *out_value = (double)sum;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_f16_neon_sum(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_f16_neon_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return cpu_reduce_f16_neon_sum_impl((const marmot_float16_t *)base, length, out_value);
}

static marmot_error_t
cpu_reduce_f16_neon_mean(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_f16_neon_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    status = cpu_reduce_f16_neon_sum_impl((const marmot_float16_t *)base, length, out_value);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value /= (double)length;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_f16_neon_prod(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_f16_neon_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const marmot_float16_t *data = (const marmot_float16_t *)base;
    double prod = 1.0;
    for (size_t i = 0; i < length; ++i) {
        prod *= (double)marmot_float16_to_native(data[i]);
    }
    *out_value = prod;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_f16_neon_max(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_f16_neon_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const marmot_float16_t *data = (const marmot_float16_t *)base;
    float best = (float)marmot_float16_to_native(data[0]);
    size_t i = 0;
    for (; i + 8 <= length; i += 8) {
        uint16x8_t raw = vld1q_u16((const uint16_t *)(data + i));
        float16x8_t v16 = vreinterpretq_f16_u16(raw);
        float32x4_t lo = vcvt_f32_f16(vget_low_f16(v16));
        float32x4_t hi = vcvt_f32_f16(vget_high_f16(v16));
        float candidate = cpu_reduce_f16_neon_horizontal_max_f32(lo);
        if (candidate > best) {
            best = candidate;
        }
        candidate = cpu_reduce_f16_neon_horizontal_max_f32(hi);
        if (candidate > best) {
            best = candidate;
        }
    }
    for (; i < length; ++i) {
        float value = (float)marmot_float16_to_native(data[i]);
        if (value > best) {
            best = value;
        }
    }
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_f16_neon_min(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_f16_neon_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const marmot_float16_t *data = (const marmot_float16_t *)base;
    float best = (float)marmot_float16_to_native(data[0]);
    size_t i = 0;
    for (; i + 8 <= length; i += 8) {
        uint16x8_t raw = vld1q_u16((const uint16_t *)(data + i));
        float16x8_t v16 = vreinterpretq_f16_u16(raw);
        float32x4_t lo = vcvt_f32_f16(vget_low_f16(v16));
        float32x4_t hi = vcvt_f32_f16(vget_high_f16(v16));
        float candidate = cpu_reduce_f16_neon_horizontal_min_f32(lo);
        if (candidate < best) {
            best = candidate;
        }
        candidate = cpu_reduce_f16_neon_horizontal_min_f32(hi);
        if (candidate < best) {
            best = candidate;
        }
    }
    for (; i < length; ++i) {
        float value = (float)marmot_float16_to_native(data[i]);
        if (value < best) {
            best = value;
        }
    }
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_f16_neon_argmax(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_f16_neon_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const marmot_float16_t *data = (const marmot_float16_t *)base;
    double best_value = 0.0;
    status = cpu_reduce_f16_neon_max(device_ctx, base, length, &best_value);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const float target = (float)best_value;
    uint64_t idx = 0;
    for (; idx < length; ++idx) {
        if ((float)marmot_float16_to_native(data[idx]) == target) {
            break;
        }
    }
    *out_value = best_value;
    *out_index = idx;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_f16_neon_argmin(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_f16_neon_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const marmot_float16_t *data = (const marmot_float16_t *)base;
    double best_value = 0.0;
    status = cpu_reduce_f16_neon_min(device_ctx, base, length, &best_value);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const float target = (float)best_value;
    uint64_t idx = 0;
    for (; idx < length; ++idx) {
        if ((float)marmot_float16_to_native(data[idx]) == target) {
            break;
        }
    }
    *out_value = best_value;
    *out_index = idx;
    return MARMOT_SUCCESS;
}

const cpu_reduce_traits_t cpu_reduce_f16_neon_traits = {
    .dtype = MARMOT_DTYPE_FLOAT16,
    .impl_kind = CPU_REDUCE_IMPL_NEON,
    .ops = {
        .sum = cpu_reduce_f16_neon_sum,
        .mean = cpu_reduce_f16_neon_mean,
        .prod = cpu_reduce_f16_neon_prod,
        .min = cpu_reduce_f16_neon_min,
        .max = cpu_reduce_f16_neon_max,
        .argmax = cpu_reduce_f16_neon_argmax,
        .argmin = cpu_reduce_f16_neon_argmin,
        .impl_name = "neon-f16",
    },
};

CPU_REDUCTION_REGISTER_TRAITS(cpu_reduce_f16_neon_traits)

#endif // MARMOT_ENABLE_NEON
