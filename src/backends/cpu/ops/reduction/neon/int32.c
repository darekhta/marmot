#include "cpu_backend_internal.h"

#if MARMOT_ENABLE_NEON

static marmot_error_t cpu_reduce_neon_i32_require_numeric(const void *base, double *out_value, size_t length) {
    if (base == nullptr || out_value == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in NEON int reduction kernel");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (length == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "Reduction over zero elements is undefined");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_neon_i32_require_arg(const void *base, double *out_value, uint64_t *out_index, size_t length) {
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    if (out_index == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Arg reduction requires index output");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return MARMOT_SUCCESS;
}

static inline double cpu_reduce_neon_i32_horizontal_sum(int32x4_t v) {
    int64x2_t pair = vpaddlq_s32(v);
    return (double)vgetq_lane_s64(pair, 0) + (double)vgetq_lane_s64(pair, 1);
}

static inline double cpu_reduce_neon_u32_horizontal_sum(uint32x4_t v) {
    uint64x2_t pair = vpaddlq_u32(v);
    return (double)vgetq_lane_u64(pair, 0) + (double)vgetq_lane_u64(pair, 1);
}

static inline int32x4_t cpu_reduce_neon_accumulate_s16(int32x4_t acc, int16x8_t values) {
    int32x4_t lo = vmovl_s16(vget_low_s16(values));
    int32x4_t hi = vmovl_s16(vget_high_s16(values));
    acc = vaddq_s32(acc, lo);
    acc = vaddq_s32(acc, hi);
    return acc;
}

static inline uint32x4_t cpu_reduce_neon_accumulate_u16(uint32x4_t acc, uint16x8_t values) {
    uint32x4_t lo = vmovl_u16(vget_low_u16(values));
    uint32x4_t hi = vmovl_u16(vget_high_u16(values));
    acc = vaddq_u32(acc, lo);
    acc = vaddq_u32(acc, hi);
    return acc;
}

static double cpu_reduce_neon_i32_sum_impl(const int32_t *data, size_t length) {
    int32x4_t acc = vdupq_n_s32(0);
    size_t i = 0;
    for (; i + 4 <= length; i += 4) {
        int32x4_t v = vld1q_s32(data + i);
        acc = vaddq_s32(acc, v);
    }
    double sum = cpu_reduce_neon_i32_horizontal_sum(acc);
    for (; i < length; ++i) {
        sum += (double)data[i];
    }
    return sum;
}

static marmot_error_t
cpu_reduce_i32_neon_sum(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value = cpu_reduce_neon_i32_sum_impl((const int32_t *)base, length);
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_i32_neon_mean(const void *device_ctx, const void *base, size_t length, double *out_value) {
    marmot_error_t status = cpu_reduce_i32_neon_sum(device_ctx, base, length, out_value);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value /= (double)length;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_i32_neon_prod(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int32_t *data = (const int32_t *)base;
    double prod = 1.0;
    for (size_t i = 0; i < length; ++i) {
        prod *= (double)data[i];
    }
    *out_value = prod;
    return MARMOT_SUCCESS;
}

static inline int32x4_t cpu_reduce_neon_i32_update_best(int32x4_t best, int32x4_t value, bool find_max) {
    return find_max ? vmaxq_s32(best, value) : vminq_s32(best, value);
}

static int32_t
cpu_reduce_neon_i32_finalize_best(int32x4_t best_vec, const int32_t *data, size_t start, size_t length, bool find_max) {
    int32_t best = vgetq_lane_s32(best_vec, 0);
    best = find_max ? (best > vgetq_lane_s32(best_vec, 1) ? best : vgetq_lane_s32(best_vec, 1))
                    : (best < vgetq_lane_s32(best_vec, 1) ? best : vgetq_lane_s32(best_vec, 1));
    best = find_max ? (best > vgetq_lane_s32(best_vec, 2) ? best : vgetq_lane_s32(best_vec, 2))
                    : (best < vgetq_lane_s32(best_vec, 2) ? best : vgetq_lane_s32(best_vec, 2));
    best = find_max ? (best > vgetq_lane_s32(best_vec, 3) ? best : vgetq_lane_s32(best_vec, 3))
                    : (best < vgetq_lane_s32(best_vec, 3) ? best : vgetq_lane_s32(best_vec, 3));
    for (size_t i = start; i < length; ++i) {
        int32_t value = data[i];
        best = find_max ? (value > best ? value : best) : (value < best ? value : best);
    }
    return best;
}

static int32_t cpu_reduce_neon_i32_best(const int32_t *data, size_t length, bool find_max) {
    size_t i = 0;
    int32x4_t best = vdupq_n_s32(find_max ? INT32_MIN : INT32_MAX);
    for (; i + 4 <= length; i += 4) {
        int32x4_t v = vld1q_s32(data + i);
        best = cpu_reduce_neon_i32_update_best(best, v, find_max);
    }
    return cpu_reduce_neon_i32_finalize_best(best, data, i, length, find_max);
}

static marmot_error_t
cpu_reduce_i32_neon_max(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int32_t *data = (const int32_t *)base;
    int32_t best = cpu_reduce_neon_i32_best(data, length, true);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_i32_neon_min(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int32_t *data = (const int32_t *)base;
    int32_t best = cpu_reduce_neon_i32_best(data, length, false);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_i32_neon_argmax(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int32_t *data = (const int32_t *)base;
    int32_t best = cpu_reduce_neon_i32_best(data, length, true);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_i32_neon_argmin(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int32_t *data = (const int32_t *)base;
    int32_t best = cpu_reduce_neon_i32_best(data, length, false);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

const cpu_reduce_traits_t cpu_reduce_i32_neon_traits = {
    .dtype = MARMOT_DTYPE_INT32,
    .impl_kind = CPU_REDUCE_IMPL_NEON,
    .ops = {
        .sum = cpu_reduce_i32_neon_sum,
        .mean = cpu_reduce_i32_neon_mean,
        .prod = cpu_reduce_i32_neon_prod,
        .min = cpu_reduce_i32_neon_min,
        .max = cpu_reduce_i32_neon_max,
        .argmax = cpu_reduce_i32_neon_argmax,
        .argmin = cpu_reduce_i32_neon_argmin,
        .impl_name = "neon-i32",
    },
};

CPU_REDUCTION_REGISTER_TRAITS(cpu_reduce_i32_neon_traits)

static marmot_error_t cpu_reduce_neon_u32_require_numeric(const void *base, double *out_value, size_t length) {
    return cpu_reduce_neon_i32_require_numeric(base, out_value, length);
}

static marmot_error_t
cpu_reduce_neon_u32_require_arg(const void *base, double *out_value, uint64_t *out_index, size_t length) {
    return cpu_reduce_neon_i32_require_arg(base, out_value, out_index, length);
}

static double cpu_reduce_neon_u32_sum_impl(const uint32_t *data, size_t length) {
    uint64x2_t acc = vdupq_n_u64(0);
    size_t i = 0;
    for (; i + 4 <= length; i += 4) {
        uint32x4_t v = vld1q_u32(data + i);
        acc = vaddq_u64(acc, vpaddlq_u32(v));
    }
    double sum = (double)vgetq_lane_u64(acc, 0) + (double)vgetq_lane_u64(acc, 1);
    for (; i < length; ++i) {
        sum += (double)data[i];
    }
    return sum;
}

static marmot_error_t
cpu_reduce_u32_neon_sum(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_u32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value = cpu_reduce_neon_u32_sum_impl((const uint32_t *)base, length);
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_u32_neon_mean(const void *device_ctx, const void *base, size_t length, double *out_value) {
    marmot_error_t status = cpu_reduce_u32_neon_sum(device_ctx, base, length, out_value);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value /= (double)length;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_u32_neon_prod(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_u32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint32_t *data = (const uint32_t *)base;
    double prod = 1.0;
    for (size_t i = 0; i < length; ++i) {
        prod *= (double)data[i];
    }
    *out_value = prod;
    return MARMOT_SUCCESS;
}

static inline uint32_t cpu_reduce_neon_u32_horizontal_best(uint32x4_t v, bool find_max) {
    uint32_t lanes[4];
    vst1q_u32(lanes, v);
    uint32_t best = lanes[0];
    for (int lane = 1; lane < 4; ++lane) {
        uint32_t value = lanes[lane];
        best = find_max ? (value > best ? value : best) : (value < best ? value : best);
    }
    return best;
}

static uint32_t cpu_reduce_neon_u32_best(const uint32_t *data, size_t length, bool find_max) {
    size_t i = 0;
    uint32_t best_scalar = data[0];
    if (length >= 4) {
        uint32x4_t best_vec = vld1q_u32(data);
        best_scalar = cpu_reduce_neon_u32_horizontal_best(best_vec, find_max);
        i = 4;
        for (; i + 4 <= length; i += 4) {
            uint32x4_t v = vld1q_u32(data + i);
            best_vec = find_max ? vmaxq_u32(vdupq_n_u32(best_scalar), v) : vminq_u32(vdupq_n_u32(best_scalar), v);
            best_scalar = cpu_reduce_neon_u32_horizontal_best(best_vec, find_max);
        }
    } else {
        i = 1;
    }
    for (; i < length; ++i) {
        uint32_t value = data[i];
        best_scalar =
            find_max ? (value > best_scalar ? value : best_scalar) : (value < best_scalar ? value : best_scalar);
    }
    return best_scalar;
}

static marmot_error_t
cpu_reduce_u32_neon_max(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_u32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint32_t *data = (const uint32_t *)base;
    uint32_t best = cpu_reduce_neon_u32_best(data, length, true);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_u32_neon_min(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_u32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint32_t *data = (const uint32_t *)base;
    uint32_t best = cpu_reduce_neon_u32_best(data, length, false);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_u32_neon_argmax(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_u32_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint32_t *data = (const uint32_t *)base;
    uint32_t best = cpu_reduce_neon_u32_best(data, length, true);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_u32_neon_argmin(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_u32_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint32_t *data = (const uint32_t *)base;
    uint32_t best = cpu_reduce_neon_u32_best(data, length, false);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

const cpu_reduce_traits_t cpu_reduce_u32_neon_traits = {
    .dtype = MARMOT_DTYPE_UINT32,
    .impl_kind = CPU_REDUCE_IMPL_NEON,
    .ops = {
        .sum = cpu_reduce_u32_neon_sum,
        .mean = cpu_reduce_u32_neon_mean,
        .prod = cpu_reduce_u32_neon_prod,
        .min = cpu_reduce_u32_neon_min,
        .max = cpu_reduce_u32_neon_max,
        .argmax = cpu_reduce_u32_neon_argmax,
        .argmin = cpu_reduce_u32_neon_argmin,
        .impl_name = "neon-u32",
    },
};

CPU_REDUCTION_REGISTER_TRAITS(cpu_reduce_u32_neon_traits)

static double cpu_reduce_neon_i16_sum_impl(const int16_t *data, size_t length) {
    int32x4_t acc = vdupq_n_s32(0);
    size_t i = 0;
    for (; i + 8 <= length; i += 8) {
        int16x8_t v = vld1q_s16(data + i);
        acc = cpu_reduce_neon_accumulate_s16(acc, v);
    }
    double sum = cpu_reduce_neon_i32_horizontal_sum(acc);
    for (; i < length; ++i) {
        sum += (double)data[i];
    }
    return sum;
}

static marmot_error_t
cpu_reduce_i16_neon_sum(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value = cpu_reduce_neon_i16_sum_impl((const int16_t *)base, length);
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_i16_neon_mean(const void *device_ctx, const void *base, size_t length, double *out_value) {
    marmot_error_t status = cpu_reduce_i16_neon_sum(device_ctx, base, length, out_value);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value /= (double)length;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_i16_neon_prod(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int16_t *data = (const int16_t *)base;
    double prod = 1.0;
    for (size_t i = 0; i < length; ++i) {
        prod *= (double)data[i];
    }
    *out_value = prod;
    return MARMOT_SUCCESS;
}

static inline int16_t
cpu_reduce_neon_i16_finalize_best(int16x8_t best_vec, const int16_t *data, size_t start, size_t length, bool find_max) {
    int16_t lanes[8];
    vst1q_s16(lanes, best_vec);
    int16_t best = lanes[0];
    for (int lane = 1; lane < 8; ++lane) {
        int16_t value = lanes[lane];
        best = find_max ? (value > best ? value : best) : (value < best ? value : best);
    }
    for (size_t i = start; i < length; ++i) {
        int16_t value = data[i];
        best = find_max ? (value > best ? value : best) : (value < best ? value : best);
    }
    return best;
}

static int16_t cpu_reduce_neon_i16_best(const int16_t *data, size_t length, bool find_max) {
    size_t i = 0;
    int16x8_t best_vec = vdupq_n_s16(find_max ? INT16_MIN : INT16_MAX);
    for (; i + 8 <= length; i += 8) {
        int16x8_t v = vld1q_s16(data + i);
        best_vec = find_max ? vmaxq_s16(best_vec, v) : vminq_s16(best_vec, v);
    }
    return cpu_reduce_neon_i16_finalize_best(best_vec, data, i, length, find_max);
}

static marmot_error_t
cpu_reduce_i16_neon_max(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int16_t *data = (const int16_t *)base;
    int16_t best = cpu_reduce_neon_i16_best(data, length, true);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_i16_neon_min(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int16_t *data = (const int16_t *)base;
    int16_t best = cpu_reduce_neon_i16_best(data, length, false);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_i16_neon_argmax(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int16_t *data = (const int16_t *)base;
    int16_t best = cpu_reduce_neon_i16_best(data, length, true);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_i16_neon_argmin(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int16_t *data = (const int16_t *)base;
    int16_t best = cpu_reduce_neon_i16_best(data, length, false);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

const cpu_reduce_traits_t cpu_reduce_i16_neon_traits = {
    .dtype = MARMOT_DTYPE_INT16,
    .impl_kind = CPU_REDUCE_IMPL_NEON,
    .ops = {
        .sum = cpu_reduce_i16_neon_sum,
        .mean = cpu_reduce_i16_neon_mean,
        .prod = cpu_reduce_i16_neon_prod,
        .min = cpu_reduce_i16_neon_min,
        .max = cpu_reduce_i16_neon_max,
        .argmax = cpu_reduce_i16_neon_argmax,
        .argmin = cpu_reduce_i16_neon_argmin,
        .impl_name = "neon-i16",
    },
};

CPU_REDUCTION_REGISTER_TRAITS(cpu_reduce_i16_neon_traits)

static double cpu_reduce_neon_u16_sum_impl(const uint16_t *data, size_t length) {
    uint32x4_t acc = vdupq_n_u32(0);
    size_t i = 0;
    for (; i + 8 <= length; i += 8) {
        uint16x8_t v = vld1q_u16(data + i);
        acc = cpu_reduce_neon_accumulate_u16(acc, v);
    }
    double sum = cpu_reduce_neon_u32_horizontal_sum(acc);
    for (; i < length; ++i) {
        sum += (double)data[i];
    }
    return sum;
}

static marmot_error_t
cpu_reduce_u16_neon_sum(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value = cpu_reduce_neon_u16_sum_impl((const uint16_t *)base, length);
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_u16_neon_mean(const void *device_ctx, const void *base, size_t length, double *out_value) {
    marmot_error_t status = cpu_reduce_u16_neon_sum(device_ctx, base, length, out_value);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value /= (double)length;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_u16_neon_prod(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint16_t *data = (const uint16_t *)base;
    double prod = 1.0;
    for (size_t i = 0; i < length; ++i) {
        prod *= (double)data[i];
    }
    *out_value = prod;
    return MARMOT_SUCCESS;
}

static inline uint16_t cpu_reduce_neon_u16_finalize_best(
    uint16x8_t best_vec, const uint16_t *data, size_t start, size_t length, bool find_max
) {
    uint16_t lanes[8];
    vst1q_u16(lanes, best_vec);
    uint16_t best = lanes[0];
    for (int lane = 1; lane < 8; ++lane) {
        uint16_t value = lanes[lane];
        best = find_max ? (value > best ? value : best) : (value < best ? value : best);
    }
    for (size_t i = start; i < length; ++i) {
        uint16_t value = data[i];
        best = find_max ? (value > best ? value : best) : (value < best ? value : best);
    }
    return best;
}

static uint16_t cpu_reduce_neon_u16_best(const uint16_t *data, size_t length, bool find_max) {
    size_t i = 0;
    uint16x8_t best_vec = vdupq_n_u16(find_max ? 0 : UINT16_MAX);
    for (; i + 8 <= length; i += 8) {
        uint16x8_t v = vld1q_u16(data + i);
        best_vec = find_max ? vmaxq_u16(best_vec, v) : vminq_u16(best_vec, v);
    }
    return cpu_reduce_neon_u16_finalize_best(best_vec, data, i, length, find_max);
}

static marmot_error_t
cpu_reduce_u16_neon_max(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint16_t *data = (const uint16_t *)base;
    uint16_t best = cpu_reduce_neon_u16_best(data, length, true);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_u16_neon_min(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint16_t *data = (const uint16_t *)base;
    uint16_t best = cpu_reduce_neon_u16_best(data, length, false);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_u16_neon_argmax(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint16_t *data = (const uint16_t *)base;
    uint16_t best = cpu_reduce_neon_u16_best(data, length, true);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_u16_neon_argmin(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint16_t *data = (const uint16_t *)base;
    uint16_t best = cpu_reduce_neon_u16_best(data, length, false);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

const cpu_reduce_traits_t cpu_reduce_u16_neon_traits = {
    .dtype = MARMOT_DTYPE_UINT16,
    .impl_kind = CPU_REDUCE_IMPL_NEON,
    .ops = {
        .sum = cpu_reduce_u16_neon_sum,
        .mean = cpu_reduce_u16_neon_mean,
        .prod = cpu_reduce_u16_neon_prod,
        .min = cpu_reduce_u16_neon_min,
        .max = cpu_reduce_u16_neon_max,
        .argmax = cpu_reduce_u16_neon_argmax,
        .argmin = cpu_reduce_u16_neon_argmin,
        .impl_name = "neon-u16",
    },
};

CPU_REDUCTION_REGISTER_TRAITS(cpu_reduce_u16_neon_traits)

static double cpu_reduce_neon_i8_sum_impl(const int8_t *data, size_t length) {
    int32x4_t acc = vdupq_n_s32(0);
    size_t i = 0;
    for (; i + 16 <= length; i += 16) {
        int8x16_t v = vld1q_s8(data + i);
        int16x8_t lo16 = vmovl_s8(vget_low_s8(v));
        int16x8_t hi16 = vmovl_s8(vget_high_s8(v));
        acc = cpu_reduce_neon_accumulate_s16(acc, lo16);
        acc = cpu_reduce_neon_accumulate_s16(acc, hi16);
    }
    double sum = cpu_reduce_neon_i32_horizontal_sum(acc);
    for (; i < length; ++i) {
        sum += (double)data[i];
    }
    return sum;
}

static marmot_error_t
cpu_reduce_i8_neon_sum(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value = cpu_reduce_neon_i8_sum_impl((const int8_t *)base, length);
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_i8_neon_mean(const void *device_ctx, const void *base, size_t length, double *out_value) {
    marmot_error_t status = cpu_reduce_i8_neon_sum(device_ctx, base, length, out_value);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value /= (double)length;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_i8_neon_prod(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int8_t *data = (const int8_t *)base;
    double prod = 1.0;
    for (size_t i = 0; i < length; ++i) {
        prod *= (double)data[i];
    }
    *out_value = prod;
    return MARMOT_SUCCESS;
}

static inline int8_t
cpu_reduce_neon_i8_finalize_best(int8x16_t best_vec, const int8_t *data, size_t start, size_t length, bool find_max) {
    int8_t lanes[16];
    vst1q_s8(lanes, best_vec);
    int8_t best = lanes[0];
    for (int lane = 1; lane < 16; ++lane) {
        int8_t value = lanes[lane];
        best = find_max ? (value > best ? value : best) : (value < best ? value : best);
    }
    for (size_t i = start; i < length; ++i) {
        int8_t value = data[i];
        best = find_max ? (value > best ? value : best) : (value < best ? value : best);
    }
    return best;
}

static int8_t cpu_reduce_neon_i8_best(const int8_t *data, size_t length, bool find_max) {
    size_t i = 0;
    int8x16_t best_vec = vdupq_n_s8(find_max ? INT8_MIN : INT8_MAX);
    for (; i + 16 <= length; i += 16) {
        int8x16_t v = vld1q_s8(data + i);
        best_vec = find_max ? vmaxq_s8(best_vec, v) : vminq_s8(best_vec, v);
    }
    return cpu_reduce_neon_i8_finalize_best(best_vec, data, i, length, find_max);
}

static marmot_error_t
cpu_reduce_i8_neon_max(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int8_t *data = (const int8_t *)base;
    int8_t best = cpu_reduce_neon_i8_best(data, length, true);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_i8_neon_min(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int8_t *data = (const int8_t *)base;
    int8_t best = cpu_reduce_neon_i8_best(data, length, false);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_i8_neon_argmax(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int8_t *data = (const int8_t *)base;
    int8_t best = cpu_reduce_neon_i8_best(data, length, true);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_i8_neon_argmin(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int8_t *data = (const int8_t *)base;
    int8_t best = cpu_reduce_neon_i8_best(data, length, false);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

const cpu_reduce_traits_t cpu_reduce_i8_neon_traits = {
    .dtype = MARMOT_DTYPE_INT8,
    .impl_kind = CPU_REDUCE_IMPL_NEON,
    .ops = {
        .sum = cpu_reduce_i8_neon_sum,
        .mean = cpu_reduce_i8_neon_mean,
        .prod = cpu_reduce_i8_neon_prod,
        .min = cpu_reduce_i8_neon_min,
        .max = cpu_reduce_i8_neon_max,
        .argmax = cpu_reduce_i8_neon_argmax,
        .argmin = cpu_reduce_i8_neon_argmin,
        .impl_name = "neon-i8",
    },
};

CPU_REDUCTION_REGISTER_TRAITS(cpu_reduce_i8_neon_traits)

static double cpu_reduce_neon_u8_sum_impl(const uint8_t *data, size_t length) {
    uint32x4_t acc = vdupq_n_u32(0);
    size_t i = 0;
    for (; i + 16 <= length; i += 16) {
        uint8x16_t v = vld1q_u8(data + i);
        uint16x8_t lo16 = vmovl_u8(vget_low_u8(v));
        uint16x8_t hi16 = vmovl_u8(vget_high_u8(v));
        acc = cpu_reduce_neon_accumulate_u16(acc, lo16);
        acc = cpu_reduce_neon_accumulate_u16(acc, hi16);
    }
    double sum = cpu_reduce_neon_u32_horizontal_sum(acc);
    for (; i < length; ++i) {
        sum += (double)data[i];
    }
    return sum;
}

static marmot_error_t
cpu_reduce_u8_neon_sum(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value = cpu_reduce_neon_u8_sum_impl((const uint8_t *)base, length);
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_u8_neon_mean(const void *device_ctx, const void *base, size_t length, double *out_value) {
    marmot_error_t status = cpu_reduce_u8_neon_sum(device_ctx, base, length, out_value);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value /= (double)length;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_u8_neon_prod(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint8_t *data = (const uint8_t *)base;
    double prod = 1.0;
    for (size_t i = 0; i < length; ++i) {
        prod *= (double)data[i];
    }
    *out_value = prod;
    return MARMOT_SUCCESS;
}

static inline uint8_t
cpu_reduce_neon_u8_finalize_best(uint8x16_t best_vec, const uint8_t *data, size_t start, size_t length, bool find_max) {
    uint8_t lanes[16];
    vst1q_u8(lanes, best_vec);
    uint8_t best = lanes[0];
    for (int lane = 1; lane < 16; ++lane) {
        uint8_t value = lanes[lane];
        best = find_max ? (value > best ? value : best) : (value < best ? value : best);
    }
    for (size_t i = start; i < length; ++i) {
        uint8_t value = data[i];
        best = find_max ? (value > best ? value : best) : (value < best ? value : best);
    }
    return best;
}

static uint8_t cpu_reduce_neon_u8_best(const uint8_t *data, size_t length, bool find_max) {
    size_t i = 0;
    uint8x16_t best_vec = vdupq_n_u8(find_max ? 0 : UINT8_MAX);
    for (; i + 16 <= length; i += 16) {
        uint8x16_t v = vld1q_u8(data + i);
        best_vec = find_max ? vmaxq_u8(best_vec, v) : vminq_u8(best_vec, v);
    }
    return cpu_reduce_neon_u8_finalize_best(best_vec, data, i, length, find_max);
}

static marmot_error_t
cpu_reduce_u8_neon_max(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint8_t *data = (const uint8_t *)base;
    uint8_t best = cpu_reduce_neon_u8_best(data, length, true);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_u8_neon_min(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint8_t *data = (const uint8_t *)base;
    uint8_t best = cpu_reduce_neon_u8_best(data, length, false);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_u8_neon_argmax(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint8_t *data = (const uint8_t *)base;
    uint8_t best = cpu_reduce_neon_u8_best(data, length, true);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_u8_neon_argmin(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_neon_i32_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint8_t *data = (const uint8_t *)base;
    uint8_t best = cpu_reduce_neon_u8_best(data, length, false);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

const cpu_reduce_traits_t cpu_reduce_u8_neon_traits = {
    .dtype = MARMOT_DTYPE_UINT8,
    .impl_kind = CPU_REDUCE_IMPL_NEON,
    .ops = {
        .sum = cpu_reduce_u8_neon_sum,
        .mean = cpu_reduce_u8_neon_mean,
        .prod = cpu_reduce_u8_neon_prod,
        .min = cpu_reduce_u8_neon_min,
        .max = cpu_reduce_u8_neon_max,
        .argmax = cpu_reduce_u8_neon_argmax,
        .argmin = cpu_reduce_u8_neon_argmin,
        .impl_name = "neon-u8",
    },
};

CPU_REDUCTION_REGISTER_TRAITS(cpu_reduce_u8_neon_traits)

#endif // MARMOT_ENABLE_NEON
