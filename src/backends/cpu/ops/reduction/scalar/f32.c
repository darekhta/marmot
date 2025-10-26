#include "cpu_backend_internal.h"

static marmot_error_t cpu_reduce_require_args(const void *base, double *out_value, size_t length) {
    if (base == nullptr || out_value == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in scalar reduction kernel");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (length == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "Reduction over zero elements is undefined");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_f32_scalar_sum(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_require_args(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const float *data = (const float *)base;
    double sum = 0.0;
    for (size_t i = 0; i < length; ++i) {
        sum += (double)data[i];
    }
    *out_value = sum;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_f32_scalar_prod(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_require_args(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const float *data = (const float *)base;
    double prod = 1.0;
    for (size_t i = 0; i < length; ++i) {
        prod *= (double)data[i];
    }
    *out_value = prod;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_f32_scalar_min(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_require_args(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const float *data = (const float *)base;
    float best = data[0];
    for (size_t i = 1; i < length; ++i) {
        if (data[i] < best) {
            best = data[i];
        }
    }
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_f32_scalar_max(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_require_args(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const float *data = (const float *)base;
    float best = data[0];
    for (size_t i = 1; i < length; ++i) {
        if (data[i] > best) {
            best = data[i];
        }
    }
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_f32_scalar_argmax(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_require_args(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    if (out_index == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Arg reduction requires index output");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const float *data = (const float *)base;
    float best = data[0];
    uint64_t best_idx = 0;
    for (size_t i = 1; i < length; ++i) {
        float value = data[i];
        if (value > best) {
            best = value;
            best_idx = (uint64_t)i;
        }
    }
    *out_value = (double)best;
    *out_index = best_idx;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_f32_scalar_argmin(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_require_args(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    if (out_index == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Arg reduction requires index output");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const float *data = (const float *)base;
    float best = data[0];
    uint64_t best_idx = 0;
    for (size_t i = 1; i < length; ++i) {
        float value = data[i];
        if (value < best) {
            best = value;
            best_idx = (uint64_t)i;
        }
    }
    *out_value = (double)best;
    *out_index = best_idx;
    return MARMOT_SUCCESS;
}

const cpu_reduce_traits_t cpu_reduce_f32_scalar_traits = {
    .dtype = MARMOT_DTYPE_FLOAT32,
    .impl_kind = CPU_REDUCE_IMPL_SCALAR,
    .ops = {
        .sum = cpu_reduce_f32_scalar_sum,
        .mean = nullptr,
        .prod = cpu_reduce_f32_scalar_prod,
        .min = cpu_reduce_f32_scalar_min,
        .max = cpu_reduce_f32_scalar_max,
        .argmax = cpu_reduce_f32_scalar_argmax,
        .argmin = cpu_reduce_f32_scalar_argmin,
        .impl_name = "scalar-f32",
    },
};

CPU_REDUCTION_REGISTER_TRAITS(cpu_reduce_f32_scalar_traits)
