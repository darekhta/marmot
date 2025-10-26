#include "cpu_backend_internal.h"

static marmot_error_t cpu_reduce_scalar_int_require_numeric(const void *base, double *out_value, size_t length) {
    if (base == nullptr || out_value == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in scalar int reduction kernel");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (length == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "Reduction over zero elements is undefined");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_scalar_int_require_arg(const void *base, double *out_value, uint64_t *out_index, size_t length) {
    marmot_error_t status = cpu_reduce_scalar_int_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    if (out_index == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Arg reduction requires index output");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return MARMOT_SUCCESS;
}

#define DEFINE_SCALAR_INT_KERNELS(SUFFIX, VALUE_T, FIELD, IMPL_NAME)                                                   \
    static marmot_error_t cpu_reduce_##SUFFIX##_scalar_sum(                                                            \
        const void *device_ctx, const void *base, size_t length, double *out_value                                     \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        marmot_error_t status = cpu_reduce_scalar_int_require_numeric(base, out_value, length);                        \
        if (status != MARMOT_SUCCESS) {                                                                                \
            return status;                                                                                             \
        }                                                                                                              \
        const VALUE_T *data = (const VALUE_T *)base;                                                                   \
        double sum = 0.0;                                                                                              \
        for (size_t i = 0; i < length; ++i) {                                                                          \
            sum += (double)data[i].FIELD;                                                                              \
        }                                                                                                              \
        *out_value = sum;                                                                                              \
        return MARMOT_SUCCESS;                                                                                         \
    }                                                                                                                  \
                                                                                                                       \
    static marmot_error_t cpu_reduce_##SUFFIX##_scalar_mean(                                                           \
        const void *device_ctx, const void *base, size_t length, double *out_value                                     \
    ) {                                                                                                                \
        marmot_error_t status = cpu_reduce_##SUFFIX##_scalar_sum(device_ctx, base, length, out_value);                 \
        if (status != MARMOT_SUCCESS) {                                                                                \
            return status;                                                                                             \
        }                                                                                                              \
        *out_value /= (double)length;                                                                                  \
        return MARMOT_SUCCESS;                                                                                         \
    }                                                                                                                  \
                                                                                                                       \
    static marmot_error_t cpu_reduce_##SUFFIX##_scalar_prod(                                                           \
        const void *device_ctx, const void *base, size_t length, double *out_value                                     \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        marmot_error_t status = cpu_reduce_scalar_int_require_numeric(base, out_value, length);                        \
        if (status != MARMOT_SUCCESS) {                                                                                \
            return status;                                                                                             \
        }                                                                                                              \
        const VALUE_T *data = (const VALUE_T *)base;                                                                   \
        double prod = 1.0;                                                                                             \
        for (size_t i = 0; i < length; ++i) {                                                                          \
            prod *= (double)data[i].FIELD;                                                                             \
        }                                                                                                              \
        *out_value = prod;                                                                                             \
        return MARMOT_SUCCESS;                                                                                         \
    }                                                                                                                  \
                                                                                                                       \
    static marmot_error_t cpu_reduce_##SUFFIX##_scalar_max(                                                            \
        const void *device_ctx, const void *base, size_t length, double *out_value                                     \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        marmot_error_t status = cpu_reduce_scalar_int_require_numeric(base, out_value, length);                        \
        if (status != MARMOT_SUCCESS) {                                                                                \
            return status;                                                                                             \
        }                                                                                                              \
        const VALUE_T *data = (const VALUE_T *)base;                                                                   \
        typedef typeof(((VALUE_T *)0)->FIELD) cpu_reduce_##SUFFIX##_scalar_t;                                          \
        cpu_reduce_##SUFFIX##_scalar_t best = data[0].FIELD;                                                           \
        for (size_t i = 1; i < length; ++i) {                                                                          \
            cpu_reduce_##SUFFIX##_scalar_t value = data[i].FIELD;                                                      \
            if (value > best) {                                                                                        \
                best = value;                                                                                          \
            }                                                                                                          \
        }                                                                                                              \
        *out_value = (double)best;                                                                                     \
        return MARMOT_SUCCESS;                                                                                         \
    }                                                                                                                  \
                                                                                                                       \
    static marmot_error_t cpu_reduce_##SUFFIX##_scalar_min(                                                            \
        const void *device_ctx, const void *base, size_t length, double *out_value                                     \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        marmot_error_t status = cpu_reduce_scalar_int_require_numeric(base, out_value, length);                        \
        if (status != MARMOT_SUCCESS) {                                                                                \
            return status;                                                                                             \
        }                                                                                                              \
        const VALUE_T *data = (const VALUE_T *)base;                                                                   \
        typedef typeof(((VALUE_T *)0)->FIELD) cpu_reduce_##SUFFIX##_scalar_t;                                          \
        cpu_reduce_##SUFFIX##_scalar_t best = data[0].FIELD;                                                           \
        for (size_t i = 1; i < length; ++i) {                                                                          \
            cpu_reduce_##SUFFIX##_scalar_t value = data[i].FIELD;                                                      \
            if (value < best) {                                                                                        \
                best = value;                                                                                          \
            }                                                                                                          \
        }                                                                                                              \
        *out_value = (double)best;                                                                                     \
        return MARMOT_SUCCESS;                                                                                         \
    }                                                                                                                  \
                                                                                                                       \
    static marmot_error_t cpu_reduce_##SUFFIX##_scalar_argmax(                                                         \
        const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index                \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        marmot_error_t status = cpu_reduce_scalar_int_require_arg(base, out_value, out_index, length);                 \
        if (status != MARMOT_SUCCESS) {                                                                                \
            return status;                                                                                             \
        }                                                                                                              \
        const VALUE_T *data = (const VALUE_T *)base;                                                                   \
        typedef typeof(((VALUE_T *)0)->FIELD) cpu_reduce_##SUFFIX##_scalar_t;                                          \
        cpu_reduce_##SUFFIX##_scalar_t best = data[0].FIELD;                                                           \
        uint64_t best_idx = 0;                                                                                         \
        for (size_t i = 1; i < length; ++i) {                                                                          \
            cpu_reduce_##SUFFIX##_scalar_t value = data[i].FIELD;                                                      \
            if (value > best) {                                                                                        \
                best = value;                                                                                          \
                best_idx = (uint64_t)i;                                                                                \
            }                                                                                                          \
        }                                                                                                              \
        *out_value = (double)best;                                                                                     \
        *out_index = best_idx;                                                                                         \
        return MARMOT_SUCCESS;                                                                                         \
    }                                                                                                                  \
                                                                                                                       \
    static marmot_error_t cpu_reduce_##SUFFIX##_scalar_argmin(                                                         \
        const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index                \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        marmot_error_t status = cpu_reduce_scalar_int_require_arg(base, out_value, out_index, length);                 \
        if (status != MARMOT_SUCCESS) {                                                                                \
            return status;                                                                                             \
        }                                                                                                              \
        const VALUE_T *data = (const VALUE_T *)base;                                                                   \
        typedef typeof(((VALUE_T *)0)->FIELD) cpu_reduce_##SUFFIX##_scalar_t;                                          \
        cpu_reduce_##SUFFIX##_scalar_t best = data[0].FIELD;                                                           \
        uint64_t best_idx = 0;                                                                                         \
        for (size_t i = 1; i < length; ++i) {                                                                          \
            cpu_reduce_##SUFFIX##_scalar_t value = data[i].FIELD;                                                      \
            if (value < best) {                                                                                        \
                best = value;                                                                                          \
                best_idx = (uint64_t)i;                                                                                \
            }                                                                                                          \
        }                                                                                                              \
        *out_value = (double)best;                                                                                     \
        *out_index = best_idx;                                                                                         \
        return MARMOT_SUCCESS;                                                                                         \
    }                                                                                                                  \
                                                                                                                       \
    const cpu_reduce_traits_t cpu_reduce_##SUFFIX##_scalar_traits = {                                                  \
        .dtype = MARMOT_DTYPE_##IMPL_NAME,                                                                             \
        .impl_kind = CPU_REDUCE_IMPL_SCALAR,                                                                           \
        .ops = {                                                                                                       \
            .sum = cpu_reduce_##SUFFIX##_scalar_sum,                                                                   \
            .mean = cpu_reduce_##SUFFIX##_scalar_mean,                                                                 \
            .prod = cpu_reduce_##SUFFIX##_scalar_prod,                                                                 \
            .min = cpu_reduce_##SUFFIX##_scalar_min,                                                                   \
            .max = cpu_reduce_##SUFFIX##_scalar_max,                                                                   \
            .argmax = cpu_reduce_##SUFFIX##_scalar_argmax,                                                             \
            .argmin = cpu_reduce_##SUFFIX##_scalar_argmin,                                                             \
            .impl_name = "scalar-" #SUFFIX,                                                                            \
        },                                                                                                             \
    };                                                                                                                 \
                                                                                                                       \
    CPU_REDUCTION_REGISTER_TRAITS(cpu_reduce_##SUFFIX##_scalar_traits)

DEFINE_SCALAR_INT_KERNELS(i32, marmot_int32_t, value, INT32)
DEFINE_SCALAR_INT_KERNELS(u32, marmot_uint32_t, value, UINT32)
DEFINE_SCALAR_INT_KERNELS(i16, marmot_int16_t, value, INT16)
DEFINE_SCALAR_INT_KERNELS(u16, marmot_uint16_t, value, UINT16)
DEFINE_SCALAR_INT_KERNELS(i8, marmot_int8_t, value, INT8)
DEFINE_SCALAR_INT_KERNELS(u8, marmot_uint8_t, value, UINT8)
