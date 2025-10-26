#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#include "cpu_backend_internal.h"

extern const cpu_reduce_traits_t cpu_reduce_f32_scalar_traits;
extern const cpu_reduce_traits_t cpu_reduce_i32_scalar_traits;
extern const cpu_reduce_traits_t cpu_reduce_u32_scalar_traits;
extern const cpu_reduce_traits_t cpu_reduce_i16_scalar_traits;
extern const cpu_reduce_traits_t cpu_reduce_u16_scalar_traits;
extern const cpu_reduce_traits_t cpu_reduce_i8_scalar_traits;
extern const cpu_reduce_traits_t cpu_reduce_u8_scalar_traits;

#if HAS_NEON
extern const cpu_reduce_traits_t cpu_reduce_f32_neon_traits;
extern const cpu_reduce_traits_t cpu_reduce_f16_neon_traits;
extern const cpu_reduce_traits_t cpu_reduce_bf16_neon_traits;
extern const cpu_reduce_traits_t cpu_reduce_i32_neon_traits;
extern const cpu_reduce_traits_t cpu_reduce_u32_neon_traits;
extern const cpu_reduce_traits_t cpu_reduce_i16_neon_traits;
extern const cpu_reduce_traits_t cpu_reduce_u16_neon_traits;
extern const cpu_reduce_traits_t cpu_reduce_i8_neon_traits;
extern const cpu_reduce_traits_t cpu_reduce_u8_neon_traits;
#endif

#if HAS_AVX2
extern const cpu_reduce_traits_t cpu_reduce_i16_avx2_traits;
extern const cpu_reduce_traits_t cpu_reduce_u16_avx2_traits;
extern const cpu_reduce_traits_t cpu_reduce_i8_avx2_traits;
extern const cpu_reduce_traits_t cpu_reduce_u8_avx2_traits;
#endif

static const cpu_reduce_ops_t *const k_cpu_reduce_scalar_ops[MARMOT_DTYPE_COUNT] = {
    [MARMOT_DTYPE_FLOAT32] = &cpu_reduce_f32_scalar_traits.ops,
    [MARMOT_DTYPE_INT32] = &cpu_reduce_i32_scalar_traits.ops,
    [MARMOT_DTYPE_UINT32] = &cpu_reduce_u32_scalar_traits.ops,
    [MARMOT_DTYPE_INT16] = &cpu_reduce_i16_scalar_traits.ops,
    [MARMOT_DTYPE_UINT16] = &cpu_reduce_u16_scalar_traits.ops,
    [MARMOT_DTYPE_INT8] = &cpu_reduce_i8_scalar_traits.ops,
    [MARMOT_DTYPE_UINT8] = &cpu_reduce_u8_scalar_traits.ops,
};

#if HAS_NEON
static const cpu_reduce_ops_t *const k_cpu_reduce_neon_ops[MARMOT_DTYPE_COUNT] = {
    [MARMOT_DTYPE_FLOAT32] = &cpu_reduce_f32_neon_traits.ops,
    [MARMOT_DTYPE_FLOAT16] = &cpu_reduce_f16_neon_traits.ops,
    [MARMOT_DTYPE_BFLOAT16] = &cpu_reduce_bf16_neon_traits.ops,
    [MARMOT_DTYPE_INT32] = &cpu_reduce_i32_neon_traits.ops,
    [MARMOT_DTYPE_UINT32] = &cpu_reduce_u32_neon_traits.ops,
    [MARMOT_DTYPE_INT16] = &cpu_reduce_i16_neon_traits.ops,
    [MARMOT_DTYPE_UINT16] = &cpu_reduce_u16_neon_traits.ops,
    [MARMOT_DTYPE_INT8] = &cpu_reduce_i8_neon_traits.ops,
    [MARMOT_DTYPE_UINT8] = &cpu_reduce_u8_neon_traits.ops,
};
#endif

#if HAS_AVX2
static const cpu_reduce_ops_t *const k_cpu_reduce_avx2_ops[MARMOT_DTYPE_COUNT] = {
    [MARMOT_DTYPE_INT16] = &cpu_reduce_i16_avx2_traits.ops,
    [MARMOT_DTYPE_UINT16] = &cpu_reduce_u16_avx2_traits.ops,
    [MARMOT_DTYPE_INT8] = &cpu_reduce_i8_avx2_traits.ops,
    [MARMOT_DTYPE_UINT8] = &cpu_reduce_u8_avx2_traits.ops,
};
#endif

static cpu_reduce_numeric_fn
cpu_reduce_numeric_fn_from_ops(const cpu_reduce_ops_t *ops, marmot_device_reduction_op_t op, bool *mean_via_sum) {
    if (mean_via_sum != nullptr) {
        *mean_via_sum = false;
    }
    if (ops == nullptr) {
        return nullptr;
    }

    switch (op) {
    case MARMOT_DEVICE_REDUCTION_SUM:
        return ops->sum;
    case MARMOT_DEVICE_REDUCTION_MEAN:
        if (ops->mean != nullptr) {
            return ops->mean;
        }
        if (ops->sum != nullptr) {
            if (mean_via_sum != nullptr) {
                *mean_via_sum = true;
            }
            return ops->sum;
        }
        return nullptr;
    case MARMOT_DEVICE_REDUCTION_PROD:
        return ops->prod;
    case MARMOT_DEVICE_REDUCTION_MIN:
        return ops->min;
    case MARMOT_DEVICE_REDUCTION_MAX:
        return ops->max;
    default:
        return nullptr;
    }
}

static cpu_reduce_arg_fn cpu_reduce_arg_fn_from_ops(const cpu_reduce_ops_t *ops, marmot_device_reduction_op_t op) {
    if (ops == nullptr) {
        return nullptr;
    }
    switch (op) {
    case MARMOT_DEVICE_REDUCTION_ARGMAX:
        return ops->argmax;
    case MARMOT_DEVICE_REDUCTION_ARGMIN:
        return ops->argmin;
    default:
        return nullptr;
    }
}

static cpu_reduce_numeric_fn cpu_reduce_resolve_numeric_fn(
    const void *device_ctx, marmot_dtype_t dtype, marmot_device_reduction_op_t op, bool *mean_via_sum
) {
    if (mean_via_sum != nullptr) {
        *mean_via_sum = false;
    }
    if (dtype >= MARMOT_DTYPE_COUNT) {
        return nullptr;
    }

#if HAS_AVX2
    if (has_avx2(device_ctx)) {
        const cpu_reduce_ops_t *ops = k_cpu_reduce_avx2_ops[dtype];
        bool candidate_mean_via_sum = false;
        cpu_reduce_numeric_fn fn = cpu_reduce_numeric_fn_from_ops(ops, op, &candidate_mean_via_sum);
        if (fn != nullptr) {
            if (mean_via_sum != nullptr) {
                *mean_via_sum = candidate_mean_via_sum;
            }
            return fn;
        }
    }
#endif

#if HAS_NEON
    if (has_neon(device_ctx)) {
        const cpu_reduce_ops_t *ops = k_cpu_reduce_neon_ops[dtype];
        bool candidate_mean_via_sum = false;
        cpu_reduce_numeric_fn fn = cpu_reduce_numeric_fn_from_ops(ops, op, &candidate_mean_via_sum);
        if (fn != nullptr) {
            if (mean_via_sum != nullptr) {
                *mean_via_sum = candidate_mean_via_sum;
            }
            return fn;
        }
    }
#endif

    return cpu_reduce_numeric_fn_from_ops(k_cpu_reduce_scalar_ops[dtype], op, mean_via_sum);
}

static cpu_reduce_arg_fn
cpu_reduce_resolve_arg_fn(const void *device_ctx, marmot_dtype_t dtype, marmot_device_reduction_op_t op) {
    if (dtype >= MARMOT_DTYPE_COUNT) {
        return nullptr;
    }

#if HAS_AVX2
    if (has_avx2(device_ctx)) {
        const cpu_reduce_ops_t *ops = k_cpu_reduce_avx2_ops[dtype];
        cpu_reduce_arg_fn fn = cpu_reduce_arg_fn_from_ops(ops, op);
        if (fn != nullptr) {
            return fn;
        }
    }
#endif

#if HAS_NEON
    if (has_neon(device_ctx)) {
        const cpu_reduce_ops_t *ops = k_cpu_reduce_neon_ops[dtype];
        cpu_reduce_arg_fn fn = cpu_reduce_arg_fn_from_ops(ops, op);
        if (fn != nullptr) {
            return fn;
        }
    }
#endif

    return cpu_reduce_arg_fn_from_ops(k_cpu_reduce_scalar_ops[dtype], op);
}

typedef struct {
    bool keepdims;
    size_t reduce_ndim;
    size_t reduce_axes[MARMOT_MAX_DIMS];
    size_t reduce_shape[MARMOT_MAX_DIMS];
    size_t reduce_linear_strides[MARMOT_MAX_DIMS];
    size_t reduce_total;
    size_t kept_ndim;
    size_t kept_axes[MARMOT_MAX_DIMS];
    size_t expected_out_ndim;
    size_t expected_out_shape[MARMOT_MAX_DIMS];
} cpu_reduction_geometry_t;

static inline bool cpu_reduction_is_arg_op(marmot_device_reduction_op_t op) {
    return op == MARMOT_DEVICE_REDUCTION_ARGMAX || op == MARMOT_DEVICE_REDUCTION_ARGMIN;
}

static inline bool cpu_reduction_increment(size_t *indices, const size_t *shape, size_t ndim) {
    if (ndim == 0) {
        return false;
    }

    for (ssize_t dim = (ssize_t)ndim - 1; dim >= 0; --dim) {
        indices[dim]++;
        if (indices[dim] < shape[dim]) {
            return true;
        }
        indices[dim] = 0;
    }
    return false;
}

static inline void cpu_reduction_linear_to_indices(size_t linear, const marmot_tensor_t *tensor, size_t *indices) {
    size_t ndim = tensor->shape.ndim;
    if (ndim == 0) {
        return;
    }

    for (ssize_t dim = (ssize_t)ndim - 1; dim >= 0; --dim) {
        size_t extent = tensor->shape.shape[dim];
        if (extent == 0) {
            indices[dim] = 0;
            continue;
        }
        indices[dim] = linear % extent;
        linear /= extent;
    }
}

static inline size_t cpu_reduction_compute_out_offset(const marmot_tensor_t *tensor, const size_t *indices) {
    size_t offset = 0;
    for (size_t dim = 0; dim < tensor->shape.ndim; ++dim) {
        offset += indices[dim] * tensor->shape.strides[dim];
    }
    return offset;
}

static inline size_t cpu_reduction_compute_base_offset(
    const cpu_reduction_geometry_t *geo, const marmot_tensor_t *input, const size_t *out_indices
) {
    size_t offset = 0;

    if (geo->keepdims) {
        for (size_t dim = 0; dim < input->shape.ndim; ++dim) {
            offset += out_indices[dim] * input->shape.strides[dim];
        }
        return offset;
    }

    if (geo->kept_ndim == 0) {
        return 0;
    }

    for (size_t i = 0; i < geo->kept_ndim; ++i) {
        size_t axis = geo->kept_axes[i];
        offset += out_indices[i] * input->shape.strides[axis];
    }
    return offset;
}

static inline void cpu_reduction_sort_axes(size_t *axes, size_t count) {
    for (size_t i = 1; i < count; ++i) {
        size_t key = axes[i];
        size_t j = i;
        while (j > 0 && axes[j - 1] > key) {
            axes[j] = axes[j - 1];
            --j;
        }
        axes[j] = key;
    }
}

static marmot_error_t cpu_prepare_reduction_geometry(
    const marmot_tensor_t *input, const marmot_reduction_params_t *params, const marmot_tensor_t *out_values,
    const marmot_tensor_t *out_indices, bool require_indices, cpu_reduction_geometry_t *geo
) {
    if (input == nullptr || out_values == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensor pointer in reduction setup");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t ndim = input->shape.ndim;
    if (ndim == 0 || ndim > MARMOT_MAX_DIMS) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unsupported tensor rank for reduction");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    bool reduce_mask[MARMOT_MAX_DIMS] = {false};
    size_t reduced_axes[MARMOT_MAX_DIMS];
    size_t reduce_ndim = 0;

    geo->keepdims = params != nullptr && params->keepdims;

    if (params == nullptr || params->axes == nullptr || params->num_axes == 0) {
        for (size_t axis = 0; axis < ndim; ++axis) {
            reduce_mask[axis] = true;
            reduced_axes[reduce_ndim++] = axis;
        }
    } else {
        for (size_t i = 0; i < params->num_axes; ++i) {
            int32_t axis = params->axes[i];
            if (axis < 0) {
                axis += (int32_t)ndim;
            }
            if (axis < 0 || axis >= (int32_t)ndim) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Reduction axis out of range");
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            size_t axis_u = (size_t)axis;
            if (!reduce_mask[axis_u]) {
                reduce_mask[axis_u] = true;
                reduced_axes[reduce_ndim++] = axis_u;
            }
        }
    }

    cpu_reduction_sort_axes(reduced_axes, reduce_ndim);

    geo->reduce_ndim = reduce_ndim;
    geo->reduce_total = 1;
    for (size_t i = 0; i < reduce_ndim; ++i) {
        size_t axis = reduced_axes[i];
        geo->reduce_axes[i] = axis;
        geo->reduce_shape[i] = input->shape.shape[axis];
        geo->reduce_total *= input->shape.shape[axis];
    }

    size_t stride = 1;
    for (ssize_t i = (ssize_t)reduce_ndim - 1; i >= 0; --i) {
        geo->reduce_linear_strides[i] = stride;
        stride *= geo->reduce_shape[i];
    }

    geo->kept_ndim = 0;
    for (size_t axis = 0; axis < ndim; ++axis) {
        if (!reduce_mask[axis]) {
            geo->kept_axes[geo->kept_ndim++] = axis;
        }
    }

    if (geo->keepdims) {
        geo->expected_out_ndim = ndim;
        for (size_t axis = 0; axis < ndim; ++axis) {
            geo->expected_out_shape[axis] = reduce_mask[axis] ? 1 : input->shape.shape[axis];
        }
    } else if (geo->kept_ndim > 0) {
        geo->expected_out_ndim = geo->kept_ndim;
        for (size_t i = 0; i < geo->kept_ndim; ++i) {
            geo->expected_out_shape[i] = input->shape.shape[geo->kept_axes[i]];
        }
    } else {
        geo->expected_out_ndim = 1;
        geo->expected_out_shape[0] = 1;
    }

    if (out_values->shape.ndim != geo->expected_out_ndim) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Reduction output rank mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    for (size_t axis = 0; axis < geo->expected_out_ndim; ++axis) {
        if (out_values->shape.shape[axis] != geo->expected_out_shape[axis]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Reduction output shape mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }

    if (require_indices) {
        if (out_indices == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Arg reduction requires indices tensor");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (out_indices->dtype != MARMOT_DTYPE_UINT64) {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Arg reduction indices must be UINT64");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        if (out_indices->shape.ndim != geo->expected_out_ndim) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Arg reduction indices rank mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        for (size_t axis = 0; axis < geo->expected_out_ndim; ++axis) {
            if (out_indices->shape.shape[axis] != geo->expected_out_shape[axis]) {
                marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Arg reduction indices shape mismatch");
                return MARMOT_ERROR_DIMENSION_MISMATCH;
            }
        }
    }

    return MARMOT_SUCCESS;
}

static inline int64_t cpu_reduction_round_to_i64(double value, int64_t min_val, int64_t max_val) {
    if (isnan(value)) {
        value = 0.0;
    }
    if (value < (double)min_val) {
        value = (double)min_val;
    }
    if (value > (double)max_val) {
        value = (double)max_val;
    }
    double rounded = nearbyint(value);
    if (rounded < (double)min_val) {
        rounded = (double)min_val;
    }
    if (rounded > (double)max_val) {
        rounded = (double)max_val;
    }
    return (int64_t)rounded;
}

static inline uint64_t cpu_reduction_round_to_u64(double value, uint64_t max_val) {
    if (isnan(value) || value < 0.0) {
        value = 0.0;
    }
    if (value > (double)max_val) {
        value = (double)max_val;
    }
    double rounded = nearbyint(value);
    if (rounded < 0.0) {
        rounded = 0.0;
    }
    if (rounded > (double)max_val) {
        rounded = (double)max_val;
    }
    return (uint64_t)rounded;
}

static bool cpu_reduction_load_element(marmot_dtype_t dtype, const void *data, size_t index, double *out_value) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT64:
        *out_value = ((const double *)data)[index];
        return true;
    case MARMOT_DTYPE_FLOAT32:
        *out_value = ((const float *)data)[index];
        return true;
    case MARMOT_DTYPE_FLOAT16:
        *out_value = (double)marmot_float16_to_native(((const marmot_float16_t *)data)[index]);
        return true;
    case MARMOT_DTYPE_BFLOAT16:
        *out_value = (double)marmot_bfloat16_to_native(((const marmot_bfloat16_t *)data)[index]);
        return true;
    case MARMOT_DTYPE_INT64:
        *out_value = (double)((const marmot_int64_t *)data)[index].value;
        return true;
    case MARMOT_DTYPE_INT32:
        *out_value = (double)((const marmot_int32_t *)data)[index].value;
        return true;
    case MARMOT_DTYPE_INT16:
        *out_value = (double)((const marmot_int16_t *)data)[index].value;
        return true;
    case MARMOT_DTYPE_INT8:
        *out_value = (double)((const marmot_int8_t *)data)[index].value;
        return true;
    case MARMOT_DTYPE_UINT8:
        *out_value = (double)((const marmot_uint8_t *)data)[index].value;
        return true;
    case MARMOT_DTYPE_UINT16:
        *out_value = (double)((const marmot_uint16_t *)data)[index].value;
        return true;
    case MARMOT_DTYPE_UINT32:
        *out_value = (double)((const marmot_uint32_t *)data)[index].value;
        return true;
    case MARMOT_DTYPE_UINT64:
        *out_value = (double)((const marmot_uint64_t *)data)[index].value;
        return true;
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
        *out_value = (double)marmot_fp8_e4m3_to_native(((const marmot_float8_e4m3_t *)data)[index]);
        return true;
    case MARMOT_DTYPE_FLOAT8_E5M2:
        *out_value = (double)marmot_fp8_e5m2_to_native(((const marmot_float8_e5m2_t *)data)[index]);
        return true;
#endif
    default:
        return false;
    }
}

static bool cpu_reduction_store_result(marmot_dtype_t dtype, void *data, size_t index, double value) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT64:
        ((double *)data)[index] = value;
        return true;
    case MARMOT_DTYPE_FLOAT32:
        ((float *)data)[index] = (float)value;
        return true;
    case MARMOT_DTYPE_FLOAT16:
        ((marmot_float16_t *)data)[index] = marmot_native_to_float16((_Float16)value);
        return true;
    case MARMOT_DTYPE_BFLOAT16:
        ((marmot_bfloat16_t *)data)[index] = marmot_native_to_bfloat16((float)value);
        return true;
    case MARMOT_DTYPE_INT64:
        ((marmot_int64_t *)data)[index].value = cpu_reduction_round_to_i64(value, INT64_MIN, INT64_MAX);
        return true;
    case MARMOT_DTYPE_INT32:
        ((marmot_int32_t *)data)[index].value = (int32_t)cpu_reduction_round_to_i64(value, INT32_MIN, INT32_MAX);
        return true;
    case MARMOT_DTYPE_INT16:
        ((marmot_int16_t *)data)[index].value = (int16_t)cpu_reduction_round_to_i64(value, INT16_MIN, INT16_MAX);
        return true;
    case MARMOT_DTYPE_INT8:
        ((marmot_int8_t *)data)[index].value = (int8_t)cpu_reduction_round_to_i64(value, INT8_MIN, INT8_MAX);
        return true;
    case MARMOT_DTYPE_UINT8:
        ((marmot_uint8_t *)data)[index].value = (uint8_t)cpu_reduction_round_to_u64(value, UINT8_MAX);
        return true;
    case MARMOT_DTYPE_UINT16:
        ((marmot_uint16_t *)data)[index].value = (uint16_t)cpu_reduction_round_to_u64(value, UINT16_MAX);
        return true;
    case MARMOT_DTYPE_UINT32:
        ((marmot_uint32_t *)data)[index].value = (uint32_t)cpu_reduction_round_to_u64(value, UINT32_MAX);
        return true;
    case MARMOT_DTYPE_UINT64:
        ((marmot_uint64_t *)data)[index].value = cpu_reduction_round_to_u64(value, UINT64_MAX);
        return true;
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
        ((marmot_float8_e4m3_t *)data)[index] = marmot_native_to_fp8_e4m3((_Float16)value);
        return true;
    case MARMOT_DTYPE_FLOAT8_E5M2:
        ((marmot_float8_e5m2_t *)data)[index] = marmot_native_to_fp8_e5m2((_Float16)value);
        return true;
#endif
    default:
        return false;
    }
}

static marmot_error_t cpu_reduction_try_microkernel(
    const void *device_ctx, marmot_device_reduction_op_t op, marmot_dtype_t dtype, const void *base_ptr, size_t length,
    double *out_value, uint64_t *out_index
) {
    if (device_ctx == nullptr) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    if (op < 0 || op >= MARMOT_DEVICE_REDUCTION_COUNT) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    if (cpu_reduction_is_arg_op(op)) {
        cpu_reduce_arg_fn arg_fn = cpu_reduce_resolve_arg_fn(device_ctx, dtype, op);
        if (arg_fn == nullptr) {
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        if (out_index == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        return arg_fn(device_ctx, base_ptr, length, out_value, out_index);
    }

    bool mean_via_sum = false;
    cpu_reduce_numeric_fn numeric_fn = cpu_reduce_resolve_numeric_fn(device_ctx, dtype, op, &mean_via_sum);
    if (numeric_fn == nullptr) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    marmot_error_t status = numeric_fn(device_ctx, base_ptr, length, out_value);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    if (op == MARMOT_DEVICE_REDUCTION_MEAN && mean_via_sum) {
        if (length == 0) {
            marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "Reduction over zero elements is undefined");
            return MARMOT_ERROR_INVALID_OPERATION;
        }
        *out_value /= (double)length;
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduction_run_numeric(
    cpu_context_t *ctx, marmot_device_reduction_op_t op, const marmot_tensor_t *input,
    const marmot_reduction_params_t *params, const cpu_reduction_geometry_t *geo, marmot_tensor_t *out_values,
    marmot_tensor_t *out_indices
) {
    const void *in_data = input->data;
    void *out_data = out_values->data;
    void *indices_data = out_indices != nullptr ? out_indices->data : nullptr;

    size_t out_count = marmot_tensor_num_elements(out_values);
    if (out_count == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Reduction output tensor is empty");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t out_coords[MARMOT_MAX_DIMS] = {0};
    size_t reduce_coords[MARMOT_MAX_DIMS] = {0};

    for (size_t out_linear = 0; out_linear < out_count; ++out_linear) {
        cpu_reduction_linear_to_indices(out_linear, out_values, out_coords);
        size_t out_offset = cpu_reduction_compute_out_offset(out_values, out_coords);
        size_t base_offset = cpu_reduction_compute_base_offset(geo, input, out_coords);

        if (geo->reduce_ndim == 0 || geo->reduce_total == 0) {
            double value;
            if (!cpu_reduction_load_element(input->dtype, in_data, base_offset, &value)) {
                marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Reduction dtype not supported on CPU");
                return MARMOT_ERROR_UNSUPPORTED_DTYPE;
            }

            if (!cpu_reduction_store_result(out_values->dtype, out_data, out_offset, value)) {
                marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Reduction result dtype not supported");
                return MARMOT_ERROR_UNSUPPORTED_DTYPE;
            }

            if (cpu_reduction_is_arg_op(op) && indices_data != nullptr) {
                size_t idx_offset = cpu_reduction_compute_out_offset(out_indices, out_coords);
                ((uint64_t *)indices_data)[idx_offset] = 0;
            }
            continue;
        }

        memset(reduce_coords, 0, geo->reduce_ndim * sizeof(size_t));

        if (ctx != nullptr && geo->reduce_ndim == 1 && input->shape.strides[geo->reduce_axes[0]] == 1) {
            const size_t dtype_bytes = marmot_dtype_size(input->dtype);
            if (dtype_bytes > 0) {
                const size_t length = geo->reduce_shape[0];
                const uint8_t *base_bytes = ((const uint8_t *)in_data) + base_offset * dtype_bytes;
                double fast_value = 0.0;
                uint64_t fast_index = 0;
                uint64_t *fast_index_ptr = cpu_reduction_is_arg_op(op) ? &fast_index : nullptr;
                marmot_error_t fast_status = cpu_reduction_try_microkernel(
                    ctx, op, input->dtype, base_bytes, length, &fast_value, fast_index_ptr
                );
                if (fast_status == MARMOT_SUCCESS) {
                    if (!cpu_reduction_store_result(out_values->dtype, out_data, out_offset, fast_value)) {
                        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Reduction result dtype not supported");
                        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
                    }
                    if (fast_index_ptr != nullptr && indices_data != nullptr) {
                        size_t idx_offset = cpu_reduction_compute_out_offset(out_indices, out_coords);
                        ((uint64_t *)indices_data)[idx_offset] = fast_index;
                    }
                    continue;
                }
                if (fast_status != MARMOT_ERROR_NOT_IMPLEMENTED) {
                    return fast_status;
                }
            }
        }

        double sum = 0.0;
        double prod = 1.0;
        double sum_abs = 0.0;
        double sum_sq = 0.0;
        double mean = 0.0;
        double m2 = 0.0;
        double best = 0.0;
        size_t best_linear = 0;
        bool have_best = false;
        bool any = false;
        bool all = true;
        size_t n = 0;

        for (;;) {
            size_t offset = base_offset;
            size_t reduce_linear = 0;
            for (size_t r = 0; r < geo->reduce_ndim; ++r) {
                offset += reduce_coords[r] * input->shape.strides[geo->reduce_axes[r]];
                reduce_linear += reduce_coords[r] * geo->reduce_linear_strides[r];
            }

            double value;
            if (!cpu_reduction_load_element(input->dtype, in_data, offset, &value)) {
                marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Reduction dtype not supported on CPU");
                return MARMOT_ERROR_UNSUPPORTED_DTYPE;
            }

            switch (op) {
            case MARMOT_DEVICE_REDUCTION_SUM:
            case MARMOT_DEVICE_REDUCTION_MEAN:
                sum += value;
                break;
            case MARMOT_DEVICE_REDUCTION_PROD:
                prod *= value;
                break;
            case MARMOT_DEVICE_REDUCTION_MAX:
            case MARMOT_DEVICE_REDUCTION_ARGMAX:
                if (!have_best || value > best || (value == best && reduce_linear < best_linear)) {
                    best = value;
                    best_linear = reduce_linear;
                    have_best = true;
                }
                break;
            case MARMOT_DEVICE_REDUCTION_MIN:
            case MARMOT_DEVICE_REDUCTION_ARGMIN:
                if (!have_best || value < best || (value == best && reduce_linear < best_linear)) {
                    best = value;
                    best_linear = reduce_linear;
                    have_best = true;
                }
                break;
            case MARMOT_DEVICE_REDUCTION_ANY:
                if (!any && value != 0.0) {
                    any = true;
                }
                break;
            case MARMOT_DEVICE_REDUCTION_ALL:
                if (all && value == 0.0) {
                    all = false;
                }
                break;
            case MARMOT_DEVICE_REDUCTION_VARIANCE:
            case MARMOT_DEVICE_REDUCTION_STD: {
                double delta = value - mean;
                mean += delta / (double)(n + 1);
                double delta2 = value - mean;
                m2 += delta * delta2;
                break;
            }
            case MARMOT_DEVICE_REDUCTION_NORM_L1:
                sum_abs += fabs(value);
                break;
            case MARMOT_DEVICE_REDUCTION_NORM_L2:
                sum_sq += value * value;
                break;
            default:
                marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Reduction operation not implemented on CPU");
                return MARMOT_ERROR_NOT_IMPLEMENTED;
            }

            n++;

            if (!cpu_reduction_increment(reduce_coords, geo->reduce_shape, geo->reduce_ndim)) {
                break;
            }
        }

        if (n == 0) {
            marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "Reduction over zero elements is undefined");
            return MARMOT_ERROR_INVALID_OPERATION;
        }

        double result = 0.0;

        switch (op) {
        case MARMOT_DEVICE_REDUCTION_SUM:
            result = sum;
            break;
        case MARMOT_DEVICE_REDUCTION_MEAN:
            result = sum / (double)n;
            break;
        case MARMOT_DEVICE_REDUCTION_PROD:
            result = prod;
            break;
        case MARMOT_DEVICE_REDUCTION_MAX:
        case MARMOT_DEVICE_REDUCTION_ARGMAX:
            result = have_best ? best : 0.0;
            break;
        case MARMOT_DEVICE_REDUCTION_MIN:
        case MARMOT_DEVICE_REDUCTION_ARGMIN:
            result = have_best ? best : 0.0;
            break;
        case MARMOT_DEVICE_REDUCTION_ANY:
            result = any ? 1.0 : 0.0;
            break;
        case MARMOT_DEVICE_REDUCTION_ALL:
            result = all ? 1.0 : 0.0;
            break;
        case MARMOT_DEVICE_REDUCTION_VARIANCE:
        case MARMOT_DEVICE_REDUCTION_STD: {
            double epsilon = params != nullptr ? (double)params->epsilon : 0.0;
            bool unbiased = params != nullptr && params->unbiased;
            double denom = unbiased && n > 1 ? (double)(n - 1) : (double)n;
            if (denom <= 0.0) {
                denom = 1.0;
            }
            double variance = (n > 0 ? (m2 / denom) : 0.0) + epsilon;
            if (op == MARMOT_DEVICE_REDUCTION_STD) {
                variance = sqrt(variance);
            }
            result = variance;
            break;
        }
        case MARMOT_DEVICE_REDUCTION_NORM_L1:
            result = sum_abs;
            break;
        case MARMOT_DEVICE_REDUCTION_NORM_L2:
            result = sqrt(sum_sq);
            break;
        default:
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Reduction operation not implemented on CPU");
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }

        if (!cpu_reduction_store_result(out_values->dtype, out_data, out_offset, result)) {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Reduction result dtype not supported");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }

        if (cpu_reduction_is_arg_op(op) && indices_data != nullptr) {
            size_t idx_offset = cpu_reduction_compute_out_offset(out_indices, out_coords);
            ((uint64_t *)indices_data)[idx_offset] = (uint64_t)best_linear;
        }
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduction_execute(
    const void *device_ctx, marmot_device_reduction_op_t op, const marmot_tensor_t *input, marmot_tensor_t *out_values,
    marmot_tensor_t *out_indices, const marmot_reduction_params_t *params
) {
    cpu_context_t *ctx = get_cpu_context(device_ctx);

    if (input == nullptr || out_values == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensor pointer in reduction");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (!marmot_dtype_supports_reduction(input->dtype)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Input dtype does not support reductions");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    bool require_indices = cpu_reduction_is_arg_op(op);
    cpu_reduction_geometry_t geo;
    marmot_error_t status =
        cpu_prepare_reduction_geometry(input, params, out_values, out_indices, require_indices, &geo);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    return cpu_reduction_run_numeric(ctx, op, input, params, &geo, out_values, out_indices);
}

marmot_error_t cpu_reduction_sum_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    return cpu_reduction_execute(device_ctx, MARMOT_DEVICE_REDUCTION_SUM, input, out_values, out_indices, params);
}

marmot_error_t cpu_reduction_mean_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    return cpu_reduction_execute(device_ctx, MARMOT_DEVICE_REDUCTION_MEAN, input, out_values, out_indices, params);
}

marmot_error_t cpu_reduction_max_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    return cpu_reduction_execute(device_ctx, MARMOT_DEVICE_REDUCTION_MAX, input, out_values, out_indices, params);
}

marmot_error_t cpu_reduction_min_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    return cpu_reduction_execute(device_ctx, MARMOT_DEVICE_REDUCTION_MIN, input, out_values, out_indices, params);
}

marmot_error_t cpu_reduction_variance_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    return cpu_reduction_execute(device_ctx, MARMOT_DEVICE_REDUCTION_VARIANCE, input, out_values, out_indices, params);
}

marmot_error_t cpu_reduction_std_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    return cpu_reduction_execute(device_ctx, MARMOT_DEVICE_REDUCTION_STD, input, out_values, out_indices, params);
}

marmot_error_t cpu_reduction_norm_l1_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    return cpu_reduction_execute(device_ctx, MARMOT_DEVICE_REDUCTION_NORM_L1, input, out_values, out_indices, params);
}

marmot_error_t cpu_reduction_norm_l2_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    return cpu_reduction_execute(device_ctx, MARMOT_DEVICE_REDUCTION_NORM_L2, input, out_values, out_indices, params);
}

marmot_error_t cpu_reduction_prod_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    return cpu_reduction_execute(device_ctx, MARMOT_DEVICE_REDUCTION_PROD, input, out_values, out_indices, params);
}

marmot_error_t cpu_reduction_argmax_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    return cpu_reduction_execute(device_ctx, MARMOT_DEVICE_REDUCTION_ARGMAX, input, out_values, out_indices, params);
}

marmot_error_t cpu_reduction_argmin_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    return cpu_reduction_execute(device_ctx, MARMOT_DEVICE_REDUCTION_ARGMIN, input, out_values, out_indices, params);
}

marmot_error_t cpu_reduction_any_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    return cpu_reduction_execute(device_ctx, MARMOT_DEVICE_REDUCTION_ANY, input, out_values, out_indices, params);
}

marmot_error_t cpu_reduction_all_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    return cpu_reduction_execute(device_ctx, MARMOT_DEVICE_REDUCTION_ALL, input, out_values, out_indices, params);
}

marmot_error_t cpu_reduction(
    const void *device_ctx, marmot_device_reduction_op_t op, const marmot_tensor_t *input, marmot_tensor_t *out_values,
    marmot_tensor_t *out_indices, const marmot_reduction_params_t *params
) {
    switch (op) {
    case MARMOT_DEVICE_REDUCTION_SUM:
        return cpu_reduction_sum_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_MEAN:
        return cpu_reduction_mean_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_MAX:
        return cpu_reduction_max_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_MIN:
        return cpu_reduction_min_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_VARIANCE:
        return cpu_reduction_variance_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_STD:
        return cpu_reduction_std_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_NORM_L1:
        return cpu_reduction_norm_l1_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_NORM_L2:
        return cpu_reduction_norm_l2_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_PROD:
        return cpu_reduction_prod_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_ARGMAX:
        return cpu_reduction_argmax_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_ARGMIN:
        return cpu_reduction_argmin_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_ANY:
        return cpu_reduction_any_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_ALL:
        return cpu_reduction_all_impl(device_ctx, input, out_values, out_indices, params);
    default:
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Reduction operation not implemented on CPU");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
}
