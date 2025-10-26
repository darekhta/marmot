#include <stdlib.h>

#include <string.h>

#include "core/helpers/elementwise.h"
#include "cpu_backend_internal.h"

extern const cpu_unary_traits_t cpu_unary_f32_scalar_traits;
extern const cpu_unary_traits_t cpu_unary_f64_scalar_traits;
extern const cpu_unary_traits_t cpu_unary_f16_scalar_traits;
extern const cpu_unary_traits_t cpu_unary_bf16_scalar_traits;
#if MARMOT_ENABLE_FP8
extern const cpu_unary_traits_t cpu_unary_fp8_e4m3_scalar_traits;
extern const cpu_unary_traits_t cpu_unary_fp8_e5m2_scalar_traits;
#endif
extern const cpu_unary_traits_t cpu_unary_i8_scalar_traits;
extern const cpu_unary_traits_t cpu_unary_i16_scalar_traits;
extern const cpu_unary_traits_t cpu_unary_i32_scalar_traits;
extern const cpu_unary_traits_t cpu_unary_i64_scalar_traits;
extern const cpu_unary_traits_t cpu_unary_u8_scalar_traits;
extern const cpu_unary_traits_t cpu_unary_u16_scalar_traits;
extern const cpu_unary_traits_t cpu_unary_u32_scalar_traits;
extern const cpu_unary_traits_t cpu_unary_u64_scalar_traits;

#if HAS_NEON
extern const cpu_unary_traits_t cpu_unary_f32_neon_traits;
#endif

#if HAS_AVX2
extern const cpu_unary_traits_t cpu_unary_f32_avx2_traits;
extern const cpu_unary_traits_t cpu_unary_f16_avx2_traits;
extern const cpu_unary_traits_t cpu_unary_bf16_avx2_traits;
#if MARMOT_ENABLE_FP8
extern const cpu_unary_traits_t cpu_unary_fp8_e4m3_avx2_traits;
extern const cpu_unary_traits_t cpu_unary_fp8_e5m2_avx2_traits;
#endif
#endif

#if MARMOT_ENABLE_ACCELERATE
extern const cpu_unary_traits_t cpu_unary_f32_accelerate_traits;
extern const cpu_unary_traits_t cpu_unary_f64_accelerate_traits;
extern const cpu_unary_traits_t cpu_unary_f16_accelerate_traits;
extern const cpu_unary_traits_t cpu_unary_bf16_accelerate_traits;
#endif

static const cpu_unary_ops_t *const k_cpu_unary_scalar_ops[MARMOT_DTYPE_COUNT] = {
    [MARMOT_DTYPE_FLOAT32] = &cpu_unary_f32_scalar_traits.ops,
    [MARMOT_DTYPE_FLOAT64] = &cpu_unary_f64_scalar_traits.ops,
    [MARMOT_DTYPE_FLOAT16] = &cpu_unary_f16_scalar_traits.ops,
    [MARMOT_DTYPE_BFLOAT16] = &cpu_unary_bf16_scalar_traits.ops,
#if MARMOT_ENABLE_FP8
    [MARMOT_DTYPE_FLOAT8_E4M3] = &cpu_unary_fp8_e4m3_scalar_traits.ops,
    [MARMOT_DTYPE_FLOAT8_E5M2] = &cpu_unary_fp8_e5m2_scalar_traits.ops,
#endif
    [MARMOT_DTYPE_INT8] = &cpu_unary_i8_scalar_traits.ops,
    [MARMOT_DTYPE_INT16] = &cpu_unary_i16_scalar_traits.ops,
    [MARMOT_DTYPE_INT32] = &cpu_unary_i32_scalar_traits.ops,
    [MARMOT_DTYPE_INT64] = &cpu_unary_i64_scalar_traits.ops,
    [MARMOT_DTYPE_UINT8] = &cpu_unary_u8_scalar_traits.ops,
    [MARMOT_DTYPE_UINT16] = &cpu_unary_u16_scalar_traits.ops,
    [MARMOT_DTYPE_UINT32] = &cpu_unary_u32_scalar_traits.ops,
    [MARMOT_DTYPE_UINT64] = &cpu_unary_u64_scalar_traits.ops,
};

#if HAS_NEON
static const cpu_unary_ops_t *const k_cpu_unary_neon_ops[MARMOT_DTYPE_COUNT] = {
    [MARMOT_DTYPE_FLOAT32] = &cpu_unary_f32_neon_traits.ops,
};
#endif

#if HAS_AVX2
static const cpu_unary_ops_t *const k_cpu_unary_avx2_ops[MARMOT_DTYPE_COUNT] = {
    [MARMOT_DTYPE_FLOAT32] = &cpu_unary_f32_avx2_traits.ops,
    [MARMOT_DTYPE_FLOAT16] = &cpu_unary_f16_avx2_traits.ops,
    [MARMOT_DTYPE_BFLOAT16] = &cpu_unary_bf16_avx2_traits.ops,
#if MARMOT_ENABLE_FP8
    [MARMOT_DTYPE_FLOAT8_E4M3] = &cpu_unary_fp8_e4m3_avx2_traits.ops,
    [MARMOT_DTYPE_FLOAT8_E5M2] = &cpu_unary_fp8_e5m2_avx2_traits.ops,
#endif
};
#endif

#if MARMOT_ENABLE_ACCELERATE
static const cpu_unary_ops_t *const k_cpu_unary_accelerate_ops[MARMOT_DTYPE_COUNT] = {
    [MARMOT_DTYPE_FLOAT32] = &cpu_unary_f32_accelerate_traits.ops,
    [MARMOT_DTYPE_FLOAT64] = &cpu_unary_f64_accelerate_traits.ops,
    [MARMOT_DTYPE_FLOAT16] = &cpu_unary_f16_accelerate_traits.ops,
    [MARMOT_DTYPE_BFLOAT16] = &cpu_unary_bf16_accelerate_traits.ops,
};
#endif

static cpu_unary_simple_fn cpu_unary_simple_fn_from_ops(const cpu_unary_ops_t *ops, marmot_device_unary_op_t op) {
    if (ops == nullptr) {
        return nullptr;
    }
    switch (op) {
    case MARMOT_DEVICE_UNARY_ABS:
        return ops->abs;
    case MARMOT_DEVICE_UNARY_NEG:
        return ops->neg;
    case MARMOT_DEVICE_UNARY_SIGN:
        return ops->sign;
    case MARMOT_DEVICE_UNARY_SQRT:
        return ops->sqrt;
    case MARMOT_DEVICE_UNARY_EXP:
        return ops->exp;
    case MARMOT_DEVICE_UNARY_LOG:
        return ops->log;
    case MARMOT_DEVICE_UNARY_BITWISE_NOT:
        return ops->bitwise_not;
    default:
        return nullptr;
    }
}

static cpu_activation_fn cpu_unary_activation_fn_from_ops(const cpu_unary_ops_t *ops, marmot_device_unary_op_t op) {
    if (ops == nullptr) {
        return nullptr;
    }
    switch (op) {
    case MARMOT_DEVICE_UNARY_RELU:
        return ops->relu;
    case MARMOT_DEVICE_UNARY_GELU:
        return ops->gelu;
    case MARMOT_DEVICE_UNARY_GELU_TANH:
        return ops->gelu_tanh;
    case MARMOT_DEVICE_UNARY_SILU:
        return ops->silu;
    case MARMOT_DEVICE_UNARY_SIGMOID:
        return ops->sigmoid;
    case MARMOT_DEVICE_UNARY_TANH:
        return ops->tanh_act;
    case MARMOT_DEVICE_UNARY_MISH:
        return ops->mish;
    case MARMOT_DEVICE_UNARY_ELU:
        return ops->elu;
    case MARMOT_DEVICE_UNARY_SELU:
        return ops->selu;
    case MARMOT_DEVICE_UNARY_LEAKY_RELU:
        return ops->leaky_relu;
    case MARMOT_DEVICE_UNARY_PRELU:
        return ops->prelu;
    default:
        return nullptr;
    }
}

static cpu_fused_bias_activation_fn
cpu_unary_fused_bias_fn_from_ops(const cpu_unary_ops_t *ops, marmot_device_unary_op_t op) {
    if (ops == nullptr) {
        return nullptr;
    }
    switch (op) {
    case MARMOT_DEVICE_UNARY_RELU:
        return ops->fused_bias_relu;
    case MARMOT_DEVICE_UNARY_GELU:
        return ops->fused_bias_gelu;
    case MARMOT_DEVICE_UNARY_GELU_TANH:
        return ops->fused_bias_gelu_tanh;
    case MARMOT_DEVICE_UNARY_SILU:
        return ops->fused_bias_silu;
    case MARMOT_DEVICE_UNARY_SIGMOID:
        return ops->fused_bias_sigmoid;
    case MARMOT_DEVICE_UNARY_TANH:
        return ops->fused_bias_tanh;
    case MARMOT_DEVICE_UNARY_MISH:
        return ops->fused_bias_mish;
    case MARMOT_DEVICE_UNARY_ELU:
        return ops->fused_bias_elu;
    case MARMOT_DEVICE_UNARY_SELU:
        return ops->fused_bias_selu;
    case MARMOT_DEVICE_UNARY_LEAKY_RELU:
        return ops->fused_bias_leaky_relu;
    case MARMOT_DEVICE_UNARY_PRELU:
        return ops->fused_bias_prelu;
    default:
        return nullptr;
    }
}

static cpu_unary_simple_fn
cpu_unary_resolve_simple_fn(const void *device_ctx, marmot_dtype_t dtype, marmot_device_unary_op_t op) {
    if (dtype >= MARMOT_DTYPE_COUNT) {
        return nullptr;
    }

#if MARMOT_ENABLE_ACCELERATE
    if (has_accelerate(device_ctx)) {
        const cpu_unary_ops_t *ops = k_cpu_unary_accelerate_ops[dtype];
        cpu_unary_simple_fn fn = cpu_unary_simple_fn_from_ops(ops, op);
        if (fn != nullptr) {
            return fn;
        }
    }
#endif

#if HAS_AVX2
    if (has_avx2(device_ctx)) {
        const cpu_unary_ops_t *ops = k_cpu_unary_avx2_ops[dtype];
        cpu_unary_simple_fn fn = cpu_unary_simple_fn_from_ops(ops, op);
        if (fn != nullptr) {
            return fn;
        }
    }
#endif

#if HAS_NEON
    if (has_neon(device_ctx)) {
        const cpu_unary_ops_t *ops = k_cpu_unary_neon_ops[dtype];
        cpu_unary_simple_fn fn = cpu_unary_simple_fn_from_ops(ops, op);
        if (fn != nullptr) {
            return fn;
        }
    }
#endif

    return cpu_unary_simple_fn_from_ops(k_cpu_unary_scalar_ops[dtype], op);
}

static cpu_activation_fn
cpu_unary_resolve_activation_fn(const void *device_ctx, marmot_dtype_t dtype, marmot_device_unary_op_t op) {
    if (dtype >= MARMOT_DTYPE_COUNT) {
        return nullptr;
    }

#if MARMOT_ENABLE_ACCELERATE
    if (has_accelerate(device_ctx)) {
        const cpu_unary_ops_t *ops = k_cpu_unary_accelerate_ops[dtype];
        cpu_activation_fn fn = cpu_unary_activation_fn_from_ops(ops, op);
        if (fn != nullptr) {
            return fn;
        }
    }
#endif

#if HAS_AVX2
    if (has_avx2(device_ctx)) {
        const cpu_unary_ops_t *ops = k_cpu_unary_avx2_ops[dtype];
        cpu_activation_fn fn = cpu_unary_activation_fn_from_ops(ops, op);
        if (fn != nullptr) {
            return fn;
        }
    }
#endif

#if HAS_NEON
    if (has_neon(device_ctx)) {
        const cpu_unary_ops_t *ops = k_cpu_unary_neon_ops[dtype];
        cpu_activation_fn fn = cpu_unary_activation_fn_from_ops(ops, op);
        if (fn != nullptr) {
            return fn;
        }
    }
#endif

    return cpu_unary_activation_fn_from_ops(k_cpu_unary_scalar_ops[dtype], op);
}

static cpu_fused_bias_activation_fn
cpu_unary_resolve_fused_fn(const void *device_ctx, marmot_dtype_t dtype, marmot_device_unary_op_t op) {
    if (dtype >= MARMOT_DTYPE_COUNT) {
        return nullptr;
    }

#if MARMOT_ENABLE_ACCELERATE
    if (has_accelerate(device_ctx)) {
        const cpu_unary_ops_t *ops = k_cpu_unary_accelerate_ops[dtype];
        cpu_fused_bias_activation_fn fn = cpu_unary_fused_bias_fn_from_ops(ops, op);
        if (fn != nullptr) {
            return fn;
        }
    }
#endif

#if HAS_AVX2
    if (has_avx2(device_ctx)) {
        const cpu_unary_ops_t *ops = k_cpu_unary_avx2_ops[dtype];
        cpu_fused_bias_activation_fn fn = cpu_unary_fused_bias_fn_from_ops(ops, op);
        if (fn != nullptr) {
            return fn;
        }
    }
#endif

#if HAS_NEON
    if (has_neon(device_ctx)) {
        const cpu_unary_ops_t *ops = k_cpu_unary_neon_ops[dtype];
        cpu_fused_bias_activation_fn fn = cpu_unary_fused_bias_fn_from_ops(ops, op);
        if (fn != nullptr) {
            return fn;
        }
    }
#endif

    return cpu_unary_fused_bias_fn_from_ops(k_cpu_unary_scalar_ops[dtype], op);
}

static size_t cpu_unary_tensor_offset_linear(const marmot_tensor_t *tensor, size_t linear_index) {
    if (tensor == nullptr || tensor->shape.ndim == 0) {
        return 0;
    }
    size_t offset = 0;
    size_t remaining = linear_index;
    for (size_t axis = tensor->shape.ndim; axis-- > 0;) {
        size_t dim = tensor->shape.shape[axis];
        if (dim == 0) {
            return 0;
        }
        size_t idx = remaining % dim;
        remaining /= dim;
        offset += idx * tensor->shape.strides[axis];
    }
    return offset;
}

static bool cpu_unary_last_dim_contiguous(const marmot_tensor_t *tensor) {
    if (tensor == nullptr || tensor->shape.ndim == 0) {
        return true;
    }
    return tensor->shape.strides[tensor->shape.ndim - 1] == 1;
}

static bool cpu_unary_tensor_is_contiguous(const marmot_tensor_t *tensor) {
    if (tensor == nullptr || tensor->shape.ndim == 0) {
        return true;
    }
    size_t expected_stride = 1;
    for (size_t axis = tensor->shape.ndim; axis-- > 0;) {
        size_t dim = tensor->shape.shape[axis];
        if (dim == 0) {
            return true;
        }
        if (tensor->shape.strides[axis] != expected_stride) {
            return false;
        }
        expected_stride *= dim;
    }
    return true;
}

static marmot_error_t cpu_unary_apply_identity(const marmot_tensor_t *x, marmot_tensor_t *out) {
    if (x->dtype != out->dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Identity requires matching dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    size_t x_elems = marmot_tensor_num_elements(x);
    size_t out_elems = marmot_tensor_num_elements(out);
    if (x_elems != out_elems) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Identity requires matching element counts");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    size_t bytes = marmot_tensor_size_bytes(x);
    if (bytes == 0 || x->data == nullptr || out->data == nullptr) {
        return MARMOT_SUCCESS;
    }
    if (x->data == out->data) {
        return MARMOT_SUCCESS;
    }
    memcpy(out->data, x->data, bytes);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_bias_fallback(
    const void *device_ctx, cpu_activation_fn activation_fn, const marmot_tensor_t *x, marmot_tensor_t *out,
    const marmot_tensor_t *bias, bool bias_is_scalar, size_t feature_dim, const marmot_activation_params_t *params
) {
    size_t total = marmot_tensor_num_elements(x);
    size_t dtype_bytes = marmot_dtype_size(x->dtype);
    if (dtype_bytes == 0) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unary operation received invalid dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (total == 0) {
        return MARMOT_SUCCESS;
    }
    uint8_t *staging = (uint8_t *)malloc(dtype_bytes * total);
    if (staging == nullptr) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate unary staging buffer");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    float bias_scalar_value = 0.0f;
    if (bias_is_scalar) {
        bias_scalar_value = cpu_load_as_f32(bias->dtype, bias->data, 0);
    }

    for (size_t idx = 0; idx < total; ++idx) {
        size_t x_offset = cpu_unary_tensor_offset_linear(x, idx);
        float value = cpu_load_as_f32(x->dtype, x->data, x_offset);
        float bias_value;
        if (bias_is_scalar) {
            bias_value = bias_scalar_value;
        } else {
            size_t bias_index = feature_dim == 0 ? 0 : (idx % feature_dim);
            size_t bias_offset = cpu_unary_tensor_offset_linear(bias, bias_index);
            bias_value = cpu_load_as_f32(bias->dtype, bias->data, bias_offset);
        }
        float biased = value + bias_value;
        cpu_store_from_f32(x->dtype, staging, idx, biased);
    }

    marmot_activation_params_t params_copy;
    const marmot_activation_params_t *activation_params = params;
    if (activation_params != nullptr && activation_params->bias != nullptr) {
        params_copy = *activation_params;
        params_copy.bias = nullptr;
        activation_params = &params_copy;
    }

    if (activation_fn == nullptr) {
        free(staging);
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Activation unavailable for fallback path");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    marmot_error_t status = activation_fn(device_ctx, staging, activation_params, staging, total);
    if (status != MARMOT_SUCCESS) {
        free(staging);
        return status;
    }

    for (size_t idx = 0; idx < total; ++idx) {
        size_t out_offset = cpu_unary_tensor_offset_linear(out, idx);
        float value = cpu_load_as_f32(x->dtype, staging, idx);
        cpu_store_from_f32(out->dtype, out->data, out_offset, value);
    }

    free(staging);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_simple_fallback(
    const void *device_ctx, cpu_unary_simple_fn simple, const marmot_tensor_t *x, marmot_tensor_t *out, size_t n
) {
    if (simple == nullptr) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unary operation unsupported for dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    size_t dtype_bytes = marmot_dtype_size(x->dtype);
    if (dtype_bytes == 0) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unary operation received invalid dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    uint8_t *tmp_in = (uint8_t *)malloc(dtype_bytes * n);
    uint8_t *tmp_out = (uint8_t *)malloc(dtype_bytes * n);
    if (tmp_in == nullptr || tmp_out == nullptr) {
        free(tmp_in);
        free(tmp_out);
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate unary staging buffer");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    const uint8_t *src_bytes = (const uint8_t *)x->data;
    for (size_t idx = 0; idx < n; ++idx) {
        size_t offset = cpu_unary_tensor_offset_linear(x, idx);
        memcpy(tmp_in + idx * dtype_bytes, src_bytes + offset * dtype_bytes, dtype_bytes);
    }

    marmot_error_t status = simple(device_ctx, tmp_in, tmp_out, n);
    if (status != MARMOT_SUCCESS) {
        free(tmp_in);
        free(tmp_out);
        if (status == MARMOT_ERROR_NOT_IMPLEMENTED) {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unary operation unsupported for dtype");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        return status;
    }

    uint8_t *dst_bytes = (uint8_t *)out->data;
    size_t out_capacity = out->capacity_bytes;
    for (size_t idx = 0; idx < n; ++idx) {
        size_t offset = cpu_unary_tensor_offset_linear(out, idx);
        size_t byte_offset = offset * dtype_bytes;
        if (byte_offset + dtype_bytes > out_capacity) {
            free(tmp_in);
            free(tmp_out);
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Unary output tensor is smaller than the input tensor");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        memcpy(dst_bytes + byte_offset, tmp_out + idx * dtype_bytes, dtype_bytes);
    }

    free(tmp_in);
    free(tmp_out);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_activation_fallback(
    const void *device_ctx, cpu_activation_fn activation_fn, const marmot_tensor_t *x, marmot_tensor_t *out, size_t n,
    const marmot_activation_params_t *params
) {
    if (activation_fn == nullptr) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Activation unsupported for dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    size_t dtype_bytes = marmot_dtype_size(x->dtype);
    if (dtype_bytes == 0) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unary operation received invalid dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }

    uint8_t *tmp_in = (uint8_t *)malloc(dtype_bytes * n);
    uint8_t *tmp_out = (uint8_t *)malloc(dtype_bytes * n);
    if (tmp_in == nullptr || tmp_out == nullptr) {
        free(tmp_in);
        free(tmp_out);
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate unary staging buffer");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    const uint8_t *src_bytes = (const uint8_t *)x->data;
    for (size_t idx = 0; idx < n; ++idx) {
        size_t offset = cpu_unary_tensor_offset_linear(x, idx);
        memcpy(tmp_in + idx * dtype_bytes, src_bytes + offset * dtype_bytes, dtype_bytes);
    }

    marmot_error_t status = activation_fn(device_ctx, tmp_in, params, tmp_out, n);
    if (status != MARMOT_SUCCESS) {
        free(tmp_in);
        free(tmp_out);
        if (status == MARMOT_ERROR_NOT_IMPLEMENTED) {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Activation unsupported for dtype");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        return status;
    }

    uint8_t *dst_bytes = (uint8_t *)out->data;
    size_t out_capacity = out->capacity_bytes;
    for (size_t idx = 0; idx < n; ++idx) {
        size_t offset = cpu_unary_tensor_offset_linear(out, idx);
        size_t byte_offset = offset * dtype_bytes;
        if (byte_offset + dtype_bytes > out_capacity) {
            free(tmp_in);
            free(tmp_out);
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Unary output tensor is smaller than the input tensor");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        memcpy(dst_bytes + byte_offset, tmp_out + idx * dtype_bytes, dtype_bytes);
    }

    free(tmp_in);
    free(tmp_out);
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_unary_apply(
    const void *device_ctx, marmot_device_unary_op_t op, const marmot_tensor_t *x,
    const marmot_activation_params_t *params, marmot_tensor_t *out
) {
    VALIDATE_TENSORS_2(x, out);
    if (op == MARMOT_DEVICE_UNARY_IDENTITY) {
        return cpu_unary_apply_identity(x, out);
    }
    if (x->dtype >= MARMOT_DTYPE_COUNT || out->dtype >= MARMOT_DTYPE_COUNT) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unary operation received invalid dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (x->dtype != out->dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unary operation requires matching input/output dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    size_t in_elems = marmot_tensor_num_elements(x);
    size_t dtype_bytes = marmot_dtype_size(x->dtype);
    if (dtype_bytes == 0) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unary operation received invalid dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    size_t required_bytes = in_elems * dtype_bytes;
    if (out->capacity_bytes < required_bytes) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Unary output tensor is smaller than the input tensor");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (in_elems == 0) {
        return MARMOT_SUCCESS;
    }
    if (x->data == nullptr || out->data == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unary tensor data pointers cannot be null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    cpu_context_t *ctx = get_cpu_context(device_ctx);
    if (ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    (void)ctx;

    marmot_activation_params_t prepared_params;
    const marmot_activation_params_t *effective_params = params;
    if (marmot_unary_op_requires_params(op) || params != nullptr) {
        marmot_error_t prep_err = marmot_unary_prepare_activation_params(op, params, &prepared_params);
        if (prep_err != MARMOT_SUCCESS) {
            return prep_err;
        }
        effective_params = &prepared_params;
    }
    const marmot_tensor_t *bias_tensor = (effective_params != nullptr) ? effective_params->bias : nullptr;
    bool has_bias = bias_tensor != nullptr;
    if (has_bias && !marmot_elementwise_unary_supports_bias(op)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Bias is only supported for activation functions");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (has_bias) {
        if (bias_tensor->dtype != x->dtype) {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Bias tensor must match activation dtype");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        if (bias_tensor->data == nullptr && marmot_tensor_num_elements(bias_tensor) > 0) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Bias tensor data cannot be null");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
    }

    cpu_unary_simple_fn simple = cpu_unary_resolve_simple_fn(device_ctx, x->dtype, op);
    if (simple != nullptr) {
        if (has_bias) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Bias is not supported for this unary operation");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        bool contiguous = cpu_unary_tensor_is_contiguous(x) && cpu_unary_tensor_is_contiguous(out);
        if (contiguous) {
            marmot_error_t status = simple(device_ctx, x->data, out->data, in_elems);
            if (status == MARMOT_ERROR_NOT_IMPLEMENTED) {
                marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unary operation unsupported for dtype");
                return MARMOT_ERROR_UNSUPPORTED_DTYPE;
            }
            return status;
        }
        return cpu_unary_simple_fallback(device_ctx, simple, x, out, in_elems);
    }

    cpu_activation_fn activation_fn = cpu_unary_resolve_activation_fn(device_ctx, x->dtype, op);
    if (!has_bias) {
        bool contiguous = cpu_unary_tensor_is_contiguous(x) && cpu_unary_tensor_is_contiguous(out);
        if (contiguous) {
            if (activation_fn == nullptr) {
                marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Activation unsupported for this dtype");
                return MARMOT_ERROR_NOT_IMPLEMENTED;
            }
            marmot_error_t status = activation_fn(device_ctx, x->data, effective_params, out->data, in_elems);
            if (status == MARMOT_ERROR_NOT_IMPLEMENTED) {
                marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Activation unsupported for this dtype");
                return MARMOT_ERROR_UNSUPPORTED_DTYPE;
            }
            return status;
        }

        marmot_error_t status =
            cpu_unary_activation_fallback(device_ctx, activation_fn, x, out, in_elems, effective_params);
        if (status == MARMOT_ERROR_NOT_IMPLEMENTED) {
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Activation unsupported for this dtype");
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        return status;
    }

    size_t feature_dim = 0;
    bool bias_is_scalar = false;
    marmot_error_t bias_status = marmot_elementwise_bias_info(x, bias_tensor, &feature_dim, &bias_is_scalar);
    if (bias_status != MARMOT_SUCCESS) {
        return bias_status;
    }
    bool last_dim_contiguous = cpu_unary_last_dim_contiguous(x) && cpu_unary_last_dim_contiguous(out);
    bool bias_contiguous = bias_is_scalar || cpu_unary_last_dim_contiguous(bias_tensor);
    bool can_use_fast_bias = (bias_is_scalar || (last_dim_contiguous && bias_contiguous));
    cpu_fused_bias_activation_fn fused_fn = cpu_unary_resolve_fused_fn(device_ctx, x->dtype, op);
    if (fused_fn != nullptr && can_use_fast_bias) {
        marmot_error_t status = fused_fn(
            device_ctx, x->data, bias_tensor->data, feature_dim, bias_is_scalar, effective_params, out->data, in_elems
        );
        if (status != MARMOT_ERROR_NOT_IMPLEMENTED) {
            return status;
        }
    }

    return cpu_unary_bias_fallback(
        device_ctx, activation_fn, x, out, bias_tensor, bias_is_scalar, feature_dim, effective_params
    );
}
