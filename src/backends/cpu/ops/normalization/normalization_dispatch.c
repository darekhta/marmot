#include "marmot/tensor.h"

#include "core/helpers/norm.h"
#include "ops/normalization/normalization_internal.h"

static inline const void *cpu_norm_tensor_data(const marmot_tensor_t *tensor) {
    return tensor != nullptr ? tensor->data : nullptr;
}

extern const cpu_norm_traits_t cpu_norm_f32_scalar_traits;
extern const cpu_norm_traits_t cpu_norm_f64_scalar_traits;
extern const cpu_norm_traits_t cpu_norm_f16_scalar_traits;
extern const cpu_norm_traits_t cpu_norm_bf16_scalar_traits;
#if MARMOT_ENABLE_FP8
extern const cpu_norm_traits_t cpu_norm_fp8_e4m3_traits;
extern const cpu_norm_traits_t cpu_norm_fp8_e5m2_traits;
#endif
#if HAS_NEON
extern const cpu_norm_traits_t cpu_norm_f32_neon_traits;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
extern const cpu_norm_traits_t cpu_norm_f16_neon_traits;
#endif
extern const cpu_norm_traits_t cpu_norm_bf16_neon_traits;
#endif
#if HAS_AVX2
extern const cpu_norm_traits_t cpu_norm_f32_avx2_traits;
extern const cpu_norm_traits_t cpu_norm_bf16_avx2_traits;
#endif

static const cpu_norm_ops_t *const k_cpu_norm_scalar_ops[MARMOT_DTYPE_COUNT] = {
    [MARMOT_DTYPE_FLOAT32] = &cpu_norm_f32_scalar_traits.ops,
    [MARMOT_DTYPE_FLOAT64] = &cpu_norm_f64_scalar_traits.ops,
    [MARMOT_DTYPE_FLOAT16] = &cpu_norm_f16_scalar_traits.ops,
    [MARMOT_DTYPE_BFLOAT16] = &cpu_norm_bf16_scalar_traits.ops,
#if MARMOT_ENABLE_FP8
    [MARMOT_DTYPE_FLOAT8_E4M3] = &cpu_norm_fp8_e4m3_traits.ops,
    [MARMOT_DTYPE_FLOAT8_E5M2] = &cpu_norm_fp8_e5m2_traits.ops,
#endif
};

#if HAS_NEON
static const cpu_norm_ops_t *const k_cpu_norm_neon_ops[MARMOT_DTYPE_COUNT] = {
    [MARMOT_DTYPE_FLOAT32] = &cpu_norm_f32_neon_traits.ops,
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    [MARMOT_DTYPE_FLOAT16] = &cpu_norm_f16_neon_traits.ops,
#endif
    [MARMOT_DTYPE_BFLOAT16] = &cpu_norm_bf16_neon_traits.ops,
};
#endif

#if HAS_AVX2
static const cpu_norm_ops_t *const k_cpu_norm_avx2_ops[MARMOT_DTYPE_COUNT] = {
    [MARMOT_DTYPE_FLOAT32] = &cpu_norm_f32_avx2_traits.ops,
    [MARMOT_DTYPE_BFLOAT16] = &cpu_norm_bf16_avx2_traits.ops,
};
#endif

typedef enum {
    CPU_NORM_OP_LAYERNORM = 0,
    CPU_NORM_OP_RMSNORM = 1,
} cpu_norm_op_type_t;

static bool cpu_norm_ops_supports(const cpu_norm_ops_t *ops, cpu_norm_op_type_t op) {
    if (ops == nullptr) {
        return false;
    }
    switch (op) {
    case CPU_NORM_OP_LAYERNORM:
        return ops->layernorm != nullptr;
    case CPU_NORM_OP_RMSNORM:
        return ops->rmsnorm != nullptr;
    default:
        return false;
    }
}

static const cpu_norm_ops_t *cpu_norm_resolve_ops(const void *device_ctx, marmot_dtype_t dtype, cpu_norm_op_type_t op) {
    if (dtype >= MARMOT_DTYPE_COUNT) {
        return nullptr;
    }

#if HAS_AVX2
    if (has_avx2(device_ctx)) {
        const cpu_norm_ops_t *ops = k_cpu_norm_avx2_ops[dtype];
        if (cpu_norm_ops_supports(ops, op)) {
            return ops;
        }
    }
#endif

#if HAS_NEON
    if (has_neon(device_ctx)) {
        const cpu_norm_ops_t *ops = k_cpu_norm_neon_ops[dtype];
        if (cpu_norm_ops_supports(ops, op)) {
            return ops;
        }
    }
#endif

    const cpu_norm_ops_t *ops = k_cpu_norm_scalar_ops[dtype];
    return cpu_norm_ops_supports(ops, op) ? ops : nullptr;
}

static cpu_layernorm_fn cpu_norm_resolve_layernorm_fn(const void *device_ctx, marmot_dtype_t dtype) {
    const cpu_norm_ops_t *ops = cpu_norm_resolve_ops(device_ctx, dtype, CPU_NORM_OP_LAYERNORM);
    return ops != nullptr ? ops->layernorm : nullptr;
}

static cpu_rmsnorm_fn cpu_norm_resolve_rmsnorm_fn(const void *device_ctx, marmot_dtype_t dtype) {
    const cpu_norm_ops_t *ops = cpu_norm_resolve_ops(device_ctx, dtype, CPU_NORM_OP_RMSNORM);
    return ops != nullptr ? ops->rmsnorm : nullptr;
}

static const marmot_norm_validation_opts_t k_cpu_layernorm_opts = {
    .allow_residual = true,
    .allow_weight = true,
    .require_weight = false,
    .allow_bias = true,
};

static const marmot_norm_validation_opts_t k_cpu_rmsnorm_opts = {
    .allow_residual = true,
    .allow_weight = true,
    .require_weight = false,
    .allow_bias = false,
};

static marmot_error_t
cpu_rmsnorm_impl_with_weight_offset(const void *device_ctx, const marmot_rmsnorm_desc_t *desc, float weight_offset) {
    if (desc == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RMSNorm descriptor is null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const marmot_tensor_t *x = desc->x;
    const marmot_tensor_t *residual = desc->residual;
    const marmot_tensor_t *weight = desc->weight;
    marmot_tensor_t *out = desc->out;
    const float eps = desc->eps;
    marmot_norm_shape_t shape;
    marmot_error_t status = marmot_norm_validate(x, residual, weight, nullptr, out, &k_cpu_rmsnorm_opts, &shape);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const size_t norm_size = shape.norm_size;
    const size_t outer_size = shape.outer_size;

    if (unlikely(device_ctx == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "CPU context is null for rmsnorm");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const void *input_data = x->data;
    const void *residual_data = residual != nullptr ? residual->data : nullptr;

    if ((x->dtype == MARMOT_DTYPE_FLOAT16 || x->dtype == MARMOT_DTYPE_BFLOAT16) && weight != nullptr &&
        weight->dtype == MARMOT_DTYPE_FLOAT32) {
        return cpu_rmsnorm_mixed_vector_f32(
            x->dtype, input_data, residual_data, (const float *)weight->data, out->data, outer_size, norm_size, eps,
            weight_offset
        );
    }

    cpu_rmsnorm_fn rmsnorm_fn = cpu_norm_resolve_rmsnorm_fn(device_ctx, x->dtype);
    if (rmsnorm_fn == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "RMSNorm not implemented for dtype");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    cpu_rmsnorm_params_t params = {
        .x = input_data,
        .residual = residual_data,
        .weight = cpu_norm_tensor_data(weight),
        .out = out->data,
        .outer_size = outer_size,
        .norm_size = norm_size,
        .eps = eps,
        .weight_offset = weight_offset,
    };

    return rmsnorm_fn(device_ctx, &params);
}

marmot_error_t cpu_layernorm_impl(const void *device_ctx, const marmot_layernorm_desc_t *desc) {
    if (desc == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Layernorm descriptor is null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const marmot_tensor_t *x = desc->x;
    const marmot_tensor_t *residual = desc->residual;
    const marmot_tensor_t *weight = desc->weight;
    const marmot_tensor_t *bias = desc->bias;
    marmot_tensor_t *out = desc->out;
    const float eps = desc->eps;
    marmot_norm_shape_t shape;
    marmot_error_t status = marmot_norm_validate(x, residual, weight, bias, out, &k_cpu_layernorm_opts, &shape);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const size_t norm_size = shape.norm_size;
    const size_t outer_size = shape.outer_size;

    if (unlikely(device_ctx == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "CPU context is null for layernorm");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const void *input_data = x->data;
    const void *residual_data = residual != nullptr ? residual->data : nullptr;

    const bool use_mixed_vector_f32 = (x->dtype == MARMOT_DTYPE_FLOAT16 || x->dtype == MARMOT_DTYPE_BFLOAT16) &&
        ((weight != nullptr && weight->dtype == MARMOT_DTYPE_FLOAT32) ||
         (bias != nullptr && bias->dtype == MARMOT_DTYPE_FLOAT32));
    if (use_mixed_vector_f32) {
        return cpu_layernorm_mixed_vector_f32(
            x->dtype, input_data, residual_data, weight != nullptr ? (const float *)weight->data : nullptr,
            bias != nullptr ? (const float *)bias->data : nullptr, out->data, outer_size, norm_size, eps
        );
    }

    cpu_layernorm_fn layernorm_fn = cpu_norm_resolve_layernorm_fn(device_ctx, x->dtype);
    if (layernorm_fn == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "LayerNorm not implemented for dtype");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    cpu_layernorm_params_t params = {
        .x = input_data,
        .residual = residual_data,
        .weight = cpu_norm_tensor_data(weight),
        .bias = cpu_norm_tensor_data(bias),
        .out = out->data,
        .outer_size = outer_size,
        .norm_size = norm_size,
        .eps = eps,
    };

    return layernorm_fn(device_ctx, &params);
}

marmot_error_t cpu_rmsnorm_impl(const void *device_ctx, const marmot_rmsnorm_desc_t *desc) {
    return cpu_rmsnorm_impl_with_weight_offset(device_ctx, desc, 0.0f);
}

marmot_error_t cpu_rmsnorm_gemma_impl(const void *device_ctx, const marmot_rmsnorm_desc_t *desc) {
    return cpu_rmsnorm_impl_with_weight_offset(device_ctx, desc, 1.0f);
}

marmot_error_t cpu_rmsnorm(const void *device_ctx, const marmot_rmsnorm_desc_t *desc) {
    if (desc == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RMSNorm descriptor is null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return cpu_rmsnorm_impl(device_ctx, desc);
}
