#include <string.h>

#include "core/helpers/norm.h"
#include "metal_backend_internal.h"

#ifdef __APPLE__

static marmot_error_t metal_layernorm_dispatch(
    metal_context_t *ctx, const marmot_tensor_t *x, const marmot_tensor_t *weight, const marmot_tensor_t *bias,
    marmot_tensor_t *out, float eps
);
static marmot_error_t metal_fused_residual_layernorm_dispatch(
    metal_context_t *ctx, const marmot_tensor_t *x, const marmot_tensor_t *residual, const marmot_tensor_t *weight,
    const marmot_tensor_t *bias, marmot_tensor_t *out, float eps
);
static marmot_error_t metal_rmsnorm_dispatch(
    metal_context_t *ctx, const marmot_tensor_t *x, const marmot_tensor_t *weight, marmot_tensor_t *out, float eps
);
static marmot_error_t metal_fused_residual_rmsnorm_dispatch(
    metal_context_t *ctx, const marmot_tensor_t *x, const marmot_tensor_t *residual, const marmot_tensor_t *weight,
    marmot_tensor_t *out, float eps
);
static marmot_error_t metal_rmsnorm_gemma_dispatch(
    metal_context_t *ctx, const marmot_tensor_t *x, const marmot_tensor_t *weight, marmot_tensor_t *out, float eps
);
static marmot_error_t metal_fused_residual_rmsnorm_gemma_dispatch(
    metal_context_t *ctx, const marmot_tensor_t *x, const marmot_tensor_t *residual, const marmot_tensor_t *weight,
    marmot_tensor_t *out, float eps
);
static marmot_error_t
metal_softmax_gpu_dispatch(metal_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out, int32_t axis);

static const metal_norm_ops_t *
metal_norm_get_impl(const metal_context_t *ctx, marmot_dtype_t dtype, metal_norm_impl_kind_t kind);

static inline const metal_norm_ops_t *
metal_norm_select_ops(const metal_context_t *ctx, marmot_dtype_t dtype, metal_norm_impl_kind_t kind) {
    return metal_norm_get_impl(ctx, dtype, kind);
}

static metal_layernorm_fn
metal_norm_select_layernorm(const metal_context_t *ctx, marmot_dtype_t dtype, metal_norm_impl_kind_t kind) {
    const metal_norm_ops_t *ops = metal_norm_select_ops(ctx, dtype, kind);
    return ops != nullptr ? ops->layernorm : nullptr;
}

static metal_rmsnorm_fn
metal_norm_select_rmsnorm(const metal_context_t *ctx, marmot_dtype_t dtype, metal_norm_impl_kind_t kind) {
    const metal_norm_ops_t *ops = metal_norm_select_ops(ctx, dtype, kind);
    return ops != nullptr ? ops->rmsnorm : nullptr;
}

static metal_softmax_fn
metal_norm_select_softmax(const metal_context_t *ctx, marmot_dtype_t dtype, metal_norm_impl_kind_t kind) {
    const metal_norm_ops_t *ops = metal_norm_select_ops(ctx, dtype, kind);
    return ops != nullptr ? ops->softmax : nullptr;
}

static inline marmot_dtype_t
metal_norm_vector_dtype(const marmot_tensor_t *weight, const marmot_tensor_t *bias, marmot_dtype_t fallback) {
    if (weight != nullptr) {
        return weight->dtype;
    }
    if (bias != nullptr) {
        return bias->dtype;
    }
    return fallback;
}

static inline const char *metal_layernorm_kernel_name(
    const metal_norm_ops_t *ops, marmot_dtype_t dtype, marmot_dtype_t vector_dtype, bool fused
) {
    if (ops == nullptr) {
        return nullptr;
    }
    if (vector_dtype == MARMOT_DTYPE_FLOAT32) {
        if (dtype == MARMOT_DTYPE_FLOAT16) {
            return fused ? "fused_residual_layernorm_f16_wf32" : "layernorm_f16_wf32";
        }
        if (dtype == MARMOT_DTYPE_BFLOAT16) {
            return fused ? "fused_residual_layernorm_bf16_wf32" : "layernorm_bf16_wf32";
        }
    }
    return fused ? ops->fused_layernorm_kernel : ops->layernorm_kernel;
}

static inline const char *
metal_rmsnorm_kernel_name(const metal_norm_ops_t *ops, marmot_dtype_t dtype, marmot_dtype_t vector_dtype, bool fused) {
    if (ops == nullptr) {
        return nullptr;
    }
    if (vector_dtype == MARMOT_DTYPE_FLOAT32) {
        if (dtype == MARMOT_DTYPE_FLOAT16) {
            return fused ? "fused_residual_rmsnorm_f16_wf32" : "rmsnorm_f16_wf32";
        }
        if (dtype == MARMOT_DTYPE_BFLOAT16) {
            return fused ? "fused_residual_rmsnorm_bf16_wf32" : "rmsnorm_bf16_wf32";
        }
    }
    return fused ? ops->fused_rmsnorm_kernel : ops->rmsnorm_kernel;
}

static inline const char *
metal_rmsnorm_gemma_kernel_name(marmot_dtype_t dtype, marmot_dtype_t vector_dtype, bool fused) {
    if (vector_dtype == MARMOT_DTYPE_FLOAT32) {
        if (dtype == MARMOT_DTYPE_FLOAT16) {
            return fused ? "fused_residual_gemma_rmsnorm_f16_wf32" : "gemma_rmsnorm_f16_wf32";
        }
        if (dtype == MARMOT_DTYPE_BFLOAT16) {
            return fused ? "fused_residual_gemma_rmsnorm_bf16_wf32" : "gemma_rmsnorm_bf16_wf32";
        }
    }

    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return fused ? "fused_residual_gemma_rmsnorm_f32" : "gemma_rmsnorm_f32";
    case MARMOT_DTYPE_FLOAT16:
        return fused ? "fused_residual_gemma_rmsnorm_f16" : "gemma_rmsnorm_f16";
    case MARMOT_DTYPE_BFLOAT16:
        return fused ? "fused_residual_gemma_rmsnorm_bf16" : "gemma_rmsnorm_bf16";
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
        return fused ? nullptr : "gemma_rmsnorm_fp8_e4m3";
    case MARMOT_DTYPE_FLOAT8_E5M2:
        return fused ? nullptr : "gemma_rmsnorm_fp8_e5m2";
#endif
    default:
        return nullptr;
    }
}

static const marmot_norm_validation_opts_t k_metal_layernorm_opts = {
    .allow_residual = true,
    .allow_weight = true,
    .require_weight = false,
    .allow_bias = true,
};

static const marmot_norm_validation_opts_t k_metal_rmsnorm_opts = {
    .allow_residual = true,
    .allow_weight = true,
    .require_weight = false,
    .allow_bias = false,
};

static marmot_error_t metal_layernorm_gpu(
    const void *device_ctx, const marmot_tensor_t *x, const marmot_tensor_t *residual, const marmot_tensor_t *weight,
    const marmot_tensor_t *bias, marmot_tensor_t *out, float eps
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (residual != nullptr) {
        return metal_fused_residual_layernorm_dispatch(ctx, x, residual, weight, bias, out, eps);
    }
    return metal_layernorm_dispatch(ctx, x, weight, bias, out, eps);
}

static marmot_error_t metal_rmsnorm_gpu(
    const void *device_ctx, const marmot_tensor_t *x, const marmot_tensor_t *residual, const marmot_tensor_t *weight,
    marmot_tensor_t *out, float eps
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (residual != nullptr) {
        return metal_fused_residual_rmsnorm_dispatch(ctx, x, residual, weight, out, eps);
    }
    return metal_rmsnorm_dispatch(ctx, x, weight, out, eps);
}

static marmot_error_t metal_rmsnorm_gemma_gpu(
    const void *device_ctx, const marmot_tensor_t *x, const marmot_tensor_t *residual, const marmot_tensor_t *weight,
    marmot_tensor_t *out, float eps
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (residual != nullptr) {
        return metal_fused_residual_rmsnorm_gemma_dispatch(ctx, x, residual, weight, out, eps);
    }
    return metal_rmsnorm_gemma_dispatch(ctx, x, weight, out, eps);
}

static marmot_error_t metal_softmax_gpu(const void *device_ctx, const marmot_softmax_desc_t *desc) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return metal_softmax_gpu_dispatch(ctx, desc->x, desc->out, desc->axis);
}

static const metal_norm_traits_t metal_norm_gpu_f32_traits = {
    .dtype = MARMOT_DTYPE_FLOAT32,
    .impl_kind = METAL_NORM_IMPL_GPU,
    .ops = {
        .layernorm = metal_layernorm_gpu,
        .rmsnorm = metal_rmsnorm_gpu,
        .softmax = metal_softmax_gpu,
        .layernorm_kernel = "layernorm_f32",
        .fused_layernorm_kernel = "fused_residual_layernorm_f32",
        .rmsnorm_kernel = "rmsnorm_f32",
        .fused_rmsnorm_kernel = "fused_residual_rmsnorm_f32",
        .softmax_kernel = "softmax_f32",
        .softmax_strided_kernel = "softmax_strided_f32",
        .impl_name = "metal-gpu",
    },
};
static const metal_norm_traits_t metal_norm_gpu_f16_traits = {
    .dtype = MARMOT_DTYPE_FLOAT16,
    .impl_kind = METAL_NORM_IMPL_GPU,
    .ops = {
        .layernorm = metal_layernorm_gpu,
        .rmsnorm = metal_rmsnorm_gpu,
        .softmax = metal_softmax_gpu,
        .layernorm_kernel = "layernorm_f16",
        .fused_layernorm_kernel = "fused_residual_layernorm_f16",
        .rmsnorm_kernel = "rmsnorm_f16",
        .fused_rmsnorm_kernel = "fused_residual_rmsnorm_f16",
        .softmax_kernel = "softmax_f16",
        .softmax_strided_kernel = "softmax_strided_f16",
        .impl_name = "metal-gpu",
    },
};
static const metal_norm_traits_t metal_norm_gpu_bf16_traits = {
    .dtype = MARMOT_DTYPE_BFLOAT16,
    .impl_kind = METAL_NORM_IMPL_GPU,
    .ops = {
        .layernorm = metal_layernorm_gpu,
        .rmsnorm = metal_rmsnorm_gpu,
        .softmax = metal_softmax_gpu,
        .layernorm_kernel = "layernorm_bf16",
        .fused_layernorm_kernel = "fused_residual_layernorm_bf16",
        .rmsnorm_kernel = "rmsnorm_bf16",
        .fused_rmsnorm_kernel = "fused_residual_rmsnorm_bf16",
        .softmax_kernel = "softmax_bf16",
        .softmax_strided_kernel = "softmax_strided_bf16",
        .impl_name = "metal-gpu",
    },
};

#if MARMOT_ENABLE_FP8
static const metal_norm_traits_t metal_norm_gpu_fp8_e4m3_traits = {
    .dtype = MARMOT_DTYPE_FLOAT8_E4M3,
    .impl_kind = METAL_NORM_IMPL_GPU,
    .ops = {
        .layernorm = metal_layernorm_gpu,
        .rmsnorm = metal_rmsnorm_gpu,
        .softmax = metal_softmax_gpu,
        .layernorm_kernel = "layernorm_fp8_e4m3",
        .fused_layernorm_kernel = nullptr,
        .rmsnorm_kernel = "rmsnorm_fp8_e4m3",
        .fused_rmsnorm_kernel = nullptr,
        .softmax_kernel = "softmax_fp8_e4m3",
        .softmax_strided_kernel = "softmax_strided_fp8_e4m3",
        .impl_name = "metal-gpu",
    },
};

static const metal_norm_traits_t metal_norm_gpu_fp8_e5m2_traits = {
    .dtype = MARMOT_DTYPE_FLOAT8_E5M2,
    .impl_kind = METAL_NORM_IMPL_GPU,
    .ops = {
        .layernorm = metal_layernorm_gpu,
        .rmsnorm = metal_rmsnorm_gpu,
        .softmax = metal_softmax_gpu,
        .layernorm_kernel = "layernorm_fp8_e5m2",
        .fused_layernorm_kernel = nullptr,
        .rmsnorm_kernel = "rmsnorm_fp8_e5m2",
        .fused_rmsnorm_kernel = nullptr,
        .softmax_kernel = "softmax_fp8_e5m2",
        .softmax_strided_kernel = "softmax_strided_fp8_e5m2",
        .impl_name = "metal-gpu",
    },
};
#endif

static const metal_norm_traits_t *k_metal_norm_gpu_traits[] = {
    &metal_norm_gpu_f32_traits,      &metal_norm_gpu_f16_traits,      &metal_norm_gpu_bf16_traits,
#if MARMOT_ENABLE_FP8
    &metal_norm_gpu_fp8_e4m3_traits, &metal_norm_gpu_fp8_e5m2_traits,
#endif
};

void metal_norm_context_build(metal_context_t *ctx) {
    (void)ctx;
}

static const metal_norm_ops_t *
metal_norm_get_impl(const metal_context_t *ctx, marmot_dtype_t dtype, metal_norm_impl_kind_t kind) {
    if ((size_t)kind >= METAL_NORM_IMPL_COUNT) {
        return nullptr;
    }
    (void)ctx;
    for (size_t i = 0; i < sizeof(k_metal_norm_gpu_traits) / sizeof(k_metal_norm_gpu_traits[0]); ++i) {
        const metal_norm_traits_t *traits = k_metal_norm_gpu_traits[i];
        if (traits == nullptr || traits->dtype != dtype || traits->impl_kind != kind) {
            continue;
        }
        return &traits->ops;
    }
    return nullptr;
}

static marmot_error_t metal_layernorm_dispatch(
    metal_context_t *ctx, const marmot_tensor_t *x, const marmot_tensor_t *weight, const marmot_tensor_t *bias,
    marmot_tensor_t *out, float eps
) {
    marmot_norm_shape_t shape;
    marmot_error_t status = marmot_norm_validate(x, nullptr, weight, bias, out, &k_metal_layernorm_opts, &shape);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    const metal_norm_ops_t *ops = metal_norm_select_ops(ctx, x->dtype, METAL_NORM_IMPL_GPU);
    const marmot_dtype_t vector_dtype = metal_norm_vector_dtype(weight, bias, x->dtype);
    const char *kernel_name = metal_layernorm_kernel_name(ops, x->dtype, vector_dtype, false);
    if (kernel_name == nullptr) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    const size_t dim = shape.norm_size;
    const size_t num_rows = shape.outer_size;
    const size_t num_elements = dim * num_rows;
    (void)num_elements;

    size_t bytes_main = marmot_tensor_size_bytes(x);
    size_t bytes_weight = (weight != nullptr) ? marmot_tensor_size_bytes(weight) : 0;
    size_t bytes_bias = (bias != nullptr) ? marmot_tensor_size_bytes(bias) : 0;

    // Try existing residency first, then acquire_compute if needed
    id<MTLBuffer> bufferX = metal_residency_acquire_existing(ctx, x, x->dtype);
    if (bufferX == nil) {
        bufferX = metal_residency_acquire_compute(ctx, x, x->dtype, nullptr);
    }
    if (bufferX == nil) {
        bufferX = metal_buffer_acquire(ctx, x->data, bytes_main);
    }

    bool out_private = false;
    bool out_is_new = false;
    id<MTLBuffer> bufferOut = metal_residency_acquire_compute(ctx, out, out->dtype, &out_is_new);
    if (bufferOut == nil) {
        bufferOut = metal_buffer_acquire(ctx, out->data, bytes_main);
    } else {
        out_private = true;
    }
    (void)out_private;
    id<MTLBuffer> bufferWeight = nil;
    id<MTLBuffer> bufferBias = nil;
    // For weight/bias, try existing first (these rarely change), then acquire_compute
    if (weight != nullptr && weight->data != nullptr) {
        bufferWeight = metal_residency_acquire_existing(ctx, weight, weight->dtype);
        if (bufferWeight == nil) {
            bufferWeight = metal_residency_acquire_compute(ctx, weight, weight->dtype, nullptr);
        }
        if (bufferWeight == nil) {
            bufferWeight = metal_buffer_acquire(ctx, weight->data, bytes_weight);
        }
    }
    if (bias != nullptr && bias->data != nullptr) {
        bufferBias = metal_residency_acquire_existing(ctx, bias, bias->dtype);
        if (bufferBias == nil) {
            bufferBias = metal_residency_acquire_compute(ctx, bias, bias->dtype, nullptr);
        }
        if (bufferBias == nil) {
            bufferBias = metal_buffer_acquire(ctx, bias->data, bytes_bias);
        }
    }

    if (bufferX == nil || bufferOut == nil || (weight != nullptr && bufferWeight == nil) ||
        (bias != nullptr && bufferBias == nil)) {
        if (bufferX != nil)
            [bufferX release];
        if (bufferOut != nil)
            [bufferOut release];
        if (bufferWeight != nil)
            [bufferWeight release];
        if (bufferBias != nil)
            [bufferBias release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        [bufferX release];
        [bufferOut release];
        if (bufferWeight != nil)
            [bufferWeight release];
        if (bufferBias != nil)
            [bufferBias release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    uint32_t dim_u32 = (uint32_t)dim;
    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferX release];
        [bufferOut release];
        if (bufferWeight != nil)
            [bufferWeight release];
        if (bufferBias != nil)
            [bufferBias release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_profiling_set_label(ctx, "layernorm");
    metal_profiling_begin(ctx);

    [encoder setBuffer:bufferX offset:0 atIndex:0];
    [encoder setBuffer:bufferWeight offset:0 atIndex:1];
    [encoder setBuffer:bufferBias offset:0 atIndex:2];
    [encoder setBuffer:bufferOut offset:0 atIndex:3];
    [encoder setBytes:&dim_u32 length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&eps length:sizeof(float) atIndex:5];

    NSUInteger threadGroupSize = metal_threadgroup_size_1d(pipeline, 256);
    MTLSize grid = MTLSizeMake(num_rows, 1, 1);
    MTLSize threads = MTLSizeMake(threadGroupSize, 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];

    metal_profiling_end(ctx);

    metal_command_stream_flush(ctx, false);

    [pipeline release];
    [bufferX release];
    [bufferOut release];
    if (bufferWeight != nil)
        [bufferWeight release];
    if (bufferBias != nil)
        [bufferBias release];
    if (out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_fused_residual_layernorm_dispatch(
    metal_context_t *ctx, const marmot_tensor_t *x, const marmot_tensor_t *residual, const marmot_tensor_t *weight,
    const marmot_tensor_t *bias, marmot_tensor_t *out, float eps
) {
    marmot_norm_shape_t shape;
    marmot_error_t status = marmot_norm_validate(x, residual, weight, bias, out, &k_metal_layernorm_opts, &shape);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    const metal_norm_ops_t *ops = metal_norm_select_ops(ctx, x->dtype, METAL_NORM_IMPL_GPU);
    const marmot_dtype_t vector_dtype = metal_norm_vector_dtype(weight, bias, x->dtype);
    const char *kernel_name = metal_layernorm_kernel_name(ops, x->dtype, vector_dtype, true);
    if (kernel_name == nullptr) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    const size_t dim = shape.norm_size;
    const size_t num_rows = shape.outer_size;
    const size_t num_elements = dim * num_rows;
    (void)num_elements;

    size_t bytes_main = marmot_tensor_size_bytes(x);
    size_t bytes_residual = marmot_tensor_size_bytes(residual);
    size_t bytes_weight = (weight != nullptr) ? marmot_tensor_size_bytes(weight) : 0;
    size_t bytes_bias = (bias != nullptr) ? marmot_tensor_size_bytes(bias) : 0;

    id<MTLBuffer> bufferX = metal_residency_acquire_existing(ctx, x, x->dtype);
    if (bufferX == nil) {
        bufferX = metal_residency_acquire_compute(ctx, x, x->dtype, nullptr);
    }
    if (bufferX == nil) {
        bufferX = metal_buffer_acquire(ctx, x->data, bytes_main);
    }

    id<MTLBuffer> bufferResidual = metal_residency_acquire_existing(ctx, residual, residual->dtype);
    if (bufferResidual == nil) {
        bufferResidual = metal_residency_acquire_compute(ctx, residual, residual->dtype, nullptr);
    }
    if (bufferResidual == nil) {
        bufferResidual = metal_buffer_acquire(ctx, residual->data, bytes_residual);
    }

    bool out_private = false;
    bool out_is_new = false;
    id<MTLBuffer> bufferOut = metal_residency_acquire_compute(ctx, out, out->dtype, &out_is_new);
    if (bufferOut == nil) {
        bufferOut = metal_buffer_acquire(ctx, out->data, bytes_main);
    } else {
        out_private = true;
    }

    id<MTLBuffer> bufferWeight = nil;
    if (weight != nullptr && weight->data != nullptr) {
        bufferWeight = metal_residency_acquire_existing(ctx, weight, weight->dtype);
        if (bufferWeight == nil) {
            bufferWeight = metal_residency_acquire_compute(ctx, weight, weight->dtype, nullptr);
        }
        if (bufferWeight == nil) {
            bufferWeight = metal_buffer_acquire(ctx, weight->data, bytes_weight);
        }
    }

    id<MTLBuffer> bufferBias = nil;
    if (bias != nullptr && bias->data != nullptr) {
        bufferBias = metal_residency_acquire_existing(ctx, bias, bias->dtype);
        if (bufferBias == nil) {
            bufferBias = metal_residency_acquire_compute(ctx, bias, bias->dtype, nullptr);
        }
        if (bufferBias == nil) {
            bufferBias = metal_buffer_acquire(ctx, bias->data, bytes_bias);
        }
    }

    if (bufferX == nil || bufferResidual == nil || bufferOut == nil || (weight != nullptr && bufferWeight == nil) ||
        (bias != nullptr && bufferBias == nil)) {
        if (bufferX != nil)
            [bufferX release];
        if (bufferResidual != nil)
            [bufferResidual release];
        if (bufferOut != nil)
            [bufferOut release];
        if (bufferWeight != nil)
            [bufferWeight release];
        if (bufferBias != nil)
            [bufferBias release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        [bufferX release];
        [bufferResidual release];
        [bufferOut release];
        if (bufferWeight != nil)
            [bufferWeight release];
        if (bufferBias != nil)
            [bufferBias release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    uint32_t dim_u32 = (uint32_t)dim;
    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferX release];
        [bufferResidual release];
        [bufferOut release];
        if (bufferWeight != nil)
            [bufferWeight release];
        if (bufferBias != nil)
            [bufferBias release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_profiling_set_label(ctx, "layernorm");
    metal_profiling_begin(ctx);

    [encoder setBuffer:bufferX offset:0 atIndex:0];
    [encoder setBuffer:bufferResidual offset:0 atIndex:1];
    [encoder setBuffer:bufferWeight offset:0 atIndex:2];
    [encoder setBuffer:bufferBias offset:0 atIndex:3];
    [encoder setBuffer:bufferOut offset:0 atIndex:4];
    [encoder setBytes:&dim_u32 length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&eps length:sizeof(float) atIndex:6];

    NSUInteger threadGroupSize = metal_threadgroup_size_1d(pipeline, 256);
    MTLSize grid = MTLSizeMake(num_rows, 1, 1);
    MTLSize threads = MTLSizeMake(threadGroupSize, 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];

    metal_profiling_end(ctx);

    metal_command_stream_flush(ctx, false);

    [pipeline release];
    [bufferX release];
    [bufferResidual release];
    [bufferOut release];
    if (bufferWeight != nil)
        [bufferWeight release];
    if (bufferBias != nil)
        [bufferBias release];
    if (out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_fused_residual_rmsnorm_dispatch(
    metal_context_t *ctx, const marmot_tensor_t *x, const marmot_tensor_t *residual, const marmot_tensor_t *weight,
    marmot_tensor_t *out, float eps
) {
    marmot_norm_shape_t shape;
    marmot_error_t status = marmot_norm_validate(x, residual, weight, nullptr, out, &k_metal_rmsnorm_opts, &shape);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    const metal_norm_ops_t *ops = metal_norm_select_ops(ctx, x->dtype, METAL_NORM_IMPL_GPU);
    const marmot_dtype_t vector_dtype = metal_norm_vector_dtype(weight, nullptr, x->dtype);
    const char *kernel_name = metal_rmsnorm_kernel_name(ops, x->dtype, vector_dtype, true);
    if (kernel_name == nullptr) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    const size_t dim = shape.norm_size;
    const size_t num_rows = shape.outer_size;

    size_t bytes_main = marmot_tensor_size_bytes(x);
    size_t bytes_residual = marmot_tensor_size_bytes(residual);
    size_t bytes_weight = (weight != nullptr) ? marmot_tensor_size_bytes(weight) : 0;

    id<MTLBuffer> bufferX = metal_residency_acquire_existing(ctx, x, x->dtype);
    if (bufferX == nil) {
        bufferX = metal_residency_acquire_compute(ctx, x, x->dtype, nullptr);
    }
    if (bufferX == nil) {
        bufferX = metal_buffer_acquire(ctx, x->data, bytes_main);
    }

    id<MTLBuffer> bufferResidual = metal_residency_acquire_existing(ctx, residual, residual->dtype);
    if (bufferResidual == nil) {
        bufferResidual = metal_residency_acquire_compute(ctx, residual, residual->dtype, nullptr);
    }
    if (bufferResidual == nil) {
        bufferResidual = metal_buffer_acquire(ctx, residual->data, bytes_residual);
    }

    bool out_private = false;
    bool out_is_new = false;
    id<MTLBuffer> bufferOut = metal_residency_acquire_compute(ctx, out, out->dtype, &out_is_new);
    if (bufferOut == nil) {
        bufferOut = metal_buffer_acquire(ctx, out->data, bytes_main);
    } else {
        out_private = true;
    }

    id<MTLBuffer> bufferWeight = nil;
    if (weight != nullptr && weight->data != nullptr) {
        bufferWeight = metal_residency_acquire_existing(ctx, weight, weight->dtype);
        if (bufferWeight == nil) {
            bufferWeight = metal_residency_acquire_compute(ctx, weight, weight->dtype, nullptr);
        }
        if (bufferWeight == nil) {
            bufferWeight = metal_buffer_acquire(ctx, weight->data, bytes_weight);
        }
    }

    if (bufferX == nil || bufferResidual == nil || bufferOut == nil || (weight != nullptr && bufferWeight == nil)) {
        if (bufferX != nil)
            [bufferX release];
        if (bufferResidual != nil)
            [bufferResidual release];
        if (bufferOut != nil)
            [bufferOut release];
        if (bufferWeight != nil)
            [bufferWeight release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        [bufferX release];
        [bufferResidual release];
        [bufferOut release];
        if (bufferWeight != nil)
            [bufferWeight release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    uint32_t dim_u32 = (uint32_t)dim;
    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferX release];
        [bufferResidual release];
        [bufferOut release];
        if (bufferWeight != nil)
            [bufferWeight release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_profiling_set_label(ctx, "rmsnorm");
    metal_profiling_begin(ctx);

    [encoder setBuffer:bufferX offset:0 atIndex:0];
    [encoder setBuffer:bufferResidual offset:0 atIndex:1];
    [encoder setBuffer:bufferWeight offset:0 atIndex:2];
    [encoder setBuffer:bufferOut offset:0 atIndex:3];
    [encoder setBytes:&dim_u32 length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&eps length:sizeof(float) atIndex:5];

    NSUInteger threadGroupSize = metal_threadgroup_size_1d(pipeline, 256);
    MTLSize grid = MTLSizeMake(num_rows, 1, 1);
    MTLSize threads = MTLSizeMake(threadGroupSize, 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];

    metal_profiling_end(ctx);

    metal_command_stream_flush(ctx, false);

    [pipeline release];
    [bufferX release];
    [bufferResidual release];
    [bufferOut release];
    if (bufferWeight != nil)
        [bufferWeight release];
    if (out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_fused_residual_rmsnorm_gemma_dispatch(
    metal_context_t *ctx, const marmot_tensor_t *x, const marmot_tensor_t *residual, const marmot_tensor_t *weight,
    marmot_tensor_t *out, float eps
) {
    marmot_norm_shape_t shape;
    marmot_error_t status = marmot_norm_validate(x, residual, weight, nullptr, out, &k_metal_rmsnorm_opts, &shape);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    const marmot_dtype_t vector_dtype = metal_norm_vector_dtype(weight, nullptr, x->dtype);
    const char *kernel_name = metal_rmsnorm_gemma_kernel_name(x->dtype, vector_dtype, true);
    if (kernel_name == nullptr) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    const size_t dim = shape.norm_size;
    const size_t num_rows = shape.outer_size;

    size_t bytes_main = marmot_tensor_size_bytes(x);
    size_t bytes_residual = marmot_tensor_size_bytes(residual);
    size_t bytes_weight = (weight != nullptr) ? marmot_tensor_size_bytes(weight) : 0;

    id<MTLBuffer> bufferX = metal_residency_acquire_existing(ctx, x, x->dtype);
    if (bufferX == nil) {
        bufferX = metal_residency_acquire_compute(ctx, x, x->dtype, nullptr);
    }
    if (bufferX == nil) {
        bufferX = metal_buffer_acquire(ctx, x->data, bytes_main);
    }

    id<MTLBuffer> bufferResidual = metal_residency_acquire_existing(ctx, residual, residual->dtype);
    if (bufferResidual == nil) {
        bufferResidual = metal_residency_acquire_compute(ctx, residual, residual->dtype, nullptr);
    }
    if (bufferResidual == nil) {
        bufferResidual = metal_buffer_acquire(ctx, residual->data, bytes_residual);
    }

    bool out_private = false;
    bool out_is_new = false;
    id<MTLBuffer> bufferOut = metal_residency_acquire_compute(ctx, out, out->dtype, &out_is_new);
    if (bufferOut == nil) {
        bufferOut = metal_buffer_acquire(ctx, out->data, bytes_main);
    } else {
        out_private = true;
    }

    id<MTLBuffer> bufferWeight = nil;
    if (weight != nullptr && weight->data != nullptr) {
        bufferWeight = metal_residency_acquire_existing(ctx, weight, weight->dtype);
        if (bufferWeight == nil) {
            bufferWeight = metal_residency_acquire_compute(ctx, weight, weight->dtype, nullptr);
        }
        if (bufferWeight == nil) {
            bufferWeight = metal_buffer_acquire(ctx, weight->data, bytes_weight);
        }
    }

    if (bufferX == nil || bufferResidual == nil || bufferOut == nil || (weight != nullptr && bufferWeight == nil)) {
        if (bufferX != nil)
            [bufferX release];
        if (bufferResidual != nil)
            [bufferResidual release];
        if (bufferOut != nil)
            [bufferOut release];
        if (bufferWeight != nil)
            [bufferWeight release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        [bufferX release];
        [bufferResidual release];
        [bufferOut release];
        if (bufferWeight != nil)
            [bufferWeight release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    uint32_t dim_u32 = (uint32_t)dim;
    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferX release];
        [bufferResidual release];
        [bufferOut release];
        if (bufferWeight != nil)
            [bufferWeight release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_profiling_set_label(ctx, "rmsnorm_gemma");
    metal_profiling_begin(ctx);

    [encoder setBuffer:bufferX offset:0 atIndex:0];
    [encoder setBuffer:bufferResidual offset:0 atIndex:1];
    [encoder setBuffer:bufferWeight offset:0 atIndex:2];
    [encoder setBuffer:bufferOut offset:0 atIndex:3];
    [encoder setBytes:&dim_u32 length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&eps length:sizeof(float) atIndex:5];

    NSUInteger threadGroupSize = metal_threadgroup_size_1d(pipeline, 256);
    MTLSize grid = MTLSizeMake(num_rows, 1, 1);
    MTLSize threads = MTLSizeMake(threadGroupSize, 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];

    metal_profiling_end(ctx);

    metal_command_stream_flush(ctx, false);

    [pipeline release];
    [bufferX release];
    [bufferResidual release];
    [bufferOut release];
    if (bufferWeight != nil)
        [bufferWeight release];
    if (out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_rmsnorm_dispatch(
    metal_context_t *ctx, const marmot_tensor_t *x, const marmot_tensor_t *weight, marmot_tensor_t *out, float eps
) {
    marmot_norm_shape_t shape;
    marmot_error_t status = marmot_norm_validate(x, nullptr, weight, nullptr, out, &k_metal_rmsnorm_opts, &shape);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    const metal_norm_ops_t *ops = metal_norm_select_ops(ctx, x->dtype, METAL_NORM_IMPL_GPU);
    const marmot_dtype_t vector_dtype = metal_norm_vector_dtype(weight, nullptr, x->dtype);
    const char *kernel_name = metal_rmsnorm_kernel_name(ops, x->dtype, vector_dtype, false);
    if (kernel_name == nullptr) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    const size_t dim = shape.norm_size;
    const size_t num_rows = shape.outer_size;
    const size_t num_elements = dim * num_rows;
    (void)num_elements;

    size_t bytes_main = marmot_tensor_size_bytes(x);
    size_t bytes_weight = (weight != nullptr) ? marmot_tensor_size_bytes(weight) : 0;

    id<MTLBuffer> bufferX = metal_residency_acquire_existing(ctx, x, x->dtype);
    if (bufferX == nil) {
        bufferX = metal_residency_acquire_compute(ctx, x, x->dtype, nullptr);
    }
    if (bufferX == nil) {
        bufferX = metal_buffer_acquire(ctx, x->data, bytes_main);
    }

    bool out_private = false;
    bool out_is_new = false;
    id<MTLBuffer> bufferOut = metal_residency_acquire_compute(ctx, out, out->dtype, &out_is_new);
    if (bufferOut == nil) {
        bufferOut = metal_buffer_acquire(ctx, out->data, bytes_main);
    } else {
        out_private = true;
    }
    id<MTLBuffer> bufferWeight = nil;
    if (weight != nullptr && weight->data != nullptr) {
        bufferWeight = metal_residency_acquire_existing(ctx, weight, weight->dtype);
        if (bufferWeight == nil) {
            bufferWeight = metal_residency_acquire_compute(ctx, weight, weight->dtype, nullptr);
        }
        if (bufferWeight == nil) {
            bufferWeight = metal_buffer_acquire(ctx, weight->data, bytes_weight);
        }
    }

    if (bufferX == nil || bufferOut == nil || (weight != nullptr && bufferWeight == nil)) {
        if (bufferX != nil)
            [bufferX release];
        if (bufferOut != nil)
            [bufferOut release];
        if (bufferWeight != nil)
            [bufferWeight release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        [bufferX release];
        [bufferOut release];
        if (bufferWeight != nil)
            [bufferWeight release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    uint32_t dim_u32 = (uint32_t)dim;
    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferX release];
        [bufferOut release];
        if (bufferWeight != nil)
            [bufferWeight release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }
    metal_profiling_set_label(ctx, "rmsnorm");
    metal_profiling_begin(ctx);
    [encoder setBuffer:bufferX offset:0 atIndex:0];
    [encoder setBuffer:bufferWeight offset:0 atIndex:1];
    [encoder setBuffer:bufferOut offset:0 atIndex:2];
    [encoder setBytes:&dim_u32 length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&eps length:sizeof(float) atIndex:4];

    NSUInteger threadGroupSize = metal_threadgroup_size_1d(pipeline, 256);
    MTLSize grid = MTLSizeMake(num_rows, 1, 1);
    MTLSize threads = MTLSizeMake(threadGroupSize, 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    metal_profiling_end(ctx);

    metal_command_stream_flush(ctx, false);

    [pipeline release];
    [bufferX release];
    [bufferOut release];
    if (bufferWeight != nil)
        [bufferWeight release];
    if (out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_rmsnorm_gemma_dispatch(
    metal_context_t *ctx, const marmot_tensor_t *x, const marmot_tensor_t *weight, marmot_tensor_t *out, float eps
) {
    marmot_norm_shape_t shape;
    marmot_error_t status = marmot_norm_validate(x, nullptr, weight, nullptr, out, &k_metal_rmsnorm_opts, &shape);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    const marmot_dtype_t vector_dtype = metal_norm_vector_dtype(weight, nullptr, x->dtype);
    const char *kernel_name = metal_rmsnorm_gemma_kernel_name(x->dtype, vector_dtype, false);
    if (kernel_name == nullptr) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    const size_t dim = shape.norm_size;
    const size_t num_rows = shape.outer_size;
    const size_t num_elements = dim * num_rows;
    (void)num_elements;

    size_t bytes_main = marmot_tensor_size_bytes(x);
    size_t bytes_weight = (weight != nullptr) ? marmot_tensor_size_bytes(weight) : 0;

    id<MTLBuffer> bufferX = metal_residency_acquire_existing(ctx, x, x->dtype);
    if (bufferX == nil) {
        bufferX = metal_residency_acquire_compute(ctx, x, x->dtype, nullptr);
    }
    if (bufferX == nil) {
        bufferX = metal_buffer_acquire(ctx, x->data, bytes_main);
    }

    bool out_private = false;
    bool out_is_new = false;
    id<MTLBuffer> bufferOut = metal_residency_acquire_compute(ctx, out, out->dtype, &out_is_new);
    if (bufferOut == nil) {
        bufferOut = metal_buffer_acquire(ctx, out->data, bytes_main);
    } else {
        out_private = true;
    }
    id<MTLBuffer> bufferWeight = nil;
    if (weight != nullptr && weight->data != nullptr) {
        bufferWeight = metal_residency_acquire_existing(ctx, weight, weight->dtype);
        if (bufferWeight == nil) {
            bufferWeight = metal_residency_acquire_compute(ctx, weight, weight->dtype, nullptr);
        }
        if (bufferWeight == nil) {
            bufferWeight = metal_buffer_acquire(ctx, weight->data, bytes_weight);
        }
    }

    if (bufferX == nil || bufferOut == nil || (weight != nullptr && bufferWeight == nil)) {
        if (bufferX != nil)
            [bufferX release];
        if (bufferOut != nil)
            [bufferOut release];
        if (bufferWeight != nil)
            [bufferWeight release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        [bufferX release];
        [bufferOut release];
        if (bufferWeight != nil)
            [bufferWeight release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    uint32_t dim_u32 = (uint32_t)dim;
    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferX release];
        [bufferOut release];
        if (bufferWeight != nil)
            [bufferWeight release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }
    metal_profiling_set_label(ctx, "rmsnorm_gemma");
    metal_profiling_begin(ctx);
    [encoder setBuffer:bufferX offset:0 atIndex:0];
    [encoder setBuffer:bufferWeight offset:0 atIndex:1];
    [encoder setBuffer:bufferOut offset:0 atIndex:2];
    [encoder setBytes:&dim_u32 length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&eps length:sizeof(float) atIndex:4];

    NSUInteger threadGroupSize = metal_threadgroup_size_1d(pipeline, 256);
    MTLSize grid = MTLSizeMake(num_rows, 1, 1);
    MTLSize threads = MTLSizeMake(threadGroupSize, 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    metal_profiling_end(ctx);

    metal_command_stream_flush(ctx, false);

    [pipeline release];
    [bufferX release];
    [bufferOut release];
    if (bufferWeight != nil)
        [bufferWeight release];
    if (out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t
metal_softmax_gpu_dispatch(metal_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out, int32_t axis) {
    marmot_softmax_shape_t shape;
    marmot_error_t status = marmot_softmax_prepare(x, out, axis, &shape);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    const metal_norm_ops_t *ops = metal_norm_select_ops(ctx, x->dtype, METAL_NORM_IMPL_GPU);
    const char *kernel_name = (ops != nullptr) ? ops->softmax_kernel : nullptr;
    const char *strided_kernel = (ops != nullptr) ? ops->softmax_strided_kernel : nullptr;

    bool axis_is_last = (shape.axis == (int)(x->shape.ndim - 1));
    size_t axis_size = shape.axis_size;
    size_t inner_stride = shape.inner_stride;
    size_t num_rows = shape.row_count;

    size_t bytes = marmot_tensor_size_bytes(x);
    id<MTLBuffer> bufferX = metal_residency_acquire_existing(ctx, x, x->dtype);
    if (bufferX == nil) {
        bufferX = metal_residency_acquire_compute(ctx, x, x->dtype, nullptr);
    }
    if (bufferX == nil) {
        bufferX = metal_buffer_acquire(ctx, x->data, bytes);
    }

    bool out_private = false;
    bool out_is_new = false;
    id<MTLBuffer> bufferOut = metal_residency_acquire_compute(ctx, out, out->dtype, &out_is_new);
    if (bufferOut == nil) {
        bufferOut = metal_buffer_acquire(ctx, out->data, bytes);
    } else {
        out_private = true;
    }

    if (bufferX == nil || bufferOut == nil) {
        if (bufferX != nil)
            [bufferX release];
        if (bufferOut != nil)
            [bufferOut release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    const char *selected_kernel = axis_is_last ? kernel_name : strided_kernel;
    if (selected_kernel == nullptr) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, selected_kernel);
    if (pipeline == nil) {
        [bufferX release];
        [bufferOut release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    uint32_t dim_u32 = (uint32_t)axis_size;
    uint32_t rows_u32 = (uint32_t)num_rows;
    uint32_t inner_stride_u32 = (uint32_t)inner_stride;
    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferX release];
        [bufferOut release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }
    metal_profiling_set_label(ctx, "softmax");
    metal_profiling_begin(ctx);
    [encoder setBuffer:bufferX offset:0 atIndex:0];
    [encoder setBuffer:bufferOut offset:0 atIndex:1];

    if (axis_is_last) {
        [encoder setBytes:&dim_u32 length:sizeof(uint32_t) atIndex:2];
        NSUInteger threadGroupSize = metal_threadgroup_size_1d(pipeline, 256);
        MTLSize grid = MTLSizeMake(num_rows, 1, 1);
        MTLSize threads = MTLSizeMake(threadGroupSize, 1, 1);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    } else {
        [encoder setBytes:&rows_u32 length:sizeof(uint32_t) atIndex:2];
        [encoder setBytes:&dim_u32 length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&inner_stride_u32 length:sizeof(uint32_t) atIndex:4];
        MTLSize grid = MTLSizeMake(num_rows, 1, 1);
        NSUInteger threadGroupSize = metal_threadgroup_size_1d(pipeline, 512);
        if (threadGroupSize > num_rows) {
            threadGroupSize = num_rows;
        }
        if (threadGroupSize == 0) {
            threadGroupSize = 1;
        }
        MTLSize threads = MTLSizeMake(threadGroupSize, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    }

    metal_profiling_end(ctx);

    metal_command_stream_flush(ctx, false);

    [pipeline release];
    [bufferX release];
    [bufferOut release];
    if (out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t metal_layernorm_impl(const void *device_ctx, const marmot_layernorm_desc_t *desc) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr || desc == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_tensor_t *x = desc->x;
    const marmot_tensor_t *residual = desc->residual;
    const marmot_tensor_t *weight = desc->weight;
    const marmot_tensor_t *bias = desc->bias;
    marmot_tensor_t *out = desc->out;
    const float eps = desc->eps;

    const size_t problem_bytes = x != nullptr ? marmot_tensor_size_bytes(x) : 0;
    marmot_dtype_t dtype = x != nullptr ? x->dtype : MARMOT_DTYPE_FLOAT32;
    metal_layernorm_fn gpu_fn = metal_norm_select_layernorm(ctx, dtype, METAL_NORM_IMPL_GPU);
    if (gpu_fn == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal layernorm not implemented for dtype");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_NORMALIZATION, "layernorm", problem_bytes, true, "gpu");
    marmot_error_t err = gpu_fn(ctx, x, residual, weight, bias, out, eps);
    return err;
}

marmot_error_t metal_rmsnorm_impl(const void *device_ctx, const marmot_rmsnorm_desc_t *desc) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr || desc == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_tensor_t *x = desc->x;
    const marmot_tensor_t *residual = desc->residual;
    const marmot_tensor_t *weight = desc->weight;
    marmot_tensor_t *out = desc->out;
    const float eps = desc->eps;

    const size_t problem_bytes = x != nullptr ? marmot_tensor_size_bytes(x) : 0;
    marmot_dtype_t dtype = x != nullptr ? x->dtype : MARMOT_DTYPE_FLOAT32;
    metal_rmsnorm_fn gpu_fn = metal_norm_select_rmsnorm(ctx, dtype, METAL_NORM_IMPL_GPU);
    if (gpu_fn == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal rmsnorm not implemented for dtype");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_NORMALIZATION, "rmsnorm", problem_bytes, true, "gpu");
    marmot_error_t err = gpu_fn(ctx, x, residual, weight, out, eps);
    return err;
}

marmot_error_t metal_rmsnorm_gemma_impl(const void *device_ctx, const marmot_rmsnorm_desc_t *desc) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr || desc == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_tensor_t *x = desc->x;
    const marmot_tensor_t *residual = desc->residual;
    const marmot_tensor_t *weight = desc->weight;
    marmot_tensor_t *out = desc->out;
    const float eps = desc->eps;

    const size_t problem_bytes = x != nullptr ? marmot_tensor_size_bytes(x) : 0;
    marmot_dtype_t dtype = x != nullptr ? x->dtype : MARMOT_DTYPE_FLOAT32;
    metal_rmsnorm_fn gpu_fn = metal_norm_select_rmsnorm(ctx, dtype, METAL_NORM_IMPL_GPU);
    if (gpu_fn == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal rmsnorm not implemented for dtype");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_NORMALIZATION, "rmsnorm_gemma", problem_bytes, true, "gpu");
    return metal_rmsnorm_gemma_gpu(ctx, x, residual, weight, out, eps);
}

marmot_error_t metal_softmax_impl(const void *device_ctx, const marmot_softmax_desc_t *desc) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr || desc == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_tensor_t *x = desc->x;
    const size_t problem_bytes = x != nullptr ? marmot_tensor_size_bytes(x) : 0;
    marmot_dtype_t dtype = x != nullptr ? x->dtype : MARMOT_DTYPE_FLOAT32;
    metal_softmax_fn gpu_fn = metal_norm_select_softmax(ctx, dtype, METAL_NORM_IMPL_GPU);
    if (gpu_fn == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal softmax not implemented for dtype");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_NORMALIZATION, "softmax", problem_bytes, true, "gpu");
    marmot_error_t err = gpu_fn(ctx, desc);
    return err;
}

marmot_error_t metal_rmsnorm(const void *device_ctx, const marmot_rmsnorm_desc_t *desc) {
    return metal_rmsnorm_impl(device_ctx, desc);
}

#endif // __APPLE__
