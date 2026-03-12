#include "marmot/ops/matmul.h"

#include "marmot/quant_block.h"

#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#import <os/log.h>
#include <string.h>
#include <TargetConditionals.h>

#include "core/helpers/matmul.h"
#include "core/helpers/matmul_packers.h"
#include "core/helpers/quant.h"
#include "core/helpers/rope.h"
#include "internal/metal_matmul_internal.h"
#include "internal/metal_matmul_quant_dispatch.h"
#include "metal_backend_internal.h"

static void metal_matmul_default_activation_support(bool supported[MARMOT_DEVICE_UNARY_COUNT]) {
    supported[MARMOT_DEVICE_UNARY_RELU] = true;
    supported[MARMOT_DEVICE_UNARY_GELU] = true;
    supported[MARMOT_DEVICE_UNARY_GELU_TANH] = true;
    supported[MARMOT_DEVICE_UNARY_SILU] = true;
    supported[MARMOT_DEVICE_UNARY_SIGMOID] = true;
    supported[MARMOT_DEVICE_UNARY_TANH] = true;
    supported[MARMOT_DEVICE_UNARY_MISH] = true;
    supported[MARMOT_DEVICE_UNARY_ELU] = true;
    supported[MARMOT_DEVICE_UNARY_SELU] = true;
    supported[MARMOT_DEVICE_UNARY_LEAKY_RELU] = true;
    supported[MARMOT_DEVICE_UNARY_PRELU] = true;
    supported[MARMOT_DEVICE_UNARY_ABS] = true;
    supported[MARMOT_DEVICE_UNARY_NEG] = true;
    supported[MARMOT_DEVICE_UNARY_SIGN] = true;
    supported[MARMOT_DEVICE_UNARY_SQRT] = true;
    supported[MARMOT_DEVICE_UNARY_EXP] = true;
    supported[MARMOT_DEVICE_UNARY_LOG] = true;
    supported[MARMOT_DEVICE_UNARY_BITWISE_NOT] = true;
    supported[MARMOT_DEVICE_UNARY_IDENTITY] = true;
}

static metal_matmul_epilogue_traits_t
metal_matmul_make_epilogue_traits(marmot_dtype_t dtype, const char *kernel_name, const char *rope_kernel_name) {
    metal_matmul_epilogue_traits_t traits = {};
    traits.dtype = dtype;
    traits.supports_bias = true;
    metal_matmul_default_activation_support(traits.activation_supported);
    traits.kernel_name = kernel_name;
    traits.rope_kernel_name = rope_kernel_name;
    return traits;
}

static const metal_matmul_epilogue_traits_t k_metal_matmul_f32_epilogue_traits = metal_matmul_make_epilogue_traits(
    MARMOT_DTYPE_FLOAT32, "fused_bias_activation_f32", "fused_bias_rope_activation_f32"
);

static const metal_matmul_epilogue_traits_t k_metal_matmul_f16_epilogue_traits = metal_matmul_make_epilogue_traits(
    MARMOT_DTYPE_FLOAT16, "fused_bias_activation_f16", "fused_bias_rope_activation_f16"
);

static const metal_matmul_epilogue_traits_t k_metal_matmul_bf16_epilogue_traits = metal_matmul_make_epilogue_traits(
    MARMOT_DTYPE_BFLOAT16, "fused_bias_activation_bf16", "fused_bias_rope_activation_bf16"
);

id<MTLBuffer>
metal_matmul_create_positions_buffer(metal_context_t *ctx, const marmot_tensor_t *positions, size_t rows) {
    if (positions == nullptr || ctx == nullptr || rows == 0) {
        return nil;
    }
    if (positions->shape.ndim != 1 || positions->shape.shape[0] != rows || positions->shape.strides[0] != 1) {
        return nil;
    }
    size_t bytes = rows * sizeof(float);
    if (positions->dtype == MARMOT_DTYPE_FLOAT32) {
        if (positions->data == nullptr) {
            return nil;
        }
        return metal_buffer_acquire(ctx, positions->data, bytes);
    }
    float *converted = (float *)malloc(bytes);
    if (converted == nullptr) {
        return nil;
    }
    switch (positions->dtype) {
    case MARMOT_DTYPE_INT32: {
        const int32_t *src = (const int32_t *)positions->data;
        for (size_t i = 0; i < rows; ++i) {
            converted[i] = (float)src[i];
        }
        break;
    }
    case MARMOT_DTYPE_INT64: {
        const int64_t *src = (const int64_t *)positions->data;
        for (size_t i = 0; i < rows; ++i) {
            converted[i] = (float)src[i];
        }
        break;
    }
    default:
        free(converted);
        return nil;
    }

    id<MTLBuffer> buffer = [ctx->device newBufferWithBytesNoCopy:converted
                                                          length:bytes
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:^(void *ptr, NSUInteger length) {
                                                       (void)length;
                                                       free(ptr);
                                                     }];
    if (buffer == nil) {
        free(converted);
    }
    return buffer;
}

id<MTLBuffer> metal_matmul_prepare_freq_buffer(
    metal_context_t *ctx, size_t dim, const marmot_rope_params_t *params, float *attn_scale_out
) {
    if (attn_scale_out != nullptr) {
        *attn_scale_out = 1.0f;
    }
    if (ctx == nullptr || params == nullptr || dim < 2) {
        return nil;
    }
    size_t freq_count = dim / 2;
    if (freq_count == 0) {
        return nil;
    }
    const size_t freq_bytes = freq_count * sizeof(float);
    if (ctx->rope_cache.buffer != nil && ctx->rope_cache.dim == dim && ctx->rope_cache.theta == params->theta &&
        ctx->rope_cache.freq_scale == params->freq_scale && ctx->rope_cache.ext_factor == params->ext_factor &&
        ctx->rope_cache.attn_factor == params->attn_factor && ctx->rope_cache.beta_fast == params->beta_fast &&
        ctx->rope_cache.beta_slow == params->beta_slow && ctx->rope_cache.orig_ctx_len == params->orig_ctx_len &&
        ctx->rope_cache.scaling_type == params->scaling_type && ctx->rope_cache.capacity_bytes >= freq_bytes) {
        if (attn_scale_out != nullptr) {
            *attn_scale_out = ctx->rope_cache.attn_scale;
        }
        return [ctx->rope_cache.buffer retain];
    }

    marmot_rope_freq_span_t span = {};
    marmot_error_t freq_status = marmot_rope_freq_cache_ensure(nullptr, dim, params, &span);
    if (freq_status != MARMOT_SUCCESS || span.freqs == nullptr) {
        if (span.owns_buffer && span.freqs != nullptr) {
            free((void *)span.freqs);
        }
        return nil;
    }

    id<MTLBuffer> buffer = nil;
    if (span.owns_buffer) {
        buffer = [ctx->device newBufferWithBytesNoCopy:(void *)span.freqs
                                                length:freq_bytes
                                               options:MTLResourceStorageModeShared
                                           deallocator:^(void *ptr, NSUInteger length) {
                                             (void)length;
                                             free(ptr);
                                           }];
    } else {
        buffer = [ctx->device newBufferWithBytesNoCopy:(void *)span.freqs
                                                length:freq_bytes
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
    }
    if (buffer == nil) {
        if (span.owns_buffer && span.freqs != nullptr) {
            free((void *)span.freqs);
        }
    }
    if (buffer == nil) {
        return nil;
    }
    if (ctx->rope_cache.buffer != nil) {
        [ctx->rope_cache.buffer release];
    }
    ctx->rope_cache.buffer = [buffer retain];
    ctx->rope_cache.dim = dim;
    ctx->rope_cache.theta = params->theta;
    ctx->rope_cache.freq_scale = params->freq_scale;
    ctx->rope_cache.ext_factor = params->ext_factor;
    ctx->rope_cache.attn_factor = params->attn_factor;
    ctx->rope_cache.beta_fast = params->beta_fast;
    ctx->rope_cache.beta_slow = params->beta_slow;
    ctx->rope_cache.orig_ctx_len = params->orig_ctx_len;
    ctx->rope_cache.scaling_type = params->scaling_type;
    ctx->rope_cache.attn_scale = span.attn_scale;
    ctx->rope_cache.capacity_bytes = freq_bytes;
    if (attn_scale_out != nullptr) {
        *attn_scale_out = span.attn_scale;
    }
    return buffer;
}

const metal_matmul_epilogue_traits_t *metal_matmul_select_epilogue(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return &k_metal_matmul_f32_epilogue_traits;
    case MARMOT_DTYPE_FLOAT16:
        return &k_metal_matmul_f16_epilogue_traits;
    case MARMOT_DTYPE_BFLOAT16:
        return &k_metal_matmul_bf16_epilogue_traits;
    default:
        return nullptr;
    }
}

#undef METAL_MATMUL_DEFAULT_ACTIVATIONS_INIT

#ifdef __APPLE__

typedef struct {
    uint32_t total_elements;
    uint32_t bias_length;
    uint32_t activation;
    uint32_t flags;
    metal_matmul_activation_params_t params;
    uint32_t rope_dim;
    uint32_t rope_pairs;
    uint32_t rope_rows;
    uint32_t rope_flags;
    uint32_t rope_type;
    float rope_attn_scale;
} metal_matmul_fused_uniforms_t;

enum {
    METAL_MATMUL_FUSED_FLAG_SCALAR_BIAS = 1u << 0,
    METAL_MATMUL_FUSED_FLAG_HAS_BIAS = 1u << 1,
    METAL_MATMUL_FUSED_FLAG_HAS_RESIDUAL = 1u << 2,
    METAL_MATMUL_FUSED_FLAG_HAS_ROPE = 1u << 3,
};

bool metal_matmul_bias_dtype_supported(marmot_dtype_t out_dtype, marmot_dtype_t bias_dtype) {
    if (out_dtype == bias_dtype) {
        return true;
    }
    if (bias_dtype == MARMOT_DTYPE_FLOAT32 &&
        (out_dtype == MARMOT_DTYPE_FLOAT16 || out_dtype == MARMOT_DTYPE_BFLOAT16)) {
        return true;
    }
    return false;
}

static bool metal_matmul_bias_matches(
    const marmot_tensor_t *out, const marmot_tensor_t *bias, size_t *feature_dim, bool *is_scalar
) {
    if (out == nullptr || bias == nullptr || feature_dim == nullptr || is_scalar == nullptr) {
        return false;
    }
    if (!metal_matmul_bias_dtype_supported(out->dtype, bias->dtype)) {
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

bool metal_matmul_residual_matches(const marmot_tensor_t *out, const marmot_tensor_t *residual) {
    if (out == nullptr || residual == nullptr) {
        return false;
    }
    if (out->dtype != residual->dtype) {
        return false;
    }
    if (out->shape.ndim != residual->shape.ndim) {
        return false;
    }
    for (size_t i = 0; i < out->shape.ndim; ++i) {
        if (out->shape.shape[i] != residual->shape.shape[i]) {
            return false;
        }
    }
    return true;
}

metal_matmul_activation_params_t
metal_matmul_build_activation_params(marmot_device_unary_op_t op, const marmot_activation_params_t *params) {
    metal_matmul_activation_params_t result = {
        .alpha = 0.0f,
        .beta = 0.0f,
        .gamma = 0.0f,
        .delta = 0.0f,
    };
    if (params == nullptr) {
        return result;
    }
    switch (op) {
    case MARMOT_DEVICE_UNARY_ELU:
        result.alpha = params->alpha;
        break;
    case MARMOT_DEVICE_UNARY_SELU:
        result.alpha = params->alpha;
        result.beta = params->beta;
        break;
    case MARMOT_DEVICE_UNARY_LEAKY_RELU:
        result.alpha = params->alpha;
        break;
    case MARMOT_DEVICE_UNARY_PRELU:
        result.alpha = params->alpha;
        break;
    default:
        break;
    }
    return result;
}

static bool metal_matmul_activation_supported(marmot_dtype_t dtype, marmot_device_unary_op_t op) {
    if (op < 0 || op >= MARMOT_DEVICE_UNARY_COUNT) {
        return false;
    }
    const metal_matmul_epilogue_traits_t *traits = metal_matmul_select_epilogue(dtype);
    if (traits == nullptr) {
        return false;
    }
    return traits->activation_supported[op];
}

bool metal_matmul_epilogue_supported(
    const marmot_tensor_t *out, const marmot_matmul_epilogue_t *epilogue, size_t *feature_dim, bool *bias_is_scalar
) {
    if (out == nullptr || epilogue == nullptr) {
        return false;
    }
    const metal_matmul_epilogue_traits_t *traits = metal_matmul_select_epilogue(out->dtype);
    if (traits == nullptr || !traits->supports_bias) {
        return false;
    }
    if (epilogue->enable_output_cast) {
        return false;
    }
    if (epilogue->bias == nullptr) {
        return false;
    }
    size_t norm_feature_dim = out->shape.ndim > 0 ? out->shape.shape[out->shape.ndim - 1] : 1;
    if (feature_dim != nullptr) {
        *feature_dim = norm_feature_dim;
    }
    if (bias_is_scalar != nullptr) {
        *bias_is_scalar = false;
    }
    return metal_matmul_bias_matches(out, epilogue->bias, feature_dim, bias_is_scalar);
}

marmot_error_t metal_matmul_apply_epilogue(
    metal_context_t *ctx, const marmot_tensor_t *out, id<MTLBuffer> bufferOut, size_t bufferOutOffset,
    size_t total_elements, size_t feature_dim, bool bias_is_scalar, const marmot_matmul_epilogue_t *epilogue
) {
    if (ctx == nullptr || out == nullptr || bufferOut == nil || epilogue == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const marmot_tensor_t *bias = epilogue->bias;
    if (bias == nullptr) {
        return MARMOT_SUCCESS;
    }
    if (total_elements == 0) {
        return MARMOT_SUCCESS;
    }
    if (total_elements > UINT32_MAX || feature_dim > UINT32_MAX) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const metal_matmul_epilogue_traits_t *traits = metal_matmul_select_epilogue(out->dtype);
    if (traits == nullptr || traits->kernel_name == nullptr) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const char *kernel_name = traits->kernel_name;

    id<MTLBuffer> bufferBias = nil;
    size_t bias_elements = marmot_tensor_num_elements(bias);
    size_t bias_bytes = marmot_tensor_size_bytes(bias);
    bufferBias = metal_residency_acquire_existing(ctx, bias, bias->dtype);
    if (bufferBias == nil) {
        bufferBias = metal_buffer_acquire(ctx, bias->data, bias_bytes);
    }
    if (bufferBias == nil) {
        bufferBias = metal_residency_acquire_compute(ctx, bias, bias->dtype, nullptr);
    }
    if (bufferBias == nil) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLBuffer> bufferResidual = [bufferOut retain];

    const bool needs_bias_conversion = (bias->dtype != out->dtype);
    marmot_allocation_t converted_bias_alloc =
        {.ptr = nullptr, .size = 0, .alignment = 0, .type = MARMOT_ALLOC_HEAP, .alloc_id = 0};
    id<MTLBuffer> converted_bias_buffer = nil;
    if (needs_bias_conversion) {
        if (!(bias->dtype == MARMOT_DTYPE_FLOAT32 &&
              (out->dtype == MARMOT_DTYPE_FLOAT16 || out->dtype == MARMOT_DTYPE_BFLOAT16))) {
            [bufferResidual release];
            [bufferBias release];
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unsupported Metal matmul bias dtype conversion");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        size_t cached_elements = 0;
        bool have_cached = metal_bias_cache_fetch(ctx, bias->data, out->dtype, &converted_bias_alloc, &cached_elements);
        if (have_cached && cached_elements == bias_elements) {
            converted_bias_buffer = metal_buffer_lookup(ctx, converted_bias_alloc.ptr);
            if (converted_bias_buffer == nil) {
                have_cached = false;
            }
        }
        if (!have_cached) {
            size_t converted_bytes = bias_elements * marmot_dtype_size(out->dtype);
            if (metal_allocate_tracked(ctx, converted_bytes, MARMOT_ALLOC_GPU_SHARED, &converted_bias_alloc) !=
                MARMOT_SUCCESS) {
                [bufferResidual release];
                [bufferBias release];
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            converted_bias_buffer = metal_buffer_lookup(ctx, converted_bias_alloc.ptr);
            if (converted_bias_buffer == nil) {
                metal_allocator_ops.free(ctx, &converted_bias_alloc);
                [bufferResidual release];
                [bufferBias release];
                return MARMOT_ERROR_BACKEND_INIT_FAILED;
            }
            metal_bias_cache_store(ctx, bias->data, out->dtype, &converted_bias_alloc, bias_elements);
        }
        marmot_error_t bias_convert_status =
            metal_convert_dispatch(ctx, out->dtype, converted_bias_alloc.ptr, bias->dtype, bias->data, bias_elements);
        if (bias_convert_status != MARMOT_SUCCESS) {
            [bufferResidual release];
            [bufferBias release];
            if (converted_bias_buffer != nil) {
                [converted_bias_buffer release];
            }
            return bias_convert_status;
        }
    }

    metal_matmul_activation_params_t params = {
        .alpha = 0.0f,
        .beta = 0.0f,
        .gamma = 0.0f,
        .delta = 0.0f,
    };
    uint32_t bias_length = (uint32_t)(bias_is_scalar ? 1u : feature_dim);
    uint32_t flags = METAL_MATMUL_FUSED_FLAG_HAS_BIAS;
    if (bias_is_scalar) {
        flags |= METAL_MATMUL_FUSED_FLAG_SCALAR_BIAS;
    }
    metal_matmul_fused_uniforms_t uniforms = {
        .total_elements = (uint32_t)total_elements,
        .bias_length = bias_length,
        .activation = (uint32_t)MARMOT_DEVICE_UNARY_IDENTITY,
        .flags = flags,
        .params = params,
        .rope_dim = 0,
        .rope_pairs = 0,
        .rope_rows = 0,
        .rope_flags = 0,
        .rope_type = (uint32_t)MARMOT_ROPE_TYPE_NORM,
        .rope_attn_scale = 1.0f,
    };

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        [bufferResidual release];
        [bufferBias release];
        if (converted_bias_buffer != nil) {
            [converted_bias_buffer release];
        }
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferResidual release];
        [bufferBias release];
        if (converted_bias_buffer != nil) {
            [converted_bias_buffer release];
        }
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLBuffer> activeBias = nil;
    if (needs_bias_conversion) {
        activeBias = converted_bias_buffer;
    } else {
        activeBias = bufferBias;
        bufferBias = nil;
    }

    [encoder setBuffer:bufferOut offset:bufferOutOffset atIndex:0];
    [encoder setBuffer:activeBias offset:0 atIndex:1];
    [encoder setBuffer:bufferResidual offset:bufferOutOffset atIndex:2];
    [encoder setBuffer:bufferOut offset:bufferOutOffset atIndex:3];
    [encoder setBytes:&uniforms length:sizeof(metal_matmul_fused_uniforms_t) atIndex:4];

    const size_t work_items_size = total_elements;
    MTLSize threads_per_threadgroup = metal_threads_for_elements(pipeline, (NSUInteger)work_items_size, 512);
    [encoder dispatchThreads:MTLSizeMake(work_items_size, 1, 1) threadsPerThreadgroup:threads_per_threadgroup];

    metal_command_stream_flush(ctx, false);

    [pipeline release];
    [activeBias release];
    [bufferResidual release];
    if (bufferBias != nil) {
        [bufferBias release];
    }
    metal_residency_mark_dirty(ctx, out, out->dtype);
    return MARMOT_SUCCESS;
}

extern "C" marmot_error_t metal_matmul_generic(
    metal_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out, const char *kernel_name
) {
    if (kernel_name == nullptr) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    const bool is_nn_kernel = strstr(kernel_name, "_nn") != nullptr;
    size_t feature_dim = 0;
    bool bias_is_scalar = false;
    bool apply_epilogue = false;
    if (epilogue != nullptr) {
        if (!metal_matmul_epilogue_supported(out, epilogue, &feature_dim, &bias_is_scalar)) {
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        apply_epilogue = true;
    }

    size_t rows = input->shape.shape[0];
    size_t inner = input->shape.shape[1];
    size_t cols = weight->shape.shape[0];
    if (is_nn_kernel) {
        cols = weight->shape.shape[1];
    }

    size_t bytesInput = marmot_tensor_size_bytes(input);
    size_t bytesWeight = marmot_tensor_size_bytes(weight);
    size_t bytesOut = marmot_tensor_size_bytes(out);

    metal_tensor_buffer_t input_view = metal_buffer_acquire_view(ctx, input, input->dtype, bytesInput);
    metal_tensor_buffer_t weight_view = metal_buffer_acquire_view(ctx, weight, weight->dtype, bytesWeight);
    metal_tensor_buffer_t out_view = metal_buffer_acquire_view(ctx, out, out->dtype, bytesOut);
    id<MTLBuffer> bufferInput = input_view.buffer;
    id<MTLBuffer> bufferWeight = weight_view.buffer;
    id<MTLBuffer> bufferOut = out_view.buffer;
    const size_t input_offset = input_view.offset;
    const size_t weight_offset = weight_view.offset;
    const size_t out_offset = out_view.offset;
    const bool out_private = out_view.is_private;

    if (bufferInput == nil || bufferWeight == nil || bufferOut == nil) {
        if (bufferInput != nil) {
            [bufferInput release];
        }
        if (bufferWeight != nil) {
            [bufferWeight release];
        }
        if (bufferOut != nil) {
            [bufferOut release];
        }
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        [bufferInput release];
        [bufferWeight release];
        [bufferOut release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    uint32_t rows_u32 = (uint32_t)rows;
    uint32_t inner_u32 = (uint32_t)inner;
    uint32_t cols_u32 = (uint32_t)cols;

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferInput release];
        [bufferWeight release];
        [bufferOut release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }
    [encoder setBuffer:bufferInput offset:input_offset atIndex:0];
    [encoder setBuffer:bufferWeight offset:weight_offset atIndex:1];
    [encoder setBuffer:bufferOut offset:out_offset atIndex:2];
    if (is_nn_kernel) {
        [encoder setBytes:&rows_u32 length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&inner_u32 length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&cols_u32 length:sizeof(uint32_t) atIndex:5];
    } else {
        [encoder setBytes:&rows_u32 length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&inner_u32 length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&cols_u32 length:sizeof(uint32_t) atIndex:5];
    }

    metal_profiling_set_label(ctx, "matmul");
    metal_profiling_begin(ctx);

    const NSUInteger tileM = 16;
    const NSUInteger tileN = 16;
    NSUInteger groupsX = (cols + tileN - 1) / tileN;
    NSUInteger groupsY = (rows + tileM - 1) / tileM;
    if (groupsX == 0) {
        groupsX = 1;
    }
    if (groupsY == 0) {
        groupsY = 1;
    }
    MTLSize threadgroups = MTLSizeMake(groupsX, groupsY, 1);
    MTLSize threadsPerGroup = MTLSizeMake(tileM, tileN, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];

    metal_profiling_end(ctx);

    bool epilogue_applied = false;
    if (apply_epilogue) {
        id<MTLBuffer> ep_buffer = [bufferOut retain];
        marmot_error_t ep_status = metal_matmul_apply_epilogue(
            ctx, out, ep_buffer, out_offset, rows * cols, feature_dim, bias_is_scalar, epilogue
        );
        [ep_buffer release];
        if (ep_status == MARMOT_SUCCESS) {
            epilogue_applied = true;
        } else {
            [pipeline release];
            [bufferInput release];
            [bufferWeight release];
            [bufferOut release];
            metal_command_stream_discard(ctx);
            return ep_status;
        }
    }

    if (!epilogue_applied) {
        metal_command_stream_flush(ctx, false);
    }

    [pipeline release];
    [bufferInput release];
    [bufferWeight release];
    [bufferOut release];
    if (!epilogue_applied && out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

typedef struct marmot_metal_gemm_params {
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t lda;
    int32_t ldb;
    int32_t ldd;
    int32_t tiles_n;
    int32_t tiles_m;
    int32_t swizzle_log;
    int32_t gemm_k_iterations_aligned;
} marmot_metal_gemm_params_t;

static id<MTLComputePipelineState> marmot_metal_pipeline_get_gemm(
    metal_context_t *ctx, const char *function_name, bool align_m, bool align_n, bool align_k
) {
    if (ctx == nullptr || function_name == nullptr) {
        return nil;
    }

    NSString *key = [NSString
        stringWithFormat:@"%s_am%d_an%d_ak%d", function_name, align_m ? 1 : 0, align_n ? 1 : 0, align_k ? 1 : 0];
    if (key == nil) {
        return nil;
    }

    pthread_mutex_lock(&ctx->pipeline_mutex);
    id<MTLComputePipelineState> cached = ctx->pipeline_cache[key];
    if (cached != nil) {
        [cached retain];
        pthread_mutex_unlock(&ctx->pipeline_mutex);
        return cached;
    }

    NSString *fn = [[NSString alloc] initWithUTF8String:function_name];
    if (fn == nil) {
        pthread_mutex_unlock(&ctx->pipeline_mutex);
        return nil;
    }

    MTLFunctionConstantValues *fc = [[MTLFunctionConstantValues alloc] init];
    if (fc == nil) {
        [fn release];
        pthread_mutex_unlock(&ctx->pipeline_mutex);
        return nil;
    }

    const BOOL am = align_m ? YES : NO;
    const BOOL an = align_n ? YES : NO;
    const BOOL ak = align_k ? YES : NO;
    [fc setConstantValue:&am type:MTLDataTypeBool atIndex:200];
    [fc setConstantValue:&an type:MTLDataTypeBool atIndex:201];
    [fc setConstantValue:&ak type:MTLDataTypeBool atIndex:202];

    NSError *error = nil;
    id<MTLFunction> function = [ctx->library newFunctionWithName:fn constantValues:fc error:&error];
    [fc release];
    [fn release];

    if (function == nil) {
        pthread_mutex_unlock(&ctx->pipeline_mutex);
        os_log_error(OS_LOG_DEFAULT, "Metal: failed to specialize %{public}s (%{public}@)", function_name, error);
        return nil;
    }

    id<MTLComputePipelineState> pipeline = [ctx->device newComputePipelineStateWithFunction:function error:&error];
    [function release];
    if (pipeline == nil) {
        pthread_mutex_unlock(&ctx->pipeline_mutex);
        os_log_error(OS_LOG_DEFAULT, "Metal: failed to create pipeline %{public}s (%{public}@)", function_name, error);
        return nil;
    }

    ctx->pipeline_cache[key] = pipeline;
    pthread_mutex_unlock(&ctx->pipeline_mutex);

    id<MTLComputePipelineState> result = [pipeline retain];
    [pipeline release];
    return result;
}

static const char *marmot_metal_gemm_dtype_name(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT16:
        return "f16";
    case MARMOT_DTYPE_BFLOAT16:
        return "bf16";
    case MARMOT_DTYPE_FLOAT32:
        return "f32";
    default:
        return nullptr;
    }
}

static void marmot_metal_gemm_select_params(
    metal_context_t *ctx, marmot_dtype_t dtype, bool transpose_b, size_t M, size_t N, size_t K, int *bm, int *bn,
    int *bk, int *wm, int *wn
) {
    (void)ctx;

    int bm_local = 64;
    int bn_local = 64;
    int bk_local = 16;
    int wm_local = 2;
    int wn_local = 2;

    NSString *device_name = ctx->device.name.lowercaseString;
    const bool is_large = [device_name containsString:@"max"] || [device_name containsString:@"ultra"];
    const bool is_small = !is_large && ![device_name containsString:@"pro"];
    const bool large_problem = (M * N) >= (1ull << 20);

    if (is_small) {
        if (transpose_b) {
            bm_local = 64;
            bn_local = 32;
            bk_local = 32;
            wm_local = 2;
            wn_local = 2;
        } else if (dtype != MARMOT_DTYPE_FLOAT32) {
            bm_local = 64;
            bn_local = 64;
            bk_local = 16;
            wm_local = 1;
            wn_local = 2;
        }
    } else if (is_large) {
        if (large_problem) {
            if (dtype != MARMOT_DTYPE_FLOAT32) {
                if ((2 * ((M > N) ? M : N)) > K) {
                    bm_local = 64;
                    bn_local = 64;
                    bk_local = 16;
                    wm_local = 1;
                    wn_local = 2;
                } else if (transpose_b) {
                    bm_local = 64;
                    bn_local = 32;
                    bk_local = 32;
                    wm_local = 2;
                    wn_local = 2;
                } else {
                    bm_local = 32;
                    bn_local = 64;
                    bk_local = 16;
                    wm_local = 1;
                    wn_local = 2;
                }
            }
        } else {
            if (dtype != MARMOT_DTYPE_FLOAT32) {
                if (transpose_b) {
                    bm_local = 64;
                    bn_local = 32;
                    bk_local = 32;
                    wm_local = 2;
                    wn_local = 2;
                } else {
                    bm_local = 64;
                    bn_local = 64;
                    bk_local = 16;
                    wm_local = 1;
                    wn_local = 2;
                }
            } else {
                if (transpose_b) {
                    bm_local = 32;
                    bn_local = 64;
                    bk_local = 16;
                    wm_local = 1;
                    wn_local = 2;
                } else {
                    bm_local = 64;
                    bn_local = 32;
                    bk_local = 32;
                    wm_local = 2;
                    wn_local = 2;
                }
            }
        }
    }

    *bm = bm_local;
    *bn = bn_local;
    *bk = bk_local;
    *wm = wm_local;
    *wn = wn_local;
}

extern "C" marmot_error_t marmot_metal_gemm(
    metal_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out, bool transpose_b
) {
    if (ctx == nullptr || input == nullptr || weight == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_dtype_t dtype = out->dtype;
    const char *dtype_name = marmot_metal_gemm_dtype_name(dtype);
    if (dtype_name == nullptr) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    size_t feature_dim = 0;
    bool bias_is_scalar = false;
    bool apply_epilogue = false;
    if (epilogue != nullptr) {
        if (!metal_matmul_epilogue_supported(out, epilogue, &feature_dim, &bias_is_scalar)) {
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        apply_epilogue = true;
    }

    const size_t M = input->shape.shape[0];
    const size_t K = input->shape.shape[1];
    const size_t N = transpose_b ? weight->shape.shape[0] : weight->shape.shape[1];

    if (M > INT32_MAX || N > INT32_MAX || K > INT32_MAX) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const int lda = (int)K;
    const int ldb = transpose_b ? (int)K : (int)N;
    const int ldd = (int)N;

    int bm = 0;
    int bn = 0;
    int bk = 0;
    int wm = 0;
    int wn = 0;
    marmot_metal_gemm_select_params(ctx, dtype, transpose_b, M, N, K, &bm, &bn, &bk, &wm, &wn);

    char kernel_name[192] = {0};
    snprintf(
        kernel_name, sizeof(kernel_name), "marmot_gemm_%s_%s_%s_bm%d_bn%d_bk%d_wm%d_wn%d", transpose_b ? "nt" : "nn",
        dtype_name, dtype_name, bm, bn, bk, wm, wn
    );

    const bool align_m = ((M % (size_t)bm) == 0);
    const bool align_n = ((N % (size_t)bn) == 0);
    const bool align_k = ((K % (size_t)bk) == 0);

    const int tiles_n = (int)((N + (size_t)bn - 1) / (size_t)bn);
    const int tiles_m = (int)((M + (size_t)bm - 1) / (size_t)bm);
    const int swizzle_log = (tiles_m <= 3) ? 0 : 1;

    marmot_metal_gemm_params_t params = {
        .M = (int32_t)M,
        .N = (int32_t)N,
        .K = (int32_t)K,
        .lda = (int32_t)lda,
        .ldb = (int32_t)ldb,
        .ldd = (int32_t)ldd,
        .tiles_n = (int32_t)tiles_n,
        .tiles_m = (int32_t)tiles_m,
        .swizzle_log = (int32_t)swizzle_log,
        .gemm_k_iterations_aligned = (int32_t)(K / (size_t)bk),
    };

    size_t bytesInput = marmot_tensor_size_bytes(input);
    size_t bytesWeight = marmot_tensor_size_bytes(weight);
    size_t bytesOut = marmot_tensor_size_bytes(out);

    metal_tensor_buffer_t input_view = metal_buffer_acquire_view(ctx, input, input->dtype, bytesInput);
    metal_tensor_buffer_t weight_view = metal_buffer_acquire_view(ctx, weight, weight->dtype, bytesWeight);
    metal_tensor_buffer_t out_view = metal_buffer_acquire_view(ctx, out, out->dtype, bytesOut);
    id<MTLBuffer> bufferInput = input_view.buffer;
    id<MTLBuffer> bufferWeight = weight_view.buffer;
    id<MTLBuffer> bufferOut = out_view.buffer;
    const size_t input_offset = input_view.offset;
    const size_t weight_offset = weight_view.offset;
    const size_t out_offset = out_view.offset;
    const bool out_private = out_view.is_private;

    if (bufferInput == nil || bufferWeight == nil || bufferOut == nil) {
        if (bufferInput != nil) {
            [bufferInput release];
        }
        if (bufferWeight != nil) {
            [bufferWeight release];
        }
        if (bufferOut != nil) {
            [bufferOut release];
        }
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputePipelineState> pipeline = marmot_metal_pipeline_get_gemm(ctx, kernel_name, align_m, align_n, align_k);
    if (pipeline == nil) {
        [bufferInput release];
        [bufferWeight release];
        [bufferOut release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferInput release];
        [bufferWeight release];
        [bufferOut release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    [encoder setBuffer:bufferInput offset:input_offset atIndex:0];
    [encoder setBuffer:bufferWeight offset:weight_offset atIndex:1];
    [encoder setBuffer:bufferOut offset:out_offset atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    metal_profiling_set_label(ctx, "matmul");
    metal_profiling_begin(ctx);

    const int tile = 1 << swizzle_log;
    int grid_m = tiles_m;
    int grid_n = tiles_n;
    grid_m = (grid_m + tile - 1) / tile;
    grid_n = grid_n * tile;

    MTLSize threadgroups = MTLSizeMake((NSUInteger)grid_n, (NSUInteger)grid_m, 1);
    MTLSize threadsPerGroup = MTLSizeMake(32, (NSUInteger)wn, (NSUInteger)wm);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];

    metal_profiling_end(ctx);

    bool epilogue_applied = false;
    if (apply_epilogue) {
        id<MTLBuffer> ep_buffer = [bufferOut retain];
        marmot_error_t ep_status =
            metal_matmul_apply_epilogue(ctx, out, ep_buffer, out_offset, M * N, feature_dim, bias_is_scalar, epilogue);
        [ep_buffer release];
        if (ep_status == MARMOT_SUCCESS) {
            epilogue_applied = true;
        } else {
            [pipeline release];
            [bufferInput release];
            [bufferWeight release];
            [bufferOut release];
            metal_command_stream_discard(ctx);
            return ep_status;
        }
    }

    if (!epilogue_applied) {
        metal_command_stream_flush(ctx, false);
    }

    [pipeline release];
    [bufferInput release];
    [bufferWeight release];
    [bufferOut release];

    if (!epilogue_applied && out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t metal_matmul_quantized(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (device_ctx == nullptr || input == nullptr || weight == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_matmul_dims_t dims = {0, 0, 0};
    marmot_matmul_activation_profile_t profile = {false, false, false, false};
    const marmot_quant_kind_traits_t *quant_traits = nullptr;
    marmot_error_t validate_status =
        marmot_matmul_validate_quantized(input, weight, out, &dims, &profile, &quant_traits);
    if (validate_status != MARMOT_SUCCESS) {
        return validate_status;
    }

    const size_t N = dims.N;
    const size_t K = dims.K;
    const size_t M = dims.M;
    const size_t weight_block_values = quant_traits->block_values;
    const size_t weight_blocks_per_row = (K + weight_block_values - 1) / weight_block_values;
    const bool uses_super_blocks = (weight_block_values != MARMOT_QUANT_BLOCK_SIZE);

    size_t ep_feature_dim = 0;
    bool ep_bias_scalar = false;
    const marmot_matmul_epilogue_t *ep_to_apply = epilogue;
    const bool epilogue_supported =
        (epilogue == nullptr || metal_matmul_epilogue_supported(out, ep_to_apply, &ep_feature_dim, &ep_bias_scalar));
    if (!epilogue_supported) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Quantized matmul epilogue not supported on Metal backend");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    // Direct dispatch via generated switch - no traits lookup needed
    return metal_matmul_quantized_dispatch(
        ctx, weight->quant_kind, input->dtype, out->dtype, input, weight, out, N, K, M, weight_blocks_per_row,
        uses_super_blocks, ep_to_apply, ep_feature_dim, ep_bias_scalar, nullptr
    );
}

#endif // __APPLE__
