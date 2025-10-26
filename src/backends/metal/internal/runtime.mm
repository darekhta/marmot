#include "core/helpers/elementwise.h"
#include "metal_backend_internal.h"
#include "metal_kernel_runtime.h"

#ifdef __APPLE__

#import <os/log.h>

static const char *metal_elementwise_binary_vec4_kernel(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return "elementwise_arith_f32_vec4";
    case MARMOT_DTYPE_FLOAT16:
        return "elementwise_arith_f16_vec4";
    case MARMOT_DTYPE_BFLOAT16:
        return "elementwise_arith_bf16_vec4";
    default:
        return nullptr;
    }
}

const char *metal_kernel_name_for_where(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT16:
        return "where_metal_f16";
    case MARMOT_DTYPE_FLOAT32:
        return "where_metal_f32";
    case MARMOT_DTYPE_BFLOAT16:
        return "where_metal_bf16";
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
        return "where_metal_fp8_e4m3";
    case MARMOT_DTYPE_FLOAT8_E5M2:
        return "where_metal_fp8_e5m2";
#endif
    default:
        return nullptr;
    }
}

marmot_error_t metal_elementwise_run_binary_kernel(
    metal_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out,
    const char *kernel_name, marmot_device_binary_op_t op
) {
    if (kernel_name == nullptr) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    size_t elements = marmot_tensor_num_elements(a);
    if (elements != marmot_tensor_num_elements(b) || elements != marmot_tensor_num_elements(out)) {
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    size_t elem_size = marmot_dtype_size(a->dtype);
    size_t bytes_a = elem_size * elements;
    size_t bytes_b = marmot_dtype_size(b->dtype) * elements;
    size_t bytes_out = marmot_dtype_size(out->dtype) * elements;

    metal_tensor_buffer_t viewA = metal_buffer_acquire_view(ctx, a, a->dtype, bytes_a);
    metal_tensor_buffer_t viewB = metal_buffer_acquire_view(ctx, b, b->dtype, bytes_b);
    metal_tensor_buffer_t viewOut = metal_buffer_acquire_view(ctx, out, out->dtype, bytes_out);
    id<MTLBuffer> bufferA = viewA.buffer;
    id<MTLBuffer> bufferB = viewB.buffer;
    id<MTLBuffer> bufferOut = viewOut.buffer;
    const size_t offsetA = viewA.offset;
    const size_t offsetB = viewB.offset;
    const size_t offsetOut = viewOut.offset;
    const bool out_private = viewOut.is_private;
    if (bufferA == nil || bufferB == nil || bufferOut == nil) {
        if (bufferA != nil)
            [bufferA release];
        if (bufferB != nil)
            [bufferB release];
        if (bufferOut != nil)
            [bufferOut release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    uint32_t op_value = (uint32_t)op;
    const char *vec4_kernel_name = metal_elementwise_binary_vec4_kernel(a->dtype);
    bool is_arithmetic = (op >= MARMOT_DEVICE_BINARY_ADD) && (op <= MARMOT_DEVICE_BINARY_MOD);
    if (!is_arithmetic) {
        is_arithmetic = (op == MARMOT_DEVICE_BINARY_SWIGLU) || (op == MARMOT_DEVICE_BINARY_GEGLU);
    }
    bool use_vectorized = is_arithmetic && (elements >= 4) && (vec4_kernel_name != nullptr);

    if (use_vectorized) {
        size_t vec4_count = elements / 4;
        size_t remainder = elements % 4;

        id<MTLComputePipelineState> vec4_pipeline = metal_pipeline_get(ctx, vec4_kernel_name);
        if (vec4_pipeline != nil) {
            id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, vec4_pipeline);
            if (encoder != nil) {
                uint32_t vec4_count_u32 = (uint32_t)vec4_count;
                [encoder setBuffer:bufferA offset:offsetA atIndex:0];
                [encoder setBuffer:bufferB offset:offsetB atIndex:1];
                [encoder setBuffer:bufferOut offset:offsetOut atIndex:2];
                [encoder setBytes:&op_value length:sizeof(uint32_t) atIndex:3];
                [encoder setBytes:&vec4_count_u32 length:sizeof(uint32_t) atIndex:4];

                MTLSize vec4_threadgroupSize = metal_threads_for_elements(vec4_pipeline, (NSUInteger)vec4_count, 512);
                [encoder dispatchThreads:MTLSizeMake(vec4_count, 1, 1) threadsPerThreadgroup:vec4_threadgroupSize];
            }
            [vec4_pipeline release];
        }

        if (remainder > 0) {
            id<MTLComputePipelineState> scalar_pipeline = metal_pipeline_get(ctx, kernel_name);
            if (scalar_pipeline != nil) {
                id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, scalar_pipeline);
                if (encoder != nil) {
                    size_t offset_bytes = vec4_count * 4 * elem_size;
                    [encoder setBuffer:bufferA offset:offsetA + offset_bytes atIndex:0];
                    [encoder setBuffer:bufferB offset:offsetB + offset_bytes atIndex:1];
                    [encoder setBuffer:bufferOut offset:offsetOut + offset_bytes atIndex:2];
                    [encoder setBytes:&op_value length:sizeof(uint32_t) atIndex:3];

                    MTLSize remainder_threadgroupSize =
                        metal_threads_for_elements(scalar_pipeline, (NSUInteger)remainder, 512);
                    [encoder dispatchThreads:MTLSizeMake(remainder, 1, 1)
                        threadsPerThreadgroup:remainder_threadgroupSize];
                }
                [scalar_pipeline release];
            }
        }
    } else {
        id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
        if (pipeline == nil) {
            [bufferA release];
            [bufferB release];
            [bufferOut release];
            return MARMOT_ERROR_BACKEND_INIT_FAILED;
        }

        id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
        if (encoder == nil) {
            [pipeline release];
            [bufferA release];
            [bufferB release];
            [bufferOut release];
            return MARMOT_ERROR_BACKEND_INIT_FAILED;
        }
        [encoder setBuffer:bufferA offset:offsetA atIndex:0];
        [encoder setBuffer:bufferB offset:offsetB atIndex:1];
        [encoder setBuffer:bufferOut offset:offsetOut atIndex:2];
        [encoder setBytes:&op_value length:sizeof(uint32_t) atIndex:3];

        MTLSize threadgroupSize = metal_threads_for_elements(pipeline, (NSUInteger)elements, 512);
        [encoder dispatchThreads:MTLSizeMake(elements, 1, 1) threadsPerThreadgroup:threadgroupSize];
        [pipeline release];
    }

    metal_command_stream_flush(ctx, false);

    [bufferA release];
    [bufferB release];
    [bufferOut release];
    if (out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t metal_elementwise_run_binary_kernel_row_strided(
    metal_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out,
    const char *kernel_name, marmot_device_binary_op_t op, uint32_t rows, uint32_t cols, size_t a_row_stride,
    size_t b_row_stride, size_t out_row_stride
) {
    if (kernel_name == nullptr) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (rows == 0 || cols == 0) {
        return MARMOT_SUCCESS;
    }

    size_t a_elem_size = marmot_dtype_size(a->dtype);
    size_t b_elem_size = marmot_dtype_size(b->dtype);
    size_t out_elem_size = marmot_dtype_size(out->dtype);
    if (a_elem_size == 0 || b_elem_size == 0 || out_elem_size == 0) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    size_t a_elems = ((size_t)rows - 1) * a_row_stride + cols;
    size_t b_elems = ((size_t)rows - 1) * b_row_stride + cols;
    size_t out_elems = ((size_t)rows - 1) * out_row_stride + cols;
    size_t bytes_a = a_elems * a_elem_size;
    size_t bytes_b = b_elems * b_elem_size;
    size_t bytes_out = out_elems * out_elem_size;

    if ((a->capacity_bytes != 0 && bytes_a > a->capacity_bytes) ||
        (b->capacity_bytes != 0 && bytes_b > b->capacity_bytes) ||
        (out->capacity_bytes != 0 && bytes_out > out->capacity_bytes)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    metal_tensor_buffer_t viewA = metal_buffer_acquire_view(ctx, a, a->dtype, bytes_a);
    metal_tensor_buffer_t viewB = metal_buffer_acquire_view(ctx, b, b->dtype, bytes_b);
    metal_tensor_buffer_t viewOut = metal_buffer_acquire_view(ctx, out, out->dtype, bytes_out);
    id<MTLBuffer> bufferA = viewA.buffer;
    id<MTLBuffer> bufferB = viewB.buffer;
    id<MTLBuffer> bufferOut = viewOut.buffer;
    const size_t offsetA = viewA.offset;
    const size_t offsetB = viewB.offset;
    const size_t offsetOut = viewOut.offset;
    const bool out_private = viewOut.is_private;
    if (bufferA == nil || bufferB == nil || bufferOut == nil) {
        if (bufferA != nil)
            [bufferA release];
        if (bufferB != nil)
            [bufferB release];
        if (bufferOut != nil)
            [bufferOut release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        [bufferA release];
        [bufferB release];
        [bufferOut release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferA release];
        [bufferB release];
        [bufferOut release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    uint32_t op_value = (uint32_t)op;
    const uint32_t total = rows * cols;
    [encoder setBuffer:bufferA offset:offsetA atIndex:0];
    [encoder setBuffer:bufferB offset:offsetB atIndex:1];
    [encoder setBuffer:bufferOut offset:offsetOut atIndex:2];
    [encoder setBytes:&op_value length:sizeof(op_value) atIndex:3];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:5];
    [encoder setBytes:&a_row_stride length:sizeof(a_row_stride) atIndex:6];
    [encoder setBytes:&b_row_stride length:sizeof(b_row_stride) atIndex:7];
    [encoder setBytes:&out_row_stride length:sizeof(out_row_stride) atIndex:8];

    MTLSize threads = metal_threads_for_elements(pipeline, (NSUInteger)total, 512);
    [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:threads];

    metal_command_stream_flush(ctx, false);

    [pipeline release];
    [bufferA release];
    [bufferB release];
    [bufferOut release];
    if (out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t metal_elementwise_run_unary_kernel(
    metal_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const char *kernel_name,
    const char *vec4_kernel_name, const metal_activation_params_t *args, bool has_args
) {
    if (kernel_name == nullptr) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    size_t elements = marmot_tensor_num_elements(x);
    if (elements != marmot_tensor_num_elements(out)) {
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    size_t elem_size = marmot_dtype_size(x->dtype);
    size_t bytes = elem_size * elements;
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

    size_t processed = 0;
    if (vec4_kernel_name != nullptr && elements >= 4 && !has_args) {
        size_t vec4_count = elements / 4;
        id<MTLComputePipelineState> vec4_pipeline = metal_pipeline_get(ctx, vec4_kernel_name);
        if (vec4_pipeline != nil) {
            id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, vec4_pipeline);
            if (encoder != nil) {
                uint32_t vec4_count_u32 = (uint32_t)vec4_count;
                [encoder setBuffer:bufferX offset:0 atIndex:0];
                [encoder setBuffer:bufferOut offset:0 atIndex:1];
                [encoder setBytes:&vec4_count_u32 length:sizeof(uint32_t) atIndex:2];

                MTLSize vec4_threadgroupSize = metal_threads_for_elements(vec4_pipeline, (NSUInteger)vec4_count, 512);
                [encoder dispatchThreads:MTLSizeMake(vec4_count, 1, 1) threadsPerThreadgroup:vec4_threadgroupSize];
            }
            [vec4_pipeline release];
            processed = vec4_count * 4;
        }
    }

    size_t remainder = elements - processed;
    if (remainder > 0) {
        id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
        if (pipeline == nil) {
            [bufferX release];
            [bufferOut release];
            return MARMOT_ERROR_BACKEND_INIT_FAILED;
        }

        id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
        if (encoder == nil) {
            [pipeline release];
            [bufferX release];
            [bufferOut release];
            return MARMOT_ERROR_BACKEND_INIT_FAILED;
        }
        size_t offset_bytes = processed * elem_size;
        [encoder setBuffer:bufferX offset:offset_bytes atIndex:0];
        [encoder setBuffer:bufferOut offset:offset_bytes atIndex:1];
        if (has_args) {
            [encoder setBytes:args length:sizeof(metal_activation_params_t) atIndex:2];
        }

        MTLSize threadgroupSize = metal_threads_for_elements(pipeline, (NSUInteger)remainder, 512);
        [encoder dispatchThreads:MTLSizeMake(remainder, 1, 1) threadsPerThreadgroup:threadgroupSize];
        [pipeline release];
    }

    metal_command_stream_flush(ctx, false);

    [bufferX release];
    [bufferOut release];
    if (out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t metal_elementwise_run_where_kernel(
    metal_context_t *ctx, const marmot_tensor_t *mask, const marmot_tensor_t *a, const marmot_tensor_t *b,
    marmot_tensor_t *out, const char *kernel_name
) {
    if (kernel_name == nullptr) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    size_t elements = marmot_tensor_num_elements(mask);
    if (elements != marmot_tensor_num_elements(a) || elements != marmot_tensor_num_elements(b) ||
        elements != marmot_tensor_num_elements(out)) {
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    size_t mask_bytes = marmot_dtype_size(mask->dtype) * elements;
    size_t elem_size = marmot_dtype_size(a->dtype);
    size_t bytes_a = elem_size * elements;
    size_t bytes_b = marmot_dtype_size(b->dtype) * elements;
    size_t bytes_out = marmot_dtype_size(out->dtype) * elements;

    id<MTLBuffer> bufferMask = metal_residency_acquire_existing(ctx, mask, mask->dtype);
    if (bufferMask == nil) {
        bufferMask = metal_residency_acquire_compute(ctx, mask, mask->dtype, nullptr);
    }
    if (bufferMask == nil) {
        bufferMask = metal_buffer_acquire(ctx, mask->data, mask_bytes);
    }

    id<MTLBuffer> bufferA = metal_residency_acquire_existing(ctx, a, a->dtype);
    if (bufferA == nil) {
        bufferA = metal_residency_acquire_compute(ctx, a, a->dtype, nullptr);
    }
    if (bufferA == nil) {
        bufferA = metal_buffer_acquire(ctx, a->data, bytes_a);
    }

    id<MTLBuffer> bufferB = metal_residency_acquire_existing(ctx, b, b->dtype);
    if (bufferB == nil) {
        bufferB = metal_residency_acquire_compute(ctx, b, b->dtype, nullptr);
    }
    if (bufferB == nil) {
        bufferB = metal_buffer_acquire(ctx, b->data, bytes_b);
    }

    bool out_private = false;
    bool out_is_new = false;
    id<MTLBuffer> bufferOut = metal_residency_acquire_compute(ctx, out, out->dtype, &out_is_new);
    if (bufferOut == nil) {
        bufferOut = metal_buffer_acquire(ctx, out->data, bytes_out);
    } else {
        out_private = true;
    }

    if (bufferMask == nil || bufferA == nil || bufferB == nil || bufferOut == nil) {
        if (bufferMask != nil)
            [bufferMask release];
        if (bufferA != nil)
            [bufferA release];
        if (bufferB != nil)
            [bufferB release];
        if (bufferOut != nil)
            [bufferOut release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        [bufferMask release];
        [bufferA release];
        [bufferB release];
        [bufferOut release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferMask release];
        [bufferA release];
        [bufferB release];
        [bufferOut release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    [encoder setBuffer:bufferMask offset:0 atIndex:0];
    [encoder setBuffer:bufferA offset:0 atIndex:1];
    [encoder setBuffer:bufferB offset:0 atIndex:2];
    [encoder setBuffer:bufferOut offset:0 atIndex:3];

    MTLSize threadgroupSize = metal_threads_for_elements(pipeline, (NSUInteger)elements, 512);
    [encoder dispatchThreads:MTLSizeMake(elements, 1, 1) threadsPerThreadgroup:threadgroupSize];

    metal_command_stream_flush(ctx, false);

    [pipeline release];
    [bufferMask release];
    [bufferA release];
    [bufferB release];
    [bufferOut release];
    if (out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t metal_elementwise_run_fused_bias_activation(
    metal_context_t *ctx, marmot_dtype_t dtype, const char *kernel_name, marmot_device_unary_op_t op,
    const marmot_tensor_t *x, const marmot_tensor_t *bias, marmot_tensor_t *out,
    const metal_activation_params_t *activation_params
) {
    if (!marmot_elementwise_unary_supports_bias(op)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (x == nullptr || bias == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (x->dtype != out->dtype) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (bias->dtype != x->dtype) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    size_t feature_dim = 0;
    bool bias_scalar = false;
    marmot_error_t bias_status = marmot_elementwise_bias_info(x, bias, &feature_dim, &bias_scalar);
    if (bias_status != MARMOT_SUCCESS) {
        return bias_status;
    }
    size_t elements = marmot_tensor_num_elements(x);
    if (elements != marmot_tensor_num_elements(out)) {
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (elements > UINT32_MAX || feature_dim > UINT32_MAX) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    if (kernel_name == nullptr) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    size_t bytes_x = marmot_tensor_size_bytes(x);
    size_t bytes_bias = marmot_tensor_size_bytes(bias);
    size_t bytes_out = marmot_tensor_size_bytes(out);

    id<MTLBuffer> bufferX = metal_residency_acquire_existing(ctx, x, x->dtype);
    if (bufferX == nil) {
        bufferX = metal_residency_acquire_compute(ctx, x, x->dtype, nullptr);
    }
    if (bufferX == nil) {
        bufferX = metal_buffer_acquire(ctx, x->data, bytes_x);
    }

    id<MTLBuffer> bufferBias = metal_residency_acquire_existing(ctx, bias, bias->dtype);
    if (bufferBias == nil) {
        bufferBias = metal_residency_acquire_compute(ctx, bias, bias->dtype, nullptr);
    }
    if (bufferBias == nil) {
        bufferBias = metal_buffer_acquire(ctx, bias->data, bytes_bias);
    }

    bool out_private = false;
    bool out_is_new = false;
    id<MTLBuffer> bufferOut = metal_residency_acquire_compute(ctx, out, out->dtype, &out_is_new);
    if (bufferOut == nil) {
        bufferOut = metal_buffer_acquire(ctx, out->data, bytes_out);
    } else {
        out_private = true;
    }

    if (bufferX == nil || bufferBias == nil || bufferOut == nil) {
        if (bufferX != nil) {
            [bufferX release];
        }
        if (bufferBias != nil) {
            [bufferBias release];
        }
        if (bufferOut != nil) {
            [bufferOut release];
        }
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        [bufferX release];
        [bufferBias release];
        [bufferOut release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLBuffer> bufferResidual = [bufferOut retain];

    metal_activation_params_t params = {.alpha = 0.0f, .beta = 0.0f, .gamma = 0.0f, .delta = 0.0f};
    if (activation_params != nullptr) {
        params = *activation_params;
    }
    metal_fused_bias_activation_uniforms_t uniforms = {
        .total_elements = (uint32_t)elements,
        .bias_length = (uint32_t)(bias_scalar ? 1u : feature_dim),
        .activation = (uint32_t)op,
        .flags = (bias_scalar ? METAL_FUSED_BIAS_FLAG_SCALAR : 0u) | METAL_FUSED_BIAS_FLAG_HAS_BIAS,
        .params = params,
    };

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferX release];
        [bufferBias release];
        [bufferResidual release];
        [bufferOut release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    [encoder setBuffer:bufferX offset:0 atIndex:0];
    [encoder setBuffer:bufferBias offset:0 atIndex:1];
    [encoder setBuffer:bufferResidual offset:0 atIndex:2];
    [encoder setBuffer:bufferOut offset:0 atIndex:3];
    [encoder setBytes:&uniforms length:sizeof(metal_fused_bias_activation_uniforms_t) atIndex:4];

    MTLSize threadgroupSize = metal_threads_for_elements(pipeline, (NSUInteger)elements, 512);
    [encoder dispatchThreads:MTLSizeMake(elements, 1, 1) threadsPerThreadgroup:threadgroupSize];

    metal_command_stream_flush(ctx, false);

    [pipeline release];
    [bufferX release];
    [bufferBias release];
    [bufferResidual release];
    [bufferOut release];
    if (out_private) {
        metal_residency_mark_dirty(ctx, out, dtype);
    }
    return MARMOT_SUCCESS;
}

#endif // __APPLE__
