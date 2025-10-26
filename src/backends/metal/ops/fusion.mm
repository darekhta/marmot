#include "marmot/error.h"
#include "marmot/tensor.h"

#include <stdint.h>

#include "metal_backend_internal.h"
#include "metal_fusion.h"

#ifdef __APPLE__

#define METAL_FUSION_MAX_INPUTS 4

typedef struct {
    const char *f32_kernel;
    const char *f16_kernel;
    const char *error_msg;
} metal_fusion_kernel_entry_t;

static const metal_fusion_kernel_entry_t kFusionAddRelu = {
    .f32_kernel = "metal_add_relu_fused_f32",
    .f16_kernel = "metal_add_relu_fused_f16",
    .error_msg = "ADD+RELU fusion supports f16/f32 only",
};

static const metal_fusion_kernel_entry_t kFusionAddGelu = {
    .f32_kernel = "metal_add_gelu_fused_f32",
    .f16_kernel = "metal_add_gelu_fused_f16",
    .error_msg = "ADD+GELU fusion supports f16/f32 only",
};

static const metal_fusion_kernel_entry_t kFusionAddSilu = {
    .f32_kernel = "metal_add_silu_fused_f32",
    .f16_kernel = "metal_add_silu_fused_f16",
    .error_msg = "ADD+SILU fusion supports f16/f32 only",
};

static const char *metal_fusion_select_kernel(const metal_fusion_kernel_entry_t *entry, marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return entry->f32_kernel;
    case MARMOT_DTYPE_FLOAT16:
        return entry->f16_kernel;
    default:
        return nullptr;
    }
}

static bool metal_tensor_shapes_match(const marmot_tensor_t *a, const marmot_tensor_t *b) {
    if (a == nullptr || b == nullptr || a->shape.ndim != b->shape.ndim) {
        return false;
    }
    for (size_t i = 0; i < a->shape.ndim; ++i) {
        if (a->shape.shape[i] != b->shape.shape[i]) {
            return false;
        }
    }
    return true;
}

static marmot_error_t
metal_fusion_validate_inputs(const marmot_tensor_t **inputs, size_t input_count, marmot_tensor_t *out) {
    if (input_count == 0 || inputs == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_tensor_t *first = inputs[0];
    if (first == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    for (size_t i = 1; i < input_count; ++i) {
        if (inputs[i] == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (inputs[i]->dtype != first->dtype) {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Fusion tensors must share dtype");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        if (!metal_tensor_shapes_match(first, inputs[i])) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Fusion tensors must share shape");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }

    if (out->dtype != first->dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Fusion tensors must share dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (!metal_tensor_shapes_match(first, out)) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Fusion tensors must share shape");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t metal_launch_fusion_kernel(
    const marmot_context_t *ctx, const char *kernel_name, const marmot_tensor_t **inputs, size_t input_count,
    marmot_tensor_t *out
) {
    if (kernel_name == nullptr) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    marmot_error_t validate = metal_fusion_validate_inputs(inputs, input_count, out);
    if (validate != MARMOT_SUCCESS) {
        return validate;
    }

    if (ctx == nullptr || ctx->device_ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    metal_context_t *metal_ctx = (metal_context_t *)ctx->device_ctx;
    if (metal_ctx->device == nil) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t elements = marmot_tensor_num_elements(inputs[0]);
    if (elements == 0) {
        return MARMOT_SUCCESS;
    }
    if (elements > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fusion element count exceeds limit");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t elem_size = marmot_dtype_size(inputs[0]->dtype);
    size_t bytes = elem_size * elements;

    id<MTLBuffer> buffers[METAL_FUSION_MAX_INPUTS + 1] = {nil};
    for (size_t i = 0; i < input_count; ++i) {
        buffers[i] = metal_buffer_acquire(metal_ctx, inputs[i]->data, bytes);
    }

    bool out_private = false;
    buffers[input_count] = metal_residency_acquire_compute(metal_ctx, out, out->dtype, &out_private);
    if (buffers[input_count] == nil) {
        buffers[input_count] = metal_buffer_acquire(metal_ctx, out->data, bytes);
    }

    bool any_nil = false;
    for (size_t i = 0; i <= input_count; ++i) {
        if (buffers[i] == nil) {
            any_nil = true;
            break;
        }
    }

    if (any_nil) {
        for (size_t i = 0; i <= input_count; ++i) {
            if (buffers[i] != nil) {
                [buffers[i] release];
            }
        }
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(metal_ctx, kernel_name);
    if (pipeline == nil) {
        for (size_t i = 0; i <= input_count; ++i) {
            [buffers[i] release];
        }
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(metal_ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        for (size_t i = 0; i <= input_count; ++i) {
            [buffers[i] release];
        }
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    for (size_t i = 0; i <= input_count; ++i) {
        [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    }
    uint32_t n = (uint32_t)elements;
    [encoder setBytes:&n length:sizeof(uint32_t) atIndex:input_count + 1];

    MTLSize grid = MTLSizeMake(elements, 1, 1);
    MTLSize threadgroup = metal_threads_for_elements(pipeline, (NSUInteger)elements, 512);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];

    metal_command_stream_flush(metal_ctx, false);

    [pipeline release];
    for (size_t i = 0; i <= input_count; ++i) {
        [buffers[i] release];
    }

    if (out_private) {
        metal_residency_mark_dirty(metal_ctx, out, out->dtype);
    }

    return MARMOT_SUCCESS;
}

marmot_error_t metal_add_relu_fused(
    const marmot_context_t *ctx, const marmot_tensor_t *input_a, const marmot_tensor_t *input_b, marmot_tensor_t *output
) {
    if (input_a == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const char *kernel = metal_fusion_select_kernel(&kFusionAddRelu, input_a->dtype);
    if (kernel == nullptr) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, kFusionAddRelu.error_msg);
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    const marmot_tensor_t *inputs[] = {input_a, input_b};
    return metal_launch_fusion_kernel(ctx, kernel, inputs, 2, output);
}

marmot_error_t metal_add_gelu_fused(
    const marmot_context_t *ctx, const marmot_tensor_t *input_a, const marmot_tensor_t *input_b, marmot_tensor_t *output
) {
    if (input_a == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const char *kernel = metal_fusion_select_kernel(&kFusionAddGelu, input_a->dtype);
    if (kernel == nullptr) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, kFusionAddGelu.error_msg);
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    const marmot_tensor_t *inputs[] = {input_a, input_b};
    return metal_launch_fusion_kernel(ctx, kernel, inputs, 2, output);
}

marmot_error_t metal_add_silu_fused(
    const marmot_context_t *ctx, const marmot_tensor_t *input_a, const marmot_tensor_t *input_b, marmot_tensor_t *output
) {
    if (input_a == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const char *kernel = metal_fusion_select_kernel(&kFusionAddSilu, input_a->dtype);
    if (kernel == nullptr) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, kFusionAddSilu.error_msg);
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    const marmot_tensor_t *inputs[] = {input_a, input_b};
    return metal_launch_fusion_kernel(ctx, kernel, inputs, 2, output);
}

marmot_error_t metal_mul_add_fused(
    const marmot_context_t *ctx, const marmot_tensor_t *input_a, const marmot_tensor_t *input_b,
    const marmot_tensor_t *input_c, marmot_tensor_t *output
) {
    if (input_a == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input_a->dtype != MARMOT_DTYPE_FLOAT32) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "MUL+ADD fusion supports f32 only");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    const marmot_tensor_t *inputs[] = {input_a, input_b, input_c};
    return metal_launch_fusion_kernel(ctx, "metal_mul_add_fused_f32", inputs, 3, output);
}

#endif
