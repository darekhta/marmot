#include <math.h>
#include <string.h>

#include "metal_quant_schemes.h"

#ifdef __APPLE__

#include "marmot/quant_block.h"

typedef marmot_error_t (*metal_quant_handler_t)(metal_context_t *, const marmot_tensor_t *, marmot_tensor_t *);

typedef struct {
    marmot_quant_kind_t kind;
    metal_quant_handler_t quantize;
    metal_quant_handler_t dequantize;
} metal_quant_scheme_entry_t;

static const metal_quant_scheme_entry_t kMetalQuantSchemes[] = {
    {MARMOT_QUANT_KIND_Q4_0, metal_quantize_q4_0_impl, metal_dequantize_q4_0_impl},
    {MARMOT_QUANT_KIND_Q4_1, metal_quantize_q4_1_impl, metal_dequantize_q4_1_impl},
    {MARMOT_QUANT_KIND_Q5_0, metal_quantize_q5_0_impl, metal_dequantize_q5_0_impl},
    {MARMOT_QUANT_KIND_Q5_1, metal_quantize_q5_1_impl, metal_dequantize_q5_1_impl},
    {MARMOT_QUANT_KIND_Q8_0, metal_quantize_q8_0_impl, metal_dequantize_q8_0_impl},
    {MARMOT_QUANT_KIND_Q8_1, metal_quantize_q8_1_impl, metal_dequantize_q8_1_impl},
    {MARMOT_QUANT_KIND_Q2_K, metal_quantize_q2_k_impl, metal_dequantize_q2_k_impl},
    {MARMOT_QUANT_KIND_Q3_K, metal_quantize_q3_k_impl, metal_dequantize_q3_k_impl},
    {MARMOT_QUANT_KIND_Q4_K, metal_quantize_q4_k_impl, metal_dequantize_q4_k_impl},
    {MARMOT_QUANT_KIND_Q5_K, metal_quantize_q5_k_impl, metal_dequantize_q5_k_impl},
    {MARMOT_QUANT_KIND_Q6_K, metal_quantize_q6_k_impl, metal_dequantize_q6_k_impl},
    {MARMOT_QUANT_KIND_Q8_K, metal_quantize_q8_k_impl, metal_dequantize_q8_k_impl},
};

static const metal_quant_scheme_entry_t *metal_find_quant_scheme(marmot_quant_kind_t kind) {
    for (size_t i = 0; i < sizeof(kMetalQuantSchemes) / sizeof(kMetalQuantSchemes[0]); ++i) {
        if (kMetalQuantSchemes[i].kind == kind) {
            return &kMetalQuantSchemes[i];
        }
    }
    return nullptr;
}

// Generic INT8/UINT8 quantization

static marmot_error_t metal_quantize_generic_impl(
    metal_context_t *ctx, const marmot_tensor_t *input, const marmot_quant_params_t *quant_params,
    marmot_tensor_t *output
) {
    if (quant_params == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Quantize requires quantization parameters");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (output->dtype != MARMOT_DTYPE_INT8 && output->dtype != MARMOT_DTYPE_UINT8) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Generic quantize supports INT8/UINT8 outputs only");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    if (input->shape.ndim != output->shape.ndim) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Input and output shapes must match for quantize");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    for (size_t i = 0; i < input->shape.ndim; ++i) {
        if (input->shape.shape[i] != output->shape.shape[i]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Input and output shapes must match for quantize");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }

    size_t num_elements = marmot_tensor_num_elements(input);
    if (num_elements != marmot_tensor_num_elements(output)) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Input and output sizes differ for quantize");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    id<MTLBuffer> buffer_input = metal_residency_acquire_existing(ctx, input, input->dtype);
    if (buffer_input == nil) {
        buffer_input = metal_buffer_acquire(ctx, input->data, num_elements * sizeof(float));
    }

    bool out_private = false;
    bool out_is_new = false;
    id<MTLBuffer> buffer_output = metal_residency_acquire_compute(ctx, output, output->dtype, &out_is_new);
    if (buffer_output == nil) {
        buffer_output = metal_buffer_acquire(ctx, output->data, marmot_tensor_size_bytes(output));
    } else {
        out_private = true;
    }
    if (buffer_input == nil || buffer_output == nil) {
        if (buffer_input != nil)
            [buffer_input release];
        if (buffer_output != nil)
            [buffer_output release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    const char *kernel_name = (output->dtype == MARMOT_DTYPE_INT8) ? "quantize_int8" : "quantize_uint8";
    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        [buffer_input release];
        [buffer_output release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    if (output->quant_params == nullptr) {
        output->quant_params = (marmot_quant_params_t *)malloc(sizeof(marmot_quant_params_t));
        if (output->quant_params == nullptr) {
            [pipeline release];
            [buffer_input release];
            [buffer_output release];
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate quant params");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }

    float scale = quant_params->scale;
    float zero_point = (float)quant_params->zero_point;
    int qmin = (output->dtype == MARMOT_DTYPE_INT8) ? -128 : 0;
    int qmax = (output->dtype == MARMOT_DTYPE_INT8) ? 127 : 255;
    uint32_t count = (uint32_t)num_elements;

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [buffer_input release];
        [buffer_output release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }
    [encoder setBuffer:buffer_input offset:0 atIndex:0];
    [encoder setBuffer:buffer_output offset:0 atIndex:1];
    [encoder setBytes:&scale length:sizeof(float) atIndex:2];
    [encoder setBytes:&zero_point length:sizeof(float) atIndex:3];
    [encoder setBytes:&qmin length:sizeof(int) atIndex:4];
    [encoder setBytes:&qmax length:sizeof(int) atIndex:5];
    [encoder setBytes:&count length:sizeof(uint32_t) atIndex:6];

    MTLSize tpg = metal_threads_for_elements(pipeline, (NSUInteger)num_elements, 512);
    [encoder dispatchThreads:MTLSizeMake(num_elements, 1, 1) threadsPerThreadgroup:tpg];

    metal_command_stream_flush(ctx, false);

    memcpy(output->quant_params, quant_params, sizeof(marmot_quant_params_t));

    [pipeline release];
    [buffer_input release];
    [buffer_output release];
    if (out_private) {
        metal_residency_mark_dirty(ctx, output, output->dtype);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_dequantize_generic_impl(
    metal_context_t *ctx, const marmot_tensor_t *input, const marmot_quant_params_t *quant_params,
    marmot_tensor_t *output
) {
    if (quant_params == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Input tensor missing quantization parameters");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (input->dtype != MARMOT_DTYPE_INT8 && input->dtype != MARMOT_DTYPE_UINT8) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Generic dequantize supports INT8/UINT8 inputs only");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    if (input->shape.ndim != output->shape.ndim) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Input and output shapes must match for dequantize");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    for (size_t i = 0; i < input->shape.ndim; ++i) {
        if (input->shape.shape[i] != output->shape.shape[i]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Input and output shapes must match for dequantize");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }

    size_t num_elements = marmot_tensor_num_elements(input);
    if (num_elements != marmot_tensor_num_elements(output)) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Input and output sizes differ for dequantize");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    id<MTLBuffer> buffer_input = metal_residency_acquire_existing(ctx, input, input->dtype);
    if (buffer_input == nil) {
        buffer_input = metal_buffer_acquire(ctx, input->data, marmot_tensor_size_bytes(input));
    }

    bool out_private = false;
    bool out_is_new = false;
    id<MTLBuffer> buffer_output = metal_residency_acquire_compute(ctx, output, output->dtype, &out_is_new);
    if (buffer_output == nil) {
        buffer_output = metal_buffer_acquire(ctx, output->data, num_elements * sizeof(float));
    } else {
        out_private = true;
    }
    if (buffer_input == nil || buffer_output == nil) {
        if (buffer_input != nil)
            [buffer_input release];
        if (buffer_output != nil)
            [buffer_output release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    const char *kernel_name = (input->dtype == MARMOT_DTYPE_INT8) ? "dequantize_int8" : "dequantize_uint8";
    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        [buffer_input release];
        [buffer_output release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    float scale = quant_params->scale;
    float zero_point = (float)quant_params->zero_point;
    uint32_t count = (uint32_t)num_elements;

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [buffer_input release];
        [buffer_output release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }
    [encoder setBuffer:buffer_input offset:0 atIndex:0];
    [encoder setBuffer:buffer_output offset:0 atIndex:1];
    [encoder setBytes:&scale length:sizeof(float) atIndex:2];
    [encoder setBytes:&zero_point length:sizeof(float) atIndex:3];
    [encoder setBytes:&count length:sizeof(uint32_t) atIndex:4];

    MTLSize tpg = metal_threads_for_elements(pipeline, (NSUInteger)num_elements, 512);
    [encoder dispatchThreads:MTLSizeMake(num_elements, 1, 1) threadsPerThreadgroup:tpg];

    metal_command_stream_flush(ctx, false);

    [pipeline release];
    [buffer_input release];
    [buffer_output release];
    if (out_private) {
        metal_residency_mark_dirty(ctx, output, output->dtype);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t metal_compute_quant_params(
    const void *device_ctx, const marmot_tensor_t *tensor, marmot_dtype_t target_dtype, size_t block_size,
    marmot_quant_params_t *out_params
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr || tensor == nullptr || out_params == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (tensor->dtype != MARMOT_DTYPE_FLOAT32) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Compute quant params requires FLOAT32 input");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    if (target_dtype != MARMOT_DTYPE_INT8 && target_dtype != MARMOT_DTYPE_UINT8) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Quantization only supports INT8/UINT8 targets on Metal");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    if (block_size != 0) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Per-block quantization not implemented on Metal");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    size_t num_elements = marmot_tensor_num_elements(tensor);
    if (num_elements == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Compute quant params requires non-empty tensor");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    id<MTLBuffer> buffer_input = metal_residency_acquire_existing(ctx, tensor, tensor->dtype);
    if (buffer_input == nil) {
        buffer_input = metal_buffer_acquire(ctx, tensor->data, marmot_tensor_size_bytes(tensor));
    }
    if (buffer_input == nil) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, "compute_quant_params_int8");
    if (pipeline == nil) {
        [buffer_input release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLBuffer> params_buffer = [ctx->device newBufferWithLength:sizeof(float) * 2
                                                           options:MTLResourceStorageModeShared];
    if (params_buffer == nil) {
        [pipeline release];
        [buffer_input release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    uint32_t element_count = (uint32_t)num_elements;
    uint32_t is_unsigned = target_dtype == MARMOT_DTYPE_UINT8 ? 1u : 0u;

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [params_buffer release];
        [pipeline release];
        [buffer_input release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    [encoder setBuffer:buffer_input offset:0 atIndex:0];
    [encoder setBuffer:params_buffer offset:0 atIndex:1];
    [encoder setBytes:&element_count length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&is_unsigned length:sizeof(uint32_t) atIndex:3];

    MTLSize threads = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreads:threads threadsPerThreadgroup:threads];

    metal_command_stream_flush(ctx, true);

    const float *params = (const float *)[params_buffer contents];
    out_params->scale = params[0];
    out_params->zero_point = params[1];
    out_params->block_size = block_size;

    [params_buffer release];
    [pipeline release];
    [buffer_input release];
    return MARMOT_SUCCESS;
}

marmot_error_t metal_quantize_dispatch(
    const void *device_ctx, marmot_quant_kind_t kind, marmot_quant_layout_t layout, const marmot_tensor_t *input,
    const marmot_quant_params_t *quant_params, marmot_tensor_t *output
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr || input == nullptr || output == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (input->dtype != MARMOT_DTYPE_FLOAT32) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Metal quantize requires FLOAT32 input");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    if (kind != MARMOT_QUANT_KIND_GENERIC && layout != MARMOT_QUANT_LAYOUT_GGUF) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Unsupported Metal quantization layout");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    if (kind == MARMOT_QUANT_KIND_GENERIC) {
        return metal_quantize_generic_impl(ctx, input, quant_params, output);
    }

    const metal_quant_scheme_entry_t *scheme = metal_find_quant_scheme(kind);
    if (scheme == nullptr || scheme->quantize == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal quantization scheme not available");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    return scheme->quantize(ctx, input, output);
}

marmot_error_t metal_dequantize_dispatch(
    const void *device_ctx, marmot_quant_kind_t kind, marmot_quant_layout_t layout, const marmot_tensor_t *input,
    marmot_tensor_t *output
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr || input == nullptr || output == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (kind != MARMOT_QUANT_KIND_GENERIC && layout != MARMOT_QUANT_LAYOUT_GGUF) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Unsupported Metal dequantization layout");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    if (kind == MARMOT_QUANT_KIND_GENERIC) {
        const marmot_quant_params_t *params = input->quant_params;
        if (params == nullptr) {
            params = output->quant_params;
        }
        return metal_dequantize_generic_impl(ctx, input, params, output);
    }

    const metal_quant_scheme_entry_t *scheme = metal_find_quant_scheme(kind);
    if (scheme == nullptr || scheme->dequantize == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal dequantization scheme not available");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    return scheme->dequantize(ctx, input, output);
}

#endif // __APPLE__
