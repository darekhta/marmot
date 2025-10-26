#include "metal_quant_schemes.h"

#ifdef __APPLE__

#include "marmot/quant_block.h"

#include "metal_quant_utils.h"

static marmot_error_t metal_quantize_block32_impl(
    metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output, const char *kernel_name,
    size_t block_bytes
) {
    metal_quant_row_config_t cfg;
    metal_quant_compute_row_config(input, MARMOT_QUANT_BLOCK_SIZE, &cfg);
    const size_t expected_output_size = (size_t)cfg.num_blocks * block_bytes;
    const size_t input_bytes = cfg.num_elements * sizeof(float);

    metal_quant_buffer_bundle_t buffers = {};
    marmot_error_t status =
        metal_quant_acquire_buffers(ctx, input, input_bytes, output, expected_output_size, &buffers);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        metal_quant_release_buffers(ctx, output, &buffers);
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        metal_quant_release_buffers(ctx, output, &buffers);
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    [encoder setBuffer:buffers.input offset:0 atIndex:0];
    [encoder setBuffer:buffers.output offset:0 atIndex:1];
    [encoder setBytes:&cfg.num_rows length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&cfg.row_size length:sizeof(uint32_t) atIndex:3];

    MTLSize tpg = metal_threads_for_elements(pipeline, cfg.num_blocks, 256);
    [encoder dispatchThreads:MTLSizeMake(cfg.num_blocks, 1, 1) threadsPerThreadgroup:tpg];

    metal_command_stream_flush(ctx, false);

    [pipeline release];
    metal_quant_release_buffers(ctx, output, &buffers);
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_dequantize_block32_impl(
    metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output, const char *kernel_name,
    size_t block_bytes
) {
    metal_quant_row_config_t cfg;
    metal_quant_compute_row_config(output, MARMOT_QUANT_BLOCK_SIZE, &cfg);
    const size_t expected_input_size = (size_t)cfg.num_blocks * block_bytes;
    const size_t output_bytes = cfg.num_elements * sizeof(float);

    metal_quant_buffer_bundle_t buffers = {};
    marmot_error_t status =
        metal_quant_acquire_buffers(ctx, input, expected_input_size, output, output_bytes, &buffers);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        metal_quant_release_buffers(ctx, output, &buffers);
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        metal_quant_release_buffers(ctx, output, &buffers);
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    [encoder setBuffer:buffers.input offset:0 atIndex:0];
    [encoder setBuffer:buffers.output offset:0 atIndex:1];
    [encoder setBytes:&cfg.num_rows length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&cfg.row_size length:sizeof(uint32_t) atIndex:3];

    MTLSize tpg = metal_threads_for_elements(pipeline, cfg.num_blocks, 256);
    [encoder dispatchThreads:MTLSizeMake(cfg.num_blocks, 1, 1) threadsPerThreadgroup:tpg];

    metal_command_stream_flush(ctx, false);

    [pipeline release];
    metal_quant_release_buffers(ctx, output, &buffers);
    return MARMOT_SUCCESS;
}

// Q4_0, Q4_1
marmot_error_t metal_quantize_q4_0_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    return metal_quantize_block32_impl(ctx, input, output, "quantize_q4_0", sizeof(marmot_q4_0_block_t));
}

marmot_error_t metal_dequantize_q4_0_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    return metal_dequantize_block32_impl(ctx, input, output, "dequantize_q4_0", sizeof(marmot_q4_0_block_t));
}

marmot_error_t metal_quantize_q4_1_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    return metal_quantize_block32_impl(ctx, input, output, "quantize_q4_1", sizeof(marmot_q4_1_block_t));
}

marmot_error_t metal_dequantize_q4_1_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    return metal_dequantize_block32_impl(ctx, input, output, "dequantize_q4_1", sizeof(marmot_q4_1_block_t));
}

// Q5_0, Q5_1
marmot_error_t metal_quantize_q5_0_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    return metal_quantize_block32_impl(ctx, input, output, "quantize_q5_0", sizeof(marmot_q5_0_block_t));
}

marmot_error_t metal_dequantize_q5_0_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    return metal_dequantize_block32_impl(ctx, input, output, "dequantize_q5_0", sizeof(marmot_q5_0_block_t));
}

marmot_error_t metal_quantize_q5_1_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    return metal_quantize_block32_impl(ctx, input, output, "quantize_q5_1", sizeof(marmot_q5_1_block_t));
}

marmot_error_t metal_dequantize_q5_1_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    return metal_dequantize_block32_impl(ctx, input, output, "dequantize_q5_1", sizeof(marmot_q5_1_block_t));
}

// Q8_0, Q8_1
marmot_error_t metal_quantize_q8_0_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    return metal_quantize_block32_impl(ctx, input, output, "quantize_q8_0", sizeof(marmot_q8_0_block_t));
}

marmot_error_t metal_dequantize_q8_0_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    return metal_dequantize_block32_impl(ctx, input, output, "dequantize_q8_0", sizeof(marmot_q8_0_block_t));
}

marmot_error_t metal_quantize_q8_1_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    return metal_quantize_block32_impl(ctx, input, output, "quantize_q8_1", sizeof(marmot_q8_1_block_t));
}

marmot_error_t metal_dequantize_q8_1_impl(metal_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    return metal_dequantize_block32_impl(ctx, input, output, "dequantize_q8_1", sizeof(marmot_q8_1_block_t));
}

#endif // __APPLE__
