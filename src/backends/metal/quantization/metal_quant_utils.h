#pragma once

#ifdef __APPLE__

#import <Metal/Metal.h>

#include "metal_backend_internal.h"

typedef struct {
    uint32_t num_rows;
    uint32_t row_size;
    uint32_t blocks_per_row;
    uint32_t num_blocks;
    size_t num_elements;
} metal_quant_row_config_t;

typedef struct {
    id<MTLBuffer> input;
    id<MTLBuffer> output;
    bool output_private;
} metal_quant_buffer_bundle_t;

void metal_quant_compute_row_config(const marmot_tensor_t *tensor, uint32_t block_size, metal_quant_row_config_t *out);

marmot_error_t metal_quant_acquire_buffers(
    metal_context_t *ctx, const marmot_tensor_t *input, size_t input_bytes, const marmot_tensor_t *output,
    size_t output_bytes, metal_quant_buffer_bundle_t *bundle
);

void metal_quant_release_buffers(
    metal_context_t *ctx, const marmot_tensor_t *output, metal_quant_buffer_bundle_t *bundle
);

#endif // __APPLE__
