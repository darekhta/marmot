#ifndef METAL_MATMUL_QUANT_BUFFERS_H
#define METAL_MATMUL_QUANT_BUFFERS_H

#include "marmot/ops/matmul.h"

#include "metal_backend_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct metal_matmul_quant_buffers {
    id<MTLBuffer> weight;
    id<MTLBuffer> input;
    id<MTLBuffer> out;
    size_t weight_offset;
    size_t input_offset;
    size_t out_offset;
    bool out_private;
} metal_matmul_quant_buffers_t;

void metal_matmul_quant_buffers_release(metal_matmul_quant_buffers_t *buffers);

marmot_error_t metal_matmul_quant_buffers_acquire(
    metal_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, marmot_tensor_t *out,
    metal_matmul_quant_buffers_t *buffers
);

id<MTLBuffer> metal_matmul_quant_new_activation_buffer(metal_context_t *ctx, size_t length);

MTLSize metal_matmul_quant_threadgroup_size(id<MTLComputePipelineState> pipeline);

NSUInteger metal_matmul_quant_clamp_threads(NSUInteger value, NSUInteger maximum);

#ifdef __cplusplus
}
#endif

#endif // METAL_MATMUL_QUANT_BUFFERS_H
