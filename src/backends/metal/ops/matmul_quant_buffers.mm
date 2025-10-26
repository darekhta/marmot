#include "internal/metal_matmul_quant_buffers.h"
#include "metal_backend_internal.h"

#ifdef __APPLE__

void metal_matmul_quant_buffers_release(metal_matmul_quant_buffers_t *buffers) {
    if (buffers == nullptr) {
        return;
    }
    if (buffers->weight != nil) {
        [buffers->weight release];
        buffers->weight = nil;
    }
    if (buffers->input != nil) {
        [buffers->input release];
        buffers->input = nil;
    }
    if (buffers->out != nil) {
        [buffers->out release];
        buffers->out = nil;
    }
}

marmot_error_t metal_matmul_quant_buffers_acquire(
    metal_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, marmot_tensor_t *out,
    metal_matmul_quant_buffers_t *buffers
) {
    if (ctx == nullptr || weight == nullptr || out == nullptr || buffers == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    buffers->weight = metal_residency_acquire_existing(ctx, weight, weight->dtype);
    if (buffers->weight == nil) {
        buffers->weight = metal_residency_acquire_compute(ctx, weight, weight->dtype, nullptr);
    }
    if (buffers->weight == nil) {
        size_t weight_bytes = marmot_tensor_size_bytes(weight);
        buffers->weight = metal_buffer_acquire(ctx, weight->data, weight_bytes);
    }
    if (buffers->weight == nil) {
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
    }

    if (input != nullptr) {
        buffers->input = metal_residency_acquire_existing(ctx, input, input->dtype);
        if (buffers->input == nil) {
            buffers->input = metal_residency_acquire_compute(ctx, input, input->dtype, nullptr);
        }
        if (buffers->input == nil) {
            size_t input_bytes = marmot_tensor_size_bytes(input);
            buffers->input = metal_buffer_acquire(ctx, input->data, input_bytes);
        }
        if (buffers->input == nil) {
            metal_matmul_quant_buffers_release(buffers);
            return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
        }
    } else {
        buffers->input = nil;
    }

    bool out_is_staging = false;
    buffers->out = metal_residency_acquire_compute(ctx, out, out->dtype, &out_is_staging);
    if (buffers->out == nil) {
        size_t out_bytes = marmot_tensor_size_bytes(out);
        buffers->out = metal_buffer_acquire(ctx, out->data, out_bytes);
    }
    if (buffers->out == nil) {
        metal_matmul_quant_buffers_release(buffers);
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
    }
    return MARMOT_SUCCESS;
}

id<MTLBuffer> metal_matmul_quant_new_activation_buffer(metal_context_t *ctx, size_t length) {
    if (ctx == nullptr || length == 0) {
        return nil;
    }
    return [ctx->device newBufferWithLength:length options:MTLResourceStorageModeShared];
}

NSUInteger metal_matmul_quant_clamp_threads(NSUInteger value, NSUInteger maximum) {
    if (maximum == 0) {
        return value;
    }
    return value < maximum ? value : maximum;
}

MTLSize metal_matmul_quant_threadgroup_size(id<MTLComputePipelineState> pipeline) {
    if (pipeline == nil) {
        return MTLSizeMake(1, 1, 1);
    }
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    const NSUInteger threads_x = metal_matmul_quant_clamp_threads((NSUInteger)16, max_threads > 0 ? max_threads : 1);
    NSUInteger threads_y = (threads_x == 0) ? 1 : max_threads / (threads_x == 0 ? 1 : threads_x);
    if (threads_y == 0) {
        threads_y = 1;
    }
    if (threads_y > 16) {
        threads_y = 16;
    }
    return MTLSizeMake(threads_x == 0 ? 1 : threads_x, threads_y, 1);
}

#endif // __APPLE__
