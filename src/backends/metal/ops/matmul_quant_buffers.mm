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

    buffers->weight_offset = 0;
    buffers->input_offset = 0;
    buffers->out_offset = 0;
    buffers->out_private = false;

    const size_t weight_bytes = marmot_tensor_quant_storage_bytes(weight);
    const size_t weight_span = weight_bytes != 0 ? weight_bytes : marmot_tensor_size_bytes(weight);
    metal_tensor_buffer_t weight_view = metal_buffer_acquire_view(ctx, weight, weight->dtype, weight_span);
    buffers->weight = weight_view.buffer;
    buffers->weight_offset = weight_view.offset;
    if (buffers->weight == nil) {
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
    }

    if (input != nullptr) {
        const size_t input_bytes = marmot_tensor_size_bytes(input);
        metal_tensor_buffer_t input_view = metal_buffer_acquire_view(ctx, input, input->dtype, input_bytes);
        buffers->input = input_view.buffer;
        buffers->input_offset = input_view.offset;
        if (buffers->input == nil) {
            metal_matmul_quant_buffers_release(buffers);
            return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
        }
    } else {
        buffers->input = nil;
    }

    const size_t out_bytes = marmot_tensor_size_bytes(out);
    metal_tensor_buffer_t out_view = metal_buffer_acquire_view(ctx, out, out->dtype, out_bytes);
    buffers->out = out_view.buffer;
    buffers->out_offset = out_view.offset;
    buffers->out_private = out_view.is_private;
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
