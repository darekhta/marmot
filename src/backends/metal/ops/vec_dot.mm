#include "metal_backend_internal.h"

#ifdef __APPLE__

#include "marmot/quant_block.h"

#include <stdint.h>
#include <stdlib.h>

#include <limits.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    marmot_quant_kind_t weight_kind;
    marmot_quant_kind_t activation_kind;
    size_t weight_block_bytes;
    size_t activation_block_bytes;
    const char *kernel_name;
} metal_vecdot_traits_t;

static const metal_vecdot_traits_t k_vecdot_traits[] = {
    {MARMOT_QUANT_KIND_Q4_0, MARMOT_QUANT_KIND_Q8_0, sizeof(marmot_q4_0_block_t), sizeof(marmot_q8_0_block_t),
     "vec_dot_q4_0_q8_0"},
    {MARMOT_QUANT_KIND_Q4_1, MARMOT_QUANT_KIND_Q8_0, sizeof(marmot_q4_1_block_t), sizeof(marmot_q8_0_block_t),
     "vec_dot_q4_1_q8_0"},
    {MARMOT_QUANT_KIND_Q5_0, MARMOT_QUANT_KIND_Q8_0, sizeof(marmot_q5_0_block_t), sizeof(marmot_q8_0_block_t),
     "vec_dot_q5_0_q8_0"},
    {MARMOT_QUANT_KIND_Q5_1, MARMOT_QUANT_KIND_Q8_0, sizeof(marmot_q5_1_block_t), sizeof(marmot_q8_0_block_t),
     "vec_dot_q5_1_q8_0"},
    {MARMOT_QUANT_KIND_Q8_0, MARMOT_QUANT_KIND_Q8_0, sizeof(marmot_q8_0_block_t), sizeof(marmot_q8_0_block_t),
     "vec_dot_q8_0_q8_0"},
    {MARMOT_QUANT_KIND_Q8_1, MARMOT_QUANT_KIND_Q8_0, sizeof(marmot_q8_1_block_t), sizeof(marmot_q8_0_block_t),
     "vec_dot_q8_1_q8_0"},
    {MARMOT_QUANT_KIND_Q2_K, MARMOT_QUANT_KIND_Q8_K, sizeof(marmot_q2_k_block_t), sizeof(marmot_q8_k_block_t),
     "vec_dot_q2_k_q8_k"},
    {MARMOT_QUANT_KIND_Q3_K, MARMOT_QUANT_KIND_Q8_K, sizeof(marmot_q3_k_block_t), sizeof(marmot_q8_k_block_t),
     "vec_dot_q3_k_q8_k"},
    {MARMOT_QUANT_KIND_Q4_K, MARMOT_QUANT_KIND_Q8_K, sizeof(marmot_q4_k_block_t), sizeof(marmot_q8_k_block_t),
     "vec_dot_q4_k_q8_k"},
    {MARMOT_QUANT_KIND_Q5_K, MARMOT_QUANT_KIND_Q8_K, sizeof(marmot_q5_k_block_t), sizeof(marmot_q8_k_block_t),
     "vec_dot_q5_k_q8_k"},
    {MARMOT_QUANT_KIND_Q6_K, MARMOT_QUANT_KIND_Q8_K, sizeof(marmot_q6_k_block_t), sizeof(marmot_q8_k_block_t),
     "vec_dot_q6_k_q8_k"},
    {MARMOT_QUANT_KIND_Q8_K, MARMOT_QUANT_KIND_Q8_K, sizeof(marmot_q8_k_block_t), sizeof(marmot_q8_k_block_t),
     "vec_dot_q8_k_q8_k"},
};

static const metal_vecdot_traits_t *
metal_vecdot_find_traits(marmot_quant_kind_t weight_kind, marmot_quant_kind_t activation_kind) {
    for (size_t i = 0; i < sizeof(k_vecdot_traits) / sizeof(k_vecdot_traits[0]); ++i) {
        if (k_vecdot_traits[i].weight_kind == weight_kind && k_vecdot_traits[i].activation_kind == activation_kind) {
            return &k_vecdot_traits[i];
        }
    }
    return nullptr;
}

static size_t bytes_for_weights(
    marmot_quant_kind_t weight_kind, marmot_quant_kind_t activation_kind, size_t num_blocks, bool *out_invalid
) {
    const metal_vecdot_traits_t *traits = metal_vecdot_find_traits(weight_kind, activation_kind);
    if (traits == nullptr) {
        if (out_invalid != nullptr) {
            *out_invalid = true;
        }
        return 0;
    }
    if (num_blocks > SIZE_MAX / traits->weight_block_bytes) {
        if (out_invalid != nullptr) {
            *out_invalid = true;
        }
        return 0;
    }
    if (out_invalid != nullptr) {
        *out_invalid = false;
    }
    return num_blocks * traits->weight_block_bytes;
}

static size_t bytes_for_activations(
    marmot_quant_kind_t weight_kind, marmot_quant_kind_t activation_kind, size_t num_blocks, bool *out_invalid
) {
    const metal_vecdot_traits_t *traits = metal_vecdot_find_traits(weight_kind, activation_kind);
    if (traits == nullptr) {
        if (out_invalid != nullptr) {
            *out_invalid = true;
        }
        return 0;
    }
    if (num_blocks > SIZE_MAX / traits->activation_block_bytes) {
        if (out_invalid != nullptr) {
            *out_invalid = true;
        }
        return 0;
    }
    if (out_invalid != nullptr) {
        *out_invalid = false;
    }
    return num_blocks * traits->activation_block_bytes;
}

static const char *kernel_name_for_vec_dot(marmot_quant_kind_t weight_kind, marmot_quant_kind_t activation_kind) {
    const metal_vecdot_traits_t *traits = metal_vecdot_find_traits(weight_kind, activation_kind);
    return traits != nullptr ? traits->kernel_name : nullptr;
}

static bool metal_nocopy_ok(const void *ptr) {
    if (ptr == nullptr) {
        return false;
    }
    const size_t page_size = (size_t)getpagesize();
    if (page_size == 0) {
        return false;
    }
    return ((uintptr_t)ptr % page_size) == 0;
}

static marmot_error_t metal_vec_dot_launch(
    metal_context_t *ctx, const char *kernel_name, const void *weights, size_t weights_bytes, const void *activations,
    size_t activations_bytes, size_t num_blocks, float *result
) {
    if (kernel_name == nullptr || weights == nullptr || activations == nullptr || result == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (weights_bytes == 0 || activations_bytes == 0) {
        *result = 0.0f;
        return MARMOT_SUCCESS;
    }
    if (num_blocks > UINT_MAX) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Vec dot block count exceeds Metal kernel limits");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    id<MTLBuffer> bufferWeights = nil;
    if (metal_nocopy_ok(weights)) {
        bufferWeights = [ctx->device newBufferWithBytesNoCopy:(void *)weights
                                                       length:weights_bytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
    }
    if (bufferWeights == nil) {
        bufferWeights = [ctx->device newBufferWithBytes:weights
                                                 length:weights_bytes
                                                options:MTLResourceStorageModeShared];
    }

    id<MTLBuffer> bufferActivations = nil;
    if (metal_nocopy_ok(activations)) {
        bufferActivations = [ctx->device newBufferWithBytesNoCopy:(void *)activations
                                                           length:activations_bytes
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
    }
    if (bufferActivations == nil) {
        bufferActivations = [ctx->device newBufferWithBytes:activations
                                                     length:activations_bytes
                                                    options:MTLResourceStorageModeShared];
    }
    size_t partial_bytes = num_blocks * sizeof(float);
    float *partials_host = nullptr;
    id<MTLBuffer> bufferPartials = nil;
    if (partial_bytes > 0) {
        partials_host = (float *)malloc(partial_bytes);
        if (partials_host != nullptr) {
            memset(partials_host, 0, partial_bytes);
            bufferPartials = [ctx->device newBufferWithBytesNoCopy:partials_host
                                                            length:partial_bytes
                                                           options:MTLResourceStorageModeShared
                                                       deallocator:nil];
        }
    }
    if (bufferWeights == nil || bufferActivations == nil ||
        (partial_bytes > 0 && (partials_host == nullptr || bufferPartials == nil))) {
        if (bufferWeights != nil)
            [bufferWeights release];
        if (bufferActivations != nil)
            [bufferActivations release];
        if (bufferPartials != nil)
            [bufferPartials release];
        free(partials_host);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        [bufferWeights release];
        [bufferActivations release];
        if (bufferPartials != nil)
            [bufferPartials release];
        free(partials_host);
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferWeights release];
        [bufferActivations release];
        if (bufferPartials != nil)
            [bufferPartials release];
        free(partials_host);
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    [encoder setBuffer:bufferWeights offset:0 atIndex:0];
    [encoder setBuffer:bufferActivations offset:0 atIndex:1];
    [encoder setBuffer:bufferPartials offset:0 atIndex:2];

    uint32_t block_count = (uint32_t)num_blocks;
    [encoder setBytes:&block_count length:sizeof(uint32_t) atIndex:3];

    const uint32_t threads_per_block = 32u;
    NSUInteger total_threads = (NSUInteger)(block_count * threads_per_block);
    if (total_threads > 0) {
        MTLSize grid = MTLSizeMake(total_threads, 1, 1);
        MTLSize threadgroup = MTLSizeMake(threads_per_block, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
    }

    metal_command_stream_flush(ctx, true);

    float value = 0.0f;
    if (partials_host != nullptr) {
        for (size_t i = 0; i < num_blocks; ++i) {
            value += partials_host[i];
        }
    }
    *result = value;

    [pipeline release];
    [bufferWeights release];
    [bufferActivations release];
    if (bufferPartials != nil)
        [bufferPartials release];
    free(partials_host);

    return MARMOT_SUCCESS;
}

marmot_error_t metal_vec_dot(const void *device_ctx, const marmot_vec_dot_descriptor_t *desc, float *result) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr || desc == nullptr || result == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (desc->num_blocks == 0) {
        *result = 0.0f;
        return MARMOT_SUCCESS;
    }

    if (desc->layout != MARMOT_QUANT_LAYOUT_GGUF) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal vec dot currently supports GGUF layout only");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    bool overflow = false;
    const size_t weights_bytes =
        bytes_for_weights(desc->weight_kind, desc->activation_kind, desc->num_blocks, &overflow);
    if (overflow) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal vec dot weight size overflow");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const size_t activations_bytes =
        bytes_for_activations(desc->weight_kind, desc->activation_kind, desc->num_blocks, &overflow);
    if (overflow) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal vec dot activation size overflow");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (weights_bytes == 0 || activations_bytes == 0) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Unsupported vec dot quant combination on Metal backend");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const char *kernel_name = kernel_name_for_vec_dot(desc->weight_kind, desc->activation_kind);
    if (kernel_name == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Vec dot combination not yet supported on Metal backend");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    return metal_vec_dot_launch(
        ctx, kernel_name, desc->weights, weights_bytes, desc->activations, activations_bytes, desc->num_blocks, result
    );
}

#endif // __APPLE__
