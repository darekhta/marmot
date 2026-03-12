#include "marmot/quant_block.h"

#include "core/tensor/tensor_utils.h"
#include "internal/metal_matmul_internal.h"
#include "internal/metal_matmul_qkv_shared.h"
#include "internal/metal_matmul_quant_dispatch.h"

// QKV quantized kernel info - returned by direct dispatch
typedef struct {
    const char *pack_kernel_fp32;
    const char *pack_kernel_fp16;
    const char *fused_kernel_fp32;
    const char *fused_kernel_fp32_out16;
    const char *fused_kernel_fp16_out32;
    const char *fused_kernel_fp16_out16;
    const char *dual_kernel_fp32;
    const char *dual_kernel_fp32_out16;
    const char *dual_kernel_fp16_out32;
    const char *dual_kernel_fp16_out16;
} metal_matmul_qkv_quant_kernels_t;

static metal_matmul_qkv_quant_kernels_t metal_matmul_qkv_quant_kernels(
    const char *pack_kernel_fp32, const char *pack_kernel_fp16, const char *fused_kernel_fp32,
    const char *fused_kernel_fp32_out16, const char *fused_kernel_fp16_out32, const char *fused_kernel_fp16_out16,
    const char *dual_kernel_fp32, const char *dual_kernel_fp32_out16, const char *dual_kernel_fp16_out32,
    const char *dual_kernel_fp16_out16
) {
    metal_matmul_qkv_quant_kernels_t kernels = {};
    kernels.pack_kernel_fp32 = pack_kernel_fp32;
    kernels.pack_kernel_fp16 = pack_kernel_fp16;
    kernels.fused_kernel_fp32 = fused_kernel_fp32;
    kernels.fused_kernel_fp32_out16 = fused_kernel_fp32_out16;
    kernels.fused_kernel_fp16_out32 = fused_kernel_fp16_out32;
    kernels.fused_kernel_fp16_out16 = fused_kernel_fp16_out16;
    kernels.dual_kernel_fp32 = dual_kernel_fp32;
    kernels.dual_kernel_fp32_out16 = dual_kernel_fp32_out16;
    kernels.dual_kernel_fp16_out32 = dual_kernel_fp16_out32;
    kernels.dual_kernel_fp16_out16 = dual_kernel_fp16_out16;
    return kernels;
}

// Direct dispatch for QKV quantized matmul - no traits lookup
static bool metal_matmul_qkv_quant_get_kernels(marmot_quant_kind_t quant_kind, metal_matmul_qkv_quant_kernels_t *out) {
    if (out == nullptr) {
        return false;
    }

    // Q8_0 activation packing kernels (for non-K quants)
    static const char *pack_q8_0_fp32 = "quantize_activations_column_q8_0";
    static const char *pack_q8_0_fp16 = "quantize_activations_column_q8_0_from_f16";

    switch (quant_kind) {
    case MARMOT_QUANT_KIND_Q4_0:
        *out = metal_matmul_qkv_quant_kernels(
            pack_q8_0_fp32, pack_q8_0_fp16, "matmul_qkv_q4_0_q8_0_f32", "matmul_qkv_q4_0_q8_0_f16",
            "matmul_qkv_q4_0_q8_0_f32", "matmul_qkv_q4_0_q8_0_f16", nullptr, nullptr, nullptr, nullptr
        );
        return true;
    case MARMOT_QUANT_KIND_Q4_1:
        *out = metal_matmul_qkv_quant_kernels(
            pack_q8_0_fp32, pack_q8_0_fp16, "matmul_qkv_q4_1_q8_0_f32", "matmul_qkv_q4_1_q8_0_f16",
            "matmul_qkv_q4_1_q8_0_f32", "matmul_qkv_q4_1_q8_0_f16", nullptr, nullptr, nullptr, nullptr
        );
        return true;
    case MARMOT_QUANT_KIND_Q5_0:
        *out = metal_matmul_qkv_quant_kernels(
            pack_q8_0_fp32, pack_q8_0_fp16, "matmul_qkv_q5_0_q8_0_f32", "matmul_qkv_q5_0_q8_0_f16",
            "matmul_qkv_q5_0_q8_0_f32", "matmul_qkv_q5_0_q8_0_f16", nullptr, nullptr, nullptr, nullptr
        );
        return true;
    case MARMOT_QUANT_KIND_Q5_1:
        *out = metal_matmul_qkv_quant_kernels(
            pack_q8_0_fp32, pack_q8_0_fp16, "matmul_qkv_q5_1_q8_0_f32", "matmul_qkv_q5_1_q8_0_f16",
            "matmul_qkv_q5_1_q8_0_f32", "matmul_qkv_q5_1_q8_0_f16", nullptr, nullptr, nullptr, nullptr
        );
        return true;
    case MARMOT_QUANT_KIND_Q8_0:
        *out = metal_matmul_qkv_quant_kernels(
            pack_q8_0_fp32, pack_q8_0_fp16, "matmul_qkv_q8_0_q8_0_f32", "matmul_qkv_q8_0_q8_0_f16",
            "matmul_qkv_q8_0_q8_0_f32", "matmul_qkv_q8_0_q8_0_f16", nullptr, nullptr, nullptr, nullptr
        );
        return true;
    case MARMOT_QUANT_KIND_Q8_1:
        *out = metal_matmul_qkv_quant_kernels(
            pack_q8_0_fp32, pack_q8_0_fp16, "matmul_qkv_q8_1_q8_0_f32", "matmul_qkv_q8_1_q8_0_f16",
            "matmul_qkv_q8_1_q8_0_f32", "matmul_qkv_q8_1_q8_0_f16", nullptr, nullptr, nullptr, nullptr
        );
        return true;
    case MARMOT_QUANT_KIND_Q2_K:
        *out = metal_matmul_qkv_quant_kernels(
            nullptr, nullptr, "matmul_qkv_q2_k_f32_f32", "matmul_qkv_q2_k_f32_f16", "matmul_qkv_q2_k_f16_f32",
            "matmul_qkv_q2_k_f16_f16", "matmul_qkv_q2_k_dual_f32_f32", "matmul_qkv_q2_k_dual_f32_f16",
            "matmul_qkv_q2_k_dual_f16_f32", "matmul_qkv_q2_k_dual_f16_f16"
        );
        return true;
    case MARMOT_QUANT_KIND_Q3_K:
        *out = metal_matmul_qkv_quant_kernels(
            nullptr, nullptr, "matmul_qkv_q3_k_f32_f32", "matmul_qkv_q3_k_f32_f16", "matmul_qkv_q3_k_f16_f32",
            "matmul_qkv_q3_k_f16_f16", "matmul_qkv_q3_k_dual_f32_f32", "matmul_qkv_q3_k_dual_f32_f16",
            "matmul_qkv_q3_k_dual_f16_f32", "matmul_qkv_q3_k_dual_f16_f16"
        );
        return true;
    case MARMOT_QUANT_KIND_Q4_K:
        *out = metal_matmul_qkv_quant_kernels(
            nullptr, nullptr, "matmul_qkv_q4_k_f32_f32", "matmul_qkv_q4_k_f32_f16", "matmul_qkv_q4_k_f16_f32",
            "matmul_qkv_q4_k_f16_f16", "matmul_qkv_q4_k_dual_f32_f32", "matmul_qkv_q4_k_dual_f32_f16",
            "matmul_qkv_q4_k_dual_f16_f32", "matmul_qkv_q4_k_dual_f16_f16"
        );
        return true;
    case MARMOT_QUANT_KIND_Q5_K:
        *out = metal_matmul_qkv_quant_kernels(
            nullptr, nullptr, "matmul_qkv_q5_k_f32_f32", "matmul_qkv_q5_k_f32_f16", "matmul_qkv_q5_k_f16_f32",
            "matmul_qkv_q5_k_f16_f16", "matmul_qkv_q5_k_dual_f32_f32", "matmul_qkv_q5_k_dual_f32_f16",
            "matmul_qkv_q5_k_dual_f16_f32", "matmul_qkv_q5_k_dual_f16_f16"
        );
        return true;
    case MARMOT_QUANT_KIND_Q6_K:
        *out = metal_matmul_qkv_quant_kernels(
            nullptr, nullptr, "matmul_qkv_q6_k_f32_f32", "matmul_qkv_q6_k_f32_f16", "matmul_qkv_q6_k_f16_f32",
            "matmul_qkv_q6_k_f16_f16", "matmul_qkv_q6_k_dual_f32_f32", "matmul_qkv_q6_k_dual_f32_f16",
            "matmul_qkv_q6_k_dual_f16_f32", "matmul_qkv_q6_k_dual_f16_f16"
        );
        return true;
    case MARMOT_QUANT_KIND_Q8_K:
        *out = metal_matmul_qkv_quant_kernels(
            nullptr, nullptr, "matmul_qkv_q8_k_f32_f32", "matmul_qkv_q8_k_f32_f16", "matmul_qkv_q8_k_f16_f32",
            "matmul_qkv_q8_k_f16_f16", "matmul_qkv_q8_k_dual_f32_f32", "matmul_qkv_q8_k_dual_f32_f16",
            "matmul_qkv_q8_k_dual_f16_f32", "matmul_qkv_q8_k_dual_f16_f16"
        );
        return true;
    default:
        return false;
    }
}

// Helper to select fused QKV kernel based on dtypes
static const char *metal_matmul_qkv_quant_select_fused_kernel(
    const metal_matmul_qkv_quant_kernels_t *kernels, marmot_dtype_t input_dtype, marmot_dtype_t out_dtype
) {
    if (input_dtype == MARMOT_DTYPE_FLOAT16) {
        return (out_dtype == MARMOT_DTYPE_FLOAT16) ? kernels->fused_kernel_fp16_out16
                                                   : kernels->fused_kernel_fp16_out32;
    }
    if (input_dtype == MARMOT_DTYPE_FLOAT32) {
        return (out_dtype == MARMOT_DTYPE_FLOAT16) ? kernels->fused_kernel_fp32_out16 : kernels->fused_kernel_fp32;
    }
    return nullptr;
}

static const char *metal_matmul_qkv_quant_select_dual_kernel(
    const metal_matmul_qkv_quant_kernels_t *kernels, marmot_dtype_t input_dtype, marmot_dtype_t out_dtype
) {
    if (input_dtype == MARMOT_DTYPE_FLOAT16) {
        return (out_dtype == MARMOT_DTYPE_FLOAT16) ? kernels->dual_kernel_fp16_out16 : kernels->dual_kernel_fp16_out32;
    }
    if (input_dtype == MARMOT_DTYPE_FLOAT32) {
        return (out_dtype == MARMOT_DTYPE_FLOAT16) ? kernels->dual_kernel_fp32_out16 : kernels->dual_kernel_fp32;
    }
    return nullptr;
}

static const char *metal_matmul_qkv_quant_select_dual_mv_kernel(
    marmot_quant_kind_t quant_kind, marmot_dtype_t input_dtype, marmot_dtype_t out_dtype
) {
    switch (quant_kind) {
    case MARMOT_QUANT_KIND_Q6_K:
        if (input_dtype == MARMOT_DTYPE_FLOAT16) {
            return out_dtype == MARMOT_DTYPE_FLOAT16 ? "matmul_qkv_q6_k_dual_f16_f16_mv"
                                                     : "matmul_qkv_q6_k_dual_f16_f32_mv";
        }
        if (input_dtype == MARMOT_DTYPE_FLOAT32) {
            return out_dtype == MARMOT_DTYPE_FLOAT16 ? nullptr : "matmul_qkv_q6_k_dual_f32_f32_mv";
        }
        return nullptr;
    default:
        return nullptr;
    }
}

typedef struct {
    uint32_t rope_enabled;
    uint32_t rope_apply_q;
    uint32_t rope_apply_k;
    uint32_t rope_head_dim;
    float rope_attn_scale;
} metal_matmul_qkv_quant_uniforms_t;

static NSUInteger metal_matmul_qkv_clamp_threads(NSUInteger value, NSUInteger maximum) {
    if (maximum == 0) {
        return value;
    }
    return value < maximum ? value : maximum;
}

static MTLSize metal_matmul_qkv_quant_threadgroup(id<MTLComputePipelineState> pipeline) {
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger threads_x = metal_matmul_qkv_clamp_threads((NSUInteger)16, max_threads > 0 ? max_threads : 1);
    if (threads_x == 0) {
        threads_x = 1;
    }
    if ((threads_x & 1u) != 0u) {
        if (threads_x > 1) {
            threads_x -= 1;
        } else if (max_threads >= 2) {
            threads_x = 2;
        }
    } else if (threads_x < 2 && max_threads >= 2) {
        threads_x = 2;
    }
    NSUInteger threads_y = max_threads / threads_x;
    if (threads_y == 0) {
        threads_y = 1;
    }
    if (threads_y > 16) {
        threads_y = 16;
    }
    return MTLSizeMake(threads_x, threads_y, 1);
}

static marmot_error_t metal_matmul_qkv_dispatch_quant_fused(
    metal_context_t *ctx, id<MTLComputePipelineState> matmul_pipeline, const char *kernel_name,
    const marmot_tensor_t *weights[3], marmot_tensor_t *outs[3], id<MTLBuffer> input_buffer, size_t N, size_t M,
    size_t K, uint32_t stride_n, uint32_t stride_k, const metal_matmul_qkv_epilogue_config_t ep_cfgs[3],
    id<MTLBuffer> rope_positions_buffer, id<MTLBuffer> rope_freqs_buffer, bool rope_apply_q, bool rope_apply_k,
    float rope_attn_scale, uint32_t rope_head_dim
) {
    id<MTLBuffer> weight_buffers[3] = {nil, nil, nil};
    id<MTLBuffer> out_buffers[3] = {nil, nil, nil};
    bool out_is_private[3] = {false, false, false};
    for (size_t i = 0; i < 3; ++i) {
        const marmot_tensor_t *weight_tensor = weights[i];
        if (weight_tensor == nullptr) {
            weight_buffers[i] = [metal_get_dummy_buffer(ctx) retain];
            continue;
        }
        weight_buffers[i] = metal_residency_acquire_existing(ctx, weight_tensor, weight_tensor->dtype);
        if (weight_buffers[i] == nil) {
            weight_buffers[i] = metal_residency_acquire_compute(ctx, weight_tensor, weight_tensor->dtype, nullptr);
        }
        if (weight_buffers[i] == nil) {
            size_t weight_bytes = marmot_tensor_size_bytes(weight_tensor);
            weight_buffers[i] = metal_buffer_acquire(ctx, weight_tensor->data, weight_bytes);
        }
        if (weight_buffers[i] == nil) {
            for (size_t j = 0; j <= i; ++j) {
                if (weight_buffers[j] != nil) {
                    [weight_buffers[j] release];
                }
            }
            return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
        }
    }

    for (size_t i = 0; i < 3; ++i) {
        if (outs[i] == nullptr) {
            out_buffers[i] = [metal_get_dummy_buffer(ctx) retain];
            continue;
        }
        bool is_private = false;
        out_buffers[i] = metal_residency_acquire_compute(ctx, outs[i], outs[i]->dtype, &is_private);
        out_is_private[i] = is_private;
        if (out_buffers[i] == nil) {
            size_t out_bytes = marmot_tensor_size_bytes(outs[i]);
            out_buffers[i] = metal_buffer_acquire(ctx, outs[i]->data, out_bytes);
        }
        if (out_buffers[i] == nil) {
            for (size_t j = 0; j < 3; ++j) {
                if (out_buffers[j] != nil) {
                    [out_buffers[j] release];
                }
                if (weight_buffers[j] != nil) {
                    [weight_buffers[j] release];
                }
            }
            return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
        }
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, matmul_pipeline);
    if (encoder == nil) {
        for (size_t i = 0; i < 3; ++i) {
            [out_buffers[i] release];
            [weight_buffers[i] release];
        }
        marmot_set_error(MARMOT_ERROR_DEVICE_NOT_AVAILABLE, "Failed to acquire Metal compute encoder");
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
    }

    [encoder setBuffer:weight_buffers[0] offset:0 atIndex:0];
    [encoder setBuffer:weight_buffers[1] offset:0 atIndex:1];
    [encoder setBuffer:weight_buffers[2] offset:0 atIndex:2];
    [encoder setBuffer:input_buffer offset:0 atIndex:3];
    [encoder setBuffer:out_buffers[0] offset:0 atIndex:4];
    [encoder setBuffer:out_buffers[1] offset:0 atIndex:5];
    [encoder setBuffer:out_buffers[2] offset:0 atIndex:6];

    const uint32_t N_u32 = (uint32_t)N;
    const uint32_t K_u32 = (uint32_t)K;
    const uint32_t M_u32 = (uint32_t)M;
    [encoder setBytes:&N_u32 length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&K_u32 length:sizeof(uint32_t) atIndex:8];
    [encoder setBytes:&M_u32 length:sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&stride_n length:sizeof(uint32_t) atIndex:10];
    [encoder setBytes:&stride_k length:sizeof(uint32_t) atIndex:11];
    if (rope_positions_buffer != nil && rope_freqs_buffer != nil) {
        [encoder setBuffer:rope_positions_buffer offset:0 atIndex:12];
        [encoder setBuffer:rope_freqs_buffer offset:0 atIndex:13];
    } else {
        [encoder setBuffer:metal_get_dummy_buffer(ctx) offset:0 atIndex:12];
        [encoder setBuffer:metal_get_dummy_buffer(ctx) offset:0 atIndex:13];
    }
    metal_matmul_qkv_quant_uniforms_t uniforms = {
        .rope_enabled = (uint32_t)((rope_positions_buffer != nil && rope_freqs_buffer != nil) ? 1u : 0u),
        .rope_apply_q = rope_apply_q ? 1u : 0u,
        .rope_apply_k = rope_apply_k ? 1u : 0u,
        .rope_head_dim = rope_head_dim,
        .rope_attn_scale = rope_attn_scale,
    };
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:14];

    const MTLSize grid = MTLSizeMake(M, N, 1);
    const MTLSize threads = metal_matmul_qkv_quant_threadgroup(matmul_pipeline);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];

    marmot_error_t status = MARMOT_SUCCESS;
    bool did_flush = false;
    bool needs_flush = false;
    for (size_t i = 0; i < 3; ++i) {
        if (outs[i] == nullptr) {
            continue;
        }
        const marmot_matmul_epilogue_t *ep = ep_cfgs[i].ep;
        if (ep != nullptr) {
            id<MTLBuffer> ep_buffer = [out_buffers[i] retain];
            status = metal_matmul_apply_epilogue(
                ctx, outs[i], ep_buffer, 0, N * M, ep_cfgs[i].feature_dim, ep_cfgs[i].bias_scalar, ep
            );
            [ep_buffer release];
            if (status != MARMOT_SUCCESS) {
                break;
            }
            did_flush = true;
        } else {
            metal_residency_mark_dirty(ctx, outs[i], outs[i]->dtype);
            needs_flush = true;
        }
    }
    if (status == MARMOT_SUCCESS && needs_flush && !did_flush) {
        metal_command_stream_flush(ctx, false);
    }

    for (size_t i = 0; i < 3; ++i) {
        [out_buffers[i] release];
        [weight_buffers[i] release];
        if (status != MARMOT_SUCCESS && out_is_private[i]) {
            metal_residency_mark_dirty(ctx, outs[i], outs[i]->dtype);
        }
    }
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_MATMUL, "matmul_quant", N * M, true, kernel_name);
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_matmul_qkv_dispatch_quant_dual_mv(
    metal_context_t *ctx, id<MTLComputePipelineState> matmul_pipeline, const char *kernel_name,
    const marmot_tensor_t *weight_q, const marmot_tensor_t *weight_k, const marmot_tensor_t *input,
    marmot_tensor_t *out_q, marmot_tensor_t *out_k, size_t N, size_t M, size_t K, size_t weight_blocks_per_row
) {
    const size_t weight_q_bytes = marmot_tensor_size_bytes(weight_q);
    const size_t weight_k_bytes = marmot_tensor_size_bytes(weight_k);
    const size_t input_bytes = marmot_tensor_size_bytes(input);
    const size_t out_q_bytes = marmot_tensor_size_bytes(out_q);
    const size_t out_k_bytes = marmot_tensor_size_bytes(out_k);

    metal_tensor_buffer_t weight_q_view = metal_buffer_acquire_view(ctx, weight_q, weight_q->dtype, weight_q_bytes);
    metal_tensor_buffer_t weight_k_view = metal_buffer_acquire_view(ctx, weight_k, weight_k->dtype, weight_k_bytes);
    metal_tensor_buffer_t input_view = metal_buffer_acquire_view(ctx, input, input->dtype, input_bytes);
    metal_tensor_buffer_t out_q_view = metal_buffer_acquire_view(ctx, out_q, out_q->dtype, out_q_bytes);
    metal_tensor_buffer_t out_k_view = metal_buffer_acquire_view(ctx, out_k, out_k->dtype, out_k_bytes);
    if (weight_q_view.buffer == nil || weight_k_view.buffer == nil || input_view.buffer == nil ||
        out_q_view.buffer == nil || out_k_view.buffer == nil) {
        if (weight_q_view.buffer != nil) {
            [weight_q_view.buffer release];
        }
        if (weight_k_view.buffer != nil) {
            [weight_k_view.buffer release];
        }
        if (input_view.buffer != nil) {
            [input_view.buffer release];
        }
        if (out_q_view.buffer != nil) {
            [out_q_view.buffer release];
        }
        if (out_k_view.buffer != nil) {
            [out_k_view.buffer release];
        }
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, matmul_pipeline);
    if (encoder == nil) {
        [weight_q_view.buffer release];
        [weight_k_view.buffer release];
        [input_view.buffer release];
        [out_q_view.buffer release];
        [out_k_view.buffer release];
        marmot_set_error(MARMOT_ERROR_DEVICE_NOT_AVAILABLE, "Failed to acquire Metal compute encoder");
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
    }

    const uint32_t N_u32 = (uint32_t)N;
    const uint32_t K_u32 = (uint32_t)K;
    const uint32_t M_u32 = (uint32_t)M;
    const uint32_t stride_n = (uint32_t)input->shape.strides[0];
    const uint32_t stride_k = (uint32_t)input->shape.strides[1];
    const uint32_t weight_blocks = (uint32_t)weight_blocks_per_row;

    [encoder setBuffer:weight_q_view.buffer offset:weight_q_view.offset atIndex:0];
    [encoder setBuffer:weight_k_view.buffer offset:weight_k_view.offset atIndex:1];
    [encoder setBuffer:metal_get_dummy_buffer(ctx) offset:0 atIndex:2];
    [encoder setBuffer:input_view.buffer offset:input_view.offset atIndex:3];
    [encoder setBuffer:out_q_view.buffer offset:out_q_view.offset atIndex:4];
    [encoder setBuffer:out_k_view.buffer offset:out_k_view.offset atIndex:5];
    [encoder setBuffer:metal_get_dummy_buffer(ctx) offset:0 atIndex:6];
    [encoder setBytes:&N_u32 length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&K_u32 length:sizeof(uint32_t) atIndex:8];
    [encoder setBytes:&M_u32 length:sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&stride_n length:sizeof(uint32_t) atIndex:10];
    [encoder setBytes:&stride_k length:sizeof(uint32_t) atIndex:11];
    [encoder setBytes:&weight_blocks length:sizeof(uint32_t) atIndex:12];

    metal_profiling_set_label(ctx, "matmul_quant_k");
    metal_profiling_begin(ctx);
    MTLSize threads = MTLSizeMake(64, 1, 1);
    MTLSize groups = MTLSizeMake((M + 3u) / 4u, N, 1);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    metal_profiling_end(ctx);
    metal_command_stream_flush(ctx, false);

    metal_residency_mark_dirty(ctx, out_q, out_q->dtype);
    metal_residency_mark_dirty(ctx, out_k, out_k->dtype);

    [weight_q_view.buffer release];
    [weight_k_view.buffer release];
    [input_view.buffer release];
    [out_q_view.buffer release];
    [out_k_view.buffer release];

    metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_MATMUL, "matmul_quant", N * M, true, kernel_name);
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_matmul_qkv_run_quantized_route_b(
    metal_context_t *ctx, const marmot_matmul_qkv_desc_t *desc, const metal_matmul_qkv_dims_t *dims
) {
    const marmot_tensor_t *weights[3] = {desc->separate.wq, desc->separate.wk, desc->separate.wv};
    const marmot_tensor_t *biases[3] = {desc->separate.bq, desc->separate.bk, desc->separate.bv};
    marmot_tensor_t *outs[3] = {desc->out_q, desc->out_k, desc->out_v};
    if (outs[0]->dtype != outs[1]->dtype || outs[0]->dtype != outs[2]->dtype) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const marmot_dtype_t input_dtype = desc->input->dtype;
    const marmot_dtype_t out_dtype = outs[0]->dtype;
    if ((input_dtype != MARMOT_DTYPE_FLOAT32 && input_dtype != MARMOT_DTYPE_FLOAT16) ||
        (out_dtype != MARMOT_DTYPE_FLOAT32 && out_dtype != MARMOT_DTYPE_FLOAT16)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    // Direct kernel lookup - no traits indirection
    const marmot_quant_kind_t quant_kind = (marmot_quant_kind_t)weights[0]->quant_kind;
    metal_matmul_qkv_quant_kernels_t qkv_kernels;
    if (!metal_matmul_qkv_quant_get_kernels(quant_kind, &qkv_kernels)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const char *fused_kernel_name = metal_matmul_qkv_quant_select_fused_kernel(&qkv_kernels, input_dtype, out_dtype);
    if (fused_kernel_name == nullptr || qkv_kernels.pack_kernel_fp32 != nullptr ||
        qkv_kernels.pack_kernel_fp16 != nullptr) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const marmot_matmul_epilogue_t *base_ep = desc->epilogue;
    const marmot_rope_params_t *rope = desc->rope_params;
    const uint32_t stride_n = (uint32_t)desc->input->shape.strides[0];
    const uint32_t stride_k = (uint32_t)desc->input->shape.strides[1];

    marmot_error_t status = MARMOT_SUCCESS;
    id<MTLBuffer> input_buffer = nil;
    id<MTLBuffer> rope_positions_buffer = nil;
    id<MTLBuffer> rope_freqs_buffer = nil;
    float rope_attn_scale = 1.0f;
    const marmot_rope_params_t *active_rope = rope;
    bool apply_rope_q = false;
    bool apply_rope_k = false;
    bool rope_inline = false;
    bool inline_apply_q = false;
    bool inline_apply_k = false;
    uint32_t rope_head_dim = 0;
    const char *kernel_name_to_use = fused_kernel_name;
    id<MTLComputePipelineState> matmul_pipeline = nil;
    input_buffer = metal_residency_acquire_existing(ctx, desc->input, desc->input->dtype);
    if (input_buffer == nil) {
        input_buffer = metal_residency_acquire_compute(ctx, desc->input, desc->input->dtype, nullptr);
    }
    if (input_buffer == nil) {
        size_t bytes_input = marmot_tensor_size_bytes(desc->input);
        input_buffer = metal_buffer_acquire(ctx, desc->input->data, bytes_input);
    }
    if (input_buffer == nil) {
        status = MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
        goto cleanup;
    }

    metal_matmul_qkv_epilogue_config_t ep_cfgs[3];
    for (size_t slice = 0; slice < 3; ++slice) {
        const bool apply_q = (slice == 0);
        const bool apply_k = (slice == 1);
        ep_cfgs[slice] = metal_matmul_qkv_epilogue_config_t{};
        status = metal_matmul_qkv_prepare_epilogue(
            outs[slice], base_ep, biases[slice], rope, apply_q, apply_k, true, true, &ep_cfgs[slice]
        );
        if (status != MARMOT_SUCCESS) {
            goto cleanup;
        }
    }
    apply_rope_q = (ep_cfgs[0].rope != nullptr);
    apply_rope_k = (ep_cfgs[1].rope != nullptr);

    if (active_rope == nullptr && apply_rope_q) {
        active_rope = ep_cfgs[0].rope;
    } else if (active_rope == nullptr && apply_rope_k) {
        active_rope = ep_cfgs[1].rope;
    }

    rope_head_dim = metal_matmul_qkv_resolve_head_dim(dims->M, active_rope);
    rope_inline = false;
    inline_apply_q = rope_inline && apply_rope_q;
    inline_apply_k = rope_inline && apply_rope_k;
    matmul_pipeline = metal_pipeline_get(ctx, kernel_name_to_use);
    if (matmul_pipeline == nil) {
        status = MARMOT_ERROR_NOT_IMPLEMENTED;
        goto cleanup;
    }
    if ((inline_apply_q || inline_apply_k) && active_rope != nullptr && dims->M >= 2) {
        rope_positions_buffer = metal_matmul_create_positions_buffer(ctx, active_rope->positions, dims->N);
        if (rope_positions_buffer == nil) {
            status = MARMOT_ERROR_OUT_OF_MEMORY;
            goto cleanup_buffers;
        }
        rope_freqs_buffer = metal_matmul_prepare_freq_buffer(ctx, rope_head_dim, active_rope, &rope_attn_scale);
        if (rope_freqs_buffer == nil) {
            [rope_positions_buffer release];
            status = MARMOT_ERROR_OUT_OF_MEMORY;
            goto cleanup;
        }
    }
    status = metal_matmul_qkv_dispatch_quant_fused(
        ctx, matmul_pipeline, kernel_name_to_use, weights, outs, input_buffer, dims->N, dims->M, dims->K, stride_n,
        stride_k, ep_cfgs, rope_positions_buffer, rope_freqs_buffer, inline_apply_q, inline_apply_k, rope_attn_scale,
        rope_head_dim
    );
cleanup_buffers:
    if (rope_positions_buffer != nil) {
        [rope_positions_buffer release];
    }
    if (rope_freqs_buffer != nil) {
        [rope_freqs_buffer release];
    }
    if (status == MARMOT_SUCCESS && (apply_rope_q || apply_rope_k) && active_rope != nullptr) {
        status = metal_matmul_qkv_apply_rope_gpu(ctx, desc, dims);
    }

cleanup:
    if (input_buffer != nil) {
        [input_buffer release];
    }
    if (matmul_pipeline != nil) {
        [matmul_pipeline release];
    }
    return status;
}

marmot_error_t metal_matmul_qkv_run_quantized(
    metal_context_t *ctx, const marmot_matmul_qkv_desc_t *desc, const metal_matmul_qkv_dims_t *dims
) {
    if (ctx == nullptr || desc == nullptr || desc->layout != MARMOT_QKV_LAYOUT_SEPARATE) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_error_t shared_status = metal_matmul_qkv_run_quantized_route_b(ctx, desc, dims);
    if (shared_status == MARMOT_SUCCESS) {
        return shared_status;
    }
    if (shared_status != MARMOT_ERROR_NOT_IMPLEMENTED && shared_status != MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        return shared_status;
    }

    const marmot_tensor_t *weights[3] = {desc->separate.wq, desc->separate.wk, desc->separate.wv};
    const marmot_tensor_t *biases[3] = {desc->separate.bq, desc->separate.bk, desc->separate.bv};
    marmot_tensor_t *outs[3] = {desc->out_q, desc->out_k, desc->out_v};
    const marmot_matmul_epilogue_t *base_ep = desc->epilogue;
    const marmot_rope_params_t *rope = desc->rope_params;

    for (size_t slice = 0; slice < 3; ++slice) {
        const bool apply_q = (slice == 0);
        const bool apply_k = (slice == 1);
        metal_matmul_qkv_epilogue_config_t ep_cfg = {};
        marmot_error_t prep_status = metal_matmul_qkv_prepare_epilogue(
            outs[slice], base_ep, biases[slice], rope, apply_q, apply_k, false, false, &ep_cfg
        );
        if (prep_status != MARMOT_SUCCESS) {
            return prep_status;
        }
        marmot_error_t status = metal_matmul_quantized(ctx, desc->input, weights[slice], ep_cfg.ep, outs[slice]);
        if (status != MARMOT_SUCCESS) {
            return status;
        }
    }
    if (rope != nullptr && (rope->apply_to_q || rope->apply_to_k)) {
        marmot_error_t rope_status = metal_matmul_qkv_apply_rope_gpu(ctx, desc, dims);
        if (rope_status != MARMOT_SUCCESS) {
            return rope_status;
        }
    }
    return MARMOT_SUCCESS;
}

marmot_error_t metal_matmul_qkv_run_quantized_dual_output(
    metal_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight_q,
    const marmot_tensor_t *weight_k, marmot_tensor_t *out_q, marmot_tensor_t *out_k
) {
    if (ctx == nullptr || input == nullptr || weight_q == nullptr || weight_k == nullptr || out_q == nullptr ||
        out_k == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input->shape.ndim != 2 || weight_q->shape.ndim != 2 || weight_k->shape.ndim != 2 || out_q->shape.ndim != 2 ||
        out_k->shape.ndim != 2) {
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (!marmot_tensor_is_block_quantized_weight(weight_q) || !marmot_tensor_is_block_quantized_weight(weight_k)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (weight_q->quant_kind != weight_k->quant_kind || weight_q->quant_layout != weight_k->quant_layout ||
        weight_q->dtype != weight_k->dtype) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const size_t N = input->shape.shape[0];
    const size_t K = input->shape.shape[1];
    const size_t M = weight_q->shape.shape[0];
    if (weight_q->shape.shape[1] != K || weight_k->shape.shape[1] != K || weight_k->shape.shape[0] != M) {
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (weight_q->shape.strides[1] != 1 || weight_k->shape.strides[1] != 1) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (out_q->shape.shape[0] != N || out_q->shape.shape[1] != M || out_k->shape.shape[0] != N ||
        out_k->shape.shape[1] != M) {
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    const marmot_dtype_t input_dtype = input->dtype;
    const marmot_dtype_t out_dtype = out_q->dtype;
    if ((input_dtype != MARMOT_DTYPE_FLOAT32 && input_dtype != MARMOT_DTYPE_FLOAT16) ||
        (out_dtype != MARMOT_DTYPE_FLOAT32 && out_dtype != MARMOT_DTYPE_FLOAT16) || out_k->dtype != out_dtype) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    metal_matmul_qkv_quant_kernels_t qkv_kernels;
    if (!metal_matmul_qkv_quant_get_kernels((marmot_quant_kind_t)weight_q->quant_kind, &qkv_kernels)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (qkv_kernels.pack_kernel_fp32 != nullptr || qkv_kernels.pack_kernel_fp16 != nullptr) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const char *kernel_name = metal_matmul_qkv_quant_select_dual_kernel(&qkv_kernels, input_dtype, out_dtype);
    if (kernel_name == nullptr) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const marmot_quant_kind_t quant_kind = (marmot_quant_kind_t)weight_q->quant_kind;
    if (N < 4) {
        const char *dual_mv_kernel = metal_matmul_qkv_quant_select_dual_mv_kernel(quant_kind, input_dtype, out_dtype);
        if (dual_mv_kernel != nullptr) {
            id<MTLComputePipelineState> dual_mv_pipeline = metal_pipeline_get(ctx, dual_mv_kernel);
            if (dual_mv_pipeline != nil) {
                const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(quant_kind);
                if (traits != nullptr && traits->block_values != 0) {
                    const size_t weight_blocks_per_row = (K + traits->block_values - 1u) / traits->block_values;
                    marmot_error_t mv_status = metal_matmul_qkv_dispatch_quant_dual_mv(
                        ctx, dual_mv_pipeline, dual_mv_kernel, weight_q, weight_k, input, out_q, out_k, N, M, K,
                        weight_blocks_per_row
                    );
                    [dual_mv_pipeline release];
                    if (mv_status == MARMOT_SUCCESS) {
                        return mv_status;
                    }
                    if (mv_status != MARMOT_ERROR_NOT_IMPLEMENTED && mv_status != MARMOT_ERROR_UNSUPPORTED_DTYPE) {
                        return mv_status;
                    }
                } else {
                    [dual_mv_pipeline release];
                }
            }
        }
    }

    id<MTLComputePipelineState> matmul_pipeline = metal_pipeline_get(ctx, kernel_name);
    if (matmul_pipeline == nil) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    id<MTLBuffer> input_buffer = metal_residency_acquire_existing(ctx, input, input->dtype);
    if (input_buffer == nil) {
        input_buffer = metal_residency_acquire_compute(ctx, input, input->dtype, nullptr);
    }
    if (input_buffer == nil) {
        input_buffer = metal_buffer_acquire(ctx, input->data, marmot_tensor_size_bytes(input));
    }
    if (input_buffer == nil) {
        [matmul_pipeline release];
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
    }

    const marmot_tensor_t *weights[3] = {weight_q, weight_k, nullptr};
    marmot_tensor_t *outs[3] = {out_q, out_k, nullptr};
    metal_matmul_qkv_epilogue_config_t ep_cfgs[3] = {};
    const uint32_t stride_n = (uint32_t)input->shape.strides[0];
    const uint32_t stride_k = (uint32_t)input->shape.strides[1];
    marmot_error_t status = metal_matmul_qkv_dispatch_quant_fused(
        ctx, matmul_pipeline, kernel_name, weights, outs, input_buffer, N, M, K, stride_n, stride_k, ep_cfgs, nil, nil,
        false, false, 1.0f, 0
    );
    [input_buffer release];
    [matmul_pipeline release];
    return status;
}
