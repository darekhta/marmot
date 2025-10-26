#include "marmot/quant_block.h"

#include <stdio.h>

#include <string.h>

#include "core/helpers/quant.h"
#include "internal/metal_matmul_internal.h"
#include "internal/metal_matmul_quant_buffers.h"
#include "internal/metal_matmul_quant_dispatch.h"
#include "metal_backend_internal.h"

#ifdef __APPLE__

static const char *metal_matmul_quant_log_label(const marmot_tensor_t *out, const char *kernel_name) {
    (void)out;
    return kernel_name != nullptr ? kernel_name : "matmul_quant";
}

static bool metal_log_routes(void) {
    static bool initialized = false;
    static bool value = false;
    if (!initialized) {
        const char *env = getenv("MARMOT_METAL_LOG_MATMUL_QUANT");
        value = (env != nullptr && env[0] != '\0' && env[0] != '0');
        initialized = true;
    }
    return value;
}

static bool metal_force_mm(void) {
    static bool initialized = false;
    static bool value = false;
    if (!initialized) {
        const char *env = getenv("MARMOT_METAL_FORCE_MM");
        value = (env != nullptr && env[0] == '1');
        initialized = true;
    }
    return value;
}

static bool metal_force_mv(void) {
    static bool initialized = false;
    static bool value = false;
    if (!initialized) {
        const char *env = getenv("MARMOT_METAL_FORCE_MV");
        value = (env != nullptr && env[0] == '1');
        initialized = true;
    }
    return value;
}

static void
metal_matmul_quant_log(metal_context_t *ctx, const char *label, const marmot_tensor_t *out, const char *why) {
    const size_t problem_bytes = marmot_tensor_size_bytes(out);
    metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_MATMUL, label, problem_bytes, true, why);
    if (metal_log_routes()) {
        fprintf(stderr, "[marmot metal matmul_quant] %s: %s\n", label != nullptr ? label : "matmul_quant", why);
    }
}

static bool metal_kernel_has_suffix(const char *kernel_name, const char *suffix) {
    if (kernel_name == nullptr || suffix == nullptr) {
        return false;
    }
    const size_t klen = strlen(kernel_name);
    const size_t slen = strlen(suffix);
    if (klen < slen) {
        return false;
    }
    return memcmp(kernel_name + (klen - slen), suffix, slen) == 0;
}

static marmot_error_t metal_matmul_quant_dispatch_direct_impl(
    metal_context_t *ctx, id<MTLComputePipelineState> pipeline, const char *log_label, const char *kernel_name,
    const marmot_tensor_t *input, const marmot_tensor_t *weight, marmot_tensor_t *out, size_t N, size_t K, size_t M,
    size_t weight_blocks_per_row, const marmot_matmul_epilogue_t *epilogue, size_t ep_feature_dim, bool ep_bias_scalar,
    const marmot_rope_params_t *rope
) {
    metal_matmul_quant_buffers_t buffers = {.weight = nil, .input = nil, .out = nil};
    marmot_error_t status = metal_matmul_quant_buffers_acquire(ctx, input, weight, out, &buffers);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        metal_matmul_quant_buffers_release(&buffers);
        marmot_set_error(MARMOT_ERROR_DEVICE_NOT_AVAILABLE, "Failed to acquire Metal compute encoder");
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
    }

    const uint32_t N_u32 = (uint32_t)N;
    const uint32_t K_u32 = (uint32_t)K;
    const uint32_t M_u32 = (uint32_t)M;
    const uint32_t stride_n_u32 = (uint32_t)input->shape.strides[0];
    const uint32_t stride_k_u32 = (uint32_t)input->shape.strides[1];
    const uint32_t weight_blocks_u32 = (uint32_t)weight_blocks_per_row;

    [encoder setBuffer:buffers.weight offset:0 atIndex:0];
    [encoder setBuffer:buffers.input offset:0 atIndex:1];
    [encoder setBuffer:buffers.out offset:0 atIndex:2];
    [encoder setBytes:&N_u32 length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&K_u32 length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&M_u32 length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&stride_n_u32 length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&stride_k_u32 length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&weight_blocks_u32 length:sizeof(uint32_t) atIndex:8];

    metal_profiling_set_label(ctx, "matmul_quant");
    metal_profiling_begin(ctx);

    MTLSize grid = MTLSizeMake(M, N, 1);
    MTLSize threads = metal_matmul_quant_threadgroup_size(pipeline);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];

    metal_profiling_end(ctx);

    marmot_error_t ep_status = MARMOT_SUCCESS;
    if (rope != nullptr && rope->positions != nullptr && (rope->apply_to_q || rope->apply_to_k)) {
        ep_status = metal_rope(ctx, out, rope, out);
        if (ep_status != MARMOT_SUCCESS) {
            metal_matmul_quant_buffers_release(&buffers);
            metal_matmul_quant_log(ctx, log_label, out, metal_matmul_quant_log_label(out, kernel_name));
            return ep_status;
        }
    }
    if (epilogue != nullptr) {
        id<MTLBuffer> ep_buffer = [buffers.out retain];
        ep_status = metal_matmul_apply_epilogue(ctx, out, ep_buffer, N * M, ep_feature_dim, ep_bias_scalar, epilogue);
        [ep_buffer release];
    } else {
        metal_residency_mark_dirty(ctx, out, out->dtype);
        metal_command_stream_flush(ctx, false);
    }

    metal_matmul_quant_buffers_release(&buffers);
    char route_msg[96];
    (void)snprintf(
        route_msg, sizeof(route_msg), "direct %s N=%zu M=%zu K=%zu",
        kernel_name != nullptr ? kernel_name : "matmul_quant", N, M, K
    );
    metal_matmul_quant_log(ctx, log_label, out, route_msg);
    return ep_status;
}

marmot_error_t metal_matmul_quant_dispatch_direct(
    metal_context_t *ctx, const char *kernel_name, const char *log_label, const char *missing_kernel_msg,
    const marmot_tensor_t *input, const marmot_tensor_t *weight, marmot_tensor_t *out, size_t N, size_t K, size_t M,
    size_t weight_blocks_per_row, const marmot_matmul_epilogue_t *epilogue, size_t ep_feature_dim, bool ep_bias_scalar,
    const marmot_rope_params_t *rope
) {
    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        marmot_set_error(MARMOT_ERROR_DEVICE_NOT_AVAILABLE, missing_kernel_msg);
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
    }
    marmot_error_t status = metal_matmul_quant_dispatch_direct_impl(
        ctx, pipeline, log_label, kernel_name, input, weight, out, N, K, M, weight_blocks_per_row, epilogue,
        ep_feature_dim, ep_bias_scalar, rope
    );
    [pipeline release];
    return status;
}

static marmot_error_t metal_matmul_quant_dispatch_packed_impl(
    metal_context_t *ctx, id<MTLComputePipelineState> quant_pipeline, id<MTLComputePipelineState> matmul_pipeline,
    const char *log_label, const char *kernel_name, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    marmot_tensor_t *out, size_t N, size_t K, size_t M, size_t activation_blocks_per_row, size_t activation_block_bytes,
    size_t weight_blocks_per_row, bool uses_super_blocks, const marmot_matmul_epilogue_t *epilogue,
    size_t ep_feature_dim, bool ep_bias_scalar, const marmot_rope_params_t *rope
) {
    const char *route_tag = kernel_name;
    const size_t input_q8_bytes = N * activation_blocks_per_row * activation_block_bytes;
    id<MTLBuffer> input_q8_buffer = metal_matmul_quant_new_activation_buffer(ctx, input_q8_bytes);
    if (input_q8_buffer == nil) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate input quantization scratch buffer");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    metal_matmul_quant_buffers_t buffers = {.weight = nil, .input = nil, .out = nil};
    marmot_error_t status = metal_matmul_quant_buffers_acquire(ctx, input, weight, out, &buffers);
    if (status != MARMOT_SUCCESS) {
        [input_q8_buffer release];
        return status;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, quant_pipeline);
    if (encoder == nil) {
        metal_matmul_quant_buffers_release(&buffers);
        [input_q8_buffer release];
        marmot_set_error(MARMOT_ERROR_DEVICE_NOT_AVAILABLE, "Failed to acquire Metal compute encoder");
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
    }

    const uint32_t K_u32 = (uint32_t)K;
    const uint32_t N_u32 = (uint32_t)N;
    const uint32_t stride_k = (uint32_t)input->shape.strides[1];
    const uint32_t stride_n = (uint32_t)input->shape.strides[0];
    const uint32_t activation_blocks_u32 = (uint32_t)activation_blocks_per_row;

    [encoder setBuffer:buffers.input offset:0 atIndex:0];
    [encoder setBuffer:input_q8_buffer offset:0 atIndex:1];
    [encoder setBytes:&K_u32 length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&N_u32 length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&stride_k length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&stride_n length:sizeof(uint32_t) atIndex:5];
    MTLSize quant_grid = MTLSizeMake(activation_blocks_per_row, N_u32, 1);
    const NSUInteger q_threads = metal_matmul_quant_clamp_threads(
        (NSUInteger)activation_blocks_per_row, quant_pipeline.maxTotalThreadsPerThreadgroup
    );
    MTLSize quant_threads = MTLSizeMake(q_threads == 0 ? 1 : q_threads, 1, 1);
    [encoder dispatchThreads:quant_grid threadsPerThreadgroup:quant_threads];

    encoder = metal_command_acquire_compute_encoder(ctx, matmul_pipeline);
    if (encoder == nil) {
        metal_matmul_quant_buffers_release(&buffers);
        [input_q8_buffer release];
        marmot_set_error(MARMOT_ERROR_DEVICE_NOT_AVAILABLE, "Failed to acquire Metal compute encoder");
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
    }

    const uint32_t weight_blocks_u32 = (uint32_t)weight_blocks_per_row;
    const uint32_t M_u32 = (uint32_t)M;

    [encoder setBuffer:buffers.weight offset:0 atIndex:0];
    [encoder setBuffer:input_q8_buffer offset:0 atIndex:1];
    [encoder setBuffer:buffers.out offset:0 atIndex:2];
    [encoder setBytes:&N_u32 length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&activation_blocks_u32 length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&M_u32 length:sizeof(uint32_t) atIndex:5];
    if (uses_super_blocks) {
        [encoder setBytes:&weight_blocks_u32 length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&K_u32 length:sizeof(uint32_t) atIndex:7];
    }

    metal_profiling_set_label(ctx, "matmul_quant_packed");
    metal_profiling_begin(ctx);

    MTLSize matmul_grid = MTLSizeMake(M, N, 1);
    MTLSize matmul_threads = metal_matmul_quant_threadgroup_size(matmul_pipeline);
    [encoder dispatchThreads:matmul_grid threadsPerThreadgroup:matmul_threads];

    metal_profiling_end(ctx);

    marmot_error_t ep_status = MARMOT_SUCCESS;
    if (rope != nullptr && rope->positions != nullptr && (rope->apply_to_q || rope->apply_to_k)) {
        ep_status = metal_rope(ctx, out, rope, out);
        if (ep_status != MARMOT_SUCCESS) {
            metal_matmul_quant_buffers_release(&buffers);
            [input_q8_buffer release];
            metal_matmul_quant_log(ctx, log_label, out, kernel_name);
            return ep_status;
        }
    }
    if (epilogue != nullptr) {
        id<MTLBuffer> ep_buffer = [buffers.out retain];
        ep_status = metal_matmul_apply_epilogue(ctx, out, ep_buffer, N * M, ep_feature_dim, ep_bias_scalar, epilogue);
        [ep_buffer release];
    } else {
        metal_residency_mark_dirty(ctx, out, out->dtype);
        metal_command_stream_flush(ctx, false);
    }

    metal_matmul_quant_buffers_release(&buffers);
    [input_q8_buffer release];
    metal_matmul_quant_log(ctx, log_label, out, route_tag);
    return ep_status;
}

marmot_error_t metal_matmul_quant_dispatch_k_direct(
    metal_context_t *ctx, const metal_kquant_kernels_t *kernels, const char *log_label, const marmot_tensor_t *input,
    const marmot_tensor_t *weight, marmot_tensor_t *out, size_t N, size_t K, size_t M, size_t weight_blocks_per_row,
    const marmot_matmul_epilogue_t *epilogue, size_t ep_feature_dim, bool ep_bias_scalar,
    const marmot_rope_params_t *rope
) {
    if (kernels == nullptr || kernels->kernel_opt == nullptr || kernels->kernel_small == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Missing Metal K-quant kernel names");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const bool small_k = K <= (size_t)MARMOT_QK_K_VALUES;
    const bool allow_mm =
        !metal_force_mv() && (metal_force_mm() || (ctx->device_caps.has_simdgroup_mm && M >= 64 && K >= 64));
    const bool mm_requested = allow_mm && !small_k && (metal_force_mm() || N > 8);
    const bool mm16_candidate = mm_requested && kernels->kernel_mm16 != nullptr && N <= 16;
    const bool mm32_candidate = mm_requested && !mm16_candidate && kernels->kernel_mm != nullptr;
    const bool mm_candidate = mm16_candidate || mm32_candidate;
    const char *mm_kernel = mm16_candidate ? kernels->kernel_mm16 : (mm32_candidate ? kernels->kernel_mm : nullptr);
    const bool mv_ext_candidate = !small_k && !mm_candidate && kernels->kernel_mv_ext != nullptr && N >= 4 && N <= 8 &&
        input->shape.strides[1] == 1 && (input->shape.strides[0] % 16u) == 0u && (K % 16u) == 0u;
    const bool use_nr2 = !small_k && !mm_candidate && !mv_ext_candidate && kernels->kernel_nr2 != nullptr && N >= 2;
    const char *kernel_name = small_k
        ? kernels->kernel_small
        : (mm_candidate
               ? mm_kernel
               : (mv_ext_candidate ? kernels->kernel_mv_ext : (use_nr2 ? kernels->kernel_nr2 : kernels->kernel_opt)));
    const char *route_tag = small_k
        ? "mv_small"
        : (mm16_candidate ? "mm16"
                          : (mm32_candidate ? "mm" : (mv_ext_candidate ? "mv_ext" : (use_nr2 ? "mv_nr2" : "mv_opt"))));
    const bool needs_route_details = metal_log_routes() || ctx->routing_debug;
    const char *route_msg = route_tag;
    char route_msg_buf[128];
    if (needs_route_details) {
        (void)snprintf(
            route_msg_buf, sizeof(route_msg_buf),
            "%s N=%zu M=%zu K=%zu small=%d mm=%d mm16=%d mvx=%d nr2=%d fmm=%d fmv=%d", route_tag, N, M, K,
            small_k ? 1 : 0, mm_candidate ? 1 : 0, mm16_candidate ? 1 : 0, mv_ext_candidate ? 1 : 0, use_nr2 ? 1 : 0,
            metal_force_mm() ? 1 : 0, metal_force_mv() ? 1 : 0
        );
        route_msg = route_msg_buf;
    }

    const char *requested_kernel = kernel_name;
    id<MTLComputePipelineState> matmul_pipeline = metal_pipeline_get(ctx, kernel_name);
    if (matmul_pipeline == nil) {
        const char *fallbacks[] = {kernels->kernel_opt, kernels->kernel_small};
        for (size_t i = 0; i < sizeof(fallbacks) / sizeof(fallbacks[0]); ++i) {
            const char *candidate = fallbacks[i];
            if (candidate == nullptr || candidate == kernel_name) {
                continue;
            }
            matmul_pipeline = metal_pipeline_get(ctx, candidate);
            if (matmul_pipeline != nil) {
                kernel_name = candidate;
                break;
            }
        }
        if (matmul_pipeline != nil && (metal_log_routes() || ctx->routing_debug)) {
            fprintf(
                stderr, "[marmot metal matmul_quant] missing kernel %s -> fallback %s\n", requested_kernel, kernel_name
            );
        }
        if (matmul_pipeline == nil) {
            marmot_set_error(MARMOT_ERROR_DEVICE_NOT_AVAILABLE, "Metal quantized matmul kernels not found");
            return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
        }
    }

    const bool use_mm = mm_candidate && kernel_name == mm_kernel;
    const bool use_mm16 = use_mm && kernel_name == kernels->kernel_mm16;
    const bool use_mv_ext = mv_ext_candidate && kernel_name == kernels->kernel_mv_ext;

    metal_matmul_quant_buffers_t buffers = {.weight = nil, .input = nil, .out = nil};
    marmot_error_t status = metal_matmul_quant_buffers_acquire(ctx, input, weight, out, &buffers);
    if (status != MARMOT_SUCCESS) {
        [matmul_pipeline release];
        return status;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, matmul_pipeline);
    if (encoder == nil) {
        metal_matmul_quant_buffers_release(&buffers);
        [matmul_pipeline release];
        marmot_set_error(MARMOT_ERROR_DEVICE_NOT_AVAILABLE, "Failed to acquire Metal compute encoder");
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
    }

    const uint32_t N_u32 = (uint32_t)N;
    const uint32_t K_u32 = (uint32_t)K;
    const uint32_t M_u32 = (uint32_t)M;
    const uint32_t stride_n = (uint32_t)input->shape.strides[0];
    const uint32_t stride_k = (uint32_t)input->shape.strides[1];
    const uint32_t weight_blocks_u32 = (uint32_t)weight_blocks_per_row;

    [encoder setBuffer:buffers.weight offset:0 atIndex:0];
    [encoder setBuffer:buffers.input offset:0 atIndex:1];
    [encoder setBuffer:buffers.out offset:0 atIndex:2];
    [encoder setBytes:&N_u32 length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&K_u32 length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&M_u32 length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&stride_n length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&stride_k length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&weight_blocks_u32 length:sizeof(uint32_t) atIndex:8];

    metal_profiling_set_label(ctx, "matmul_quant_k");
    metal_profiling_begin(ctx);

    if (use_mm) {
        const NSUInteger mm_nr0 = 64;
        const NSUInteger mm_nr1 = use_mm16 ? 16 : 32;
        const NSUInteger tg_mem_bytes = use_mm16 ? 5632 : 8192;
        [encoder setThreadgroupMemoryLength:tg_mem_bytes atIndex:0];
        MTLSize matmul_threads = use_mm16 ? MTLSizeMake(64, 1, 1) : MTLSizeMake(128, 1, 1);
        MTLSize matmul_groups = MTLSizeMake((N + mm_nr1 - 1) / mm_nr1, (M + mm_nr0 - 1) / mm_nr0, 1);
        [encoder dispatchThreadgroups:matmul_groups threadsPerThreadgroup:matmul_threads];
    } else if (small_k) {
        MTLSize matmul_grid = MTLSizeMake(M, N, 1);
        MTLSize matmul_threads = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreads:matmul_grid threadsPerThreadgroup:matmul_threads];
    } else {
        if (use_mv_ext) {
            const NSUInteger mvx_r0ptg = 8;
            const NSUInteger mvx_r1ptg = 4;
            MTLSize matmul_threads = MTLSizeMake(64, 1, 1);
            MTLSize matmul_groups = MTLSizeMake((M + mvx_r0ptg - 1) / mvx_r0ptg, (N + mvx_r1ptg - 1) / mvx_r1ptg, 1);
            [encoder dispatchThreadgroups:matmul_groups threadsPerThreadgroup:matmul_threads];
        } else {
            const bool use_mv = metal_kernel_has_suffix(kernel_name, "_mv");
            if (use_mv) {
                const bool mv_r0ptg8 = strncmp(kernel_name, "matmul_q4_0_", 11) == 0 ||
                    strncmp(kernel_name, "matmul_q4_1_", 11) == 0 || strncmp(kernel_name, "matmul_q5_0_", 11) == 0 ||
                    strncmp(kernel_name, "matmul_q5_1_", 11) == 0;
                const NSUInteger mv_r0ptg = mv_r0ptg8 ? 8 : 4;
                MTLSize matmul_threads = MTLSizeMake(64, 1, 1);
                MTLSize matmul_groups = MTLSizeMake((M + mv_r0ptg - 1) / mv_r0ptg, N, 1);
                [encoder dispatchThreadgroups:matmul_groups threadsPerThreadgroup:matmul_threads];
            } else {
                MTLSize matmul_threads = use_nr2 ? MTLSizeMake(64, 1, 1) : MTLSizeMake(32, 1, 1);
                MTLSize matmul_groups = MTLSizeMake(M, use_nr2 ? ((N + 3) / 4) : N, 1);
                [encoder dispatchThreadgroups:matmul_groups threadsPerThreadgroup:matmul_threads];
            }
        }
    }

    metal_profiling_end(ctx);

    marmot_error_t ep_status = MARMOT_SUCCESS;
    if (rope != nullptr && rope->positions != nullptr && (rope->apply_to_q || rope->apply_to_k)) {
        ep_status = metal_rope(ctx, out, rope, out);
        if (ep_status != MARMOT_SUCCESS) {
            metal_matmul_quant_buffers_release(&buffers);
            [matmul_pipeline release];
            metal_matmul_quant_log(ctx, log_label, out, kernel_name);
            return ep_status;
        }
    }
    if (epilogue != nullptr) {
        id<MTLBuffer> ep_buffer = [buffers.out retain];
        ep_status = metal_matmul_apply_epilogue(ctx, out, ep_buffer, N * M, ep_feature_dim, ep_bias_scalar, epilogue);
        [ep_buffer release];
    } else {
        metal_residency_mark_dirty(ctx, out, out->dtype);
        metal_command_stream_flush(ctx, false);
    }

    metal_matmul_quant_buffers_release(&buffers);
    [matmul_pipeline release];
    metal_matmul_quant_log(ctx, log_label, out, route_msg);
    return ep_status;
}

marmot_error_t metal_matmul_quant_dispatch_packed(
    metal_context_t *ctx, const char *kernel_name, const char *quant_kernel, const char *log_label,
    const marmot_tensor_t *input, const marmot_tensor_t *weight, marmot_tensor_t *out, size_t N, size_t K, size_t M,
    size_t activation_blocks_per_row, size_t activation_block_bytes, size_t weight_blocks_per_row,
    bool uses_super_blocks, const marmot_matmul_epilogue_t *epilogue, size_t ep_feature_dim, bool ep_bias_scalar,
    const marmot_rope_params_t *rope
) {
    id<MTLComputePipelineState> matmul_pipeline = metal_pipeline_get(ctx, kernel_name);
    if (matmul_pipeline == nil) {
        marmot_set_error(MARMOT_ERROR_DEVICE_NOT_AVAILABLE, "Metal quantized matmul kernels not found");
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
    }
    id<MTLComputePipelineState> quant_pipeline = metal_pipeline_get(ctx, quant_kernel);
    if (quant_pipeline == nil) {
        [matmul_pipeline release];
        marmot_set_error(MARMOT_ERROR_DEVICE_NOT_AVAILABLE, "Metal quantized matmul kernels not found");
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
    }

    marmot_error_t status = metal_matmul_quant_dispatch_packed_impl(
        ctx, quant_pipeline, matmul_pipeline, log_label, kernel_name, input, weight, out, N, K, M,
        activation_blocks_per_row, activation_block_bytes, weight_blocks_per_row, uses_super_blocks, epilogue,
        ep_feature_dim, ep_bias_scalar, rope
    );
    [quant_pipeline release];
    [matmul_pipeline release];
    return status;
}

#endif // __APPLE__
