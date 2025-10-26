#include "internal/metal_matmul_internal.h"
#include "internal/stride_helpers.h"
#include "metal_backend_internal.h"

#ifdef __APPLE__

#include <stdlib.h>

#include <math.h>
#include <string.h>

static const char *metal_rope_kernel_name(marmot_dtype_t dtype, bool strided) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return strided ? "rope_f32_g" : "rope_f32";
    case MARMOT_DTYPE_FLOAT16:
        return strided ? "rope_f16_g" : "rope_f16";
    case MARMOT_DTYPE_BFLOAT16:
        return strided ? "rope_bf16_g" : "rope_bf16";
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
        return strided ? "rope_fp8_e4m3_g" : "rope_fp8_e4m3";
    case MARMOT_DTYPE_FLOAT8_E5M2:
        return strided ? "rope_fp8_e5m2_g" : "rope_fp8_e5m2";
#endif
    default:
        return nullptr;
    }
}

static const char *metal_paged_kv_scatter_kernel_name(marmot_dtype_t in_dtype, marmot_dtype_t kv_dtype) {
    switch (in_dtype) {
    case MARMOT_DTYPE_FLOAT32:
        switch (kv_dtype) {
        case MARMOT_DTYPE_FLOAT32:
            return "paged_kv_scatter_f32";
        case MARMOT_DTYPE_FLOAT16:
            return "paged_kv_scatter_f32_kv_f16";
        case MARMOT_DTYPE_BFLOAT16:
            return "paged_kv_scatter_f32_kv_bf16";
        default:
            return nullptr;
        }
    case MARMOT_DTYPE_FLOAT16:
        switch (kv_dtype) {
        case MARMOT_DTYPE_FLOAT16:
            return "paged_kv_scatter_f16";
        case MARMOT_DTYPE_FLOAT32:
            return "paged_kv_scatter_f16_kv_f32";
        case MARMOT_DTYPE_BFLOAT16:
            return "paged_kv_scatter_f16_kv_bf16";
        default:
            return nullptr;
        }
    case MARMOT_DTYPE_BFLOAT16:
        switch (kv_dtype) {
        case MARMOT_DTYPE_BFLOAT16:
            return "paged_kv_scatter_bf16";
        case MARMOT_DTYPE_FLOAT32:
            return "paged_kv_scatter_bf16_kv_f32";
        case MARMOT_DTYPE_FLOAT16:
            return "paged_kv_scatter_bf16_kv_f16";
        default:
            return nullptr;
        }
    default:
        return nullptr;
    }
}

static const char *
metal_paged_attention_kernel_name(marmot_dtype_t q_dtype, marmot_dtype_t kv_dtype, uint32_t simd_groups) {
    const bool use_sg4 = simd_groups == 4;
    const bool use_sg8 = simd_groups == 8;
    if (!use_sg4 && !use_sg8) {
        return nullptr;
    }
    switch (q_dtype) {
    case MARMOT_DTYPE_FLOAT32:
        switch (kv_dtype) {
        case MARMOT_DTYPE_FLOAT32:
            return use_sg4 ? "paged_attention_sg4_f32" : "paged_attention_sg8_f32";
        case MARMOT_DTYPE_FLOAT16:
            return use_sg4 ? "paged_attention_sg4_f32_kv_f16" : "paged_attention_sg8_f32_kv_f16";
        case MARMOT_DTYPE_BFLOAT16:
            return use_sg4 ? "paged_attention_sg4_f32_kv_bf16" : "paged_attention_sg8_f32_kv_bf16";
        default:
            return nullptr;
        }
    case MARMOT_DTYPE_FLOAT16:
        switch (kv_dtype) {
        case MARMOT_DTYPE_FLOAT16:
            return use_sg4 ? "paged_attention_sg4_f16" : "paged_attention_sg8_f16";
        case MARMOT_DTYPE_FLOAT32:
            return use_sg4 ? "paged_attention_sg4_f16_kv_f32" : "paged_attention_sg8_f16_kv_f32";
        case MARMOT_DTYPE_BFLOAT16:
            return use_sg4 ? "paged_attention_sg4_f16_kv_bf16" : "paged_attention_sg8_f16_kv_bf16";
        default:
            return nullptr;
        }
    case MARMOT_DTYPE_BFLOAT16:
        switch (kv_dtype) {
        case MARMOT_DTYPE_BFLOAT16:
            return use_sg4 ? "paged_attention_sg4_bf16" : "paged_attention_sg8_bf16";
        case MARMOT_DTYPE_FLOAT32:
            return use_sg4 ? "paged_attention_sg4_bf16_kv_f32" : "paged_attention_sg8_bf16_kv_f32";
        case MARMOT_DTYPE_FLOAT16:
            return use_sg4 ? "paged_attention_sg4_bf16_kv_f16" : "paged_attention_sg8_bf16_kv_f16";
        default:
            return nullptr;
        }
    default:
        return nullptr;
    }
}

static constexpr uint32_t k_metal_decode_attention_simd_width = 32;
static constexpr uint32_t k_metal_decode_attention_simd_groups_small = 4;
static constexpr uint32_t k_metal_decode_attention_simd_groups_large = 8;
static constexpr size_t k_metal_decode_attention_dim_max = 256;
static constexpr size_t k_metal_decode_attention_seq_k_large = 64;

static constexpr size_t k_metal_flash_attention_dim_small = 128;
static constexpr size_t k_metal_flash_attention_dim_max = 256;
static constexpr size_t k_metal_flash_attention_dim_narrow = 192;

static constexpr uint32_t k_paged_token_flag_prefill = 1u << 0;
static constexpr uint32_t k_paged_token_flag_decode = 1u << 1;

static int metal_decode_attention_simd_groups_override(void) {
    static int cached = -2;
    if (cached != -2) {
        return cached;
    }
    cached = -1;
    const char *env = getenv("MARMOT_METAL_DECODE_SIMD_GROUPS");
    if (env == nullptr || env[0] == '\0') {
        return cached;
    }
    char *end = nullptr;
    long value = strtol(env, &end, 10);
    if (end != env && (value == 4 || value == 8)) {
        cached = (int)value;
    }
    return cached;
}

enum metal_paged_flash_tuning_mode {
    k_metal_paged_flash_tune_auto = 0,
    k_metal_paged_flash_tune_default = 1,
    k_metal_paged_flash_tune_narrow = 2,
};

static metal_paged_flash_tuning_mode metal_paged_flash_tuning_mode_value(void) {
    static bool cached = false;
    static metal_paged_flash_tuning_mode mode = k_metal_paged_flash_tune_auto;
    if (cached) {
        return mode;
    }
    cached = true;
    const char *env = getenv("MARMOT_METAL_PAGED_FLASH_VARIANT");
    if (env == nullptr || env[0] == '\0') {
        return mode;
    }
    if (strcmp(env, "narrow") == 0 || strcmp(env, "1") == 0) {
        mode = k_metal_paged_flash_tune_narrow;
    } else if (strcmp(env, "default") == 0 || strcmp(env, "0") == 0) {
        mode = k_metal_paged_flash_tune_default;
    } else if (strcmp(env, "auto") == 0) {
        mode = k_metal_paged_flash_tune_auto;
    }
    return mode;
}

static bool metal_paged_flash_use_narrow_variant(size_t dim, marmot_dtype_t dtype, marmot_dtype_t kv_dtype) {
    if (dim == 0 || dim > k_metal_flash_attention_dim_max) {
        return false;
    }
    if (dtype != MARMOT_DTYPE_FLOAT32 && dtype != MARMOT_DTYPE_FLOAT16 && dtype != MARMOT_DTYPE_BFLOAT16) {
        return false;
    }
    const metal_paged_flash_tuning_mode mode = metal_paged_flash_tuning_mode_value();
    if (mode == k_metal_paged_flash_tune_narrow) {
        return true;
    }
    if (mode == k_metal_paged_flash_tune_default) {
        return false;
    }
    if (kv_dtype == MARMOT_DTYPE_FLOAT32 && (dtype == MARMOT_DTYPE_FLOAT16 || dtype == MARMOT_DTYPE_BFLOAT16)) {
        return true;
    }
    return dim >= k_metal_flash_attention_dim_narrow;
}

static const char *metal_paged_flash_attention_kernel_name(marmot_dtype_t dtype, size_t dim) {
    if (dim == 0 || dim > k_metal_flash_attention_dim_max) {
        return nullptr;
    }
    const bool wide = dim > k_metal_flash_attention_dim_small;
    const bool narrow = metal_paged_flash_use_narrow_variant(dim, dtype, dtype);
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        if (narrow) {
            return wide ? "paged_flash_attention_f32_wide_narrow" : "paged_flash_attention_f32_narrow";
        }
        return wide ? "paged_flash_attention_f32_wide" : "paged_flash_attention_f32";
    case MARMOT_DTYPE_FLOAT16:
        if (narrow) {
            return wide ? "paged_flash_attention_f16_wide_narrow" : "paged_flash_attention_f16_narrow";
        }
        return wide ? "paged_flash_attention_f16_wide" : "paged_flash_attention_f16";
    case MARMOT_DTYPE_BFLOAT16:
        if (narrow) {
            return wide ? "paged_flash_attention_bf16_wide_narrow" : "paged_flash_attention_bf16_narrow";
        }
        return wide ? "paged_flash_attention_bf16_wide" : "paged_flash_attention_bf16";
    default:
        return nullptr;
    }
}

static const char *metal_paged_flash_attention_kernel_name(marmot_dtype_t dtype, marmot_dtype_t kv_dtype, size_t dim) {
    if (kv_dtype == dtype) {
        return metal_paged_flash_attention_kernel_name(dtype, dim);
    }
    if (dim == 0 || dim > k_metal_flash_attention_dim_max) {
        return nullptr;
    }
    const bool wide = dim > k_metal_flash_attention_dim_small;
    const bool narrow = metal_paged_flash_use_narrow_variant(dim, dtype, kv_dtype);
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        switch (kv_dtype) {
        case MARMOT_DTYPE_FLOAT16:
            if (narrow) {
                return wide ? "paged_flash_attention_f32_kv_f16_wide_narrow"
                            : "paged_flash_attention_f32_kv_f16_narrow";
            }
            return wide ? "paged_flash_attention_f32_kv_f16_wide" : "paged_flash_attention_f32_kv_f16";
        case MARMOT_DTYPE_BFLOAT16:
            if (narrow) {
                return wide ? "paged_flash_attention_f32_kv_bf16_wide_narrow"
                            : "paged_flash_attention_f32_kv_bf16_narrow";
            }
            return wide ? "paged_flash_attention_f32_kv_bf16_wide" : "paged_flash_attention_f32_kv_bf16";
        default:
            return nullptr;
        }
    case MARMOT_DTYPE_FLOAT16:
        switch (kv_dtype) {
        case MARMOT_DTYPE_FLOAT32:
            if (narrow) {
                return wide ? "paged_flash_attention_f16_kv_f32_wide_narrow"
                            : "paged_flash_attention_f16_kv_f32_narrow";
            }
            return wide ? "paged_flash_attention_f16_kv_f32_wide" : "paged_flash_attention_f16_kv_f32";
        case MARMOT_DTYPE_BFLOAT16:
            if (narrow) {
                return wide ? "paged_flash_attention_f16_kv_bf16_wide_narrow"
                            : "paged_flash_attention_f16_kv_bf16_narrow";
            }
            return wide ? "paged_flash_attention_f16_kv_bf16_wide" : "paged_flash_attention_f16_kv_bf16";
        default:
            return nullptr;
        }
    case MARMOT_DTYPE_BFLOAT16:
        switch (kv_dtype) {
        case MARMOT_DTYPE_FLOAT32:
            if (narrow) {
                return wide ? "paged_flash_attention_bf16_kv_f32_wide_narrow"
                            : "paged_flash_attention_bf16_kv_f32_narrow";
            }
            return wide ? "paged_flash_attention_bf16_kv_f32_wide" : "paged_flash_attention_bf16_kv_f32";
        case MARMOT_DTYPE_FLOAT16:
            if (narrow) {
                return wide ? "paged_flash_attention_bf16_kv_f16_wide_narrow"
                            : "paged_flash_attention_bf16_kv_f16_narrow";
            }
            return wide ? "paged_flash_attention_bf16_kv_f16_wide" : "paged_flash_attention_bf16_kv_f16";
        default:
            return nullptr;
        }
    default:
        return nullptr;
    }
}

static uint32_t metal_flash_attention_block_m(marmot_dtype_t dtype, size_t dim) {
    const bool wide = dim > k_metal_flash_attention_dim_small;
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return wide ? 4 : 8;
    case MARMOT_DTYPE_FLOAT16:
    case MARMOT_DTYPE_BFLOAT16:
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
    case MARMOT_DTYPE_FLOAT8_E5M2:
#endif
        return wide ? 8 : 16;
    default:
        return 0;
    }
}

static uint32_t metal_flash_attention_block_n(marmot_dtype_t dtype, size_t dim) {
    const bool wide = dim > k_metal_flash_attention_dim_small;
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return wide ? 16 : 32;
    case MARMOT_DTYPE_FLOAT16:
    case MARMOT_DTYPE_BFLOAT16:
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
    case MARMOT_DTYPE_FLOAT8_E5M2:
#endif
        return 32;
    default:
        return 0;
    }
}

static uint32_t metal_paged_flash_attention_block_n(marmot_dtype_t dtype, marmot_dtype_t kv_dtype, size_t dim) {
    if (!metal_paged_flash_use_narrow_variant(dim, dtype, kv_dtype)) {
        return metal_flash_attention_block_n(dtype, dim);
    }
    const bool wide = dim > k_metal_flash_attention_dim_small;
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return wide ? 8 : 16;
    case MARMOT_DTYPE_FLOAT16:
    case MARMOT_DTYPE_BFLOAT16:
        return 16;
    default:
        return 0;
    }
}

static size_t metal_flash_attention_max_dim(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
    case MARMOT_DTYPE_FLOAT16:
    case MARMOT_DTYPE_BFLOAT16:
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
    case MARMOT_DTYPE_FLOAT8_E5M2:
#endif
        return k_metal_flash_attention_dim_max;
    default:
        return 0;
    }
}

static bool metal_should_use_paged_flash_attention(
    size_t seq_q, size_t seq_k, size_t dim, marmot_dtype_t dtype, marmot_dtype_t kv_dtype
) {
    (void)seq_q;
    // Use paged Flash Attention when:
    // 1. KV length is long enough to benefit from tiling (>= 128)
    // 2. Dimension fits within kernel limits
    size_t max_dim = metal_flash_attention_max_dim(dtype);
    uint32_t block_m = metal_flash_attention_block_m(dtype, dim);
    uint32_t block_n = metal_paged_flash_attention_block_n(dtype, kv_dtype, dim);
    if (seq_k < 128 || max_dim == 0 || dim > max_dim || block_m == 0 || block_n == 0) {
        return false;
    }
    return metal_paged_flash_attention_kernel_name(dtype, kv_dtype, dim) != nullptr;
}

static uint32_t metal_decode_attention_simd_groups(size_t seq_k) {
    int override = metal_decode_attention_simd_groups_override();
    if (override > 0) {
        return (uint32_t) override;
    }
    return seq_k >= k_metal_decode_attention_seq_k_large ? k_metal_decode_attention_simd_groups_large
                                                         : k_metal_decode_attention_simd_groups_small;
}

static bool metal_paged_attention_dtype_supported(marmot_dtype_t dtype) {
    return dtype == MARMOT_DTYPE_FLOAT32 || dtype == MARMOT_DTYPE_FLOAT16 || dtype == MARMOT_DTYPE_BFLOAT16;
}

static bool metal_is_power_of_two_u32(uint32_t value) {
    return value != 0u && (value & (value - 1u)) == 0u;
}

static uint32_t metal_log2_u32(uint32_t value) {
    uint32_t shift = 0;
    while (value > 1u) {
        value >>= 1u;
        shift++;
    }
    return shift;
}

marmot_error_t
metal_rope(const void *device_ctx, const marmot_tensor_t *x, const marmot_rope_params_t *params, marmot_tensor_t *out) {
    if (x == nullptr || params == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const marmot_tensor_t *positions = params->positions;
    if (positions == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr) {
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    if (x->shape.ndim < 2 || out->shape.ndim != x->shape.ndim) {
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    const auto x_stride = marmot::metal::get_stride_info(x);
    const auto out_stride = marmot::metal::get_stride_info(out);
    const bool use_strided = (!x_stride.is_contiguous || !out_stride.is_contiguous);
    if (use_strided) {
        const size_t ndim = x->shape.ndim;
        if (x->shape.strides[ndim - 1] != 1 || out->shape.strides[ndim - 1] != 1) {
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal RoPE requires contiguous inner dimension");
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
    }

    const char *kernel_name = metal_rope_kernel_name(x->dtype, use_strided);
    if (kernel_name == nullptr || out->dtype != x->dtype) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    const bool positions_is_float = (positions->dtype == MARMOT_DTYPE_FLOAT32);
    const bool positions_is_int = (positions->dtype == MARMOT_DTYPE_INT32);
    if (!positions_is_float && !positions_is_int) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    size_t dim = x->shape.shape[x->shape.ndim - 1];
    if (dim % 2 != 0) {
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    size_t seq_len = x->shape.shape[x->shape.ndim - 2];
    size_t total_tokens = marmot_tensor_num_elements(x);
    if (seq_len == 0 || dim == 0 || total_tokens == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    size_t total_seqs = total_tokens / (seq_len * dim);

    size_t expected_positions = total_seqs * seq_len;
    if (marmot_tensor_num_elements(positions) != expected_positions) {
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    for (size_t i = 0; i < x->shape.ndim; ++i) {
        if (x->shape.shape[i] != out->shape.shape[i]) {
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }

    size_t bytes = marmot::metal::tensor_span_bytes(x);
    size_t out_bytes = marmot::metal::tensor_span_bytes(out);
    size_t pos_float_bytes = expected_positions * sizeof(float);
    if (bytes == 0 || out_bytes == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_tensor_buffer_t viewX = metal_buffer_acquire_view(ctx, x, x->dtype, bytes);
    id<MTLBuffer> bufferX = viewX.buffer;
    const size_t offsetX = viewX.offset;
    id<MTLBuffer> bufferPos = nil;
    constexpr size_t k_rope_positions_inline_limit = 4096;
    const bool use_positions_bytes =
        positions_is_float && pos_float_bytes > 0 && pos_float_bytes <= k_rope_positions_inline_limit;
    if (positions_is_float) {
        if (!use_positions_bytes) {
            bufferPos = metal_buffer_acquire(ctx, positions->data, marmot_tensor_size_bytes(positions));
        }
    } else {
        const int32_t *positions_data = (const int32_t *)positions->data;
        float *positions_temp = (float *)malloc(pos_float_bytes);
        if (positions_temp == nullptr) {
            if (bufferX != nil)
                [bufferX release];
            [pipeline release];
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        for (size_t i = 0; i < expected_positions; ++i) {
            positions_temp[i] = (float)positions_data[i];
        }
        bufferPos = [ctx->device newBufferWithBytesNoCopy:positions_temp
                                                   length:pos_float_bytes
                                                  options:MTLResourceStorageModeShared
                                              deallocator:^(void *ptr, NSUInteger length) {
                                                (void)length;
                                                free(ptr);
                                              }];
        if (bufferPos == nil) {
            free(positions_temp);
        }
    }

    metal_tensor_buffer_t viewOut = metal_buffer_acquire_view(ctx, out, out->dtype, out_bytes);
    id<MTLBuffer> bufferOut = viewOut.buffer;
    const size_t offsetOut = viewOut.offset;
    const bool out_private = viewOut.is_private;
    if (bufferX == nil || bufferOut == nil || (!use_positions_bytes && bufferPos == nil)) {
        if (bufferX != nil)
            [bufferX release];
        if (bufferPos != nil)
            [bufferPos release];
        if (bufferOut != nil)
            [bufferOut release];
        [pipeline release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    float rope_attn_scale = 1.0f;
    id<MTLBuffer> bufferFreqs = metal_matmul_prepare_freq_buffer(ctx, dim, params, &rope_attn_scale);
    if (bufferFreqs == nil) {
        [bufferX release];
        [bufferPos release];
        [bufferOut release];
        [pipeline release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    uint32_t seq_len_u32 = (uint32_t)seq_len;
    uint32_t dim_u32 = (uint32_t)dim;
    uint32_t total_seqs_u32 = (uint32_t)total_seqs;
    uint32_t rope_type_u32 = (uint32_t)params->rope_type;

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferX release];
        if (bufferPos != nil) {
            [bufferPos release];
        }
        [bufferOut release];
        [bufferFreqs release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }
    [encoder setBuffer:bufferX offset:offsetX atIndex:0];
    if (use_positions_bytes) {
        [encoder setBytes:positions->data length:pos_float_bytes atIndex:1];
    } else {
        [encoder setBuffer:bufferPos offset:0 atIndex:1];
    }
    [encoder setBuffer:bufferOut offset:offsetOut atIndex:2];
    [encoder setBytes:&seq_len_u32 length:sizeof(seq_len_u32) atIndex:3];
    [encoder setBytes:&dim_u32 length:sizeof(dim_u32) atIndex:4];
    [encoder setBytes:&total_seqs_u32 length:sizeof(total_seqs_u32) atIndex:5];
    [encoder setBuffer:bufferFreqs offset:0 atIndex:6];
    [encoder setBytes:&rope_attn_scale length:sizeof(rope_attn_scale) atIndex:7];
    [encoder setBytes:&rope_type_u32 length:sizeof(rope_type_u32) atIndex:8];
    if (use_strided) {
        const uint32_t ndim_u32 = x_stride.ndim;
        if (ndim_u32 < 2) {
            [pipeline release];
            [bufferX release];
            [bufferPos release];
            [bufferOut release];
            [bufferFreqs release];
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        [encoder setBytes:&x_stride.shape length:sizeof(x_stride.shape) atIndex:9];
        [encoder setBytes:&x_stride.strides length:sizeof(x_stride.strides) atIndex:10];
        [encoder setBytes:&out_stride.strides length:sizeof(out_stride.strides) atIndex:11];
        [encoder setBytes:&ndim_u32 length:sizeof(ndim_u32) atIndex:12];
    }

    const NSUInteger total_tokens_u = (NSUInteger)(total_seqs * seq_len);
    NSUInteger threadsPerGroup = metal_threadgroup_size_1d(pipeline, total_tokens_u);

    MTLSize grid = MTLSizeMake(total_tokens_u, 1, 1);
    MTLSize threads = MTLSizeMake(threadsPerGroup, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];

    metal_command_stream_flush(ctx, false);

    [pipeline release];
    [bufferX release];
    if (bufferPos != nil) {
        [bufferPos release];
    }
    [bufferOut release];
    if (bufferFreqs != nil) {
        [bufferFreqs release];
    }
    if (out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t metal_paged_attention_impl(const void *device_ctx, const marmot_paged_attention_desc_t *desc) {
    if (desc == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention descriptor invalid");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!marmot_paged_attention_desc_is_valid(desc)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention descriptor invalid");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr) {
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    if (desc->token_count == 0) {
        return MARMOT_SUCCESS;
    }
    if (desc->num_q_heads == 0 || desc->num_kv_heads == 0 || desc->head_dim == 0 || desc->block_size == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention requires non-zero dimensions");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!metal_is_power_of_two_u32(desc->block_size)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention block_size must be power of two");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (desc->num_q_heads % desc->num_kv_heads != 0) {
        marmot_set_error(
            MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention requires num_q_heads divisible by num_kv_heads"
        );
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_tensor_t *token_meta = desc->token_meta;
    const marmot_tensor_t *q = desc->q;
    const marmot_tensor_t *k_new = desc->k_new;
    const marmot_tensor_t *v_new = desc->v_new;
    const marmot_tensor_t *kv_k = desc->kv_k;
    const marmot_tensor_t *kv_v = desc->kv_v;
    const marmot_tensor_t *block_table = desc->block_table;
    marmot_tensor_t *out = desc->out;

    if (token_meta->dtype != MARMOT_DTYPE_UINT32 || block_table->dtype != MARMOT_DTYPE_UINT32) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "paged_attention requires uint32 metadata");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (!metal_paged_attention_dtype_supported(q->dtype) || !metal_paged_attention_dtype_supported(k_new->dtype) ||
        !metal_paged_attention_dtype_supported(v_new->dtype) || !metal_paged_attention_dtype_supported(kv_k->dtype) ||
        !metal_paged_attention_dtype_supported(kv_v->dtype) || !metal_paged_attention_dtype_supported(out->dtype)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "paged_attention dtype not supported");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (q->dtype != k_new->dtype || q->dtype != v_new->dtype || q->dtype != out->dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "paged_attention requires matching q/k_new/v_new/out dtypes");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    if (token_meta->shape.ndim != 2 || token_meta->shape.shape[1] != 4 ||
        token_meta->shape.shape[0] != desc->token_count) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention token_meta shape invalid");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (block_table->shape.ndim != 2) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention block_table must be 2D");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (q->shape.ndim != 3 || k_new->shape.ndim != 3 || v_new->shape.ndim != 3 || out->shape.ndim != 3) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention expects 3D q/k_new/v_new/out tensors");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (q->shape.shape[0] < desc->token_count || k_new->shape.shape[0] < desc->token_count ||
        v_new->shape.shape[0] < desc->token_count || out->shape.shape[0] < desc->token_count) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention token_count exceeds tensor rows");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (q->shape.shape[1] != desc->num_q_heads || out->shape.shape[1] != desc->num_q_heads) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention num_q_heads mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (k_new->shape.shape[1] != desc->num_kv_heads || v_new->shape.shape[1] != desc->num_kv_heads) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention num_kv_heads mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (q->shape.shape[2] != desc->head_dim || k_new->shape.shape[2] != desc->head_dim ||
        v_new->shape.shape[2] != desc->head_dim || out->shape.shape[2] != desc->head_dim) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention head_dim mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (kv_k->shape.ndim != 5 || kv_v->shape.ndim != 5) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention kv cache must be 5D");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (kv_k->shape.shape[1] <= desc->layer_idx) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention layer_idx out of range");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (kv_k->shape.shape[2] != desc->num_kv_heads || kv_v->shape.shape[2] != desc->num_kv_heads) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention kv head mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (kv_k->shape.shape[3] != desc->block_size || kv_v->shape.shape[3] != desc->block_size) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention block_size mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (kv_k->shape.shape[4] != desc->head_dim || kv_v->shape.shape[4] != desc->head_dim) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention kv head_dim mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (kv_k->dtype != kv_v->dtype) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention kv dtype mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (desc->head_dim > k_metal_decode_attention_dim_max) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention head_dim exceeds metal support");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const char *scatter_kernel = metal_paged_kv_scatter_kernel_name(q->dtype, kv_k->dtype);
    if (scatter_kernel == nullptr) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "paged_attention unsupported dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    size_t token_meta_bytes = marmot::metal::tensor_span_bytes(token_meta);
    size_t block_table_bytes = marmot::metal::tensor_span_bytes(block_table);
    size_t q_bytes = marmot::metal::tensor_span_bytes(q);
    size_t k_new_bytes = marmot::metal::tensor_span_bytes(k_new);
    size_t v_new_bytes = marmot::metal::tensor_span_bytes(v_new);
    size_t kv_k_bytes = marmot::metal::tensor_span_bytes(kv_k);
    size_t kv_v_bytes = marmot::metal::tensor_span_bytes(kv_v);
    size_t out_bytes = marmot::metal::tensor_span_bytes(out);
    if (token_meta_bytes == 0 || block_table_bytes == 0 || q_bytes == 0 || k_new_bytes == 0 || v_new_bytes == 0 ||
        kv_k_bytes == 0 || kv_v_bytes == 0 || out_bytes == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention invalid tensor spans");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    metal_tensor_buffer_t viewMeta = metal_buffer_acquire_view(ctx, token_meta, token_meta->dtype, token_meta_bytes);
    metal_tensor_buffer_t viewBlock =
        metal_buffer_acquire_view(ctx, block_table, block_table->dtype, block_table_bytes);
    metal_tensor_buffer_t viewQ = metal_buffer_acquire_view(ctx, q, q->dtype, q_bytes);
    metal_tensor_buffer_t viewK = metal_buffer_acquire_view(ctx, k_new, k_new->dtype, k_new_bytes);
    metal_tensor_buffer_t viewV = metal_buffer_acquire_view(ctx, v_new, v_new->dtype, v_new_bytes);
    metal_tensor_buffer_t viewKvK = metal_buffer_acquire_view(ctx, kv_k, kv_k->dtype, kv_k_bytes);
    metal_tensor_buffer_t viewKvV = metal_buffer_acquire_view(ctx, kv_v, kv_v->dtype, kv_v_bytes);
    metal_tensor_buffer_t viewOut = metal_buffer_acquire_view(ctx, out, out->dtype, out_bytes);

    id<MTLBuffer> bufferMeta = viewMeta.buffer;
    id<MTLBuffer> bufferBlock = viewBlock.buffer;
    id<MTLBuffer> bufferQ = viewQ.buffer;
    id<MTLBuffer> bufferK = viewK.buffer;
    id<MTLBuffer> bufferV = viewV.buffer;
    id<MTLBuffer> bufferKvK = viewKvK.buffer;
    id<MTLBuffer> bufferKvV = viewKvV.buffer;
    id<MTLBuffer> bufferOut = viewOut.buffer;
    if (bufferMeta == nil || bufferBlock == nil || bufferQ == nil || bufferK == nil || bufferV == nil ||
        bufferKvK == nil || bufferKvV == nil || bufferOut == nil) {
        if (bufferMeta != nil)
            [bufferMeta release];
        if (bufferBlock != nil)
            [bufferBlock release];
        if (bufferQ != nil)
            [bufferQ release];
        if (bufferK != nil)
            [bufferK release];
        if (bufferV != nil)
            [bufferV release];
        if (bufferKvK != nil)
            [bufferKvK release];
        if (bufferKvV != nil)
            [bufferKvV release];
        if (bufferOut != nil)
            [bufferOut release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    const size_t meta_stride0 = token_meta->shape.strides[0];
    const size_t meta_stride1 = token_meta->shape.strides[1];
    const size_t block_stride0 = block_table->shape.strides[0];
    const size_t block_stride1 = block_table->shape.strides[1];
    if (meta_stride0 > UINT32_MAX || meta_stride1 > UINT32_MAX || block_stride0 > UINT32_MAX ||
        block_stride1 > UINT32_MAX || block_table->shape.shape[1] > UINT32_MAX || desc->token_count > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention stride overflow");
        [bufferMeta release];
        [bufferBlock release];
        [bufferQ release];
        [bufferK release];
        [bufferV release];
        [bufferKvK release];
        [bufferKvV release];
        [bufferOut release];
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    struct metal_paged_attention_uniforms_t {
        uint32_t token_count;
        uint32_t token_offset;
        uint32_t num_q_heads;
        uint32_t num_kv_heads;
        uint32_t head_dim;
        uint32_t block_size;
        uint32_t block_shift;
        uint32_t layer_idx;
        uint32_t gqa_group_size;
        uint32_t max_blocks_per_seq;
        uint32_t meta_stride0;
        uint32_t meta_stride1;
        uint32_t block_stride0;
        uint32_t block_stride1;
        float scale;
    };

    struct metal_paged_attention_strides_t {
        uint64_t q_stride0;
        uint64_t q_stride1;
        uint64_t q_stride2;
        uint64_t k_stride0;
        uint64_t k_stride1;
        uint64_t k_stride2;
        uint64_t v_stride0;
        uint64_t v_stride1;
        uint64_t v_stride2;
        uint64_t out_stride0;
        uint64_t out_stride1;
        uint64_t out_stride2;
        uint64_t kv_stride0;
        uint64_t kv_stride1;
        uint64_t kv_stride2;
        uint64_t kv_stride3;
        uint64_t kv_stride4;
    };

    metal_paged_attention_strides_t strides = {
        .q_stride0 = (uint64_t)q->shape.strides[0],
        .q_stride1 = (uint64_t)q->shape.strides[1],
        .q_stride2 = (uint64_t)q->shape.strides[2],
        .k_stride0 = (uint64_t)k_new->shape.strides[0],
        .k_stride1 = (uint64_t)k_new->shape.strides[1],
        .k_stride2 = (uint64_t)k_new->shape.strides[2],
        .v_stride0 = (uint64_t)v_new->shape.strides[0],
        .v_stride1 = (uint64_t)v_new->shape.strides[1],
        .v_stride2 = (uint64_t)v_new->shape.strides[2],
        .out_stride0 = (uint64_t)out->shape.strides[0],
        .out_stride1 = (uint64_t)out->shape.strides[1],
        .out_stride2 = (uint64_t)out->shape.strides[2],
        .kv_stride0 = (uint64_t)kv_k->shape.strides[0],
        .kv_stride1 = (uint64_t)kv_k->shape.strides[1],
        .kv_stride2 = (uint64_t)kv_k->shape.strides[2],
        .kv_stride3 = (uint64_t)kv_k->shape.strides[3],
        .kv_stride4 = (uint64_t)kv_k->shape.strides[4],
    };

    const uint32_t block_shift = metal_log2_u32(desc->block_size);
    const uint32_t gqa_group = desc->num_q_heads / desc->num_kv_heads;
    metal_paged_attention_uniforms_t base_uniforms = {
        .token_count = desc->token_count,
        .token_offset = 0,
        .num_q_heads = desc->num_q_heads,
        .num_kv_heads = desc->num_kv_heads,
        .head_dim = desc->head_dim,
        .block_size = desc->block_size,
        .block_shift = block_shift,
        .layer_idx = desc->layer_idx,
        .gqa_group_size = gqa_group,
        .max_blocks_per_seq = (uint32_t)block_table->shape.shape[1],
        .meta_stride0 = (uint32_t)meta_stride0,
        .meta_stride1 = (uint32_t)meta_stride1,
        .block_stride0 = (uint32_t)block_stride0,
        .block_stride1 = (uint32_t)block_stride1,
        .scale = desc->scale,
    };

    metal_profiling_set_label(ctx, "paged_attention");
    metal_profiling_begin(ctx);

    id<MTLComputePipelineState> scatter_pipe = metal_pipeline_get(ctx, scatter_kernel);
    if (scatter_pipe == nil) {
        [bufferMeta release];
        [bufferBlock release];
        [bufferQ release];
        [bufferK release];
        [bufferV release];
        [bufferKvK release];
        [bufferKvV release];
        [bufferOut release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, scatter_pipe);
    if (encoder == nil) {
        [scatter_pipe release];
        [bufferMeta release];
        [bufferBlock release];
        [bufferQ release];
        [bufferK release];
        [bufferV release];
        [bufferKvK release];
        [bufferKvV release];
        [bufferOut release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    [encoder setBuffer:bufferMeta offset:viewMeta.offset atIndex:0];
    [encoder setBuffer:bufferK offset:viewK.offset atIndex:1];
    [encoder setBuffer:bufferV offset:viewV.offset atIndex:2];
    [encoder setBuffer:bufferKvK offset:viewKvK.offset atIndex:3];
    [encoder setBuffer:bufferKvV offset:viewKvV.offset atIndex:4];
    [encoder setBytes:&base_uniforms length:sizeof(base_uniforms) atIndex:5];
    [encoder setBytes:&strides length:sizeof(strides) atIndex:6];

    const uint64_t scatter_total =
        (uint64_t)desc->token_count * (uint64_t)desc->num_kv_heads * (uint64_t)desc->head_dim;
    if (scatter_total > 0) {
        NSUInteger elements = (NSUInteger)scatter_total;
        MTLSize grid = MTLSizeMake(elements, 1, 1);
        MTLSize tpg = metal_threads_for_elements(scatter_pipe, elements, UINT32_MAX);
        [encoder dispatchThreads:grid threadsPerThreadgroup:tpg];
    }

    [scatter_pipe release];

    const marmot_uint32_t *meta_data = (const marmot_uint32_t *)token_meta->data;
    bool mixed = false;
    bool have_phase = false;
    bool first_is_decode = true;
    bool current_decode = true;
    size_t boundary = desc->token_count;
    uint32_t max_pos_decode = 0;
    uint32_t max_pos_prefill = 0;
    if (meta_data != nullptr) {
        for (size_t t = 0; t < desc->token_count; ++t) {
            const size_t base = t * meta_stride0;
            uint32_t flags = meta_data[base + 3 * meta_stride1].value;
            uint32_t pos = meta_data[base + 1 * meta_stride1].value;
            bool is_decode = (flags & k_paged_token_flag_decode) != 0;
            bool is_prefill = (flags & k_paged_token_flag_prefill) != 0;
            if (!is_decode && !is_prefill) {
                is_decode = true;
            }
            if (!have_phase) {
                first_is_decode = is_decode;
                current_decode = is_decode;
                have_phase = true;
            }
            if (is_decode != current_decode) {
                if (boundary == desc->token_count) {
                    boundary = t;
                    current_decode = is_decode;
                } else {
                    mixed = true;
                    break;
                }
            }
            if (is_decode) {
                if (pos > max_pos_decode) {
                    max_pos_decode = pos;
                }
            } else {
                if (pos > max_pos_prefill) {
                    max_pos_prefill = pos;
                }
            }
        }
    }

    size_t decode_offset = 0;
    size_t decode_count = desc->token_count;
    size_t prefill_offset = 0;
    size_t prefill_count = 0;
    uint32_t decode_max_pos = max_pos_decode;
    uint32_t prefill_max_pos = max_pos_prefill;
    if (!mixed && have_phase && boundary != desc->token_count) {
        const size_t first_len = boundary;
        const size_t second_len = desc->token_count - boundary;
        if (first_is_decode) {
            decode_offset = 0;
            decode_count = first_len;
            prefill_offset = boundary;
            prefill_count = second_len;
        } else {
            prefill_offset = 0;
            prefill_count = first_len;
            decode_offset = boundary;
            decode_count = second_len;
        }
    } else if (!mixed && have_phase) {
        if (first_is_decode) {
            decode_offset = 0;
            decode_count = desc->token_count;
            prefill_count = 0;
        } else {
            prefill_offset = 0;
            prefill_count = desc->token_count;
            decode_count = 0;
        }
    }

    if (mixed || meta_data == nullptr) {
        decode_offset = 0;
        decode_count = desc->token_count;
        prefill_count = 0;
        decode_max_pos = max_pos_decode > max_pos_prefill ? max_pos_decode : max_pos_prefill;
    }

    auto encode_attention_segment = [&](size_t token_offset, size_t token_count, uint32_t max_pos) -> marmot_error_t {
        if (token_count == 0) {
            return MARMOT_SUCCESS;
        }
        const uint32_t kv_len = max_pos + 1u;
        uint32_t simd_groups = metal_decode_attention_simd_groups(kv_len);
        const char *kernel_name = metal_paged_attention_kernel_name(q->dtype, kv_k->dtype, simd_groups);
        if (kernel_name == nullptr) {
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        id<MTLComputePipelineState> pipe = metal_pipeline_get(ctx, kernel_name);
        if (pipe == nil) {
            return MARMOT_ERROR_BACKEND_INIT_FAILED;
        }
        const NSUInteger required_threads = (NSUInteger)k_metal_decode_attention_simd_width * (NSUInteger)simd_groups;
        if (pipe.maxTotalThreadsPerThreadgroup < required_threads) {
            [pipe release];
            simd_groups = k_metal_decode_attention_simd_groups_small;
            kernel_name = metal_paged_attention_kernel_name(q->dtype, kv_k->dtype, simd_groups);
            pipe = kernel_name != nullptr ? metal_pipeline_get(ctx, kernel_name) : nil;
            if (pipe == nil) {
                return MARMOT_ERROR_BACKEND_INIT_FAILED;
            }
        }

        metal_paged_attention_uniforms_t uniforms = base_uniforms;
        uniforms.token_offset = (uint32_t)token_offset;
        uniforms.token_count = (uint32_t)token_count;

        id<MTLComputeCommandEncoder> enc = metal_command_acquire_compute_encoder(ctx, pipe);
        if (enc == nil) {
            [pipe release];
            return MARMOT_ERROR_BACKEND_INIT_FAILED;
        }

        [enc setBuffer:bufferMeta offset:viewMeta.offset atIndex:0];
        [enc setBuffer:bufferBlock offset:viewBlock.offset atIndex:1];
        [enc setBuffer:bufferQ offset:viewQ.offset atIndex:2];
        [enc setBuffer:bufferKvK offset:viewKvK.offset atIndex:3];
        [enc setBuffer:bufferKvV offset:viewKvV.offset atIndex:4];
        [enc setBuffer:bufferOut offset:viewOut.offset atIndex:5];
        [enc setBytes:&uniforms length:sizeof(uniforms) atIndex:6];
        [enc setBytes:&strides length:sizeof(strides) atIndex:7];

        MTLSize threadgroups = MTLSizeMake((NSUInteger)token_count, (NSUInteger)desc->num_q_heads, 1);
        MTLSize threads = MTLSizeMake(k_metal_decode_attention_simd_width, simd_groups, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads];

        [pipe release];
        return MARMOT_SUCCESS;
    };

    auto encode_flash_segment = [&](size_t token_offset, size_t token_count, uint32_t max_pos) -> marmot_error_t {
        if (token_count == 0) {
            return MARMOT_SUCCESS;
        }
        const uint32_t kv_len = max_pos + 1u;
        if (!metal_should_use_paged_flash_attention(token_count, kv_len, desc->head_dim, q->dtype, kv_k->dtype)) {
            return encode_attention_segment(token_offset, token_count, max_pos);
        }
        const char *kernel_name = metal_paged_flash_attention_kernel_name(q->dtype, kv_k->dtype, desc->head_dim);
        uint32_t block_m = metal_flash_attention_block_m(q->dtype, desc->head_dim);
        uint32_t block_n = metal_paged_flash_attention_block_n(q->dtype, kv_k->dtype, desc->head_dim);
        if (kernel_name == nullptr || block_m == 0 || block_n == 0) {
            return encode_attention_segment(token_offset, token_count, max_pos);
        }
        id<MTLComputePipelineState> pipe = metal_pipeline_get(ctx, kernel_name);
        if (pipe == nil) {
            return MARMOT_ERROR_BACKEND_INIT_FAILED;
        }
        const NSUInteger required_threads = (NSUInteger)block_m * (NSUInteger)block_n;
        if (pipe.maxTotalThreadsPerThreadgroup < required_threads) {
            [pipe release];
            return encode_attention_segment(token_offset, token_count, max_pos);
        }

        metal_paged_attention_uniforms_t uniforms = base_uniforms;
        uniforms.token_offset = (uint32_t)token_offset;
        uniforms.token_count = (uint32_t)token_count;

        id<MTLComputeCommandEncoder> enc = metal_command_acquire_compute_encoder(ctx, pipe);
        if (enc == nil) {
            [pipe release];
            return MARMOT_ERROR_BACKEND_INIT_FAILED;
        }

        [enc setBuffer:bufferMeta offset:viewMeta.offset atIndex:0];
        [enc setBuffer:bufferBlock offset:viewBlock.offset atIndex:1];
        [enc setBuffer:bufferQ offset:viewQ.offset atIndex:2];
        [enc setBuffer:bufferKvK offset:viewKvK.offset atIndex:3];
        [enc setBuffer:bufferKvV offset:viewKvV.offset atIndex:4];
        [enc setBuffer:bufferOut offset:viewOut.offset atIndex:5];
        [enc setBytes:&uniforms length:sizeof(uniforms) atIndex:6];
        [enc setBytes:&strides length:sizeof(strides) atIndex:7];

        uint32_t num_q_blocks = (uint32_t)((token_count + block_m - 1u) / block_m);
        MTLSize threadgroups = MTLSizeMake(num_q_blocks, (NSUInteger)desc->num_q_heads, 1);
        MTLSize threads = MTLSizeMake(block_n, block_m, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads];

        [pipe release];
        return MARMOT_SUCCESS;
    };

    marmot_error_t status = encode_attention_segment(decode_offset, decode_count, decode_max_pos);
    if (status == MARMOT_SUCCESS && prefill_count > 0) {
        if (meta_data == nullptr || mixed) {
            status = encode_attention_segment(prefill_offset, prefill_count, prefill_max_pos);
        } else {
            size_t range_start = prefill_offset;
            uint32_t range_seq = 0;
            uint32_t range_max_pos = 0;
            if (prefill_count > 0) {
                const size_t base = range_start * meta_stride0;
                range_seq = meta_data[base + 0 * meta_stride1].value;
                range_max_pos = meta_data[base + 1 * meta_stride1].value;
            }
            const size_t range_end = prefill_offset + prefill_count;
            for (size_t t = prefill_offset; t < range_end; ++t) {
                const size_t base = t * meta_stride0;
                uint32_t seq = meta_data[base + 0 * meta_stride1].value;
                uint32_t pos = meta_data[base + 1 * meta_stride1].value;
                if (seq != range_seq) {
                    const size_t range_len = t - range_start;
                    status = encode_flash_segment(range_start, range_len, range_max_pos);
                    if (status != MARMOT_SUCCESS) {
                        break;
                    }
                    range_start = t;
                    range_seq = seq;
                    range_max_pos = pos;
                } else if (pos > range_max_pos) {
                    range_max_pos = pos;
                }
            }
            if (status == MARMOT_SUCCESS && range_start < range_end) {
                const size_t range_len = range_end - range_start;
                status = encode_flash_segment(range_start, range_len, range_max_pos);
            }
        }
    }

    metal_profiling_end(ctx);
    metal_command_stream_flush(ctx, false);

    [bufferMeta release];
    [bufferBlock release];
    [bufferQ release];
    [bufferK release];
    [bufferV release];
    [bufferKvK release];
    [bufferKvV release];
    [bufferOut release];

    metal_residency_mark_dirty(ctx, (marmot_tensor_t *)kv_k, kv_k->dtype);
    metal_residency_mark_dirty(ctx, (marmot_tensor_t *)kv_v, kv_v->dtype);
    metal_residency_mark_dirty(ctx, out, out->dtype);

    return status;
}

#endif // __APPLE__
