#include "core/helpers/quant.h"
#include "metal_backend_internal.h"

#ifdef __APPLE__

#include <dispatch/dispatch.h>
#include <limits.h>
#include <string.h>

static bool metal_embedding_is_block_quant(const marmot_tensor_t *weights) {
    if (weights == nullptr)
        return false;
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(weights->quant_kind);
    if (traits == nullptr)
        return false;
    if (!traits->is_block_quantized)
        return false;
    if (!marmot_quant_storage_dtype_compatible(traits, weights->dtype))
        return false;
    if (weights->quant_layout != traits->layout)
        return false;
    if (weights->shape.ndim != 2)
        return false;
    return true;
}

typedef struct {
    marmot_dtype_t weight_dtype;
    marmot_dtype_t out_dtype;
    const char *kernel_name;
} metal_embedding_dense_entry_t;

typedef struct {
    marmot_quant_kind_t quant_kind;
    marmot_dtype_t out_dtype;
    const char *kernel_name;
} metal_embedding_quant_entry_t;

static const metal_embedding_dense_entry_t k_metal_embedding_dense_entries[] = {
    {MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT32, "embedding_gather_f32_to_f32"},
    {MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT16, "embedding_gather_f32_to_f16"},
    {MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_BFLOAT16, "embedding_gather_f32_to_bf16"},
    {MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT32, "embedding_gather_f16_to_f32"},
    {MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT16, "embedding_gather_f16_to_f16"},
    {MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_BFLOAT16, "embedding_gather_f16_to_bf16"},
    {MARMOT_DTYPE_BFLOAT16, MARMOT_DTYPE_FLOAT32, "embedding_gather_bf16_to_f32"},
    {MARMOT_DTYPE_BFLOAT16, MARMOT_DTYPE_FLOAT16, "embedding_gather_bf16_to_f16"},
    {MARMOT_DTYPE_BFLOAT16, MARMOT_DTYPE_BFLOAT16, "embedding_gather_bf16_to_bf16"},
};

static const metal_embedding_quant_entry_t k_metal_embedding_quant_entries[] = {
    {MARMOT_QUANT_KIND_Q4_0, MARMOT_DTYPE_FLOAT32, "embedding_gather_q4_0_to_f32"},
    {MARMOT_QUANT_KIND_Q4_0, MARMOT_DTYPE_FLOAT16, "embedding_gather_q4_0_to_f16_opt"},
    {MARMOT_QUANT_KIND_Q4_0, MARMOT_DTYPE_BFLOAT16, "embedding_gather_q4_0_to_bf16"},
    {MARMOT_QUANT_KIND_Q4_1, MARMOT_DTYPE_FLOAT32, "embedding_gather_q4_1_to_f32"},
    {MARMOT_QUANT_KIND_Q4_1, MARMOT_DTYPE_FLOAT16, "embedding_gather_q4_1_to_f16_opt"},
    {MARMOT_QUANT_KIND_Q4_1, MARMOT_DTYPE_BFLOAT16, "embedding_gather_q4_1_to_bf16"},
    {MARMOT_QUANT_KIND_Q5_0, MARMOT_DTYPE_FLOAT32, "embedding_gather_q5_0_to_f32"},
    {MARMOT_QUANT_KIND_Q5_0, MARMOT_DTYPE_FLOAT16, "embedding_gather_q5_0_to_f16_opt"},
    {MARMOT_QUANT_KIND_Q5_0, MARMOT_DTYPE_BFLOAT16, "embedding_gather_q5_0_to_bf16"},
    {MARMOT_QUANT_KIND_Q5_1, MARMOT_DTYPE_FLOAT32, "embedding_gather_q5_1_to_f32"},
    {MARMOT_QUANT_KIND_Q5_1, MARMOT_DTYPE_FLOAT16, "embedding_gather_q5_1_to_f16_opt"},
    {MARMOT_QUANT_KIND_Q5_1, MARMOT_DTYPE_BFLOAT16, "embedding_gather_q5_1_to_bf16"},
    {MARMOT_QUANT_KIND_Q8_0, MARMOT_DTYPE_FLOAT32, "embedding_gather_q8_0_to_f32"},
    {MARMOT_QUANT_KIND_Q8_0, MARMOT_DTYPE_FLOAT16, "embedding_gather_q8_0_to_f16_opt"},
    {MARMOT_QUANT_KIND_Q8_0, MARMOT_DTYPE_BFLOAT16, "embedding_gather_q8_0_to_bf16"},
    {MARMOT_QUANT_KIND_Q2_K, MARMOT_DTYPE_FLOAT32, "embedding_gather_q2_k_to_f32"},
    {MARMOT_QUANT_KIND_Q2_K, MARMOT_DTYPE_FLOAT16, "embedding_gather_q2_k_to_f16"},
    {MARMOT_QUANT_KIND_Q2_K, MARMOT_DTYPE_BFLOAT16, "embedding_gather_q2_k_to_bf16"},
    {MARMOT_QUANT_KIND_Q3_K, MARMOT_DTYPE_FLOAT32, "embedding_gather_q3_k_to_f32"},
    {MARMOT_QUANT_KIND_Q3_K, MARMOT_DTYPE_FLOAT16, "embedding_gather_q3_k_to_f16"},
    {MARMOT_QUANT_KIND_Q3_K, MARMOT_DTYPE_BFLOAT16, "embedding_gather_q3_k_to_bf16"},
    {MARMOT_QUANT_KIND_Q4_K, MARMOT_DTYPE_FLOAT32, "embedding_gather_q4_k_to_f32"},
    {MARMOT_QUANT_KIND_Q4_K, MARMOT_DTYPE_FLOAT16, "embedding_gather_q4_k_to_f16"},
    {MARMOT_QUANT_KIND_Q4_K, MARMOT_DTYPE_BFLOAT16, "embedding_gather_q4_k_to_bf16"},
    {MARMOT_QUANT_KIND_Q5_K, MARMOT_DTYPE_FLOAT32, "embedding_gather_q5_k_to_f32"},
    {MARMOT_QUANT_KIND_Q5_K, MARMOT_DTYPE_FLOAT16, "embedding_gather_q5_k_to_f16"},
    {MARMOT_QUANT_KIND_Q5_K, MARMOT_DTYPE_BFLOAT16, "embedding_gather_q5_k_to_bf16"},
    {MARMOT_QUANT_KIND_Q6_K, MARMOT_DTYPE_FLOAT32, "embedding_gather_q6_k_to_f32"},
    {MARMOT_QUANT_KIND_Q6_K, MARMOT_DTYPE_FLOAT16, "embedding_gather_q6_k_to_f16"},
    {MARMOT_QUANT_KIND_Q6_K, MARMOT_DTYPE_BFLOAT16, "embedding_gather_q6_k_to_bf16"},
    {MARMOT_QUANT_KIND_Q8_K, MARMOT_DTYPE_FLOAT32, "embedding_gather_q8_k_to_f32"},
    {MARMOT_QUANT_KIND_Q8_K, MARMOT_DTYPE_FLOAT16, "embedding_gather_q8_k_to_f16"},
    {MARMOT_QUANT_KIND_Q8_K, MARMOT_DTYPE_BFLOAT16, "embedding_gather_q8_k_to_bf16"},
};

static const char *k_metal_embedding_dense_kernels[MARMOT_DTYPE_COUNT][MARMOT_DTYPE_COUNT];
static const char *k_metal_embedding_quant_kernels[MARMOT_QUANT_KIND_COUNT][MARMOT_DTYPE_COUNT];
static dispatch_once_t g_metal_embedding_tables_once;

static void metal_embedding_build_tables(void) {
    for (size_t i = 0; i < sizeof(k_metal_embedding_dense_entries) / sizeof(k_metal_embedding_dense_entries[0]); ++i) {
        const metal_embedding_dense_entry_t *entry = &k_metal_embedding_dense_entries[i];
        if (entry->weight_dtype < MARMOT_DTYPE_COUNT && entry->out_dtype < MARMOT_DTYPE_COUNT) {
            k_metal_embedding_dense_kernels[entry->weight_dtype][entry->out_dtype] = entry->kernel_name;
        }
    }
    for (size_t i = 0; i < sizeof(k_metal_embedding_quant_entries) / sizeof(k_metal_embedding_quant_entries[0]); ++i) {
        const metal_embedding_quant_entry_t *entry = &k_metal_embedding_quant_entries[i];
        if (entry->quant_kind < MARMOT_QUANT_KIND_COUNT && entry->out_dtype < MARMOT_DTYPE_COUNT) {
            k_metal_embedding_quant_kernels[entry->quant_kind][entry->out_dtype] = entry->kernel_name;
        }
    }
}

static inline void metal_embedding_ensure_tables(void) {
    dispatch_once(&g_metal_embedding_tables_once, ^{
      metal_embedding_build_tables();
    });
}

static const char *metal_embedding_dense_kernel(marmot_dtype_t weight_dtype, marmot_dtype_t out_dtype) {
    metal_embedding_ensure_tables();
    if (weight_dtype >= MARMOT_DTYPE_COUNT || out_dtype >= MARMOT_DTYPE_COUNT) {
        return nullptr;
    }
    return k_metal_embedding_dense_kernels[weight_dtype][out_dtype];
}

static const char *metal_embedding_quant_kernel(marmot_quant_kind_t kind, marmot_dtype_t out_dtype) {
    metal_embedding_ensure_tables();
    if (kind >= MARMOT_QUANT_KIND_COUNT || out_dtype >= MARMOT_DTYPE_COUNT) {
        return nullptr;
    }
    return k_metal_embedding_quant_kernels[kind][out_dtype];
}

static const char *kernel_resolve(const marmot_tensor_t *w, marmot_dtype_t odt, bool *is_quant) {
    bool iq = metal_embedding_is_block_quant(w);
    if (is_quant != nullptr) {
        *is_quant = iq;
    }
    return iq ? metal_embedding_quant_kernel(w->quant_kind, odt) : metal_embedding_dense_kernel(w->dtype, odt);
}

static bool read_token(const marmot_tensor_t *tok, size_t i, int32_t *out) {
    switch (tok->dtype) {
    case MARMOT_DTYPE_INT32:
        *out = ((const int32_t *)tok->data)[i];
        return true;
    case MARMOT_DTYPE_UINT32:
        *out = (int32_t)((const uint32_t *)tok->data)[i];
        return true;
    case MARMOT_DTYPE_INT16:
        *out = (int32_t)((const int16_t *)tok->data)[i];
        return true;
    case MARMOT_DTYPE_UINT16:
        *out = (int32_t)((const uint16_t *)tok->data)[i];
        return true;
    default:
        return false;
    }
}

static marmot_error_t build_id_buffer(
    metal_context_t *ctx, const marmot_tensor_t *tok, size_t n, int32_t padding_id, bool bounds_check, size_t vocab,
    id<MTLBuffer> *out_ids
) {
    if (ctx == nullptr || tok == nullptr || out_ids == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null arguments to build_id_buffer");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (n == 0) {
        *out_ids = nil;
        return MARMOT_SUCCESS;
    }
    if (n > (SIZE_MAX / sizeof(int32_t))) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Embedding id buffer size overflow");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t bytes = n * sizeof(int32_t);
    const bool reuse_allowed = !ctx->has_in_flight_work;
    id<MTLBuffer> buffer = nil;

    if (reuse_allowed && ctx->embedding_ids_buffer != nil && ctx->embedding_ids_capacity >= bytes) {
        buffer = [ctx->embedding_ids_buffer retain];
    } else if (reuse_allowed) {
        if (ctx->embedding_ids_buffer != nil) {
            [ctx->embedding_ids_buffer release];
            ctx->embedding_ids_buffer = nil;
            ctx->embedding_ids_capacity = 0;
        }
        id<MTLBuffer> new_buffer = [ctx->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (new_buffer == nil) {
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        ctx->embedding_ids_buffer = [new_buffer retain];
        ctx->embedding_ids_capacity = bytes;
        buffer = new_buffer;
    } else {
        buffer = [ctx->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }

    int32_t *ids = (int32_t *)[buffer contents];
    if (ids == nullptr) {
        [buffer release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < n; ++i) {
        int32_t v = 0;
        if (!read_token(tok, i, &v)) {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unsupported token dtype for Metal embedding");
            [buffer release];
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        if (padding_id != -1 && v == padding_id) {
            ids[i] = -1;
        } else if (v < 0 || (size_t)v >= vocab) {
            if (bounds_check) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Token id out of range during embedding lookup");
                [buffer release];
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            ids[i] = -1;
        } else {
            ids[i] = v;
        }
    }

    *out_ids = buffer;
    return MARMOT_SUCCESS;
}

marmot_error_t metal_embedding_gather(const void *device_ctx, const marmot_embedding_gather_desc_t *desc) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (!ctx || !desc || !desc->weights || !desc->token_ids || !desc->out)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    const marmot_tensor_t *w = desc->weights;
    const marmot_tensor_t *tok = desc->token_ids;
    marmot_tensor_t *out = desc->out;

    if (w->shape.ndim != 2)
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    size_t vocab = w->shape.shape[0];
    size_t dim = w->shape.shape[1];
    size_t n = marmot_tensor_num_elements(tok);
    if (n == 0 || dim == 0)
        return MARMOT_SUCCESS;
    if (dim > UINT32_MAX || n > UINT32_MAX)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    // Flatten token ids to 1D view if necessary
    marmot_tensor_t tok_view;
    if (tok->shape.ndim != 1) {
        memset(&tok_view, 0, sizeof(tok_view));
        memcpy(&tok_view, tok, sizeof(tok_view));
        tok_view.shape.ndim = 1;
        tok_view.shape.shape[0] = n;
        tok_view.shape.strides[0] = 1;
        tok = &tok_view;
    }

    // Reshape output to [n, dim] if needed
    marmot_tensor_t out_view;
    if (out->shape.ndim != 2 || out->shape.shape[1] != dim) {
        memset(&out_view, 0, sizeof(out_view));
        memcpy(&out_view, out, sizeof(out_view));
        out_view.shape.ndim = 2;
        out_view.shape.shape[0] = n;
        out_view.shape.shape[1] = dim;
        out_view.shape.strides[1] = 1;
        out_view.shape.strides[0] = dim;
        out = &out_view;
    }

    const int32_t padding_id = desc->padding_id;
    const bool bounds_check = desc->bounds_check;
    const float scale = desc->scale;

    bool tok_stale_on_host = tok->needs_sync && tok->memory_location == MARMOT_MEMORY_DEVICE;
    const bool tok_use_device_ids =
        tok_stale_on_host && tok->dtype == MARMOT_DTYPE_INT32 && padding_id == -1 && !bounds_check;
    if (tok_stale_on_host && !tok_use_device_ids) {
        marmot_error_t sync_err = metal_memcpy_from_device(ctx, tok->data, tok->data, marmot_tensor_size_bytes(tok));
        if (sync_err != MARMOT_SUCCESS) {
            return sync_err;
        }
        tok_stale_on_host = false;
    }

    if (!metal_embedding_is_block_quant(w) && w->dtype == MARMOT_DTYPE_FLOAT32 && out->dtype == MARMOT_DTYPE_FLOAT32 &&
        tok->dtype == MARMOT_DTYPE_INT32 && !tok_stale_on_host) {
        const float *w_ptr = (const float *)w->data;
        const int32_t *ids = (const int32_t *)tok->data;
        float *out_ptr = (float *)out->data;
        for (size_t i = 0; i < n; ++i) {
            int32_t v = ids[i];
            if (padding_id != -1 && v == padding_id) {
                memset(out_ptr + i * dim, 0, dim * sizeof(float));
                continue;
            }
            if (v < 0 || (size_t)v >= vocab) {
                if (bounds_check) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Token id out of range during embedding lookup");
                    return MARMOT_ERROR_INVALID_ARGUMENT;
                }
                memset(out_ptr + i * dim, 0, dim * sizeof(float));
                continue;
            }
            const float *src = w_ptr + (size_t)v * dim;
            float *dst = out_ptr + i * dim;
            if (scale == 1.0f) {
                memcpy(dst, src, dim * sizeof(float));
            } else {
                for (size_t d = 0; d < dim; ++d) {
                    dst[d] = src[d] * scale;
                }
            }
        }
        metal_residency_invalidate(ctx, out->data);
        return MARMOT_SUCCESS;
    }

    bool is_quant = false;
    const char *kname = kernel_resolve(w, out->dtype, &is_quant);
    if (!kname)
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    if (is_quant && w->quant_layout != MARMOT_QUANT_LAYOUT_GGUF)
        return MARMOT_ERROR_NOT_IMPLEMENTED;

    uint32_t blocks_per_row_u32 = 0;
    if (is_quant) {
        const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(w->quant_kind);
        if (traits == nullptr || traits->block_values == 0) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        blocks_per_row_u32 = (uint32_t)((dim + traits->block_values - 1u) / traits->block_values);
    }

    int32_t single_id = 0;
    bool use_single_id = false;
    id<MTLBuffer> bufIds = nil;
    if (tok_stale_on_host) {
        const size_t bytesTok = marmot_tensor_size_bytes(tok);
        bufIds = metal_residency_acquire_existing(ctx, tok, tok->dtype);
        if (bufIds == nil) {
            bufIds = metal_buffer_acquire(ctx, tok->data, bytesTok);
        }
        if (bufIds == nil) {
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    } else if (n == 1) {
        int32_t v = 0;
        if (!read_token(tok, 0, &v)) {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unsupported token dtype for Metal embedding");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        if (padding_id != -1 && v == padding_id) {
            single_id = -1;
        } else if (v < 0 || (size_t)v >= vocab) {
            if (bounds_check) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Token id out of range during embedding lookup");
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            single_id = -1;
        } else {
            single_id = v;
        }
        use_single_id = true;
    } else {
        marmot_error_t e = build_id_buffer(ctx, tok, n, padding_id, bounds_check, vocab, &bufIds);
        if (e != MARMOT_SUCCESS) {
            return e;
        }
    }

    size_t bytesW = marmot_tensor_size_bytes(w);
    size_t bytesOut = marmot_tensor_size_bytes(out);

    id<MTLBuffer> bufW = nil;
    const bool prefer_gpu_private = marmot_preference_resolve(desc->prefer_gpu_private, false);
    if (prefer_gpu_private) {
        bufW = metal_residency_acquire_existing(ctx, w, w->dtype);
        if (bufW == nil) {
            bool staged_new = false;
            bufW = metal_residency_acquire_compute(ctx, w, w->dtype, &staged_new);
        }
        if (bufW == nil) {
            bufW = metal_buffer_acquire(ctx, w->data, bytesW);
        }
    } else {
        bufW = metal_residency_acquire_existing(ctx, w, w->dtype);
        if (bufW == nil) {
            bufW = metal_buffer_acquire(ctx, w->data, bytesW);
        }
    }
    bool out_is_new = false;
    id<MTLBuffer> bufOut = metal_residency_acquire_compute(ctx, out, out->dtype, &out_is_new);
    if (bufOut == nil)
        bufOut = metal_buffer_acquire(ctx, out->data, bytesOut);
    if (bufW == nil || bufOut == nil) {
        if (bufW)
            [bufW release];
        if (bufOut)
            [bufOut release];
        if (!use_single_id && bufIds != nil)
            [bufIds release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputePipelineState> pipe = metal_pipeline_get(ctx, kname);
    if (pipe == nil) {
        [bufW release];
        [bufOut release];
        if (!use_single_id && bufIds != nil)
            [bufIds release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> enc = metal_command_acquire_compute_encoder(ctx, pipe);
    if (enc == nil) {
        [pipe release];
        [bufW release];
        [bufOut release];
        if (!use_single_id && bufIds != nil)
            [bufIds release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    uint32_t dim_u32 = (uint32_t)dim;
    uint32_t rows_u32 = (uint32_t)n;
    [enc setBuffer:bufW offset:0 atIndex:0];
    if (use_single_id) {
        [enc setBytes:&single_id length:sizeof(single_id) atIndex:1];
    } else {
        [enc setBuffer:bufIds offset:0 atIndex:1];
    }
    [enc setBuffer:bufOut offset:0 atIndex:2];
    [enc setBytes:&dim_u32 length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&rows_u32 length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&scale length:sizeof(float) atIndex:5];

    if (is_quant) {
        [enc setBytes:&blocks_per_row_u32 length:sizeof(uint32_t) atIndex:6];
    }

    bool use_opt = (strstr(kname, "_opt") != nullptr);
    if (use_opt) {
        uint64_t blocks = (uint64_t)rows_u32 * (uint64_t)blocks_per_row_u32;
        if (blocks > 0) {
            MTLSize tg = MTLSizeMake(32, 1, 1);
            MTLSize grid = MTLSizeMake((NSUInteger)(blocks * 32ull), 1, 1);
            metal_profiling_set_label(ctx, "embedding");
            metal_profiling_begin(ctx);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            metal_profiling_end(ctx);
        }
    } else {
        uint64_t total = (uint64_t)dim * (uint64_t)n;
        if (total > 0) {
            NSUInteger elements = (NSUInteger)total;
            MTLSize grid = MTLSizeMake(elements, 1, 1);
            MTLSize tg = metal_threads_for_elements(pipe, elements, UINT32_MAX);
            metal_profiling_set_label(ctx, "embedding");
            metal_profiling_begin(ctx);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            metal_profiling_end(ctx);
        }
    }

    metal_command_stream_flush(ctx, false);

    [pipe release];
    [bufW release];
    [bufOut release];
    if (!use_single_id && bufIds != nil)
        [bufIds release];
    metal_residency_mark_dirty(ctx, out, out->dtype);
    return MARMOT_SUCCESS;
}

#endif // __APPLE__
