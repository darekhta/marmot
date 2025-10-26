// SPDX-License-Identifier: MIT
// CPU QKV matmul implementation - handles separate and fused weight layouts,
// quantized and dense weights, RoPE application, and epilogues.

#include "marmot/ops/matmul.h"

#include <stdlib.h>

#include <string.h>

#include "core/tensor/tensor_utils.h"
#include "cpu_backend_internal.h"
#include "matmul_qkv_rope.h"
#include "ops/matmul/matmul_kernels.h"
#include "ops/matmul/quantized/matmul_quant_activation.h"
#include "ops/matmul/quantized/matmul_quant_kernels.h"
#include "quantization/format_metadata.h"

#if HAS_NEON
marmot_error_t cpu_matmul_qkv_neon_run_separate_f32(
    const void *device_ctx, const marmot_matmul_qkv_desc_t *desc, size_t N, size_t K, size_t M
);
#endif

// ============================================================================
// Internal types
// ============================================================================

typedef struct {
    size_t N;
    size_t K;
    size_t M;
} cpu_matmul_qkv_dims_t;

typedef struct {
    const marmot_rope_params_t *params;
    const float *freqs;
    float attn_scale;
    const float *sincos;
    size_t sincos_stride;
    size_t sincos_cached_positions;
    const int32_t *positions_i32;
    const int64_t *positions_i64;
    size_t head_dim;
    bool use_sincos_cache;
    bool owns_buffer;
} cpu_matmul_qkv_rope_state_t;

typedef struct {
    marmot_tensor_t *tensor;
    void *data;
    marmot_dtype_t dtype;
    size_t row_stride;
    size_t col_stride;
    size_t cols;
    bool needs_scratch;
    float *scratch;
} cpu_matmul_qkv_row_buffer_t;

typedef struct {
    marmot_tensor_t tensor;
    bool owns_buffer;
} cpu_matmul_qkv_dense_weight_t;

// ============================================================================
// Forward declarations
// ============================================================================

static bool cpu_matmul_qkv_desc_has_quantized_weights(const marmot_matmul_qkv_desc_t *desc);
static marmot_error_t cpu_matmul_qkv_validate_desc(const marmot_matmul_qkv_desc_t *desc, cpu_matmul_qkv_dims_t *dims);
static marmot_error_t cpu_matmul_qkv_run_fallback(
    const void *device_ctx, const marmot_matmul_qkv_desc_t *desc, const cpu_matmul_qkv_dims_t *dims
);
static marmot_error_t cpu_matmul_qkv_apply_epilogue(
    const void *device_ctx, const marmot_matmul_qkv_desc_t *desc, const cpu_matmul_qkv_dims_t *dims
);

// ============================================================================
// Value load/store helpers
// ============================================================================

static inline float cpu_matmul_qkv_load_value(const void *data, marmot_dtype_t dtype, size_t idx) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return ((const float *)data)[idx];
    case MARMOT_DTYPE_FLOAT16:
        return (float)marmot_float16_to_native(((const marmot_float16_t *)data)[idx]);
    case MARMOT_DTYPE_BFLOAT16:
        return (float)marmot_bfloat16_to_native(((const marmot_bfloat16_t *)data)[idx]);
    default:
        return 0.0f;
    }
}

static inline void cpu_matmul_qkv_store_value(void *data, marmot_dtype_t dtype, size_t idx, float value) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        ((float *)data)[idx] = value;
        break;
    case MARMOT_DTYPE_FLOAT16:
        ((marmot_float16_t *)data)[idx] = marmot_native_to_float16((_Float16)value);
        break;
    case MARMOT_DTYPE_BFLOAT16:
        ((marmot_bfloat16_t *)data)[idx] = marmot_native_to_bfloat16(value);
        break;
    default:
        break;
    }
}

// ============================================================================
// Weight helpers
// ============================================================================

static bool cpu_matmul_qkv_desc_has_quantized_weights(const marmot_matmul_qkv_desc_t *desc) {
    if (desc == nullptr || desc->layout != MARMOT_QKV_LAYOUT_SEPARATE) {
        return false;
    }
    const marmot_tensor_t *wq = desc->separate.wq;
    const marmot_tensor_t *wk = desc->separate.wk;
    const marmot_tensor_t *wv = desc->separate.wv;
    if (wq == nullptr || wk == nullptr || wv == nullptr) {
        return false;
    }
    if (!marmot_tensor_is_block_quantized_weight(wq) || !marmot_tensor_is_block_quantized_weight(wk) ||
        !marmot_tensor_is_block_quantized_weight(wv)) {
        return false;
    }
    const bool kind_match = (wq->quant_kind == wk->quant_kind) && (wq->quant_kind == wv->quant_kind);
    const bool layout_match = (wq->quant_layout == wk->quant_layout) && (wq->quant_layout == wv->quant_layout);
    return kind_match && layout_match;
}

static bool cpu_tensor_is_row_major(const marmot_tensor_t *tensor, marmot_dtype_t dtype, size_t rows, size_t cols) {
    if (tensor == nullptr || tensor->dtype != dtype || tensor->shape.ndim != 2) {
        return false;
    }
    return tensor->shape.shape[0] == rows && tensor->shape.shape[1] == cols && tensor->shape.strides[1] == 1 &&
        tensor->shape.strides[0] == cols;
}

static bool cpu_tensor_is_dense_vector(const marmot_tensor_t *tensor, marmot_dtype_t dtype, size_t length) {
    if (tensor == nullptr) {
        return false;
    }
    if (tensor->shape.ndim != 1 || tensor->shape.shape[0] != length || tensor->shape.strides[0] != 1) {
        return false;
    }
    if (tensor->dtype == dtype) {
        return true;
    }
    if ((dtype == MARMOT_DTYPE_FLOAT16 || dtype == MARMOT_DTYPE_BFLOAT16) && tensor->dtype == MARMOT_DTYPE_FLOAT32) {
        return true;
    }
    return false;
}

static void cpu_matmul_qkv_dense_weight_cleanup(cpu_matmul_qkv_dense_weight_t *weights, size_t count) {
    if (weights == nullptr) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        if (weights[i].owns_buffer && weights[i].tensor.data != nullptr) {
            free(weights[i].tensor.data);
            weights[i].tensor.data = nullptr;
            weights[i].owns_buffer = false;
        }
    }
}

static marmot_error_t cpu_matmul_qkv_prepare_dense_weight(
    const void *device_ctx, const marmot_tensor_t *src, size_t rows, size_t cols,
    cpu_matmul_qkv_dense_weight_t *out_weight
) {
    if (src == nullptr || out_weight == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (src->shape.ndim != 2 || src->shape.shape[0] != rows || src->shape.shape[1] != cols) {
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    if (src->quant_kind == MARMOT_QUANT_KIND_GENERIC) {
        if (src->dtype != MARMOT_DTYPE_FLOAT32 && src->dtype != MARMOT_DTYPE_FLOAT16 &&
            src->dtype != MARMOT_DTYPE_BFLOAT16) {
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        if (!cpu_tensor_is_row_major(src, src->dtype, rows, cols)) {
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        out_weight->tensor = *src;
        out_weight->owns_buffer = false;
        return MARMOT_SUCCESS;
    }

    if (!marmot_tensor_is_block_quantized_weight(src)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const size_t elements = rows * cols;
    float *buffer = (float *)malloc(elements * sizeof(float));
    if (buffer == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    marmot_tensor_t dense = {0};
    dense.shape.ndim = 2;
    dense.shape.shape[0] = rows;
    dense.shape.shape[1] = cols;
    dense.shape.strides[1] = 1;
    dense.shape.strides[0] = cols;
    dense.dtype = MARMOT_DTYPE_FLOAT32;
    dense.data = buffer;
    dense.capacity_bytes = elements * sizeof(float);
    dense.owns_data = true;
    dense.backend = MARMOT_BACKEND_CPU;

    marmot_error_t dequant_status =
        cpu_dequantize_with_kind(device_ctx, src->quant_kind, src->quant_layout, src, &dense);
    if (dequant_status != MARMOT_SUCCESS) {
        free(buffer);
        return dequant_status;
    }

    out_weight->tensor = dense;
    out_weight->owns_buffer = true;
    return MARMOT_SUCCESS;
}

// ============================================================================
// RoPE helpers
// ============================================================================

static marmot_error_t cpu_matmul_qkv_apply_rope_tensor_rows(
    marmot_tensor_t *tensor, size_t rows, size_t dim, size_t head_dim, const marmot_tensor_t *positions,
    const float *freqs, float attn_scale, marmot_rope_type_t rope_type, const float *sincos_base, size_t sincos_stride,
    size_t sincos_cached_positions, const int32_t *positions_i32, const int64_t *positions_i64
) {
    if (tensor == nullptr || positions == nullptr || freqs == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const size_t row_stride = tensor->shape.strides[0];
    const size_t col_stride = tensor->shape.strides[1];
    if (col_stride != 1 || row_stride != dim) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "RoPE fusion requires contiguous QKV outputs");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    switch (tensor->dtype) {
    case MARMOT_DTYPE_FLOAT32: {
        float *data = (float *)tensor->data;
        for (size_t n = 0; n < rows; ++n) {
            const float *sincos = cpu_rope_sincos_lookup(
                sincos_base, sincos_stride, sincos_cached_positions, positions_i32, positions_i64, n
            );
            if (sincos != nullptr) {
                cpu_matmul_qkv_rotate_row_f32_sincos_headed(data + n * row_stride, dim, head_dim, sincos, rope_type);
            } else {
                const float pos = cpu_rope_position_as_f32(positions, n);
                cpu_matmul_qkv_rotate_row_f32_headed(
                    data + n * row_stride, dim, head_dim, pos, freqs, attn_scale, rope_type
                );
            }
        }
        return MARMOT_SUCCESS;
    }
    case MARMOT_DTYPE_FLOAT16: {
        marmot_float16_t *data = (marmot_float16_t *)tensor->data;
        for (size_t n = 0; n < rows; ++n) {
            const float *sincos = cpu_rope_sincos_lookup(
                sincos_base, sincos_stride, sincos_cached_positions, positions_i32, positions_i64, n
            );
            if (sincos != nullptr) {
                cpu_matmul_qkv_rotate_row_f16_sincos_headed(data + n * row_stride, dim, head_dim, sincos, rope_type);
            } else {
                const float pos = cpu_rope_position_as_f32(positions, n);
                cpu_matmul_qkv_rotate_row_f16_headed(
                    data + n * row_stride, dim, head_dim, pos, freqs, attn_scale, rope_type
                );
            }
        }
        return MARMOT_SUCCESS;
    }
    case MARMOT_DTYPE_BFLOAT16: {
        marmot_bfloat16_t *data = (marmot_bfloat16_t *)tensor->data;
        for (size_t n = 0; n < rows; ++n) {
            const float *sincos = cpu_rope_sincos_lookup(
                sincos_base, sincos_stride, sincos_cached_positions, positions_i32, positions_i64, n
            );
            if (sincos != nullptr) {
                cpu_matmul_qkv_rotate_row_bf16_sincos_headed(data + n * row_stride, dim, head_dim, sincos, rope_type);
            } else {
                const float pos = cpu_rope_position_as_f32(positions, n);
                cpu_matmul_qkv_rotate_row_bf16_headed(
                    data + n * row_stride, dim, head_dim, pos, freqs, attn_scale, rope_type
                );
            }
        }
        return MARMOT_SUCCESS;
    }
    default:
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "RoPE fusion unsupported dtype on CPU fallback");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
}

static marmot_error_t cpu_matmul_qkv_apply_rope_tensor_rows_pair(
    marmot_tensor_t *tensor_a, marmot_tensor_t *tensor_b, size_t rows, size_t dim, size_t head_dim,
    const marmot_tensor_t *positions, const float *freqs, float attn_scale, marmot_rope_type_t rope_type,
    const float *sincos_base, size_t sincos_stride, size_t sincos_cached_positions, const int32_t *positions_i32,
    const int64_t *positions_i64
) {
    if (tensor_a == nullptr || tensor_b == nullptr || positions == nullptr || freqs == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (tensor_a->dtype != tensor_b->dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "RoPE fusion requires matching Q/K dtypes");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    const size_t row_stride_a = tensor_a->shape.strides[0];
    const size_t col_stride_a = tensor_a->shape.strides[1];
    const size_t row_stride_b = tensor_b->shape.strides[0];
    const size_t col_stride_b = tensor_b->shape.strides[1];
    if (col_stride_a != 1 || row_stride_a != dim || col_stride_b != 1 || row_stride_b != dim) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "RoPE fusion requires contiguous QKV outputs");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    switch (tensor_a->dtype) {
    case MARMOT_DTYPE_FLOAT32: {
        float *data_a = (float *)tensor_a->data;
        float *data_b = (float *)tensor_b->data;
        for (size_t n = 0; n < rows; ++n) {
            const float *sincos = cpu_rope_sincos_lookup(
                sincos_base, sincos_stride, sincos_cached_positions, positions_i32, positions_i64, n
            );
            if (sincos != nullptr) {
                cpu_matmul_qkv_rotate_row_f32_sincos_headed(
                    data_a + n * row_stride_a, dim, head_dim, sincos, rope_type
                );
                cpu_matmul_qkv_rotate_row_f32_sincos_headed(
                    data_b + n * row_stride_b, dim, head_dim, sincos, rope_type
                );
            } else {
                const float pos = cpu_rope_position_as_f32(positions, n);
                cpu_matmul_qkv_rotate_rows_f32_headed(
                    data_a + n * row_stride_a, data_b + n * row_stride_b, dim, head_dim, pos, freqs, attn_scale,
                    rope_type
                );
            }
        }
        return MARMOT_SUCCESS;
    }
    case MARMOT_DTYPE_FLOAT16: {
        marmot_float16_t *data_a = (marmot_float16_t *)tensor_a->data;
        marmot_float16_t *data_b = (marmot_float16_t *)tensor_b->data;
        for (size_t n = 0; n < rows; ++n) {
            const float *sincos = cpu_rope_sincos_lookup(
                sincos_base, sincos_stride, sincos_cached_positions, positions_i32, positions_i64, n
            );
            if (sincos != nullptr) {
                cpu_matmul_qkv_rotate_row_f16_sincos_headed(
                    data_a + n * row_stride_a, dim, head_dim, sincos, rope_type
                );
                cpu_matmul_qkv_rotate_row_f16_sincos_headed(
                    data_b + n * row_stride_b, dim, head_dim, sincos, rope_type
                );
            } else {
                const float pos = cpu_rope_position_as_f32(positions, n);
                cpu_matmul_qkv_rotate_rows_f16_headed(
                    data_a + n * row_stride_a, data_b + n * row_stride_b, dim, head_dim, pos, freqs, attn_scale,
                    rope_type
                );
            }
        }
        return MARMOT_SUCCESS;
    }
    case MARMOT_DTYPE_BFLOAT16: {
        marmot_bfloat16_t *data_a = (marmot_bfloat16_t *)tensor_a->data;
        marmot_bfloat16_t *data_b = (marmot_bfloat16_t *)tensor_b->data;
        for (size_t n = 0; n < rows; ++n) {
            const float *sincos = cpu_rope_sincos_lookup(
                sincos_base, sincos_stride, sincos_cached_positions, positions_i32, positions_i64, n
            );
            if (sincos != nullptr) {
                cpu_matmul_qkv_rotate_row_bf16_sincos_headed(
                    data_a + n * row_stride_a, dim, head_dim, sincos, rope_type
                );
                cpu_matmul_qkv_rotate_row_bf16_sincos_headed(
                    data_b + n * row_stride_b, dim, head_dim, sincos, rope_type
                );
            } else {
                const float pos = cpu_rope_position_as_f32(positions, n);
                cpu_matmul_qkv_rotate_rows_bf16_headed(
                    data_a + n * row_stride_a, data_b + n * row_stride_b, dim, head_dim, pos, freqs, attn_scale,
                    rope_type
                );
            }
        }
        return MARMOT_SUCCESS;
    }
    default:
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "RoPE fusion unsupported dtype on CPU fallback");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
}

static marmot_error_t cpu_matmul_qkv_apply_rope_outputs(
    const void *device_ctx, const marmot_matmul_qkv_desc_t *desc, const cpu_matmul_qkv_dims_t *dims
) {
    const marmot_rope_params_t *rope = desc->rope_params;
    if (rope == nullptr || (!rope->apply_to_q && !rope->apply_to_k)) {
        return MARMOT_SUCCESS;
    }

    const size_t head_dim = cpu_matmul_qkv_resolve_head_dim(dims->M, rope);
    cpu_context_t *ctx = (cpu_context_t *)device_ctx;
    marmot_rope_freq_span_t span = {0};
    marmot_error_t status =
        marmot_rope_freq_cache_ensure(ctx != nullptr ? &ctx->rope_cache : nullptr, head_dim, rope, &span);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    bool use_sincos_cache = false;
    const float *sincos_base = nullptr;
    size_t sincos_stride = 0;
    size_t sincos_cached_positions = 0;
    const int32_t *positions_i32 = nullptr;
    const int64_t *positions_i64 = nullptr;
    if (ctx != nullptr && rope->positions != nullptr) {
        marmot_error_t cache_status =
            cpu_rope_sincos_cache_ensure(ctx, &span, rope->positions, dims->N, &use_sincos_cache);
        if (cache_status != MARMOT_SUCCESS) {
            if (span.owns_buffer) {
                free((void *)span.freqs);
            }
            return cache_status;
        }
        if (use_sincos_cache) {
            sincos_base = ctx->rope_sincos_cache.sincos;
            sincos_stride = ctx->rope_sincos_cache.pair_count * 2;
            sincos_cached_positions = ctx->rope_sincos_cache.cached_positions;
            if (rope->positions->dtype == MARMOT_DTYPE_INT32) {
                positions_i32 = (const int32_t *)rope->positions->data;
            } else if (rope->positions->dtype == MARMOT_DTYPE_INT64) {
                positions_i64 = (const int64_t *)rope->positions->data;
            }
        }
    }

    if (rope->apply_to_q && rope->apply_to_k) {
        status = cpu_matmul_qkv_apply_rope_tensor_rows_pair(
            desc->out_q, desc->out_k, dims->N, dims->M, head_dim, rope->positions, span.freqs, span.attn_scale,
            rope->rope_type, sincos_base, sincos_stride, sincos_cached_positions, positions_i32, positions_i64
        );
    } else if (rope->apply_to_q) {
        status = cpu_matmul_qkv_apply_rope_tensor_rows(
            desc->out_q, dims->N, dims->M, head_dim, rope->positions, span.freqs, span.attn_scale, rope->rope_type,
            sincos_base, sincos_stride, sincos_cached_positions, positions_i32, positions_i64
        );
    } else if (rope->apply_to_k) {
        status = cpu_matmul_qkv_apply_rope_tensor_rows(
            desc->out_k, dims->N, dims->M, head_dim, rope->positions, span.freqs, span.attn_scale, rope->rope_type,
            sincos_base, sincos_stride, sincos_cached_positions, positions_i32, positions_i64
        );
    }

    if (span.owns_buffer) {
        free((void *)span.freqs);
    }
    return status;
}

static marmot_error_t cpu_matmul_qkv_rope_state_init(
    const void *device_ctx, const marmot_matmul_qkv_desc_t *desc, const cpu_matmul_qkv_dims_t *dims,
    cpu_matmul_qkv_rope_state_t *state
) {
    if (state == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    state->params = nullptr;
    state->freqs = nullptr;
    state->attn_scale = 1.0f;
    state->sincos = nullptr;
    state->sincos_stride = 0;
    state->sincos_cached_positions = 0;
    state->positions_i32 = nullptr;
    state->positions_i64 = nullptr;
    state->head_dim = 0;
    state->use_sincos_cache = false;
    state->owns_buffer = false;

    if (desc == nullptr || dims == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_rope_params_t *rope = desc->rope_params;
    if (rope == nullptr || (!rope->apply_to_q && !rope->apply_to_k)) {
        return MARMOT_SUCCESS;
    }

    const size_t head_dim = cpu_matmul_qkv_resolve_head_dim(dims->M, rope);
    cpu_context_t *ctx = (cpu_context_t *)device_ctx;
    marmot_rope_freq_span_t span = {0};
    marmot_error_t status =
        marmot_rope_freq_cache_ensure(ctx != nullptr ? &ctx->rope_cache : nullptr, head_dim, rope, &span);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    state->params = rope;
    state->freqs = span.freqs;
    state->attn_scale = span.attn_scale;
    state->owns_buffer = span.owns_buffer;
    state->head_dim = head_dim;
    if (ctx != nullptr && rope->positions != nullptr) {
        bool use_sincos_cache = false;
        marmot_error_t cache_status =
            cpu_rope_sincos_cache_ensure(ctx, &span, rope->positions, dims->N, &use_sincos_cache);
        if (cache_status != MARMOT_SUCCESS) {
            if (span.owns_buffer) {
                free((void *)span.freqs);
            }
            state->params = nullptr;
            state->freqs = nullptr;
            state->attn_scale = 1.0f;
            state->owns_buffer = false;
            return cache_status;
        }
        if (use_sincos_cache) {
            state->sincos = ctx->rope_sincos_cache.sincos;
            state->sincos_stride = ctx->rope_sincos_cache.pair_count * 2;
            state->sincos_cached_positions = ctx->rope_sincos_cache.cached_positions;
            if (rope->positions->dtype == MARMOT_DTYPE_INT32) {
                state->positions_i32 = (const int32_t *)rope->positions->data;
            } else if (rope->positions->dtype == MARMOT_DTYPE_INT64) {
                state->positions_i64 = (const int64_t *)rope->positions->data;
            }
            state->use_sincos_cache = true;
        }
    }
    return MARMOT_SUCCESS;
}

static void cpu_matmul_qkv_rope_state_destroy(cpu_matmul_qkv_rope_state_t *state) {
    if (state == nullptr) {
        return;
    }
    if (state->owns_buffer && state->freqs != nullptr) {
        free((void *)state->freqs);
    }
    state->params = nullptr;
    state->freqs = nullptr;
    state->attn_scale = 1.0f;
    state->sincos = nullptr;
    state->sincos_stride = 0;
    state->sincos_cached_positions = 0;
    state->positions_i32 = nullptr;
    state->positions_i64 = nullptr;
    state->head_dim = 0;
    state->use_sincos_cache = false;
    state->owns_buffer = false;
}

// ============================================================================
// Row buffer helpers
// ============================================================================

static marmot_error_t
cpu_matmul_qkv_row_buffer_init(cpu_matmul_qkv_row_buffer_t *buf, marmot_tensor_t *tensor, bool needs_scratch) {
    if (buf == nullptr || tensor == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    buf->tensor = tensor;
    buf->data = tensor->data;
    buf->dtype = tensor->dtype;
    buf->row_stride = tensor->shape.strides[0];
    buf->col_stride = tensor->shape.strides[1];
    buf->cols = tensor->shape.shape[1];
    buf->needs_scratch = needs_scratch;
    buf->scratch = nullptr;

    if (needs_scratch) {
        buf->scratch = (float *)malloc(sizeof(float) * buf->cols);
        if (buf->scratch == nullptr) {
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }
    return MARMOT_SUCCESS;
}

static void cpu_matmul_qkv_row_buffer_cleanup(cpu_matmul_qkv_row_buffer_t *buf) {
    if (buf == nullptr) {
        return;
    }
    free(buf->scratch);
    buf->scratch = nullptr;
}

static float *cpu_matmul_qkv_row_buffer_begin(cpu_matmul_qkv_row_buffer_t *buf, size_t row) {
    if (buf == nullptr) {
        return nullptr;
    }
    if (buf->needs_scratch) {
        return buf->scratch;
    }
    if (buf->dtype == MARMOT_DTYPE_FLOAT32 && buf->col_stride == 1) {
        return (float *)buf->data + row * buf->row_stride;
    }
    return nullptr;
}

static void cpu_matmul_qkv_row_buffer_store(
    cpu_matmul_qkv_row_buffer_t *buf, float *row_data, size_t row, size_t col, float value
) {
    if (buf == nullptr) {
        return;
    }
    if (row_data != nullptr) {
        row_data[col] = value;
        return;
    }
    const size_t offset = row * buf->row_stride + col * buf->col_stride;
    cpu_matmul_qkv_store_value(buf->data, buf->dtype, offset, value);
}

static void
cpu_matmul_qkv_row_buffer_commit(const cpu_matmul_qkv_row_buffer_t *buf, const float *row_data, size_t row) {
    if (buf == nullptr || !buf->needs_scratch || row_data == nullptr) {
        return;
    }
    const size_t cols = buf->cols;
    const size_t base = row * buf->row_stride;
    for (size_t col = 0; col < cols; ++col) {
        const size_t offset = base + col * buf->col_stride;
        cpu_matmul_qkv_store_value(buf->data, buf->dtype, offset, row_data[col]);
    }
}

static marmot_error_t cpu_matmul_qkv_apply_rope_row(
    const cpu_matmul_qkv_rope_state_t *state, float *row_data, size_t dim, size_t row_index, bool apply_rope
) {
    if (!apply_rope) {
        return MARMOT_SUCCESS;
    }
    if (state == nullptr || state->params == nullptr || state->freqs == nullptr || row_data == nullptr) {
        marmot_set_error(
            MARMOT_ERROR_NOT_IMPLEMENTED, "RoPE fusion requires float32 accumulation buffer for the requested tensor"
        );
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const marmot_rope_params_t *rope = state->params;
    const size_t head_dim = state->head_dim > 0 ? state->head_dim : dim;
    const float *sincos = nullptr;
    if (state->use_sincos_cache) {
        sincos = cpu_rope_sincos_lookup(
            state->sincos, state->sincos_stride, state->sincos_cached_positions, state->positions_i32,
            state->positions_i64, row_index
        );
    }
    if (sincos != nullptr) {
        cpu_matmul_qkv_rotate_row_f32_sincos_headed(row_data, dim, head_dim, sincos, rope->rope_type);
    } else {
        const float pos = cpu_rope_position_as_f32(rope->positions, row_index);
        cpu_matmul_qkv_rotate_row_f32_headed(
            row_data, dim, head_dim, pos, state->freqs, state->attn_scale, rope->rope_type
        );
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_matmul_qkv_finalize_row(
    const cpu_matmul_qkv_rope_state_t *state, cpu_matmul_qkv_row_buffer_t *buf, float *row_data, size_t row_index,
    size_t dim, bool apply_rope
) {
    marmot_error_t status = cpu_matmul_qkv_apply_rope_row(state, row_data, dim, row_index, apply_rope);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    cpu_matmul_qkv_row_buffer_commit(buf, row_data, row_index);
    return MARMOT_SUCCESS;
}

// ============================================================================
// Quantized matmul path
// ============================================================================

// Internal implementation that takes a pre-selected kernel config
static marmot_error_t cpu_matmul_qkv_run_separate_quantized_impl(
    const void *device_ctx, const marmot_matmul_qkv_desc_t *desc, const cpu_matmul_qkv_dims_t *dims,
    const cpu_matmul_quant_kernel_t *kernel
) {
    if (device_ctx == nullptr || desc == nullptr || dims == nullptr || kernel == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_tensor_t *input = desc->input;
    const marmot_tensor_t *wq = desc->separate.wq;
    const marmot_tensor_t *wk = desc->separate.wk;
    const marmot_tensor_t *wv = desc->separate.wv;
    const marmot_dtype_t dtype = input->dtype;
    if (dtype != MARMOT_DTYPE_FLOAT32 && dtype != MARMOT_DTYPE_FLOAT16) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const size_t N = dims->N;
    const size_t K = dims->K;
    const size_t M = dims->M;

    if (!cpu_tensor_is_row_major(input, dtype, N, K) || !cpu_tensor_is_row_major(desc->out_q, dtype, N, M) ||
        !cpu_tensor_is_row_major(desc->out_k, dtype, N, M) || !cpu_tensor_is_row_major(desc->out_v, dtype, N, M)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    if (!marmot_tensor_is_block_quantized_weight(wq) || !marmot_tensor_is_block_quantized_weight(wk) ||
        !marmot_tensor_is_block_quantized_weight(wv)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    if (wq->quant_layout != MARMOT_QUANT_LAYOUT_GGUF || wk->quant_layout != MARMOT_QUANT_LAYOUT_GGUF ||
        wv->quant_layout != MARMOT_QUANT_LAYOUT_GGUF) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    if (wq->quant_kind != wk->quant_kind || wq->quant_kind != wv->quant_kind) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    // Kernel config is passed in directly - no runtime lookup needed
    if (dtype == MARMOT_DTYPE_FLOAT16 && !kernel->supports_fp16_input) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    const cpu_quant_format_info_t *format = kernel->format;
    if (format == nullptr || format->kind != wq->quant_kind) {
        format = cpu_quant_format_info(wq->quant_kind);
    }
    if (format == nullptr) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const bool use_q8_k_activation = format->activation_packer == MARMOT_MATMUL_ACTIVATION_PACKER_Q8_K;
    const size_t blocks_per_row = (K + format->block_values - 1) / format->block_values;
    const size_t row_bytes = format->block_bytes * blocks_per_row;
    const uint8_t *wq_bytes = (const uint8_t *)wq->data;
    const uint8_t *wk_bytes = (const uint8_t *)wk->data;
    const uint8_t *wv_bytes = (const uint8_t *)wv->data;

    cpu_matmul_qkv_rope_state_t rope_state = {0};
    const marmot_rope_params_t *rope = desc->rope_params;
    const bool apply_rope_q = rope != nullptr && rope->apply_to_q;
    const bool apply_rope_k = rope != nullptr && rope->apply_to_k;
    marmot_error_t status = MARMOT_SUCCESS;
    if (apply_rope_q || apply_rope_k) {
        status = cpu_matmul_qkv_rope_state_init(device_ctx, desc, dims, &rope_state);
        if (status != MARMOT_SUCCESS) {
            return status;
        }
    }

    const bool q_needs_scratch = apply_rope_q && desc->out_q->dtype != MARMOT_DTYPE_FLOAT32;
    const bool k_needs_scratch = apply_rope_k && desc->out_k->dtype != MARMOT_DTYPE_FLOAT32;
    cpu_matmul_qkv_row_buffer_t q_buf = {0};
    cpu_matmul_qkv_row_buffer_t k_buf = {0};
    cpu_matmul_qkv_row_buffer_t v_buf = {0};

    status = cpu_matmul_qkv_row_buffer_init(&q_buf, desc->out_q, q_needs_scratch);
    if (status != MARMOT_SUCCESS) {
        cpu_matmul_qkv_rope_state_destroy(&rope_state);
        return status;
    }
    status = cpu_matmul_qkv_row_buffer_init(&k_buf, desc->out_k, k_needs_scratch);
    if (status != MARMOT_SUCCESS) {
        cpu_matmul_qkv_row_buffer_cleanup(&q_buf);
        cpu_matmul_qkv_rope_state_destroy(&rope_state);
        return status;
    }
    status = cpu_matmul_qkv_row_buffer_init(&v_buf, desc->out_v, false);
    if (status != MARMOT_SUCCESS) {
        cpu_matmul_qkv_row_buffer_cleanup(&q_buf);
        cpu_matmul_qkv_row_buffer_cleanup(&k_buf);
        cpu_matmul_qkv_rope_state_destroy(&rope_state);
        return status;
    }

    const marmot_tensor_t *biases[3] = {desc->separate.bq, desc->separate.bk, desc->separate.bv};
    const void *bias_data[3] = {
        biases[0] != nullptr ? biases[0]->data : nullptr,
        biases[1] != nullptr ? biases[1]->data : nullptr,
        biases[2] != nullptr ? biases[2]->data : nullptr,
    };
    const marmot_dtype_t bias_dtype[3] = {
        biases[0] != nullptr ? biases[0]->dtype : dtype,
        biases[1] != nullptr ? biases[1]->dtype : dtype,
        biases[2] != nullptr ? biases[2]->dtype : dtype,
    };

    marmot_q8_0_block_t *activation_blocks_q8_0 = nullptr;
    marmot_q8_k_block_t *activation_blocks_q8_k = nullptr;
    if (use_q8_k_activation) {
        activation_blocks_q8_k = (marmot_q8_k_block_t *)malloc(sizeof(marmot_q8_k_block_t) * blocks_per_row);
        if (activation_blocks_q8_k == nullptr) {
            status = MARMOT_ERROR_OUT_OF_MEMORY;
            goto quant_cleanup;
        }
    } else {
        activation_blocks_q8_0 = (marmot_q8_0_block_t *)malloc(sizeof(marmot_q8_0_block_t) * blocks_per_row);
        if (activation_blocks_q8_0 == nullptr) {
            status = MARMOT_ERROR_OUT_OF_MEMORY;
            goto quant_cleanup;
        }
    }

    cpu_matmul_quant_pack_fp32_fn pack_f32_default =
        use_q8_k_activation ? cpu_matmul_quant_pack_q8_k_f32 : cpu_matmul_quant_pack_q8_0_f32;
    cpu_matmul_quant_pack_fp16_fn pack_f16_default =
        use_q8_k_activation ? cpu_matmul_quant_pack_q8_k_f16 : cpu_matmul_quant_pack_q8_0_f16;
    cpu_matmul_quant_pack_fp32_fn pack_f32 =
        kernel->ops.pack_activations_f32 != nullptr ? kernel->ops.pack_activations_f32 : pack_f32_default;
    cpu_matmul_quant_pack_fp16_fn pack_f16 =
        kernel->ops.pack_activations_f16 != nullptr ? kernel->ops.pack_activations_f16 : pack_f16_default;

    const size_t stride_k = input->shape.strides[1];
    const size_t stride_n = input->shape.strides[0];
    const float *input_f32 = (const float *)input->data;
    const marmot_float16_t *input_f16 = (const marmot_float16_t *)input->data;

    for (size_t n = 0; n < N; ++n) {
        void *dst_blocks = use_q8_k_activation ? (void *)activation_blocks_q8_k : (void *)activation_blocks_q8_0;
        if (dtype == MARMOT_DTYPE_FLOAT16) {
            pack_f16(input_f16, stride_k, stride_n, n, K, blocks_per_row, dst_blocks);
        } else {
            pack_f32(input_f32, stride_k, stride_n, n, K, blocks_per_row, dst_blocks);
        }

        float *row_q = cpu_matmul_qkv_row_buffer_begin(&q_buf, n);
        float *row_k = cpu_matmul_qkv_row_buffer_begin(&k_buf, n);
        float *row_v = cpu_matmul_qkv_row_buffer_begin(&v_buf, n);

        for (size_t m = 0; m < M; ++m) {
            float acc_q = bias_data[0] != nullptr ? cpu_matmul_qkv_load_value(bias_data[0], bias_dtype[0], m) : 0.0f;
            float acc_k = bias_data[1] != nullptr ? cpu_matmul_qkv_load_value(bias_data[1], bias_dtype[1], m) : 0.0f;
            float acc_v = bias_data[2] != nullptr ? cpu_matmul_qkv_load_value(bias_data[2], bias_dtype[2], m) : 0.0f;

            const uint8_t *wq_row = wq_bytes + m * row_bytes;
            const uint8_t *wk_row = wk_bytes + m * row_bytes;
            const uint8_t *wv_row = wv_bytes + m * row_bytes;

            if (use_q8_k_activation) {
                if (kernel->ops.dot_q8_k == nullptr) {
                    status = MARMOT_ERROR_NOT_IMPLEMENTED;
                    goto quant_cleanup;
                }
                acc_q += kernel->ops.dot_q8_k(wq_row, activation_blocks_q8_k, blocks_per_row);
                acc_k += kernel->ops.dot_q8_k(wk_row, activation_blocks_q8_k, blocks_per_row);
                acc_v += kernel->ops.dot_q8_k(wv_row, activation_blocks_q8_k, blocks_per_row);
            } else {
                if (kernel->ops.dot_q8_0 == nullptr) {
                    status = MARMOT_ERROR_NOT_IMPLEMENTED;
                    goto quant_cleanup;
                }
                acc_q += kernel->ops.dot_q8_0(wq_row, activation_blocks_q8_0, blocks_per_row);
                acc_k += kernel->ops.dot_q8_0(wk_row, activation_blocks_q8_0, blocks_per_row);
                acc_v += kernel->ops.dot_q8_0(wv_row, activation_blocks_q8_0, blocks_per_row);
            }

            cpu_matmul_qkv_row_buffer_store(&q_buf, row_q, n, m, acc_q);
            cpu_matmul_qkv_row_buffer_store(&k_buf, row_k, n, m, acc_k);
            cpu_matmul_qkv_row_buffer_store(&v_buf, row_v, n, m, acc_v);
        }

        status = cpu_matmul_qkv_finalize_row(&rope_state, &q_buf, row_q, n, M, apply_rope_q);
        if (status != MARMOT_SUCCESS) {
            goto quant_cleanup;
        }
        status = cpu_matmul_qkv_finalize_row(&rope_state, &k_buf, row_k, n, M, apply_rope_k);
        if (status != MARMOT_SUCCESS) {
            goto quant_cleanup;
        }
        cpu_matmul_qkv_row_buffer_commit(&v_buf, row_v, n);
    }

quant_cleanup:
    free(activation_blocks_q8_k);
    free(activation_blocks_q8_0);
    cpu_matmul_qkv_row_buffer_cleanup(&v_buf);
    cpu_matmul_qkv_row_buffer_cleanup(&k_buf);
    cpu_matmul_qkv_row_buffer_cleanup(&q_buf);
    cpu_matmul_qkv_rope_state_destroy(&rope_state);
    return status;
}

// ============================================================================
// Dense matmul path
// ============================================================================

static marmot_error_t cpu_matmul_qkv_run_separate_dense(
    const void *device_ctx, const marmot_matmul_qkv_desc_t *desc, const cpu_matmul_qkv_dims_t *dims
) {
    const size_t N = dims->N;
    const size_t K = dims->K;
    const size_t M = dims->M;
    const marmot_dtype_t dtype = desc->input->dtype;
    if (dtype != MARMOT_DTYPE_FLOAT32 && dtype != MARMOT_DTYPE_FLOAT16 && dtype != MARMOT_DTYPE_BFLOAT16) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (desc->out_q->dtype != dtype || desc->out_k->dtype != dtype || desc->out_v->dtype != dtype) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (!cpu_tensor_is_row_major(desc->input, dtype, N, K) || !cpu_tensor_is_row_major(desc->out_q, dtype, N, M) ||
        !cpu_tensor_is_row_major(desc->out_k, dtype, N, M) || !cpu_tensor_is_row_major(desc->out_v, dtype, N, M)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const marmot_tensor_t *raw_weights[3] = {desc->separate.wq, desc->separate.wk, desc->separate.wv};
    cpu_matmul_qkv_dense_weight_t resolved[3];
    memset(resolved, 0, sizeof(resolved));
    const marmot_tensor_t *weights[3] = {nullptr, nullptr, nullptr};

    for (size_t slice = 0; slice < 3; ++slice) {
        marmot_error_t status =
            cpu_matmul_qkv_prepare_dense_weight(device_ctx, raw_weights[slice], M, K, &resolved[slice]);
        if (status != MARMOT_SUCCESS) {
            cpu_matmul_qkv_dense_weight_cleanup(resolved, 3);
            return status;
        }
        weights[slice] = &resolved[slice].tensor;
        if (!cpu_tensor_is_row_major(weights[slice], weights[slice]->dtype, M, K)) {
            cpu_matmul_qkv_dense_weight_cleanup(resolved, 3);
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
    }

    const marmot_tensor_t *bq = desc->separate.bq;
    const marmot_tensor_t *bk = desc->separate.bk;
    const marmot_tensor_t *bv = desc->separate.bv;
    if ((bq != nullptr && !cpu_tensor_is_dense_vector(bq, dtype, M)) ||
        (bk != nullptr && !cpu_tensor_is_dense_vector(bk, dtype, M)) ||
        (bv != nullptr && !cpu_tensor_is_dense_vector(bv, dtype, M))) {
        cpu_matmul_qkv_dense_weight_cleanup(resolved, 3);
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const void *input = desc->input->data;
    const void *weight_data[3] = {weights[0]->data, weights[1]->data, weights[2]->data};
    const marmot_dtype_t weight_dtype[3] = {weights[0]->dtype, weights[1]->dtype, weights[2]->dtype};
    const void *bias_data[3] = {
        bq != nullptr ? bq->data : nullptr,
        bk != nullptr ? bk->data : nullptr,
        bv != nullptr ? bv->data : nullptr,
    };
    const marmot_dtype_t bias_dtype[3] = {
        bq != nullptr ? bq->dtype : dtype,
        bk != nullptr ? bk->dtype : dtype,
        bv != nullptr ? bv->dtype : dtype,
    };

#if HAS_NEON
    if (dtype == MARMOT_DTYPE_FLOAT32 && weights[0]->dtype == MARMOT_DTYPE_FLOAT32 &&
        weights[1]->dtype == MARMOT_DTYPE_FLOAT32 && weights[2]->dtype == MARMOT_DTYPE_FLOAT32 &&
        (bq == nullptr || bq->dtype == MARMOT_DTYPE_FLOAT32) && (bk == nullptr || bk->dtype == MARMOT_DTYPE_FLOAT32) &&
        (bv == nullptr || bv->dtype == MARMOT_DTYPE_FLOAT32)) {
        marmot_matmul_qkv_desc_t neon_desc = *desc;
        neon_desc.separate.wq = weights[0];
        neon_desc.separate.wk = weights[1];
        neon_desc.separate.wv = weights[2];
        neon_desc.separate.bq = bq;
        neon_desc.separate.bk = bk;
        neon_desc.separate.bv = bv;
        marmot_error_t neon_status = cpu_matmul_qkv_neon_run_separate_f32(device_ctx, &neon_desc, N, K, M);
        if (neon_status == MARMOT_SUCCESS) {
            cpu_matmul_qkv_dense_weight_cleanup(resolved, 3);
            return MARMOT_SUCCESS;
        }
        if (neon_status != MARMOT_ERROR_NOT_IMPLEMENTED) {
            cpu_matmul_qkv_dense_weight_cleanup(resolved, 3);
            return neon_status;
        }
    }
#endif

    cpu_matmul_qkv_rope_state_t rope_state = {0};
    const marmot_rope_params_t *rope = desc->rope_params;
    const bool apply_rope_q = rope != nullptr && rope->apply_to_q;
    const bool apply_rope_k = rope != nullptr && rope->apply_to_k;
    marmot_error_t status = MARMOT_SUCCESS;
    if (apply_rope_q || apply_rope_k) {
        status = cpu_matmul_qkv_rope_state_init(device_ctx, desc, dims, &rope_state);
        if (status != MARMOT_SUCCESS) {
            cpu_matmul_qkv_dense_weight_cleanup(resolved, 3);
            return status;
        }
    }

    const bool q_needs_scratch = apply_rope_q && desc->out_q->dtype != MARMOT_DTYPE_FLOAT32;
    const bool k_needs_scratch = apply_rope_k && desc->out_k->dtype != MARMOT_DTYPE_FLOAT32;

    cpu_matmul_qkv_row_buffer_t q_buf = {0};
    cpu_matmul_qkv_row_buffer_t k_buf = {0};
    cpu_matmul_qkv_row_buffer_t v_buf = {0};

    status = cpu_matmul_qkv_row_buffer_init(&q_buf, desc->out_q, q_needs_scratch);
    if (status != MARMOT_SUCCESS) {
        goto dense_cleanup;
    }
    status = cpu_matmul_qkv_row_buffer_init(&k_buf, desc->out_k, k_needs_scratch);
    if (status != MARMOT_SUCCESS) {
        goto dense_cleanup;
    }
    status = cpu_matmul_qkv_row_buffer_init(&v_buf, desc->out_v, false);
    if (status != MARMOT_SUCCESS) {
        goto dense_cleanup;
    }

    for (size_t n = 0; n < N; ++n) {
        float *row_q = cpu_matmul_qkv_row_buffer_begin(&q_buf, n);
        float *row_k = cpu_matmul_qkv_row_buffer_begin(&k_buf, n);
        float *row_v = cpu_matmul_qkv_row_buffer_begin(&v_buf, n);

        for (size_t m = 0; m < M; ++m) {
            float acc_q = bias_data[0] != nullptr ? cpu_matmul_qkv_load_value(bias_data[0], bias_dtype[0], m) : 0.0f;
            float acc_k = bias_data[1] != nullptr ? cpu_matmul_qkv_load_value(bias_data[1], bias_dtype[1], m) : 0.0f;
            float acc_v = bias_data[2] != nullptr ? cpu_matmul_qkv_load_value(bias_data[2], bias_dtype[2], m) : 0.0f;

            for (size_t k = 0; k < K; ++k) {
                const size_t input_idx = n * K + k;
                const size_t weight_idx = m * K + k;
                const float a = cpu_matmul_qkv_load_value(input, dtype, input_idx);
                const float wq_val = cpu_matmul_qkv_load_value(weight_data[0], weight_dtype[0], weight_idx);
                const float wk_val = cpu_matmul_qkv_load_value(weight_data[1], weight_dtype[1], weight_idx);
                const float wv_val = cpu_matmul_qkv_load_value(weight_data[2], weight_dtype[2], weight_idx);
                acc_q += a * wq_val;
                acc_k += a * wk_val;
                acc_v += a * wv_val;
            }

            cpu_matmul_qkv_row_buffer_store(&q_buf, row_q, n, m, acc_q);
            cpu_matmul_qkv_row_buffer_store(&k_buf, row_k, n, m, acc_k);
            cpu_matmul_qkv_row_buffer_store(&v_buf, row_v, n, m, acc_v);
        }

        status = cpu_matmul_qkv_finalize_row(&rope_state, &q_buf, row_q, n, M, apply_rope_q);
        if (status != MARMOT_SUCCESS) {
            goto dense_cleanup;
        }
        status = cpu_matmul_qkv_finalize_row(&rope_state, &k_buf, row_k, n, M, apply_rope_k);
        if (status != MARMOT_SUCCESS) {
            goto dense_cleanup;
        }
        cpu_matmul_qkv_row_buffer_commit(&v_buf, row_v, n);
    }

    status = MARMOT_SUCCESS;

dense_cleanup:
    cpu_matmul_qkv_row_buffer_cleanup(&v_buf);
    cpu_matmul_qkv_row_buffer_cleanup(&k_buf);
    cpu_matmul_qkv_row_buffer_cleanup(&q_buf);
    cpu_matmul_qkv_rope_state_destroy(&rope_state);
    cpu_matmul_qkv_dense_weight_cleanup(resolved, 3);
    return status;
}

// ============================================================================
// Fused weight path (converts separate to fused)
// ============================================================================

// NOTE: fused-weight path removed; fused layouts are handled by the C-API fallback.

// ============================================================================
// Epilogue handling
// ============================================================================

static marmot_error_t cpu_matmul_qkv_apply_epilogue(
    const void *device_ctx, const marmot_matmul_qkv_desc_t *desc, const cpu_matmul_qkv_dims_t *dims
) {
    (void)device_ctx;
    (void)desc;
    (void)dims;
    return MARMOT_SUCCESS;
}

// ============================================================================
// Validation
// ============================================================================

static marmot_error_t cpu_matmul_qkv_validate_desc(const marmot_matmul_qkv_desc_t *desc, cpu_matmul_qkv_dims_t *dims) {
    if (desc == nullptr || desc->input == nullptr || desc->out_q == nullptr || desc->out_k == nullptr ||
        desc->out_v == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (desc->layout == MARMOT_QKV_LAYOUT_FUSED) {
        if (desc->fused.weight == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fused QKV weight tensor is required for fused layout");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        const marmot_tensor_t *input = desc->input;
        const marmot_tensor_t *weight = desc->fused.weight;
        const marmot_tensor_t *out_q = desc->out_q;
        const marmot_tensor_t *out_k = desc->out_k;
        const marmot_tensor_t *out_v = desc->out_v;
        const marmot_tensor_t *bias = desc->fused.bias;

        if (input->dtype != weight->dtype || input->dtype != out_q->dtype || input->dtype != out_k->dtype ||
            input->dtype != out_v->dtype) {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Fused QKV tensors must share dtype");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }

        if (input->shape.ndim != 2 || weight->shape.ndim != 2 || out_q->shape.ndim != 2 || out_k->shape.ndim != 2 ||
            out_v->shape.ndim != 2) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Fused QKV expects 2D tensors");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }

        const size_t N = input->shape.shape[0];
        const size_t K = input->shape.shape[1];
        const size_t fused_rows = weight->shape.shape[0];
        if (fused_rows % 3 != 0) {
            marmot_set_error(
                MARMOT_ERROR_DIMENSION_MISMATCH, "Fused QKV weight first dimension must be divisible by 3"
            );
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        const size_t M = fused_rows / 3;

        if (weight->shape.shape[1] != K) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Fused QKV weight K dimension mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (out_q->shape.shape[0] != N || out_q->shape.shape[1] != M || out_k->shape.shape[0] != N ||
            out_k->shape.shape[1] != M || out_v->shape.shape[0] != N || out_v->shape.shape[1] != M) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Fused QKV output shape mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (weight->shape.strides[1] != 1) {
            marmot_set_error(
                MARMOT_ERROR_NOT_IMPLEMENTED, "Fused QKV weight tensor must be contiguous in the last axis"
            );
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        if (weight->quant_kind != MARMOT_QUANT_KIND_GENERIC) {
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Quantized fused QKV weights are not supported yet");
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }

        if (bias != nullptr) {
            if (bias->dtype != input->dtype) {
                marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Fused QKV bias dtype must match activations");
                return MARMOT_ERROR_UNSUPPORTED_DTYPE;
            }
            if (bias->shape.ndim != 1 || bias->shape.shape[0] != fused_rows) {
                marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Fused QKV bias must be 1D and match weight rows");
                return MARMOT_ERROR_DIMENSION_MISMATCH;
            }
            if (bias->shape.strides[0] != 1) {
                marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Fused QKV bias must be contiguous");
                return MARMOT_ERROR_NOT_IMPLEMENTED;
            }
        }
        if (desc->epilogue != nullptr && desc->epilogue->bias != nullptr && bias == nullptr) {
            marmot_set_error(
                MARMOT_ERROR_INVALID_ARGUMENT, "Fused QKV epilogue bias must be supplied via the descriptor bias tensor"
            );
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        if (dims != nullptr) {
            dims->N = N;
            dims->K = K;
            dims->M = M;
        }
        return MARMOT_SUCCESS;
    }

    if (desc->layout == MARMOT_QKV_LAYOUT_SEPARATE) {
        const marmot_tensor_t *input = desc->input;
        const marmot_tensor_t *wq = desc->separate.wq;
        const marmot_tensor_t *wk = desc->separate.wk;
        const marmot_tensor_t *wv = desc->separate.wv;
        const marmot_tensor_t *out_q = desc->out_q;
        const marmot_tensor_t *out_k = desc->out_k;
        const marmot_tensor_t *out_v = desc->out_v;

        if (input->dtype != out_q->dtype || input->dtype != out_k->dtype || input->dtype != out_v->dtype) {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "QKV tensors must share dtype");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        if (input->shape.ndim != 2 || wq->shape.ndim != 2 || wk->shape.ndim != 2 || wv->shape.ndim != 2 ||
            out_q->shape.ndim != 2 || out_k->shape.ndim != 2 || out_v->shape.ndim != 2) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Separate QKV expects 2D tensors");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }

        const size_t N = input->shape.shape[0];
        const size_t K = input->shape.shape[1];
        if (wq->shape.shape[1] != K || wk->shape.shape[1] != K || wv->shape.shape[1] != K) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Separate QKV weights must share the input dimension");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (wq->shape.shape[0] != wk->shape.shape[0] || wq->shape.shape[0] != wv->shape.shape[0]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Separate QKV weights must share the output dimension");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        const size_t M = wq->shape.shape[0];

        if (out_q->shape.shape[0] != N || out_q->shape.shape[1] != M || out_k->shape.shape[0] != N ||
            out_k->shape.shape[1] != M || out_v->shape.shape[0] != N || out_v->shape.shape[1] != M) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Separate QKV output shape mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }

        const marmot_tensor_t *biases[3] = {desc->separate.bq, desc->separate.bk, desc->separate.bv};
        for (size_t idx = 0; idx < 3; ++idx) {
            const marmot_tensor_t *bias = biases[idx];
            if (bias == nullptr) {
                continue;
            }
            if (bias->dtype != input->dtype) {
                marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Separate QKV bias dtype must match activations");
                return MARMOT_ERROR_UNSUPPORTED_DTYPE;
            }
            if (bias->shape.ndim != 1 || bias->shape.shape[0] != M) {
                marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Separate QKV bias must be 1D and match rows");
                return MARMOT_ERROR_DIMENSION_MISMATCH;
            }
            if (bias->shape.strides[0] != 1) {
                marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Separate QKV bias must be contiguous");
                return MARMOT_ERROR_NOT_IMPLEMENTED;
            }
        }

        if (desc->epilogue != nullptr && desc->epilogue->bias != nullptr) {
            marmot_set_error(
                MARMOT_ERROR_INVALID_ARGUMENT,
                "Separate QKV epilogue bias must be supplied via the descriptor bias tensors"
            );
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        if (dims != nullptr) {
            dims->N = N;
            dims->K = K;
            dims->M = M;
        }
        return MARMOT_SUCCESS;
    }

    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid QKV layout for CPU backend");
    return MARMOT_ERROR_INVALID_ARGUMENT;
}

// ============================================================================
// Fallback path
// ============================================================================

static marmot_error_t cpu_matmul_qkv_run_fallback(
    const void *device_ctx, const marmot_matmul_qkv_desc_t *desc, const cpu_matmul_qkv_dims_t *dims
) {
    cpu_matmul_qkv_dims_t local_dims = {0};
    const cpu_matmul_qkv_dims_t *resolved_dims = dims;
    if (resolved_dims == nullptr) {
        marmot_error_t status = cpu_matmul_qkv_validate_desc(desc, &local_dims);
        if (status != MARMOT_SUCCESS) {
            return status;
        }
        resolved_dims = &local_dims;
    }

    marmot_tensor_t *outputs[3] = {desc->out_q, desc->out_k, desc->out_v};
    const marmot_matmul_epilogue_t *desc_ep = desc->epilogue;

    if (desc->layout == MARMOT_QKV_LAYOUT_FUSED) {
        const marmot_tensor_t *weight = desc->fused.weight;
        const marmot_tensor_t *bias = desc->fused.bias;
        const bool has_bias = (bias != nullptr);
        const size_t row_stride = weight->shape.strides[0];
        const size_t element_size = marmot_dtype_size(weight->dtype);
        const char *weight_bytes = (const char *)weight->data;

        const char *bias_bytes = has_bias ? (const char *)bias->data : nullptr;
        const size_t bias_stride = has_bias ? bias->shape.strides[0] : 0;
        const size_t bias_elt_size = has_bias ? marmot_dtype_size(bias->dtype) : 0;

        for (size_t slice = 0; slice < 3; ++slice) {
            marmot_tensor_t weight_view = *weight;
            weight_view.shape.shape[0] = resolved_dims->M;
            weight_view.shape.shape[1] = resolved_dims->K;
            weight_view.data = (void *)(weight_bytes + slice * resolved_dims->M * row_stride * element_size);

            marmot_matmul_epilogue_t ep = {.bias = nullptr};
            marmot_tensor_t bias_view = {0};
            if (has_bias) {
                bias_view = *bias;
                bias_view.shape.shape[0] = resolved_dims->M;
                bias_view.data = (void *)(bias_bytes + slice * resolved_dims->M * bias_stride * bias_elt_size);
                ep.bias = &bias_view;
            }

            marmot_matmul_epilogue_t fused_ep = ep;
            const marmot_matmul_epilogue_t *ep_ptr = nullptr;
            if (desc_ep != nullptr) {
                fused_ep = *desc_ep;
                fused_ep.bias = ep.bias != nullptr ? ep.bias : desc_ep->bias;
                ep_ptr = &fused_ep;
            } else if (ep.bias != nullptr) {
                ep_ptr = &ep;
            }

            marmot_error_t status = cpu_matmul_direct(device_ctx, desc->input, &weight_view, ep_ptr, outputs[slice]);
            if (status != MARMOT_SUCCESS) {
                return status;
            }
        }
    } else {
        const marmot_tensor_t *weights[3] = {desc->separate.wq, desc->separate.wk, desc->separate.wv};
        const marmot_tensor_t *biases[3] = {desc->separate.bq, desc->separate.bk, desc->separate.bv};

        for (size_t slice = 0; slice < 3; ++slice) {
            marmot_matmul_epilogue_t ep = {.bias = biases[slice]};

            marmot_matmul_epilogue_t fused_ep = ep;
            const marmot_matmul_epilogue_t *ep_ptr = nullptr;
            if (desc_ep != nullptr) {
                fused_ep = *desc_ep;
                fused_ep.bias = ep.bias != nullptr ? ep.bias : desc_ep->bias;
                ep_ptr = &fused_ep;
            } else if (ep.bias != nullptr) {
                ep_ptr = &ep;
            }

            marmot_error_t status = cpu_matmul_direct(device_ctx, desc->input, weights[slice], ep_ptr, outputs[slice]);
            if (status != MARMOT_SUCCESS) {
                return status;
            }
        }
    }

    if (desc->rope_params != nullptr) {
        marmot_error_t rope_status = cpu_matmul_qkv_apply_rope_outputs(device_ctx, desc, resolved_dims);
        if (rope_status != MARMOT_SUCCESS) {
            return rope_status;
        }
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_matmul_qkv_dense_entry(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc, cpu_matmul_qkv_dims_t *dims) {
    if (device_ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (desc->layout != MARMOT_QKV_LAYOUT_SEPARATE) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (cpu_matmul_qkv_desc_has_quantized_weights(desc)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    marmot_error_t status = cpu_matmul_qkv_run_separate_dense(device_ctx, desc, dims);
    if (status == MARMOT_SUCCESS) {
        return cpu_matmul_qkv_apply_epilogue(device_ctx, desc, dims);
    }
    if (status == MARMOT_ERROR_NOT_IMPLEMENTED || status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        return cpu_matmul_qkv_run_fallback(device_ctx, desc, dims);
    }
    return status;
}

static marmot_error_t cpu_matmul_qkv_quant_entry(
    const void *device_ctx, const marmot_matmul_qkv_desc_t *desc, cpu_matmul_qkv_dims_t *dims, marmot_quant_kind_t kind
) {
    if (device_ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (desc->layout != MARMOT_QKV_LAYOUT_SEPARATE) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (!cpu_matmul_qkv_desc_has_quantized_weights(desc) || desc->separate.wq->quant_kind != kind) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const cpu_matmul_quant_kernel_t *kernel = cpu_matmul_quant_select_kernel(device_ctx, kind);
    if (kernel == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "No kernel for quant kind");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    marmot_error_t status = cpu_matmul_qkv_run_separate_quantized_impl(device_ctx, desc, dims, kernel);
    if (status == MARMOT_SUCCESS) {
        return cpu_matmul_qkv_apply_epilogue(device_ctx, desc, dims);
    }
    if (status == MARMOT_ERROR_NOT_IMPLEMENTED || status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        return cpu_matmul_qkv_run_fallback(device_ctx, desc, dims);
    }
    return status;
}

marmot_error_t cpu_matmul_qkv_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    cpu_matmul_qkv_dims_t dims = {0};
    marmot_error_t status = cpu_matmul_qkv_validate_desc(desc, &dims);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return cpu_matmul_qkv_dense_entry(device_ctx, desc, &dims);
}

marmot_error_t cpu_matmul_qkv_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    cpu_matmul_qkv_dims_t dims = {0};
    marmot_error_t status = cpu_matmul_qkv_validate_desc(desc, &dims);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return cpu_matmul_qkv_dense_entry(device_ctx, desc, &dims);
}

marmot_error_t cpu_matmul_qkv_bf16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    cpu_matmul_qkv_dims_t dims = {0};
    marmot_error_t status = cpu_matmul_qkv_validate_desc(desc, &dims);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return cpu_matmul_qkv_dense_entry(device_ctx, desc, &dims);
}

marmot_error_t cpu_matmul_qkv_q2_k(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    cpu_matmul_qkv_dims_t dims = {0};
    marmot_error_t status = cpu_matmul_qkv_validate_desc(desc, &dims);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return cpu_matmul_qkv_quant_entry(device_ctx, desc, &dims, MARMOT_QUANT_KIND_Q2_K);
}

marmot_error_t cpu_matmul_qkv_q3_k(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    cpu_matmul_qkv_dims_t dims = {0};
    marmot_error_t status = cpu_matmul_qkv_validate_desc(desc, &dims);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return cpu_matmul_qkv_quant_entry(device_ctx, desc, &dims, MARMOT_QUANT_KIND_Q3_K);
}

marmot_error_t cpu_matmul_qkv_q4_k(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    cpu_matmul_qkv_dims_t dims = {0};
    marmot_error_t status = cpu_matmul_qkv_validate_desc(desc, &dims);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return cpu_matmul_qkv_quant_entry(device_ctx, desc, &dims, MARMOT_QUANT_KIND_Q4_K);
}

marmot_error_t cpu_matmul_qkv_q5_k(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    cpu_matmul_qkv_dims_t dims = {0};
    marmot_error_t status = cpu_matmul_qkv_validate_desc(desc, &dims);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return cpu_matmul_qkv_quant_entry(device_ctx, desc, &dims, MARMOT_QUANT_KIND_Q5_K);
}

marmot_error_t cpu_matmul_qkv_q6_k(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    cpu_matmul_qkv_dims_t dims = {0};
    marmot_error_t status = cpu_matmul_qkv_validate_desc(desc, &dims);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return cpu_matmul_qkv_quant_entry(device_ctx, desc, &dims, MARMOT_QUANT_KIND_Q6_K);
}

marmot_error_t cpu_matmul_qkv_q8_0(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    cpu_matmul_qkv_dims_t dims = {0};
    marmot_error_t status = cpu_matmul_qkv_validate_desc(desc, &dims);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return cpu_matmul_qkv_quant_entry(device_ctx, desc, &dims, MARMOT_QUANT_KIND_Q8_0);
}

marmot_error_t cpu_matmul_qkv_q8_k(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    cpu_matmul_qkv_dims_t dims = {0};
    marmot_error_t status = cpu_matmul_qkv_validate_desc(desc, &dims);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return cpu_matmul_qkv_quant_entry(device_ctx, desc, &dims, MARMOT_QUANT_KIND_Q8_K);
}
