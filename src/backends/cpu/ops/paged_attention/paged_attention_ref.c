#include "marmot/dispatch.h"
#include "marmot/error.h"

#include <stdlib.h>

#include <math.h>
#include <string.h>

#include "cpu_backend_internal.h"
#include "ops/paged_attention/paged_attention_kernels.h"

static constexpr size_t CPU_PAGED_ATTENTION_MIN_BLOCK_N = 32;
static constexpr size_t CPU_PAGED_ATTENTION_MAX_BLOCK_N = 256;
static constexpr size_t CPU_PAGED_ATTENTION_TARGET_TILE_BYTES = 64 * 1024;
static constexpr uint32_t CPU_PAGED_ATTENTION_PREFETCH_DISTANCE = 4;
static constexpr float CPU_PAGED_ATTENTION_FP8_E4M3_MAX = 240.0f;

static bool cpu_paged_attention_is_power_of_two_u32(uint32_t value) {
    return value != 0u && (value & (value - 1u)) == 0u;
}

static uint32_t cpu_paged_attention_log2_u32(uint32_t value) {
    uint32_t shift = 0;
    while (value > 1u) {
        value >>= 1u;
        shift++;
    }
    return shift;
}

static size_t cpu_paged_attention_choose_block_n(size_t head_dim) {
    if (head_dim == 0) {
        return CPU_PAGED_ATTENTION_MIN_BLOCK_N;
    }
    const size_t bytes_per_row = head_dim * sizeof(float);
    if (bytes_per_row == 0) {
        return CPU_PAGED_ATTENTION_MIN_BLOCK_N;
    }
    size_t rows = CPU_PAGED_ATTENTION_TARGET_TILE_BYTES / (2 * bytes_per_row);
    if (rows == 0) {
        rows = 1;
    }
    if (rows < CPU_PAGED_ATTENTION_MIN_BLOCK_N) {
        rows = CPU_PAGED_ATTENTION_MIN_BLOCK_N;
    }
    if (rows > CPU_PAGED_ATTENTION_MAX_BLOCK_N) {
        rows = CPU_PAGED_ATTENTION_MAX_BLOCK_N;
    }
    return rows;
}

static cpu_paged_attention_f32_ops_t cpu_paged_attention_select_f32_ops(const cpu_context_t *ctx) {
    cpu_paged_attention_f32_ops_t ops = {
        .dot_f32 = cpu_paged_attention_dot_f32_scalar,
        .scale_f32 = cpu_paged_attention_scale_f32_scalar,
        .block_sum_f32 = cpu_paged_attention_block_sum_f32_scalar,
    };
#if HAS_AVX2
    if (ctx != nullptr && cpu_ctx_has_avx2(ctx)) {
        ops.dot_f32 = cpu_paged_attention_dot_f32_avx2;
        ops.scale_f32 = cpu_paged_attention_scale_f32_avx2;
        ops.block_sum_f32 = cpu_paged_attention_block_sum_f32_avx2;
        return ops;
    }
#endif
#if HAS_NEON
    if (ctx != nullptr && cpu_ctx_has_neon(ctx)) {
        ops.dot_f32 = cpu_paged_attention_dot_f32_neon;
        ops.scale_f32 = cpu_paged_attention_scale_f32_neon;
        ops.block_sum_f32 = cpu_paged_attention_block_sum_f32_neon;
        return ops;
    }
#endif
#if !HAS_AVX2 && !HAS_NEON
    (void)ctx;
#endif
    return ops;
}

static inline void cpu_paged_attention_load_row_f32(
    const void *device_ctx, marmot_dtype_t dtype, const void *data, size_t row_offset, size_t inner_stride, size_t len,
    float *dst
) {
    if (inner_stride == 1) {
        switch (dtype) {
        case MARMOT_DTYPE_FLOAT32:
            memcpy(dst, ((const float *)data) + row_offset, len * sizeof(float));
            return;
        case MARMOT_DTYPE_FLOAT16:
            cpu_convert_f16_to_f32(device_ctx, dst, ((const marmot_float16_t *)data) + row_offset, len);
            return;
        case MARMOT_DTYPE_BFLOAT16:
            cpu_convert_bf16_to_f32(device_ctx, dst, ((const marmot_bfloat16_t *)data) + row_offset, len);
            return;
        default:
            break;
        }
    }
    for (size_t d = 0; d < len; ++d) {
        dst[d] = cpu_load_as_f32(dtype, data, row_offset + d * inner_stride);
    }
}

static inline void cpu_paged_attention_store_row_f32(
    const void *device_ctx, marmot_dtype_t dtype, void *data, size_t row_offset, size_t inner_stride, size_t len,
    const float *src
) {
    if (inner_stride == 1) {
        switch (dtype) {
        case MARMOT_DTYPE_FLOAT32:
            memcpy(((float *)data) + row_offset, src, len * sizeof(float));
            return;
        case MARMOT_DTYPE_FLOAT16:
            cpu_convert_f32_to_f16(device_ctx, ((marmot_float16_t *)data) + row_offset, src, len);
            return;
        case MARMOT_DTYPE_BFLOAT16:
            cpu_convert_f32_to_bf16(device_ctx, ((marmot_bfloat16_t *)data) + row_offset, src, len);
            return;
        default:
            break;
        }
    }
    for (size_t d = 0; d < len; ++d) {
        cpu_store_from_f32(dtype, data, row_offset + d * inner_stride, src[d]);
    }
}

static inline float cpu_paged_attention_row_max_abs(
    marmot_dtype_t dtype, const void *data, size_t row_offset, size_t inner_stride, size_t len
) {
    float max_abs = 0.0f;
    for (size_t d = 0; d < len; ++d) {
        float value = cpu_load_as_f32(dtype, data, row_offset + d * inner_stride);
        float abs_value = fabsf(value);
        if (abs_value > max_abs) {
            max_abs = abs_value;
        }
    }
    return max_abs;
}

#if MARMOT_ENABLE_FP8
static inline float cpu_paged_attention_fp8_scale_from_max(float max_abs) {
    if (!(max_abs > 0.0f)) {
        return 1.0f;
    }
    return max_abs / CPU_PAGED_ATTENTION_FP8_E4M3_MAX;
}

static inline marmot_float8_e4m3_t cpu_paged_attention_fp8_quant(float value, float scale) {
    if (!(scale > 0.0f)) {
        return marmot_make_fp8_e4m3(0);
    }
    return marmot_f32_to_fp8_e4m3_ref(value / scale);
}

static inline float cpu_paged_attention_fp8_dequant(marmot_float8_e4m3_t value, float scale) {
    if (!(scale > 0.0f)) {
        return 0.0f;
    }
    return marmot_fp8_e4m3_to_f32_ref(value) * scale;
}

static inline void cpu_paged_attention_load_row_fp8_scaled(
    const marmot_float8_e4m3_t *data, size_t row_offset, size_t inner_stride, size_t len, float scale, float *dst
) {
    for (size_t d = 0; d < len; ++d) {
        const marmot_float8_e4m3_t value = data[row_offset + d * inner_stride];
        dst[d] = cpu_paged_attention_fp8_dequant(value, scale);
    }
}

static inline void cpu_paged_attention_store_row_fp8_scaled(
    marmot_float8_e4m3_t *dst, size_t dst_offset, size_t dst_stride, marmot_dtype_t src_dtype, const void *src,
    size_t src_offset, size_t src_stride, size_t len, float scale
) {
    for (size_t d = 0; d < len; ++d) {
        float value = cpu_load_as_f32(src_dtype, src, src_offset + d * src_stride);
        dst[dst_offset + d * dst_stride] = cpu_paged_attention_fp8_quant(value, scale);
    }
}
#endif

static inline void cpu_paged_attention_copy_row_mixed(
    const void *device_ctx, marmot_dtype_t dst_dtype, void *dst, size_t dst_offset, size_t dst_stride,
    marmot_dtype_t src_dtype, const void *src, size_t src_offset, size_t src_stride, size_t len
) {
    if (dst_stride == 1 && src_stride == 1) {
        if (dst_dtype == src_dtype) {
            switch (dst_dtype) {
            case MARMOT_DTYPE_FLOAT32:
                memcpy(((float *)dst) + dst_offset, ((const float *)src) + src_offset, len * sizeof(float));
                return;
            case MARMOT_DTYPE_FLOAT16:
                memcpy(
                    ((marmot_float16_t *)dst) + dst_offset, ((const marmot_float16_t *)src) + src_offset,
                    len * sizeof(marmot_float16_t)
                );
                return;
            case MARMOT_DTYPE_BFLOAT16:
                memcpy(
                    ((marmot_bfloat16_t *)dst) + dst_offset, ((const marmot_bfloat16_t *)src) + src_offset,
                    len * sizeof(marmot_bfloat16_t)
                );
                return;
            default:
                break;
            }
        } else if (dst_dtype == MARMOT_DTYPE_FLOAT32 && src_dtype == MARMOT_DTYPE_FLOAT16) {
            cpu_convert_f16_to_f32(
                device_ctx, ((float *)dst) + dst_offset, ((const marmot_float16_t *)src) + src_offset, len
            );
            return;
        } else if (dst_dtype == MARMOT_DTYPE_FLOAT32 && src_dtype == MARMOT_DTYPE_BFLOAT16) {
            cpu_convert_bf16_to_f32(
                device_ctx, ((float *)dst) + dst_offset, ((const marmot_bfloat16_t *)src) + src_offset, len
            );
            return;
        } else if (dst_dtype == MARMOT_DTYPE_FLOAT16 && src_dtype == MARMOT_DTYPE_FLOAT32) {
            cpu_convert_f32_to_f16(
                device_ctx, ((marmot_float16_t *)dst) + dst_offset, ((const float *)src) + src_offset, len
            );
            return;
        } else if (dst_dtype == MARMOT_DTYPE_BFLOAT16 && src_dtype == MARMOT_DTYPE_FLOAT32) {
            cpu_convert_f32_to_bf16(
                device_ctx, ((marmot_bfloat16_t *)dst) + dst_offset, ((const float *)src) + src_offset, len
            );
            return;
        }
    }
    for (size_t d = 0; d < len; ++d) {
        float value = cpu_load_as_f32(src_dtype, src, src_offset + d * src_stride);
        cpu_store_from_f32(dst_dtype, dst, dst_offset + d * dst_stride, value);
    }
}

typedef struct {
    float *q_vec;
    float *out_vec;
    float *k_block;
    float *v_block;
    float *scores;
    size_t head_dim_cap;
    size_t block_n_cap;
} cpu_paged_attention_workspace_t;

static void cpu_paged_attention_workspace_release(cpu_paged_attention_workspace_t *ws) {
    if (ws == nullptr) {
        return;
    }
    free(ws->q_vec);
    free(ws->out_vec);
    free(ws->k_block);
    free(ws->v_block);
    free(ws->scores);
    ws->q_vec = nullptr;
    ws->out_vec = nullptr;
    ws->k_block = nullptr;
    ws->v_block = nullptr;
    ws->scores = nullptr;
    ws->head_dim_cap = 0;
    ws->block_n_cap = 0;
}

static bool cpu_paged_attention_workspace_ensure(cpu_paged_attention_workspace_t *ws, size_t block_n, size_t head_dim) {
    if (ws == nullptr) {
        return false;
    }
    if (ws->q_vec != nullptr && ws->head_dim_cap >= head_dim && ws->block_n_cap >= block_n) {
        return true;
    }

    const size_t next_head_dim = head_dim > ws->head_dim_cap ? head_dim : ws->head_dim_cap;
    const size_t next_block_n = block_n > ws->block_n_cap ? block_n : ws->block_n_cap;

    cpu_paged_attention_workspace_t next = {0};
    next.q_vec = (float *)marmot_aligned_alloc(64, next_head_dim * sizeof(float));
    next.out_vec = (float *)marmot_aligned_alloc(64, next_head_dim * sizeof(float));
    next.k_block = (float *)marmot_aligned_alloc(64, next_block_n * next_head_dim * sizeof(float));
    next.v_block = (float *)marmot_aligned_alloc(64, next_block_n * next_head_dim * sizeof(float));
    next.scores = (float *)marmot_aligned_alloc(64, next_block_n * sizeof(float));
    if (next.q_vec == nullptr || next.out_vec == nullptr || next.k_block == nullptr || next.v_block == nullptr ||
        next.scores == nullptr) {
        cpu_paged_attention_workspace_release(&next);
        return false;
    }

    next.head_dim_cap = next_head_dim;
    next.block_n_cap = next_block_n;
    cpu_paged_attention_workspace_release(ws);
    *ws = next;
    return true;
}

static thread_local cpu_paged_attention_workspace_t cpu_paged_attention_tls = {0};

typedef struct {
    const marmot_uint32_t *meta_data;
    const void *k_data;
    const void *v_data;
    void *kv_k_data;
    void *kv_v_data;
    size_t meta_stride0;
    size_t meta_stride1;
    size_t k_stride0;
    size_t k_stride1;
    size_t k_stride2;
    size_t v_stride0;
    size_t v_stride1;
    size_t v_stride2;
    size_t kv_stride0;
    size_t kv_stride1;
    size_t kv_stride2;
    size_t kv_stride3;
    size_t kv_stride4;
    size_t num_kv_heads;
    size_t head_dim;
    uint32_t block_shift;
    uint32_t block_mask;
    uint32_t layer_idx;
    marmot_dtype_t k_dtype;
    marmot_dtype_t v_dtype;
    marmot_dtype_t kv_dtype;
    const void *device_ctx;
} cpu_paged_attention_kv_ctx_t;

static void cpu_paged_attention_kv_write_range(void *ctx, size_t start, size_t end) {
    const cpu_paged_attention_kv_ctx_t *c = (const cpu_paged_attention_kv_ctx_t *)ctx;
    for (size_t t = start; t < end; ++t) {
        const size_t meta_base = t * c->meta_stride0;
        const uint32_t kv_slot = c->meta_data[meta_base + 2 * c->meta_stride1].value;
        const uint32_t block_id = kv_slot >> c->block_shift;
        const uint32_t offset = kv_slot & c->block_mask;
        const size_t kv_base_block =
            (size_t)block_id * c->kv_stride0 + (size_t)c->layer_idx * c->kv_stride1 + (size_t)offset * c->kv_stride3;
        for (size_t kv_head = 0; kv_head < c->num_kv_heads; ++kv_head) {
            const size_t k_base = t * c->k_stride0 + kv_head * c->k_stride1;
            const size_t v_base = t * c->v_stride0 + kv_head * c->v_stride1;
            const size_t kv_head_base = kv_base_block + kv_head * c->kv_stride2;
            cpu_paged_attention_copy_row_mixed(
                c->device_ctx, c->kv_dtype, c->kv_k_data, kv_head_base, c->kv_stride4, c->k_dtype, c->k_data, k_base,
                c->k_stride2, c->head_dim
            );
            cpu_paged_attention_copy_row_mixed(
                c->device_ctx, c->kv_dtype, c->kv_v_data, kv_head_base, c->kv_stride4, c->v_dtype, c->v_data, v_base,
                c->v_stride2, c->head_dim
            );
        }
    }
}

typedef struct {
    const marmot_uint32_t *meta_data;
    const marmot_uint32_t *table_data;
    const void *q_data;
    const void *kv_k_data;
    const void *kv_v_data;
    void *out_data;
    size_t meta_stride0;
    size_t meta_stride1;
    size_t block_stride0;
    size_t block_stride1;
    size_t q_stride0;
    size_t q_stride1;
    size_t q_stride2;
    size_t out_stride0;
    size_t out_stride1;
    size_t out_stride2;
    size_t kv_stride0;
    size_t kv_stride1;
    size_t kv_stride2;
    size_t kv_stride3;
    size_t kv_stride4;
    size_t num_q_heads;
    size_t num_kv_heads;
    size_t head_dim;
    size_t max_blocks_per_seq;
    size_t block_n;
    size_t gqa_group;
    uint32_t block_shift;
    uint32_t block_mask;
    uint32_t layer_idx;
    marmot_dtype_t q_dtype;
    marmot_dtype_t kv_dtype;
    marmot_dtype_t out_dtype;
    float scale;
    bool kv_fp8;
    cpu_paged_attention_f32_ops_t ops;
    const float *kv_k_scale;
    const float *kv_v_scale;
    size_t kv_scale_stride0;
    size_t kv_scale_stride1;
    size_t kv_scale_stride2;
    const void *device_ctx;
} cpu_paged_attention_attn_ctx_t;

static marmot_error_t cpu_paged_attention_attention_range(void *ctx, size_t start, size_t end) {
    const cpu_paged_attention_attn_ctx_t *c = (const cpu_paged_attention_attn_ctx_t *)ctx;
    cpu_paged_attention_workspace_t *ws = &cpu_paged_attention_tls;
    if (!cpu_paged_attention_workspace_ensure(ws, c->block_n, c->head_dim)) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    const size_t kv_elem_size = marmot_dtype_size(c->kv_dtype);
    const bool allow_prefetch = kv_elem_size != 0;

    for (size_t t = start; t < end; ++t) {
        const size_t meta_base = t * c->meta_stride0;
        const uint32_t seq_slot = c->meta_data[meta_base + 0 * c->meta_stride1].value;
        const uint32_t pos = c->meta_data[meta_base + 1 * c->meta_stride1].value;
        const uint32_t kv_len = pos + 1u;

        for (size_t q_head = 0; q_head < c->num_q_heads; ++q_head) {
            const size_t kv_head = (c->gqa_group == 0) ? q_head : (q_head / c->gqa_group);
            const size_t q_base = t * c->q_stride0 + q_head * c->q_stride1;
            const size_t out_base = t * c->out_stride0 + q_head * c->out_stride1;
#if MARMOT_ENABLE_FP8
            uint32_t last_table_block = 0xFFFFFFFFu;
            float last_k_scale = 1.0f;
            float last_v_scale = 1.0f;
#endif

            cpu_paged_attention_load_row_f32(
                c->device_ctx, c->q_dtype, c->q_data, q_base, c->q_stride2, c->head_dim, ws->q_vec
            );
            if (c->scale != 1.0f) {
                c->ops.scale_f32(ws->q_vec, c->head_dim, c->scale);
            }

            memset(ws->out_vec, 0, c->head_dim * sizeof(float));

            float max_score = -INFINITY;
            float sum = 0.0f;
            bool has_any = false;

            for (uint32_t k_start = 0; k_start < kv_len; k_start += (uint32_t)c->block_n) {
                uint32_t k_limit = k_start + (uint32_t)c->block_n;
                if (k_limit > kv_len) {
                    k_limit = kv_len;
                }

                size_t k_count = 0;
                for (uint32_t p = k_start; p < k_limit; ++p) {
                    if (allow_prefetch) {
                        const uint32_t p_prefetch = p + CPU_PAGED_ATTENTION_PREFETCH_DISTANCE;
                        if (p_prefetch < k_limit) {
                            const uint32_t logical_block_pf = p_prefetch >> c->block_shift;
                            if (logical_block_pf < c->max_blocks_per_seq) {
                                const size_t block_index_pf =
                                    (size_t)seq_slot * c->block_stride0 + (size_t)logical_block_pf * c->block_stride1;
                                const uint32_t table_block_pf = c->table_data[block_index_pf].value;
                                if (table_block_pf != 0xFFFFFFFFu) {
                                    const uint32_t p_offset_pf = p_prefetch & c->block_mask;
                                    const size_t kv_base_pf = (size_t)table_block_pf * c->kv_stride0 +
                                        (size_t)c->layer_idx * c->kv_stride1 + kv_head * c->kv_stride2 +
                                        (size_t)p_offset_pf * c->kv_stride3;
                                    const uint8_t *kv_k_ptr = (const uint8_t *)c->kv_k_data + kv_base_pf * kv_elem_size;
                                    const uint8_t *kv_v_ptr = (const uint8_t *)c->kv_v_data + kv_base_pf * kv_elem_size;
                                    MARMOT_PREFETCH(kv_k_ptr);
                                    MARMOT_PREFETCH(kv_v_ptr);
                                }
                            }
                        }
                    }
                    const uint32_t logical_block = p >> c->block_shift;
                    if (logical_block >= c->max_blocks_per_seq) {
                        break;
                    }
                    const size_t block_index =
                        (size_t)seq_slot * c->block_stride0 + (size_t)logical_block * c->block_stride1;
                    const uint32_t table_block = c->table_data[block_index].value;
                    if (table_block == 0xFFFFFFFFu) {
                        continue;
                    }
                    const uint32_t p_offset = p & c->block_mask;
                    const size_t kv_base = (size_t)table_block * c->kv_stride0 + (size_t)c->layer_idx * c->kv_stride1 +
                        kv_head * c->kv_stride2 + (size_t)p_offset * c->kv_stride3;
                    if (c->kv_fp8) {
#if MARMOT_ENABLE_FP8
                        float k_scale = last_k_scale;
                        float v_scale = last_v_scale;
                        if (table_block != last_table_block) {
                            last_table_block = table_block;
                            const size_t scale_index = (size_t)table_block * c->kv_scale_stride0 +
                                (size_t)c->layer_idx * c->kv_scale_stride1 + kv_head * c->kv_scale_stride2;
                            k_scale = c->kv_k_scale[scale_index];
                            v_scale = c->kv_v_scale[scale_index];
                            last_k_scale = k_scale;
                            last_v_scale = v_scale;
                        }
                        cpu_paged_attention_load_row_fp8_scaled(
                            (const marmot_float8_e4m3_t *)c->kv_k_data, kv_base, c->kv_stride4, c->head_dim, k_scale,
                            ws->k_block + k_count * c->head_dim
                        );
                        cpu_paged_attention_load_row_fp8_scaled(
                            (const marmot_float8_e4m3_t *)c->kv_v_data, kv_base, c->kv_stride4, c->head_dim, v_scale,
                            ws->v_block + k_count * c->head_dim
                        );
#endif
                    } else {
                        cpu_paged_attention_load_row_f32(
                            c->device_ctx, c->kv_dtype, c->kv_k_data, kv_base, c->kv_stride4, c->head_dim,
                            ws->k_block + k_count * c->head_dim
                        );
                        cpu_paged_attention_load_row_f32(
                            c->device_ctx, c->kv_dtype, c->kv_v_data, kv_base, c->kv_stride4, c->head_dim,
                            ws->v_block + k_count * c->head_dim
                        );
                    }
                    ++k_count;
                }

                if (k_count == 0) {
                    continue;
                }

                float row_max = -INFINITY;
                for (size_t kj = 0; kj < k_count; ++kj) {
                    const float *k_row = ws->k_block + kj * c->head_dim;
                    float dot = c->ops.dot_f32(ws->q_vec, k_row, c->head_dim);
                    ws->scores[kj] = dot;
                    if (dot > row_max) {
                        row_max = dot;
                    }
                }

                float next_max = max_score > row_max ? max_score : row_max;
                float exp_prev = max_score == -INFINITY ? 0.0f : expf(max_score - next_max);
                if (exp_prev != 1.0f) {
                    c->ops.scale_f32(ws->out_vec, c->head_dim, exp_prev);
                }

                float block_sum =
                    c->ops.block_sum_f32(ws->out_vec, ws->v_block, ws->scores, k_count, c->head_dim, next_max);

                sum = sum * exp_prev + block_sum;
                max_score = next_max;
                has_any = true;
            }

            if (!has_any) {
                for (size_t d = 0; d < c->head_dim; ++d) {
                    cpu_store_from_f32(c->out_dtype, c->out_data, out_base + d * c->out_stride2, 0.0f);
                }
                continue;
            }

            float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
            if (inv_sum != 1.0f) {
                c->ops.scale_f32(ws->out_vec, c->head_dim, inv_sum);
            }
            cpu_paged_attention_store_row_f32(
                c->device_ctx, c->out_dtype, c->out_data, out_base, c->out_stride2, c->head_dim, ws->out_vec
            );
        }
    }

    return MARMOT_SUCCESS;
}

static bool cpu_paged_attention_activation_dtype_supported(marmot_dtype_t dtype) {
    return dtype == MARMOT_DTYPE_FLOAT32 || dtype == MARMOT_DTYPE_FLOAT16 || dtype == MARMOT_DTYPE_BFLOAT16;
}

static bool cpu_paged_attention_kv_dtype_supported(marmot_dtype_t dtype) {
    if (dtype == MARMOT_DTYPE_FLOAT32 || dtype == MARMOT_DTYPE_FLOAT16 || dtype == MARMOT_DTYPE_BFLOAT16) {
        return true;
    }
#if MARMOT_ENABLE_FP8
    return dtype == MARMOT_DTYPE_FLOAT8_E4M3;
#else
    (void)dtype;
    return false;
#endif
}

marmot_error_t cpu_paged_attention_impl(const void *device_ctx, const marmot_paged_attention_desc_t *desc) {
    if (desc == nullptr || !marmot_paged_attention_desc_is_valid(desc)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention descriptor invalid");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (desc->token_count == 0) {
        return MARMOT_SUCCESS;
    }
    if (desc->num_q_heads == 0 || desc->num_kv_heads == 0 || desc->head_dim == 0 || desc->block_size == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention requires non-zero dimensions");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!cpu_paged_attention_is_power_of_two_u32(desc->block_size)) {
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
    const marmot_paged_attention_kv_scale_ext_t *scale_ext =
        (desc->pnext != nullptr) ? (const marmot_paged_attention_kv_scale_ext_t *)desc->pnext : nullptr;
    const marmot_tensor_t *kv_k_scale = nullptr;
    const marmot_tensor_t *kv_v_scale = nullptr;
#if MARMOT_ENABLE_FP8
    if (scale_ext != nullptr && scale_ext->struct_version == MARMOT_PAGED_ATTENTION_KV_SCALE_EXT_VERSION &&
        scale_ext->struct_size >= sizeof(*scale_ext)) {
        kv_k_scale = scale_ext->kv_k_scale;
        kv_v_scale = scale_ext->kv_v_scale;
    }
#else
    (void)scale_ext;
#endif

    if (token_meta->dtype != MARMOT_DTYPE_UINT32 || block_table->dtype != MARMOT_DTYPE_UINT32) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "paged_attention requires uint32 metadata");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (!cpu_paged_attention_activation_dtype_supported(q->dtype) ||
        !cpu_paged_attention_activation_dtype_supported(k_new->dtype) ||
        !cpu_paged_attention_activation_dtype_supported(v_new->dtype) ||
        !cpu_paged_attention_kv_dtype_supported(kv_k->dtype) || !cpu_paged_attention_kv_dtype_supported(kv_v->dtype) ||
        !cpu_paged_attention_activation_dtype_supported(out->dtype)) {
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
#if MARMOT_ENABLE_FP8
    if (kv_k->dtype == MARMOT_DTYPE_FLOAT8_E4M3) {
        if (kv_k_scale == nullptr || kv_v_scale == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention FP8 KV requires scale tensors");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (kv_k_scale->dtype != MARMOT_DTYPE_FLOAT32 || kv_v_scale->dtype != MARMOT_DTYPE_FLOAT32) {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "paged_attention FP8 scale dtype must be float32");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        if (kv_k_scale->shape.ndim != 3 || kv_v_scale->shape.ndim != 3) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention FP8 scale tensors must be 3D");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (kv_k_scale->shape.shape[0] != kv_k->shape.shape[0] || kv_k_scale->shape.shape[1] != kv_k->shape.shape[1] ||
            kv_k_scale->shape.shape[2] != kv_k->shape.shape[2]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention FP8 scale tensor shape mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (kv_v_scale->shape.shape[0] != kv_v->shape.shape[0] || kv_v_scale->shape.shape[1] != kv_v->shape.shape[1] ||
            kv_v_scale->shape.shape[2] != kv_v->shape.shape[2]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention FP8 scale tensor shape mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (kv_k_scale->data == nullptr || kv_v_scale->data == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention FP8 scale tensors missing data");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
    } else
#endif
        if (kv_k_scale != nullptr || kv_v_scale != nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention scale tensors require FP8 KV");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t token_count = desc->token_count;
    const size_t num_q_heads = desc->num_q_heads;
    const size_t num_kv_heads = desc->num_kv_heads;
    const size_t head_dim = desc->head_dim;
    const size_t max_blocks_per_seq = block_table->shape.shape[1];
    const uint32_t block_shift = cpu_paged_attention_log2_u32(desc->block_size);
    const uint32_t block_mask = desc->block_size - 1u;
    const size_t gqa_group = desc->num_q_heads / desc->num_kv_heads;

    const size_t meta_stride0 = token_meta->shape.strides[0];
    const size_t meta_stride1 = token_meta->shape.strides[1];
    const size_t block_stride0 = block_table->shape.strides[0];
    const size_t block_stride1 = block_table->shape.strides[1];

    const size_t q_stride0 = q->shape.strides[0];
    const size_t q_stride1 = q->shape.strides[1];
    const size_t q_stride2 = q->shape.strides[2];
    const size_t k_stride0 = k_new->shape.strides[0];
    const size_t k_stride1 = k_new->shape.strides[1];
    const size_t k_stride2 = k_new->shape.strides[2];
    const size_t v_stride0 = v_new->shape.strides[0];
    const size_t v_stride1 = v_new->shape.strides[1];
    const size_t v_stride2 = v_new->shape.strides[2];
    const size_t out_stride0 = out->shape.strides[0];
    const size_t out_stride1 = out->shape.strides[1];
    const size_t out_stride2 = out->shape.strides[2];
    const size_t kv_stride0 = kv_k->shape.strides[0];
    const size_t kv_stride1 = kv_k->shape.strides[1];
    const size_t kv_stride2 = kv_k->shape.strides[2];
    const size_t kv_stride3 = kv_k->shape.strides[3];
    const size_t kv_stride4 = kv_k->shape.strides[4];
    size_t kv_scale_stride0 = 0;
    size_t kv_scale_stride1 = 0;
    size_t kv_scale_stride2 = 0;
    float *kv_k_scale_data = nullptr;
    float *kv_v_scale_data = nullptr;
    bool kv_fp8 = false;
#if MARMOT_ENABLE_FP8
    if (kv_k->dtype == MARMOT_DTYPE_FLOAT8_E4M3) {
        kv_fp8 = true;
        kv_k_scale_data = (float *)kv_k_scale->data;
        kv_v_scale_data = (float *)kv_v_scale->data;
        kv_scale_stride0 = kv_k_scale->shape.strides[0];
        kv_scale_stride1 = kv_k_scale->shape.strides[1];
        kv_scale_stride2 = kv_k_scale->shape.strides[2];
    }
#endif

    const marmot_uint32_t *meta_data = (const marmot_uint32_t *)token_meta->data;
    const marmot_uint32_t *table_data = (const marmot_uint32_t *)block_table->data;
    const void *q_data = q->data;
    const void *k_data = k_new->data;
    const void *v_data = v_new->data;
    void *kv_k_data = kv_k->data;
    void *kv_v_data = kv_v->data;
    void *out_data = out->data;
    if (meta_data == nullptr || table_data == nullptr || q_data == nullptr || k_data == nullptr || v_data == nullptr ||
        kv_k_data == nullptr || kv_v_data == nullptr || out_data == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention requires non-null tensor data");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t block_n = cpu_paged_attention_choose_block_n(head_dim);
    const cpu_context_t *cpu_ctx = get_cpu_context(device_ctx);
    const cpu_paged_attention_f32_ops_t ops = cpu_paged_attention_select_f32_ops(cpu_ctx);

    const size_t max_seq_slots = block_table->shape.shape[0];
    const size_t max_kv_blocks = kv_k->shape.shape[0];
    for (size_t t = 0; t < token_count; ++t) {
        const size_t meta_base = t * meta_stride0;
        const uint32_t seq_slot = meta_data[meta_base + 0 * meta_stride1].value;
        const uint32_t kv_slot = meta_data[meta_base + 2 * meta_stride1].value;
        const uint32_t block_id = kv_slot >> block_shift;
        if (seq_slot >= max_seq_slots) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention seq_slot out of range");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (block_id >= max_kv_blocks) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention kv_slot block out of range");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }
#if MARMOT_ENABLE_FP8
    if (kv_fp8) {
        const size_t num_blocks = kv_k->shape.shape[0];
        marmot_error_t fp8_status = MARMOT_SUCCESS;
        if (num_blocks > SIZE_MAX / num_kv_heads) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention fp8 scale overflow");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        uint32_t *block_min_offset = (uint32_t *)malloc(num_blocks * sizeof(uint32_t));
        uint8_t *block_touched = (uint8_t *)calloc(num_blocks, sizeof(uint8_t));
        float *new_k_max = (float *)calloc(num_blocks * num_kv_heads, sizeof(float));
        float *new_v_max = (float *)calloc(num_blocks * num_kv_heads, sizeof(float));
        if (block_min_offset == nullptr || block_touched == nullptr || new_k_max == nullptr || new_v_max == nullptr) {
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "paged_attention fp8 scale allocation failed");
            fp8_status = MARMOT_ERROR_OUT_OF_MEMORY;
            free(block_min_offset);
            free(block_touched);
            free(new_k_max);
            free(new_v_max);
            return fp8_status;
        }
        for (size_t i = 0; i < num_blocks; ++i) {
            block_min_offset[i] = desc->block_size;
        }

        for (size_t t = 0; t < token_count; ++t) {
            const size_t meta_base = t * meta_stride0;
            const uint32_t kv_slot = meta_data[meta_base + 2 * meta_stride1].value;
            const uint32_t block_id = kv_slot >> block_shift;
            const uint32_t offset = kv_slot & block_mask;
            if (block_id >= num_blocks) {
                marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention kv_slot block out of range");
                fp8_status = MARMOT_ERROR_DIMENSION_MISMATCH;
                break;
            }
            block_touched[block_id] = 1;
            if (offset < block_min_offset[block_id]) {
                block_min_offset[block_id] = offset;
            }
            for (size_t kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
                const size_t k_base = t * k_stride0 + kv_head * k_stride1;
                const size_t v_base = t * v_stride0 + kv_head * v_stride1;
                const float k_max = cpu_paged_attention_row_max_abs(k_new->dtype, k_data, k_base, k_stride2, head_dim);
                const float v_max = cpu_paged_attention_row_max_abs(v_new->dtype, v_data, v_base, v_stride2, head_dim);
                const size_t max_index = (size_t)block_id * num_kv_heads + kv_head;
                if (k_max > new_k_max[max_index]) {
                    new_k_max[max_index] = k_max;
                }
                if (v_max > new_v_max[max_index]) {
                    new_v_max[max_index] = v_max;
                }
            }
        }

        marmot_float8_e4m3_t *kv_k_fp8 = (marmot_float8_e4m3_t *)kv_k_data;
        marmot_float8_e4m3_t *kv_v_fp8 = (marmot_float8_e4m3_t *)kv_v_data;

        if (fp8_status == MARMOT_SUCCESS) {
            for (size_t block_id = 0; block_id < num_blocks; ++block_id) {
                if (block_touched[block_id] == 0) {
                    continue;
                }
                const uint32_t existing_count = block_min_offset[block_id];
                const size_t scale_base = block_id * kv_scale_stride0 + (size_t)desc->layer_idx * kv_scale_stride1;
                for (size_t kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
                    const size_t scale_index = scale_base + kv_head * kv_scale_stride2;
                    const float old_k_scale = kv_k_scale_data[scale_index];
                    const float old_v_scale = kv_v_scale_data[scale_index];
                    float existing_k_max = 0.0f;
                    float existing_v_max = 0.0f;
                    if (existing_count > 0 && old_k_scale > 0.0f) {
                        for (uint32_t pos = 0; pos < existing_count; ++pos) {
                            const size_t kv_base = (size_t)block_id * kv_stride0 +
                                (size_t)desc->layer_idx * kv_stride1 + kv_head * kv_stride2 + (size_t)pos * kv_stride3;
                            for (size_t d = 0; d < head_dim; ++d) {
                                const marmot_float8_e4m3_t value = kv_k_fp8[kv_base + d * kv_stride4];
                                const float fval = cpu_paged_attention_fp8_dequant(value, old_k_scale);
                                const float abs_val = fabsf(fval);
                                if (abs_val > existing_k_max) {
                                    existing_k_max = abs_val;
                                }
                            }
                        }
                    }
                    if (existing_count > 0 && old_v_scale > 0.0f) {
                        for (uint32_t pos = 0; pos < existing_count; ++pos) {
                            const size_t kv_base = (size_t)block_id * kv_stride0 +
                                (size_t)desc->layer_idx * kv_stride1 + kv_head * kv_stride2 + (size_t)pos * kv_stride3;
                            for (size_t d = 0; d < head_dim; ++d) {
                                const marmot_float8_e4m3_t value = kv_v_fp8[kv_base + d * kv_stride4];
                                const float fval = cpu_paged_attention_fp8_dequant(value, old_v_scale);
                                const float abs_val = fabsf(fval);
                                if (abs_val > existing_v_max) {
                                    existing_v_max = abs_val;
                                }
                            }
                        }
                    }

                    const size_t max_index = block_id * num_kv_heads + kv_head;
                    float total_k_max = existing_k_max;
                    float total_v_max = existing_v_max;
                    if (new_k_max[max_index] > total_k_max) {
                        total_k_max = new_k_max[max_index];
                    }
                    if (new_v_max[max_index] > total_v_max) {
                        total_v_max = new_v_max[max_index];
                    }

                    const float new_k_scale = cpu_paged_attention_fp8_scale_from_max(total_k_max);
                    const float new_v_scale = cpu_paged_attention_fp8_scale_from_max(total_v_max);

                    if (existing_count > 0 && new_k_scale != old_k_scale) {
                        for (uint32_t pos = 0; pos < existing_count; ++pos) {
                            const size_t kv_base = (size_t)block_id * kv_stride0 +
                                (size_t)desc->layer_idx * kv_stride1 + kv_head * kv_stride2 + (size_t)pos * kv_stride3;
                            for (size_t d = 0; d < head_dim; ++d) {
                                const marmot_float8_e4m3_t value = kv_k_fp8[kv_base + d * kv_stride4];
                                const float fval = cpu_paged_attention_fp8_dequant(value, old_k_scale);
                                kv_k_fp8[kv_base + d * kv_stride4] = cpu_paged_attention_fp8_quant(fval, new_k_scale);
                            }
                        }
                    }
                    if (existing_count > 0 && new_v_scale != old_v_scale) {
                        for (uint32_t pos = 0; pos < existing_count; ++pos) {
                            const size_t kv_base = (size_t)block_id * kv_stride0 +
                                (size_t)desc->layer_idx * kv_stride1 + kv_head * kv_stride2 + (size_t)pos * kv_stride3;
                            for (size_t d = 0; d < head_dim; ++d) {
                                const marmot_float8_e4m3_t value = kv_v_fp8[kv_base + d * kv_stride4];
                                const float fval = cpu_paged_attention_fp8_dequant(value, old_v_scale);
                                kv_v_fp8[kv_base + d * kv_stride4] = cpu_paged_attention_fp8_quant(fval, new_v_scale);
                            }
                        }
                    }

                    kv_k_scale_data[scale_index] = new_k_scale;
                    kv_v_scale_data[scale_index] = new_v_scale;
                }
            }
        }

        if (fp8_status == MARMOT_SUCCESS) {
            for (size_t t = 0; t < token_count; ++t) {
                const size_t meta_base = t * meta_stride0;
                const uint32_t kv_slot = meta_data[meta_base + 2 * meta_stride1].value;
                const uint32_t block_id = kv_slot >> block_shift;
                const uint32_t offset = kv_slot & block_mask;
                const size_t kv_base_block =
                    (size_t)block_id * kv_stride0 + (size_t)desc->layer_idx * kv_stride1 + (size_t)offset * kv_stride3;
                const size_t scale_base =
                    (size_t)block_id * kv_scale_stride0 + (size_t)desc->layer_idx * kv_scale_stride1;
                for (size_t kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
                    const size_t k_base = t * k_stride0 + kv_head * k_stride1;
                    const size_t v_base = t * v_stride0 + kv_head * v_stride1;
                    const size_t kv_head_base = kv_base_block + kv_head * kv_stride2;
                    const size_t scale_index = scale_base + kv_head * kv_scale_stride2;
                    const float k_scale = kv_k_scale_data[scale_index];
                    const float v_scale = kv_v_scale_data[scale_index];
                    cpu_paged_attention_store_row_fp8_scaled(
                        kv_k_fp8, kv_head_base, kv_stride4, k_new->dtype, k_data, k_base, k_stride2, head_dim, k_scale
                    );
                    cpu_paged_attention_store_row_fp8_scaled(
                        kv_v_fp8, kv_head_base, kv_stride4, v_new->dtype, v_data, v_base, v_stride2, head_dim, v_scale
                    );
                }
            }
        }

        free(block_min_offset);
        free(block_touched);
        free(new_k_max);
        free(new_v_max);

        if (fp8_status != MARMOT_SUCCESS) {
            return fp8_status;
        }
    } else
#endif
    {
        cpu_paged_attention_kv_ctx_t kv_ctx = {
            .meta_data = meta_data,
            .k_data = k_data,
            .v_data = v_data,
            .kv_k_data = kv_k_data,
            .kv_v_data = kv_v_data,
            .meta_stride0 = meta_stride0,
            .meta_stride1 = meta_stride1,
            .k_stride0 = k_stride0,
            .k_stride1 = k_stride1,
            .k_stride2 = k_stride2,
            .v_stride0 = v_stride0,
            .v_stride1 = v_stride1,
            .v_stride2 = v_stride2,
            .kv_stride0 = kv_stride0,
            .kv_stride1 = kv_stride1,
            .kv_stride2 = kv_stride2,
            .kv_stride3 = kv_stride3,
            .kv_stride4 = kv_stride4,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .block_shift = block_shift,
            .block_mask = block_mask,
            .layer_idx = desc->layer_idx,
            .k_dtype = k_new->dtype,
            .v_dtype = v_new->dtype,
            .kv_dtype = kv_k->dtype,
            .device_ctx = device_ctx,
        };

        marmot_dispatch_parallel_for_range(
            MARMOT_DISPATCH_PRIORITY_HIGH, token_count, 1, &kv_ctx, cpu_paged_attention_kv_write_range
        );
    }

    cpu_paged_attention_attn_ctx_t attn_ctx = {
        .meta_data = meta_data,
        .table_data = table_data,
        .q_data = q_data,
        .kv_k_data = kv_k_data,
        .kv_v_data = kv_v_data,
        .out_data = out_data,
        .meta_stride0 = meta_stride0,
        .meta_stride1 = meta_stride1,
        .block_stride0 = block_stride0,
        .block_stride1 = block_stride1,
        .q_stride0 = q_stride0,
        .q_stride1 = q_stride1,
        .q_stride2 = q_stride2,
        .out_stride0 = out_stride0,
        .out_stride1 = out_stride1,
        .out_stride2 = out_stride2,
        .kv_stride0 = kv_stride0,
        .kv_stride1 = kv_stride1,
        .kv_stride2 = kv_stride2,
        .kv_stride3 = kv_stride3,
        .kv_stride4 = kv_stride4,
        .num_q_heads = num_q_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .max_blocks_per_seq = max_blocks_per_seq,
        .block_n = block_n,
        .gqa_group = gqa_group,
        .block_shift = block_shift,
        .block_mask = block_mask,
        .layer_idx = desc->layer_idx,
        .q_dtype = q->dtype,
        .kv_dtype = kv_k->dtype,
        .out_dtype = out->dtype,
        .scale = desc->scale,
        .kv_fp8 = kv_fp8,
        .ops = ops,
        .kv_k_scale = kv_k_scale_data,
        .kv_v_scale = kv_v_scale_data,
        .kv_scale_stride0 = kv_scale_stride0,
        .kv_scale_stride1 = kv_scale_stride1,
        .kv_scale_stride2 = kv_scale_stride2,
        .device_ctx = device_ctx,
    };

    marmot_error_t status = marmot_dispatch_parallel_for_range_with_error(
        MARMOT_DISPATCH_PRIORITY_HIGH, token_count, 1, &attn_ctx, cpu_paged_attention_attention_range
    );
    if (status != MARMOT_SUCCESS) {
        if (status == MARMOT_ERROR_OUT_OF_MEMORY) {
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "paged_attention scratch allocation failed");
        } else {
            marmot_set_error(status, "paged_attention execution failed");
        }
        return status;
    }

    return MARMOT_SUCCESS;
}
