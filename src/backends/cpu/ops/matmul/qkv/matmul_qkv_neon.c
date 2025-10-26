#include "cpu_backend_internal.h"
#include "matmul_qkv_rope.h"

#if HAS_NEON

#include <arm_neon.h>

#define MATMUL_QKV_NEON_MIN_WORK_F32 (256 * 1024)                // tuned threshold
#define MATMUL_QKV_NEON_MIN_WORK_F16 (2ULL * 1024ULL * 1024ULL)  // tuned threshold
#define MATMUL_QKV_NEON_MIN_WORK_BF16 (3ULL * 1024ULL * 1024ULL) // tuned threshold

static inline float cpu_matmul_qkv_neon_sum(float32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float32x2_t lo = vget_low_f32(v);
    float32x2_t hi = vget_high_f32(v);
    lo = vadd_f32(lo, hi);
    lo = vpadd_f32(lo, lo);
    return vget_lane_f32(lo, 0);
#endif
}

static inline bool cpu_matmul_qkv_neon_can_use_contiguous(
    const marmot_tensor_t *input, const marmot_tensor_t *weight, const marmot_tensor_t *out_q,
    const marmot_tensor_t *out_k, const marmot_tensor_t *out_v, const marmot_tensor_t *bias, size_t K, size_t M
) {
    if (input == nullptr || weight == nullptr || out_q == nullptr || out_k == nullptr || out_v == nullptr) {
        return false;
    }
    if (input->shape.strides[1] != 1 || weight->shape.strides[1] != 1 || out_q->shape.strides[1] != 1 ||
        out_k->shape.strides[1] != 1 || out_v->shape.strides[1] != 1) {
        return false;
    }
    const ptrdiff_t in_row_stride = input->shape.strides[0];
    const ptrdiff_t weight_row_stride = weight->shape.strides[0];
    const ptrdiff_t out_q_row_stride = out_q->shape.strides[0];
    const ptrdiff_t out_k_row_stride = out_k->shape.strides[0];
    const ptrdiff_t out_v_row_stride = out_v->shape.strides[0];
    if (in_row_stride != (ptrdiff_t)K || weight_row_stride != (ptrdiff_t)K || out_q_row_stride != (ptrdiff_t)M ||
        out_k_row_stride != (ptrdiff_t)M || out_v_row_stride != (ptrdiff_t)M) {
        return false;
    }
    if (bias != nullptr) {
        if (bias->shape.ndim != 1 || bias->shape.shape[0] != 3 * M || bias->shape.strides[0] != 1) {
            return false;
        }
    }
    return true;
}

static inline size_t cpu_matmul_qkv_neon_select_block_columns(size_t M, size_t K) {
    if (M >= 16 && K >= 64) {
        return 4;
    }
    if (M >= 8 && K >= 32) {
        return 2;
    }
    return 1;
}

#if HAS_NEON
static bool cpu_matmul_qkv_neon_is_row_major_f32(const marmot_tensor_t *tensor, size_t rows, size_t cols) {
    if (tensor == nullptr || tensor->dtype != MARMOT_DTYPE_FLOAT32 || tensor->shape.ndim != 2) {
        return false;
    }
    return tensor->shape.shape[0] == rows && tensor->shape.shape[1] == cols && tensor->shape.strides[1] == 1 &&
        tensor->shape.strides[0] == cols;
}

static bool cpu_matmul_qkv_neon_is_bias_vector_f32(const marmot_tensor_t *tensor, size_t length) {
    if (tensor == nullptr) {
        return true;
    }
    return tensor->dtype == MARMOT_DTYPE_FLOAT32 && tensor->shape.ndim == 1 && tensor->shape.shape[0] == length &&
        tensor->shape.strides[0] == 1;
}
#endif

marmot_error_t cpu_matmul_qkv_neon_run_separate_f32(
    const void *device_ctx, const marmot_matmul_qkv_desc_t *desc, size_t N, size_t K, size_t M
) {
#if !HAS_NEON
    (void)device_ctx;
    (void)desc;
    (void)N;
    (void)K;
    (void)M;
    return MARMOT_ERROR_NOT_IMPLEMENTED;
#else
    cpu_context_t *ctx = (cpu_context_t *)device_ctx;
    if (ctx == nullptr || !ctx->runtime_caps.has_neon || desc == nullptr) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (desc->layout != MARMOT_QKV_LAYOUT_SEPARATE) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const marmot_tensor_t *input = desc->input;
    const marmot_tensor_t *wq = desc->separate.wq;
    const marmot_tensor_t *wk = desc->separate.wk;
    const marmot_tensor_t *wv = desc->separate.wv;
    const marmot_tensor_t *bq = desc->separate.bq;
    const marmot_tensor_t *bk = desc->separate.bk;
    const marmot_tensor_t *bv = desc->separate.bv;
    marmot_tensor_t *out_q = desc->out_q;
    marmot_tensor_t *out_k = desc->out_k;
    marmot_tensor_t *out_v = desc->out_v;

    if (!cpu_matmul_qkv_neon_is_row_major_f32(input, N, K) || !cpu_matmul_qkv_neon_is_row_major_f32(out_q, N, M) ||
        !cpu_matmul_qkv_neon_is_row_major_f32(out_k, N, M) || !cpu_matmul_qkv_neon_is_row_major_f32(out_v, N, M)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (!cpu_matmul_qkv_neon_is_row_major_f32(wq, M, K) || !cpu_matmul_qkv_neon_is_row_major_f32(wk, M, K) ||
        !cpu_matmul_qkv_neon_is_row_major_f32(wv, M, K)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (!cpu_matmul_qkv_neon_is_bias_vector_f32(bq, M) || !cpu_matmul_qkv_neon_is_bias_vector_f32(bk, M) ||
        !cpu_matmul_qkv_neon_is_bias_vector_f32(bv, M)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const float *input_data = (const float *)input->data;
    const float *wq_data = (const float *)wq->data;
    const float *wk_data = (const float *)wk->data;
    const float *wv_data = (const float *)wv->data;
    const float *bq_data = bq != nullptr ? (const float *)bq->data : nullptr;
    const float *bk_data = bk != nullptr ? (const float *)bk->data : nullptr;
    const float *bv_data = bv != nullptr ? (const float *)bv->data : nullptr;
    float *out_q_data = (float *)out_q->data;
    float *out_k_data = (float *)out_k->data;
    float *out_v_data = (float *)out_v->data;

    const marmot_rope_params_t *rope = desc->rope_params;
    const bool apply_rope_q = rope != nullptr && rope->apply_to_q;
    const bool apply_rope_k = rope != nullptr && rope->apply_to_k;
    const marmot_tensor_t *positions = rope != nullptr ? rope->positions : nullptr;
    const size_t rope_head_dim = cpu_matmul_qkv_resolve_head_dim(M, rope);
    marmot_rope_freq_span_t rope_span = {0};
    const float *rope_freqs = nullptr;
    const float *sincos_base = nullptr;
    size_t sincos_stride = 0;
    size_t sincos_cached_positions = 0;
    const int32_t *positions_i32 = nullptr;
    const int64_t *positions_i64 = nullptr;
    float rope_attn_scale = 1.0f;
    if ((apply_rope_q || apply_rope_k) && positions != nullptr) {
        marmot_error_t rope_status =
            marmot_rope_freq_cache_ensure(ctx != nullptr ? &ctx->rope_cache : nullptr, rope_head_dim, rope, &rope_span);
        if (rope_status != MARMOT_SUCCESS) {
            return rope_status;
        }
        rope_freqs = rope_span.freqs;
        rope_attn_scale = rope_span.attn_scale;
        if (ctx != nullptr) {
            bool use_sincos_cache = false;
            marmot_error_t cache_status =
                cpu_rope_sincos_cache_ensure(ctx, &rope_span, positions, N, &use_sincos_cache);
            if (cache_status != MARMOT_SUCCESS) {
                if (rope_span.owns_buffer) {
                    free((void *)rope_span.freqs);
                }
                return cache_status;
            }
            if (use_sincos_cache) {
                sincos_base = ctx->rope_sincos_cache.sincos;
                sincos_stride = ctx->rope_sincos_cache.pair_count * 2;
                sincos_cached_positions = ctx->rope_sincos_cache.cached_positions;
                if (positions->dtype == MARMOT_DTYPE_INT32) {
                    positions_i32 = (const int32_t *)positions->data;
                } else if (positions->dtype == MARMOT_DTYPE_INT64) {
                    positions_i64 = (const int64_t *)positions->data;
                }
            }
        }
    }

    for (size_t n = 0; n < N; ++n) {
        const float *input_row = input_data + n * K;
        float *out_q_row = out_q_data + n * M;
        float *out_k_row = out_k_data + n * M;
        float *out_v_row = out_v_data + n * M;

        size_t m = 0;
        for (; m + 3 < M; m += 4) {
            const float *wq_row0 = wq_data + (m + 0) * K;
            const float *wq_row1 = wq_row0 + K;
            const float *wq_row2 = wq_row1 + K;
            const float *wq_row3 = wq_row2 + K;

            const float *wk_row0 = wk_data + (m + 0) * K;
            const float *wk_row1 = wk_row0 + K;
            const float *wk_row2 = wk_row1 + K;
            const float *wk_row3 = wk_row2 + K;

            const float *wv_row0 = wv_data + (m + 0) * K;
            const float *wv_row1 = wv_row0 + K;
            const float *wv_row2 = wv_row1 + K;
            const float *wv_row3 = wv_row2 + K;

            float32x4_t acc_q0_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_k0_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_v0_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_q1_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_k1_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_v1_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_q2_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_k2_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_v2_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_q3_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_k3_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_v3_vec = vdupq_n_f32(0.0f);

            size_t kk = 0;
            for (; kk + 4 <= K; kk += 4) {
                float32x4_t a_vec = vld1q_f32(input_row + kk);
                float32x4_t wq0_vec = vld1q_f32(wq_row0 + kk);
                float32x4_t wk0_vec = vld1q_f32(wk_row0 + kk);
                float32x4_t wv0_vec = vld1q_f32(wv_row0 + kk);
                float32x4_t wq1_vec = vld1q_f32(wq_row1 + kk);
                float32x4_t wk1_vec = vld1q_f32(wk_row1 + kk);
                float32x4_t wv1_vec = vld1q_f32(wv_row1 + kk);
                float32x4_t wq2_vec = vld1q_f32(wq_row2 + kk);
                float32x4_t wk2_vec = vld1q_f32(wk_row2 + kk);
                float32x4_t wv2_vec = vld1q_f32(wv_row2 + kk);
                float32x4_t wq3_vec = vld1q_f32(wq_row3 + kk);
                float32x4_t wk3_vec = vld1q_f32(wk_row3 + kk);
                float32x4_t wv3_vec = vld1q_f32(wv_row3 + kk);
#if defined(__aarch64__)
                acc_q0_vec = vfmaq_f32(acc_q0_vec, a_vec, wq0_vec);
                acc_k0_vec = vfmaq_f32(acc_k0_vec, a_vec, wk0_vec);
                acc_v0_vec = vfmaq_f32(acc_v0_vec, a_vec, wv0_vec);
                acc_q1_vec = vfmaq_f32(acc_q1_vec, a_vec, wq1_vec);
                acc_k1_vec = vfmaq_f32(acc_k1_vec, a_vec, wk1_vec);
                acc_v1_vec = vfmaq_f32(acc_v1_vec, a_vec, wv1_vec);
                acc_q2_vec = vfmaq_f32(acc_q2_vec, a_vec, wq2_vec);
                acc_k2_vec = vfmaq_f32(acc_k2_vec, a_vec, wk2_vec);
                acc_v2_vec = vfmaq_f32(acc_v2_vec, a_vec, wv2_vec);
                acc_q3_vec = vfmaq_f32(acc_q3_vec, a_vec, wq3_vec);
                acc_k3_vec = vfmaq_f32(acc_k3_vec, a_vec, wk3_vec);
                acc_v3_vec = vfmaq_f32(acc_v3_vec, a_vec, wv3_vec);
#else
                acc_q0_vec = vmlaq_f32(acc_q0_vec, a_vec, wq0_vec);
                acc_k0_vec = vmlaq_f32(acc_k0_vec, a_vec, wk0_vec);
                acc_v0_vec = vmlaq_f32(acc_v0_vec, a_vec, wv0_vec);
                acc_q1_vec = vmlaq_f32(acc_q1_vec, a_vec, wq1_vec);
                acc_k1_vec = vmlaq_f32(acc_k1_vec, a_vec, wk1_vec);
                acc_v1_vec = vmlaq_f32(acc_v1_vec, a_vec, wv1_vec);
                acc_q2_vec = vmlaq_f32(acc_q2_vec, a_vec, wq2_vec);
                acc_k2_vec = vmlaq_f32(acc_k2_vec, a_vec, wk2_vec);
                acc_v2_vec = vmlaq_f32(acc_v2_vec, a_vec, wv2_vec);
                acc_q3_vec = vmlaq_f32(acc_q3_vec, a_vec, wq3_vec);
                acc_k3_vec = vmlaq_f32(acc_k3_vec, a_vec, wk3_vec);
                acc_v3_vec = vmlaq_f32(acc_v3_vec, a_vec, wv3_vec);
#endif
            }

            float acc_q0 = cpu_matmul_qkv_neon_sum(acc_q0_vec);
            float acc_k0 = cpu_matmul_qkv_neon_sum(acc_k0_vec);
            float acc_v0 = cpu_matmul_qkv_neon_sum(acc_v0_vec);
            float acc_q1 = cpu_matmul_qkv_neon_sum(acc_q1_vec);
            float acc_k1 = cpu_matmul_qkv_neon_sum(acc_k1_vec);
            float acc_v1 = cpu_matmul_qkv_neon_sum(acc_v1_vec);
            float acc_q2 = cpu_matmul_qkv_neon_sum(acc_q2_vec);
            float acc_k2 = cpu_matmul_qkv_neon_sum(acc_k2_vec);
            float acc_v2 = cpu_matmul_qkv_neon_sum(acc_v2_vec);
            float acc_q3 = cpu_matmul_qkv_neon_sum(acc_q3_vec);
            float acc_k3 = cpu_matmul_qkv_neon_sum(acc_k3_vec);
            float acc_v3 = cpu_matmul_qkv_neon_sum(acc_v3_vec);

            for (; kk < K; ++kk) {
                const float a = input_row[kk];
                acc_q0 += a * wq_row0[kk];
                acc_k0 += a * wk_row0[kk];
                acc_v0 += a * wv_row0[kk];
                acc_q1 += a * wq_row1[kk];
                acc_k1 += a * wk_row1[kk];
                acc_v1 += a * wv_row1[kk];
                acc_q2 += a * wq_row2[kk];
                acc_k2 += a * wk_row2[kk];
                acc_v2 += a * wv_row2[kk];
                acc_q3 += a * wq_row3[kk];
                acc_k3 += a * wk_row3[kk];
                acc_v3 += a * wv_row3[kk];
            }

            if (bq_data != nullptr) {
                acc_q0 += bq_data[m + 0];
                acc_q1 += bq_data[m + 1];
                acc_q2 += bq_data[m + 2];
                acc_q3 += bq_data[m + 3];
            }
            if (bk_data != nullptr) {
                acc_k0 += bk_data[m + 0];
                acc_k1 += bk_data[m + 1];
                acc_k2 += bk_data[m + 2];
                acc_k3 += bk_data[m + 3];
            }
            if (bv_data != nullptr) {
                acc_v0 += bv_data[m + 0];
                acc_v1 += bv_data[m + 1];
                acc_v2 += bv_data[m + 2];
                acc_v3 += bv_data[m + 3];
            }

            out_q_row[m + 0] = acc_q0;
            out_k_row[m + 0] = acc_k0;
            out_v_row[m + 0] = acc_v0;
            out_q_row[m + 1] = acc_q1;
            out_k_row[m + 1] = acc_k1;
            out_v_row[m + 1] = acc_v1;
            out_q_row[m + 2] = acc_q2;
            out_k_row[m + 2] = acc_k2;
            out_v_row[m + 2] = acc_v2;
            out_q_row[m + 3] = acc_q3;
            out_k_row[m + 3] = acc_k3;
            out_v_row[m + 3] = acc_v3;
        }

        for (; m + 1 < M; m += 2) {
            const float *wq_row0 = wq_data + (m + 0) * K;
            const float *wq_row1 = wq_row0 + K;
            const float *wk_row0 = wk_data + (m + 0) * K;
            const float *wk_row1 = wk_row0 + K;
            const float *wv_row0 = wv_data + (m + 0) * K;
            const float *wv_row1 = wv_row0 + K;

            float32x4_t acc_q0_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_k0_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_v0_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_q1_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_k1_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_v1_vec = vdupq_n_f32(0.0f);

            size_t kk = 0;
            for (; kk + 4 <= K; kk += 4) {
                float32x4_t a_vec = vld1q_f32(input_row + kk);
                float32x4_t wq0_vec = vld1q_f32(wq_row0 + kk);
                float32x4_t wk0_vec = vld1q_f32(wk_row0 + kk);
                float32x4_t wv0_vec = vld1q_f32(wv_row0 + kk);
                float32x4_t wq1_vec = vld1q_f32(wq_row1 + kk);
                float32x4_t wk1_vec = vld1q_f32(wk_row1 + kk);
                float32x4_t wv1_vec = vld1q_f32(wv_row1 + kk);
#if defined(__aarch64__)
                acc_q0_vec = vfmaq_f32(acc_q0_vec, a_vec, wq0_vec);
                acc_k0_vec = vfmaq_f32(acc_k0_vec, a_vec, wk0_vec);
                acc_v0_vec = vfmaq_f32(acc_v0_vec, a_vec, wv0_vec);
                acc_q1_vec = vfmaq_f32(acc_q1_vec, a_vec, wq1_vec);
                acc_k1_vec = vfmaq_f32(acc_k1_vec, a_vec, wk1_vec);
                acc_v1_vec = vfmaq_f32(acc_v1_vec, a_vec, wv1_vec);
#else
                acc_q0_vec = vmlaq_f32(acc_q0_vec, a_vec, wq0_vec);
                acc_k0_vec = vmlaq_f32(acc_k0_vec, a_vec, wk0_vec);
                acc_v0_vec = vmlaq_f32(acc_v0_vec, a_vec, wv0_vec);
                acc_q1_vec = vmlaq_f32(acc_q1_vec, a_vec, wq1_vec);
                acc_k1_vec = vmlaq_f32(acc_k1_vec, a_vec, wk1_vec);
                acc_v1_vec = vmlaq_f32(acc_v1_vec, a_vec, wv1_vec);
#endif
            }

            float acc_q0 = cpu_matmul_qkv_neon_sum(acc_q0_vec);
            float acc_k0 = cpu_matmul_qkv_neon_sum(acc_k0_vec);
            float acc_v0 = cpu_matmul_qkv_neon_sum(acc_v0_vec);
            float acc_q1 = cpu_matmul_qkv_neon_sum(acc_q1_vec);
            float acc_k1 = cpu_matmul_qkv_neon_sum(acc_k1_vec);
            float acc_v1 = cpu_matmul_qkv_neon_sum(acc_v1_vec);

            for (; kk < K; ++kk) {
                const float a = input_row[kk];
                acc_q0 += a * wq_row0[kk];
                acc_k0 += a * wk_row0[kk];
                acc_v0 += a * wv_row0[kk];
                acc_q1 += a * wq_row1[kk];
                acc_k1 += a * wk_row1[kk];
                acc_v1 += a * wv_row1[kk];
            }

            if (bq_data != nullptr) {
                acc_q0 += bq_data[m + 0];
                acc_q1 += bq_data[m + 1];
            }
            if (bk_data != nullptr) {
                acc_k0 += bk_data[m + 0];
                acc_k1 += bk_data[m + 1];
            }
            if (bv_data != nullptr) {
                acc_v0 += bv_data[m + 0];
                acc_v1 += bv_data[m + 1];
            }

            out_q_row[m + 0] = acc_q0;
            out_k_row[m + 0] = acc_k0;
            out_v_row[m + 0] = acc_v0;
            out_q_row[m + 1] = acc_q1;
            out_k_row[m + 1] = acc_k1;
            out_v_row[m + 1] = acc_v1;
        }

        for (; m < M; ++m) {
            const float *wq_row = wq_data + m * K;
            const float *wk_row = wk_data + m * K;
            const float *wv_row = wv_data + m * K;

            float32x4_t acc_q_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_k_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_v_vec = vdupq_n_f32(0.0f);

            size_t kk = 0;
            for (; kk + 4 <= K; kk += 4) {
                float32x4_t a_vec = vld1q_f32(input_row + kk);
                float32x4_t wq_vec = vld1q_f32(wq_row + kk);
                float32x4_t wk_vec = vld1q_f32(wk_row + kk);
                float32x4_t wv_vec = vld1q_f32(wv_row + kk);
#if defined(__aarch64__)
                acc_q_vec = vfmaq_f32(acc_q_vec, a_vec, wq_vec);
                acc_k_vec = vfmaq_f32(acc_k_vec, a_vec, wk_vec);
                acc_v_vec = vfmaq_f32(acc_v_vec, a_vec, wv_vec);
#else
                acc_q_vec = vmlaq_f32(acc_q_vec, a_vec, wq_vec);
                acc_k_vec = vmlaq_f32(acc_k_vec, a_vec, wk_vec);
                acc_v_vec = vmlaq_f32(acc_v_vec, a_vec, wv_vec);
#endif
            }

            float acc_q = cpu_matmul_qkv_neon_sum(acc_q_vec);
            float acc_k = cpu_matmul_qkv_neon_sum(acc_k_vec);
            float acc_v = cpu_matmul_qkv_neon_sum(acc_v_vec);

            for (; kk < K; ++kk) {
                const float a = input_row[kk];
                acc_q += a * wq_row[kk];
                acc_k += a * wk_row[kk];
                acc_v += a * wv_row[kk];
            }

            if (bq_data != nullptr) {
                acc_q += bq_data[m];
            }
            if (bk_data != nullptr) {
                acc_k += bk_data[m];
            }
            if (bv_data != nullptr) {
                acc_v += bv_data[m];
            }

            out_q_row[m] = acc_q;
            out_k_row[m] = acc_k;
            out_v_row[m] = acc_v;
        }

        if (rope_freqs != nullptr && positions != nullptr) {
            const float *sincos = cpu_rope_sincos_lookup(
                sincos_base, sincos_stride, sincos_cached_positions, positions_i32, positions_i64, n
            );
            if (sincos != nullptr) {
                if (apply_rope_q) {
                    cpu_matmul_qkv_rotate_row_f32_sincos_headed(out_q_row, M, rope_head_dim, sincos, rope->rope_type);
                }
                if (apply_rope_k) {
                    cpu_matmul_qkv_rotate_row_f32_sincos_headed(out_k_row, M, rope_head_dim, sincos, rope->rope_type);
                }
            } else {
                const float pos = cpu_rope_position_as_f32(positions, n);
                if (apply_rope_q && apply_rope_k) {
                    cpu_matmul_qkv_rotate_rows_f32_headed(
                        out_q_row, out_k_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope->rope_type
                    );
                } else if (apply_rope_q) {
                    cpu_matmul_qkv_rotate_row_f32_headed(
                        out_q_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope->rope_type
                    );
                } else if (apply_rope_k) {
                    cpu_matmul_qkv_rotate_row_f32_headed(
                        out_k_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope->rope_type
                    );
                }
            }
        }
    }

    if (rope_span.owns_buffer) {
        free((void *)rope_span.freqs);
    }
    return MARMOT_SUCCESS;
#endif
}

marmot_error_t cpu_matmul_qkv_kernel_f32_neon(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    const marmot_tensor_t *bias, marmot_tensor_t *out_q, marmot_tensor_t *out_k, marmot_tensor_t *out_v,
    const marmot_rope_params_t *rope_params
) {
    cpu_context_t *ctx = (cpu_context_t *)device_ctx;
    if (!cpu_matmul_qkv_neon_can_use_contiguous(input, weight, out_q, out_k, out_v, bias, K, M)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const float *input_data = (const float *)input->data;
    const float *weight_data = (const float *)weight->data;
    const float *bias_data = bias != nullptr ? (const float *)bias->data : nullptr;

    float *out_q_data = (float *)out_q->data;
    float *out_k_data = (float *)out_k->data;
    float *out_v_data = (float *)out_v->data;

    const marmot_tensor_t *positions = rope_params != nullptr ? rope_params->positions : nullptr;
    const bool apply_rope_q = rope_params != nullptr && rope_params->apply_to_q;
    const bool apply_rope_k = rope_params != nullptr && rope_params->apply_to_k;
    const size_t rope_head_dim = cpu_matmul_qkv_resolve_head_dim(M, rope_params);
    marmot_rope_freq_span_t rope_span = {0};
    const float *rope_freqs = nullptr;
    const float *sincos_base = nullptr;
    size_t sincos_stride = 0;
    size_t sincos_cached_positions = 0;
    const int32_t *positions_i32 = nullptr;
    const int64_t *positions_i64 = nullptr;
    float rope_attn_scale = 1.0f;
    if (rope_params != nullptr && (apply_rope_q || apply_rope_k)) {
        marmot_error_t freq_status = marmot_rope_freq_cache_ensure(
            ctx != nullptr ? &ctx->rope_cache : nullptr, rope_head_dim, rope_params, &rope_span
        );
        if (freq_status != MARMOT_SUCCESS) {
            return freq_status;
        }
        rope_freqs = rope_span.freqs;
        rope_attn_scale = rope_span.attn_scale;
        if (ctx != nullptr && positions != nullptr) {
            bool use_sincos_cache = false;
            marmot_error_t cache_status =
                cpu_rope_sincos_cache_ensure(ctx, &rope_span, positions, N, &use_sincos_cache);
            if (cache_status != MARMOT_SUCCESS) {
                if (rope_span.owns_buffer) {
                    free((void *)rope_span.freqs);
                }
                return cache_status;
            }
            if (use_sincos_cache) {
                sincos_base = ctx->rope_sincos_cache.sincos;
                sincos_stride = ctx->rope_sincos_cache.pair_count * 2;
                sincos_cached_positions = ctx->rope_sincos_cache.cached_positions;
                if (positions->dtype == MARMOT_DTYPE_INT32) {
                    positions_i32 = (const int32_t *)positions->data;
                } else if (positions->dtype == MARMOT_DTYPE_INT64) {
                    positions_i64 = (const int64_t *)positions->data;
                }
            }
        }
    }

    for (size_t n = 0; n < N; ++n) {
        const float *input_row = input_data + n * K;
        float *out_q_row = out_q_data + n * M;
        float *out_k_row = out_k_data + n * M;
        float *out_v_row = out_v_data + n * M;

        size_t block_cols = cpu_matmul_qkv_neon_select_block_columns(M, K);
        size_t m = 0;

        if (block_cols >= 4) {
            for (; m + 3 < M; m += 4) {
                const float *wq_row0 = weight_data + m * K;
                const float *wq_row1 = wq_row0 + K;
                const float *wq_row2 = wq_row1 + K;
                const float *wq_row3 = wq_row2 + K;

                const float *wk_row0 = weight_data + (M + m) * K;
                const float *wk_row1 = wk_row0 + K;
                const float *wk_row2 = wk_row1 + K;
                const float *wk_row3 = wk_row2 + K;

                const float *wv_row0 = weight_data + (2 * M + m) * K;
                const float *wv_row1 = wv_row0 + K;
                const float *wv_row2 = wv_row1 + K;
                const float *wv_row3 = wv_row2 + K;

                float32x4_t acc_q0_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_k0_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_v0_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_q1_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_k1_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_v1_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_q2_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_k2_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_v2_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_q3_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_k3_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_v3_vec = vdupq_n_f32(0.0f);

                size_t k = 0;
                for (; k + 4 <= K; k += 4) {
                    float32x4_t a_vec = vld1q_f32(input_row + k);
                    float32x4_t wq0_vec = vld1q_f32(wq_row0 + k);
                    float32x4_t wk0_vec = vld1q_f32(wk_row0 + k);
                    float32x4_t wv0_vec = vld1q_f32(wv_row0 + k);
                    float32x4_t wq1_vec = vld1q_f32(wq_row1 + k);
                    float32x4_t wk1_vec = vld1q_f32(wk_row1 + k);
                    float32x4_t wv1_vec = vld1q_f32(wv_row1 + k);
                    float32x4_t wq2_vec = vld1q_f32(wq_row2 + k);
                    float32x4_t wk2_vec = vld1q_f32(wk_row2 + k);
                    float32x4_t wv2_vec = vld1q_f32(wv_row2 + k);
                    float32x4_t wq3_vec = vld1q_f32(wq_row3 + k);
                    float32x4_t wk3_vec = vld1q_f32(wk_row3 + k);
                    float32x4_t wv3_vec = vld1q_f32(wv_row3 + k);
#if defined(__aarch64__)
                    acc_q0_vec = vfmaq_f32(acc_q0_vec, a_vec, wq0_vec);
                    acc_k0_vec = vfmaq_f32(acc_k0_vec, a_vec, wk0_vec);
                    acc_v0_vec = vfmaq_f32(acc_v0_vec, a_vec, wv0_vec);
                    acc_q1_vec = vfmaq_f32(acc_q1_vec, a_vec, wq1_vec);
                    acc_k1_vec = vfmaq_f32(acc_k1_vec, a_vec, wk1_vec);
                    acc_v1_vec = vfmaq_f32(acc_v1_vec, a_vec, wv1_vec);
                    acc_q2_vec = vfmaq_f32(acc_q2_vec, a_vec, wq2_vec);
                    acc_k2_vec = vfmaq_f32(acc_k2_vec, a_vec, wk2_vec);
                    acc_v2_vec = vfmaq_f32(acc_v2_vec, a_vec, wv2_vec);
                    acc_q3_vec = vfmaq_f32(acc_q3_vec, a_vec, wq3_vec);
                    acc_k3_vec = vfmaq_f32(acc_k3_vec, a_vec, wk3_vec);
                    acc_v3_vec = vfmaq_f32(acc_v3_vec, a_vec, wv3_vec);
#else
                    acc_q0_vec = vmlaq_f32(acc_q0_vec, a_vec, wq0_vec);
                    acc_k0_vec = vmlaq_f32(acc_k0_vec, a_vec, wk0_vec);
                    acc_v0_vec = vmlaq_f32(acc_v0_vec, a_vec, wv0_vec);
                    acc_q1_vec = vmlaq_f32(acc_q1_vec, a_vec, wq1_vec);
                    acc_k1_vec = vmlaq_f32(acc_k1_vec, a_vec, wk1_vec);
                    acc_v1_vec = vmlaq_f32(acc_v1_vec, a_vec, wv1_vec);
                    acc_q2_vec = vmlaq_f32(acc_q2_vec, a_vec, wq2_vec);
                    acc_k2_vec = vmlaq_f32(acc_k2_vec, a_vec, wk2_vec);
                    acc_v2_vec = vmlaq_f32(acc_v2_vec, a_vec, wv2_vec);
                    acc_q3_vec = vmlaq_f32(acc_q3_vec, a_vec, wq3_vec);
                    acc_k3_vec = vmlaq_f32(acc_k3_vec, a_vec, wk3_vec);
                    acc_v3_vec = vmlaq_f32(acc_v3_vec, a_vec, wv3_vec);
#endif
                }

                float acc_q0 = cpu_matmul_qkv_neon_sum(acc_q0_vec);
                float acc_k0 = cpu_matmul_qkv_neon_sum(acc_k0_vec);
                float acc_v0 = cpu_matmul_qkv_neon_sum(acc_v0_vec);
                float acc_q1 = cpu_matmul_qkv_neon_sum(acc_q1_vec);
                float acc_k1 = cpu_matmul_qkv_neon_sum(acc_k1_vec);
                float acc_v1 = cpu_matmul_qkv_neon_sum(acc_v1_vec);
                float acc_q2 = cpu_matmul_qkv_neon_sum(acc_q2_vec);
                float acc_k2 = cpu_matmul_qkv_neon_sum(acc_k2_vec);
                float acc_v2 = cpu_matmul_qkv_neon_sum(acc_v2_vec);
                float acc_q3 = cpu_matmul_qkv_neon_sum(acc_q3_vec);
                float acc_k3 = cpu_matmul_qkv_neon_sum(acc_k3_vec);
                float acc_v3 = cpu_matmul_qkv_neon_sum(acc_v3_vec);

                for (; k < K; ++k) {
                    const float a = input_row[k];
                    acc_q0 += a * wq_row0[k];
                    acc_k0 += a * wk_row0[k];
                    acc_v0 += a * wv_row0[k];
                    acc_q1 += a * wq_row1[k];
                    acc_k1 += a * wk_row1[k];
                    acc_v1 += a * wv_row1[k];
                    acc_q2 += a * wq_row2[k];
                    acc_k2 += a * wk_row2[k];
                    acc_v2 += a * wv_row2[k];
                    acc_q3 += a * wq_row3[k];
                    acc_k3 += a * wk_row3[k];
                    acc_v3 += a * wv_row3[k];
                }

                if (bias_data != nullptr) {
                    acc_q0 += bias_data[m + 0];
                    acc_k0 += bias_data[M + m + 0];
                    acc_v0 += bias_data[2 * M + m + 0];
                    acc_q1 += bias_data[m + 1];
                    acc_k1 += bias_data[M + m + 1];
                    acc_v1 += bias_data[2 * M + m + 1];
                    acc_q2 += bias_data[m + 2];
                    acc_k2 += bias_data[M + m + 2];
                    acc_v2 += bias_data[2 * M + m + 2];
                    acc_q3 += bias_data[m + 3];
                    acc_k3 += bias_data[M + m + 3];
                    acc_v3 += bias_data[2 * M + m + 3];
                }

                out_q_row[m + 0] = acc_q0;
                out_k_row[m + 0] = acc_k0;
                out_v_row[m + 0] = acc_v0;
                out_q_row[m + 1] = acc_q1;
                out_k_row[m + 1] = acc_k1;
                out_v_row[m + 1] = acc_v1;
                out_q_row[m + 2] = acc_q2;
                out_k_row[m + 2] = acc_k2;
                out_v_row[m + 2] = acc_v2;
                out_q_row[m + 3] = acc_q3;
                out_k_row[m + 3] = acc_k3;
                out_v_row[m + 3] = acc_v3;
            }
        }

        if (block_cols >= 2) {
            for (; m + 1 < M; m += 2) {
                const float *wq_row0 = weight_data + m * K;
                const float *wk_row0 = weight_data + (M + m) * K;
                const float *wv_row0 = weight_data + (2 * M + m) * K;
                const float *wq_row1 = wq_row0 + K;
                const float *wk_row1 = wk_row0 + K;
                const float *wv_row1 = wv_row0 + K;

                float32x4_t acc_q0_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_k0_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_v0_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_q1_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_k1_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_v1_vec = vdupq_n_f32(0.0f);

                size_t k = 0;
                for (; k + 4 <= K; k += 4) {
                    float32x4_t a_vec = vld1q_f32(input_row + k);
                    float32x4_t wq0_vec = vld1q_f32(wq_row0 + k);
                    float32x4_t wk0_vec = vld1q_f32(wk_row0 + k);
                    float32x4_t wv0_vec = vld1q_f32(wv_row0 + k);
                    float32x4_t wq1_vec = vld1q_f32(wq_row1 + k);
                    float32x4_t wk1_vec = vld1q_f32(wk_row1 + k);
                    float32x4_t wv1_vec = vld1q_f32(wv_row1 + k);
#if defined(__aarch64__)
                    acc_q0_vec = vfmaq_f32(acc_q0_vec, a_vec, wq0_vec);
                    acc_k0_vec = vfmaq_f32(acc_k0_vec, a_vec, wk0_vec);
                    acc_v0_vec = vfmaq_f32(acc_v0_vec, a_vec, wv0_vec);
                    acc_q1_vec = vfmaq_f32(acc_q1_vec, a_vec, wq1_vec);
                    acc_k1_vec = vfmaq_f32(acc_k1_vec, a_vec, wk1_vec);
                    acc_v1_vec = vfmaq_f32(acc_v1_vec, a_vec, wv1_vec);
#else
                    acc_q0_vec = vmlaq_f32(acc_q0_vec, a_vec, wq0_vec);
                    acc_k0_vec = vmlaq_f32(acc_k0_vec, a_vec, wk0_vec);
                    acc_v0_vec = vmlaq_f32(acc_v0_vec, a_vec, wv0_vec);
                    acc_q1_vec = vmlaq_f32(acc_q1_vec, a_vec, wq1_vec);
                    acc_k1_vec = vmlaq_f32(acc_k1_vec, a_vec, wk1_vec);
                    acc_v1_vec = vmlaq_f32(acc_v1_vec, a_vec, wv1_vec);
#endif
                }

                float acc_q0 = cpu_matmul_qkv_neon_sum(acc_q0_vec);
                float acc_k0 = cpu_matmul_qkv_neon_sum(acc_k0_vec);
                float acc_v0 = cpu_matmul_qkv_neon_sum(acc_v0_vec);
                float acc_q1 = cpu_matmul_qkv_neon_sum(acc_q1_vec);
                float acc_k1 = cpu_matmul_qkv_neon_sum(acc_k1_vec);
                float acc_v1 = cpu_matmul_qkv_neon_sum(acc_v1_vec);

                for (; k < K; ++k) {
                    const float a = input_row[k];
                    acc_q0 += a * wq_row0[k];
                    acc_k0 += a * wk_row0[k];
                    acc_v0 += a * wv_row0[k];
                    acc_q1 += a * wq_row1[k];
                    acc_k1 += a * wk_row1[k];
                    acc_v1 += a * wv_row1[k];
                }

                if (bias_data != nullptr) {
                    acc_q0 += bias_data[m];
                    acc_k0 += bias_data[M + m];
                    acc_v0 += bias_data[2 * M + m];
                    acc_q1 += bias_data[m + 1];
                    acc_k1 += bias_data[M + m + 1];
                    acc_v1 += bias_data[2 * M + m + 1];
                }

                out_q_row[m] = acc_q0;
                out_k_row[m] = acc_k0;
                out_v_row[m] = acc_v0;
                out_q_row[m + 1] = acc_q1;
                out_k_row[m + 1] = acc_k1;
                out_v_row[m + 1] = acc_v1;
            }
        }

        for (; m < M; ++m) {
            const float *wq_row = weight_data + m * K;
            const float *wk_row = weight_data + (M + m) * K;
            const float *wv_row = weight_data + (2 * M + m) * K;

            float32x4_t acc_q_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_k_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_v_vec = vdupq_n_f32(0.0f);

            size_t k = 0;
            for (; k + 4 <= K; k += 4) {
                float32x4_t a_vec = vld1q_f32(input_row + k);
                float32x4_t wq_vec = vld1q_f32(wq_row + k);
                float32x4_t wk_vec = vld1q_f32(wk_row + k);
                float32x4_t wv_vec = vld1q_f32(wv_row + k);
#if defined(__aarch64__)
                acc_q_vec = vfmaq_f32(acc_q_vec, a_vec, wq_vec);
                acc_k_vec = vfmaq_f32(acc_k_vec, a_vec, wk_vec);
                acc_v_vec = vfmaq_f32(acc_v_vec, a_vec, wv_vec);
#else
                acc_q_vec = vmlaq_f32(acc_q_vec, a_vec, wq_vec);
                acc_k_vec = vmlaq_f32(acc_k_vec, a_vec, wk_vec);
                acc_v_vec = vmlaq_f32(acc_v_vec, a_vec, wv_vec);
#endif
            }

            float acc_q = cpu_matmul_qkv_neon_sum(acc_q_vec);
            float acc_k = cpu_matmul_qkv_neon_sum(acc_k_vec);
            float acc_v = cpu_matmul_qkv_neon_sum(acc_v_vec);

            for (; k < K; ++k) {
                const float a = input_row[k];
                acc_q += a * wq_row[k];
                acc_k += a * wk_row[k];
                acc_v += a * wv_row[k];
            }

            if (bias_data != nullptr) {
                acc_q += bias_data[m];
                acc_k += bias_data[M + m];
                acc_v += bias_data[2 * M + m];
            }

            out_q_row[m] = acc_q;
            out_k_row[m] = acc_k;
            out_v_row[m] = acc_v;
        }

        if (rope_freqs != nullptr && positions != nullptr) {
            const float *sincos = cpu_rope_sincos_lookup(
                sincos_base, sincos_stride, sincos_cached_positions, positions_i32, positions_i64, n
            );
            if (sincos != nullptr) {
                if (apply_rope_q) {
                    cpu_matmul_qkv_rotate_row_f32_sincos_headed(
                        out_q_row, M, rope_head_dim, sincos, rope_params->rope_type
                    );
                }
                if (apply_rope_k) {
                    cpu_matmul_qkv_rotate_row_f32_sincos_headed(
                        out_k_row, M, rope_head_dim, sincos, rope_params->rope_type
                    );
                }
            } else {
                const float pos = cpu_rope_position_as_f32(positions, n);
                if (apply_rope_q && apply_rope_k) {
                    cpu_matmul_qkv_rotate_rows_f32_headed(
                        out_q_row, out_k_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope_params->rope_type
                    );
                } else if (apply_rope_q) {
                    cpu_matmul_qkv_rotate_row_f32_headed(
                        out_q_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope_params->rope_type
                    );
                } else if (apply_rope_k) {
                    cpu_matmul_qkv_rotate_row_f32_headed(
                        out_k_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope_params->rope_type
                    );
                }
            }
        }
    }

    if (rope_span.owns_buffer) {
        free((void *)rope_span.freqs);
    }

    return MARMOT_SUCCESS;
}

#if defined(__aarch64__)
static inline float32x4_t cpu_matmul_qkv_neon_load_f32_from_f16(const marmot_float16_t *ptr) {
    uint16x4_t bits = vld1_u16((const uint16_t *)ptr);
    float16x4_t half = vreinterpret_f16_u16(bits);
    return vcvt_f32_f16(half);
}
#endif

marmot_error_t cpu_matmul_qkv_kernel_f16_neon(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    const marmot_tensor_t *bias, marmot_tensor_t *out_q, marmot_tensor_t *out_k, marmot_tensor_t *out_v,
    const marmot_rope_params_t *rope_params
) {
#if !defined(__aarch64__)
    (void)device_ctx;
    (void)rope_params;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)bias;
    (void)out_q;
    (void)out_k;
    (void)out_v;
    return MARMOT_ERROR_NOT_IMPLEMENTED;
#else
    cpu_context_t *ctx = (cpu_context_t *)device_ctx;
    if (!cpu_matmul_qkv_neon_can_use_contiguous(input, weight, out_q, out_k, out_v, bias, K, M)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const marmot_float16_t *input_data = (const marmot_float16_t *)input->data;
    const marmot_float16_t *weight_data = (const marmot_float16_t *)weight->data;
    const marmot_float16_t *bias_data = bias != nullptr ? (const marmot_float16_t *)bias->data : nullptr;
    marmot_float16_t *out_q_data = (marmot_float16_t *)out_q->data;
    marmot_float16_t *out_k_data = (marmot_float16_t *)out_k->data;
    marmot_float16_t *out_v_data = (marmot_float16_t *)out_v->data;

    const marmot_tensor_t *positions = rope_params != nullptr ? rope_params->positions : nullptr;
    const bool apply_rope_q = rope_params != nullptr && rope_params->apply_to_q;
    const bool apply_rope_k = rope_params != nullptr && rope_params->apply_to_k;
    const size_t rope_head_dim = cpu_matmul_qkv_resolve_head_dim(M, rope_params);
    marmot_rope_freq_span_t rope_span = {0};
    const float *rope_freqs = nullptr;
    const float *sincos_base = nullptr;
    size_t sincos_stride = 0;
    size_t sincos_cached_positions = 0;
    const int32_t *positions_i32 = nullptr;
    const int64_t *positions_i64 = nullptr;
    float rope_attn_scale = 1.0f;
    if (rope_params != nullptr && (apply_rope_q || apply_rope_k)) {
        marmot_error_t freq_status = marmot_rope_freq_cache_ensure(
            ctx != nullptr ? &ctx->rope_cache : nullptr, rope_head_dim, rope_params, &rope_span
        );
        if (freq_status != MARMOT_SUCCESS) {
            return freq_status;
        }
        rope_freqs = rope_span.freqs;
        rope_attn_scale = rope_span.attn_scale;
        if (ctx != nullptr && positions != nullptr) {
            bool use_sincos_cache = false;
            marmot_error_t cache_status =
                cpu_rope_sincos_cache_ensure(ctx, &rope_span, positions, N, &use_sincos_cache);
            if (cache_status != MARMOT_SUCCESS) {
                if (rope_span.owns_buffer) {
                    free((void *)rope_span.freqs);
                }
                return cache_status;
            }
            if (use_sincos_cache) {
                sincos_base = ctx->rope_sincos_cache.sincos;
                sincos_stride = ctx->rope_sincos_cache.pair_count * 2;
                sincos_cached_positions = ctx->rope_sincos_cache.cached_positions;
                if (positions->dtype == MARMOT_DTYPE_INT32) {
                    positions_i32 = (const int32_t *)positions->data;
                } else if (positions->dtype == MARMOT_DTYPE_INT64) {
                    positions_i64 = (const int64_t *)positions->data;
                }
            }
        }
    }

    for (size_t n = 0; n < N; ++n) {
        const marmot_float16_t *input_row = input_data + n * K;
        marmot_float16_t *out_q_row = out_q_data + n * M;
        marmot_float16_t *out_k_row = out_k_data + n * M;
        marmot_float16_t *out_v_row = out_v_data + n * M;

        size_t block_cols = cpu_matmul_qkv_neon_select_block_columns(M, K);
        size_t m = 0;

        if (block_cols >= 4) {
            for (; m + 3 < M; m += 4) {
                const marmot_float16_t *wq_row0 = weight_data + m * K;
                const marmot_float16_t *wq_row1 = wq_row0 + K;
                const marmot_float16_t *wq_row2 = wq_row1 + K;
                const marmot_float16_t *wq_row3 = wq_row2 + K;

                const marmot_float16_t *wk_row0 = weight_data + (M + m) * K;
                const marmot_float16_t *wk_row1 = wk_row0 + K;
                const marmot_float16_t *wk_row2 = wk_row1 + K;
                const marmot_float16_t *wk_row3 = wk_row2 + K;

                const marmot_float16_t *wv_row0 = weight_data + (2 * M + m) * K;
                const marmot_float16_t *wv_row1 = wv_row0 + K;
                const marmot_float16_t *wv_row2 = wv_row1 + K;
                const marmot_float16_t *wv_row3 = wv_row2 + K;

                float32x4_t acc_q0_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_k0_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_v0_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_q1_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_k1_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_v1_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_q2_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_k2_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_v2_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_q3_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_k3_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_v3_vec = vdupq_n_f32(0.0f);

                size_t k = 0;
                for (; k + 4 <= K; k += 4) {
                    float32x4_t a_vec = cpu_matmul_qkv_neon_load_f32_from_f16(input_row + k);
                    float32x4_t wq0_vec = cpu_matmul_qkv_neon_load_f32_from_f16(wq_row0 + k);
                    float32x4_t wk0_vec = cpu_matmul_qkv_neon_load_f32_from_f16(wk_row0 + k);
                    float32x4_t wv0_vec = cpu_matmul_qkv_neon_load_f32_from_f16(wv_row0 + k);
                    float32x4_t wq1_vec = cpu_matmul_qkv_neon_load_f32_from_f16(wq_row1 + k);
                    float32x4_t wk1_vec = cpu_matmul_qkv_neon_load_f32_from_f16(wk_row1 + k);
                    float32x4_t wv1_vec = cpu_matmul_qkv_neon_load_f32_from_f16(wv_row1 + k);
                    float32x4_t wq2_vec = cpu_matmul_qkv_neon_load_f32_from_f16(wq_row2 + k);
                    float32x4_t wk2_vec = cpu_matmul_qkv_neon_load_f32_from_f16(wk_row2 + k);
                    float32x4_t wv2_vec = cpu_matmul_qkv_neon_load_f32_from_f16(wv_row2 + k);
                    float32x4_t wq3_vec = cpu_matmul_qkv_neon_load_f32_from_f16(wq_row3 + k);
                    float32x4_t wk3_vec = cpu_matmul_qkv_neon_load_f32_from_f16(wk_row3 + k);
                    float32x4_t wv3_vec = cpu_matmul_qkv_neon_load_f32_from_f16(wv_row3 + k);
#if defined(__aarch64__)
                    acc_q0_vec = vfmaq_f32(acc_q0_vec, a_vec, wq0_vec);
                    acc_k0_vec = vfmaq_f32(acc_k0_vec, a_vec, wk0_vec);
                    acc_v0_vec = vfmaq_f32(acc_v0_vec, a_vec, wv0_vec);
                    acc_q1_vec = vfmaq_f32(acc_q1_vec, a_vec, wq1_vec);
                    acc_k1_vec = vfmaq_f32(acc_k1_vec, a_vec, wk1_vec);
                    acc_v1_vec = vfmaq_f32(acc_v1_vec, a_vec, wv1_vec);
                    acc_q2_vec = vfmaq_f32(acc_q2_vec, a_vec, wq2_vec);
                    acc_k2_vec = vfmaq_f32(acc_k2_vec, a_vec, wk2_vec);
                    acc_v2_vec = vfmaq_f32(acc_v2_vec, a_vec, wv2_vec);
                    acc_q3_vec = vfmaq_f32(acc_q3_vec, a_vec, wq3_vec);
                    acc_k3_vec = vfmaq_f32(acc_k3_vec, a_vec, wk3_vec);
                    acc_v3_vec = vfmaq_f32(acc_v3_vec, a_vec, wv3_vec);
#else
                    acc_q0_vec = vmlaq_f32(acc_q0_vec, a_vec, wq0_vec);
                    acc_k0_vec = vmlaq_f32(acc_k0_vec, a_vec, wk0_vec);
                    acc_v0_vec = vmlaq_f32(acc_v0_vec, a_vec, wv0_vec);
                    acc_q1_vec = vmlaq_f32(acc_q1_vec, a_vec, wq1_vec);
                    acc_k1_vec = vmlaq_f32(acc_k1_vec, a_vec, wk1_vec);
                    acc_v1_vec = vmlaq_f32(acc_v1_vec, a_vec, wv1_vec);
                    acc_q2_vec = vmlaq_f32(acc_q2_vec, a_vec, wq2_vec);
                    acc_k2_vec = vmlaq_f32(acc_k2_vec, a_vec, wk2_vec);
                    acc_v2_vec = vmlaq_f32(acc_v2_vec, a_vec, wv2_vec);
                    acc_q3_vec = vmlaq_f32(acc_q3_vec, a_vec, wq3_vec);
                    acc_k3_vec = vmlaq_f32(acc_k3_vec, a_vec, wk3_vec);
                    acc_v3_vec = vmlaq_f32(acc_v3_vec, a_vec, wv3_vec);
#endif
                }

                float acc_q0 = cpu_matmul_qkv_neon_sum(acc_q0_vec);
                float acc_k0 = cpu_matmul_qkv_neon_sum(acc_k0_vec);
                float acc_v0 = cpu_matmul_qkv_neon_sum(acc_v0_vec);
                float acc_q1 = cpu_matmul_qkv_neon_sum(acc_q1_vec);
                float acc_k1 = cpu_matmul_qkv_neon_sum(acc_k1_vec);
                float acc_v1 = cpu_matmul_qkv_neon_sum(acc_v1_vec);
                float acc_q2 = cpu_matmul_qkv_neon_sum(acc_q2_vec);
                float acc_k2 = cpu_matmul_qkv_neon_sum(acc_k2_vec);
                float acc_v2 = cpu_matmul_qkv_neon_sum(acc_v2_vec);
                float acc_q3 = cpu_matmul_qkv_neon_sum(acc_q3_vec);
                float acc_k3 = cpu_matmul_qkv_neon_sum(acc_k3_vec);
                float acc_v3 = cpu_matmul_qkv_neon_sum(acc_v3_vec);

                for (; k < K; ++k) {
                    const float a = (float)marmot_float16_to_native(input_row[k]);
                    acc_q0 += a * (float)marmot_float16_to_native(wq_row0[k]);
                    acc_k0 += a * (float)marmot_float16_to_native(wk_row0[k]);
                    acc_v0 += a * (float)marmot_float16_to_native(wv_row0[k]);
                    acc_q1 += a * (float)marmot_float16_to_native(wq_row1[k]);
                    acc_k1 += a * (float)marmot_float16_to_native(wk_row1[k]);
                    acc_v1 += a * (float)marmot_float16_to_native(wv_row1[k]);
                    acc_q2 += a * (float)marmot_float16_to_native(wq_row2[k]);
                    acc_k2 += a * (float)marmot_float16_to_native(wk_row2[k]);
                    acc_v2 += a * (float)marmot_float16_to_native(wv_row2[k]);
                    acc_q3 += a * (float)marmot_float16_to_native(wq_row3[k]);
                    acc_k3 += a * (float)marmot_float16_to_native(wk_row3[k]);
                    acc_v3 += a * (float)marmot_float16_to_native(wv_row3[k]);
                }

                if (bias_data != nullptr) {
                    acc_q0 += (float)marmot_float16_to_native(bias_data[m + 0]);
                    acc_k0 += (float)marmot_float16_to_native(bias_data[M + m + 0]);
                    acc_v0 += (float)marmot_float16_to_native(bias_data[2 * M + m + 0]);
                    acc_q1 += (float)marmot_float16_to_native(bias_data[m + 1]);
                    acc_k1 += (float)marmot_float16_to_native(bias_data[M + m + 1]);
                    acc_v1 += (float)marmot_float16_to_native(bias_data[2 * M + m + 1]);
                    acc_q2 += (float)marmot_float16_to_native(bias_data[m + 2]);
                    acc_k2 += (float)marmot_float16_to_native(bias_data[M + m + 2]);
                    acc_v2 += (float)marmot_float16_to_native(bias_data[2 * M + m + 2]);
                    acc_q3 += (float)marmot_float16_to_native(bias_data[m + 3]);
                    acc_k3 += (float)marmot_float16_to_native(bias_data[M + m + 3]);
                    acc_v3 += (float)marmot_float16_to_native(bias_data[2 * M + m + 3]);
                }

                out_q_row[m + 0] = marmot_native_to_float16((_Float16)acc_q0);
                out_k_row[m + 0] = marmot_native_to_float16((_Float16)acc_k0);
                out_v_row[m + 0] = marmot_native_to_float16((_Float16)acc_v0);
                out_q_row[m + 1] = marmot_native_to_float16((_Float16)acc_q1);
                out_k_row[m + 1] = marmot_native_to_float16((_Float16)acc_k1);
                out_v_row[m + 1] = marmot_native_to_float16((_Float16)acc_v1);
                out_q_row[m + 2] = marmot_native_to_float16((_Float16)acc_q2);
                out_k_row[m + 2] = marmot_native_to_float16((_Float16)acc_k2);
                out_v_row[m + 2] = marmot_native_to_float16((_Float16)acc_v2);
                out_q_row[m + 3] = marmot_native_to_float16((_Float16)acc_q3);
                out_k_row[m + 3] = marmot_native_to_float16((_Float16)acc_k3);
                out_v_row[m + 3] = marmot_native_to_float16((_Float16)acc_v3);
            }
        }

        for (; m < M; ++m) {
            const marmot_float16_t *wq_row = weight_data + m * K;
            const marmot_float16_t *wk_row = weight_data + (M + m) * K;
            const marmot_float16_t *wv_row = weight_data + (2 * M + m) * K;

            float32x4_t acc_q_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_k_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_v_vec = vdupq_n_f32(0.0f);

            size_t k = 0;
            for (; k + 4 <= K; k += 4) {
                float32x4_t a_vec = cpu_matmul_qkv_neon_load_f32_from_f16(input_row + k);
                float32x4_t wq_vec = cpu_matmul_qkv_neon_load_f32_from_f16(wq_row + k);
                float32x4_t wk_vec = cpu_matmul_qkv_neon_load_f32_from_f16(wk_row + k);
                float32x4_t wv_vec = cpu_matmul_qkv_neon_load_f32_from_f16(wv_row + k);
#if defined(__aarch64__)
                acc_q_vec = vfmaq_f32(acc_q_vec, a_vec, wq_vec);
                acc_k_vec = vfmaq_f32(acc_k_vec, a_vec, wk_vec);
                acc_v_vec = vfmaq_f32(acc_v_vec, a_vec, wv_vec);
#else
                acc_q_vec = vmlaq_f32(acc_q_vec, a_vec, wq_vec);
                acc_k_vec = vmlaq_f32(acc_k_vec, a_vec, wk_vec);
                acc_v_vec = vmlaq_f32(acc_v_vec, a_vec, wv_vec);
#endif
            }

            float acc_q = cpu_matmul_qkv_neon_sum(acc_q_vec);
            float acc_k = cpu_matmul_qkv_neon_sum(acc_k_vec);
            float acc_v = cpu_matmul_qkv_neon_sum(acc_v_vec);

            for (; k < K; ++k) {
                const float a = (float)marmot_float16_to_native(input_row[k]);
                acc_q += a * (float)marmot_float16_to_native(wq_row[k]);
                acc_k += a * (float)marmot_float16_to_native(wk_row[k]);
                acc_v += a * (float)marmot_float16_to_native(wv_row[k]);
            }

            if (bias_data != nullptr) {
                acc_q += (float)marmot_float16_to_native(bias_data[m]);
                acc_k += (float)marmot_float16_to_native(bias_data[M + m]);
                acc_v += (float)marmot_float16_to_native(bias_data[2 * M + m]);
            }

            out_q_row[m] = marmot_native_to_float16((_Float16)acc_q);
            out_k_row[m] = marmot_native_to_float16((_Float16)acc_k);
            out_v_row[m] = marmot_native_to_float16((_Float16)acc_v);
        }

        if (rope_freqs != nullptr && positions != nullptr) {
            const float *sincos = cpu_rope_sincos_lookup(
                sincos_base, sincos_stride, sincos_cached_positions, positions_i32, positions_i64, n
            );
            if (sincos != nullptr) {
                if (apply_rope_q) {
                    cpu_matmul_qkv_rotate_row_f16_sincos_headed(
                        out_q_row, M, rope_head_dim, sincos, rope_params->rope_type
                    );
                }
                if (apply_rope_k) {
                    cpu_matmul_qkv_rotate_row_f16_sincos_headed(
                        out_k_row, M, rope_head_dim, sincos, rope_params->rope_type
                    );
                }
            } else {
                const float pos = cpu_rope_position_as_f32(positions, n);
                if (apply_rope_q && apply_rope_k) {
                    cpu_matmul_qkv_rotate_rows_f16_headed(
                        out_q_row, out_k_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope_params->rope_type
                    );
                } else if (apply_rope_q) {
                    cpu_matmul_qkv_rotate_row_f16_headed(
                        out_q_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope_params->rope_type
                    );
                } else if (apply_rope_k) {
                    cpu_matmul_qkv_rotate_row_f16_headed(
                        out_k_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope_params->rope_type
                    );
                }
            }
        }
    }

    if (rope_span.owns_buffer) {
        free((void *)rope_span.freqs);
    }

    return MARMOT_SUCCESS;
#endif
}

static inline float32x4_t cpu_matmul_qkv_neon_load_f32_from_bf16(const marmot_bfloat16_t *ptr) {
    uint16x4_t bits = vld1_u16((const uint16_t *)ptr);
    uint32x4_t shifted = vshll_n_u16(bits, 16);
    return vreinterpretq_f32_u32(shifted);
}

marmot_error_t cpu_matmul_qkv_kernel_bf16_neon(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    const marmot_tensor_t *bias, marmot_tensor_t *out_q, marmot_tensor_t *out_k, marmot_tensor_t *out_v,
    const marmot_rope_params_t *rope_params
) {
    cpu_context_t *ctx = (cpu_context_t *)device_ctx;
    if (!cpu_matmul_qkv_neon_can_use_contiguous(input, weight, out_q, out_k, out_v, bias, K, M)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const marmot_bfloat16_t *input_data = (const marmot_bfloat16_t *)input->data;
    const marmot_bfloat16_t *weight_data = (const marmot_bfloat16_t *)weight->data;
    const marmot_bfloat16_t *bias_data = bias != nullptr ? (const marmot_bfloat16_t *)bias->data : nullptr;
    marmot_bfloat16_t *out_q_data = (marmot_bfloat16_t *)out_q->data;
    marmot_bfloat16_t *out_k_data = (marmot_bfloat16_t *)out_k->data;
    marmot_bfloat16_t *out_v_data = (marmot_bfloat16_t *)out_v->data;

    const marmot_tensor_t *positions = rope_params != nullptr ? rope_params->positions : nullptr;
    const bool apply_rope_q = rope_params != nullptr && rope_params->apply_to_q;
    const bool apply_rope_k = rope_params != nullptr && rope_params->apply_to_k;
    const size_t rope_head_dim = cpu_matmul_qkv_resolve_head_dim(M, rope_params);
    marmot_rope_freq_span_t rope_span = {0};
    const float *rope_freqs = nullptr;
    const float *sincos_base = nullptr;
    size_t sincos_stride = 0;
    size_t sincos_cached_positions = 0;
    const int32_t *positions_i32 = nullptr;
    const int64_t *positions_i64 = nullptr;
    float rope_attn_scale = 1.0f;
    if (rope_params != nullptr && (apply_rope_q || apply_rope_k)) {
        marmot_error_t freq_status = marmot_rope_freq_cache_ensure(
            ctx != nullptr ? &ctx->rope_cache : nullptr, rope_head_dim, rope_params, &rope_span
        );
        if (freq_status != MARMOT_SUCCESS) {
            return freq_status;
        }
        rope_freqs = rope_span.freqs;
        rope_attn_scale = rope_span.attn_scale;
        if (ctx != nullptr && positions != nullptr) {
            bool use_sincos_cache = false;
            marmot_error_t cache_status =
                cpu_rope_sincos_cache_ensure(ctx, &rope_span, positions, N, &use_sincos_cache);
            if (cache_status != MARMOT_SUCCESS) {
                if (rope_span.owns_buffer) {
                    free((void *)rope_span.freqs);
                }
                return cache_status;
            }
            if (use_sincos_cache) {
                sincos_base = ctx->rope_sincos_cache.sincos;
                sincos_stride = ctx->rope_sincos_cache.pair_count * 2;
                sincos_cached_positions = ctx->rope_sincos_cache.cached_positions;
                if (positions->dtype == MARMOT_DTYPE_INT32) {
                    positions_i32 = (const int32_t *)positions->data;
                } else if (positions->dtype == MARMOT_DTYPE_INT64) {
                    positions_i64 = (const int64_t *)positions->data;
                }
            }
        }
    }

    for (size_t n = 0; n < N; ++n) {
        const marmot_bfloat16_t *input_row = input_data + n * K;
        marmot_bfloat16_t *out_q_row = out_q_data + n * M;
        marmot_bfloat16_t *out_k_row = out_k_data + n * M;
        marmot_bfloat16_t *out_v_row = out_v_data + n * M;

        size_t block_cols = cpu_matmul_qkv_neon_select_block_columns(M, K);
        size_t m = 0;

        if (block_cols >= 4) {
            for (; m + 3 < M; m += 4) {
                const marmot_bfloat16_t *wq_row0 = weight_data + m * K;
                const marmot_bfloat16_t *wq_row1 = wq_row0 + K;
                const marmot_bfloat16_t *wq_row2 = wq_row1 + K;
                const marmot_bfloat16_t *wq_row3 = wq_row2 + K;

                const marmot_bfloat16_t *wk_row0 = weight_data + (M + m) * K;
                const marmot_bfloat16_t *wk_row1 = wk_row0 + K;
                const marmot_bfloat16_t *wk_row2 = wk_row1 + K;
                const marmot_bfloat16_t *wk_row3 = wk_row2 + K;

                const marmot_bfloat16_t *wv_row0 = weight_data + (2 * M + m) * K;
                const marmot_bfloat16_t *wv_row1 = wv_row0 + K;
                const marmot_bfloat16_t *wv_row2 = wv_row1 + K;
                const marmot_bfloat16_t *wv_row3 = wv_row2 + K;

                float32x4_t acc_q0_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_k0_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_v0_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_q1_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_k1_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_v1_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_q2_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_k2_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_v2_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_q3_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_k3_vec = vdupq_n_f32(0.0f);
                float32x4_t acc_v3_vec = vdupq_n_f32(0.0f);

                size_t k = 0;
                for (; k + 4 <= K; k += 4) {
                    float32x4_t a_vec = cpu_matmul_qkv_neon_load_f32_from_bf16(input_row + k);
                    float32x4_t wq0_vec = cpu_matmul_qkv_neon_load_f32_from_bf16(wq_row0 + k);
                    float32x4_t wk0_vec = cpu_matmul_qkv_neon_load_f32_from_bf16(wk_row0 + k);
                    float32x4_t wv0_vec = cpu_matmul_qkv_neon_load_f32_from_bf16(wv_row0 + k);
                    float32x4_t wq1_vec = cpu_matmul_qkv_neon_load_f32_from_bf16(wq_row1 + k);
                    float32x4_t wk1_vec = cpu_matmul_qkv_neon_load_f32_from_bf16(wk_row1 + k);
                    float32x4_t wv1_vec = cpu_matmul_qkv_neon_load_f32_from_bf16(wv_row1 + k);
                    float32x4_t wq2_vec = cpu_matmul_qkv_neon_load_f32_from_bf16(wq_row2 + k);
                    float32x4_t wk2_vec = cpu_matmul_qkv_neon_load_f32_from_bf16(wk_row2 + k);
                    float32x4_t wv2_vec = cpu_matmul_qkv_neon_load_f32_from_bf16(wv_row2 + k);
                    float32x4_t wq3_vec = cpu_matmul_qkv_neon_load_f32_from_bf16(wq_row3 + k);
                    float32x4_t wk3_vec = cpu_matmul_qkv_neon_load_f32_from_bf16(wk_row3 + k);
                    float32x4_t wv3_vec = cpu_matmul_qkv_neon_load_f32_from_bf16(wv_row3 + k);
#if defined(__aarch64__)
                    acc_q0_vec = vfmaq_f32(acc_q0_vec, a_vec, wq0_vec);
                    acc_k0_vec = vfmaq_f32(acc_k0_vec, a_vec, wk0_vec);
                    acc_v0_vec = vfmaq_f32(acc_v0_vec, a_vec, wv0_vec);
                    acc_q1_vec = vfmaq_f32(acc_q1_vec, a_vec, wq1_vec);
                    acc_k1_vec = vfmaq_f32(acc_k1_vec, a_vec, wk1_vec);
                    acc_v1_vec = vfmaq_f32(acc_v1_vec, a_vec, wv1_vec);
                    acc_q2_vec = vfmaq_f32(acc_q2_vec, a_vec, wq2_vec);
                    acc_k2_vec = vfmaq_f32(acc_k2_vec, a_vec, wk2_vec);
                    acc_v2_vec = vfmaq_f32(acc_v2_vec, a_vec, wv2_vec);
                    acc_q3_vec = vfmaq_f32(acc_q3_vec, a_vec, wq3_vec);
                    acc_k3_vec = vfmaq_f32(acc_k3_vec, a_vec, wk3_vec);
                    acc_v3_vec = vfmaq_f32(acc_v3_vec, a_vec, wv3_vec);
#else
                    acc_q0_vec = vmlaq_f32(acc_q0_vec, a_vec, wq0_vec);
                    acc_k0_vec = vmlaq_f32(acc_k0_vec, a_vec, wk0_vec);
                    acc_v0_vec = vmlaq_f32(acc_v0_vec, a_vec, wv0_vec);
                    acc_q1_vec = vmlaq_f32(acc_q1_vec, a_vec, wq1_vec);
                    acc_k1_vec = vmlaq_f32(acc_k1_vec, a_vec, wk1_vec);
                    acc_v1_vec = vmlaq_f32(acc_v1_vec, a_vec, wv1_vec);
                    acc_q2_vec = vmlaq_f32(acc_q2_vec, a_vec, wq2_vec);
                    acc_k2_vec = vmlaq_f32(acc_k2_vec, a_vec, wk2_vec);
                    acc_v2_vec = vmlaq_f32(acc_v2_vec, a_vec, wv2_vec);
                    acc_q3_vec = vmlaq_f32(acc_q3_vec, a_vec, wq3_vec);
                    acc_k3_vec = vmlaq_f32(acc_k3_vec, a_vec, wk3_vec);
                    acc_v3_vec = vmlaq_f32(acc_v3_vec, a_vec, wv3_vec);
#endif
                }

                float acc_q0 = cpu_matmul_qkv_neon_sum(acc_q0_vec);
                float acc_k0 = cpu_matmul_qkv_neon_sum(acc_k0_vec);
                float acc_v0 = cpu_matmul_qkv_neon_sum(acc_v0_vec);
                float acc_q1 = cpu_matmul_qkv_neon_sum(acc_q1_vec);
                float acc_k1 = cpu_matmul_qkv_neon_sum(acc_k1_vec);
                float acc_v1 = cpu_matmul_qkv_neon_sum(acc_v1_vec);
                float acc_q2 = cpu_matmul_qkv_neon_sum(acc_q2_vec);
                float acc_k2 = cpu_matmul_qkv_neon_sum(acc_k2_vec);
                float acc_v2 = cpu_matmul_qkv_neon_sum(acc_v2_vec);
                float acc_q3 = cpu_matmul_qkv_neon_sum(acc_q3_vec);
                float acc_k3 = cpu_matmul_qkv_neon_sum(acc_k3_vec);
                float acc_v3 = cpu_matmul_qkv_neon_sum(acc_v3_vec);

                for (; k < K; ++k) {
                    const float a = marmot_bfloat16_to_native(input_row[k]);
                    acc_q0 += a * marmot_bfloat16_to_native(wq_row0[k]);
                    acc_k0 += a * marmot_bfloat16_to_native(wk_row0[k]);
                    acc_v0 += a * marmot_bfloat16_to_native(wv_row0[k]);
                    acc_q1 += a * marmot_bfloat16_to_native(wq_row1[k]);
                    acc_k1 += a * marmot_bfloat16_to_native(wk_row1[k]);
                    acc_v1 += a * marmot_bfloat16_to_native(wv_row1[k]);
                    acc_q2 += a * marmot_bfloat16_to_native(wq_row2[k]);
                    acc_k2 += a * marmot_bfloat16_to_native(wk_row2[k]);
                    acc_v2 += a * marmot_bfloat16_to_native(wv_row2[k]);
                    acc_q3 += a * marmot_bfloat16_to_native(wq_row3[k]);
                    acc_k3 += a * marmot_bfloat16_to_native(wk_row3[k]);
                    acc_v3 += a * marmot_bfloat16_to_native(wv_row3[k]);
                }

                if (bias_data != nullptr) {
                    acc_q0 += marmot_bfloat16_to_native(bias_data[m + 0]);
                    acc_k0 += marmot_bfloat16_to_native(bias_data[M + m + 0]);
                    acc_v0 += marmot_bfloat16_to_native(bias_data[2 * M + m + 0]);
                    acc_q1 += marmot_bfloat16_to_native(bias_data[m + 1]);
                    acc_k1 += marmot_bfloat16_to_native(bias_data[M + m + 1]);
                    acc_v1 += marmot_bfloat16_to_native(bias_data[2 * M + m + 1]);
                    acc_q2 += marmot_bfloat16_to_native(bias_data[m + 2]);
                    acc_k2 += marmot_bfloat16_to_native(bias_data[M + m + 2]);
                    acc_v2 += marmot_bfloat16_to_native(bias_data[2 * M + m + 2]);
                    acc_q3 += marmot_bfloat16_to_native(bias_data[m + 3]);
                    acc_k3 += marmot_bfloat16_to_native(bias_data[M + m + 3]);
                    acc_v3 += marmot_bfloat16_to_native(bias_data[2 * M + m + 3]);
                }

                out_q_row[m + 0] = marmot_native_to_bfloat16(acc_q0);
                out_k_row[m + 0] = marmot_native_to_bfloat16(acc_k0);
                out_v_row[m + 0] = marmot_native_to_bfloat16(acc_v0);
                out_q_row[m + 1] = marmot_native_to_bfloat16(acc_q1);
                out_k_row[m + 1] = marmot_native_to_bfloat16(acc_k1);
                out_v_row[m + 1] = marmot_native_to_bfloat16(acc_v1);
                out_q_row[m + 2] = marmot_native_to_bfloat16(acc_q2);
                out_k_row[m + 2] = marmot_native_to_bfloat16(acc_k2);
                out_v_row[m + 2] = marmot_native_to_bfloat16(acc_v2);
                out_q_row[m + 3] = marmot_native_to_bfloat16(acc_q3);
                out_k_row[m + 3] = marmot_native_to_bfloat16(acc_k3);
                out_v_row[m + 3] = marmot_native_to_bfloat16(acc_v3);
            }
        }

        for (; m < M; ++m) {
            const marmot_bfloat16_t *wq_row = weight_data + m * K;
            const marmot_bfloat16_t *wk_row = weight_data + (M + m) * K;
            const marmot_bfloat16_t *wv_row = weight_data + (2 * M + m) * K;

            float32x4_t acc_q_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_k_vec = vdupq_n_f32(0.0f);
            float32x4_t acc_v_vec = vdupq_n_f32(0.0f);

            size_t k = 0;
            for (; k + 4 <= K; k += 4) {
                float32x4_t a_vec = cpu_matmul_qkv_neon_load_f32_from_bf16(input_row + k);
                float32x4_t wq_vec = cpu_matmul_qkv_neon_load_f32_from_bf16(wq_row + k);
                float32x4_t wk_vec = cpu_matmul_qkv_neon_load_f32_from_bf16(wk_row + k);
                float32x4_t wv_vec = cpu_matmul_qkv_neon_load_f32_from_bf16(wv_row + k);
#if defined(__aarch64__)
                acc_q_vec = vfmaq_f32(acc_q_vec, a_vec, wq_vec);
                acc_k_vec = vfmaq_f32(acc_k_vec, a_vec, wk_vec);
                acc_v_vec = vfmaq_f32(acc_v_vec, a_vec, wv_vec);
#else
                acc_q_vec = vmlaq_f32(acc_q_vec, a_vec, wq_vec);
                acc_k_vec = vmlaq_f32(acc_k_vec, a_vec, wk_vec);
                acc_v_vec = vmlaq_f32(acc_v_vec, a_vec, wv_vec);
#endif
            }

            float acc_q = cpu_matmul_qkv_neon_sum(acc_q_vec);
            float acc_k = cpu_matmul_qkv_neon_sum(acc_k_vec);
            float acc_v = cpu_matmul_qkv_neon_sum(acc_v_vec);

            for (; k < K; ++k) {
                const float a = marmot_bfloat16_to_native(input_row[k]);
                acc_q += a * marmot_bfloat16_to_native(wq_row[k]);
                acc_k += a * marmot_bfloat16_to_native(wk_row[k]);
                acc_v += a * marmot_bfloat16_to_native(wv_row[k]);
            }

            if (bias_data != nullptr) {
                acc_q += marmot_bfloat16_to_native(bias_data[m]);
                acc_k += marmot_bfloat16_to_native(bias_data[M + m]);
                acc_v += marmot_bfloat16_to_native(bias_data[2 * M + m]);
            }

            out_q_row[m] = marmot_native_to_bfloat16(acc_q);
            out_k_row[m] = marmot_native_to_bfloat16(acc_k);
            out_v_row[m] = marmot_native_to_bfloat16(acc_v);
        }

        if (rope_freqs != nullptr && positions != nullptr) {
            const float *sincos = cpu_rope_sincos_lookup(
                sincos_base, sincos_stride, sincos_cached_positions, positions_i32, positions_i64, n
            );
            if (sincos != nullptr) {
                if (apply_rope_q) {
                    cpu_matmul_qkv_rotate_row_bf16_sincos_headed(
                        out_q_row, M, rope_head_dim, sincos, rope_params->rope_type
                    );
                }
                if (apply_rope_k) {
                    cpu_matmul_qkv_rotate_row_bf16_sincos_headed(
                        out_k_row, M, rope_head_dim, sincos, rope_params->rope_type
                    );
                }
            } else {
                const float pos = cpu_rope_position_as_f32(positions, n);
                if (apply_rope_q && apply_rope_k) {
                    cpu_matmul_qkv_rotate_rows_bf16_headed(
                        out_q_row, out_k_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope_params->rope_type
                    );
                } else if (apply_rope_q) {
                    cpu_matmul_qkv_rotate_row_bf16_headed(
                        out_q_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope_params->rope_type
                    );
                } else if (apply_rope_k) {
                    cpu_matmul_qkv_rotate_row_bf16_headed(
                        out_k_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope_params->rope_type
                    );
                }
            }
        }
    }

    if (rope_span.owns_buffer) {
        free((void *)rope_span.freqs);
    }

    return MARMOT_SUCCESS;
}

#endif // HAS_NEON
