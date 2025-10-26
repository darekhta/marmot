#include "cpu_backend_internal.h"
#include "matmul_qkv_rope.h"

#if HAS_AVX2

#include <immintrin.h>

#define MATMUL_QKV_AVX2_MIN_WORK_F32 (1024 * 1024 * 4ULL)
#define MATMUL_QKV_AVX2_MIN_WORK_F16 (1024 * 1024 * 4ULL)
#define MATMUL_QKV_AVX2_MIN_WORK_BF16 (1024 * 1024 * 6ULL)

static inline float cpu_matmul_qkv_avx2_sum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static inline bool cpu_matmul_qkv_avx2_can_use_contiguous(
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

marmot_error_t cpu_matmul_qkv_kernel_f32_avx2(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    const marmot_tensor_t *bias, marmot_tensor_t *out_q, marmot_tensor_t *out_k, marmot_tensor_t *out_v,
    const marmot_rope_params_t *rope_params
) {
    if (!cpu_matmul_qkv_avx2_can_use_contiguous(input, weight, out_q, out_k, out_v, bias, K, M)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    cpu_context_t *ctx = (cpu_context_t *)device_ctx;

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

        size_t m = 0;
        for (; m + 1 < M; m += 2) {
            const float *wq_row0 = weight_data + m * K;
            const float *wk_row0 = weight_data + (M + m) * K;
            const float *wv_row0 = weight_data + (2 * M + m) * K;
            const float *wq_row1 = wq_row0 + K;
            const float *wk_row1 = wk_row0 + K;
            const float *wv_row1 = wv_row0 + K;

            __m256 acc_q0_vec = _mm256_setzero_ps();
            __m256 acc_k0_vec = _mm256_setzero_ps();
            __m256 acc_v0_vec = _mm256_setzero_ps();
            __m256 acc_q1_vec = _mm256_setzero_ps();
            __m256 acc_k1_vec = _mm256_setzero_ps();
            __m256 acc_v1_vec = _mm256_setzero_ps();

            size_t k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(input_row + k);
                __m256 wq0_vec = _mm256_loadu_ps(wq_row0 + k);
                __m256 wk0_vec = _mm256_loadu_ps(wk_row0 + k);
                __m256 wv0_vec = _mm256_loadu_ps(wv_row0 + k);
                __m256 wq1_vec = _mm256_loadu_ps(wq_row1 + k);
                __m256 wk1_vec = _mm256_loadu_ps(wk_row1 + k);
                __m256 wv1_vec = _mm256_loadu_ps(wv_row1 + k);
                acc_q0_vec = _mm256_fmadd_ps(a_vec, wq0_vec, acc_q0_vec);
                acc_k0_vec = _mm256_fmadd_ps(a_vec, wk0_vec, acc_k0_vec);
                acc_v0_vec = _mm256_fmadd_ps(a_vec, wv0_vec, acc_v0_vec);
                acc_q1_vec = _mm256_fmadd_ps(a_vec, wq1_vec, acc_q1_vec);
                acc_k1_vec = _mm256_fmadd_ps(a_vec, wk1_vec, acc_k1_vec);
                acc_v1_vec = _mm256_fmadd_ps(a_vec, wv1_vec, acc_v1_vec);
            }

            float acc_q0 = cpu_matmul_qkv_avx2_sum(acc_q0_vec);
            float acc_k0 = cpu_matmul_qkv_avx2_sum(acc_k0_vec);
            float acc_v0 = cpu_matmul_qkv_avx2_sum(acc_v0_vec);
            float acc_q1 = cpu_matmul_qkv_avx2_sum(acc_q1_vec);
            float acc_k1 = cpu_matmul_qkv_avx2_sum(acc_k1_vec);
            float acc_v1 = cpu_matmul_qkv_avx2_sum(acc_v1_vec);

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

        for (; m < M; ++m) {
            const float *wq_row = weight_data + m * K;
            const float *wk_row = weight_data + (M + m) * K;
            const float *wv_row = weight_data + (2 * M + m) * K;

            __m256 acc_q_vec = _mm256_setzero_ps();
            __m256 acc_k_vec = _mm256_setzero_ps();
            __m256 acc_v_vec = _mm256_setzero_ps();

            size_t k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(input_row + k);
                __m256 wq_vec = _mm256_loadu_ps(wq_row + k);
                __m256 wk_vec = _mm256_loadu_ps(wk_row + k);
                __m256 wv_vec = _mm256_loadu_ps(wv_row + k);
                acc_q_vec = _mm256_fmadd_ps(a_vec, wq_vec, acc_q_vec);
                acc_k_vec = _mm256_fmadd_ps(a_vec, wk_vec, acc_k_vec);
                acc_v_vec = _mm256_fmadd_ps(a_vec, wv_vec, acc_v_vec);
            }

            float acc_q = cpu_matmul_qkv_avx2_sum(acc_q_vec);
            float acc_k = cpu_matmul_qkv_avx2_sum(acc_k_vec);
            float acc_v = cpu_matmul_qkv_avx2_sum(acc_v_vec);

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

#if defined(__F16C__)
static inline __m256 cpu_matmul_qkv_avx2_load_f32_from_f16(const marmot_float16_t *ptr) {
    __m128i bits = _mm_loadu_si128((const __m128i *)ptr);
    return _mm256_cvtph_ps(bits);
}
#endif

marmot_error_t cpu_matmul_qkv_kernel_f16_avx2(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    const marmot_tensor_t *bias, marmot_tensor_t *out_q, marmot_tensor_t *out_k, marmot_tensor_t *out_v,
    const marmot_rope_params_t *rope_params
) {
#if !defined(__F16C__)
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
    if (!cpu_matmul_qkv_avx2_can_use_contiguous(input, weight, out_q, out_k, out_v, bias, K, M)) {
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

        for (size_t m = 0; m < M; ++m) {
            const marmot_float16_t *wq_row = weight_data + m * K;
            const marmot_float16_t *wk_row = weight_data + (M + m) * K;
            const marmot_float16_t *wv_row = weight_data + (2 * M + m) * K;

            __m256 acc_q_vec = _mm256_setzero_ps();
            __m256 acc_k_vec = _mm256_setzero_ps();
            __m256 acc_v_vec = _mm256_setzero_ps();

            size_t k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 a_vec = cpu_matmul_qkv_avx2_load_f32_from_f16(input_row + k);
                __m256 wq_vec = cpu_matmul_qkv_avx2_load_f32_from_f16(wq_row + k);
                __m256 wk_vec = cpu_matmul_qkv_avx2_load_f32_from_f16(wk_row + k);
                __m256 wv_vec = cpu_matmul_qkv_avx2_load_f32_from_f16(wv_row + k);
                acc_q_vec = _mm256_fmadd_ps(a_vec, wq_vec, acc_q_vec);
                acc_k_vec = _mm256_fmadd_ps(a_vec, wk_vec, acc_k_vec);
                acc_v_vec = _mm256_fmadd_ps(a_vec, wv_vec, acc_v_vec);
            }

            float acc_q = cpu_matmul_qkv_avx2_sum(acc_q_vec);
            float acc_k = cpu_matmul_qkv_avx2_sum(acc_k_vec);
            float acc_v = cpu_matmul_qkv_avx2_sum(acc_v_vec);

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

static inline __m256 cpu_matmul_qkv_avx2_load_f32_from_bf16(const marmot_bfloat16_t *ptr) {
    __m128i half = _mm_loadu_si128((const __m128i *)ptr);
    __m256i shifted = _mm256_slli_epi32(_mm256_cvtepu16_epi32(half), 16);
    return _mm256_castsi256_ps(shifted);
}

marmot_error_t cpu_matmul_qkv_kernel_bf16_avx2(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    const marmot_tensor_t *bias, marmot_tensor_t *out_q, marmot_tensor_t *out_k, marmot_tensor_t *out_v,
    const marmot_rope_params_t *rope_params
) {
    cpu_context_t *ctx = (cpu_context_t *)device_ctx;
    if (!cpu_matmul_qkv_avx2_can_use_contiguous(input, weight, out_q, out_k, out_v, bias, K, M)) {
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

        for (size_t m = 0; m < M; ++m) {
            const marmot_bfloat16_t *wq_row = weight_data + m * K;
            const marmot_bfloat16_t *wk_row = weight_data + (M + m) * K;
            const marmot_bfloat16_t *wv_row = weight_data + (2 * M + m) * K;

            __m256 acc_q_vec = _mm256_setzero_ps();
            __m256 acc_k_vec = _mm256_setzero_ps();
            __m256 acc_v_vec = _mm256_setzero_ps();

            size_t k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 a_vec = cpu_matmul_qkv_avx2_load_f32_from_bf16(input_row + k);
                __m256 wq_vec = cpu_matmul_qkv_avx2_load_f32_from_bf16(wq_row + k);
                __m256 wk_vec = cpu_matmul_qkv_avx2_load_f32_from_bf16(wk_row + k);
                __m256 wv_vec = cpu_matmul_qkv_avx2_load_f32_from_bf16(wv_row + k);
                acc_q_vec = _mm256_fmadd_ps(a_vec, wq_vec, acc_q_vec);
                acc_k_vec = _mm256_fmadd_ps(a_vec, wk_vec, acc_k_vec);
                acc_v_vec = _mm256_fmadd_ps(a_vec, wv_vec, acc_v_vec);
            }

            float acc_q = cpu_matmul_qkv_avx2_sum(acc_q_vec);
            float acc_k = cpu_matmul_qkv_avx2_sum(acc_k_vec);
            float acc_v = cpu_matmul_qkv_avx2_sum(acc_v_vec);

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

#endif // HAS_AVX2
