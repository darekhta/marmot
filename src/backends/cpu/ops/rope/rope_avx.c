#include <math.h>

#include "cpu_backend_internal.h"
#include "ops/matmul/qkv/matmul_qkv_rope.h"
#include "rope_kernels.h"

#if HAS_AVX2

#include <immintrin.h>

static inline __m256 cpu_avx_load_bf16(const marmot_bfloat16_t *src) {
    __m128i bf16_vec = _mm_loadu_si128((const __m128i *)src);
    __m256i f32_bits = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_vec), 16);
    return _mm256_castsi256_ps(f32_bits);
}

static inline void cpu_avx_store_bf16(marmot_bfloat16_t *dst, __m256 value) {
    const __m256i rounding_bias = _mm256_set1_epi32(0x7FFF);
    __m256i f32_bits = _mm256_castps_si256(value);
    __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(f32_bits, 16), _mm256_set1_epi32(1));
    __m256i bias = _mm256_add_epi32(rounding_bias, lsb);
    __m256i rounded = _mm256_add_epi32(f32_bits, bias);
    __m256i shifted = _mm256_srli_epi32(rounded, 16);
    __m128i lo = _mm256_castsi256_si128(shifted);
    __m128i hi = _mm256_extracti128_si256(shifted, 1);
    __m128i bf16_vec = _mm_packus_epi32(lo, hi);
    bf16_vec = _mm_permute4x64_epi64(bf16_vec, 0xD8);
    _mm_storeu_si128((__m128i *)dst, bf16_vec);
}

static marmot_error_t cpu_rope_kernel_avx_run(
    cpu_context_t *ctx, const marmot_tensor_t *x, const marmot_tensor_t *positions, const float *freqs,
    float attn_scale, marmot_rope_type_t rope_type, size_t seq_len, size_t dim, size_t total_seqs, marmot_tensor_t *out
) {
    if (x->dtype != MARMOT_DTYPE_FLOAT32 || out->dtype != MARMOT_DTYPE_FLOAT32) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (freqs == nullptr || dim == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t pair_count = dim / 2;
    const size_t half_dim = dim / 2;
    const size_t total_tokens = total_seqs * seq_len;
    const float *input = (const float *)x->data;
    float *output = (float *)out->data;
    const cpu_rope_sincos_cache_t *cache = ctx != nullptr ? &ctx->rope_sincos_cache : nullptr;
    const bool cache_ok = cache != nullptr && cache->sincos != nullptr && cache->pair_count == pair_count &&
        cache->dim == dim && cache->cached_positions > 0;
    const float *sincos_base = cache_ok ? cache->sincos : nullptr;
    const size_t sincos_stride = cache_ok ? cache->pair_count * 2 : 0;
    const size_t sincos_cached_positions = cache_ok ? cache->cached_positions : 0;
    const int32_t *positions_i32 = cache_ok && positions != nullptr && positions->dtype == MARMOT_DTYPE_INT32
        ? (const int32_t *)positions->data
        : nullptr;
    const int64_t *positions_i64 = cache_ok && positions != nullptr && positions->dtype == MARMOT_DTYPE_INT64
        ? (const int64_t *)positions->data
        : nullptr;

    for (size_t token = 0; token < total_tokens; ++token) {
        const float *sincos = cpu_rope_sincos_lookup(
            sincos_base, sincos_stride, sincos_cached_positions, positions_i32, positions_i64, token
        );
        float pos = 0.0f;
        if (sincos == nullptr) {
            pos = cpu_rope_position_as_f32(positions, token);
        }
        const float *row_in = input + token * dim;
        float *row_out = output + token * dim;

        if (rope_type == MARMOT_ROPE_TYPE_NEOX) {
            size_t i = 0;
            for (; i + 7 < pair_count; i += 8) {
                float cos_vals[8];
                float sin_vals[8];
                if (sincos != nullptr) {
                    for (size_t lane = 0; lane < 8; ++lane) {
                        cos_vals[lane] = sincos[2 * (i + lane)];
                        sin_vals[lane] = sincos[2 * (i + lane) + 1];
                    }
                } else {
                    for (size_t lane = 0; lane < 8; ++lane) {
                        const float angle = pos * freqs[i + lane];
                        float sin_theta = 0.0f;
                        float cos_theta = 0.0f;
                        cpu_sincosf(angle, &sin_theta, &cos_theta);
                        cos_vals[lane] = cos_theta * attn_scale;
                        sin_vals[lane] = sin_theta * attn_scale;
                    }
                }

                const __m256 cos_vec = _mm256_loadu_ps(cos_vals);
                const __m256 sin_vec = _mm256_loadu_ps(sin_vals);
                const __m256 even_vec = _mm256_loadu_ps(row_in + i);
                const __m256 odd_vec = _mm256_loadu_ps(row_in + half_dim + i);

                const __m256 even_cos = _mm256_mul_ps(even_vec, cos_vec);
                const __m256 odd_sin = _mm256_mul_ps(odd_vec, sin_vec);
                const __m256 out_even = _mm256_sub_ps(even_cos, odd_sin);

                const __m256 even_sin = _mm256_mul_ps(even_vec, sin_vec);
                const __m256 odd_cos = _mm256_mul_ps(odd_vec, cos_vec);
                const __m256 out_odd = _mm256_add_ps(even_sin, odd_cos);

                _mm256_storeu_ps(row_out + i, out_even);
                _mm256_storeu_ps(row_out + half_dim + i, out_odd);
            }

            for (; i < pair_count; ++i) {
                float c = 0.0f;
                float s = 0.0f;
                if (sincos != nullptr) {
                    c = sincos[2 * i];
                    s = sincos[2 * i + 1];
                } else {
                    const float angle = pos * freqs[i];
                    cpu_sincosf(angle, &s, &c);
                    c *= attn_scale;
                    s *= attn_scale;
                }
                const size_t even_index = i;
                const size_t odd_index = i + half_dim;
                const float even_val = row_in[even_index];
                const float odd_val = row_in[odd_index];
                row_out[even_index] = even_val * c - odd_val * s;
                row_out[odd_index] = even_val * s + odd_val * c;
            }
        } else {
            size_t i = 0;
            for (; i + 3 < pair_count; i += 4) {
                float cos_vals[4];
                float sin_vals[4];
                if (sincos != nullptr) {
                    for (size_t lane = 0; lane < 4; ++lane) {
                        cos_vals[lane] = sincos[2 * (i + lane)];
                        sin_vals[lane] = sincos[2 * (i + lane) + 1];
                    }
                } else {
                    for (size_t lane = 0; lane < 4; ++lane) {
                        const float angle = pos * freqs[i + lane];
                        float sin_theta = 0.0f;
                        float cos_theta = 0.0f;
                        cpu_sincosf(angle, &sin_theta, &cos_theta);
                        cos_vals[lane] = cos_theta * attn_scale;
                        sin_vals[lane] = sin_theta * attn_scale;
                    }
                }

                float cos_dup_vals[8] = {
                    cos_vals[0], cos_vals[0], cos_vals[1], cos_vals[1],
                    cos_vals[2], cos_vals[2], cos_vals[3], cos_vals[3],
                };
                float sin_dup_vals[8] = {
                    sin_vals[0], sin_vals[0], sin_vals[1], sin_vals[1],
                    sin_vals[2], sin_vals[2], sin_vals[3], sin_vals[3],
                };

                const __m256 cos_dup = _mm256_loadu_ps(cos_dup_vals);
                const __m256 sin_dup = _mm256_loadu_ps(sin_dup_vals);
                const __m256 data = _mm256_loadu_ps(row_in + 2 * i);

                const __m256 real_vals = _mm256_moveldup_ps(data);
                const __m256 imag_vals = _mm256_movehdup_ps(data);

                const __m256 real_cos = _mm256_mul_ps(real_vals, cos_dup);
                const __m256 imag_sin = _mm256_mul_ps(imag_vals, sin_dup);
                const __m256 out_real = _mm256_sub_ps(real_cos, imag_sin);

                const __m256 real_sin = _mm256_mul_ps(real_vals, sin_dup);
                const __m256 imag_cos = _mm256_mul_ps(imag_vals, cos_dup);
                const __m256 out_imag = _mm256_add_ps(real_sin, imag_cos);

                const __m256 low = _mm256_unpacklo_ps(out_real, out_imag);
                const __m256 high = _mm256_unpackhi_ps(out_real, out_imag);
                const __m256 packed = _mm256_shuffle_ps(low, high, _MM_SHUFFLE(1, 0, 1, 0));
                _mm256_storeu_ps(row_out + 2 * i, packed);
            }

            for (; i < pair_count; ++i) {
                float c = 0.0f;
                float s = 0.0f;
                if (sincos != nullptr) {
                    c = sincos[2 * i];
                    s = sincos[2 * i + 1];
                } else {
                    const float angle = pos * freqs[i];
                    cpu_sincosf(angle, &s, &c);
                    c *= attn_scale;
                    s *= attn_scale;
                }
                const size_t even_index = 2 * i;
                const size_t odd_index = even_index + 1;
                const float even_val = row_in[even_index];
                const float odd_val = row_in[odd_index];
                row_out[even_index] = even_val * c - odd_val * s;
                row_out[odd_index] = even_val * s + odd_val * c;
            }
        }
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_rope_kernel_avx_f16_run(
    cpu_context_t *ctx, const marmot_tensor_t *x, const marmot_tensor_t *positions, const float *freqs,
    float attn_scale, marmot_rope_type_t rope_type, size_t seq_len, size_t dim, size_t total_seqs, marmot_tensor_t *out
) {
#if HAS_F16C
    if (x->dtype != MARMOT_DTYPE_FLOAT16 || out->dtype != MARMOT_DTYPE_FLOAT16) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (ctx != nullptr && !cpu_ctx_has_f16c(ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (freqs == nullptr || dim == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t pair_count = dim / 2;
    const size_t half_dim = dim / 2;
    const size_t total_tokens = total_seqs * seq_len;
    const marmot_float16_t *input = (const marmot_float16_t *)x->data;
    marmot_float16_t *output = (marmot_float16_t *)out->data;
    const cpu_rope_sincos_cache_t *cache = ctx != nullptr ? &ctx->rope_sincos_cache : nullptr;
    const bool cache_ok = cache != nullptr && cache->sincos != nullptr && cache->pair_count == pair_count &&
        cache->dim == dim && cache->cached_positions > 0;
    const float *sincos_base = cache_ok ? cache->sincos : nullptr;
    const size_t sincos_stride = cache_ok ? cache->pair_count * 2 : 0;
    const size_t sincos_cached_positions = cache_ok ? cache->cached_positions : 0;
    const int32_t *positions_i32 = cache_ok && positions != nullptr && positions->dtype == MARMOT_DTYPE_INT32
        ? (const int32_t *)positions->data
        : nullptr;
    const int64_t *positions_i64 = cache_ok && positions != nullptr && positions->dtype == MARMOT_DTYPE_INT64
        ? (const int64_t *)positions->data
        : nullptr;

    for (size_t token = 0; token < total_tokens; ++token) {
        const float *sincos = cpu_rope_sincos_lookup(
            sincos_base, sincos_stride, sincos_cached_positions, positions_i32, positions_i64, token
        );
        float pos = 0.0f;
        if (sincos == nullptr) {
            pos = cpu_rope_position_as_f32(positions, token);
        }
        const marmot_float16_t *row_in = input + token * dim;
        marmot_float16_t *row_out = output + token * dim;

        if (rope_type == MARMOT_ROPE_TYPE_NEOX) {
            size_t i = 0;
            for (; i + 7 < pair_count; i += 8) {
                float cos_vals[8];
                float sin_vals[8];
                if (sincos != nullptr) {
                    for (size_t lane = 0; lane < 8; ++lane) {
                        cos_vals[lane] = sincos[2 * (i + lane)];
                        sin_vals[lane] = sincos[2 * (i + lane) + 1];
                    }
                } else {
                    for (size_t lane = 0; lane < 8; ++lane) {
                        const float angle = pos * freqs[i + lane];
                        float sin_theta = 0.0f;
                        float cos_theta = 0.0f;
                        cpu_sincosf(angle, &sin_theta, &cos_theta);
                        cos_vals[lane] = cos_theta * attn_scale;
                        sin_vals[lane] = sin_theta * attn_scale;
                    }
                }

                const __m256 cos_vec = _mm256_loadu_ps(cos_vals);
                const __m256 sin_vec = _mm256_loadu_ps(sin_vals);
                const __m256 even_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row_in + i)));
                const __m256 odd_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row_in + half_dim + i)));

                const __m256 even_cos = _mm256_mul_ps(even_vec, cos_vec);
                const __m256 odd_sin = _mm256_mul_ps(odd_vec, sin_vec);
                const __m256 out_even = _mm256_sub_ps(even_cos, odd_sin);

                const __m256 even_sin = _mm256_mul_ps(even_vec, sin_vec);
                const __m256 odd_cos = _mm256_mul_ps(odd_vec, cos_vec);
                const __m256 out_odd = _mm256_add_ps(even_sin, odd_cos);

                _mm_storeu_si128((__m128i *)(row_out + i), _mm256_cvtps_ph(out_even, _MM_FROUND_TO_NEAREST_INT));
                _mm_storeu_si128(
                    (__m128i *)(row_out + half_dim + i), _mm256_cvtps_ph(out_odd, _MM_FROUND_TO_NEAREST_INT)
                );
            }

            for (; i < pair_count; ++i) {
                float c = 0.0f;
                float s = 0.0f;
                if (sincos != nullptr) {
                    c = sincos[2 * i];
                    s = sincos[2 * i + 1];
                } else {
                    const float angle = pos * freqs[i];
                    cpu_sincosf(angle, &s, &c);
                    c *= attn_scale;
                    s *= attn_scale;
                }
                const size_t even_index = i;
                const size_t odd_index = i + half_dim;
                const float even_val = (float)marmot_float16_to_native(row_in[even_index]);
                const float odd_val = (float)marmot_float16_to_native(row_in[odd_index]);
                row_out[even_index] = marmot_native_to_float16((_Float16)(even_val * c - odd_val * s));
                row_out[odd_index] = marmot_native_to_float16((_Float16)(even_val * s + odd_val * c));
            }
        } else {
            size_t i = 0;
            for (; i + 3 < pair_count; i += 4) {
                float cos_vals[4];
                float sin_vals[4];
                if (sincos != nullptr) {
                    for (size_t lane = 0; lane < 4; ++lane) {
                        cos_vals[lane] = sincos[2 * (i + lane)];
                        sin_vals[lane] = sincos[2 * (i + lane) + 1];
                    }
                } else {
                    for (size_t lane = 0; lane < 4; ++lane) {
                        const float angle = pos * freqs[i + lane];
                        float sin_theta = 0.0f;
                        float cos_theta = 0.0f;
                        cpu_sincosf(angle, &sin_theta, &cos_theta);
                        cos_vals[lane] = cos_theta * attn_scale;
                        sin_vals[lane] = sin_theta * attn_scale;
                    }
                }

                float cos_dup_vals[8] = {
                    cos_vals[0], cos_vals[0], cos_vals[1], cos_vals[1],
                    cos_vals[2], cos_vals[2], cos_vals[3], cos_vals[3],
                };
                float sin_dup_vals[8] = {
                    sin_vals[0], sin_vals[0], sin_vals[1], sin_vals[1],
                    sin_vals[2], sin_vals[2], sin_vals[3], sin_vals[3],
                };

                const __m256 cos_dup = _mm256_loadu_ps(cos_dup_vals);
                const __m256 sin_dup = _mm256_loadu_ps(sin_dup_vals);
                const __m256 data = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row_in + 2 * i)));

                const __m256 real_vals = _mm256_moveldup_ps(data);
                const __m256 imag_vals = _mm256_movehdup_ps(data);

                const __m256 real_cos = _mm256_mul_ps(real_vals, cos_dup);
                const __m256 imag_sin = _mm256_mul_ps(imag_vals, sin_dup);
                const __m256 out_real = _mm256_sub_ps(real_cos, imag_sin);

                const __m256 real_sin = _mm256_mul_ps(real_vals, sin_dup);
                const __m256 imag_cos = _mm256_mul_ps(imag_vals, cos_dup);
                const __m256 out_imag = _mm256_add_ps(real_sin, imag_cos);

                const __m256 low = _mm256_unpacklo_ps(out_real, out_imag);
                const __m256 high = _mm256_unpackhi_ps(out_real, out_imag);
                const __m256 packed = _mm256_shuffle_ps(low, high, _MM_SHUFFLE(1, 0, 1, 0));

                _mm_storeu_si128((__m128i *)(row_out + 2 * i), _mm256_cvtps_ph(packed, _MM_FROUND_TO_NEAREST_INT));
            }

            for (; i < pair_count; ++i) {
                float c = 0.0f;
                float s = 0.0f;
                if (sincos != nullptr) {
                    c = sincos[2 * i];
                    s = sincos[2 * i + 1];
                } else {
                    const float angle = pos * freqs[i];
                    cpu_sincosf(angle, &s, &c);
                    c *= attn_scale;
                    s *= attn_scale;
                }
                const size_t even_index = 2 * i;
                const size_t odd_index = even_index + 1;
                const float even_val = (float)marmot_float16_to_native(row_in[even_index]);
                const float odd_val = (float)marmot_float16_to_native(row_in[odd_index]);
                row_out[even_index] = marmot_native_to_float16((_Float16)(even_val * c - odd_val * s));
                row_out[odd_index] = marmot_native_to_float16((_Float16)(even_val * s + odd_val * c));
            }
        }
    }

    return MARMOT_SUCCESS;
#else
    (void)ctx;
    (void)x;
    (void)positions;
    (void)freqs;
    (void)attn_scale;
    (void)rope_type;
    (void)seq_len;
    (void)dim;
    (void)total_seqs;
    (void)out;
    return MARMOT_ERROR_NOT_IMPLEMENTED;
#endif
}

static marmot_error_t cpu_rope_kernel_avx_bf16_run(
    cpu_context_t *ctx, const marmot_tensor_t *x, const marmot_tensor_t *positions, const float *freqs,
    float attn_scale, marmot_rope_type_t rope_type, size_t seq_len, size_t dim, size_t total_seqs, marmot_tensor_t *out
) {
    if (x->dtype != MARMOT_DTYPE_BFLOAT16 || out->dtype != MARMOT_DTYPE_BFLOAT16) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (freqs == nullptr || dim == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t pair_count = dim / 2;
    const size_t half_dim = dim / 2;
    const size_t total_tokens = total_seqs * seq_len;
    const marmot_bfloat16_t *input = (const marmot_bfloat16_t *)x->data;
    marmot_bfloat16_t *output = (marmot_bfloat16_t *)out->data;
    const cpu_rope_sincos_cache_t *cache = ctx != nullptr ? &ctx->rope_sincos_cache : nullptr;
    const bool cache_ok = cache != nullptr && cache->sincos != nullptr && cache->pair_count == pair_count &&
        cache->dim == dim && cache->cached_positions > 0;
    const float *sincos_base = cache_ok ? cache->sincos : nullptr;
    const size_t sincos_stride = cache_ok ? cache->pair_count * 2 : 0;
    const size_t sincos_cached_positions = cache_ok ? cache->cached_positions : 0;
    const int32_t *positions_i32 = cache_ok && positions != nullptr && positions->dtype == MARMOT_DTYPE_INT32
        ? (const int32_t *)positions->data
        : nullptr;
    const int64_t *positions_i64 = cache_ok && positions != nullptr && positions->dtype == MARMOT_DTYPE_INT64
        ? (const int64_t *)positions->data
        : nullptr;

    for (size_t token = 0; token < total_tokens; ++token) {
        const float *sincos = cpu_rope_sincos_lookup(
            sincos_base, sincos_stride, sincos_cached_positions, positions_i32, positions_i64, token
        );
        float pos = 0.0f;
        if (sincos == nullptr) {
            pos = cpu_rope_position_as_f32(positions, token);
        }
        const marmot_bfloat16_t *row_in = input + token * dim;
        marmot_bfloat16_t *row_out = output + token * dim;

        if (rope_type == MARMOT_ROPE_TYPE_NEOX) {
            size_t i = 0;
            for (; i + 7 < pair_count; i += 8) {
                float cos_vals[8];
                float sin_vals[8];
                if (sincos != nullptr) {
                    for (size_t lane = 0; lane < 8; ++lane) {
                        cos_vals[lane] = sincos[2 * (i + lane)];
                        sin_vals[lane] = sincos[2 * (i + lane) + 1];
                    }
                } else {
                    for (size_t lane = 0; lane < 8; ++lane) {
                        const float angle = pos * freqs[i + lane];
                        float sin_theta = 0.0f;
                        float cos_theta = 0.0f;
                        cpu_sincosf(angle, &sin_theta, &cos_theta);
                        cos_vals[lane] = cos_theta * attn_scale;
                        sin_vals[lane] = sin_theta * attn_scale;
                    }
                }

                const __m256 cos_vec = _mm256_loadu_ps(cos_vals);
                const __m256 sin_vec = _mm256_loadu_ps(sin_vals);
                const __m256 even_vec = cpu_avx_load_bf16(row_in + i);
                const __m256 odd_vec = cpu_avx_load_bf16(row_in + half_dim + i);

                const __m256 even_cos = _mm256_mul_ps(even_vec, cos_vec);
                const __m256 odd_sin = _mm256_mul_ps(odd_vec, sin_vec);
                const __m256 out_even = _mm256_sub_ps(even_cos, odd_sin);

                const __m256 even_sin = _mm256_mul_ps(even_vec, sin_vec);
                const __m256 odd_cos = _mm256_mul_ps(odd_vec, cos_vec);
                const __m256 out_odd = _mm256_add_ps(even_sin, odd_cos);

                cpu_avx_store_bf16(row_out + i, out_even);
                cpu_avx_store_bf16(row_out + half_dim + i, out_odd);
            }

            for (; i < pair_count; ++i) {
                float c = 0.0f;
                float s = 0.0f;
                if (sincos != nullptr) {
                    c = sincos[2 * i];
                    s = sincos[2 * i + 1];
                } else {
                    const float angle = pos * freqs[i];
                    cpu_sincosf(angle, &s, &c);
                    c *= attn_scale;
                    s *= attn_scale;
                }
                const size_t even_index = i;
                const size_t odd_index = i + half_dim;
                const float even_val = marmot_bfloat16_to_native(row_in[even_index]);
                const float odd_val = marmot_bfloat16_to_native(row_in[odd_index]);
                row_out[even_index] = marmot_native_to_bfloat16(even_val * c - odd_val * s);
                row_out[odd_index] = marmot_native_to_bfloat16(even_val * s + odd_val * c);
            }
        } else {
            size_t i = 0;
            for (; i + 3 < pair_count; i += 4) {
                float cos_vals[4];
                float sin_vals[4];
                if (sincos != nullptr) {
                    for (size_t lane = 0; lane < 4; ++lane) {
                        cos_vals[lane] = sincos[2 * (i + lane)];
                        sin_vals[lane] = sincos[2 * (i + lane) + 1];
                    }
                } else {
                    for (size_t lane = 0; lane < 4; ++lane) {
                        const float angle = pos * freqs[i + lane];
                        float sin_theta = 0.0f;
                        float cos_theta = 0.0f;
                        cpu_sincosf(angle, &sin_theta, &cos_theta);
                        cos_vals[lane] = cos_theta * attn_scale;
                        sin_vals[lane] = sin_theta * attn_scale;
                    }
                }

                float cos_dup_vals[8] = {
                    cos_vals[0], cos_vals[0], cos_vals[1], cos_vals[1],
                    cos_vals[2], cos_vals[2], cos_vals[3], cos_vals[3],
                };
                float sin_dup_vals[8] = {
                    sin_vals[0], sin_vals[0], sin_vals[1], sin_vals[1],
                    sin_vals[2], sin_vals[2], sin_vals[3], sin_vals[3],
                };

                const __m256 cos_dup = _mm256_loadu_ps(cos_dup_vals);
                const __m256 sin_dup = _mm256_loadu_ps(sin_dup_vals);
                const __m256 data = cpu_avx_load_bf16(row_in + 2 * i);

                const __m256 real_vals = _mm256_moveldup_ps(data);
                const __m256 imag_vals = _mm256_movehdup_ps(data);

                const __m256 real_cos = _mm256_mul_ps(real_vals, cos_dup);
                const __m256 imag_sin = _mm256_mul_ps(imag_vals, sin_dup);
                const __m256 out_real = _mm256_sub_ps(real_cos, imag_sin);

                const __m256 real_sin = _mm256_mul_ps(real_vals, sin_dup);
                const __m256 imag_cos = _mm256_mul_ps(imag_vals, cos_dup);
                const __m256 out_imag = _mm256_add_ps(real_sin, imag_cos);

                const __m256 low = _mm256_unpacklo_ps(out_real, out_imag);
                const __m256 high = _mm256_unpackhi_ps(out_real, out_imag);
                const __m256 packed = _mm256_shuffle_ps(low, high, _MM_SHUFFLE(1, 0, 1, 0));
                cpu_avx_store_bf16(row_out + 2 * i, packed);
            }

            for (; i < pair_count; ++i) {
                float c = 0.0f;
                float s = 0.0f;
                if (sincos != nullptr) {
                    c = sincos[2 * i];
                    s = sincos[2 * i + 1];
                } else {
                    const float angle = pos * freqs[i];
                    cpu_sincosf(angle, &s, &c);
                    c *= attn_scale;
                    s *= attn_scale;
                }
                const size_t even_index = 2 * i;
                const size_t odd_index = even_index + 1;
                const float even_val = marmot_bfloat16_to_native(row_in[even_index]);
                const float odd_val = marmot_bfloat16_to_native(row_in[odd_index]);
                row_out[even_index] = marmot_native_to_bfloat16(even_val * c - odd_val * s);
                row_out[odd_index] = marmot_native_to_bfloat16(even_val * s + odd_val * c);
            }
        }
    }

    return MARMOT_SUCCESS;
}

const cpu_rope_traits_t cpu_rope_avx_traits = {
    .dtype = MARMOT_DTYPE_FLOAT32,
    .impl_kind = CPU_ROPE_IMPL_AVX2,
    .min_dim = 0,
    .kernel = cpu_rope_kernel_avx_run,
};

const cpu_rope_traits_t cpu_rope_avx_f16_traits = {
    .dtype = MARMOT_DTYPE_FLOAT16,
    .impl_kind = CPU_ROPE_IMPL_AVX2,
    .min_dim = 0,
    .kernel = cpu_rope_kernel_avx_f16_run,
};

const cpu_rope_traits_t cpu_rope_avx_bf16_traits = {
    .dtype = MARMOT_DTYPE_BFLOAT16,
    .impl_kind = CPU_ROPE_IMPL_AVX2,
    .min_dim = 0,
    .kernel = cpu_rope_kernel_avx_bf16_run,
};

#endif
