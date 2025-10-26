#include <math.h>

#include "cpu_backend_internal.h"
#include "ops/matmul/qkv/matmul_qkv_rope.h"
#include "rope_kernels.h"

#if HAS_NEON

#include <arm_neon.h>

static inline float32x4_t cpu_neon_load_f16(const marmot_float16_t *src) {
    float16x4_t f16_vec = vreinterpret_f16_u16(vld1_u16((const uint16_t *)src));
    return vcvt_f32_f16(f16_vec);
}

static inline void cpu_neon_store_f16(marmot_float16_t *dst, float32x4_t value) {
    float16x4_t f16_vec = vcvt_f16_f32(value);
    vst1_u16((uint16_t *)dst, vreinterpret_u16_f16(f16_vec));
}

static inline float32x4_t cpu_neon_load_bf16(const marmot_bfloat16_t *src) {
    uint16x4_t bf16_vec = vld1_u16((const uint16_t *)src);
    uint32x4_t f32_bits = vshll_n_u16(bf16_vec, 16);
    return vreinterpretq_f32_u32(f32_bits);
}

static inline void cpu_neon_store_bf16(marmot_bfloat16_t *dst, float32x4_t value, uint32x4_t rounding_bias) {
    uint32x4_t f32_bits = vreinterpretq_u32_f32(value);
    uint32x4_t lsb = vshrq_n_u32(f32_bits, 16);
    lsb = vandq_u32(lsb, vdupq_n_u32(1));
    uint32x4_t bias = vaddq_u32(rounding_bias, lsb);
    uint32x4_t rounded = vaddq_u32(f32_bits, bias);
    uint16x4_t bf16_vec = vmovn_u32(vshrq_n_u32(rounded, 16));
    vst1_u16((uint16_t *)dst, bf16_vec);
}

static marmot_error_t cpu_rope_kernel_neon_run(
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

                const float32x4_t cos_vec = vld1q_f32(cos_vals);
                const float32x4_t sin_vec = vld1q_f32(sin_vals);
                const float32x4_t even_vec = vld1q_f32(row_in + i);
                const float32x4_t odd_vec = vld1q_f32(row_in + half_dim + i);

                const float32x4_t even_cos = vmulq_f32(even_vec, cos_vec);
                const float32x4_t odd_sin = vmulq_f32(odd_vec, sin_vec);
                const float32x4_t out_even = vsubq_f32(even_cos, odd_sin);

                const float32x4_t even_sin = vmulq_f32(even_vec, sin_vec);
                const float32x4_t odd_cos = vmulq_f32(odd_vec, cos_vec);
                const float32x4_t out_odd = vaddq_f32(even_sin, odd_cos);

                vst1q_f32(row_out + i, out_even);
                vst1q_f32(row_out + half_dim + i, out_odd);
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
            for (; i + 1 < pair_count; i += 2) {
                float cos_vals[2];
                float sin_vals[2];
                if (sincos != nullptr) {
                    cos_vals[0] = sincos[2 * i];
                    sin_vals[0] = sincos[2 * i + 1];
                    cos_vals[1] = sincos[2 * (i + 1)];
                    sin_vals[1] = sincos[2 * (i + 1) + 1];
                } else {
                    for (size_t lane = 0; lane < 2; ++lane) {
                        const float angle = pos * freqs[i + lane];
                        float sin_theta = 0.0f;
                        float cos_theta = 0.0f;
                        cpu_sincosf(angle, &sin_theta, &cos_theta);
                        cos_vals[lane] = cos_theta * attn_scale;
                        sin_vals[lane] = sin_theta * attn_scale;
                    }
                }
                const float32x2_t cos_low = vld1_f32(cos_vals);
                const float32x2_t sin_low = vld1_f32(sin_vals);
                const float32x4_t cos_dup = vcombine_f32(cos_low, cos_low);
                const float32x4_t sin_dup = vcombine_f32(sin_low, sin_low);

                const float32x4_t data = vld1q_f32(row_in + 2 * i);
                const float32x4x2_t split = vuzpq_f32(data, data);
                const float32x4_t real_vals = split.val[0];
                const float32x4_t imag_vals = split.val[1];

                const float32x4_t real_cos = vmulq_f32(real_vals, cos_dup);
                const float32x4_t imag_sin = vmulq_f32(imag_vals, sin_dup);
                const float32x4_t out_real = vsubq_f32(real_cos, imag_sin);

                const float32x4_t real_sin = vmulq_f32(real_vals, sin_dup);
                const float32x4_t imag_cos = vmulq_f32(imag_vals, cos_dup);
                const float32x4_t out_imag = vaddq_f32(real_sin, imag_cos);

                const float32x4x2_t interleaved = vzipq_f32(out_real, out_imag);
                vst1q_f32(row_out + 2 * i, interleaved.val[0]);
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

static marmot_error_t cpu_rope_kernel_neon_f16_run(
    cpu_context_t *ctx, const marmot_tensor_t *x, const marmot_tensor_t *positions, const float *freqs,
    float attn_scale, marmot_rope_type_t rope_type, size_t seq_len, size_t dim, size_t total_seqs, marmot_tensor_t *out
) {
    if (x->dtype != MARMOT_DTYPE_FLOAT16 || out->dtype != MARMOT_DTYPE_FLOAT16) {
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

                const float32x4_t cos_vec = vld1q_f32(cos_vals);
                const float32x4_t sin_vec = vld1q_f32(sin_vals);
                const float32x4_t even_vec = cpu_neon_load_f16(row_in + i);
                const float32x4_t odd_vec = cpu_neon_load_f16(row_in + half_dim + i);

                const float32x4_t even_cos = vmulq_f32(even_vec, cos_vec);
                const float32x4_t odd_sin = vmulq_f32(odd_vec, sin_vec);
                const float32x4_t out_even = vsubq_f32(even_cos, odd_sin);

                const float32x4_t even_sin = vmulq_f32(even_vec, sin_vec);
                const float32x4_t odd_cos = vmulq_f32(odd_vec, cos_vec);
                const float32x4_t out_odd = vaddq_f32(even_sin, odd_cos);

                cpu_neon_store_f16(row_out + i, out_even);
                cpu_neon_store_f16(row_out + half_dim + i, out_odd);
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
            for (; i + 1 < pair_count; i += 2) {
                float cos_vals[2];
                float sin_vals[2];
                if (sincos != nullptr) {
                    cos_vals[0] = sincos[2 * i];
                    sin_vals[0] = sincos[2 * i + 1];
                    cos_vals[1] = sincos[2 * (i + 1)];
                    sin_vals[1] = sincos[2 * (i + 1) + 1];
                } else {
                    for (size_t lane = 0; lane < 2; ++lane) {
                        const float angle = pos * freqs[i + lane];
                        float sin_theta = 0.0f;
                        float cos_theta = 0.0f;
                        cpu_sincosf(angle, &sin_theta, &cos_theta);
                        cos_vals[lane] = cos_theta * attn_scale;
                        sin_vals[lane] = sin_theta * attn_scale;
                    }
                }
                const float32x2_t cos_low = vld1_f32(cos_vals);
                const float32x2_t sin_low = vld1_f32(sin_vals);
                const float32x4_t cos_dup = vcombine_f32(cos_low, cos_low);
                const float32x4_t sin_dup = vcombine_f32(sin_low, sin_low);

                const float32x4_t data = cpu_neon_load_f16(row_in + 2 * i);
                const float32x4x2_t split = vuzpq_f32(data, data);
                const float32x4_t real_vals = split.val[0];
                const float32x4_t imag_vals = split.val[1];

                const float32x4_t real_cos = vmulq_f32(real_vals, cos_dup);
                const float32x4_t imag_sin = vmulq_f32(imag_vals, sin_dup);
                const float32x4_t out_real = vsubq_f32(real_cos, imag_sin);

                const float32x4_t real_sin = vmulq_f32(real_vals, sin_dup);
                const float32x4_t imag_cos = vmulq_f32(imag_vals, cos_dup);
                const float32x4_t out_imag = vaddq_f32(real_sin, imag_cos);

                const float32x4x2_t interleaved = vzipq_f32(out_real, out_imag);
                cpu_neon_store_f16(row_out + 2 * i, interleaved.val[0]);
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
}

static marmot_error_t cpu_rope_kernel_neon_bf16_run(
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
    const uint32x4_t rounding_bias = vdupq_n_u32(0x7FFF);

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

                const float32x4_t cos_vec = vld1q_f32(cos_vals);
                const float32x4_t sin_vec = vld1q_f32(sin_vals);
                const float32x4_t even_vec = cpu_neon_load_bf16(row_in + i);
                const float32x4_t odd_vec = cpu_neon_load_bf16(row_in + half_dim + i);

                const float32x4_t even_cos = vmulq_f32(even_vec, cos_vec);
                const float32x4_t odd_sin = vmulq_f32(odd_vec, sin_vec);
                const float32x4_t out_even = vsubq_f32(even_cos, odd_sin);

                const float32x4_t even_sin = vmulq_f32(even_vec, sin_vec);
                const float32x4_t odd_cos = vmulq_f32(odd_vec, cos_vec);
                const float32x4_t out_odd = vaddq_f32(even_sin, odd_cos);

                cpu_neon_store_bf16(row_out + i, out_even, rounding_bias);
                cpu_neon_store_bf16(row_out + half_dim + i, out_odd, rounding_bias);
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
            for (; i + 1 < pair_count; i += 2) {
                float cos_vals[2];
                float sin_vals[2];
                if (sincos != nullptr) {
                    cos_vals[0] = sincos[2 * i];
                    sin_vals[0] = sincos[2 * i + 1];
                    cos_vals[1] = sincos[2 * (i + 1)];
                    sin_vals[1] = sincos[2 * (i + 1) + 1];
                } else {
                    for (size_t lane = 0; lane < 2; ++lane) {
                        const float angle = pos * freqs[i + lane];
                        float sin_theta = 0.0f;
                        float cos_theta = 0.0f;
                        cpu_sincosf(angle, &sin_theta, &cos_theta);
                        cos_vals[lane] = cos_theta * attn_scale;
                        sin_vals[lane] = sin_theta * attn_scale;
                    }
                }
                const float32x2_t cos_low = vld1_f32(cos_vals);
                const float32x2_t sin_low = vld1_f32(sin_vals);
                const float32x4_t cos_dup = vcombine_f32(cos_low, cos_low);
                const float32x4_t sin_dup = vcombine_f32(sin_low, sin_low);

                const float32x4_t data = cpu_neon_load_bf16(row_in + 2 * i);
                const float32x4x2_t split = vuzpq_f32(data, data);
                const float32x4_t real_vals = split.val[0];
                const float32x4_t imag_vals = split.val[1];

                const float32x4_t real_cos = vmulq_f32(real_vals, cos_dup);
                const float32x4_t imag_sin = vmulq_f32(imag_vals, sin_dup);
                const float32x4_t out_real = vsubq_f32(real_cos, imag_sin);

                const float32x4_t real_sin = vmulq_f32(real_vals, sin_dup);
                const float32x4_t imag_cos = vmulq_f32(imag_vals, cos_dup);
                const float32x4_t out_imag = vaddq_f32(real_sin, imag_cos);

                const float32x4x2_t interleaved = vzipq_f32(out_real, out_imag);
                cpu_neon_store_bf16(row_out + 2 * i, interleaved.val[0], rounding_bias);
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

const cpu_rope_traits_t cpu_rope_neon_traits = {
    .dtype = MARMOT_DTYPE_FLOAT32,
    .impl_kind = CPU_ROPE_IMPL_NEON,
    .min_dim = 0,
    .kernel = cpu_rope_kernel_neon_run,
};

const cpu_rope_traits_t cpu_rope_neon_f16_traits = {
    .dtype = MARMOT_DTYPE_FLOAT16,
    .impl_kind = CPU_ROPE_IMPL_NEON,
    .min_dim = 0,
    .kernel = cpu_rope_kernel_neon_f16_run,
};

const cpu_rope_traits_t cpu_rope_neon_bf16_traits = {
    .dtype = MARMOT_DTYPE_BFLOAT16,
    .impl_kind = CPU_ROPE_IMPL_NEON,
    .min_dim = 0,
    .kernel = cpu_rope_kernel_neon_bf16_run,
};

#endif
