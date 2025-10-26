#include "quantization/common/block.h"

#include "marmot/tensor.h"

#include <float.h>
#include <math.h>

#if HAS_NEON
#include <arm_neon.h>
#endif

#if HAS_AVX2
#include <immintrin.h>
#endif

static inline __attribute__((unused)) float max_abs_scalar(const float *data, size_t len) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        float v = fabsf(data[i]);
        if (v > max_abs) {
            max_abs = v;
        }
    }
    return max_abs;
}

static inline __attribute__((unused)) marmot_block_minmax_t minmax_scalar(const float *data, size_t len) {
    marmot_block_minmax_t result = {.min_val = data[0], .max_val = data[0]};
    for (size_t i = 1; i < len; ++i) {
        float v = data[i];
        if (v < result.min_val) {
            result.min_val = v;
        }
        if (v > result.max_val) {
            result.max_val = v;
        }
    }
    return result;
}

#if HAS_NEON
static inline float max_abs_neon(const float *data, size_t len) {
    size_t i = 0;
    float32x4_t max_vec = vdupq_n_f32(0.0f);
    for (; i + 4 <= len; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        float32x4_t abs_v = vabsq_f32(v);
        max_vec = vmaxq_f32(max_vec, abs_v);
    }

    float max_abs = 0.0f;
    if (i >= 4) {
        float32x2_t max_low = vmax_f32(vget_low_f32(max_vec), vget_high_f32(max_vec));
        max_low = vpmax_f32(max_low, max_low);
        max_abs = vget_lane_f32(max_low, 0);
    }

    for (; i < len; ++i) {
        float v = fabsf(data[i]);
        if (v > max_abs) {
            max_abs = v;
        }
    }
    return max_abs;
}

static inline marmot_block_minmax_t minmax_neon(const float *data, size_t len) {
    marmot_block_minmax_t result = {.min_val = data[0], .max_val = data[0]};
    size_t i = 0;
    if (len >= 4) {
        float32x4_t min_vec = vdupq_n_f32(result.min_val);
        float32x4_t max_vec = vdupq_n_f32(result.max_val);
        for (; i + 4 <= len; i += 4) {
            float32x4_t v = vld1q_f32(data + i);
            min_vec = vminq_f32(min_vec, v);
            max_vec = vmaxq_f32(max_vec, v);
        }

        float32x2_t min_low = vmin_f32(vget_low_f32(min_vec), vget_high_f32(min_vec));
        min_low = vpmin_f32(min_low, min_low);
        float32x2_t max_low = vmax_f32(vget_low_f32(max_vec), vget_high_f32(max_vec));
        max_low = vpmax_f32(max_low, max_low);
        result.min_val = vget_lane_f32(min_low, 0);
        result.max_val = vget_lane_f32(max_low, 0);
    }

    for (; i < len; ++i) {
        float v = data[i];
        if (v < result.min_val) {
            result.min_val = v;
        }
        if (v > result.max_val) {
            result.max_val = v;
        }
    }
    return result;
}
#endif

#if HAS_AVX2
static inline float horizontal_max_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 max128 = _mm_max_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(max128);
    max128 = _mm_max_ps(max128, shuf);
    shuf = _mm_movehl_ps(shuf, max128);
    max128 = _mm_max_ps(max128, shuf);
    return _mm_cvtss_f32(max128);
}

static inline float horizontal_min_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 min128 = _mm_min_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(min128);
    min128 = _mm_min_ps(min128, shuf);
    shuf = _mm_movehl_ps(shuf, min128);
    min128 = _mm_min_ps(min128, shuf);
    return _mm_cvtss_f32(min128);
}

static inline float max_abs_avx2(const float *data, size_t len) {
    size_t i = 0;
    __m256 max_vec = _mm256_setzero_ps();
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    for (; i + 8 <= len; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        __m256 abs_v = _mm256_and_ps(v, sign_mask);
        max_vec = _mm256_max_ps(max_vec, abs_v);
    }

    float max_abs = 0.0f;
    if (i >= 8) {
        max_abs = horizontal_max_ps(max_vec);
    }

    for (; i < len; ++i) {
        float v = fabsf(data[i]);
        if (v > max_abs) {
            max_abs = v;
        }
    }
    return max_abs;
}

static inline marmot_block_minmax_t minmax_avx2(const float *data, size_t len) {
    marmot_block_minmax_t result = {.min_val = data[0], .max_val = data[0]};
    size_t i = 0;
    if (len >= 8) {
        __m256 min_vec = _mm256_set1_ps(result.min_val);
        __m256 max_vec = _mm256_set1_ps(result.max_val);
        for (; i + 8 <= len; i += 8) {
            __m256 v = _mm256_loadu_ps(data + i);
            min_vec = _mm256_min_ps(min_vec, v);
            max_vec = _mm256_max_ps(max_vec, v);
        }
        result.min_val = horizontal_min_ps(min_vec);
        result.max_val = horizontal_max_ps(max_vec);
    }

    for (; i < len; ++i) {
        float v = data[i];
        if (v < result.min_val) {
            result.min_val = v;
        }
        if (v > result.max_val) {
            result.max_val = v;
        }
    }
    return result;
}
#endif

float cpu_quant_block_max_abs(const float *data, size_t len) {
#if HAS_AVX2
    return max_abs_avx2(data, len);
#elif HAS_NEON
    return max_abs_neon(data, len);
#else
    return max_abs_scalar(data, len);
#endif
}

marmot_block_minmax_t cpu_quant_block_minmax(const float *data, size_t len) {
#if HAS_AVX2
    return minmax_avx2(data, len);
#elif HAS_NEON
    return minmax_neon(data, len);
#else
    return minmax_scalar(data, len);
#endif
}

static inline void quantize_q5_0_scalar(const float *data, size_t len, float inv_scale, int8_t *out) {
    for (size_t i = 0; i < len; ++i) {
        float shifted = data[i] * inv_scale + 16.5f;
        int32_t q = (int32_t)shifted;
        if (q < 0) {
            q = 0;
        } else if (q > 31) {
            q = 31;
        }
        out[i] = (int8_t)(q - 16);
    }
}

static inline void quantize_q5_1_scalar(const float *data, size_t len, float min_val, float inv_scale, uint8_t *out) {
    for (size_t i = 0; i < len; ++i) {
        float value = (data[i] - min_val) * inv_scale;
        int32_t q = (int32_t)roundf(value);
        if (q < 0) {
            q = 0;
        } else if (q > 31) {
            q = 31;
        }
        out[i] = (uint8_t)q;
    }
}

static inline void quantize_q8_0_scalar(const float *data, size_t len, float inv_scale, int8_t *out) {
    for (size_t i = 0; i < len; ++i) {
        float value = data[i] * inv_scale;
        int32_t q = (int32_t)roundf(value);
        if (q < -127) {
            q = -127;
        } else if (q > 127) {
            q = 127;
        }
        out[i] = (int8_t)q;
    }
}

#if HAS_NEON
static inline void quantize_q5_0_neon(const float *data, size_t len, float inv_scale, int8_t *out) {
    const float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
    const float32x4_t offset_vec = vdupq_n_f32(16.5f);
    const int32x4_t clamp_min = vdupq_n_s32(0);
    const int32x4_t clamp_max = vdupq_n_s32(31);
    const int32x4_t shift_vec = vdupq_n_s32(16);
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        float32x4_t v0 = vaddq_f32(vmulq_f32(vld1q_f32(data + i), inv_scale_vec), offset_vec);
        float32x4_t v1 = vaddq_f32(vmulq_f32(vld1q_f32(data + i + 4), inv_scale_vec), offset_vec);
        int32x4_t q0 = vcvtq_s32_f32(v0);
        int32x4_t q1 = vcvtq_s32_f32(v1);
        q0 = vmaxq_s32(q0, clamp_min);
        q0 = vminq_s32(q0, clamp_max);
        q1 = vmaxq_s32(q1, clamp_min);
        q1 = vminq_s32(q1, clamp_max);
        q0 = vsubq_s32(q0, shift_vec);
        q1 = vsubq_s32(q1, shift_vec);
        int16x4_t q16_0 = vqmovn_s32(q0);
        int16x4_t q16_1 = vqmovn_s32(q1);
        int16x8_t q16 = vcombine_s16(q16_0, q16_1);
        int8x8_t packed = vqmovn_s16(q16);
        vst1_s8(out + i, packed);
    }
    if (i < len) {
        quantize_q5_0_scalar(data + i, len - i, inv_scale, out + i);
    }
}

static inline void quantize_q5_1_neon(const float *data, size_t len, float min_val, float inv_scale, uint8_t *out) {
    const float32x4_t min_vec = vdupq_n_f32(min_val);
    const float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
    const int32x4_t clamp_min = vdupq_n_s32(0);
    const int32x4_t clamp_max = vdupq_n_s32(31);
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        float32x4_t v0 = vld1q_f32(data + i);
        float32x4_t v1 = vld1q_f32(data + i + 4);
        v0 = vmulq_f32(vsubq_f32(v0, min_vec), inv_scale_vec);
        v1 = vmulq_f32(vsubq_f32(v1, min_vec), inv_scale_vec);
        int32x4_t q0 = vcvtnq_s32_f32(v0);
        int32x4_t q1 = vcvtnq_s32_f32(v1);
        q0 = vmaxq_s32(q0, clamp_min);
        q0 = vminq_s32(q0, clamp_max);
        q1 = vmaxq_s32(q1, clamp_min);
        q1 = vminq_s32(q1, clamp_max);
        int16x4_t q16_0 = vqmovn_s32(q0);
        int16x4_t q16_1 = vqmovn_s32(q1);
        int16x8_t q16 = vcombine_s16(q16_0, q16_1);
        int8x8_t packed = vqmovn_s16(q16);
        vst1_u8(out + i, vreinterpret_u8_s8(packed));
    }
    if (i < len) {
        quantize_q5_1_scalar(data + i, len - i, min_val, inv_scale, out + i);
    }
}

static inline void quantize_q8_0_neon(const float *data, size_t len, float inv_scale, int8_t *out) {
    const float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
    const int32x4_t clamp_min = vdupq_n_s32(-127);
    const int32x4_t clamp_max = vdupq_n_s32(127);
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        float32x4_t v0 = vld1q_f32(data + i);
        float32x4_t v1 = vld1q_f32(data + i + 4);
        int32x4_t q0 = vcvtnq_s32_f32(vmulq_f32(v0, inv_scale_vec));
        int32x4_t q1 = vcvtnq_s32_f32(vmulq_f32(v1, inv_scale_vec));
        q0 = vmaxq_s32(q0, clamp_min);
        q0 = vminq_s32(q0, clamp_max);
        q1 = vmaxq_s32(q1, clamp_min);
        q1 = vminq_s32(q1, clamp_max);
        int16x4_t q16_0 = vqmovn_s32(q0);
        int16x4_t q16_1 = vqmovn_s32(q1);
        int16x8_t q16 = vcombine_s16(q16_0, q16_1);
        int8x8_t packed = vqmovn_s16(q16);
        vst1_s8(out + i, packed);
    }
    if (i < len) {
        quantize_q8_0_scalar(data + i, len - i, inv_scale, out + i);
    }
}
#endif

#if HAS_AVX2
static inline void quantize_q5_0_avx2(const float *data, size_t len, float inv_scale, int8_t *out) {
    const __m256 inv_scale_vec = _mm256_set1_ps(inv_scale);
    const __m256 offset_vec = _mm256_set1_ps(16.5f);
    const __m256i clamp_min = _mm256_set1_epi32(0);
    const __m256i clamp_max = _mm256_set1_epi32(31);
    const __m256i shift_vec = _mm256_set1_epi32(16);
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 v = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(data + i), inv_scale_vec), offset_vec);
        __m256i q = _mm256_cvttps_epi32(v);
        q = _mm256_max_epi32(q, clamp_min);
        q = _mm256_min_epi32(q, clamp_max);
        q = _mm256_sub_epi32(q, shift_vec);
        __m128i lo = _mm256_castsi256_si128(q);
        __m128i hi = _mm256_extracti128_si256(q, 1);
        __m128i q16 = _mm_packs_epi32(lo, hi);
        __m128i q8 = _mm_packs_epi16(q16, q16);
        _mm_storel_epi64((__m128i *)(out + i), q8);
    }
    if (i < len) {
        quantize_q5_0_scalar(data + i, len - i, inv_scale, out + i);
    }
}

static inline void quantize_q5_1_avx2(const float *data, size_t len, float min_val, float inv_scale, uint8_t *out) {
    const __m256 min_vec = _mm256_set1_ps(min_val);
    const __m256 inv_scale_vec = _mm256_set1_ps(inv_scale);
    const __m256i clamp_min = _mm256_set1_epi32(0);
    const __m256i clamp_max = _mm256_set1_epi32(31);
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 shifted = _mm256_sub_ps(_mm256_loadu_ps(data + i), min_vec);
        shifted = _mm256_mul_ps(shifted, inv_scale_vec);
        __m256i q = _mm256_cvtps_epi32(shifted);
        q = _mm256_max_epi32(q, clamp_min);
        q = _mm256_min_epi32(q, clamp_max);
        __m128i lo = _mm256_castsi256_si128(q);
        __m128i hi = _mm256_extracti128_si256(q, 1);
        __m128i q16 = _mm_packs_epi32(lo, hi);
        __m128i q8 = _mm_packs_epi16(q16, q16);
        _mm_storel_epi64((__m128i *)(out + i), q8);
    }
    if (i < len) {
        quantize_q5_1_scalar(data + i, len - i, min_val, inv_scale, out + i);
    }
}

static inline void quantize_q8_0_avx2(const float *data, size_t len, float inv_scale, int8_t *out) {
    const __m256 inv_scale_vec = _mm256_set1_ps(inv_scale);
    const __m256i clamp_min = _mm256_set1_epi32(-127);
    const __m256i clamp_max = _mm256_set1_epi32(127);
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 scaled = _mm256_mul_ps(_mm256_loadu_ps(data + i), inv_scale_vec);
        __m256i q = _mm256_cvtps_epi32(scaled);
        q = _mm256_max_epi32(q, clamp_min);
        q = _mm256_min_epi32(q, clamp_max);
        __m128i lo = _mm256_castsi256_si128(q);
        __m128i hi = _mm256_extracti128_si256(q, 1);
        __m128i q16 = _mm_packs_epi32(lo, hi);
        __m128i q8 = _mm_packs_epi16(q16, q16);
        _mm_storel_epi64((__m128i *)(out + i), q8);
    }
    if (i < len) {
        quantize_q8_0_scalar(data + i, len - i, inv_scale, out + i);
    }
}
#endif

void cpu_quantize_q5_0_pack(const float *data, size_t len, float inv_scale, int8_t *out) {
#if HAS_AVX2
    quantize_q5_0_avx2(data, len, inv_scale, out);
#elif HAS_NEON
    quantize_q5_0_neon(data, len, inv_scale, out);
#else
    quantize_q5_0_scalar(data, len, inv_scale, out);
#endif
}

void cpu_quantize_q5_1_pack(const float *data, size_t len, float min_val, float inv_scale, uint8_t *out) {
#if HAS_AVX2
    quantize_q5_1_avx2(data, len, min_val, inv_scale, out);
#elif HAS_NEON
    quantize_q5_1_neon(data, len, min_val, inv_scale, out);
#else
    quantize_q5_1_scalar(data, len, min_val, inv_scale, out);
#endif
}

void cpu_quantize_q8_0_pack(const float *data, size_t len, float inv_scale, int8_t *out) {
#if HAS_AVX2
    quantize_q8_0_avx2(data, len, inv_scale, out);
#elif HAS_NEON
    quantize_q8_0_neon(data, len, inv_scale, out);
#else
    quantize_q8_0_scalar(data, len, inv_scale, out);
#endif
}
