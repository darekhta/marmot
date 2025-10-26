#ifndef CPU_SOFTMAX_INTERNAL_H
#define CPU_SOFTMAX_INTERNAL_H

#include <float.h>
#include <math.h>

#include "cpu_backend_internal.h"
#include "ops/cpu_neon_math.h"

#if HAS_AVX2
#include "ops/softmax/avx2/avx2_math.h"
#endif

static constexpr float SOFTMAX_EXP_FLUSH_THRESHOLD = -88.0f;
#if HAS_NEON
static constexpr size_t SOFTMAX_TWO_PASS_MIN_BYTES = 1024 * 1024 * 1024;
#else
static constexpr size_t SOFTMAX_TWO_PASS_MIN_BYTES = 128 * 1024;
#endif

static inline float softmax_safe_expf(float diff) {
    if (diff <= SOFTMAX_EXP_FLUSH_THRESHOLD) {
        return 0.0f;
    }
    return expf(diff);
}

static inline bool softmax_should_use_two_pass(size_t n) {
    return n >= (SOFTMAX_TWO_PASS_MIN_BYTES / sizeof(float));
}

typedef struct {
    float max_val;
    float sum_exp;
} softmax_accum_f32_t;

static inline softmax_accum_f32_t softmax_accumulate_f32(const float *x, size_t n) {
    softmax_accum_f32_t acc = {
        .max_val = -FLT_MAX,
        .sum_exp = 0.0f,
    };

    for (size_t i = 0; i < n; ++i) {
        float xi = x[i];
        if (xi > acc.max_val) {
            float scale = softmax_safe_expf(acc.max_val - xi);
            acc.sum_exp = acc.sum_exp * scale + 1.0f;
            acc.max_val = xi;
        } else {
            acc.sum_exp += softmax_safe_expf(xi - acc.max_val);
        }
    }

    return acc;
}

#if HAS_NEON
static inline float softmax_neon_reduce_max_f32(float32x4_t v) {
#if defined(__aarch64__)
    return vmaxvq_f32(v);
#else
    float32x2_t lo = vget_low_f32(v);
    float32x2_t hi = vget_high_f32(v);
    float32x2_t max_pair = vpmax_f32(lo, hi);
    max_pair = vpmax_f32(max_pair, max_pair);
    return vget_lane_f32(max_pair, 0);
#endif
}

static inline float softmax_neon_reduce_sum_f32(float32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float32x2_t lo = vget_low_f32(v);
    float32x2_t hi = vget_high_f32(v);
    float32x2_t sum_pair = vadd_f32(lo, hi);
    sum_pair = vpadd_f32(sum_pair, sum_pair);
    return vget_lane_f32(sum_pair, 0);
#endif
}

static inline float32x4_t softmax_exp_neon(float32x4_t diff) {
    const float32x4_t threshold = vdupq_n_f32(SOFTMAX_EXP_FLUSH_THRESHOLD);
    uint32x4_t mask = vcgtq_f32(diff, threshold);
    float32x4_t exp_val = cpu_neon_exp_vec(diff);
    return vbslq_f32(mask, exp_val, vdupq_n_f32(0.0f));
}
#endif

#if HAS_AVX2
static inline float softmax_horizontal_sum_m256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static inline float softmax_horizontal_max_m256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 max_vec = _mm_max_ps(lo, hi);
    __m128 shuf = _mm_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(2, 3, 0, 1));
    max_vec = _mm_max_ps(max_vec, shuf);
    shuf = _mm_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(1, 0, 3, 2));
    max_vec = _mm_max_ps(max_vec, shuf);
    return _mm_cvtss_f32(max_vec);
}

static inline __m256 softmax_exp_avx2(__m256 diff) {
    const __m256 threshold = _mm256_set1_ps(SOFTMAX_EXP_FLUSH_THRESHOLD);
    __m256 exp_val = cpu_avx2_exp_vec(diff);
    __m256 mask = _mm256_cmp_ps(diff, threshold, _CMP_GT_OQ);
    return _mm256_and_ps(exp_val, mask);
}
#endif

static inline float softmax_find_max_f16(cpu_context_t *ctx, const marmot_float16_t *row, size_t n) {
    float max_val = -FLT_MAX;

#if HAS_NEON
    if (ctx != nullptr && cpu_ctx_has_neon(ctx)) {
        float32x4_t max_lo = vdupq_n_f32(-FLT_MAX);
        float32x4_t max_hi = vdupq_n_f32(-FLT_MAX);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            uint16x8_t bits = vld1q_u16((const uint16_t *)(row + i));
            float16x8_t vec = vreinterpretq_f16_u16(bits);
            float32x4_t lo = vcvt_f32_f16(vget_low_f16(vec));
            float32x4_t hi = vcvt_f32_f16(vget_high_f16(vec));
            max_lo = vmaxq_f32(max_lo, lo);
            max_hi = vmaxq_f32(max_hi, hi);
        }
        max_val = fmaxf(softmax_neon_reduce_max_f32(max_lo), softmax_neon_reduce_max_f32(max_hi));
        for (; i < n; i++) {
            float value = marmot_f16_to_f32_ref(row[i]);
            if (value > max_val) {
                max_val = value;
            }
        }
        return max_val;
    }
#endif

#if HAS_F16C
    if (ctx != nullptr && cpu_ctx_has_f16c(ctx)) {
        __m256 max_vec = _mm256_set1_ps(-FLT_MAX);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m128i vals = _mm_loadu_si128((const __m128i *)(row + i));
            __m256 vals_f32 = _mm256_cvtph_ps(vals);
            max_vec = _mm256_max_ps(max_vec, vals_f32);
        }
        max_val = softmax_horizontal_max_m256(max_vec);
        for (; i < n; i++) {
            float value = marmot_f16_to_f32_ref(row[i]);
            if (value > max_val) {
                max_val = value;
            }
        }
        return max_val;
    }
#endif

    for (size_t i = 0; i < n; i++) {
        float value = marmot_f16_to_f32_ref(row[i]);
        if (value > max_val) {
            max_val = value;
        }
    }
    return max_val;
}

static inline void softmax_scale_f16(cpu_context_t *ctx, marmot_float16_t *row, size_t n, float inv_sum) {
#if HAS_NEON
    if (ctx != nullptr && cpu_ctx_has_neon(ctx)) {
        float32x4_t inv_vec = vdupq_n_f32(inv_sum);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            uint16x8_t bits = vld1q_u16((const uint16_t *)(row + i));
            float16x8_t vals = vreinterpretq_f16_u16(bits);
            float32x4_t lo = vcvt_f32_f16(vget_low_f16(vals));
            float32x4_t hi = vcvt_f32_f16(vget_high_f16(vals));
            lo = vmulq_f32(lo, inv_vec);
            hi = vmulq_f32(hi, inv_vec);
            float16x4_t out_lo = vcvt_f16_f32(lo);
            float16x4_t out_hi = vcvt_f16_f32(hi);
            float16x8_t packed = vcombine_f16(out_lo, out_hi);
            vst1q_u16((uint16_t *)(row + i), vreinterpretq_u16_f16(packed));
        }
        for (; i < n; i++) {
            float value = marmot_f16_to_f32_ref(row[i]);
            row[i] = marmot_f32_to_f16_ref(value * inv_sum);
        }
        return;
    }
#endif

#if HAS_F16C
    if (ctx != nullptr && cpu_ctx_has_f16c(ctx)) {
        __m256 inv_vec = _mm256_set1_ps(inv_sum);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m128i vals = _mm_loadu_si128((const __m128i *)(row + i));
            __m256 vals_f32 = _mm256_cvtph_ps(vals);
            __m256 scaled = _mm256_mul_ps(vals_f32, inv_vec);
            __m128i packed = _mm256_cvtps_ph(scaled, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)(row + i), packed);
        }
        for (; i < n; i++) {
            float value = marmot_f16_to_f32_ref(row[i]);
            row[i] = marmot_f32_to_f16_ref(value * inv_sum);
        }
        return;
    }
#endif

    for (size_t i = 0; i < n; i++) {
        float value = marmot_f16_to_f32_ref(row[i]);
        row[i] = marmot_f32_to_f16_ref(value * inv_sum);
    }
}

static inline float softmax_find_max_bf16(cpu_context_t *ctx, const marmot_bfloat16_t *row, size_t n) {
    float max_val = -FLT_MAX;

#if HAS_NEON
    if (ctx != nullptr && cpu_ctx_has_neon(ctx)) {
        float32x4_t max_lo = vdupq_n_f32(-FLT_MAX);
        float32x4_t max_hi = vdupq_n_f32(-FLT_MAX);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            uint16x8_t bits = vld1q_u16((const uint16_t *)(row + i));
            float32x4_t lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bits), 16));
            float32x4_t hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bits), 16));
            max_lo = vmaxq_f32(max_lo, lo);
            max_hi = vmaxq_f32(max_hi, hi);
        }
        max_val = fmaxf(softmax_neon_reduce_max_f32(max_lo), softmax_neon_reduce_max_f32(max_hi));
        for (; i < n; i++) {
            float value = marmot_bf16_to_f32_ref(row[i]);
            if (value > max_val) {
                max_val = value;
            }
        }
        return max_val;
    }
#endif

#if HAS_AVX2
    if (ctx != nullptr && cpu_ctx_has_avx2(ctx)) {
        __m256 max_vec = _mm256_set1_ps(-FLT_MAX);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m128i vals = _mm_loadu_si128((const __m128i *)(row + i));
            __m256i vals_shifted = _mm256_slli_epi32(_mm256_cvtepu16_epi32(vals), 16);
            __m256 vals_f32 = _mm256_castsi256_ps(vals_shifted);
            max_vec = _mm256_max_ps(max_vec, vals_f32);
        }
        max_val = softmax_horizontal_max_m256(max_vec);
        for (; i < n; i++) {
            float value = marmot_bf16_to_f32_ref(row[i]);
            if (value > max_val) {
                max_val = value;
            }
        }
        return max_val;
    }
#endif

    for (size_t i = 0; i < n; i++) {
        float value = marmot_bf16_to_f32_ref(row[i]);
        if (value > max_val) {
            max_val = value;
        }
    }
    return max_val;
}

static inline void softmax_scale_bf16(cpu_context_t *ctx, marmot_bfloat16_t *row, size_t n, float inv_sum) {
#if HAS_NEON
    if (ctx != nullptr && cpu_ctx_has_neon(ctx)) {
        float32x4_t inv_vec = vdupq_n_f32(inv_sum);
        uint32x4_t bias = vdupq_n_u32(0x7FFF);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            uint16x8_t bits = vld1q_u16((const uint16_t *)(row + i));
            float32x4_t lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bits), 16));
            float32x4_t hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bits), 16));
            lo = vmulq_f32(lo, inv_vec);
            hi = vmulq_f32(hi, inv_vec);
            uint32x4_t lo_bits = vreinterpretq_u32_f32(lo);
            uint32x4_t hi_bits = vreinterpretq_u32_f32(hi);
            uint32x4_t lsb_lo = vandq_u32(vshrq_n_u32(lo_bits, 16), vdupq_n_u32(1));
            uint32x4_t lsb_hi = vandq_u32(vshrq_n_u32(hi_bits, 16), vdupq_n_u32(1));
            uint32x4_t rounded_lo = vaddq_u32(lo_bits, vaddq_u32(bias, lsb_lo));
            uint32x4_t rounded_hi = vaddq_u32(hi_bits, vaddq_u32(bias, lsb_hi));
            uint16x4_t out_lo = vmovn_u32(vshrq_n_u32(rounded_lo, 16));
            uint16x4_t out_hi = vmovn_u32(vshrq_n_u32(rounded_hi, 16));
            uint16x8_t packed = vcombine_u16(out_lo, out_hi);
            vst1q_u16((uint16_t *)(row + i), packed);
        }
        for (; i < n; i++) {
            float value = marmot_bf16_to_f32_ref(row[i]);
            row[i] = marmot_f32_to_bf16_ref(value * inv_sum);
        }
        return;
    }
#endif

#if HAS_AVX2
    if (ctx != nullptr && cpu_ctx_has_avx2(ctx)) {
        __m256 inv_vec = _mm256_set1_ps(inv_sum);
        __m256i bias = _mm256_set1_epi32(0x7FFF);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m128i vals = _mm_loadu_si128((const __m128i *)(row + i));
            __m256i vals_u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(vals), 16);
            __m256 vals_f32 = _mm256_castsi256_ps(vals_u32);
            __m256 scaled = _mm256_mul_ps(vals_f32, inv_vec);
            __m256i scaled_bits = _mm256_castps_si256(scaled);
            __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(scaled_bits, 16), _mm256_set1_epi32(1));
            __m256i rounded = _mm256_add_epi32(scaled_bits, _mm256_add_epi32(bias, lsb));
            __m256i shifted = _mm256_srli_epi32(rounded, 16);
            __m128i lo = _mm256_castsi256_si128(shifted);
            __m128i hi = _mm256_extracti128_si256(shifted, 1);
            __m128i packed = _mm_packus_epi32(lo, hi);
            packed = _mm_permute4x64_epi64(packed, 0xD8);
            _mm_storeu_si128((__m128i *)(row + i), packed);
        }
        for (; i < n; i++) {
            float value = marmot_bf16_to_f32_ref(row[i]);
            row[i] = marmot_f32_to_bf16_ref(value * inv_sum);
        }
        return;
    }
#endif

    for (size_t i = 0; i < n; i++) {
        float value = marmot_bf16_to_f32_ref(row[i]);
        row[i] = marmot_f32_to_bf16_ref(value * inv_sum);
    }
}

#endif // CPU_SOFTMAX_INTERNAL_H
