#include "marmot/error.h"
#include "marmot/macros.h"
#include "marmot/quant_block.h"
#include "marmot/quant_traits.h"
#include "marmot/tensor.h"

#include <float.h>
#include <math.h>
#include <string.h>

#include "core/helpers/int4.h"
#include "cpu_backend_internal.h"
#include "quantization/common/block.h"
#include "quantization/format_metadata.h"

#define Q4_0_BLOCK_SIZE MARMOT_QUANT_BLOCK_SIZE
#define Q4_1_BLOCK_SIZE MARMOT_QUANT_BLOCK_SIZE

static marmot_error_t q4_0_quantize_block(const float *values, uint32_t count, void *block_out, const void *params);
static marmot_error_t q4_1_quantize_block(const float *values, uint32_t count, void *block_out, const void *params);
static marmot_error_t q4_0_dequantize_block(const void *block, uint32_t count, float *values_out, const void *params);
static marmot_error_t q4_1_dequantize_block(const void *block, uint32_t count, float *values_out, const void *params);

static const marmot_quant_traits_t q4_0_traits = {
    .kind = MARMOT_QUANT_KIND_Q4_0,
    .name = "Q4_0",
    .block_size = CPU_QUANT_FORMAT_BLOCK_VALUES(MARMOT_QUANT_KIND_Q4_0),
    .block_bytes = CPU_QUANT_FORMAT_BLOCK_BYTES(MARMOT_QUANT_KIND_Q4_0),
    .weight_bits = 4,
    .has_zero_point = false,
    .requires_calibration = false,
    .layout = CPU_QUANT_FORMAT_LAYOUT(MARMOT_QUANT_KIND_Q4_0),
    .compute_params = nullptr,
    .quantize_block = q4_0_quantize_block,
    .dequantize_block = q4_0_dequantize_block,
    .vec_dot_block = nullptr,
};

static const marmot_quant_traits_t q4_1_traits = {
    .kind = MARMOT_QUANT_KIND_Q4_1,
    .name = "Q4_1",
    .block_size = CPU_QUANT_FORMAT_BLOCK_VALUES(MARMOT_QUANT_KIND_Q4_1),
    .block_bytes = CPU_QUANT_FORMAT_BLOCK_BYTES(MARMOT_QUANT_KIND_Q4_1),
    .weight_bits = 4,
    .has_zero_point = true,
    .requires_calibration = false,
    .layout = CPU_QUANT_FORMAT_LAYOUT(MARMOT_QUANT_KIND_Q4_1),
    .compute_params = nullptr,
    .quantize_block = q4_1_quantize_block,
    .dequantize_block = q4_1_dequantize_block,
    .vec_dot_block = nullptr,
};

MARMOT_REGISTER_QUANT_SCHEME(q4_0_traits)
MARMOT_REGISTER_QUANT_SCHEME(q4_1_traits)

#if HAS_NEON
static inline size_t neon_quantize_q4_0(const float *data, size_t len, float inv_scale, uint8_t *out) {
    const float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
    const float32x4_t bias_vec = vdupq_n_f32(8.5f);
    const int32x4_t clamp_min = vdupq_n_s32(0);
    const int32x4_t clamp_max = vdupq_n_s32(15);
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        float32x4_t v0 = vmulq_f32(vld1q_f32(data + i), inv_scale_vec);
        float32x4_t v1 = vmulq_f32(vld1q_f32(data + i + 4), inv_scale_vec);
        v0 = vaddq_f32(v0, bias_vec);
        v1 = vaddq_f32(v1, bias_vec);
        int32x4_t q0 = vcvtq_s32_f32(v0);
        int32x4_t q1 = vcvtq_s32_f32(v1);
        q0 = vmaxq_s32(q0, clamp_min);
        q0 = vminq_s32(q0, clamp_max);
        q1 = vmaxq_s32(q1, clamp_min);
        q1 = vminq_s32(q1, clamp_max);
        int16x4_t q16_0 = vqmovn_s32(q0);
        int16x4_t q16_1 = vqmovn_s32(q1);
        int16x8_t q16 = vcombine_s16(q16_0, q16_1);
        int8x8_t q8 = vqmovn_s16(q16);
        vst1_u8(out + i, vreinterpret_u8_s8(q8));
    }
    return i;
}

static inline size_t neon_quantize_q4_1(const float *data, size_t len, float min_val, float inv_scale, uint8_t *out) {
    const float32x4_t min_vec = vdupq_n_f32(min_val);
    const float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
    const float32x4_t bias_vec = vdupq_n_f32(0.5f);
    const int32x4_t clamp_min = vdupq_n_s32(0);
    const int32x4_t clamp_max = vdupq_n_s32(15);
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        float32x4_t v0 = vmulq_f32(vsubq_f32(vld1q_f32(data + i), min_vec), inv_scale_vec);
        float32x4_t v1 = vmulq_f32(vsubq_f32(vld1q_f32(data + i + 4), min_vec), inv_scale_vec);
        v0 = vaddq_f32(v0, bias_vec);
        v1 = vaddq_f32(v1, bias_vec);
        int32x4_t q0 = vcvtq_s32_f32(v0);
        int32x4_t q1 = vcvtq_s32_f32(v1);
        q0 = vmaxq_s32(q0, clamp_min);
        q0 = vminq_s32(q0, clamp_max);
        q1 = vmaxq_s32(q1, clamp_min);
        q1 = vminq_s32(q1, clamp_max);
        int16x4_t q16_0 = vqmovn_s32(q0);
        int16x4_t q16_1 = vqmovn_s32(q1);
        int16x8_t q16 = vcombine_s16(q16_0, q16_1);
        int8x8_t q8 = vqmovn_s16(q16);
        vst1_u8(out + i, vreinterpret_u8_s8(q8));
    }
    return i;
}

static inline void neon_dequantize_q4_0(const marmot_q4_0_block_t *block, float scale, float *dst) {
    const float32x4_t scale_vec = vdupq_n_f32(scale);
    const int16x8_t offset = vdupq_n_s16(8);

    uint8x16_t packed = vld1q_u8(block->qs);
    uint8x16_t lo = vandq_u8(packed, vdupq_n_u8(0x0F));
    uint8x16_t hi = vshrq_n_u8(packed, 4);

    uint8x8_t lo0_u8 = vget_low_u8(lo);
    uint8x8_t lo1_u8 = vget_high_u8(lo);
    int16x8_t lo0_s16 = vreinterpretq_s16_u16(vmovl_u8(lo0_u8));
    int16x8_t lo1_s16 = vreinterpretq_s16_u16(vmovl_u8(lo1_u8));
    lo0_s16 = vsubq_s16(lo0_s16, offset);
    lo1_s16 = vsubq_s16(lo1_s16, offset);

    int32x4_t lo0_s32_lo = vmovl_s16(vget_low_s16(lo0_s16));
    int32x4_t lo0_s32_hi = vmovl_s16(vget_high_s16(lo0_s16));
    int32x4_t lo1_s32_lo = vmovl_s16(vget_low_s16(lo1_s16));
    int32x4_t lo1_s32_hi = vmovl_s16(vget_high_s16(lo1_s16));

    vst1q_f32(dst + 0, vmulq_f32(vcvtq_f32_s32(lo0_s32_lo), scale_vec));
    vst1q_f32(dst + 4, vmulq_f32(vcvtq_f32_s32(lo0_s32_hi), scale_vec));
    vst1q_f32(dst + 8, vmulq_f32(vcvtq_f32_s32(lo1_s32_lo), scale_vec));
    vst1q_f32(dst + 12, vmulq_f32(vcvtq_f32_s32(lo1_s32_hi), scale_vec));

    uint8x8_t hi0_u8 = vget_low_u8(hi);
    uint8x8_t hi1_u8 = vget_high_u8(hi);
    int16x8_t hi0_s16 = vreinterpretq_s16_u16(vmovl_u8(hi0_u8));
    int16x8_t hi1_s16 = vreinterpretq_s16_u16(vmovl_u8(hi1_u8));
    hi0_s16 = vsubq_s16(hi0_s16, offset);
    hi1_s16 = vsubq_s16(hi1_s16, offset);

    int32x4_t hi0_s32_lo = vmovl_s16(vget_low_s16(hi0_s16));
    int32x4_t hi0_s32_hi = vmovl_s16(vget_high_s16(hi0_s16));
    int32x4_t hi1_s32_lo = vmovl_s16(vget_low_s16(hi1_s16));
    int32x4_t hi1_s32_hi = vmovl_s16(vget_high_s16(hi1_s16));

    vst1q_f32(dst + 16, vmulq_f32(vcvtq_f32_s32(hi0_s32_lo), scale_vec));
    vst1q_f32(dst + 20, vmulq_f32(vcvtq_f32_s32(hi0_s32_hi), scale_vec));
    vst1q_f32(dst + 24, vmulq_f32(vcvtq_f32_s32(hi1_s32_lo), scale_vec));
    vst1q_f32(dst + 28, vmulq_f32(vcvtq_f32_s32(hi1_s32_hi), scale_vec));
}

static inline void neon_dequantize_q4_1(const marmot_q4_1_block_t *block, float scale, float min_val, float *dst) {
    const float32x4_t scale_vec = vdupq_n_f32(scale);
    const float32x4_t min_vec = vdupq_n_f32(min_val);

    uint8x16_t packed = vld1q_u8(block->qs);
    uint8x16_t lo = vandq_u8(packed, vdupq_n_u8(0x0F));
    uint8x16_t hi = vshrq_n_u8(packed, 4);

    uint8x8_t lo0_u8 = vget_low_u8(lo);
    uint8x8_t lo1_u8 = vget_high_u8(lo);
    int16x8_t lo0_s16 = vreinterpretq_s16_u16(vmovl_u8(lo0_u8));
    int16x8_t lo1_s16 = vreinterpretq_s16_u16(vmovl_u8(lo1_u8));

    int32x4_t lo0_s32_lo = vmovl_s16(vget_low_s16(lo0_s16));
    int32x4_t lo0_s32_hi = vmovl_s16(vget_high_s16(lo0_s16));
    int32x4_t lo1_s32_lo = vmovl_s16(vget_low_s16(lo1_s16));
    int32x4_t lo1_s32_hi = vmovl_s16(vget_high_s16(lo1_s16));

    vst1q_f32(dst + 0, vaddq_f32(vmulq_f32(vcvtq_f32_s32(lo0_s32_lo), scale_vec), min_vec));
    vst1q_f32(dst + 4, vaddq_f32(vmulq_f32(vcvtq_f32_s32(lo0_s32_hi), scale_vec), min_vec));
    vst1q_f32(dst + 8, vaddq_f32(vmulq_f32(vcvtq_f32_s32(lo1_s32_lo), scale_vec), min_vec));
    vst1q_f32(dst + 12, vaddq_f32(vmulq_f32(vcvtq_f32_s32(lo1_s32_hi), scale_vec), min_vec));

    uint8x8_t hi0_u8 = vget_low_u8(hi);
    uint8x8_t hi1_u8 = vget_high_u8(hi);
    int16x8_t hi0_s16 = vreinterpretq_s16_u16(vmovl_u8(hi0_u8));
    int16x8_t hi1_s16 = vreinterpretq_s16_u16(vmovl_u8(hi1_u8));

    int32x4_t hi0_s32_lo = vmovl_s16(vget_low_s16(hi0_s16));
    int32x4_t hi0_s32_hi = vmovl_s16(vget_high_s16(hi0_s16));
    int32x4_t hi1_s32_lo = vmovl_s16(vget_low_s16(hi1_s16));
    int32x4_t hi1_s32_hi = vmovl_s16(vget_high_s16(hi1_s16));

    vst1q_f32(dst + 16, vaddq_f32(vmulq_f32(vcvtq_f32_s32(hi0_s32_lo), scale_vec), min_vec));
    vst1q_f32(dst + 20, vaddq_f32(vmulq_f32(vcvtq_f32_s32(hi0_s32_hi), scale_vec), min_vec));
    vst1q_f32(dst + 24, vaddq_f32(vmulq_f32(vcvtq_f32_s32(hi1_s32_lo), scale_vec), min_vec));
    vst1q_f32(dst + 28, vaddq_f32(vmulq_f32(vcvtq_f32_s32(hi1_s32_hi), scale_vec), min_vec));
}
#endif

#if HAS_AVX2
static inline void avx2_dequantize_q4_0(const marmot_q4_0_block_t *block, float scale, float *dst) {
    const __m128i bytes = _mm_loadu_si128((const __m128i *)block->qs);
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i offset = _mm_set1_epi16(8);
    const __m128i lo = _mm_and_si128(bytes, mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);

    __m128i lo0 = _mm_cvtepu8_epi16(lo);
    __m128i lo1 = _mm_cvtepu8_epi16(_mm_srli_si128(lo, 8));
    __m128i hi0 = _mm_cvtepu8_epi16(hi);
    __m128i hi1 = _mm_cvtepu8_epi16(_mm_srli_si128(hi, 8));

    lo0 = _mm_sub_epi16(lo0, offset);
    lo1 = _mm_sub_epi16(lo1, offset);
    hi0 = _mm_sub_epi16(hi0, offset);
    hi1 = _mm_sub_epi16(hi1, offset);

    const __m256 scale_vec = _mm256_set1_ps(scale);
    __m256 f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo0)), scale_vec);
    __m256 f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo1)), scale_vec);
    __m256 f2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi0)), scale_vec);
    __m256 f3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi1)), scale_vec);

    _mm256_storeu_ps(dst + 0, f0);
    _mm256_storeu_ps(dst + 8, f1);
    _mm256_storeu_ps(dst + 16, f2);
    _mm256_storeu_ps(dst + 24, f3);
}

static inline void avx2_dequantize_q4_1(const marmot_q4_1_block_t *block, float scale, float min_val, float *dst) {
    const __m128i bytes = _mm_loadu_si128((const __m128i *)block->qs);
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(bytes, mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);

    __m128i lo0 = _mm_cvtepu8_epi16(lo);
    __m128i lo1 = _mm_cvtepu8_epi16(_mm_srli_si128(lo, 8));
    __m128i hi0 = _mm_cvtepu8_epi16(hi);
    __m128i hi1 = _mm_cvtepu8_epi16(_mm_srli_si128(hi, 8));

    const __m256 scale_vec = _mm256_set1_ps(scale);
    const __m256 min_vec = _mm256_set1_ps(min_val);
    __m256 f0 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo0)), scale_vec), min_vec);
    __m256 f1 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo1)), scale_vec), min_vec);
    __m256 f2 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi0)), scale_vec), min_vec);
    __m256 f3 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi1)), scale_vec), min_vec);

    _mm256_storeu_ps(dst + 0, f0);
    _mm256_storeu_ps(dst + 8, f1);
    _mm256_storeu_ps(dst + 16, f2);
    _mm256_storeu_ps(dst + 24, f3);
}
#endif

static marmot_error_t
q4_0_quantize_block(const float *values, uint32_t count, void *block_out, [[maybe_unused]] const void *params) {
    if (values == nullptr || block_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = count > MARMOT_QUANT_BLOCK_SIZE ? MARMOT_QUANT_BLOCK_SIZE : count;
    marmot_q4_0_block_t *block = (marmot_q4_0_block_t *)block_out;

    float block_values[MARMOT_QUANT_BLOCK_SIZE] = {0};
    if (elems > 0) {
        memcpy(block_values, values, elems * sizeof(float));
    }

    float amax = 0.0f;
    float max_val = 0.0f;
    for (uint32_t i = 0; i < MARMOT_QUANT_BLOCK_SIZE; ++i) {
        const float v = block_values[i];
        const float abs_v = fabsf(v);
        if (abs_v > amax) {
            amax = abs_v;
            max_val = v;
        }
    }

    float scale = 0.0f;
    float inv_scale = 0.0f;
    if (amax > 0.0f) {
        scale = -max_val / 8.0f;
        if (scale != 0.0f) {
            inv_scale = 1.0f / scale;
        }
    }

    block->scale = marmot_native_to_float16((_Float16)scale);
    memset(block->qs, 0, sizeof(block->qs));

    uint8_t qvals[MARMOT_QUANT_BLOCK_SIZE] = {0};
    uint32_t processed = 0;

#if HAS_NEON
    processed = (uint32_t)neon_quantize_q4_0(block_values, MARMOT_QUANT_BLOCK_SIZE, inv_scale, qvals);
#endif

    for (; processed < MARMOT_QUANT_BLOCK_SIZE; ++processed) {
        float x = block_values[processed] * inv_scale;
        int32_t q = (int32_t)(int8_t)(x + 8.5f);
        if (q < 0) {
            q = 0;
        } else if (q > 15) {
            q = 15;
        }
        qvals[processed] = (uint8_t)q;
    }

    const uint32_t half = MARMOT_QUANT_BLOCK_SIZE / 2;
    for (uint32_t j = 0; j < half; ++j) {
        const uint8_t q0 = (uint8_t)(qvals[j] & 0x0F);
        const uint8_t q1 = (uint8_t)(qvals[j + half] & 0x0F);
        block->qs[j] = (uint8_t)((q1 << 4) | q0);
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t
q4_1_quantize_block(const float *values, uint32_t count, void *block_out, [[maybe_unused]] const void *params) {
    if (values == nullptr || block_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = count > MARMOT_QUANT_BLOCK_SIZE ? MARMOT_QUANT_BLOCK_SIZE : count;
    marmot_q4_1_block_t *block = (marmot_q4_1_block_t *)block_out;

    marmot_block_minmax_t minmax = cpu_quant_block_minmax(values, elems);
    float min_val = minmax.min_val;
    float max_val = minmax.max_val;
    float scale = (max_val - min_val) / 15.0f;
    if (scale < FLT_MIN) {
        scale = 1.0f;
    }
    const float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

    block->scale = marmot_native_to_float16((_Float16)scale);
    block->min = marmot_native_to_float16((_Float16)min_val);
    memset(block->qs, 0, sizeof(block->qs));

    uint8_t qvals[MARMOT_QUANT_BLOCK_SIZE] = {0};
    uint32_t processed = 0;

#if HAS_NEON
    if (elems >= 8) {
        processed = (uint32_t)neon_quantize_q4_1(values, elems, min_val, inv_scale, qvals);
    }
#endif
    for (; processed < elems; ++processed) {
        float x = (values[processed] - min_val) * inv_scale;
        int32_t q = (int32_t)(int8_t)(x + 0.5f);
        if (q < 0) {
            q = 0;
        } else if (q > 15) {
            q = 15;
        }
        qvals[processed] = (uint8_t)q;
    }

    const uint32_t half = MARMOT_QUANT_BLOCK_SIZE / 2;
    for (uint32_t j = 0; j < half; ++j) {
        const uint32_t idx0 = j;
        const uint32_t idx1 = j + half;
        const uint8_t q0 = (idx0 < elems) ? (uint8_t)(qvals[idx0] & 0x0F) : 0;
        const uint8_t q1 = (idx1 < elems) ? (uint8_t)(qvals[idx1] & 0x0F) : 0;
        block->qs[j] = (uint8_t)((q1 << 4) | q0);
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t
q4_0_dequantize_block(const void *block_in, uint32_t count, float *values_out, [[maybe_unused]] const void *params) {
    if (block_in == nullptr || values_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = count > MARMOT_QUANT_BLOCK_SIZE ? MARMOT_QUANT_BLOCK_SIZE : count;
    const marmot_q4_0_block_t *block = (const marmot_q4_0_block_t *)block_in;
    const float scale = (float)marmot_float16_to_native(block->scale);

    float decoded[MARMOT_QUANT_BLOCK_SIZE];
#if HAS_AVX2
    avx2_dequantize_q4_0(block, scale, decoded);
#elif HAS_NEON
    neon_dequantize_q4_0(block, scale, decoded);
#else
    const size_t half = MARMOT_QUANT_BLOCK_SIZE / 2;
    for (size_t j = 0; j < half; ++j) {
        const uint8_t packed = block->qs[j];
        const uint8_t q0 = packed & 0x0F;
        const uint8_t q1 = packed >> 4;
        decoded[j] = ((int8_t)q0 - 8) * scale;
        decoded[j + half] = ((int8_t)q1 - 8) * scale;
    }
#endif
    memcpy(values_out, decoded, elems * sizeof(float));
    return MARMOT_SUCCESS;
}

static marmot_error_t
q4_1_dequantize_block(const void *block_in, uint32_t count, float *values_out, [[maybe_unused]] const void *params) {
    if (block_in == nullptr || values_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = count > MARMOT_QUANT_BLOCK_SIZE ? MARMOT_QUANT_BLOCK_SIZE : count;
    const marmot_q4_1_block_t *block = (const marmot_q4_1_block_t *)block_in;
    const float scale = (float)marmot_float16_to_native(block->scale);
    const float min_val = (float)marmot_float16_to_native(block->min);

    float decoded[MARMOT_QUANT_BLOCK_SIZE];
#if HAS_AVX2
    avx2_dequantize_q4_1(block, scale, min_val, decoded);
#elif HAS_NEON
    neon_dequantize_q4_1(block, scale, min_val, decoded);
#else
    const size_t half = MARMOT_QUANT_BLOCK_SIZE / 2;
    for (size_t j = 0; j < half; ++j) {
        const uint8_t packed = block->qs[j];
        const uint8_t q0 = packed & 0x0F;
        const uint8_t q1 = packed >> 4;
        decoded[j] = (float)q0 * scale + min_val;
        decoded[j + half] = (float)q1 * scale + min_val;
    }
#endif
    memcpy(values_out, decoded, elems * sizeof(float));
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_quantize_q4_0(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = marmot_get_quant_traits(MARMOT_QUANT_KIND_Q4_0);
    if (traits == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Q4_0 traits not registered");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    return cpu_quantize_blockwise(traits, input, output);
}

marmot_error_t cpu_dequantize_q4_0(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = marmot_get_quant_traits(MARMOT_QUANT_KIND_Q4_0);
    if (traits == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Q4_0 traits not registered");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    return cpu_dequantize_blockwise(traits, input, output);
}

marmot_error_t cpu_quantize_q4_1(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = marmot_get_quant_traits(MARMOT_QUANT_KIND_Q4_1);
    if (traits == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Q4_1 traits not registered");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    return cpu_quantize_blockwise(traits, input, output);
}

marmot_error_t cpu_dequantize_q4_1(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = marmot_get_quant_traits(MARMOT_QUANT_KIND_Q4_1);
    if (traits == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Q4_1 traits not registered");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    return cpu_dequantize_blockwise(traits, input, output);
}
