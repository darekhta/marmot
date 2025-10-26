#include "marmot/error.h"
#include "marmot/macros.h"
#include "marmot/quant_block.h"
#include "marmot/quant_traits.h"
#include "marmot/tensor.h"

#include <float.h>
#include <math.h>
#include <string.h>

#include "cpu_backend_internal.h"
#include "quantization/common/block.h"
#include "quantization/format_metadata.h"

#define Q5_0_BLOCK_SIZE MARMOT_QUANT_BLOCK_SIZE
#define Q5_1_BLOCK_SIZE MARMOT_QUANT_BLOCK_SIZE

static marmot_error_t
q5_0_quantize_block(const float *values, uint32_t count, void *block_out, [[maybe_unused]] const void *params);
static marmot_error_t
q5_1_quantize_block(const float *values, uint32_t count, void *block_out, [[maybe_unused]] const void *params);
static marmot_error_t
q5_0_dequantize_block(const void *block, uint32_t count, float *values_out, [[maybe_unused]] const void *params);
static marmot_error_t
q5_1_dequantize_block(const void *block, uint32_t count, float *values_out, [[maybe_unused]] const void *params);

static const marmot_quant_traits_t q5_0_traits = {
    .kind = MARMOT_QUANT_KIND_Q5_0,
    .name = "Q5_0",
    .block_size = CPU_QUANT_FORMAT_BLOCK_VALUES(MARMOT_QUANT_KIND_Q5_0),
    .block_bytes = CPU_QUANT_FORMAT_BLOCK_BYTES(MARMOT_QUANT_KIND_Q5_0),
    .weight_bits = 5,
    .has_zero_point = false,
    .requires_calibration = false,
    .layout = CPU_QUANT_FORMAT_LAYOUT(MARMOT_QUANT_KIND_Q5_0),
    .compute_params = nullptr,
    .quantize_block = q5_0_quantize_block,
    .dequantize_block = q5_0_dequantize_block,
    .vec_dot_block = nullptr,
};

static const marmot_quant_traits_t q5_1_traits = {
    .kind = MARMOT_QUANT_KIND_Q5_1,
    .name = "Q5_1",
    .block_size = CPU_QUANT_FORMAT_BLOCK_VALUES(MARMOT_QUANT_KIND_Q5_1),
    .block_bytes = CPU_QUANT_FORMAT_BLOCK_BYTES(MARMOT_QUANT_KIND_Q5_1),
    .weight_bits = 5,
    .has_zero_point = true,
    .requires_calibration = false,
    .layout = CPU_QUANT_FORMAT_LAYOUT(MARMOT_QUANT_KIND_Q5_1),
    .compute_params = nullptr,
    .quantize_block = q5_1_quantize_block,
    .dequantize_block = q5_1_dequantize_block,
    .vec_dot_block = nullptr,
};

MARMOT_REGISTER_QUANT_SCHEME(q5_0_traits)
MARMOT_REGISTER_QUANT_SCHEME(q5_1_traits)

#if HAS_NEON
static inline void neon_q5_0_to_f32(const uint8_t *qvals, float scale, float *dst) {
    const float32x4_t scale_vec = vdupq_n_f32(scale);
    const int16x8_t offset = vdupq_n_s16(16);
    for (size_t i = 0; i < Q5_0_BLOCK_SIZE; i += 8) {
        uint8x8_t v_u8 = vld1_u8(qvals + i);
        int16x8_t v_s16 = vreinterpretq_s16_u16(vmovl_u8(v_u8));
        v_s16 = vsubq_s16(v_s16, offset);
        int32x4_t v_s32_lo = vmovl_s16(vget_low_s16(v_s16));
        int32x4_t v_s32_hi = vmovl_s16(vget_high_s16(v_s16));
        vst1q_f32(dst + i, vmulq_f32(vcvtq_f32_s32(v_s32_lo), scale_vec));
        vst1q_f32(dst + i + 4, vmulq_f32(vcvtq_f32_s32(v_s32_hi), scale_vec));
    }
}

static inline void neon_q5_1_to_f32(const uint8_t *qvals, float scale, float min_val, float *dst) {
    const float32x4_t scale_vec = vdupq_n_f32(scale);
    const float32x4_t min_vec = vdupq_n_f32(min_val);
    for (size_t i = 0; i < Q5_1_BLOCK_SIZE; i += 8) {
        uint8x8_t v_u8 = vld1_u8(qvals + i);
        int16x8_t v_s16 = vreinterpretq_s16_u16(vmovl_u8(v_u8));
        int32x4_t v_s32_lo = vmovl_s16(vget_low_s16(v_s16));
        int32x4_t v_s32_hi = vmovl_s16(vget_high_s16(v_s16));
        vst1q_f32(dst + i, vaddq_f32(vmulq_f32(vcvtq_f32_s32(v_s32_lo), scale_vec), min_vec));
        vst1q_f32(dst + i + 4, vaddq_f32(vmulq_f32(vcvtq_f32_s32(v_s32_hi), scale_vec), min_vec));
    }
}
#endif

#if HAS_AVX2
static inline void avx2_q5_0_to_f32(const uint8_t *qvals, float scale, float *dst) {
    const __m256 scale_vec = _mm256_set1_ps(scale);
    const __m128i offset = _mm_set1_epi16(16);
    for (size_t i = 0; i < Q5_0_BLOCK_SIZE; i += 16) {
        __m128i bytes = _mm_loadu_si128((const __m128i *)(qvals + i));
        __m128i lo = _mm_cvtepu8_epi16(bytes);
        __m128i hi = _mm_cvtepu8_epi16(_mm_srli_si128(bytes, 8));
        lo = _mm_sub_epi16(lo, offset);
        hi = _mm_sub_epi16(hi, offset);
        __m256 f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo)), scale_vec);
        __m256 f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi)), scale_vec);
        _mm256_storeu_ps(dst + i, f0);
        _mm256_storeu_ps(dst + i + 8, f1);
    }
}

static inline void avx2_q5_1_to_f32(const uint8_t *qvals, float scale, float min_val, float *dst) {
    const __m256 scale_vec = _mm256_set1_ps(scale);
    const __m256 min_vec = _mm256_set1_ps(min_val);
    for (size_t i = 0; i < Q5_1_BLOCK_SIZE; i += 16) {
        __m128i bytes = _mm_loadu_si128((const __m128i *)(qvals + i));
        __m128i lo = _mm_cvtepu8_epi16(bytes);
        __m128i hi = _mm_cvtepu8_epi16(_mm_srli_si128(bytes, 8));
        __m256 f0 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo)), scale_vec), min_vec);
        __m256 f1 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi)), scale_vec), min_vec);
        _mm256_storeu_ps(dst + i, f0);
        _mm256_storeu_ps(dst + i + 8, f1);
    }
}
#endif

static marmot_error_t
q5_0_quantize_block(const float *values, uint32_t count, void *block_out, [[maybe_unused]] const void *params) {
    if (values == nullptr || block_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = count > Q5_0_BLOCK_SIZE ? Q5_0_BLOCK_SIZE : count;
    marmot_q5_0_block_t *block = (marmot_q5_0_block_t *)block_out;

    float amax = 0.0f;
    float max_val = 0.0f;
    for (uint32_t i = 0; i < elems; ++i) {
        const float v = values[i];
        const float abs_v = fabsf(v);
        if (abs_v > amax) {
            amax = abs_v;
            max_val = v;
        }
    }

    const float d = max_val / -16.0f;
    float scale = (d != 0.0f) ? d : 1.0f;
    const float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

    memset(block, 0, sizeof(*block));
    block->scale = marmot_native_to_float16((_Float16)scale);

    int8_t qtmp[Q5_0_BLOCK_SIZE] = {0};
    if (elems > 0) {
        cpu_quantize_q5_0_pack(values, elems, inv_scale, qtmp);
    }

    uint32_t qh = 0;
    const size_t half = Q5_0_BLOCK_SIZE / 2;
    for (size_t j = 0; j < half; ++j) {
        const size_t idx0 = j;
        const size_t idx1 = j + half;

        uint8_t q0 = 0;
        uint8_t q1 = 0;
        if (idx0 < elems) {
            int32_t v = (int32_t)qtmp[idx0] + 16;
            if (v < 0) {
                v = 0;
            } else if (v > 31) {
                v = 31;
            }
            q0 = (uint8_t)v;
        }
        if (idx1 < elems) {
            int32_t v = (int32_t)qtmp[idx1] + 16;
            if (v < 0) {
                v = 0;
            } else if (v > 31) {
                v = 31;
            }
            q1 = (uint8_t)v;
        }

        block->qs[j] = (uint8_t)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
        qh |= ((uint32_t)((q0 & 0x10u) >> 4)) << j;
        qh |= ((uint32_t)((q1 & 0x10u) >> 4)) << (j + half);
    }

    memcpy(block->qh, &qh, sizeof(block->qh));
    return MARMOT_SUCCESS;
}

static marmot_error_t
q5_1_quantize_block(const float *values, uint32_t count, void *block_out, [[maybe_unused]] const void *params) {
    if (values == nullptr || block_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = count > Q5_1_BLOCK_SIZE ? Q5_1_BLOCK_SIZE : count;
    marmot_q5_1_block_t *block = (marmot_q5_1_block_t *)block_out;

    marmot_block_minmax_t minmax = cpu_quant_block_minmax(values, elems);
    float min_val = minmax.min_val;
    float max_val = minmax.max_val;
    float scale = (max_val - min_val) / 31.0f;
    if (scale < FLT_MIN) {
        scale = 1.0f;
    }
    const float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

    memset(block, 0, sizeof(*block));
    block->scale = marmot_native_to_float16((_Float16)scale);
    block->min = marmot_native_to_float16((_Float16)min_val);

    uint8_t qtmp[Q5_1_BLOCK_SIZE] = {0};
    if (elems > 0) {
        cpu_quantize_q5_1_pack(values, elems, min_val, inv_scale, qtmp);
    }

    uint32_t qh = 0;
    const size_t half = Q5_1_BLOCK_SIZE / 2;
    for (size_t j = 0; j < half; ++j) {
        const size_t idx0 = j;
        const size_t idx1 = j + half;

        uint8_t q0 = 0;
        uint8_t q1 = 0;
        if (idx0 < elems) {
            uint8_t v = qtmp[idx0] > 31 ? 31 : qtmp[idx0];
            q0 = v;
        }
        if (idx1 < elems) {
            uint8_t v = qtmp[idx1] > 31 ? 31 : qtmp[idx1];
            q1 = v;
        }

        block->qs[j] = (uint8_t)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
        qh |= ((uint32_t)((q0 & 0x10u) >> 4)) << j;
        qh |= ((uint32_t)((q1 & 0x10u) >> 4)) << (j + half);
    }

    memcpy(block->qh, &qh, sizeof(block->qh));
    return MARMOT_SUCCESS;
}

static marmot_error_t
q5_0_dequantize_block(const void *block_in, uint32_t count, float *values_out, [[maybe_unused]] const void *params) {
    if (block_in == nullptr || values_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = count > Q5_0_BLOCK_SIZE ? Q5_0_BLOCK_SIZE : count;
    const marmot_q5_0_block_t *block = (const marmot_q5_0_block_t *)block_in;
    const float scale = (float)marmot_float16_to_native(block->scale);

    uint8_t qvals[Q5_0_BLOCK_SIZE];
    uint32_t qh;
    memcpy(&qh, block->qh, sizeof(qh));
    const size_t half = Q5_0_BLOCK_SIZE / 2;
    for (size_t j = 0; j < half; ++j) {
        const uint8_t qs_byte = block->qs[j];
        uint8_t q0 = qs_byte & 0x0F;
        uint8_t q1 = qs_byte >> 4;
        q0 |= (uint8_t)(((qh >> (j + 0)) & 0x1u) << 4);
        q1 |= (uint8_t)(((qh >> (j + half)) & 0x1u) << 4);
        qvals[j] = q0;
        qvals[j + half] = q1;
    }

    float decoded[Q5_0_BLOCK_SIZE];
#if HAS_AVX2
    avx2_q5_0_to_f32(qvals, scale, decoded);
#elif HAS_NEON
    neon_q5_0_to_f32(qvals, scale, decoded);
#else
    for (size_t i = 0; i < Q5_0_BLOCK_SIZE; ++i) {
        int32_t v = (int32_t)qvals[i] - 16;
        decoded[i] = (float)v * scale;
    }
#endif
    memcpy(values_out, decoded, elems * sizeof(float));
    return MARMOT_SUCCESS;
}

static marmot_error_t
q5_1_dequantize_block(const void *block_in, uint32_t count, float *values_out, [[maybe_unused]] const void *params) {
    if (block_in == nullptr || values_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = count > Q5_1_BLOCK_SIZE ? Q5_1_BLOCK_SIZE : count;
    const marmot_q5_1_block_t *block = (const marmot_q5_1_block_t *)block_in;
    const float scale = (float)marmot_float16_to_native(block->scale);
    const float min_val = (float)marmot_float16_to_native(block->min);

    uint8_t qvals[Q5_1_BLOCK_SIZE];
    uint32_t qh;
    memcpy(&qh, block->qh, sizeof(qh));
    const size_t half = Q5_1_BLOCK_SIZE / 2;
    for (size_t j = 0; j < half; ++j) {
        const uint8_t qs_byte = block->qs[j];
        uint8_t q0 = qs_byte & 0x0F;
        uint8_t q1 = qs_byte >> 4;
        q0 |= (uint8_t)(((qh >> (j + 0)) & 0x1u) << 4);
        q1 |= (uint8_t)(((qh >> (j + half)) & 0x1u) << 4);
        qvals[j] = q0;
        qvals[j + half] = q1;
    }

    float decoded[Q5_1_BLOCK_SIZE];
#if HAS_AVX2
    avx2_q5_1_to_f32(qvals, scale, min_val, decoded);
#elif HAS_NEON
    neon_q5_1_to_f32(qvals, scale, min_val, decoded);
#else
    for (size_t i = 0; i < Q5_1_BLOCK_SIZE; ++i) {
        decoded[i] = (float)qvals[i] * scale + min_val;
    }
#endif
    memcpy(values_out, decoded, elems * sizeof(float));
    return MARMOT_SUCCESS;
}

static marmot_error_t
require_q5_traits(marmot_quant_kind_t kind, const marmot_quant_traits_t **out_traits, const char *error_message) {
    const marmot_quant_traits_t *traits = marmot_get_quant_traits(kind);
    if (traits == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, error_message);
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    *out_traits = traits;
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_quantize_q5_0(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = nullptr;
    marmot_error_t err = require_q5_traits(MARMOT_QUANT_KIND_Q5_0, &traits, "Q5_0 traits not registered");
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return cpu_quantize_blockwise(traits, input, output);
}

marmot_error_t cpu_dequantize_q5_0(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = nullptr;
    marmot_error_t err = require_q5_traits(MARMOT_QUANT_KIND_Q5_0, &traits, "Q5_0 traits not registered");
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return cpu_dequantize_blockwise(traits, input, output);
}

marmot_error_t cpu_quantize_q5_1(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = nullptr;
    marmot_error_t err = require_q5_traits(MARMOT_QUANT_KIND_Q5_1, &traits, "Q5_1 traits not registered");
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return cpu_quantize_blockwise(traits, input, output);
}

marmot_error_t cpu_dequantize_q5_1(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = nullptr;
    marmot_error_t err = require_q5_traits(MARMOT_QUANT_KIND_Q5_1, &traits, "Q5_1 traits not registered");
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return cpu_dequantize_blockwise(traits, input, output);
}
