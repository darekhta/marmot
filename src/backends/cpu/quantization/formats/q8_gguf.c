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
#include "quantization/common/quant_utils.h"
#include "quantization/format_metadata.h"

#define Q8_BLOCK_SIZE MARMOT_QUANT_BLOCK_SIZE

static marmot_error_t
q8_0_quantize_block(const float *values, uint32_t count, void *block_out, [[maybe_unused]] const void *params);
static marmot_error_t
q8_1_quantize_block(const float *values, uint32_t count, void *block_out, [[maybe_unused]] const void *params);
static marmot_error_t
q8_0_dequantize_block(const void *block, uint32_t count, float *values_out, [[maybe_unused]] const void *params);
static marmot_error_t
q8_1_dequantize_block(const void *block, uint32_t count, float *values_out, [[maybe_unused]] const void *params);

static const marmot_quant_traits_t q8_0_traits = {
    .kind = MARMOT_QUANT_KIND_Q8_0,
    .name = "Q8_0",
    .block_size = CPU_QUANT_FORMAT_BLOCK_VALUES(MARMOT_QUANT_KIND_Q8_0),
    .block_bytes = CPU_QUANT_FORMAT_BLOCK_BYTES(MARMOT_QUANT_KIND_Q8_0),
    .weight_bits = 8,
    .has_zero_point = false,
    .requires_calibration = false,
    .layout = CPU_QUANT_FORMAT_LAYOUT(MARMOT_QUANT_KIND_Q8_0),
    .compute_params = nullptr,
    .quantize_block = q8_0_quantize_block,
    .dequantize_block = q8_0_dequantize_block,
    .vec_dot_block = nullptr,
};

static const marmot_quant_traits_t q8_1_traits = {
    .kind = MARMOT_QUANT_KIND_Q8_1,
    .name = "Q8_1",
    .block_size = CPU_QUANT_FORMAT_BLOCK_VALUES(MARMOT_QUANT_KIND_Q8_1),
    .block_bytes = CPU_QUANT_FORMAT_BLOCK_BYTES(MARMOT_QUANT_KIND_Q8_1),
    .weight_bits = 8,
    .has_zero_point = false,
    .requires_calibration = false,
    .layout = CPU_QUANT_FORMAT_LAYOUT(MARMOT_QUANT_KIND_Q8_1),
    .compute_params = nullptr,
    .quantize_block = q8_1_quantize_block,
    .dequantize_block = q8_1_dequantize_block,
    .vec_dot_block = nullptr,
};

MARMOT_REGISTER_QUANT_SCHEME(q8_0_traits)
MARMOT_REGISTER_QUANT_SCHEME(q8_1_traits)

#if HAS_NEON
static inline void q8_dequantize_neon(const int8_t *weights, uint32_t elems, float scale, float *dst) {
    const float32x4_t scale_vec = vdupq_n_f32(scale);
    size_t i = 0;
    for (; i + 16 <= elems; i += 16) {
        int8x16_t v = vld1q_s8(weights + i);
        int16x8_t lo16 = vmovl_s8(vget_low_s8(v));
        int16x8_t hi16 = vmovl_s8(vget_high_s8(v));

        int32x4_t lo32_a = vmovl_s16(vget_low_s16(lo16));
        int32x4_t lo32_b = vmovl_s16(vget_high_s16(lo16));
        int32x4_t hi32_a = vmovl_s16(vget_low_s16(hi16));
        int32x4_t hi32_b = vmovl_s16(vget_high_s16(hi16));

        vst1q_f32(dst + i + 0, vmulq_f32(vcvtq_f32_s32(lo32_a), scale_vec));
        vst1q_f32(dst + i + 4, vmulq_f32(vcvtq_f32_s32(lo32_b), scale_vec));
        vst1q_f32(dst + i + 8, vmulq_f32(vcvtq_f32_s32(hi32_a), scale_vec));
        vst1q_f32(dst + i + 12, vmulq_f32(vcvtq_f32_s32(hi32_b), scale_vec));
    }
    for (; i < elems; ++i) {
        dst[i] = (float)weights[i] * scale;
    }
}
#endif

#if HAS_AVX2
static inline void q8_dequantize_avx2(const int8_t *weights, uint32_t elems, float scale, float *dst) {
    const __m256 scale_vec = _mm256_set1_ps(scale);
    uint32_t i = 0;
    for (; i + 32 <= elems; i += 32) {
        __m128i b0 = _mm_loadu_si128((const __m128i *)(weights + i));
        __m128i b1 = _mm_loadu_si128((const __m128i *)(weights + i + 16));

        __m256i v0 = _mm256_cvtepi8_epi32(b0);
        __m256i v1 = _mm256_cvtepi8_epi32(_mm_srli_si128(b0, 8));
        __m256i v2 = _mm256_cvtepi8_epi32(b1);
        __m256i v3 = _mm256_cvtepi8_epi32(_mm_srli_si128(b1, 8));

        _mm256_storeu_ps(dst + i + 0, _mm256_mul_ps(_mm256_cvtepi32_ps(v0), scale_vec));
        _mm256_storeu_ps(dst + i + 8, _mm256_mul_ps(_mm256_cvtepi32_ps(v1), scale_vec));
        _mm256_storeu_ps(dst + i + 16, _mm256_mul_ps(_mm256_cvtepi32_ps(v2), scale_vec));
        _mm256_storeu_ps(dst + i + 24, _mm256_mul_ps(_mm256_cvtepi32_ps(v3), scale_vec));
    }
    for (; i < elems; ++i) {
        dst[i] = (float)weights[i] * scale;
    }
}
#endif

#if !HAS_NEON
static inline void q8_dequantize_scalar(const int8_t *weights, uint32_t elems, float scale, float *dst) {
    for (uint32_t i = 0; i < elems; ++i) {
        dst[i] = (float)weights[i] * scale;
    }
}
#endif

static void q8_dequantize_values(const int8_t *weights, uint32_t elems, float scale, float *dst) {
#if HAS_AVX2
    q8_dequantize_avx2(weights, elems, scale, dst);
#elif HAS_NEON
    q8_dequantize_neon(weights, elems, scale, dst);
#else
    q8_dequantize_scalar(weights, elems, scale, dst);
#endif
}

static marmot_error_t
q8_0_quantize_block(const float *values, uint32_t count, void *block_out, [[maybe_unused]] const void *params) {
    if (values == nullptr || block_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = count > Q8_BLOCK_SIZE ? Q8_BLOCK_SIZE : count;
    marmot_q8_0_block_t *block = (marmot_q8_0_block_t *)block_out;

    float scale = marmot_quant_compute_positive_scale(values, elems, 127.0f);
    if (scale < FLT_MIN) {
        scale = 1.0f;
    }
    const float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

    memset(block, 0, sizeof(*block));
    block->scale = marmot_native_to_float16((_Float16)scale);

    if (elems > 0) {
        cpu_quantize_q8_0_pack(values, elems, inv_scale, block->qs);
    }
    if (elems < Q8_BLOCK_SIZE) {
        memset(block->qs + elems, 0, (Q8_BLOCK_SIZE - elems) * sizeof(int8_t));
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t
q8_1_quantize_block(const float *values, uint32_t count, void *block_out, [[maybe_unused]] const void *params) {
    if (values == nullptr || block_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = count > Q8_BLOCK_SIZE ? Q8_BLOCK_SIZE : count;
    marmot_q8_1_block_t *block = (marmot_q8_1_block_t *)block_out;

    float scale = marmot_quant_compute_positive_scale(values, elems, 127.0f);
    if (scale < FLT_MIN) {
        scale = 1.0f;
    }
    const float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

    memset(block, 0, sizeof(*block));
    block->scale = marmot_native_to_float16((_Float16)scale);

    if (elems > 0) {
        cpu_quantize_q8_0_pack(values, elems, inv_scale, block->qs);
    }
    if (elems < Q8_BLOCK_SIZE) {
        memset(block->qs + elems, 0, (Q8_BLOCK_SIZE - elems) * sizeof(int8_t));
    }

    int32_t sum = 0;
    for (uint32_t i = 0; i < elems; ++i) {
        sum += block->qs[i];
    }
    block->sum = marmot_native_to_float16((_Float16)((float)sum * scale));

    return MARMOT_SUCCESS;
}

static marmot_error_t
q8_0_dequantize_block(const void *block_in, uint32_t count, float *values_out, [[maybe_unused]] const void *params) {
    if (block_in == nullptr || values_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = count > Q8_BLOCK_SIZE ? Q8_BLOCK_SIZE : count;
    const marmot_q8_0_block_t *block = (const marmot_q8_0_block_t *)block_in;
    const float scale = (float)marmot_float16_to_native(block->scale);
    q8_dequantize_values(block->qs, elems, scale, values_out);
    return MARMOT_SUCCESS;
}

static marmot_error_t
q8_1_dequantize_block(const void *block_in, uint32_t count, float *values_out, [[maybe_unused]] const void *params) {
    if (block_in == nullptr || values_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = count > Q8_BLOCK_SIZE ? Q8_BLOCK_SIZE : count;
    const marmot_q8_1_block_t *block = (const marmot_q8_1_block_t *)block_in;
    const float scale = (float)marmot_float16_to_native(block->scale);
    q8_dequantize_values(block->qs, elems, scale, values_out);
    return MARMOT_SUCCESS;
}

static marmot_error_t
require_q8_traits(marmot_quant_kind_t kind, const marmot_quant_traits_t **out_traits, const char *error_message) {
    const marmot_quant_traits_t *traits = marmot_get_quant_traits(kind);
    if (traits == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, error_message);
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    *out_traits = traits;
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_quantize_q8_0(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = nullptr;
    marmot_error_t err = require_q8_traits(MARMOT_QUANT_KIND_Q8_0, &traits, "Q8_0 traits not registered");
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return cpu_quantize_blockwise(traits, input, output);
}

marmot_error_t cpu_dequantize_q8_0(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = nullptr;
    marmot_error_t err = require_q8_traits(MARMOT_QUANT_KIND_Q8_0, &traits, "Q8_0 traits not registered");
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return cpu_dequantize_blockwise(traits, input, output);
}

marmot_error_t cpu_quantize_q8_1(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = nullptr;
    marmot_error_t err = require_q8_traits(MARMOT_QUANT_KIND_Q8_1, &traits, "Q8_1 traits not registered");
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return cpu_quantize_blockwise(traits, input, output);
}

marmot_error_t cpu_dequantize_q8_1(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = nullptr;
    marmot_error_t err = require_q8_traits(MARMOT_QUANT_KIND_Q8_1, &traits, "Q8_1 traits not registered");
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return cpu_dequantize_blockwise(traits, input, output);
}
