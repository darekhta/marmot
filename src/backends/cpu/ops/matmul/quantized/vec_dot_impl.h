#ifndef VEC_DOT_NAME
#error "VEC_DOT_NAME(base) must be defined before including vec_dot_impl.h"
#endif

#define FN(base) VEC_DOT_NAME(base)

#define cpu_vec_dot_q4_0_q8_0 FN(q4_0_q8_0)
#define cpu_vec_dot_q4_0_f16 FN(q4_0_f16)
#define cpu_vec_dot_q4_1_q8_0 FN(q4_1_q8_0)
#define cpu_vec_dot_q4_1_f16 FN(q4_1_f16)
#define cpu_vec_dot_q5_0_q8_0 FN(q5_0_q8_0)
#define cpu_vec_dot_q5_1_q8_0 FN(q5_1_q8_0)
#define cpu_vec_dot_q5_0_f16 FN(q5_0_f16)
#define cpu_vec_dot_q5_1_f16 FN(q5_1_f16)
#define cpu_vec_dot_q8_0_f16 FN(q8_0_f16)
#define cpu_vec_dot_q8_0_q8_0 FN(q8_0_q8_0)
#define cpu_vec_dot_q8_1_q8_0 FN(q8_1_q8_0)
#define cpu_vec_dot_q8_1_f16 FN(q8_1_f16)
#define cpu_vec_dot_q2_k_q8_k FN(q2_k_q8_k)
#define cpu_vec_dot_q3_k_q8_k FN(q3_k_q8_k)
#define cpu_vec_dot_q4_k_q8_k FN(q4_k_q8_k)
#define cpu_vec_dot_q5_k_q8_k FN(q5_k_q8_k)
#define cpu_vec_dot_q6_k_q8_k FN(q6_k_q8_k)
#define cpu_vec_dot_q8_k_q8_k FN(q8_k_q8_k)
#define cpu_vec_dot_q2_k_f16 FN(q2_k_f16)
#define cpu_vec_dot_q3_k_f16 FN(q3_k_f16)
#define cpu_vec_dot_q4_k_f16 FN(q4_k_f16)
#define cpu_vec_dot_q5_k_f16 FN(q5_k_f16)
#define cpu_vec_dot_q6_k_f16 FN(q6_k_f16)
#define cpu_vec_dot_q8_k_f16 FN(q8_k_f16)

#include <stddef.h>
#include <stdint.h>

#include <string.h>

static inline int32_t dot_int16_scalar(const int16_t *lhs, const int16_t *rhs, size_t len) {
    int32_t sum = 0;
    for (size_t i = 0; i < len; ++i) {
        sum += (int32_t)lhs[i] * (int32_t)rhs[i];
    }
    return sum;
}

#if HAS_NEON
static inline int32_t dot_int16_neon(const int16_t *lhs, const int16_t *rhs, size_t len) {
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    for (size_t i = 0; i < len; i += 8) {
        int16x8_t lv = vld1q_s16(lhs + i);
        int16x8_t rv = vld1q_s16(rhs + i);
        acc0 = vmlal_s16(acc0, vget_low_s16(lv), vget_low_s16(rv));
        acc1 = vmlal_s16(acc1, vget_high_s16(lv), vget_high_s16(rv));
    }
    int32x4_t sum = vaddq_s32(acc0, acc1);
#if defined(__aarch64__)
    return vaddvq_s32(sum);
#else
    int64x2_t sum64 = vpaddlq_s32(sum);
    return (int32_t)(vgetq_lane_s64(sum64, 0) + vgetq_lane_s64(sum64, 1));
#endif
}
#endif

#if HAS_AVX2
static inline int32_t dot_int16_avx2(const int16_t *lhs, const int16_t *rhs, size_t len) {
    __m256i acc = _mm256_setzero_si256();
    for (size_t i = 0; i < len; i += 16) {
        __m256i lv = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i rv = _mm256_loadu_si256((const __m256i *)(rhs + i));
        __m256i madd = _mm256_madd_epi16(lv, rv);
        acc = _mm256_add_epi32(acc, madd);
    }
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(sum128);
}
#endif

static inline int32_t dot_int16(const int16_t *lhs, const int16_t *rhs, size_t len) {
#if HAS_AVX2
    if ((len & 15u) == 0) {
        return dot_int16_avx2(lhs, rhs, len);
    }
#endif
#if HAS_NEON
    if ((len & 7u) == 0) {
        return dot_int16_neon(lhs, rhs, len);
    }
#endif
    return dot_int16_scalar(lhs, rhs, len);
}

static inline int32_t sum_int16_scalar(const int16_t *values, size_t len) {
    int32_t sum = 0;
    for (size_t i = 0; i < len; ++i) {
        sum += (int32_t)values[i];
    }
    return sum;
}

static inline void dequant_q4_0_to_i16(const marmot_q4_0_block_t *block, int16_t *dst) {
    const size_t half = MARMOT_QUANT_BLOCK_SIZE / 2;
    for (size_t i = 0; i < MARMOT_Q4_PACKED_BYTES; ++i) {
        const uint8_t packed = block->qs[i];
        dst[i] = (int16_t)((packed & 0x0f) - 8);
        dst[i + half] = (int16_t)(((packed >> 4) & 0x0f) - 8);
    }
}

static inline void dequant_q4_1_to_i16(const marmot_q4_1_block_t *block, int16_t *dst) {
    const size_t half = MARMOT_QUANT_BLOCK_SIZE / 2;
    for (size_t i = 0; i < MARMOT_Q4_PACKED_BYTES; ++i) {
        const uint8_t packed = block->qs[i];
        dst[i] = (int16_t)(packed & 0x0f);
        dst[i + half] = (int16_t)((packed >> 4) & 0x0f);
    }
}

static inline void dequant_q5_0_to_i16(const marmot_q5_0_block_t *block, int16_t *dst) {
    uint32_t qh;
    memcpy(&qh, block->qh, sizeof(qh));
    const size_t half = MARMOT_QUANT_BLOCK_SIZE / 2;
    for (size_t j = 0; j < half; ++j) {
        const uint8_t packed = block->qs[j];
        uint8_t lo = packed & 0x0f;
        uint8_t hi = packed >> 4;
        lo |= (uint8_t)(((qh >> (j + 0)) & 0x1u) << 4);
        hi |= (uint8_t)(((qh >> (j + half)) & 0x1u) << 4);
        dst[j] = (int16_t)((int32_t)lo - 16);
        dst[j + half] = (int16_t)((int32_t)hi - 16);
    }
}

static inline void dequant_q5_1_to_i16(const marmot_q5_1_block_t *block, int16_t *dst) {
    uint32_t qh;
    memcpy(&qh, block->qh, sizeof(qh));
    const size_t half = MARMOT_QUANT_BLOCK_SIZE / 2;
    for (size_t j = 0; j < half; ++j) {
        const uint8_t packed = block->qs[j];
        uint8_t lo = packed & 0x0f;
        uint8_t hi = packed >> 4;
        lo |= (uint8_t)(((qh >> (j + 0)) & 0x1u) << 4);
        hi |= (uint8_t)(((qh >> (j + half)) & 0x1u) << 4);
        dst[j] = (int16_t)lo;
        dst[j + half] = (int16_t)hi;
    }
}

static inline void q8_block_to_i16(const marmot_q8_0_block_t *block, int16_t *dst) {
    for (size_t i = 0; i < MARMOT_QUANT_BLOCK_SIZE; ++i) {
        dst[i] = (int16_t)block->qs[i];
    }
}

float cpu_vec_dot_q4_0_q8_0(
    const marmot_q4_0_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    int16_t w16[MARMOT_QUANT_BLOCK_SIZE];
    int16_t a16[MARMOT_QUANT_BLOCK_SIZE];

    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q4_0_block_t *w_block = weights + block_index;
        const marmot_q8_0_block_t *a_block = activations + block_index;

        dequant_q4_0_to_i16(w_block, w16);
        q8_block_to_i16(a_block, a16);

        const float scale_w = (float)marmot_float16_to_native(w_block->scale);
        const float scale_a = (float)marmot_float16_to_native(a_block->scale);
        const float block_scale = scale_w * scale_a;

        int32_t block_sum = dot_int16(w16, a16, MARMOT_QUANT_BLOCK_SIZE);
        total += (float)block_sum * block_scale;
    }

    return total;
}

float cpu_vec_dot_q4_1_q8_0(
    const marmot_q4_1_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    int16_t q16[MARMOT_QUANT_BLOCK_SIZE];
    int16_t a16[MARMOT_QUANT_BLOCK_SIZE];

    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q4_1_block_t *w_block = weights + block_index;
        const marmot_q8_0_block_t *a_block = activations + block_index;

        dequant_q4_1_to_i16(w_block, q16);
        q8_block_to_i16(a_block, a16);

        const float scale_w = (float)marmot_float16_to_native(w_block->scale);
        const float min_w = (float)marmot_float16_to_native(w_block->min);
        const float scale_a = (float)marmot_float16_to_native(a_block->scale);

        const float block_scale = scale_w * scale_a;
        const float bias_scale = min_w * scale_a;

        int32_t sum_q = dot_int16(q16, a16, MARMOT_QUANT_BLOCK_SIZE);
        int32_t sum_a = sum_int16_scalar(a16, MARMOT_QUANT_BLOCK_SIZE);

        total += block_scale * (float)sum_q + bias_scale * (float)sum_a;
    }

    return total;
}

float cpu_vec_dot_q4_0_f16(
    const marmot_q4_0_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,
    size_t K
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    int16_t w16[MARMOT_QUANT_BLOCK_SIZE];

    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q4_0_block_t *w_block = weights + block_index;
        dequant_q4_0_to_i16(w_block, w16);

        const float scale_w = (float)marmot_float16_to_native(w_block->scale);
        const marmot_float16_t *act_block = activations + block_index * MARMOT_QUANT_BLOCK_SIZE * stride_k;
        const size_t block_start = block_index * MARMOT_QUANT_BLOCK_SIZE;
        size_t block_end = block_start + MARMOT_QUANT_BLOCK_SIZE;
        if (block_end > K)
            block_end = K;
        const size_t block_len = block_end - block_start;

        for (size_t i = 0; i < block_len; ++i) {
            const marmot_float16_t *a_elem = act_block + i * stride_k;
            const float av = (float)marmot_float16_to_native(*a_elem);
            total += ((float)w16[i]) * scale_w * av;
        }
    }

    return total;
}

float cpu_vec_dot_q4_1_f16(
    const marmot_q4_1_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,
    size_t K
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    int16_t q16[MARMOT_QUANT_BLOCK_SIZE];

    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q4_1_block_t *w_block = weights + block_index;
        dequant_q4_1_to_i16(w_block, q16);

        const float scale_w = (float)marmot_float16_to_native(w_block->scale);
        const float min_w = (float)marmot_float16_to_native(w_block->min);
        const marmot_float16_t *act_block = activations + block_index * MARMOT_QUANT_BLOCK_SIZE * stride_k;
        const size_t block_start = block_index * MARMOT_QUANT_BLOCK_SIZE;
        size_t block_end = block_start + MARMOT_QUANT_BLOCK_SIZE;
        if (block_end > K)
            block_end = K;
        const size_t block_len = block_end - block_start;

        for (size_t i = 0; i < block_len; ++i) {
            const marmot_float16_t *a_elem = act_block + i * stride_k;
            const float av = (float)marmot_float16_to_native(*a_elem);
            const float w = scale_w * (float)q16[i] + min_w;
            total += w * av;
        }
    }

    return total;
}

float cpu_vec_dot_q5_0_q8_0(
    const marmot_q5_0_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    int16_t w16[MARMOT_QUANT_BLOCK_SIZE];
    int16_t a16[MARMOT_QUANT_BLOCK_SIZE];

    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q5_0_block_t *w_block = weights + block_index;
        const marmot_q8_0_block_t *a_block = activations + block_index;

        dequant_q5_0_to_i16(w_block, w16);
        q8_block_to_i16(a_block, a16);

        const float scale_w = (float)marmot_float16_to_native(w_block->scale);
        const float scale_a = (float)marmot_float16_to_native(a_block->scale);
        const float block_scale = scale_w * scale_a;

        int32_t block_sum = dot_int16(w16, a16, MARMOT_QUANT_BLOCK_SIZE);
        total += (float)block_sum * block_scale;
    }

    return total;
}

float cpu_vec_dot_q5_1_q8_0(
    const marmot_q5_1_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    int16_t q16[MARMOT_QUANT_BLOCK_SIZE];
    int16_t a16[MARMOT_QUANT_BLOCK_SIZE];

    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q5_1_block_t *w_block = weights + block_index;
        const marmot_q8_0_block_t *a_block = activations + block_index;

        dequant_q5_1_to_i16(w_block, q16);
        q8_block_to_i16(a_block, a16);

        const float scale_w = (float)marmot_float16_to_native(w_block->scale);
        const float min_w = (float)marmot_float16_to_native(w_block->min);
        const float scale_a = (float)marmot_float16_to_native(a_block->scale);

        const float block_scale = scale_w * scale_a;
        const float bias_scale = min_w * scale_a;

        int32_t sum_q = dot_int16(q16, a16, MARMOT_QUANT_BLOCK_SIZE);
        int32_t sum_a = sum_int16_scalar(a16, MARMOT_QUANT_BLOCK_SIZE);

        total += block_scale * (float)sum_q + bias_scale * (float)sum_a;
    }

    return total;
}

float cpu_vec_dot_q5_0_f16(
    const marmot_q5_0_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,
    size_t K
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    int16_t w16[MARMOT_QUANT_BLOCK_SIZE];

    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q5_0_block_t *w_block = weights + block_index;
        dequant_q5_0_to_i16(w_block, w16);

        const float scale_w = (float)marmot_float16_to_native(w_block->scale);
        const marmot_float16_t *act_block = activations + block_index * MARMOT_QUANT_BLOCK_SIZE * stride_k;
        const size_t block_start = block_index * MARMOT_QUANT_BLOCK_SIZE;
        size_t block_end = block_start + MARMOT_QUANT_BLOCK_SIZE;
        if (block_end > K)
            block_end = K;
        const size_t block_len = block_end - block_start;

        for (size_t i = 0; i < block_len; ++i) {
            const marmot_float16_t *a_elem = act_block + i * stride_k;
            const float av = (float)marmot_float16_to_native(*a_elem);
            total += ((float)w16[i]) * scale_w * av;
        }
    }

    return total;
}

float cpu_vec_dot_q5_1_f16(
    const marmot_q5_1_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,
    size_t K
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    int16_t q16[MARMOT_QUANT_BLOCK_SIZE];

    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q5_1_block_t *w_block = weights + block_index;
        dequant_q5_1_to_i16(w_block, q16);

        const float scale_w = (float)marmot_float16_to_native(w_block->scale);
        const float min_w = (float)marmot_float16_to_native(w_block->min);
        const marmot_float16_t *act_block = activations + block_index * MARMOT_QUANT_BLOCK_SIZE * stride_k;
        const size_t block_start = block_index * MARMOT_QUANT_BLOCK_SIZE;
        size_t block_end = block_start + MARMOT_QUANT_BLOCK_SIZE;
        if (block_end > K)
            block_end = K;
        const size_t block_len = block_end - block_start;

        for (size_t i = 0; i < block_len; ++i) {
            const marmot_float16_t *a_elem = act_block + i * stride_k;
            const float av = (float)marmot_float16_to_native(*a_elem);
            const float w = scale_w * (float)q16[i] + min_w;
            total += w * av;
        }
    }

    return total;
}

float cpu_vec_dot_q8_0_f16(
    const marmot_q8_0_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,
    size_t K
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    int16_t w16[MARMOT_QUANT_BLOCK_SIZE];

    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q8_0_block_t *w_block = weights + block_index;
        q8_block_to_i16(w_block, w16);

        const float scale_w = (float)marmot_float16_to_native(w_block->scale);
        const marmot_float16_t *act_block = activations + block_index * MARMOT_QUANT_BLOCK_SIZE * stride_k;
        const size_t block_start = block_index * MARMOT_QUANT_BLOCK_SIZE;
        size_t block_end = block_start + MARMOT_QUANT_BLOCK_SIZE;
        if (block_end > K)
            block_end = K;
        const size_t block_len = block_end - block_start;

        for (size_t i = 0; i < block_len; ++i) {
            const marmot_float16_t *a_elem = act_block + i * stride_k;
            const float av = (float)marmot_float16_to_native(*a_elem);
            total += ((float)w16[i]) * scale_w * av;
        }
    }

    return total;
}

float cpu_vec_dot_q8_0_q8_0(
    const marmot_q8_0_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    int16_t w16[MARMOT_QUANT_BLOCK_SIZE];
    int16_t a16[MARMOT_QUANT_BLOCK_SIZE];

    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q8_0_block_t *w_block = weights + block_index;
        const marmot_q8_0_block_t *a_block = activations + block_index;

        q8_block_to_i16(w_block, w16);
        q8_block_to_i16(a_block, a16);

        const float scale_w = (float)marmot_float16_to_native(w_block->scale);
        const float scale_a = (float)marmot_float16_to_native(a_block->scale);
        const float block_scale = scale_w * scale_a;

        int32_t block_sum = dot_int16(w16, a16, MARMOT_QUANT_BLOCK_SIZE);
        total += (float)block_sum * block_scale;
    }

    return total;
}

float cpu_vec_dot_q8_1_q8_0(
    const marmot_q8_1_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    int16_t w16[MARMOT_QUANT_BLOCK_SIZE];
    int16_t a16[MARMOT_QUANT_BLOCK_SIZE];

    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q8_1_block_t *w_block = weights + block_index;
        const marmot_q8_0_block_t *a_block = activations + block_index;

        for (size_t i = 0; i < MARMOT_QUANT_BLOCK_SIZE; ++i) {
            w16[i] = (int16_t)w_block->qs[i];
        }
        q8_block_to_i16(a_block, a16);

        const float scale_w = (float)marmot_float16_to_native(w_block->scale);
        const float scale_a = (float)marmot_float16_to_native(a_block->scale);
        const float block_scale = scale_w * scale_a;

        int32_t block_sum = dot_int16(w16, a16, MARMOT_QUANT_BLOCK_SIZE);
        total += (float)block_sum * block_scale;
    }

    return total;
}

float cpu_vec_dot_q8_1_f16(
    const marmot_q8_1_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,
    size_t K
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q8_1_block_t *w_block = weights + block_index;
        const float scale_w = (float)marmot_float16_to_native(w_block->scale);
        const marmot_float16_t *act_block = activations + block_index * MARMOT_QUANT_BLOCK_SIZE * stride_k;
        const size_t block_start = block_index * MARMOT_QUANT_BLOCK_SIZE;
        size_t block_end = block_start + MARMOT_QUANT_BLOCK_SIZE;
        if (block_end > K)
            block_end = K;
        const size_t block_len = block_end - block_start;

        for (size_t i = 0; i < block_len; ++i) {
            const marmot_float16_t *a_elem = act_block + i * stride_k;
            const float av = (float)marmot_float16_to_native(*a_elem);
            total += ((float)w_block->qs[i]) * scale_w * av;
        }
    }

    return total;
}

static inline void cpu_vec_dot_unpack_k4_scales(const uint8_t *packed, uint8_t *scales_out, uint8_t *mins_out) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];
    memcpy(utmp, packed, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    memcpy(scales_out, &utmp[0], 8);
    memcpy(mins_out, &utmp[2], 8);
}

static inline int32_t cpu_vec_dot_k4_sumi(const int16_t *bsums, const uint8_t *mins) {
    int32_t sumi = 0;
    for (size_t j = 0; j < MARMOT_QK_K_VALUES / 16; ++j) {
        sumi += (int32_t)bsums[j] * (int32_t)mins[j / 2];
    }
    return sumi;
}

float cpu_vec_dot_q2_k_q8_k(
    const marmot_q2_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float sum_all = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q2_k_block_t *w_block = &weights[block_index];
        const marmot_q8_k_block_t *a_block = &activations[block_index];

        const uint8_t *q2 = w_block->qs;
        const uint8_t *sc = w_block->scales;
        const int8_t *q8 = a_block->qs;

        int32_t summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += (int32_t)a_block->bsums[j] * (int32_t)(sc[j] >> 4);
        }

        const float dall = a_block->d * (float)marmot_float16_to_native(w_block->d);
        const float dmin = a_block->d * (float)marmot_float16_to_native(w_block->dmin);

        int32_t isum = 0;
        int is = 0;
        for (size_t chunk = 0; chunk < MARMOT_QK_K_VALUES / 128; ++chunk) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                int32_t d0 = sc[is++] & 0xF;
                int32_t dot0 = 0;
                for (int l = 0; l < 16; ++l) {
                    dot0 += (int32_t)q8[l] * (int32_t)((q2[l] >> shift) & 3);
                }
                isum += d0 * dot0;

                int32_t d1 = sc[is++] & 0xF;
                int32_t dot1 = 0;
                for (int l = 0; l < 16; ++l) {
                    dot1 += (int32_t)q8[l + 16] * (int32_t)((q2[l + 16] >> shift) & 3);
                }
                isum += d1 * dot1;

                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }

        sum_all += dall * (float)isum - dmin * (float)summs;
    }

    return sum_all;
}

float cpu_vec_dot_q3_k_q8_k(
    const marmot_q3_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    static const uint32_t kmask1 = 0x03030303;
    static const uint32_t kmask2 = 0x0f0f0f0f;

    float sum_all = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q3_k_block_t *w_block = &weights[block_index];
        const marmot_q8_k_block_t *a_block = &activations[block_index];

        const uint8_t *q3 = w_block->qs;
        const uint8_t *hm = w_block->hmask;
        const int8_t *q8 = a_block->qs;

        uint32_t auxs[4];
        memcpy(auxs, w_block->scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t *scales = (const int8_t *)auxs;

        int32_t block_sum = 0;
        uint8_t m = 1;
        int scale_index = 0;
        for (size_t chunk = 0; chunk < MARMOT_QK_K_VALUES / 128; ++chunk) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                int32_t dot0 = 0;
                for (int l = 0; l < 16; ++l) {
                    int8_t val = (int8_t)((q3[l] >> shift) & 3);
                    if ((hm[l] & m) == 0) {
                        val -= 4;
                    }
                    dot0 += (int32_t)val * (int32_t)q8[l];
                }
                block_sum += (int32_t)(scales[scale_index] - 32) * dot0;
                scale_index++;

                int32_t dot1 = 0;
                for (int l = 0; l < 16; ++l) {
                    int8_t val = (int8_t)((q3[l + 16] >> shift) & 3);
                    if ((hm[l + 16] & m) == 0) {
                        val -= 4;
                    }
                    dot1 += (int32_t)val * (int32_t)q8[l + 16];
                }
                block_sum += (int32_t)(scales[scale_index] - 32) * dot1;
                scale_index++;

                q8 += 32;
                shift += 2;
                m <<= 1;
            }
            q3 += 32;
        }

        const float d = a_block->d * (float)marmot_float16_to_native(w_block->d);
        sum_all += d * (float)block_sum;
    }
    return sum_all;
}

float cpu_vec_dot_q4_k_q8_k(
    const marmot_q4_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float sum_all = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q4_k_block_t *w_block = &weights[block_index];
        const marmot_q8_k_block_t *a_block = &activations[block_index];

        const int8_t *q8 = a_block->qs;

        uint8_t scales[8];
        uint8_t mins[8];
        cpu_vec_dot_unpack_k4_scales(w_block->scales, scales, mins);

        const int32_t sumi = cpu_vec_dot_k4_sumi(a_block->bsums, mins);
        int32_t block_sum = 0;

        for (int sg = 0; sg < 8; ++sg) {
            const uint8_t *q4_block = w_block->qs + (sg / 2) * 32;
            const int8_t *q8_block = q8 + sg * 32;
            const bool high = (sg & 1) != 0;
            int32_t dot = 0;

            for (int l = 0; l < 16; ++l) {
                const uint8_t byte0 = q4_block[l];
                const uint8_t byte1 = q4_block[l + 16];
                const int32_t w0 = high ? (byte0 >> 4) : (byte0 & 0x0F);
                const int32_t w1 = high ? (byte1 >> 4) : (byte1 & 0x0F);
                dot += w0 * (int32_t)q8_block[l] + w1 * (int32_t)q8_block[l + 16];
            }

            block_sum += (int32_t)scales[sg] * dot;
        }

        const float d = a_block->d * (float)marmot_float16_to_native(w_block->d);
        const float dmin = a_block->d * (float)marmot_float16_to_native(w_block->dmin);
        sum_all += d * (float)block_sum - dmin * (float)sumi;
    }
    return sum_all;
}

float cpu_vec_dot_q5_k_q8_k(
    const marmot_q5_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float sum_all = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q5_k_block_t *w_block = &weights[block_index];
        const marmot_q8_k_block_t *a_block = &activations[block_index];

        const uint8_t *qh = w_block->qh;
        const int8_t *q8 = a_block->qs;

        uint8_t scales[8];
        uint8_t mins[8];
        cpu_vec_dot_unpack_k4_scales(w_block->scales, scales, mins);

        const int32_t sumi = cpu_vec_dot_k4_sumi(a_block->bsums, mins);
        int32_t block_sum = 0;

        for (int sg = 0; sg < 8; ++sg) {
            const uint8_t *q4_block = w_block->qs + (sg / 2) * 32;
            const int8_t *q8_block = q8 + sg * 32;
            const bool high = (sg & 1) != 0;
            const uint8_t mask = (uint8_t)(1u << sg);
            int32_t dot = 0;

            for (int l = 0; l < 16; ++l) {
                const uint8_t byte0 = q4_block[l];
                const uint8_t byte1 = q4_block[l + 16];
                const int32_t hi0 = (qh[l] & mask) ? 16 : 0;
                const int32_t hi1 = (qh[l + 16] & mask) ? 16 : 0;
                const int32_t w0 = (high ? (byte0 >> 4) : (byte0 & 0x0F)) + hi0;
                const int32_t w1 = (high ? (byte1 >> 4) : (byte1 & 0x0F)) + hi1;
                dot += w0 * (int32_t)q8_block[l] + w1 * (int32_t)q8_block[l + 16];
            }

            block_sum += (int32_t)scales[sg] * dot;
        }

        const float d = a_block->d * (float)marmot_float16_to_native(w_block->d);
        const float dmin = a_block->d * (float)marmot_float16_to_native(w_block->dmin);
        sum_all += d * (float)block_sum - dmin * (float)sumi;
    }
    return sum_all;
}

float cpu_vec_dot_q6_k_q8_k(
    const marmot_q6_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float sum_all = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q6_k_block_t *w_block = &weights[block_index];
        const marmot_q8_k_block_t *a_block = &activations[block_index];

        const int8_t *q8 = a_block->qs;

        int32_t block_sum = 0;
        for (size_t sg = 0; sg < MARMOT_QK_K_VALUES / 16; ++sg) {
            const size_t group32 = sg / 2;
            const size_t half = group32 / 4;
            const size_t group_in_half = group32 & 3;
            const size_t part = sg & 1;
            const uint8_t *ql = w_block->ql + half * 64 + ((group_in_half & 1) ? 32 : 0) + (part * 16);
            const uint8_t *qh = w_block->qh + half * 32 + (part * 16);
            const int shift = (int)(group_in_half * 2);
            const bool high_nibble = group_in_half >= 2;
            const int32_t scale = (int32_t)w_block->scales[sg];
            int32_t dot = 0;

            for (int l = 0; l < 16; ++l) {
                const uint8_t ql_byte = ql[l];
                const uint8_t qh_byte = qh[l];
                const uint8_t ql_val = high_nibble ? (ql_byte >> 4) : (ql_byte & 0x0F);
                const uint8_t qh_val = (qh_byte >> shift) & 0x03;
                const int8_t q = (int8_t)((ql_val | (qh_val << 4)) - 32);
                dot += (int32_t)q * (int32_t)q8[l];
            }

            block_sum += scale * dot;
            q8 += 16;
        }

        const float d = a_block->d * (float)marmot_float16_to_native(w_block->d);
        sum_all += d * (float)block_sum;
    }

    return sum_all;
}

float cpu_vec_dot_q8_k_q8_k(
    const marmot_q8_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float sum_all = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q8_k_block_t *w_block = &weights[block_index];
        const marmot_q8_k_block_t *a_block = &activations[block_index];

        const float d = w_block->d * a_block->d;
        const int8_t *qw = w_block->qs;
        const int8_t *qa = a_block->qs;

        int32_t block_sum = 0;
        for (size_t j = 0; j < MARMOT_QK_K_VALUES; ++j) {
            block_sum += (int32_t)qw[j] * (int32_t)qa[j];
        }
        sum_all += d * (float)block_sum;
    }

    return sum_all;
}

static inline void get_scale_min_k4_f16(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

float cpu_vec_dot_q2_k_f16(
    const marmot_q2_k_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,
    size_t K
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q2_k_block_t *w_block = &weights[block_index];
        const float d = (float)marmot_float16_to_native(w_block->d);
        const float dmin = (float)marmot_float16_to_native(w_block->dmin);
        const uint8_t *q = w_block->qs;

        const marmot_float16_t *act_block = activations + block_index * MARMOT_QK_K_VALUES * stride_k;
        const size_t block_start = block_index * MARMOT_QK_K_VALUES;
        size_t block_end = block_start + MARMOT_QK_K_VALUES;
        if (block_end > K)
            block_end = K;
        const size_t block_len = block_end - block_start;

        size_t written = 0;
        for (size_t j = 0; j < MARMOT_QK_K_VALUES && written < block_len; j += 128) {
            size_t index = j / 128;
            uint8_t sc = w_block->scales[index] & 0xF;
            uint8_t m = w_block->scales[index] >> 4;
            const float dl = d * (float)sc;
            const float ml = dmin * (float)m;

            for (size_t i = 0; i < 128 && written < block_len; ++i, ++written) {
                size_t q_idx = (j + i) / 4;
                size_t q_shift = ((j + i) % 4) * 2;
                uint8_t q_val = (q[q_idx] >> q_shift) & 0x3;
                const marmot_float16_t *a_elem = act_block + written * stride_k;
                const float av = (float)marmot_float16_to_native(*a_elem);
                const float w_val = dl * (float)q_val - ml;
                total += w_val * av;
            }
        }
    }
    return total;
}

float cpu_vec_dot_q3_k_f16(
    const marmot_q3_k_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,
    size_t K
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q3_k_block_t *w_block = &weights[block_index];
        const float d_all = (float)marmot_float16_to_native(w_block->d);
        const uint8_t *q = w_block->qs;
        const uint8_t *hm = w_block->hmask;

        const marmot_float16_t *act_block = activations + block_index * MARMOT_QK_K_VALUES * stride_k;
        const size_t block_start = block_index * MARMOT_QK_K_VALUES;
        size_t block_end = block_start + MARMOT_QK_K_VALUES;
        if (block_end > K)
            block_end = K;
        const size_t block_len = block_end - block_start;

        uint32_t aux[4];
        memcpy(aux, w_block->scales, 12);
        const uint32_t kmask1 = 0x03030303;
        const uint32_t kmask2 = 0x0f0f0f0f;
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t *scales = (const int8_t *)aux;

        size_t written = 0;
        int is = 0;
        uint8_t m = 1;
        for (size_t n = 0; n < MARMOT_QK_K_VALUES && written < block_len; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (float)(scales[is++] - 32);
                for (int l = 0; l < 16 && written < block_len; ++l, ++written) {
                    const marmot_float16_t *a_elem = act_block + written * stride_k;
                    const float av = (float)marmot_float16_to_native(*a_elem);
                    int8_t q_val = ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                    total += dl * (float)q_val * av;
                }

                dl = d_all * (float)(scales[is++] - 32);
                for (int l = 0; l < 16 && written < block_len; ++l, ++written) {
                    const marmot_float16_t *a_elem = act_block + written * stride_k;
                    const float av = (float)marmot_float16_to_native(*a_elem);
                    int8_t q_val = ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                    total += dl * (float)q_val * av;
                }
                shift += 2;
            }
            q += 32;
            m <<= 1;
        }
    }
    return total;
}

float cpu_vec_dot_q4_k_f16(
    const marmot_q4_k_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,
    size_t K
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q4_k_block_t *w_block = &weights[block_index];
        const uint8_t *q = w_block->qs;
        const float d = (float)marmot_float16_to_native(w_block->d);
        const float dmin = (float)marmot_float16_to_native(w_block->dmin);

        const marmot_float16_t *act_block = activations + block_index * MARMOT_QK_K_VALUES * stride_k;
        const size_t block_start = block_index * MARMOT_QK_K_VALUES;
        size_t block_end = block_start + MARMOT_QK_K_VALUES;
        if (block_end > K)
            block_end = K;
        const size_t block_len = block_end - block_start;

        size_t written = 0;
        int is = 0;
        for (size_t j = 0; j < MARMOT_QK_K_VALUES && written < block_len; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4_f16(is + 0, w_block->scales, &sc, &m);
            const float d1 = d * (float)sc;
            const float m1 = dmin * (float)m;
            get_scale_min_k4_f16(is + 1, w_block->scales, &sc, &m);
            const float d2 = d * (float)sc;
            const float m2 = dmin * (float)m;

            for (int l = 0; l < 32 && written < block_len; ++l, ++written) {
                const marmot_float16_t *a_elem = act_block + written * stride_k;
                const float av = (float)marmot_float16_to_native(*a_elem);
                const float w_val = d1 * (float)(q[l] & 0xF) - m1;
                total += w_val * av;
            }
            for (int l = 0; l < 32 && written < block_len; ++l, ++written) {
                const marmot_float16_t *a_elem = act_block + written * stride_k;
                const float av = (float)marmot_float16_to_native(*a_elem);
                const float w_val = d2 * (float)(q[l] >> 4) - m2;
                total += w_val * av;
            }
            q += 32;
            is += 2;
        }
    }
    return total;
}

float cpu_vec_dot_q5_k_f16(
    const marmot_q5_k_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,
    size_t K
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q5_k_block_t *w_block = &weights[block_index];
        const uint8_t *ql = w_block->qs;
        const uint8_t *qh = w_block->qh;
        const float d = (float)marmot_float16_to_native(w_block->d);
        const float dmin = (float)marmot_float16_to_native(w_block->dmin);

        const marmot_float16_t *act_block = activations + block_index * MARMOT_QK_K_VALUES * stride_k;
        const size_t block_start = block_index * MARMOT_QK_K_VALUES;
        size_t block_end = block_start + MARMOT_QK_K_VALUES;
        if (block_end > K)
            block_end = K;
        const size_t block_len = block_end - block_start;

        size_t written = 0;
        int is = 0;
        for (size_t j = 0; j < MARMOT_QK_K_VALUES && written < block_len; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4_f16(is + 0, w_block->scales, &sc, &m);
            const float d1 = d * (float)sc;
            const float m1 = dmin * (float)m;
            get_scale_min_k4_f16(is + 1, w_block->scales, &sc, &m);
            const float d2 = d * (float)sc;
            const float m2 = dmin * (float)m;

            for (int l = 0; l < 32 && written < block_len; ++l, ++written) {
                uint8_t h = qh[j / 8 + l] >> (j % 8) & 1;
                uint8_t q_val = (ql[l] & 0xF) | (h << 4);
                const marmot_float16_t *a_elem = act_block + written * stride_k;
                const float av = (float)marmot_float16_to_native(*a_elem);
                const float w_val = d1 * (float)q_val - m1;
                total += w_val * av;
            }
            for (int l = 0; l < 32 && written < block_len; ++l, ++written) {
                uint8_t h = qh[j / 8 + l + 32] >> (j % 8) & 1;
                uint8_t q_val = (ql[l] >> 4) | (h << 4);
                const marmot_float16_t *a_elem = act_block + written * stride_k;
                const float av = (float)marmot_float16_to_native(*a_elem);
                const float w_val = d2 * (float)q_val - m2;
                total += w_val * av;
            }
            ql += 32;
            is += 2;
        }
    }
    return total;
}

float cpu_vec_dot_q6_k_f16(
    const marmot_q6_k_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,
    size_t K
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q6_k_block_t *w_block = &weights[block_index];
        const uint8_t *ql = w_block->ql;
        const uint8_t *qh = w_block->qh;
        const int8_t *sc = w_block->scales;
        const float d = (float)marmot_float16_to_native(w_block->d);

        const marmot_float16_t *act_block = activations + block_index * MARMOT_QK_K_VALUES * stride_k;
        const size_t block_start = block_index * MARMOT_QK_K_VALUES;
        size_t block_end = block_start + MARMOT_QK_K_VALUES;
        if (block_end > K)
            block_end = K;
        const size_t block_len = block_end - block_start;

        size_t written = 0;
        for (size_t j = 0; j < MARMOT_QK_K_VALUES && written < block_len; j += 128) {
            size_t sc_idx = j / 128;
            float scale = d * (float)sc[sc_idx];

            for (size_t i = 0; i < 128 && written < block_len; ++i, ++written) {
                size_t idx = j + i;
                size_t ql_idx = idx / 2;
                size_t qh_idx = idx / 4;
                uint8_t ql_val = (i % 2 == 0) ? (ql[ql_idx] & 0xF) : (ql[ql_idx] >> 4);
                uint8_t qh_shift = (idx % 4) * 2;
                uint8_t qh_val = (qh[qh_idx] >> qh_shift) & 0x3;
                uint8_t combined = ql_val | (qh_val << 4);
                int8_t signed_q = (int8_t)combined - 32;

                const marmot_float16_t *a_elem = act_block + written * stride_k;
                const float av = (float)marmot_float16_to_native(*a_elem);
                const float w_val = scale * (float)signed_q;
                total += w_val * av;
            }
        }
    }
    return total;
}

float cpu_vec_dot_q8_k_f16(
    const marmot_q8_k_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,
    size_t K
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q8_k_block_t *w_block = &weights[block_index];
        const float d = w_block->d;

        const marmot_float16_t *act_block = activations + block_index * MARMOT_QK_K_VALUES * stride_k;
        const size_t block_start = block_index * MARMOT_QK_K_VALUES;
        size_t block_end = block_start + MARMOT_QK_K_VALUES;
        if (block_end > K)
            block_end = K;
        const size_t block_len = block_end - block_start;

        for (size_t i = 0; i < block_len; ++i) {
            const marmot_float16_t *a_elem = act_block + i * stride_k;
            const float av = (float)marmot_float16_to_native(*a_elem);
            const float w_val = d * (float)w_block->qs[i];
            total += w_val * av;
        }
    }
    return total;
}

#undef cpu_vec_dot_q8_k_f16
#undef cpu_vec_dot_q6_k_f16
#undef cpu_vec_dot_q5_k_f16
#undef cpu_vec_dot_q4_k_f16
#undef cpu_vec_dot_q3_k_f16
#undef cpu_vec_dot_q2_k_f16
#undef cpu_vec_dot_q8_k_q8_k
#undef cpu_vec_dot_q6_k_q8_k
#undef cpu_vec_dot_q5_k_q8_k
#undef cpu_vec_dot_q4_k_q8_k
#undef cpu_vec_dot_q3_k_q8_k
#undef cpu_vec_dot_q2_k_q8_k
#undef cpu_vec_dot_q8_1_f16
#undef cpu_vec_dot_q8_1_q8_0
#undef cpu_vec_dot_q8_0_q8_0
#undef cpu_vec_dot_q8_0_f16
#undef cpu_vec_dot_q5_1_f16
#undef cpu_vec_dot_q5_0_f16
#undef cpu_vec_dot_q5_1_q8_0
#undef cpu_vec_dot_q5_0_q8_0
#undef cpu_vec_dot_q4_1_f16
#undef cpu_vec_dot_q4_1_q8_0
#undef cpu_vec_dot_q4_0_f16
#undef cpu_vec_dot_q4_0_q8_0
#undef FN
