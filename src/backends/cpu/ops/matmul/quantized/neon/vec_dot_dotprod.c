#include "ops/matmul/quantized/internal/vec_dot.h"

#if MARMOT_ENABLE_NEON && defined(__aarch64__)

#if defined(__ARM_FEATURE_DOTPROD)

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

static inline uint8x16_t cpu_vec_dot_shift_mask_2bit(uint8x16_t values, int shift) {
    const uint8x16_t mask = vdupq_n_u8(0x03);
    switch (shift) {
    case 0:
        return vandq_u8(values, mask);
    case 2:
        return vandq_u8(vshrq_n_u8(values, 2), mask);
    case 4:
        return vandq_u8(vshrq_n_u8(values, 4), mask);
    default:
        return vandq_u8(vshrq_n_u8(values, 6), mask);
    }
}

static inline int8x16_t cpu_vec_dot_q3_k_decode(uint8x16_t q3, uint8x16_t hm, int shift, uint8_t mask) {
    const uint8x16_t mask_vec = vdupq_n_u8(mask);
    const uint8x16_t sub_mask = vdupq_n_u8(4);
    uint8x16_t low;
    switch (shift) {
    case 0:
        low = vandq_u8(q3, vdupq_n_u8(0x03));
        break;
    case 2:
        low = vandq_u8(vshrq_n_u8(q3, 2), vdupq_n_u8(0x03));
        break;
    case 4:
        low = vandq_u8(vshrq_n_u8(q3, 4), vdupq_n_u8(0x03));
        break;
    default:
        low = vandq_u8(vshrq_n_u8(q3, 6), vdupq_n_u8(0x03));
        break;
    }
    const uint8x16_t has_bit = vtstq_u8(hm, mask_vec);
    const uint8x16_t sub = vandq_u8(vmvnq_u8(has_bit), sub_mask);
    return vsubq_s8(vreinterpretq_s8_u8(low), vreinterpretq_s8_u8(sub));
}

static inline uint8x16_t cpu_vec_dot_q6_k_qh_bits(uint8x16_t qh, int shift) {
    const uint8x16_t mask = vdupq_n_u8(0x03);
    switch (shift) {
    case 0:
        return vandq_u8(qh, mask);
    case 2:
        return vandq_u8(vshrq_n_u8(qh, 2), mask);
    case 4:
        return vandq_u8(vshrq_n_u8(qh, 4), mask);
    default:
        return vandq_u8(vshrq_n_u8(qh, 6), mask);
    }
}

static inline int8x16_t cpu_vec_dot_q6_k_decode_16(const uint8_t *ql, const uint8_t *qh, int shift, bool high_nibble) {
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    uint8x16_t ql_bytes = vld1q_u8(ql);
    uint8x16_t qh_bytes = vld1q_u8(qh);
    uint8x16_t ql_vals = high_nibble ? vshrq_n_u8(ql_bytes, 4) : vandq_u8(ql_bytes, mask);
    uint8x16_t qh_vals = cpu_vec_dot_q6_k_qh_bits(qh_bytes, shift);
    uint8x16_t merged = vorrq_u8(ql_vals, vshlq_n_u8(qh_vals, 4));
    int8x16_t signed_vals = vreinterpretq_s8_u8(merged);
    return vsubq_s8(signed_vals, vdupq_n_s8(32));
}

static inline int32_t cpu_vec_dot_q6_k_block_dotprod(const marmot_q6_k_block_t *w_block, const int8_t *q8) {
    int32x4_t block_acc = vdupq_n_s32(0);
    const int8_t *a_ptr = q8;
    for (size_t sg = 0; sg < MARMOT_QK_K_VALUES / 16; ++sg) {
        const size_t group32 = sg / 2;
        const size_t half = group32 / 4;
        const size_t group_in_half = group32 & 3;
        const size_t part = sg & 1;
        const uint8_t *ql = w_block->ql + half * 64 + ((group_in_half & 1) ? 32 : 0) + (part * 16);
        const uint8_t *qh = w_block->qh + half * 32 + (part * 16);
        const int shift = (int)(group_in_half * 2);
        const bool high_nibble = group_in_half >= 2;
        const int8_t scale = w_block->scales[sg];
        int32x4_t acc = vdupq_n_s32(0);
        int8x16_t wv = cpu_vec_dot_q6_k_decode_16(ql, qh, shift, high_nibble);
        int8x16_t av = vld1q_s8(a_ptr);
        acc = vdotq_s32(acc, wv, av);
        block_acc = vaddq_s32(block_acc, vmulq_n_s32(acc, (int32_t)scale));
        a_ptr += 16;
    }
    return vaddvq_s32(block_acc);
}

float cpu_vec_dot_q8_0_q8_0_neon_dotprod(
    const marmot_q8_0_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q8_0_block_t *w_block = weights + block_index;
        const marmot_q8_0_block_t *a_block = activations + block_index;

        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);

        int8x16_t w0 = vld1q_s8(w_block->qs);
        int8x16_t w1 = vld1q_s8(w_block->qs + 16);
        int8x16_t a0 = vld1q_s8(a_block->qs);
        int8x16_t a1 = vld1q_s8(a_block->qs + 16);

        acc0 = vdotq_s32(acc0, w0, a0);
        acc1 = vdotq_s32(acc1, w1, a1);

        const float block_scale =
            (float)marmot_float16_to_native(w_block->scale) * (float)marmot_float16_to_native(a_block->scale);
        const int32_t block_sum = vaddvq_s32(vaddq_s32(acc0, acc1));
        total += (float)block_sum * block_scale;
    }

    return total;
}

float cpu_vec_dot_q8_1_q8_0_neon_dotprod(
    const marmot_q8_1_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q8_1_block_t *w_block = weights + block_index;
        const marmot_q8_0_block_t *a_block = activations + block_index;

        int32x4_t acc = vdupq_n_s32(0);

        int8x16_t w0 = vld1q_s8(w_block->qs);
        int8x16_t w1 = vld1q_s8(w_block->qs + 16);
        int8x16_t a0 = vld1q_s8(a_block->qs);
        int8x16_t a1 = vld1q_s8(a_block->qs + 16);

        acc = vdotq_s32(acc, w0, a0);
        acc = vdotq_s32(acc, w1, a1);

        const float block_scale =
            (float)marmot_float16_to_native(w_block->scale) * (float)marmot_float16_to_native(a_block->scale);
        const int32_t block_sum = vaddvq_s32(acc);
        total += (float)block_sum * block_scale;
    }

    return total;
}

float cpu_vec_dot_q4_0_q8_0_neon_dotprod(
    const marmot_q4_0_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const int8x16_t offset = vdupq_n_s8(8);
    float total = 0.0f;

    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q4_0_block_t *w_block = weights + block_index;
        const marmot_q8_0_block_t *a_block = activations + block_index;

        const uint8x16_t packed = vld1q_u8(w_block->qs);
        const int8x16_t q4_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(packed, mask)), offset);
        const int8x16_t q4_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), offset);

        int8x16_t a0 = vld1q_s8(a_block->qs);
        int8x16_t a1 = vld1q_s8(a_block->qs + 16);

        int32x4_t acc = vdotq_s32(vdupq_n_s32(0), q4_lo, a0);
        acc = vdotq_s32(acc, q4_hi, a1);

        const float block_scale =
            (float)marmot_float16_to_native(w_block->scale) * (float)marmot_float16_to_native(a_block->scale);
        const int32_t block_sum = vaddvq_s32(acc);
        total += (float)block_sum * block_scale;
    }

    return total;
}

float cpu_vec_dot_q4_1_q8_0_neon_dotprod(
    const marmot_q4_1_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    const uint8x16_t mask = vdupq_n_u8(0x0F);
    float total = 0.0f;

    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q4_1_block_t *w_block = weights + block_index;
        const marmot_q8_0_block_t *a_block = activations + block_index;

        const uint8x16_t packed = vld1q_u8(w_block->qs);
        const int8x16_t q4_lo = vreinterpretq_s8_u8(vandq_u8(packed, mask));
        const int8x16_t q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));

        int8x16_t a0 = vld1q_s8(a_block->qs);
        int8x16_t a1 = vld1q_s8(a_block->qs + 16);

        int32x4_t dot_acc = vdotq_s32(vdupq_n_s32(0), q4_lo, a0);
        dot_acc = vdotq_s32(dot_acc, q4_hi, a1);
        const int32_t dot_sum = vaddvq_s32(dot_acc);

        int16x8_t a0_16 = vpaddlq_s8(a0);
        int16x8_t a1_16 = vpaddlq_s8(a1);
        int32x4_t a_sum_32 = vpaddlq_s16(vaddq_s16(a0_16, a1_16));
        const int32_t a_sum = vaddvq_s32(a_sum_32);

        const float scale_w = (float)marmot_float16_to_native(w_block->scale);
        const float min_w = (float)marmot_float16_to_native(w_block->min);
        const float scale_a = (float)marmot_float16_to_native(a_block->scale);

        total += scale_a * (scale_w * (float)dot_sum + min_w * (float)a_sum);
    }

    return total;
}

float cpu_vec_dot_q5_0_q8_0_neon_dotprod(
    const marmot_q5_0_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const int8x16_t offset = vdupq_n_s8(16);
    const uint8x8_t one = vdup_n_u8(1);
    static const int8_t shift_values[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const int8x8_t shifts = vneg_s8(vld1_s8(shift_values));

    float total = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q5_0_block_t *w_block = weights + block_index;
        const marmot_q8_0_block_t *a_block = activations + block_index;

        uint32_t qh_bits;
        memcpy(&qh_bits, w_block->qh, sizeof(qh_bits));

        const uint8x16_t packed = vld1q_u8(w_block->qs);
        const uint8x16_t q4_lo = vandq_u8(packed, mask);
        const uint8x16_t q4_hi = vshrq_n_u8(packed, 4);

        const uint8x8_t qh0 = vdup_n_u8((uint8_t)(qh_bits >> 0));
        const uint8x8_t qh1 = vdup_n_u8((uint8_t)(qh_bits >> 8));
        const uint8x8_t qh2 = vdup_n_u8((uint8_t)(qh_bits >> 16));
        const uint8x8_t qh3 = vdup_n_u8((uint8_t)(qh_bits >> 24));

        const uint8x8_t bits0 = vand_u8(vshl_u8(qh0, shifts), one);
        const uint8x8_t bits1 = vand_u8(vshl_u8(qh1, shifts), one);
        const uint8x8_t bits2 = vand_u8(vshl_u8(qh2, shifts), one);
        const uint8x8_t bits3 = vand_u8(vshl_u8(qh3, shifts), one);

        const uint8x16_t high_lo = vcombine_u8(bits0, bits1);
        const uint8x16_t high_hi = vcombine_u8(bits2, bits3);

        const uint8x16_t q5_lo_u8 = vorrq_u8(q4_lo, vshlq_n_u8(high_lo, 4));
        const uint8x16_t q5_hi_u8 = vorrq_u8(q4_hi, vshlq_n_u8(high_hi, 4));

        const int8x16_t w0 = vsubq_s8(vreinterpretq_s8_u8(q5_lo_u8), offset);
        const int8x16_t w1 = vsubq_s8(vreinterpretq_s8_u8(q5_hi_u8), offset);

        int8x16_t a0 = vld1q_s8(a_block->qs);
        int8x16_t a1 = vld1q_s8(a_block->qs + 16);

        int32x4_t acc = vdotq_s32(vdupq_n_s32(0), w0, a0);
        acc = vdotq_s32(acc, w1, a1);

        const float block_scale =
            (float)marmot_float16_to_native(w_block->scale) * (float)marmot_float16_to_native(a_block->scale);
        const int32_t block_sum = vaddvq_s32(acc);
        total += (float)block_sum * block_scale;
    }

    return total;
}

float cpu_vec_dot_q5_1_q8_0_neon_dotprod(
    const marmot_q5_1_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const uint8x8_t one = vdup_n_u8(1);
    static const int8_t shift_values[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const int8x8_t shifts = vneg_s8(vld1_s8(shift_values));

    float total = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q5_1_block_t *w_block = weights + block_index;
        const marmot_q8_0_block_t *a_block = activations + block_index;

        uint32_t qh_bits;
        memcpy(&qh_bits, w_block->qh, sizeof(qh_bits));

        const uint8x16_t packed = vld1q_u8(w_block->qs);
        const uint8x16_t q4_lo = vandq_u8(packed, mask);
        const uint8x16_t q4_hi = vshrq_n_u8(packed, 4);

        const uint8x8_t qh0 = vdup_n_u8((uint8_t)(qh_bits >> 0));
        const uint8x8_t qh1 = vdup_n_u8((uint8_t)(qh_bits >> 8));
        const uint8x8_t qh2 = vdup_n_u8((uint8_t)(qh_bits >> 16));
        const uint8x8_t qh3 = vdup_n_u8((uint8_t)(qh_bits >> 24));

        const uint8x8_t bits0 = vand_u8(vshl_u8(qh0, shifts), one);
        const uint8x8_t bits1 = vand_u8(vshl_u8(qh1, shifts), one);
        const uint8x8_t bits2 = vand_u8(vshl_u8(qh2, shifts), one);
        const uint8x8_t bits3 = vand_u8(vshl_u8(qh3, shifts), one);

        const uint8x16_t high_lo = vcombine_u8(bits0, bits1);
        const uint8x16_t high_hi = vcombine_u8(bits2, bits3);

        const int8x16_t w0 = vreinterpretq_s8_u8(vorrq_u8(q4_lo, vshlq_n_u8(high_lo, 4)));
        const int8x16_t w1 = vreinterpretq_s8_u8(vorrq_u8(q4_hi, vshlq_n_u8(high_hi, 4)));

        int8x16_t a0 = vld1q_s8(a_block->qs);
        int8x16_t a1 = vld1q_s8(a_block->qs + 16);

        int32x4_t dot_acc = vdotq_s32(vdupq_n_s32(0), w0, a0);
        dot_acc = vdotq_s32(dot_acc, w1, a1);
        const int32_t dot_sum = vaddvq_s32(dot_acc);

        int16x8_t a0_16 = vpaddlq_s8(a0);
        int16x8_t a1_16 = vpaddlq_s8(a1);
        int32x4_t a_sum_32 = vpaddlq_s16(vaddq_s16(a0_16, a1_16));
        const int32_t a_sum = vaddvq_s32(a_sum_32);

        const float scale_w = (float)marmot_float16_to_native(w_block->scale);
        const float min_w = (float)marmot_float16_to_native(w_block->min);
        const float scale_a = (float)marmot_float16_to_native(a_block->scale);

        total += scale_a * (scale_w * (float)dot_sum + min_w * (float)a_sum);
    }

    return total;
}

float cpu_vec_dot_q8_k_q8_k_neon_dotprod(
    const marmot_q8_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float sum_all = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q8_k_block_t *w_block = weights + block_index;
        const marmot_q8_k_block_t *a_block = activations + block_index;

        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);

        const int8_t *qw = w_block->qs;
        const int8_t *qa = a_block->qs;

        for (size_t j = 0; j < MARMOT_QK_K_VALUES; j += 64) {
            int8x16_t w0 = vld1q_s8(qw + j + 0);
            int8x16_t a0 = vld1q_s8(qa + j + 0);
            acc0 = vdotq_s32(acc0, w0, a0);

            int8x16_t w1 = vld1q_s8(qw + j + 16);
            int8x16_t a1 = vld1q_s8(qa + j + 16);
            acc1 = vdotq_s32(acc1, w1, a1);

            int8x16_t w2 = vld1q_s8(qw + j + 32);
            int8x16_t a2 = vld1q_s8(qa + j + 32);
            acc2 = vdotq_s32(acc2, w2, a2);

            int8x16_t w3 = vld1q_s8(qw + j + 48);
            int8x16_t a3 = vld1q_s8(qa + j + 48);
            acc3 = vdotq_s32(acc3, w3, a3);
        }

        const float d = w_block->d * a_block->d;
        const int32_t block_sum = vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3)));
        sum_all += d * (float)block_sum;
    }

    return sum_all;
}

float cpu_vec_dot_q4_k_q8_k_neon_dotprod(
    const marmot_q4_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    const uint8x16_t nibble_mask = vdupq_n_u8(0x0F);
    float total = 0.0f;

    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q4_k_block_t *w_block = &weights[block_index];
        const marmot_q8_k_block_t *a_block = &activations[block_index];
        const int8_t *q8 = a_block->qs;

        uint8_t scales[8];
        uint8_t mins[8];
        cpu_vec_dot_unpack_k4_scales(w_block->scales, scales, mins);

        const int32_t sumi = cpu_vec_dot_k4_sumi(a_block->bsums, mins);
        int32x4_t block_acc = vdupq_n_s32(0);

        for (int sg = 0; sg < 8; ++sg) {
            const uint8_t *q4_block = w_block->qs + (sg / 2) * 32;
            const int8_t *q8_block = q8 + sg * 32;
            const bool high = (sg & 1) != 0;

            uint8x16_t bytes0 = vld1q_u8(q4_block);
            uint8x16_t bytes1 = vld1q_u8(q4_block + 16);
            uint8x16_t vals0 = high ? vshrq_n_u8(bytes0, 4) : vandq_u8(bytes0, nibble_mask);
            uint8x16_t vals1 = high ? vshrq_n_u8(bytes1, 4) : vandq_u8(bytes1, nibble_mask);

            int8x16_t w0 = vreinterpretq_s8_u8(vals0);
            int8x16_t w1 = vreinterpretq_s8_u8(vals1);
            int8x16_t a0 = vld1q_s8(q8_block);
            int8x16_t a1 = vld1q_s8(q8_block + 16);

            int32x4_t acc = vdotq_s32(vdupq_n_s32(0), w0, a0);
            acc = vdotq_s32(acc, w1, a1);
            block_acc = vaddq_s32(block_acc, vmulq_n_s32(acc, (int32_t)scales[sg]));
        }

        const float d = a_block->d * (float)marmot_float16_to_native(w_block->d);
        const float dmin = a_block->d * (float)marmot_float16_to_native(w_block->dmin);
        const int32_t block_sum = vaddvq_s32(block_acc);
        total += d * (float)block_sum - dmin * (float)sumi;
    }

    return total;
}

float cpu_vec_dot_q5_k_q8_k_neon_dotprod(
    const marmot_q5_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    const uint8x16_t nibble_mask = vdupq_n_u8(0x0F);
    const uint8x16_t add_mask = vdupq_n_u8(16);
    float total = 0.0f;

    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q5_k_block_t *w_block = &weights[block_index];
        const marmot_q8_k_block_t *a_block = &activations[block_index];
        const uint8_t *qh = w_block->qh;
        const int8_t *q8 = a_block->qs;

        uint8_t scales[8];
        uint8_t mins[8];
        cpu_vec_dot_unpack_k4_scales(w_block->scales, scales, mins);

        const int32_t sumi = cpu_vec_dot_k4_sumi(a_block->bsums, mins);
        int32x4_t block_acc = vdupq_n_s32(0);

        for (int sg = 0; sg < 8; ++sg) {
            const uint8_t *q4_block = w_block->qs + (sg / 2) * 32;
            const int8_t *q8_block = q8 + sg * 32;
            const bool high = (sg & 1) != 0;
            const uint8_t mask = (uint8_t)(1u << sg);
            const uint8x16_t mask_vec = vdupq_n_u8(mask);

            uint8x16_t bytes0 = vld1q_u8(q4_block);
            uint8x16_t bytes1 = vld1q_u8(q4_block + 16);
            uint8x16_t qh0 = vld1q_u8(qh);
            uint8x16_t qh1 = vld1q_u8(qh + 16);

            uint8x16_t vals0 = high ? vshrq_n_u8(bytes0, 4) : vandq_u8(bytes0, nibble_mask);
            uint8x16_t vals1 = high ? vshrq_n_u8(bytes1, 4) : vandq_u8(bytes1, nibble_mask);
            uint8x16_t add0 = vandq_u8(vtstq_u8(qh0, mask_vec), add_mask);
            uint8x16_t add1 = vandq_u8(vtstq_u8(qh1, mask_vec), add_mask);

            int8x16_t w0 = vreinterpretq_s8_u8(vaddq_u8(vals0, add0));
            int8x16_t w1 = vreinterpretq_s8_u8(vaddq_u8(vals1, add1));
            int8x16_t a0 = vld1q_s8(q8_block);
            int8x16_t a1 = vld1q_s8(q8_block + 16);

            int32x4_t acc = vdotq_s32(vdupq_n_s32(0), w0, a0);
            acc = vdotq_s32(acc, w1, a1);
            block_acc = vaddq_s32(block_acc, vmulq_n_s32(acc, (int32_t)scales[sg]));
        }

        const float d = a_block->d * (float)marmot_float16_to_native(w_block->d);
        const float dmin = a_block->d * (float)marmot_float16_to_native(w_block->dmin);
        const int32_t block_sum = vaddvq_s32(block_acc);
        total += d * (float)block_sum - dmin * (float)sumi;
    }

    return total;
}

float cpu_vec_dot_q3_k_q8_k_neon_dotprod(
    const marmot_q3_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    static const uint32_t kmask1 = 0x03030303;
    static const uint32_t kmask2 = 0x0f0f0f0f;

    float total = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q3_k_block_t *w_block = &weights[block_index];
        const marmot_q8_k_block_t *a_block = &activations[block_index];
        const uint8_t *q3 = w_block->qs;
        const uint8_t *hm = w_block->hmask;
        const int8_t *q8 = a_block->qs;

        uint32_t auxs[4];
        memcpy(auxs, w_block->scales, 12);
        const uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t *scales = (const int8_t *)auxs;

        int32x4_t block_acc = vdupq_n_s32(0);
        uint8_t m = 1;
        int scale_index = 0;
        for (size_t chunk = 0; chunk < MARMOT_QK_K_VALUES / 128; ++chunk) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8x16_t q3_0 = vld1q_u8(q3);
                uint8x16_t q3_1 = vld1q_u8(q3 + 16);
                uint8x16_t hm0 = vld1q_u8(hm);
                uint8x16_t hm1 = vld1q_u8(hm + 16);

                int8x16_t w0 = cpu_vec_dot_q3_k_decode(q3_0, hm0, shift, m);
                int8x16_t w1 = cpu_vec_dot_q3_k_decode(q3_1, hm1, shift, m);
                int8x16_t a0 = vld1q_s8(q8);
                int8x16_t a1 = vld1q_s8(q8 + 16);

                int32x4_t acc0 = vdotq_s32(vdupq_n_s32(0), w0, a0);
                int32x4_t acc1 = vdotq_s32(vdupq_n_s32(0), w1, a1);

                const int32_t scale0 = (int32_t)scales[scale_index] - 32;
                block_acc = vaddq_s32(block_acc, vmulq_n_s32(acc0, scale0));
                scale_index++;
                const int32_t scale1 = (int32_t)scales[scale_index] - 32;
                block_acc = vaddq_s32(block_acc, vmulq_n_s32(acc1, scale1));
                scale_index++;

                shift += 2;
                q8 += 32;
                m <<= 1;
            }
            q3 += 32;
        }

        const float d = a_block->d * (float)marmot_float16_to_native(w_block->d);
        const int32_t block_sum = vaddvq_s32(block_acc);
        total += d * (float)block_sum;
    }

    return total;
}

float cpu_vec_dot_q2_k_q8_k_neon_dotprod(
    const marmot_q2_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;

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

        int32x4_t block_acc = vdupq_n_s32(0);
        int is = 0;

        for (size_t chunk = 0; chunk < MARMOT_QK_K_VALUES / 128; ++chunk) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                const int d0 = sc[is++] & 0xF;
                const int d1 = sc[is++] & 0xF;

                uint8x16_t q2_0 = vld1q_u8(q2);
                uint8x16_t q2_1 = vld1q_u8(q2 + 16);
                int8x16_t w0 = vreinterpretq_s8_u8(cpu_vec_dot_shift_mask_2bit(q2_0, shift));
                int8x16_t w1 = vreinterpretq_s8_u8(cpu_vec_dot_shift_mask_2bit(q2_1, shift));

                int8x16_t a0 = vld1q_s8(q8);
                int8x16_t a1 = vld1q_s8(q8 + 16);

                int32x4_t acc0 = vdotq_s32(vdupq_n_s32(0), w0, a0);
                int32x4_t acc1 = vdotq_s32(vdupq_n_s32(0), w1, a1);

                block_acc = vaddq_s32(block_acc, vmulq_n_s32(acc0, d0));
                block_acc = vaddq_s32(block_acc, vmulq_n_s32(acc1, d1));

                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }

        const int32_t isum = vaddvq_s32(block_acc);
        total += dall * (float)isum - dmin * (float)summs;
    }

    return total;
}

float cpu_vec_dot_q6_k_q8_k_neon_dotprod(
    const marmot_q6_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;

    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q6_k_block_t *w_block = &weights[block_index];
        const marmot_q8_k_block_t *a_block = &activations[block_index];

        const int32_t block_sum = cpu_vec_dot_q6_k_block_dotprod(w_block, a_block->qs);
        const float d = a_block->d * (float)marmot_float16_to_native(w_block->d);
        total += d * (float)block_sum;
    }

    return total;
}

#endif // __ARM_FEATURE_DOTPROD

#if defined(__ARM_FEATURE_I8MM) || defined(__ARM_FEATURE_MATMUL_INT8)
static inline int32_t cpu_vec_dot_i8mm_sum(int32x4_t acc) {
    return vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 3);
}

float cpu_vec_dot_q8_0_q8_0_neon_i8mm(
    const marmot_q8_0_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q8_0_block_t *w_block = weights + block_index;
        const marmot_q8_0_block_t *a_block = activations + block_index;

        int8x16_t w0 = vld1q_s8(w_block->qs);
        int8x16_t w1 = vld1q_s8(w_block->qs + 16);
        int8x16_t a0 = vld1q_s8(a_block->qs);
        int8x16_t a1 = vld1q_s8(a_block->qs + 16);

        int32x4_t acc = vdupq_n_s32(0);
        acc = vmmlaq_s32(acc, w0, a0);
        acc = vmmlaq_s32(acc, w1, a1);

        const float block_scale =
            (float)marmot_float16_to_native(w_block->scale) * (float)marmot_float16_to_native(a_block->scale);
        const int32_t block_sum = cpu_vec_dot_i8mm_sum(acc);
        total += (float)block_sum * block_scale;
    }

    return total;
}

float cpu_vec_dot_q8_1_q8_0_neon_i8mm(
    const marmot_q8_1_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q8_1_block_t *w_block = weights + block_index;
        const marmot_q8_0_block_t *a_block = activations + block_index;

        int8x16_t w0 = vld1q_s8(w_block->qs);
        int8x16_t w1 = vld1q_s8(w_block->qs + 16);
        int8x16_t a0 = vld1q_s8(a_block->qs);
        int8x16_t a1 = vld1q_s8(a_block->qs + 16);

        int32x4_t acc = vdupq_n_s32(0);
        acc = vmmlaq_s32(acc, w0, a0);
        acc = vmmlaq_s32(acc, w1, a1);

        const float block_scale =
            (float)marmot_float16_to_native(w_block->scale) * (float)marmot_float16_to_native(a_block->scale);
        const int32_t block_sum = cpu_vec_dot_i8mm_sum(acc);
        total += (float)block_sum * block_scale;
    }

    return total;
}

float cpu_vec_dot_q8_k_q8_k_neon_i8mm(
    const marmot_q8_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
) {
    if (weights == nullptr || activations == nullptr || num_blocks == 0) {
        return 0.0f;
    }

    float total = 0.0f;
    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const marmot_q8_k_block_t *w_block = weights + block_index;
        const marmot_q8_k_block_t *a_block = activations + block_index;

        int32x4_t acc = vdupq_n_s32(0);
        const int8_t *qw = w_block->qs;
        const int8_t *qa = a_block->qs;

        for (size_t j = 0; j < MARMOT_QK_K_VALUES; j += 16) {
            int8x16_t wv = vld1q_s8(qw + j);
            int8x16_t av = vld1q_s8(qa + j);
            acc = vmmlaq_s32(acc, wv, av);
        }

        const float d = w_block->d * a_block->d;
        const int32_t block_sum = cpu_vec_dot_i8mm_sum(acc);
        total += d * (float)block_sum;
    }

    return total;
}
#endif

#endif // MARMOT_ENABLE_NEON && __aarch64__
