#if MARMOT_ENABLE_NEON

#include "marmot/tensor.h"

#include <stdint.h>

#include <arm_neon.h>

#include "cpu_backend_internal.h"
#include "ops/elementwise/elementwise_int_common.h"

static inline void cpu_store_mask4(uint8_t *dst, size_t offset, uint32x4_t mask) {
    uint32_t tmp[4];
    vst1q_u32(tmp, mask);
    for (size_t lane = 0; lane < 4; ++lane) {
        dst[offset + lane] = (uint8_t)(tmp[lane] ? 1U : 0U);
    }
}

marmot_error_t
cpu_add_i32_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    int32_t *dst = (int32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(lhs + i);
        int32x4_t vb = vld1q_s32(rhs + i);
        vst1q_s32(dst + i, vaddq_s32(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] + rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_sub_i32_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    int32_t *dst = (int32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(lhs + i);
        int32x4_t vb = vld1q_s32(rhs + i);
        vst1q_s32(dst + i, vsubq_s32(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] - rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_min_i32_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    int32_t *dst = (int32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(lhs + i);
        int32x4_t vb = vld1q_s32(rhs + i);
        vst1q_s32(dst + i, vminq_s32(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] < rhs[i] ? lhs[i] : rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_max_i32_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    int32_t *dst = (int32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(lhs + i);
        int32x4_t vb = vld1q_s32(rhs + i);
        vst1q_s32(dst + i, vmaxq_s32(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] > rhs[i] ? lhs[i] : rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_and_i32_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    int32_t *dst = (int32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(lhs + i);
        int32x4_t vb = vld1q_s32(rhs + i);
        vst1q_s32(dst + i, vandq_s32(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] & rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_or_i32_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    int32_t *dst = (int32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(lhs + i);
        int32x4_t vb = vld1q_s32(rhs + i);
        vst1q_s32(dst + i, vorrq_s32(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] | rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_xor_i32_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    int32_t *dst = (int32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(lhs + i);
        int32x4_t vb = vld1q_s32(rhs + i);
        vst1q_s32(dst + i, veorq_s32(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] ^ rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_shl_i32_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    int32_t *dst = (int32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    const int32x4_t zero_vec = vdupq_n_s32(0);
    const int32x4_t bits_vec = vdupq_n_s32(32);
    const int32x4_t max_vec = vdupq_n_s32(31);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(lhs + i);
        int32x4_t vamt_raw = vld1q_s32(rhs + i);
        int32x4_t vamt_clamped = vmaxq_s32(vamt_raw, zero_vec);
        uint32x4_t valid = vreinterpretq_u32_s32(vcltq_s32(vamt_clamped, bits_vec));
        int32x4_t vamt_safe = vminq_s32(vamt_clamped, max_vec);
        int32x4_t vshifted = vshlq_s32(va, vamt_safe);
        uint32x4_t masked = vandq_u32(valid, vreinterpretq_u32_s32(vshifted));
        vst1q_s32(dst + i, vreinterpretq_s32_u32(masked));
    }
    for (; i < n; ++i) {
        unsigned amount = cpu_ew_shift_amount((int64_t)rhs[i], 32);
        dst[i] = amount >= 32 ? 0 : (lhs[i] << (int)amount);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_shr_i32_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    int32_t *dst = (int32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    const int32x4_t zero_vec = vdupq_n_s32(0);
    const int32x4_t bits_vec = vdupq_n_s32(32);
    const int32x4_t max_vec = vdupq_n_s32(31);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(lhs + i);
        int32x4_t vamt_raw = vld1q_s32(rhs + i);
        int32x4_t vamt_clamped = vmaxq_s32(vamt_raw, zero_vec);
        uint32x4_t valid = vreinterpretq_u32_s32(vcltq_s32(vamt_clamped, bits_vec));
        int32x4_t vamt_safe = vminq_s32(vamt_clamped, max_vec);
        int32x4_t vneg_amt = vnegq_s32(vamt_safe);
        int32x4_t vshifted = vshlq_s32(va, vneg_amt);
        int32x4_t sign_fill = vshrq_n_s32(va, 31);
        int32x4_t blended = vbslq_s32(valid, vshifted, sign_fill);
        vst1q_s32(dst + i, blended);
    }
    for (; i < n; ++i) {
        unsigned amount = cpu_ew_shift_amount((int64_t)rhs[i], 32);
        if (amount >= 32) {
            dst[i] = lhs[i] < 0 ? -1 : 0;
        } else {
            dst[i] = lhs[i] >> (int)amount;
        }
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_shr_logical_i32_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const uint32_t *lhs = (const uint32_t *)a->data;
    const uint32_t *rhs = (const uint32_t *)b->data;
    uint32_t *dst = (uint32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    const uint32x4_t zero_vec = vdupq_n_u32(0);
    const uint32x4_t bits_vec = vdupq_n_u32(32);
    const uint32x4_t max_vec = vdupq_n_u32(31);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint32x4_t va = vld1q_u32(lhs + i);
        uint32x4_t vamt_raw = vld1q_u32(rhs + i);
        uint32x4_t vamt_clamped = vmaxq_u32(vamt_raw, zero_vec);
        uint32x4_t valid = vcltq_u32(vamt_clamped, bits_vec);
        uint32x4_t vamt_safe = vminq_u32(vamt_clamped, max_vec);
        int32x4_t vneg = vnegq_s32(vreinterpretq_s32_u32(vamt_safe));
        uint32x4_t vshifted = vshlq_u32(va, vneg);
        uint32x4_t masked = vandq_u32(valid, vshifted);
        vst1q_u32(dst + i, masked);
    }
    for (; i < n; ++i) {
        unsigned amount = cpu_ew_shift_amount((int64_t)rhs[i], 32);
        dst[i] = amount >= 32 ? 0U : (lhs[i] >> amount);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_eq_i32_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(lhs + i);
        int32x4_t vb = vld1q_s32(rhs + i);
        cpu_store_mask4(dst, i, vreinterpretq_u32_s32(vceqq_s32(va, vb)));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] == rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_ne_i32_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(lhs + i);
        int32x4_t vb = vld1q_s32(rhs + i);
        uint32x4_t cmp = vmvnq_u32(vreinterpretq_u32_s32(vceqq_s32(va, vb)));
        cpu_store_mask4(dst, i, cmp);
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] != rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_lt_i32_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(lhs + i);
        int32x4_t vb = vld1q_s32(rhs + i);
        cpu_store_mask4(dst, i, vreinterpretq_u32_s32(vcltq_s32(va, vb)));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] < rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_le_i32_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(lhs + i);
        int32x4_t vb = vld1q_s32(rhs + i);
        cpu_store_mask4(dst, i, vreinterpretq_u32_s32(vcleq_s32(va, vb)));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] <= rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_gt_i32_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(lhs + i);
        int32x4_t vb = vld1q_s32(rhs + i);
        cpu_store_mask4(dst, i, vreinterpretq_u32_s32(vcgtq_s32(va, vb)));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] > rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_ge_i32_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(lhs + i);
        int32x4_t vb = vld1q_s32(rhs + i);
        cpu_store_mask4(dst, i, vreinterpretq_u32_s32(vcgeq_s32(va, vb)));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] >= rhs[i]);
    }
    return MARMOT_SUCCESS;
}

#endif // MARMOT_ENABLE_NEON
