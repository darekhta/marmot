#if MARMOT_ENABLE_NEON && defined(__aarch64__)

#include "marmot/tensor.h"

#include <stdint.h>

#include <arm_neon.h>

#include "cpu_backend_internal.h"
#include "ops/elementwise/elementwise_int_common.h"

static inline void cpu_store_mask2(uint8_t *dst, size_t offset, uint64x2_t mask) {
    uint64_t tmp[2];
    vst1q_u64(tmp, mask);
    for (size_t lane = 0; lane < 2; ++lane) {
        dst[offset + lane] = (uint8_t)(tmp[lane] ? 1U : 0U);
    }
}

marmot_error_t
cpu_add_i64_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    int64_t *dst = (int64_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t va = vld1q_s64(lhs + i);
        int64x2_t vb = vld1q_s64(rhs + i);
        vst1q_s64(dst + i, vaddq_s64(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] + rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_sub_i64_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    int64_t *dst = (int64_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t va = vld1q_s64(lhs + i);
        int64x2_t vb = vld1q_s64(rhs + i);
        vst1q_s64(dst + i, vsubq_s64(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] - rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_min_i64_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    int64_t *dst = (int64_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t va = vld1q_s64(lhs + i);
        int64x2_t vb = vld1q_s64(rhs + i);
        uint64x2_t mask = vreinterpretq_u64_s64(vcltq_s64(va, vb));
        vst1q_s64(dst + i, vbslq_s64(mask, va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] < rhs[i] ? lhs[i] : rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_max_i64_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    int64_t *dst = (int64_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t va = vld1q_s64(lhs + i);
        int64x2_t vb = vld1q_s64(rhs + i);
        uint64x2_t mask = vreinterpretq_u64_s64(vcgtq_s64(va, vb));
        vst1q_s64(dst + i, vbslq_s64(mask, va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] > rhs[i] ? lhs[i] : rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_and_i64_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    int64_t *dst = (int64_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t va = vld1q_s64(lhs + i);
        int64x2_t vb = vld1q_s64(rhs + i);
        vst1q_s64(dst + i, vandq_s64(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] & rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_or_i64_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    int64_t *dst = (int64_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t va = vld1q_s64(lhs + i);
        int64x2_t vb = vld1q_s64(rhs + i);
        vst1q_s64(dst + i, vorrq_s64(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] | rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_xor_i64_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    int64_t *dst = (int64_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t va = vld1q_s64(lhs + i);
        int64x2_t vb = vld1q_s64(rhs + i);
        vst1q_s64(dst + i, veorq_s64(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] ^ rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_shl_i64_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    int64_t *dst = (int64_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        unsigned amount = cpu_ew_shift_amount((int64_t)rhs[i], 64);
        dst[i] = amount >= 64 ? 0 : (int64_t)((uint64_t)lhs[i] << amount);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_shr_i64_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    int64_t *dst = (int64_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        unsigned amount = cpu_ew_shift_amount((int64_t)rhs[i], 64);
        if (amount >= 64) {
            dst[i] = lhs[i] < 0 ? -1 : 0;
        } else {
            dst[i] = lhs[i] >> (int)amount;
        }
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_shr_logical_i64_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const uint64_t *lhs = (const uint64_t *)a->data;
    const uint64_t *rhs = (const uint64_t *)b->data;
    uint64_t *dst = (uint64_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        unsigned amount = cpu_ew_shift_amount((int64_t)rhs[i], 64);
        dst[i] = amount >= 64 ? 0U : (lhs[i] >> amount);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_eq_i64_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t va = vld1q_s64(lhs + i);
        int64x2_t vb = vld1q_s64(rhs + i);
        cpu_store_mask2(dst, i, vreinterpretq_u64_s64(vceqq_s64(va, vb)));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] == rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_ne_i64_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t va = vld1q_s64(lhs + i);
        int64x2_t vb = vld1q_s64(rhs + i);
        uint64x2_t cmp = veorq_u64(vreinterpretq_u64_s64(vceqq_s64(va, vb)), vdupq_n_u64(UINT64_MAX));
        cpu_store_mask2(dst, i, cmp);
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] != rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_lt_i64_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t va = vld1q_s64(lhs + i);
        int64x2_t vb = vld1q_s64(rhs + i);
        cpu_store_mask2(dst, i, vreinterpretq_u64_s64(vcltq_s64(va, vb)));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] < rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_le_i64_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t va = vld1q_s64(lhs + i);
        int64x2_t vb = vld1q_s64(rhs + i);
        cpu_store_mask2(dst, i, vreinterpretq_u64_s64(vcleq_s64(va, vb)));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] <= rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_gt_i64_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t va = vld1q_s64(lhs + i);
        int64x2_t vb = vld1q_s64(rhs + i);
        cpu_store_mask2(dst, i, vreinterpretq_u64_s64(vcgtq_s64(va, vb)));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] > rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_ge_i64_neon(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t va = vld1q_s64(lhs + i);
        int64x2_t vb = vld1q_s64(rhs + i);
        cpu_store_mask2(dst, i, vreinterpretq_u64_s64(vcgeq_s64(va, vb)));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] >= rhs[i]);
    }
    return MARMOT_SUCCESS;
}

#endif // MARMOT_ENABLE_NEON && defined(__aarch64__)
