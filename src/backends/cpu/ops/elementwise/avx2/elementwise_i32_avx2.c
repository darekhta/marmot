#if MARMOT_ENABLE_AVX2

#include "marmot/tensor.h"

#include <stdint.h>

#include <immintrin.h>

#include "cpu_backend_internal.h"
#include "ops/elementwise/elementwise_int_common.h"

static inline void cpu_store_mask8(uint8_t *dst, size_t offset, __m256i mask) {
    int32_t tmp[8];
    _mm256_storeu_si256((__m256i *)tmp, mask);
    for (size_t lane = 0; lane < 8; ++lane) {
        dst[offset + lane] = (uint8_t)(((uint32_t)tmp[lane] >> 31) ? 1U : 0U);
    }
}

marmot_error_t
cpu_add_i32_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    int32_t *dst = (int32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        _mm256_storeu_si256((__m256i *)(dst + i), _mm256_add_epi32(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] + rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_sub_i32_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    int32_t *dst = (int32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        _mm256_storeu_si256((__m256i *)(dst + i), _mm256_sub_epi32(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] - rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_min_i32_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    int32_t *dst = (int32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        _mm256_storeu_si256((__m256i *)(dst + i), _mm256_min_epi32(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] < rhs[i] ? lhs[i] : rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_max_i32_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    int32_t *dst = (int32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        _mm256_storeu_si256((__m256i *)(dst + i), _mm256_max_epi32(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] > rhs[i] ? lhs[i] : rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_and_i32_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    int32_t *dst = (int32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        _mm256_storeu_si256((__m256i *)(dst + i), _mm256_and_si256(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] & rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_or_i32_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    int32_t *dst = (int32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        _mm256_storeu_si256((__m256i *)(dst + i), _mm256_or_si256(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] | rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_xor_i32_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    int32_t *dst = (int32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        _mm256_storeu_si256((__m256i *)(dst + i), _mm256_xor_si256(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] ^ rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_shl_i32_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    int32_t *dst = (int32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vamt = _mm256_loadu_si256((const __m256i *)(rhs + i));
        __m256i zero = _mm256_setzero_si256();
        __m256i bits = _mm256_set1_epi32(32);
        __m256i max = _mm256_set1_epi32(31);
        __m256i clamped = _mm256_max_epi32(vamt, zero);
        __m256i valid = _mm256_cmpgt_epi32(bits, clamped);
        __m256i safe = _mm256_min_epi32(clamped, max);
        __m256i shifted = _mm256_sllv_epi32(va, safe);
        __m256i masked = _mm256_and_si256(shifted, valid);
        _mm256_storeu_si256((__m256i *)(dst + i), masked);
    }
    for (; i < n; ++i) {
        unsigned amount = cpu_ew_shift_amount((int64_t)rhs[i], 32);
        dst[i] = amount >= 32 ? 0 : (lhs[i] << (int)amount);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_shr_i32_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    int32_t *dst = (int32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vamt = _mm256_loadu_si256((const __m256i *)(rhs + i));
        __m256i zero = _mm256_setzero_si256();
        __m256i bits = _mm256_set1_epi32(32);
        __m256i max = _mm256_set1_epi32(31);
        __m256i clamped = _mm256_max_epi32(vamt, zero);
        __m256i valid = _mm256_cmpgt_epi32(bits, clamped);
        __m256i safe = _mm256_min_epi32(clamped, max);
        __m256i shifted = _mm256_srav_epi32(va, safe);
        __m256i sign_fill = _mm256_srai_epi32(va, 31);
        __m256i blended = _mm256_blendv_epi8(sign_fill, shifted, valid);
        _mm256_storeu_si256((__m256i *)(dst + i), blended);
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

marmot_error_t cpu_bitwise_shr_logical_i32_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const uint32_t *lhs = (const uint32_t *)a->data;
    const uint32_t *rhs = (const uint32_t *)b->data;
    uint32_t *dst = (uint32_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vamt = _mm256_loadu_si256((const __m256i *)(rhs + i));
        __m256i zero = _mm256_setzero_si256();
        __m256i bits = _mm256_set1_epi32(32);
        __m256i max = _mm256_set1_epi32(31);
        __m256i clamped = _mm256_max_epi32(vamt, zero);
        __m256i valid = _mm256_cmpgt_epi32(bits, clamped);
        __m256i safe = _mm256_min_epi32(clamped, max);
        __m256i shifted = _mm256_srlv_epi32(va, safe);
        __m256i masked = _mm256_and_si256(shifted, valid);
        _mm256_storeu_si256((__m256i *)(dst + i), masked);
    }
    for (; i < n; ++i) {
        unsigned amount = cpu_ew_shift_amount((int64_t)rhs[i], 32);
        dst[i] = amount >= 32 ? 0U : (lhs[i] >> amount);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_eq_i32_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        cpu_store_mask8(dst, i, _mm256_cmpeq_epi32(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] == rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_ne_i32_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        __m256i cmp = _mm256_xor_si256(_mm256_cmpeq_epi32(va, vb), _mm256_set1_epi32(-1));
        cpu_store_mask8(dst, i, cmp);
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] != rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_lt_i32_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        cpu_store_mask8(dst, i, _mm256_cmpgt_epi32(vb, va));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] < rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_le_i32_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        __m256i lt = _mm256_cmpgt_epi32(vb, va);
        __m256i eq = _mm256_cmpeq_epi32(va, vb);
        cpu_store_mask8(dst, i, _mm256_or_si256(lt, eq));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] <= rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_gt_i32_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        cpu_store_mask8(dst, i, _mm256_cmpgt_epi32(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] > rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_ge_i32_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int32_t *lhs = (const int32_t *)a->data;
    const int32_t *rhs = (const int32_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        __m256i gt = _mm256_cmpgt_epi32(va, vb);
        __m256i eq = _mm256_cmpeq_epi32(va, vb);
        cpu_store_mask8(dst, i, _mm256_or_si256(gt, eq));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] >= rhs[i]);
    }
    return MARMOT_SUCCESS;
}

#endif // MARMOT_ENABLE_AVX2
