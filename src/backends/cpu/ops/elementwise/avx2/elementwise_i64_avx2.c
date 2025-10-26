#if MARMOT_ENABLE_AVX2

#include "marmot/tensor.h"

#include <stdint.h>

#include <immintrin.h>

#include "cpu_backend_internal.h"
#include "ops/elementwise/elementwise_int_common.h"

static inline void cpu_store_mask4(uint8_t *dst, size_t offset, __m256i mask) {
    int64_t tmp[4];
    _mm256_storeu_si256((__m256i *)tmp, mask);
    for (size_t lane = 0; lane < 4; ++lane) {
        dst[offset + lane] = (uint8_t)(((uint64_t)tmp[lane] >> 63) ? 1U : 0U);
    }
}

marmot_error_t
cpu_add_i64_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    int64_t *dst = (int64_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        _mm256_storeu_si256((__m256i *)(dst + i), _mm256_add_epi64(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] + rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_sub_i64_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    int64_t *dst = (int64_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        _mm256_storeu_si256((__m256i *)(dst + i), _mm256_sub_epi64(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] - rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_min_i64_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    int64_t *dst = (int64_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        __m256i mask = _mm256_cmpgt_epi64(vb, va);
        _mm256_storeu_si256((__m256i *)(dst + i), _mm256_blendv_epi8(vb, va, mask));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] < rhs[i] ? lhs[i] : rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_max_i64_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    int64_t *dst = (int64_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        __m256i mask = _mm256_cmpgt_epi64(va, vb);
        _mm256_storeu_si256((__m256i *)(dst + i), _mm256_blendv_epi8(vb, va, mask));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] > rhs[i] ? lhs[i] : rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_and_i64_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    int64_t *dst = (int64_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        _mm256_storeu_si256((__m256i *)(dst + i), _mm256_and_si256(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] & rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_or_i64_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    int64_t *dst = (int64_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        _mm256_storeu_si256((__m256i *)(dst + i), _mm256_or_si256(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] | rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_xor_i64_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    int64_t *dst = (int64_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        _mm256_storeu_si256((__m256i *)(dst + i), _mm256_xor_si256(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] ^ rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_bitwise_shl_i64_avx2(
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

marmot_error_t cpu_bitwise_shr_i64_avx2(
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

marmot_error_t cpu_bitwise_shr_logical_i64_avx2(
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

marmot_error_t cpu_compare_eq_i64_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        cpu_store_mask4(dst, i, _mm256_cmpeq_epi64(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] == rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_ne_i64_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        __m256i cmp = _mm256_xor_si256(_mm256_cmpeq_epi64(va, vb), _mm256_set1_epi64x(-1));
        cpu_store_mask4(dst, i, cmp);
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] != rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_lt_i64_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        cpu_store_mask4(dst, i, _mm256_cmpgt_epi64(vb, va));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] < rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_le_i64_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        __m256i lt = _mm256_cmpgt_epi64(vb, va);
        __m256i eq = _mm256_cmpeq_epi64(va, vb);
        cpu_store_mask4(dst, i, _mm256_or_si256(lt, eq));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] <= rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_gt_i64_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        cpu_store_mask4(dst, i, _mm256_cmpgt_epi64(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] > rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_ge_i64_avx2(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const int64_t *lhs = (const int64_t *)a->data;
    const int64_t *rhs = (const int64_t *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        __m256i gt = _mm256_cmpgt_epi64(va, vb);
        __m256i eq = _mm256_cmpeq_epi64(va, vb);
        cpu_store_mask4(dst, i, _mm256_or_si256(gt, eq));
    }
    for (; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] >= rhs[i]);
    }
    return MARMOT_SUCCESS;
}

#endif // MARMOT_ENABLE_AVX2
