#if MARMOT_ENABLE_AVX2

#include "marmot/tensor.h"

#include <immintrin.h>
#include <math.h>

#include "cpu_backend_internal.h"

marmot_error_t
cpu_add_f32_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(lhs + i);
        __m256 vb = _mm256_loadu_ps(rhs + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] + rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_sub_f32_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(lhs + i);
        __m256 vb = _mm256_loadu_ps(rhs + i);
        _mm256_storeu_ps(dst + i, _mm256_sub_ps(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] - rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_mul_f32_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(lhs + i);
        __m256 vb = _mm256_loadu_ps(rhs + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] * rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_div_f32_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(lhs + i);
        __m256 vb = _mm256_loadu_ps(rhs + i);
        _mm256_storeu_ps(dst + i, _mm256_div_ps(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] / rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_min_f32_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(lhs + i);
        __m256 vb = _mm256_loadu_ps(rhs + i);
        _mm256_storeu_ps(dst + i, _mm256_min_ps(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = fminf(lhs[i], rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_max_f32_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(lhs + i);
        __m256 vb = _mm256_loadu_ps(rhs + i);
        _mm256_storeu_ps(dst + i, _mm256_max_ps(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = fmaxf(lhs[i], rhs[i]);
    }
    return MARMOT_SUCCESS;
}

#endif // MARMOT_ENABLE_AVX2
