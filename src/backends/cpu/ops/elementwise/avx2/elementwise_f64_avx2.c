#if MARMOT_ENABLE_AVX2

#include "marmot/tensor.h"

#include <immintrin.h>
#include <math.h>

#include "cpu_backend_internal.h"

marmot_error_t
cpu_add_f64_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const double *lhs = (const double *)a->data;
    const double *rhs = (const double *)b->data;
    double *dst = (double *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(lhs + i);
        __m256d vb = _mm256_loadu_pd(rhs + i);
        _mm256_storeu_pd(dst + i, _mm256_add_pd(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] + rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_sub_f64_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const double *lhs = (const double *)a->data;
    const double *rhs = (const double *)b->data;
    double *dst = (double *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(lhs + i);
        __m256d vb = _mm256_loadu_pd(rhs + i);
        _mm256_storeu_pd(dst + i, _mm256_sub_pd(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] - rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_mul_f64_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const double *lhs = (const double *)a->data;
    const double *rhs = (const double *)b->data;
    double *dst = (double *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(lhs + i);
        __m256d vb = _mm256_loadu_pd(rhs + i);
        _mm256_storeu_pd(dst + i, _mm256_mul_pd(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] * rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_div_f64_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const double *lhs = (const double *)a->data;
    const double *rhs = (const double *)b->data;
    double *dst = (double *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(lhs + i);
        __m256d vb = _mm256_loadu_pd(rhs + i);
        _mm256_storeu_pd(dst + i, _mm256_div_pd(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] / rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_min_f64_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const double *lhs = (const double *)a->data;
    const double *rhs = (const double *)b->data;
    double *dst = (double *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(lhs + i);
        __m256d vb = _mm256_loadu_pd(rhs + i);
        _mm256_storeu_pd(dst + i, _mm256_min_pd(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = fmin(lhs[i], rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_max_f64_avx2(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const double *lhs = (const double *)a->data;
    const double *rhs = (const double *)b->data;
    double *dst = (double *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(lhs + i);
        __m256d vb = _mm256_loadu_pd(rhs + i);
        _mm256_storeu_pd(dst + i, _mm256_max_pd(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = fmax(lhs[i], rhs[i]);
    }
    return MARMOT_SUCCESS;
}

#endif // MARMOT_ENABLE_AVX2
