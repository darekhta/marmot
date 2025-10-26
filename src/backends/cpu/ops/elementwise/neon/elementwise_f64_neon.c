#if MARMOT_ENABLE_NEON && defined(__aarch64__)

#include "marmot/tensor.h"

#include <arm_neon.h>
#include <math.h>

#include "cpu_backend_internal.h"

marmot_error_t
cpu_add_f64_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const double *lhs = (const double *)a->data;
    const double *rhs = (const double *)b->data;
    double *dst = (double *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(lhs + i);
        float64x2_t vb = vld1q_f64(rhs + i);
        vst1q_f64(dst + i, vaddq_f64(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] + rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_sub_f64_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const double *lhs = (const double *)a->data;
    const double *rhs = (const double *)b->data;
    double *dst = (double *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(lhs + i);
        float64x2_t vb = vld1q_f64(rhs + i);
        vst1q_f64(dst + i, vsubq_f64(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] - rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_mul_f64_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const double *lhs = (const double *)a->data;
    const double *rhs = (const double *)b->data;
    double *dst = (double *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(lhs + i);
        float64x2_t vb = vld1q_f64(rhs + i);
        vst1q_f64(dst + i, vmulq_f64(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] * rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_div_f64_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const double *lhs = (const double *)a->data;
    const double *rhs = (const double *)b->data;
    double *dst = (double *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(lhs + i);
        float64x2_t vb = vld1q_f64(rhs + i);
        vst1q_f64(dst + i, vdivq_f64(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] / rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_min_f64_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const double *lhs = (const double *)a->data;
    const double *rhs = (const double *)b->data;
    double *dst = (double *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(lhs + i);
        float64x2_t vb = vld1q_f64(rhs + i);
        vst1q_f64(dst + i, vminq_f64(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = fmin(lhs[i], rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_max_f64_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const double *lhs = (const double *)a->data;
    const double *rhs = (const double *)b->data;
    double *dst = (double *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(lhs + i);
        float64x2_t vb = vld1q_f64(rhs + i);
        vst1q_f64(dst + i, vmaxq_f64(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = fmax(lhs[i], rhs[i]);
    }
    return MARMOT_SUCCESS;
}

#endif // MARMOT_ENABLE_NEON && defined(__aarch64__)
