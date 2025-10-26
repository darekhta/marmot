#if MARMOT_ENABLE_NEON

#include "marmot/tensor.h"

#include <arm_neon.h>
#include <math.h>

#include "cpu_backend_internal.h"

marmot_error_t
cpu_add_f32_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(lhs + i);
        float32x4_t vb = vld1q_f32(rhs + i);
        vst1q_f32(dst + i, vaddq_f32(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] + rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_sub_f32_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(lhs + i);
        float32x4_t vb = vld1q_f32(rhs + i);
        vst1q_f32(dst + i, vsubq_f32(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] - rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_mul_f32_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(lhs + i);
        float32x4_t vb = vld1q_f32(rhs + i);
        vst1q_f32(dst + i, vmulq_f32(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] * rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_div_f32_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(lhs + i);
        float32x4_t vb = vld1q_f32(rhs + i);
#if defined(__aarch64__)
        float32x4_t vout = vdivq_f32(va, vb);
#else
        float32x4_t recip = vrecpeq_f32(vb);
        recip = vmulq_f32(vrecpsq_f32(vb, recip), recip);
        recip = vmulq_f32(vrecpsq_f32(vb, recip), recip);
        float32x4_t vout = vmulq_f32(va, recip);
#endif
        vst1q_f32(dst + i, vout);
    }
    for (; i < n; ++i) {
        dst[i] = lhs[i] / rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_min_f32_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(lhs + i);
        float32x4_t vb = vld1q_f32(rhs + i);
        vst1q_f32(dst + i, vminq_f32(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = fminf(lhs[i], rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_max_f32_neon(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(lhs + i);
        float32x4_t vb = vld1q_f32(rhs + i);
        vst1q_f32(dst + i, vmaxq_f32(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = fmaxf(lhs[i], rhs[i]);
    }
    return MARMOT_SUCCESS;
}

#endif // MARMOT_ENABLE_NEON
