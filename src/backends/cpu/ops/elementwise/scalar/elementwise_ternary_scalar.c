#include "marmot/tensor.h"

#include <math.h>

#include "cpu_backend_internal.h"

// -----------------------------------------------------------------------------
// Scalar ternary kernels (FMA, WHERE)
// -----------------------------------------------------------------------------

marmot_error_t cpu_fma_f32_scalar(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, const marmot_tensor_t *c,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    const size_t n = marmot_tensor_num_elements(out);
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    const float *cval = (const float *)c->data;
    float *dst = (float *)out->data;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = fmaf(lhs[i], rhs[i], cval[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_fma_f64_scalar(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, const marmot_tensor_t *c,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    const size_t n = marmot_tensor_num_elements(out);
    const double *lhs = (const double *)a->data;
    const double *rhs = (const double *)b->data;
    const double *cval = (const double *)c->data;
    double *dst = (double *)out->data;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = fma(lhs[i], rhs[i], cval[i]);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_fma_via_f32(marmot_dtype_t dtype, const void *a, const void *b, const void *c, void *out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float av = cpu_load_as_f32(dtype, a, i);
        float bv = cpu_load_as_f32(dtype, b, i);
        float cv = cpu_load_as_f32(dtype, c, i);
        cpu_store_from_f32(dtype, out, i, fmaf(av, bv, cv));
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_fma_f16_scalar(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, const marmot_tensor_t *c,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    return cpu_fma_via_f32(MARMOT_DTYPE_FLOAT16, a->data, b->data, c->data, out->data, marmot_tensor_num_elements(out));
}

marmot_error_t cpu_fma_bf16_scalar(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, const marmot_tensor_t *c,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    return cpu_fma_via_f32(
        MARMOT_DTYPE_BFLOAT16, a->data, b->data, c->data, out->data, marmot_tensor_num_elements(out)
    );
}

#if MARMOT_ENABLE_FP8
marmot_error_t cpu_fma_fp8_e4m3_scalar(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, const marmot_tensor_t *c,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    return cpu_fma_via_f32(
        MARMOT_DTYPE_FLOAT8_E4M3, a->data, b->data, c->data, out->data, marmot_tensor_num_elements(out)
    );
}

marmot_error_t cpu_fma_fp8_e5m2_scalar(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, const marmot_tensor_t *c,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    return cpu_fma_via_f32(
        MARMOT_DTYPE_FLOAT8_E5M2, a->data, b->data, c->data, out->data, marmot_tensor_num_elements(out)
    );
}
#endif

marmot_error_t cpu_where_f32_scalar(
    const void *device_ctx, const marmot_tensor_t *mask, const marmot_tensor_t *a, const marmot_tensor_t *b,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    const size_t n = marmot_tensor_num_elements(out);
    const uint8_t *m = (const uint8_t *)mask->data;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = m[i] ? lhs[i] : rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_where_f64_scalar(
    const void *device_ctx, const marmot_tensor_t *mask, const marmot_tensor_t *a, const marmot_tensor_t *b,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    const size_t n = marmot_tensor_num_elements(out);
    const uint8_t *m = (const uint8_t *)mask->data;
    const double *lhs = (const double *)a->data;
    const double *rhs = (const double *)b->data;
    double *dst = (double *)out->data;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = m[i] ? lhs[i] : rhs[i];
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_where_via_f32(marmot_dtype_t dtype, const void *mask, const void *a, const void *b, void *out, size_t n) {
    const uint8_t *m = (const uint8_t *)mask;
    for (size_t i = 0; i < n; ++i) {
        float av = cpu_load_as_f32(dtype, a, i);
        float bv = cpu_load_as_f32(dtype, b, i);
        cpu_store_from_f32(dtype, out, i, m[i] ? av : bv);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_where_f16_scalar(
    const void *device_ctx, const marmot_tensor_t *mask, const marmot_tensor_t *a, const marmot_tensor_t *b,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    return cpu_where_via_f32(
        MARMOT_DTYPE_FLOAT16, mask->data, a->data, b->data, out->data, marmot_tensor_num_elements(out)
    );
}

marmot_error_t cpu_where_bf16_scalar(
    const void *device_ctx, const marmot_tensor_t *mask, const marmot_tensor_t *a, const marmot_tensor_t *b,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    return cpu_where_via_f32(
        MARMOT_DTYPE_BFLOAT16, mask->data, a->data, b->data, out->data, marmot_tensor_num_elements(out)
    );
}

#if MARMOT_ENABLE_FP8
marmot_error_t cpu_where_fp8_e4m3_scalar(
    const void *device_ctx, const marmot_tensor_t *mask, const marmot_tensor_t *a, const marmot_tensor_t *b,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    return cpu_where_via_f32(
        MARMOT_DTYPE_FLOAT8_E4M3, mask->data, a->data, b->data, out->data, marmot_tensor_num_elements(out)
    );
}

marmot_error_t cpu_where_fp8_e5m2_scalar(
    const void *device_ctx, const marmot_tensor_t *mask, const marmot_tensor_t *a, const marmot_tensor_t *b,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    return cpu_where_via_f32(
        MARMOT_DTYPE_FLOAT8_E5M2, mask->data, a->data, b->data, out->data, marmot_tensor_num_elements(out)
    );
}
#endif
