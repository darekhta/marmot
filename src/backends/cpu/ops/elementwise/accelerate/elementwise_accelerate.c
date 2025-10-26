#if MARMOT_ENABLE_ACCELERATE

#include "marmot/tensor.h"

#include <Accelerate/Accelerate.h>

#include <stdlib.h>

#include <limits.h>
#include <math.h>

#include "cpu_backend_internal.h"

typedef void (*cpu_ew_vforce_binary_f32)(float *, const float *, const float *, const int *);
typedef void (*cpu_ew_vforce_binary_f64)(double *, const double *, const double *, const int *);

typedef marmot_error_t (*cpu_ew_binary_fn)(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n);

static inline bool cpu_ew_accelerate_f32_use_scalar(size_t n) {
    return n < 128;
}

static inline bool cpu_ew_accelerate_f64_use_scalar(size_t n) {
    return n < 128;
}

static inline bool cpu_ew_accelerate_length_ok(size_t n) {
    return n <= (size_t)INT32_MAX;
}

static float *cpu_ew_accelerate_alloc_f32(size_t n) {
    return (float *)marmot_aligned_alloc(64, n * sizeof(float));
}

static marmot_error_t
cpu_ew_accelerate_f32_scalar_binary(const void *lhs, const void *rhs, void *out, size_t n, float (*fn)(float, float)) {
    const float *a = (const float *)lhs;
    const float *b = (const float *)rhs;
    float *dst = (float *)out;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = fn(a[i], b[i]);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_ew_accelerate_f64_scalar_binary(
    const void *lhs, const void *rhs, void *out, size_t n, double (*fn)(double, double)
) {
    const double *a = (const double *)lhs;
    const double *b = (const double *)rhs;
    double *dst = (double *)out;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = fn(a[i], b[i]);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_ew_accelerate_f32_vforce_binary(cpu_ew_vforce_binary_f32 fn, float *out, const float *a, const float *b, size_t n) {
    while (n > 0) {
        int chunk = (n > (size_t)INT32_MAX) ? INT32_MAX : (int)n;
        fn(out, a, b, &chunk);
        size_t step = (size_t)chunk;
        out += step;
        a += step;
        b += step;
        n -= step;
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_ew_accelerate_f64_vforce_binary(
    cpu_ew_vforce_binary_f64 fn, double *out, const double *a, const double *b, size_t n
) {
    while (n > 0) {
        int chunk = (n > (size_t)INT32_MAX) ? INT32_MAX : (int)n;
        fn(out, a, b, &chunk);
        size_t step = (size_t)chunk;
        out += step;
        a += step;
        b += step;
        n -= step;
    }
    return MARMOT_SUCCESS;
}

static float cpu_ew_accelerate_f32_add_scalar(float a, float b) {
    return a + b;
}

static float cpu_ew_accelerate_f32_sub_scalar(float a, float b) {
    return a - b;
}

static float cpu_ew_accelerate_f32_mul_scalar(float a, float b) {
    return a * b;
}

static float cpu_ew_accelerate_f32_div_scalar(float a, float b) {
    return a / b;
}

static float cpu_ew_accelerate_f32_min_scalar(float a, float b) {
    return a < b ? a : b;
}

static float cpu_ew_accelerate_f32_max_scalar(float a, float b) {
    return a > b ? a : b;
}

static float cpu_ew_accelerate_f32_pow_scalar(float a, float b) {
    return powf(a, b);
}

static float cpu_ew_accelerate_f32_mod_scalar(float a, float b) {
    return fmodf(a, b);
}

static double cpu_ew_accelerate_f64_add_scalar(double a, double b) {
    return a + b;
}

static double cpu_ew_accelerate_f64_sub_scalar(double a, double b) {
    return a - b;
}

static double cpu_ew_accelerate_f64_mul_scalar(double a, double b) {
    return a * b;
}

static double cpu_ew_accelerate_f64_div_scalar(double a, double b) {
    return a / b;
}

static double cpu_ew_accelerate_f64_min_scalar(double a, double b) {
    return a < b ? a : b;
}

static double cpu_ew_accelerate_f64_max_scalar(double a, double b) {
    return a > b ? a : b;
}

static double cpu_ew_accelerate_f64_pow_scalar(double a, double b) {
    return pow(a, b);
}

static double cpu_ew_accelerate_f64_mod_scalar(double a, double b) {
    return fmod(a, b);
}

static marmot_error_t
cpu_ew_f32_accelerate_add(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    (void)ctx;
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (!cpu_ew_accelerate_f32_use_scalar(n) && cpu_ew_accelerate_length_ok(n)) {
        if (out == lhs) {
            cblas_saxpy((int)n, 1.0f, (const float *)rhs, 1, (float *)out, 1);
            return MARMOT_SUCCESS;
        }
        if (out == rhs) {
            cblas_saxpy((int)n, 1.0f, (const float *)lhs, 1, (float *)out, 1);
            return MARMOT_SUCCESS;
        }
    }
    if (cpu_ew_accelerate_f32_use_scalar(n)) {
        return cpu_ew_accelerate_f32_scalar_binary(lhs, rhs, out, n, cpu_ew_accelerate_f32_add_scalar);
    }
    vDSP_vadd((const float *)lhs, 1, (const float *)rhs, 1, (float *)out, 1, (vDSP_Length)n);
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_ew_f32_accelerate_sub(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    (void)ctx;
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (!cpu_ew_accelerate_f32_use_scalar(n) && cpu_ew_accelerate_length_ok(n) && out == lhs) {
        cblas_saxpy((int)n, -1.0f, (const float *)rhs, 1, (float *)out, 1);
        return MARMOT_SUCCESS;
    }
    if (cpu_ew_accelerate_f32_use_scalar(n)) {
        return cpu_ew_accelerate_f32_scalar_binary(lhs, rhs, out, n, cpu_ew_accelerate_f32_sub_scalar);
    }
    vDSP_vsub((const float *)rhs, 1, (const float *)lhs, 1, (float *)out, 1, (vDSP_Length)n);
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_ew_f32_accelerate_mul(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    (void)ctx;
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (cpu_ew_accelerate_f32_use_scalar(n)) {
        return cpu_ew_accelerate_f32_scalar_binary(lhs, rhs, out, n, cpu_ew_accelerate_f32_mul_scalar);
    }
    vDSP_vmul((const float *)lhs, 1, (const float *)rhs, 1, (float *)out, 1, (vDSP_Length)n);
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_ew_f32_accelerate_div(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    (void)ctx;
    if (cpu_ew_accelerate_f32_use_scalar(n)) {
        return cpu_ew_accelerate_f32_scalar_binary(lhs, rhs, out, n, cpu_ew_accelerate_f32_div_scalar);
    }
    return cpu_ew_accelerate_f32_vforce_binary(vvdivf, (float *)out, (const float *)lhs, (const float *)rhs, n);
}

static marmot_error_t
cpu_ew_f32_accelerate_pow(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    (void)ctx;
    if (cpu_ew_accelerate_f32_use_scalar(n)) {
        return cpu_ew_accelerate_f32_scalar_binary(lhs, rhs, out, n, cpu_ew_accelerate_f32_pow_scalar);
    }
    const float *base = (const float *)lhs;
    const float *exp = (const float *)rhs;
    float *dst = (float *)out;
    size_t remaining = n;
    while (remaining > 0) {
        int chunk = (remaining > (size_t)INT32_MAX) ? INT32_MAX : (int)remaining;
        vvpowf(dst, exp, base, &chunk);
        size_t step = (size_t)chunk;
        dst += step;
        base += step;
        exp += step;
        remaining -= step;
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_ew_f32_accelerate_mod(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    (void)ctx;
    if (cpu_ew_accelerate_f32_use_scalar(n)) {
        return cpu_ew_accelerate_f32_scalar_binary(lhs, rhs, out, n, cpu_ew_accelerate_f32_mod_scalar);
    }
    return cpu_ew_accelerate_f32_vforce_binary(vvfmodf, (float *)out, (const float *)lhs, (const float *)rhs, n);
}

static marmot_error_t
cpu_ew_f32_accelerate_min(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    (void)ctx;
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (cpu_ew_accelerate_f32_use_scalar(n)) {
        return cpu_ew_accelerate_f32_scalar_binary(lhs, rhs, out, n, cpu_ew_accelerate_f32_min_scalar);
    }
    vDSP_vmin((const float *)lhs, 1, (const float *)rhs, 1, (float *)out, 1, (vDSP_Length)n);
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_ew_f32_accelerate_max(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    (void)ctx;
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (cpu_ew_accelerate_f32_use_scalar(n)) {
        return cpu_ew_accelerate_f32_scalar_binary(lhs, rhs, out, n, cpu_ew_accelerate_f32_max_scalar);
    }
    vDSP_vmax((const float *)lhs, 1, (const float *)rhs, 1, (float *)out, 1, (vDSP_Length)n);
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_ew_f64_accelerate_add(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    (void)ctx;
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (!cpu_ew_accelerate_f64_use_scalar(n) && cpu_ew_accelerate_length_ok(n)) {
        if (out == lhs) {
            cblas_daxpy((int)n, 1.0, (const double *)rhs, 1, (double *)out, 1);
            return MARMOT_SUCCESS;
        }
        if (out == rhs) {
            cblas_daxpy((int)n, 1.0, (const double *)lhs, 1, (double *)out, 1);
            return MARMOT_SUCCESS;
        }
    }
    if (cpu_ew_accelerate_f64_use_scalar(n)) {
        return cpu_ew_accelerate_f64_scalar_binary(lhs, rhs, out, n, cpu_ew_accelerate_f64_add_scalar);
    }
    vDSP_vaddD((const double *)lhs, 1, (const double *)rhs, 1, (double *)out, 1, (vDSP_Length)n);
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_ew_f64_accelerate_sub(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    (void)ctx;
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (!cpu_ew_accelerate_f64_use_scalar(n) && cpu_ew_accelerate_length_ok(n) && out == lhs) {
        cblas_daxpy((int)n, -1.0, (const double *)rhs, 1, (double *)out, 1);
        return MARMOT_SUCCESS;
    }
    if (cpu_ew_accelerate_f64_use_scalar(n)) {
        return cpu_ew_accelerate_f64_scalar_binary(lhs, rhs, out, n, cpu_ew_accelerate_f64_sub_scalar);
    }
    vDSP_vsubD((const double *)rhs, 1, (const double *)lhs, 1, (double *)out, 1, (vDSP_Length)n);
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_ew_f64_accelerate_mul(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    (void)ctx;
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (cpu_ew_accelerate_f64_use_scalar(n)) {
        return cpu_ew_accelerate_f64_scalar_binary(lhs, rhs, out, n, cpu_ew_accelerate_f64_mul_scalar);
    }
    vDSP_vmulD((const double *)lhs, 1, (const double *)rhs, 1, (double *)out, 1, (vDSP_Length)n);
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_ew_f64_accelerate_div(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    (void)ctx;
    if (cpu_ew_accelerate_f64_use_scalar(n)) {
        return cpu_ew_accelerate_f64_scalar_binary(lhs, rhs, out, n, cpu_ew_accelerate_f64_div_scalar);
    }
    return cpu_ew_accelerate_f64_vforce_binary(vvdiv, (double *)out, (const double *)lhs, (const double *)rhs, n);
}

static marmot_error_t
cpu_ew_f64_accelerate_pow(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    (void)ctx;
    if (cpu_ew_accelerate_f64_use_scalar(n)) {
        return cpu_ew_accelerate_f64_scalar_binary(lhs, rhs, out, n, cpu_ew_accelerate_f64_pow_scalar);
    }
    const double *base = (const double *)lhs;
    const double *exp = (const double *)rhs;
    double *dst = (double *)out;
    size_t remaining = n;
    while (remaining > 0) {
        int chunk = (remaining > (size_t)INT32_MAX) ? INT32_MAX : (int)remaining;
        vvpow(dst, exp, base, &chunk);
        size_t step = (size_t)chunk;
        dst += step;
        base += step;
        exp += step;
        remaining -= step;
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_ew_f64_accelerate_mod(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    (void)ctx;
    if (cpu_ew_accelerate_f64_use_scalar(n)) {
        return cpu_ew_accelerate_f64_scalar_binary(lhs, rhs, out, n, cpu_ew_accelerate_f64_mod_scalar);
    }
    return cpu_ew_accelerate_f64_vforce_binary(vvfmod, (double *)out, (const double *)lhs, (const double *)rhs, n);
}

static marmot_error_t
cpu_ew_f64_accelerate_min(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    (void)ctx;
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (cpu_ew_accelerate_f64_use_scalar(n)) {
        return cpu_ew_accelerate_f64_scalar_binary(lhs, rhs, out, n, cpu_ew_accelerate_f64_min_scalar);
    }
    vDSP_vminD((const double *)lhs, 1, (const double *)rhs, 1, (double *)out, 1, (vDSP_Length)n);
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_ew_f64_accelerate_max(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    (void)ctx;
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (cpu_ew_accelerate_f64_use_scalar(n)) {
        return cpu_ew_accelerate_f64_scalar_binary(lhs, rhs, out, n, cpu_ew_accelerate_f64_max_scalar);
    }
    vDSP_vmaxD((const double *)lhs, 1, (const double *)rhs, 1, (double *)out, 1, (vDSP_Length)n);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_ew_f16_accelerate_binary_impl(
    const void *ctx, const void *lhs, const void *rhs, void *out, size_t n, cpu_ew_binary_fn f32_kernel
) {
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    float *lhs_f32 = cpu_ew_accelerate_alloc_f32(n);
    float *rhs_f32 = cpu_ew_accelerate_alloc_f32(n);
    float *out_f32 = cpu_ew_accelerate_alloc_f32(n);
    if (lhs_f32 == nullptr || rhs_f32 == nullptr || out_f32 == nullptr) {
        free(lhs_f32);
        free(rhs_f32);
        free(out_f32);
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Accelerate fp16 bridge allocation failed");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    cpu_convert_f16_to_f32(ctx, lhs_f32, (const marmot_float16_t *)lhs, n);
    cpu_convert_f16_to_f32(ctx, rhs_f32, (const marmot_float16_t *)rhs, n);
    marmot_error_t err = f32_kernel(ctx, lhs_f32, rhs_f32, out_f32, n);
    if (err == MARMOT_SUCCESS) {
        cpu_convert_f32_to_f16(ctx, (marmot_float16_t *)out, out_f32, n);
    }

    free(lhs_f32);
    free(rhs_f32);
    free(out_f32);
    return err;
}

static marmot_error_t cpu_ew_bf16_accelerate_binary_impl(
    const void *ctx, const void *lhs, const void *rhs, void *out, size_t n, cpu_ew_binary_fn f32_kernel
) {
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    float *lhs_f32 = cpu_ew_accelerate_alloc_f32(n);
    float *rhs_f32 = cpu_ew_accelerate_alloc_f32(n);
    float *out_f32 = cpu_ew_accelerate_alloc_f32(n);
    if (lhs_f32 == nullptr || rhs_f32 == nullptr || out_f32 == nullptr) {
        free(lhs_f32);
        free(rhs_f32);
        free(out_f32);
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Accelerate bf16 bridge allocation failed");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    cpu_convert_bf16_to_f32(ctx, lhs_f32, (const marmot_bfloat16_t *)lhs, n);
    cpu_convert_bf16_to_f32(ctx, rhs_f32, (const marmot_bfloat16_t *)rhs, n);
    marmot_error_t err = f32_kernel(ctx, lhs_f32, rhs_f32, out_f32, n);
    if (err == MARMOT_SUCCESS) {
        cpu_convert_f32_to_bf16(ctx, (marmot_bfloat16_t *)out, out_f32, n);
    }

    free(lhs_f32);
    free(rhs_f32);
    free(out_f32);
    return err;
}

static marmot_error_t
cpu_ew_f16_accelerate_add(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    return cpu_ew_f16_accelerate_binary_impl(ctx, lhs, rhs, out, n, cpu_ew_f32_accelerate_add);
}

static marmot_error_t
cpu_ew_f16_accelerate_sub(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    return cpu_ew_f16_accelerate_binary_impl(ctx, lhs, rhs, out, n, cpu_ew_f32_accelerate_sub);
}

static marmot_error_t
cpu_ew_f16_accelerate_mul(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    return cpu_ew_f16_accelerate_binary_impl(ctx, lhs, rhs, out, n, cpu_ew_f32_accelerate_mul);
}

static marmot_error_t
cpu_ew_f16_accelerate_div(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    return cpu_ew_f16_accelerate_binary_impl(ctx, lhs, rhs, out, n, cpu_ew_f32_accelerate_div);
}

static marmot_error_t
cpu_ew_f16_accelerate_pow(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    return cpu_ew_f16_accelerate_binary_impl(ctx, lhs, rhs, out, n, cpu_ew_f32_accelerate_pow);
}

static marmot_error_t
cpu_ew_f16_accelerate_mod(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    return cpu_ew_f16_accelerate_binary_impl(ctx, lhs, rhs, out, n, cpu_ew_f32_accelerate_mod);
}

static marmot_error_t
cpu_ew_f16_accelerate_min(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    return cpu_ew_f16_accelerate_binary_impl(ctx, lhs, rhs, out, n, cpu_ew_f32_accelerate_min);
}

static marmot_error_t
cpu_ew_f16_accelerate_max(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    return cpu_ew_f16_accelerate_binary_impl(ctx, lhs, rhs, out, n, cpu_ew_f32_accelerate_max);
}

static marmot_error_t
cpu_ew_bf16_accelerate_add(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    return cpu_ew_bf16_accelerate_binary_impl(ctx, lhs, rhs, out, n, cpu_ew_f32_accelerate_add);
}

static marmot_error_t
cpu_ew_bf16_accelerate_sub(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    return cpu_ew_bf16_accelerate_binary_impl(ctx, lhs, rhs, out, n, cpu_ew_f32_accelerate_sub);
}

static marmot_error_t
cpu_ew_bf16_accelerate_mul(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    return cpu_ew_bf16_accelerate_binary_impl(ctx, lhs, rhs, out, n, cpu_ew_f32_accelerate_mul);
}

static marmot_error_t
cpu_ew_bf16_accelerate_div(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    return cpu_ew_bf16_accelerate_binary_impl(ctx, lhs, rhs, out, n, cpu_ew_f32_accelerate_div);
}

static marmot_error_t
cpu_ew_bf16_accelerate_pow(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    return cpu_ew_bf16_accelerate_binary_impl(ctx, lhs, rhs, out, n, cpu_ew_f32_accelerate_pow);
}

static marmot_error_t
cpu_ew_bf16_accelerate_mod(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    return cpu_ew_bf16_accelerate_binary_impl(ctx, lhs, rhs, out, n, cpu_ew_f32_accelerate_mod);
}

static marmot_error_t
cpu_ew_bf16_accelerate_min(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    return cpu_ew_bf16_accelerate_binary_impl(ctx, lhs, rhs, out, n, cpu_ew_f32_accelerate_min);
}

static marmot_error_t
cpu_ew_bf16_accelerate_max(const void *ctx, const void *lhs, const void *rhs, void *out, size_t n) {
    return cpu_ew_bf16_accelerate_binary_impl(ctx, lhs, rhs, out, n, cpu_ew_f32_accelerate_max);
}

#define CPU_EW_ACCELERATE_DEFINE_KERNEL(name, suffix, raw_fn)                                                          \
    marmot_error_t cpu_##name##_##suffix##_accelerate(                                                                 \
        const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out               \
    ) {                                                                                                                \
        return raw_fn(device_ctx, a->data, b->data, out->data, marmot_tensor_num_elements(a));                         \
    }

CPU_EW_ACCELERATE_DEFINE_KERNEL(add, f16, cpu_ew_f16_accelerate_add)
CPU_EW_ACCELERATE_DEFINE_KERNEL(sub, f16, cpu_ew_f16_accelerate_sub)
CPU_EW_ACCELERATE_DEFINE_KERNEL(mul, f16, cpu_ew_f16_accelerate_mul)
CPU_EW_ACCELERATE_DEFINE_KERNEL(div, f16, cpu_ew_f16_accelerate_div)
CPU_EW_ACCELERATE_DEFINE_KERNEL(min, f16, cpu_ew_f16_accelerate_min)
CPU_EW_ACCELERATE_DEFINE_KERNEL(max, f16, cpu_ew_f16_accelerate_max)
CPU_EW_ACCELERATE_DEFINE_KERNEL(pow, f16, cpu_ew_f16_accelerate_pow)
CPU_EW_ACCELERATE_DEFINE_KERNEL(mod, f16, cpu_ew_f16_accelerate_mod)

CPU_EW_ACCELERATE_DEFINE_KERNEL(add, f32, cpu_ew_f32_accelerate_add)
CPU_EW_ACCELERATE_DEFINE_KERNEL(sub, f32, cpu_ew_f32_accelerate_sub)
CPU_EW_ACCELERATE_DEFINE_KERNEL(mul, f32, cpu_ew_f32_accelerate_mul)
CPU_EW_ACCELERATE_DEFINE_KERNEL(div, f32, cpu_ew_f32_accelerate_div)
CPU_EW_ACCELERATE_DEFINE_KERNEL(min, f32, cpu_ew_f32_accelerate_min)
CPU_EW_ACCELERATE_DEFINE_KERNEL(max, f32, cpu_ew_f32_accelerate_max)
CPU_EW_ACCELERATE_DEFINE_KERNEL(pow, f32, cpu_ew_f32_accelerate_pow)
CPU_EW_ACCELERATE_DEFINE_KERNEL(mod, f32, cpu_ew_f32_accelerate_mod)

CPU_EW_ACCELERATE_DEFINE_KERNEL(add, bf16, cpu_ew_bf16_accelerate_add)
CPU_EW_ACCELERATE_DEFINE_KERNEL(sub, bf16, cpu_ew_bf16_accelerate_sub)
CPU_EW_ACCELERATE_DEFINE_KERNEL(mul, bf16, cpu_ew_bf16_accelerate_mul)
CPU_EW_ACCELERATE_DEFINE_KERNEL(div, bf16, cpu_ew_bf16_accelerate_div)
CPU_EW_ACCELERATE_DEFINE_KERNEL(min, bf16, cpu_ew_bf16_accelerate_min)
CPU_EW_ACCELERATE_DEFINE_KERNEL(max, bf16, cpu_ew_bf16_accelerate_max)
CPU_EW_ACCELERATE_DEFINE_KERNEL(pow, bf16, cpu_ew_bf16_accelerate_pow)
CPU_EW_ACCELERATE_DEFINE_KERNEL(mod, bf16, cpu_ew_bf16_accelerate_mod)

CPU_EW_ACCELERATE_DEFINE_KERNEL(add, f64, cpu_ew_f64_accelerate_add)
CPU_EW_ACCELERATE_DEFINE_KERNEL(sub, f64, cpu_ew_f64_accelerate_sub)
CPU_EW_ACCELERATE_DEFINE_KERNEL(mul, f64, cpu_ew_f64_accelerate_mul)
CPU_EW_ACCELERATE_DEFINE_KERNEL(div, f64, cpu_ew_f64_accelerate_div)
CPU_EW_ACCELERATE_DEFINE_KERNEL(min, f64, cpu_ew_f64_accelerate_min)
CPU_EW_ACCELERATE_DEFINE_KERNEL(max, f64, cpu_ew_f64_accelerate_max)
CPU_EW_ACCELERATE_DEFINE_KERNEL(pow, f64, cpu_ew_f64_accelerate_pow)
CPU_EW_ACCELERATE_DEFINE_KERNEL(mod, f64, cpu_ew_f64_accelerate_mod)

#undef CPU_EW_ACCELERATE_DEFINE_KERNEL

#endif // MARMOT_ENABLE_ACCELERATE
