#include "ops/softmax/softmax_kernels.h"

extern const cpu_softmax_traits_t cpu_softmax_f64_scalar_traits;
extern const cpu_softmax_traits_t cpu_softmax_f32_scalar_traits;
extern const cpu_softmax_traits_t cpu_softmax_f16_scalar_traits;
extern const cpu_softmax_traits_t cpu_softmax_bf16_scalar_traits;
#if MARMOT_ENABLE_FP8
extern const cpu_softmax_traits_t cpu_softmax_fp8_e4m3_scalar_traits;
extern const cpu_softmax_traits_t cpu_softmax_fp8_e5m2_scalar_traits;
#endif
#if HAS_NEON
extern const cpu_softmax_traits_t cpu_softmax_f32_neon_traits;
extern const cpu_softmax_traits_t cpu_softmax_f16_neon_traits;
#endif
#if HAS_AVX2
extern const cpu_softmax_traits_t cpu_softmax_f32_avx2_traits;
#endif

static const cpu_softmax_traits_t *const k_cpu_softmax_scalar_ops[MARMOT_DTYPE_COUNT] = {
    [MARMOT_DTYPE_FLOAT64] = &cpu_softmax_f64_scalar_traits,
    [MARMOT_DTYPE_FLOAT32] = &cpu_softmax_f32_scalar_traits,
    [MARMOT_DTYPE_FLOAT16] = &cpu_softmax_f16_scalar_traits,
    [MARMOT_DTYPE_BFLOAT16] = &cpu_softmax_bf16_scalar_traits,
#if MARMOT_ENABLE_FP8
    [MARMOT_DTYPE_FLOAT8_E4M3] = &cpu_softmax_fp8_e4m3_scalar_traits,
    [MARMOT_DTYPE_FLOAT8_E5M2] = &cpu_softmax_fp8_e5m2_scalar_traits,
#endif
};

#if HAS_NEON
static const cpu_softmax_traits_t *const k_cpu_softmax_neon_ops[MARMOT_DTYPE_COUNT] = {
    [MARMOT_DTYPE_FLOAT32] = &cpu_softmax_f32_neon_traits,
    [MARMOT_DTYPE_FLOAT16] = &cpu_softmax_f16_neon_traits,
};
#endif

#if HAS_AVX2
static const cpu_softmax_traits_t *const k_cpu_softmax_avx2_ops[MARMOT_DTYPE_COUNT] = {
    [MARMOT_DTYPE_FLOAT32] = &cpu_softmax_f32_avx2_traits,
};
#endif

static cpu_softmax_kernel_fn cpu_softmax_resolve_kernel(const void *device_ctx, marmot_dtype_t dtype) {
#if HAS_AVX2
    if (has_avx2(device_ctx)) {
        const cpu_softmax_traits_t *traits = k_cpu_softmax_avx2_ops[dtype];
        if (traits != nullptr && traits->fn != nullptr) {
            return traits->fn;
        }
    }
#endif

#if HAS_NEON
    if (has_neon(device_ctx)) {
        const cpu_softmax_traits_t *traits = k_cpu_softmax_neon_ops[dtype];
        if (traits != nullptr && traits->fn != nullptr) {
            return traits->fn;
        }
    }
#endif

    const cpu_softmax_traits_t *traits = k_cpu_softmax_scalar_ops[dtype];
    return traits != nullptr ? traits->fn : nullptr;
}

marmot_error_t cpu_softmax_impl(const void *device_ctx, const marmot_softmax_desc_t *desc) {
    if (desc == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Softmax descriptor is null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (desc->x == nullptr || desc->out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Softmax input or output is null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    cpu_context_t *ctx = (cpu_context_t *)device_ctx;
    if (ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "CPU context is null for softmax");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_tensor_t *x = desc->x;
    marmot_tensor_t *out = desc->out;
    const int32_t axis = desc->axis;

    marmot_softmax_shape_t shape;
    marmot_error_t status = marmot_softmax_prepare(x, out, axis, &shape);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    cpu_softmax_kernel_fn kernel = cpu_softmax_resolve_kernel(ctx, x->dtype);
    if (kernel == nullptr) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Softmax unsupported for dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    return kernel(ctx, x, &shape, out);
}
