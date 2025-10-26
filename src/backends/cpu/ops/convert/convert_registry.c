#include "convert_registry.h"

#include <stdbool.h>

#include "cpu_backend_internal.h"

extern const cpu_convert_traits_t cpu_convert_f32_to_f16_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_f16_to_f32_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_f32_to_bf16_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_bf16_to_f32_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_f16_to_bf16_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_bf16_to_f16_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_f32_to_f64_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_f64_to_f32_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_f32_to_i64_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_i64_to_f32_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_f64_to_i64_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_i64_to_f64_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_f64_to_f16_custom_traits;
extern const cpu_convert_traits_t cpu_convert_f64_to_bf16_custom_traits;
extern const cpu_convert_traits_t cpu_convert_f16_to_f64_custom_traits;
extern const cpu_convert_traits_t cpu_convert_bf16_to_f64_custom_traits;
#if MARMOT_ENABLE_FP8
extern const cpu_convert_traits_t cpu_convert_f32_to_fp8_e4m3_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_fp8_e4m3_to_f32_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_f32_to_fp8_e5m2_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_fp8_e5m2_to_f32_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_f16_to_fp8_e4m3_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_fp8_e4m3_to_f16_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_f16_to_fp8_e5m2_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_fp8_e5m2_to_f16_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_bf16_to_fp8_e4m3_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_fp8_e4m3_to_bf16_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_bf16_to_fp8_e5m2_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_fp8_e5m2_to_bf16_scalar_traits;
extern const cpu_convert_traits_t cpu_convert_f64_to_fp8_e4m3_custom_traits;
extern const cpu_convert_traits_t cpu_convert_f64_to_fp8_e5m2_custom_traits;
extern const cpu_convert_traits_t cpu_convert_fp8_e4m3_to_f64_custom_traits;
extern const cpu_convert_traits_t cpu_convert_fp8_e5m2_to_f64_custom_traits;
#endif
#if MARMOT_ENABLE_ACCELERATE
extern const cpu_convert_traits_t cpu_convert_f32_to_f16_accel_traits;
extern const cpu_convert_traits_t cpu_convert_f16_to_f32_accel_traits;
#endif
#if HAS_NEON
extern const cpu_convert_traits_t cpu_convert_f32_to_f16_neon_traits;
extern const cpu_convert_traits_t cpu_convert_f16_to_f32_neon_traits;
extern const cpu_convert_traits_t cpu_convert_f32_to_bf16_neon_traits;
extern const cpu_convert_traits_t cpu_convert_bf16_to_f32_neon_traits;
extern const cpu_convert_traits_t cpu_convert_f16_to_bf16_neon_traits;
extern const cpu_convert_traits_t cpu_convert_bf16_to_f16_neon_traits;
#endif
#if HAS_AVX2
extern const cpu_convert_traits_t cpu_convert_f32_to_f16_avx2_traits;
extern const cpu_convert_traits_t cpu_convert_f16_to_f32_avx2_traits;
extern const cpu_convert_traits_t cpu_convert_f32_to_bf16_avx2_traits;
extern const cpu_convert_traits_t cpu_convert_bf16_to_f32_avx2_traits;
#endif

#define CPU_CONVERT_ENTRY(dst, src, trait) [dst][src] = &(trait)

static const cpu_convert_traits_t *const k_cpu_convert_scalar_ops[MARMOT_DTYPE_COUNT][MARMOT_DTYPE_COUNT] = {
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT32, cpu_convert_f32_to_f16_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT16, cpu_convert_f16_to_f32_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_BFLOAT16, MARMOT_DTYPE_FLOAT32, cpu_convert_f32_to_bf16_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_BFLOAT16, cpu_convert_bf16_to_f32_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_BFLOAT16, MARMOT_DTYPE_FLOAT16, cpu_convert_f16_to_bf16_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_BFLOAT16, cpu_convert_bf16_to_f16_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT64, MARMOT_DTYPE_FLOAT32, cpu_convert_f32_to_f64_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT64, cpu_convert_f64_to_f32_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_INT64, MARMOT_DTYPE_FLOAT32, cpu_convert_f32_to_i64_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_INT64, cpu_convert_i64_to_f32_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_INT64, MARMOT_DTYPE_FLOAT64, cpu_convert_f64_to_i64_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT64, MARMOT_DTYPE_INT64, cpu_convert_i64_to_f64_scalar_traits),
#if MARMOT_ENABLE_FP8
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT8_E4M3, MARMOT_DTYPE_FLOAT32, cpu_convert_f32_to_fp8_e4m3_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT8_E4M3, cpu_convert_fp8_e4m3_to_f32_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT8_E5M2, MARMOT_DTYPE_FLOAT32, cpu_convert_f32_to_fp8_e5m2_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT8_E5M2, cpu_convert_fp8_e5m2_to_f32_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT8_E4M3, MARMOT_DTYPE_FLOAT16, cpu_convert_f16_to_fp8_e4m3_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT8_E4M3, cpu_convert_fp8_e4m3_to_f16_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT8_E5M2, MARMOT_DTYPE_FLOAT16, cpu_convert_f16_to_fp8_e5m2_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT8_E5M2, cpu_convert_fp8_e5m2_to_f16_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT8_E4M3, MARMOT_DTYPE_BFLOAT16, cpu_convert_bf16_to_fp8_e4m3_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_BFLOAT16, MARMOT_DTYPE_FLOAT8_E4M3, cpu_convert_fp8_e4m3_to_bf16_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT8_E5M2, MARMOT_DTYPE_BFLOAT16, cpu_convert_bf16_to_fp8_e5m2_scalar_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_BFLOAT16, MARMOT_DTYPE_FLOAT8_E5M2, cpu_convert_fp8_e5m2_to_bf16_scalar_traits),
#endif
};

static const cpu_convert_traits_t *const k_cpu_convert_custom_ops[MARMOT_DTYPE_COUNT][MARMOT_DTYPE_COUNT] = {
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT64, cpu_convert_f64_to_f16_custom_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_BFLOAT16, MARMOT_DTYPE_FLOAT64, cpu_convert_f64_to_bf16_custom_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT64, MARMOT_DTYPE_FLOAT16, cpu_convert_f16_to_f64_custom_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT64, MARMOT_DTYPE_BFLOAT16, cpu_convert_bf16_to_f64_custom_traits),
#if MARMOT_ENABLE_FP8
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT8_E4M3, MARMOT_DTYPE_FLOAT64, cpu_convert_f64_to_fp8_e4m3_custom_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT8_E5M2, MARMOT_DTYPE_FLOAT64, cpu_convert_f64_to_fp8_e5m2_custom_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT64, MARMOT_DTYPE_FLOAT8_E4M3, cpu_convert_fp8_e4m3_to_f64_custom_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT64, MARMOT_DTYPE_FLOAT8_E5M2, cpu_convert_fp8_e5m2_to_f64_custom_traits),
#endif
};

#if HAS_NEON
static const cpu_convert_traits_t *const k_cpu_convert_neon_ops[MARMOT_DTYPE_COUNT][MARMOT_DTYPE_COUNT] = {
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT32, cpu_convert_f32_to_f16_neon_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT16, cpu_convert_f16_to_f32_neon_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_BFLOAT16, MARMOT_DTYPE_FLOAT32, cpu_convert_f32_to_bf16_neon_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_BFLOAT16, cpu_convert_bf16_to_f32_neon_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_BFLOAT16, MARMOT_DTYPE_FLOAT16, cpu_convert_f16_to_bf16_neon_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_BFLOAT16, cpu_convert_bf16_to_f16_neon_traits),
};
#endif

#if HAS_AVX2
static const cpu_convert_traits_t *const k_cpu_convert_avx2_ops[MARMOT_DTYPE_COUNT][MARMOT_DTYPE_COUNT] = {
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT32, cpu_convert_f32_to_f16_avx2_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT16, cpu_convert_f16_to_f32_avx2_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_BFLOAT16, MARMOT_DTYPE_FLOAT32, cpu_convert_f32_to_bf16_avx2_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_BFLOAT16, cpu_convert_bf16_to_f32_avx2_traits),
};
#endif

#if MARMOT_ENABLE_ACCELERATE
static const cpu_convert_traits_t *const k_cpu_convert_accelerate_ops[MARMOT_DTYPE_COUNT][MARMOT_DTYPE_COUNT] = {
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT32, cpu_convert_f32_to_f16_accel_traits),
    CPU_CONVERT_ENTRY(MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT16, cpu_convert_f16_to_f32_accel_traits),
};
#endif

#undef CPU_CONVERT_ENTRY

static bool cpu_convert_impl_supported(cpu_convert_impl_kind_t kind, const cpu_context_t *ctx) {
    if (ctx == nullptr) {
        return false;
    }
    switch (kind) {
    case CPU_CONVERT_IMPL_SCALAR:
    case CPU_CONVERT_IMPL_CUSTOM:
        return true;
    case CPU_CONVERT_IMPL_NEON:
        return ctx->runtime_caps.has_neon;
    case CPU_CONVERT_IMPL_AVX2:
        return ctx->runtime_caps.has_avx2;
    case CPU_CONVERT_IMPL_ACCELERATE:
        return ctx->runtime_caps.has_accelerate;
    default:
        return false;
    }
}

cpu_convert_fn
cpu_convert_resolve_fn(const cpu_context_t *ctx, marmot_dtype_t dst, marmot_dtype_t src, const char **impl_name) {
    if (impl_name != nullptr) {
        *impl_name = nullptr;
    }
    if (ctx == nullptr || dst >= MARMOT_DTYPE_COUNT || src >= MARMOT_DTYPE_COUNT) {
        return nullptr;
    }

    const cpu_convert_traits_t *selected = nullptr;
    const cpu_convert_traits_t *trait = k_cpu_convert_scalar_ops[dst][src];
    if (trait != nullptr) {
        selected = trait;
    }

#if HAS_NEON
    if (cpu_convert_impl_supported(CPU_CONVERT_IMPL_NEON, ctx)) {
        trait = k_cpu_convert_neon_ops[dst][src];
        if (trait != nullptr) {
            selected = trait;
        }
    }
#endif

#if HAS_AVX2
    if (cpu_convert_impl_supported(CPU_CONVERT_IMPL_AVX2, ctx)) {
        trait = k_cpu_convert_avx2_ops[dst][src];
        if (trait != nullptr) {
            selected = trait;
        }
    }
#endif

#if MARMOT_ENABLE_ACCELERATE
    if (cpu_convert_impl_supported(CPU_CONVERT_IMPL_ACCELERATE, ctx)) {
        trait = k_cpu_convert_accelerate_ops[dst][src];
        if (trait != nullptr) {
            selected = trait;
        }
    }
#endif

    if (cpu_convert_impl_supported(CPU_CONVERT_IMPL_CUSTOM, ctx)) {
        trait = k_cpu_convert_custom_ops[dst][src];
        if (trait != nullptr) {
            selected = trait;
        }
    }

    if (impl_name != nullptr && selected != nullptr) {
        *impl_name = selected->impl_name;
    }
    return selected != nullptr ? selected->fn : nullptr;
}
