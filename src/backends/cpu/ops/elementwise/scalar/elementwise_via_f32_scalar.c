#include "marmot/tensor.h"

#include <math.h>

#include "cpu_backend_internal.h"

// -----------------------------------------------------------------------------
// Scalar float16/bfloat16/fp8 kernels via float32
// -----------------------------------------------------------------------------

static inline float cpu_silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

static inline float cpu_gelu_tanh_f32(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static inline bool cpu_row_strided_2d(
    const marmot_tensor_t *a, const marmot_tensor_t *b, const marmot_tensor_t *out, size_t *rows, size_t *cols,
    size_t *a_row_stride, size_t *b_row_stride, size_t *out_row_stride
) {
    if (a->shape.ndim != 2 || b->shape.ndim != 2 || out->shape.ndim != 2) {
        return false;
    }
    if (a->shape.shape[0] != b->shape.shape[0] || a->shape.shape[1] != b->shape.shape[1] ||
        a->shape.shape[0] != out->shape.shape[0] || a->shape.shape[1] != out->shape.shape[1]) {
        return false;
    }
    if (a->shape.strides[1] != 1 || b->shape.strides[1] != 1 || out->shape.strides[1] != 1) {
        return false;
    }
    if (a->shape.strides[0] < a->shape.shape[1] || b->shape.strides[0] < b->shape.shape[1] ||
        out->shape.strides[0] < out->shape.shape[1]) {
        return false;
    }
    *rows = a->shape.shape[0];
    *cols = a->shape.shape[1];
    *a_row_stride = a->shape.strides[0];
    *b_row_stride = b->shape.strides[0];
    *out_row_stride = out->shape.strides[0];
    return true;
}

#define CPU_EW_DEFINE_VIA_F32_BIN(op_name, suffix, dtype_enum, expr)                                                   \
    marmot_error_t cpu_##op_name##_##suffix##_scalar(                                                                  \
        const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out               \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        const marmot_dtype_t dtype = MARMOT_DTYPE_##dtype_enum;                                                        \
        size_t rows = 0;                                                                                               \
        size_t cols = 0;                                                                                               \
        size_t a_row_stride = 0;                                                                                       \
        size_t b_row_stride = 0;                                                                                       \
        size_t out_row_stride = 0;                                                                                     \
        if (cpu_row_strided_2d(a, b, out, &rows, &cols, &a_row_stride, &b_row_stride, &out_row_stride) &&              \
            (a_row_stride != cols || b_row_stride != cols || out_row_stride != cols)) {                                \
            const void *lhs = a->data;                                                                                 \
            const void *rhs = b->data;                                                                                 \
            void *dst = out->data;                                                                                     \
            for (size_t row = 0; row < rows; ++row) {                                                                  \
                const size_t a_base = row * a_row_stride;                                                              \
                const size_t b_base = row * b_row_stride;                                                              \
                const size_t out_base = row * out_row_stride;                                                          \
                for (size_t col = 0; col < cols; ++col) {                                                              \
                    float av = cpu_load_as_f32(dtype, lhs, a_base + col);                                              \
                    float bv = cpu_load_as_f32(dtype, rhs, b_base + col);                                              \
                    cpu_store_from_f32(dtype, dst, out_base + col, (expr));                                            \
                }                                                                                                      \
            }                                                                                                          \
        } else {                                                                                                       \
            const size_t n = marmot_tensor_num_elements(a);                                                            \
            const void *lhs = a->data;                                                                                 \
            const void *rhs = b->data;                                                                                 \
            void *dst = out->data;                                                                                     \
            for (size_t i = 0; i < n; ++i) {                                                                           \
                float av = cpu_load_as_f32(dtype, lhs, i);                                                             \
                float bv = cpu_load_as_f32(dtype, rhs, i);                                                             \
                cpu_store_from_f32(dtype, dst, i, (expr));                                                             \
            }                                                                                                          \
        }                                                                                                              \
        return MARMOT_SUCCESS;                                                                                         \
    }

#define CPU_EW_DEFINE_VIA_F32_CMP(op_name, suffix, dtype_enum, pred)                                                   \
    marmot_error_t cpu_##op_name##_##suffix##_scalar(                                                                  \
        const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out               \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        const marmot_dtype_t dtype = MARMOT_DTYPE_##dtype_enum;                                                        \
        size_t rows = 0;                                                                                               \
        size_t cols = 0;                                                                                               \
        size_t a_row_stride = 0;                                                                                       \
        size_t b_row_stride = 0;                                                                                       \
        size_t out_row_stride = 0;                                                                                     \
        if (cpu_row_strided_2d(a, b, out, &rows, &cols, &a_row_stride, &b_row_stride, &out_row_stride) &&              \
            (a_row_stride != cols || b_row_stride != cols || out_row_stride != cols)) {                                \
            const void *lhs = a->data;                                                                                 \
            const void *rhs = b->data;                                                                                 \
            uint8_t *dst = (uint8_t *)out->data;                                                                       \
            for (size_t row = 0; row < rows; ++row) {                                                                  \
                const size_t a_base = row * a_row_stride;                                                              \
                const size_t b_base = row * b_row_stride;                                                              \
                const size_t out_base = row * out_row_stride;                                                          \
                for (size_t col = 0; col < cols; ++col) {                                                              \
                    float av = cpu_load_as_f32(dtype, lhs, a_base + col);                                              \
                    float bv = cpu_load_as_f32(dtype, rhs, b_base + col);                                              \
                    dst[out_base + col] = (uint8_t)(pred);                                                             \
                }                                                                                                      \
            }                                                                                                          \
        } else {                                                                                                       \
            const size_t n = marmot_tensor_num_elements(a);                                                            \
            const void *lhs = a->data;                                                                                 \
            const void *rhs = b->data;                                                                                 \
            uint8_t *dst = (uint8_t *)out->data;                                                                       \
            for (size_t i = 0; i < n; ++i) {                                                                           \
                float av = cpu_load_as_f32(dtype, lhs, i);                                                             \
                float bv = cpu_load_as_f32(dtype, rhs, i);                                                             \
                dst[i] = (uint8_t)(pred);                                                                              \
            }                                                                                                          \
        }                                                                                                              \
        return MARMOT_SUCCESS;                                                                                         \
    }

CPU_EW_DEFINE_VIA_F32_BIN(add, f16, FLOAT16, av + bv)
CPU_EW_DEFINE_VIA_F32_BIN(sub, f16, FLOAT16, av - bv)
CPU_EW_DEFINE_VIA_F32_BIN(mul, f16, FLOAT16, av *bv)
CPU_EW_DEFINE_VIA_F32_BIN(swiglu, f16, FLOAT16, cpu_silu_f32(av) * bv)
CPU_EW_DEFINE_VIA_F32_BIN(geglu, f16, FLOAT16, cpu_gelu_tanh_f32(av) * bv)
CPU_EW_DEFINE_VIA_F32_BIN(div, f16, FLOAT16, av / bv)
CPU_EW_DEFINE_VIA_F32_BIN(min, f16, FLOAT16, fminf(av, bv))
CPU_EW_DEFINE_VIA_F32_BIN(max, f16, FLOAT16, fmaxf(av, bv))
CPU_EW_DEFINE_VIA_F32_BIN(pow, f16, FLOAT16, powf(av, bv))
CPU_EW_DEFINE_VIA_F32_BIN(mod, f16, FLOAT16, fmodf(av, bv))
CPU_EW_DEFINE_VIA_F32_CMP(compare_eq, f16, FLOAT16, av == bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_ne, f16, FLOAT16, av != bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_lt, f16, FLOAT16, av < bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_le, f16, FLOAT16, av <= bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_gt, f16, FLOAT16, av > bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_ge, f16, FLOAT16, av >= bv)

CPU_EW_DEFINE_VIA_F32_BIN(add, bf16, BFLOAT16, av + bv)
CPU_EW_DEFINE_VIA_F32_BIN(sub, bf16, BFLOAT16, av - bv)
CPU_EW_DEFINE_VIA_F32_BIN(mul, bf16, BFLOAT16, av *bv)
CPU_EW_DEFINE_VIA_F32_BIN(swiglu, bf16, BFLOAT16, cpu_silu_f32(av) * bv)
CPU_EW_DEFINE_VIA_F32_BIN(geglu, bf16, BFLOAT16, cpu_gelu_tanh_f32(av) * bv)
CPU_EW_DEFINE_VIA_F32_BIN(div, bf16, BFLOAT16, av / bv)
CPU_EW_DEFINE_VIA_F32_BIN(min, bf16, BFLOAT16, fminf(av, bv))
CPU_EW_DEFINE_VIA_F32_BIN(max, bf16, BFLOAT16, fmaxf(av, bv))
CPU_EW_DEFINE_VIA_F32_BIN(pow, bf16, BFLOAT16, powf(av, bv))
CPU_EW_DEFINE_VIA_F32_BIN(mod, bf16, BFLOAT16, fmodf(av, bv))
CPU_EW_DEFINE_VIA_F32_CMP(compare_eq, bf16, BFLOAT16, av == bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_ne, bf16, BFLOAT16, av != bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_lt, bf16, BFLOAT16, av < bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_le, bf16, BFLOAT16, av <= bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_gt, bf16, BFLOAT16, av > bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_ge, bf16, BFLOAT16, av >= bv)

#if MARMOT_ENABLE_FP8
CPU_EW_DEFINE_VIA_F32_BIN(add, fp8_e4m3, FLOAT8_E4M3, av + bv)
CPU_EW_DEFINE_VIA_F32_BIN(sub, fp8_e4m3, FLOAT8_E4M3, av - bv)
CPU_EW_DEFINE_VIA_F32_BIN(mul, fp8_e4m3, FLOAT8_E4M3, av *bv)
CPU_EW_DEFINE_VIA_F32_BIN(swiglu, fp8_e4m3, FLOAT8_E4M3, cpu_silu_f32(av) * bv)
CPU_EW_DEFINE_VIA_F32_BIN(geglu, fp8_e4m3, FLOAT8_E4M3, cpu_gelu_tanh_f32(av) * bv)
CPU_EW_DEFINE_VIA_F32_BIN(div, fp8_e4m3, FLOAT8_E4M3, av / bv)
CPU_EW_DEFINE_VIA_F32_BIN(min, fp8_e4m3, FLOAT8_E4M3, fminf(av, bv))
CPU_EW_DEFINE_VIA_F32_BIN(max, fp8_e4m3, FLOAT8_E4M3, fmaxf(av, bv))
CPU_EW_DEFINE_VIA_F32_BIN(pow, fp8_e4m3, FLOAT8_E4M3, powf(av, bv))
CPU_EW_DEFINE_VIA_F32_BIN(mod, fp8_e4m3, FLOAT8_E4M3, fmodf(av, bv))
CPU_EW_DEFINE_VIA_F32_CMP(compare_eq, fp8_e4m3, FLOAT8_E4M3, av == bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_ne, fp8_e4m3, FLOAT8_E4M3, av != bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_lt, fp8_e4m3, FLOAT8_E4M3, av < bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_le, fp8_e4m3, FLOAT8_E4M3, av <= bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_gt, fp8_e4m3, FLOAT8_E4M3, av > bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_ge, fp8_e4m3, FLOAT8_E4M3, av >= bv)

CPU_EW_DEFINE_VIA_F32_BIN(add, fp8_e5m2, FLOAT8_E5M2, av + bv)
CPU_EW_DEFINE_VIA_F32_BIN(sub, fp8_e5m2, FLOAT8_E5M2, av - bv)
CPU_EW_DEFINE_VIA_F32_BIN(mul, fp8_e5m2, FLOAT8_E5M2, av *bv)
CPU_EW_DEFINE_VIA_F32_BIN(swiglu, fp8_e5m2, FLOAT8_E5M2, cpu_silu_f32(av) * bv)
CPU_EW_DEFINE_VIA_F32_BIN(geglu, fp8_e5m2, FLOAT8_E5M2, cpu_gelu_tanh_f32(av) * bv)
CPU_EW_DEFINE_VIA_F32_BIN(div, fp8_e5m2, FLOAT8_E5M2, av / bv)
CPU_EW_DEFINE_VIA_F32_BIN(min, fp8_e5m2, FLOAT8_E5M2, fminf(av, bv))
CPU_EW_DEFINE_VIA_F32_BIN(max, fp8_e5m2, FLOAT8_E5M2, fmaxf(av, bv))
CPU_EW_DEFINE_VIA_F32_BIN(pow, fp8_e5m2, FLOAT8_E5M2, powf(av, bv))
CPU_EW_DEFINE_VIA_F32_BIN(mod, fp8_e5m2, FLOAT8_E5M2, fmodf(av, bv))
CPU_EW_DEFINE_VIA_F32_CMP(compare_eq, fp8_e5m2, FLOAT8_E5M2, av == bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_ne, fp8_e5m2, FLOAT8_E5M2, av != bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_lt, fp8_e5m2, FLOAT8_E5M2, av < bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_le, fp8_e5m2, FLOAT8_E5M2, av <= bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_gt, fp8_e5m2, FLOAT8_E5M2, av > bv)
CPU_EW_DEFINE_VIA_F32_CMP(compare_ge, fp8_e5m2, FLOAT8_E5M2, av >= bv)
#endif

#undef CPU_EW_DEFINE_VIA_F32_CMP
#undef CPU_EW_DEFINE_VIA_F32_BIN
