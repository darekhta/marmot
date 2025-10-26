#include <math.h>

#include "cpu_backend_internal.h"

typedef float (*cpu_activation_compute_fn)(float, const marmot_activation_params_t *);
typedef double (*cpu_unary_simple_float_fn)(double);

static inline bool cpu_unary_dtype_is_float(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT64:
    case MARMOT_DTYPE_FLOAT32:
    case MARMOT_DTYPE_FLOAT16:
    case MARMOT_DTYPE_BFLOAT16:
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
    case MARMOT_DTYPE_FLOAT8_E5M2:
#endif
        return true;
    default:
        return false;
    }
}

static inline double cpu_unary_abs_eval(double x) {
    return fabs(x);
}

static inline double cpu_unary_neg_eval(double x) {
    return -x;
}

static inline double cpu_unary_sqrt_eval(double x) {
    return sqrt(x);
}

static inline double cpu_unary_exp_eval(double x) {
    return exp(x);
}

static inline double cpu_unary_log_eval(double x) {
    return log(x);
}

static marmot_error_t
cpu_unary_apply_simple_float(marmot_dtype_t dtype, const void *x, void *out, size_t n, cpu_unary_simple_float_fn fn) {
    if (!cpu_unary_dtype_is_float(dtype)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unary operation requires floating-point dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    for (size_t i = 0; i < n; ++i) {
        double value = cpu_load_as_f64(dtype, x, i);
        cpu_store_from_f64(dtype, out, i, fn(value));
    }
    return MARMOT_SUCCESS;
}

#define CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(name, suffix, dtype_enum, fn)                                                 \
    static marmot_error_t cpu_unary_##name##_##suffix(const void *device_ctx, const void *x, void *out, size_t n) {    \
        (void)device_ctx;                                                                                              \
        return cpu_unary_apply_simple_float((dtype_enum), x, out, n, (fn));                                            \
    }

CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(abs, f64, MARMOT_DTYPE_FLOAT64, cpu_unary_abs_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(abs, f32, MARMOT_DTYPE_FLOAT32, cpu_unary_abs_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(abs, f16, MARMOT_DTYPE_FLOAT16, cpu_unary_abs_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(abs, bf16, MARMOT_DTYPE_BFLOAT16, cpu_unary_abs_eval)
#if MARMOT_ENABLE_FP8
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(abs, fp8_e4m3, MARMOT_DTYPE_FLOAT8_E4M3, cpu_unary_abs_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(abs, fp8_e5m2, MARMOT_DTYPE_FLOAT8_E5M2, cpu_unary_abs_eval)
#endif

CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(neg, f64, MARMOT_DTYPE_FLOAT64, cpu_unary_neg_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(neg, f32, MARMOT_DTYPE_FLOAT32, cpu_unary_neg_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(neg, f16, MARMOT_DTYPE_FLOAT16, cpu_unary_neg_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(neg, bf16, MARMOT_DTYPE_BFLOAT16, cpu_unary_neg_eval)
#if MARMOT_ENABLE_FP8
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(neg, fp8_e4m3, MARMOT_DTYPE_FLOAT8_E4M3, cpu_unary_neg_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(neg, fp8_e5m2, MARMOT_DTYPE_FLOAT8_E5M2, cpu_unary_neg_eval)
#endif

CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(sqrt, f64, MARMOT_DTYPE_FLOAT64, cpu_unary_sqrt_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(sqrt, f32, MARMOT_DTYPE_FLOAT32, cpu_unary_sqrt_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(sqrt, f16, MARMOT_DTYPE_FLOAT16, cpu_unary_sqrt_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(sqrt, bf16, MARMOT_DTYPE_BFLOAT16, cpu_unary_sqrt_eval)
#if MARMOT_ENABLE_FP8
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(sqrt, fp8_e4m3, MARMOT_DTYPE_FLOAT8_E4M3, cpu_unary_sqrt_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(sqrt, fp8_e5m2, MARMOT_DTYPE_FLOAT8_E5M2, cpu_unary_sqrt_eval)
#endif

CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(exp, f64, MARMOT_DTYPE_FLOAT64, cpu_unary_exp_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(exp, f32, MARMOT_DTYPE_FLOAT32, cpu_unary_exp_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(exp, f16, MARMOT_DTYPE_FLOAT16, cpu_unary_exp_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(exp, bf16, MARMOT_DTYPE_BFLOAT16, cpu_unary_exp_eval)
#if MARMOT_ENABLE_FP8
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(exp, fp8_e4m3, MARMOT_DTYPE_FLOAT8_E4M3, cpu_unary_exp_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(exp, fp8_e5m2, MARMOT_DTYPE_FLOAT8_E5M2, cpu_unary_exp_eval)
#endif

CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(log, f64, MARMOT_DTYPE_FLOAT64, cpu_unary_log_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(log, f32, MARMOT_DTYPE_FLOAT32, cpu_unary_log_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(log, f16, MARMOT_DTYPE_FLOAT16, cpu_unary_log_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(log, bf16, MARMOT_DTYPE_BFLOAT16, cpu_unary_log_eval)
#if MARMOT_ENABLE_FP8
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(log, fp8_e4m3, MARMOT_DTYPE_FLOAT8_E4M3, cpu_unary_log_eval)
CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP(log, fp8_e5m2, MARMOT_DTYPE_FLOAT8_E5M2, cpu_unary_log_eval)
#endif

#undef CPU_UNARY_DEFINE_SIMPLE_FLOAT_OP

#define CPU_UNARY_DEFINE_INT_NEG(name, suffix, struct_t, base_t, min_const)                                            \
    static marmot_error_t cpu_unary_neg_##suffix(const void *device_ctx, const void *x, void *out, size_t n) {         \
        (void)device_ctx;                                                                                              \
        const struct_t *src = (const struct_t *)x;                                                                     \
        struct_t *dst = (struct_t *)out;                                                                               \
        for (size_t i = 0; i < n; ++i) {                                                                               \
            base_t v = src[i].value;                                                                                   \
            dst[i].value = (v == (min_const)) ? (min_const) : (base_t)(-v);                                            \
        }                                                                                                              \
        return MARMOT_SUCCESS;                                                                                         \
    }

#define CPU_UNARY_DEFINE_INT_ABS(name, suffix, struct_t, base_t, min_const)                                            \
    static marmot_error_t cpu_unary_abs_##suffix(const void *device_ctx, const void *x, void *out, size_t n) {         \
        (void)device_ctx;                                                                                              \
        const struct_t *src = (const struct_t *)x;                                                                     \
        struct_t *dst = (struct_t *)out;                                                                               \
        for (size_t i = 0; i < n; ++i) {                                                                               \
            base_t v = src[i].value;                                                                                   \
            if (v < 0) {                                                                                               \
                dst[i].value = (v == (min_const)) ? (min_const) : (base_t)(-v);                                        \
            } else {                                                                                                   \
                dst[i].value = v;                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        return MARMOT_SUCCESS;                                                                                         \
    }

#define CPU_UNARY_DEFINE_UINT_ABS(name, suffix, struct_t, base_t)                                                      \
    static marmot_error_t cpu_unary_abs_##suffix(const void *device_ctx, const void *x, void *out, size_t n) {         \
        (void)device_ctx;                                                                                              \
        const struct_t *src = (const struct_t *)x;                                                                     \
        struct_t *dst = (struct_t *)out;                                                                               \
        for (size_t i = 0; i < n; ++i) {                                                                               \
            dst[i].value = (base_t)src[i].value;                                                                       \
        }                                                                                                              \
        return MARMOT_SUCCESS;                                                                                         \
    }

#define CPU_UNARY_DEFINE_BITWISE_NOT(suffix, struct_t, base_t)                                                         \
    static marmot_error_t cpu_unary_bitwise_not_##suffix(const void *device_ctx, const void *x, void *out, size_t n) { \
        (void)device_ctx;                                                                                              \
        const struct_t *src = (const struct_t *)x;                                                                     \
        struct_t *dst = (struct_t *)out;                                                                               \
        for (size_t i = 0; i < n; ++i) {                                                                               \
            dst[i].value = (base_t)(~src[i].value);                                                                    \
        }                                                                                                              \
        return MARMOT_SUCCESS;                                                                                         \
    }

CPU_UNARY_DEFINE_INT_NEG(i8, i8, marmot_int8_t, int8_t, INT8_MIN)
CPU_UNARY_DEFINE_INT_ABS(i8, i8, marmot_int8_t, int8_t, INT8_MIN)
CPU_UNARY_DEFINE_BITWISE_NOT(i8, marmot_int8_t, int8_t)

CPU_UNARY_DEFINE_INT_NEG(i16, i16, marmot_int16_t, int16_t, INT16_MIN)
CPU_UNARY_DEFINE_INT_ABS(i16, i16, marmot_int16_t, int16_t, INT16_MIN)
CPU_UNARY_DEFINE_BITWISE_NOT(i16, marmot_int16_t, int16_t)

CPU_UNARY_DEFINE_INT_NEG(i32, i32, marmot_int32_t, int32_t, INT32_MIN)
CPU_UNARY_DEFINE_INT_ABS(i32, i32, marmot_int32_t, int32_t, INT32_MIN)
CPU_UNARY_DEFINE_BITWISE_NOT(i32, marmot_int32_t, int32_t)

CPU_UNARY_DEFINE_INT_NEG(i64, i64, marmot_int64_t, int64_t, INT64_MIN)
CPU_UNARY_DEFINE_INT_ABS(i64, i64, marmot_int64_t, int64_t, INT64_MIN)
CPU_UNARY_DEFINE_BITWISE_NOT(i64, marmot_int64_t, int64_t)

CPU_UNARY_DEFINE_UINT_ABS(u8, u8, marmot_uint8_t, uint8_t)
CPU_UNARY_DEFINE_BITWISE_NOT(u8, marmot_uint8_t, uint8_t)

CPU_UNARY_DEFINE_UINT_ABS(u16, u16, marmot_uint16_t, uint16_t)
CPU_UNARY_DEFINE_BITWISE_NOT(u16, marmot_uint16_t, uint16_t)

CPU_UNARY_DEFINE_UINT_ABS(u32, u32, marmot_uint32_t, uint32_t)
CPU_UNARY_DEFINE_BITWISE_NOT(u32, marmot_uint32_t, uint32_t)

CPU_UNARY_DEFINE_UINT_ABS(u64, u64, marmot_uint64_t, uint64_t)
CPU_UNARY_DEFINE_BITWISE_NOT(u64, marmot_uint64_t, uint64_t)

#undef CPU_UNARY_DEFINE_INT_NEG
#undef CPU_UNARY_DEFINE_INT_ABS
#undef CPU_UNARY_DEFINE_UINT_ABS
#undef CPU_UNARY_DEFINE_BITWISE_NOT

static inline float relu_scalar_general(float x, const marmot_activation_params_t *params) {
    (void)params;
    return x > 0.0f ? x : 0.0f;
}

static inline float gelu_scalar(float x, const marmot_activation_params_t *params) {
    (void)params;
    const float inv_sqrt2 = 0.7071067811865475f;
    return x * 0.5f * (1.0f + erff(x * inv_sqrt2));
}

static inline float gelu_tanh_scalar(float x, const marmot_activation_params_t *params) {
    (void)params;
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static inline float silu_scalar_fn(float x, const marmot_activation_params_t *params) {
    (void)params;
    return x / (1.0f + expf(-x));
}

static inline float sigmoid_scalar(float x, const marmot_activation_params_t *params) {
    (void)params;
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

static inline float tanh_scalar(float x, const marmot_activation_params_t *params) {
    (void)params;
    return tanhf(x);
}

static inline float mish_scalar(float x, const marmot_activation_params_t *params) {
    (void)params;
    float abs_x = fabsf(x);
    float softplus = log1pf(expf(-abs_x)) + (x > 0.0f ? x : 0.0f);
    return x * tanhf(softplus);
}

static inline float elu_scalar(float x, const marmot_activation_params_t *params) {
    float alpha = params != nullptr ? params->alpha : 1.0f;
    return x >= 0.0f ? x : alpha * (expf(x) - 1.0f);
}

static inline float selu_scalar(float x, const marmot_activation_params_t *params) {
    const float default_alpha = 1.6732632423543772f;
    const float default_lambda = 1.0507009873554804f;
    float alpha = params != nullptr ? params->alpha : default_alpha;
    float lambda = params != nullptr ? params->beta : default_lambda;
    float inner = x >= 0.0f ? x : alpha * (expf(x) - 1.0f);
    return lambda * inner;
}

static inline float leaky_relu_scalar(float x, const marmot_activation_params_t *params) {
    float slope = params != nullptr ? params->alpha : 0.01f;
    return x >= 0.0f ? x : slope * x;
}

static inline float prelu_scalar(float x, const marmot_activation_params_t *params) {
    float slope = params != nullptr ? params->alpha : 0.25f;
    return x >= 0.0f ? x : slope * x;
}

static marmot_error_t cpu_unary_apply_activation_float(
    marmot_dtype_t dtype, const void *x, const marmot_activation_params_t *params, void *out, size_t n,
    cpu_activation_compute_fn fn
) {
    if (!cpu_unary_dtype_is_float(dtype)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Activation requires floating-point dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    for (size_t i = 0; i < n; ++i) {
        float value = cpu_load_as_f32(dtype, x, i);
        float result = fn(value, params);
        cpu_store_from_f32(dtype, out, i, result);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_apply_activation_float_with_bias(
    marmot_dtype_t dtype, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n, cpu_activation_compute_fn fn
) {
    if (!cpu_unary_dtype_is_float(dtype)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Activation requires floating-point dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (bias == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Activation bias pointer cannot be null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!bias_is_scalar && feature_dim == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Activation bias feature dimension cannot be zero");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    for (size_t idx = 0; idx < n; ++idx) {
        size_t bias_index = bias_is_scalar ? 0 : (idx % feature_dim);
        float value = cpu_load_as_f32(dtype, x, idx);
        float bias_value = cpu_load_as_f32(dtype, bias, bias_index);
        float result = fn(value + bias_value, params);
        cpu_store_from_f32(dtype, out, idx, result);
    }
    return MARMOT_SUCCESS;
}

#if HAS_NEON
static void relu_f16_neon(const marmot_float16_t *x, marmot_float16_t *out, size_t n) {
    size_t i = 0;
    const float16x8_t zero = vdupq_n_f16((_Float16)0.0f);
    for (; i + 8 <= n; i += 8) {
        float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16((const uint16_t *)(x + i)));
        float16x8_t vout = vmaxq_f16(vx, zero);
        vst1q_u16((uint16_t *)(out + i), vreinterpretq_u16_f16(vout));
    }
    for (; i < n; ++i) {
        _Float16 value = marmot_float16_to_native(x[i]);
        out[i] = marmot_native_to_float16(value > (_Float16)0.0f ? value : (_Float16)0.0f);
    }
}
#endif

static marmot_error_t cpu_unary_relu_impl(
    const void *device_ctx, marmot_dtype_t dtype, const void *x, const marmot_activation_params_t *params, void *out,
    size_t n
) {
    if (dtype == MARMOT_DTYPE_FLOAT16 && n != 0) {
#if HAS_NEON
        if (has_neon(device_ctx)) {
            relu_f16_neon((const marmot_float16_t *)x, (marmot_float16_t *)out, n);
            return MARMOT_SUCCESS;
        }
#endif
    }
    return cpu_unary_apply_activation_float(dtype, x, params, out, n, relu_scalar_general);
}

static marmot_error_t cpu_unary_fused_bias_relu_impl(
    marmot_dtype_t dtype, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    return cpu_unary_apply_activation_float_with_bias(
        dtype, x, bias, feature_dim, bias_is_scalar, params, out, n, relu_scalar_general
    );
}

#define CPU_UNARY_DEFINE_RELU_FUNCS(dtype_enum, suffix)                                                                \
    static marmot_error_t cpu_unary_relu_##suffix(                                                                     \
        const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n           \
    ) {                                                                                                                \
        return cpu_unary_relu_impl(device_ctx, dtype_enum, x, params, out, n);                                         \
    }                                                                                                                  \
    static marmot_error_t cpu_unary_fused_bias_relu_##suffix(                                                          \
        const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,              \
        const marmot_activation_params_t *params, void *out, size_t n                                                  \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        return cpu_unary_fused_bias_relu_impl(dtype_enum, x, bias, feature_dim, bias_is_scalar, params, out, n);       \
    }

#define CPU_UNARY_DEFINE_FLOAT_ACTIVATION(op_name, fn_name, dtype_enum, suffix)                                        \
    static marmot_error_t cpu_unary_##op_name##_##suffix(                                                              \
        const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n           \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        return cpu_unary_apply_activation_float(dtype_enum, x, params, out, n, fn_name);                               \
    }

#define CPU_UNARY_DEFINE_FLOAT_FUSED(op_name, fn_name, dtype_enum, suffix)                                             \
    static marmot_error_t cpu_unary_fused_bias_##op_name##_##suffix(                                                   \
        const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,              \
        const marmot_activation_params_t *params, void *out, size_t n                                                  \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        return cpu_unary_apply_activation_float_with_bias(                                                             \
            dtype_enum, x, bias, feature_dim, bias_is_scalar, params, out, n, fn_name                                  \
        );                                                                                                             \
    }

#define CPU_UNARY_DEFINE_FLOAT_ACTIVATION_SET(dtype_enum, suffix)                                                      \
    CPU_UNARY_DEFINE_FLOAT_ACTIVATION(gelu, gelu_scalar, dtype_enum, suffix)                                           \
    CPU_UNARY_DEFINE_FLOAT_ACTIVATION(gelu_tanh, gelu_tanh_scalar, dtype_enum, suffix)                                 \
    CPU_UNARY_DEFINE_FLOAT_ACTIVATION(silu, silu_scalar_fn, dtype_enum, suffix)                                        \
    CPU_UNARY_DEFINE_FLOAT_ACTIVATION(sigmoid, sigmoid_scalar, dtype_enum, suffix)                                     \
    CPU_UNARY_DEFINE_FLOAT_ACTIVATION(tanh_act, tanh_scalar, dtype_enum, suffix)                                       \
    CPU_UNARY_DEFINE_FLOAT_ACTIVATION(mish, mish_scalar, dtype_enum, suffix)                                           \
    CPU_UNARY_DEFINE_FLOAT_ACTIVATION(elu, elu_scalar, dtype_enum, suffix)                                             \
    CPU_UNARY_DEFINE_FLOAT_ACTIVATION(selu, selu_scalar, dtype_enum, suffix)                                           \
    CPU_UNARY_DEFINE_FLOAT_ACTIVATION(leaky_relu, leaky_relu_scalar, dtype_enum, suffix)

#define CPU_UNARY_DEFINE_FLOAT_FUSED_SET(dtype_enum, suffix)                                                           \
    CPU_UNARY_DEFINE_FLOAT_FUSED(gelu, gelu_scalar, dtype_enum, suffix)                                                \
    CPU_UNARY_DEFINE_FLOAT_FUSED(gelu_tanh, gelu_tanh_scalar, dtype_enum, suffix)                                      \
    CPU_UNARY_DEFINE_FLOAT_FUSED(silu, silu_scalar_fn, dtype_enum, suffix)                                             \
    CPU_UNARY_DEFINE_FLOAT_FUSED(sigmoid, sigmoid_scalar, dtype_enum, suffix)                                          \
    CPU_UNARY_DEFINE_FLOAT_FUSED(tanh, tanh_scalar, dtype_enum, suffix)                                                \
    CPU_UNARY_DEFINE_FLOAT_FUSED(mish, mish_scalar, dtype_enum, suffix)                                                \
    CPU_UNARY_DEFINE_FLOAT_FUSED(elu, elu_scalar, dtype_enum, suffix)                                                  \
    CPU_UNARY_DEFINE_FLOAT_FUSED(selu, selu_scalar, dtype_enum, suffix)                                                \
    CPU_UNARY_DEFINE_FLOAT_FUSED(leaky_relu, leaky_relu_scalar, dtype_enum, suffix)

#define CPU_UNARY_DEFINE_FLOAT_ACTIVATION_SETS(dtype_enum, suffix)                                                     \
    CPU_UNARY_DEFINE_FLOAT_ACTIVATION_SET(dtype_enum, suffix)                                                          \
    CPU_UNARY_DEFINE_FLOAT_FUSED_SET(dtype_enum, suffix)

CPU_UNARY_DEFINE_RELU_FUNCS(MARMOT_DTYPE_FLOAT64, f64)
CPU_UNARY_DEFINE_FLOAT_ACTIVATION_SETS(MARMOT_DTYPE_FLOAT64, f64)
CPU_UNARY_DEFINE_RELU_FUNCS(MARMOT_DTYPE_FLOAT32, f32)
CPU_UNARY_DEFINE_FLOAT_ACTIVATION_SETS(MARMOT_DTYPE_FLOAT32, f32)
CPU_UNARY_DEFINE_FLOAT_ACTIVATION_SETS(MARMOT_DTYPE_FLOAT16, f16)
CPU_UNARY_DEFINE_RELU_FUNCS(MARMOT_DTYPE_BFLOAT16, bf16)
CPU_UNARY_DEFINE_FLOAT_ACTIVATION_SETS(MARMOT_DTYPE_BFLOAT16, bf16)
#if MARMOT_ENABLE_FP8
CPU_UNARY_DEFINE_RELU_FUNCS(MARMOT_DTYPE_FLOAT8_E4M3, fp8_e4m3)
CPU_UNARY_DEFINE_FLOAT_ACTIVATION_SETS(MARMOT_DTYPE_FLOAT8_E4M3, fp8_e4m3)
CPU_UNARY_DEFINE_RELU_FUNCS(MARMOT_DTYPE_FLOAT8_E5M2, fp8_e5m2)
CPU_UNARY_DEFINE_FLOAT_ACTIVATION_SETS(MARMOT_DTYPE_FLOAT8_E5M2, fp8_e5m2)
#endif

CPU_UNARY_DEFINE_RELU_FUNCS(MARMOT_DTYPE_FLOAT16, f16)

#define CPU_UNARY_DEFINE_PRELU_FUNCS(dtype_enum, suffix)                                                               \
    static marmot_error_t cpu_unary_prelu_##suffix(                                                                    \
        const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n           \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        if (params == nullptr) {                                                                                       \
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "PReLU requires scalar slope parameter");                  \
            return MARMOT_ERROR_INVALID_ARGUMENT;                                                                      \
        }                                                                                                              \
        if (params->parameter_tensor != nullptr) {                                                                     \
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Tensor slopes are not implemented for CPU PReLU");         \
            return MARMOT_ERROR_NOT_IMPLEMENTED;                                                                       \
        }                                                                                                              \
        return cpu_unary_apply_activation_float(dtype_enum, x, params, out, n, prelu_scalar);                          \
    }                                                                                                                  \
    static marmot_error_t cpu_unary_fused_bias_prelu_##suffix(                                                         \
        const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,              \
        const marmot_activation_params_t *params, void *out, size_t n                                                  \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        if (params == nullptr) {                                                                                       \
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "PReLU requires scalar slope parameter");                  \
            return MARMOT_ERROR_INVALID_ARGUMENT;                                                                      \
        }                                                                                                              \
        if (params->parameter_tensor != nullptr) {                                                                     \
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Tensor slopes are not implemented for CPU PReLU");         \
            return MARMOT_ERROR_NOT_IMPLEMENTED;                                                                       \
        }                                                                                                              \
        return cpu_unary_apply_activation_float_with_bias(                                                             \
            dtype_enum, x, bias, feature_dim, bias_is_scalar, params, out, n, prelu_scalar                             \
        );                                                                                                             \
    }

CPU_UNARY_DEFINE_PRELU_FUNCS(MARMOT_DTYPE_FLOAT64, f64)
CPU_UNARY_DEFINE_PRELU_FUNCS(MARMOT_DTYPE_FLOAT32, f32)
CPU_UNARY_DEFINE_PRELU_FUNCS(MARMOT_DTYPE_FLOAT16, f16)
CPU_UNARY_DEFINE_PRELU_FUNCS(MARMOT_DTYPE_BFLOAT16, bf16)
#if MARMOT_ENABLE_FP8
CPU_UNARY_DEFINE_PRELU_FUNCS(MARMOT_DTYPE_FLOAT8_E4M3, fp8_e4m3)
CPU_UNARY_DEFINE_PRELU_FUNCS(MARMOT_DTYPE_FLOAT8_E5M2, fp8_e5m2)
#endif

static marmot_error_t cpu_unary_sign_float(marmot_dtype_t dtype, const void *x, void *out, size_t n) {
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    for (size_t i = 0; i < n; ++i) {
        float value = cpu_load_as_f32(dtype, x, i);
        float sign = (value > 0.0f) - (value < 0.0f);
        cpu_store_from_f32(dtype, out, i, sign);
    }
    return MARMOT_SUCCESS;
}

#define CPU_UNARY_DEFINE_SIGN_FLOAT(dtype_enum, suffix)                                                                \
    static marmot_error_t cpu_unary_sign_##suffix(const void *device_ctx, const void *x, void *out, size_t n) {        \
        (void)device_ctx;                                                                                              \
        if (!cpu_unary_dtype_is_float(dtype_enum)) {                                                                   \
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Sign unsupported for dtype");                            \
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;                                                                     \
        }                                                                                                              \
        return cpu_unary_sign_float(dtype_enum, x, out, n);                                                            \
    }

CPU_UNARY_DEFINE_SIGN_FLOAT(MARMOT_DTYPE_FLOAT64, f64)
CPU_UNARY_DEFINE_SIGN_FLOAT(MARMOT_DTYPE_FLOAT32, f32)
CPU_UNARY_DEFINE_SIGN_FLOAT(MARMOT_DTYPE_FLOAT16, f16)
CPU_UNARY_DEFINE_SIGN_FLOAT(MARMOT_DTYPE_BFLOAT16, bf16)
#if MARMOT_ENABLE_FP8
CPU_UNARY_DEFINE_SIGN_FLOAT(MARMOT_DTYPE_FLOAT8_E4M3, fp8_e4m3)
CPU_UNARY_DEFINE_SIGN_FLOAT(MARMOT_DTYPE_FLOAT8_E5M2, fp8_e5m2)
#endif

#define CPU_UNARY_DEFINE_SIGN_SIGNED(dtype_enum, suffix, ctype, field)                                                 \
    static marmot_error_t cpu_unary_sign_##suffix(const void *device_ctx, const void *x, void *out, size_t n) {        \
        (void)device_ctx;                                                                                              \
        const ctype *src = (const ctype *)x;                                                                           \
        ctype *dst = (ctype *)out;                                                                                     \
        for (size_t i = 0; i < n; ++i) {                                                                               \
            int64_t value = src[i].field;                                                                              \
            dst[i].field = (value > 0) - (value < 0);                                                                  \
        }                                                                                                              \
        return MARMOT_SUCCESS;                                                                                         \
    }

#define CPU_UNARY_DEFINE_SIGN_UNSIGNED(dtype_enum, suffix, ctype, field)                                               \
    static marmot_error_t cpu_unary_sign_##suffix(const void *device_ctx, const void *x, void *out, size_t n) {        \
        (void)device_ctx;                                                                                              \
        const ctype *src = (const ctype *)x;                                                                           \
        ctype *dst = (ctype *)out;                                                                                     \
        for (size_t i = 0; i < n; ++i) {                                                                               \
            dst[i].field = (src[i].field > 0) ? 1 : 0;                                                                 \
        }                                                                                                              \
        return MARMOT_SUCCESS;                                                                                         \
    }

CPU_UNARY_DEFINE_SIGN_SIGNED(MARMOT_DTYPE_INT8, i8, marmot_int8_t, value)
CPU_UNARY_DEFINE_SIGN_SIGNED(MARMOT_DTYPE_INT16, i16, marmot_int16_t, value)
CPU_UNARY_DEFINE_SIGN_SIGNED(MARMOT_DTYPE_INT32, i32, marmot_int32_t, value)
CPU_UNARY_DEFINE_SIGN_SIGNED(MARMOT_DTYPE_INT64, i64, marmot_int64_t, value)
CPU_UNARY_DEFINE_SIGN_UNSIGNED(MARMOT_DTYPE_UINT8, u8, marmot_uint8_t, value)
CPU_UNARY_DEFINE_SIGN_UNSIGNED(MARMOT_DTYPE_UINT16, u16, marmot_uint16_t, value)
CPU_UNARY_DEFINE_SIGN_UNSIGNED(MARMOT_DTYPE_UINT32, u32, marmot_uint32_t, value)
CPU_UNARY_DEFINE_SIGN_UNSIGNED(MARMOT_DTYPE_UINT64, u64, marmot_uint64_t, value)

const cpu_unary_traits_t cpu_unary_f64_scalar_traits = {
    .dtype = MARMOT_DTYPE_FLOAT64,
    .impl_kind = CPU_UNARY_IMPL_SCALAR,
    .ops = {
        .abs = cpu_unary_abs_f64,
        .neg = cpu_unary_neg_f64,
        .sqrt = cpu_unary_sqrt_f64,
        .exp = cpu_unary_exp_f64,
        .log = cpu_unary_log_f64,
        .relu = cpu_unary_relu_f64,
        .gelu = cpu_unary_gelu_f64,
        .gelu_tanh = cpu_unary_gelu_tanh_f64,
        .silu = cpu_unary_silu_f64,
        .sigmoid = cpu_unary_sigmoid_f64,
        .tanh_act = cpu_unary_tanh_act_f64,
        .mish = cpu_unary_mish_f64,
        .elu = cpu_unary_elu_f64,
        .selu = cpu_unary_selu_f64,
        .leaky_relu = cpu_unary_leaky_relu_f64,
        .prelu = cpu_unary_prelu_f64,
        .fused_bias_relu = cpu_unary_fused_bias_relu_f64,
        .fused_bias_gelu = cpu_unary_fused_bias_gelu_f64,
        .fused_bias_gelu_tanh = cpu_unary_fused_bias_gelu_tanh_f64,
        .fused_bias_silu = cpu_unary_fused_bias_silu_f64,
        .fused_bias_sigmoid = cpu_unary_fused_bias_sigmoid_f64,
        .fused_bias_tanh = cpu_unary_fused_bias_tanh_f64,
        .fused_bias_mish = cpu_unary_fused_bias_mish_f64,
        .fused_bias_elu = cpu_unary_fused_bias_elu_f64,
        .fused_bias_selu = cpu_unary_fused_bias_selu_f64,
        .fused_bias_leaky_relu = cpu_unary_fused_bias_leaky_relu_f64,
        .fused_bias_prelu = cpu_unary_fused_bias_prelu_f64,
        .sign = cpu_unary_sign_f64,
        .impl_name = "scalar:f64",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_f64_scalar_traits)

const cpu_unary_traits_t cpu_unary_f32_scalar_traits = {
    .dtype = MARMOT_DTYPE_FLOAT32,
    .impl_kind = CPU_UNARY_IMPL_SCALAR,
    .ops = {
        .abs = cpu_unary_abs_f32,
        .neg = cpu_unary_neg_f32,
        .sqrt = cpu_unary_sqrt_f32,
        .exp = cpu_unary_exp_f32,
        .log = cpu_unary_log_f32,
        .relu = cpu_unary_relu_f32,
        .gelu = cpu_unary_gelu_f32,
        .gelu_tanh = cpu_unary_gelu_tanh_f32,
        .silu = cpu_unary_silu_f32,
        .sigmoid = cpu_unary_sigmoid_f32,
        .tanh_act = cpu_unary_tanh_act_f32,
        .mish = cpu_unary_mish_f32,
        .elu = cpu_unary_elu_f32,
        .selu = cpu_unary_selu_f32,
        .leaky_relu = cpu_unary_leaky_relu_f32,
        .prelu = cpu_unary_prelu_f32,
        .fused_bias_relu = cpu_unary_fused_bias_relu_f32,
        .fused_bias_gelu = cpu_unary_fused_bias_gelu_f32,
        .fused_bias_gelu_tanh = cpu_unary_fused_bias_gelu_tanh_f32,
        .fused_bias_silu = cpu_unary_fused_bias_silu_f32,
        .fused_bias_sigmoid = cpu_unary_fused_bias_sigmoid_f32,
        .fused_bias_tanh = cpu_unary_fused_bias_tanh_f32,
        .fused_bias_mish = cpu_unary_fused_bias_mish_f32,
        .fused_bias_elu = cpu_unary_fused_bias_elu_f32,
        .fused_bias_selu = cpu_unary_fused_bias_selu_f32,
        .fused_bias_leaky_relu = cpu_unary_fused_bias_leaky_relu_f32,
        .fused_bias_prelu = cpu_unary_fused_bias_prelu_f32,
        .sign = cpu_unary_sign_f32,
        .impl_name = "scalar:f32",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_f32_scalar_traits)

const cpu_unary_traits_t cpu_unary_f16_scalar_traits = {
    .dtype = MARMOT_DTYPE_FLOAT16,
    .impl_kind = CPU_UNARY_IMPL_SCALAR,
    .ops = {
        .abs = cpu_unary_abs_f16,
        .neg = cpu_unary_neg_f16,
        .sqrt = cpu_unary_sqrt_f16,
        .exp = cpu_unary_exp_f16,
        .log = cpu_unary_log_f16,
        .relu = cpu_unary_relu_f16,
        .gelu = cpu_unary_gelu_f16,
        .gelu_tanh = cpu_unary_gelu_tanh_f16,
        .silu = cpu_unary_silu_f16,
        .sigmoid = cpu_unary_sigmoid_f16,
        .tanh_act = cpu_unary_tanh_act_f16,
        .mish = cpu_unary_mish_f16,
        .elu = cpu_unary_elu_f16,
        .selu = cpu_unary_selu_f16,
        .leaky_relu = cpu_unary_leaky_relu_f16,
        .prelu = cpu_unary_prelu_f16,
        .fused_bias_relu = cpu_unary_fused_bias_relu_f16,
        .fused_bias_gelu = cpu_unary_fused_bias_gelu_f16,
        .fused_bias_gelu_tanh = cpu_unary_fused_bias_gelu_tanh_f16,
        .fused_bias_silu = cpu_unary_fused_bias_silu_f16,
        .fused_bias_sigmoid = cpu_unary_fused_bias_sigmoid_f16,
        .fused_bias_tanh = cpu_unary_fused_bias_tanh_f16,
        .fused_bias_mish = cpu_unary_fused_bias_mish_f16,
        .fused_bias_elu = cpu_unary_fused_bias_elu_f16,
        .fused_bias_selu = cpu_unary_fused_bias_selu_f16,
        .fused_bias_leaky_relu = cpu_unary_fused_bias_leaky_relu_f16,
        .fused_bias_prelu = cpu_unary_fused_bias_prelu_f16,
        .sign = cpu_unary_sign_f16,
        .impl_name = "scalar:f16",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_f16_scalar_traits)

const cpu_unary_traits_t cpu_unary_bf16_scalar_traits = {
    .dtype = MARMOT_DTYPE_BFLOAT16,
    .impl_kind = CPU_UNARY_IMPL_SCALAR,
    .ops = {
        .abs = cpu_unary_abs_bf16,
        .neg = cpu_unary_neg_bf16,
        .sqrt = cpu_unary_sqrt_bf16,
        .exp = cpu_unary_exp_bf16,
        .log = cpu_unary_log_bf16,
        .relu = cpu_unary_relu_bf16,
        .gelu = cpu_unary_gelu_bf16,
        .gelu_tanh = cpu_unary_gelu_tanh_bf16,
        .silu = cpu_unary_silu_bf16,
        .sigmoid = cpu_unary_sigmoid_bf16,
        .tanh_act = cpu_unary_tanh_act_bf16,
        .mish = cpu_unary_mish_bf16,
        .elu = cpu_unary_elu_bf16,
        .selu = cpu_unary_selu_bf16,
        .leaky_relu = cpu_unary_leaky_relu_bf16,
        .prelu = cpu_unary_prelu_bf16,
        .fused_bias_relu = cpu_unary_fused_bias_relu_bf16,
        .fused_bias_gelu = cpu_unary_fused_bias_gelu_bf16,
        .fused_bias_gelu_tanh = cpu_unary_fused_bias_gelu_tanh_bf16,
        .fused_bias_silu = cpu_unary_fused_bias_silu_bf16,
        .fused_bias_sigmoid = cpu_unary_fused_bias_sigmoid_bf16,
        .fused_bias_tanh = cpu_unary_fused_bias_tanh_bf16,
        .fused_bias_mish = cpu_unary_fused_bias_mish_bf16,
        .fused_bias_elu = cpu_unary_fused_bias_elu_bf16,
        .fused_bias_selu = cpu_unary_fused_bias_selu_bf16,
        .fused_bias_leaky_relu = cpu_unary_fused_bias_leaky_relu_bf16,
        .fused_bias_prelu = cpu_unary_fused_bias_prelu_bf16,
        .sign = cpu_unary_sign_bf16,
        .impl_name = "scalar:bf16",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_bf16_scalar_traits)

#if MARMOT_ENABLE_FP8
const cpu_unary_traits_t cpu_unary_fp8_e4m3_scalar_traits = {
    .dtype = MARMOT_DTYPE_FLOAT8_E4M3,
    .impl_kind = CPU_UNARY_IMPL_SCALAR,
    .ops = {
        .abs = cpu_unary_abs_fp8_e4m3,
        .neg = cpu_unary_neg_fp8_e4m3,
        .sqrt = cpu_unary_sqrt_fp8_e4m3,
        .exp = cpu_unary_exp_fp8_e4m3,
        .log = cpu_unary_log_fp8_e4m3,
        .relu = cpu_unary_relu_fp8_e4m3,
        .gelu = cpu_unary_gelu_fp8_e4m3,
        .gelu_tanh = cpu_unary_gelu_tanh_fp8_e4m3,
        .silu = cpu_unary_silu_fp8_e4m3,
        .sigmoid = cpu_unary_sigmoid_fp8_e4m3,
        .tanh_act = cpu_unary_tanh_act_fp8_e4m3,
        .mish = cpu_unary_mish_fp8_e4m3,
        .elu = cpu_unary_elu_fp8_e4m3,
        .selu = cpu_unary_selu_fp8_e4m3,
        .leaky_relu = cpu_unary_leaky_relu_fp8_e4m3,
        .prelu = cpu_unary_prelu_fp8_e4m3,
        .fused_bias_relu = cpu_unary_fused_bias_relu_fp8_e4m3,
        .fused_bias_gelu = cpu_unary_fused_bias_gelu_fp8_e4m3,
        .fused_bias_gelu_tanh = cpu_unary_fused_bias_gelu_tanh_fp8_e4m3,
        .fused_bias_silu = cpu_unary_fused_bias_silu_fp8_e4m3,
        .fused_bias_sigmoid = cpu_unary_fused_bias_sigmoid_fp8_e4m3,
        .fused_bias_tanh = cpu_unary_fused_bias_tanh_fp8_e4m3,
        .fused_bias_mish = cpu_unary_fused_bias_mish_fp8_e4m3,
        .fused_bias_elu = cpu_unary_fused_bias_elu_fp8_e4m3,
        .fused_bias_selu = cpu_unary_fused_bias_selu_fp8_e4m3,
        .fused_bias_leaky_relu = cpu_unary_fused_bias_leaky_relu_fp8_e4m3,
        .fused_bias_prelu = cpu_unary_fused_bias_prelu_fp8_e4m3,
        .sign = cpu_unary_sign_fp8_e4m3,
        .impl_name = "scalar:fp8_e4m3",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_fp8_e4m3_scalar_traits)

const cpu_unary_traits_t cpu_unary_fp8_e5m2_scalar_traits = {
    .dtype = MARMOT_DTYPE_FLOAT8_E5M2,
    .impl_kind = CPU_UNARY_IMPL_SCALAR,
    .ops = {
        .abs = cpu_unary_abs_fp8_e5m2,
        .neg = cpu_unary_neg_fp8_e5m2,
        .sqrt = cpu_unary_sqrt_fp8_e5m2,
        .exp = cpu_unary_exp_fp8_e5m2,
        .log = cpu_unary_log_fp8_e5m2,
        .relu = cpu_unary_relu_fp8_e5m2,
        .gelu = cpu_unary_gelu_fp8_e5m2,
        .gelu_tanh = cpu_unary_gelu_tanh_fp8_e5m2,
        .silu = cpu_unary_silu_fp8_e5m2,
        .sigmoid = cpu_unary_sigmoid_fp8_e5m2,
        .tanh_act = cpu_unary_tanh_act_fp8_e5m2,
        .mish = cpu_unary_mish_fp8_e5m2,
        .elu = cpu_unary_elu_fp8_e5m2,
        .selu = cpu_unary_selu_fp8_e5m2,
        .leaky_relu = cpu_unary_leaky_relu_fp8_e5m2,
        .prelu = cpu_unary_prelu_fp8_e5m2,
        .fused_bias_relu = cpu_unary_fused_bias_relu_fp8_e5m2,
        .fused_bias_gelu = cpu_unary_fused_bias_gelu_fp8_e5m2,
        .fused_bias_gelu_tanh = cpu_unary_fused_bias_gelu_tanh_fp8_e5m2,
        .fused_bias_silu = cpu_unary_fused_bias_silu_fp8_e5m2,
        .fused_bias_sigmoid = cpu_unary_fused_bias_sigmoid_fp8_e5m2,
        .fused_bias_tanh = cpu_unary_fused_bias_tanh_fp8_e5m2,
        .fused_bias_mish = cpu_unary_fused_bias_mish_fp8_e5m2,
        .fused_bias_elu = cpu_unary_fused_bias_elu_fp8_e5m2,
        .fused_bias_selu = cpu_unary_fused_bias_selu_fp8_e5m2,
        .fused_bias_leaky_relu = cpu_unary_fused_bias_leaky_relu_fp8_e5m2,
        .fused_bias_prelu = cpu_unary_fused_bias_prelu_fp8_e5m2,
        .sign = cpu_unary_sign_fp8_e5m2,
        .impl_name = "scalar:fp8_e5m2",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_fp8_e5m2_scalar_traits)
#endif

const cpu_unary_traits_t cpu_unary_i8_scalar_traits = {
    .dtype = MARMOT_DTYPE_INT8,
    .impl_kind = CPU_UNARY_IMPL_SCALAR,
    .ops = {
        .abs = cpu_unary_abs_i8,
        .neg = cpu_unary_neg_i8,
        .bitwise_not = cpu_unary_bitwise_not_i8,
        .sign = cpu_unary_sign_i8,
        .impl_name = "scalar:i8",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_i8_scalar_traits)

const cpu_unary_traits_t cpu_unary_i16_scalar_traits = {
    .dtype = MARMOT_DTYPE_INT16,
    .impl_kind = CPU_UNARY_IMPL_SCALAR,
    .ops = {
        .abs = cpu_unary_abs_i16,
        .neg = cpu_unary_neg_i16,
        .bitwise_not = cpu_unary_bitwise_not_i16,
        .sign = cpu_unary_sign_i16,
        .impl_name = "scalar:i16",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_i16_scalar_traits)

const cpu_unary_traits_t cpu_unary_i32_scalar_traits = {
    .dtype = MARMOT_DTYPE_INT32,
    .impl_kind = CPU_UNARY_IMPL_SCALAR,
    .ops = {
        .abs = cpu_unary_abs_i32,
        .neg = cpu_unary_neg_i32,
        .bitwise_not = cpu_unary_bitwise_not_i32,
        .sign = cpu_unary_sign_i32,
        .impl_name = "scalar:i32",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_i32_scalar_traits)

const cpu_unary_traits_t cpu_unary_i64_scalar_traits = {
    .dtype = MARMOT_DTYPE_INT64,
    .impl_kind = CPU_UNARY_IMPL_SCALAR,
    .ops = {
        .abs = cpu_unary_abs_i64,
        .neg = cpu_unary_neg_i64,
        .bitwise_not = cpu_unary_bitwise_not_i64,
        .sign = cpu_unary_sign_i64,
        .impl_name = "scalar:i64",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_i64_scalar_traits)

const cpu_unary_traits_t cpu_unary_u8_scalar_traits = {
    .dtype = MARMOT_DTYPE_UINT8,
    .impl_kind = CPU_UNARY_IMPL_SCALAR,
    .ops = {
        .abs = cpu_unary_abs_u8,
        .bitwise_not = cpu_unary_bitwise_not_u8,
        .sign = cpu_unary_sign_u8,
        .impl_name = "scalar:u8",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_u8_scalar_traits)

const cpu_unary_traits_t cpu_unary_u16_scalar_traits = {
    .dtype = MARMOT_DTYPE_UINT16,
    .impl_kind = CPU_UNARY_IMPL_SCALAR,
    .ops = {
        .abs = cpu_unary_abs_u16,
        .bitwise_not = cpu_unary_bitwise_not_u16,
        .sign = cpu_unary_sign_u16,
        .impl_name = "scalar:u16",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_u16_scalar_traits)

const cpu_unary_traits_t cpu_unary_u32_scalar_traits = {
    .dtype = MARMOT_DTYPE_UINT32,
    .impl_kind = CPU_UNARY_IMPL_SCALAR,
    .ops = {
        .abs = cpu_unary_abs_u32,
        .bitwise_not = cpu_unary_bitwise_not_u32,
        .sign = cpu_unary_sign_u32,
        .impl_name = "scalar:u32",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_u32_scalar_traits)

const cpu_unary_traits_t cpu_unary_u64_scalar_traits = {
    .dtype = MARMOT_DTYPE_UINT64,
    .impl_kind = CPU_UNARY_IMPL_SCALAR,
    .ops = {
        .abs = cpu_unary_abs_u64,
        .bitwise_not = cpu_unary_bitwise_not_u64,
        .sign = cpu_unary_sign_u64,
        .impl_name = "scalar:u64",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_u64_scalar_traits)
