#include "cpu_backend_internal.h"

#if HAS_AVX2

typedef float (*cpu_unary_compute_fn)(float, const marmot_activation_params_t *);

static inline float cpu_unary_gelu_eval(float x, const marmot_activation_params_t *params) {
    (void)params;
    const float inv_sqrt2 = 0.7071067811865475f;
    return x * 0.5f * (1.0f + erff(x * inv_sqrt2));
}

static inline float cpu_unary_gelu_tanh_eval(float x, const marmot_activation_params_t *params) {
    (void)params;
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static inline float cpu_unary_silu_eval(float x, const marmot_activation_params_t *params) {
    (void)params;
    return x / (1.0f + expf(-x));
}

static inline float cpu_unary_sigmoid_eval(float x, const marmot_activation_params_t *params) {
    (void)params;
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

static inline float cpu_unary_tanh_eval(float x, const marmot_activation_params_t *params) {
    (void)params;
    return tanhf(x);
}

static inline float cpu_unary_mish_eval(float x, const marmot_activation_params_t *params) {
    (void)params;
    float abs_x = fabsf(x);
    float softplus = log1pf(expf(-abs_x)) + (x > 0.0f ? x : 0.0f);
    return x * tanhf(softplus);
}

static inline float cpu_unary_elu_eval(float x, const marmot_activation_params_t *params) {
    float alpha = params != nullptr ? params->alpha : 1.0f;
    return x >= 0.0f ? x : alpha * (expf(x) - 1.0f);
}

static inline float cpu_unary_selu_eval(float x, const marmot_activation_params_t *params) {
    const float default_alpha = 1.6732632423543772f;
    const float default_lambda = 1.0507009873554804f;
    float alpha = params != nullptr ? params->alpha : default_alpha;
    float lambda = params != nullptr ? params->beta : default_lambda;
    float inner = x >= 0.0f ? x : alpha * (expf(x) - 1.0f);
    return lambda * inner;
}

typedef void (*cpu_unary_to_f32_fn)(const void *device_ctx, float *dst, const void *src, size_t n);
typedef void (*cpu_unary_from_f32_fn)(const void *device_ctx, void *dst, const float *src, size_t n);

static const size_t k_cpu_unary_avx2_via_f32_chunk = 8192;

#define CPU_UNARY_TAG_TYPE_f16 marmot_float16_t
#define CPU_UNARY_TAG_TYPE_bf16 marmot_bfloat16_t
#if MARMOT_ENABLE_FP8
#define CPU_UNARY_TAG_TYPE_fp8_e4m3 marmot_float8_e4m3_t
#define CPU_UNARY_TAG_TYPE_fp8_e5m2 marmot_float8_e5m2_t
#endif

#define CPU_UNARY_ELEM_SIZE(tag) (sizeof(CPU_UNARY_TAG_TYPE_##tag))

static void cpu_unary_to_f32_f16(const void *device_ctx, float *dst, const void *src, size_t n) {
    cpu_convert_f16_to_f32(device_ctx, dst, (const marmot_float16_t *)src, n);
}

static void cpu_unary_from_f32_f16(const void *device_ctx, void *dst, const float *src, size_t n) {
    cpu_convert_f32_to_f16(device_ctx, (marmot_float16_t *)dst, src, n);
}

static void cpu_unary_to_f32_bf16(const void *device_ctx, float *dst, const void *src, size_t n) {
    cpu_convert_bf16_to_f32(device_ctx, dst, (const marmot_bfloat16_t *)src, n);
}

static void cpu_unary_from_f32_bf16(const void *device_ctx, void *dst, const float *src, size_t n) {
    cpu_convert_f32_to_bf16(device_ctx, (marmot_bfloat16_t *)dst, src, n);
}

#if MARMOT_ENABLE_FP8
static void cpu_unary_to_f32_fp8_e4m3(const void *device_ctx, float *dst, const void *src, size_t n) {
    cpu_convert_fp8_e4m3_to_f32(device_ctx, dst, (const marmot_float8_e4m3_t *)src, n);
}

static void cpu_unary_from_f32_fp8_e4m3(const void *device_ctx, void *dst, const float *src, size_t n) {
    cpu_convert_f32_to_fp8_e4m3(device_ctx, (marmot_float8_e4m3_t *)dst, src, n);
}

static void cpu_unary_to_f32_fp8_e5m2(const void *device_ctx, float *dst, const void *src, size_t n) {
    cpu_convert_fp8_e5m2_to_f32(device_ctx, dst, (const marmot_float8_e5m2_t *)src, n);
}

static void cpu_unary_from_f32_fp8_e5m2(const void *device_ctx, void *dst, const float *src, size_t n) {
    cpu_convert_f32_to_fp8_e5m2(device_ctx, (marmot_float8_e5m2_t *)dst, src, n);
}
#endif

static marmot_error_t cpu_unary_via_f32_activation(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n,
    cpu_activation_fn f32_kernel, cpu_unary_to_f32_fn to_f32, cpu_unary_from_f32_fn from_f32, size_t elem_size
) {
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    size_t chunk = n < k_cpu_unary_avx2_via_f32_chunk ? n : k_cpu_unary_avx2_via_f32_chunk;
    float *tmp_in = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    float *tmp_out = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    if (tmp_in == nullptr || tmp_out == nullptr) {
        free(tmp_in);
        free(tmp_out);
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Unary AVX2 via-f32 allocation failed");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    const uint8_t *src = (const uint8_t *)x;
    uint8_t *dst = (uint8_t *)out;
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        to_f32(device_ctx, tmp_in, src + offset * elem_size, block);
        marmot_error_t err = f32_kernel(device_ctx, tmp_in, params, tmp_out, block);
        if (err != MARMOT_SUCCESS) {
            free(tmp_in);
            free(tmp_out);
            return err;
        }
        from_f32(device_ctx, dst + offset * elem_size, tmp_out, block);
    }
    free(tmp_in);
    free(tmp_out);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_via_f32_fused_bias(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n, cpu_fused_bias_activation_fn f32_kernel,
    cpu_unary_to_f32_fn to_f32, cpu_unary_from_f32_fn from_f32, size_t elem_size
) {
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (bias == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unary fused bias pointer cannot be null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!bias_is_scalar && feature_dim == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unary fused bias feature dimension cannot be zero");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    size_t chunk = n < k_cpu_unary_avx2_via_f32_chunk ? n : k_cpu_unary_avx2_via_f32_chunk;
    float *tmp_in = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    float *tmp_out = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    size_t bias_elems = bias_is_scalar ? 1 : feature_dim;
    float *bias_f32 = (float *)marmot_aligned_alloc(64, bias_elems * sizeof(float));
    if (tmp_in == nullptr || tmp_out == nullptr || bias_f32 == nullptr) {
        free(tmp_in);
        free(tmp_out);
        free(bias_f32);
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Unary AVX2 fused bias via-f32 allocation failed");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    to_f32(device_ctx, bias_f32, bias, bias_elems);
    const uint8_t *src = (const uint8_t *)x;
    uint8_t *dst = (uint8_t *)out;
    marmot_error_t err = MARMOT_SUCCESS;
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        to_f32(device_ctx, tmp_in, src + offset * elem_size, block);
        err = f32_kernel(device_ctx, tmp_in, bias_f32, feature_dim, bias_is_scalar, params, tmp_out, block);
        if (err != MARMOT_SUCCESS) {
            break;
        }
        from_f32(device_ctx, dst + offset * elem_size, tmp_out, block);
    }
    free(tmp_in);
    free(tmp_out);
    free(bias_f32);
    return err;
}

#define CPU_UNARY_TO_F32_FN(tag) cpu_unary_to_f32_##tag
#define CPU_UNARY_FROM_F32_FN(tag) cpu_unary_from_f32_##tag

#define CPU_UNARY_DEFINE_VIA_F32_ACT(tag, name, base_fn)                                                               \
    static marmot_error_t cpu_unary_##name##_##tag##_avx2(                                                             \
        const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n           \
    ) {                                                                                                                \
        return cpu_unary_via_f32_activation(                                                                           \
            device_ctx, x, params, out, n, base_fn, CPU_UNARY_TO_F32_FN(tag), CPU_UNARY_FROM_F32_FN(tag),              \
            CPU_UNARY_ELEM_SIZE(tag)                                                                                   \
        );                                                                                                             \
    }

#define CPU_UNARY_DEFINE_VIA_F32_FUSED(tag, name, base_fn)                                                             \
    static marmot_error_t cpu_unary_fused_bias_##name##_##tag##_avx2(                                                  \
        const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,              \
        const marmot_activation_params_t *params, void *out, size_t n                                                  \
    ) {                                                                                                                \
        return cpu_unary_via_f32_fused_bias(                                                                           \
            device_ctx, x, bias, feature_dim, bias_is_scalar, params, out, n, base_fn, CPU_UNARY_TO_F32_FN(tag),       \
            CPU_UNARY_FROM_F32_FN(tag), CPU_UNARY_ELEM_SIZE(tag)                                                       \
        );                                                                                                             \
    }

#define CPU_UNARY_FOR_EACH_ACT(MACRO, tag)                                                                             \
    MACRO(tag, relu, cpu_unary_relu_f32_avx2)                                                                          \
    MACRO(tag, gelu, cpu_unary_gelu_f32_avx2)                                                                          \
    MACRO(tag, gelu_tanh, cpu_unary_gelu_tanh_f32_avx2)                                                                \
    MACRO(tag, silu, cpu_unary_silu_f32_avx2)                                                                          \
    MACRO(tag, sigmoid, cpu_unary_sigmoid_f32_avx2)                                                                    \
    MACRO(tag, tanh_act, cpu_unary_tanh_act_f32_avx2)                                                                  \
    MACRO(tag, mish, cpu_unary_mish_f32_avx2)                                                                          \
    MACRO(tag, elu, cpu_unary_elu_f32_avx2)                                                                            \
    MACRO(tag, selu, cpu_unary_selu_f32_avx2)                                                                          \
    MACRO(tag, leaky_relu, cpu_unary_leaky_relu_f32_avx2)                                                              \
    MACRO(tag, prelu, cpu_unary_prelu_f32_avx2)

#define CPU_UNARY_FOR_EACH_FUSED(MACRO, tag)                                                                           \
    MACRO(tag, relu, cpu_unary_fused_bias_relu_f32_avx2)                                                               \
    MACRO(tag, gelu, cpu_unary_fused_bias_gelu_f32_avx2)                                                               \
    MACRO(tag, gelu_tanh, cpu_unary_fused_bias_gelu_tanh_f32_avx2)                                                     \
    MACRO(tag, silu, cpu_unary_fused_bias_silu_f32_avx2)                                                               \
    MACRO(tag, sigmoid, cpu_unary_fused_bias_sigmoid_f32_avx2)                                                         \
    MACRO(tag, tanh, cpu_unary_fused_bias_tanh_f32_avx2)                                                               \
    MACRO(tag, mish, cpu_unary_fused_bias_mish_f32_avx2)                                                               \
    MACRO(tag, elu, cpu_unary_fused_bias_elu_f32_avx2)                                                                 \
    MACRO(tag, selu, cpu_unary_fused_bias_selu_f32_avx2)                                                               \
    MACRO(tag, leaky_relu, cpu_unary_fused_bias_leaky_relu_f32_avx2)                                                   \
    MACRO(tag, prelu, cpu_unary_fused_bias_prelu_f32_avx2)

#define CPU_UNARY_AVX2_ACT_OPS(tag)                                                                                    \
    .relu = cpu_unary_relu_##tag##_avx2, .gelu = cpu_unary_gelu_##tag##_avx2,                                          \
    .gelu_tanh = cpu_unary_gelu_tanh_##tag##_avx2, .silu = cpu_unary_silu_##tag##_avx2,                                \
    .sigmoid = cpu_unary_sigmoid_##tag##_avx2, .tanh_act = cpu_unary_tanh_act_##tag##_avx2,                            \
    .mish = cpu_unary_mish_##tag##_avx2, .elu = cpu_unary_elu_##tag##_avx2, .selu = cpu_unary_selu_##tag##_avx2,       \
    .leaky_relu = cpu_unary_leaky_relu_##tag##_avx2, .prelu = cpu_unary_prelu_##tag##_avx2,

#define CPU_UNARY_AVX2_FUSED_OPS(tag)                                                                                  \
    .fused_bias_relu = cpu_unary_fused_bias_relu_##tag##_avx2,                                                         \
    .fused_bias_gelu = cpu_unary_fused_bias_gelu_##tag##_avx2,                                                         \
    .fused_bias_gelu_tanh = cpu_unary_fused_bias_gelu_tanh_##tag##_avx2,                                               \
    .fused_bias_silu = cpu_unary_fused_bias_silu_##tag##_avx2,                                                         \
    .fused_bias_sigmoid = cpu_unary_fused_bias_sigmoid_##tag##_avx2,                                                   \
    .fused_bias_tanh = cpu_unary_fused_bias_tanh_##tag##_avx2,                                                         \
    .fused_bias_mish = cpu_unary_fused_bias_mish_##tag##_avx2,                                                         \
    .fused_bias_elu = cpu_unary_fused_bias_elu_##tag##_avx2,                                                           \
    .fused_bias_selu = cpu_unary_fused_bias_selu_##tag##_avx2,                                                         \
    .fused_bias_leaky_relu = cpu_unary_fused_bias_leaky_relu_##tag##_avx2,                                             \
    .fused_bias_prelu = cpu_unary_fused_bias_prelu_##tag##_avx2,

static marmot_error_t cpu_unary_activation_loop_f32_avx2(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n,
    cpu_unary_compute_fn fn
) {
    if (!has_avx2(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const float *src = (const float *)x;
    float *dst = (float *)out;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = fn(src[i], params);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_loop_f32_avx2(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n, cpu_unary_compute_fn fn
) {
    if (!has_avx2(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const float *src = (const float *)x;
    const float *bias_data = (const float *)bias;
    float *dst = (float *)out;
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (bias_is_scalar) {
        float bias_value = bias_data[0];
        for (size_t i = 0; i < n; ++i) {
            dst[i] = fn(src[i] + bias_value, params);
        }
        return MARMOT_SUCCESS;
    }
    if (feature_dim == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < n; ++i) {
        size_t bias_index = feature_dim == 0 ? 0 : (i % feature_dim);
        dst[i] = fn(src[i] + bias_data[bias_index], params);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_relu_f32_avx2(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!has_avx2(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const float *src = (const float *)x;
    float *dst = (float *)out;
    __m256 vzero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(src + i);
        __m256 vout = _mm256_max_ps(vx, vzero);
        _mm256_storeu_ps(dst + i, vout);
    }
    for (; i < n; ++i) {
        float value = src[i];
        dst[i] = value > 0.0f ? value : 0.0f;
    }
    return MARMOT_SUCCESS;
}

static inline void cpu_unary_avx2_sloped_relu(const float *src, float *dst, size_t n, float slope) {
    __m256 vslope = _mm256_set1_ps(slope);
    __m256 vzero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(src + i);
        __m256 vpos = _mm256_max_ps(vx, vzero);
        __m256 vneg = _mm256_min_ps(vx, vzero);
        vneg = _mm256_mul_ps(vneg, vslope);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(vpos, vneg));
    }
    for (; i < n; ++i) {
        float value = src[i];
        dst[i] = value >= 0.0f ? value : slope * value;
    }
}

static marmot_error_t cpu_unary_leaky_relu_f32_avx2(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    if (!has_avx2(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    float slope = params != nullptr ? params->alpha : 0.01f;
    cpu_unary_avx2_sloped_relu((const float *)x, (float *)out, n, slope);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_prelu_f32_avx2(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    if (!has_avx2(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (params == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "PReLU requires scalar slope parameter");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (params->parameter_tensor != nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Tensor slopes are not implemented for CPU PReLU");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    cpu_unary_avx2_sloped_relu((const float *)x, (float *)out, n, params->alpha);
    return MARMOT_SUCCESS;
}

static void cpu_unary_avx2_sloped_relu_with_bias(
    const float *x, const float *bias, float *out, size_t n, float slope, bool bias_is_scalar, size_t feature_dim
) {
    if (n == 0) {
        return;
    }
    __m256 vslope = _mm256_set1_ps(slope);
    __m256 vzero = _mm256_setzero_ps();

    if (bias_is_scalar) {
        __m256 vbias = _mm256_set1_ps(bias[0]);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 vx = _mm256_add_ps(_mm256_loadu_ps(x + i), vbias);
            __m256 vpos = _mm256_max_ps(vx, vzero);
            __m256 vneg = _mm256_min_ps(vx, vzero);
            vneg = _mm256_mul_ps(vneg, vslope);
            _mm256_storeu_ps(out + i, _mm256_add_ps(vpos, vneg));
        }
        for (; i < n; ++i) {
            float sum = x[i] + bias[0];
            out[i] = sum >= 0.0f ? sum : slope * sum;
        }
        return;
    }

    if (feature_dim == 0) {
        return;
    }

    size_t outer = n / feature_dim;
    for (size_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
        size_t base = outer_idx * feature_dim;
        size_t i = 0;
        for (; i + 8 <= feature_dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + base + i);
            __m256 vb = _mm256_loadu_ps(bias + i);
            __m256 vsum = _mm256_add_ps(vx, vb);
            __m256 vpos = _mm256_max_ps(vsum, vzero);
            __m256 vneg = _mm256_min_ps(vsum, vzero);
            vneg = _mm256_mul_ps(vneg, vslope);
            _mm256_storeu_ps(out + base + i, _mm256_add_ps(vpos, vneg));
        }
        for (; i < feature_dim; ++i) {
            float sum = x[base + i] + bias[i];
            out[base + i] = sum >= 0.0f ? sum : slope * sum;
        }
    }

    size_t processed = outer * feature_dim;
    for (size_t i = processed; i < n; ++i) {
        size_t bias_index = feature_dim == 0 ? 0 : (i % feature_dim);
        float sum = x[i] + bias[bias_index];
        out[i] = sum >= 0.0f ? sum : slope * sum;
    }
}

static marmot_error_t cpu_unary_fused_bias_relu_f32_avx2(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!has_avx2(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    cpu_unary_avx2_sloped_relu_with_bias(
        (const float *)x, (const float *)bias, (float *)out, n, 0.0f, bias_is_scalar, feature_dim
    );
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_leaky_relu_f32_avx2(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    if (!has_avx2(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    float slope = params != nullptr ? params->alpha : 0.01f;
    cpu_unary_avx2_sloped_relu_with_bias(
        (const float *)x, (const float *)bias, (float *)out, n, slope, bias_is_scalar, feature_dim
    );
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_prelu_f32_avx2(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    if (!has_avx2(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (params == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "PReLU requires scalar slope parameter");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (params->parameter_tensor != nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Tensor slopes are not implemented for CPU PReLU");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    cpu_unary_avx2_sloped_relu_with_bias(
        (const float *)x, (const float *)bias, (float *)out, n, params->alpha, bias_is_scalar, feature_dim
    );
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_gelu_f32_avx2(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    return cpu_unary_activation_loop_f32_avx2(device_ctx, x, params, out, n, cpu_unary_gelu_eval);
}

static marmot_error_t cpu_unary_gelu_tanh_f32_avx2(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    return cpu_unary_activation_loop_f32_avx2(device_ctx, x, params, out, n, cpu_unary_gelu_tanh_eval);
}

static marmot_error_t cpu_unary_silu_f32_avx2(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    return cpu_unary_activation_loop_f32_avx2(device_ctx, x, params, out, n, cpu_unary_silu_eval);
}

static marmot_error_t cpu_unary_sigmoid_f32_avx2(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    return cpu_unary_activation_loop_f32_avx2(device_ctx, x, params, out, n, cpu_unary_sigmoid_eval);
}

static marmot_error_t cpu_unary_tanh_act_f32_avx2(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    return cpu_unary_activation_loop_f32_avx2(device_ctx, x, params, out, n, cpu_unary_tanh_eval);
}

static marmot_error_t cpu_unary_mish_f32_avx2(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    return cpu_unary_activation_loop_f32_avx2(device_ctx, x, params, out, n, cpu_unary_mish_eval);
}

static marmot_error_t cpu_unary_elu_f32_avx2(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    return cpu_unary_activation_loop_f32_avx2(device_ctx, x, params, out, n, cpu_unary_elu_eval);
}

static marmot_error_t cpu_unary_selu_f32_avx2(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    return cpu_unary_activation_loop_f32_avx2(device_ctx, x, params, out, n, cpu_unary_selu_eval);
}

static marmot_error_t cpu_unary_fused_bias_gelu_f32_avx2(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    return cpu_unary_fused_bias_loop_f32_avx2(
        device_ctx, x, bias, feature_dim, bias_is_scalar, params, out, n, cpu_unary_gelu_eval
    );
}

static marmot_error_t cpu_unary_fused_bias_gelu_tanh_f32_avx2(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    return cpu_unary_fused_bias_loop_f32_avx2(
        device_ctx, x, bias, feature_dim, bias_is_scalar, params, out, n, cpu_unary_gelu_tanh_eval
    );
}

static marmot_error_t cpu_unary_fused_bias_silu_f32_avx2(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    return cpu_unary_fused_bias_loop_f32_avx2(
        device_ctx, x, bias, feature_dim, bias_is_scalar, params, out, n, cpu_unary_silu_eval
    );
}

static marmot_error_t cpu_unary_fused_bias_sigmoid_f32_avx2(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    return cpu_unary_fused_bias_loop_f32_avx2(
        device_ctx, x, bias, feature_dim, bias_is_scalar, params, out, n, cpu_unary_sigmoid_eval
    );
}

static marmot_error_t cpu_unary_fused_bias_tanh_f32_avx2(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    return cpu_unary_fused_bias_loop_f32_avx2(
        device_ctx, x, bias, feature_dim, bias_is_scalar, params, out, n, cpu_unary_tanh_eval
    );
}

static marmot_error_t cpu_unary_fused_bias_mish_f32_avx2(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    return cpu_unary_fused_bias_loop_f32_avx2(
        device_ctx, x, bias, feature_dim, bias_is_scalar, params, out, n, cpu_unary_mish_eval
    );
}

static marmot_error_t cpu_unary_fused_bias_elu_f32_avx2(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    return cpu_unary_fused_bias_loop_f32_avx2(
        device_ctx, x, bias, feature_dim, bias_is_scalar, params, out, n, cpu_unary_elu_eval
    );
}

static marmot_error_t cpu_unary_fused_bias_selu_f32_avx2(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    return cpu_unary_fused_bias_loop_f32_avx2(
        device_ctx, x, bias, feature_dim, bias_is_scalar, params, out, n, cpu_unary_selu_eval
    );
}

const cpu_unary_traits_t cpu_unary_f32_avx2_traits = {
    .dtype = MARMOT_DTYPE_FLOAT32,
    .impl_kind = CPU_UNARY_IMPL_AVX2,
    .ops = {
        .relu = cpu_unary_relu_f32_avx2,
        .gelu = cpu_unary_gelu_f32_avx2,
        .gelu_tanh = cpu_unary_gelu_tanh_f32_avx2,
        .silu = cpu_unary_silu_f32_avx2,
        .sigmoid = cpu_unary_sigmoid_f32_avx2,
        .tanh_act = cpu_unary_tanh_act_f32_avx2,
        .mish = cpu_unary_mish_f32_avx2,
        .elu = cpu_unary_elu_f32_avx2,
        .selu = cpu_unary_selu_f32_avx2,
        .leaky_relu = cpu_unary_leaky_relu_f32_avx2,
        .prelu = cpu_unary_prelu_f32_avx2,
        .fused_bias_relu = cpu_unary_fused_bias_relu_f32_avx2,
        .fused_bias_gelu = cpu_unary_fused_bias_gelu_f32_avx2,
        .fused_bias_gelu_tanh = cpu_unary_fused_bias_gelu_tanh_f32_avx2,
        .fused_bias_silu = cpu_unary_fused_bias_silu_f32_avx2,
        .fused_bias_sigmoid = cpu_unary_fused_bias_sigmoid_f32_avx2,
        .fused_bias_tanh = cpu_unary_fused_bias_tanh_f32_avx2,
        .fused_bias_mish = cpu_unary_fused_bias_mish_f32_avx2,
        .fused_bias_elu = cpu_unary_fused_bias_elu_f32_avx2,
        .fused_bias_selu = cpu_unary_fused_bias_selu_f32_avx2,
        .fused_bias_leaky_relu = cpu_unary_fused_bias_leaky_relu_f32_avx2,
        .fused_bias_prelu = cpu_unary_fused_bias_prelu_f32_avx2,
        .impl_name = "avx2:f32",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_f32_avx2_traits)

CPU_UNARY_FOR_EACH_ACT(CPU_UNARY_DEFINE_VIA_F32_ACT, f16)
CPU_UNARY_FOR_EACH_FUSED(CPU_UNARY_DEFINE_VIA_F32_FUSED, f16)

const cpu_unary_traits_t cpu_unary_f16_avx2_traits = {
    .dtype = MARMOT_DTYPE_FLOAT16,
    .impl_kind = CPU_UNARY_IMPL_AVX2,
    .ops = {
        CPU_UNARY_AVX2_ACT_OPS(f16) CPU_UNARY_AVX2_FUSED_OPS(f16).impl_name = "avx2:f16-via-f32",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_f16_avx2_traits)

CPU_UNARY_FOR_EACH_ACT(CPU_UNARY_DEFINE_VIA_F32_ACT, bf16)
CPU_UNARY_FOR_EACH_FUSED(CPU_UNARY_DEFINE_VIA_F32_FUSED, bf16)

const cpu_unary_traits_t cpu_unary_bf16_avx2_traits = {
    .dtype = MARMOT_DTYPE_BFLOAT16,
    .impl_kind = CPU_UNARY_IMPL_AVX2,
    .ops = {
        CPU_UNARY_AVX2_ACT_OPS(bf16) CPU_UNARY_AVX2_FUSED_OPS(bf16).impl_name = "avx2:bf16-via-f32",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_bf16_avx2_traits)

#if MARMOT_ENABLE_FP8
CPU_UNARY_FOR_EACH_ACT(CPU_UNARY_DEFINE_VIA_F32_ACT, fp8_e4m3)
CPU_UNARY_FOR_EACH_FUSED(CPU_UNARY_DEFINE_VIA_F32_FUSED, fp8_e4m3)

const cpu_unary_traits_t cpu_unary_fp8_e4m3_avx2_traits = {
    .dtype = MARMOT_DTYPE_FLOAT8_E4M3,
    .impl_kind = CPU_UNARY_IMPL_AVX2,
    .ops = {
        CPU_UNARY_AVX2_ACT_OPS(fp8_e4m3) CPU_UNARY_AVX2_FUSED_OPS(fp8_e4m3).impl_name = "avx2:fp8_e4m3-via-f32",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_fp8_e4m3_avx2_traits)

CPU_UNARY_FOR_EACH_ACT(CPU_UNARY_DEFINE_VIA_F32_ACT, fp8_e5m2)
CPU_UNARY_FOR_EACH_FUSED(CPU_UNARY_DEFINE_VIA_F32_FUSED, fp8_e5m2)

const cpu_unary_traits_t cpu_unary_fp8_e5m2_avx2_traits = {
    .dtype = MARMOT_DTYPE_FLOAT8_E5M2,
    .impl_kind = CPU_UNARY_IMPL_AVX2,
    .ops = {
        CPU_UNARY_AVX2_ACT_OPS(fp8_e5m2) CPU_UNARY_AVX2_FUSED_OPS(fp8_e5m2).impl_name = "avx2:fp8_e5m2-via-f32",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_fp8_e5m2_avx2_traits)
#endif

#endif // HAS_AVX2
