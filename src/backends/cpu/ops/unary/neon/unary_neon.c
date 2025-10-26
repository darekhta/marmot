#include "cpu_backend_internal.h"
#include "ops/cpu_neon_math.h"

#if HAS_NEON

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

typedef float32x4_t (*cpu_unary_neon_vec_fn)(float32x4_t);

static inline float32x4_t cpu_unary_neon_sigmoid_vec(float32x4_t x) {
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t denom = vaddq_f32(one, cpu_neon_exp_vec(vnegq_f32(x)));
    return cpu_neon_div_vec(one, denom);
}

static inline float32x4_t cpu_unary_neon_silu_vec(float32x4_t x) {
    return vmulq_f32(x, cpu_unary_neon_sigmoid_vec(x));
}

static inline float32x4_t cpu_unary_neon_tanh_vec(float32x4_t x) {
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t two = vdupq_n_f32(2.0f);
    float32x4_t exp2x = cpu_neon_exp_vec(vmulq_f32(x, two));
    float32x4_t numerator = vsubq_f32(exp2x, one);
    float32x4_t denominator = vaddq_f32(exp2x, one);
    return cpu_neon_div_vec(numerator, denominator);
}

static inline float32x4_t cpu_unary_neon_erf_vec(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t p = vdupq_n_f32(0.3275911f);
    const float32x4_t a1 = vdupq_n_f32(0.254829592f);
    const float32x4_t a2 = vdupq_n_f32(-0.284496736f);
    const float32x4_t a3 = vdupq_n_f32(1.421413741f);
    const float32x4_t a4 = vdupq_n_f32(-1.453152027f);
    const float32x4_t a5 = vdupq_n_f32(1.061405429f);

    uint32x4_t sign_mask = vcltq_f32(x, vdupq_n_f32(0.0f));
    float32x4_t ax = vabsq_f32(x);
    float32x4_t t = cpu_neon_div_vec(one, vaddq_f32(one, vmulq_f32(p, ax)));

    float32x4_t poly = a5;
    poly = vmlaq_f32(a4, poly, t);
    poly = vmlaq_f32(a3, poly, t);
    poly = vmlaq_f32(a2, poly, t);
    poly = vmlaq_f32(a1, poly, t);
    poly = vmulq_f32(poly, t);

    float32x4_t exp_term = cpu_neon_exp_vec(vnegq_f32(vmulq_f32(ax, ax)));
    float32x4_t y = vsubq_f32(one, vmulq_f32(poly, exp_term));
    float32x4_t neg_y = vnegq_f32(y);
    return vbslq_f32(sign_mask, neg_y, y);
}

static inline float32x4_t cpu_unary_neon_gelu_vec(float32x4_t x) {
    float32x4_t inv_sqrt2 = vdupq_n_f32(0.7071067811865475f);
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t half = vdupq_n_f32(0.5f);
    float32x4_t erf_val = cpu_unary_neon_erf_vec(vmulq_f32(x, inv_sqrt2));
    return vmulq_f32(vmulq_f32(x, half), vaddq_f32(one, erf_val));
}

static inline float32x4_t cpu_unary_neon_gelu_tanh_vec(float32x4_t x) {
    float32x4_t sqrt_2_over_pi = vdupq_n_f32(0.7978845608028654f);
    float32x4_t coeff = vdupq_n_f32(0.044715f);
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t half = vdupq_n_f32(0.5f);
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x3 = vmulq_f32(x2, x);
    float32x4_t inner = vaddq_f32(x, vmulq_f32(coeff, x3));
    inner = vmulq_f32(inner, sqrt_2_over_pi);
    float32x4_t tanh_inner = cpu_unary_neon_tanh_vec(inner);
    return vmulq_f32(vmulq_f32(x, half), vaddq_f32(one, tanh_inner));
}

static inline float32x4_t cpu_unary_neon_log1p_scalar_vec(float32x4_t x) {
    float tmp[4];
    vst1q_f32(tmp, x);
    tmp[0] = log1pf(tmp[0]);
    tmp[1] = log1pf(tmp[1]);
    tmp[2] = log1pf(tmp[2]);
    tmp[3] = log1pf(tmp[3]);
    return vld1q_f32(tmp);
}

static inline float32x4_t cpu_unary_neon_mish_vec(float32x4_t x) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t abs_x = vabsq_f32(x);
    float32x4_t exp_neg_abs = cpu_neon_exp_vec(vnegq_f32(abs_x));
    float32x4_t log1p_exp = cpu_unary_neon_log1p_scalar_vec(exp_neg_abs);
    float32x4_t softplus = vaddq_f32(log1p_exp, vmaxq_f32(x, zero));
    float32x4_t tanh_softplus = cpu_unary_neon_tanh_vec(softplus);
    return vmulq_f32(x, tanh_softplus);
}

static inline float32x4_t cpu_unary_neon_elu_vec(float32x4_t x, float32x4_t valpha) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t one = vdupq_n_f32(1.0f);
    uint32x4_t mask = vcgeq_f32(x, zero);
    float32x4_t neg = vmulq_f32(valpha, vsubq_f32(cpu_neon_exp_vec(x), one));
    return vbslq_f32(mask, x, neg);
}

static inline float32x4_t cpu_unary_neon_selu_vec(float32x4_t x, float32x4_t valpha, float32x4_t vlambda) {
    float32x4_t inner = cpu_unary_neon_elu_vec(x, valpha);
    return vmulq_f32(inner, vlambda);
}

static inline float32x4_t cpu_unary_neon_sloped_relu_vec(float32x4_t vx, float32x4_t vslope) {
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    float32x4_t vpos = vmaxq_f32(vx, vzero);
    float32x4_t vneg = vmulq_f32(vminq_f32(vx, vzero), vslope);
    return vaddq_f32(vpos, vneg);
}

static void cpu_unary_neon_sloped_relu(const float *src, float *dst, size_t n, float slope) {
    const float32x4_t vslope = vdupq_n_f32(slope);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(src + i);
        vst1q_f32(dst + i, cpu_unary_neon_sloped_relu_vec(vx, vslope));
    }
    for (; i < n; ++i) {
        float value = src[i];
        dst[i] = value >= 0.0f ? value : slope * value;
    }
}

static marmot_error_t cpu_unary_relu_f32_neon(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const float *src = (const float *)x;
    float *dst = (float *)out;
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(src + i);
        vst1q_f32(dst + i, vmaxq_f32(vx, vzero));
    }
    for (; i < n; ++i) {
        float value = src[i];
        dst[i] = value > 0.0f ? value : 0.0f;
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_leaky_relu_f32_neon(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    float slope = params != nullptr ? params->alpha : 0.01f;
    cpu_unary_neon_sloped_relu((const float *)x, (float *)out, n, slope);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_prelu_f32_neon(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (params == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "PReLU requires a scalar slope parameter");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (params->parameter_tensor != nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Tensor slopes are not implemented for CPU PReLU");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    cpu_unary_neon_sloped_relu((const float *)x, (float *)out, n, params->alpha);
    return MARMOT_SUCCESS;
}

static void cpu_unary_neon_sloped_relu_with_bias(
    const float *src, const float *bias, float *dst, size_t n, float slope, bool bias_is_scalar, size_t feature_dim
) {
    if (n == 0) {
        return;
    }
    const float32x4_t vslope = vdupq_n_f32(slope);

    if (bias_is_scalar) {
        float bias_value = bias[0];
        float32x4_t vbias = vdupq_n_f32(bias_value);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            float32x4_t vx = vaddq_f32(vld1q_f32(src + i), vbias);
            vst1q_f32(dst + i, cpu_unary_neon_sloped_relu_vec(vx, vslope));
        }
        for (; i < n; ++i) {
            float sum = src[i] + bias_value;
            dst[i] = sum >= 0.0f ? sum : slope * sum;
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
        for (; i + 4 <= feature_dim; i += 4) {
            float32x4_t vx = vld1q_f32(src + base + i);
            float32x4_t vb = vld1q_f32(bias + i);
            float32x4_t vsum = vaddq_f32(vx, vb);
            vst1q_f32(dst + base + i, cpu_unary_neon_sloped_relu_vec(vsum, vslope));
        }
        for (; i < feature_dim; ++i) {
            float sum = src[base + i] + bias[i];
            dst[base + i] = sum >= 0.0f ? sum : slope * sum;
        }
    }

    size_t processed = outer * feature_dim;
    for (size_t i = processed; i < n; ++i) {
        size_t bias_index = feature_dim == 0 ? 0 : (i % feature_dim);
        float sum = src[i] + bias[bias_index];
        dst[i] = sum >= 0.0f ? sum : slope * sum;
    }
}

static void cpu_unary_neon_apply_vec(
    const float *src, float *dst, size_t n, cpu_unary_neon_vec_fn vec_fn, cpu_unary_compute_fn scalar_fn,
    const marmot_activation_params_t *params
) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t v0 = vld1q_f32(src + i);
        float32x4_t v1 = vld1q_f32(src + i + 4);
        float32x4_t v2 = vld1q_f32(src + i + 8);
        float32x4_t v3 = vld1q_f32(src + i + 12);

        v0 = vec_fn(v0);
        v1 = vec_fn(v1);
        v2 = vec_fn(v2);
        v3 = vec_fn(v3);

        vst1q_f32(dst + i, v0);
        vst1q_f32(dst + i + 4, v1);
        vst1q_f32(dst + i + 8, v2);
        vst1q_f32(dst + i + 12, v3);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(src + i);
        vst1q_f32(dst + i, vec_fn(vx));
    }
    for (; i < n; ++i) {
        dst[i] = scalar_fn(src[i], params);
    }
}

static void cpu_unary_neon_apply_vec_with_bias(
    const float *src, const float *bias, float *dst, size_t n, bool bias_is_scalar, size_t feature_dim,
    cpu_unary_neon_vec_fn vec_fn, cpu_unary_compute_fn scalar_fn, const marmot_activation_params_t *params
) {
    if (n == 0) {
        return;
    }
    if (bias_is_scalar) {
        float bias_value = bias[0];
        float32x4_t vbias = vdupq_n_f32(bias_value);
        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            float32x4_t v0 = vaddq_f32(vld1q_f32(src + i), vbias);
            float32x4_t v1 = vaddq_f32(vld1q_f32(src + i + 4), vbias);
            float32x4_t v2 = vaddq_f32(vld1q_f32(src + i + 8), vbias);
            float32x4_t v3 = vaddq_f32(vld1q_f32(src + i + 12), vbias);

            v0 = vec_fn(v0);
            v1 = vec_fn(v1);
            v2 = vec_fn(v2);
            v3 = vec_fn(v3);

            vst1q_f32(dst + i, v0);
            vst1q_f32(dst + i + 4, v1);
            vst1q_f32(dst + i + 8, v2);
            vst1q_f32(dst + i + 12, v3);
        }
        for (; i + 4 <= n; i += 4) {
            float32x4_t vx = vaddq_f32(vld1q_f32(src + i), vbias);
            vst1q_f32(dst + i, vec_fn(vx));
        }
        for (; i < n; ++i) {
            dst[i] = scalar_fn(src[i] + bias_value, params);
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
        for (; i + 16 <= feature_dim; i += 16) {
            float32x4_t v0 = vaddq_f32(vld1q_f32(src + base + i), vld1q_f32(bias + i));
            float32x4_t v1 = vaddq_f32(vld1q_f32(src + base + i + 4), vld1q_f32(bias + i + 4));
            float32x4_t v2 = vaddq_f32(vld1q_f32(src + base + i + 8), vld1q_f32(bias + i + 8));
            float32x4_t v3 = vaddq_f32(vld1q_f32(src + base + i + 12), vld1q_f32(bias + i + 12));

            v0 = vec_fn(v0);
            v1 = vec_fn(v1);
            v2 = vec_fn(v2);
            v3 = vec_fn(v3);

            vst1q_f32(dst + base + i, v0);
            vst1q_f32(dst + base + i + 4, v1);
            vst1q_f32(dst + base + i + 8, v2);
            vst1q_f32(dst + base + i + 12, v3);
        }
        for (; i + 4 <= feature_dim; i += 4) {
            float32x4_t vx = vaddq_f32(vld1q_f32(src + base + i), vld1q_f32(bias + i));
            vst1q_f32(dst + base + i, vec_fn(vx));
        }
        for (; i < feature_dim; ++i) {
            float sum = src[base + i] + bias[i];
            dst[base + i] = scalar_fn(sum, params);
        }
    }

    size_t processed = outer * feature_dim;
    for (size_t i = processed; i < n; ++i) {
        size_t bias_index = feature_dim == 0 ? 0 : (i % feature_dim);
        float sum = src[i] + bias[bias_index];
        dst[i] = scalar_fn(sum, params);
    }
}

static void cpu_unary_neon_sigmoid(const float *src, float *dst, size_t n) {
    cpu_unary_neon_apply_vec(src, dst, n, cpu_unary_neon_sigmoid_vec, cpu_unary_sigmoid_eval, nullptr);
}

static void cpu_unary_neon_silu(const float *src, float *dst, size_t n) {
    cpu_unary_neon_apply_vec(src, dst, n, cpu_unary_neon_silu_vec, cpu_unary_silu_eval, nullptr);
}

static void cpu_unary_neon_sigmoid_with_bias(
    const float *src, const float *bias, float *dst, size_t n, bool bias_is_scalar, size_t feature_dim
) {
    cpu_unary_neon_apply_vec_with_bias(
        src, bias, dst, n, bias_is_scalar, feature_dim, cpu_unary_neon_sigmoid_vec, cpu_unary_sigmoid_eval, nullptr
    );
}

static void cpu_unary_neon_silu_with_bias(
    const float *src, const float *bias, float *dst, size_t n, bool bias_is_scalar, size_t feature_dim
) {
    cpu_unary_neon_apply_vec_with_bias(
        src, bias, dst, n, bias_is_scalar, feature_dim, cpu_unary_neon_silu_vec, cpu_unary_silu_eval, nullptr
    );
}

static void cpu_unary_neon_tanh(const float *src, float *dst, size_t n) {
    cpu_unary_neon_apply_vec(src, dst, n, cpu_unary_neon_tanh_vec, cpu_unary_tanh_eval, nullptr);
}

static void cpu_unary_neon_gelu(const float *src, float *dst, size_t n) {
    cpu_unary_neon_apply_vec(src, dst, n, cpu_unary_neon_gelu_vec, cpu_unary_gelu_eval, nullptr);
}

static void cpu_unary_neon_gelu_tanh(const float *src, float *dst, size_t n) {
    cpu_unary_neon_apply_vec(src, dst, n, cpu_unary_neon_gelu_tanh_vec, cpu_unary_gelu_tanh_eval, nullptr);
}

static void cpu_unary_neon_mish(const float *src, float *dst, size_t n) {
    cpu_unary_neon_apply_vec(src, dst, n, cpu_unary_neon_mish_vec, cpu_unary_mish_eval, nullptr);
}

static void cpu_unary_neon_tanh_with_bias(
    const float *src, const float *bias, float *dst, size_t n, bool bias_is_scalar, size_t feature_dim
) {
    cpu_unary_neon_apply_vec_with_bias(
        src, bias, dst, n, bias_is_scalar, feature_dim, cpu_unary_neon_tanh_vec, cpu_unary_tanh_eval, nullptr
    );
}

static void cpu_unary_neon_gelu_with_bias(
    const float *src, const float *bias, float *dst, size_t n, bool bias_is_scalar, size_t feature_dim
) {
    cpu_unary_neon_apply_vec_with_bias(
        src, bias, dst, n, bias_is_scalar, feature_dim, cpu_unary_neon_gelu_vec, cpu_unary_gelu_eval, nullptr
    );
}

static void cpu_unary_neon_gelu_tanh_with_bias(
    const float *src, const float *bias, float *dst, size_t n, bool bias_is_scalar, size_t feature_dim
) {
    cpu_unary_neon_apply_vec_with_bias(
        src, bias, dst, n, bias_is_scalar, feature_dim, cpu_unary_neon_gelu_tanh_vec, cpu_unary_gelu_tanh_eval, nullptr
    );
}

static void cpu_unary_neon_mish_with_bias(
    const float *src, const float *bias, float *dst, size_t n, bool bias_is_scalar, size_t feature_dim
) {
    cpu_unary_neon_apply_vec_with_bias(
        src, bias, dst, n, bias_is_scalar, feature_dim, cpu_unary_neon_mish_vec, cpu_unary_mish_eval, nullptr
    );
}

static void cpu_unary_neon_elu(const float *src, float *dst, size_t n, const marmot_activation_params_t *params) {
    float alpha = params != nullptr ? params->alpha : 1.0f;
    float32x4_t valpha = vdupq_n_f32(alpha);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t v0 = vld1q_f32(src + i);
        float32x4_t v1 = vld1q_f32(src + i + 4);
        float32x4_t v2 = vld1q_f32(src + i + 8);
        float32x4_t v3 = vld1q_f32(src + i + 12);

        v0 = cpu_unary_neon_elu_vec(v0, valpha);
        v1 = cpu_unary_neon_elu_vec(v1, valpha);
        v2 = cpu_unary_neon_elu_vec(v2, valpha);
        v3 = cpu_unary_neon_elu_vec(v3, valpha);

        vst1q_f32(dst + i, v0);
        vst1q_f32(dst + i + 4, v1);
        vst1q_f32(dst + i + 8, v2);
        vst1q_f32(dst + i + 12, v3);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(src + i);
        vst1q_f32(dst + i, cpu_unary_neon_elu_vec(vx, valpha));
    }
    for (; i < n; ++i) {
        dst[i] = cpu_unary_elu_eval(src[i], params);
    }
}

static void cpu_unary_neon_selu(const float *src, float *dst, size_t n, const marmot_activation_params_t *params) {
    const float default_alpha = 1.6732632423543772f;
    const float default_lambda = 1.0507009873554804f;
    float alpha = params != nullptr ? params->alpha : default_alpha;
    float lambda = params != nullptr ? params->beta : default_lambda;
    float32x4_t valpha = vdupq_n_f32(alpha);
    float32x4_t vlambda = vdupq_n_f32(lambda);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t v0 = vld1q_f32(src + i);
        float32x4_t v1 = vld1q_f32(src + i + 4);
        float32x4_t v2 = vld1q_f32(src + i + 8);
        float32x4_t v3 = vld1q_f32(src + i + 12);

        v0 = cpu_unary_neon_selu_vec(v0, valpha, vlambda);
        v1 = cpu_unary_neon_selu_vec(v1, valpha, vlambda);
        v2 = cpu_unary_neon_selu_vec(v2, valpha, vlambda);
        v3 = cpu_unary_neon_selu_vec(v3, valpha, vlambda);

        vst1q_f32(dst + i, v0);
        vst1q_f32(dst + i + 4, v1);
        vst1q_f32(dst + i + 8, v2);
        vst1q_f32(dst + i + 12, v3);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(src + i);
        vst1q_f32(dst + i, cpu_unary_neon_selu_vec(vx, valpha, vlambda));
    }
    for (; i < n; ++i) {
        dst[i] = cpu_unary_selu_eval(src[i], params);
    }
}

static void cpu_unary_neon_elu_with_bias(
    const float *src, const float *bias, float *dst, size_t n, bool bias_is_scalar, size_t feature_dim,
    const marmot_activation_params_t *params
) {
    float alpha = params != nullptr ? params->alpha : 1.0f;
    float32x4_t valpha = vdupq_n_f32(alpha);
    if (n == 0) {
        return;
    }
    if (bias_is_scalar) {
        float bias_value = bias[0];
        float32x4_t vbias = vdupq_n_f32(bias_value);
        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            float32x4_t v0 = vaddq_f32(vld1q_f32(src + i), vbias);
            float32x4_t v1 = vaddq_f32(vld1q_f32(src + i + 4), vbias);
            float32x4_t v2 = vaddq_f32(vld1q_f32(src + i + 8), vbias);
            float32x4_t v3 = vaddq_f32(vld1q_f32(src + i + 12), vbias);

            v0 = cpu_unary_neon_elu_vec(v0, valpha);
            v1 = cpu_unary_neon_elu_vec(v1, valpha);
            v2 = cpu_unary_neon_elu_vec(v2, valpha);
            v3 = cpu_unary_neon_elu_vec(v3, valpha);

            vst1q_f32(dst + i, v0);
            vst1q_f32(dst + i + 4, v1);
            vst1q_f32(dst + i + 8, v2);
            vst1q_f32(dst + i + 12, v3);
        }
        for (; i + 4 <= n; i += 4) {
            float32x4_t vx = vaddq_f32(vld1q_f32(src + i), vbias);
            vst1q_f32(dst + i, cpu_unary_neon_elu_vec(vx, valpha));
        }
        for (; i < n; ++i) {
            dst[i] = cpu_unary_elu_eval(src[i] + bias_value, params);
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
        for (; i + 16 <= feature_dim; i += 16) {
            float32x4_t v0 = vaddq_f32(vld1q_f32(src + base + i), vld1q_f32(bias + i));
            float32x4_t v1 = vaddq_f32(vld1q_f32(src + base + i + 4), vld1q_f32(bias + i + 4));
            float32x4_t v2 = vaddq_f32(vld1q_f32(src + base + i + 8), vld1q_f32(bias + i + 8));
            float32x4_t v3 = vaddq_f32(vld1q_f32(src + base + i + 12), vld1q_f32(bias + i + 12));

            v0 = cpu_unary_neon_elu_vec(v0, valpha);
            v1 = cpu_unary_neon_elu_vec(v1, valpha);
            v2 = cpu_unary_neon_elu_vec(v2, valpha);
            v3 = cpu_unary_neon_elu_vec(v3, valpha);

            vst1q_f32(dst + base + i, v0);
            vst1q_f32(dst + base + i + 4, v1);
            vst1q_f32(dst + base + i + 8, v2);
            vst1q_f32(dst + base + i + 12, v3);
        }
        for (; i + 4 <= feature_dim; i += 4) {
            float32x4_t vx = vaddq_f32(vld1q_f32(src + base + i), vld1q_f32(bias + i));
            vst1q_f32(dst + base + i, cpu_unary_neon_elu_vec(vx, valpha));
        }
        for (; i < feature_dim; ++i) {
            float sum = src[base + i] + bias[i];
            dst[base + i] = cpu_unary_elu_eval(sum, params);
        }
    }

    size_t processed = outer * feature_dim;
    for (size_t i = processed; i < n; ++i) {
        size_t bias_index = feature_dim == 0 ? 0 : (i % feature_dim);
        float sum = src[i] + bias[bias_index];
        dst[i] = cpu_unary_elu_eval(sum, params);
    }
}

static void cpu_unary_neon_selu_with_bias(
    const float *src, const float *bias, float *dst, size_t n, bool bias_is_scalar, size_t feature_dim,
    const marmot_activation_params_t *params
) {
    const float default_alpha = 1.6732632423543772f;
    const float default_lambda = 1.0507009873554804f;
    float alpha = params != nullptr ? params->alpha : default_alpha;
    float lambda = params != nullptr ? params->beta : default_lambda;
    float32x4_t valpha = vdupq_n_f32(alpha);
    float32x4_t vlambda = vdupq_n_f32(lambda);
    if (n == 0) {
        return;
    }
    if (bias_is_scalar) {
        float bias_value = bias[0];
        float32x4_t vbias = vdupq_n_f32(bias_value);
        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            float32x4_t v0 = vaddq_f32(vld1q_f32(src + i), vbias);
            float32x4_t v1 = vaddq_f32(vld1q_f32(src + i + 4), vbias);
            float32x4_t v2 = vaddq_f32(vld1q_f32(src + i + 8), vbias);
            float32x4_t v3 = vaddq_f32(vld1q_f32(src + i + 12), vbias);

            v0 = cpu_unary_neon_selu_vec(v0, valpha, vlambda);
            v1 = cpu_unary_neon_selu_vec(v1, valpha, vlambda);
            v2 = cpu_unary_neon_selu_vec(v2, valpha, vlambda);
            v3 = cpu_unary_neon_selu_vec(v3, valpha, vlambda);

            vst1q_f32(dst + i, v0);
            vst1q_f32(dst + i + 4, v1);
            vst1q_f32(dst + i + 8, v2);
            vst1q_f32(dst + i + 12, v3);
        }
        for (; i + 4 <= n; i += 4) {
            float32x4_t vx = vaddq_f32(vld1q_f32(src + i), vbias);
            vst1q_f32(dst + i, cpu_unary_neon_selu_vec(vx, valpha, vlambda));
        }
        for (; i < n; ++i) {
            dst[i] = cpu_unary_selu_eval(src[i] + bias_value, params);
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
        for (; i + 16 <= feature_dim; i += 16) {
            float32x4_t v0 = vaddq_f32(vld1q_f32(src + base + i), vld1q_f32(bias + i));
            float32x4_t v1 = vaddq_f32(vld1q_f32(src + base + i + 4), vld1q_f32(bias + i + 4));
            float32x4_t v2 = vaddq_f32(vld1q_f32(src + base + i + 8), vld1q_f32(bias + i + 8));
            float32x4_t v3 = vaddq_f32(vld1q_f32(src + base + i + 12), vld1q_f32(bias + i + 12));

            v0 = cpu_unary_neon_selu_vec(v0, valpha, vlambda);
            v1 = cpu_unary_neon_selu_vec(v1, valpha, vlambda);
            v2 = cpu_unary_neon_selu_vec(v2, valpha, vlambda);
            v3 = cpu_unary_neon_selu_vec(v3, valpha, vlambda);

            vst1q_f32(dst + base + i, v0);
            vst1q_f32(dst + base + i + 4, v1);
            vst1q_f32(dst + base + i + 8, v2);
            vst1q_f32(dst + base + i + 12, v3);
        }
        for (; i + 4 <= feature_dim; i += 4) {
            float32x4_t vx = vaddq_f32(vld1q_f32(src + base + i), vld1q_f32(bias + i));
            vst1q_f32(dst + base + i, cpu_unary_neon_selu_vec(vx, valpha, vlambda));
        }
        for (; i < feature_dim; ++i) {
            float sum = src[base + i] + bias[i];
            dst[base + i] = cpu_unary_selu_eval(sum, params);
        }
    }

    size_t processed = outer * feature_dim;
    for (size_t i = processed; i < n; ++i) {
        size_t bias_index = feature_dim == 0 ? 0 : (i % feature_dim);
        float sum = src[i] + bias[bias_index];
        dst[i] = cpu_unary_selu_eval(sum, params);
    }
}

static marmot_error_t cpu_unary_fused_bias_relu_f32_neon(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (bias_is_scalar) {
        cpu_unary_neon_sloped_relu_with_bias(
            (const float *)x, (const float *)bias, (float *)out, n, 0.0f, true, feature_dim
        );
        return MARMOT_SUCCESS;
    }
    cpu_unary_neon_sloped_relu_with_bias(
        (const float *)x, (const float *)bias, (float *)out, n, 0.0f, false, feature_dim
    );
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_leaky_relu_f32_neon(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    float slope = params != nullptr ? params->alpha : 0.01f;
    cpu_unary_neon_sloped_relu_with_bias(
        (const float *)x, (const float *)bias, (float *)out, n, slope, bias_is_scalar, feature_dim
    );
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_prelu_f32_neon(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (params == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "PReLU requires a scalar slope parameter");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (params->parameter_tensor != nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Tensor slopes are not implemented for CPU PReLU");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    cpu_unary_neon_sloped_relu_with_bias(
        (const float *)x, (const float *)bias, (float *)out, n, params->alpha, bias_is_scalar, feature_dim
    );
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_gelu_f32_neon(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    cpu_unary_neon_gelu((const float *)x, (float *)out, n);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_gelu_tanh_f32_neon(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    cpu_unary_neon_gelu_tanh((const float *)x, (float *)out, n);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_silu_f32_neon(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    cpu_unary_neon_silu((const float *)x, (float *)out, n);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_sigmoid_f32_neon(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    cpu_unary_neon_sigmoid((const float *)x, (float *)out, n);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_tanh_act_f32_neon(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    cpu_unary_neon_tanh((const float *)x, (float *)out, n);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_mish_f32_neon(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    cpu_unary_neon_mish((const float *)x, (float *)out, n);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_elu_f32_neon(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    cpu_unary_neon_elu((const float *)x, (float *)out, n, params);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_selu_f32_neon(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    cpu_unary_neon_selu((const float *)x, (float *)out, n, params);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_gelu_f32_neon(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (!bias_is_scalar && feature_dim == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    cpu_unary_neon_gelu_with_bias((const float *)x, (const float *)bias, (float *)out, n, bias_is_scalar, feature_dim);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_gelu_tanh_f32_neon(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (!bias_is_scalar && feature_dim == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    cpu_unary_neon_gelu_tanh_with_bias(
        (const float *)x, (const float *)bias, (float *)out, n, bias_is_scalar, feature_dim
    );
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_silu_f32_neon(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (!bias_is_scalar && feature_dim == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    cpu_unary_neon_silu_with_bias((const float *)x, (const float *)bias, (float *)out, n, bias_is_scalar, feature_dim);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_sigmoid_f32_neon(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (!bias_is_scalar && feature_dim == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    cpu_unary_neon_sigmoid_with_bias(
        (const float *)x, (const float *)bias, (float *)out, n, bias_is_scalar, feature_dim
    );
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_tanh_f32_neon(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (!bias_is_scalar && feature_dim == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    cpu_unary_neon_tanh_with_bias((const float *)x, (const float *)bias, (float *)out, n, bias_is_scalar, feature_dim);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_mish_f32_neon(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (!bias_is_scalar && feature_dim == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    cpu_unary_neon_mish_with_bias((const float *)x, (const float *)bias, (float *)out, n, bias_is_scalar, feature_dim);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_elu_f32_neon(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (!bias_is_scalar && feature_dim == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    cpu_unary_neon_elu_with_bias(
        (const float *)x, (const float *)bias, (float *)out, n, bias_is_scalar, feature_dim, params
    );
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_selu_f32_neon(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    if (!has_neon(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (!bias_is_scalar && feature_dim == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    cpu_unary_neon_selu_with_bias(
        (const float *)x, (const float *)bias, (float *)out, n, bias_is_scalar, feature_dim, params
    );
    return MARMOT_SUCCESS;
}

const cpu_unary_traits_t cpu_unary_f32_neon_traits = {
    .dtype = MARMOT_DTYPE_FLOAT32,
    .impl_kind = CPU_UNARY_IMPL_NEON,
    .ops = {
        .relu = cpu_unary_relu_f32_neon,
        .gelu = cpu_unary_gelu_f32_neon,
        .gelu_tanh = cpu_unary_gelu_tanh_f32_neon,
        .silu = cpu_unary_silu_f32_neon,
        .sigmoid = cpu_unary_sigmoid_f32_neon,
        .tanh_act = cpu_unary_tanh_act_f32_neon,
        .mish = cpu_unary_mish_f32_neon,
        .elu = cpu_unary_elu_f32_neon,
        .selu = cpu_unary_selu_f32_neon,
        .leaky_relu = cpu_unary_leaky_relu_f32_neon,
        .prelu = cpu_unary_prelu_f32_neon,
        .fused_bias_relu = cpu_unary_fused_bias_relu_f32_neon,
        .fused_bias_gelu = cpu_unary_fused_bias_gelu_f32_neon,
        .fused_bias_gelu_tanh = cpu_unary_fused_bias_gelu_tanh_f32_neon,
        .fused_bias_silu = cpu_unary_fused_bias_silu_f32_neon,
        .fused_bias_sigmoid = cpu_unary_fused_bias_sigmoid_f32_neon,
        .fused_bias_tanh = cpu_unary_fused_bias_tanh_f32_neon,
        .fused_bias_mish = cpu_unary_fused_bias_mish_f32_neon,
        .fused_bias_elu = cpu_unary_fused_bias_elu_f32_neon,
        .fused_bias_selu = cpu_unary_fused_bias_selu_f32_neon,
        .fused_bias_leaky_relu = cpu_unary_fused_bias_leaky_relu_f32_neon,
        .fused_bias_prelu = cpu_unary_fused_bias_prelu_f32_neon,
        .impl_name = "neon:f32",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_f32_neon_traits)

#endif // HAS_NEON
