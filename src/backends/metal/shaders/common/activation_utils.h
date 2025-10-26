#pragma once

#include "common/defines.h"

#include "common/math_utils.h"

static inline float relu_op(float x) {
    return fmax(x, 0.0f);
}

static inline float gelu_exact(float x) {
    const float inv_sqrt2 = 0.70710678118654752440f;
    return x * 0.5f * (1.0f + erf_approx(x * inv_sqrt2));
}

static inline float silu_exact(float x) {
    return x / (1.0f + exp(-x));
}

static inline float tanh_exact(float x);

static inline float gelu_tanh_exact(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5f * x * (1.0f + tanh_exact(inner));
}

static inline float sigmoid_exact(float x) {
    if (x >= 0.0f) {
        float z = exp(-x);
        return 1.0f / (1.0f + z);
    }
    float z = exp(x);
    return z / (1.0f + z);
}

static inline float tanh_exact(float x) {
    const float limit = 10.0f;
    if (x > limit) {
        return 1.0f;
    }
    if (x < -limit) {
        return -1.0f;
    }
    return precise::tanh(x);
}

static inline float mish_exact(float x) {
    float abs_x = fabs(x);
    float softplus = log(1.0f + exp(-abs_x)) + max(x, 0.0f);
    return x * tanh_exact(softplus);
}

static inline float elu_with_params(float x, constant ActivationParams &params) {
    float alpha = params.alpha;
    return x >= 0.0f ? x : alpha * (exp(x) - 1.0f);
}

static inline float selu_with_params(float x, constant ActivationParams &params) {
    float alpha = params.alpha;
    float lambda = params.beta;
    float inner = x >= 0.0f ? x : alpha * (exp(x) - 1.0f);
    return lambda * inner;
}

static inline float leaky_relu_with_params(float x, constant ActivationParams &params) {
    float slope = params.alpha;
    return x >= 0.0f ? x : slope * x;
}

static inline float prelu_with_params(float x, constant ActivationParams &params) {
    float slope = params.alpha;
    return x >= 0.0f ? x : slope * x;
}

static inline float apply_fused_activation(uint activation, float value, constant ActivationParams &params) {
    switch (activation) {
    case UnaryActivationIdentity:
        return value;
    case UnaryActivationRelu:
        return relu_op(value);
    case UnaryActivationGelu:
        return gelu_exact(value);
    case UnaryActivationGeluTanh:
        return gelu_tanh_exact(value);
    case UnaryActivationSilu:
        return silu_exact(value);
    case UnaryActivationSigmoid:
        return sigmoid_exact(value);
    case UnaryActivationTanh:
        return tanh_exact(value);
    case UnaryActivationMish:
        return mish_exact(value);
    case UnaryActivationElu:
        return elu_with_params(value, params);
    case UnaryActivationSelu:
        return selu_with_params(value, params);
    case UnaryActivationLeakyRelu:
        return leaky_relu_with_params(value, params);
    case UnaryActivationPrelu:
        return prelu_with_params(value, params);
    default:
        return value;
    }
}
