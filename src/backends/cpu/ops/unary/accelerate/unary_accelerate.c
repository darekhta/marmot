#include "cpu_backend_internal.h"

#if MARMOT_ENABLE_ACCELERATE

#include <math.h>
#include <string.h>

static const size_t k_cpu_unary_accelerate_chunk = 8192;

typedef void (*cpu_unary_acc_to_f32_fn)(const void *device_ctx, float *dst, const void *src, size_t n);
typedef void (*cpu_unary_acc_from_f32_fn)(const void *device_ctx, void *dst, const float *src, size_t n);

static void cpu_unary_acc_to_f32_f16(const void *device_ctx, float *dst, const void *src, size_t n) {
    cpu_convert_f16_to_f32(device_ctx, dst, (const marmot_float16_t *)src, n);
}

static void cpu_unary_acc_from_f32_f16(const void *device_ctx, void *dst, const float *src, size_t n) {
    cpu_convert_f32_to_f16(device_ctx, (marmot_float16_t *)dst, src, n);
}

static void cpu_unary_acc_to_f32_bf16(const void *device_ctx, float *dst, const void *src, size_t n) {
    cpu_convert_bf16_to_f32(device_ctx, dst, (const marmot_bfloat16_t *)src, n);
}

static void cpu_unary_acc_from_f32_bf16(const void *device_ctx, void *dst, const float *src, size_t n) {
    cpu_convert_f32_to_bf16(device_ctx, (marmot_bfloat16_t *)dst, src, n);
}

static bool cpu_unary_have_accelerate(const void *device_ctx) {
    return has_accelerate(device_ctx);
}

static marmot_error_t cpu_unary_relu_accelerate_f32(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!cpu_unary_have_accelerate(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    const float zero = 0.0f;
    vDSP_vthr((const float *)x, 1, &zero, (float *)out, 1, (vDSP_Length)n);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_relu_accelerate_f64(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!cpu_unary_have_accelerate(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    const double zero = 0.0;
    vDSP_vthrD((const double *)x, 1, &zero, (double *)out, 1, (vDSP_Length)n);
    return MARMOT_SUCCESS;
}

static void cpu_unary_accelerate_sigmoid_block(float *dst, const float *src, size_t count, float *scratch) {
    const float neg_one = -1.0f;
    const float one = 1.0f;
    vDSP_vsmul(src, 1, &neg_one, scratch, 1, count);
    int len = (int)count;
    vvexpf(scratch, scratch, &len);
    vDSP_vsadd(scratch, 1, &one, scratch, 1, count);
    vDSP_svdiv(&one, scratch, 1, dst, 1, count);
}

static void cpu_unary_accelerate_load_bias(
    float *dst, const float *src, const float *bias, size_t feature_dim, bool bias_is_scalar, size_t block,
    size_t global_index
) {
    if (bias_is_scalar) {
        float scalar = bias[0];
        vDSP_vsadd(src, 1, &scalar, dst, 1, block);
        return;
    }
    if (feature_dim == 0) {
        memset(dst, 0, block * sizeof(float));
        return;
    }
    size_t bias_offset = global_index % feature_dim;
    size_t produced = 0;
    while (produced < block) {
        size_t remaining = block - produced;
        size_t bias_chunk = feature_dim - bias_offset;
        if (bias_chunk > remaining) {
            bias_chunk = remaining;
        }
        vDSP_vadd(src + produced, 1, bias + bias_offset, 1, dst + produced, 1, bias_chunk);
        produced += bias_chunk;
        bias_offset += bias_chunk;
        if (bias_offset == feature_dim) {
            bias_offset = 0;
        }
    }
}

static inline float cpu_unary_gelu_eval(float x) {
    const float inv_sqrt2 = 0.7071067811865475f;
    return x * 0.5f * (1.0f + erff(x * inv_sqrt2));
}

static inline float cpu_unary_elu_eval(float x, float alpha) {
    return x >= 0.0f ? x : alpha * (expf(x) - 1.0f);
}

static inline float cpu_unary_selu_eval(float x, float alpha, float lambda) {
    float inner = cpu_unary_elu_eval(x, alpha);
    return lambda * inner;
}

static inline float cpu_unary_leaky_eval(float x, float slope) {
    return x >= 0.0f ? x : slope * x;
}

static void cpu_unary_gelu_tanh_block(float *dst, const float *src, size_t count, float *scratch) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    const float half = 0.5f;
    const float one = 1.0f;
    vDSP_vsq(src, 1, scratch, 1, count);
    vDSP_vmul(scratch, 1, src, 1, scratch, 1, count);
    vDSP_vsmul(scratch, 1, &coeff, scratch, 1, count);
    vDSP_vadd(src, 1, scratch, 1, scratch, 1, count);
    vDSP_vsmul(scratch, 1, &sqrt_2_over_pi, scratch, 1, count);
    int len = (int)count;
    vvtanhf(scratch, scratch, &len);
    vDSP_vsadd(scratch, 1, &one, scratch, 1, count);
    vDSP_vmul(scratch, 1, src, 1, scratch, 1, count);
    vDSP_vsmul(scratch, 1, &half, dst, 1, count);
}

static marmot_error_t cpu_unary_sigmoid_accelerate_f32(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!cpu_unary_have_accelerate(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    size_t chunk = k_cpu_unary_accelerate_chunk;
    float *scratch = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    if (scratch == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    const float *src = (const float *)x;
    float *dst = (float *)out;
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        cpu_unary_accelerate_sigmoid_block(dst + offset, src + offset, block, scratch);
    }
    free(scratch);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_tanh_accelerate_f32(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!cpu_unary_have_accelerate(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    const float *src = (const float *)x;
    float *dst = (float *)out;
    size_t chunk = k_cpu_unary_accelerate_chunk;
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        int len = (int)block;
        vvtanhf(dst + offset, src + offset, &len);
    }
    return MARMOT_SUCCESS;
}

static void cpu_unary_accelerate_apply_affine(float *dst, const float *src, size_t count, float slope, bool relu_like) {
    if (relu_like && slope == 0.0f) {
        const float zero = 0.0f;
        vDSP_vthr(src, 1, &zero, dst, 1, (vDSP_Length)count);
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        dst[i] = cpu_unary_leaky_eval(src[i], slope);
    }
}

static marmot_error_t cpu_unary_leaky_relu_accelerate_f32(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    if (!cpu_unary_have_accelerate(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    float slope = params != nullptr ? params->alpha : 0.01f;
    const float *src = (const float *)x;
    float *dst = (float *)out;
    size_t chunk = k_cpu_unary_accelerate_chunk;
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        cpu_unary_accelerate_apply_affine(dst + offset, src + offset, block, slope, slope == 0.0f);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_prelu_accelerate_f32(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    if (!cpu_unary_have_accelerate(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (params == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "PReLU requires scalar slope parameter");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (params->parameter_tensor != nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Tensor slopes are not implemented for Accelerate PReLU");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    float slope = params->alpha;
    const float *src = (const float *)x;
    float *dst = (float *)out;
    size_t chunk = k_cpu_unary_accelerate_chunk;
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        cpu_unary_accelerate_apply_affine(dst + offset, src + offset, block, slope, false);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_elu_accelerate_f32(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    if (!cpu_unary_have_accelerate(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    float alpha = params != nullptr ? params->alpha : 1.0f;
    const float *src = (const float *)x;
    float *dst = (float *)out;
    size_t chunk = k_cpu_unary_accelerate_chunk;
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        for (size_t i = 0; i < block; ++i) {
            dst[offset + i] = cpu_unary_elu_eval(src[offset + i], alpha);
        }
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_selu_accelerate_f32(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    if (!cpu_unary_have_accelerate(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    const float default_alpha = 1.6732632423543772f;
    const float default_lambda = 1.0507009873554804f;
    float alpha = params != nullptr ? params->alpha : default_alpha;
    float lambda = params != nullptr ? params->beta : default_lambda;
    const float *src = (const float *)x;
    float *dst = (float *)out;
    size_t chunk = k_cpu_unary_accelerate_chunk;
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        for (size_t i = 0; i < block; ++i) {
            dst[offset + i] = cpu_unary_selu_eval(src[offset + i], alpha, lambda);
        }
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_apply_affine(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n, float slope, bool relu_like
) {
    (void)params;
    if (!cpu_unary_have_accelerate(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    size_t chunk = k_cpu_unary_accelerate_chunk;
    float *tmp = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    if (tmp == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    const float *src = (const float *)x;
    const float *bias_f32 = (const float *)bias;
    float *dst = (float *)out;
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        cpu_unary_accelerate_load_bias(tmp, src + offset, bias_f32, feature_dim, bias_is_scalar, block, offset);
        cpu_unary_accelerate_apply_affine(dst + offset, tmp, block, slope, relu_like);
    }
    free(tmp);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_leaky_relu_accelerate_f32(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    float slope = params != nullptr ? params->alpha : 0.01f;
    return cpu_unary_fused_bias_apply_affine(
        device_ctx, x, bias, feature_dim, bias_is_scalar, params, out, n, slope, slope == 0.0f
    );
}

static marmot_error_t cpu_unary_fused_bias_prelu_accelerate_f32(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    if (params == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "PReLU requires scalar slope parameter");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (params->parameter_tensor != nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Tensor slopes are not implemented for Accelerate PReLU");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    return cpu_unary_fused_bias_apply_affine(
        device_ctx, x, bias, feature_dim, bias_is_scalar, params, out, n, params->alpha, false
    );
}

static marmot_error_t cpu_unary_fused_bias_elu_accelerate_f32(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    if (!cpu_unary_have_accelerate(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    float alpha = params != nullptr ? params->alpha : 1.0f;
    size_t chunk = k_cpu_unary_accelerate_chunk;
    float *tmp = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    if (tmp == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    const float *src = (const float *)x;
    const float *bias_f32 = (const float *)bias;
    float *dst = (float *)out;
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        cpu_unary_accelerate_load_bias(tmp, src + offset, bias_f32, feature_dim, bias_is_scalar, block, offset);
        for (size_t i = 0; i < block; ++i) {
            dst[offset + i] = cpu_unary_elu_eval(tmp[i], alpha);
        }
    }
    free(tmp);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_selu_accelerate_f32(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    if (!cpu_unary_have_accelerate(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    const float default_alpha = 1.6732632423543772f;
    const float default_lambda = 1.0507009873554804f;
    float alpha = params != nullptr ? params->alpha : default_alpha;
    float lambda = params != nullptr ? params->beta : default_lambda;
    size_t chunk = k_cpu_unary_accelerate_chunk;
    float *tmp = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    if (tmp == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    const float *src = (const float *)x;
    const float *bias_f32 = (const float *)bias;
    float *dst = (float *)out;
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        cpu_unary_accelerate_load_bias(tmp, src + offset, bias_f32, feature_dim, bias_is_scalar, block, offset);
        for (size_t i = 0; i < block; ++i) {
            dst[offset + i] = cpu_unary_selu_eval(tmp[i], alpha, lambda);
        }
    }
    free(tmp);
    return MARMOT_SUCCESS;
}
static marmot_error_t cpu_unary_silu_accelerate_f32(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!cpu_unary_have_accelerate(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    size_t chunk = k_cpu_unary_accelerate_chunk;
    float *scratch = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    if (scratch == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    const float *src = (const float *)x;
    float *dst = (float *)out;
    bool alias = (dst == src);
    float *src_copy = nullptr;
    if (alias) {
        src_copy = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
        if (src_copy == nullptr) {
            free(scratch);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        const float *block_src = src + offset;
        if (alias) {
            memcpy(src_copy, block_src, block * sizeof(float));
            block_src = src_copy;
        }
        cpu_unary_accelerate_sigmoid_block(dst + offset, block_src, block, scratch);
        vDSP_vmul(dst + offset, 1, block_src, 1, dst + offset, 1, block);
    }
    free(src_copy);
    free(scratch);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_gelu_tanh_accelerate_f32(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!cpu_unary_have_accelerate(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    size_t chunk = k_cpu_unary_accelerate_chunk;
    float *scratch = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    if (scratch == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    const float *src = (const float *)x;
    float *dst = (float *)out;
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        cpu_unary_gelu_tanh_block(dst + offset, src + offset, block, scratch);
    }
    free(scratch);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_gelu_accelerate_f32(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!cpu_unary_have_accelerate(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    const float *src = (const float *)x;
    float *dst = (float *)out;
    size_t chunk = k_cpu_unary_accelerate_chunk;
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        for (size_t i = 0; i < block; ++i) {
            dst[offset + i] = cpu_unary_gelu_eval(src[offset + i]);
        }
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_relu_accelerate_f32(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!cpu_unary_have_accelerate(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    const float zero = 0.0f;
    size_t chunk = k_cpu_unary_accelerate_chunk;
    float *tmp = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    if (tmp == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    const float *src = (const float *)x;
    const float *bias_f32 = (const float *)bias;
    float *dst = (float *)out;
    size_t offset = 0;
    while (offset < n) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        cpu_unary_accelerate_load_bias(tmp, src + offset, bias_f32, feature_dim, bias_is_scalar, block, offset);
        vDSP_vthr(tmp, 1, &zero, dst + offset, 1, (vDSP_Length)block);
        offset += block;
    }
    free(tmp);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_accelerate_via_f32_activation(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n,
    cpu_activation_fn f32_kernel, cpu_unary_acc_to_f32_fn to_f32, cpu_unary_acc_from_f32_fn from_f32, size_t elem_size
) {
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    size_t chunk = n < k_cpu_unary_accelerate_chunk ? n : k_cpu_unary_accelerate_chunk;
    float *tmp_in = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    float *tmp_out = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    if (tmp_in == nullptr || tmp_out == nullptr) {
        free(tmp_in);
        free(tmp_out);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    const uint8_t *src_bytes = (const uint8_t *)x;
    uint8_t *dst_bytes = (uint8_t *)out;
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        to_f32(device_ctx, tmp_in, src_bytes + offset * elem_size, block);
        marmot_error_t err = f32_kernel(device_ctx, tmp_in, params, tmp_out, block);
        if (err != MARMOT_SUCCESS) {
            free(tmp_in);
            free(tmp_out);
            return err;
        }
        from_f32(device_ctx, dst_bytes + offset * elem_size, tmp_out, block);
    }
    free(tmp_in);
    free(tmp_out);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_accelerate_via_f32_fused(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n, cpu_fused_bias_activation_fn f32_kernel,
    cpu_unary_acc_to_f32_fn to_f32, cpu_unary_acc_from_f32_fn from_f32, size_t elem_size
) {
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    size_t chunk = n < k_cpu_unary_accelerate_chunk ? n : k_cpu_unary_accelerate_chunk;
    size_t bias_elems = bias_is_scalar ? 1 : feature_dim;
    float *bias_f32 = (float *)marmot_aligned_alloc(64, bias_elems * sizeof(float));
    float *tmp_in = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    float *tmp_out = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    if (bias_f32 == nullptr || tmp_in == nullptr || tmp_out == nullptr) {
        free(bias_f32);
        free(tmp_in);
        free(tmp_out);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    to_f32(device_ctx, bias_f32, bias, bias_elems);
    const uint8_t *src_bytes = (const uint8_t *)x;
    uint8_t *dst_bytes = (uint8_t *)out;
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        to_f32(device_ctx, tmp_in, src_bytes + offset * elem_size, block);
        marmot_error_t err =
            f32_kernel(device_ctx, tmp_in, bias_f32, feature_dim, bias_is_scalar, params, tmp_out, block);
        if (err != MARMOT_SUCCESS) {
            free(bias_f32);
            free(tmp_in);
            free(tmp_out);
            return err;
        }
        from_f32(device_ctx, dst_bytes + offset * elem_size, tmp_out, block);
    }
    free(bias_f32);
    free(tmp_in);
    free(tmp_out);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_sigmoid_accelerate_f32(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!cpu_unary_have_accelerate(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    size_t chunk = k_cpu_unary_accelerate_chunk;
    float *tmp = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    float *scratch = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    if (tmp == nullptr || scratch == nullptr) {
        free(tmp);
        free(scratch);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    const float *src = (const float *)x;
    const float *bias_f32 = (const float *)bias;
    float *dst = (float *)out;
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        cpu_unary_accelerate_load_bias(tmp, src + offset, bias_f32, feature_dim, bias_is_scalar, block, offset);
        cpu_unary_accelerate_sigmoid_block(dst + offset, tmp, block, scratch);
    }
    free(tmp);
    free(scratch);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_silu_accelerate_f32(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!cpu_unary_have_accelerate(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    size_t chunk = k_cpu_unary_accelerate_chunk;
    float *tmp = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    float *scratch = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    if (tmp == nullptr || scratch == nullptr) {
        free(tmp);
        free(scratch);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    const float *src = (const float *)x;
    const float *bias_f32 = (const float *)bias;
    float *dst = (float *)out;
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        cpu_unary_accelerate_load_bias(tmp, src + offset, bias_f32, feature_dim, bias_is_scalar, block, offset);
        cpu_unary_accelerate_sigmoid_block(dst + offset, tmp, block, scratch);
        vDSP_vmul(dst + offset, 1, tmp, 1, dst + offset, 1, block);
    }
    free(tmp);
    free(scratch);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_gelu_tanh_accelerate_f32(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!cpu_unary_have_accelerate(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    size_t chunk = k_cpu_unary_accelerate_chunk;
    float *tmp = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    float *scratch = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    if (tmp == nullptr || scratch == nullptr) {
        free(tmp);
        free(scratch);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    const float *src = (const float *)x;
    const float *bias_f32 = (const float *)bias;
    float *dst = (float *)out;
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        cpu_unary_accelerate_load_bias(tmp, src + offset, bias_f32, feature_dim, bias_is_scalar, block, offset);
        cpu_unary_gelu_tanh_block(dst + offset, tmp, block, scratch);
    }
    free(tmp);
    free(scratch);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_unary_fused_bias_gelu_accelerate_f32(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
) {
    (void)params;
    if (!cpu_unary_have_accelerate(device_ctx)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    size_t chunk = k_cpu_unary_accelerate_chunk;
    float *tmp = (float *)marmot_aligned_alloc(64, chunk * sizeof(float));
    if (tmp == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    const float *src = (const float *)x;
    const float *bias_f32 = (const float *)bias;
    float *dst = (float *)out;
    for (size_t offset = 0; offset < n; offset += chunk) {
        size_t block = chunk;
        if (block > n - offset) {
            block = n - offset;
        }
        cpu_unary_accelerate_load_bias(tmp, src + offset, bias_f32, feature_dim, bias_is_scalar, block, offset);
        for (size_t i = 0; i < block; ++i) {
            dst[offset + i] = cpu_unary_gelu_eval(tmp[i]);
        }
    }
    free(tmp);
    return MARMOT_SUCCESS;
}

static inline size_t cpu_unary_acc_elem_size_f16(void) {
    return sizeof(marmot_float16_t);
}

static inline size_t cpu_unary_acc_elem_size_bf16(void) {
    return sizeof(marmot_bfloat16_t);
}

#define CPU_UNARY_ACCELERATE_TO_F32_FN(tag) cpu_unary_acc_to_f32_##tag
#define CPU_UNARY_ACCELERATE_FROM_F32_FN(tag) cpu_unary_acc_from_f32_##tag
#define CPU_UNARY_ACCELERATE_ELEM_SIZE(tag) cpu_unary_acc_elem_size_##tag()

#define CPU_UNARY_ACCELERATE_DEFINE_VIA_ACT(tag, name, base_fn)                                                        \
    static marmot_error_t cpu_unary_##name##_##tag##_accelerate(                                                       \
        const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n           \
    ) {                                                                                                                \
        return cpu_unary_accelerate_via_f32_activation(                                                                \
            device_ctx, x, params, out, n, base_fn, CPU_UNARY_ACCELERATE_TO_F32_FN(tag),                               \
            CPU_UNARY_ACCELERATE_FROM_F32_FN(tag), CPU_UNARY_ACCELERATE_ELEM_SIZE(tag)                                 \
        );                                                                                                             \
    }

#define CPU_UNARY_ACCELERATE_DEFINE_VIA_FUSED(tag, name, base_fn)                                                      \
    static marmot_error_t cpu_unary_fused_bias_##name##_##tag##_accelerate(                                            \
        const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,              \
        const marmot_activation_params_t *params, void *out, size_t n                                                  \
    ) {                                                                                                                \
        return cpu_unary_accelerate_via_f32_fused(                                                                     \
            device_ctx, x, bias, feature_dim, bias_is_scalar, params, out, n, base_fn,                                 \
            CPU_UNARY_ACCELERATE_TO_F32_FN(tag), CPU_UNARY_ACCELERATE_FROM_F32_FN(tag),                                \
            CPU_UNARY_ACCELERATE_ELEM_SIZE(tag)                                                                        \
        );                                                                                                             \
    }

#define CPU_UNARY_ACCELERATE_FOR_EACH_ACT(MACRO, tag)                                                                  \
    MACRO(tag, relu, cpu_unary_relu_accelerate_f32)                                                                    \
    MACRO(tag, gelu, cpu_unary_gelu_accelerate_f32)                                                                    \
    MACRO(tag, gelu_tanh, cpu_unary_gelu_tanh_accelerate_f32)                                                          \
    MACRO(tag, silu, cpu_unary_silu_accelerate_f32)                                                                    \
    MACRO(tag, sigmoid, cpu_unary_sigmoid_accelerate_f32)                                                              \
    MACRO(tag, tanh_act, cpu_unary_tanh_accelerate_f32)                                                                \
    MACRO(tag, elu, cpu_unary_elu_accelerate_f32)                                                                      \
    MACRO(tag, selu, cpu_unary_selu_accelerate_f32)                                                                    \
    MACRO(tag, leaky_relu, cpu_unary_leaky_relu_accelerate_f32)                                                        \
    MACRO(tag, prelu, cpu_unary_prelu_accelerate_f32)

#define CPU_UNARY_ACCELERATE_FOR_EACH_FUSED(MACRO, tag)                                                                \
    MACRO(tag, relu, cpu_unary_fused_bias_relu_accelerate_f32)                                                         \
    MACRO(tag, gelu, cpu_unary_fused_bias_gelu_accelerate_f32)                                                         \
    MACRO(tag, gelu_tanh, cpu_unary_fused_bias_gelu_tanh_accelerate_f32)                                               \
    MACRO(tag, silu, cpu_unary_fused_bias_silu_accelerate_f32)                                                         \
    MACRO(tag, sigmoid, cpu_unary_fused_bias_sigmoid_accelerate_f32)                                                   \
    MACRO(tag, elu, cpu_unary_fused_bias_elu_accelerate_f32)                                                           \
    MACRO(tag, selu, cpu_unary_fused_bias_selu_accelerate_f32)                                                         \
    MACRO(tag, leaky_relu, cpu_unary_fused_bias_leaky_relu_accelerate_f32)                                             \
    MACRO(tag, prelu, cpu_unary_fused_bias_prelu_accelerate_f32)

CPU_UNARY_ACCELERATE_FOR_EACH_ACT(CPU_UNARY_ACCELERATE_DEFINE_VIA_ACT, f16)
CPU_UNARY_ACCELERATE_FOR_EACH_FUSED(CPU_UNARY_ACCELERATE_DEFINE_VIA_FUSED, f16)
CPU_UNARY_ACCELERATE_FOR_EACH_ACT(CPU_UNARY_ACCELERATE_DEFINE_VIA_ACT, bf16)
CPU_UNARY_ACCELERATE_FOR_EACH_FUSED(CPU_UNARY_ACCELERATE_DEFINE_VIA_FUSED, bf16)

const cpu_unary_traits_t cpu_unary_f32_accelerate_traits = {
    .dtype = MARMOT_DTYPE_FLOAT32,
    .impl_kind = CPU_UNARY_IMPL_ACCELERATE,
    .ops = {
        .relu = cpu_unary_relu_accelerate_f32,
        .gelu = cpu_unary_gelu_accelerate_f32,
        .gelu_tanh = cpu_unary_gelu_tanh_accelerate_f32,
        .silu = cpu_unary_silu_accelerate_f32,
        .sigmoid = cpu_unary_sigmoid_accelerate_f32,
        .tanh_act = cpu_unary_tanh_accelerate_f32,
        .elu = cpu_unary_elu_accelerate_f32,
        .selu = cpu_unary_selu_accelerate_f32,
        .leaky_relu = cpu_unary_leaky_relu_accelerate_f32,
        .prelu = cpu_unary_prelu_accelerate_f32,
        .fused_bias_relu = cpu_unary_fused_bias_relu_accelerate_f32,
        .fused_bias_gelu = cpu_unary_fused_bias_gelu_accelerate_f32,
        .fused_bias_gelu_tanh = cpu_unary_fused_bias_gelu_tanh_accelerate_f32,
        .fused_bias_silu = cpu_unary_fused_bias_silu_accelerate_f32,
        .fused_bias_sigmoid = cpu_unary_fused_bias_sigmoid_accelerate_f32,
        .fused_bias_elu = cpu_unary_fused_bias_elu_accelerate_f32,
        .fused_bias_selu = cpu_unary_fused_bias_selu_accelerate_f32,
        .fused_bias_leaky_relu = cpu_unary_fused_bias_leaky_relu_accelerate_f32,
        .fused_bias_prelu = cpu_unary_fused_bias_prelu_accelerate_f32,
        .impl_name = "accelerate:relu",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_f32_accelerate_traits)

const cpu_unary_traits_t cpu_unary_f64_accelerate_traits = {
    .dtype = MARMOT_DTYPE_FLOAT64,
    .impl_kind = CPU_UNARY_IMPL_ACCELERATE,
    .ops = {
        .relu = cpu_unary_relu_accelerate_f64,
        .impl_name = "accelerate:relu",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_f64_accelerate_traits)

const cpu_unary_traits_t cpu_unary_f16_accelerate_traits = {
    .dtype = MARMOT_DTYPE_FLOAT16,
    .impl_kind = CPU_UNARY_IMPL_ACCELERATE,
    .ops = {
        .relu = cpu_unary_relu_f16_accelerate,
        .gelu = cpu_unary_gelu_f16_accelerate,
        .gelu_tanh = cpu_unary_gelu_tanh_f16_accelerate,
        .silu = cpu_unary_silu_f16_accelerate,
        .sigmoid = cpu_unary_sigmoid_f16_accelerate,
        .tanh_act = cpu_unary_tanh_act_f16_accelerate,
        .elu = cpu_unary_elu_f16_accelerate,
        .selu = cpu_unary_selu_f16_accelerate,
        .leaky_relu = cpu_unary_leaky_relu_f16_accelerate,
        .prelu = cpu_unary_prelu_f16_accelerate,
        .fused_bias_relu = cpu_unary_fused_bias_relu_f16_accelerate,
        .fused_bias_gelu = cpu_unary_fused_bias_gelu_f16_accelerate,
        .fused_bias_gelu_tanh = cpu_unary_fused_bias_gelu_tanh_f16_accelerate,
        .fused_bias_silu = cpu_unary_fused_bias_silu_f16_accelerate,
        .fused_bias_sigmoid = cpu_unary_fused_bias_sigmoid_f16_accelerate,
        .fused_bias_elu = cpu_unary_fused_bias_elu_f16_accelerate,
        .fused_bias_selu = cpu_unary_fused_bias_selu_f16_accelerate,
        .fused_bias_leaky_relu = cpu_unary_fused_bias_leaky_relu_f16_accelerate,
        .fused_bias_prelu = cpu_unary_fused_bias_prelu_f16_accelerate,
        .impl_name = "accelerate:f16-via-f32",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_f16_accelerate_traits)

const cpu_unary_traits_t cpu_unary_bf16_accelerate_traits = {
    .dtype = MARMOT_DTYPE_BFLOAT16,
    .impl_kind = CPU_UNARY_IMPL_ACCELERATE,
    .ops = {
        .relu = cpu_unary_relu_bf16_accelerate,
        .gelu = cpu_unary_gelu_bf16_accelerate,
        .gelu_tanh = cpu_unary_gelu_tanh_bf16_accelerate,
        .silu = cpu_unary_silu_bf16_accelerate,
        .sigmoid = cpu_unary_sigmoid_bf16_accelerate,
        .tanh_act = cpu_unary_tanh_act_bf16_accelerate,
        .elu = cpu_unary_elu_bf16_accelerate,
        .selu = cpu_unary_selu_bf16_accelerate,
        .leaky_relu = cpu_unary_leaky_relu_bf16_accelerate,
        .prelu = cpu_unary_prelu_bf16_accelerate,
        .fused_bias_relu = cpu_unary_fused_bias_relu_bf16_accelerate,
        .fused_bias_gelu = cpu_unary_fused_bias_gelu_bf16_accelerate,
        .fused_bias_gelu_tanh = cpu_unary_fused_bias_gelu_tanh_bf16_accelerate,
        .fused_bias_silu = cpu_unary_fused_bias_silu_bf16_accelerate,
        .fused_bias_sigmoid = cpu_unary_fused_bias_sigmoid_bf16_accelerate,
        .fused_bias_elu = cpu_unary_fused_bias_elu_bf16_accelerate,
        .fused_bias_selu = cpu_unary_fused_bias_selu_bf16_accelerate,
        .fused_bias_leaky_relu = cpu_unary_fused_bias_leaky_relu_bf16_accelerate,
        .fused_bias_prelu = cpu_unary_fused_bias_prelu_bf16_accelerate,
        .impl_name = "accelerate:bf16-via-f32",
    },
};
CPU_UNARY_REGISTER_TRAITS(cpu_unary_bf16_accelerate_traits)

#endif // MARMOT_ENABLE_ACCELERATE
