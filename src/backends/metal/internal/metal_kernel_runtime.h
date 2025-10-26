#ifndef METAL_KERNEL_RUNTIME_H
#define METAL_KERNEL_RUNTIME_H

#include "marmot/device.h"
#include "marmot/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct metal_context metal_context_t;

typedef struct {
    float alpha;
    float beta;
    float gamma;
    float delta;
} metal_activation_params_t;

typedef struct {
    uint32_t total_elements;
    uint32_t bias_length;
    uint32_t activation;
    uint32_t flags;
    metal_activation_params_t params;
} metal_fused_bias_activation_uniforms_t;

enum {
    METAL_FUSED_BIAS_FLAG_SCALAR = 1u << 0,
    METAL_FUSED_BIAS_FLAG_HAS_BIAS = 1u << 1,
    METAL_FUSED_BIAS_FLAG_HAS_RESIDUAL = 1u << 2,
};

marmot_error_t metal_elementwise_run_binary_kernel(
    metal_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out,
    const char *kernel_name, marmot_device_binary_op_t op
);

marmot_error_t metal_elementwise_run_binary_kernel_row_strided(
    metal_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out,
    const char *kernel_name, marmot_device_binary_op_t op, uint32_t rows, uint32_t cols, size_t a_row_stride,
    size_t b_row_stride, size_t out_row_stride
);

marmot_error_t metal_elementwise_run_unary_kernel(
    metal_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const char *kernel_name,
    const char *vec4_kernel_name, const metal_activation_params_t *args, bool has_args
);

marmot_error_t metal_elementwise_run_where_kernel(
    metal_context_t *ctx, const marmot_tensor_t *mask, const marmot_tensor_t *a, const marmot_tensor_t *b,
    marmot_tensor_t *out, const char *kernel_name
);

const char *metal_kernel_name_for_where(marmot_dtype_t dtype);

marmot_error_t metal_elementwise_run_fused_bias_activation(
    metal_context_t *ctx, marmot_dtype_t dtype, const char *kernel_name, marmot_device_unary_op_t op,
    const marmot_tensor_t *x, const marmot_tensor_t *bias, marmot_tensor_t *out,
    const metal_activation_params_t *activation_params
);

#ifdef __cplusplus
}
#endif

#endif // METAL_KERNEL_RUNTIME_H
