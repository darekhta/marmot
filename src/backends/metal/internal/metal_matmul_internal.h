#ifndef METAL_MATMUL_INTERNAL_H
#define METAL_MATMUL_INTERNAL_H

#include "metal_backend_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct metal_matmul_activation_params {
    float alpha;
    float beta;
    float gamma;
    float delta;
} metal_matmul_activation_params_t;

typedef struct metal_matmul_epilogue_traits {
    marmot_dtype_t dtype;
    bool supports_bias;
    bool activation_supported[MARMOT_DEVICE_UNARY_COUNT];
    const char *kernel_name;
    const char *rope_kernel_name;
} metal_matmul_epilogue_traits_t;

bool metal_matmul_bias_dtype_supported(marmot_dtype_t out_dtype, marmot_dtype_t bias_dtype);
id<MTLBuffer> metal_matmul_create_positions_buffer(metal_context_t *ctx, const marmot_tensor_t *positions, size_t rows);
id<MTLBuffer> metal_matmul_prepare_freq_buffer(
    metal_context_t *ctx, size_t dim, const marmot_rope_params_t *params, float *attn_scale_out
);
metal_matmul_activation_params_t
metal_matmul_build_activation_params(marmot_device_unary_op_t op, const marmot_activation_params_t *params);
const metal_matmul_epilogue_traits_t *metal_matmul_select_epilogue(marmot_dtype_t dtype);

#ifdef __cplusplus
}
#endif

#endif // METAL_MATMUL_INTERNAL_H
