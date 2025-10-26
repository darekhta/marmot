#ifndef METAL_MATMUL_QUANT_DISPATCH_H
#define METAL_MATMUL_QUANT_DISPATCH_H

#include "marmot/ops/matmul.h"

#include "metal_backend_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct metal_kquant_kernels {
    const char *kernel_opt;
    const char *kernel_nr2;
    const char *kernel_small;
    const char *kernel_mv_ext;
    const char *kernel_mm;
    const char *kernel_mm16;
} metal_kquant_kernels_t;

marmot_error_t metal_matmul_quant_dispatch(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);

marmot_error_t metal_matmul_quant_dispatch_direct(
    metal_context_t *ctx, const char *kernel_name, const char *log_label, const char *missing_kernel_msg,
    const marmot_tensor_t *input, const marmot_tensor_t *weight, marmot_tensor_t *out, size_t N, size_t K, size_t M,
    size_t weight_blocks_per_row, const marmot_matmul_epilogue_t *epilogue, size_t ep_feature_dim, bool ep_bias_scalar,
    const marmot_rope_params_t *rope
);

marmot_error_t metal_matmul_quant_dispatch_packed(
    metal_context_t *ctx, const char *kernel_name, const char *quant_kernel, const char *log_label,
    const marmot_tensor_t *input, const marmot_tensor_t *weight, marmot_tensor_t *out, size_t N, size_t K, size_t M,
    size_t activation_blocks_per_row, size_t activation_block_bytes, size_t weight_blocks_per_row,
    bool uses_super_blocks, const marmot_matmul_epilogue_t *epilogue, size_t ep_feature_dim, bool ep_bias_scalar,
    const marmot_rope_params_t *rope
);

marmot_error_t metal_matmul_quant_dispatch_k_direct(
    metal_context_t *ctx, const metal_kquant_kernels_t *kernels, const char *log_label, const marmot_tensor_t *input,
    const marmot_tensor_t *weight, marmot_tensor_t *out, size_t N, size_t K, size_t M, size_t weight_blocks_per_row,
    const marmot_matmul_epilogue_t *epilogue, size_t ep_feature_dim, bool ep_bias_scalar,
    const marmot_rope_params_t *rope
);

// Generated dispatch function - direct kernel calls without traits lookup
marmot_error_t metal_matmul_quantized_dispatch(
    metal_context_t *ctx, marmot_quant_kind_t quant_kind, marmot_dtype_t input_dtype, marmot_dtype_t output_dtype,
    const marmot_tensor_t *input, const marmot_tensor_t *weight, marmot_tensor_t *out, size_t N, size_t K, size_t M,
    size_t weight_blocks_per_row, bool uses_super_blocks, const marmot_matmul_epilogue_t *epilogue,
    size_t ep_feature_dim, bool ep_bias_scalar, const marmot_rope_params_t *rope
);

#ifdef __cplusplus
}
#endif

#endif // METAL_MATMUL_QUANT_DISPATCH_H
