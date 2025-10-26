// SPDX-License-Identifier: MIT
// Public declarations for Metal matmul kernels called directly by generated dispatch

#ifndef METAL_MATMUL_KERNELS_H
#define METAL_MATMUL_KERNELS_H

#include "marmot/error.h"
#include "marmot/matmul_types.h"
#include "marmot/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// NT layout kernels (weight transposed, standard linear layer format)
// input(N×K) @ weight(M×K).T = output(N×M)
marmot_error_t metal_matmul_f32_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t metal_matmul_f16_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t metal_matmul_bf16_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);

// NN layout kernels (PyTorch matmul convention)
// input(N×K) @ weight(K×M) = output(N×M)
marmot_error_t metal_matmul_f32_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t metal_matmul_f16_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t metal_matmul_bf16_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);

// FP8 kernels (always declared for codegen compatibility, stub when FP8 disabled)
marmot_error_t metal_matmul_fp8_e4m3_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t metal_matmul_fp8_e5m2_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t metal_matmul_fp8_e4m3_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t metal_matmul_fp8_e5m2_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);

// Direct dispatch helper - selects kernel based on dtype (NT layout assumed)
// This replaces metal_matmul() for internal use without runtime layout detection
marmot_error_t metal_matmul_direct(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);

#ifdef __cplusplus
}
#endif

#endif // METAL_MATMUL_KERNELS_H
