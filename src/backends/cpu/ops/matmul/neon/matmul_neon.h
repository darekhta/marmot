#ifndef MARMOT_MATMUL_NEON_H
#define MARMOT_MATMUL_NEON_H

#include "marmot/error.h"
#include "marmot/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

marmot_error_t cpu_matmul_f32_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);

marmot_error_t cpu_matmul_f32_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);

marmot_error_t cpu_matmul_f64_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);

marmot_error_t cpu_matmul_f64_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);

marmot_error_t cpu_matmul_bf16_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);

marmot_error_t cpu_matmul_bf16_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);

marmot_error_t cpu_matmul_f16_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);

marmot_error_t cpu_matmul_f16_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);

marmot_error_t cpu_matmul_fp8_e4m3_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);

marmot_error_t cpu_matmul_fp8_e4m3_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);

marmot_error_t cpu_matmul_fp8_e5m2_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);

marmot_error_t cpu_matmul_fp8_e5m2_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);

#ifdef __cplusplus
}
#endif

#endif
