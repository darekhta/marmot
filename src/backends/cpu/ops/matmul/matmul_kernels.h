// SPDX-License-Identifier: MIT
// Public declarations for CPU matmul kernels called directly by generated dispatch

#ifndef CPU_MATMUL_KERNELS_H
#define CPU_MATMUL_KERNELS_H

#include "marmot/error.h"
#include "marmot/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Scalar kernels - NT layout (weight transposed, standard linear layer format)
marmot_error_t cpu_matmul_f32_scalar(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_f64_scalar(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_f16_scalar(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_bf16_scalar(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);

// Scalar kernels - NN layout (PyTorch matmul convention)
marmot_error_t cpu_matmul_f32_scalar_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_f64_scalar_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_f16_scalar_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_bf16_scalar_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);

// NEON blocked kernels - optimized for ARM (Apple Silicon)
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

// Accelerate kernels - Apple Accelerate framework (cblas_sgemm/cblas_dgemm)
marmot_error_t cpu_matmul_bf16_accelerate(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_f16_accelerate(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_f32_accelerate(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_f64_accelerate(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_f32_accelerate_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_f64_accelerate_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_bf16_accelerate_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_f16_accelerate_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);

// AVX2 kernels - x86-64 SIMD
marmot_error_t cpu_matmul_f16_avx2(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_bf16_avx2(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);

// FP8 kernels (always declared for codegen compatibility, stub when FP8 disabled)
marmot_error_t cpu_matmul_fp8_e4m3_scalar(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_fp8_e5m2_scalar(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_fp8_e4m3_scalar_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_fp8_e5m2_scalar_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_fp8_e4m3_accelerate(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_fp8_e5m2_accelerate(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_fp8_e4m3_accelerate_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);
marmot_error_t cpu_matmul_fp8_e5m2_accelerate_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);

// Epilogue application (bias, activation, residual)
marmot_error_t
cpu_matmul_apply_epilogue(const void *device_ctx, marmot_tensor_t *out, const marmot_matmul_epilogue_t *epilogue);

// Direct dispatch helper - selects kernel based on dtype and layout (NT assumed)
// This replaces cpu_matmul() for internal use without registry lookup
marmot_error_t cpu_matmul_direct(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);

#ifdef __cplusplus
}
#endif

#endif // CPU_MATMUL_KERNELS_H
