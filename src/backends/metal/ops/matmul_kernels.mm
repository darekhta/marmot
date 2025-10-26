// SPDX-License-Identifier: MIT
// Metal matmul kernel implementations for direct dispatch

#include "matmul_kernels.h"

#include "internal/metal_matmul_internal.h"
#include "metal_backend_internal.h"

#ifdef __APPLE__

// Forward declarations for generic kernel dispatch
extern "C" marmot_error_t metal_matmul_generic(
    metal_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out, const char *kernel_name
);
extern "C" marmot_error_t marmot_metal_gemm(
    metal_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out, bool transpose_b
);

// NT layout kernels (weight stored as [M, K], standard linear layer format)
marmot_error_t metal_matmul_f32_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    marmot_error_t status = marmot_metal_gemm(ctx, input, weight, epilogue, out, true);
    if (status == MARMOT_SUCCESS) {
        return status;
    }
    return metal_matmul_generic(ctx, input, weight, epilogue, out, "matmul_f32_nt");
}

marmot_error_t metal_matmul_f16_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    marmot_error_t status = marmot_metal_gemm(ctx, input, weight, epilogue, out, true);
    if (status == MARMOT_SUCCESS) {
        return status;
    }
    return metal_matmul_generic(ctx, input, weight, epilogue, out, "matmul_f16_nt");
}

marmot_error_t metal_matmul_bf16_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    marmot_error_t status = marmot_metal_gemm(ctx, input, weight, epilogue, out, true);
    if (status == MARMOT_SUCCESS) {
        return status;
    }
    return metal_matmul_generic(ctx, input, weight, epilogue, out, "matmul_bf16_nt");
}

// NN layout kernels (weight stored as [K, M], torch-style matmul)
marmot_error_t metal_matmul_f32_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    marmot_error_t status = marmot_metal_gemm(ctx, input, weight, epilogue, out, false);
    if (status == MARMOT_SUCCESS) {
        return status;
    }
    return metal_matmul_generic(ctx, input, weight, epilogue, out, "matmul_f32_nn");
}

marmot_error_t metal_matmul_f16_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    marmot_error_t status = marmot_metal_gemm(ctx, input, weight, epilogue, out, false);
    if (status == MARMOT_SUCCESS) {
        return status;
    }
    return metal_matmul_generic(ctx, input, weight, epilogue, out, "matmul_f16_nn");
}

marmot_error_t metal_matmul_bf16_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    marmot_error_t status = marmot_metal_gemm(ctx, input, weight, epilogue, out, false);
    if (status == MARMOT_SUCCESS) {
        return status;
    }
    return metal_matmul_generic(ctx, input, weight, epilogue, out, "matmul_bf16_nn");
}

// FP8 kernels
#if MARMOT_ENABLE_FP8
marmot_error_t metal_matmul_fp8_e4m3_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    return metal_matmul_generic(ctx, input, weight, epilogue, out, "matmul_fp8_e4m3_nt");
}

marmot_error_t metal_matmul_fp8_e5m2_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    return metal_matmul_generic(ctx, input, weight, epilogue, out, "matmul_fp8_e5m2_nt");
}

marmot_error_t metal_matmul_fp8_e4m3_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    return metal_matmul_generic(ctx, input, weight, epilogue, out, "matmul_fp8_e4m3_nn");
}

marmot_error_t metal_matmul_fp8_e5m2_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    return metal_matmul_generic(ctx, input, weight, epilogue, out, "matmul_fp8_e5m2_nn");
}
#else
// Stub implementations when FP8 is disabled
marmot_error_t metal_matmul_fp8_e4m3_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)epilogue;
    (void)out;
    marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "FP8 matmul not enabled in this build");
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

marmot_error_t metal_matmul_fp8_e5m2_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)epilogue;
    (void)out;
    marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "FP8 matmul not enabled in this build");
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

marmot_error_t metal_matmul_fp8_e4m3_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)epilogue;
    (void)out;
    marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "FP8 matmul not enabled in this build");
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

marmot_error_t metal_matmul_fp8_e5m2_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)epilogue;
    (void)out;
    marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "FP8 matmul not enabled in this build");
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}
#endif

// Direct dispatch helper for QKV fallback - avoids runtime layout detection
// Assumes NT layout (weight transposed) which is standard for linear layers
marmot_error_t metal_matmul_direct(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    if (device_ctx == nullptr || input == nullptr || weight == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    switch (input->dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return metal_matmul_f32_nt(device_ctx, input, weight, epilogue, out);
    case MARMOT_DTYPE_FLOAT16:
        return metal_matmul_f16_nt(device_ctx, input, weight, epilogue, out);
    case MARMOT_DTYPE_BFLOAT16:
        return metal_matmul_bf16_nt(device_ctx, input, weight, epilogue, out);
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
        return metal_matmul_fp8_e4m3_nt(device_ctx, input, weight, epilogue, out);
    case MARMOT_DTYPE_FLOAT8_E5M2:
        return metal_matmul_fp8_e5m2_nt(device_ctx, input, weight, epilogue, out);
#endif
    default:
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unsupported dtype for Metal matmul");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
}

#endif // __APPLE__
