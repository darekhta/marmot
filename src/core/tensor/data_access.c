#include "tensor_internal.h"

//------------------------------------------------------------------------------
// Read-Only Data Access (GPU → Host sync)
//------------------------------------------------------------------------------

const void *marmot_tensor_data(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(ctx == nullptr || tensor == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "tensor_data requires non-null arguments");
        return nullptr;
    }

    // Fast path for CPU backend - no sync ever needed
    if (ctx->backend_type == MARMOT_BACKEND_CPU) {
        return tensor->data;
    }

    // GPU path - delegate to to_host (it handles early-return if already synced)
    marmot_error_t err = marmot_tensor_to_host(ctx, tensor);
    if (err != MARMOT_SUCCESS) {
        return nullptr;
    }

    return tensor->data;
}

const float *marmot_tensor_data_f32(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_FLOAT32)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected FLOAT32 dtype");
        return nullptr;
    }
    return (const float *)marmot_tensor_data(ctx, tensor);
}

const marmot_float16_t *marmot_tensor_data_f16(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_FLOAT16)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected FLOAT16 dtype");
        return nullptr;
    }
    return (const marmot_float16_t *)marmot_tensor_data(ctx, tensor);
}

const marmot_bfloat16_t *marmot_tensor_data_bf16(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_BFLOAT16)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected BFLOAT16 dtype");
        return nullptr;
    }
    return (const marmot_bfloat16_t *)marmot_tensor_data(ctx, tensor);
}

const marmot_int32_t *marmot_tensor_data_i32(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_INT32)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected INT32 dtype");
        return nullptr;
    }
    return (const marmot_int32_t *)marmot_tensor_data(ctx, tensor);
}

const marmot_uint8_t *marmot_tensor_data_u8(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_UINT8)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected UINT8 dtype");
        return nullptr;
    }
    return (const marmot_uint8_t *)marmot_tensor_data(ctx, tensor);
}

const marmot_int8_t *marmot_tensor_data_i8(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_INT8)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected INT8 dtype");
        return nullptr;
    }
    return (const marmot_int8_t *)marmot_tensor_data(ctx, tensor);
}

const marmot_int16_t *marmot_tensor_data_i16(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_INT16)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected INT16 dtype");
        return nullptr;
    }
    return (const marmot_int16_t *)marmot_tensor_data(ctx, tensor);
}

const marmot_uint16_t *marmot_tensor_data_u16(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_UINT16)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected UINT16 dtype");
        return nullptr;
    }
    return (const marmot_uint16_t *)marmot_tensor_data(ctx, tensor);
}

const marmot_uint32_t *marmot_tensor_data_u32(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_UINT32)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected UINT32 dtype");
        return nullptr;
    }
    return (const marmot_uint32_t *)marmot_tensor_data(ctx, tensor);
}

const marmot_int64_t *marmot_tensor_data_i64(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_INT64)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected INT64 dtype");
        return nullptr;
    }
    return (const marmot_int64_t *)marmot_tensor_data(ctx, tensor);
}

const marmot_uint64_t *marmot_tensor_data_u64(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_UINT64)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected UINT64 dtype");
        return nullptr;
    }
    return (const marmot_uint64_t *)marmot_tensor_data(ctx, tensor);
}

const double *marmot_tensor_data_f64(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_FLOAT64)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected FLOAT64 dtype");
        return nullptr;
    }
    return (const double *)marmot_tensor_data(ctx, tensor);
}

#if MARMOT_ENABLE_FP8
const marmot_float8_e4m3_t *marmot_tensor_data_fp8_e4m3(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_FLOAT8_E4M3)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected FLOAT8_E4M3 dtype");
        return nullptr;
    }
    return (const marmot_float8_e4m3_t *)marmot_tensor_data(ctx, tensor);
}

const marmot_float8_e5m2_t *marmot_tensor_data_fp8_e5m2(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_FLOAT8_E5M2)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected FLOAT8_E5M2 dtype");
        return nullptr;
    }
    return (const marmot_float8_e5m2_t *)marmot_tensor_data(ctx, tensor);
}
#endif

//------------------------------------------------------------------------------
// Mutable Data Access (GPU → Host sync, then mark host-dirty)
//------------------------------------------------------------------------------

void *marmot_tensor_data_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(ctx == nullptr || tensor == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "tensor_data_mut requires non-null arguments");
        return nullptr;
    }

    // Fast path for CPU backend
    if (ctx->backend_type == MARMOT_BACKEND_CPU) {
        return tensor->data;
    }

    // Sync from GPU first (if needed)
    marmot_error_t err = marmot_tensor_to_host(ctx, tensor);
    if (err != MARMOT_SUCCESS) {
        return nullptr;
    }

    // Mark as host-dirty so next GPU op will re-upload
    mark_host_written(tensor);

    return tensor->data;
}

float *marmot_tensor_data_f32_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_FLOAT32)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected FLOAT32 dtype");
        return nullptr;
    }
    return (float *)marmot_tensor_data_mut(ctx, tensor);
}

marmot_float16_t *marmot_tensor_data_f16_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_FLOAT16)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected FLOAT16 dtype");
        return nullptr;
    }
    return (marmot_float16_t *)marmot_tensor_data_mut(ctx, tensor);
}

marmot_bfloat16_t *marmot_tensor_data_bf16_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_BFLOAT16)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected BFLOAT16 dtype");
        return nullptr;
    }
    return (marmot_bfloat16_t *)marmot_tensor_data_mut(ctx, tensor);
}

marmot_int32_t *marmot_tensor_data_i32_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_INT32)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected INT32 dtype");
        return nullptr;
    }
    return (marmot_int32_t *)marmot_tensor_data_mut(ctx, tensor);
}

marmot_uint8_t *marmot_tensor_data_u8_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_UINT8)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected UINT8 dtype");
        return nullptr;
    }
    return (marmot_uint8_t *)marmot_tensor_data_mut(ctx, tensor);
}

marmot_int8_t *marmot_tensor_data_i8_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_INT8)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected INT8 dtype");
        return nullptr;
    }
    return (marmot_int8_t *)marmot_tensor_data_mut(ctx, tensor);
}

marmot_int16_t *marmot_tensor_data_i16_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_INT16)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected INT16 dtype");
        return nullptr;
    }
    return (marmot_int16_t *)marmot_tensor_data_mut(ctx, tensor);
}

marmot_uint16_t *marmot_tensor_data_u16_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_UINT16)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected UINT16 dtype");
        return nullptr;
    }
    return (marmot_uint16_t *)marmot_tensor_data_mut(ctx, tensor);
}

marmot_uint32_t *marmot_tensor_data_u32_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_UINT32)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected UINT32 dtype");
        return nullptr;
    }
    return (marmot_uint32_t *)marmot_tensor_data_mut(ctx, tensor);
}

marmot_int64_t *marmot_tensor_data_i64_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_INT64)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected INT64 dtype");
        return nullptr;
    }
    return (marmot_int64_t *)marmot_tensor_data_mut(ctx, tensor);
}

marmot_uint64_t *marmot_tensor_data_u64_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_UINT64)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected UINT64 dtype");
        return nullptr;
    }
    return (marmot_uint64_t *)marmot_tensor_data_mut(ctx, tensor);
}

double *marmot_tensor_data_f64_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_FLOAT64)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected FLOAT64 dtype");
        return nullptr;
    }
    return (double *)marmot_tensor_data_mut(ctx, tensor);
}

#if MARMOT_ENABLE_FP8
marmot_float8_e4m3_t *marmot_tensor_data_fp8_e4m3_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_FLOAT8_E4M3)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected FLOAT8_E4M3 dtype");
        return nullptr;
    }
    return (marmot_float8_e4m3_t *)marmot_tensor_data_mut(ctx, tensor);
}

marmot_float8_e5m2_t *marmot_tensor_data_fp8_e5m2_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(tensor != nullptr && tensor->dtype != MARMOT_DTYPE_FLOAT8_E5M2)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected FLOAT8_E5M2 dtype");
        return nullptr;
    }
    return (marmot_float8_e5m2_t *)marmot_tensor_data_mut(ctx, tensor);
}
#endif
