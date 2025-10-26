#include "marmot/device.h"

#include <stdatomic.h>
#include <stdlib.h>

#include <string.h>

#include "tensor_debug.h"
#include "tensor_internal.h"
#include "utils/dtype_ref.h"

static atomic_size_t g_tensor_alloc_bytes = 0;
static atomic_size_t g_tensor_free_bytes = 0;
static atomic_size_t g_tensor_live_bytes = 0;
static atomic_size_t g_tensor_peak_bytes = 0;
static atomic_size_t g_tensor_allocs = 0;
static atomic_size_t g_tensor_frees = 0;

void marmot_tensor_debug_record_alloc(size_t bytes) {
    atomic_fetch_add_explicit(&g_tensor_allocs, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_tensor_alloc_bytes, bytes, memory_order_relaxed);
    size_t live = atomic_fetch_add_explicit(&g_tensor_live_bytes, bytes, memory_order_relaxed) + bytes;
    size_t peak = atomic_load_explicit(&g_tensor_peak_bytes, memory_order_relaxed);
    while (live > peak) {
        if (atomic_compare_exchange_weak_explicit(
                &g_tensor_peak_bytes, &peak, live, memory_order_relaxed, memory_order_relaxed
            )) {
            break;
        }
    }
}

void marmot_tensor_debug_record_free(size_t bytes) {
    atomic_fetch_add_explicit(&g_tensor_frees, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_tensor_free_bytes, bytes, memory_order_relaxed);
    size_t current = atomic_load_explicit(&g_tensor_live_bytes, memory_order_relaxed);
    size_t next = current >= bytes ? current - bytes : 0;
    while (!atomic_compare_exchange_weak_explicit(
        &g_tensor_live_bytes, &current, next, memory_order_relaxed, memory_order_relaxed
    )) {
        next = current >= bytes ? current - bytes : 0;
    }
}

void marmot_tensor_debug_snapshot(marmot_tensor_debug_stats_t *out) {
    if (out == nullptr) {
        return;
    }
    out->live_bytes = atomic_load_explicit(&g_tensor_live_bytes, memory_order_relaxed);
    out->peak_bytes = atomic_load_explicit(&g_tensor_peak_bytes, memory_order_relaxed);
    out->alloc_bytes = atomic_load_explicit(&g_tensor_alloc_bytes, memory_order_relaxed);
    out->free_bytes = atomic_load_explicit(&g_tensor_free_bytes, memory_order_relaxed);
    out->allocs = atomic_load_explicit(&g_tensor_allocs, memory_order_relaxed);
    out->frees = atomic_load_explicit(&g_tensor_frees, memory_order_relaxed);
}

marmot_tensor_t *
marmot_tensor_create(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_dtype_t dtype) {
    marmot_clear_error();

    if (shape == nullptr || ndim == 0 || ndim > MARMOT_MAX_DIMS) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid shape or ndim");
        return nullptr;
    }

    marmot_tensor_t *tensor = calloc(1, sizeof(marmot_tensor_t));
    if (tensor == nullptr) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate tensor");
        return nullptr;
    }

    tensor->shape.ndim = ndim;
    memcpy(tensor->shape.shape, shape, ndim * sizeof(size_t));

    tensor->shape.strides[ndim - 1] = 1;
    for (ssize_t i = (ssize_t)ndim - 2; i >= 0; i--) {
        tensor->shape.strides[i] = tensor->shape.strides[i + 1] * shape[i + 1];
    }

    tensor->dtype = dtype;
    tensor->backend = ctx != nullptr ? ctx->backend_type : MARMOT_BACKEND_CPU;
    tensor->ctx = (marmot_context_t *)ctx;
    tensor->owns_data = true;
    tensor->quant_params = nullptr;
    tensor->quant_kind = MARMOT_QUANT_KIND_GENERIC;
    tensor->quant_layout = MARMOT_QUANT_LAYOUT_GENERIC;
    tensor->memory_location = MARMOT_MEMORY_HOST;
    tensor->needs_sync = false;
    tensor->packed_data = nullptr;
    tensor->packed_src_data = nullptr;
    tensor->packed_bytes = 0;
    tensor->packed_row_bytes = 0;
    tensor->packed_rows = 0;

    size_t size = marmot_tensor_size_bytes(tensor);
    if (size == 0 && marmot_get_last_error() != MARMOT_SUCCESS) {
        free(tensor);
        return nullptr;
    }

    tensor->data = malloc(size);
    if (tensor->data == nullptr && size > 0) {
        free(tensor);
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate tensor data");
        return nullptr;
    }
    tensor->capacity_bytes = size;

    if (size > 0) {
        memset(tensor->data, 0, size);
        marmot_tensor_debug_record_alloc(size);
    }
    return tensor;
}

marmot_tensor_t *marmot_tensor_create_quantized(
    const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_quant_kind_t quant_kind
) {
    marmot_clear_error();

    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(quant_kind);
    if (traits == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid or unsupported quant_kind");
        return nullptr;
    }

    if (shape == nullptr || ndim != 2) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Quantized tensors require 2D shape");
        return nullptr;
    }

    marmot_tensor_t *tensor = calloc(1, sizeof(marmot_tensor_t));
    if (tensor == nullptr) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate tensor");
        return nullptr;
    }

    tensor->shape.ndim = ndim;
    memcpy(tensor->shape.shape, shape, ndim * sizeof(size_t));
    tensor->shape.strides[ndim - 1] = 1;
    for (ssize_t i = (ssize_t)ndim - 2; i >= 0; i--) {
        tensor->shape.strides[i] = tensor->shape.strides[i + 1] * shape[i + 1];
    }

    tensor->dtype = traits->storage_dtype;
    tensor->backend = ctx != nullptr ? ctx->backend_type : MARMOT_BACKEND_CPU;
    tensor->ctx = (marmot_context_t *)ctx;
    tensor->owns_data = true;
    tensor->quant_params = nullptr;
    tensor->quant_kind = quant_kind;
    tensor->quant_layout = traits->layout;
    tensor->memory_location = MARMOT_MEMORY_HOST;
    tensor->needs_sync = false;
    tensor->packed_data = nullptr;
    tensor->packed_src_data = nullptr;
    tensor->packed_bytes = 0;
    tensor->packed_row_bytes = 0;
    tensor->packed_rows = 0;

    size_t size = marmot_tensor_size_bytes(tensor);
    if (size == 0 && marmot_get_last_error() != MARMOT_SUCCESS) {
        free(tensor);
        return nullptr;
    }

    tensor->data = malloc(size);
    if (tensor->data == nullptr && size > 0) {
        free(tensor);
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate tensor data");
        return nullptr;
    }
    tensor->capacity_bytes = size;

    if (size > 0) {
        memset(tensor->data, 0, size);
        marmot_tensor_debug_record_alloc(size);
    }
    return tensor;
}

marmot_tensor_t *
marmot_tensor_zeros(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_dtype_t dtype) {
    return marmot_tensor_create(ctx, shape, ndim, dtype);
}

marmot_tensor_t *
marmot_tensor_ones(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_dtype_t dtype) {
    marmot_tensor_t *tensor = marmot_tensor_create(ctx, shape, ndim, dtype);
    if (tensor == nullptr) {
        return nullptr;
    }

    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        marmot_tensor_fill(tensor, 1.0f);
        break;
    case MARMOT_DTYPE_FLOAT16:
        marmot_tensor_fill(tensor, MARMOT_F16(0x3C00));
        break;
    case MARMOT_DTYPE_BFLOAT16:
        marmot_tensor_fill(tensor, MARMOT_BF16(0x3F80));
        break;
    case MARMOT_DTYPE_INT32:
        marmot_tensor_fill(tensor, MARMOT_I32(1));
        break;
    case MARMOT_DTYPE_INT16:
        marmot_tensor_fill(tensor, MARMOT_I16(1));
        break;
    case MARMOT_DTYPE_INT8:
        marmot_tensor_fill(tensor, MARMOT_I8(1));
        break;
    case MARMOT_DTYPE_UINT8:
        marmot_tensor_fill(tensor, MARMOT_U8(1));
        break;
    case MARMOT_DTYPE_UINT16:
        marmot_tensor_fill(tensor, MARMOT_U16(1));
        break;
    case MARMOT_DTYPE_UINT32:
        marmot_tensor_fill(tensor, MARMOT_U32(1));
        break;
    case MARMOT_DTYPE_UINT64:
        marmot_tensor_fill(tensor, MARMOT_U64(1));
        break;
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3: {
        marmot_float8_e4m3_t one = marmot_f32_to_fp8_e4m3_ref(1.0f);
        marmot_tensor_fill(tensor, one);
        break;
    }
    case MARMOT_DTYPE_FLOAT8_E5M2: {
        marmot_float8_e5m2_t one = marmot_f32_to_fp8_e5m2_ref(1.0f);
        marmot_tensor_fill(tensor, one);
        break;
    }
#endif
    default:
        marmot_tensor_destroy(tensor);
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unsupported dtype for ones");
        return nullptr;
    }

    return tensor;
}

marmot_tensor_t *marmot_tensor_like(const marmot_tensor_t *tensor) {
    if (unlikely(tensor == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Input tensor is nullptr");
        return nullptr;
    }
    marmot_tensor_t *result = marmot_tensor_create(tensor->ctx, tensor->shape.shape, tensor->shape.ndim, tensor->dtype);
    if (result != nullptr) {
        result->quant_kind = tensor->quant_kind;
        result->quant_layout = tensor->quant_layout;
        if (tensor->quant_params != nullptr) {
            result->quant_params = malloc(sizeof(marmot_quant_params_t));
            if (result->quant_params == nullptr) {
                marmot_tensor_destroy(result);
                marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate quant params");
                return nullptr;
            }
            memcpy(result->quant_params, tensor->quant_params, sizeof(marmot_quant_params_t));
        }
    }
    return result;
}

marmot_tensor_t *marmot_tensor_zeros_like(const marmot_tensor_t *tensor) {
    if (unlikely(tensor == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Input tensor is nullptr");
        return nullptr;
    }
    return marmot_tensor_zeros(tensor->ctx, tensor->shape.shape, tensor->shape.ndim, tensor->dtype);
}

marmot_tensor_t *marmot_tensor_ones_like(const marmot_tensor_t *tensor) {
    if (unlikely(tensor == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Input tensor is nullptr");
        return nullptr;
    }
    return marmot_tensor_ones(tensor->ctx, tensor->shape.shape, tensor->shape.ndim, tensor->dtype);
}

void marmot_tensor_destroy(marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return;
    }

    if (tensor->packed_data != nullptr) {
        free(tensor->packed_data);
    }

    if (tensor->ctx != nullptr && tensor->ctx->ops != nullptr && tensor->ctx->ops->on_host_ptr_freed != nullptr &&
        tensor->data != nullptr) {
        tensor->ctx->ops->on_host_ptr_freed(tensor->ctx->device_ctx, tensor->data);
    }

    if (tensor->owns_data && tensor->data != nullptr) {
        if (tensor->capacity_bytes > 0) {
            marmot_tensor_debug_record_free(tensor->capacity_bytes);
        }
        free(tensor->data);
    }

    if (tensor->quant_params != nullptr) {
        free(tensor->quant_params);
    }

    free(tensor);
}
