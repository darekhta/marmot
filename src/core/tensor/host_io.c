#include <string.h>

#include "tensor_internal.h"

marmot_error_t
marmot_tensor_copy_to_host_buffer(const marmot_context_t *ctx, const marmot_tensor_t *tensor, void *dst, size_t bytes) {
    if (unlikely(ctx == nullptr || tensor == nullptr || dst == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "tensor_copy_to_host_buffer requires non-null arguments");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t effective = tensor_effective_bytes(tensor);
    if (bytes == 0) {
        bytes = effective;
    } else if (effective != 0 && bytes > effective) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Requested copy exceeds tensor storage size");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_error_t err = marmot_tensor_to_host(ctx, (marmot_tensor_t *)tensor);
    if (err != MARMOT_SUCCESS) {
        return err;
    }

    if (bytes > 0) {
        memcpy(dst, tensor->data, bytes);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_tensor_copy_from_host_buffer(
    const marmot_context_t *ctx, marmot_tensor_t *tensor, const void *src, size_t bytes
) {
    if (unlikely(ctx == nullptr || tensor == nullptr || src == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "tensor_copy_from_host_buffer requires non-null arguments");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t effective = tensor_effective_bytes(tensor);
    if (bytes == 0) {
        bytes = effective;
    } else if (effective != 0 && bytes > effective) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Requested copy exceeds tensor storage size");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (bytes > tensor->capacity_bytes) {
        marmot_error_t reserve_err = marmot_tensor_ensure_capacity(tensor, bytes);
        if (reserve_err != MARMOT_SUCCESS) {
            return reserve_err;
        }
    }

    if (bytes > 0) {
        memcpy(tensor->data, src, bytes);
    }

    if (ctx->backend_type != MARMOT_BACKEND_CPU && ctx->ops != nullptr && ctx->ops->memcpy_to_device != nullptr) {
        marmot_error_t err = ctx->ops->memcpy_to_device(ctx->device_ctx, tensor->data, src, bytes);
        if (err != MARMOT_SUCCESS) {
            marmot_set_error(err, "tensor_copy_from_host_buffer device upload failed");
            return err;
        }
        tensor->memory_location = MARMOT_MEMORY_DEVICE;
        tensor->needs_sync = false;
    } else {
        tensor->memory_location = MARMOT_MEMORY_HOST;
        tensor->needs_sync = false;
    }
    return MARMOT_SUCCESS;
}
