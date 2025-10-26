#include "tensor_internal.h"

marmot_error_t marmot_tensor_to_device(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(ctx == nullptr || tensor == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "tensor_to_device requires non-null context and tensor");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t bytes = tensor_effective_bytes(tensor);
    if (bytes == 0) {
        tensor->memory_location = (ctx->backend_type == MARMOT_BACKEND_CPU) ? MARMOT_MEMORY_HOST : MARMOT_MEMORY_DEVICE;
        tensor->needs_sync = false;
        return MARMOT_SUCCESS;
    }

    if (ctx->backend_type == MARMOT_BACKEND_CPU || ctx->ops == nullptr || ctx->ops->memcpy_to_device == nullptr) {
        tensor->memory_location = (ctx->backend_type == MARMOT_BACKEND_CPU) ? MARMOT_MEMORY_HOST : MARMOT_MEMORY_DEVICE;
        tensor->needs_sync = false;
        return MARMOT_SUCCESS;
    }

    marmot_error_t err = ctx->ops->memcpy_to_device(ctx->device_ctx, tensor->data, tensor->data, bytes);
    if (err != MARMOT_SUCCESS) {
        marmot_set_error(err, "tensor_to_device memcpy failed");
        return err;
    }

    tensor->memory_location = MARMOT_MEMORY_DEVICE;
    tensor->needs_sync = false;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_tensor_to_host(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (unlikely(ctx == nullptr || tensor == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "tensor_to_host requires non-null context and tensor");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (tensor->data == nullptr) {
        tensor->memory_location = MARMOT_MEMORY_HOST;
        tensor->needs_sync = false;
        return MARMOT_SUCCESS;
    }

    if (tensor->memory_location == MARMOT_MEMORY_HOST && !tensor->needs_sync) {
        return MARMOT_SUCCESS;
    }

    if (ctx->backend_type != MARMOT_BACKEND_CPU && ctx->ops != nullptr && ctx->ops->memcpy_from_device != nullptr) {
        marmot_error_t err =
            ctx->ops->memcpy_from_device(ctx->device_ctx, tensor->data, tensor->data, tensor_effective_bytes(tensor));
        if (err != MARMOT_SUCCESS) {
            marmot_set_error(err, "tensor_to_host memcpy failed");
            return err;
        }
    } else {
        marmot_error_t err = marmot_device_synchronize(ctx);
        if (err != MARMOT_SUCCESS) {
            marmot_set_error(err, "tensor_to_host sync failed");
            return err;
        }
    }

    tensor->memory_location = MARMOT_MEMORY_HOST;
    tensor->needs_sync = false;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_tensor_to_device_async(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    return marmot_tensor_to_device(ctx, tensor);
}

marmot_error_t marmot_tensor_to_host_async(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    return marmot_tensor_to_host(ctx, tensor);
}

marmot_memory_location_t marmot_tensor_memory_location(const marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return MARMOT_MEMORY_UNKNOWN;
    }
    return tensor->memory_location;
}

bool marmot_tensor_is_ready(const marmot_context_t *ctx, const marmot_tensor_t *tensor) {
    (void)ctx;
    if (tensor == nullptr) {
        return false;
    }
    return !tensor->needs_sync;
}
