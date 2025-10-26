#ifndef MARMOT_CORE_TENSOR_INTERNAL_H
#define MARMOT_CORE_TENSOR_INTERNAL_H

#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/macros.h"
#include "marmot/tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

static inline size_t tensor_effective_bytes(const marmot_tensor_t *tensor) {
    size_t quant_bytes = marmot_tensor_quant_storage_bytes(tensor);
    if (quant_bytes != 0) {
        return quant_bytes;
    }
    return marmot_tensor_size_bytes(tensor);
}

static inline void mark_host_written(marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return;
    }
    tensor->memory_location = MARMOT_MEMORY_HOST;
    tensor->needs_sync = (tensor->backend != MARMOT_BACKEND_CPU);
}

static inline marmot_error_t marmot_tensor_ensure_capacity(marmot_tensor_t *tensor, size_t bytes) {
    if (tensor == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (bytes == 0 || tensor->capacity_bytes >= bytes) {
        return MARMOT_SUCCESS;
    }
    if (!tensor->owns_data) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Tensor view cannot grow storage");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    void *new_data = realloc(tensor->data, bytes);
    if (new_data == nullptr) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to grow tensor storage");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    tensor->data = new_data;
    tensor->capacity_bytes = bytes;
    return MARMOT_SUCCESS;
}

#endif
