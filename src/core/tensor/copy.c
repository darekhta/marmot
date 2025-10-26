#include <stdlib.h>

#include <string.h>

#include "tensor_internal.h"

marmot_error_t marmot_tensor_copy(marmot_tensor_t *dst, const marmot_tensor_t *src) {
    if (unlikely(dst == nullptr || src == nullptr)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (unlikely(dst->dtype != src->dtype)) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    if (unlikely(dst->shape.ndim != src->shape.ndim)) {
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    for (size_t i = 0; i < dst->shape.ndim; ++i) {
        if (dst->shape.shape[i] != src->shape.shape[i]) {
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }

    size_t bytes = marmot_tensor_size_bytes(src);
    if (bytes > 0) {
        marmot_error_t reserve_err = marmot_tensor_ensure_capacity(dst, bytes);
        if (reserve_err != MARMOT_SUCCESS) {
            return reserve_err;
        }
    }
    if (bytes > 0 && dst->data != nullptr && src->data != nullptr) {
        memcpy(dst->data, src->data, bytes);
    }

    dst->quant_kind = src->quant_kind;
    dst->quant_layout = src->quant_layout;
    dst->memory_location = src->memory_location;
    dst->needs_sync = src->needs_sync;
    if (dst->packed_data != nullptr) {
        free(dst->packed_data);
        dst->packed_data = nullptr;
    }
    dst->packed_src_data = nullptr;
    dst->packed_bytes = 0;
    dst->packed_row_bytes = 0;
    dst->packed_rows = 0;

    if (dst->quant_params != nullptr) {
        free(dst->quant_params);
        dst->quant_params = nullptr;
    }
    if (src->quant_params != nullptr) {
        dst->quant_params = malloc(sizeof(marmot_quant_params_t));
        if (dst->quant_params == nullptr) {
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        memcpy(dst->quant_params, src->quant_params, sizeof(marmot_quant_params_t));
    }

    return MARMOT_SUCCESS;
}

marmot_tensor_t *marmot_tensor_clone(const marmot_tensor_t *src) {
    if (unlikely(src == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Input tensor is nullptr");
        return nullptr;
    }

    marmot_tensor_t *dst = marmot_tensor_like(src);
    if (dst == nullptr) {
        return nullptr;
    }

    marmot_error_t err = marmot_tensor_copy(dst, src);
    if (err != MARMOT_SUCCESS) {
        marmot_tensor_destroy(dst);
        marmot_set_error(err, "Failed to clone tensor");
        return nullptr;
    }

    return dst;
}
