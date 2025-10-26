#ifndef MARMOT_STRIDE_UTILS_H
#define MARMOT_STRIDE_UTILS_H

#include "marmot/graph/op_signature.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline bool marmot_is_contiguous_layout(size_t ndim, const size_t *shape, const size_t *strides) {
    if (ndim == 0) {
        return true;
    }
    if (shape == nullptr || strides == nullptr) {
        return false;
    }
    size_t expected_stride = 1;
    for (size_t i = ndim; i > 0; --i) {
        const size_t dim_idx = i - 1;
        if (strides[dim_idx] != expected_stride) {
            return false;
        }
        expected_stride *= shape[dim_idx];
    }
    return true;
}

static inline bool marmot_is_row_strided_layout(size_t ndim, const size_t *shape, const size_t *strides) {
    if (ndim == 0) {
        return true;
    }
    if (shape == nullptr || strides == nullptr) {
        return false;
    }
    if (strides[ndim - 1] != 1) {
        return false;
    }
    size_t min_stride = 1;
    for (size_t i = ndim; i > 0; --i) {
        const size_t dim_idx = i - 1;
        if (strides[dim_idx] < min_stride) {
            return false;
        }
        min_stride *= shape[dim_idx];
    }
    return true;
}

static inline marmot_stride_mode_t
marmot_stride_mode_from_layout(size_t ndim, const size_t *shape, const size_t *strides) {
    if (marmot_is_contiguous_layout(ndim, shape, strides)) {
        return MARMOT_STRIDE_MODE_CONTIGUOUS;
    }
    if (marmot_is_row_strided_layout(ndim, shape, strides)) {
        return MARMOT_STRIDE_MODE_ROW_STRIDED;
    }
    return MARMOT_STRIDE_MODE_STRIDED;
}

#ifdef __cplusplus
}
#endif

#endif // MARMOT_STRIDE_UTILS_H
