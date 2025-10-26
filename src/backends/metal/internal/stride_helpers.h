#ifndef MARMOT_METAL_STRIDE_HELPERS_H
#define MARMOT_METAL_STRIDE_HELPERS_H

#include "marmot/tensor.h"

#include <cstddef>

namespace marmot::metal {

inline bool is_contiguous(const marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return false;
    }
    const size_t ndim = tensor->shape.ndim;
    if (ndim == 0) {
        return true;
    }
    size_t expected_stride = 1;
    for (size_t i = ndim; i > 0; --i) {
        const size_t dim_idx = i - 1;
        if (tensor->shape.strides[dim_idx] != expected_stride) {
            return false;
        }
        expected_stride *= tensor->shape.shape[dim_idx];
    }
    return true;
}

inline bool is_row_strided(const marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return false;
    }
    const size_t ndim = tensor->shape.ndim;
    if (ndim < 2) {
        return is_contiguous(tensor);
    }
    if (tensor->shape.strides[ndim - 1] != 1) {
        return false;
    }
    return tensor->shape.strides[ndim - 2] >= tensor->shape.shape[ndim - 1];
}

struct StrideInfo {
    uint32_t shape[MARMOT_MAX_DIMS]{};
    size_t strides[MARMOT_MAX_DIMS]{};
    uint32_t ndim{0};
    size_t row_stride{0};
    bool is_contiguous{false};
    bool is_row_strided{false};
};

inline StrideInfo get_stride_info(const marmot_tensor_t *tensor) {
    StrideInfo info{};
    if (tensor == nullptr) {
        return info;
    }
    const size_t ndim = tensor->shape.ndim;
    info.ndim = static_cast<uint32_t>(ndim);
    for (size_t i = 0; i < ndim && i < MARMOT_MAX_DIMS; ++i) {
        info.shape[i] = static_cast<uint32_t>(tensor->shape.shape[i]);
        info.strides[i] = tensor->shape.strides[i];
    }
    info.is_contiguous = is_contiguous(tensor);
    info.is_row_strided = is_row_strided(tensor);
    if (info.is_row_strided && ndim >= 2) {
        info.row_stride = tensor->shape.strides[ndim - 2];
    }
    return info;
}

inline size_t tensor_span_elements(const marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return 0;
    }
    const size_t ndim = tensor->shape.ndim;
    if (ndim == 0) {
        return 0;
    }
    size_t max_offset = 0;
    for (size_t i = 0; i < ndim; ++i) {
        const size_t dim = tensor->shape.shape[i];
        const size_t stride = tensor->shape.strides[i];
        if (dim == 0) {
            return 0;
        }
        if (stride == 0) {
            return 0;
        }
        const size_t span = dim - 1;
        if (stride != 0 && span > (SIZE_MAX - max_offset) / stride) {
            return 0;
        }
        max_offset += span * stride;
    }
    return max_offset + 1;
}

inline size_t tensor_span_bytes(const marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return 0;
    }
    const size_t elements = tensor_span_elements(tensor);
    if (elements == 0) {
        return 0;
    }
    if (tensor->dtype == MARMOT_DTYPE_INT4 || tensor->dtype == MARMOT_DTYPE_UINT4) {
        return (elements + 1) / 2;
    }
    const size_t elem_size = marmot_dtype_size(tensor->dtype);
    if (elem_size == 0 || elements > SIZE_MAX / elem_size) {
        return 0;
    }
    return elements * elem_size;
}

} // namespace marmot::metal

#endif // MARMOT_METAL_STRIDE_HELPERS_H
