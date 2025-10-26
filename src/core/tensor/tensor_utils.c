#include "tensor_utils.h"

#include "marmot/stride_utils.h"

#include <stdint.h>

#include "core/helpers/quant.h"

bool marmot_tensor_is_contiguous(const marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return false;
    }
    return marmot_is_contiguous_layout(tensor->shape.ndim, tensor->shape.shape, tensor->shape.strides);
}

bool marmot_tensor_is_row_strided(const marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return false;
    }
    return marmot_is_row_strided_layout(tensor->shape.ndim, tensor->shape.shape, tensor->shape.strides);
}

bool marmot_tensors_same_shape(const marmot_tensor_t *lhs, const marmot_tensor_t *rhs) {
    if (lhs == nullptr || rhs == nullptr) {
        return false;
    }
    if (lhs->shape.ndim != rhs->shape.ndim) {
        return false;
    }
    for (size_t i = 0; i < lhs->shape.ndim; ++i) {
        if (lhs->shape.shape[i] != rhs->shape.shape[i]) {
            return false;
        }
    }
    return true;
}

bool marmot_tensor_is_block_quantized_weight(const marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return false;
    }
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(tensor->quant_kind);
    if (traits == nullptr) {
        return false;
    }
    if (!traits->is_block_quantized) {
        return false;
    }
    if (!marmot_quant_storage_dtype_compatible(traits, tensor->dtype)) {
        return false;
    }
    if (tensor->quant_layout != traits->layout) {
        return false;
    }
    return true;
}

bool marmot_buffers_overlap(const void *dst, size_t dst_bytes, const void *src, size_t src_bytes) {
    if (dst == nullptr || src == nullptr || dst_bytes == 0 || src_bytes == 0) {
        return false;
    }
    uintptr_t dst_begin = (uintptr_t)dst;
    uintptr_t src_begin = (uintptr_t)src;
    if (dst_bytes > UINTPTR_MAX - dst_begin || src_bytes > UINTPTR_MAX - src_begin) {
        return true;
    }
    uintptr_t dst_end = dst_begin + dst_bytes;
    uintptr_t src_end = src_begin + src_bytes;
    return dst_begin < src_end && src_begin < dst_end;
}

bool marmot_is_power_of_two_u32(uint32_t value) {
    return value != 0u && (value & (value - 1u)) == 0u;
}
