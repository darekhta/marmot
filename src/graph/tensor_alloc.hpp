#pragma once

#include "marmot/error.h"
#include "marmot/graph/graph_types.h"
#include "marmot/tensor.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "core/tensor/tensor_debug.h"

namespace marmot::graph {

inline bool graph_desc_compute_strides(const marmot_graph_tensor_desc_t &desc, size_t *out_strides) {
    if (desc.ndim == 0 || desc.ndim > MARMOT_MAX_DIMS || out_strides == nullptr) {
        return false;
    }

    bool has_stride = false;
    bool has_zero = false;
    for (uint32_t i = 0; i < desc.ndim; ++i) {
        if (desc.strides[i] == 0) {
            has_zero = true;
        } else {
            has_stride = true;
        }
    }

    if (!has_stride) {
        out_strides[desc.ndim - 1] = 1;
        for (size_t i = desc.ndim - 1; i-- > 0;) {
            out_strides[i] = out_strides[i + 1] * desc.shape[i + 1];
        }
        return true;
    }

    if (has_zero) {
        return false;
    }

    for (uint32_t i = 0; i < desc.ndim; ++i) {
        out_strides[i] = desc.strides[i];
    }
    return true;
}

inline bool graph_desc_span_elements(const marmot_graph_tensor_desc_t &desc, const size_t *strides, size_t &out_elems) {
    if (desc.ndim == 0 || desc.ndim > MARMOT_MAX_DIMS || strides == nullptr) {
        return false;
    }

    size_t max_offset = 0;
    for (uint32_t i = 0; i < desc.ndim; ++i) {
        const size_t dim = desc.shape[i];
        const size_t stride = strides[i];
        if (dim == 0 || stride == 0) {
            return false;
        }
        const size_t span = dim - 1;
        if (span != 0 && stride > (SIZE_MAX - max_offset) / span) {
            return false;
        }
        max_offset += span * stride;
    }

    if (max_offset == SIZE_MAX) {
        return false;
    }
    out_elems = max_offset + 1;
    return true;
}

inline bool graph_desc_span_bytes(const marmot_graph_tensor_desc_t &desc, const size_t *strides, size_t &out_bytes) {
    size_t span_elems = 0;
    if (!graph_desc_span_elements(desc, strides, span_elems)) {
        return false;
    }

    if (desc.dtype == MARMOT_DTYPE_INT4 || desc.dtype == MARMOT_DTYPE_UINT4) {
        out_bytes = (span_elems + 1) / 2;
        return true;
    }

    const size_t elem_size = marmot_dtype_size(desc.dtype);
    if (elem_size == 0 || span_elems > SIZE_MAX / elem_size) {
        return false;
    }
    out_bytes = span_elems * elem_size;
    return true;
}

inline marmot_tensor_t *
allocate_tensor_for_desc(const marmot_graph_tensor_desc_t &desc, marmot_backend_type_t backend) {
    size_t strides[MARMOT_MAX_DIMS] = {};
    if (!graph_desc_compute_strides(desc, strides)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid tensor strides");
        return nullptr;
    }

    size_t bytes = 0;
    if (!graph_desc_span_bytes(desc, strides, bytes)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid tensor span");
        return nullptr;
    }

    marmot_tensor_t *tensor = (marmot_tensor_t *)calloc(1, sizeof(*tensor));
    if (tensor == nullptr) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate tensor");
        return nullptr;
    }

    tensor->shape.ndim = desc.ndim;
    for (uint32_t i = 0; i < desc.ndim; ++i) {
        tensor->shape.shape[i] = desc.shape[i];
        tensor->shape.strides[i] = strides[i];
    }

    tensor->dtype = desc.dtype;
    tensor->backend = backend;
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

    tensor->data = bytes > 0 ? malloc(bytes) : nullptr;
    if (bytes > 0 && tensor->data == nullptr) {
        free(tensor);
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate tensor data");
        return nullptr;
    }
    tensor->capacity_bytes = bytes;

    if (bytes > 0) {
        memset(tensor->data, 0, bytes);
        marmot_tensor_debug_record_alloc(bytes);
    }

    return tensor;
}

} // namespace marmot::graph
