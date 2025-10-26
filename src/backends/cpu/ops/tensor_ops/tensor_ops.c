#include "cpu_backend_internal.h"

// ==================================================================
// CPU Backend Tensor Manipulation Operations
// ==================================================================
// Basic tensor manipulation: reshape, transpose, concat, slice
// Currently supports simple cases (1D, 2D), general N-D TODO
// ==================================================================

static bool cpu_tensor_is_contiguous(const marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return false;
    }
    if (tensor->shape.ndim == 0) {
        return true;
    }
    size_t expected_stride = 1;
    for (size_t i = tensor->shape.ndim; i > 0; --i) {
        const size_t dim_idx = i - 1;
        if (tensor->shape.strides[dim_idx] != expected_stride) {
            return false;
        }
        expected_stride *= tensor->shape.shape[dim_idx];
    }
    return true;
}

// Reshape: change tensor shape (total elements must match)
marmot_error_t cpu_reshape(
    const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const size_t *new_shape, size_t new_ndim
) {
    (void)device_ctx;

    if (unlikely(x == nullptr || out == nullptr || new_shape == nullptr || new_ndim == 0)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in reshape");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    // Verify total elements match
    size_t old_elements = marmot_tensor_num_elements(x);
    size_t new_elements = 1;
    for (size_t i = 0; i < new_ndim; i++) {
        new_elements *= new_shape[i];
    }

    if (unlikely(old_elements != new_elements)) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Reshape element count mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    if (out->owns_data && out->data != nullptr && out->data != x->data) {
        free(out->data);
    }

    out->shape.ndim = (uint32_t)new_ndim;
    for (size_t i = 0; i < new_ndim; ++i) {
        out->shape.shape[i] = new_shape[i];
    }

    bool has_stride = false;
    for (size_t i = 0; i < new_ndim; ++i) {
        if (out->shape.strides[i] != 0) {
            has_stride = true;
            break;
        }
    }
    if (!has_stride) {
        out->shape.strides[new_ndim - 1] = 1;
        for (size_t i = new_ndim - 1; i-- > 0;) {
            out->shape.strides[i] = out->shape.strides[i + 1] * out->shape.shape[i + 1];
        }
    }

    out->dtype = x->dtype;
    out->data = x->data;
    out->capacity_bytes = x->capacity_bytes;
    out->owns_data = false;
    out->quant_params = x->quant_params;
    out->quant_kind = x->quant_kind;
    out->quant_layout = x->quant_layout;
    out->backend = x->backend;
    out->memory_location = x->memory_location;
    out->needs_sync = x->needs_sync;
    out->packed_data = nullptr;
    out->packed_src_data = nullptr;
    out->packed_bytes = 0;
    out->packed_row_bytes = 0;
    out->packed_rows = 0;

    return MARMOT_SUCCESS;
}

marmot_error_t cpu_view(const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out, size_t byte_offset) {
    (void)device_ctx;

    if (unlikely(x == nullptr || out == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in view");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (unlikely(x->dtype != out->dtype)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "View requires matching dtypes");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (unlikely(x->backend != out->backend)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "View requires matching backends");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t elem_size = marmot_dtype_size(x->dtype);
    if (elem_size != 0 && (byte_offset % elem_size) != 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "View byte offset must align to dtype size");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_tensor_t out_probe = *out;
    out_probe.quant_kind = x->quant_kind;
    out_probe.quant_layout = x->quant_layout;
    size_t out_bytes = marmot_tensor_size_bytes(&out_probe);
    if (x->capacity_bytes < byte_offset) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "View byte offset exceeds input capacity");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (out_bytes > x->capacity_bytes - byte_offset) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "View exceeds input capacity");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (out_bytes != 0 && x->data == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "View requires non-null input data");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (out->owns_data && out->data != nullptr) {
        free(out->data);
    }
    if (out->packed_data != nullptr) {
        free(out->packed_data);
    }
    if (out->quant_params != nullptr && out->quant_params != x->quant_params) {
        free(out->quant_params);
    }

    out->data = (uint8_t *)x->data + byte_offset;
    out->capacity_bytes = x->capacity_bytes > byte_offset ? x->capacity_bytes - byte_offset : 0;
    out->owns_data = false;
    out->quant_params = x->quant_params;
    out->quant_kind = x->quant_kind;
    out->quant_layout = x->quant_layout;
    out->backend = x->backend;
    out->memory_location = x->memory_location;
    out->needs_sync = x->needs_sync;
    out->packed_data = nullptr;
    out->packed_src_data = nullptr;
    out->packed_bytes = 0;
    out->packed_row_bytes = 0;
    out->packed_rows = 0;

    return MARMOT_SUCCESS;
}

marmot_error_t cpu_contiguous(const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out) {
    (void)device_ctx;

    if (unlikely(x == nullptr || out == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in contiguous");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (unlikely(x->dtype != out->dtype)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Contiguous requires matching dtypes");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (unlikely(x->shape.ndim != out->shape.ndim)) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Contiguous rank mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    for (size_t i = 0; i < x->shape.ndim; ++i) {
        if (x->shape.shape[i] != out->shape.shape[i]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Contiguous shape mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }

    size_t total = marmot_tensor_num_elements(out);
    if (total == 0) {
        return MARMOT_SUCCESS;
    }

    size_t elem_size = marmot_dtype_size(out->dtype);
    if (elem_size == 0) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Contiguous dtype unsupported");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    if (unlikely(x->data == nullptr || out->data == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Contiguous requires non-null tensor data");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (cpu_tensor_is_contiguous(x) && cpu_tensor_is_contiguous(out)) {
        if (x->data != out->data) {
            memcpy(out->data, x->data, total * elem_size);
        }
        return MARMOT_SUCCESS;
    }

    size_t divisors[MARMOT_MAX_DIMS];
    size_t divisor = 1;
    for (size_t i = out->shape.ndim; i > 0; --i) {
        const size_t dim_idx = i - 1;
        divisors[dim_idx] = divisor;
        divisor *= out->shape.shape[dim_idx];
    }

    const uint8_t *src = (const uint8_t *)x->data;
    uint8_t *dst = (uint8_t *)out->data;
    for (size_t idx = 0; idx < total; ++idx) {
        size_t remaining = idx;
        size_t src_index = 0;
        size_t dst_index = 0;
        for (size_t axis = 0; axis < out->shape.ndim; ++axis) {
            size_t stride = (axis + 1 < out->shape.ndim) ? divisors[axis] : 1;
            size_t coord = (axis + 1 < out->shape.ndim) ? (remaining / stride) : remaining;
            remaining -= coord * stride;
            src_index += coord * x->shape.strides[axis];
            dst_index += coord * out->shape.strides[axis];
        }
        memcpy(dst + dst_index * elem_size, src + src_index * elem_size, elem_size);
    }

    return MARMOT_SUCCESS;
}

// Transpose: permute dimensions (2D only currently)
marmot_error_t cpu_transpose(const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const int *perm) {
    (void)device_ctx;

    if (unlikely(x == nullptr || out == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in transpose");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t ndim = x->shape.ndim;
    if (ndim != out->shape.ndim) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Transpose rank mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    if (ndim == 0) {
        return MARMOT_SUCCESS;
    }

    if (ndim > 8) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Transpose supports up to 8 dimensions");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    int perm_buffer[8];
    if (perm != nullptr) {
        for (size_t i = 0; i < ndim; ++i) {
            perm_buffer[i] = perm[i];
        }
    } else {
        for (size_t i = 0; i < ndim; ++i) {
            perm_buffer[i] = (int)(ndim - 1 - i);
        }
    }

    bool seen[8] = {false};
    for (size_t i = 0; i < ndim; ++i) {
        int axis = perm_buffer[i];
        if (axis < 0 || axis >= (int)ndim || seen[axis]) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid permutation for transpose");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        seen[axis] = true;
        if (out->shape.shape[i] != x->shape.shape[axis]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Transpose output shape mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }

    size_t elem_size = marmot_dtype_size(x->dtype);
    size_t total = marmot_tensor_num_elements(x);
    if (total == 0) {
        return MARMOT_SUCCESS;
    }

    uint32_t src_strides[8];
    src_strides[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; --i) {
        src_strides[i] = src_strides[i + 1] * (uint32_t)x->shape.shape[i + 1];
    }

    uint32_t dst_divisors[8];
    uint32_t divisor = 1;
    for (int i = (int)ndim - 1; i >= 0; --i) {
        dst_divisors[i] = divisor;
        divisor *= (uint32_t)out->shape.shape[i];
    }

    uint32_t src_strides_perm[8];
    bool identity = true;
    for (size_t i = 0; i < ndim; ++i) {
        src_strides_perm[i] = src_strides[perm_buffer[i]];
        if (perm_buffer[i] != (int)i) {
            identity = false;
        }
    }

    const uint8_t *src = (const uint8_t *)x->data;
    uint8_t *dst = (uint8_t *)out->data;

    if (identity) {
        memcpy(dst, src, total * elem_size);
        return MARMOT_SUCCESS;
    }

    for (size_t idx = 0; idx < total; ++idx) {
        size_t remaining = idx;
        size_t src_index = 0;
        for (size_t axis = 0; axis < ndim; ++axis) {
            size_t stride = (axis + 1 < ndim) ? dst_divisors[axis] : 1;
            size_t coord = (axis + 1 < ndim) ? (remaining / stride) : remaining;
            remaining -= coord * stride;
            src_index += coord * src_strides_perm[axis];
        }
        memcpy(dst + idx * elem_size, src + src_index * elem_size, elem_size);
    }

    return MARMOT_SUCCESS;
}

// Concat: concatenate tensors along axis (1D only currently)
marmot_error_t cpu_concat(
    const void *device_ctx, const marmot_tensor_t *const *tensors, size_t num_tensors, marmot_tensor_t *out, int axis
) {
    (void)device_ctx;

    if (unlikely(tensors == nullptr || out == nullptr || num_tensors == 0)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer or zero tensors in concat");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    // Handle negative axis
    if (axis < 0) {
        axis = (int)tensors[0]->shape.ndim + axis;
    }

    if (unlikely(axis < 0 || axis >= (int)tensors[0]->shape.ndim)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Axis out of range");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t ndim = tensors[0]->shape.ndim;
    if (out->shape.ndim != ndim) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Concat rank mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    size_t elem_size = marmot_dtype_size(tensors[0]->dtype);
    size_t inner = 1;
    for (size_t i = (size_t)axis + 1; i < ndim; ++i) {
        inner *= tensors[0]->shape.shape[i];
    }
    size_t outer = 1;
    for (size_t i = 0; i < (size_t)axis; ++i) {
        outer *= tensors[0]->shape.shape[i];
    }

    size_t out_axis = out->shape.shape[axis];
    size_t dst_axis_stride = out_axis * inner;
    size_t axis_offset = 0;

    for (size_t t = 0; t < num_tensors; ++t) {
        const marmot_tensor_t *tensor = tensors[t];
        for (size_t dim = 0; dim < ndim; ++dim) {
            if (dim == (size_t)axis) {
                continue;
            }
            if (tensor->shape.shape[dim] != tensors[0]->shape.shape[dim]) {
                marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Concat tensor shape mismatch");
                return MARMOT_ERROR_DIMENSION_MISMATCH;
            }
        }

        size_t axis_len = tensor->shape.shape[axis];
        size_t copy_elems = axis_len * inner;
        for (size_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
            size_t dst_offset = (outer_idx * dst_axis_stride + axis_offset) * elem_size;
            size_t src_offset = (outer_idx * copy_elems) * elem_size;
            memcpy(
                (uint8_t *)out->data + dst_offset, (const uint8_t *)tensor->data + src_offset, copy_elems * elem_size
            );
        }
        axis_offset += copy_elems;
    }

    return MARMOT_SUCCESS;
}

// Slice: extract sub-tensor (1D only currently)
marmot_error_t cpu_slice(
    const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const size_t *starts, const size_t *sizes
) {
    (void)device_ctx;

    if (unlikely(x == nullptr || out == nullptr || starts == nullptr || sizes == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in slice");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t ndim = x->shape.ndim;
    size_t elem_size = marmot_dtype_size(x->dtype);

    if (ndim == 0) {
        return MARMOT_SUCCESS;
    }

    size_t src_strides[8];
    size_t dst_strides[8];
    src_strides[ndim - 1] = 1;
    dst_strides[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; --i) {
        src_strides[i] = src_strides[i + 1] * x->shape.shape[i + 1];
        dst_strides[i] = dst_strides[i + 1] * out->shape.shape[i + 1];
    }

    size_t copy_block = sizes[ndim - 1] * elem_size;
    size_t loops = (ndim > 1) ? ndim - 1 : 0;
    size_t counters[8] = {0};

    while (true) {
        size_t src_index = starts[ndim - 1] * src_strides[ndim - 1];
        size_t dst_index = 0;
        for (size_t d = 0; d < loops; ++d) {
            src_index += (starts[d] + counters[d]) * src_strides[d];
            dst_index += counters[d] * dst_strides[d];
        }

        memcpy(
            (uint8_t *)out->data + dst_index * elem_size, (const uint8_t *)x->data + src_index * elem_size, copy_block
        );

        if (loops == 0) {
            break;
        }

        int dim = (int)loops - 1;
        while (dim >= 0) {
            counters[dim]++;
            if (counters[dim] < sizes[dim]) {
                break;
            }
            counters[dim] = 0;
            dim--;
        }
        if (dim < 0) {
            break;
        }
    }

    return MARMOT_SUCCESS;
}

marmot_error_t cpu_gather_rows(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *indices, marmot_tensor_t *out
) {
    (void)device_ctx;

    if (unlikely(input == nullptr || indices == nullptr || out == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in gather_rows");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input->shape.ndim != 2 || out->shape.ndim != 2) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "gather_rows expects 2D input/output tensors");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (indices->shape.ndim != 1) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "gather_rows indices must be 1D");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (input->dtype != out->dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "gather_rows requires matching input/output dtypes");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    const size_t rows_in = input->shape.shape[0];
    const size_t cols = input->shape.shape[1];
    const size_t rows_out = out->shape.shape[0];
    if (rows_out != indices->shape.shape[0] || cols != out->shape.shape[1]) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "gather_rows output shape mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (rows_out == 0 || cols == 0) {
        return MARMOT_SUCCESS;
    }
    const size_t elem_size = marmot_dtype_size(input->dtype);
    if (elem_size == 0) {
        return MARMOT_SUCCESS;
    }

    if (input->data == nullptr || out->data == nullptr || indices->data == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "gather_rows requires non-null tensor data");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t in_stride0 = input->shape.strides[0];
    const size_t in_stride1 = input->shape.strides[1];
    const size_t out_stride0 = out->shape.strides[0];
    const size_t out_stride1 = out->shape.strides[1];
    const size_t idx_stride0 = indices->shape.strides[0];

    const bool row_contiguous = in_stride1 == 1 && out_stride1 == 1;
    const size_t row_bytes = cols * elem_size;

    for (size_t r = 0; r < rows_out; ++r) {
        size_t index_offset = r * idx_stride0;
        int64_t row_idx = -1;
        if (indices->dtype == MARMOT_DTYPE_INT32) {
            row_idx = ((const marmot_int32_t *)indices->data)[index_offset].value;
        } else if (indices->dtype == MARMOT_DTYPE_UINT32) {
            row_idx = (int64_t)((const marmot_uint32_t *)indices->data)[index_offset].value;
        } else {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "gather_rows indices dtype unsupported");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }

        if (row_idx < 0 || (size_t)row_idx >= rows_in) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "gather_rows index out of range");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        const size_t src_row = (size_t)row_idx;
        if (row_contiguous) {
            const size_t src_offset = src_row * in_stride0 * elem_size;
            const size_t dst_offset = r * out_stride0 * elem_size;
            memcpy((uint8_t *)out->data + dst_offset, (const uint8_t *)input->data + src_offset, row_bytes);
        } else {
            for (size_t c = 0; c < cols; ++c) {
                const size_t src_offset = (src_row * in_stride0 + c * in_stride1) * elem_size;
                const size_t dst_offset = (r * out_stride0 + c * out_stride1) * elem_size;
                memcpy((uint8_t *)out->data + dst_offset, (const uint8_t *)input->data + src_offset, elem_size);
            }
        }
    }

    return MARMOT_SUCCESS;
}

marmot_error_t cpu_scatter_u64_to_i32(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *indices, marmot_tensor_t *out
) {
    (void)device_ctx;

    if (unlikely(input == nullptr || indices == nullptr || out == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in scatter_u64_to_i32");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input->dtype != MARMOT_DTYPE_UINT64 || out->dtype != MARMOT_DTYPE_INT32 ||
        indices->dtype != MARMOT_DTYPE_UINT32) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "scatter_u64_to_i32 dtype mismatch");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (input->shape.ndim != 1 || indices->shape.ndim != 1 || out->shape.ndim != 1) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "scatter_u64_to_i32 expects 1D tensors");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (input->shape.shape[0] != indices->shape.shape[0]) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "scatter_u64_to_i32 input/indices length mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    const size_t count = input->shape.shape[0];
    if (count == 0 || out->shape.shape[0] == 0) {
        return MARMOT_SUCCESS;
    }

    if (input->data == nullptr || indices->data == nullptr || out->data == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "scatter_u64_to_i32 requires non-null tensor data");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t in_stride = input->shape.strides[0];
    const size_t idx_stride = indices->shape.strides[0];
    const size_t out_stride = out->shape.strides[0];
    const size_t out_len = out->shape.shape[0];

    const marmot_uint64_t *src = (const marmot_uint64_t *)input->data;
    const marmot_uint32_t *idx = (const marmot_uint32_t *)indices->data;
    marmot_int32_t *dst = (marmot_int32_t *)out->data;

    for (size_t i = 0; i < count; ++i) {
        const uint32_t out_index = idx[i * idx_stride].value;
        if (out_index >= out_len) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "scatter_u64_to_i32 index out of range");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        dst[out_index * out_stride].value = (int32_t)src[i * in_stride].value;
    }

    return MARMOT_SUCCESS;
}
