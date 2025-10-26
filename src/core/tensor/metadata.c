#include "tensor_internal.h"

size_t marmot_tensor_num_elements(const marmot_tensor_t *tensor) {
    if (tensor == nullptr || tensor->shape.ndim == 0) {
        return 0;
    }

    size_t total = 1;
    for (size_t i = 0; i < tensor->shape.ndim; i++) {
        if (tensor->shape.shape[i] > 0 && total > SIZE_MAX / tensor->shape.shape[i]) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Tensor size overflow");
            return 0;
        }
        total *= tensor->shape.shape[i];
    }
    return total;
}

size_t marmot_tensor_ndim(const marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return 0;
    }
    return tensor->shape.ndim;
}

size_t marmot_tensor_shape_at(const marmot_tensor_t *tensor, size_t dim) {
    if (tensor == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensor passed to shape query");
        return 0;
    }
    if (dim >= tensor->shape.ndim) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Tensor dimension out of range");
        return 0;
    }
    return tensor->shape.shape[dim];
}

size_t marmot_tensor_stride_at(const marmot_tensor_t *tensor, size_t dim) {
    if (tensor == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensor passed to stride query");
        return 0;
    }
    if (dim >= tensor->shape.ndim) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Tensor stride dimension out of range");
        return 0;
    }
    return tensor->shape.strides[dim];
}

size_t marmot_tensor_numel(const marmot_tensor_t *tensor) {
    return marmot_tensor_num_elements(tensor);
}

bool marmot_tensor_is_logical_quant_weight(const marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return false;
    }
    const marmot_quant_kind_traits_t *qtraits = marmot_get_quant_kind_traits(tensor->quant_kind);
    if (qtraits == nullptr) {
        return false;
    }
    if (!qtraits->is_block_quantized) {
        return false;
    }
    if (tensor->quant_layout != qtraits->layout) {
        return false;
    }
    if (tensor->shape.ndim != 2) {
        return false;
    }
    return true;
}

size_t marmot_tensor_quant_storage_bytes(const marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return 0;
    }
    if (!marmot_tensor_is_logical_quant_weight(tensor)) {
        return 0;
    }

    const marmot_quant_kind_traits_t *qtraits = marmot_get_quant_kind_traits(tensor->quant_kind);
    const size_t block_vals = qtraits->block_values;
    const size_t block_bytes = qtraits->header_bytes + qtraits->payload_bytes;

    const size_t rows = tensor->shape.shape[0];
    const size_t cols = tensor->shape.shape[1];
    const size_t blocks_per_row = (cols + block_vals - 1) / block_vals;
    return rows * blocks_per_row * block_bytes;
}

size_t marmot_tensor_size_bytes(const marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return 0;
    }

    size_t num_elements = marmot_tensor_num_elements(tensor);
    if (num_elements == 0) {
        return 0;
    }

    size_t quant_bytes = marmot_tensor_quant_storage_bytes(tensor);
    if (quant_bytes != 0) {
        return quant_bytes;
    }

    size_t dtype_bytes = marmot_dtype_size(tensor->dtype);

    if (tensor->dtype == MARMOT_DTYPE_INT4 || tensor->dtype == MARMOT_DTYPE_UINT4) {
        return (num_elements + 1) / 2;
    }

    if (num_elements > SIZE_MAX / dtype_bytes) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Tensor size overflow");
        return 0;
    }

    return num_elements * dtype_bytes;
}
