#include "core/helpers/quant.h"
#include "tensor_internal.h"

// NOTE: Internal element-wise, normalization, and quantization operations live inside backend
// implementations (e.g., src/backends/cpu/dispatch/device_ops.c, metal_backend.mm).
// Public APIs route normalization ops (layernorm/rmsnorm/softmax) through universal dispatch instead of device vtables.

// ===================================================================
// Vec-dot helper: build descriptor from 1D quantized tensors
// ===================================================================

static inline bool dtype_matches_quant_storage(marmot_dtype_t dtype, marmot_quant_kind_t kind) {
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(kind);
    return marmot_quant_storage_dtype_compatible(traits, dtype);
}

static bool
quant_vector_block_count(const marmot_tensor_t *tensor, const marmot_quant_kind_traits_t *traits, size_t *out_blocks) {
    if (tensor == nullptr || traits == nullptr || out_blocks == nullptr) {
        return false;
    }

    const size_t block_bytes = traits->header_bytes + traits->payload_bytes;
    size_t storage_bytes = 0;

    if (tensor->shape.ndim == 1) {
        const size_t elements = tensor->shape.shape[0];
        const size_t dtype_bytes = marmot_dtype_size(tensor->dtype);
        if (dtype_bytes == 0 || elements > SIZE_MAX / dtype_bytes) {
            return false;
        }
        storage_bytes = elements * dtype_bytes;
    } else if (tensor->shape.ndim == 2) {
        if (tensor->shape.shape[0] != 1) {
            return false;
        }
        const size_t cols = tensor->shape.shape[1];
        const size_t blocks_per_row = (cols + traits->block_values - 1) / traits->block_values;
        if (blocks_per_row == 0) {
            return false;
        }
        if (blocks_per_row > SIZE_MAX / block_bytes) {
            return false;
        }
        storage_bytes = blocks_per_row * block_bytes;
    } else {
        return false;
    }

    if (storage_bytes == 0 || storage_bytes % block_bytes != 0) {
        return false;
    }

    *out_blocks = storage_bytes / block_bytes;
    return true;
}

static bool vec_dot_tensor_is_vector(const marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return false;
    }
    if (tensor->shape.ndim == 1) {
        return true;
    }
    if (tensor->shape.ndim == 2 && tensor->shape.shape[0] == 1) {
        return true;
    }
    return false;
}

marmot_error_t marmot_vec_dot_descriptor_from_tensors(
    const marmot_tensor_t *input, const marmot_tensor_t *weight, marmot_vec_dot_descriptor_t *out
) {
    if (input == nullptr || weight == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null argument to vec_dot descriptor builder");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (!vec_dot_tensor_is_vector(input) || !vec_dot_tensor_is_vector(weight)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "vec_dot expects 1D logical tensors or 1xK views");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (input->quant_layout != MARMOT_QUANT_LAYOUT_GGUF || weight->quant_layout != MARMOT_QUANT_LAYOUT_GGUF) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unsupported quant layout for vec_dot (expect GGUF)");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (!dtype_matches_quant_storage(input->dtype, input->quant_kind) ||
        !dtype_matches_quant_storage(weight->dtype, weight->quant_kind)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Quantized tensor dtype does not match storage kind");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_quant_kind_traits_t *input_traits = marmot_get_quant_kind_traits(input->quant_kind);
    const marmot_quant_kind_traits_t *weight_traits = marmot_get_quant_kind_traits(weight->quant_kind);
    if (input_traits == nullptr || weight_traits == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unknown quant kind for vec_dot descriptor");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t in_blocks = 0;
    size_t wt_blocks = 0;
    if (!quant_vector_block_count(input, input_traits, &in_blocks) ||
        !quant_vector_block_count(weight, weight_traits, &wt_blocks)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid tensor shape for vec_dot descriptor");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (in_blocks == 0 || wt_blocks == 0 || in_blocks != wt_blocks) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Mismatched or zero GGUF block counts for vec_dot");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    out->weights = weight->data;
    out->activations = input->data;
    out->num_blocks = wt_blocks;
    out->weight_kind = weight->quant_kind;
    out->activation_kind = input->quant_kind;
    out->layout = MARMOT_QUANT_LAYOUT_GGUF;
    return MARMOT_SUCCESS;
}
