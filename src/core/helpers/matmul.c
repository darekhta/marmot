#include "matmul.h"

#include "marmot/error.h"

#include "core/helpers/quant.h"

static bool matmul_output_dtype_supported(marmot_dtype_t input_dtype, marmot_dtype_t out_dtype) {
    if (input_dtype == out_dtype) {
        return true;
    }
#if MARMOT_ENABLE_FP8
    if ((input_dtype == MARMOT_DTYPE_FLOAT8_E4M3 || input_dtype == MARMOT_DTYPE_FLOAT8_E5M2) &&
        out_dtype == MARMOT_DTYPE_FLOAT32) {
        return true;
    }
#endif
    return false;
}

static marmot_error_t matmul_validate_common(
    const marmot_tensor_t *input, const marmot_tensor_t *weight, const marmot_tensor_t *out, marmot_matmul_dims_t *dims,
    bool weight_transposed
) {
    if (input == nullptr || weight == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input->dtype != weight->dtype || !matmul_output_dtype_supported(input->dtype, out->dtype)) {
        marmot_set_error(
            MARMOT_ERROR_UNSUPPORTED_DTYPE, "Matmul tensors must share dtype (FP8 requires FLOAT32 output)"
        );
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (input->shape.ndim != 2 || weight->shape.ndim != 2 || out->shape.ndim != 2) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Matmul expects 2D tensors");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    const size_t out_rows = input->shape.shape[0];
    const size_t k = input->shape.shape[1];
    const size_t out_cols = weight_transposed ? weight->shape.shape[0] : weight->shape.shape[1];
    const size_t weight_k = weight_transposed ? weight->shape.shape[1] : weight->shape.shape[0];

    if (weight_k != k || out->shape.shape[0] != out_rows || out->shape.shape[1] != out_cols) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Matmul tensor shape mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    if (dims != nullptr) {
        dims->N = out_rows;
        dims->K = k;
        dims->M = out_cols;
    }
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_matmul_validate_dense(
    const marmot_tensor_t *input, const marmot_tensor_t *weight, const marmot_tensor_t *out, marmot_matmul_dims_t *dims
) {
    return matmul_validate_common(input, weight, out, dims, true);
}

marmot_error_t marmot_matmul_validate_nn(
    const marmot_tensor_t *input, const marmot_tensor_t *weight, const marmot_tensor_t *out, marmot_matmul_dims_t *dims
) {
    return matmul_validate_common(input, weight, out, dims, false);
}

marmot_error_t marmot_matmul_validate_quantized(
    const marmot_tensor_t *input, const marmot_tensor_t *weight, const marmot_tensor_t *out, marmot_matmul_dims_t *dims,
    marmot_matmul_activation_profile_t *profile, const marmot_quant_kind_traits_t **weight_traits
) {
    if (weight == nullptr || weight_traits == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(weight->quant_kind);
    if (traits == nullptr || !traits->is_block_quantized) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Weight tensor is not block-quantized");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (weight->quant_layout != traits->layout) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Quantized matmul only supports the GGUF layout");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (!marmot_quant_storage_dtype_compatible(traits, weight->dtype)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Quantized weight uses an unexpected storage dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    if (input == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input->shape.ndim != 2 || weight->shape.ndim != 2 || out->shape.ndim != 2) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Quantized matmul expects 2D tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_matmul_activation_profile_t local_profile = {
        .input_is_fp32 = (input->dtype == MARMOT_DTYPE_FLOAT32),
        .input_is_fp16 = (input->dtype == MARMOT_DTYPE_FLOAT16),
        .output_is_fp32 = (out->dtype == MARMOT_DTYPE_FLOAT32),
        .output_is_fp16 = (out->dtype == MARMOT_DTYPE_FLOAT16),
    };

    if (!((local_profile.input_is_fp32 && local_profile.output_is_fp32) ||
          (local_profile.input_is_fp16 && (local_profile.output_is_fp32 || local_profile.output_is_fp16)))) {
        marmot_set_error(
            MARMOT_ERROR_UNSUPPORTED_DTYPE,
            "Quantized matmul requires FLOAT32 input or FLOAT16 activations with FLOAT16/FLOAT32 output"
        );
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    const size_t N = input->shape.shape[0];
    const size_t K = input->shape.shape[1];
    const size_t M = weight->shape.shape[0];

    if (weight->shape.shape[1] != K) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Quantized matmul dimension mismatch on K");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (out->shape.shape[0] != N || out->shape.shape[1] != M) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Quantized matmul output shape mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    if (dims != nullptr) {
        dims->N = N;
        dims->K = K;
        dims->M = M;
    }
    if (profile != nullptr) {
        *profile = local_profile;
    }
    *weight_traits = traits;
    return MARMOT_SUCCESS;
}
