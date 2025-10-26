#include "elementwise.h"

#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/op_metadata.gen.h"
#include "marmot/tensor.h"

bool marmot_elementwise_unary_supports_bias(marmot_device_unary_op_t op) {
    return marmot_op_metadata_unary_supports_bias(op);
}

marmot_error_t marmot_elementwise_bias_info(
    const marmot_tensor_t *x, const marmot_tensor_t *bias, size_t *feature_dim, bool *bias_is_scalar
) {
    if (x == nullptr || bias == nullptr || feature_dim == nullptr || bias_is_scalar == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    size_t total = marmot_tensor_num_elements(x);
    size_t bias_elems = marmot_tensor_num_elements(bias);
    if (bias_elems == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Activation bias tensor is empty");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    *bias_is_scalar = (bias_elems == 1);
    *feature_dim = (x->shape.ndim == 0) ? 1 : x->shape.shape[x->shape.ndim - 1];
    if (!*bias_is_scalar) {
        if (bias->shape.ndim != 1 || bias->shape.shape[0] != *feature_dim) {
            marmot_set_error(
                MARMOT_ERROR_DIMENSION_MISMATCH, "Activation bias vector must match the final tensor dimension"
            );
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }
    if (*feature_dim == 0 && total != 0) {
        marmot_set_error(
            MARMOT_ERROR_DIMENSION_MISMATCH, "Activation tensors with empty trailing dim require scalar bias"
        );
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    return MARMOT_SUCCESS;
}

bool marmot_unary_op_requires_params(marmot_device_unary_op_t op) {
    return marmot_op_metadata_unary_requires_params(op);
}

marmot_error_t marmot_unary_prepare_activation_params(
    marmot_device_unary_op_t op, const marmot_activation_params_t *input, marmot_activation_params_t *out_params
) {
    if (out_params == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_activation_params_t prepared = {
        .parameter_tensor = input != nullptr ? input->parameter_tensor : nullptr,
        .bias = input != nullptr ? input->bias : nullptr,
        .alpha = input != nullptr ? input->alpha : 0.0f,
        .beta = input != nullptr ? input->beta : 0.0f,
        .gamma = input != nullptr ? input->gamma : 0.0f,
    };

    marmot_unary_params_metadata_t meta = marmot_op_metadata_unary_params(op);
    if (meta.requires_input && input == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Activation parameters are required for this operation");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!meta.allow_param_tensor && prepared.parameter_tensor != nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Tensor activation parameters are not implemented");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    if (input == nullptr) {
        if (meta.has_default_alpha) {
            prepared.alpha = meta.alpha_default;
        }
        if (meta.has_default_beta) {
            prepared.beta = meta.beta_default;
        }
    } else {
        if (meta.has_default_alpha && meta.alpha_default_on_zero && prepared.alpha == 0.0f) {
            prepared.alpha = meta.alpha_default;
        }
        if (meta.has_default_beta && meta.beta_default_on_zero && prepared.beta == 0.0f) {
            prepared.beta = meta.beta_default;
        }
    }

    *out_params = prepared;
    return MARMOT_SUCCESS;
}
