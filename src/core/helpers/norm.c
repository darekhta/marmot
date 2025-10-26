#include "norm.h"

#include "marmot/error.h"

static bool marmot_norm_vector_dtype_allowed(marmot_dtype_t activation_dtype, marmot_dtype_t vector_dtype) {
    if (vector_dtype == activation_dtype) {
        return true;
    }
    const bool allow_f32_vectors =
        activation_dtype == MARMOT_DTYPE_FLOAT16 || activation_dtype == MARMOT_DTYPE_BFLOAT16;
    return allow_f32_vectors && vector_dtype == MARMOT_DTYPE_FLOAT32;
}

static bool marmot_norm_is_vector_of(const marmot_tensor_t *tensor, marmot_dtype_t activation_dtype, size_t length) {
    if (tensor == nullptr) {
        return true;
    }
    if (!marmot_norm_vector_dtype_allowed(activation_dtype, tensor->dtype)) {
        return false;
    }
    if (tensor->shape.ndim != 1) {
        return false;
    }
    return tensor->shape.shape[0] == length;
}

static bool marmot_norm_shapes_equal(const marmot_tensor_t *a, const marmot_tensor_t *b) {
    if (a == nullptr || b == nullptr) {
        return a == b;
    }
    if (a->shape.ndim != b->shape.ndim) {
        return false;
    }
    for (size_t i = 0; i < a->shape.ndim; ++i) {
        if (a->shape.shape[i] != b->shape.shape[i]) {
            return false;
        }
    }
    return true;
}

marmot_error_t marmot_norm_validate(
    const marmot_tensor_t *x, const marmot_tensor_t *residual, const marmot_tensor_t *weight,
    const marmot_tensor_t *bias, marmot_tensor_t *out, const marmot_norm_validation_opts_t *opts,
    marmot_norm_shape_t *shape
) {
    if (x == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensor pointer for normalization");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts == nullptr || shape == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Normalization options or shape is null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (x->dtype >= MARMOT_DTYPE_COUNT || out->dtype != x->dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Normalization dtype mismatch");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    if (!opts->allow_residual && residual != nullptr) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Residual tensor not supported for this norm variant");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (!opts->allow_weight && weight != nullptr) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Weight tensor not supported for this norm variant");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (opts->require_weight && weight == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Weight tensor required for this norm variant");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!opts->allow_bias && bias != nullptr) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Bias tensor not supported for this norm variant");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    if (x->shape.ndim == 0) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Normalization requires at least 1D tensor");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    if (!marmot_norm_shapes_equal(x, out)) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Normalization output shape mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (residual != nullptr) {
        if (residual->dtype != x->dtype || !marmot_norm_shapes_equal(x, residual)) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Residual tensor shape or dtype mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }

    if (weight != nullptr && bias != nullptr && weight->dtype != bias->dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Normalization weight and bias must have matching dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    size_t norm_size = x->shape.shape[x->shape.ndim - 1];
    if (norm_size == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Normalization last dimension must be non-zero");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    size_t total = marmot_tensor_num_elements(x);
    if (total % norm_size != 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Normalization element count not divisible by feature dim");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (!marmot_norm_is_vector_of(weight, x->dtype, norm_size)) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Weight tensor shape or dtype mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (!marmot_norm_is_vector_of(bias, x->dtype, norm_size)) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Bias tensor shape or dtype mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    shape->norm_size = norm_size;
    shape->outer_size = total / norm_size;
    return MARMOT_SUCCESS;
}

marmot_error_t
marmot_softmax_prepare(const marmot_tensor_t *x, const marmot_tensor_t *out, int axis, marmot_softmax_shape_t *shape) {
    if (x == nullptr || out == nullptr || shape == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Softmax requires non-null tensors and shape descriptor");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (x->dtype != out->dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Softmax input and output dtype mismatch");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (!marmot_norm_shapes_equal(x, out)) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Softmax input/output shape mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (x->shape.ndim == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Softmax requires at least 1D tensor");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    int normalized_axis = axis;
    if (normalized_axis < 0) {
        normalized_axis += (int)x->shape.ndim;
    }
    if (normalized_axis < 0 || normalized_axis >= (int)x->shape.ndim) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Softmax axis out of range");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t axis_size = x->shape.shape[normalized_axis];
    if (axis_size == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Softmax axis dimension must be non-zero");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t inner_stride = 1;
    for (size_t i = (size_t)normalized_axis + 1; i < x->shape.ndim; ++i) {
        inner_stride *= x->shape.shape[i];
    }

    size_t row_extent = axis_size * inner_stride;
    if (row_extent == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Softmax row extent is zero");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t total = marmot_tensor_num_elements(x);
    if (total % row_extent != 0) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Softmax tensor element count mismatches axis extent");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    shape->axis = normalized_axis;
    shape->axis_size = axis_size;
    shape->inner_stride = inner_stride;
    shape->outer_size = total / row_extent;
    shape->row_count = shape->outer_size * inner_stride;
    return MARMOT_SUCCESS;
}
