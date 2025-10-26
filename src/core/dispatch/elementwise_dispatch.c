#include "marmot/graph/op_signature.h"
#include "marmot/stride_utils.h"

#include "core/dispatch/dispatch_build.h"
#include "core/dispatch/dispatch_execute.h"
#include "core/dispatch/fusion_flags.h"
#include "core/dispatch/signature_utils.h"
#include "core/helpers/elementwise.h"
#include "core/tensor/tensor_utils.h"
#include "graph/kernel_dispatch_args.gen.h"

static bool tensor_is_row_strided_2d(const marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return false;
    }
    if (tensor->shape.ndim != 2) {
        return false;
    }
    return marmot_tensor_is_row_strided(tensor);
}

static marmot_stride_mode_t
elementwise_stride_mode_2d(const marmot_tensor_t *a, const marmot_tensor_t *b, const marmot_tensor_t *out) {
    if (marmot_tensor_is_contiguous(a) && marmot_tensor_is_contiguous(b) && marmot_tensor_is_contiguous(out)) {
        return MARMOT_STRIDE_MODE_CONTIGUOUS;
    }
    if (tensor_is_row_strided_2d(a) && tensor_is_row_strided_2d(b) && tensor_is_row_strided_2d(out)) {
        return MARMOT_STRIDE_MODE_ROW_STRIDED;
    }
    return MARMOT_STRIDE_MODE_STRIDED;
}

static marmot_error_t
marmot_validate_unary(const marmot_context_t *ctx, const marmot_tensor_t *x, const marmot_tensor_t *out) {
    if (ctx == nullptr || x == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null argument to unary op");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (x->dtype != out->dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unary result dtype must match input");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    size_t in_elems = marmot_tensor_num_elements(x);
    size_t dtype_bytes = marmot_dtype_size(x->dtype);
    if (dtype_bytes == 0) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unary operation received invalid dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    size_t required_bytes = in_elems * dtype_bytes;
    if (out->capacity_bytes < required_bytes) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Unary output is smaller than the input tensor");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (in_elems != 0 && (x->data == nullptr || out->data == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unary tensor data pointers cannot be null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t marmot_validate_elementwise(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, const marmot_tensor_t *out,
    bool allow_bool_out
) {
    if (ctx == nullptr || a == nullptr || b == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null argument to elementwise op");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!marmot_tensors_same_shape(a, b) || !marmot_tensors_same_shape(a, out)) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Elementwise shape mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (a->dtype != b->dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Elementwise operands must match dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (!allow_bool_out) {
        if (a->dtype != out->dtype) {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Elementwise result dtype must match operands");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
    } else if (out->dtype != a->dtype && out->dtype != MARMOT_DTYPE_UINT8) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Comparison output must be input dtype or UINT8");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    return MARMOT_SUCCESS;
}

static marmot_dtype_t
ternary_accum_dtype(marmot_device_ternary_op_t op, marmot_dtype_t input_dtype, marmot_dtype_t output_dtype) {
    switch (op) {
    case MARMOT_DEVICE_TERNARY_FMA:
        return marmot_elementwise_accum_dtype(input_dtype);
    case MARMOT_DEVICE_TERNARY_WHERE:
        return output_dtype;
    default:
        return output_dtype;
    }
}

marmot_error_t marmot_unary_build(
    const marmot_context_t *ctx, marmot_device_unary_op_t device_op, marmot_op_id_t op_id, const marmot_tensor_t *x,
    const marmot_activation_params_t *params, marmot_tensor_t *out, marmot_op_signature_t *sig_out,
    marmot_kernel_args_unary_t *packed_out
) {
    if (sig_out == nullptr || packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unary build requires non-null output buffers");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_error_t status = marmot_validate_unary(ctx, x, out);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    if (params != nullptr && params->bias != nullptr && !marmot_elementwise_unary_supports_bias(device_op)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Bias is not supported for this unary operation");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_signature_t sig = {
        .op_id = op_id,
        .profile_id = ctx->best_profile,
        .input_dtype = x->dtype,
        .weight_dtype = x->dtype,
        .output_dtype = out->dtype,
        .accum_dtype = marmot_elementwise_accum_dtype(x->dtype),
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .quant_block = {0},
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
        .dims = {.elementwise = {.n_elems = (uint32_t)marmot_tensor_num_elements(x)}},
    };
    *packed_out = (marmot_kernel_args_unary_t){
        .ctx = ctx,
        .input = x,
        .params = params,
        .output = out,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_binary_build(
    const marmot_context_t *ctx, marmot_op_id_t op_id, const marmot_tensor_t *a, const marmot_tensor_t *b,
    marmot_tensor_t *out, bool allow_bool_out, bool use_stride_mode_2d, marmot_op_signature_t *sig_out,
    marmot_kernel_args_binary_t *packed_out
) {
    if (sig_out == nullptr || packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Elementwise build requires non-null output buffers");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_error_t status = marmot_validate_elementwise(ctx, a, b, out, allow_bool_out);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    marmot_op_signature_t sig = {
        .op_id = op_id,
        .profile_id = ctx->best_profile,
        .input_dtype = a->dtype,
        .weight_dtype = a->dtype,
        .output_dtype = out->dtype,
        .accum_dtype = marmot_elementwise_accum_dtype(a->dtype),
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .quant_block = {0},
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .stride_mode = use_stride_mode_2d ? elementwise_stride_mode_2d(a, b, out) : MARMOT_STRIDE_MODE_ANY,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
        .dims = {.elementwise = {.n_elems = (uint32_t)marmot_tensor_num_elements(a)}},
    };
    *packed_out = (marmot_kernel_args_binary_t){
        .ctx = ctx,
        .input_a = a,
        .input_b = b,
        .output = out,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_ternary_build(
    const marmot_context_t *ctx, marmot_device_ternary_op_t op, marmot_op_id_t op_id, const marmot_tensor_t *a,
    const marmot_tensor_t *b, const marmot_tensor_t *c, marmot_tensor_t *out, marmot_op_signature_t *sig_out,
    marmot_kernel_args_ternary_t *packed_out
) {
    if (sig_out == nullptr || packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Ternary build requires non-null output buffers");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (ctx == nullptr || a == nullptr || b == nullptr || c == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null argument to ternary op");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_signature_t sig = {
        .op_id = op_id,
        .profile_id = ctx->best_profile,
        .input_dtype = a->dtype,
        .weight_dtype = b->dtype,
        .output_dtype = out->dtype,
        .accum_dtype = ternary_accum_dtype(op, a->dtype, out->dtype),
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .quant_block = {0},
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
        .dims = {.elementwise = {.n_elems = (uint32_t)marmot_tensor_num_elements(out)}},
    };

    *packed_out = (marmot_kernel_args_ternary_t){
        .ctx = ctx,
        .input_a = a,
        .input_b = b,
        .input_c = c,
        .output = out,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_dispatch_unary_uniform(
    const marmot_context_t *ctx, marmot_device_unary_op_t device_op, marmot_op_id_t op_id, const marmot_tensor_t *x,
    const marmot_activation_params_t *params, marmot_tensor_t *out, const char *op_name
) {
    marmot_activation_params_t params_copy = {0};
    const marmot_activation_params_t *params_ptr = params;
    if (params != nullptr) {
        params_copy = *params;
        params_ptr = &params_copy;
    }

    marmot_op_signature_t sig = {0};
    marmot_kernel_args_unary_t packed = {0};
    marmot_error_t status = marmot_unary_build(ctx, device_op, op_id, x, params_ptr, out, &sig, &packed);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    return marmot_execute_signature(ctx, &sig, &packed, op_name);
}

marmot_error_t marmot_dispatch_binary(
    const marmot_context_t *ctx, marmot_op_id_t op_id, const marmot_tensor_t *a, const marmot_tensor_t *b,
    marmot_tensor_t *out, bool allow_bool_out, bool use_stride_mode_2d, const char *label
) {
    marmot_op_signature_t sig = {0};
    marmot_kernel_args_binary_t packed = {0};
    marmot_error_t status =
        marmot_binary_build(ctx, op_id, a, b, out, allow_bool_out, use_stride_mode_2d, &sig, &packed);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    return marmot_execute_signature(ctx, &sig, &packed, label);
}

marmot_error_t marmot_dispatch_ternary(
    const marmot_context_t *ctx, marmot_device_ternary_op_t op, marmot_op_id_t op_id, const marmot_tensor_t *a,
    const marmot_tensor_t *b, const marmot_tensor_t *c, marmot_tensor_t *out, marmot_dtype_t lookup_dtype,
    const char *op_name
) {
    marmot_op_signature_t sig = {0};
    marmot_kernel_args_ternary_t packed = {0};
    marmot_error_t status = marmot_ternary_build(ctx, op, op_id, a, b, c, out, &sig, &packed);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    return marmot_execute_signature(ctx, &sig, &packed, op_name);
}
