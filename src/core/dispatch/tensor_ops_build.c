#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/graph/op_signature.h"

#include "core/dispatch/fusion_flags.h"
#include "core/tensor/tensor_utils.h"
#include "graph/kernel_dispatch_args.gen.h"

static marmot_op_signature_t tensor_op_signature(
    marmot_op_id_t op_id, marmot_dtype_t input_dtype, marmot_dtype_t weight_dtype, marmot_dtype_t output_dtype,
    uint32_t n_elems
) {
    marmot_op_signature_t sig = {
        .op_id = op_id,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = input_dtype,
        .weight_dtype = weight_dtype,
        .output_dtype = output_dtype,
        .accum_dtype = MARMOT_DTYPE_COUNT,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .quant_block = {0},
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
        .dims = {.elementwise = {.n_elems = n_elems}},
    };
    return sig;
}

marmot_error_t marmot_reshape_build(
    const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output, const size_t *new_shape,
    size_t new_ndim, marmot_op_signature_t *sig_out, marmot_kernel_args_reshape_t *packed_out
) {
    if (ctx == nullptr || input == nullptr || output == nullptr || new_shape == nullptr || sig_out == nullptr ||
        packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Reshape requires non-null context, tensors, and shape");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_signature_t sig = tensor_op_signature(
        MARMOT_OP_RESHAPE, input->dtype, input->dtype, output->dtype, (uint32_t)marmot_tensor_num_elements(output)
    );

    *packed_out = (marmot_kernel_args_reshape_t){
        .ctx = ctx,
        .input = input,
        .output = output,
        .new_shape = new_shape,
        .new_ndim = new_ndim,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_view_build(
    const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output, size_t byte_offset,
    marmot_op_signature_t *sig_out, marmot_kernel_args_view_t *packed_out
) {
    if (ctx == nullptr || input == nullptr || output == nullptr || sig_out == nullptr || packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "View requires non-null context and tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_signature_t sig = tensor_op_signature(
        MARMOT_OP_VIEW, input->dtype, input->dtype, output->dtype, (uint32_t)marmot_tensor_num_elements(output)
    );

    *packed_out = (marmot_kernel_args_view_t){
        .ctx = ctx,
        .input = input,
        .output = output,
        .byte_offset = byte_offset,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_contiguous_build(
    const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output, marmot_op_signature_t *sig_out,
    marmot_kernel_args_contiguous_t *packed_out
) {
    if (ctx == nullptr || input == nullptr || output == nullptr || sig_out == nullptr || packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Contiguous requires non-null context and tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_signature_t sig = tensor_op_signature(
        MARMOT_OP_CONTIGUOUS, input->dtype, input->dtype, output->dtype, (uint32_t)marmot_tensor_num_elements(output)
    );

    *packed_out = (marmot_kernel_args_contiguous_t){
        .ctx = ctx,
        .input = input,
        .output = output,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_transpose_build(
    const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output, const int *perm,
    marmot_op_signature_t *sig_out, marmot_kernel_args_transpose_t *packed_out
) {
    if (ctx == nullptr || input == nullptr || output == nullptr || sig_out == nullptr || packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Transpose requires non-null context and tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_signature_t sig = tensor_op_signature(
        MARMOT_OP_TRANSPOSE, input->dtype, input->dtype, output->dtype, (uint32_t)marmot_tensor_num_elements(output)
    );

    *packed_out = (marmot_kernel_args_transpose_t){.ctx = ctx, .input = input, .output = output, .perm = perm};
    *sig_out = sig;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_concat_build(
    const marmot_context_t *ctx, const marmot_tensor_t *const *inputs, size_t num_inputs, marmot_tensor_t *output,
    int axis, marmot_op_signature_t *sig_out, marmot_kernel_args_concat_t *packed_out
) {
    if (ctx == nullptr || inputs == nullptr || output == nullptr || sig_out == nullptr || packed_out == nullptr ||
        num_inputs == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Concat requires non-null context, tensors, and outputs");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_dtype_t input_dtype = inputs[0] != nullptr ? inputs[0]->dtype : MARMOT_DTYPE_COUNT;
    marmot_op_signature_t sig = tensor_op_signature(
        MARMOT_OP_CONCAT, input_dtype, input_dtype, output->dtype, (uint32_t)marmot_tensor_num_elements(output)
    );

    *packed_out = (marmot_kernel_args_concat_t){
        .ctx = ctx,
        .inputs = inputs,
        .num_inputs = num_inputs,
        .output = output,
        .axis = axis,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_slice_build(
    const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output, const size_t *starts,
    const size_t *sizes, marmot_op_signature_t *sig_out, marmot_kernel_args_slice_t *packed_out
) {
    if (ctx == nullptr || input == nullptr || output == nullptr || starts == nullptr || sizes == nullptr ||
        sig_out == nullptr || packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Slice requires non-null context, tensors, and slice metadata");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_signature_t sig = tensor_op_signature(
        MARMOT_OP_SLICE, input->dtype, input->dtype, output->dtype, (uint32_t)marmot_tensor_num_elements(output)
    );

    *packed_out = (marmot_kernel_args_slice_t){
        .ctx = ctx,
        .input = input,
        .output = output,
        .starts = starts,
        .sizes = sizes,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_gather_rows_build(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *indices, marmot_tensor_t *output,
    marmot_op_signature_t *sig_out, marmot_kernel_args_gather_rows_t *packed_out
) {
    if (ctx == nullptr || input == nullptr || indices == nullptr || output == nullptr || sig_out == nullptr ||
        packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Gather rows requires non-null context and tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_signature_t sig = tensor_op_signature(
        MARMOT_OP_GATHER_ROWS, input->dtype, input->dtype, output->dtype, (uint32_t)marmot_tensor_num_elements(output)
    );

    *packed_out = (marmot_kernel_args_gather_rows_t){
        .ctx = ctx,
        .input = input,
        .indices = indices,
        .output = output,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_scatter_u64_to_i32_build(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *indices, marmot_tensor_t *output,
    marmot_op_signature_t *sig_out, marmot_kernel_args_gather_rows_t *packed_out
) {
    if (ctx == nullptr || input == nullptr || indices == nullptr || output == nullptr || sig_out == nullptr ||
        packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Scatter u64->i32 requires non-null context and tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input->dtype != MARMOT_DTYPE_UINT64) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Scatter u64->i32 requires UINT64 input");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (output->dtype != MARMOT_DTYPE_INT32) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Scatter u64->i32 requires INT32 output");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (indices->dtype != MARMOT_DTYPE_UINT32) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Scatter u64->i32 requires UINT32 indices");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (input->shape.ndim != 1 || indices->shape.ndim != 1 || output->shape.ndim != 1) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Scatter u64->i32 expects 1D input/indices/output");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (input->shape.shape[0] != indices->shape.shape[0]) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Scatter u64->i32 input/indices length mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (output->shape.shape[0] == 0 || input->shape.shape[0] == 0) {
        marmot_op_signature_t sig = tensor_op_signature(
            MARMOT_OP_SCATTER_U64_TO_I32, input->dtype, input->dtype, output->dtype,
            (uint32_t)marmot_tensor_num_elements(output)
        );
        *packed_out = (marmot_kernel_args_gather_rows_t){
            .ctx = ctx,
            .input = input,
            .indices = indices,
            .output = output,
        };
        *sig_out = sig;
        return MARMOT_SUCCESS;
    }

    marmot_op_signature_t sig = tensor_op_signature(
        MARMOT_OP_SCATTER_U64_TO_I32, input->dtype, input->dtype, output->dtype,
        (uint32_t)marmot_tensor_num_elements(output)
    );

    *packed_out = (marmot_kernel_args_gather_rows_t){
        .ctx = ctx,
        .input = input,
        .indices = indices,
        .output = output,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}
