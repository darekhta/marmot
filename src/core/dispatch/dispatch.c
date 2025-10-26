#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/graph/op_signature.h"
#include "marmot/op_metadata.gen.h"

#include <stdio.h>

#include "core/dispatch/fusion_flags.h"
#include "dispatch_build.h"
#include "dispatch_execute.h"
#include "graph/kernel_dispatch_args.gen.h"

marmot_error_t marmot_reduction_build(
    const marmot_context_t *ctx, marmot_device_reduction_op_t op, const marmot_tensor_t *input,
    marmot_tensor_t *out_values, marmot_tensor_t *out_indices, const marmot_reduction_params_t *params,
    const char *op_name, marmot_op_signature_t *sig_out, marmot_kernel_args_reduction_t *packed_out
) {
    if (ctx == nullptr) {
        char msg[112];
        snprintf(msg, sizeof(msg), "%s reduction requires non-null context", op_name);
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, msg);
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input == nullptr || out_values == nullptr || sig_out == nullptr || packed_out == nullptr) {
        char msg[112];
        snprintf(msg, sizeof(msg), "%s reduction requires non-null tensors", op_name);
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, msg);
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_id_t op_id = marmot_op_metadata_reduction_op_id(op);
    if (op_id == MARMOT_OP_INVALID) {
        char msg[112];
        snprintf(msg, sizeof(msg), "Unknown reduction operation for %s", op_name);
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, msg);
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    marmot_op_signature_t sig = {
        .op_id = op_id,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = input->dtype,
        .weight_dtype = input->dtype,
        .output_dtype = out_values->dtype,
        .accum_dtype = marmot_dtype_reduction_accum_dtype(input->dtype),
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .quant_block = {0},
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    *packed_out = (marmot_kernel_args_reduction_t){
        .ctx = ctx,
        .input = input,
        .out_values = out_values,
        .out_indices = out_indices,
        .params = params,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_dispatch_reduction(
    const marmot_context_t *ctx, marmot_device_reduction_op_t op, const marmot_tensor_t *input,
    marmot_tensor_t *out_values, marmot_tensor_t *out_indices, const marmot_reduction_params_t *params,
    const char *op_name
) {
    marmot_op_signature_t sig = {0};
    marmot_kernel_args_reduction_t packed = {0};
    marmot_error_t build_status =
        marmot_reduction_build(ctx, op, input, out_values, out_indices, params, op_name, &sig, &packed);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return marmot_execute_signature(ctx, &sig, &packed, op_name);
}
