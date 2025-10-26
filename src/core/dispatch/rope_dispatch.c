#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/graph/op_signature.h"

#include "core/dispatch/fusion_flags.h"
#include "core/tensor/tensor_utils.h"
#include "graph/kernel_dispatch_args.gen.h"

static marmot_stride_mode_t rope_stride_mode(const marmot_tensor_t *input, const marmot_tensor_t *output) {
    if (marmot_tensor_is_contiguous(input) && marmot_tensor_is_contiguous(output)) {
        return MARMOT_STRIDE_MODE_CONTIGUOUS;
    }
    if (marmot_tensor_is_row_strided(input) && marmot_tensor_is_row_strided(output)) {
        return MARMOT_STRIDE_MODE_ROW_STRIDED;
    }
    return MARMOT_STRIDE_MODE_STRIDED;
}

marmot_error_t marmot_rope_build(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_rope_params_t *params,
    marmot_tensor_t *output, marmot_op_signature_t *sig_out, marmot_kernel_args_rope_t *packed_out
) {
    if (ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE operation requires non-null context");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input == nullptr || output == nullptr || params == nullptr || sig_out == nullptr || packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE operation requires non-null tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (params->positions == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE operation requires positions tensor");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_ROPE,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = input->dtype,
        .weight_dtype = input->dtype,
        .output_dtype = output->dtype,
        .accum_dtype = MARMOT_DTYPE_COUNT,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .quant_block = {0},
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .stride_mode = rope_stride_mode(input, output),
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    *packed_out = (marmot_kernel_args_rope_t){
        .ctx = ctx,
        .input = input,
        .output = output,
        .rope_params = params,
        .n_past = 0,
        .n_rot = 0,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}
