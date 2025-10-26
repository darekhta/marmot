#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/graph/op_signature.h"

#include "core/dispatch/fusion_flags.h"
#include "core/helpers/quant.h"
#include "graph/kernel_dispatch_args.gen.h"

marmot_error_t marmot_quantize_build(
    const marmot_context_t *ctx, marmot_quant_kind_t kind, const marmot_tensor_t *input,
    const marmot_quant_params_t *params, marmot_tensor_t *output, marmot_op_signature_t *sig_out,
    marmot_kernel_args_quantize_t *packed_out, marmot_quant_layout_t *layout_out
) {
    if (ctx == nullptr || input == nullptr || output == nullptr || sig_out == nullptr || packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Quantize requires non-null context and tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (kind == MARMOT_QUANT_KIND_GENERIC && params == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Generic quantization requires quantization parameters");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_QUANTIZE,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = input->dtype,
        .weight_dtype = input->dtype,
        .output_dtype = output->dtype,
        .accum_dtype = MARMOT_DTYPE_COUNT,
        .qscheme_id = marmot_quant_kind_to_qscheme(kind),
        .quant_block = {0},
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
        .dims = {.elementwise = {.n_elems = (uint32_t)marmot_tensor_num_elements(input)}},
    };

    marmot_quant_layout_t layout = marmot_quant_kind_to_layout(kind);
    *packed_out = (marmot_kernel_args_quantize_t){
        .ctx = ctx,
        .input = input,
        .quant_params = params,
        .output = output,
        .kind = kind,
        .layout = layout,
    };
    *sig_out = sig;
    if (layout_out != nullptr) {
        *layout_out = layout;
    }
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_dequantize_build(
    const marmot_context_t *ctx, marmot_quant_kind_t kind, const marmot_tensor_t *input, marmot_tensor_t *output,
    marmot_op_signature_t *sig_out, marmot_kernel_args_dequantize_t *packed_out, marmot_quant_layout_t *layout_out
) {
    if (ctx == nullptr || input == nullptr || output == nullptr || sig_out == nullptr || packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Dequantize requires non-null context and tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_DEQUANTIZE,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = input->dtype,
        .weight_dtype = input->dtype,
        .output_dtype = output->dtype,
        .accum_dtype = MARMOT_DTYPE_COUNT,
        .qscheme_id = marmot_quant_kind_to_qscheme(kind),
        .quant_block = {0},
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
        .dims = {.elementwise = {.n_elems = (uint32_t)marmot_tensor_num_elements(output)}},
    };

    marmot_quant_layout_t layout = marmot_quant_kind_to_layout(kind);
    *packed_out = (marmot_kernel_args_dequantize_t){
        .ctx = ctx,
        .input = input,
        .output = output,
        .kind = kind,
        .layout = layout,
    };
    *sig_out = sig;
    if (layout_out != nullptr) {
        *layout_out = layout;
    }
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_compute_quant_params_build(
    const marmot_context_t *ctx, const marmot_tensor_t *tensor, marmot_dtype_t target_dtype, size_t block_size,
    marmot_quant_params_t *out_params, marmot_op_signature_t *sig_out, marmot_kernel_args_compute_qparams_t *packed_out
) {
    if (ctx == nullptr || tensor == nullptr || out_params == nullptr || sig_out == nullptr || packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Compute quant params requires non-null inputs");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_COMPUTE_QUANT_PARAMS,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = tensor->dtype,
        .weight_dtype = tensor->dtype,
        .output_dtype = target_dtype,
        .accum_dtype = MARMOT_DTYPE_COUNT,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .quant_block = {0},
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
        .dims = {.elementwise = {.n_elems = (uint32_t)marmot_tensor_num_elements(tensor)}},
    };

    *packed_out = (marmot_kernel_args_compute_qparams_t){
        .ctx = ctx,
        .tensor = tensor,
        .target_dtype = target_dtype,
        .block_size = block_size,
        .out_params = out_params,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_vec_dot_build(
    const marmot_context_t *ctx, const marmot_vec_dot_descriptor_t *desc, float *result, marmot_op_signature_t *sig_out,
    marmot_kernel_args_vec_dot_t *packed_out
) {
    if (desc == nullptr || result == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null descriptor or result buffer for vec dot");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null context for vec dot");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (sig_out == nullptr || packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Vec dot requires non-null output buffers");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_VEC_DOT,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = MARMOT_DTYPE_FLOAT32,
        .weight_dtype = MARMOT_DTYPE_FLOAT32,
        .output_dtype = MARMOT_DTYPE_FLOAT32,
        .accum_dtype = MARMOT_DTYPE_FLOAT32,
        .qscheme_id = marmot_quant_kind_to_qscheme(desc->weight_kind),
        .quant_block = {0},
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    *packed_out = (marmot_kernel_args_vec_dot_t){.ctx = ctx, .desc = desc, .result = result};

    *sig_out = sig;
    return MARMOT_SUCCESS;
}
