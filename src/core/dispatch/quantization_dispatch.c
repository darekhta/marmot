#include "marmot/graph/op_signature.h"

#include "core/dispatch/dispatch_build.h"
#include "core/dispatch/dispatch_execute.h"
#include "core/helpers/quant.h"
#include "graph/kernel_dispatch_args.gen.h"

marmot_error_t marmot_quantize_dispatch(
    const marmot_context_t *ctx, marmot_quant_kind_t kind, const marmot_tensor_t *input,
    const marmot_quant_params_t *params, marmot_tensor_t *output
) {
    marmot_quant_layout_t layout = marmot_quant_kind_to_layout(kind);

    marmot_op_signature_t sig = {0};
    marmot_kernel_args_quantize_t packed = {0};
    marmot_error_t build_status = marmot_quantize_build(ctx, kind, input, params, output, &sig, &packed, &layout);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    marmot_error_t err = marmot_execute_signature(ctx, &sig, &packed, "Quantize");
    if (err == MARMOT_SUCCESS && output != nullptr) {
        output->quant_kind = kind;
        output->quant_layout = layout;
    }
    return err;
}

marmot_error_t marmot_dequantize_dispatch(
    const marmot_context_t *ctx, marmot_quant_kind_t kind, const marmot_tensor_t *input, marmot_tensor_t *output
) {
    marmot_quant_layout_t layout = marmot_quant_kind_to_layout(kind);

    marmot_op_signature_t sig = {0};
    marmot_kernel_args_dequantize_t packed = {0};
    marmot_error_t build_status = marmot_dequantize_build(ctx, kind, input, output, &sig, &packed, &layout);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return marmot_execute_signature(ctx, &sig, &packed, "Dequantize");
}

marmot_error_t marmot_compute_quant_params_dispatch(
    const marmot_context_t *ctx, const marmot_tensor_t *tensor, marmot_dtype_t target_dtype, size_t block_size,
    marmot_quant_params_t *out_params
) {
    marmot_op_signature_t sig = {0};
    marmot_kernel_args_compute_qparams_t packed = {0};
    marmot_error_t build_status =
        marmot_compute_quant_params_build(ctx, tensor, target_dtype, block_size, out_params, &sig, &packed);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return marmot_execute_signature(ctx, &sig, &packed, "Compute quant params");
}

marmot_error_t
marmot_vec_dot_dispatch(const marmot_context_t *ctx, const marmot_vec_dot_descriptor_t *desc, float *result) {
    if (ctx == nullptr || desc == nullptr || result == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Vec dot requires non-null context, descriptor, and result");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_signature_t sig = {0};
    marmot_kernel_args_vec_dot_t packed = {0};
    marmot_error_t build_status = marmot_vec_dot_build(ctx, desc, result, &sig, &packed);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return marmot_execute_signature(ctx, &sig, &packed, "vec_dot");
}
