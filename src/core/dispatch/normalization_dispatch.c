#include "marmot/graph/op_signature.h"

#include "core/dispatch/dispatch_build.h"
#include "core/dispatch/dispatch_execute.h"
#include "core/dispatch/fusion_flags.h"
#include "core/helpers/norm.h"
#include "graph/kernel_dispatch_args.gen.h"

static marmot_dtype_t norm_accum_dtype(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT64:
        return MARMOT_DTYPE_FLOAT64;
    default:
        return MARMOT_DTYPE_FLOAT32;
    }
}

marmot_error_t marmot_layernorm_build(
    const marmot_context_t *ctx, const marmot_layernorm_desc_t *desc, marmot_op_signature_t *sig_out,
    marmot_kernel_args_layernorm_t *packed_out
) {
    if (ctx == nullptr || desc == nullptr || sig_out == nullptr || packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Layernorm requires non-null inputs");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (desc->x == nullptr || desc->out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Layernorm requires input and output tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_LAYERNORM,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = desc->x->dtype,
        .weight_dtype = desc->weight != nullptr ? desc->weight->dtype : desc->x->dtype,
        .output_dtype = desc->out->dtype,
        .accum_dtype = norm_accum_dtype(desc->x->dtype),
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .quant_block = {0},
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = desc->residual != nullptr ? MARMOT_FUSION_RESIDUAL_ADD : MARMOT_FUSION_NONE,
    };
    *packed_out = (marmot_kernel_args_layernorm_t){
        .ctx = ctx,
        .input = desc->x,
        .weight = desc->weight,
        .bias = desc->bias,
        .residual = desc->residual,
        .output = desc->out,
        .eps = desc->eps,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_rmsnorm_build(
    const marmot_context_t *ctx, const marmot_rmsnorm_desc_t *desc, marmot_op_id_t op_id,
    marmot_op_signature_t *sig_out, marmot_kernel_args_rms_norm_t *packed_out
) {
    if (ctx == nullptr || desc == nullptr || sig_out == nullptr || packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RMSNorm requires non-null inputs");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (desc->x == nullptr || desc->out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RMSNorm requires input and output tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_signature_t sig = {
        .op_id = op_id,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = desc->x->dtype,
        .weight_dtype = desc->weight != nullptr ? desc->weight->dtype : desc->x->dtype,
        .output_dtype = desc->out->dtype,
        .accum_dtype = norm_accum_dtype(desc->x->dtype),
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .quant_block = {0},
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
    };
    *packed_out = (marmot_kernel_args_rms_norm_t){
        .ctx = ctx,
        .input = desc->x,
        .weight = desc->weight,
        .residual = desc->residual,
        .output = desc->out,
        .eps = desc->eps,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_softmax_build(
    const marmot_context_t *ctx, const marmot_softmax_desc_t *desc, marmot_op_signature_t *sig_out,
    marmot_kernel_args_softmax_t *packed_out
) {
    if (ctx == nullptr || desc == nullptr || sig_out == nullptr || packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Softmax requires non-null inputs");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (desc->x == nullptr || desc->out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Softmax requires input and output tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_softmax_shape_t shape;
    marmot_error_t prep_status = marmot_softmax_prepare(desc->x, desc->out, desc->axis, &shape);
    if (prep_status != MARMOT_SUCCESS) {
        return prep_status;
    }

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_SOFTMAX,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = desc->x->dtype,
        .weight_dtype = desc->x->dtype,
        .output_dtype = desc->out->dtype,
        .accum_dtype = norm_accum_dtype(desc->x->dtype),
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .quant_block = {0},
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
        .dims = {
            .softmax = {
                .axis_size = (uint32_t)shape.axis_size,
                .inner_stride = (uint32_t)shape.inner_stride,
                .outer_size = (uint32_t)shape.outer_size,
                .row_count = (uint32_t)shape.row_count,
            },
        },
    };
    *packed_out = (marmot_kernel_args_softmax_t){
        .ctx = ctx,
        .input = desc->x,
        .output = desc->out,
        .axis = desc->axis,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_layernorm_dispatch(const marmot_context_t *ctx, const marmot_layernorm_desc_t *desc) {
    marmot_op_signature_t sig = {0};
    marmot_kernel_args_layernorm_t packed = {0};
    marmot_error_t build_status = marmot_layernorm_build(ctx, desc, &sig, &packed);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return marmot_execute_signature(ctx, &sig, &packed, "Layernorm");
}

marmot_error_t marmot_rmsnorm_dispatch(const marmot_context_t *ctx, const marmot_rmsnorm_desc_t *desc) {
    marmot_op_signature_t sig = {0};
    marmot_kernel_args_rms_norm_t packed = {0};
    marmot_error_t build_status = marmot_rmsnorm_build(ctx, desc, MARMOT_OP_RMS_NORM, &sig, &packed);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return marmot_execute_signature(ctx, &sig, &packed, "RMSNorm");
}

marmot_error_t marmot_rmsnorm_gemma_dispatch(const marmot_context_t *ctx, const marmot_rmsnorm_desc_t *desc) {
    marmot_op_signature_t sig = {0};
    marmot_kernel_args_rms_norm_t packed = {0};
    marmot_error_t build_status = marmot_rmsnorm_build(ctx, desc, MARMOT_OP_RMS_NORM_GEMMA, &sig, &packed);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return marmot_execute_signature(ctx, &sig, &packed, "RMSNormGemma");
}

marmot_error_t marmot_softmax_dispatch(const marmot_context_t *ctx, const marmot_softmax_desc_t *desc) {
    marmot_op_signature_t sig = {0};
    marmot_kernel_args_softmax_t packed = {0};
    marmot_error_t build_status = marmot_softmax_build(ctx, desc, &sig, &packed);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return marmot_execute_signature(ctx, &sig, &packed, "Softmax");
}
