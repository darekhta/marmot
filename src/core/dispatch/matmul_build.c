#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/graph/op_signature.h"
#include "marmot/ops/matmul.h"

#include <stdlib.h>

#include "core/dispatch/fusion_flags.h"
#include "core/dispatch/signature_utils.h"
#include "core/helpers/quant.h"
#include "core/tensor/tensor_utils.h"
#include "graph/kernel_dispatch_args.gen.h"

static bool matmul_force_scalar(const marmot_context_t *ctx) {
    if (ctx == nullptr || ctx->backend_type != MARMOT_BACKEND_CPU) {
        return false;
    }
    static thread_local int cached = -1;
    if (cached < 0) {
        const char *env = getenv("MARMOT_CPU_FORCE_SCALAR_MATMUL");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return cached == 1;
}

static uint32_t matmul_epilogue_to_flags(const marmot_matmul_epilogue_t *epilogue) {
    uint32_t flags = MARMOT_EPILOGUE_NONE;
    if (epilogue == nullptr) {
        return flags;
    }
    if (epilogue->bias != nullptr) {
        flags |= MARMOT_EPILOGUE_BIAS;
    }
    return flags;
}

static marmot_device_unary_op_t matmul_activation_from_op_id(marmot_op_id_t op_id) {
    switch (op_id) {
    case MARMOT_OP_MATMUL_BIAS_RELU:
        return MARMOT_DEVICE_UNARY_RELU;
    case MARMOT_OP_MATMUL_BIAS_GELU:
        return MARMOT_DEVICE_UNARY_GELU;
    case MARMOT_OP_MATMUL_BIAS_SILU:
        return MARMOT_DEVICE_UNARY_SILU;
    default:
        return MARMOT_DEVICE_UNARY_COUNT;
    }
}

static uint32_t qkv_epilogue_to_flags(const marmot_matmul_qkv_desc_t *desc) {
    uint32_t flags = MARMOT_EPILOGUE_NONE;
    if (desc == nullptr) {
        return flags;
    }
    if (desc->separate.bq != nullptr || desc->separate.bk != nullptr || desc->separate.bv != nullptr ||
        desc->fused.bias != nullptr) {
        flags |= MARMOT_EPILOGUE_BIAS;
    }
    const marmot_matmul_epilogue_t *ep = desc->epilogue;
    if (ep != nullptr) {
        if (ep->bias != nullptr) {
            flags |= MARMOT_EPILOGUE_BIAS;
        }
    }
    if (desc->rope_params != nullptr) {
        flags |= MARMOT_EPILOGUE_ROPE;
    }
    return flags;
}

static marmot_weight_layout_t marmot_resolve_weight_layout(
    const marmot_context_t *ctx, marmot_qscheme_id_t qscheme_id, marmot_weight_layout_t weight_layout
) {
    if (weight_layout != MARMOT_WEIGHT_LAYOUT_INVALID) {
        return weight_layout;
    }
    if (qscheme_id == MARMOT_QSCHEME_NONE) {
        return MARMOT_WEIGHT_LAYOUT_INVALID;
    }
    if (ctx != nullptr && ctx->policy.matmul_prefer_packed_weights) {
        return MARMOT_WEIGHT_LAYOUT_PACKED_3MK;
    }
    return MARMOT_WEIGHT_LAYOUT_SEPARATE;
}

marmot_error_t marmot_matmul_build(
    const marmot_context_t *ctx, marmot_matmul_layout_t matmul_layout, marmot_op_id_t op_id,
    const marmot_tensor_t *input, const marmot_tensor_t *weight, const marmot_matmul_epilogue_t *epilogue,
    marmot_tensor_t *output, marmot_qscheme_id_t qscheme_id, marmot_weight_layout_t weight_layout,
    marmot_op_signature_t *sig_out, marmot_kernel_args_matmul_t *packed_out
) {
    if (ctx == nullptr || input == nullptr || weight == nullptr || output == nullptr || sig_out == nullptr ||
        packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Matmul build requires non-null arguments");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (matmul_layout != MARMOT_MATMUL_LAYOUT_NN && matmul_layout != MARMOT_MATMUL_LAYOUT_NT) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Matmul build requires NN or NT layout");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t N = input->shape.ndim >= 1 ? input->shape.shape[0] : 1;
    const size_t K = input->shape.ndim >= 2 ? input->shape.shape[1] : 1;
    size_t M = 1;
    if (matmul_layout == MARMOT_MATMUL_LAYOUT_NN) {
        M = weight->shape.ndim >= 2 ? weight->shape.shape[1] : 1;
    } else {
        M = weight->shape.ndim >= 1 ? weight->shape.shape[0] : 1;
    }

    marmot_device_unary_op_t activation = matmul_activation_from_op_id(op_id);

    marmot_weight_layout_t resolved_weight_layout = marmot_resolve_weight_layout(ctx, qscheme_id, weight_layout);
    marmot_profile_id_t profile = ctx->best_profile;
    if (matmul_force_scalar(ctx)) {
        profile = MARMOT_PROFILE_SCALAR;
    }

    marmot_op_signature_t sig = {
        .op_id = op_id,
        .profile_id = profile,
        .matmul_layout = matmul_layout,
        .input_dtype = input->dtype,
        .weight_dtype = weight->dtype,
        .output_dtype = output->dtype,
        .accum_dtype = marmot_matmul_accum_dtype(input->dtype),
        .qscheme_id = qscheme_id,
        .quant_block = {0},
        .weight_layout = resolved_weight_layout,
        .epilogue_flags = matmul_epilogue_to_flags(epilogue),
        .activation = activation,
        .variant_flags = MARMOT_FUSION_NONE,
        .dims = {.matmul = {.N = (uint32_t)N, .K = (uint32_t)K, .M = (uint32_t)M}},
    };

    *packed_out = (marmot_kernel_args_matmul_t){
        .ctx = ctx,
        .input = input,
        .weight = weight,
        .epilogue = epilogue,
        .output = output,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_matmul_qkv_build(
    const marmot_context_t *ctx, const marmot_matmul_qkv_desc_t *desc, marmot_op_id_t op_id,
    marmot_matmul_layout_t matmul_layout, marmot_weight_layout_t weight_layout, marmot_op_signature_t *sig_out,
    marmot_kernel_args_qkv_t *packed_out
) {
    if (ctx == nullptr || desc == nullptr || sig_out == nullptr || packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Matmul QKV build requires non-null arguments");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (desc->input == nullptr || desc->separate.wq == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Matmul QKV build requires input and weights");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_tensor_t *input = desc->input;
    const marmot_tensor_t *wq = desc->separate.wq;

    const size_t N = input->shape.ndim >= 1 ? input->shape.shape[0] : 1;
    const size_t K = input->shape.ndim >= 2 ? input->shape.shape[1] : 1;
    const size_t M = wq->shape.ndim >= 1 ? wq->shape.shape[0] : 1;

    const bool weight_quantized = marmot_tensor_is_block_quantized_weight(wq);
    marmot_qscheme_id_t qscheme_id =
        weight_quantized ? marmot_quant_kind_to_qscheme(wq->quant_kind) : MARMOT_QSCHEME_NONE;
    marmot_dtype_t weight_dtype = weight_quantized ? input->dtype : wq->dtype;

    uint32_t epilogue_flags = qkv_epilogue_to_flags(desc);
    marmot_device_unary_op_t activation = MARMOT_DEVICE_UNARY_COUNT;

    marmot_weight_layout_t resolved_weight_layout = marmot_resolve_weight_layout(ctx, qscheme_id, weight_layout);
    marmot_profile_id_t profile = ctx->best_profile;
    if (matmul_force_scalar(ctx)) {
        profile = MARMOT_PROFILE_SCALAR;
    }

    marmot_op_signature_t sig = {
        .op_id = op_id,
        .profile_id = profile,
        .matmul_layout = matmul_layout,
        .input_dtype = input->dtype,
        .weight_dtype = weight_dtype,
        .output_dtype = desc->out_q->dtype,
        .accum_dtype = marmot_matmul_accum_dtype(input->dtype),
        .qscheme_id = qscheme_id,
        .quant_block = {0},
        .weight_layout = resolved_weight_layout,
        .epilogue_flags = epilogue_flags,
        .activation = activation,
        .variant_flags = MARMOT_FUSION_NONE,
        .dims = {.matmul = {.N = (uint32_t)N, .K = (uint32_t)K, .M = (uint32_t)M}},
    };

    *packed_out = (marmot_kernel_args_qkv_t){
        .ctx = ctx,
        .input = desc->input,
        .wq = desc->separate.wq,
        .wk = desc->separate.wk,
        .wv = desc->separate.wv,
        .bq = desc->separate.bq,
        .bk = desc->separate.bk,
        .bv = desc->separate.bv,
        .epilogue = desc->epilogue,
        .rope_params = desc->rope_params,
        .out_q = desc->out_q,
        .out_k = desc->out_k,
        .out_v = desc->out_v,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}
