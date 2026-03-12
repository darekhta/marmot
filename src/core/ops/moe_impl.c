#include "marmot/error.h"
#include "marmot/ops/neural.h"

#include <limits.h>

#include "core/dispatch/dispatch_execute.h"
#include "core/dispatch/fusion_flags.h"
#include "core/helpers/quant.h"
#include "core/tensor/tensor_utils.h"
#include "graph/kernel_dispatch_args.gen.h"

static bool moe_value_dtype_supported(marmot_dtype_t dtype) {
    return dtype == MARMOT_DTYPE_FLOAT32 || dtype == MARMOT_DTYPE_FLOAT16;
}

static marmot_error_t canonical_last_axis(const marmot_tensor_t *tensor, int32_t axis, int32_t *out_axis) {
    if (tensor == nullptr || out_axis == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (tensor->shape.ndim == 0 || tensor->shape.ndim > INT32_MAX) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "TopK requires a tensor with rank");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    int32_t resolved = axis;
    if (resolved < 0) {
        resolved += (int32_t)tensor->shape.ndim;
    }
    if (resolved < 0 || resolved >= (int32_t)tensor->shape.ndim) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "TopK axis is out of range");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (resolved != (int32_t)tensor->shape.ndim - 1) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "TopK currently supports only the last axis");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    *out_axis = resolved;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_topk_impl(const marmot_context_t *ctx, const marmot_topk_desc_t *desc) {
    if (ctx == nullptr || !marmot_topk_desc_is_valid(desc)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "TopK requires non-null context and tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!moe_value_dtype_supported(desc->x->dtype) || desc->values_out->dtype != desc->x->dtype ||
        desc->indices_out->dtype != MARMOT_DTYPE_INT32) {
        marmot_set_error(
            MARMOT_ERROR_UNSUPPORTED_DTYPE,
            "TopK currently supports FLOAT16/FLOAT32 values with matching output dtype and INT32 indices"
        );
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (desc->x->shape.ndim != 2 || desc->values_out->shape.ndim != 2 || desc->indices_out->shape.ndim != 2) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "TopK expects 2D tensors");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    int32_t axis = -1;
    marmot_error_t axis_status = canonical_last_axis(desc->x, desc->axis, &axis);
    if (axis_status != MARMOT_SUCCESS) {
        return axis_status;
    }

    const size_t rows = desc->x->shape.shape[0];
    const size_t cols = desc->x->shape.shape[1];
    if (desc->k > cols || desc->values_out->shape.shape[0] != rows || desc->indices_out->shape.shape[0] != rows ||
        desc->values_out->shape.shape[1] != desc->k || desc->indices_out->shape.shape[1] != desc->k) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "TopK tensor shapes do not match");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_TOPK,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = desc->x->dtype,
        .weight_dtype = desc->indices_out->dtype,
        .output_dtype = desc->values_out->dtype,
        .accum_dtype = MARMOT_DTYPE_FLOAT32,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .stride_mode = MARMOT_STRIDE_MODE_STRIDED,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
    };
    marmot_kernel_args_topk_t packed = {
        .ctx = ctx,
        .input = desc->x,
        .values_out = desc->values_out,
        .indices_out = desc->indices_out,
        .axis = axis,
        .k = desc->k,
    };
    return marmot_execute_signature(ctx, &sig, &packed, "TopK");
}

marmot_error_t marmot_moe_experts_impl(const marmot_context_t *ctx, const marmot_moe_experts_desc_t *desc) {
    if (ctx == nullptr || !marmot_moe_experts_desc_is_valid(desc)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "MoE experts requires non-null context and tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (desc->router_weight_policy >= MARMOT_ROUTER_WEIGHT_POLICY_COUNT) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "MoE experts requires a valid router weight policy");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!moe_value_dtype_supported(desc->hidden_states->dtype) ||
        desc->topk_weights->dtype != desc->hidden_states->dtype || desc->out->dtype != desc->hidden_states->dtype ||
        desc->topk_ids->dtype != MARMOT_DTYPE_INT32) {
        marmot_set_error(
            MARMOT_ERROR_UNSUPPORTED_DTYPE,
            "MoE experts currently supports FLOAT16/FLOAT32 activations with matching output/router weights and INT32 "
            "expert ids"
        );
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (desc->hidden_states->shape.ndim != 2 || desc->topk_ids->shape.ndim != 2 ||
        desc->topk_weights->shape.ndim != 2 || desc->out->shape.ndim != 2 || desc->gate_exps->shape.ndim != 3 ||
        desc->up_exps->shape.ndim != 3 || desc->down_exps->shape.ndim != 3) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "MoE experts expects 2D activations and 3D expert weights");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    const size_t tokens = desc->hidden_states->shape.shape[0];
    const size_t hidden = desc->hidden_states->shape.shape[1];
    const size_t experts_per_token = desc->topk_ids->shape.shape[1];
    const size_t experts = desc->gate_exps->shape.shape[2];
    const size_t ff_length = desc->gate_exps->shape.shape[1];

    if (desc->topk_ids->shape.shape[0] != tokens || desc->topk_weights->shape.shape[0] != tokens ||
        desc->topk_weights->shape.shape[1] != experts_per_token || desc->out->shape.shape[0] != tokens ||
        desc->out->shape.shape[1] != hidden) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "MoE experts token dimensions do not match");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (desc->gate_exps->shape.shape[0] != hidden || desc->up_exps->shape.shape[0] != hidden ||
        desc->up_exps->shape.shape[1] != ff_length || desc->up_exps->shape.shape[2] != experts ||
        desc->down_exps->shape.shape[0] != ff_length || desc->down_exps->shape.shape[1] != hidden ||
        desc->down_exps->shape.shape[2] != experts) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "MoE expert tensor shapes do not match");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (desc->ffn_type != MARMOT_FFN_SWIGLU && desc->ffn_type != MARMOT_FFN_GEGLU) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "MoE experts currently supports SwiGLU and GeGLU only");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const bool gate_quantized = marmot_tensor_is_block_quantized_weight(desc->gate_exps);
    const bool up_quantized = marmot_tensor_is_block_quantized_weight(desc->up_exps);
    const bool down_quantized = marmot_tensor_is_block_quantized_weight(desc->down_exps);
    if (gate_quantized || up_quantized || down_quantized) {
        if (!(gate_quantized && up_quantized && down_quantized)) {
            marmot_set_error(
                MARMOT_ERROR_INVALID_ARGUMENT, "MoE expert quantization must be consistent across weights"
            );
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
    } else if (desc->gate_exps->dtype != desc->hidden_states->dtype ||
               desc->up_exps->dtype != desc->hidden_states->dtype ||
               desc->down_exps->dtype != desc->hidden_states->dtype) {
        marmot_set_error(
            MARMOT_ERROR_UNSUPPORTED_DTYPE, "Dense MoE experts currently requires activation and weight dtypes to match"
        );
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    marmot_qscheme_id_t qscheme_id = MARMOT_QSCHEME_NONE;
    if (gate_quantized) {
        qscheme_id = marmot_quant_kind_to_qscheme(desc->gate_exps->quant_kind);
    }

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_MOE_EXPERTS,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = desc->hidden_states->dtype,
        .weight_dtype = desc->gate_exps->dtype,
        .output_dtype = desc->out->dtype,
        .accum_dtype = MARMOT_DTYPE_FLOAT32,
        .qscheme_id = qscheme_id,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .stride_mode = MARMOT_STRIDE_MODE_STRIDED,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
    };
    marmot_kernel_args_moe_experts_t packed = {
        .ctx = ctx,
        .hidden_states = desc->hidden_states,
        .gate_exps = desc->gate_exps,
        .up_exps = desc->up_exps,
        .down_exps = desc->down_exps,
        .topk_ids = desc->topk_ids,
        .topk_weights = desc->topk_weights,
        .out = desc->out,
        .ffn_type = desc->ffn_type,
        .weights_scale = desc->weights_scale,
        .router_weight_policy = desc->router_weight_policy,
    };
    return marmot_execute_signature(ctx, &sig, &packed, "MoEExperts");
}
