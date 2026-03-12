#ifndef MARMOT_CORE_OPS_INTERNAL_H
#define MARMOT_CORE_OPS_INTERNAL_H

#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/ops/matmul.h"
#include "marmot/ops/neural.h"
#include "marmot/ops/paged_attention.h"

#include "core/dispatch/dispatch_build.h"
#include "core/dispatch/dispatch_execute.h"
#include "core/helpers/embedding.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline marmot_reduction_params_t marmot_desc_to_params(const marmot_reduction_desc_t *desc) {
    marmot_reduction_params_t params = {
        .axes = desc->axes,
        .num_axes = desc->num_axes,
        .keepdims = desc->keepdims,
        .unbiased = desc->unbiased,
        .epsilon = desc->epsilon,
    };
    return params;
}

marmot_error_t marmot_dispatch_unary_uniform(
    const marmot_context_t *ctx, marmot_device_unary_op_t device_op, marmot_op_id_t op_id, const marmot_tensor_t *x,
    const marmot_activation_params_t *params, marmot_tensor_t *out, const char *op_name
);

marmot_error_t marmot_dispatch_binary(
    const marmot_context_t *ctx, marmot_op_id_t op_id, const marmot_tensor_t *a, const marmot_tensor_t *b,
    marmot_tensor_t *out, bool allow_bool_out, bool use_stride_mode_2d, const char *label
);

marmot_error_t marmot_dispatch_ternary(
    const marmot_context_t *ctx, marmot_device_ternary_op_t op, marmot_op_id_t op_id, const marmot_tensor_t *a,
    const marmot_tensor_t *b, const marmot_tensor_t *c, marmot_tensor_t *out, marmot_dtype_t lookup_dtype,
    const char *op_name
);

marmot_error_t marmot_dispatch_reduction(
    const marmot_context_t *ctx, marmot_device_reduction_op_t op, const marmot_tensor_t *input,
    marmot_tensor_t *out_values, marmot_tensor_t *out_indices, const marmot_reduction_params_t *params,
    const char *op_name
);

marmot_error_t marmot_convert_dispatch(
    const marmot_context_t *ctx, marmot_dtype_t dst_dtype, void *dst, marmot_dtype_t src_dtype, const void *src,
    size_t n
);

marmot_error_t marmot_quantize_dispatch(
    const marmot_context_t *ctx, marmot_quant_kind_t kind, const marmot_tensor_t *input,
    const marmot_quant_params_t *params, marmot_tensor_t *output
);

marmot_error_t marmot_dequantize_dispatch(
    const marmot_context_t *ctx, marmot_quant_kind_t kind, const marmot_tensor_t *input, marmot_tensor_t *output
);

marmot_error_t marmot_compute_quant_params_dispatch(
    const marmot_context_t *ctx, const marmot_tensor_t *tensor, marmot_dtype_t target_dtype, size_t block_size,
    marmot_quant_params_t *out_params
);

marmot_error_t
marmot_vec_dot_dispatch(const marmot_context_t *ctx, const marmot_vec_dot_descriptor_t *desc, float *result);

marmot_error_t marmot_layernorm_dispatch(const marmot_context_t *ctx, const marmot_layernorm_desc_t *desc);

marmot_error_t marmot_rmsnorm_dispatch(const marmot_context_t *ctx, const marmot_rmsnorm_desc_t *desc);

marmot_error_t marmot_rmsnorm_gemma_dispatch(const marmot_context_t *ctx, const marmot_rmsnorm_desc_t *desc);

marmot_error_t marmot_softmax_dispatch(const marmot_context_t *ctx, const marmot_softmax_desc_t *desc);

marmot_error_t marmot_topk_impl(const marmot_context_t *ctx, const marmot_topk_desc_t *desc);

marmot_error_t marmot_moe_experts_impl(const marmot_context_t *ctx, const marmot_moe_experts_desc_t *desc);

marmot_error_t marmot_embedding_lookup_impl(const marmot_context_t *ctx, const marmot_embedding_desc_t *desc);

marmot_error_t marmot_linear_impl(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);

marmot_error_t marmot_matmul_impl(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);

marmot_error_t marmot_matmul_bias_impl(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);

marmot_error_t marmot_matmul_bias_relu_impl(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);

marmot_error_t marmot_matmul_bias_gelu_impl(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);

marmot_error_t marmot_matmul_bias_silu_impl(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);

marmot_error_t marmot_matmul_prepack_quant_weight_impl(const marmot_context_t *ctx, const marmot_tensor_t *weight);

marmot_error_t marmot_matmul_qkv_impl(const marmot_context_t *ctx, const marmot_matmul_qkv_desc_t *desc);

marmot_error_t marmot_matmul_qkv_shared_input_impl(const marmot_context_t *ctx, const marmot_matmul_qkv_desc_t *desc);

marmot_error_t marmot_matmul_qkv_projection_impl(const marmot_context_t *ctx, const marmot_matmul_qkv_desc_t *desc);

marmot_error_t marmot_paged_attention_impl(const marmot_context_t *ctx, const marmot_paged_attention_desc_t *desc);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_CORE_OPS_INTERNAL_H
