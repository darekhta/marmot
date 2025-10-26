#ifndef MARMOT_CORE_DISPATCH_BUILD_H
#define MARMOT_CORE_DISPATCH_BUILD_H

#include "marmot/device.h"
#include "marmot/graph/op_signature.h"
#include "marmot/ops/matmul.h"
#include "marmot/ops/neural.h"
#include "marmot/ops/paged_attention.h"
#include "marmot/ops_types.h"
#include "marmot/tensor.h"

#include "graph/kernel_dispatch_args.gen.h"

#ifdef __cplusplus
extern "C" {
#endif

marmot_error_t marmot_quantize_build(
    const marmot_context_t *ctx, marmot_quant_kind_t kind, const marmot_tensor_t *input,
    const marmot_quant_params_t *params, marmot_tensor_t *output, marmot_op_signature_t *sig_out,
    marmot_kernel_args_quantize_t *packed_out, marmot_quant_layout_t *layout_out
);

marmot_error_t marmot_dequantize_build(
    const marmot_context_t *ctx, marmot_quant_kind_t kind, const marmot_tensor_t *input, marmot_tensor_t *output,
    marmot_op_signature_t *sig_out, marmot_kernel_args_dequantize_t *packed_out, marmot_quant_layout_t *layout_out
);

marmot_error_t marmot_compute_quant_params_build(
    const marmot_context_t *ctx, const marmot_tensor_t *tensor, marmot_dtype_t target_dtype, size_t block_size,
    marmot_quant_params_t *out_params, marmot_op_signature_t *sig_out, marmot_kernel_args_compute_qparams_t *packed_out
);

marmot_error_t marmot_vec_dot_build(
    const marmot_context_t *ctx, const marmot_vec_dot_descriptor_t *desc, float *result, marmot_op_signature_t *sig_out,
    marmot_kernel_args_vec_dot_t *packed_out
);

marmot_error_t marmot_unary_build(
    const marmot_context_t *ctx, marmot_device_unary_op_t device_op, marmot_op_id_t op_id, const marmot_tensor_t *x,
    const marmot_activation_params_t *params, marmot_tensor_t *out, marmot_op_signature_t *sig_out,
    marmot_kernel_args_unary_t *packed_out
);

marmot_error_t marmot_binary_build(
    const marmot_context_t *ctx, marmot_op_id_t op_id, const marmot_tensor_t *a, const marmot_tensor_t *b,
    marmot_tensor_t *out, bool allow_bool_out, bool use_stride_mode_2d, marmot_op_signature_t *sig_out,
    marmot_kernel_args_binary_t *packed_out
);

marmot_error_t marmot_ternary_build(
    const marmot_context_t *ctx, marmot_device_ternary_op_t op, marmot_op_id_t op_id, const marmot_tensor_t *a,
    const marmot_tensor_t *b, const marmot_tensor_t *c, marmot_tensor_t *out, marmot_op_signature_t *sig_out,
    marmot_kernel_args_ternary_t *packed_out
);

marmot_error_t marmot_convert_build(
    const marmot_context_t *ctx, marmot_dtype_t dst_dtype, void *dst, marmot_dtype_t src_dtype, const void *src,
    size_t n, marmot_op_signature_t *sig_out, marmot_kernel_args_convert_t *packed_out
);

marmot_error_t marmot_embedding_gather_build(
    const marmot_context_t *ctx, const marmot_embedding_gather_desc_t *desc, marmot_op_signature_t *sig_out,
    marmot_kernel_args_embedding_t *packed_out, marmot_dtype_t *resolved_dtype_out
);

marmot_error_t marmot_matmul_build(
    const marmot_context_t *ctx, marmot_matmul_layout_t matmul_layout, marmot_op_id_t op_id,
    const marmot_tensor_t *input, const marmot_tensor_t *weight, const marmot_matmul_epilogue_t *epilogue,
    marmot_tensor_t *output, marmot_qscheme_id_t qscheme_id, marmot_weight_layout_t weight_layout,
    marmot_op_signature_t *sig_out, marmot_kernel_args_matmul_t *packed_out
);

marmot_error_t marmot_matmul_qkv_build(
    const marmot_context_t *ctx, const marmot_matmul_qkv_desc_t *desc, marmot_op_id_t op_id,
    marmot_matmul_layout_t matmul_layout, marmot_weight_layout_t weight_layout, marmot_op_signature_t *sig_out,
    marmot_kernel_args_qkv_t *packed_out
);

marmot_error_t marmot_paged_attention_build(
    const marmot_context_t *ctx, const marmot_paged_attention_desc_t *desc, marmot_op_signature_t *sig_out,
    marmot_kernel_args_paged_attention_t *packed_out
);

marmot_error_t marmot_rope_build(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_rope_params_t *params,
    marmot_tensor_t *output, marmot_op_signature_t *sig_out, marmot_kernel_args_rope_t *packed_out
);

marmot_error_t marmot_reshape_build(
    const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output, const size_t *new_shape,
    size_t new_ndim, marmot_op_signature_t *sig_out, marmot_kernel_args_reshape_t *packed_out
);

marmot_error_t marmot_view_build(
    const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output, size_t byte_offset,
    marmot_op_signature_t *sig_out, marmot_kernel_args_view_t *packed_out
);

marmot_error_t marmot_contiguous_build(
    const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output, marmot_op_signature_t *sig_out,
    marmot_kernel_args_contiguous_t *packed_out
);

marmot_error_t marmot_transpose_build(
    const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output, const int *perm,
    marmot_op_signature_t *sig_out, marmot_kernel_args_transpose_t *packed_out
);

marmot_error_t marmot_concat_build(
    const marmot_context_t *ctx, const marmot_tensor_t *const *inputs, size_t num_inputs, marmot_tensor_t *output,
    int axis, marmot_op_signature_t *sig_out, marmot_kernel_args_concat_t *packed_out
);

marmot_error_t marmot_slice_build(
    const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output, const size_t *starts,
    const size_t *sizes, marmot_op_signature_t *sig_out, marmot_kernel_args_slice_t *packed_out
);

marmot_error_t marmot_gather_rows_build(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *indices, marmot_tensor_t *output,
    marmot_op_signature_t *sig_out, marmot_kernel_args_gather_rows_t *packed_out
);

marmot_error_t marmot_scatter_u64_to_i32_build(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *indices, marmot_tensor_t *output,
    marmot_op_signature_t *sig_out, marmot_kernel_args_gather_rows_t *packed_out
);

marmot_error_t marmot_reduction_build(
    const marmot_context_t *ctx, marmot_device_reduction_op_t op, const marmot_tensor_t *input,
    marmot_tensor_t *out_values, marmot_tensor_t *out_indices, const marmot_reduction_params_t *params,
    const char *op_name, marmot_op_signature_t *sig_out, marmot_kernel_args_reduction_t *packed_out
);

marmot_error_t marmot_layernorm_build(
    const marmot_context_t *ctx, const marmot_layernorm_desc_t *desc, marmot_op_signature_t *sig_out,
    marmot_kernel_args_layernorm_t *packed_out
);

marmot_error_t marmot_rmsnorm_build(
    const marmot_context_t *ctx, const marmot_rmsnorm_desc_t *desc, marmot_op_id_t op_id,
    marmot_op_signature_t *sig_out, marmot_kernel_args_rms_norm_t *packed_out
);

marmot_error_t marmot_softmax_build(
    const marmot_context_t *ctx, const marmot_softmax_desc_t *desc, marmot_op_signature_t *sig_out,
    marmot_kernel_args_softmax_t *packed_out
);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_CORE_DISPATCH_BUILD_H
