#include "marmot/error.h"
#include "marmot/graph/op_signature.h"
#include "marmot/ops/neural.h"

#include "core/dispatch/fusion_flags.h"
#include "core/helpers/embedding.h"
#include "core/helpers/quant.h"
#include "core/tensor/tensor_utils.h"
#include "graph/kernel_dispatch_args.gen.h"

static marmot_qscheme_id_t marmot_embedding_weight_qscheme(const marmot_tensor_t *weights) {
    if (!marmot_tensor_is_block_quantized_weight(weights)) {
        return MARMOT_QSCHEME_NONE;
    }
    return marmot_quant_kind_to_qscheme(weights->quant_kind);
}

marmot_error_t marmot_embedding_gather_build(
    const marmot_context_t *ctx, const marmot_embedding_gather_desc_t *desc, marmot_op_signature_t *sig_out,
    marmot_kernel_args_embedding_t *packed_out, marmot_dtype_t *resolved_dtype_out
) {
    if (sig_out == nullptr || packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Embedding gather requires non-null signature and args");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Embedding gather requires non-null context");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (desc == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Embedding gather requires non-null descriptor");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_embedding_gather_desc_t resolved_desc;
    marmot_embedding_resolve_gather_desc(ctx, desc, &resolved_desc);
    const bool prefer_gpu_private = marmot_preference_resolve(resolved_desc.prefer_gpu_private, false);
    const bool allow_quant_decode = marmot_preference_resolve(resolved_desc.allow_quant_decode_on_the_fly, false);

    marmot_embedding_desc_t validate_desc = marmot_embedding_desc_default();
    validate_desc.weights = resolved_desc.weights;
    validate_desc.token_ids = resolved_desc.token_ids;
    validate_desc.out = resolved_desc.out;
    validate_desc.dtype_out = resolved_desc.dtype_out;
    validate_desc.scale = resolved_desc.scale;
    validate_desc.padding_id = resolved_desc.padding_id;
    validate_desc.bounds_check = resolved_desc.bounds_check;
    validate_desc.ragged = false;
    validate_desc.row_offsets = nullptr;
    validate_desc.num_row_offsets = 0;
    validate_desc.prefer_gpu_private = resolved_desc.prefer_gpu_private;
    validate_desc.allow_quant_decode_on_the_fly = resolved_desc.allow_quant_decode_on_the_fly;

    size_t vocab = 0;
    size_t dim = 0;
    size_t token_count = 0;
    marmot_dtype_t resolved_dtype = (marmot_dtype_t)MARMOT_DTYPE_COUNT;
    marmot_error_t err =
        marmot_embedding_validate_desc_common(ctx, &validate_desc, &vocab, &dim, &token_count, &resolved_dtype);
    if (err != MARMOT_SUCCESS) {
        return err;
    }

    marmot_dtype_t weight_dtype = resolved_desc.weights->dtype;
    if (marmot_tensor_is_block_quantized_weight(resolved_desc.weights)) {
        weight_dtype = resolved_dtype;
    }

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_EMBEDDING,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = resolved_desc.token_ids->dtype,
        .weight_dtype = weight_dtype,
        .output_dtype = resolved_dtype,
        .accum_dtype = MARMOT_DTYPE_COUNT,
        .qscheme_id = marmot_embedding_weight_qscheme(resolved_desc.weights),
        .quant_block = {0},
        .weight_layout = MARMOT_WEIGHT_LAYOUT_SEPARATE,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    *packed_out = (marmot_kernel_args_embedding_t){
        .ctx = ctx,
        .weights = resolved_desc.weights,
        .token_ids = resolved_desc.token_ids,
        .out = resolved_desc.out,
        .dtype_out = resolved_dtype,
        .scale = resolved_desc.scale,
        .padding_id = resolved_desc.padding_id,
        .bounds_check = resolved_desc.bounds_check,
        .prefer_gpu_private = prefer_gpu_private,
        .allow_quant_decode_on_the_fly = allow_quant_decode,
    };
    *sig_out = sig;
    if (resolved_dtype_out != nullptr) {
        *resolved_dtype_out = resolved_dtype;
    }
    return MARMOT_SUCCESS;
}
