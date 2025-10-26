#include "marmot/error.h"
#include "marmot/ops/neural.h"

#include <string.h>

#include "core/helpers/embedding.h"

marmot_error_t marmot_embedding_lookup_impl(const marmot_context_t *ctx, const marmot_embedding_desc_t *desc) {
    if (ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Embedding lookup requires non-null context");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t vocab = 0;
    size_t dim = 0;
    size_t token_count = 0;
    marmot_dtype_t resolved_dtype = (marmot_dtype_t)MARMOT_DTYPE_COUNT;
    marmot_error_t err = marmot_embedding_validate_desc_common(ctx, desc, &vocab, &dim, &token_count, &resolved_dtype);
    if (err != MARMOT_SUCCESS) {
        return err;
    }

    const marmot_tensor_t *token_ids_ptr = desc->token_ids;
    marmot_tensor_t token_flat;
    if (!desc->ragged && desc->token_ids->shape.ndim != 1) {
        memset(&token_flat, 0, sizeof(token_flat));
        memcpy(&token_flat, desc->token_ids, sizeof(token_flat));
        token_flat.shape.ndim = 1;
        token_flat.shape.shape[0] = token_count;
        token_flat.shape.strides[0] = 1;
        token_ids_ptr = (const marmot_tensor_t *)&token_flat;
    }

    marmot_tensor_t out_flat;
    marmot_tensor_t *out_ptr = desc->out;
    if (desc->out->shape.ndim != 2) {
        memset(&out_flat, 0, sizeof(out_flat));
        memcpy(&out_flat, desc->out, sizeof(out_flat));
        out_flat.shape.ndim = 2;
        out_flat.shape.shape[0] = token_count;
        out_flat.shape.shape[1] = dim;
        out_flat.shape.strides[1] = 1;
        out_flat.shape.strides[0] = dim;
        out_ptr = &out_flat;
    }

    marmot_embedding_gather_desc_t gather_desc = {
        .weights = desc->weights,
        .token_ids = token_ids_ptr,
        .out = out_ptr,
        .dtype_out = resolved_dtype,
        .scale = desc->scale,
        .padding_id = desc->padding_id,
        .bounds_check = desc->bounds_check,
        .prefer_gpu_private = desc->prefer_gpu_private,
        .allow_quant_decode_on_the_fly = desc->allow_quant_decode_on_the_fly,
    };

    return marmot_embedding_gather(ctx, &gather_desc);
}
