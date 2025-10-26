#include "core/helpers/embedding.h"

#include "marmot/device.h"
#include "marmot/error.h"

#include "core/tensor/tensor_utils.h"

static bool marmot_dtype_is_supported_token_id(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_INT32:
    case MARMOT_DTYPE_INT16:
    case MARMOT_DTYPE_UINT32:
    case MARMOT_DTYPE_UINT16:
    case MARMOT_DTYPE_INT64:
    case MARMOT_DTYPE_UINT64:
        return true;
    default:
        return false;
    }
}

static marmot_dtype_t marmot_embedding_resolve_dtype_out(
    const marmot_context_t *ctx, const marmot_tensor_t *weights, const marmot_embedding_desc_t *desc
) {
    if (desc->dtype_out != (marmot_dtype_t)MARMOT_DTYPE_COUNT) {
        return desc->dtype_out;
    }
    if (!marmot_tensor_is_block_quantized_weight(weights)) {
        return weights->dtype;
    }
    if (ctx == nullptr) {
        return MARMOT_DTYPE_FLOAT32;
    }
    return ctx->policy.embedding_quant_output_dtype;
}

static bool marmot_embedding_default_prefer_gpu_private(const marmot_context_t *ctx) {
    if (ctx == nullptr) {
        return false;
    }
    return ctx->policy.embedding_prefer_gpu_private;
}

static bool marmot_embedding_default_allow_quant_decode(const marmot_context_t *ctx) {
    if (ctx == nullptr) {
        return true;
    }
    return ctx->policy.embedding_allow_quant_decode_on_the_fly;
}

static marmot_error_t
marmot_embedding_validate_row_offsets(const marmot_embedding_desc_t *desc, size_t token_count, size_t *out_batch) {
    if (!desc->ragged) {
        if (desc->row_offsets != nullptr && desc->num_row_offsets != 0) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "row_offsets provided but ragged flag is false");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        *out_batch = 0;
        return MARMOT_SUCCESS;
    }

    if (desc->row_offsets == nullptr || desc->num_row_offsets < 2) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "ragged embeddings require row_offsets with length >= 2");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t previous = 0;
    if (desc->row_offsets[0] != 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "row_offsets[0] must be 0 for ragged embeddings");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    for (size_t i = 1; i < desc->num_row_offsets; ++i) {
        const int32_t raw_value = desc->row_offsets[i];
        if (raw_value < 0) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "row_offsets must be non-negative");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        const size_t current = (size_t)raw_value;
        if (current < previous) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "row_offsets must be non-decreasing");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        previous = current;
    }

    if (previous != token_count) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "row_offsets length does not match number of token ids");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    *out_batch = desc->num_row_offsets - 1;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_embedding_validate_desc_common(
    const marmot_context_t *ctx, const marmot_embedding_desc_t *desc, size_t *out_vocab, size_t *out_dim,
    size_t *out_token_count, marmot_dtype_t *resolved_dtype
) {
    if (desc == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null descriptor provided to marmot_embedding_lookup");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (desc->weights == nullptr || desc->token_ids == nullptr || desc->out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "weights, token_ids, and out tensors must be non-null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_tensor_t *weights = desc->weights;
    if (weights->shape.ndim != 2) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Embedding weights must be a 2D tensor [vocab, dim]");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const size_t vocab = weights->shape.shape[0];
    const size_t dim = weights->shape.shape[1];

    if (dim == 0 || vocab == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Embedding weight dimensions must be non-zero");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_tensor_t *token_ids = desc->token_ids;
    if (!marmot_dtype_is_supported_token_id(token_ids->dtype)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "token_ids tensor must be INT32/INT16/UINT32/UINT16");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t token_count = marmot_tensor_num_elements(token_ids);
    if (token_count == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "token_ids tensor must have at least one element");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_tensor_t *out = desc->out;
    if (out->shape.ndim == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Output tensor must have rank >= 2");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_dtype_t target_dtype = marmot_embedding_resolve_dtype_out(ctx, weights, desc);
    if (out->dtype != target_dtype) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Output tensor dtype does not match requested dtype_out");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t out_ndim = out->shape.ndim;
    const size_t out_last_dim = out->shape.shape[out_ndim - 1];
    if (out_last_dim != dim) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Output tensor's last dimension must match embedding dim");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t prefix_elems = 1;
    for (size_t i = 0; i + 1 < out_ndim; ++i) {
        prefix_elems *= out->shape.shape[i];
    }

    if (prefix_elems != token_count) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Output tensor shape does not match flattened token count");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const bool allow_decode = marmot_preference_resolve(
        desc->allow_quant_decode_on_the_fly, marmot_embedding_default_allow_quant_decode(ctx)
    );
    if (!allow_decode && marmot_tensor_is_block_quantized_weight(weights)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Quantized embeddings require allow_quant_decode_on_the_fly");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t ragged_batch = 0;
    marmot_error_t offsets_err = marmot_embedding_validate_row_offsets(desc, token_count, &ragged_batch);
    if (offsets_err != MARMOT_SUCCESS) {
        return offsets_err;
    }

    if (!desc->ragged && token_ids->shape.ndim > 2) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "token_ids must be rank 1 or 2 when ragged is false");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (desc->ragged && token_ids->shape.ndim != 1) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Ragged embedding lookup expects 1D token_ids buffer");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (desc->ragged) {
        if (desc->row_offsets == nullptr || desc->num_row_offsets == 0) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "row_offsets required when ragged flag is set");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
    } else if (desc->row_offsets != nullptr || desc->num_row_offsets != 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "row_offsets only allowed when ragged flag is true");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (out_vocab != nullptr) {
        *out_vocab = vocab;
    }
    if (out_dim != nullptr) {
        *out_dim = dim;
    }
    if (out_token_count != nullptr) {
        *out_token_count = token_count;
    }
    if (resolved_dtype != nullptr) {
        *resolved_dtype = target_dtype;
    }

    return MARMOT_SUCCESS;
}

void marmot_embedding_resolve_gather_desc(
    const marmot_context_t *ctx, const marmot_embedding_gather_desc_t *desc,
    marmot_embedding_gather_desc_t *resolved_out
) {
    if (desc == nullptr || resolved_out == nullptr) {
        return;
    }

    *resolved_out = *desc;
    const bool prefer_gpu_private =
        marmot_preference_resolve(desc->prefer_gpu_private, marmot_embedding_default_prefer_gpu_private(ctx));
    const bool allow_decode = marmot_preference_resolve(
        desc->allow_quant_decode_on_the_fly, marmot_embedding_default_allow_quant_decode(ctx)
    );
    resolved_out->prefer_gpu_private = prefer_gpu_private ? MARMOT_PREFERENCE_ENABLE : MARMOT_PREFERENCE_DISABLE;
    resolved_out->allow_quant_decode_on_the_fly = allow_decode ? MARMOT_PREFERENCE_ENABLE : MARMOT_PREFERENCE_DISABLE;
}
