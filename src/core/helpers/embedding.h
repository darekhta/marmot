#ifndef MARMOT_CORE_HELPERS_EMBEDDING_H
#define MARMOT_CORE_HELPERS_EMBEDDING_H

#include "marmot/ops/neural.h"

#ifdef __cplusplus
extern "C" {
#endif

marmot_error_t marmot_embedding_validate_desc_common(
    const marmot_context_t *ctx, const marmot_embedding_desc_t *desc, size_t *out_vocab, size_t *out_dim,
    size_t *out_token_count, marmot_dtype_t *resolved_dtype
);

void marmot_embedding_resolve_gather_desc(
    const marmot_context_t *ctx, const marmot_embedding_gather_desc_t *desc,
    marmot_embedding_gather_desc_t *resolved_out
);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_CORE_HELPERS_EMBEDDING_H
