#ifndef MARMOT_OPS_ROPE_H
#define MARMOT_OPS_ROPE_H

#include "../ops_types.h"
#include "../tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

MARMOT_NODISCARD marmot_error_t marmot_rope(
    const marmot_context_t *ctx, const marmot_tensor_t *x, const marmot_rope_params_t *params, marmot_tensor_t *out
);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_OPS_ROPE_H
