#ifndef MARMOT_ERROR_CAPTURE_H
#define MARMOT_ERROR_CAPTURE_H

#include "allocator.h"
#include "error.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

MARMOT_NODISCARD marmot_error_t
marmot_init_capture(marmot_backend_type_t backend, marmot_context_t **out_ctx, marmot_error_info_t *out_error_info);

MARMOT_NODISCARD marmot_error_t marmot_allocator_get_usage_capture(
    const marmot_context_t *ctx, marmot_allocator_usage_t *usage, marmot_error_info_t *out_error_info
);

MARMOT_NODISCARD marmot_error_t marmot_tensor_create_capture(
    const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_dtype_t dtype, marmot_tensor_t **out_tensor,
    marmot_error_info_t *out_error_info
);

MARMOT_NODISCARD marmot_error_t marmot_matmul_capture(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out, marmot_error_info_t *out_error_info
);

MARMOT_NODISCARD marmot_error_t marmot_linear_capture(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out, marmot_error_info_t *out_error_info
);

MARMOT_NODISCARD marmot_error_t marmot_layernorm_capture(
    const marmot_context_t *ctx, const marmot_layernorm_desc_t *desc, marmot_error_info_t *out_error_info
);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_ERROR_CAPTURE_H
