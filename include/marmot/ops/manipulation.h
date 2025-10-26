#ifndef MARMOT_OPS_MANIPULATION_H
#define MARMOT_OPS_MANIPULATION_H

#include "../ops_types.h"
#include "../tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Reshape tensor to new shape (total elements must match)
MARMOT_NODISCARD marmot_error_t marmot_reshape(
    const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const size_t *new_shape,
    size_t new_ndim
);

MARMOT_NODISCARD marmot_error_t
marmot_view(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out, size_t byte_offset);

// Copy tensor into contiguous layout (same shape/dtype)
MARMOT_NODISCARD marmot_error_t
marmot_contiguous(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out);

// Transpose tensor by permuting dimensions
MARMOT_NODISCARD marmot_error_t
marmot_transpose(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const int *perm);

// Concatenate tensors along a given axis
MARMOT_NODISCARD marmot_error_t marmot_concat(
    const marmot_context_t *ctx, const marmot_tensor_t *const *tensors, size_t num_tensors, marmot_tensor_t *out,
    int axis
);

// Slice tensor along dimensions
MARMOT_NODISCARD marmot_error_t marmot_slice(
    const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const size_t *starts,
    const size_t *sizes
);

// Gather rows by index
MARMOT_NODISCARD marmot_error_t marmot_gather_rows(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *indices, marmot_tensor_t *output
);

// Scatter UINT64 values into INT32 output using indices.
MARMOT_NODISCARD marmot_error_t marmot_scatter_u64_to_i32(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *indices, marmot_tensor_t *output
);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_OPS_MANIPULATION_H
