#ifndef MARMOT_CORE_HELPERS_MATMUL_H
#define MARMOT_CORE_HELPERS_MATMUL_H

#include "marmot/device.h"
#include "marmot/graph/op_signature.h"
#include "marmot/quant_traits.h"
#include "marmot/tensor.h"

#include <stdbool.h>

typedef struct {
    bool input_is_fp32;
    bool input_is_fp16;
    bool output_is_fp32;
    bool output_is_fp16;
} marmot_matmul_activation_profile_t;

#ifdef __cplusplus
extern "C" {
#endif

marmot_error_t marmot_matmul_validate_dense(
    const marmot_tensor_t *input, const marmot_tensor_t *weight, const marmot_tensor_t *out, marmot_matmul_dims_t *dims
);

marmot_error_t marmot_matmul_validate_nn(
    const marmot_tensor_t *input, const marmot_tensor_t *weight, const marmot_tensor_t *out, marmot_matmul_dims_t *dims
);

marmot_error_t marmot_matmul_validate_quantized(
    const marmot_tensor_t *input, const marmot_tensor_t *weight, const marmot_tensor_t *out, marmot_matmul_dims_t *dims,
    marmot_matmul_activation_profile_t *profile, const marmot_quant_kind_traits_t **weight_traits
);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_CORE_HELPERS_MATMUL_H
