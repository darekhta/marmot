#ifndef MARMOT_CORE_HELPERS_ELEMENTWISE_H
#define MARMOT_CORE_HELPERS_ELEMENTWISE_H

#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/tensor.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

bool marmot_elementwise_unary_supports_bias(marmot_device_unary_op_t op);
marmot_error_t marmot_elementwise_bias_info(
    const marmot_tensor_t *x, const marmot_tensor_t *bias, size_t *feature_dim, bool *bias_is_scalar
);
bool marmot_unary_op_requires_params(marmot_device_unary_op_t op);
marmot_error_t marmot_unary_prepare_activation_params(
    marmot_device_unary_op_t op, const marmot_activation_params_t *input, marmot_activation_params_t *out_params
);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_CORE_HELPERS_ELEMENTWISE_H
