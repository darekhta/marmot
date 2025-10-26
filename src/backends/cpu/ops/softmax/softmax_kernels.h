#ifndef CPU_SOFTMAX_KERNELS_H
#define CPU_SOFTMAX_KERNELS_H

#include "marmot/error.h"
#include "marmot/types.h"

#include "core/helpers/norm.h"
#include "cpu_backend_internal.h"

typedef marmot_error_t (*cpu_softmax_kernel_fn)(
    cpu_context_t *ctx, const marmot_tensor_t *x, const marmot_softmax_shape_t *shape, marmot_tensor_t *out
);

typedef enum {
    SOFTMAX_IMPL_SCALAR = 0,
    SOFTMAX_IMPL_NEON = 1,
    SOFTMAX_IMPL_AVX2 = 2,
} cpu_softmax_impl_kind_t;

typedef struct {
    marmot_dtype_t dtype;
    cpu_softmax_impl_kind_t impl_kind;
    cpu_softmax_kernel_fn fn;
    const char *impl_name;
} cpu_softmax_traits_t;

#define SOFTMAX_REGISTER_TRAITS(symbol)

#endif // CPU_SOFTMAX_KERNELS_H
