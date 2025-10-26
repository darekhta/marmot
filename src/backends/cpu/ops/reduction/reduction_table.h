#ifndef CPU_REDUCTION_TABLE_H
#define CPU_REDUCTION_TABLE_H

#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/types.h"

#include <stddef.h>
#include <stdint.h>

typedef marmot_error_t (*marmot_reduction_numeric_fn)(
    const void *device_ctx, const void *base, size_t length, double *out_value
);

typedef marmot_error_t (*marmot_reduction_arg_fn)(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
);

typedef struct marmot_reduction_ops {
    marmot_reduction_numeric_fn sum;
    marmot_reduction_numeric_fn mean;
    marmot_reduction_numeric_fn prod;
    marmot_reduction_numeric_fn min;
    marmot_reduction_numeric_fn max;

    marmot_reduction_arg_fn argmax;
    marmot_reduction_arg_fn argmin;

    const char *impl_name;

    marmot_reduction_numeric_fn numeric[MARMOT_DEVICE_REDUCTION_COUNT];
    marmot_reduction_arg_fn arg[MARMOT_DEVICE_REDUCTION_COUNT];
} marmot_reduction_ops_t;

typedef enum {
    CPU_REDUCE_IMPL_SCALAR = 0,
    CPU_REDUCE_IMPL_NEON = 1,
    CPU_REDUCE_IMPL_AVX2 = 2,
    CPU_REDUCE_IMPL_ACCELERATE = 3,
    CPU_REDUCE_IMPL_CUSTOM = 4,
} cpu_reduce_impl_kind_t;

typedef marmot_reduction_numeric_fn cpu_reduce_numeric_fn;
typedef marmot_reduction_arg_fn cpu_reduce_arg_fn;
typedef marmot_reduction_ops_t cpu_reduce_ops_t;

typedef struct {
    marmot_dtype_t dtype;
    cpu_reduce_impl_kind_t impl_kind;
    cpu_reduce_ops_t ops;
} cpu_reduce_traits_t;

#define CPU_REDUCTION_REGISTER_TRAITS(symbol)

#endif // CPU_REDUCTION_TABLE_H
