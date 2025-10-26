#ifndef CPU_NORMALIZATION_TABLE_H
#define CPU_NORMALIZATION_TABLE_H

#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/types.h"

#include <stddef.h>

typedef struct {
    const void *x;
    const void *residual;
    const void *weight;
    const void *bias;
    void *out;
    size_t outer_size;
    size_t norm_size;
    float eps;
} cpu_layernorm_params_t;

typedef struct {
    const void *x;
    const void *residual;
    const void *weight;
    void *out;
    size_t outer_size;
    size_t norm_size;
    float eps;
    float weight_offset;
} cpu_rmsnorm_params_t;

typedef marmot_error_t (*cpu_layernorm_fn)(const void *device_ctx, const cpu_layernorm_params_t *params);
typedef marmot_error_t (*cpu_rmsnorm_fn)(const void *device_ctx, const cpu_rmsnorm_params_t *params);

typedef struct cpu_norm_ops {
    cpu_layernorm_fn layernorm;
    cpu_rmsnorm_fn rmsnorm;
    const char *impl_name;
} cpu_norm_ops_t;

typedef enum {
    CPU_NORM_IMPL_SCALAR = 0,
    CPU_NORM_IMPL_NEON = 1,
    CPU_NORM_IMPL_AVX2 = 2,
    CPU_NORM_IMPL_ACCELERATE = 3,
    CPU_NORM_IMPL_CUSTOM = 4,
} cpu_norm_impl_kind_t;

typedef struct {
    marmot_dtype_t dtype;
    cpu_norm_impl_kind_t impl_kind;
    cpu_norm_ops_t ops;
} cpu_norm_traits_t;

#define CPU_NORM_REGISTER_TRAITS(symbol)

#endif // CPU_NORMALIZATION_TABLE_H
