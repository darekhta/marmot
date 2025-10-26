#ifndef CPU_UNARY_TABLE_H
#define CPU_UNARY_TABLE_H

#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/types.h"

#include <stdbool.h>
#include <stddef.h>

typedef marmot_error_t (*marmot_unary_simple_fn)(const void *device_ctx, const void *x, void *out, size_t n);

typedef marmot_error_t (*marmot_unary_activation_fn)(
    const void *device_ctx, const void *x, const marmot_activation_params_t *params, void *out, size_t n
);

typedef marmot_error_t (*marmot_unary_fused_activation_fn)(
    const void *device_ctx, const void *x, const void *bias, size_t feature_dim, bool bias_is_scalar,
    const marmot_activation_params_t *params, void *out, size_t n
);

typedef struct marmot_unary_ops {
    marmot_unary_simple_fn abs;
    marmot_unary_simple_fn neg;
    marmot_unary_simple_fn sign;
    marmot_unary_simple_fn sqrt;
    marmot_unary_simple_fn exp;
    marmot_unary_simple_fn log;
    marmot_unary_simple_fn bitwise_not;

    marmot_unary_activation_fn relu;
    marmot_unary_activation_fn gelu;
    marmot_unary_activation_fn gelu_tanh;
    marmot_unary_activation_fn silu;
    marmot_unary_activation_fn sigmoid;
    marmot_unary_activation_fn tanh_act;
    marmot_unary_activation_fn mish;
    marmot_unary_activation_fn elu;
    marmot_unary_activation_fn selu;
    marmot_unary_activation_fn leaky_relu;
    marmot_unary_activation_fn prelu;

    marmot_unary_fused_activation_fn fused_bias_relu;
    marmot_unary_fused_activation_fn fused_bias_gelu;
    marmot_unary_fused_activation_fn fused_bias_gelu_tanh;
    marmot_unary_fused_activation_fn fused_bias_silu;
    marmot_unary_fused_activation_fn fused_bias_sigmoid;
    marmot_unary_fused_activation_fn fused_bias_tanh;
    marmot_unary_fused_activation_fn fused_bias_mish;
    marmot_unary_fused_activation_fn fused_bias_elu;
    marmot_unary_fused_activation_fn fused_bias_selu;
    marmot_unary_fused_activation_fn fused_bias_leaky_relu;
    marmot_unary_fused_activation_fn fused_bias_prelu;

    const char *impl_name;

    marmot_unary_simple_fn simple[MARMOT_DEVICE_UNARY_COUNT];
    marmot_unary_activation_fn activation_table[MARMOT_DEVICE_UNARY_COUNT];
    marmot_unary_fused_activation_fn fused_table[MARMOT_DEVICE_UNARY_COUNT];
    bool is_simple[MARMOT_DEVICE_UNARY_COUNT];
} marmot_unary_ops_t;

typedef marmot_unary_simple_fn cpu_unary_simple_fn;
typedef marmot_unary_activation_fn cpu_activation_fn;
typedef marmot_unary_fused_activation_fn cpu_fused_bias_activation_fn;
typedef marmot_unary_ops_t cpu_unary_ops_t;

typedef enum {
    CPU_UNARY_IMPL_SCALAR = 0,
    CPU_UNARY_IMPL_NEON = 1,
    CPU_UNARY_IMPL_AVX2 = 2,
    CPU_UNARY_IMPL_ACCELERATE = 3,
    CPU_UNARY_IMPL_CUSTOM = 4,
} cpu_unary_impl_kind_t;

typedef struct cpu_unary_traits {
    marmot_dtype_t dtype;
    cpu_unary_impl_kind_t impl_kind;
    cpu_unary_ops_t ops;
} cpu_unary_traits_t;

#define CPU_UNARY_REGISTER_TRAITS(symbol)

#endif // CPU_UNARY_TABLE_H
