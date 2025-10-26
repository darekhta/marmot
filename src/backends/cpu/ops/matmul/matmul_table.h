#ifndef CPU_MATMUL_TABLE_H
#define CPU_MATMUL_TABLE_H

#include "marmot/device.h"
#include "marmot/tensor.h"
#include "marmot/types.h"

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    CPU_MATMUL_LAYOUT_NT = 0,
    CPU_MATMUL_LAYOUT_NN = 1,
} cpu_matmul_layout_t;

typedef struct cpu_matmul_profile_key {
    marmot_dtype_t input_dtype;
    marmot_dtype_t weight_dtype;
    marmot_dtype_t output_dtype;
    marmot_quant_kind_t weight_quant_kind;
    marmot_quant_layout_t weight_quant_layout;
    cpu_matmul_layout_t layout;
    bool is_batched;
    bool has_bias;
    marmot_device_unary_op_t activation;
} cpu_matmul_profile_key_t;

typedef enum {
    CPU_MATMUL_PROFILE_F32 = 0,
    CPU_MATMUL_PROFILE_F64 = 1,
    CPU_MATMUL_PROFILE_F16 = 2,
    CPU_MATMUL_PROFILE_BF16 = 3,
#if MARMOT_ENABLE_FP8
    CPU_MATMUL_PROFILE_FP8_E4M3 = 4,
    CPU_MATMUL_PROFILE_FP8_E5M2 = 5,
#endif
    CPU_MATMUL_PROFILE_F32_BIAS_IDENTITY,
    CPU_MATMUL_PROFILE_F32_BIAS_GELU,
    CPU_MATMUL_PROFILE_F32_BIAS_RELU,
    CPU_MATMUL_PROFILE_F16_BIAS_IDENTITY,
    CPU_MATMUL_PROFILE_F16_BIAS_RELU,
    CPU_MATMUL_PROFILE_BF16_BIAS_IDENTITY,
    CPU_MATMUL_PROFILE_BF16_BIAS_GELU,
    CPU_MATMUL_PROFILE_F32_NN,
    CPU_MATMUL_PROFILE_F64_NN,
    CPU_MATMUL_PROFILE_F16_NN,
    CPU_MATMUL_PROFILE_BF16_NN,
#if MARMOT_ENABLE_FP8
    CPU_MATMUL_PROFILE_FP8_E4M3_NN,
    CPU_MATMUL_PROFILE_FP8_E5M2_NN,
#endif
    CPU_MATMUL_PROFILE_F32_NN_BIAS_IDENTITY,
    CPU_MATMUL_PROFILE_F32_NN_BIAS_GELU,
    CPU_MATMUL_PROFILE_F32_NN_BIAS_RELU,
    CPU_MATMUL_PROFILE_F16_NN_BIAS_IDENTITY,
    CPU_MATMUL_PROFILE_F16_NN_BIAS_RELU,
    CPU_MATMUL_PROFILE_BF16_NN_BIAS_IDENTITY,
    CPU_MATMUL_PROFILE_BF16_NN_BIAS_GELU,
    CPU_MATMUL_PROFILE_COUNT,
} cpu_matmul_profile_id_t;

#define CPU_MATMUL_PROFILE_INVALID ((cpu_matmul_profile_id_t) - 1)

cpu_matmul_profile_key_t cpu_matmul_profile_key_dense(
    const marmot_tensor_t *input, const marmot_tensor_t *weight, const marmot_tensor_t *out,
    const marmot_matmul_epilogue_t *epilogue, cpu_matmul_layout_t layout
);
cpu_matmul_profile_key_t cpu_matmul_profile_key_quantized(
    const marmot_tensor_t *input, const marmot_tensor_t *weight, const marmot_tensor_t *out,
    const marmot_matmul_epilogue_t *epilogue
);
#define CPU_MATMUL_PROFILE_KEY_DENSE_EX(dtype_value, bias_flag, activation_value)                                      \
    ((cpu_matmul_profile_key_t){                                                                                       \
        .input_dtype = (dtype_value),                                                                                  \
        .weight_dtype = (dtype_value),                                                                                 \
        .output_dtype = (dtype_value),                                                                                 \
        .weight_quant_kind = MARMOT_QUANT_KIND_GENERIC,                                                                \
        .weight_quant_layout = MARMOT_QUANT_LAYOUT_GENERIC,                                                            \
        .layout = CPU_MATMUL_LAYOUT_NT,                                                                                \
        .is_batched = false,                                                                                           \
        .has_bias = (bias_flag),                                                                                       \
        .activation = (activation_value),                                                                              \
    })

#define CPU_MATMUL_PROFILE_KEY_DENSE(dtype_value)                                                                      \
    CPU_MATMUL_PROFILE_KEY_DENSE_EX(dtype_value, false, MARMOT_DEVICE_UNARY_IDENTITY)

#define CPU_MATMUL_PROFILE_KEY_DENSE_EX_LAYOUT(dtype_value, bias_flag, activation_value, layout_value)                 \
    ((cpu_matmul_profile_key_t){                                                                                       \
        .input_dtype = (dtype_value),                                                                                  \
        .weight_dtype = (dtype_value),                                                                                 \
        .output_dtype = (dtype_value),                                                                                 \
        .weight_quant_kind = MARMOT_QUANT_KIND_GENERIC,                                                                \
        .weight_quant_layout = MARMOT_QUANT_LAYOUT_GENERIC,                                                            \
        .layout = (layout_value),                                                                                      \
        .is_batched = false,                                                                                           \
        .has_bias = (bias_flag),                                                                                       \
        .activation = (activation_value),                                                                              \
    })

#define CPU_MATMUL_PROFILE_KEY_DENSE_LAYOUT(dtype_value, layout_value)                                                 \
    CPU_MATMUL_PROFILE_KEY_DENSE_EX_LAYOUT(dtype_value, false, MARMOT_DEVICE_UNARY_IDENTITY, layout_value)

cpu_matmul_profile_id_t cpu_matmul_profile_resolve_dense(const cpu_matmul_profile_key_t *key);

typedef marmot_error_t (*cpu_matmul_kernel_fn)(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
);

typedef struct cpu_matmul_ops {
    cpu_matmul_kernel_fn kernel;
    const char *impl_name;
    cpu_matmul_profile_key_t profile;
} cpu_matmul_ops_t;

#endif // CPU_MATMUL_TABLE_H
