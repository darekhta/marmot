#ifndef CPU_MATMUL_QUANT_TABLE_H
#define CPU_MATMUL_QUANT_TABLE_H

#include "marmot/quant_block.h"
#include "marmot/types.h"

#include <stdbool.h>
#include <stddef.h>

typedef struct cpu_quant_format_info cpu_quant_format_info_t;
typedef struct cpu_matmul_quant_kernel cpu_matmul_quant_kernel_t;

typedef float (*cpu_matmul_quant_vec_dot_fn)(
    const void *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
);

typedef float (*cpu_matmul_quant_vec_dot_fp16_fn)(
    const void *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks, size_t K
);

typedef float (*cpu_matmul_quant_vec_dot_q8k_fn)(
    const void *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
);

typedef void (*cpu_matmul_quant_vec_dot_q8k_2rows_fn)(
    const void *weights_row0, const void *weights_row1, const marmot_q8_k_block_t *activations, size_t num_blocks,
    float *sum_row0, float *sum_row1
);

typedef void (*cpu_matmul_quant_pack_fp32_fn)(
    const float *activations, size_t stride_k, size_t stride_n, size_t column_index, size_t K, size_t blocks_per_row,
    void *dst_blocks
);

typedef void (*cpu_matmul_quant_pack_fp16_fn)(
    const marmot_float16_t *activations, size_t stride_k, size_t stride_n, size_t column_index, size_t K,
    size_t blocks_per_row, void *dst_blocks
);

typedef struct {
    cpu_matmul_quant_vec_dot_fn dot_q8_0;
    cpu_matmul_quant_vec_dot_fp16_fn dot_fp16;
    cpu_matmul_quant_vec_dot_q8k_fn dot_q8_k;
    cpu_matmul_quant_vec_dot_q8k_2rows_fn dot_q8_k_2rows;
    cpu_matmul_quant_pack_fp32_fn pack_activations_f32;
    cpu_matmul_quant_pack_fp16_fn pack_activations_f16;
} cpu_matmul_quant_ops_t;

struct cpu_matmul_quant_kernel {
    const cpu_quant_format_info_t *format;
    cpu_matmul_quant_ops_t ops;
    bool supports_fp16_input;
    const char *impl_name;
};

#endif // CPU_MATMUL_QUANT_TABLE_H
