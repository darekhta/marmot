#ifndef CPU_MATMUL_QUANT_ACTIVATION_H
#define CPU_MATMUL_QUANT_ACTIVATION_H

#include "marmot/quant_block.h"
#include "marmot/types.h"

void cpu_matmul_quant_pack_q8_0_f32(
    const float *activations, size_t stride_k, size_t stride_n, size_t column_index, size_t K, size_t blocks_per_row,
    void *dst_blocks
);

void cpu_matmul_quant_pack_q8_0_f16(
    const marmot_float16_t *activations, size_t stride_k, size_t stride_n, size_t column_index, size_t K,
    size_t blocks_per_row, void *dst_blocks
);

void cpu_matmul_quant_pack_q8_k_f32(
    const float *activations, size_t stride_k, size_t stride_n, size_t column_index, size_t K, size_t blocks_per_row,
    void *dst_blocks
);

void cpu_matmul_quant_pack_q8_k_f16(
    const marmot_float16_t *activations, size_t stride_k, size_t stride_n, size_t column_index, size_t K,
    size_t blocks_per_row, void *dst_blocks
);

#endif // CPU_MATMUL_QUANT_ACTIVATION_H
