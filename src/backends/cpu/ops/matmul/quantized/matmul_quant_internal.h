#ifndef CPU_MATMUL_QUANT_INTERNAL_H
#define CPU_MATMUL_QUANT_INTERNAL_H

#include "marmot/quant_block.h"

#include "cpu_backend_internal.h"
#include "ops/matmul/quantized/matmul_quant_kernels.h"

#define CPU_QUANT_MATMUL_TILE_COLS 16u
#define CPU_QUANT_MATMUL_BLOCK_ROWS 64u
#define CPU_QUANT_KC_BLOCKS 64u
#define CPU_QUANT_PREFETCH_BLOCKS 2u
#if HAS_NEON
#define CPU_QUANT_MATMUL_MR 8u
#define CPU_QUANT_MATMUL_NR 8u
#else
#define CPU_QUANT_MATMUL_MR 8u
#define CPU_QUANT_MATMUL_NR 8u
#endif

void cpu_quant_matmul_dispatch_fp16_tile(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes, size_t row_bytes, size_t blocks_per_row,
    const marmot_float16_t *const *activation_rows, size_t act_stride_k, size_t cols_in_tile, size_t n0,
    float *out_data_f32, marmot_float16_t *out_data_f16, size_t out_stride_m, size_t out_stride_n, size_t row_count,
    size_t K, size_t thread_cap
);

void cpu_quant_matmul_dispatch_quant_tile(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes, size_t row_bytes, size_t blocks_per_row,
    const marmot_q8_0_block_t *activation_q8_0, const marmot_q8_k_block_t *activation_q8_k,
    const marmot_q8_k_block_t *activation_panel_q8_k, size_t activation_panel_cols, size_t cols_in_tile, size_t n0,
    float *out_data_f32, marmot_float16_t *out_data_f16, size_t out_stride_m, size_t out_stride_n, size_t row_start,
    size_t row_end, size_t thread_cap, marmot_quant_kind_t quant_kind
);

#endif // CPU_MATMUL_QUANT_INTERNAL_H
