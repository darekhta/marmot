#ifndef CPU_MATMUL_QUANT_INTERNAL_H
#define CPU_MATMUL_QUANT_INTERNAL_H

#include "marmot/quant_block.h"

#include <stdint.h>

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
#define CPU_QUANT_DECODE_PANEL_ROWS CPU_QUANT_MATMUL_MR
#define CPU_QUANT_MATMUL_HINT_OUTPUT_PROJECTION (1u << 0)
#define CPU_QUANT_MATMUL_HINT_PREFER_RAW (1u << 1)

#ifdef __cplusplus
extern "C" {
#endif

void cpu_quant_matmul_dispatch_fp16_tile(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes, size_t row_bytes, size_t blocks_per_row,
    const marmot_float16_t *const *activation_rows, size_t act_stride_k, size_t cols_in_tile, size_t n0,
    float *out_data_f32, marmot_float16_t *out_data_f16, size_t out_stride_m, size_t out_stride_n, size_t row_count,
    size_t K, size_t thread_cap
);

void cpu_quant_matmul_dispatch_fp16_tile_dual(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes_a, const uint8_t *weight_bytes_b,
    size_t row_bytes, size_t blocks_per_row, const marmot_float16_t *const *activation_rows, size_t act_stride_k,
    size_t cols_in_tile, size_t n0, float *out_a_data_f32, marmot_float16_t *out_a_data_f16, float *out_b_data_f32,
    marmot_float16_t *out_b_data_f16, size_t out_stride_m, size_t out_stride_n, size_t row_count, size_t K,
    size_t thread_cap
);

void cpu_quant_matmul_dispatch_quant_tile(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes, size_t row_bytes, size_t blocks_per_row,
    const marmot_q8_0_block_t *activation_q8_0, const marmot_q8_k_block_t *activation_q8_k,
    const marmot_q8_k_block_t *activation_panel_q8_k, size_t activation_panel_cols, size_t cols_in_tile, size_t n0,
    float *out_data_f32, marmot_float16_t *out_data_f16, size_t out_stride_m, size_t out_stride_n, size_t row_start,
    size_t row_end, size_t thread_cap, marmot_quant_kind_t quant_kind
);

void cpu_quant_matmul_dispatch_quant_tile_dual(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes_a, const uint8_t *weight_bytes_b,
    size_t row_bytes, size_t blocks_per_row, const marmot_q8_0_block_t *activation_q8_0,
    const marmot_q8_k_block_t *activation_q8_k, const marmot_q8_k_block_t *activation_panel_q8_k,
    size_t activation_panel_cols, size_t cols_in_tile, size_t n0, float *out_a_data_f32,
    marmot_float16_t *out_a_data_f16, float *out_b_data_f32, marmot_float16_t *out_b_data_f16, size_t out_stride_m,
    size_t out_stride_n, size_t row_start, size_t row_end, size_t thread_cap, marmot_quant_kind_t quant_kind
);

void cpu_quant_matmul_dispatch_quant_tile_row_panel(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes, size_t row_count, size_t row_bytes,
    size_t blocks_per_row, const marmot_q8_k_block_t *activation_q8_k, const marmot_q8_k_block_t *activation_panel_q8_k,
    size_t activation_panel_cols, size_t cols_in_tile, size_t n0, float *out_data_f32, marmot_float16_t *out_data_f16,
    size_t out_stride_m, size_t out_stride_n, size_t panel_rows, marmot_quant_kind_t quant_kind
);

void cpu_quant_matmul_dispatch_quant_tile_row_panel_dual(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes_a, const uint8_t *weight_bytes_b,
    size_t row_count, size_t row_bytes, size_t blocks_per_row, const marmot_q8_k_block_t *activation_q8_k,
    const marmot_q8_k_block_t *activation_panel_q8_k, size_t activation_panel_cols, size_t cols_in_tile, size_t n0,
    float *out_a_data_f32, marmot_float16_t *out_a_data_f16, float *out_b_data_f32, marmot_float16_t *out_b_data_f16,
    size_t out_stride_m, size_t out_stride_n, size_t panel_rows, marmot_quant_kind_t quant_kind
);

void cpu_quant_matmul_dispatch_quant_tile_q4_k_row_panel_decoded(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes, size_t row_count, size_t blocks_per_row,
    const marmot_q8_k_block_t *activation_q8_k, const marmot_q8_k_block_t *activation_panel_q8_k,
    size_t activation_panel_cols, size_t cols_in_tile, size_t n0, float *out_data_f32, marmot_float16_t *out_data_f16,
    size_t out_stride_m, size_t out_stride_n, size_t panel_rows
);

void cpu_quant_matmul_dispatch_quant_tile_q4_k_row_panel_decoded_dual(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes_a, const uint8_t *weight_bytes_b,
    size_t row_count, size_t blocks_per_row, const marmot_q8_k_block_t *activation_q8_k,
    const marmot_q8_k_block_t *activation_panel_q8_k, size_t activation_panel_cols, size_t cols_in_tile, size_t n0,
    float *out_a_data_f32, marmot_float16_t *out_a_data_f16, float *out_b_data_f32, marmot_float16_t *out_b_data_f16,
    size_t out_stride_m, size_t out_stride_n, size_t panel_rows
);

void cpu_quant_matmul_dispatch_quant_tile_q6_k_row_panel_decoded(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes, size_t row_count, size_t blocks_per_row,
    const marmot_q8_k_block_t *activation_q8_k, const marmot_q8_k_block_t *activation_panel_q8_k,
    size_t activation_panel_cols, size_t cols_in_tile, size_t n0, float *out_data_f32, marmot_float16_t *out_data_f16,
    size_t out_stride_m, size_t out_stride_n, size_t panel_rows
);

void cpu_quant_matmul_dispatch_quant_tile_q6_k_row_panel_decoded_dual(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes_a, const uint8_t *weight_bytes_b,
    size_t row_count, size_t blocks_per_row, const marmot_q8_k_block_t *activation_q8_k,
    const marmot_q8_k_block_t *activation_panel_q8_k, size_t activation_panel_cols, size_t cols_in_tile, size_t n0,
    float *out_a_data_f32, marmot_float16_t *out_a_data_f16, float *out_b_data_f32, marmot_float16_t *out_b_data_f16,
    size_t out_stride_m, size_t out_stride_n, size_t panel_rows
);

marmot_error_t cpu_matmul_quantized_dual_output(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight_a,
    const marmot_tensor_t *weight_b, marmot_tensor_t *out_a, marmot_tensor_t *out_b
);

marmot_error_t cpu_matmul_quantized_with_hints(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out, uint32_t hints
);

void cpu_quant_matmul_set_thread_cap_override(size_t thread_cap);

#ifdef __cplusplus
}
#endif

#endif // CPU_MATMUL_QUANT_INTERNAL_H
