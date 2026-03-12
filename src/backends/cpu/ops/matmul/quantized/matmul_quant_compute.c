#include "marmot/dispatch.h"
#include "marmot/quant_block.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <string.h>

#include "cpu_backend_internal.h"
#include "ops/matmul/quantized/internal/vec_dot.h"
#include "ops/matmul/quantized/matmul_quant_internal.h"
#include "ops/matmul/quantized/matmul_quant_kernels.h"
#include "quantization/format_metadata.h"

typedef struct {
    const cpu_matmul_quant_kernel_t *kernel;
    const uint8_t *weight_bytes;
    size_t row_bytes;
    size_t blocks_per_row;
    const marmot_q8_0_block_t *activation_tile_q8_0;
    const marmot_q8_k_block_t *activation_tile_q8_k;
    const marmot_q8_k_block_t *activation_panel_q8_k;
    size_t activation_panel_cols;
    size_t activation_panel_col0;
    size_t cols_in_tile;
    size_t n0;
    float *out_data_f32;
    marmot_float16_t *out_data_f16;
    size_t block_bytes;
    size_t out_stride_m;
    size_t out_stride_n;
    size_t row_base;
    size_t total_row_count;
    size_t row_start;
    size_t row_end;
    marmot_quant_kind_t quant_kind;
} cpu_quant_matmul_worker_args_t;

typedef struct {
    const cpu_matmul_quant_kernel_t *kernel;
    const uint8_t *weight_bytes_a;
    const uint8_t *weight_bytes_b;
    size_t row_bytes;
    size_t blocks_per_row;
    const marmot_q8_0_block_t *activation_tile_q8_0;
    const marmot_q8_k_block_t *activation_tile_q8_k;
    const marmot_q8_k_block_t *activation_panel_q8_k;
    size_t activation_panel_cols;
    size_t activation_panel_col0;
    size_t cols_in_tile;
    size_t n0;
    float *out_a_data_f32;
    marmot_float16_t *out_a_data_f16;
    float *out_b_data_f32;
    marmot_float16_t *out_b_data_f16;
    size_t block_bytes;
    size_t out_stride_m;
    size_t out_stride_n;
    size_t row_base;
    size_t total_row_count;
    size_t row_start;
    size_t row_end;
    marmot_quant_kind_t quant_kind;
} cpu_quant_matmul_dual_worker_args_t;

#if MARMOT_ENABLE_NEON && defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
static inline int32_t
cpu_quant_q8_0_block_dotprod(const marmot_q8_0_block_t *w_block, const marmot_q8_0_block_t *a_block) {
    int8x16_t a0 = vld1q_s8(a_block->qs);
    int8x16_t a1 = vld1q_s8(a_block->qs + 16);

    int8x16_t w0 = vld1q_s8(w_block->qs);
    int8x16_t w1 = vld1q_s8(w_block->qs + 16);

    int32x4_t acc = vdotq_s32(vdupq_n_s32(0), w0, a0);
    acc = vdotq_s32(acc, w1, a1);
    return vaddvq_s32(acc);
}

static void cpu_quant_matmul_compute_rows_q8_0_dotprod(const cpu_quant_matmul_worker_args_t *args) {
    const uint8_t *weight_bytes = args->weight_bytes;
    const size_t row_bytes = args->row_bytes;
    const size_t blocks_per_row = args->blocks_per_row;
    const size_t cols_in_tile = args->cols_in_tile;
    const size_t n0 = args->n0;
    float *out_data_f32 = args->out_data_f32;
    marmot_float16_t *out_data_f16 = args->out_data_f16;
    const size_t out_stride_m = args->out_stride_m;
    const size_t out_stride_n = args->out_stride_n;
    const size_t row_base = args->row_base;

    for (size_t c = 0; c < cols_in_tile; ++c) {
        const marmot_q8_0_block_t *act_blocks = args->activation_tile_q8_0 + c * blocks_per_row;
        float *out_col_f32 = out_data_f32 != nullptr ? out_data_f32 + (n0 + c) * out_stride_m : nullptr;
        marmot_float16_t *out_col_f16 = out_data_f16 != nullptr ? out_data_f16 + (n0 + c) * out_stride_m : nullptr;

        size_t m = args->row_start;
        for (; m + CPU_QUANT_MATMUL_MR <= args->row_end; m += CPU_QUANT_MATMUL_MR) {
            float acc[CPU_QUANT_MATMUL_MR] = {0};
            const marmot_q8_0_block_t *w_rows[CPU_QUANT_MATMUL_MR];

            for (size_t r = 0; r < CPU_QUANT_MATMUL_MR; ++r) {
                w_rows[r] = (const marmot_q8_0_block_t *)(weight_bytes + (m + r - row_base) * row_bytes);
            }

            for (size_t b = 0; b < blocks_per_row; ++b) {
                const marmot_q8_0_block_t *a_block = act_blocks + b;
                const float act_scale = (float)marmot_float16_to_native(a_block->scale);

                if (b + CPU_QUANT_PREFETCH_BLOCKS < blocks_per_row) {
                    MARMOT_PREFETCH(act_blocks + b + CPU_QUANT_PREFETCH_BLOCKS);
                    for (size_t r = 0; r < CPU_QUANT_MATMUL_MR; ++r) {
                        MARMOT_PREFETCH(w_rows[r] + b + CPU_QUANT_PREFETCH_BLOCKS);
                    }
                }

                for (size_t r = 0; r < CPU_QUANT_MATMUL_MR; ++r) {
                    const int32_t d = cpu_quant_q8_0_block_dotprod(w_rows[r] + b, a_block);
                    const float w_scale = (float)marmot_float16_to_native((w_rows[r] + b)->scale);
                    acc[r] += (float)d * (w_scale * act_scale);
                }
            }

            for (size_t r = 0; r < CPU_QUANT_MATMUL_MR; ++r) {
                const size_t row_idx = m + r;
                if (out_col_f32 != nullptr) {
                    out_col_f32[row_idx * out_stride_n] = acc[r];
                }
                if (out_col_f16 != nullptr) {
                    out_col_f16[row_idx * out_stride_n] = marmot_native_to_float16((_Float16)acc[r]);
                }
            }
        }

        for (; m < args->row_end; ++m) {
            const uint8_t *row_ptr = weight_bytes + (m - row_base) * row_bytes;
            const float dot = args->kernel->ops.dot_q8_0(row_ptr, act_blocks, blocks_per_row);
            if (out_col_f32 != nullptr) {
                out_col_f32[m * out_stride_n] = dot;
            }
            if (out_col_f16 != nullptr) {
                out_col_f16[m * out_stride_n] = marmot_native_to_float16((_Float16)dot);
            }
        }
    }
}
#endif

#if MARMOT_ENABLE_NEON && defined(__aarch64__) && (defined(__ARM_FEATURE_I8MM) || defined(__ARM_FEATURE_MATMUL_INT8))
static inline int32_t
cpu_quant_q8_0_block_i8mm(const marmot_q8_0_block_t *w_block, const marmot_q8_0_block_t *a_block) {
    int8x16_t a0 = vld1q_s8(a_block->qs);
    int8x16_t a1 = vld1q_s8(a_block->qs + 16);

    int8x16_t w0 = vld1q_s8(w_block->qs);
    int8x16_t w1 = vld1q_s8(w_block->qs + 16);

    int32x4_t acc = vdupq_n_s32(0);
    acc = vmmlaq_s32(acc, w0, a0);
    acc = vmmlaq_s32(acc, w1, a1);
    return vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 3);
}

static void cpu_quant_matmul_compute_rows_q8_0_i8mm(const cpu_quant_matmul_worker_args_t *args) {
    const uint8_t *weight_bytes = args->weight_bytes;
    const size_t row_bytes = args->row_bytes;
    const size_t blocks_per_row = args->blocks_per_row;
    const size_t cols_in_tile = args->cols_in_tile;
    const size_t n0 = args->n0;
    float *out_data_f32 = args->out_data_f32;
    marmot_float16_t *out_data_f16 = args->out_data_f16;
    const size_t out_stride_m = args->out_stride_m;
    const size_t out_stride_n = args->out_stride_n;
    const size_t row_base = args->row_base;

    for (size_t c = 0; c < cols_in_tile; ++c) {
        const marmot_q8_0_block_t *act_blocks = args->activation_tile_q8_0 + c * blocks_per_row;
        float *out_col_f32 = out_data_f32 != nullptr ? out_data_f32 + (n0 + c) * out_stride_m : nullptr;
        marmot_float16_t *out_col_f16 = out_data_f16 != nullptr ? out_data_f16 + (n0 + c) * out_stride_m : nullptr;

        size_t m = args->row_start;
        for (; m + CPU_QUANT_MATMUL_MR <= args->row_end; m += CPU_QUANT_MATMUL_MR) {
            float acc[CPU_QUANT_MATMUL_MR] = {0};
            const marmot_q8_0_block_t *w_rows[CPU_QUANT_MATMUL_MR];

            for (size_t r = 0; r < CPU_QUANT_MATMUL_MR; ++r) {
                w_rows[r] = (const marmot_q8_0_block_t *)(weight_bytes + (m + r - row_base) * row_bytes);
            }

            for (size_t b = 0; b < blocks_per_row; ++b) {
                const marmot_q8_0_block_t *a_block = act_blocks + b;
                const float act_scale = (float)marmot_float16_to_native(a_block->scale);

                if (b + CPU_QUANT_PREFETCH_BLOCKS < blocks_per_row) {
                    MARMOT_PREFETCH(act_blocks + b + CPU_QUANT_PREFETCH_BLOCKS);
                    for (size_t r = 0; r < CPU_QUANT_MATMUL_MR; ++r) {
                        MARMOT_PREFETCH(w_rows[r] + b + CPU_QUANT_PREFETCH_BLOCKS);
                    }
                }

                for (size_t r = 0; r < CPU_QUANT_MATMUL_MR; ++r) {
                    const int32_t d = cpu_quant_q8_0_block_i8mm(w_rows[r] + b, a_block);
                    const float w_scale = (float)marmot_float16_to_native((w_rows[r] + b)->scale);
                    acc[r] += (float)d * (w_scale * act_scale);
                }
            }

            for (size_t r = 0; r < CPU_QUANT_MATMUL_MR; ++r) {
                const size_t row_idx = m + r;
                if (out_col_f32 != nullptr) {
                    out_col_f32[row_idx * out_stride_n] = acc[r];
                }
                if (out_col_f16 != nullptr) {
                    out_col_f16[row_idx * out_stride_n] = marmot_native_to_float16((_Float16)acc[r]);
                }
            }
        }

        for (; m < args->row_end; ++m) {
            const uint8_t *row_ptr = weight_bytes + (m - row_base) * row_bytes;
            const float dot = args->kernel->ops.dot_q8_0(row_ptr, act_blocks, blocks_per_row);
            if (out_col_f32 != nullptr) {
                out_col_f32[m * out_stride_n] = dot;
            }
            if (out_col_f16 != nullptr) {
                out_col_f16[m * out_stride_n] = marmot_native_to_float16((_Float16)dot);
            }
        }
    }
}
#endif

#if MARMOT_ENABLE_NEON && defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
static void cpu_quant_matmul_compute_rows_q8_0_dotprod_blocked(const cpu_quant_matmul_worker_args_t *args) {
    const uint8_t *weight_bytes = args->weight_bytes;
    const size_t row_bytes = args->row_bytes;
    const size_t blocks_per_row = args->blocks_per_row;
    const size_t cols_in_tile = args->cols_in_tile;
    const size_t n0 = args->n0;
    float *out_data_f32 = args->out_data_f32;
    marmot_float16_t *out_data_f16 = args->out_data_f16;
    const size_t block_bytes = args->block_bytes;
    const size_t out_stride_m = args->out_stride_m;
    const size_t out_stride_n = args->out_stride_n;
    const size_t row_base = args->row_base;

    for (size_t c0 = 0; c0 < cols_in_tile; c0 += CPU_QUANT_MATMUL_NR) {
        const size_t col_block = (c0 + CPU_QUANT_MATMUL_NR <= cols_in_tile) ? CPU_QUANT_MATMUL_NR : (cols_in_tile - c0);
        size_t m = args->row_start;
        for (; m + CPU_QUANT_MATMUL_MR <= args->row_end; m += CPU_QUANT_MATMUL_MR) {
            float acc[CPU_QUANT_MATMUL_MR][CPU_QUANT_MATMUL_NR] = {{0}};

            for (size_t kb = 0; kb < blocks_per_row; kb += CPU_QUANT_KC_BLOCKS) {
                const size_t chunk =
                    (kb + CPU_QUANT_KC_BLOCKS <= blocks_per_row) ? CPU_QUANT_KC_BLOCKS : (blocks_per_row - kb);
                float act_scales[CPU_QUANT_MATMUL_NR][CPU_QUANT_KC_BLOCKS] = {{0}};
                for (size_t c = 0; c < col_block; ++c) {
                    const marmot_q8_0_block_t *a_base = args->activation_tile_q8_0 + (c0 + c) * blocks_per_row + kb;
                    for (size_t b = 0; b < chunk; ++b) {
                        if (b + CPU_QUANT_PREFETCH_BLOCKS < chunk) {
                            MARMOT_PREFETCH(a_base + b + CPU_QUANT_PREFETCH_BLOCKS);
                        }
                        act_scales[c][b] = (float)marmot_float16_to_native((a_base + b)->scale);
                    }
                }

                float w_scales[CPU_QUANT_MATMUL_MR][CPU_QUANT_KC_BLOCKS] = {{0}};
                for (size_t r = 0; r < CPU_QUANT_MATMUL_MR; ++r) {
                    const size_t row_offset = m + r - row_base;
                    const marmot_q8_0_block_t *w_row =
                        (const marmot_q8_0_block_t *)(weight_bytes + row_offset * row_bytes + kb * block_bytes);
                    for (size_t b = 0; b < chunk; ++b) {
                        if (b + CPU_QUANT_PREFETCH_BLOCKS < chunk) {
                            MARMOT_PREFETCH(w_row + b + CPU_QUANT_PREFETCH_BLOCKS);
                        }
                        w_scales[r][b] = (float)marmot_float16_to_native((w_row + b)->scale);
                    }
                }

                for (size_t r = 0; r < CPU_QUANT_MATMUL_MR; ++r) {
                    const size_t row_offset = m + r - row_base;
                    const marmot_q8_0_block_t *w_row =
                        (const marmot_q8_0_block_t *)(weight_bytes + row_offset * row_bytes + kb * block_bytes);
                    for (size_t c = 0; c < col_block; ++c) {
                        const marmot_q8_0_block_t *a_base = args->activation_tile_q8_0 + (c0 + c) * blocks_per_row + kb;
                        for (size_t b = 0; b < chunk; ++b) {
                            const int32_t dot = cpu_quant_q8_0_block_dotprod(w_row + b, a_base + b);
                            acc[r][c] += (float)dot * (w_scales[r][b] * act_scales[c][b]);
                        }
                    }
                }
            }

            for (size_t c = 0; c < col_block; ++c) {
                float *out_col_f32 = out_data_f32 != nullptr ? out_data_f32 + (n0 + c0 + c) * out_stride_m : nullptr;
                marmot_float16_t *out_col_f16 =
                    out_data_f16 != nullptr ? out_data_f16 + (n0 + c0 + c) * out_stride_m : nullptr;
                for (size_t r = 0; r < CPU_QUANT_MATMUL_MR; ++r) {
                    const size_t row_idx = m + r;
                    if (out_col_f32 != nullptr) {
                        out_col_f32[row_idx * out_stride_n] = acc[r][c];
                    }
                    if (out_col_f16 != nullptr) {
                        out_col_f16[row_idx * out_stride_n] = marmot_native_to_float16((_Float16)acc[r][c]);
                    }
                }
            }
        }

        if (m < args->row_end) {
            cpu_quant_matmul_worker_args_t tail_args = *args;
            tail_args.row_start = m;
            tail_args.cols_in_tile = col_block;
            tail_args.n0 = n0 + c0;
            tail_args.activation_tile_q8_0 = args->activation_tile_q8_0 + c0 * blocks_per_row;
            cpu_quant_matmul_compute_rows_q8_0_dotprod(&tail_args);
        }
    }
}
#endif

#if MARMOT_ENABLE_NEON && defined(__aarch64__) && (defined(__ARM_FEATURE_I8MM) || defined(__ARM_FEATURE_MATMUL_INT8))
static void cpu_quant_matmul_compute_rows_q8_0_i8mm_blocked(const cpu_quant_matmul_worker_args_t *args) {
    const uint8_t *weight_bytes = args->weight_bytes;
    const size_t row_bytes = args->row_bytes;
    const size_t blocks_per_row = args->blocks_per_row;
    const size_t cols_in_tile = args->cols_in_tile;
    const size_t n0 = args->n0;
    float *out_data_f32 = args->out_data_f32;
    marmot_float16_t *out_data_f16 = args->out_data_f16;
    const size_t block_bytes = args->block_bytes;
    const size_t out_stride_m = args->out_stride_m;
    const size_t out_stride_n = args->out_stride_n;
    const size_t row_base = args->row_base;

    for (size_t c0 = 0; c0 < cols_in_tile; c0 += CPU_QUANT_MATMUL_NR) {
        const size_t col_block = (c0 + CPU_QUANT_MATMUL_NR <= cols_in_tile) ? CPU_QUANT_MATMUL_NR : (cols_in_tile - c0);
        size_t m = args->row_start;
        for (; m + CPU_QUANT_MATMUL_MR <= args->row_end; m += CPU_QUANT_MATMUL_MR) {
            float acc[CPU_QUANT_MATMUL_MR][CPU_QUANT_MATMUL_NR] = {{0}};

            for (size_t kb = 0; kb < blocks_per_row; kb += CPU_QUANT_KC_BLOCKS) {
                const size_t chunk =
                    (kb + CPU_QUANT_KC_BLOCKS <= blocks_per_row) ? CPU_QUANT_KC_BLOCKS : (blocks_per_row - kb);
                float act_scales[CPU_QUANT_MATMUL_NR][CPU_QUANT_KC_BLOCKS] = {{0}};
                for (size_t c = 0; c < col_block; ++c) {
                    const marmot_q8_0_block_t *a_base = args->activation_tile_q8_0 + (c0 + c) * blocks_per_row + kb;
                    for (size_t b = 0; b < chunk; ++b) {
                        if (b + CPU_QUANT_PREFETCH_BLOCKS < chunk) {
                            MARMOT_PREFETCH(a_base + b + CPU_QUANT_PREFETCH_BLOCKS);
                        }
                        act_scales[c][b] = (float)marmot_float16_to_native((a_base + b)->scale);
                    }
                }

                float w_scales[CPU_QUANT_MATMUL_MR][CPU_QUANT_KC_BLOCKS] = {{0}};
                for (size_t r = 0; r < CPU_QUANT_MATMUL_MR; ++r) {
                    const size_t row_offset = m + r - row_base;
                    const marmot_q8_0_block_t *w_row =
                        (const marmot_q8_0_block_t *)(weight_bytes + row_offset * row_bytes + kb * block_bytes);
                    for (size_t b = 0; b < chunk; ++b) {
                        if (b + CPU_QUANT_PREFETCH_BLOCKS < chunk) {
                            MARMOT_PREFETCH(w_row + b + CPU_QUANT_PREFETCH_BLOCKS);
                        }
                        w_scales[r][b] = (float)marmot_float16_to_native((w_row + b)->scale);
                    }
                }

                for (size_t r = 0; r < CPU_QUANT_MATMUL_MR; ++r) {
                    const size_t row_offset = m + r - row_base;
                    const marmot_q8_0_block_t *w_row =
                        (const marmot_q8_0_block_t *)(weight_bytes + row_offset * row_bytes + kb * block_bytes);
                    for (size_t c = 0; c < col_block; ++c) {
                        const marmot_q8_0_block_t *a_base = args->activation_tile_q8_0 + (c0 + c) * blocks_per_row + kb;
                        for (size_t b = 0; b < chunk; ++b) {
                            const int32_t dot = cpu_quant_q8_0_block_i8mm(w_row + b, a_base + b);
                            acc[r][c] += (float)dot * (w_scales[r][b] * act_scales[c][b]);
                        }
                    }
                }
            }

            for (size_t c = 0; c < col_block; ++c) {
                float *out_col_f32 = out_data_f32 != nullptr ? out_data_f32 + (n0 + c0 + c) * out_stride_m : nullptr;
                marmot_float16_t *out_col_f16 =
                    out_data_f16 != nullptr ? out_data_f16 + (n0 + c0 + c) * out_stride_m : nullptr;
                for (size_t r = 0; r < CPU_QUANT_MATMUL_MR; ++r) {
                    const size_t row_idx = m + r;
                    if (out_col_f32 != nullptr) {
                        out_col_f32[row_idx * out_stride_n] = acc[r][c];
                    }
                    if (out_col_f16 != nullptr) {
                        out_col_f16[row_idx * out_stride_n] = marmot_native_to_float16((_Float16)acc[r][c]);
                    }
                }
            }
        }

        if (m < args->row_end) {
            cpu_quant_matmul_worker_args_t tail_args = *args;
            tail_args.row_start = m;
            tail_args.cols_in_tile = col_block;
            tail_args.n0 = n0 + c0;
            tail_args.activation_tile_q8_0 = args->activation_tile_q8_0 + c0 * blocks_per_row;
            cpu_quant_matmul_compute_rows_q8_0_i8mm(&tail_args);
        }
    }
}
#endif

static void cpu_quant_matmul_compute_rows_q8_0_blocked_generic(const cpu_quant_matmul_worker_args_t *args) {
    const cpu_matmul_quant_kernel_t *kernel = args->kernel;
    const uint8_t *weight_bytes = args->weight_bytes;
    const size_t row_bytes = args->row_bytes;
    const size_t blocks_per_row = args->blocks_per_row;
    const size_t cols_in_tile = args->cols_in_tile;
    const size_t n0 = args->n0;
    float *out_data_f32 = args->out_data_f32;
    marmot_float16_t *out_data_f16 = args->out_data_f16;
    const size_t block_bytes = args->block_bytes;
    const size_t out_stride_m = args->out_stride_m;
    const size_t out_stride_n = args->out_stride_n;
    const size_t row_base = args->row_base;

    for (size_t c0 = 0; c0 < cols_in_tile; c0 += CPU_QUANT_MATMUL_NR) {
        const size_t col_block = (c0 + CPU_QUANT_MATMUL_NR <= cols_in_tile) ? CPU_QUANT_MATMUL_NR : (cols_in_tile - c0);
        size_t m = args->row_start;
        while (m < args->row_end) {
            const size_t rows_this =
                (m + CPU_QUANT_MATMUL_MR <= args->row_end) ? CPU_QUANT_MATMUL_MR : (args->row_end - m);
            float acc[CPU_QUANT_MATMUL_MR][CPU_QUANT_MATMUL_NR] = {{0}};

            for (size_t r = 0; r < rows_this; ++r) {
                const size_t row_offset = m + r - row_base;
                const uint8_t *row_ptr = weight_bytes + row_offset * row_bytes;
                for (size_t c = 0; c < col_block; ++c) {
                    const marmot_q8_0_block_t *a_block = args->activation_tile_q8_0 + (c0 + c) * blocks_per_row;
                    for (size_t kb = 0; kb < blocks_per_row; kb += CPU_QUANT_KC_BLOCKS) {
                        const size_t chunk =
                            (kb + CPU_QUANT_KC_BLOCKS <= blocks_per_row) ? CPU_QUANT_KC_BLOCKS : (blocks_per_row - kb);
                        const uint8_t *row_block_ptr = row_ptr + kb * block_bytes;
                        acc[r][c] += kernel->ops.dot_q8_0(row_block_ptr, a_block + kb, chunk);
                    }
                }
            }

            for (size_t c = 0; c < col_block; ++c) {
                float *out_col_f32 = out_data_f32 != nullptr ? out_data_f32 + (n0 + c0 + c) * out_stride_m : nullptr;
                marmot_float16_t *out_col_f16 =
                    out_data_f16 != nullptr ? out_data_f16 + (n0 + c0 + c) * out_stride_m : nullptr;
                for (size_t r = 0; r < rows_this; ++r) {
                    const size_t row_idx = m + r;
                    if (out_col_f32 != nullptr) {
                        out_col_f32[row_idx * out_stride_n] = acc[r][c];
                    }
                    if (out_col_f16 != nullptr) {
                        out_col_f16[row_idx * out_stride_n] = marmot_native_to_float16((_Float16)acc[r][c]);
                    }
                }
            }

            m += rows_this;
        }
    }
}

#if MARMOT_ENABLE_NEON && defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
typedef struct {
    float d;
    float dmin;
    uint8_t scales[8];
    uint8_t mins[8];
    uint8_t qs[MARMOT_QK_K_QS_BYTES];
} cpu_q4_k_row_panel_decoded_block_t;

static_assert(sizeof(cpu_q4_k_row_panel_decoded_block_t) == 152, "decoded Q4_K row-panel block size mismatch");

typedef struct {
    float d;
    int8_t scales[MARMOT_QK_K_VALUES / 16];
    int8_t qs[MARMOT_QK_K_VALUES];
} cpu_q6_k_row_panel_decoded_block_t;

static_assert(sizeof(cpu_q6_k_row_panel_decoded_block_t) == 276, "decoded Q6_K row-panel block size mismatch");

static inline void cpu_quant_q4_k_decode_block(
    const marmot_q4_k_block_t *src, uint8_t *scales_out, uint8_t *mins_out, float *d_out, float *dmin_out
) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];
    memcpy(utmp, src->scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    memcpy(scales_out, &utmp[0], 8);
    memcpy(mins_out, &utmp[2], 8);

    *d_out = (float)marmot_float16_to_native(src->d);
    *dmin_out = (float)marmot_float16_to_native(src->dmin);
}

static inline int32_t cpu_quant_q4_k_compute_sumi(const int16_t *bsums, const uint8_t *mins) {
    int32_t sumi = 0;
    for (size_t j = 0; j < MARMOT_QK_K_VALUES / 16; ++j) {
        sumi += (int32_t)bsums[j] * (int32_t)mins[j / 2];
    }
    return sumi;
}

static inline int32_t cpu_quant_q4_k_block_dotprod_qs(const uint8_t *q4, const int8_t *q8, const uint8_t *scales) {
    int32x4_t block_acc = vdupq_n_s32(0);
    const uint8x16_t mask = vdupq_n_u8(0x0F);

    for (size_t segment = 0; segment < 4; ++segment) {
        const uint8_t *q4_block = q4 + segment * 32;
        const int8_t *q8_block = q8 + segment * 64;

        uint8x16_t bytes0 = vld1q_u8(q4_block);
        uint8x16_t bytes1 = vld1q_u8(q4_block + 16);

        uint8x16_t low0 = vandq_u8(bytes0, mask);
        uint8x16_t low1 = vandq_u8(bytes1, mask);
        int32x4_t acc = vdupq_n_s32(0);
        int8x16_t w0 = vreinterpretq_s8_u8(low0);
        int8x16_t w1 = vreinterpretq_s8_u8(low1);
        int8x16_t a0 = vld1q_s8(q8_block + 0);
        int8x16_t a1 = vld1q_s8(q8_block + 16);

        acc = vdotq_s32(acc, w0, a0);
        acc = vdotq_s32(acc, w1, a1);

        block_acc = vaddq_s32(block_acc, vmulq_n_s32(acc, (int32_t)scales[segment * 2]));

        uint8x16_t high0 = vshrq_n_u8(bytes0, 4);
        uint8x16_t high1 = vshrq_n_u8(bytes1, 4);
        int32x4_t acc_hi = vdupq_n_s32(0);
        int8x16_t w2 = vreinterpretq_s8_u8(high0);
        int8x16_t w3 = vreinterpretq_s8_u8(high1);
        int8x16_t a2 = vld1q_s8(q8_block + 32);
        int8x16_t a3 = vld1q_s8(q8_block + 48);

        acc_hi = vdotq_s32(acc_hi, w2, a2);
        acc_hi = vdotq_s32(acc_hi, w3, a3);

        block_acc = vaddq_s32(block_acc, vmulq_n_s32(acc_hi, (int32_t)scales[segment * 2 + 1]));
    }
    return vaddvq_s32(block_acc);
}

static inline int32_t
cpu_quant_q4_k_block_dotprod(const marmot_q4_k_block_t *w_block, const int8_t *q8, const uint8_t *scales) {
    return cpu_quant_q4_k_block_dotprod_qs(w_block->qs, q8, scales);
}

static inline uint8x16_t cpu_quant_q6_k_qh_bits(uint8x16_t qh, int shift) {
    const uint8x16_t mask = vdupq_n_u8(0x03);
    switch (shift) {
    case 0:
        return vandq_u8(qh, mask);
    case 2:
        return vandq_u8(vshrq_n_u8(qh, 2), mask);
    case 4:
        return vandq_u8(vshrq_n_u8(qh, 4), mask);
    default:
        return vandq_u8(vshrq_n_u8(qh, 6), mask);
    }
}

static inline int8x16_t cpu_quant_q6_k_decode_16(const uint8_t *ql, const uint8_t *qh, int shift, bool high_nibble) {
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    uint8x16_t ql_bytes = vld1q_u8(ql);
    uint8x16_t qh_bytes = vld1q_u8(qh);
    uint8x16_t ql_vals = high_nibble ? vshrq_n_u8(ql_bytes, 4) : vandq_u8(ql_bytes, mask);
    uint8x16_t qh_vals = cpu_quant_q6_k_qh_bits(qh_bytes, shift);
    uint8x16_t merged = vorrq_u8(ql_vals, vshlq_n_u8(qh_vals, 4));
    int8x16_t signed_vals = vreinterpretq_s8_u8(merged);
    return vsubq_s8(signed_vals, vdupq_n_s8(32));
}

static inline int32_t cpu_quant_q6_k_block_dotprod(const marmot_q6_k_block_t *w_block, const int8_t *q8) {
    int32x4_t block_acc = vdupq_n_s32(0);
    const int8_t *a_ptr = q8;
    for (size_t sg = 0; sg < MARMOT_QK_K_VALUES / 16; ++sg) {
        const size_t group32 = sg / 2;
        const size_t half = group32 / 4;
        const size_t group_in_half = group32 & 3;
        const size_t part = sg & 1;
        const uint8_t *ql = w_block->ql + half * 64 + ((group_in_half & 1) ? 32 : 0) + (part * 16);
        const uint8_t *qh = w_block->qh + half * 32 + (part * 16);
        const int shift = (int)(group_in_half * 2);
        const bool high_nibble = group_in_half >= 2;
        const int8_t scale = w_block->scales[sg];
        int32x4_t acc = vdupq_n_s32(0);
        int8x16_t wv = cpu_quant_q6_k_decode_16(ql, qh, shift, high_nibble);
        int8x16_t av = vld1q_s8(a_ptr);
        acc = vdotq_s32(acc, wv, av);
        block_acc = vaddq_s32(block_acc, vmulq_n_s32(acc, (int32_t)scale));
        a_ptr += 16;
    }
    return vaddvq_s32(block_acc);
}

static inline int32_t
cpu_quant_q6_k_decoded_block_dotprod(const cpu_q6_k_row_panel_decoded_block_t *w_block, const int8_t *q8) {
    int32_t total = 0;
    for (size_t sg = 0; sg < MARMOT_QK_K_VALUES / 16; ++sg) {
        const int8x16_t wv = vld1q_s8(w_block->qs + sg * 16);
        const int8x16_t av = vld1q_s8(q8 + sg * 16);
        const int32x4_t acc = vdotq_s32(vdupq_n_s32(0), wv, av);
        total += vaddvq_s32(acc) * (int32_t)w_block->scales[sg];
    }
    return total;
}

static inline bool cpu_quant_matmul_should_use_small_q8_k_vecdot(
    marmot_quant_kind_t quant_kind, size_t col_block, size_t total_row_count
) {
    if (quant_kind != MARMOT_QUANT_KIND_Q4_K && quant_kind != MARMOT_QUANT_KIND_Q6_K) {
        return false;
    }
    if (col_block > 2) {
        return false;
    }
    if (col_block == 1 && total_row_count >= 8192 && quant_kind != MARMOT_QUANT_KIND_Q6_K) {
        return false;
    }
    return true;
}

static void cpu_quant_matmul_compute_rows_q8_k_blocked(const cpu_quant_matmul_worker_args_t *args) {
    const uint8_t *weight_bytes = args->weight_bytes;
    const size_t row_bytes = args->row_bytes;
    const size_t blocks_per_row = args->blocks_per_row;
    const size_t cols_in_tile = args->cols_in_tile;
    const size_t n0 = args->n0;
    float *out_data_f32 = args->out_data_f32;
    marmot_float16_t *out_data_f16 = args->out_data_f16;
    const size_t block_bytes = args->block_bytes;
    const size_t out_stride_m = args->out_stride_m;
    const size_t out_stride_n = args->out_stride_n;
    const size_t row_base = args->row_base;
    const marmot_q8_k_block_t *activation_panel = args->activation_panel_q8_k;
    const size_t activation_panel_cols = args->activation_panel_cols;
    const size_t activation_panel_col0 = args->activation_panel_col0;

    for (size_t c0 = 0; c0 < cols_in_tile; c0 += CPU_QUANT_MATMUL_NR) {
        const size_t col_block = (c0 + CPU_QUANT_MATMUL_NR <= cols_in_tile) ? CPU_QUANT_MATMUL_NR : (cols_in_tile - c0);
        const bool use_small_q8_k_pair_vecdot = col_block == 1 && args->kernel->ops.dot_q8_k_2rows != nullptr;
        size_t m = args->row_start;
        while (m < args->row_end) {
            const size_t rows_this =
                (m + CPU_QUANT_MATMUL_MR <= args->row_end) ? CPU_QUANT_MATMUL_MR : (args->row_end - m);
            float acc[CPU_QUANT_MATMUL_MR][CPU_QUANT_MATMUL_NR] = {{0}};

            for (size_t kb = 0; kb < blocks_per_row; kb += CPU_QUANT_KC_BLOCKS) {
                const size_t chunk =
                    (kb + CPU_QUANT_KC_BLOCKS <= blocks_per_row) ? CPU_QUANT_KC_BLOCKS : (blocks_per_row - kb);
                if (use_small_q8_k_pair_vecdot ||
                    cpu_quant_matmul_should_use_small_q8_k_vecdot(args->quant_kind, col_block, args->total_row_count)) {
                    if (use_small_q8_k_pair_vecdot) {
                        const size_t col_index = activation_panel_col0 + c0;
                        const marmot_q8_k_block_t *a_block = activation_panel != nullptr
                            ? activation_panel + kb * activation_panel_cols + col_index
                            : args->activation_tile_q8_k + c0 * blocks_per_row + kb;
                        size_t r = 0;
                        for (; r + 1 < rows_this; r += 2) {
                            const size_t row_offset0 = m + r - row_base;
                            const size_t row_offset1 = m + r + 1 - row_base;
                            const uint8_t *row_ptr0 = weight_bytes + row_offset0 * row_bytes + kb * block_bytes;
                            const uint8_t *row_ptr1 = weight_bytes + row_offset1 * row_bytes + kb * block_bytes;
                            float sum0 = 0.0f;
                            float sum1 = 0.0f;
                            args->kernel->ops.dot_q8_k_2rows(row_ptr0, row_ptr1, a_block, chunk, &sum0, &sum1);
                            acc[r][0] += sum0;
                            acc[r + 1][0] += sum1;
                        }
                        for (; r < rows_this; ++r) {
                            const size_t row_offset = m + r - row_base;
                            const uint8_t *row_ptr = weight_bytes + row_offset * row_bytes + kb * block_bytes;
                            acc[r][0] += args->kernel->ops.dot_q8_k(row_ptr, a_block, chunk);
                        }
                        continue;
                    }
                    for (size_t r = 0; r < rows_this; ++r) {
                        const size_t row_offset = m + r - row_base;
                        const uint8_t *row_ptr = weight_bytes + row_offset * row_bytes + kb * block_bytes;
                        for (size_t c = 0; c < col_block; ++c) {
                            const size_t col_index = activation_panel_col0 + c0 + c;
                            const marmot_q8_k_block_t *a_block = activation_panel != nullptr
                                ? activation_panel + kb * activation_panel_cols + col_index
                                : args->activation_tile_q8_k + (c0 + c) * blocks_per_row + kb;
                            acc[r][c] += args->kernel->ops.dot_q8_k(row_ptr, a_block, chunk);
                        }
                    }
                    continue;
                }
                if (args->quant_kind == MARMOT_QUANT_KIND_Q4_K) {
                    for (size_t b = 0; b < chunk; ++b) {
                        uint8_t scales[CPU_QUANT_MATMUL_MR][8];
                        uint8_t mins[CPU_QUANT_MATMUL_MR][8];
                        float w_d[CPU_QUANT_MATMUL_MR];
                        float w_dmin[CPU_QUANT_MATMUL_MR];
                        const marmot_q4_k_block_t *w_blocks[CPU_QUANT_MATMUL_MR];

                        for (size_t r = 0; r < rows_this; ++r) {
                            const size_t row_offset = m + r - row_base;
                            if (b + CPU_QUANT_PREFETCH_BLOCKS < chunk) {
                                const size_t prefetch_offset = (kb + b + CPU_QUANT_PREFETCH_BLOCKS) * block_bytes;
                                MARMOT_PREFETCH(weight_bytes + row_offset * row_bytes + prefetch_offset);
                            }
                            const marmot_q4_k_block_t *w_block =
                                (const marmot_q4_k_block_t *)(weight_bytes + row_offset * row_bytes +
                                                              (kb + b) * block_bytes);
                            w_blocks[r] = w_block;
                            cpu_quant_q4_k_decode_block(w_block, scales[r], mins[r], &w_d[r], &w_dmin[r]);
                        }

                        for (size_t c = 0; c < col_block; ++c) {
                            const size_t col_index = activation_panel_col0 + c0 + c;
                            if (b + CPU_QUANT_PREFETCH_BLOCKS < chunk) {
                                const size_t prefetch_b = kb + b + CPU_QUANT_PREFETCH_BLOCKS;
                                const marmot_q8_k_block_t *prefetch_block = activation_panel != nullptr
                                    ? activation_panel + prefetch_b * activation_panel_cols + col_index
                                    : args->activation_tile_q8_k + (c0 + c) * blocks_per_row + prefetch_b;
                                MARMOT_PREFETCH(prefetch_block);
                            }
                            const marmot_q8_k_block_t *a_block = activation_panel != nullptr
                                ? activation_panel + (kb + b) * activation_panel_cols + col_index
                                : args->activation_tile_q8_k + (c0 + c) * blocks_per_row + kb + b;
                            const float act_d = a_block->d;
                            const int16_t *bsums = a_block->bsums;

                            for (size_t r = 0; r < rows_this; ++r) {
                                const int32_t block_sum =
                                    cpu_quant_q4_k_block_dotprod(w_blocks[r], a_block->qs, scales[r]);
                                const int32_t sumi = cpu_quant_q4_k_compute_sumi(bsums, mins[r]);
                                const float d = act_d * w_d[r];
                                const float dmin = act_d * w_dmin[r];
                                acc[r][c] += d * (float)block_sum - dmin * (float)sumi;
                            }
                        }
                    }
                    continue;
                }

                if (args->quant_kind == MARMOT_QUANT_KIND_Q6_K) {
                    for (size_t b = 0; b < chunk; ++b) {
                        float w_d[CPU_QUANT_MATMUL_MR];
                        const marmot_q6_k_block_t *w_blocks[CPU_QUANT_MATMUL_MR];

                        for (size_t r = 0; r < rows_this; ++r) {
                            const size_t row_offset = m + r - row_base;
                            if (b + CPU_QUANT_PREFETCH_BLOCKS < chunk) {
                                const size_t prefetch_offset = (kb + b + CPU_QUANT_PREFETCH_BLOCKS) * block_bytes;
                                MARMOT_PREFETCH(weight_bytes + row_offset * row_bytes + prefetch_offset);
                            }
                            const marmot_q6_k_block_t *w_block =
                                (const marmot_q6_k_block_t *)(weight_bytes + row_offset * row_bytes +
                                                              (kb + b) * block_bytes);
                            w_blocks[r] = w_block;
                            w_d[r] = (float)marmot_float16_to_native(w_block->d);
                        }

                        for (size_t c = 0; c < col_block; ++c) {
                            const size_t col_index = activation_panel_col0 + c0 + c;
                            if (b + CPU_QUANT_PREFETCH_BLOCKS < chunk) {
                                const size_t prefetch_b = kb + b + CPU_QUANT_PREFETCH_BLOCKS;
                                const marmot_q8_k_block_t *prefetch_block = activation_panel != nullptr
                                    ? activation_panel + prefetch_b * activation_panel_cols + col_index
                                    : args->activation_tile_q8_k + (c0 + c) * blocks_per_row + prefetch_b;
                                MARMOT_PREFETCH(prefetch_block);
                            }
                            const marmot_q8_k_block_t *a_block = activation_panel != nullptr
                                ? activation_panel + (kb + b) * activation_panel_cols + col_index
                                : args->activation_tile_q8_k + (c0 + c) * blocks_per_row + kb + b;
                            const float act_d = a_block->d;

                            for (size_t r = 0; r < rows_this; ++r) {
                                const int32_t block_sum = cpu_quant_q6_k_block_dotprod(w_blocks[r], a_block->qs);
                                const float d = act_d * w_d[r];
                                acc[r][c] += d * (float)block_sum;
                            }
                        }
                    }
                    continue;
                }
                for (size_t r = 0; r < rows_this; ++r) {
                    const size_t row_offset = m + r - row_base;
                    const uint8_t *row_ptr = weight_bytes + row_offset * row_bytes + kb * block_bytes;
                    for (size_t c = 0; c < col_block; ++c) {
                        const marmot_q8_k_block_t *a_block =
                            args->activation_tile_q8_k + (c0 + c) * blocks_per_row + kb;
                        acc[r][c] += args->kernel->ops.dot_q8_k(row_ptr, a_block, chunk);
                    }
                }
            }

            for (size_t c = 0; c < col_block; ++c) {
                float *out_col_f32 = out_data_f32 != nullptr ? out_data_f32 + (n0 + c0 + c) * out_stride_m : nullptr;
                marmot_float16_t *out_col_f16 =
                    out_data_f16 != nullptr ? out_data_f16 + (n0 + c0 + c) * out_stride_m : nullptr;
                for (size_t r = 0; r < rows_this; ++r) {
                    const size_t row_idx = m + r;
                    if (out_col_f32 != nullptr) {
                        out_col_f32[row_idx * out_stride_n] = acc[r][c];
                    }
                    if (out_col_f16 != nullptr) {
                        out_col_f16[row_idx * out_stride_n] = marmot_native_to_float16((_Float16)acc[r][c]);
                    }
                }
            }

            m += rows_this;
        }
    }
}
#endif

static void cpu_quant_matmul_compute_rows_q8_k_blocked_dual(const cpu_quant_matmul_dual_worker_args_t *args) {
    const uint8_t *weight_bytes_a = args->weight_bytes_a;
    const uint8_t *weight_bytes_b = args->weight_bytes_b;
    const size_t row_bytes = args->row_bytes;
    const size_t blocks_per_row = args->blocks_per_row;
    const size_t cols_in_tile = args->cols_in_tile;
    const size_t n0 = args->n0;
    float *out_a_data_f32 = args->out_a_data_f32;
    marmot_float16_t *out_a_data_f16 = args->out_a_data_f16;
    float *out_b_data_f32 = args->out_b_data_f32;
    marmot_float16_t *out_b_data_f16 = args->out_b_data_f16;
    const size_t block_bytes = args->block_bytes;
    const size_t out_stride_m = args->out_stride_m;
    const size_t out_stride_n = args->out_stride_n;
    const size_t row_base = args->row_base;
    const marmot_q8_k_block_t *activation_panel = args->activation_panel_q8_k;
    const size_t activation_panel_cols = args->activation_panel_cols;
    const size_t activation_panel_col0 = args->activation_panel_col0;

    for (size_t c0 = 0; c0 < cols_in_tile; c0 += CPU_QUANT_MATMUL_NR) {
        const size_t col_block = (c0 + CPU_QUANT_MATMUL_NR <= cols_in_tile) ? CPU_QUANT_MATMUL_NR : (cols_in_tile - c0);
        size_t m = args->row_start;
        while (m < args->row_end) {
            const size_t rows_this =
                (m + CPU_QUANT_MATMUL_MR <= args->row_end) ? CPU_QUANT_MATMUL_MR : (args->row_end - m);
            float acc_a[CPU_QUANT_MATMUL_MR][CPU_QUANT_MATMUL_NR] = {{0}};
            float acc_b[CPU_QUANT_MATMUL_MR][CPU_QUANT_MATMUL_NR] = {{0}};

            for (size_t kb = 0; kb < blocks_per_row; kb += CPU_QUANT_KC_BLOCKS) {
                const size_t chunk =
                    (kb + CPU_QUANT_KC_BLOCKS <= blocks_per_row) ? CPU_QUANT_KC_BLOCKS : (blocks_per_row - kb);
                if (cpu_quant_matmul_should_use_small_q8_k_vecdot(args->quant_kind, col_block, args->total_row_count)) {
                    for (size_t r = 0; r < rows_this; ++r) {
                        const size_t row_offset = m + r - row_base;
                        const uint8_t *row_ptr_a = weight_bytes_a + row_offset * row_bytes + kb * block_bytes;
                        const uint8_t *row_ptr_b = weight_bytes_b + row_offset * row_bytes + kb * block_bytes;
                        for (size_t c = 0; c < col_block; ++c) {
                            const size_t col_index = activation_panel_col0 + c0 + c;
                            const marmot_q8_k_block_t *a_block = activation_panel != nullptr
                                ? activation_panel + kb * activation_panel_cols + col_index
                                : args->activation_tile_q8_k + (c0 + c) * blocks_per_row + kb;
                            acc_a[r][c] += args->kernel->ops.dot_q8_k(row_ptr_a, a_block, chunk);
                            acc_b[r][c] += args->kernel->ops.dot_q8_k(row_ptr_b, a_block, chunk);
                        }
                    }
                    continue;
                }
                if (args->quant_kind == MARMOT_QUANT_KIND_Q4_K) {
                    for (size_t b = 0; b < chunk; ++b) {
                        uint8_t scales_a[CPU_QUANT_MATMUL_MR][8];
                        uint8_t mins_a[CPU_QUANT_MATMUL_MR][8];
                        float w_d_a[CPU_QUANT_MATMUL_MR];
                        float w_dmin_a[CPU_QUANT_MATMUL_MR];
                        const marmot_q4_k_block_t *w_blocks_a[CPU_QUANT_MATMUL_MR];
                        uint8_t scales_b[CPU_QUANT_MATMUL_MR][8];
                        uint8_t mins_b[CPU_QUANT_MATMUL_MR][8];
                        float w_d_b[CPU_QUANT_MATMUL_MR];
                        float w_dmin_b[CPU_QUANT_MATMUL_MR];
                        const marmot_q4_k_block_t *w_blocks_b[CPU_QUANT_MATMUL_MR];

                        for (size_t r = 0; r < rows_this; ++r) {
                            const size_t row_offset = m + r - row_base;
                            if (b + CPU_QUANT_PREFETCH_BLOCKS < chunk) {
                                const size_t prefetch_offset = (kb + b + CPU_QUANT_PREFETCH_BLOCKS) * block_bytes;
                                MARMOT_PREFETCH(weight_bytes_a + row_offset * row_bytes + prefetch_offset);
                                MARMOT_PREFETCH(weight_bytes_b + row_offset * row_bytes + prefetch_offset);
                            }
                            const marmot_q4_k_block_t *w_block_a =
                                (const marmot_q4_k_block_t *)(weight_bytes_a + row_offset * row_bytes +
                                                              (kb + b) * block_bytes);
                            const marmot_q4_k_block_t *w_block_b =
                                (const marmot_q4_k_block_t *)(weight_bytes_b + row_offset * row_bytes +
                                                              (kb + b) * block_bytes);
                            w_blocks_a[r] = w_block_a;
                            w_blocks_b[r] = w_block_b;
                            cpu_quant_q4_k_decode_block(w_block_a, scales_a[r], mins_a[r], &w_d_a[r], &w_dmin_a[r]);
                            cpu_quant_q4_k_decode_block(w_block_b, scales_b[r], mins_b[r], &w_d_b[r], &w_dmin_b[r]);
                        }

                        for (size_t c = 0; c < col_block; ++c) {
                            const size_t col_index = activation_panel_col0 + c0 + c;
                            if (b + CPU_QUANT_PREFETCH_BLOCKS < chunk) {
                                const size_t prefetch_b = kb + b + CPU_QUANT_PREFETCH_BLOCKS;
                                const marmot_q8_k_block_t *prefetch_block = activation_panel != nullptr
                                    ? activation_panel + prefetch_b * activation_panel_cols + col_index
                                    : args->activation_tile_q8_k + (c0 + c) * blocks_per_row + prefetch_b;
                                MARMOT_PREFETCH(prefetch_block);
                            }
                            const marmot_q8_k_block_t *a_block = activation_panel != nullptr
                                ? activation_panel + (kb + b) * activation_panel_cols + col_index
                                : args->activation_tile_q8_k + (c0 + c) * blocks_per_row + kb + b;
                            const float act_d = a_block->d;
                            const int16_t *bsums = a_block->bsums;

                            for (size_t r = 0; r < rows_this; ++r) {
                                const int32_t block_sum_a =
                                    cpu_quant_q4_k_block_dotprod(w_blocks_a[r], a_block->qs, scales_a[r]);
                                const int32_t block_sum_b =
                                    cpu_quant_q4_k_block_dotprod(w_blocks_b[r], a_block->qs, scales_b[r]);
                                const int32_t sumi_a = cpu_quant_q4_k_compute_sumi(bsums, mins_a[r]);
                                const int32_t sumi_b = cpu_quant_q4_k_compute_sumi(bsums, mins_b[r]);
                                const float d_a = act_d * w_d_a[r];
                                const float d_b = act_d * w_d_b[r];
                                const float dmin_a = act_d * w_dmin_a[r];
                                const float dmin_b = act_d * w_dmin_b[r];
                                acc_a[r][c] += d_a * (float)block_sum_a - dmin_a * (float)sumi_a;
                                acc_b[r][c] += d_b * (float)block_sum_b - dmin_b * (float)sumi_b;
                            }
                        }
                    }
                    continue;
                }

                if (args->quant_kind == MARMOT_QUANT_KIND_Q6_K) {
                    for (size_t b = 0; b < chunk; ++b) {
                        float w_d_a[CPU_QUANT_MATMUL_MR];
                        const marmot_q6_k_block_t *w_blocks_a[CPU_QUANT_MATMUL_MR];
                        float w_d_b[CPU_QUANT_MATMUL_MR];
                        const marmot_q6_k_block_t *w_blocks_b[CPU_QUANT_MATMUL_MR];

                        for (size_t r = 0; r < rows_this; ++r) {
                            const size_t row_offset = m + r - row_base;
                            if (b + CPU_QUANT_PREFETCH_BLOCKS < chunk) {
                                const size_t prefetch_offset = (kb + b + CPU_QUANT_PREFETCH_BLOCKS) * block_bytes;
                                MARMOT_PREFETCH(weight_bytes_a + row_offset * row_bytes + prefetch_offset);
                                MARMOT_PREFETCH(weight_bytes_b + row_offset * row_bytes + prefetch_offset);
                            }
                            const marmot_q6_k_block_t *w_block_a =
                                (const marmot_q6_k_block_t *)(weight_bytes_a + row_offset * row_bytes +
                                                              (kb + b) * block_bytes);
                            const marmot_q6_k_block_t *w_block_b =
                                (const marmot_q6_k_block_t *)(weight_bytes_b + row_offset * row_bytes +
                                                              (kb + b) * block_bytes);
                            w_blocks_a[r] = w_block_a;
                            w_blocks_b[r] = w_block_b;
                            w_d_a[r] = (float)marmot_float16_to_native(w_block_a->d);
                            w_d_b[r] = (float)marmot_float16_to_native(w_block_b->d);
                        }

                        for (size_t c = 0; c < col_block; ++c) {
                            const size_t col_index = activation_panel_col0 + c0 + c;
                            if (b + CPU_QUANT_PREFETCH_BLOCKS < chunk) {
                                const size_t prefetch_b = kb + b + CPU_QUANT_PREFETCH_BLOCKS;
                                const marmot_q8_k_block_t *prefetch_block = activation_panel != nullptr
                                    ? activation_panel + prefetch_b * activation_panel_cols + col_index
                                    : args->activation_tile_q8_k + (c0 + c) * blocks_per_row + prefetch_b;
                                MARMOT_PREFETCH(prefetch_block);
                            }
                            const marmot_q8_k_block_t *a_block = activation_panel != nullptr
                                ? activation_panel + (kb + b) * activation_panel_cols + col_index
                                : args->activation_tile_q8_k + (c0 + c) * blocks_per_row + kb + b;
                            const float act_d = a_block->d;

                            for (size_t r = 0; r < rows_this; ++r) {
                                const int32_t block_sum_a = cpu_quant_q6_k_block_dotprod(w_blocks_a[r], a_block->qs);
                                const int32_t block_sum_b = cpu_quant_q6_k_block_dotprod(w_blocks_b[r], a_block->qs);
                                acc_a[r][c] += act_d * w_d_a[r] * (float)block_sum_a;
                                acc_b[r][c] += act_d * w_d_b[r] * (float)block_sum_b;
                            }
                        }
                    }
                    continue;
                }
                for (size_t r = 0; r < rows_this; ++r) {
                    const size_t row_offset = m + r - row_base;
                    const uint8_t *row_ptr_a = weight_bytes_a + row_offset * row_bytes + kb * block_bytes;
                    const uint8_t *row_ptr_b = weight_bytes_b + row_offset * row_bytes + kb * block_bytes;
                    for (size_t c = 0; c < col_block; ++c) {
                        const marmot_q8_k_block_t *a_block =
                            args->activation_tile_q8_k + (c0 + c) * blocks_per_row + kb;
                        acc_a[r][c] += args->kernel->ops.dot_q8_k(row_ptr_a, a_block, chunk);
                        acc_b[r][c] += args->kernel->ops.dot_q8_k(row_ptr_b, a_block, chunk);
                    }
                }
            }

            for (size_t c = 0; c < col_block; ++c) {
                float *out_col_a_f32 =
                    out_a_data_f32 != nullptr ? out_a_data_f32 + (n0 + c0 + c) * out_stride_m : nullptr;
                marmot_float16_t *out_col_a_f16 =
                    out_a_data_f16 != nullptr ? out_a_data_f16 + (n0 + c0 + c) * out_stride_m : nullptr;
                float *out_col_b_f32 =
                    out_b_data_f32 != nullptr ? out_b_data_f32 + (n0 + c0 + c) * out_stride_m : nullptr;
                marmot_float16_t *out_col_b_f16 =
                    out_b_data_f16 != nullptr ? out_b_data_f16 + (n0 + c0 + c) * out_stride_m : nullptr;
                for (size_t r = 0; r < rows_this; ++r) {
                    const size_t row_idx = m + r;
                    if (out_col_a_f32 != nullptr) {
                        out_col_a_f32[row_idx * out_stride_n] = acc_a[r][c];
                    }
                    if (out_col_a_f16 != nullptr) {
                        out_col_a_f16[row_idx * out_stride_n] = marmot_native_to_float16((_Float16)acc_a[r][c]);
                    }
                    if (out_col_b_f32 != nullptr) {
                        out_col_b_f32[row_idx * out_stride_n] = acc_b[r][c];
                    }
                    if (out_col_b_f16 != nullptr) {
                        out_col_b_f16[row_idx * out_stride_n] = marmot_native_to_float16((_Float16)acc_b[r][c]);
                    }
                }
            }

            m += rows_this;
        }
    }
}

static void cpu_quant_matmul_compute_rows(const cpu_quant_matmul_worker_args_t *args) {
    const cpu_matmul_quant_kernel_t *kernel = args->kernel;
    const uint8_t *weight_bytes = args->weight_bytes;
    const size_t row_bytes = args->row_bytes;
    const size_t blocks_per_row = args->blocks_per_row;
    const bool use_q8_k = kernel->format != nullptr &&
        kernel->format->activation_packer == MARMOT_MATMUL_ACTIVATION_PACKER_Q8_K && kernel->ops.dot_q8_k != nullptr;
    const size_t cols_in_tile = args->cols_in_tile;
    const size_t n0 = args->n0;
    float *out_data_f32 = args->out_data_f32;
    marmot_float16_t *out_data_f16 = args->out_data_f16;
    const size_t block_bytes = args->block_bytes;
    const size_t out_stride_m = args->out_stride_m;
    const size_t out_stride_n = args->out_stride_n;
    const size_t row_base = args->row_base;

#if MARMOT_ENABLE_NEON && defined(__aarch64__)
#if defined(__ARM_FEATURE_I8MM) || defined(__ARM_FEATURE_MATMUL_INT8)
    if (!use_q8_k && kernel->ops.dot_q8_0 == (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q8_0_q8_0_neon_i8mm &&
        args->activation_tile_q8_0 != nullptr) {
        if (args->cols_in_tile >= 2) {
            cpu_quant_matmul_compute_rows_q8_0_i8mm_blocked(args);
        } else {
            cpu_quant_matmul_compute_rows_q8_0_i8mm(args);
        }
        return;
    }
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    if (!use_q8_k && kernel->ops.dot_q8_0 == (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q8_0_q8_0_neon_dotprod &&
        args->activation_tile_q8_0 != nullptr) {
        if (args->cols_in_tile >= 2) {
            cpu_quant_matmul_compute_rows_q8_0_dotprod_blocked(args);
        } else {
            cpu_quant_matmul_compute_rows_q8_0_dotprod(args);
        }
        return;
    }
    if (use_q8_k && args->activation_tile_q8_k != nullptr) {
        cpu_quant_matmul_compute_rows_q8_k_blocked(args);
        return;
    }
#endif
#endif
    if (!use_q8_k && args->activation_tile_q8_0 != nullptr && args->cols_in_tile >= 2) {
        cpu_quant_matmul_compute_rows_q8_0_blocked_generic(args);
        return;
    }

    for (size_t m = args->row_start; m < args->row_end; ++m) {
        const uint8_t *row_ptr = weight_bytes + (m - row_base) * row_bytes;
        for (size_t c = 0; c < cols_in_tile; ++c) {
            float dot = 0.0f;
            if (use_q8_k) {
                const marmot_q8_k_block_t *col_blocks = args->activation_tile_q8_k + c * blocks_per_row;
                for (size_t kb = 0; kb < blocks_per_row; kb += CPU_QUANT_KC_BLOCKS) {
                    const size_t chunk =
                        (kb + CPU_QUANT_KC_BLOCKS <= blocks_per_row) ? CPU_QUANT_KC_BLOCKS : (blocks_per_row - kb);
                    const uint8_t *row_block_ptr = row_ptr + kb * block_bytes;
                    dot += kernel->ops.dot_q8_k(row_block_ptr, col_blocks + kb, chunk);
                }
            } else {
                const marmot_q8_0_block_t *col_blocks = args->activation_tile_q8_0 + c * blocks_per_row;
                for (size_t kb = 0; kb < blocks_per_row; kb += CPU_QUANT_KC_BLOCKS) {
                    const size_t chunk =
                        (kb + CPU_QUANT_KC_BLOCKS <= blocks_per_row) ? CPU_QUANT_KC_BLOCKS : (blocks_per_row - kb);
                    const uint8_t *row_block_ptr = row_ptr + kb * block_bytes;
                    dot += kernel->ops.dot_q8_0(row_block_ptr, col_blocks + kb, chunk);
                }
            }
            if (out_data_f32 != nullptr) {
                float *out_row = out_data_f32 + (n0 + c) * out_stride_m;
                out_row[m * out_stride_n] = dot;
            }
            if (out_data_f16 != nullptr) {
                marmot_float16_t *out_row = out_data_f16 + (n0 + c) * out_stride_m;
                out_row[m * out_stride_n] = marmot_native_to_float16((_Float16)dot);
            }
        }
    }
}

#if MARMOT_ENABLE_NEON && defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
typedef struct {
    const cpu_matmul_quant_kernel_t *kernel;
    const uint8_t *weight_bytes_a;
    const uint8_t *weight_bytes_b;
    size_t row_bytes;
    size_t blocks_per_row;
    const marmot_q8_0_block_t *activation_tile_q8_0;
    const marmot_q8_k_block_t *activation_tile_q8_k;
    const marmot_q8_k_block_t *activation_panel_q8_k;
    size_t activation_panel_cols;
    size_t activation_panel_col0;
    size_t cols_in_tile;
    size_t n0;
    float *out_a_data_f32;
    marmot_float16_t *out_a_data_f16;
    float *out_b_data_f32;
    marmot_float16_t *out_b_data_f16;
    size_t block_bytes;
    size_t out_stride_m;
    size_t out_stride_n;
    size_t row_base;
    size_t total_row_count;
    marmot_quant_kind_t quant_kind;
} cpu_quant_matmul_dual_dispatch_ctx_t;

static void cpu_quant_matmul_dispatch_range_dual(void *ctx, size_t row_start, size_t row_end) {
    const cpu_quant_matmul_dual_dispatch_ctx_t *c = (const cpu_quant_matmul_dual_dispatch_ctx_t *)ctx;
    cpu_quant_matmul_dual_worker_args_t args = {
        .kernel = c->kernel,
        .weight_bytes_a = c->weight_bytes_a,
        .weight_bytes_b = c->weight_bytes_b,
        .row_bytes = c->row_bytes,
        .blocks_per_row = c->blocks_per_row,
        .activation_tile_q8_0 = c->activation_tile_q8_0,
        .activation_tile_q8_k = c->activation_tile_q8_k,
        .activation_panel_q8_k = c->activation_panel_q8_k,
        .activation_panel_cols = c->activation_panel_cols,
        .activation_panel_col0 = c->activation_panel_col0,
        .cols_in_tile = c->cols_in_tile,
        .n0 = c->n0,
        .out_a_data_f32 = c->out_a_data_f32,
        .out_a_data_f16 = c->out_a_data_f16,
        .out_b_data_f32 = c->out_b_data_f32,
        .out_b_data_f16 = c->out_b_data_f16,
        .block_bytes = c->block_bytes,
        .out_stride_m = c->out_stride_m,
        .out_stride_n = c->out_stride_n,
        .row_base = c->row_base,
        .total_row_count = c->total_row_count,
        .row_start = c->row_base + row_start,
        .row_end = c->row_base + row_end,
        .quant_kind = c->quant_kind,
    };
    cpu_quant_matmul_compute_rows_q8_k_blocked_dual(&args);
}
#endif

typedef struct {
    const cpu_matmul_quant_kernel_t *kernel;
    const uint8_t *weight_bytes;
    size_t row_bytes;
    size_t blocks_per_row;
    const marmot_q8_0_block_t *activation_tile_q8_0;
    const marmot_q8_k_block_t *activation_tile_q8_k;
    const marmot_q8_k_block_t *activation_panel_q8_k;
    size_t activation_panel_cols;
    size_t activation_panel_col0;
    size_t cols_in_tile;
    size_t n0;
    float *out_data_f32;
    marmot_float16_t *out_data_f16;
    size_t block_bytes;
    size_t out_stride_m;
    size_t out_stride_n;
    size_t row_base;
    size_t total_row_count;
    marmot_quant_kind_t quant_kind;
} cpu_quant_matmul_dispatch_ctx_t;

static void cpu_quant_matmul_dispatch_range(void *ctx, size_t row_start, size_t row_end) {
    const cpu_quant_matmul_dispatch_ctx_t *c = (const cpu_quant_matmul_dispatch_ctx_t *)ctx;
    cpu_quant_matmul_worker_args_t args = {
        .kernel = c->kernel,
        .weight_bytes = c->weight_bytes,
        .row_bytes = c->row_bytes,
        .blocks_per_row = c->blocks_per_row,
        .activation_tile_q8_0 = c->activation_tile_q8_0,
        .activation_tile_q8_k = c->activation_tile_q8_k,
        .activation_panel_q8_k = c->activation_panel_q8_k,
        .activation_panel_cols = c->activation_panel_cols,
        .activation_panel_col0 = c->activation_panel_col0,
        .cols_in_tile = c->cols_in_tile,
        .n0 = c->n0,
        .out_data_f32 = c->out_data_f32,
        .out_data_f16 = c->out_data_f16,
        .block_bytes = c->block_bytes,
        .out_stride_m = c->out_stride_m,
        .out_stride_n = c->out_stride_n,
        .row_base = c->row_base,
        .total_row_count = c->total_row_count,
        .row_start = c->row_base + row_start,
        .row_end = c->row_base + row_end,
        .quant_kind = c->quant_kind,
    };
    cpu_quant_matmul_compute_rows(&args);
}

static inline size_t cpu_quant_matmul_row_grain(size_t row_count, size_t cols_in_tile, size_t thread_cap) {
    size_t grain = CPU_QUANT_MATMUL_BLOCK_ROWS;
    if (thread_cap == 0) {
        thread_cap = 1;
    }
    if (cols_in_tile <= 4 && row_count > grain * thread_cap) {
        const size_t target_tasks = thread_cap * (cols_in_tile <= 1 ? 8u : 12u);
        size_t dynamic = (row_count + target_tasks - 1) / target_tasks;
        const size_t align = CPU_QUANT_MATMUL_MR;
        dynamic = ((dynamic + align - 1) / align) * align;
        if (dynamic > grain) {
            grain = dynamic;
        }
    }
    if (grain > 4096) {
        grain = 4096;
    }
    return grain;
}

#if MARMOT_ENABLE_NEON && defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
typedef struct {
    const cpu_matmul_quant_kernel_t *kernel;
    const uint8_t *weight_bytes;
    size_t row_count;
    size_t row_bytes;
    size_t blocks_per_row;
    const marmot_q8_k_block_t *activation_tile_q8_k;
    const marmot_q8_k_block_t *activation_panel_q8_k;
    size_t activation_panel_cols;
    size_t cols_in_tile;
    size_t n0;
    float *out_data_f32;
    marmot_float16_t *out_data_f16;
    size_t out_stride_m;
    size_t out_stride_n;
    size_t panel_rows;
    size_t block_bytes;
    marmot_quant_kind_t quant_kind;
} cpu_quant_matmul_row_panel_ctx_t;

typedef struct {
    const cpu_matmul_quant_kernel_t *kernel;
    const uint8_t *weight_bytes_a;
    const uint8_t *weight_bytes_b;
    size_t row_count;
    size_t row_bytes;
    size_t blocks_per_row;
    const marmot_q8_k_block_t *activation_tile_q8_k;
    const marmot_q8_k_block_t *activation_panel_q8_k;
    size_t activation_panel_cols;
    size_t cols_in_tile;
    size_t n0;
    float *out_a_data_f32;
    marmot_float16_t *out_a_data_f16;
    float *out_b_data_f32;
    marmot_float16_t *out_b_data_f16;
    size_t out_stride_m;
    size_t out_stride_n;
    size_t panel_rows;
    size_t block_bytes;
    marmot_quant_kind_t quant_kind;
} cpu_quant_matmul_row_panel_dual_ctx_t;

static inline const marmot_q8_k_block_t *
cpu_quant_matmul_row_panel_activation_block(const cpu_quant_matmul_row_panel_ctx_t *ctx, size_t block, size_t col) {
    if (ctx->activation_panel_q8_k != nullptr) {
        return ctx->activation_panel_q8_k + block * ctx->activation_panel_cols + col;
    }
    return ctx->activation_tile_q8_k + col * ctx->blocks_per_row + block;
}

static inline const marmot_q8_k_block_t *cpu_quant_matmul_row_panel_activation_block_dual(
    const cpu_quant_matmul_row_panel_dual_ctx_t *ctx, size_t block, size_t col
) {
    if (ctx->activation_panel_q8_k != nullptr) {
        return ctx->activation_panel_q8_k + block * ctx->activation_panel_cols + col;
    }
    return ctx->activation_tile_q8_k + col * ctx->blocks_per_row + block;
}

static void cpu_quant_matmul_compute_row_panel_q4_k(
    const cpu_quant_matmul_row_panel_ctx_t *ctx, size_t panel_start, size_t panel_end
) {
    const size_t panel_bytes = ctx->blocks_per_row * ctx->panel_rows * ctx->block_bytes;
    for (size_t panel = panel_start; panel < panel_end; ++panel) {
        const size_t row0 = panel * ctx->panel_rows;
        const size_t rows_this = (row0 + ctx->panel_rows <= ctx->row_count) ? ctx->panel_rows : (ctx->row_count - row0);
        const uint8_t *panel_ptr = ctx->weight_bytes + panel * panel_bytes;

        float acc[CPU_QUANT_MATMUL_MR][CPU_QUANT_MATMUL_NR] = {{0}};
        for (size_t block = 0; block < ctx->blocks_per_row; ++block) {
            const uint8_t *panel_block = panel_ptr + block * ctx->panel_rows * ctx->block_bytes;
            uint8_t scales[CPU_QUANT_MATMUL_MR][8];
            uint8_t mins[CPU_QUANT_MATMUL_MR][8];
            float w_d[CPU_QUANT_MATMUL_MR];
            float w_dmin[CPU_QUANT_MATMUL_MR];
            const marmot_q4_k_block_t *w_blocks[CPU_QUANT_MATMUL_MR];
            for (size_t row = 0; row < rows_this; ++row) {
                const marmot_q4_k_block_t *w_block =
                    (const marmot_q4_k_block_t *)(panel_block + row * ctx->block_bytes);
                w_blocks[row] = w_block;
                cpu_quant_q4_k_decode_block(w_block, scales[row], mins[row], &w_d[row], &w_dmin[row]);
            }

            for (size_t col = 0; col < ctx->cols_in_tile; ++col) {
                const marmot_q8_k_block_t *a_block = cpu_quant_matmul_row_panel_activation_block(ctx, block, col);
                const float act_d = a_block->d;
                const int16_t *bsums = a_block->bsums;
                for (size_t row = 0; row < rows_this; ++row) {
                    const int32_t block_sum = cpu_quant_q4_k_block_dotprod(w_blocks[row], a_block->qs, scales[row]);
                    const int32_t sumi = cpu_quant_q4_k_compute_sumi(bsums, mins[row]);
                    acc[row][col] += act_d * (w_d[row] * (float)block_sum - w_dmin[row] * (float)sumi);
                }
            }
        }

        for (size_t col = 0; col < ctx->cols_in_tile; ++col) {
            float *out_col_f32 =
                ctx->out_data_f32 != nullptr ? ctx->out_data_f32 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            marmot_float16_t *out_col_f16 =
                ctx->out_data_f16 != nullptr ? ctx->out_data_f16 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            for (size_t row = 0; row < rows_this; ++row) {
                const size_t row_idx = row0 + row;
                if (out_col_f32 != nullptr) {
                    out_col_f32[row_idx * ctx->out_stride_n] = acc[row][col];
                }
                if (out_col_f16 != nullptr) {
                    out_col_f16[row_idx * ctx->out_stride_n] = marmot_native_to_float16((_Float16)acc[row][col]);
                }
            }
        }
    }
}

static void cpu_quant_matmul_compute_row_panel_q4_k_decoded(
    const cpu_quant_matmul_row_panel_ctx_t *ctx, size_t panel_start, size_t panel_end
) {
    const size_t panel_bytes = ctx->blocks_per_row * ctx->panel_rows * ctx->block_bytes;
    for (size_t panel = panel_start; panel < panel_end; ++panel) {
        const size_t row0 = panel * ctx->panel_rows;
        const size_t rows_this = (row0 + ctx->panel_rows <= ctx->row_count) ? ctx->panel_rows : (ctx->row_count - row0);
        const uint8_t *panel_ptr = ctx->weight_bytes + panel * panel_bytes;

        float acc[CPU_QUANT_MATMUL_MR][CPU_QUANT_MATMUL_NR] = {{0}};
        for (size_t block = 0; block < ctx->blocks_per_row; ++block) {
            const cpu_q4_k_row_panel_decoded_block_t *panel_block =
                (const cpu_q4_k_row_panel_decoded_block_t *)(panel_ptr + block * ctx->panel_rows * ctx->block_bytes);

            for (size_t col = 0; col < ctx->cols_in_tile; ++col) {
                const marmot_q8_k_block_t *a_block = cpu_quant_matmul_row_panel_activation_block(ctx, block, col);
                const float act_d = a_block->d;
                const int16_t *bsums = a_block->bsums;
                for (size_t row = 0; row < rows_this; ++row) {
                    const cpu_q4_k_row_panel_decoded_block_t *w_block = panel_block + row;
                    const int32_t block_sum =
                        cpu_quant_q4_k_block_dotprod_qs(w_block->qs, a_block->qs, w_block->scales);
                    const int32_t sumi = cpu_quant_q4_k_compute_sumi(bsums, w_block->mins);
                    acc[row][col] += act_d * (w_block->d * (float)block_sum - w_block->dmin * (float)sumi);
                }
            }
        }

        for (size_t col = 0; col < ctx->cols_in_tile; ++col) {
            float *out_col_f32 =
                ctx->out_data_f32 != nullptr ? ctx->out_data_f32 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            marmot_float16_t *out_col_f16 =
                ctx->out_data_f16 != nullptr ? ctx->out_data_f16 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            for (size_t row = 0; row < rows_this; ++row) {
                const size_t row_idx = row0 + row;
                if (out_col_f32 != nullptr) {
                    out_col_f32[row_idx * ctx->out_stride_n] = acc[row][col];
                }
                if (out_col_f16 != nullptr) {
                    out_col_f16[row_idx * ctx->out_stride_n] = marmot_native_to_float16((_Float16)acc[row][col]);
                }
            }
        }
    }
}

static void cpu_quant_matmul_compute_row_panel_q6_k(
    const cpu_quant_matmul_row_panel_ctx_t *ctx, size_t panel_start, size_t panel_end
) {
    const size_t panel_bytes = ctx->blocks_per_row * ctx->panel_rows * ctx->block_bytes;
    for (size_t panel = panel_start; panel < panel_end; ++panel) {
        const size_t row0 = panel * ctx->panel_rows;
        const size_t rows_this = (row0 + ctx->panel_rows <= ctx->row_count) ? ctx->panel_rows : (ctx->row_count - row0);
        const uint8_t *panel_ptr = ctx->weight_bytes + panel * panel_bytes;

        float acc[CPU_QUANT_MATMUL_MR][CPU_QUANT_MATMUL_NR] = {{0}};
        for (size_t block = 0; block < ctx->blocks_per_row; ++block) {
            const uint8_t *panel_block = panel_ptr + block * ctx->panel_rows * ctx->block_bytes;
            float w_d[CPU_QUANT_MATMUL_MR];
            const marmot_q6_k_block_t *w_blocks[CPU_QUANT_MATMUL_MR];
            for (size_t row = 0; row < rows_this; ++row) {
                const marmot_q6_k_block_t *w_block =
                    (const marmot_q6_k_block_t *)(panel_block + row * ctx->block_bytes);
                w_blocks[row] = w_block;
                w_d[row] = (float)marmot_float16_to_native(w_block->d);
            }

            for (size_t col = 0; col < ctx->cols_in_tile; ++col) {
                const marmot_q8_k_block_t *a_block = cpu_quant_matmul_row_panel_activation_block(ctx, block, col);
                const float act_d = a_block->d;
                for (size_t row = 0; row < rows_this; ++row) {
                    const int32_t block_sum = cpu_quant_q6_k_block_dotprod(w_blocks[row], a_block->qs);
                    acc[row][col] += act_d * w_d[row] * (float)block_sum;
                }
            }
        }

        for (size_t col = 0; col < ctx->cols_in_tile; ++col) {
            float *out_col_f32 =
                ctx->out_data_f32 != nullptr ? ctx->out_data_f32 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            marmot_float16_t *out_col_f16 =
                ctx->out_data_f16 != nullptr ? ctx->out_data_f16 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            for (size_t row = 0; row < rows_this; ++row) {
                const size_t row_idx = row0 + row;
                if (out_col_f32 != nullptr) {
                    out_col_f32[row_idx * ctx->out_stride_n] = acc[row][col];
                }
                if (out_col_f16 != nullptr) {
                    out_col_f16[row_idx * ctx->out_stride_n] = marmot_native_to_float16((_Float16)acc[row][col]);
                }
            }
        }
    }
}

static void cpu_quant_matmul_compute_row_panel_q6_k_decoded(
    const cpu_quant_matmul_row_panel_ctx_t *ctx, size_t panel_start, size_t panel_end
) {
    const size_t panel_bytes = ctx->blocks_per_row * ctx->panel_rows * ctx->block_bytes;
    for (size_t panel = panel_start; panel < panel_end; ++panel) {
        const size_t row0 = panel * ctx->panel_rows;
        const size_t rows_this = (row0 + ctx->panel_rows <= ctx->row_count) ? ctx->panel_rows : (ctx->row_count - row0);
        const uint8_t *panel_ptr = ctx->weight_bytes + panel * panel_bytes;

        float acc[CPU_QUANT_MATMUL_MR][CPU_QUANT_MATMUL_NR] = {{0}};
        for (size_t block = 0; block < ctx->blocks_per_row; ++block) {
            const cpu_q6_k_row_panel_decoded_block_t *panel_block =
                (const cpu_q6_k_row_panel_decoded_block_t *)(panel_ptr + block * ctx->panel_rows * ctx->block_bytes);
            for (size_t col = 0; col < ctx->cols_in_tile; ++col) {
                const marmot_q8_k_block_t *a_block = cpu_quant_matmul_row_panel_activation_block(ctx, block, col);
                const float act_d = a_block->d;
                for (size_t row = 0; row < rows_this; ++row) {
                    const cpu_q6_k_row_panel_decoded_block_t *w_block = panel_block + row;
                    const int32_t block_sum = cpu_quant_q6_k_decoded_block_dotprod(w_block, a_block->qs);
                    acc[row][col] += act_d * w_block->d * (float)block_sum;
                }
            }
        }

        for (size_t col = 0; col < ctx->cols_in_tile; ++col) {
            float *out_col_f32 =
                ctx->out_data_f32 != nullptr ? ctx->out_data_f32 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            marmot_float16_t *out_col_f16 =
                ctx->out_data_f16 != nullptr ? ctx->out_data_f16 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            for (size_t row = 0; row < rows_this; ++row) {
                const size_t row_idx = row0 + row;
                if (out_col_f32 != nullptr) {
                    out_col_f32[row_idx * ctx->out_stride_n] = acc[row][col];
                }
                if (out_col_f16 != nullptr) {
                    out_col_f16[row_idx * ctx->out_stride_n] = marmot_native_to_float16((_Float16)acc[row][col]);
                }
            }
        }
    }
}

static void cpu_quant_matmul_compute_row_panel_dual_q4_k(
    const cpu_quant_matmul_row_panel_dual_ctx_t *ctx, size_t panel_start, size_t panel_end
) {
    const size_t panel_bytes = ctx->blocks_per_row * ctx->panel_rows * ctx->block_bytes;
    for (size_t panel = panel_start; panel < panel_end; ++panel) {
        const size_t row0 = panel * ctx->panel_rows;
        const size_t rows_this = (row0 + ctx->panel_rows <= ctx->row_count) ? ctx->panel_rows : (ctx->row_count - row0);
        const uint8_t *panel_ptr_a = ctx->weight_bytes_a + panel * panel_bytes;
        const uint8_t *panel_ptr_b = ctx->weight_bytes_b + panel * panel_bytes;

        float acc_a[CPU_QUANT_MATMUL_MR][CPU_QUANT_MATMUL_NR] = {{0}};
        float acc_b[CPU_QUANT_MATMUL_MR][CPU_QUANT_MATMUL_NR] = {{0}};
        for (size_t block = 0; block < ctx->blocks_per_row; ++block) {
            const uint8_t *panel_block_a = panel_ptr_a + block * ctx->panel_rows * ctx->block_bytes;
            const uint8_t *panel_block_b = panel_ptr_b + block * ctx->panel_rows * ctx->block_bytes;
            uint8_t scales_a[CPU_QUANT_MATMUL_MR][8];
            uint8_t mins_a[CPU_QUANT_MATMUL_MR][8];
            float w_d_a[CPU_QUANT_MATMUL_MR];
            float w_dmin_a[CPU_QUANT_MATMUL_MR];
            const marmot_q4_k_block_t *w_blocks_a[CPU_QUANT_MATMUL_MR];
            uint8_t scales_b[CPU_QUANT_MATMUL_MR][8];
            uint8_t mins_b[CPU_QUANT_MATMUL_MR][8];
            float w_d_b[CPU_QUANT_MATMUL_MR];
            float w_dmin_b[CPU_QUANT_MATMUL_MR];
            const marmot_q4_k_block_t *w_blocks_b[CPU_QUANT_MATMUL_MR];
            for (size_t row = 0; row < rows_this; ++row) {
                const marmot_q4_k_block_t *w_block_a =
                    (const marmot_q4_k_block_t *)(panel_block_a + row * ctx->block_bytes);
                const marmot_q4_k_block_t *w_block_b =
                    (const marmot_q4_k_block_t *)(panel_block_b + row * ctx->block_bytes);
                w_blocks_a[row] = w_block_a;
                w_blocks_b[row] = w_block_b;
                cpu_quant_q4_k_decode_block(w_block_a, scales_a[row], mins_a[row], &w_d_a[row], &w_dmin_a[row]);
                cpu_quant_q4_k_decode_block(w_block_b, scales_b[row], mins_b[row], &w_d_b[row], &w_dmin_b[row]);
            }

            for (size_t col = 0; col < ctx->cols_in_tile; ++col) {
                const marmot_q8_k_block_t *a_block = cpu_quant_matmul_row_panel_activation_block_dual(ctx, block, col);
                const float act_d = a_block->d;
                const int16_t *bsums = a_block->bsums;
                for (size_t row = 0; row < rows_this; ++row) {
                    const int32_t block_sum_a =
                        cpu_quant_q4_k_block_dotprod(w_blocks_a[row], a_block->qs, scales_a[row]);
                    const int32_t sumi_a = cpu_quant_q4_k_compute_sumi(bsums, mins_a[row]);
                    acc_a[row][col] += act_d * (w_d_a[row] * (float)block_sum_a - w_dmin_a[row] * (float)sumi_a);

                    const int32_t block_sum_b =
                        cpu_quant_q4_k_block_dotprod(w_blocks_b[row], a_block->qs, scales_b[row]);
                    const int32_t sumi_b = cpu_quant_q4_k_compute_sumi(bsums, mins_b[row]);
                    acc_b[row][col] += act_d * (w_d_b[row] * (float)block_sum_b - w_dmin_b[row] * (float)sumi_b);
                }
            }
        }

        for (size_t col = 0; col < ctx->cols_in_tile; ++col) {
            float *out_a_col_f32 =
                ctx->out_a_data_f32 != nullptr ? ctx->out_a_data_f32 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            marmot_float16_t *out_a_col_f16 =
                ctx->out_a_data_f16 != nullptr ? ctx->out_a_data_f16 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            float *out_b_col_f32 =
                ctx->out_b_data_f32 != nullptr ? ctx->out_b_data_f32 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            marmot_float16_t *out_b_col_f16 =
                ctx->out_b_data_f16 != nullptr ? ctx->out_b_data_f16 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            for (size_t row = 0; row < rows_this; ++row) {
                const size_t row_idx = row0 + row;
                if (out_a_col_f32 != nullptr) {
                    out_a_col_f32[row_idx * ctx->out_stride_n] = acc_a[row][col];
                }
                if (out_a_col_f16 != nullptr) {
                    out_a_col_f16[row_idx * ctx->out_stride_n] = marmot_native_to_float16((_Float16)acc_a[row][col]);
                }
                if (out_b_col_f32 != nullptr) {
                    out_b_col_f32[row_idx * ctx->out_stride_n] = acc_b[row][col];
                }
                if (out_b_col_f16 != nullptr) {
                    out_b_col_f16[row_idx * ctx->out_stride_n] = marmot_native_to_float16((_Float16)acc_b[row][col]);
                }
            }
        }
    }
}

static void cpu_quant_matmul_compute_row_panel_dual_q4_k_decoded(
    const cpu_quant_matmul_row_panel_dual_ctx_t *ctx, size_t panel_start, size_t panel_end
) {
    const size_t panel_bytes = ctx->blocks_per_row * ctx->panel_rows * ctx->block_bytes;
    for (size_t panel = panel_start; panel < panel_end; ++panel) {
        const size_t row0 = panel * ctx->panel_rows;
        const size_t rows_this = (row0 + ctx->panel_rows <= ctx->row_count) ? ctx->panel_rows : (ctx->row_count - row0);
        const uint8_t *panel_ptr_a = ctx->weight_bytes_a + panel * panel_bytes;
        const uint8_t *panel_ptr_b = ctx->weight_bytes_b + panel * panel_bytes;

        float acc_a[CPU_QUANT_MATMUL_MR][CPU_QUANT_MATMUL_NR] = {{0}};
        float acc_b[CPU_QUANT_MATMUL_MR][CPU_QUANT_MATMUL_NR] = {{0}};
        for (size_t block = 0; block < ctx->blocks_per_row; ++block) {
            const cpu_q4_k_row_panel_decoded_block_t *panel_block_a =
                (const cpu_q4_k_row_panel_decoded_block_t *)(panel_ptr_a + block * ctx->panel_rows * ctx->block_bytes);
            const cpu_q4_k_row_panel_decoded_block_t *panel_block_b =
                (const cpu_q4_k_row_panel_decoded_block_t *)(panel_ptr_b + block * ctx->panel_rows * ctx->block_bytes);

            for (size_t col = 0; col < ctx->cols_in_tile; ++col) {
                const marmot_q8_k_block_t *a_block = cpu_quant_matmul_row_panel_activation_block_dual(ctx, block, col);
                const float act_d = a_block->d;
                const int16_t *bsums = a_block->bsums;
                for (size_t row = 0; row < rows_this; ++row) {
                    const cpu_q4_k_row_panel_decoded_block_t *w_block_a = panel_block_a + row;
                    const cpu_q4_k_row_panel_decoded_block_t *w_block_b = panel_block_b + row;
                    const int32_t block_sum_a =
                        cpu_quant_q4_k_block_dotprod_qs(w_block_a->qs, a_block->qs, w_block_a->scales);
                    const int32_t sumi_a = cpu_quant_q4_k_compute_sumi(bsums, w_block_a->mins);
                    acc_a[row][col] += act_d * (w_block_a->d * (float)block_sum_a - w_block_a->dmin * (float)sumi_a);

                    const int32_t block_sum_b =
                        cpu_quant_q4_k_block_dotprod_qs(w_block_b->qs, a_block->qs, w_block_b->scales);
                    const int32_t sumi_b = cpu_quant_q4_k_compute_sumi(bsums, w_block_b->mins);
                    acc_b[row][col] += act_d * (w_block_b->d * (float)block_sum_b - w_block_b->dmin * (float)sumi_b);
                }
            }
        }

        for (size_t col = 0; col < ctx->cols_in_tile; ++col) {
            float *out_a_col_f32 =
                ctx->out_a_data_f32 != nullptr ? ctx->out_a_data_f32 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            marmot_float16_t *out_a_col_f16 =
                ctx->out_a_data_f16 != nullptr ? ctx->out_a_data_f16 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            float *out_b_col_f32 =
                ctx->out_b_data_f32 != nullptr ? ctx->out_b_data_f32 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            marmot_float16_t *out_b_col_f16 =
                ctx->out_b_data_f16 != nullptr ? ctx->out_b_data_f16 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            for (size_t row = 0; row < rows_this; ++row) {
                const size_t row_idx = row0 + row;
                if (out_a_col_f32 != nullptr) {
                    out_a_col_f32[row_idx * ctx->out_stride_n] = acc_a[row][col];
                }
                if (out_a_col_f16 != nullptr) {
                    out_a_col_f16[row_idx * ctx->out_stride_n] = marmot_native_to_float16((_Float16)acc_a[row][col]);
                }
                if (out_b_col_f32 != nullptr) {
                    out_b_col_f32[row_idx * ctx->out_stride_n] = acc_b[row][col];
                }
                if (out_b_col_f16 != nullptr) {
                    out_b_col_f16[row_idx * ctx->out_stride_n] = marmot_native_to_float16((_Float16)acc_b[row][col]);
                }
            }
        }
    }
}

static void cpu_quant_matmul_compute_row_panel_dual_q6_k(
    const cpu_quant_matmul_row_panel_dual_ctx_t *ctx, size_t panel_start, size_t panel_end
) {
    const size_t panel_bytes = ctx->blocks_per_row * ctx->panel_rows * ctx->block_bytes;
    for (size_t panel = panel_start; panel < panel_end; ++panel) {
        const size_t row0 = panel * ctx->panel_rows;
        const size_t rows_this = (row0 + ctx->panel_rows <= ctx->row_count) ? ctx->panel_rows : (ctx->row_count - row0);
        const uint8_t *panel_ptr_a = ctx->weight_bytes_a + panel * panel_bytes;
        const uint8_t *panel_ptr_b = ctx->weight_bytes_b + panel * panel_bytes;

        float acc_a[CPU_QUANT_MATMUL_MR][CPU_QUANT_MATMUL_NR] = {{0}};
        float acc_b[CPU_QUANT_MATMUL_MR][CPU_QUANT_MATMUL_NR] = {{0}};
        for (size_t block = 0; block < ctx->blocks_per_row; ++block) {
            const uint8_t *panel_block_a = panel_ptr_a + block * ctx->panel_rows * ctx->block_bytes;
            const uint8_t *panel_block_b = panel_ptr_b + block * ctx->panel_rows * ctx->block_bytes;
            float w_d_a[CPU_QUANT_MATMUL_MR];
            float w_d_b[CPU_QUANT_MATMUL_MR];
            const marmot_q6_k_block_t *w_blocks_a[CPU_QUANT_MATMUL_MR];
            const marmot_q6_k_block_t *w_blocks_b[CPU_QUANT_MATMUL_MR];
            for (size_t row = 0; row < rows_this; ++row) {
                const marmot_q6_k_block_t *w_block_a =
                    (const marmot_q6_k_block_t *)(panel_block_a + row * ctx->block_bytes);
                const marmot_q6_k_block_t *w_block_b =
                    (const marmot_q6_k_block_t *)(panel_block_b + row * ctx->block_bytes);
                w_blocks_a[row] = w_block_a;
                w_blocks_b[row] = w_block_b;
                w_d_a[row] = (float)marmot_float16_to_native(w_block_a->d);
                w_d_b[row] = (float)marmot_float16_to_native(w_block_b->d);
            }

            for (size_t col = 0; col < ctx->cols_in_tile; ++col) {
                const marmot_q8_k_block_t *a_block = cpu_quant_matmul_row_panel_activation_block_dual(ctx, block, col);
                const float act_d = a_block->d;
                for (size_t row = 0; row < rows_this; ++row) {
                    const int32_t block_sum_a = cpu_quant_q6_k_block_dotprod(w_blocks_a[row], a_block->qs);
                    const int32_t block_sum_b = cpu_quant_q6_k_block_dotprod(w_blocks_b[row], a_block->qs);
                    acc_a[row][col] += act_d * w_d_a[row] * (float)block_sum_a;
                    acc_b[row][col] += act_d * w_d_b[row] * (float)block_sum_b;
                }
            }
        }

        for (size_t col = 0; col < ctx->cols_in_tile; ++col) {
            float *out_a_col_f32 =
                ctx->out_a_data_f32 != nullptr ? ctx->out_a_data_f32 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            marmot_float16_t *out_a_col_f16 =
                ctx->out_a_data_f16 != nullptr ? ctx->out_a_data_f16 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            float *out_b_col_f32 =
                ctx->out_b_data_f32 != nullptr ? ctx->out_b_data_f32 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            marmot_float16_t *out_b_col_f16 =
                ctx->out_b_data_f16 != nullptr ? ctx->out_b_data_f16 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            for (size_t row = 0; row < rows_this; ++row) {
                const size_t row_idx = row0 + row;
                if (out_a_col_f32 != nullptr) {
                    out_a_col_f32[row_idx * ctx->out_stride_n] = acc_a[row][col];
                }
                if (out_a_col_f16 != nullptr) {
                    out_a_col_f16[row_idx * ctx->out_stride_n] = marmot_native_to_float16((_Float16)acc_a[row][col]);
                }
                if (out_b_col_f32 != nullptr) {
                    out_b_col_f32[row_idx * ctx->out_stride_n] = acc_b[row][col];
                }
                if (out_b_col_f16 != nullptr) {
                    out_b_col_f16[row_idx * ctx->out_stride_n] = marmot_native_to_float16((_Float16)acc_b[row][col]);
                }
            }
        }
    }
}

static void cpu_quant_matmul_compute_row_panel_dual_q6_k_decoded(
    const cpu_quant_matmul_row_panel_dual_ctx_t *ctx, size_t panel_start, size_t panel_end
) {
    const size_t panel_bytes = ctx->blocks_per_row * ctx->panel_rows * ctx->block_bytes;
    for (size_t panel = panel_start; panel < panel_end; ++panel) {
        const size_t row0 = panel * ctx->panel_rows;
        const size_t rows_this = (row0 + ctx->panel_rows <= ctx->row_count) ? ctx->panel_rows : (ctx->row_count - row0);
        const uint8_t *panel_ptr_a = ctx->weight_bytes_a + panel * panel_bytes;
        const uint8_t *panel_ptr_b = ctx->weight_bytes_b + panel * panel_bytes;

        float acc_a[CPU_QUANT_MATMUL_MR][CPU_QUANT_MATMUL_NR] = {{0}};
        float acc_b[CPU_QUANT_MATMUL_MR][CPU_QUANT_MATMUL_NR] = {{0}};
        for (size_t block = 0; block < ctx->blocks_per_row; ++block) {
            const cpu_q6_k_row_panel_decoded_block_t *panel_block_a =
                (const cpu_q6_k_row_panel_decoded_block_t *)(panel_ptr_a + block * ctx->panel_rows * ctx->block_bytes);
            const cpu_q6_k_row_panel_decoded_block_t *panel_block_b =
                (const cpu_q6_k_row_panel_decoded_block_t *)(panel_ptr_b + block * ctx->panel_rows * ctx->block_bytes);

            for (size_t col = 0; col < ctx->cols_in_tile; ++col) {
                const marmot_q8_k_block_t *a_block = cpu_quant_matmul_row_panel_activation_block_dual(ctx, block, col);
                const float act_d = a_block->d;
                for (size_t row = 0; row < rows_this; ++row) {
                    const cpu_q6_k_row_panel_decoded_block_t *w_block_a = panel_block_a + row;
                    const cpu_q6_k_row_panel_decoded_block_t *w_block_b = panel_block_b + row;
                    const int32_t block_sum_a = cpu_quant_q6_k_decoded_block_dotprod(w_block_a, a_block->qs);
                    const int32_t block_sum_b = cpu_quant_q6_k_decoded_block_dotprod(w_block_b, a_block->qs);
                    acc_a[row][col] += act_d * w_block_a->d * (float)block_sum_a;
                    acc_b[row][col] += act_d * w_block_b->d * (float)block_sum_b;
                }
            }
        }

        for (size_t col = 0; col < ctx->cols_in_tile; ++col) {
            float *out_a_col_f32 =
                ctx->out_a_data_f32 != nullptr ? ctx->out_a_data_f32 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            marmot_float16_t *out_a_col_f16 =
                ctx->out_a_data_f16 != nullptr ? ctx->out_a_data_f16 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            float *out_b_col_f32 =
                ctx->out_b_data_f32 != nullptr ? ctx->out_b_data_f32 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            marmot_float16_t *out_b_col_f16 =
                ctx->out_b_data_f16 != nullptr ? ctx->out_b_data_f16 + (ctx->n0 + col) * ctx->out_stride_m : nullptr;
            for (size_t row = 0; row < rows_this; ++row) {
                const size_t row_idx = row0 + row;
                if (out_a_col_f32 != nullptr) {
                    out_a_col_f32[row_idx * ctx->out_stride_n] = acc_a[row][col];
                }
                if (out_a_col_f16 != nullptr) {
                    out_a_col_f16[row_idx * ctx->out_stride_n] = marmot_native_to_float16((_Float16)acc_a[row][col]);
                }
                if (out_b_col_f32 != nullptr) {
                    out_b_col_f32[row_idx * ctx->out_stride_n] = acc_b[row][col];
                }
                if (out_b_col_f16 != nullptr) {
                    out_b_col_f16[row_idx * ctx->out_stride_n] = marmot_native_to_float16((_Float16)acc_b[row][col]);
                }
            }
        }
    }
}

static void cpu_quant_matmul_row_panel_dispatch_range(void *ctx, size_t panel_start, size_t panel_end) {
    const cpu_quant_matmul_row_panel_ctx_t *dctx = (const cpu_quant_matmul_row_panel_ctx_t *)ctx;
    switch (dctx->quant_kind) {
    case MARMOT_QUANT_KIND_Q4_K:
        cpu_quant_matmul_compute_row_panel_q4_k(dctx, panel_start, panel_end);
        return;
    case MARMOT_QUANT_KIND_Q6_K:
        cpu_quant_matmul_compute_row_panel_q6_k(dctx, panel_start, panel_end);
        return;
    default:
        return;
    }
}

static void cpu_quant_matmul_row_panel_dispatch_range_dual(void *ctx, size_t panel_start, size_t panel_end) {
    const cpu_quant_matmul_row_panel_dual_ctx_t *dctx = (const cpu_quant_matmul_row_panel_dual_ctx_t *)ctx;
    switch (dctx->quant_kind) {
    case MARMOT_QUANT_KIND_Q4_K:
        cpu_quant_matmul_compute_row_panel_dual_q4_k(dctx, panel_start, panel_end);
        return;
    case MARMOT_QUANT_KIND_Q6_K:
        cpu_quant_matmul_compute_row_panel_dual_q6_k(dctx, panel_start, panel_end);
        return;
    default:
        return;
    }
}

static void cpu_quant_matmul_q4_k_decoded_row_panel_dispatch_range(void *ctx, size_t panel_start, size_t panel_end) {
    cpu_quant_matmul_compute_row_panel_q4_k_decoded(
        (const cpu_quant_matmul_row_panel_ctx_t *)ctx, panel_start, panel_end
    );
}

static void
cpu_quant_matmul_q4_k_decoded_row_panel_dispatch_range_dual(void *ctx, size_t panel_start, size_t panel_end) {
    cpu_quant_matmul_compute_row_panel_dual_q4_k_decoded(
        (const cpu_quant_matmul_row_panel_dual_ctx_t *)ctx, panel_start, panel_end
    );
}

static void cpu_quant_matmul_q6_k_decoded_row_panel_dispatch_range(void *ctx, size_t panel_start, size_t panel_end) {
    cpu_quant_matmul_compute_row_panel_q6_k_decoded(
        (const cpu_quant_matmul_row_panel_ctx_t *)ctx, panel_start, panel_end
    );
}

static void
cpu_quant_matmul_q6_k_decoded_row_panel_dispatch_range_dual(void *ctx, size_t panel_start, size_t panel_end) {
    cpu_quant_matmul_compute_row_panel_dual_q6_k_decoded(
        (const cpu_quant_matmul_row_panel_dual_ctx_t *)ctx, panel_start, panel_end
    );
}

typedef struct {
    const cpu_matmul_quant_kernel_t *kernel;
    const uint8_t *weight_bytes;
    size_t row_bytes;
    size_t blocks_per_row;
    const marmot_float16_t *const *activation_rows;
    size_t act_stride_k;
    size_t cols_in_tile;
    size_t n0;
    float *out_data_f32;
    marmot_float16_t *out_data_f16;
    size_t out_stride_m;
    size_t out_stride_n;
    size_t K;
} cpu_quant_matmul_fp16_dispatch_ctx_t;

typedef struct {
    const cpu_matmul_quant_kernel_t *kernel;
    const uint8_t *weight_bytes;
    size_t row_bytes;
    size_t blocks_per_row;
    const marmot_float16_t *const *activation_rows;
    size_t act_stride_k;
    size_t cols_in_tile;
    size_t n0;
    float *out_data_f32;
    marmot_float16_t *out_data_f16;
    size_t out_stride_m;
    size_t out_stride_n;
    size_t row_start;
    size_t row_end;
    size_t K;
} cpu_quant_matmul_fp16_worker_args_t;

static void cpu_quant_matmul_compute_rows_fp16(const cpu_quant_matmul_fp16_worker_args_t *args) {
    const cpu_matmul_quant_kernel_t *kernel = args->kernel;
    const uint8_t *weight_bytes = args->weight_bytes;
    const size_t row_bytes = args->row_bytes;
    const size_t blocks_per_row = args->blocks_per_row;
    const size_t cols_in_tile = args->cols_in_tile;
    const size_t n0 = args->n0;
    float *out_data_f32 = args->out_data_f32;
    marmot_float16_t *out_data_f16 = args->out_data_f16;
    const size_t out_stride_m = args->out_stride_m;
    const size_t out_stride_n = args->out_stride_n;
    const size_t K = args->K;

    for (size_t c = 0; c < cols_in_tile; ++c) {
        const marmot_float16_t *act = args->activation_rows[c];
        float *out_col_f32 = out_data_f32 != nullptr ? out_data_f32 + (n0 + c) * out_stride_m : nullptr;
        marmot_float16_t *out_col_f16 = out_data_f16 != nullptr ? out_data_f16 + (n0 + c) * out_stride_m : nullptr;

        for (size_t m = args->row_start; m < args->row_end; ++m) {
            const uint8_t *row_ptr = weight_bytes + m * row_bytes;
            const float dot = kernel->ops.dot_fp16(row_ptr, act, args->act_stride_k, blocks_per_row, K);
            if (out_col_f32 != nullptr) {
                out_col_f32[m * out_stride_n] = dot;
            }
            if (out_col_f16 != nullptr) {
                out_col_f16[m * out_stride_n] = marmot_native_to_float16((_Float16)dot);
            }
        }
    }
}

static void cpu_quant_matmul_fp16_dispatch_range(void *ctx, size_t row_start, size_t row_end) {
    const cpu_quant_matmul_fp16_dispatch_ctx_t *c = (const cpu_quant_matmul_fp16_dispatch_ctx_t *)ctx;
    cpu_quant_matmul_fp16_worker_args_t args = {
        .kernel = c->kernel,
        .weight_bytes = c->weight_bytes,
        .row_bytes = c->row_bytes,
        .blocks_per_row = c->blocks_per_row,
        .activation_rows = c->activation_rows,
        .act_stride_k = c->act_stride_k,
        .cols_in_tile = c->cols_in_tile,
        .n0 = c->n0,
        .out_data_f32 = c->out_data_f32,
        .out_data_f16 = c->out_data_f16,
        .out_stride_m = c->out_stride_m,
        .out_stride_n = c->out_stride_n,
        .row_start = row_start,
        .row_end = row_end,
        .K = c->K,
    };
    cpu_quant_matmul_compute_rows_fp16(&args);
}

void cpu_quant_matmul_dispatch_fp16_tile(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes, size_t row_bytes, size_t blocks_per_row,
    const marmot_float16_t *const *activation_rows, size_t act_stride_k, size_t cols_in_tile, size_t n0,
    float *out_data_f32, marmot_float16_t *out_data_f16, size_t out_stride_m, size_t out_stride_n, size_t row_count,
    size_t K, size_t thread_cap
) {
    (void)thread_cap;

    cpu_quant_matmul_fp16_dispatch_ctx_t dctx = {
        .kernel = kernel,
        .weight_bytes = weight_bytes,
        .row_bytes = row_bytes,
        .blocks_per_row = blocks_per_row,
        .activation_rows = activation_rows,
        .act_stride_k = act_stride_k,
        .cols_in_tile = cols_in_tile,
        .n0 = n0,
        .out_data_f32 = out_data_f32,
        .out_data_f16 = out_data_f16,
        .out_stride_m = out_stride_m,
        .out_stride_n = out_stride_n,
        .K = K,
    };
    marmot_dispatch_parallel_for_range(
        MARMOT_DISPATCH_PRIORITY_HIGH, row_count, 64, &dctx, cpu_quant_matmul_fp16_dispatch_range
    );
}

void cpu_quant_matmul_dispatch_fp16_tile_dual(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes_a, const uint8_t *weight_bytes_b,
    size_t row_bytes, size_t blocks_per_row, const marmot_float16_t *const *activation_rows, size_t act_stride_k,
    size_t cols_in_tile, size_t n0, float *out_a_data_f32, marmot_float16_t *out_a_data_f16, float *out_b_data_f32,
    marmot_float16_t *out_b_data_f16, size_t out_stride_m, size_t out_stride_n, size_t row_count, size_t K,
    size_t thread_cap
) {
    cpu_quant_matmul_dispatch_fp16_tile(
        kernel, weight_bytes_a, row_bytes, blocks_per_row, activation_rows, act_stride_k, cols_in_tile, n0,
        out_a_data_f32, out_a_data_f16, out_stride_m, out_stride_n, row_count, K, thread_cap
    );
    cpu_quant_matmul_dispatch_fp16_tile(
        kernel, weight_bytes_b, row_bytes, blocks_per_row, activation_rows, act_stride_k, cols_in_tile, n0,
        out_b_data_f32, out_b_data_f16, out_stride_m, out_stride_n, row_count, K, thread_cap
    );
}

void cpu_quant_matmul_dispatch_quant_tile(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes, size_t row_bytes, size_t blocks_per_row,
    const marmot_q8_0_block_t *activation_q8_0, const marmot_q8_k_block_t *activation_q8_k,
    const marmot_q8_k_block_t *activation_panel_q8_k, size_t activation_panel_cols, size_t cols_in_tile, size_t n0,
    float *out_data_f32, marmot_float16_t *out_data_f16, size_t out_stride_m, size_t out_stride_n, size_t row_start,
    size_t row_end, size_t thread_cap, marmot_quant_kind_t quant_kind
) {
    cpu_quant_matmul_dispatch_ctx_t dctx = {
        .kernel = kernel,
        .weight_bytes = weight_bytes,
        .row_bytes = row_bytes,
        .blocks_per_row = blocks_per_row,
        .activation_tile_q8_0 = activation_q8_0,
        .activation_tile_q8_k = activation_q8_k,
        .activation_panel_q8_k = activation_panel_q8_k,
        .activation_panel_cols = activation_panel_cols,
        .activation_panel_col0 = 0,
        .cols_in_tile = cols_in_tile,
        .n0 = n0,
        .out_data_f32 = out_data_f32,
        .out_data_f16 = out_data_f16,
        .block_bytes = kernel->format != nullptr ? kernel->format->block_bytes : 0,
        .out_stride_m = out_stride_m,
        .out_stride_n = out_stride_n,
        .row_base = row_start,
        .total_row_count = row_end - row_start,
        .quant_kind = quant_kind,
    };
    const size_t row_grain = cpu_quant_matmul_row_grain(row_end - row_start, cols_in_tile, thread_cap);
    marmot_dispatch_parallel_for_range(
        MARMOT_DISPATCH_PRIORITY_HIGH, row_end - row_start, row_grain, &dctx, cpu_quant_matmul_dispatch_range
    );
}

void cpu_quant_matmul_dispatch_quant_tile_dual(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes_a, const uint8_t *weight_bytes_b,
    size_t row_bytes, size_t blocks_per_row, const marmot_q8_0_block_t *activation_q8_0,
    const marmot_q8_k_block_t *activation_q8_k, const marmot_q8_k_block_t *activation_panel_q8_k,
    size_t activation_panel_cols, size_t cols_in_tile, size_t n0, float *out_a_data_f32,
    marmot_float16_t *out_a_data_f16, float *out_b_data_f32, marmot_float16_t *out_b_data_f16, size_t out_stride_m,
    size_t out_stride_n, size_t row_start, size_t row_end, size_t thread_cap, marmot_quant_kind_t quant_kind
) {
    const bool use_q8_k = kernel != nullptr && kernel->format != nullptr &&
        kernel->format->activation_packer == MARMOT_MATMUL_ACTIVATION_PACKER_Q8_K && kernel->ops.dot_q8_k != nullptr &&
        activation_q8_k != nullptr;
#if !(MARMOT_ENABLE_NEON && defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD))
    (void)use_q8_k;
#endif
    if (!use_q8_k) {
        cpu_quant_matmul_dispatch_quant_tile(
            kernel, weight_bytes_a, row_bytes, blocks_per_row, activation_q8_0, activation_q8_k, activation_panel_q8_k,
            activation_panel_cols, cols_in_tile, n0, out_a_data_f32, out_a_data_f16, out_stride_m, out_stride_n,
            row_start, row_end, thread_cap, quant_kind
        );
        cpu_quant_matmul_dispatch_quant_tile(
            kernel, weight_bytes_b, row_bytes, blocks_per_row, activation_q8_0, activation_q8_k, activation_panel_q8_k,
            activation_panel_cols, cols_in_tile, n0, out_b_data_f32, out_b_data_f16, out_stride_m, out_stride_n,
            row_start, row_end, thread_cap, quant_kind
        );
        return;
    }

#if MARMOT_ENABLE_NEON && defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    cpu_quant_matmul_dual_dispatch_ctx_t dctx = {
        .kernel = kernel,
        .weight_bytes_a = weight_bytes_a,
        .weight_bytes_b = weight_bytes_b,
        .row_bytes = row_bytes,
        .blocks_per_row = blocks_per_row,
        .activation_tile_q8_0 = activation_q8_0,
        .activation_tile_q8_k = activation_q8_k,
        .activation_panel_q8_k = activation_panel_q8_k,
        .activation_panel_cols = activation_panel_cols,
        .activation_panel_col0 = 0,
        .cols_in_tile = cols_in_tile,
        .n0 = n0,
        .out_a_data_f32 = out_a_data_f32,
        .out_a_data_f16 = out_a_data_f16,
        .out_b_data_f32 = out_b_data_f32,
        .out_b_data_f16 = out_b_data_f16,
        .block_bytes = kernel->format != nullptr ? kernel->format->block_bytes : 0,
        .out_stride_m = out_stride_m,
        .out_stride_n = out_stride_n,
        .row_base = row_start,
        .total_row_count = row_end - row_start,
        .quant_kind = quant_kind,
    };
    const size_t row_grain = cpu_quant_matmul_row_grain(row_end - row_start, cols_in_tile, thread_cap);
    marmot_dispatch_parallel_for_range(
        MARMOT_DISPATCH_PRIORITY_HIGH, row_end - row_start, row_grain, &dctx, cpu_quant_matmul_dispatch_range_dual
    );
#else
    cpu_quant_matmul_dispatch_quant_tile(
        kernel, weight_bytes_a, row_bytes, blocks_per_row, activation_q8_0, activation_q8_k, activation_panel_q8_k,
        activation_panel_cols, cols_in_tile, n0, out_a_data_f32, out_a_data_f16, out_stride_m, out_stride_n, row_start,
        row_end, thread_cap, quant_kind
    );
    cpu_quant_matmul_dispatch_quant_tile(
        kernel, weight_bytes_b, row_bytes, blocks_per_row, activation_q8_0, activation_q8_k, activation_panel_q8_k,
        activation_panel_cols, cols_in_tile, n0, out_b_data_f32, out_b_data_f16, out_stride_m, out_stride_n, row_start,
        row_end, thread_cap, quant_kind
    );
#endif
}

void cpu_quant_matmul_dispatch_quant_tile_row_panel(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes, size_t row_count, size_t row_bytes,
    size_t blocks_per_row, const marmot_q8_k_block_t *activation_q8_k, const marmot_q8_k_block_t *activation_panel_q8_k,
    size_t activation_panel_cols, size_t cols_in_tile, size_t n0, float *out_data_f32, marmot_float16_t *out_data_f16,
    size_t out_stride_m, size_t out_stride_n, size_t panel_rows, marmot_quant_kind_t quant_kind
) {
    if (kernel == nullptr || weight_bytes == nullptr || row_count == 0 || row_bytes == 0 || blocks_per_row == 0 ||
        cols_in_tile == 0 || panel_rows == 0) {
        return;
    }
    cpu_quant_matmul_row_panel_ctx_t ctx = (cpu_quant_matmul_row_panel_ctx_t){
        .kernel = kernel,
        .weight_bytes = weight_bytes,
        .row_count = row_count,
        .row_bytes = row_bytes,
        .blocks_per_row = blocks_per_row,
        .activation_tile_q8_k = activation_q8_k,
        .activation_panel_q8_k = activation_panel_q8_k,
        .activation_panel_cols = activation_panel_cols,
        .cols_in_tile = cols_in_tile,
        .n0 = n0,
        .out_data_f32 = out_data_f32,
        .out_data_f16 = out_data_f16,
        .out_stride_m = out_stride_m,
        .out_stride_n = out_stride_n,
        .panel_rows = panel_rows,
        .block_bytes = kernel->format != nullptr ? kernel->format->block_bytes : 0,
        .quant_kind = quant_kind,
    };
    const size_t panel_count = (row_count + panel_rows - 1) / panel_rows;
    marmot_dispatch_parallel_for_range(
        MARMOT_DISPATCH_PRIORITY_HIGH, panel_count, 8, &ctx, cpu_quant_matmul_row_panel_dispatch_range
    );
}

void cpu_quant_matmul_dispatch_quant_tile_row_panel_dual(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes_a, const uint8_t *weight_bytes_b,
    size_t row_count, size_t row_bytes, size_t blocks_per_row, const marmot_q8_k_block_t *activation_q8_k,
    const marmot_q8_k_block_t *activation_panel_q8_k, size_t activation_panel_cols, size_t cols_in_tile, size_t n0,
    float *out_a_data_f32, marmot_float16_t *out_a_data_f16, float *out_b_data_f32, marmot_float16_t *out_b_data_f16,
    size_t out_stride_m, size_t out_stride_n, size_t panel_rows, marmot_quant_kind_t quant_kind
) {
    if (kernel == nullptr || weight_bytes_a == nullptr || weight_bytes_b == nullptr || row_count == 0 ||
        row_bytes == 0 || blocks_per_row == 0 || cols_in_tile == 0 || panel_rows == 0) {
        return;
    }
    cpu_quant_matmul_row_panel_dual_ctx_t ctx = (cpu_quant_matmul_row_panel_dual_ctx_t){
        .kernel = kernel,
        .weight_bytes_a = weight_bytes_a,
        .weight_bytes_b = weight_bytes_b,
        .row_count = row_count,
        .row_bytes = row_bytes,
        .blocks_per_row = blocks_per_row,
        .activation_tile_q8_k = activation_q8_k,
        .activation_panel_q8_k = activation_panel_q8_k,
        .activation_panel_cols = activation_panel_cols,
        .cols_in_tile = cols_in_tile,
        .n0 = n0,
        .out_a_data_f32 = out_a_data_f32,
        .out_a_data_f16 = out_a_data_f16,
        .out_b_data_f32 = out_b_data_f32,
        .out_b_data_f16 = out_b_data_f16,
        .out_stride_m = out_stride_m,
        .out_stride_n = out_stride_n,
        .panel_rows = panel_rows,
        .block_bytes = kernel->format != nullptr ? kernel->format->block_bytes : 0,
        .quant_kind = quant_kind,
    };
    const size_t panel_count = (row_count + panel_rows - 1) / panel_rows;
    marmot_dispatch_parallel_for_range(
        MARMOT_DISPATCH_PRIORITY_HIGH, panel_count, 8, &ctx, cpu_quant_matmul_row_panel_dispatch_range_dual
    );
}

void cpu_quant_matmul_dispatch_quant_tile_q4_k_row_panel_decoded(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes, size_t row_count, size_t blocks_per_row,
    const marmot_q8_k_block_t *activation_q8_k, const marmot_q8_k_block_t *activation_panel_q8_k,
    size_t activation_panel_cols, size_t cols_in_tile, size_t n0, float *out_data_f32, marmot_float16_t *out_data_f16,
    size_t out_stride_m, size_t out_stride_n, size_t panel_rows
) {
    if (kernel == nullptr || weight_bytes == nullptr || row_count == 0 || blocks_per_row == 0 || cols_in_tile == 0 ||
        panel_rows == 0) {
        return;
    }
    cpu_quant_matmul_row_panel_ctx_t ctx = (cpu_quant_matmul_row_panel_ctx_t){
        .kernel = kernel,
        .weight_bytes = weight_bytes,
        .row_count = row_count,
        .row_bytes = 0,
        .blocks_per_row = blocks_per_row,
        .activation_tile_q8_k = activation_q8_k,
        .activation_panel_q8_k = activation_panel_q8_k,
        .activation_panel_cols = activation_panel_cols,
        .cols_in_tile = cols_in_tile,
        .n0 = n0,
        .out_data_f32 = out_data_f32,
        .out_data_f16 = out_data_f16,
        .out_stride_m = out_stride_m,
        .out_stride_n = out_stride_n,
        .panel_rows = panel_rows,
        .block_bytes = sizeof(cpu_q4_k_row_panel_decoded_block_t),
        .quant_kind = MARMOT_QUANT_KIND_Q4_K,
    };
    const size_t panel_count = (row_count + panel_rows - 1) / panel_rows;
    marmot_dispatch_parallel_for_range(
        MARMOT_DISPATCH_PRIORITY_HIGH, panel_count, 8, &ctx, cpu_quant_matmul_q4_k_decoded_row_panel_dispatch_range
    );
}

void cpu_quant_matmul_dispatch_quant_tile_q4_k_row_panel_decoded_dual(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes_a, const uint8_t *weight_bytes_b,
    size_t row_count, size_t blocks_per_row, const marmot_q8_k_block_t *activation_q8_k,
    const marmot_q8_k_block_t *activation_panel_q8_k, size_t activation_panel_cols, size_t cols_in_tile, size_t n0,
    float *out_a_data_f32, marmot_float16_t *out_a_data_f16, float *out_b_data_f32, marmot_float16_t *out_b_data_f16,
    size_t out_stride_m, size_t out_stride_n, size_t panel_rows
) {
    if (kernel == nullptr || weight_bytes_a == nullptr || weight_bytes_b == nullptr || row_count == 0 ||
        blocks_per_row == 0 || cols_in_tile == 0 || panel_rows == 0) {
        return;
    }
    cpu_quant_matmul_row_panel_dual_ctx_t ctx = (cpu_quant_matmul_row_panel_dual_ctx_t){
        .kernel = kernel,
        .weight_bytes_a = weight_bytes_a,
        .weight_bytes_b = weight_bytes_b,
        .row_count = row_count,
        .row_bytes = 0,
        .blocks_per_row = blocks_per_row,
        .activation_tile_q8_k = activation_q8_k,
        .activation_panel_q8_k = activation_panel_q8_k,
        .activation_panel_cols = activation_panel_cols,
        .cols_in_tile = cols_in_tile,
        .n0 = n0,
        .out_a_data_f32 = out_a_data_f32,
        .out_a_data_f16 = out_a_data_f16,
        .out_b_data_f32 = out_b_data_f32,
        .out_b_data_f16 = out_b_data_f16,
        .out_stride_m = out_stride_m,
        .out_stride_n = out_stride_n,
        .panel_rows = panel_rows,
        .block_bytes = sizeof(cpu_q4_k_row_panel_decoded_block_t),
        .quant_kind = MARMOT_QUANT_KIND_Q4_K,
    };
    const size_t panel_count = (row_count + panel_rows - 1) / panel_rows;
    marmot_dispatch_parallel_for_range(
        MARMOT_DISPATCH_PRIORITY_HIGH, panel_count, 8, &ctx, cpu_quant_matmul_q4_k_decoded_row_panel_dispatch_range_dual
    );
}

void cpu_quant_matmul_dispatch_quant_tile_q6_k_row_panel_decoded(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes, size_t row_count, size_t blocks_per_row,
    const marmot_q8_k_block_t *activation_q8_k, const marmot_q8_k_block_t *activation_panel_q8_k,
    size_t activation_panel_cols, size_t cols_in_tile, size_t n0, float *out_data_f32, marmot_float16_t *out_data_f16,
    size_t out_stride_m, size_t out_stride_n, size_t panel_rows
) {
    if (kernel == nullptr || weight_bytes == nullptr || row_count == 0 || blocks_per_row == 0 || cols_in_tile == 0 ||
        panel_rows == 0) {
        return;
    }
    cpu_quant_matmul_row_panel_ctx_t ctx = (cpu_quant_matmul_row_panel_ctx_t){
        .kernel = kernel,
        .weight_bytes = weight_bytes,
        .row_count = row_count,
        .row_bytes = 0,
        .blocks_per_row = blocks_per_row,
        .activation_tile_q8_k = activation_q8_k,
        .activation_panel_q8_k = activation_panel_q8_k,
        .activation_panel_cols = activation_panel_cols,
        .cols_in_tile = cols_in_tile,
        .n0 = n0,
        .out_data_f32 = out_data_f32,
        .out_data_f16 = out_data_f16,
        .out_stride_m = out_stride_m,
        .out_stride_n = out_stride_n,
        .panel_rows = panel_rows,
        .block_bytes = sizeof(cpu_q6_k_row_panel_decoded_block_t),
        .quant_kind = MARMOT_QUANT_KIND_Q6_K,
    };
    const size_t panel_count = (row_count + panel_rows - 1) / panel_rows;
    marmot_dispatch_parallel_for_range(
        MARMOT_DISPATCH_PRIORITY_HIGH, panel_count, 8, &ctx, cpu_quant_matmul_q6_k_decoded_row_panel_dispatch_range
    );
}

void cpu_quant_matmul_dispatch_quant_tile_q6_k_row_panel_decoded_dual(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes_a, const uint8_t *weight_bytes_b,
    size_t row_count, size_t blocks_per_row, const marmot_q8_k_block_t *activation_q8_k,
    const marmot_q8_k_block_t *activation_panel_q8_k, size_t activation_panel_cols, size_t cols_in_tile, size_t n0,
    float *out_a_data_f32, marmot_float16_t *out_a_data_f16, float *out_b_data_f32, marmot_float16_t *out_b_data_f16,
    size_t out_stride_m, size_t out_stride_n, size_t panel_rows
) {
    if (kernel == nullptr || weight_bytes_a == nullptr || weight_bytes_b == nullptr || row_count == 0 ||
        blocks_per_row == 0 || cols_in_tile == 0 || panel_rows == 0) {
        return;
    }
    cpu_quant_matmul_row_panel_dual_ctx_t ctx = (cpu_quant_matmul_row_panel_dual_ctx_t){
        .kernel = kernel,
        .weight_bytes_a = weight_bytes_a,
        .weight_bytes_b = weight_bytes_b,
        .row_count = row_count,
        .row_bytes = 0,
        .blocks_per_row = blocks_per_row,
        .activation_tile_q8_k = activation_q8_k,
        .activation_panel_q8_k = activation_panel_q8_k,
        .activation_panel_cols = activation_panel_cols,
        .cols_in_tile = cols_in_tile,
        .n0 = n0,
        .out_a_data_f32 = out_a_data_f32,
        .out_a_data_f16 = out_a_data_f16,
        .out_b_data_f32 = out_b_data_f32,
        .out_b_data_f16 = out_b_data_f16,
        .out_stride_m = out_stride_m,
        .out_stride_n = out_stride_n,
        .panel_rows = panel_rows,
        .block_bytes = sizeof(cpu_q6_k_row_panel_decoded_block_t),
        .quant_kind = MARMOT_QUANT_KIND_Q6_K,
    };
    const size_t panel_count = (row_count + panel_rows - 1) / panel_rows;
    marmot_dispatch_parallel_for_range(
        MARMOT_DISPATCH_PRIORITY_HIGH, panel_count, 8, &ctx, cpu_quant_matmul_q6_k_decoded_row_panel_dispatch_range_dual
    );
}

#else
void cpu_quant_matmul_dispatch_quant_tile_row_panel(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes, size_t row_count, size_t row_bytes,
    size_t blocks_per_row, const marmot_q8_k_block_t *activation_q8_k, const marmot_q8_k_block_t *activation_panel_q8_k,
    size_t activation_panel_cols, size_t cols_in_tile, size_t n0, float *out_data_f32, marmot_float16_t *out_data_f16,
    size_t out_stride_m, size_t out_stride_n, size_t panel_rows, marmot_quant_kind_t quant_kind
) {
    (void)kernel;
    (void)weight_bytes;
    (void)row_count;
    (void)row_bytes;
    (void)blocks_per_row;
    (void)activation_q8_k;
    (void)activation_panel_q8_k;
    (void)activation_panel_cols;
    (void)cols_in_tile;
    (void)n0;
    (void)out_data_f32;
    (void)out_data_f16;
    (void)out_stride_m;
    (void)out_stride_n;
    (void)panel_rows;
    (void)quant_kind;
}

void cpu_quant_matmul_dispatch_quant_tile_row_panel_dual(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes_a, const uint8_t *weight_bytes_b,
    size_t row_count, size_t row_bytes, size_t blocks_per_row, const marmot_q8_k_block_t *activation_q8_k,
    const marmot_q8_k_block_t *activation_panel_q8_k, size_t activation_panel_cols, size_t cols_in_tile, size_t n0,
    float *out_a_data_f32, marmot_float16_t *out_a_data_f16, float *out_b_data_f32, marmot_float16_t *out_b_data_f16,
    size_t out_stride_m, size_t out_stride_n, size_t panel_rows, marmot_quant_kind_t quant_kind
) {
    (void)kernel;
    (void)weight_bytes_a;
    (void)weight_bytes_b;
    (void)row_count;
    (void)row_bytes;
    (void)blocks_per_row;
    (void)activation_q8_k;
    (void)activation_panel_q8_k;
    (void)activation_panel_cols;
    (void)cols_in_tile;
    (void)n0;
    (void)out_a_data_f32;
    (void)out_a_data_f16;
    (void)out_b_data_f32;
    (void)out_b_data_f16;
    (void)out_stride_m;
    (void)out_stride_n;
    (void)panel_rows;
    (void)quant_kind;
}

void cpu_quant_matmul_dispatch_quant_tile_q4_k_row_panel_decoded(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes, size_t row_count, size_t blocks_per_row,
    const marmot_q8_k_block_t *activation_q8_k, const marmot_q8_k_block_t *activation_panel_q8_k,
    size_t activation_panel_cols, size_t cols_in_tile, size_t n0, float *out_data_f32, marmot_float16_t *out_data_f16,
    size_t out_stride_m, size_t out_stride_n, size_t panel_rows
) {
    (void)kernel;
    (void)weight_bytes;
    (void)row_count;
    (void)blocks_per_row;
    (void)activation_q8_k;
    (void)activation_panel_q8_k;
    (void)activation_panel_cols;
    (void)cols_in_tile;
    (void)n0;
    (void)out_data_f32;
    (void)out_data_f16;
    (void)out_stride_m;
    (void)out_stride_n;
    (void)panel_rows;
}

void cpu_quant_matmul_dispatch_quant_tile_q4_k_row_panel_decoded_dual(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes_a, const uint8_t *weight_bytes_b,
    size_t row_count, size_t blocks_per_row, const marmot_q8_k_block_t *activation_q8_k,
    const marmot_q8_k_block_t *activation_panel_q8_k, size_t activation_panel_cols, size_t cols_in_tile, size_t n0,
    float *out_a_data_f32, marmot_float16_t *out_a_data_f16, float *out_b_data_f32, marmot_float16_t *out_b_data_f16,
    size_t out_stride_m, size_t out_stride_n, size_t panel_rows
) {
    (void)kernel;
    (void)weight_bytes_a;
    (void)weight_bytes_b;
    (void)row_count;
    (void)blocks_per_row;
    (void)activation_q8_k;
    (void)activation_panel_q8_k;
    (void)activation_panel_cols;
    (void)cols_in_tile;
    (void)n0;
    (void)out_a_data_f32;
    (void)out_a_data_f16;
    (void)out_b_data_f32;
    (void)out_b_data_f16;
    (void)out_stride_m;
    (void)out_stride_n;
    (void)panel_rows;
}

void cpu_quant_matmul_dispatch_quant_tile_q6_k_row_panel_decoded(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes, size_t row_count, size_t blocks_per_row,
    const marmot_q8_k_block_t *activation_q8_k, const marmot_q8_k_block_t *activation_panel_q8_k,
    size_t activation_panel_cols, size_t cols_in_tile, size_t n0, float *out_data_f32, marmot_float16_t *out_data_f16,
    size_t out_stride_m, size_t out_stride_n, size_t panel_rows
) {
    (void)kernel;
    (void)weight_bytes;
    (void)row_count;
    (void)blocks_per_row;
    (void)activation_q8_k;
    (void)activation_panel_q8_k;
    (void)activation_panel_cols;
    (void)cols_in_tile;
    (void)n0;
    (void)out_data_f32;
    (void)out_data_f16;
    (void)out_stride_m;
    (void)out_stride_n;
    (void)panel_rows;
}

void cpu_quant_matmul_dispatch_quant_tile_q6_k_row_panel_decoded_dual(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes_a, const uint8_t *weight_bytes_b,
    size_t row_count, size_t blocks_per_row, const marmot_q8_k_block_t *activation_q8_k,
    const marmot_q8_k_block_t *activation_panel_q8_k, size_t activation_panel_cols, size_t cols_in_tile, size_t n0,
    float *out_a_data_f32, marmot_float16_t *out_a_data_f16, float *out_b_data_f32, marmot_float16_t *out_b_data_f16,
    size_t out_stride_m, size_t out_stride_n, size_t panel_rows
) {
    (void)kernel;
    (void)weight_bytes_a;
    (void)weight_bytes_b;
    (void)row_count;
    (void)blocks_per_row;
    (void)activation_q8_k;
    (void)activation_panel_q8_k;
    (void)activation_panel_cols;
    (void)cols_in_tile;
    (void)n0;
    (void)out_a_data_f32;
    (void)out_a_data_f16;
    (void)out_b_data_f32;
    (void)out_b_data_f16;
    (void)out_stride_m;
    (void)out_stride_n;
    (void)panel_rows;
}

#endif
