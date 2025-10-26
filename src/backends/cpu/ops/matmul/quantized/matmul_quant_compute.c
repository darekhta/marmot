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
    size_t row_start;
    size_t row_end;
    marmot_quant_kind_t quant_kind;
} cpu_quant_matmul_worker_args_t;

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

static inline int32_t
cpu_quant_q4_k_block_dotprod(const marmot_q4_k_block_t *w_block, const int8_t *q8, const uint8_t *scales) {
    int32x4_t block_acc = vdupq_n_s32(0);
    const uint8_t *q4 = w_block->qs;
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
        size_t m = args->row_start;
        while (m < args->row_end) {
            const size_t rows_this =
                (m + CPU_QUANT_MATMUL_MR <= args->row_end) ? CPU_QUANT_MATMUL_MR : (args->row_end - m);
            float acc[CPU_QUANT_MATMUL_MR][CPU_QUANT_MATMUL_NR] = {{0}};

            for (size_t kb = 0; kb < blocks_per_row; kb += CPU_QUANT_KC_BLOCKS) {
                const size_t chunk =
                    (kb + CPU_QUANT_KC_BLOCKS <= blocks_per_row) ? CPU_QUANT_KC_BLOCKS : (blocks_per_row - kb);
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
        .row_start = c->row_base + row_start,
        .row_end = c->row_base + row_end,
        .quant_kind = c->quant_kind,
    };
    cpu_quant_matmul_compute_rows(&args);
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

void cpu_quant_matmul_dispatch_quant_tile(
    const cpu_matmul_quant_kernel_t *kernel, const uint8_t *weight_bytes, size_t row_bytes, size_t blocks_per_row,
    const marmot_q8_0_block_t *activation_q8_0, const marmot_q8_k_block_t *activation_q8_k,
    const marmot_q8_k_block_t *activation_panel_q8_k, size_t activation_panel_cols, size_t cols_in_tile, size_t n0,
    float *out_data_f32, marmot_float16_t *out_data_f16, size_t out_stride_m, size_t out_stride_n, size_t row_start,
    size_t row_end, size_t thread_cap, marmot_quant_kind_t quant_kind
) {
    (void)thread_cap;

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
        .quant_kind = quant_kind,
    };
    marmot_dispatch_parallel_for_range(
        MARMOT_DISPATCH_PRIORITY_HIGH, row_end - row_start, CPU_QUANT_MATMUL_BLOCK_ROWS, &dctx,
        cpu_quant_matmul_dispatch_range
    );
}
