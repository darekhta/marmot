#include "matmul_neon.h"

#include "marmot/dispatch.h"

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <arm_neon.h>
#include <string.h>

#include "cpu_backend_internal.h"
#include "cpu_caps.h"
#include "neon_matmul_params.h"
#include "neon_matmul_scratch.h"
#include "ops/matmul/matmul_kernels.h"
#include "utils/dtype_ref.h"

#if HAS_NEON

#if defined(__ARM_FEATURE_BF16) && defined(__aarch64__)
#define MARMOT_HAS_BF16_NATIVE 1
#else
#define MARMOT_HAS_BF16_NATIVE 0
#endif

#define DGEMM_NEON_TILE_M 4
#define DGEMM_NEON_TILE_N 8
#define DGEMM_NEON_BLOCK_K_DEFAULT 384
#define DGEMM_NEON_BLOCK_M_DEFAULT 192
#define DGEMM_NEON_BLOCK_N_DEFAULT 192
#define DGEMM_NEON_MIN_DIM 4
#define DGEMM_NEON_PREFETCH_K_AHEAD_DEFAULT 6

typedef struct {
    size_t block_m;
    size_t block_n;
    size_t block_k;
    size_t prefetch_k_ahead;
    bool double_buffer_pack;
} dgemm_neon_params_t;

static inline const dgemm_neon_params_t *dgemm_neon_get_params(void) {
    static dgemm_neon_params_t params = {0};
    static bool initialized = false;
    if (initialized) {
        return &params;
    }
    params.block_m =
        marmot_neon_parse_env_size("MARMOT_DGEMM_NEON_BLOCK_M", DGEMM_NEON_BLOCK_M_DEFAULT, DGEMM_NEON_TILE_M, 1024);
    params.block_n =
        marmot_neon_parse_env_size("MARMOT_DGEMM_NEON_BLOCK_N", DGEMM_NEON_BLOCK_N_DEFAULT, DGEMM_NEON_TILE_N, 1024);
    params.block_k =
        marmot_neon_parse_env_size("MARMOT_DGEMM_NEON_BLOCK_K", DGEMM_NEON_BLOCK_K_DEFAULT, DGEMM_NEON_TILE_N, 2048);
    params.prefetch_k_ahead =
        marmot_neon_parse_env_size("MARMOT_DGEMM_NEON_PREFETCH_K_AHEAD", DGEMM_NEON_PREFETCH_K_AHEAD_DEFAULT, 1, 32);
    params.double_buffer_pack =
        marmot_neon_parse_env_bool("MARMOT_DGEMM_NEON_DOUBLE_BUFFER_PACK", MARMOT_NEON_F32_DOUBLE_BUFFER_PACK_DEFAULT);
    initialized = true;
    return &params;
}

static void marmot_neon_f32_pack_a(
    const float *input, size_t lda, size_t row_start, size_t rows, size_t k_start, size_t k_block, float *packed
) {
    const size_t prefetch_k = marmot_neon_f32_get_params()->prefetch_k_ahead;
    if (rows == MARMOT_NEON_F32_TILE_N) {
        const float *r0 = input + (row_start + 0) * lda + k_start;
        const float *r1 = input + (row_start + 1) * lda + k_start;
        const float *r2 = input + (row_start + 2) * lda + k_start;
        const float *r3 = input + (row_start + 3) * lda + k_start;
        const float *r4 = input + (row_start + 4) * lda + k_start;
        const float *r5 = input + (row_start + 5) * lda + k_start;
        const float *r6 = input + (row_start + 6) * lda + k_start;
        const float *r7 = input + (row_start + 7) * lda + k_start;

        float *dst = packed;
        for (size_t k = 0; k < k_block; ++k) {
            if (k + prefetch_k < k_block) {
                const size_t next_k = k + prefetch_k;
                __builtin_prefetch(r0 + next_k, 0, 1);
                __builtin_prefetch(r1 + next_k, 0, 1);
                __builtin_prefetch(r2 + next_k, 0, 1);
                __builtin_prefetch(r3 + next_k, 0, 1);
                __builtin_prefetch(r4 + next_k, 0, 1);
                __builtin_prefetch(r5 + next_k, 0, 1);
                __builtin_prefetch(r6 + next_k, 0, 1);
                __builtin_prefetch(r7 + next_k, 0, 1);
            }
            float32x4_t v0 = {r0[k], r1[k], r2[k], r3[k]};
            float32x4_t v1 = {r4[k], r5[k], r6[k], r7[k]};
            vst1q_f32(dst, v0);
            vst1q_f32(dst + 4, v1);
            dst += MARMOT_NEON_F32_TILE_N;
        }
        return;
    }

    for (size_t k = 0; k < k_block; ++k) {
        if (k + prefetch_k < k_block) {
            const size_t next_k = k_start + k + prefetch_k;
            for (size_t r = 0; r < rows; ++r) {
                __builtin_prefetch(input + (row_start + r) * lda + next_k, 0, 1);
            }
        }
        float *dst = packed + k * MARMOT_NEON_F32_TILE_N;
        for (size_t r = 0; r < rows; ++r) {
            const float *a_row = input + (row_start + r) * lda;
            dst[r] = a_row[k_start + k];
        }
        for (size_t r = rows; r < MARMOT_NEON_F32_TILE_N; ++r) {
            dst[r] = 0.0f;
        }
    }
}

static void marmot_neon_f32_pack_b_nt(
    const float *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, float *packed
) {
    const size_t prefetch_k = marmot_neon_f32_get_params()->prefetch_k_ahead;
    const size_t stride = ldw;
    if (cols >= 8) {
        // vector path for full or wider columns
        for (size_t k = 0; k < k_block; ++k) {
            const float *base = weight + col_start * stride + (k_start + k);
            if (k + prefetch_k < k_block) {
                const float *next = weight + col_start * stride + (k_start + k + prefetch_k);
                __builtin_prefetch(next, 0, 3);
            }
            float32x4_t b0 = {base[0 * stride], base[1 * stride], base[2 * stride], base[3 * stride]};
            float32x4_t b1 = {base[4 * stride], base[5 * stride], base[6 * stride], base[7 * stride]};
            float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
            vst1q_f32(dst, b0);
            vst1q_f32(dst + 4, b1);
        }
        return;
    }

    for (size_t k = 0; k < k_block; ++k) {
        const float *w_col = weight + (k_start + k);
        if (k + prefetch_k < k_block) {
            __builtin_prefetch(weight + (k_start + k + prefetch_k), 0, 1);
        }
        float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
        size_t c = 0;
        for (; c + 3 < cols; c += 4) {
            float32x4_t v = {
                w_col[(col_start + c + 0) * ldw], w_col[(col_start + c + 1) * ldw], w_col[(col_start + c + 2) * ldw],
                w_col[(col_start + c + 3) * ldw]
            };
            vst1q_f32(dst + c, v);
        }
        for (; c < cols; ++c) {
            dst[c] = w_col[(col_start + c) * ldw];
        }
        size_t rem = MARMOT_NEON_F32_TILE_M - c;
        for (; rem >= 4; rem -= 4, c += 4) {
            vst1q_f32(dst + c, vdupq_n_f32(0.0f));
        }
        for (; rem > 0; --rem, ++c) {
            dst[c] = 0.0f;
        }
    }
}

static void marmot_neon_f32_pack_b_nn(
    const float *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, float *packed
) {
    const size_t prefetch_k = marmot_neon_f32_get_params()->prefetch_k_ahead;
    if (cols == MARMOT_NEON_F32_TILE_M) {
        for (size_t k = 0; k < k_block; ++k) {
            float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
            const float *w_row = weight + (k_start + k) * ldw + col_start;
            if (k + prefetch_k < k_block) {
                const float *next_row = weight + (k_start + k + prefetch_k) * ldw + col_start;
                __builtin_prefetch(next_row, 0, 3);
            }
            float32x4_t b0 = vld1q_f32(w_row);
            float32x4_t b1 = vld1q_f32(w_row + 4);
            vst1q_f32(dst, b0);
            vst1q_f32(dst + 4, b1);
        }
        return;
    }

    for (size_t k = 0; k < k_block; ++k) {
        float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
        const float *w_row = weight + (k_start + k) * ldw + col_start;
        if (k + prefetch_k < k_block) {
            const float *next_row = weight + (k_start + k + prefetch_k) * ldw + col_start;
            __builtin_prefetch(next_row, 0, 3);
        }
        memcpy(dst, w_row, cols * sizeof(float));
        size_t c = cols;
        size_t rem = MARMOT_NEON_F32_TILE_M - c;
        for (; rem >= 4; rem -= 4, c += 4) {
            vst1q_f32(dst + c, vdupq_n_f32(0.0f));
        }
        for (; rem > 0; --rem, ++c) {
            dst[c] = 0.0f;
        }
    }
}

typedef struct {
    const float *weight;
    size_t M;
    size_t K;
    size_t block_k;
    size_t packed_b_stride;
    float *packed_b;
    bool layout_nt;
} marmot_neon_f32_pack_b_global_ctx_t;

static void marmot_neon_f32_pack_b_global_worker(void *ctx, size_t tile_idx) {
    const marmot_neon_f32_pack_b_global_ctx_t *p = (const marmot_neon_f32_pack_b_global_ctx_t *)ctx;
    const size_t m0 = tile_idx * MARMOT_NEON_F32_TILE_M;
    const size_t m_block = marmot_neon_min_size(MARMOT_NEON_F32_TILE_M, p->M - m0);
    float *packed_base = p->packed_b + tile_idx * p->packed_b_stride;
    for (size_t k0 = 0; k0 < p->K; k0 += p->block_k) {
        const size_t k_block = marmot_neon_min_size(p->block_k, p->K - k0);
        float *packed = packed_base + k0 * MARMOT_NEON_F32_TILE_M;
        if (p->layout_nt) {
            marmot_neon_f32_pack_b_nt(p->weight, p->K, m0, m_block, k0, k_block, packed);
        } else {
            marmot_neon_f32_pack_b_nn(p->weight, p->M, m0, m_block, k0, k_block, packed);
        }
    }
}

static void marmot_neon_f32_kernel_8x8(float *c_tile, size_t k_block, const float *packed_a, const float *packed_b) {
    float32x4_t c00 = vld1q_f32(c_tile + 0);
    float32x4_t c01 = vld1q_f32(c_tile + 4);
    float32x4_t c10 = vld1q_f32(c_tile + 8);
    float32x4_t c11 = vld1q_f32(c_tile + 12);
    float32x4_t c20 = vld1q_f32(c_tile + 16);
    float32x4_t c21 = vld1q_f32(c_tile + 20);
    float32x4_t c30 = vld1q_f32(c_tile + 24);
    float32x4_t c31 = vld1q_f32(c_tile + 28);
    float32x4_t c40 = vld1q_f32(c_tile + 32);
    float32x4_t c41 = vld1q_f32(c_tile + 36);
    float32x4_t c50 = vld1q_f32(c_tile + 40);
    float32x4_t c51 = vld1q_f32(c_tile + 44);
    float32x4_t c60 = vld1q_f32(c_tile + 48);
    float32x4_t c61 = vld1q_f32(c_tile + 52);
    float32x4_t c70 = vld1q_f32(c_tile + 56);
    float32x4_t c71 = vld1q_f32(c_tile + 60);

    const float *pa = packed_a;
    const float *pb = packed_b;
    const size_t prefetch_k = marmot_neon_f32_get_params()->prefetch_k_ahead;
    const size_t a_stride = MARMOT_NEON_F32_TILE_N;
    const size_t b_stride = MARMOT_NEON_F32_TILE_M;

#if defined(__aarch64__)
#define SGEMM_8X8_ACCUM(a0, a1, b0, b1)                                                                                \
    do {                                                                                                               \
        c00 = vfmaq_laneq_f32(c00, (b0), (a0), 0);                                                                     \
        c01 = vfmaq_laneq_f32(c01, (b1), (a0), 0);                                                                     \
        c10 = vfmaq_laneq_f32(c10, (b0), (a0), 1);                                                                     \
        c11 = vfmaq_laneq_f32(c11, (b1), (a0), 1);                                                                     \
        c20 = vfmaq_laneq_f32(c20, (b0), (a0), 2);                                                                     \
        c21 = vfmaq_laneq_f32(c21, (b1), (a0), 2);                                                                     \
        c30 = vfmaq_laneq_f32(c30, (b0), (a0), 3);                                                                     \
        c31 = vfmaq_laneq_f32(c31, (b1), (a0), 3);                                                                     \
                                                                                                                       \
        c40 = vfmaq_laneq_f32(c40, (b0), (a1), 0);                                                                     \
        c41 = vfmaq_laneq_f32(c41, (b1), (a1), 0);                                                                     \
        c50 = vfmaq_laneq_f32(c50, (b0), (a1), 1);                                                                     \
        c51 = vfmaq_laneq_f32(c51, (b1), (a1), 1);                                                                     \
        c60 = vfmaq_laneq_f32(c60, (b0), (a1), 2);                                                                     \
        c61 = vfmaq_laneq_f32(c61, (b1), (a1), 2);                                                                     \
        c70 = vfmaq_laneq_f32(c70, (b0), (a1), 3);                                                                     \
        c71 = vfmaq_laneq_f32(c71, (b1), (a1), 3);                                                                     \
    } while (0)
#else
#define SGEMM_8X8_ACCUM(a0, a1, b0, b1)                                                                                \
    do {                                                                                                               \
        const float a0_lane0 = vgetq_lane_f32((a0), 0);                                                                \
        const float a0_lane1 = vgetq_lane_f32((a0), 1);                                                                \
        const float a0_lane2 = vgetq_lane_f32((a0), 2);                                                                \
        const float a0_lane3 = vgetq_lane_f32((a0), 3);                                                                \
        const float a1_lane0 = vgetq_lane_f32((a1), 0);                                                                \
        const float a1_lane1 = vgetq_lane_f32((a1), 1);                                                                \
        const float a1_lane2 = vgetq_lane_f32((a1), 2);                                                                \
        const float a1_lane3 = vgetq_lane_f32((a1), 3);                                                                \
                                                                                                                       \
        c00 = vmlaq_n_f32(c00, (b0), a0_lane0);                                                                        \
        c01 = vmlaq_n_f32(c01, (b1), a0_lane0);                                                                        \
        c10 = vmlaq_n_f32(c10, (b0), a0_lane1);                                                                        \
        c11 = vmlaq_n_f32(c11, (b1), a0_lane1);                                                                        \
        c20 = vmlaq_n_f32(c20, (b0), a0_lane2);                                                                        \
        c21 = vmlaq_n_f32(c21, (b1), a0_lane2);                                                                        \
        c30 = vmlaq_n_f32(c30, (b0), a0_lane3);                                                                        \
        c31 = vmlaq_n_f32(c31, (b1), a0_lane3);                                                                        \
                                                                                                                       \
        c40 = vmlaq_n_f32(c40, (b0), a1_lane0);                                                                        \
        c41 = vmlaq_n_f32(c41, (b1), a1_lane0);                                                                        \
        c50 = vmlaq_n_f32(c50, (b0), a1_lane1);                                                                        \
        c51 = vmlaq_n_f32(c51, (b1), a1_lane1);                                                                        \
        c60 = vmlaq_n_f32(c60, (b0), a1_lane2);                                                                        \
        c61 = vmlaq_n_f32(c61, (b1), a1_lane2);                                                                        \
        c70 = vmlaq_n_f32(c70, (b0), a1_lane3);                                                                        \
        c71 = vmlaq_n_f32(c71, (b1), a1_lane3);                                                                        \
    } while (0)
#endif

    size_t k = 0;
    for (; k + 3 < k_block; k += 4) {
        const float *pa0 = pa;
        const float *pb0 = pb;
        if (k + prefetch_k < k_block) {
            __builtin_prefetch(pa0 + prefetch_k * a_stride, 0, 3);
            __builtin_prefetch(pb0 + prefetch_k * b_stride, 0, 3);
        }
        float32x4_t a0_0 = vld1q_f32(pa0);
        float32x4_t a1_0 = vld1q_f32(pa0 + 4);
        float32x4_t b0_0 = vld1q_f32(pb0);
        float32x4_t b1_0 = vld1q_f32(pb0 + 4);

        float32x4_t a0_1 = vld1q_f32(pa0 + a_stride);
        float32x4_t a1_1 = vld1q_f32(pa0 + a_stride + 4);
        float32x4_t b0_1 = vld1q_f32(pb0 + b_stride);
        float32x4_t b1_1 = vld1q_f32(pb0 + b_stride + 4);

        SGEMM_8X8_ACCUM(a0_0, a1_0, b0_0, b1_0);

        float32x4_t a0_2 = vld1q_f32(pa0 + 2 * a_stride);
        float32x4_t a1_2 = vld1q_f32(pa0 + 2 * a_stride + 4);
        float32x4_t b0_2 = vld1q_f32(pb0 + 2 * b_stride);
        float32x4_t b1_2 = vld1q_f32(pb0 + 2 * b_stride + 4);

        SGEMM_8X8_ACCUM(a0_1, a1_1, b0_1, b1_1);

        float32x4_t a0_3 = vld1q_f32(pa0 + 3 * a_stride);
        float32x4_t a1_3 = vld1q_f32(pa0 + 3 * a_stride + 4);
        float32x4_t b0_3 = vld1q_f32(pb0 + 3 * b_stride);
        float32x4_t b1_3 = vld1q_f32(pb0 + 3 * b_stride + 4);

        SGEMM_8X8_ACCUM(a0_2, a1_2, b0_2, b1_2);
        SGEMM_8X8_ACCUM(a0_3, a1_3, b0_3, b1_3);

        pa += 4 * a_stride;
        pb += 4 * b_stride;
    }

    for (; k < k_block; ++k) {
        if (k + prefetch_k < k_block) {
            __builtin_prefetch(pa + prefetch_k * a_stride, 0, 3);
            __builtin_prefetch(pb + prefetch_k * b_stride, 0, 3);
        }
        float32x4_t a0 = vld1q_f32(pa);
        float32x4_t a1 = vld1q_f32(pa + 4);
        float32x4_t b0 = vld1q_f32(pb);
        float32x4_t b1 = vld1q_f32(pb + 4);

        SGEMM_8X8_ACCUM(a0, a1, b0, b1);

        pa += a_stride;
        pb += b_stride;
    }

#undef SGEMM_8X8_ACCUM

    vst1q_f32(c_tile + 0, c00);
    vst1q_f32(c_tile + 4, c01);
    vst1q_f32(c_tile + 8, c10);
    vst1q_f32(c_tile + 12, c11);
    vst1q_f32(c_tile + 16, c20);
    vst1q_f32(c_tile + 20, c21);
    vst1q_f32(c_tile + 24, c30);
    vst1q_f32(c_tile + 28, c31);
    vst1q_f32(c_tile + 32, c40);
    vst1q_f32(c_tile + 36, c41);
    vst1q_f32(c_tile + 40, c50);
    vst1q_f32(c_tile + 44, c51);
    vst1q_f32(c_tile + 48, c60);
    vst1q_f32(c_tile + 52, c61);
    vst1q_f32(c_tile + 56, c70);
    vst1q_f32(c_tile + 60, c71);
}

static void marmot_neon_f32_kernel_4x8(float *c_tile, size_t k_block, const float *packed_a, const float *packed_b) {
    float32x4_t c0 = vld1q_f32(c_tile + 0);
    float32x4_t c1 = vld1q_f32(c_tile + 4);
    float32x4_t c2 = vld1q_f32(c_tile + 8);
    float32x4_t c3 = vld1q_f32(c_tile + 12);
    float32x4_t c4 = vld1q_f32(c_tile + 16);
    float32x4_t c5 = vld1q_f32(c_tile + 20);
    float32x4_t c6 = vld1q_f32(c_tile + 24);
    float32x4_t c7 = vld1q_f32(c_tile + 28);

    const float *pa = packed_a;
    const float *pb = packed_b;
    for (size_t k = 0; k < k_block; ++k) {
        float32x4_t a = vld1q_f32(pa);
        float32x4_t b0 = vld1q_f32(pb);
        float32x4_t b1 = vld1q_f32(pb + 4);

#if defined(__aarch64__)
        c0 = vfmaq_laneq_f32(c0, b0, a, 0);
        c1 = vfmaq_laneq_f32(c1, b1, a, 0);
        c2 = vfmaq_laneq_f32(c2, b0, a, 1);
        c3 = vfmaq_laneq_f32(c3, b1, a, 1);
        c4 = vfmaq_laneq_f32(c4, b0, a, 2);
        c5 = vfmaq_laneq_f32(c5, b1, a, 2);
        c6 = vfmaq_laneq_f32(c6, b0, a, 3);
        c7 = vfmaq_laneq_f32(c7, b1, a, 3);
#else
        const float a0 = vgetq_lane_f32(a, 0);
        const float a1 = vgetq_lane_f32(a, 1);
        const float a2 = vgetq_lane_f32(a, 2);
        const float a3 = vgetq_lane_f32(a, 3);
        c0 = vmlaq_n_f32(c0, b0, a0);
        c1 = vmlaq_n_f32(c1, b1, a0);
        c2 = vmlaq_n_f32(c2, b0, a1);
        c3 = vmlaq_n_f32(c3, b1, a1);
        c4 = vmlaq_n_f32(c4, b0, a2);
        c5 = vmlaq_n_f32(c5, b1, a2);
        c6 = vmlaq_n_f32(c6, b0, a3);
        c7 = vmlaq_n_f32(c7, b1, a3);
#endif

        pa += MARMOT_NEON_F32_TILE_N;
        pb += MARMOT_NEON_F32_TILE_M;
    }

    vst1q_f32(c_tile + 0, c0);
    vst1q_f32(c_tile + 4, c1);
    vst1q_f32(c_tile + 8, c2);
    vst1q_f32(c_tile + 12, c3);
    vst1q_f32(c_tile + 16, c4);
    vst1q_f32(c_tile + 20, c5);
    vst1q_f32(c_tile + 24, c6);
    vst1q_f32(c_tile + 28, c7);
}

static void marmot_neon_f32_kernel_8x4(float *c_tile, size_t k_block, const float *packed_a, const float *packed_b) {
    float32x4_t c0 = vld1q_f32(c_tile + 0);
    float32x4_t c1 = vld1q_f32(c_tile + 8);
    float32x4_t c2 = vld1q_f32(c_tile + 16);
    float32x4_t c3 = vld1q_f32(c_tile + 24);
    float32x4_t c4 = vld1q_f32(c_tile + 32);
    float32x4_t c5 = vld1q_f32(c_tile + 40);
    float32x4_t c6 = vld1q_f32(c_tile + 48);
    float32x4_t c7 = vld1q_f32(c_tile + 56);

    const float *pa = packed_a;
    const float *pb = packed_b;
    for (size_t k = 0; k < k_block; ++k) {
        float32x4_t a0 = vld1q_f32(pa);
        float32x4_t a1 = vld1q_f32(pa + 4);
        float32x4_t b = vld1q_f32(pb);

#if defined(__aarch64__)
        c0 = vfmaq_laneq_f32(c0, b, a0, 0);
        c1 = vfmaq_laneq_f32(c1, b, a0, 1);
        c2 = vfmaq_laneq_f32(c2, b, a0, 2);
        c3 = vfmaq_laneq_f32(c3, b, a0, 3);
        c4 = vfmaq_laneq_f32(c4, b, a1, 0);
        c5 = vfmaq_laneq_f32(c5, b, a1, 1);
        c6 = vfmaq_laneq_f32(c6, b, a1, 2);
        c7 = vfmaq_laneq_f32(c7, b, a1, 3);
#else
        const float a0_lane0 = vgetq_lane_f32(a0, 0);
        const float a0_lane1 = vgetq_lane_f32(a0, 1);
        const float a0_lane2 = vgetq_lane_f32(a0, 2);
        const float a0_lane3 = vgetq_lane_f32(a0, 3);
        const float a1_lane0 = vgetq_lane_f32(a1, 0);
        const float a1_lane1 = vgetq_lane_f32(a1, 1);
        const float a1_lane2 = vgetq_lane_f32(a1, 2);
        const float a1_lane3 = vgetq_lane_f32(a1, 3);
        c0 = vmlaq_n_f32(c0, b, a0_lane0);
        c1 = vmlaq_n_f32(c1, b, a0_lane1);
        c2 = vmlaq_n_f32(c2, b, a0_lane2);
        c3 = vmlaq_n_f32(c3, b, a0_lane3);
        c4 = vmlaq_n_f32(c4, b, a1_lane0);
        c5 = vmlaq_n_f32(c5, b, a1_lane1);
        c6 = vmlaq_n_f32(c6, b, a1_lane2);
        c7 = vmlaq_n_f32(c7, b, a1_lane3);
#endif

        pa += MARMOT_NEON_F32_TILE_N;
        pb += MARMOT_NEON_F32_TILE_M;
    }

    vst1q_f32(c_tile + 0, c0);
    vst1q_f32(c_tile + 8, c1);
    vst1q_f32(c_tile + 16, c2);
    vst1q_f32(c_tile + 24, c3);
    vst1q_f32(c_tile + 32, c4);
    vst1q_f32(c_tile + 40, c5);
    vst1q_f32(c_tile + 48, c6);
    vst1q_f32(c_tile + 56, c7);
}

static void marmot_neon_f32_kernel_4x4(float *c_tile, size_t k_block, const float *packed_a, const float *packed_b) {
    float32x4_t c0 = vld1q_f32(c_tile + 0);
    float32x4_t c1 = vld1q_f32(c_tile + 8);
    float32x4_t c2 = vld1q_f32(c_tile + 16);
    float32x4_t c3 = vld1q_f32(c_tile + 24);

    const float *pa = packed_a;
    const float *pb = packed_b;
    for (size_t k = 0; k < k_block; ++k) {
        float32x4_t a = vld1q_f32(pa);
        float32x4_t b = vld1q_f32(pb);
#if defined(__aarch64__)
        c0 = vfmaq_laneq_f32(c0, b, a, 0);
        c1 = vfmaq_laneq_f32(c1, b, a, 1);
        c2 = vfmaq_laneq_f32(c2, b, a, 2);
        c3 = vfmaq_laneq_f32(c3, b, a, 3);
#else
        const float a0 = vgetq_lane_f32(a, 0);
        const float a1 = vgetq_lane_f32(a, 1);
        const float a2 = vgetq_lane_f32(a, 2);
        const float a3 = vgetq_lane_f32(a, 3);
        c0 = vmlaq_n_f32(c0, b, a0);
        c1 = vmlaq_n_f32(c1, b, a1);
        c2 = vmlaq_n_f32(c2, b, a2);
        c3 = vmlaq_n_f32(c3, b, a3);
#endif
        pa += MARMOT_NEON_F32_TILE_N;
        pb += MARMOT_NEON_F32_TILE_M;
    }

    vst1q_f32(c_tile + 0, c0);
    vst1q_f32(c_tile + 8, c1);
    vst1q_f32(c_tile + 16, c2);
    vst1q_f32(c_tile + 24, c3);
}

static void marmot_neon_f32_kernel_2x8(float *c_tile, size_t k_block, const float *packed_a, const float *packed_b) {
    float32x4_t c0 = vld1q_f32(c_tile + 0);
    float32x4_t c1 = vld1q_f32(c_tile + 4);
    float32x4_t c2 = vld1q_f32(c_tile + 8);
    float32x4_t c3 = vld1q_f32(c_tile + 12);

    float32x4_t c4 = vld1q_f32(c_tile + 16);
    float32x4_t c5 = vld1q_f32(c_tile + 20);
    float32x4_t c6 = vld1q_f32(c_tile + 24);
    float32x4_t c7 = vld1q_f32(c_tile + 28);

    const float *pa = packed_a;
    const float *pb = packed_b;
    for (size_t k = 0; k < k_block; ++k) {
        const float a0 = pa[0];
        const float a1 = pa[1];
        float32x4_t b0 = vld1q_f32(pb);
        float32x4_t b1 = vld1q_f32(pb + 4);

        c0 = vmlaq_n_f32(c0, b0, a0);
        c1 = vmlaq_n_f32(c1, b1, a0);
        c2 = vmlaq_n_f32(c2, b0, a1);
        c3 = vmlaq_n_f32(c3, b1, a1);

        const float a2 = pa[2];
        const float a3 = pa[3];
        c4 = vmlaq_n_f32(c4, b0, a2);
        c5 = vmlaq_n_f32(c5, b1, a2);
        c6 = vmlaq_n_f32(c6, b0, a3);
        c7 = vmlaq_n_f32(c7, b1, a3);

        pa += MARMOT_NEON_F32_TILE_N;
        pb += MARMOT_NEON_F32_TILE_M;
    }

    vst1q_f32(c_tile + 0, c0);
    vst1q_f32(c_tile + 4, c1);
    vst1q_f32(c_tile + 8, c2);
    vst1q_f32(c_tile + 12, c3);
    vst1q_f32(c_tile + 16, c4);
    vst1q_f32(c_tile + 20, c5);
    vst1q_f32(c_tile + 24, c6);
    vst1q_f32(c_tile + 28, c7);
}

static void marmot_neon_f32_kernel_1x8(float *c_tile, size_t k_block, const float *packed_a, const float *packed_b) {
    float32x4_t c0 = vld1q_f32(c_tile + 0);
    float32x4_t c1 = vld1q_f32(c_tile + 4);

    const float *pa = packed_a;
    const float *pb = packed_b;
    for (size_t k = 0; k < k_block; ++k) {
        const float a0 = pa[0];
        float32x4_t b0 = vld1q_f32(pb);
        float32x4_t b1 = vld1q_f32(pb + 4);

        c0 = vmlaq_n_f32(c0, b0, a0);
        c1 = vmlaq_n_f32(c1, b1, a0);

        pa += MARMOT_NEON_F32_TILE_N;
        pb += MARMOT_NEON_F32_TILE_M;
    }

    vst1q_f32(c_tile + 0, c0);
    vst1q_f32(c_tile + 4, c1);
}

static void marmot_neon_f32_kernel_2x4(float *c_tile, size_t k_block, const float *packed_a, const float *packed_b) {
    float32x4_t c0 = vld1q_f32(c_tile + 0);
    float32x4_t c1 = vld1q_f32(c_tile + 8);

    const float *pa = packed_a;
    const float *pb = packed_b;
    for (size_t k = 0; k < k_block; ++k) {
        float32x4_t b = vld1q_f32(pb);
        const float a0 = pa[0];
        const float a1 = pa[1];
        c0 = vmlaq_n_f32(c0, b, a0);
        c1 = vmlaq_n_f32(c1, b, a1);

        pa += MARMOT_NEON_F32_TILE_N;
        pb += MARMOT_NEON_F32_TILE_M;
    }

    vst1q_f32(c_tile + 0, c0);
    vst1q_f32(c_tile + 8, c1);
}

static void marmot_neon_f32_kernel_1x4(float *c_tile, size_t k_block, const float *packed_a, const float *packed_b) {
    float32x4_t c0 = vld1q_f32(c_tile + 0);

    const float *pa = packed_a;
    const float *pb = packed_b;
    for (size_t k = 0; k < k_block; ++k) {
        float32x4_t b = vld1q_f32(pb);
        const float a0 = pa[0];
        c0 = vmlaq_n_f32(c0, b, a0);
        pa += MARMOT_NEON_F32_TILE_N;
        pb += MARMOT_NEON_F32_TILE_M;
    }

    vst1q_f32(c_tile + 0, c0);
}
static void marmot_neon_f32_kernel_edge(
    float *c_tile, size_t n_block, size_t m_block, size_t k_block, const float *packed_a, const float *packed_b
) {
    for (size_t n = 0; n < n_block; ++n) {
        for (size_t m = 0; m < m_block; ++m) {
            float acc = c_tile[n * MARMOT_NEON_F32_TILE_M + m];
            const float *pa = packed_a + n;
            const float *pb = packed_b + m;
            for (size_t k = 0; k < k_block; ++k) {
                acc += pa[k * MARMOT_NEON_F32_TILE_N] * pb[k * MARMOT_NEON_F32_TILE_M];
            }
            c_tile[n * MARMOT_NEON_F32_TILE_M + m] = acc;
        }
    }
}

static void marmot_neon_f32_store_tile(
    float *out, size_t ldo, const float *c_tile, size_t n_block, size_t m_block, size_t row_start, size_t col_start,
    float alpha, float beta
) {
    for (size_t r = 0; r < n_block; ++r) {
        float *dst = out + (row_start + r) * ldo + col_start;
        const float *src = c_tile + r * MARMOT_NEON_F32_TILE_M;
        for (size_t c = 0; c < m_block; ++c) {
            if (beta == 0.0f) {
                dst[c] = alpha * src[c];
            } else {
                dst[c] = alpha * src[c] + beta * dst[c];
            }
        }
    }
}

static void marmot_neon_f32_small_direct_nt(
    const float *input, const float *weight, float *out, size_t N, size_t M, size_t K, float alpha, float beta
) {
    const bool beta_zero = beta == 0.0f;
    for (size_t n = 0; n < N; n += 4) {
        const size_t n_block = (n + 4 <= N) ? 4 : (N - n);
        for (size_t m = 0; m < M; m += 4) {
            const size_t m_block = (m + 4 <= M) ? 4 : (M - m);
            float32x4_t acc[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
            if (n_block == 4 && m_block == 4) {
                for (size_t k = 0; k + 4 <= K; k += 4) {
                    float32x4_t a0 = vld1q_f32(input + (n + 0) * K + k);
                    float32x4_t a1 = vld1q_f32(input + (n + 1) * K + k);
                    float32x4_t a2 = vld1q_f32(input + (n + 2) * K + k);
                    float32x4_t a3 = vld1q_f32(input + (n + 3) * K + k);
                    float32x4_t b0 = vld1q_f32(weight + (m + 0) * K + k);
                    float32x4_t b1 = vld1q_f32(weight + (m + 1) * K + k);
                    float32x4_t b2 = vld1q_f32(weight + (m + 2) * K + k);
                    float32x4_t b3 = vld1q_f32(weight + (m + 3) * K + k);
                    acc[0] = vfmaq_laneq_f32(acc[0], b0, a0, 0);
                    acc[0] = vfmaq_laneq_f32(acc[0], b1, a0, 1);
                    acc[0] = vfmaq_laneq_f32(acc[0], b2, a0, 2);
                    acc[0] = vfmaq_laneq_f32(acc[0], b3, a0, 3);
                    acc[1] = vfmaq_laneq_f32(acc[1], b0, a1, 0);
                    acc[1] = vfmaq_laneq_f32(acc[1], b1, a1, 1);
                    acc[1] = vfmaq_laneq_f32(acc[1], b2, a1, 2);
                    acc[1] = vfmaq_laneq_f32(acc[1], b3, a1, 3);
                    acc[2] = vfmaq_laneq_f32(acc[2], b0, a2, 0);
                    acc[2] = vfmaq_laneq_f32(acc[2], b1, a2, 1);
                    acc[2] = vfmaq_laneq_f32(acc[2], b2, a2, 2);
                    acc[2] = vfmaq_laneq_f32(acc[2], b3, a2, 3);
                    acc[3] = vfmaq_laneq_f32(acc[3], b0, a3, 0);
                    acc[3] = vfmaq_laneq_f32(acc[3], b1, a3, 1);
                    acc[3] = vfmaq_laneq_f32(acc[3], b2, a3, 2);
                    acc[3] = vfmaq_laneq_f32(acc[3], b3, a3, 3);
                }
                size_t k_tail = K & ~3UL;
                for (size_t k = k_tail; k < K; ++k) {
                    float a0 = input[(n + 0) * K + k];
                    float a1 = input[(n + 1) * K + k];
                    float a2 = input[(n + 2) * K + k];
                    float a3 = input[(n + 3) * K + k];
                    float b0 = weight[(m + 0) * K + k];
                    float b1 = weight[(m + 1) * K + k];
                    float b2 = weight[(m + 2) * K + k];
                    float b3 = weight[(m + 3) * K + k];
                    float32x4_t b_vec = {b0, b1, b2, b3};
                    acc[0] = vfmaq_n_f32(acc[0], b_vec, a0);
                    acc[1] = vfmaq_n_f32(acc[1], b_vec, a1);
                    acc[2] = vfmaq_n_f32(acc[2], b_vec, a2);
                    acc[3] = vfmaq_n_f32(acc[3], b_vec, a3);
                }
                float32x4_t alpha_vec = vdupq_n_f32(alpha);
                if (beta_zero) {
                    vst1q_f32(out + (n + 0) * M + m, vmulq_f32(acc[0], alpha_vec));
                    vst1q_f32(out + (n + 1) * M + m, vmulq_f32(acc[1], alpha_vec));
                    vst1q_f32(out + (n + 2) * M + m, vmulq_f32(acc[2], alpha_vec));
                    vst1q_f32(out + (n + 3) * M + m, vmulq_f32(acc[3], alpha_vec));
                } else {
                    float32x4_t beta_vec = vdupq_n_f32(beta);
                    float32x4_t c0 = vld1q_f32(out + (n + 0) * M + m);
                    float32x4_t c1 = vld1q_f32(out + (n + 1) * M + m);
                    float32x4_t c2 = vld1q_f32(out + (n + 2) * M + m);
                    float32x4_t c3 = vld1q_f32(out + (n + 3) * M + m);
                    vst1q_f32(out + (n + 0) * M + m, vfmaq_f32(vmulq_f32(c0, beta_vec), acc[0], alpha_vec));
                    vst1q_f32(out + (n + 1) * M + m, vfmaq_f32(vmulq_f32(c1, beta_vec), acc[1], alpha_vec));
                    vst1q_f32(out + (n + 2) * M + m, vfmaq_f32(vmulq_f32(c2, beta_vec), acc[2], alpha_vec));
                    vst1q_f32(out + (n + 3) * M + m, vfmaq_f32(vmulq_f32(c3, beta_vec), acc[3], alpha_vec));
                }
            } else {
                float acc_scalar[4][4] = {{0}};
                for (size_t ni = 0; ni < n_block; ++ni) {
                    const float *a_row = input + (n + ni) * K;
                    for (size_t mi = 0; mi < m_block; ++mi) {
                        const float *w_row = weight + (m + mi) * K;
                        float sum = 0.0f;
                        for (size_t k = 0; k < K; ++k) {
                            sum += a_row[k] * w_row[k];
                        }
                        acc_scalar[ni][mi] = sum;
                    }
                }
                for (size_t ni = 0; ni < n_block; ++ni) {
                    float *c_row = out + (n + ni) * M + m;
                    for (size_t mi = 0; mi < m_block; ++mi) {
                        float val = alpha * acc_scalar[ni][mi];
                        if (!beta_zero) {
                            val += beta * c_row[mi];
                        }
                        c_row[mi] = val;
                    }
                }
            }
        }
    }
}

static void marmot_neon_f32_fallback_nt(
    const float *input, const float *weight, float *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    float alpha, float beta
) {
    for (size_t n = row_start; n < row_end; ++n) {
        const float *a_row = input + n * K;
        float *c_row = out + n * M;
        for (size_t m = 0; m < M; ++m) {
            const float *w_row = weight + m * K;
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += a_row[k] * w_row[k];
            }
            float value = alpha * acc;
            if (beta != 0.0f) {
                value += beta * c_row[m];
            }
            c_row[m] = value;
        }
    }
}

static void marmot_neon_f32_fallback_nn(
    const float *input, const float *weight, float *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    float alpha, float beta
) {
    for (size_t n = row_start; n < row_end; ++n) {
        const float *a_row = input + n * K;
        float *c_row = out + n * M;
        for (size_t m = 0; m < M; ++m) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += a_row[k] * weight[k * M + m];
            }
            float value = alpha * acc;
            if (beta != 0.0f) {
                value += beta * c_row[m];
            }
            c_row[m] = value;
        }
    }
}

typedef struct {
    const float *input;
    const float *weight;
    float *out;
    size_t N;
    size_t M;
    size_t K;
    size_t row_start;
    size_t row_end;
    bool layout_nt;
    float alpha;
    float beta;
    const float *packed_b_global;
    size_t packed_b_stride;
    marmot_neon_f32_scratch_t *scratch; // Pooled scratch buffer, nullptr = allocate locally
} marmot_neon_f32_worker_args_t;

static void marmot_neon_f32_compute_rows(const marmot_neon_f32_worker_args_t *args) {
    const float *input = args->input;
    const float *weight = args->weight;
    float *out = args->out;
    const size_t N = args->N;
    const size_t M = args->M;
    const size_t K = args->K;
    const size_t row_start = args->row_start;
    const size_t row_end = args->row_end;
    const marmot_neon_f32_params_t *params = marmot_neon_f32_get_params();
    const float *packed_b_global = args->packed_b_global;
    const size_t packed_b_stride = args->packed_b_stride;

    // Try to use pooled scratch buffers if provided
    float *packed_a = nullptr;
    float *packed_b_panel = nullptr;
    float *c_panel = nullptr;
    bool local_alloc = false;
    const size_t c_tile_elems = MARMOT_NEON_F32_TILE_N * MARMOT_NEON_F32_TILE_M;

    if (args->scratch != nullptr && marmot_neon_f32_scratch_ensure(args->scratch, params, row_end - row_start, M, K)) {
        packed_a = args->scratch->packed_a;
        if (packed_b_global == nullptr) {
            packed_b_panel = args->scratch->packed_b;
        }
        c_panel = args->scratch->c_panel;
    } else {
        // Fallback to local allocation
        local_alloc = true;
        const size_t pack_a_elems = params->block_k * MARMOT_NEON_F32_TILE_N;
        packed_a = (float *)marmot_aligned_alloc(64, pack_a_elems * sizeof(float));
        if (packed_b_global == nullptr) {
            const size_t pack_b_panel_elems = params->block_k * params->block_m;
            packed_b_panel = (float *)marmot_aligned_alloc(64, pack_b_panel_elems * sizeof(float));
        }
        const size_t block_n_cap = marmot_neon_min_size(params->block_n, row_end - row_start);
        const size_t block_m_cap = marmot_neon_min_size(params->block_m, M);
        const size_t n_tiles_cap = (block_n_cap + MARMOT_NEON_F32_TILE_N - 1) / MARMOT_NEON_F32_TILE_N;
        const size_t m_tiles_cap = (block_m_cap + MARMOT_NEON_F32_TILE_M - 1) / MARMOT_NEON_F32_TILE_M;
        c_panel = (float *)marmot_aligned_alloc(64, n_tiles_cap * m_tiles_cap * c_tile_elems * sizeof(float));
    }

    if (packed_a == nullptr || c_panel == nullptr || (packed_b_global == nullptr && packed_b_panel == nullptr)) {
        if (local_alloc) {
            free(packed_a);
            if (packed_b_panel != nullptr) {
                free(packed_b_panel);
            }
            free(c_panel);
        }
        if (args->layout_nt) {
            marmot_neon_f32_fallback_nt(input, weight, out, N, M, K, row_start, row_end, args->alpha, args->beta);
        } else {
            marmot_neon_f32_fallback_nn(input, weight, out, N, M, K, row_start, row_end, args->alpha, args->beta);
        }
        return;
    }

    const float alpha = args->alpha;
    const float beta = args->beta;
    const bool beta_zero = beta == 0.0f;
    for (size_t m_outer = 0; m_outer < M; m_outer += params->block_m) {
        const size_t m_outer_end = marmot_neon_min_size(M, m_outer + params->block_m);
        const size_t m_tiles = (m_outer_end - m_outer + MARMOT_NEON_F32_TILE_M - 1) / MARMOT_NEON_F32_TILE_M;
        for (size_t n_outer = row_start; n_outer < row_end; n_outer += params->block_n) {
            const size_t n_outer_end = marmot_neon_min_size(row_end, n_outer + params->block_n);
            const size_t n_tiles = (n_outer_end - n_outer + MARMOT_NEON_F32_TILE_N - 1) / MARMOT_NEON_F32_TILE_N;
            if (beta_zero) {
                memset(c_panel, 0, n_tiles * m_tiles * c_tile_elems * sizeof(float));
            } else {
                size_t n_idx = 0;
                for (size_t n0 = n_outer; n0 < n_outer_end; n0 += MARMOT_NEON_F32_TILE_N, ++n_idx) {
                    const size_t n_block = marmot_neon_min_size(MARMOT_NEON_F32_TILE_N, n_outer_end - n0);
                    for (size_t m_idx = 0; m_idx < m_tiles; ++m_idx) {
                        const size_t m0 = m_outer + m_idx * MARMOT_NEON_F32_TILE_M;
                        const size_t m_block = marmot_neon_min_size(MARMOT_NEON_F32_TILE_M, m_outer_end - m0);
                        float *c_tile = c_panel + (n_idx * m_tiles + m_idx) * c_tile_elems;
                        size_t r = 0;
                        for (; r < n_block; ++r) {
                            const float *src = out + (n0 + r) * M + m0;
                            float *dst_row = c_tile + r * MARMOT_NEON_F32_TILE_M;
                            size_t c = 0;
                            for (; c < m_block; ++c) {
                                dst_row[c] = beta * src[c];
                            }
                            for (; c < MARMOT_NEON_F32_TILE_M; ++c) {
                                dst_row[c] = 0.0f;
                            }
                        }
                        for (; r < MARMOT_NEON_F32_TILE_N; ++r) {
                            float *dst_row = c_tile + r * MARMOT_NEON_F32_TILE_M;
                            for (size_t c = 0; c < MARMOT_NEON_F32_TILE_M; ++c) {
                                dst_row[c] = 0.0f;
                            }
                        }
                    }
                }
            }

            for (size_t k0 = 0; k0 < K; k0 += params->block_k) {
                const size_t k_block = marmot_neon_min_size(params->block_k, K - k0);
                if (packed_b_global == nullptr) {
                    for (size_t m_idx = 0; m_idx < m_tiles; ++m_idx) {
                        const size_t m0 = m_outer + m_idx * MARMOT_NEON_F32_TILE_M;
                        const size_t m_block = marmot_neon_min_size(MARMOT_NEON_F32_TILE_M, m_outer_end - m0);
                        float *packed_b = packed_b_panel + m_idx * params->block_k * MARMOT_NEON_F32_TILE_M;
                        if (args->layout_nt) {
                            marmot_neon_f32_pack_b_nt(weight, K, m0, m_block, k0, k_block, packed_b);
                        } else {
                            marmot_neon_f32_pack_b_nn(weight, M, m0, m_block, k0, k_block, packed_b);
                        }
                    }
                }

                size_t tile_idx = 0;
                for (size_t n0 = n_outer; n0 < n_outer_end; n0 += MARMOT_NEON_F32_TILE_N, ++tile_idx) {
                    const size_t n_block = marmot_neon_min_size(MARMOT_NEON_F32_TILE_N, n_outer_end - n0);
                    marmot_neon_f32_pack_a(input, K, n0, n_block, k0, k_block, packed_a);
                    float *c_base = c_panel + tile_idx * m_tiles * c_tile_elems;
                    for (size_t m_idx = 0; m_idx < m_tiles; ++m_idx) {
                        const size_t m0 = m_outer + m_idx * MARMOT_NEON_F32_TILE_M;
                        const size_t m_block = marmot_neon_min_size(MARMOT_NEON_F32_TILE_M, m_outer_end - m0);
                        float *c_tile = c_base + m_idx * c_tile_elems;
                        const float *packed_b = nullptr;
                        if (packed_b_global != nullptr) {
                            const size_t m_tile_idx = m0 / MARMOT_NEON_F32_TILE_M;
                            packed_b = packed_b_global + m_tile_idx * packed_b_stride + k0 * MARMOT_NEON_F32_TILE_M;
                        } else {
                            packed_b = packed_b_panel + m_idx * params->block_k * MARMOT_NEON_F32_TILE_M;
                        }
                        if (n_block == MARMOT_NEON_F32_TILE_N && m_block == MARMOT_NEON_F32_TILE_M) {
                            marmot_neon_f32_kernel_8x8(c_tile, k_block, packed_a, packed_b);
                        } else if (n_block == 4 && m_block == MARMOT_NEON_F32_TILE_M) {
                            marmot_neon_f32_kernel_4x8(c_tile, k_block, packed_a, packed_b);
                        } else if (n_block == MARMOT_NEON_F32_TILE_N && m_block == 4) {
                            marmot_neon_f32_kernel_8x4(c_tile, k_block, packed_a, packed_b);
                        } else if (n_block == 4 && m_block == 4) {
                            marmot_neon_f32_kernel_4x4(c_tile, k_block, packed_a, packed_b);
                        } else if (n_block == 2 && m_block == MARMOT_NEON_F32_TILE_M) {
                            marmot_neon_f32_kernel_2x8(c_tile, k_block, packed_a, packed_b);
                        } else if (n_block == 2 && m_block == 4) {
                            marmot_neon_f32_kernel_2x4(c_tile, k_block, packed_a, packed_b);
                        } else if (n_block == 1 && m_block == MARMOT_NEON_F32_TILE_M) {
                            marmot_neon_f32_kernel_1x8(c_tile, k_block, packed_a, packed_b);
                        } else if (n_block == 1 && m_block == 4) {
                            marmot_neon_f32_kernel_1x4(c_tile, k_block, packed_a, packed_b);
                        } else {
                            marmot_neon_f32_kernel_edge(c_tile, n_block, m_block, k_block, packed_a, packed_b);
                        }
                    }
                }
            }

            size_t tile_idx = 0;
            for (size_t n0 = n_outer; n0 < n_outer_end; n0 += MARMOT_NEON_F32_TILE_N, ++tile_idx) {
                const size_t n_block = marmot_neon_min_size(MARMOT_NEON_F32_TILE_N, n_outer_end - n0);
                const float *c_base = c_panel + tile_idx * m_tiles * c_tile_elems;
                for (size_t m_idx = 0; m_idx < m_tiles; ++m_idx) {
                    const size_t m0 = m_outer + m_idx * MARMOT_NEON_F32_TILE_M;
                    const size_t m_block = marmot_neon_min_size(MARMOT_NEON_F32_TILE_M, m_outer_end - m0);
                    const float *c_tile = c_base + m_idx * c_tile_elems;
                    marmot_neon_f32_store_tile(out, M, c_tile, n_block, m_block, n0, m0, alpha, 0.0f);
                }
            }
        }
    }

    if (local_alloc) {
        free(packed_a);
        if (packed_b_panel != nullptr) {
            free(packed_b_panel);
        }
        free(c_panel);
    }
}

typedef struct {
    const float *input;
    const float *weight;
    float *out;
    size_t N;
    size_t M;
    size_t K;
    bool layout_nt;
    float alpha;
    float beta;
    marmot_neon_f32_scratch_t *scratch_pool;
    size_t num_workers;
    const float *packed_b_global;
    size_t packed_b_stride;
} marmot_neon_f32_dispatch_ctx_t;

typedef void (*marmot_neon_dispatch_range_indexed_fn)(void *ctx, size_t start, size_t end, size_t worker_idx);

typedef struct {
    void *context;
    marmot_neon_dispatch_range_indexed_fn work;
    _Atomic size_t next_worker;
} marmot_neon_dispatch_range_indexed_ctx_t;

static void marmot_neon_dispatch_range_indexed_worker(void *ctx, size_t start, size_t end) {
    marmot_neon_dispatch_range_indexed_ctx_t *c = (marmot_neon_dispatch_range_indexed_ctx_t *)ctx;
    const size_t worker_idx = atomic_fetch_add(&c->next_worker, 1);
    c->work(c->context, start, end, worker_idx);
}

static void marmot_neon_dispatch_parallel_for_range_indexed(
    marmot_dispatch_priority_t priority, size_t count, size_t min_chunk_size, void *context,
    marmot_neon_dispatch_range_indexed_fn work
) {
    marmot_neon_dispatch_range_indexed_ctx_t ctx = {
        .context = context,
        .work = work,
        .next_worker = 0,
    };
    marmot_dispatch_parallel_for_range(
        priority, count, min_chunk_size, &ctx, marmot_neon_dispatch_range_indexed_worker
    );
}

static void marmot_neon_f32_dispatch_indexed(void *ctx, size_t row_start, size_t row_end, size_t worker_idx) {
    const marmot_neon_f32_dispatch_ctx_t *c = (const marmot_neon_f32_dispatch_ctx_t *)ctx;
    marmot_neon_f32_scratch_t *scratch = nullptr;
    if (c->scratch_pool != nullptr && worker_idx < c->num_workers) {
        scratch = &c->scratch_pool[worker_idx];
    }
    marmot_neon_f32_worker_args_t args = {
        .input = c->input,
        .weight = c->weight,
        .out = c->out,
        .N = c->N,
        .M = c->M,
        .K = c->K,
        .row_start = row_start,
        .row_end = row_end,
        .layout_nt = c->layout_nt,
        .alpha = c->alpha,
        .beta = c->beta,
        .packed_b_global = c->packed_b_global,
        .packed_b_stride = c->packed_b_stride,
        .scratch = scratch,
    };
    marmot_neon_f32_compute_rows(&args);
}

typedef void (*pack_a_fn)(
    const void *input, size_t lda, size_t row_start, size_t rows, size_t k_start, size_t k_block, float *packed
);
typedef void (*pack_b_fn)(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, float *packed
);
typedef void (*store_tile_fn)(
    void *out, size_t ldo, const float *c_tile, size_t n_block, size_t m_block, size_t row_start, size_t col_start,
    float alpha, float beta
);
typedef void (*init_tile_fn)(
    const void *out, size_t ldo, size_t n_block, size_t m_block, size_t row_start, size_t col_start, float beta,
    float *dst
);
typedef void (*fallback_fn)(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    float alpha, float beta
);

typedef struct {
    pack_a_fn pack_a;
    pack_b_fn pack_b_nt;
    pack_b_fn pack_b_nn;
    init_tile_fn init_tile;
    store_tile_fn store_tile;
    fallback_fn fallback_nt;
    fallback_fn fallback_nn;
} marmot_neon_f32_ops_t;

typedef struct {
    const void *input;
    const void *weight;
    void *out;
    size_t N;
    size_t M;
    size_t K;
    size_t row_start;
    size_t row_end;
    bool layout_nt;
    float alpha;
    float beta;
    const marmot_neon_f32_ops_t *ops;
} marmot_neon_f32_generic_worker_args_t;

static void marmot_neon_bf16_pack_a(
    const void *input, size_t lda, size_t row_start, size_t rows, size_t k_start, size_t k_block, float *packed
) {
    const size_t prefetch_k = marmot_neon_f32_get_params()->prefetch_k_ahead;
    const marmot_bfloat16_t *src = (const marmot_bfloat16_t *)input;
    for (size_t k = 0; k < k_block; ++k) {
        if (k + prefetch_k < k_block) {
            const size_t next_k = k_start + k + prefetch_k;
            for (size_t r = 0; r < rows; ++r) {
                __builtin_prefetch(src + (row_start + r) * lda + next_k, 0, 1);
            }
        }
        float *dst = packed + k * MARMOT_NEON_F32_TILE_N;
        for (size_t r = 0; r < rows; ++r) {
            const marmot_bfloat16_t *a_row = src + (row_start + r) * lda;
            dst[r] = marmot_bf16_to_f32_ref(a_row[k_start + k]);
        }
        for (size_t r = rows; r < MARMOT_NEON_F32_TILE_N; ++r) {
            dst[r] = 0.0f;
        }
    }
}

static void marmot_neon_bf16_pack_b_nt(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, float *packed
) {
    const size_t prefetch_k = marmot_neon_f32_get_params()->prefetch_k_ahead;
    const marmot_bfloat16_t *w = (const marmot_bfloat16_t *)weight;
    const size_t stride = ldw;
    if (cols >= 8) {
        for (size_t k = 0; k < k_block; ++k) {
            const marmot_bfloat16_t *base = w + col_start * stride + (k_start + k);
            if (k + prefetch_k < k_block) {
                const marmot_bfloat16_t *next = w + col_start * stride + (k_start + k + prefetch_k);
                __builtin_prefetch(next, 0, 3);
            }
            float32x4_t b0 = {
                marmot_bf16_to_f32_ref(base[0 * stride]), marmot_bf16_to_f32_ref(base[1 * stride]),
                marmot_bf16_to_f32_ref(base[2 * stride]), marmot_bf16_to_f32_ref(base[3 * stride])
            };
            float32x4_t b1 = {
                marmot_bf16_to_f32_ref(base[4 * stride]), marmot_bf16_to_f32_ref(base[5 * stride]),
                marmot_bf16_to_f32_ref(base[6 * stride]), marmot_bf16_to_f32_ref(base[7 * stride])
            };
            float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
            vst1q_f32(dst, b0);
            vst1q_f32(dst + 4, b1);
        }
        return;
    }

    for (size_t k = 0; k < k_block; ++k) {
        const marmot_bfloat16_t *w_col = w + (k_start + k);
        if (k + prefetch_k < k_block) {
            __builtin_prefetch(w + (k_start + k + prefetch_k), 0, 1);
        }
        float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
        size_t c = 0;
        for (; c + 3 < cols; c += 4) {
            float32x4_t v = {
                marmot_bf16_to_f32_ref(w_col[(col_start + c + 0) * ldw]),
                marmot_bf16_to_f32_ref(w_col[(col_start + c + 1) * ldw]),
                marmot_bf16_to_f32_ref(w_col[(col_start + c + 2) * ldw]),
                marmot_bf16_to_f32_ref(w_col[(col_start + c + 3) * ldw])
            };
            vst1q_f32(dst + c, v);
        }
        for (; c < cols; ++c) {
            dst[c] = marmot_bf16_to_f32_ref(w_col[(col_start + c) * ldw]);
        }
        for (; c < MARMOT_NEON_F32_TILE_M; ++c) {
            dst[c] = 0.0f;
        }
    }
}

static void marmot_neon_bf16_pack_b_nn(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, float *packed
) {
    const size_t prefetch_k = marmot_neon_f32_get_params()->prefetch_k_ahead;
    const marmot_bfloat16_t *w = (const marmot_bfloat16_t *)weight;
    if (cols == MARMOT_NEON_F32_TILE_M) {
        for (size_t k = 0; k < k_block; ++k) {
            float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
            const marmot_bfloat16_t *w_row = w + (k_start + k) * ldw + col_start;
            if (k + prefetch_k < k_block) {
                const marmot_bfloat16_t *next_row = w + (k_start + k + prefetch_k) * ldw + col_start;
                __builtin_prefetch(next_row, 0, 3);
            }
            for (size_t c = 0; c < MARMOT_NEON_F32_TILE_M; ++c) {
                dst[c] = marmot_bf16_to_f32_ref(w_row[c]);
            }
        }
        return;
    }

    for (size_t k = 0; k < k_block; ++k) {
        float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
        const marmot_bfloat16_t *w_row = w + (k_start + k) * ldw + col_start;
        if (k + prefetch_k < k_block) {
            const marmot_bfloat16_t *next_row = w + (k_start + k + prefetch_k) * ldw + col_start;
            __builtin_prefetch(next_row, 0, 3);
        }
        size_t c = 0;
        for (; c < cols; ++c) {
            dst[c] = marmot_bf16_to_f32_ref(w_row[c]);
        }
        for (; c < MARMOT_NEON_F32_TILE_M; ++c) {
            dst[c] = 0.0f;
        }
    }
}

static void marmot_neon_bf16_store_tile(
    void *out, size_t ldo, const float *c_tile, size_t n_block, size_t m_block, size_t row_start, size_t col_start,
    float alpha, float beta
) {
    marmot_bfloat16_t *dst_base = (marmot_bfloat16_t *)out;
    for (size_t r = 0; r < n_block; ++r) {
        marmot_bfloat16_t *dst = dst_base + (row_start + r) * ldo + col_start;
        const float *src = c_tile + r * MARMOT_NEON_F32_TILE_M;
        for (size_t c = 0; c < m_block; ++c) {
            float value = alpha * src[c];
            if (beta != 0.0f) {
                value += beta * marmot_bf16_to_f32_ref(dst[c]);
            }
            dst[c] = marmot_f32_to_bf16_ref(value);
        }
    }
}

static void marmot_neon_bf16_init_tile(
    const void *out, size_t ldo, size_t n_block, size_t m_block, size_t row_start, size_t col_start, float beta,
    float *dst
) {
    const marmot_bfloat16_t *src_base = (const marmot_bfloat16_t *)out + row_start * ldo + col_start;
    for (size_t r = 0; r < n_block; ++r) {
        const marmot_bfloat16_t *src = src_base + r * ldo;
        float *dst_row = dst + r * MARMOT_NEON_F32_TILE_M;
        size_t c = 0;
        for (; c < m_block; ++c) {
            dst_row[c] = beta * marmot_bf16_to_f32_ref(src[c]);
        }
        for (; c < MARMOT_NEON_F32_TILE_M; ++c) {
            dst_row[c] = 0.0f;
        }
    }
    for (size_t r = n_block; r < MARMOT_NEON_F32_TILE_N; ++r) {
        float *dst_row = dst + r * MARMOT_NEON_F32_TILE_M;
        for (size_t c = 0; c < MARMOT_NEON_F32_TILE_M; ++c) {
            dst_row[c] = 0.0f;
        }
    }
}

static void marmot_neon_f32_fallback_nt_bf16(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    float alpha, float beta
) {
    const marmot_bfloat16_t *a = (const marmot_bfloat16_t *)input;
    const marmot_bfloat16_t *w = (const marmot_bfloat16_t *)weight;
    marmot_bfloat16_t *o = (marmot_bfloat16_t *)out;
    for (size_t n = row_start; n < row_end; ++n) {
        const marmot_bfloat16_t *a_row = a + n * K;
        marmot_bfloat16_t *c_row = o + n * M;
        for (size_t m = 0; m < M; ++m) {
            const marmot_bfloat16_t *w_row = w + m * K;
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += marmot_bf16_to_f32_ref(a_row[k]) * marmot_bf16_to_f32_ref(w_row[k]);
            }
            float value = alpha * acc;
            if (beta != 0.0f) {
                value += beta * marmot_bf16_to_f32_ref(c_row[m]);
            }
            c_row[m] = marmot_f32_to_bf16_ref(value);
        }
    }
}

static void marmot_neon_f32_fallback_nn_bf16(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    float alpha, float beta
) {
    const marmot_bfloat16_t *a = (const marmot_bfloat16_t *)input;
    const marmot_bfloat16_t *w = (const marmot_bfloat16_t *)weight;
    marmot_bfloat16_t *o = (marmot_bfloat16_t *)out;
    for (size_t n = row_start; n < row_end; ++n) {
        const marmot_bfloat16_t *a_row = a + n * K;
        marmot_bfloat16_t *c_row = o + n * M;
        for (size_t m = 0; m < M; ++m) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += marmot_bf16_to_f32_ref(a_row[k]) * marmot_bf16_to_f32_ref(w[k * M + m]);
            }
            float value = alpha * acc;
            if (beta != 0.0f) {
                value += beta * marmot_bf16_to_f32_ref(c_row[m]);
            }
            c_row[m] = marmot_f32_to_bf16_ref(value);
        }
    }
}

static const marmot_neon_f32_ops_t sgemm_ops_bf16 = {
    .pack_a = marmot_neon_bf16_pack_a,
    .pack_b_nt = marmot_neon_bf16_pack_b_nt,
    .pack_b_nn = marmot_neon_bf16_pack_b_nn,
    .init_tile = marmot_neon_bf16_init_tile,
    .store_tile = marmot_neon_bf16_store_tile,
    .fallback_nt = marmot_neon_f32_fallback_nt_bf16,
    .fallback_nn = marmot_neon_f32_fallback_nn_bf16,
};

#if MARMOT_HAS_BF16_NATIVE
static void marmot_neon_bf16_small_direct_nt(
    const marmot_bfloat16_t *input, const marmot_bfloat16_t *weight, marmot_bfloat16_t *out, size_t N, size_t M,
    size_t K, float alpha, float beta
) {
    const bfloat16_t *a = (const bfloat16_t *)input;
    const bfloat16_t *w = (const bfloat16_t *)weight;
    bfloat16_t *o = (bfloat16_t *)out;
    const bool beta_zero = beta == 0.0f;
    for (size_t n = 0; n < N; n += 4) {
        const size_t n_block = (n + 4 <= N) ? 4 : (N - n);
        for (size_t m = 0; m < M; m += 4) {
            const size_t m_block = (m + 4 <= M) ? 4 : (M - m);
            float32x4_t acc[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
            if (n_block == 4 && m_block == 4) {
                for (size_t k = 0; k + 8 <= K; k += 8) {
                    bfloat16x8_t a0 = vld1q_bf16(a + (n + 0) * K + k);
                    bfloat16x8_t a1 = vld1q_bf16(a + (n + 1) * K + k);
                    bfloat16x8_t a2 = vld1q_bf16(a + (n + 2) * K + k);
                    bfloat16x8_t a3 = vld1q_bf16(a + (n + 3) * K + k);
                    bfloat16x8_t b0 = vld1q_bf16(w + (m + 0) * K + k);
                    bfloat16x8_t b1 = vld1q_bf16(w + (m + 1) * K + k);
                    bfloat16x8_t b2 = vld1q_bf16(w + (m + 2) * K + k);
                    bfloat16x8_t b3 = vld1q_bf16(w + (m + 3) * K + k);
                    float32x4_t dot00 = vbfdotq_f32(vdupq_n_f32(0), a0, b0);
                    float32x4_t dot01 = vbfdotq_f32(vdupq_n_f32(0), a0, b1);
                    float32x4_t dot02 = vbfdotq_f32(vdupq_n_f32(0), a0, b2);
                    float32x4_t dot03 = vbfdotq_f32(vdupq_n_f32(0), a0, b3);
                    float32x4_t dot10 = vbfdotq_f32(vdupq_n_f32(0), a1, b0);
                    float32x4_t dot11 = vbfdotq_f32(vdupq_n_f32(0), a1, b1);
                    float32x4_t dot12 = vbfdotq_f32(vdupq_n_f32(0), a1, b2);
                    float32x4_t dot13 = vbfdotq_f32(vdupq_n_f32(0), a1, b3);
                    float32x4_t dot20 = vbfdotq_f32(vdupq_n_f32(0), a2, b0);
                    float32x4_t dot21 = vbfdotq_f32(vdupq_n_f32(0), a2, b1);
                    float32x4_t dot22 = vbfdotq_f32(vdupq_n_f32(0), a2, b2);
                    float32x4_t dot23 = vbfdotq_f32(vdupq_n_f32(0), a2, b3);
                    float32x4_t dot30 = vbfdotq_f32(vdupq_n_f32(0), a3, b0);
                    float32x4_t dot31 = vbfdotq_f32(vdupq_n_f32(0), a3, b1);
                    float32x4_t dot32 = vbfdotq_f32(vdupq_n_f32(0), a3, b2);
                    float32x4_t dot33 = vbfdotq_f32(vdupq_n_f32(0), a3, b3);
                    float sum00 = vaddvq_f32(dot00);
                    float sum01 = vaddvq_f32(dot01);
                    float sum02 = vaddvq_f32(dot02);
                    float sum03 = vaddvq_f32(dot03);
                    float sum10 = vaddvq_f32(dot10);
                    float sum11 = vaddvq_f32(dot11);
                    float sum12 = vaddvq_f32(dot12);
                    float sum13 = vaddvq_f32(dot13);
                    float sum20 = vaddvq_f32(dot20);
                    float sum21 = vaddvq_f32(dot21);
                    float sum22 = vaddvq_f32(dot22);
                    float sum23 = vaddvq_f32(dot23);
                    float sum30 = vaddvq_f32(dot30);
                    float sum31 = vaddvq_f32(dot31);
                    float sum32 = vaddvq_f32(dot32);
                    float sum33 = vaddvq_f32(dot33);
                    float32x4_t row0 = {sum00, sum01, sum02, sum03};
                    float32x4_t row1 = {sum10, sum11, sum12, sum13};
                    float32x4_t row2 = {sum20, sum21, sum22, sum23};
                    float32x4_t row3 = {sum30, sum31, sum32, sum33};
                    acc[0] = vaddq_f32(acc[0], row0);
                    acc[1] = vaddq_f32(acc[1], row1);
                    acc[2] = vaddq_f32(acc[2], row2);
                    acc[3] = vaddq_f32(acc[3], row3);
                }
                size_t k_tail = K & ~7UL;
                for (size_t k = k_tail; k < K; ++k) {
                    float a0_f = marmot_bf16_to_f32_ref(input[(n + 0) * K + k]);
                    float a1_f = marmot_bf16_to_f32_ref(input[(n + 1) * K + k]);
                    float a2_f = marmot_bf16_to_f32_ref(input[(n + 2) * K + k]);
                    float a3_f = marmot_bf16_to_f32_ref(input[(n + 3) * K + k]);
                    float b0_f = marmot_bf16_to_f32_ref(weight[(m + 0) * K + k]);
                    float b1_f = marmot_bf16_to_f32_ref(weight[(m + 1) * K + k]);
                    float b2_f = marmot_bf16_to_f32_ref(weight[(m + 2) * K + k]);
                    float b3_f = marmot_bf16_to_f32_ref(weight[(m + 3) * K + k]);
                    float32x4_t b_vec = {b0_f, b1_f, b2_f, b3_f};
                    acc[0] = vfmaq_n_f32(acc[0], b_vec, a0_f);
                    acc[1] = vfmaq_n_f32(acc[1], b_vec, a1_f);
                    acc[2] = vfmaq_n_f32(acc[2], b_vec, a2_f);
                    acc[3] = vfmaq_n_f32(acc[3], b_vec, a3_f);
                }
                float32x4_t alpha_vec = vdupq_n_f32(alpha);
                if (beta_zero) {
                    float32x4_t r0 = vmulq_f32(acc[0], alpha_vec);
                    float32x4_t r1 = vmulq_f32(acc[1], alpha_vec);
                    float32x4_t r2 = vmulq_f32(acc[2], alpha_vec);
                    float32x4_t r3 = vmulq_f32(acc[3], alpha_vec);
                    bfloat16x4_t out0 = vcvt_bf16_f32(r0);
                    bfloat16x4_t out1 = vcvt_bf16_f32(r1);
                    bfloat16x4_t out2 = vcvt_bf16_f32(r2);
                    bfloat16x4_t out3 = vcvt_bf16_f32(r3);
                    vst1_bf16(o + (n + 0) * M + m, out0);
                    vst1_bf16(o + (n + 1) * M + m, out1);
                    vst1_bf16(o + (n + 2) * M + m, out2);
                    vst1_bf16(o + (n + 3) * M + m, out3);
                } else {
                    float32x4_t beta_vec = vdupq_n_f32(beta);
                    bfloat16x4_t c0_bf = vld1_bf16(o + (n + 0) * M + m);
                    bfloat16x4_t c1_bf = vld1_bf16(o + (n + 1) * M + m);
                    bfloat16x4_t c2_bf = vld1_bf16(o + (n + 2) * M + m);
                    bfloat16x4_t c3_bf = vld1_bf16(o + (n + 3) * M + m);
                    float32x4_t c0 = vcvt_f32_bf16(c0_bf);
                    float32x4_t c1 = vcvt_f32_bf16(c1_bf);
                    float32x4_t c2 = vcvt_f32_bf16(c2_bf);
                    float32x4_t c3 = vcvt_f32_bf16(c3_bf);
                    float32x4_t r0 = vfmaq_f32(vmulq_f32(c0, beta_vec), acc[0], alpha_vec);
                    float32x4_t r1 = vfmaq_f32(vmulq_f32(c1, beta_vec), acc[1], alpha_vec);
                    float32x4_t r2 = vfmaq_f32(vmulq_f32(c2, beta_vec), acc[2], alpha_vec);
                    float32x4_t r3 = vfmaq_f32(vmulq_f32(c3, beta_vec), acc[3], alpha_vec);
                    vst1_bf16(o + (n + 0) * M + m, vcvt_bf16_f32(r0));
                    vst1_bf16(o + (n + 1) * M + m, vcvt_bf16_f32(r1));
                    vst1_bf16(o + (n + 2) * M + m, vcvt_bf16_f32(r2));
                    vst1_bf16(o + (n + 3) * M + m, vcvt_bf16_f32(r3));
                }
            } else {
                float acc_scalar[4][4] = {{0}};
                for (size_t ni = 0; ni < n_block; ++ni) {
                    const marmot_bfloat16_t *a_row = input + (n + ni) * K;
                    for (size_t mi = 0; mi < m_block; ++mi) {
                        const marmot_bfloat16_t *w_row = weight + (m + mi) * K;
                        float sum = 0.0f;
                        for (size_t k = 0; k < K; ++k) {
                            sum += marmot_bf16_to_f32_ref(a_row[k]) * marmot_bf16_to_f32_ref(w_row[k]);
                        }
                        acc_scalar[ni][mi] = sum;
                    }
                }
                for (size_t ni = 0; ni < n_block; ++ni) {
                    marmot_bfloat16_t *c_row = out + (n + ni) * M + m;
                    for (size_t mi = 0; mi < m_block; ++mi) {
                        float val = alpha * acc_scalar[ni][mi];
                        if (!beta_zero) {
                            val += beta * marmot_bf16_to_f32_ref(c_row[mi]);
                        }
                        c_row[mi] = marmot_f32_to_bf16_ref(val);
                    }
                }
            }
        }
    }
}
#endif

static void marmot_neon_f32_pack_a_f16(
    const void *input, size_t lda, size_t row_start, size_t rows, size_t k_start, size_t k_block, float *packed
) {
    const size_t prefetch_k = marmot_neon_f32_get_params()->prefetch_k_ahead;
    const marmot_float16_t *src = (const marmot_float16_t *)input;
    for (size_t k = 0; k < k_block; ++k) {
        if (k + prefetch_k < k_block) {
            const size_t next_k = k_start + k + prefetch_k;
            for (size_t r = 0; r < rows; ++r) {
                __builtin_prefetch(src + (row_start + r) * lda + next_k, 0, 1);
            }
        }
        float *dst = packed + k * MARMOT_NEON_F32_TILE_N;
        for (size_t r = 0; r < rows; ++r) {
            const marmot_float16_t *a_row = src + (row_start + r) * lda;
            dst[r] = (float)marmot_float16_to_native(a_row[k_start + k]);
        }
        for (size_t r = rows; r < MARMOT_NEON_F32_TILE_N; ++r) {
            dst[r] = 0.0f;
        }
    }
}

static void marmot_neon_f32_pack_b_nt_f16(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, float *packed
) {
    const size_t prefetch_k = marmot_neon_f32_get_params()->prefetch_k_ahead;
    const marmot_float16_t *w = (const marmot_float16_t *)weight;
    const size_t stride = ldw;
    if (cols >= 8) {
        for (size_t k = 0; k < k_block; ++k) {
            const marmot_float16_t *base = w + col_start * stride + (k_start + k);
            if (k + prefetch_k < k_block) {
                const marmot_float16_t *next = w + col_start * stride + (k_start + k + prefetch_k);
                __builtin_prefetch(next, 0, 3);
            }
            float32x4_t b0 = {
                (float)marmot_float16_to_native(base[0 * stride]), (float)marmot_float16_to_native(base[1 * stride]),
                (float)marmot_float16_to_native(base[2 * stride]), (float)marmot_float16_to_native(base[3 * stride])
            };
            float32x4_t b1 = {
                (float)marmot_float16_to_native(base[4 * stride]), (float)marmot_float16_to_native(base[5 * stride]),
                (float)marmot_float16_to_native(base[6 * stride]), (float)marmot_float16_to_native(base[7 * stride])
            };
            float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
            vst1q_f32(dst, b0);
            vst1q_f32(dst + 4, b1);
        }
        return;
    }

    for (size_t k = 0; k < k_block; ++k) {
        const marmot_float16_t *w_col = w + (k_start + k);
        if (k + prefetch_k < k_block) {
            __builtin_prefetch(w + (k_start + k + prefetch_k), 0, 1);
        }
        float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
        size_t c = 0;
        for (; c + 3 < cols; c += 4) {
            float32x4_t v = {
                (float)marmot_float16_to_native(w_col[(col_start + c + 0) * ldw]),
                (float)marmot_float16_to_native(w_col[(col_start + c + 1) * ldw]),
                (float)marmot_float16_to_native(w_col[(col_start + c + 2) * ldw]),
                (float)marmot_float16_to_native(w_col[(col_start + c + 3) * ldw])
            };
            vst1q_f32(dst + c, v);
        }
        for (; c < cols; ++c) {
            dst[c] = (float)marmot_float16_to_native(w_col[(col_start + c) * ldw]);
        }
        for (; c < MARMOT_NEON_F32_TILE_M; ++c) {
            dst[c] = 0.0f;
        }
    }
}

static void marmot_neon_f32_pack_b_nn_f16(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, float *packed
) {
    const size_t prefetch_k = marmot_neon_f32_get_params()->prefetch_k_ahead;
    const marmot_float16_t *w = (const marmot_float16_t *)weight;
    if (cols == MARMOT_NEON_F32_TILE_M) {
        for (size_t k = 0; k < k_block; ++k) {
            float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
            const marmot_float16_t *w_row = w + (k_start + k) * ldw + col_start;
            if (k + prefetch_k < k_block) {
                const marmot_float16_t *next_row = w + (k_start + k + prefetch_k) * ldw + col_start;
                __builtin_prefetch(next_row, 0, 3);
            }
            for (size_t c = 0; c < MARMOT_NEON_F32_TILE_M; ++c) {
                dst[c] = (float)marmot_float16_to_native(w_row[c]);
            }
        }
        return;
    }

    for (size_t k = 0; k < k_block; ++k) {
        float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
        const marmot_float16_t *w_row = w + (k_start + k) * ldw + col_start;
        if (k + prefetch_k < k_block) {
            const marmot_float16_t *next_row = w + (k_start + k + prefetch_k) * ldw + col_start;
            __builtin_prefetch(next_row, 0, 3);
        }
        size_t c = 0;
        for (; c < cols; ++c) {
            dst[c] = (float)marmot_float16_to_native(w_row[c]);
        }
        for (; c < MARMOT_NEON_F32_TILE_M; ++c) {
            dst[c] = 0.0f;
        }
    }
}

static void marmot_neon_f32_store_tile_f16(
    void *out, size_t ldo, const float *c_tile, size_t n_block, size_t m_block, size_t row_start, size_t col_start,
    float alpha, float beta
) {
    marmot_float16_t *dst_base = (marmot_float16_t *)out;
    for (size_t r = 0; r < n_block; ++r) {
        marmot_float16_t *dst = dst_base + (row_start + r) * ldo + col_start;
        const float *src = c_tile + r * MARMOT_NEON_F32_TILE_M;
        for (size_t c = 0; c < m_block; ++c) {
            float value = alpha * src[c];
            if (beta != 0.0f) {
                value += beta * (float)marmot_float16_to_native(dst[c]);
            }
            dst[c] = marmot_native_to_float16((_Float16)value);
        }
    }
}

static void marmot_neon_f32_init_tile_f16(
    const void *out, size_t ldo, size_t n_block, size_t m_block, size_t row_start, size_t col_start, float beta,
    float *dst
) {
    const marmot_float16_t *src_base = (const marmot_float16_t *)out + row_start * ldo + col_start;
    for (size_t r = 0; r < n_block; ++r) {
        const marmot_float16_t *src = src_base + r * ldo;
        float *dst_row = dst + r * MARMOT_NEON_F32_TILE_M;
        size_t c = 0;
        for (; c < m_block; ++c) {
            dst_row[c] = beta * (float)marmot_float16_to_native(src[c]);
        }
        for (; c < MARMOT_NEON_F32_TILE_M; ++c) {
            dst_row[c] = 0.0f;
        }
    }
    for (size_t r = n_block; r < MARMOT_NEON_F32_TILE_N; ++r) {
        float *dst_row = dst + r * MARMOT_NEON_F32_TILE_M;
        for (size_t c = 0; c < MARMOT_NEON_F32_TILE_M; ++c) {
            dst_row[c] = 0.0f;
        }
    }
}

static void marmot_neon_f32_fallback_nt_f16(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    float alpha, float beta
) {
    const marmot_float16_t *a = (const marmot_float16_t *)input;
    const marmot_float16_t *w = (const marmot_float16_t *)weight;
    marmot_float16_t *o = (marmot_float16_t *)out;
    for (size_t n = row_start; n < row_end; ++n) {
        const marmot_float16_t *a_row = a + n * K;
        marmot_float16_t *c_row = o + n * M;
        for (size_t m = 0; m < M; ++m) {
            const marmot_float16_t *w_row = w + m * K;
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += (float)marmot_float16_to_native(a_row[k]) * (float)marmot_float16_to_native(w_row[k]);
            }
            float value = alpha * acc;
            if (beta != 0.0f) {
                value += beta * (float)marmot_float16_to_native(c_row[m]);
            }
            c_row[m] = marmot_native_to_float16((_Float16)value);
        }
    }
}

static void marmot_neon_f32_fallback_nn_f16(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    float alpha, float beta
) {
    const marmot_float16_t *a = (const marmot_float16_t *)input;
    const marmot_float16_t *w = (const marmot_float16_t *)weight;
    marmot_float16_t *o = (marmot_float16_t *)out;
    for (size_t n = row_start; n < row_end; ++n) {
        const marmot_float16_t *a_row = a + n * K;
        marmot_float16_t *c_row = o + n * M;
        for (size_t m = 0; m < M; ++m) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += (float)marmot_float16_to_native(a_row[k]) * (float)marmot_float16_to_native(w[k * M + m]);
            }
            float value = alpha * acc;
            if (beta != 0.0f) {
                value += beta * (float)marmot_float16_to_native(c_row[m]);
            }
            c_row[m] = marmot_native_to_float16((_Float16)value);
        }
    }
}

static const marmot_neon_f32_ops_t sgemm_ops_f16 = {
    .pack_a = marmot_neon_f32_pack_a_f16,
    .pack_b_nt = marmot_neon_f32_pack_b_nt_f16,
    .pack_b_nn = marmot_neon_f32_pack_b_nn_f16,
    .init_tile = marmot_neon_f32_init_tile_f16,
    .store_tile = marmot_neon_f32_store_tile_f16,
    .fallback_nt = marmot_neon_f32_fallback_nt_f16,
    .fallback_nn = marmot_neon_f32_fallback_nn_f16,
};

#if MARMOT_ENABLE_FP8

static inline void fp8_e4m3_convert_8_to_f32(const marmot_float8_e4m3_t *src, float *dst) {
    uint8x8_t raw = vld1_u8((const uint8_t *)src);
    uint16x8_t raw_u16 = vmovl_u8(raw);

    uint16x8_t exp = vandq_u16(vshrq_n_u16(raw_u16, 3), vdupq_n_u16(0xF));
    uint16x8_t mant = vandq_u16(raw_u16, vdupq_n_u16(0x7));
    uint16x8_t sign = vandq_u16(raw_u16, vdupq_n_u16(0x80));

    uint16x8_t exp_gt_zero = vcgtq_u16(exp, vdupq_n_u16(0));
    uint16x8_t exp_lt_15 = vcgtq_u16(vdupq_n_u16(15), exp);
    uint16x8_t fast_mask = vandq_u16(exp_gt_zero, exp_lt_15);

    uint16_t mask_arr[8];
    vst1q_u16(mask_arr, fast_mask);
    bool all_fast = true;
    for (int lane = 0; lane < 8; ++lane) {
        if (mask_arr[lane] != 0xFFFF) {
            all_fast = false;
            break;
        }
    }
    if (!all_fast) {
        for (size_t lane = 0; lane < 8; ++lane) {
            dst[lane] = marmot_fp8_e4m3_to_f32_ref(src[lane]);
        }
        return;
    }

    uint16x8_t sign_f16 = vshlq_n_u16(sign, 8);
    uint16x8_t exp_f16 = vaddq_u16(exp, vdupq_n_u16(8));
    uint16x8_t mant_f16 = vshlq_n_u16(mant, 7);

    uint16x8_t f16_bits = vorrq_u16(sign_f16, vorrq_u16(vshlq_n_u16(exp_f16, 10), mant_f16));

    float16x4_t f16_lo = vreinterpret_f16_u16(vget_low_u16(f16_bits));
    float16x4_t f16_hi = vreinterpret_f16_u16(vget_high_u16(f16_bits));

    float32x4_t f32_lo = vcvt_f32_f16(f16_lo);
    float32x4_t f32_hi = vcvt_f32_f16(f16_hi);

    vst1q_f32(dst, f32_lo);
    vst1q_f32(dst + 4, f32_hi);
}

static inline void fp8_e5m2_convert_8_to_f32(const marmot_float8_e5m2_t *src, float *dst) {
    uint8x8_t raw = vld1_u8((const uint8_t *)src);
    uint16x8_t raw_u16 = vmovl_u8(raw);

    uint16x8_t exp = vandq_u16(vshrq_n_u16(raw_u16, 2), vdupq_n_u16(0x1F));
    uint16x8_t mant = vandq_u16(raw_u16, vdupq_n_u16(0x3));
    uint16x8_t sign = vandq_u16(raw_u16, vdupq_n_u16(0x80));

    uint16x8_t exp_gt_zero = vcgtq_u16(exp, vdupq_n_u16(0));
    uint16x8_t exp_lt_31 = vcgtq_u16(vdupq_n_u16(31), exp);
    uint16x8_t fast_mask = vandq_u16(exp_gt_zero, exp_lt_31);

    uint16_t mask_arr[8];
    vst1q_u16(mask_arr, fast_mask);
    bool all_fast = true;
    for (int lane = 0; lane < 8; ++lane) {
        if (mask_arr[lane] != 0xFFFF) {
            all_fast = false;
            break;
        }
    }
    if (!all_fast) {
        for (size_t lane = 0; lane < 8; ++lane) {
            dst[lane] = marmot_fp8_e5m2_to_f32_ref(src[lane]);
        }
        return;
    }

    uint16x8_t sign_f16 = vshlq_n_u16(sign, 8);
    uint16x8_t exp_f16 = exp;
    uint16x8_t mant_f16 = vshlq_n_u16(mant, 8);

    uint16x8_t f16_bits = vorrq_u16(sign_f16, vorrq_u16(vshlq_n_u16(exp_f16, 10), mant_f16));

    float16x4_t f16_lo = vreinterpret_f16_u16(vget_low_u16(f16_bits));
    float16x4_t f16_hi = vreinterpret_f16_u16(vget_high_u16(f16_bits));

    float32x4_t f32_lo = vcvt_f32_f16(f16_lo);
    float32x4_t f32_hi = vcvt_f32_f16(f16_hi);

    vst1q_f32(dst, f32_lo);
    vst1q_f32(dst + 4, f32_hi);
}

static void marmot_neon_f32_pack_a_fp8_e4m3(
    const void *input, size_t lda, size_t row_start, size_t rows, size_t k_start, size_t k_block, float *packed
) {
    const size_t prefetch_k = marmot_neon_f32_get_params()->prefetch_k_ahead;
    const marmot_float8_e4m3_t *src = (const marmot_float8_e4m3_t *)input;
    for (size_t k = 0; k < k_block; ++k) {
        if (k + prefetch_k < k_block) {
            const size_t next_k = k_start + k + prefetch_k;
            for (size_t r = 0; r < rows; ++r) {
                __builtin_prefetch(src + (row_start + r) * lda + next_k, 0, 1);
            }
        }
        float *dst = packed + k * MARMOT_NEON_F32_TILE_N;
        for (size_t r = 0; r < rows; ++r) {
            const marmot_float8_e4m3_t *a_row = src + (row_start + r) * lda;
            dst[r] = marmot_fp8_e4m3_to_f32_ref(a_row[k_start + k]);
        }
        for (size_t r = rows; r < MARMOT_NEON_F32_TILE_N; ++r) {
            dst[r] = 0.0f;
        }
    }
}

static void marmot_neon_f32_pack_a_fp8_e5m2(
    const void *input, size_t lda, size_t row_start, size_t rows, size_t k_start, size_t k_block, float *packed
) {
    const size_t prefetch_k = marmot_neon_f32_get_params()->prefetch_k_ahead;
    const marmot_float8_e5m2_t *src = (const marmot_float8_e5m2_t *)input;
    for (size_t k = 0; k < k_block; ++k) {
        if (k + prefetch_k < k_block) {
            const size_t next_k = k_start + k + prefetch_k;
            for (size_t r = 0; r < rows; ++r) {
                __builtin_prefetch(src + (row_start + r) * lda + next_k, 0, 1);
            }
        }
        float *dst = packed + k * MARMOT_NEON_F32_TILE_N;
        for (size_t r = 0; r < rows; ++r) {
            const marmot_float8_e5m2_t *a_row = src + (row_start + r) * lda;
            dst[r] = marmot_fp8_e5m2_to_f32_ref(a_row[k_start + k]);
        }
        for (size_t r = rows; r < MARMOT_NEON_F32_TILE_N; ++r) {
            dst[r] = 0.0f;
        }
    }
}

static void marmot_neon_f32_pack_b_nt_fp8_e4m3(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, float *packed
) {
    const size_t prefetch_k = marmot_neon_f32_get_params()->prefetch_k_ahead;
    const marmot_float8_e4m3_t *w = (const marmot_float8_e4m3_t *)weight;
    const size_t stride = ldw;

    if (cols == MARMOT_NEON_F32_TILE_M) {
        for (size_t k = 0; k < k_block; ++k) {
            const marmot_float8_e4m3_t *base = w + col_start * stride + (k_start + k);
            if (k + prefetch_k < k_block) {
                const marmot_float8_e4m3_t *next = w + col_start * stride + (k_start + k + prefetch_k);
                __builtin_prefetch(next, 0, 3);
            }
            float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
            fp8_e4m3_convert_8_to_f32(
                (const marmot_float8_e4m3_t[8]){base[0 * stride], base[1 * stride], base[2 * stride], base[3 * stride],
                                                base[4 * stride], base[5 * stride], base[6 * stride], base[7 * stride]},
                dst
            );
        }
        return;
    }

    for (size_t k = 0; k < k_block; ++k) {
        const marmot_float8_e4m3_t *w_col = w + (k_start + k);
        if (k + prefetch_k < k_block) {
            __builtin_prefetch(w + (k_start + k + prefetch_k), 0, 1);
        }
        float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = marmot_fp8_e4m3_to_f32_ref(w_col[(col_start + c) * ldw]);
        }
        for (size_t c = cols; c < MARMOT_NEON_F32_TILE_M; ++c) {
            dst[c] = 0.0f;
        }
    }
}

static void marmot_neon_f32_pack_b_nt_fp8_e5m2(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, float *packed
) {
    const size_t prefetch_k = marmot_neon_f32_get_params()->prefetch_k_ahead;
    const marmot_float8_e5m2_t *w = (const marmot_float8_e5m2_t *)weight;
    const size_t stride = ldw;

    if (cols == MARMOT_NEON_F32_TILE_M) {
        for (size_t k = 0; k < k_block; ++k) {
            const marmot_float8_e5m2_t *base = w + col_start * stride + (k_start + k);
            if (k + prefetch_k < k_block) {
                const marmot_float8_e5m2_t *next = w + col_start * stride + (k_start + k + prefetch_k);
                __builtin_prefetch(next, 0, 3);
            }
            float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
            fp8_e5m2_convert_8_to_f32(
                (const marmot_float8_e5m2_t[8]){base[0 * stride], base[1 * stride], base[2 * stride], base[3 * stride],
                                                base[4 * stride], base[5 * stride], base[6 * stride], base[7 * stride]},
                dst
            );
        }
        return;
    }

    for (size_t k = 0; k < k_block; ++k) {
        const marmot_float8_e5m2_t *w_col = w + (k_start + k);
        if (k + prefetch_k < k_block) {
            __builtin_prefetch(w + (k_start + k + prefetch_k), 0, 1);
        }
        float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = marmot_fp8_e5m2_to_f32_ref(w_col[(col_start + c) * ldw]);
        }
        for (size_t c = cols; c < MARMOT_NEON_F32_TILE_M; ++c) {
            dst[c] = 0.0f;
        }
    }
}

static void marmot_neon_f32_pack_b_nn_fp8_e4m3(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, float *packed
) {
    const size_t prefetch_k = marmot_neon_f32_get_params()->prefetch_k_ahead;
    const marmot_float8_e4m3_t *w = (const marmot_float8_e4m3_t *)weight;

    if (cols == MARMOT_NEON_F32_TILE_M) {
        for (size_t k = 0; k < k_block; ++k) {
            float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
            const marmot_float8_e4m3_t *w_row = w + (k_start + k) * ldw + col_start;
            if (k + prefetch_k < k_block) {
                const marmot_float8_e4m3_t *next_row = w + (k_start + k + prefetch_k) * ldw + col_start;
                __builtin_prefetch(next_row, 0, 3);
            }
            fp8_e4m3_convert_8_to_f32(w_row, dst);
        }
        return;
    }

    for (size_t k = 0; k < k_block; ++k) {
        float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
        const marmot_float8_e4m3_t *w_row = w + (k_start + k) * ldw + col_start;
        if (k + prefetch_k < k_block) {
            const marmot_float8_e4m3_t *next_row = w + (k_start + k + prefetch_k) * ldw + col_start;
            __builtin_prefetch(next_row, 0, 3);
        }
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = marmot_fp8_e4m3_to_f32_ref(w_row[c]);
        }
        for (size_t c = cols; c < MARMOT_NEON_F32_TILE_M; ++c) {
            dst[c] = 0.0f;
        }
    }
}

static void marmot_neon_f32_pack_b_nn_fp8_e5m2(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, float *packed
) {
    const size_t prefetch_k = marmot_neon_f32_get_params()->prefetch_k_ahead;
    const marmot_float8_e5m2_t *w = (const marmot_float8_e5m2_t *)weight;

    if (cols == MARMOT_NEON_F32_TILE_M) {
        for (size_t k = 0; k < k_block; ++k) {
            float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
            const marmot_float8_e5m2_t *w_row = w + (k_start + k) * ldw + col_start;
            if (k + prefetch_k < k_block) {
                const marmot_float8_e5m2_t *next_row = w + (k_start + k + prefetch_k) * ldw + col_start;
                __builtin_prefetch(next_row, 0, 3);
            }
            fp8_e5m2_convert_8_to_f32(w_row, dst);
        }
        return;
    }

    for (size_t k = 0; k < k_block; ++k) {
        float *dst = packed + k * MARMOT_NEON_F32_TILE_M;
        const marmot_float8_e5m2_t *w_row = w + (k_start + k) * ldw + col_start;
        if (k + prefetch_k < k_block) {
            const marmot_float8_e5m2_t *next_row = w + (k_start + k + prefetch_k) * ldw + col_start;
            __builtin_prefetch(next_row, 0, 3);
        }
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = marmot_fp8_e5m2_to_f32_ref(w_row[c]);
        }
        for (size_t c = cols; c < MARMOT_NEON_F32_TILE_M; ++c) {
            dst[c] = 0.0f;
        }
    }
}

static void marmot_neon_f32_init_tile_fp8_f32(
    const void *out, size_t ldo, size_t n_block, size_t m_block, size_t row_start, size_t col_start, float beta,
    float *dst
) {
    const float *src_base = (const float *)out + row_start * ldo + col_start;
    for (size_t r = 0; r < n_block; ++r) {
        const float *src = src_base + r * ldo;
        float *dst_row = dst + r * MARMOT_NEON_F32_TILE_M;
        size_t c = 0;
        for (; c < m_block; ++c) {
            dst_row[c] = beta * src[c];
        }
        for (; c < MARMOT_NEON_F32_TILE_M; ++c) {
            dst_row[c] = 0.0f;
        }
    }
    for (size_t r = n_block; r < MARMOT_NEON_F32_TILE_N; ++r) {
        float *dst_row = dst + r * MARMOT_NEON_F32_TILE_M;
        for (size_t c = 0; c < MARMOT_NEON_F32_TILE_M; ++c) {
            dst_row[c] = 0.0f;
        }
    }
}

static void marmot_neon_f32_store_tile_fp8_f32(
    void *out, size_t ldo, const float *c_tile, size_t n_block, size_t m_block, size_t row_start, size_t col_start,
    float alpha, float beta
) {
    float *dst_base = (float *)out + row_start * ldo + col_start;
    for (size_t r = 0; r < n_block; ++r) {
        float *dst = dst_base + r * ldo;
        const float *src = c_tile + r * MARMOT_NEON_F32_TILE_M;
        for (size_t c = 0; c < m_block; ++c) {
            float value = alpha * src[c];
            if (beta != 0.0f) {
                value += beta * dst[c];
            }
            dst[c] = value;
        }
    }
}

static void marmot_neon_f32_fallback_nt_fp8_e4m3(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    float alpha, float beta
) {
    const marmot_float8_e4m3_t *a = (const marmot_float8_e4m3_t *)input;
    const marmot_float8_e4m3_t *w = (const marmot_float8_e4m3_t *)weight;
    float *o = (float *)out;
    for (size_t n = row_start; n < row_end && n < N; ++n) {
        const marmot_float8_e4m3_t *a_row = a + n * K;
        float *c_row = o + n * M;
        for (size_t m = 0; m < M; ++m) {
            const marmot_float8_e4m3_t *w_row = w + m * K;
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += marmot_fp8_e4m3_to_f32_ref(a_row[k]) * marmot_fp8_e4m3_to_f32_ref(w_row[k]);
            }
            float value = alpha * acc;
            if (beta != 0.0f) {
                value += beta * c_row[m];
            }
            c_row[m] = value;
        }
    }
}

static void marmot_neon_f32_fallback_nn_fp8_e4m3(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    float alpha, float beta
) {
    const marmot_float8_e4m3_t *a = (const marmot_float8_e4m3_t *)input;
    const marmot_float8_e4m3_t *w = (const marmot_float8_e4m3_t *)weight;
    float *o = (float *)out;
    for (size_t n = row_start; n < row_end && n < N; ++n) {
        const marmot_float8_e4m3_t *a_row = a + n * K;
        float *c_row = o + n * M;
        for (size_t m = 0; m < M; ++m) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += marmot_fp8_e4m3_to_f32_ref(a_row[k]) * marmot_fp8_e4m3_to_f32_ref(w[k * M + m]);
            }
            float value = alpha * acc;
            if (beta != 0.0f) {
                value += beta * c_row[m];
            }
            c_row[m] = value;
        }
    }
}

static void marmot_neon_f32_fallback_nt_fp8_e5m2(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    float alpha, float beta
) {
    const marmot_float8_e5m2_t *a = (const marmot_float8_e5m2_t *)input;
    const marmot_float8_e5m2_t *w = (const marmot_float8_e5m2_t *)weight;
    float *o = (float *)out;
    for (size_t n = row_start; n < row_end && n < N; ++n) {
        const marmot_float8_e5m2_t *a_row = a + n * K;
        float *c_row = o + n * M;
        for (size_t m = 0; m < M; ++m) {
            const marmot_float8_e5m2_t *w_row = w + m * K;
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += marmot_fp8_e5m2_to_f32_ref(a_row[k]) * marmot_fp8_e5m2_to_f32_ref(w_row[k]);
            }
            float value = alpha * acc;
            if (beta != 0.0f) {
                value += beta * c_row[m];
            }
            c_row[m] = value;
        }
    }
}

static void marmot_neon_f32_fallback_nn_fp8_e5m2(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    float alpha, float beta
) {
    const marmot_float8_e5m2_t *a = (const marmot_float8_e5m2_t *)input;
    const marmot_float8_e5m2_t *w = (const marmot_float8_e5m2_t *)weight;
    float *o = (float *)out;
    for (size_t n = row_start; n < row_end && n < N; ++n) {
        const marmot_float8_e5m2_t *a_row = a + n * K;
        float *c_row = o + n * M;
        for (size_t m = 0; m < M; ++m) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += marmot_fp8_e5m2_to_f32_ref(a_row[k]) * marmot_fp8_e5m2_to_f32_ref(w[k * M + m]);
            }
            float value = alpha * acc;
            if (beta != 0.0f) {
                value += beta * c_row[m];
            }
            c_row[m] = value;
        }
    }
}

static const marmot_neon_f32_ops_t sgemm_ops_fp8_e4m3 = {
    .pack_a = marmot_neon_f32_pack_a_fp8_e4m3,
    .pack_b_nt = marmot_neon_f32_pack_b_nt_fp8_e4m3,
    .pack_b_nn = marmot_neon_f32_pack_b_nn_fp8_e4m3,
    .init_tile = marmot_neon_f32_init_tile_fp8_f32,
    .store_tile = marmot_neon_f32_store_tile_fp8_f32,
    .fallback_nt = marmot_neon_f32_fallback_nt_fp8_e4m3,
    .fallback_nn = marmot_neon_f32_fallback_nn_fp8_e4m3,
};

static const marmot_neon_f32_ops_t sgemm_ops_fp8_e5m2 = {
    .pack_a = marmot_neon_f32_pack_a_fp8_e5m2,
    .pack_b_nt = marmot_neon_f32_pack_b_nt_fp8_e5m2,
    .pack_b_nn = marmot_neon_f32_pack_b_nn_fp8_e5m2,
    .init_tile = marmot_neon_f32_init_tile_fp8_f32,
    .store_tile = marmot_neon_f32_store_tile_fp8_f32,
    .fallback_nt = marmot_neon_f32_fallback_nt_fp8_e5m2,
    .fallback_nn = marmot_neon_f32_fallback_nn_fp8_e5m2,
};

#endif // MARMOT_ENABLE_FP8

static void marmot_neon_f32_compute_rows_generic(const marmot_neon_f32_generic_worker_args_t *args) {
    const void *input = args->input;
    const void *weight = args->weight;
    void *out = args->out;
    const size_t N = args->N;
    const size_t M = args->M;
    const size_t K = args->K;
    const size_t row_start = args->row_start;
    const size_t row_end = args->row_end;
    const marmot_neon_f32_params_t *params = marmot_neon_f32_get_params();
    const marmot_neon_f32_ops_t *ops = args->ops;

    const size_t pack_a_elems = params->block_k * MARMOT_NEON_F32_TILE_N;
    const size_t max_k_blocks = (K + params->block_k - 1) / params->block_k;
    float *packed_a = (float *)marmot_aligned_alloc(64, pack_a_elems * sizeof(float));
    float **packed_b_cache = (float **)malloc(max_k_blocks * sizeof(float *));
    const size_t block_n_cap = marmot_neon_min_size(params->block_n, row_end - row_start);
    const size_t n_tiles_cap = (block_n_cap + MARMOT_NEON_F32_TILE_N - 1) / MARMOT_NEON_F32_TILE_N;
    const size_t c_tile_elems = MARMOT_NEON_F32_TILE_N * MARMOT_NEON_F32_TILE_M;
    float *c_panel = (float *)marmot_aligned_alloc(64, n_tiles_cap * c_tile_elems * sizeof(float));
    if (packed_b_cache != nullptr) {
        for (size_t i = 0; i < max_k_blocks; ++i) {
            packed_b_cache[i] = nullptr;
        }
    }
    if (packed_a == nullptr || packed_b_cache == nullptr || c_panel == nullptr) {
        free(packed_a);
        if (packed_b_cache != nullptr) {
            free(packed_b_cache);
        }
        free(c_panel);
        if (args->layout_nt) {
            ops->fallback_nt(input, weight, out, N, M, K, row_start, row_end, args->alpha, args->beta);
        } else {
            ops->fallback_nn(input, weight, out, N, M, K, row_start, row_end, args->alpha, args->beta);
        }
        return;
    }
    const float alpha = args->alpha;
    const float beta = args->beta;
    const bool beta_zero = beta == 0.0f;

    bool pack_failed = false;
    for (size_t m_outer = 0; m_outer < M; m_outer += params->block_m) {
        const size_t m_outer_end = marmot_neon_min_size(M, m_outer + params->block_m);
        for (size_t m0 = m_outer; m0 < m_outer_end; m0 += MARMOT_NEON_F32_TILE_M) {
            const size_t m_block = marmot_neon_min_size(MARMOT_NEON_F32_TILE_M, m_outer_end - m0);
            for (size_t i = 0; i < max_k_blocks; ++i) {
                packed_b_cache[i] = nullptr;
            }
            for (size_t n_outer = row_start; n_outer < row_end; n_outer += params->block_n) {
                const size_t n_outer_end = marmot_neon_min_size(row_end, n_outer + params->block_n);
                const size_t n_tiles = (n_outer_end - n_outer + MARMOT_NEON_F32_TILE_N - 1) / MARMOT_NEON_F32_TILE_N;
                if (beta_zero) {
                    memset(c_panel, 0, n_tiles * c_tile_elems * sizeof(float));
                } else {
                    size_t init_idx = 0;
                    for (size_t n0 = n_outer; n0 < n_outer_end; n0 += MARMOT_NEON_F32_TILE_N, ++init_idx) {
                        const size_t n_block = marmot_neon_min_size(MARMOT_NEON_F32_TILE_N, n_outer_end - n0);
                        float *c_tile = c_panel + init_idx * c_tile_elems;
                        ops->init_tile(out, M, n_block, m_block, n0, m0, beta, c_tile);
                    }
                }
                size_t k_idx = 0;
                for (size_t k0 = 0; k0 < K; k0 += params->block_k, ++k_idx) {
                    const size_t k_block = marmot_neon_min_size(params->block_k, K - k0);
                    float *packed_b = packed_b_cache[k_idx];
                    if (packed_b == nullptr) {
                        packed_b = (float *)marmot_aligned_alloc(64, k_block * MARMOT_NEON_F32_TILE_M * sizeof(float));
                        if (packed_b == nullptr) {
                            pack_failed = true;
                            goto cleanup_generic;
                        }
                        packed_b_cache[k_idx] = packed_b;
                        if (args->layout_nt) {
                            ops->pack_b_nt(weight, K, m0, m_block, k0, k_block, packed_b);
                        } else {
                            ops->pack_b_nn(weight, M, m0, m_block, k0, k_block, packed_b);
                        }
                    }
                    size_t tile_idx = 0;
                    for (size_t n0 = n_outer; n0 < n_outer_end; n0 += MARMOT_NEON_F32_TILE_N, ++tile_idx) {
                        const size_t n_block = marmot_neon_min_size(MARMOT_NEON_F32_TILE_N, n_outer_end - n0);
                        float *c_tile = c_panel + tile_idx * c_tile_elems;
                        ops->pack_a(input, K, n0, n_block, k0, k_block, packed_a);
                        if (n_block == MARMOT_NEON_F32_TILE_N && m_block == MARMOT_NEON_F32_TILE_M) {
                            marmot_neon_f32_kernel_8x8(c_tile, k_block, packed_a, packed_b);
                        } else if (n_block == 4 && m_block == MARMOT_NEON_F32_TILE_M) {
                            marmot_neon_f32_kernel_4x8(c_tile, k_block, packed_a, packed_b);
                        } else if (n_block == MARMOT_NEON_F32_TILE_N && m_block == 4) {
                            marmot_neon_f32_kernel_8x4(c_tile, k_block, packed_a, packed_b);
                        } else if (n_block == 4 && m_block == 4) {
                            marmot_neon_f32_kernel_4x4(c_tile, k_block, packed_a, packed_b);
                        } else if (n_block == 2 && m_block == MARMOT_NEON_F32_TILE_M) {
                            marmot_neon_f32_kernel_2x8(c_tile, k_block, packed_a, packed_b);
                        } else if (n_block == 2 && m_block == 4) {
                            marmot_neon_f32_kernel_2x4(c_tile, k_block, packed_a, packed_b);
                        } else if (n_block == 1 && m_block == MARMOT_NEON_F32_TILE_M) {
                            marmot_neon_f32_kernel_1x8(c_tile, k_block, packed_a, packed_b);
                        } else if (n_block == 1 && m_block == 4) {
                            marmot_neon_f32_kernel_1x4(c_tile, k_block, packed_a, packed_b);
                        } else {
                            marmot_neon_f32_kernel_edge(c_tile, n_block, m_block, k_block, packed_a, packed_b);
                        }
                    }
                }
                size_t tile_idx = 0;
                for (size_t n0 = n_outer; n0 < n_outer_end; n0 += MARMOT_NEON_F32_TILE_N, ++tile_idx) {
                    const size_t n_block = marmot_neon_min_size(MARMOT_NEON_F32_TILE_N, n_outer_end - n0);
                    const float *c_tile = c_panel + tile_idx * c_tile_elems;
                    ops->store_tile(out, M, c_tile, n_block, m_block, n0, m0, alpha, 0.0f);
                }
            }
            for (size_t i = 0; i < max_k_blocks; ++i) {
                free(packed_b_cache[i]);
                packed_b_cache[i] = nullptr;
            }
        }
    }

cleanup_generic:
    free(packed_a);
    for (size_t i = 0; i < max_k_blocks; ++i) {
        free(packed_b_cache[i]);
    }
    free(packed_b_cache);
    free(c_panel);
    if (pack_failed) {
        if (args->layout_nt) {
            ops->fallback_nt(input, weight, out, N, M, K, row_start, row_end, args->alpha, args->beta);
        } else {
            ops->fallback_nn(input, weight, out, N, M, K, row_start, row_end, args->alpha, args->beta);
        }
    }
}

typedef struct {
    const void *input;
    const void *weight;
    void *out;
    size_t N;
    size_t M;
    size_t K;
    bool layout_nt;
    float alpha;
    float beta;
    const marmot_neon_f32_ops_t *ops;
} marmot_neon_f32_generic_dispatch_ctx_t;

static void marmot_neon_f32_generic_dispatch(void *ctx, size_t row_start, size_t row_end) {
    const marmot_neon_f32_generic_dispatch_ctx_t *c = (const marmot_neon_f32_generic_dispatch_ctx_t *)ctx;
    marmot_neon_f32_generic_worker_args_t args = {
        .input = c->input,
        .weight = c->weight,
        .out = c->out,
        .N = c->N,
        .M = c->M,
        .K = c->K,
        .row_start = row_start,
        .row_end = row_end,
        .layout_nt = c->layout_nt,
        .alpha = c->alpha,
        .beta = c->beta,
        .ops = c->ops,
    };
    marmot_neon_f32_compute_rows_generic(&args);
}

static void marmot_neon_f32_run_generic(
    const void *device_ctx, const void *input, const void *weight, void *out, size_t N, size_t K, size_t M,
    bool layout_nt, float alpha, float beta, const marmot_neon_f32_ops_t *ops
) {
    (void)device_ctx;

    if (N < MARMOT_NEON_F32_MIN_DIM || M < MARMOT_NEON_F32_MIN_DIM || K < MARMOT_NEON_F32_MIN_DIM) {
        if (layout_nt) {
            ops->fallback_nt(input, weight, out, N, M, K, 0, N, alpha, beta);
        } else {
            ops->fallback_nn(input, weight, out, N, M, K, 0, N, alpha, beta);
        }
        return;
    }

    (void)marmot_neon_f32_get_params();

    marmot_neon_f32_generic_dispatch_ctx_t dctx = {
        .input = input,
        .weight = weight,
        .out = out,
        .N = N,
        .M = M,
        .K = K,
        .layout_nt = layout_nt,
        .alpha = alpha,
        .beta = beta,
        .ops = ops,
    };
    marmot_dispatch_parallel_for_range(MARMOT_DISPATCH_PRIORITY_HIGH, N, 64, &dctx, marmot_neon_f32_generic_dispatch);
}

static void marmot_neon_f32_run(
    const void *device_ctx, const float *input, const float *weight, float *out, size_t N, size_t K, size_t M,
    bool layout_nt, float alpha, float beta
) {
    if (N < MARMOT_NEON_F32_MIN_DIM || M < MARMOT_NEON_F32_MIN_DIM || K < MARMOT_NEON_F32_MIN_DIM) {
        if (layout_nt) {
            marmot_neon_f32_fallback_nt(input, weight, out, N, M, K, 0, N, alpha, beta);
        } else {
            marmot_neon_f32_fallback_nn(input, weight, out, N, M, K, 0, N, alpha, beta);
        }
        return;
    }
    if (layout_nt && marmot_neon_f32_use_small_kernel(N, M, K)) {
        marmot_neon_f32_small_direct_nt(input, weight, out, N, M, K, alpha, beta);
        return;
    }

    const marmot_neon_f32_params_t *params = marmot_neon_f32_get_params();
    float *packed_b_global = nullptr;
    size_t packed_b_stride = 0;
    const size_t m_tiles = (M + MARMOT_NEON_F32_TILE_M - 1) / MARMOT_NEON_F32_TILE_M;
    if (m_tiles > 0 && K > 0 && K <= SIZE_MAX / MARMOT_NEON_F32_TILE_M) {
        const size_t stride = K * MARMOT_NEON_F32_TILE_M;
        if (stride > 0 && m_tiles <= SIZE_MAX / stride) {
            const size_t packed_b_elems = m_tiles * stride;
            packed_b_global = (float *)marmot_aligned_alloc(64, packed_b_elems * sizeof(float));
            if (packed_b_global != nullptr) {
                marmot_neon_f32_pack_b_global_ctx_t pctx = {
                    .weight = weight,
                    .M = M,
                    .K = K,
                    .block_k = params->block_k,
                    .packed_b_stride = stride,
                    .packed_b = packed_b_global,
                    .layout_nt = layout_nt,
                };
                marmot_dispatch_parallel_for(
                    MARMOT_DISPATCH_PRIORITY_HIGH, m_tiles, &pctx, marmot_neon_f32_pack_b_global_worker
                );
                packed_b_stride = stride;
            }
        }
    }

    const cpu_context_t *ctx = (const cpu_context_t *)device_ctx;
    marmot_neon_f32_dispatch_ctx_t dctx = {
        .input = input,
        .weight = weight,
        .out = out,
        .N = N,
        .M = M,
        .K = K,
        .layout_nt = layout_nt,
        .alpha = alpha,
        .beta = beta,
        .scratch_pool = ctx != nullptr ? ctx->neon_scratch_pool.f32 : nullptr,
        .num_workers = ctx != nullptr ? ctx->neon_scratch_pool.num_workers : 0,
        .packed_b_global = packed_b_global,
        .packed_b_stride = packed_b_stride,
    };
    marmot_neon_dispatch_parallel_for_range_indexed(
        MARMOT_DISPATCH_PRIORITY_HIGH, N, 64, &dctx, marmot_neon_f32_dispatch_indexed
    );
    free(packed_b_global);
}

static void dgemm_pack_a_tile(
    const double *input, size_t lda, size_t row_start, size_t rows, size_t k_start, size_t k_block, double *packed
) {
    const size_t prefetch_k = dgemm_neon_get_params()->prefetch_k_ahead;
    for (size_t k = 0; k < k_block; ++k) {
        if (k + prefetch_k < k_block) {
            const size_t next_k = k_start + k + prefetch_k;
            for (size_t r = 0; r < rows; ++r) {
                __builtin_prefetch(input + (row_start + r) * lda + next_k, 0, 1);
            }
        }
        double *dst = packed + k * DGEMM_NEON_TILE_N;
        for (size_t r = 0; r < rows; ++r) {
            const double *a_row = input + (row_start + r) * lda;
            dst[r] = a_row[k_start + k];
        }
        for (size_t r = rows; r < DGEMM_NEON_TILE_N; ++r) {
            dst[r] = 0.0;
        }
    }
}

static void dgemm_pack_b_tile_nt(
    const double *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, double *packed
) {
    const size_t prefetch_k = dgemm_neon_get_params()->prefetch_k_ahead;
    const size_t stride = ldw;
    if (cols >= DGEMM_NEON_TILE_M) {
        for (size_t k = 0; k < k_block; ++k) {
            const double *base = weight + col_start * stride + (k_start + k);
            if (k + prefetch_k < k_block) {
                const double *next = weight + col_start * stride + (k_start + k + prefetch_k);
                __builtin_prefetch(next, 0, 3);
            }
            float64x2_t b0 = {base[0 * stride], base[1 * stride]};
            float64x2_t b1 = {base[2 * stride], base[3 * stride]};
            double *dst = packed + k * DGEMM_NEON_TILE_M;
            vst1q_f64(dst, b0);
            vst1q_f64(dst + 2, b1);
        }
        return;
    }

    for (size_t k = 0; k < k_block; ++k) {
        const double *w_col = weight + (k_start + k);
        if (k + prefetch_k < k_block) {
            __builtin_prefetch(weight + (k_start + k + prefetch_k), 0, 1);
        }
        double *dst = packed + k * DGEMM_NEON_TILE_M;
        size_t c = 0;
        for (; c + 1 < cols; c += 2) {
            float64x2_t v = {w_col[(col_start + c + 0) * ldw], w_col[(col_start + c + 1) * ldw]};
            vst1q_f64(dst + c, v);
        }
        for (; c < cols; ++c) {
            dst[c] = w_col[(col_start + c) * ldw];
        }
        for (; c < DGEMM_NEON_TILE_M; ++c) {
            dst[c] = 0.0;
        }
    }
}

static void dgemm_pack_b_tile_nn(
    const double *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, double *packed
) {
    const size_t prefetch_k = dgemm_neon_get_params()->prefetch_k_ahead;
    if (cols == DGEMM_NEON_TILE_M) {
        for (size_t k = 0; k < k_block; ++k) {
            double *dst = packed + k * DGEMM_NEON_TILE_M;
            const double *w_row = weight + (k_start + k) * ldw + col_start;
            if (k + prefetch_k < k_block) {
                const double *next_row = weight + (k_start + k + prefetch_k) * ldw + col_start;
                __builtin_prefetch(next_row, 0, 3);
            }
            float64x2_t b0 = vld1q_f64(w_row);
            float64x2_t b1 = vld1q_f64(w_row + 2);
            vst1q_f64(dst, b0);
            vst1q_f64(dst + 2, b1);
        }
        return;
    }

    for (size_t k = 0; k < k_block; ++k) {
        double *dst = packed + k * DGEMM_NEON_TILE_M;
        const double *w_row = weight + (k_start + k) * ldw + col_start;
        if (k + prefetch_k < k_block) {
            const double *next_row = weight + (k_start + k + prefetch_k) * ldw + col_start;
            __builtin_prefetch(next_row, 0, 3);
        }
        memcpy(dst, w_row, cols * sizeof(double));
        for (size_t c = cols; c < DGEMM_NEON_TILE_M; ++c) {
            dst[c] = 0.0;
        }
    }
}

static void dgemm_kernel_8x4(double *c_tile, size_t k_block, const double *packed_a, const double *packed_b) {
    float64x2_t c00 = vld1q_f64(c_tile + 0);
    float64x2_t c01 = vld1q_f64(c_tile + 2);
    float64x2_t c10 = vld1q_f64(c_tile + 4);
    float64x2_t c11 = vld1q_f64(c_tile + 6);
    float64x2_t c20 = vld1q_f64(c_tile + 8);
    float64x2_t c21 = vld1q_f64(c_tile + 10);
    float64x2_t c30 = vld1q_f64(c_tile + 12);
    float64x2_t c31 = vld1q_f64(c_tile + 14);
    float64x2_t c40 = vld1q_f64(c_tile + 16);
    float64x2_t c41 = vld1q_f64(c_tile + 18);
    float64x2_t c50 = vld1q_f64(c_tile + 20);
    float64x2_t c51 = vld1q_f64(c_tile + 22);
    float64x2_t c60 = vld1q_f64(c_tile + 24);
    float64x2_t c61 = vld1q_f64(c_tile + 26);
    float64x2_t c70 = vld1q_f64(c_tile + 28);
    float64x2_t c71 = vld1q_f64(c_tile + 30);

    const double *pa = packed_a;
    const double *pb = packed_b;
    const size_t prefetch_k = dgemm_neon_get_params()->prefetch_k_ahead;
    const size_t a_stride = DGEMM_NEON_TILE_N;
    const size_t b_stride = DGEMM_NEON_TILE_M;

#if defined(__aarch64__)
#define DGEMM_8X4_ACCUM(a0, a1, a2, a3, b01, b23)                                                                      \
    do {                                                                                                               \
        c00 = vfmaq_laneq_f64(c00, (b01), (a0), 0);                                                                    \
        c01 = vfmaq_laneq_f64(c01, (b23), (a0), 0);                                                                    \
        c10 = vfmaq_laneq_f64(c10, (b01), (a0), 1);                                                                    \
        c11 = vfmaq_laneq_f64(c11, (b23), (a0), 1);                                                                    \
        c20 = vfmaq_laneq_f64(c20, (b01), (a1), 0);                                                                    \
        c21 = vfmaq_laneq_f64(c21, (b23), (a1), 0);                                                                    \
        c30 = vfmaq_laneq_f64(c30, (b01), (a1), 1);                                                                    \
        c31 = vfmaq_laneq_f64(c31, (b23), (a1), 1);                                                                    \
        c40 = vfmaq_laneq_f64(c40, (b01), (a2), 0);                                                                    \
        c41 = vfmaq_laneq_f64(c41, (b23), (a2), 0);                                                                    \
        c50 = vfmaq_laneq_f64(c50, (b01), (a2), 1);                                                                    \
        c51 = vfmaq_laneq_f64(c51, (b23), (a2), 1);                                                                    \
        c60 = vfmaq_laneq_f64(c60, (b01), (a3), 0);                                                                    \
        c61 = vfmaq_laneq_f64(c61, (b23), (a3), 0);                                                                    \
        c70 = vfmaq_laneq_f64(c70, (b01), (a3), 1);                                                                    \
        c71 = vfmaq_laneq_f64(c71, (b23), (a3), 1);                                                                    \
    } while (0)
#else
#define DGEMM_8X4_ACCUM(a0, a1, a2, a3, b01, b23)                                                                      \
    do {                                                                                                               \
        const double a0_lane0 = vgetq_lane_f64((a0), 0);                                                               \
        const double a0_lane1 = vgetq_lane_f64((a0), 1);                                                               \
        const double a1_lane0 = vgetq_lane_f64((a1), 0);                                                               \
        const double a1_lane1 = vgetq_lane_f64((a1), 1);                                                               \
        const double a2_lane0 = vgetq_lane_f64((a2), 0);                                                               \
        const double a2_lane1 = vgetq_lane_f64((a2), 1);                                                               \
        const double a3_lane0 = vgetq_lane_f64((a3), 0);                                                               \
        const double a3_lane1 = vgetq_lane_f64((a3), 1);                                                               \
        c00 = vmlaq_n_f64(c00, (b01), a0_lane0);                                                                       \
        c01 = vmlaq_n_f64(c01, (b23), a0_lane0);                                                                       \
        c10 = vmlaq_n_f64(c10, (b01), a0_lane1);                                                                       \
        c11 = vmlaq_n_f64(c11, (b23), a0_lane1);                                                                       \
        c20 = vmlaq_n_f64(c20, (b01), a1_lane0);                                                                       \
        c21 = vmlaq_n_f64(c21, (b23), a1_lane0);                                                                       \
        c30 = vmlaq_n_f64(c30, (b01), a1_lane1);                                                                       \
        c31 = vmlaq_n_f64(c31, (b23), a1_lane1);                                                                       \
        c40 = vmlaq_n_f64(c40, (b01), a2_lane0);                                                                       \
        c41 = vmlaq_n_f64(c41, (b23), a2_lane0);                                                                       \
        c50 = vmlaq_n_f64(c50, (b01), a2_lane1);                                                                       \
        c51 = vmlaq_n_f64(c51, (b23), a2_lane1);                                                                       \
        c60 = vmlaq_n_f64(c60, (b01), a3_lane0);                                                                       \
        c61 = vmlaq_n_f64(c61, (b23), a3_lane0);                                                                       \
        c70 = vmlaq_n_f64(c70, (b01), a3_lane1);                                                                       \
        c71 = vmlaq_n_f64(c71, (b23), a3_lane1);                                                                       \
    } while (0)
#endif

    size_t k = 0;
    for (; k + 3 < k_block; k += 4) {
        if (k + prefetch_k < k_block) {
            __builtin_prefetch(pa + prefetch_k * a_stride, 0, 3);
            __builtin_prefetch(pb + prefetch_k * b_stride, 0, 3);
        }
        float64x2_t b01_0 = vld1q_f64(pb);
        float64x2_t b23_0 = vld1q_f64(pb + 2);
        float64x2_t a0_0 = vld1q_f64(pa + 0);
        float64x2_t a1_0 = vld1q_f64(pa + 2);
        float64x2_t a2_0 = vld1q_f64(pa + 4);
        float64x2_t a3_0 = vld1q_f64(pa + 6);

        float64x2_t b01_1 = vld1q_f64(pb + b_stride);
        float64x2_t b23_1 = vld1q_f64(pb + b_stride + 2);
        float64x2_t a0_1 = vld1q_f64(pa + a_stride + 0);
        float64x2_t a1_1 = vld1q_f64(pa + a_stride + 2);
        float64x2_t a2_1 = vld1q_f64(pa + a_stride + 4);
        float64x2_t a3_1 = vld1q_f64(pa + a_stride + 6);

        DGEMM_8X4_ACCUM(a0_0, a1_0, a2_0, a3_0, b01_0, b23_0);

        float64x2_t b01_2 = vld1q_f64(pb + 2 * b_stride);
        float64x2_t b23_2 = vld1q_f64(pb + 2 * b_stride + 2);
        float64x2_t a0_2 = vld1q_f64(pa + 2 * a_stride + 0);
        float64x2_t a1_2 = vld1q_f64(pa + 2 * a_stride + 2);
        float64x2_t a2_2 = vld1q_f64(pa + 2 * a_stride + 4);
        float64x2_t a3_2 = vld1q_f64(pa + 2 * a_stride + 6);

        DGEMM_8X4_ACCUM(a0_1, a1_1, a2_1, a3_1, b01_1, b23_1);

        float64x2_t b01_3 = vld1q_f64(pb + 3 * b_stride);
        float64x2_t b23_3 = vld1q_f64(pb + 3 * b_stride + 2);
        float64x2_t a0_3 = vld1q_f64(pa + 3 * a_stride + 0);
        float64x2_t a1_3 = vld1q_f64(pa + 3 * a_stride + 2);
        float64x2_t a2_3 = vld1q_f64(pa + 3 * a_stride + 4);
        float64x2_t a3_3 = vld1q_f64(pa + 3 * a_stride + 6);

        DGEMM_8X4_ACCUM(a0_2, a1_2, a2_2, a3_2, b01_2, b23_2);
        DGEMM_8X4_ACCUM(a0_3, a1_3, a2_3, a3_3, b01_3, b23_3);

        pa += 4 * a_stride;
        pb += 4 * b_stride;
    }

    for (; k < k_block; ++k) {
        if (k + prefetch_k < k_block) {
            __builtin_prefetch(pa + prefetch_k * a_stride, 0, 3);
            __builtin_prefetch(pb + prefetch_k * b_stride, 0, 3);
        }
        const float64x2_t b01 = vld1q_f64(pb);
        const float64x2_t b23 = vld1q_f64(pb + 2);
        const float64x2_t a0 = vld1q_f64(pa + 0);
        const float64x2_t a1 = vld1q_f64(pa + 2);
        const float64x2_t a2 = vld1q_f64(pa + 4);
        const float64x2_t a3 = vld1q_f64(pa + 6);

        DGEMM_8X4_ACCUM(a0, a1, a2, a3, b01, b23);

        pa += a_stride;
        pb += b_stride;
    }

#undef DGEMM_8X4_ACCUM

    vst1q_f64(c_tile + 0, c00);
    vst1q_f64(c_tile + 2, c01);
    vst1q_f64(c_tile + 4, c10);
    vst1q_f64(c_tile + 6, c11);
    vst1q_f64(c_tile + 8, c20);
    vst1q_f64(c_tile + 10, c21);
    vst1q_f64(c_tile + 12, c30);
    vst1q_f64(c_tile + 14, c31);
    vst1q_f64(c_tile + 16, c40);
    vst1q_f64(c_tile + 18, c41);
    vst1q_f64(c_tile + 20, c50);
    vst1q_f64(c_tile + 22, c51);
    vst1q_f64(c_tile + 24, c60);
    vst1q_f64(c_tile + 26, c61);
    vst1q_f64(c_tile + 28, c70);
    vst1q_f64(c_tile + 30, c71);
}

static void dgemm_kernel_edge(
    double *c_tile, size_t n_block, size_t m_block, size_t k_block, const double *packed_a, const double *packed_b
) {
    for (size_t n = 0; n < n_block; ++n) {
        for (size_t m = 0; m < m_block; ++m) {
            double acc = c_tile[n * DGEMM_NEON_TILE_M + m];
            const double *pa = packed_a + n;
            const double *pb = packed_b + m;
            for (size_t k = 0; k < k_block; ++k) {
                acc += pa[k * DGEMM_NEON_TILE_N] * pb[k * DGEMM_NEON_TILE_M];
            }
            c_tile[n * DGEMM_NEON_TILE_M + m] = acc;
        }
    }
}

static void dgemm_store_tile(
    double *out, size_t ldo, const double *c_tile, size_t n_block, size_t m_block, size_t row_start, size_t col_start,
    double alpha, double beta
) {
    for (size_t r = 0; r < n_block; ++r) {
        double *dst = out + (row_start + r) * ldo + col_start;
        const double *src = c_tile + r * DGEMM_NEON_TILE_M;
        for (size_t c = 0; c < m_block; ++c) {
            if (beta == 0.0) {
                dst[c] = alpha * src[c];
            } else {
                dst[c] = alpha * src[c] + beta * dst[c];
            }
        }
    }
}

static void dgemm_init_tile(
    const double *out, size_t ldo, size_t n_block, size_t m_block, size_t row_start, size_t col_start, double beta,
    double *dst
) {
    for (size_t r = 0; r < n_block; ++r) {
        const double *src = out + (row_start + r) * ldo + col_start;
        double *row_dst = dst + r * DGEMM_NEON_TILE_M;
        size_t c = 0;
        for (; c < m_block; ++c) {
            row_dst[c] = beta * src[c];
        }
        for (; c < DGEMM_NEON_TILE_M; ++c) {
            row_dst[c] = 0.0;
        }
    }
    for (size_t r = n_block; r < DGEMM_NEON_TILE_N; ++r) {
        double *row_dst = dst + r * DGEMM_NEON_TILE_M;
        for (size_t c = 0; c < DGEMM_NEON_TILE_M; ++c) {
            row_dst[c] = 0.0;
        }
    }
}

static void dgemm_fallback_nt(
    const double *input, const double *weight, double *out, size_t N, size_t M, size_t K, size_t row_start,
    size_t row_end, double alpha, double beta
) {
    for (size_t n = row_start; n < row_end; ++n) {
        const double *a_row = input + n * K;
        double *c_row = out + n * M;
        for (size_t m = 0; m < M; ++m) {
            const double *w_row = weight + m * K;
            double acc = 0.0;
            for (size_t k = 0; k < K; ++k) {
                acc += a_row[k] * w_row[k];
            }
            double value = alpha * acc;
            if (beta != 0.0) {
                value += beta * c_row[m];
            }
            c_row[m] = value;
        }
    }
}

static void dgemm_fallback_nn(
    const double *input, const double *weight, double *out, size_t N, size_t M, size_t K, size_t row_start,
    size_t row_end, double alpha, double beta
) {
    for (size_t n = row_start; n < row_end; ++n) {
        const double *a_row = input + n * K;
        double *c_row = out + n * M;
        for (size_t m = 0; m < M; ++m) {
            double acc = 0.0;
            for (size_t k = 0; k < K; ++k) {
                const double *w_row = weight + k * M;
                acc += a_row[k] * w_row[m];
            }
            double value = alpha * acc;
            if (beta != 0.0) {
                value += beta * c_row[m];
            }
            c_row[m] = value;
        }
    }
}

typedef struct {
    const double *input;
    const double *weight;
    double *out;
    size_t N;
    size_t M;
    size_t K;
    size_t row_start;
    size_t row_end;
    bool layout_nt;
    double alpha;
    double beta;
} dgemm_worker_args_t;

static void dgemm_block_rows(const dgemm_worker_args_t *args) {
    const double *input = args->input;
    const double *weight = args->weight;
    double *out = args->out;
    const size_t N = args->N;
    const size_t M = args->M;
    const size_t K = args->K;
    const size_t row_start = args->row_start;
    const size_t row_end = args->row_end;
    const dgemm_neon_params_t *params = dgemm_neon_get_params();

    const size_t pack_a_elems = params->block_k * DGEMM_NEON_TILE_N;
    const size_t pack_b_elems = params->block_k * DGEMM_NEON_TILE_M;
    double *packed_a = (double *)marmot_aligned_alloc(64, pack_a_elems * sizeof(double));
    double *packed_b0 = (double *)marmot_aligned_alloc(64, pack_b_elems * sizeof(double));
    double *packed_b1 = nullptr;
#if MARMOT_NEON_F32_DOUBLE_BUFFER_PACK_DEFAULT
    if (params->double_buffer_pack) {
        packed_b1 = (double *)marmot_aligned_alloc(64, pack_b_elems * sizeof(double));
    }
#endif
    const size_t block_n_cap = marmot_neon_min_size(params->block_n, row_end - row_start);
    const size_t n_tiles_cap = (block_n_cap + DGEMM_NEON_TILE_N - 1) / DGEMM_NEON_TILE_N;
    const size_t c_tile_elems = DGEMM_NEON_TILE_N * DGEMM_NEON_TILE_M;
    double *c_panel = (double *)marmot_aligned_alloc(64, n_tiles_cap * c_tile_elems * sizeof(double));
    if (packed_a == nullptr || packed_b0 == nullptr || (params->double_buffer_pack && packed_b1 == nullptr) ||
        c_panel == nullptr) {
        free(packed_a);
        free(packed_b0);
        free(packed_b1);
        free(c_panel);
        if (args->layout_nt) {
            dgemm_fallback_nt(input, weight, out, N, M, K, row_start, row_end, args->alpha, args->beta);
        } else {
            dgemm_fallback_nn(input, weight, out, N, M, K, row_start, row_end, args->alpha, args->beta);
        }
        return;
    }

    double *packed_b_buffers[2] = {packed_b0, packed_b1 != nullptr ? packed_b1 : packed_b0};
    const double alpha = args->alpha;
    const double beta = args->beta;
    const bool beta_zero = beta == 0.0;

    for (size_t m_outer = 0; m_outer < M; m_outer += params->block_m) {
        const size_t m_outer_end = marmot_neon_min_size(M, m_outer + params->block_m);
        for (size_t n_outer = row_start; n_outer < row_end; n_outer += params->block_n) {
            const size_t n_outer_end = marmot_neon_min_size(row_end, n_outer + params->block_n);
            const size_t n_tiles = (n_outer_end - n_outer + DGEMM_NEON_TILE_N - 1) / DGEMM_NEON_TILE_N;
            for (size_t m0 = m_outer; m0 < m_outer_end; m0 += DGEMM_NEON_TILE_M) {
                const size_t m_block = marmot_neon_min_size(DGEMM_NEON_TILE_M, m_outer_end - m0);
                if (beta_zero) {
                    memset(c_panel, 0, n_tiles * c_tile_elems * sizeof(double));
                } else {
                    size_t init_idx = 0;
                    for (size_t n0 = n_outer; n0 < n_outer_end; n0 += DGEMM_NEON_TILE_N, ++init_idx) {
                        const size_t n_block = marmot_neon_min_size(DGEMM_NEON_TILE_N, n_outer_end - n0);
                        double *c_tile = c_panel + init_idx * c_tile_elems;
                        dgemm_init_tile(out, M, n_block, m_block, n0, m0, beta, c_tile);
                    }
                }
                size_t b_buf_idx = 0;
                for (size_t k0 = 0; k0 < K; k0 += params->block_k) {
                    const size_t k_block = marmot_neon_min_size(params->block_k, K - k0);
                    double *packed_b = packed_b_buffers[b_buf_idx];
                    if (params->double_buffer_pack) {
                        b_buf_idx ^= 1;
                    }
                    if (args->layout_nt) {
                        dgemm_pack_b_tile_nt(weight, K, m0, m_block, k0, k_block, packed_b);
                    } else {
                        dgemm_pack_b_tile_nn(weight, M, m0, m_block, k0, k_block, packed_b);
                    }
                    size_t tile_idx = 0;
                    for (size_t n0 = n_outer; n0 < n_outer_end; n0 += DGEMM_NEON_TILE_N, ++tile_idx) {
                        const size_t n_block = marmot_neon_min_size(DGEMM_NEON_TILE_N, n_outer_end - n0);
                        double *c_tile = c_panel + tile_idx * c_tile_elems;
                        dgemm_pack_a_tile(input, K, n0, n_block, k0, k_block, packed_a);
                        if (n_block == DGEMM_NEON_TILE_N && m_block == DGEMM_NEON_TILE_M) {
                            dgemm_kernel_8x4(c_tile, k_block, packed_a, packed_b);
                        } else {
                            dgemm_kernel_edge(c_tile, n_block, m_block, k_block, packed_a, packed_b);
                        }
                    }
                }
                size_t tile_idx = 0;
                for (size_t n0 = n_outer; n0 < n_outer_end; n0 += DGEMM_NEON_TILE_N, ++tile_idx) {
                    const size_t n_block = marmot_neon_min_size(DGEMM_NEON_TILE_N, n_outer_end - n0);
                    const double *c_tile = c_panel + tile_idx * c_tile_elems;
                    dgemm_store_tile(out, M, c_tile, n_block, m_block, n0, m0, alpha, 0.0);
                }
            }
        }
    }

    free(packed_a);
    free(packed_b0);
    free(packed_b1);
    free(c_panel);
}

typedef struct {
    const double *input;
    const double *weight;
    double *out;
    size_t N;
    size_t M;
    size_t K;
    bool layout_nt;
    double alpha;
    double beta;
} dgemm_dispatch_ctx_t;

static void dgemm_dispatch_range(void *ctx, size_t row_start, size_t row_end) {
    const dgemm_dispatch_ctx_t *c = (const dgemm_dispatch_ctx_t *)ctx;
    dgemm_worker_args_t args = {
        .input = c->input,
        .weight = c->weight,
        .out = c->out,
        .N = c->N,
        .M = c->M,
        .K = c->K,
        .row_start = row_start,
        .row_end = row_end,
        .layout_nt = c->layout_nt,
        .alpha = c->alpha,
        .beta = c->beta,
    };
    dgemm_block_rows(&args);
}

static void dgemm_run(
    const void *device_ctx, const double *input, const double *weight, double *out, size_t N, size_t K, size_t M,
    bool layout_nt, double alpha, double beta
) {
    (void)device_ctx;

    if (N < DGEMM_NEON_MIN_DIM || M < DGEMM_NEON_MIN_DIM || K < DGEMM_NEON_MIN_DIM) {
        if (layout_nt) {
            dgemm_fallback_nt(input, weight, out, N, M, K, 0, N, alpha, beta);
        } else {
            dgemm_fallback_nn(input, weight, out, N, M, K, 0, N, alpha, beta);
        }
        return;
    }

    (void)dgemm_neon_get_params();

    dgemm_dispatch_ctx_t dctx = {
        .input = input,
        .weight = weight,
        .out = out,
        .N = N,
        .M = M,
        .K = K,
        .layout_nt = layout_nt,
        .alpha = alpha,
        .beta = beta,
    };
    marmot_dispatch_parallel_for_range(MARMOT_DISPATCH_PRIORITY_HIGH, N, 64, &dctx, dgemm_dispatch_range);
}

marmot_error_t cpu_matmul_f32_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    if (input == nullptr || weight == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (N <= 16 && M <= 16 && K <= 16) {
        return cpu_matmul_f32_scalar(device_ctx, input, weight, N, K, M, out);
    }
    float *out_data = (float *)out->data;
    memset(out_data, 0, N * M * sizeof(float));
    marmot_neon_f32_run(
        device_ctx, (const float *)input->data, (const float *)weight->data, out_data, N, K, M, true, 1.0f, 0.0f
    );
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_matmul_f32_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    if (input == nullptr || weight == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (N <= 16 && M <= 16 && K <= 16) {
        return cpu_matmul_f32_scalar_nn(device_ctx, input, weight, N, K, M, out);
    }
    float *out_data = (float *)out->data;
    memset(out_data, 0, N * M * sizeof(float));
    marmot_neon_f32_run(
        device_ctx, (const float *)input->data, (const float *)weight->data, out_data, N, K, M, false, 1.0f, 0.0f
    );
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_matmul_bf16_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    if (input == nullptr || weight == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
#if MARMOT_HAS_BF16_NATIVE
    if (marmot_cpu_has_arm_bf16() && marmot_neon_f32_use_small_kernel(N, M, K)) {
        marmot_neon_bf16_small_direct_nt(
            (const marmot_bfloat16_t *)input->data, (const marmot_bfloat16_t *)weight->data,
            (marmot_bfloat16_t *)out->data, N, M, K, 1.0f, 0.0f
        );
        return MARMOT_SUCCESS;
    }
#endif
    marmot_neon_f32_run_generic(
        device_ctx, input->data, weight->data, out->data, N, K, M, true, 1.0f, 0.0f, &sgemm_ops_bf16
    );
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_matmul_bf16_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    if (input == nullptr || weight == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_neon_f32_run_generic(
        device_ctx, input->data, weight->data, out->data, N, K, M, false, 1.0f, 0.0f, &sgemm_ops_bf16
    );
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_matmul_f16_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    if (input == nullptr || weight == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_neon_f32_run_generic(
        device_ctx, input->data, weight->data, out->data, N, K, M, true, 1.0f, 0.0f, &sgemm_ops_f16
    );
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_matmul_f16_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    if (input == nullptr || weight == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_neon_f32_run_generic(
        device_ctx, input->data, weight->data, out->data, N, K, M, false, 1.0f, 0.0f, &sgemm_ops_f16
    );
    return MARMOT_SUCCESS;
}

#if MARMOT_ENABLE_FP8

static marmot_error_t cpu_matmul_fp8_neon_require_f32_output(
    const marmot_tensor_t *input, const marmot_tensor_t *weight, const marmot_tensor_t *out
) {
    if (input == nullptr || weight == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "FP8 NEON matmul received null tensor");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (weight->dtype != input->dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 NEON matmul requires matching input and weight dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (out->dtype != MARMOT_DTYPE_FLOAT32) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 NEON matmul requires FLOAT32 output");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_matmul_fp8_e4m3_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    marmot_error_t status = cpu_matmul_fp8_neon_require_f32_output(input, weight, out);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    marmot_neon_f32_run_generic(
        device_ctx, input->data, weight->data, out->data, N, K, M, true, 1.0f, 0.0f, &sgemm_ops_fp8_e4m3
    );
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_matmul_fp8_e4m3_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    marmot_error_t status = cpu_matmul_fp8_neon_require_f32_output(input, weight, out);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    marmot_neon_f32_run_generic(
        device_ctx, input->data, weight->data, out->data, N, K, M, false, 1.0f, 0.0f, &sgemm_ops_fp8_e4m3
    );
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_matmul_fp8_e5m2_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    marmot_error_t status = cpu_matmul_fp8_neon_require_f32_output(input, weight, out);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    marmot_neon_f32_run_generic(
        device_ctx, input->data, weight->data, out->data, N, K, M, true, 1.0f, 0.0f, &sgemm_ops_fp8_e5m2
    );
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_matmul_fp8_e5m2_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    marmot_error_t status = cpu_matmul_fp8_neon_require_f32_output(input, weight, out);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    marmot_neon_f32_run_generic(
        device_ctx, input->data, weight->data, out->data, N, K, M, false, 1.0f, 0.0f, &sgemm_ops_fp8_e5m2
    );
    return MARMOT_SUCCESS;
}

#else // MARMOT_ENABLE_FP8

marmot_error_t cpu_matmul_fp8_e4m3_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 NEON matmul not enabled");
    return MARMOT_ERROR_UNSUPPORTED_DTYPE;
}

marmot_error_t cpu_matmul_fp8_e4m3_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 NEON matmul not enabled");
    return MARMOT_ERROR_UNSUPPORTED_DTYPE;
}

marmot_error_t cpu_matmul_fp8_e5m2_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 NEON matmul not enabled");
    return MARMOT_ERROR_UNSUPPORTED_DTYPE;
}

marmot_error_t cpu_matmul_fp8_e5m2_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 NEON matmul not enabled");
    return MARMOT_ERROR_UNSUPPORTED_DTYPE;
}

#endif // MARMOT_ENABLE_FP8

marmot_error_t cpu_matmul_f64_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    if (input == nullptr || weight == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    double *out_data = (double *)out->data;
    memset(out_data, 0, N * M * sizeof(double));
    dgemm_run(device_ctx, (const double *)input->data, (const double *)weight->data, out_data, N, K, M, true, 1.0, 0.0);
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_matmul_f64_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    if (input == nullptr || weight == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    double *out_data = (double *)out->data;
    memset(out_data, 0, N * M * sizeof(double));
    dgemm_run(
        device_ctx, (const double *)input->data, (const double *)weight->data, out_data, N, K, M, false, 1.0, 0.0
    );
    return MARMOT_SUCCESS;
}

#else

// Stub implementations when NEON is disabled - fall back to scalar
marmot_error_t cpu_matmul_f32_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

marmot_error_t cpu_matmul_f32_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

marmot_error_t cpu_matmul_bf16_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

marmot_error_t cpu_matmul_bf16_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

marmot_error_t cpu_matmul_f16_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

marmot_error_t cpu_matmul_f16_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

marmot_error_t cpu_matmul_f64_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

marmot_error_t cpu_matmul_f64_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

// FP8 stub implementations when NEON is disabled
marmot_error_t cpu_matmul_fp8_e4m3_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

marmot_error_t cpu_matmul_fp8_e4m3_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

marmot_error_t cpu_matmul_fp8_e5m2_neon_blocked_nt(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

marmot_error_t cpu_matmul_fp8_e5m2_neon_blocked_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

#endif
