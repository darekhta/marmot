#include "marmot/dispatch.h"

#include <stdbool.h>
#include <stdlib.h>

#include <string.h>

#include "cpu_backend_internal.h"
#include "ops/matmul/matmul_epilogue.h"
#include "ops/matmul/matmul_table.h"

typedef void (*scalar_pack_a_fn)(
    const void *input, size_t lda, size_t row_start, size_t rows, size_t k_start, size_t k_block, size_t tile_n,
    size_t prefetch_k, void *packed
);
typedef void (*scalar_pack_b_fn)(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, size_t tile_m,
    size_t prefetch_k, void *packed
);
typedef void (*scalar_micro_kernel_fn)(
    const void *packed_a, const void *packed_b, void *c_tile, size_t k_block, size_t rows, size_t cols
);
typedef void (*scalar_init_tile_fn)(
    const void *out, size_t ldo, size_t row_start, size_t col_start, size_t n_block, size_t m_block, double beta,
    void *dst
);
typedef void (*scalar_store_tile_fn)(
    void *out, size_t ldo, const void *c_tile, size_t row_start, size_t col_start, size_t n_block, size_t m_block,
    double alpha
);
typedef void (*scalar_fallback_fn)(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    double alpha, double beta
);

typedef struct {
    scalar_pack_a_fn pack_a;
    scalar_pack_b_fn pack_b_nt;
    scalar_pack_b_fn pack_b_nn;
    scalar_micro_kernel_fn micro_kernel;
    scalar_init_tile_fn init_tile;
    scalar_store_tile_fn store_tile;
    scalar_fallback_fn fallback_nt;
    scalar_fallback_fn fallback_nn;
    size_t tile_m;
    size_t tile_n;
    size_t block_m;
    size_t block_n;
    size_t block_k;
    size_t pack_a_elem_size;
    size_t pack_b_elem_size;
    size_t accum_elem_size;
    size_t prefetch_k;
} scalar_gemm_ops_t;

static inline size_t cpu_matmul_min_size(size_t a, size_t b) {
    return a < b ? a : b;
}

static void cpu_matmul_scalar_block_rows(
    const scalar_gemm_ops_t *ops, const void *input, const void *weight, void *out, size_t N, size_t M, size_t K,
    size_t row_start, size_t row_end, cpu_matmul_layout_t layout, double alpha, double beta
) {
    const size_t pack_a_elems = ops->block_k * ops->tile_n;
    const size_t pack_b_elems = ops->block_k * ops->tile_m;
    const size_t max_n_tiles = (ops->block_n + ops->tile_n - 1) / ops->tile_n;
    void *packed_a = marmot_aligned_alloc(MARMOT_CACHE_LINE_SIZE, pack_a_elems * ops->pack_a_elem_size);
    void *packed_b = marmot_aligned_alloc(MARMOT_CACHE_LINE_SIZE, pack_b_elems * ops->pack_b_elem_size);
    void *c_panel =
        marmot_aligned_alloc(MARMOT_CACHE_LINE_SIZE, max_n_tiles * ops->tile_n * ops->tile_m * ops->accum_elem_size);
    if (packed_a == nullptr || packed_b == nullptr || c_panel == nullptr) {
        free(packed_a);
        free(packed_b);
        free(c_panel);
        if (layout == CPU_MATMUL_LAYOUT_NT) {
            ops->fallback_nt(input, weight, out, N, M, K, row_start, row_end, alpha, beta);
        } else {
            ops->fallback_nn(input, weight, out, N, M, K, row_start, row_end, alpha, beta);
        }
        return;
    }

    const size_t c_tile_elems = ops->tile_n * ops->tile_m;
    for (size_t m_outer = 0; m_outer < M; m_outer += ops->block_m) {
        const size_t m_outer_end = cpu_matmul_min_size(M, m_outer + ops->block_m);
        for (size_t n_outer = row_start; n_outer < row_end; n_outer += ops->block_n) {
            const size_t n_outer_end = cpu_matmul_min_size(row_end, n_outer + ops->block_n);
            const size_t n_tiles = (n_outer_end - n_outer + ops->tile_n - 1) / ops->tile_n;
            const size_t c_panel_elems = n_tiles * c_tile_elems;

            for (size_t m0 = m_outer; m0 < m_outer_end; m0 += ops->tile_m) {
                const size_t m_tile = cpu_matmul_min_size(ops->tile_m, m_outer_end - m0);
                if (beta == 0.0) {
                    memset(c_panel, 0, c_panel_elems * ops->accum_elem_size);
                } else {
                    size_t tile_idx = 0;
                    for (size_t n0 = n_outer; n0 < n_outer_end; n0 += ops->tile_n, ++tile_idx) {
                        const size_t n_tile = cpu_matmul_min_size(ops->tile_n, n_outer_end - n0);
                        void *c_tile = (char *)c_panel + tile_idx * c_tile_elems * ops->accum_elem_size;
                        ops->init_tile(out, M, n0, m0, n_tile, m_tile, beta, c_tile);
                    }
                }

                for (size_t k0 = 0; k0 < K; k0 += ops->block_k) {
                    const size_t k_block = cpu_matmul_min_size(ops->block_k, K - k0);
                    if (layout == CPU_MATMUL_LAYOUT_NT) {
                        ops->pack_b_nt(weight, K, m0, m_tile, k0, k_block, ops->tile_m, ops->prefetch_k, packed_b);
                    } else {
                        ops->pack_b_nn(weight, M, m0, m_tile, k0, k_block, ops->tile_m, ops->prefetch_k, packed_b);
                    }

                    size_t tile_idx = 0;
                    for (size_t n0 = n_outer; n0 < n_outer_end; n0 += ops->tile_n, ++tile_idx) {
                        const size_t n_tile = cpu_matmul_min_size(ops->tile_n, n_outer_end - n0);
                        ops->pack_a(input, K, n0, n_tile, k0, k_block, ops->tile_n, ops->prefetch_k, packed_a);
                        void *c_tile = (char *)c_panel + tile_idx * c_tile_elems * ops->accum_elem_size;
                        ops->micro_kernel(packed_a, packed_b, c_tile, k_block, n_tile, m_tile);
                    }
                }

                size_t tile_idx = 0;
                for (size_t n0 = n_outer; n0 < n_outer_end; n0 += ops->tile_n, ++tile_idx) {
                    const size_t n_tile = cpu_matmul_min_size(ops->tile_n, n_outer_end - n0);
                    const void *c_tile = (const char *)c_panel + tile_idx * c_tile_elems * ops->accum_elem_size;
                    ops->store_tile(out, M, c_tile, n0, m0, n_tile, m_tile, alpha);
                }
            }
        }
    }

    free(packed_a);
    free(packed_b);
    free(c_panel);
}

typedef struct {
    const scalar_gemm_ops_t *ops;
    const void *input;
    const void *weight;
    void *out;
    size_t N;
    size_t M;
    size_t K;
    cpu_matmul_layout_t layout;
    double alpha;
    double beta;
} cpu_matmul_scalar_dispatch_ctx_t;

static void cpu_matmul_scalar_dispatch_range(void *ctx, size_t row_start, size_t row_end) {
    const cpu_matmul_scalar_dispatch_ctx_t *c = (const cpu_matmul_scalar_dispatch_ctx_t *)ctx;
    cpu_matmul_scalar_block_rows(
        c->ops, c->input, c->weight, c->out, c->N, c->M, c->K, row_start, row_end, c->layout, c->alpha, c->beta
    );
}

static void cpu_matmul_scalar_run(
    const scalar_gemm_ops_t *ops, cpu_context_t *ctx, const void *input, const void *weight, void *out, size_t N,
    size_t K, size_t M, cpu_matmul_layout_t layout, double alpha, double beta
) {
    (void)ctx;

    cpu_matmul_scalar_dispatch_ctx_t dctx = {
        .ops = ops,
        .input = input,
        .weight = weight,
        .out = out,
        .N = N,
        .M = M,
        .K = K,
        .layout = layout,
        .alpha = alpha,
        .beta = beta,
    };
    marmot_dispatch_parallel_for_range(MARMOT_DISPATCH_PRIORITY_HIGH, N, 64, &dctx, cpu_matmul_scalar_dispatch_range);
}

#define DGEMM_SCALAR_TILE_M 4
#define DGEMM_SCALAR_TILE_N 4
#define DGEMM_SCALAR_BLOCK_M 64
#define DGEMM_SCALAR_BLOCK_N 64
#define DGEMM_SCALAR_BLOCK_K 128
#define DGEMM_SCALAR_PREFETCH_K 4

static void dgemm_scalar_pack_a(
    const void *input, size_t lda, size_t row_start, size_t rows, size_t k_start, size_t k_block, size_t tile_n,
    size_t prefetch_k, void *packed
) {
    const double *a = (const double *)input;
    double *dst_base = (double *)packed;
    for (size_t k = 0; k < k_block; ++k) {
        if (k + prefetch_k < k_block) {
            for (size_t r = 0; r < rows; ++r) {
                __builtin_prefetch(a + (row_start + r) * lda + k_start + k + prefetch_k, 0, 1);
            }
        }
        double *dst = dst_base + k * tile_n;
        for (size_t r = 0; r < rows; ++r) {
            dst[r] = a[(row_start + r) * lda + k_start + k];
        }
        for (size_t r = rows; r < tile_n; ++r) {
            dst[r] = 0.0;
        }
    }
}

static void dgemm_scalar_pack_b_nt(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, size_t tile_m,
    size_t prefetch_k, void *packed
) {
    const double *b = (const double *)weight;
    double *dst_base = (double *)packed;
    for (size_t k = 0; k < k_block; ++k) {
        if (k + prefetch_k < k_block) {
            __builtin_prefetch(b + col_start * ldw + k_start + k + prefetch_k, 0, 1);
        }
        double *dst = dst_base + k * tile_m;
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = b[(col_start + c) * ldw + k_start + k];
        }
        for (size_t c = cols; c < tile_m; ++c) {
            dst[c] = 0.0;
        }
    }
}

static void dgemm_scalar_pack_b_nn(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, size_t tile_m,
    size_t prefetch_k, void *packed
) {
    (void)prefetch_k;
    const double *b = (const double *)weight;
    double *dst_base = (double *)packed;
    for (size_t k = 0; k < k_block; ++k) {
        const double *src = b + (k_start + k) * ldw + col_start;
        double *dst = dst_base + k * tile_m;
        memcpy(dst, src, cols * sizeof(double));
        for (size_t c = cols; c < tile_m; ++c) {
            dst[c] = 0.0;
        }
    }
}

static inline void dgemm_scalar_micro_kernel(
    const void *packed_a, const void *packed_b, void *c_tile, size_t k_block, size_t rows, size_t cols
) {
    const double *a = (const double *)packed_a;
    const double *b = (const double *)packed_b;
    double *c = (double *)c_tile;
    double acc[DGEMM_SCALAR_TILE_N][DGEMM_SCALAR_TILE_M] = {{0.0}};
    const size_t a_stride = DGEMM_SCALAR_TILE_N;
    const size_t b_stride = DGEMM_SCALAR_TILE_M;

    size_t k = 0;
    for (; k + 3 < k_block; k += 4) {
        for (size_t kk = 0; kk < 4; ++kk) {
            const double *a_k = a + (k + kk) * a_stride;
            const double *b_k = b + (k + kk) * b_stride;
            if (kk == 0 && k + DGEMM_SCALAR_PREFETCH_K < k_block) {
                __builtin_prefetch(a + (k + DGEMM_SCALAR_PREFETCH_K) * a_stride, 0, 3);
                __builtin_prefetch(b + (k + DGEMM_SCALAR_PREFETCH_K) * b_stride, 0, 3);
            }
            for (size_t i = 0; i < rows; ++i) {
                const double a_val = a_k[i];
                for (size_t j = 0; j < cols; ++j) {
                    acc[i][j] += a_val * b_k[j];
                }
            }
        }
    }
    for (; k < k_block; ++k) {
        const double *a_k = a + k * a_stride;
        const double *b_k = b + k * b_stride;
        for (size_t i = 0; i < rows; ++i) {
            const double a_val = a_k[i];
            for (size_t j = 0; j < cols; ++j) {
                acc[i][j] += a_val * b_k[j];
            }
        }
    }

    for (size_t i = 0; i < rows; ++i) {
        double *c_row = c + i * DGEMM_SCALAR_TILE_M;
        for (size_t j = 0; j < cols; ++j) {
            c_row[j] += acc[i][j];
        }
    }
}

static void dgemm_scalar_init_tile(
    const void *out, size_t ldo, size_t row_start, size_t col_start, size_t n_block, size_t m_block, double beta,
    void *dst
) {
    const double *c_src = (const double *)out + row_start * ldo + col_start;
    double *c_dst = (double *)dst;
    for (size_t i = 0; i < DGEMM_SCALAR_TILE_N; ++i) {
        double *row_dst = c_dst + i * DGEMM_SCALAR_TILE_M;
        if (i < n_block) {
            const double *row_src = c_src + i * ldo;
            for (size_t j = 0; j < DGEMM_SCALAR_TILE_M; ++j) {
                if (j < m_block) {
                    row_dst[j] = beta * row_src[j];
                } else {
                    row_dst[j] = 0.0;
                }
            }
        } else {
            for (size_t j = 0; j < DGEMM_SCALAR_TILE_M; ++j) {
                row_dst[j] = 0.0;
            }
        }
    }
}

static void dgemm_scalar_store_tile(
    void *out, size_t ldo, const void *c_tile, size_t row_start, size_t col_start, size_t n_block, size_t m_block,
    double alpha
) {
    double *c_dst = (double *)out + row_start * ldo + col_start;
    const double *c_src = (const double *)c_tile;
    for (size_t i = 0; i < n_block; ++i) {
        double *row_dst = c_dst + i * ldo;
        const double *row_src = c_src + i * DGEMM_SCALAR_TILE_M;
        for (size_t j = 0; j < m_block; ++j) {
            row_dst[j] = alpha * row_src[j];
        }
    }
}

static void dgemm_scalar_fallback_nt(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    double alpha, double beta
) {
    const double *a = (const double *)input;
    const double *b = (const double *)weight;
    double *c = (double *)out;
    for (size_t n = row_start; n < row_end && n < N; ++n) {
        const double *a_row = a + n * K;
        double *c_row = c + n * M;
        for (size_t m = 0; m < M; ++m) {
            const double *b_row = b + m * K;
            double sum = 0.0;
            for (size_t k = 0; k < K; ++k) {
                sum += a_row[k] * b_row[k];
            }
            c_row[m] = alpha * sum + (beta != 0.0 ? beta * c_row[m] : 0.0);
        }
    }
}

static void dgemm_scalar_fallback_nn(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    double alpha, double beta
) {
    const double *a = (const double *)input;
    const double *b = (const double *)weight;
    double *c = (double *)out;
    for (size_t n = row_start; n < row_end && n < N; ++n) {
        const double *a_row = a + n * K;
        double *c_row = c + n * M;
        for (size_t m = 0; m < M; ++m) {
            double sum = 0.0;
            for (size_t k = 0; k < K; ++k) {
                sum += a_row[k] * b[k * M + m];
            }
            c_row[m] = alpha * sum + (beta != 0.0 ? beta * c_row[m] : 0.0);
        }
    }
}

static const scalar_gemm_ops_t scalar_ops_f64 = {
    .pack_a = dgemm_scalar_pack_a,
    .pack_b_nt = dgemm_scalar_pack_b_nt,
    .pack_b_nn = dgemm_scalar_pack_b_nn,
    .micro_kernel = dgemm_scalar_micro_kernel,
    .init_tile = dgemm_scalar_init_tile,
    .store_tile = dgemm_scalar_store_tile,
    .fallback_nt = dgemm_scalar_fallback_nt,
    .fallback_nn = dgemm_scalar_fallback_nn,
    .tile_m = DGEMM_SCALAR_TILE_M,
    .tile_n = DGEMM_SCALAR_TILE_N,
    .block_m = DGEMM_SCALAR_BLOCK_M,
    .block_n = DGEMM_SCALAR_BLOCK_N,
    .block_k = DGEMM_SCALAR_BLOCK_K,
    .pack_a_elem_size = sizeof(double),
    .pack_b_elem_size = sizeof(double),
    .accum_elem_size = sizeof(double),
    .prefetch_k = DGEMM_SCALAR_PREFETCH_K,
};

#define SGEMM_SCALAR_TILE_M 4
#define SGEMM_SCALAR_TILE_N 4
#define SGEMM_SCALAR_BLOCK_M 64
#define SGEMM_SCALAR_BLOCK_N 64
#define SGEMM_SCALAR_BLOCK_K 256
#define SGEMM_SCALAR_PREFETCH_K 6

static void sgemm_scalar_pack_a(
    const void *input, size_t lda, size_t row_start, size_t rows, size_t k_start, size_t k_block, size_t tile_n,
    size_t prefetch_k, void *packed
) {
    const float *a = (const float *)input;
    float *dst_base = (float *)packed;
    for (size_t k = 0; k < k_block; ++k) {
        if (k + prefetch_k < k_block) {
            for (size_t r = 0; r < rows; ++r) {
                __builtin_prefetch(a + (row_start + r) * lda + k_start + k + prefetch_k, 0, 1);
            }
        }
        float *dst = dst_base + k * tile_n;
        for (size_t r = 0; r < rows; ++r) {
            dst[r] = a[(row_start + r) * lda + k_start + k];
        }
        for (size_t r = rows; r < tile_n; ++r) {
            dst[r] = 0.0f;
        }
    }
}

static void sgemm_scalar_pack_b_nt(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, size_t tile_m,
    size_t prefetch_k, void *packed
) {
    const float *b = (const float *)weight;
    float *dst_base = (float *)packed;
    for (size_t k = 0; k < k_block; ++k) {
        if (k + prefetch_k < k_block) {
            __builtin_prefetch(b + col_start * ldw + k_start + k + prefetch_k, 0, 1);
        }
        float *dst = dst_base + k * tile_m;
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = b[(col_start + c) * ldw + k_start + k];
        }
        for (size_t c = cols; c < tile_m; ++c) {
            dst[c] = 0.0f;
        }
    }
}

static void sgemm_scalar_pack_b_nn(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, size_t tile_m,
    size_t prefetch_k, void *packed
) {
    (void)prefetch_k;
    const float *b = (const float *)weight;
    float *dst_base = (float *)packed;
    for (size_t k = 0; k < k_block; ++k) {
        const float *src = b + (k_start + k) * ldw + col_start;
        float *dst = dst_base + k * tile_m;
        memcpy(dst, src, cols * sizeof(float));
        for (size_t c = cols; c < tile_m; ++c) {
            dst[c] = 0.0f;
        }
    }
}

static inline void sgemm_scalar_micro_kernel(
    const void *packed_a, const void *packed_b, void *c_tile, size_t k_block, size_t rows, size_t cols
) {
    const float *a = (const float *)packed_a;
    const float *b = (const float *)packed_b;
    float *c = (float *)c_tile;
    float acc[SGEMM_SCALAR_TILE_N][SGEMM_SCALAR_TILE_M] = {{0.0f}};
    const size_t a_stride = SGEMM_SCALAR_TILE_N;
    const size_t b_stride = SGEMM_SCALAR_TILE_M;

    size_t k = 0;
    for (; k + 3 < k_block; k += 4) {
        for (size_t kk = 0; kk < 4; ++kk) {
            const float *a_k = a + (k + kk) * a_stride;
            const float *b_k = b + (k + kk) * b_stride;
            if (kk == 0 && k + SGEMM_SCALAR_PREFETCH_K < k_block) {
                __builtin_prefetch(a + (k + SGEMM_SCALAR_PREFETCH_K) * a_stride, 0, 3);
                __builtin_prefetch(b + (k + SGEMM_SCALAR_PREFETCH_K) * b_stride, 0, 3);
            }
            for (size_t i = 0; i < rows; ++i) {
                const float a_val = a_k[i];
                for (size_t j = 0; j < cols; ++j) {
                    acc[i][j] += a_val * b_k[j];
                }
            }
        }
    }
    for (; k < k_block; ++k) {
        const float *a_k = a + k * a_stride;
        const float *b_k = b + k * b_stride;
        for (size_t i = 0; i < rows; ++i) {
            const float a_val = a_k[i];
            for (size_t j = 0; j < cols; ++j) {
                acc[i][j] += a_val * b_k[j];
            }
        }
    }

    for (size_t i = 0; i < rows; ++i) {
        float *c_row = c + i * SGEMM_SCALAR_TILE_M;
        for (size_t j = 0; j < cols; ++j) {
            c_row[j] += acc[i][j];
        }
    }
}

static void sgemm_scalar_init_tile(
    const void *out, size_t ldo, size_t row_start, size_t col_start, size_t n_block, size_t m_block, double beta,
    void *dst
) {
    const float *c_src = (const float *)out + row_start * ldo + col_start;
    float *c_dst = (float *)dst;
    for (size_t i = 0; i < SGEMM_SCALAR_TILE_N; ++i) {
        float *row_dst = c_dst + i * SGEMM_SCALAR_TILE_M;
        if (i < n_block) {
            const float *row_src = c_src + i * ldo;
            for (size_t j = 0; j < SGEMM_SCALAR_TILE_M; ++j) {
                if (j < m_block) {
                    row_dst[j] = (float)(beta * row_src[j]);
                } else {
                    row_dst[j] = 0.0f;
                }
            }
        } else {
            for (size_t j = 0; j < SGEMM_SCALAR_TILE_M; ++j) {
                row_dst[j] = 0.0f;
            }
        }
    }
}

static void sgemm_scalar_store_tile(
    void *out, size_t ldo, const void *c_tile, size_t row_start, size_t col_start, size_t n_block, size_t m_block,
    double alpha
) {
    float *c_dst = (float *)out + row_start * ldo + col_start;
    const float *c_src = (const float *)c_tile;
    for (size_t i = 0; i < n_block; ++i) {
        float *row_dst = c_dst + i * ldo;
        const float *row_src = c_src + i * SGEMM_SCALAR_TILE_M;
        for (size_t j = 0; j < m_block; ++j) {
            row_dst[j] = (float)(alpha * row_src[j]);
        }
    }
}

static void sgemm_scalar_fallback_nt(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    double alpha, double beta
) {
    const float *a = (const float *)input;
    const float *b = (const float *)weight;
    float *c = (float *)out;
    for (size_t n = row_start; n < row_end && n < N; ++n) {
        const float *a_row = a + n * K;
        float *c_row = c + n * M;
        for (size_t m = 0; m < M; ++m) {
            const float *b_row = b + m * K;
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += a_row[k] * b_row[k];
            }
            c_row[m] = (float)(alpha * sum + (beta != 0.0 ? beta * c_row[m] : 0.0));
        }
    }
}

static void sgemm_scalar_fallback_nn(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    double alpha, double beta
) {
    const float *a = (const float *)input;
    const float *b = (const float *)weight;
    float *c = (float *)out;
    for (size_t n = row_start; n < row_end && n < N; ++n) {
        const float *a_row = a + n * K;
        float *c_row = c + n * M;
        for (size_t m = 0; m < M; ++m) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += a_row[k] * b[k * M + m];
            }
            c_row[m] = (float)(alpha * sum + (beta != 0.0 ? beta * c_row[m] : 0.0));
        }
    }
}

static const scalar_gemm_ops_t scalar_ops_f32 = {
    .pack_a = sgemm_scalar_pack_a,
    .pack_b_nt = sgemm_scalar_pack_b_nt,
    .pack_b_nn = sgemm_scalar_pack_b_nn,
    .micro_kernel = sgemm_scalar_micro_kernel,
    .init_tile = sgemm_scalar_init_tile,
    .store_tile = sgemm_scalar_store_tile,
    .fallback_nt = sgemm_scalar_fallback_nt,
    .fallback_nn = sgemm_scalar_fallback_nn,
    .tile_m = SGEMM_SCALAR_TILE_M,
    .tile_n = SGEMM_SCALAR_TILE_N,
    .block_m = SGEMM_SCALAR_BLOCK_M,
    .block_n = SGEMM_SCALAR_BLOCK_N,
    .block_k = SGEMM_SCALAR_BLOCK_K,
    .pack_a_elem_size = sizeof(float),
    .pack_b_elem_size = sizeof(float),
    .accum_elem_size = sizeof(float),
    .prefetch_k = SGEMM_SCALAR_PREFETCH_K,
};

#define HGEMM_SCALAR_TILE_M 4
#define HGEMM_SCALAR_TILE_N 4
#define HGEMM_SCALAR_BLOCK_M 64
#define HGEMM_SCALAR_BLOCK_N 64
#define HGEMM_SCALAR_BLOCK_K 512
#define HGEMM_SCALAR_PREFETCH_K 6

static void hgemm_scalar_pack_a(
    const void *input, size_t lda, size_t row_start, size_t rows, size_t k_start, size_t k_block, size_t tile_n,
    size_t prefetch_k, void *packed
) {
    const marmot_float16_t *a = (const marmot_float16_t *)input;
    float *dst_base = (float *)packed;
    for (size_t k = 0; k < k_block; ++k) {
        float *dst = dst_base + k * tile_n;
        if (k + prefetch_k < k_block) {
            for (size_t r = 0; r < rows; ++r) {
                __builtin_prefetch(a + (row_start + r) * lda + k_start + k + prefetch_k, 0, 1);
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            const marmot_float16_t *a_row = a + (row_start + r) * lda;
            dst[r] = (float)marmot_float16_to_native(a_row[k_start + k]);
        }
        for (size_t r = rows; r < tile_n; ++r) {
            dst[r] = 0.0f;
        }
    }
}

static void hgemm_scalar_pack_b_nt(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, size_t tile_m,
    size_t prefetch_k, void *packed
) {
    const marmot_float16_t *b = (const marmot_float16_t *)weight;
    float *dst_base = (float *)packed;
    for (size_t k = 0; k < k_block; ++k) {
        if (k + prefetch_k < k_block) {
            __builtin_prefetch(b + col_start * ldw + k_start + k + prefetch_k, 0, 1);
        }
        float *dst = dst_base + k * tile_m;
        for (size_t c = 0; c < cols; ++c) {
            const marmot_float16_t *b_row = b + (col_start + c) * ldw;
            dst[c] = (float)marmot_float16_to_native(b_row[k_start + k]);
        }
        for (size_t c = cols; c < tile_m; ++c) {
            dst[c] = 0.0f;
        }
    }
}

static void hgemm_scalar_pack_b_nn(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, size_t tile_m,
    size_t prefetch_k, void *packed
) {
    (void)prefetch_k;
    const marmot_float16_t *b = (const marmot_float16_t *)weight;
    float *dst_base = (float *)packed;
    for (size_t k = 0; k < k_block; ++k) {
        const marmot_float16_t *src_row = b + (k_start + k) * ldw + col_start;
        float *dst = dst_base + k * tile_m;
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = (float)marmot_float16_to_native(src_row[c]);
        }
        for (size_t c = cols; c < tile_m; ++c) {
            dst[c] = 0.0f;
        }
    }
}

static inline void hgemm_scalar_micro_kernel(
    const void *packed_a, const void *packed_b, void *c_tile, size_t k_block, size_t rows, size_t cols
) {
    const float *a = (const float *)packed_a;
    const float *b = (const float *)packed_b;
    float *c = (float *)c_tile;
    float acc[HGEMM_SCALAR_TILE_N][HGEMM_SCALAR_TILE_M] = {{0.0f}};
    const size_t a_stride = HGEMM_SCALAR_TILE_N;
    const size_t b_stride = HGEMM_SCALAR_TILE_M;

    for (size_t k = 0; k < k_block; ++k) {
        const float *a_k = a + k * a_stride;
        const float *b_k = b + k * b_stride;
        for (size_t i = 0; i < rows; ++i) {
            const float a_val = a_k[i];
            for (size_t j = 0; j < cols; ++j) {
                acc[i][j] += a_val * b_k[j];
            }
        }
    }

    for (size_t i = 0; i < rows; ++i) {
        float *c_row = c + i * HGEMM_SCALAR_TILE_M;
        for (size_t j = 0; j < cols; ++j) {
            c_row[j] += acc[i][j];
        }
    }
}

static void hgemm_scalar_init_tile(
    const void *out, size_t ldo, size_t row_start, size_t col_start, size_t n_block, size_t m_block, double beta,
    void *dst
) {
    const marmot_float16_t *c_src = (const marmot_float16_t *)out + row_start * ldo + col_start;
    float *c_dst = (float *)dst;
    for (size_t i = 0; i < HGEMM_SCALAR_TILE_N; ++i) {
        float *row_dst = c_dst + i * HGEMM_SCALAR_TILE_M;
        if (i < n_block) {
            const marmot_float16_t *row_src = c_src + i * ldo;
            for (size_t j = 0; j < HGEMM_SCALAR_TILE_M; ++j) {
                if (j < m_block) {
                    row_dst[j] = (float)(beta * (float)marmot_float16_to_native(row_src[j]));
                } else {
                    row_dst[j] = 0.0f;
                }
            }
        } else {
            for (size_t j = 0; j < HGEMM_SCALAR_TILE_M; ++j) {
                row_dst[j] = 0.0f;
            }
        }
    }
}

static void hgemm_scalar_store_tile(
    void *out, size_t ldo, const void *c_tile, size_t row_start, size_t col_start, size_t n_block, size_t m_block,
    double alpha
) {
    marmot_float16_t *c_dst = (marmot_float16_t *)out + row_start * ldo + col_start;
    const float *c_src = (const float *)c_tile;
    for (size_t i = 0; i < n_block; ++i) {
        marmot_float16_t *row_dst = c_dst + i * ldo;
        const float *row_src = c_src + i * HGEMM_SCALAR_TILE_M;
        for (size_t j = 0; j < m_block; ++j) {
            const float value = (float)(alpha * row_src[j]);
            row_dst[j] = marmot_native_to_float16((_Float16)value);
        }
    }
}

static void hgemm_scalar_fallback_nt(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    double alpha, double beta
) {
    const marmot_float16_t *a = (const marmot_float16_t *)input;
    const marmot_float16_t *b = (const marmot_float16_t *)weight;
    marmot_float16_t *c = (marmot_float16_t *)out;
    for (size_t n = row_start; n < row_end && n < N; ++n) {
        const marmot_float16_t *a_row = a + n * K;
        marmot_float16_t *c_row = c + n * M;
        for (size_t m = 0; m < M; ++m) {
            const marmot_float16_t *b_row = b + m * K;
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += (float)marmot_float16_to_native(a_row[k]) * (float)marmot_float16_to_native(b_row[k]);
            }
            const float prior = beta != 0.0 ? (float)marmot_float16_to_native(c_row[m]) : 0.0f;
            c_row[m] = marmot_native_to_float16((_Float16)(alpha * sum + beta * prior));
        }
    }
}

static void hgemm_scalar_fallback_nn(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    double alpha, double beta
) {
    const marmot_float16_t *a = (const marmot_float16_t *)input;
    const marmot_float16_t *b = (const marmot_float16_t *)weight;
    marmot_float16_t *c = (marmot_float16_t *)out;
    for (size_t n = row_start; n < row_end && n < N; ++n) {
        const marmot_float16_t *a_row = a + n * K;
        marmot_float16_t *c_row = c + n * M;
        for (size_t m = 0; m < M; ++m) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += (float)marmot_float16_to_native(a_row[k]) * (float)marmot_float16_to_native(b[k * M + m]);
            }
            const float prior = beta != 0.0 ? (float)marmot_float16_to_native(c_row[m]) : 0.0f;
            c_row[m] = marmot_native_to_float16((_Float16)(alpha * sum + beta * prior));
        }
    }
}

static const scalar_gemm_ops_t scalar_ops_f16 = {
    .pack_a = hgemm_scalar_pack_a,
    .pack_b_nt = hgemm_scalar_pack_b_nt,
    .pack_b_nn = hgemm_scalar_pack_b_nn,
    .micro_kernel = hgemm_scalar_micro_kernel,
    .init_tile = hgemm_scalar_init_tile,
    .store_tile = hgemm_scalar_store_tile,
    .fallback_nt = hgemm_scalar_fallback_nt,
    .fallback_nn = hgemm_scalar_fallback_nn,
    .tile_m = HGEMM_SCALAR_TILE_M,
    .tile_n = HGEMM_SCALAR_TILE_N,
    .block_m = HGEMM_SCALAR_BLOCK_M,
    .block_n = HGEMM_SCALAR_BLOCK_N,
    .block_k = HGEMM_SCALAR_BLOCK_K,
    .pack_a_elem_size = sizeof(float),
    .pack_b_elem_size = sizeof(float),
    .accum_elem_size = sizeof(float),
    .prefetch_k = HGEMM_SCALAR_PREFETCH_K,
};

#define BF16GEMM_SCALAR_TILE_M 4
#define BF16GEMM_SCALAR_TILE_N 4
#define BF16GEMM_SCALAR_BLOCK_M 64
#define BF16GEMM_SCALAR_BLOCK_N 64
#define BF16GEMM_SCALAR_BLOCK_K 512
#define BF16GEMM_SCALAR_PREFETCH_K 6

static void bf16gemm_scalar_pack_a(
    const void *input, size_t lda, size_t row_start, size_t rows, size_t k_start, size_t k_block, size_t tile_n,
    size_t prefetch_k, void *packed
) {
    const marmot_bfloat16_t *a = (const marmot_bfloat16_t *)input;
    float *dst_base = (float *)packed;
    for (size_t k = 0; k < k_block; ++k) {
        float *dst = dst_base + k * tile_n;
        if (k + prefetch_k < k_block) {
            for (size_t r = 0; r < rows; ++r) {
                __builtin_prefetch(a + (row_start + r) * lda + k_start + k + prefetch_k, 0, 1);
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            const marmot_bfloat16_t *a_row = a + (row_start + r) * lda;
            dst[r] = marmot_bf16_to_f32_ref(a_row[k_start + k]);
        }
        for (size_t r = rows; r < tile_n; ++r) {
            dst[r] = 0.0f;
        }
    }
}

static void bf16gemm_scalar_pack_b_nt(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, size_t tile_m,
    size_t prefetch_k, void *packed
) {
    const marmot_bfloat16_t *b = (const marmot_bfloat16_t *)weight;
    float *dst_base = (float *)packed;
    for (size_t k = 0; k < k_block; ++k) {
        if (k + prefetch_k < k_block) {
            __builtin_prefetch(b + col_start * ldw + k_start + k + prefetch_k, 0, 1);
        }
        float *dst = dst_base + k * tile_m;
        for (size_t c = 0; c < cols; ++c) {
            const marmot_bfloat16_t *b_row = b + (col_start + c) * ldw;
            dst[c] = marmot_bf16_to_f32_ref(b_row[k_start + k]);
        }
        for (size_t c = cols; c < tile_m; ++c) {
            dst[c] = 0.0f;
        }
    }
}

static void bf16gemm_scalar_pack_b_nn(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, size_t tile_m,
    size_t prefetch_k, void *packed
) {
    (void)prefetch_k;
    const marmot_bfloat16_t *b = (const marmot_bfloat16_t *)weight;
    float *dst_base = (float *)packed;
    for (size_t k = 0; k < k_block; ++k) {
        const marmot_bfloat16_t *src_row = b + (k_start + k) * ldw + col_start;
        float *dst = dst_base + k * tile_m;
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = marmot_bf16_to_f32_ref(src_row[c]);
        }
        for (size_t c = cols; c < tile_m; ++c) {
            dst[c] = 0.0f;
        }
    }
}

static inline void bf16gemm_scalar_micro_kernel(
    const void *packed_a, const void *packed_b, void *c_tile, size_t k_block, size_t rows, size_t cols
) {
    const float *a = (const float *)packed_a;
    const float *b = (const float *)packed_b;
    float *c = (float *)c_tile;
    float acc[BF16GEMM_SCALAR_TILE_N][BF16GEMM_SCALAR_TILE_M] = {{0.0f}};
    const size_t a_stride = BF16GEMM_SCALAR_TILE_N;
    const size_t b_stride = BF16GEMM_SCALAR_TILE_M;

    for (size_t k = 0; k < k_block; ++k) {
        const float *a_k = a + k * a_stride;
        const float *b_k = b + k * b_stride;
        for (size_t i = 0; i < rows; ++i) {
            const float a_val = a_k[i];
            for (size_t j = 0; j < cols; ++j) {
                acc[i][j] += a_val * b_k[j];
            }
        }
    }

    for (size_t i = 0; i < rows; ++i) {
        float *c_row = c + i * BF16GEMM_SCALAR_TILE_M;
        for (size_t j = 0; j < cols; ++j) {
            c_row[j] += acc[i][j];
        }
    }
}

static void bf16gemm_scalar_init_tile(
    const void *out, size_t ldo, size_t row_start, size_t col_start, size_t n_block, size_t m_block, double beta,
    void *dst
) {
    const marmot_bfloat16_t *c_src = (const marmot_bfloat16_t *)out + row_start * ldo + col_start;
    float *c_dst = (float *)dst;
    for (size_t i = 0; i < BF16GEMM_SCALAR_TILE_N; ++i) {
        float *row_dst = c_dst + i * BF16GEMM_SCALAR_TILE_M;
        if (i < n_block) {
            const marmot_bfloat16_t *row_src = c_src + i * ldo;
            for (size_t j = 0; j < BF16GEMM_SCALAR_TILE_M; ++j) {
                if (j < m_block) {
                    row_dst[j] = (float)(beta * marmot_bf16_to_f32_ref(row_src[j]));
                } else {
                    row_dst[j] = 0.0f;
                }
            }
        } else {
            for (size_t j = 0; j < BF16GEMM_SCALAR_TILE_M; ++j) {
                row_dst[j] = 0.0f;
            }
        }
    }
}

static void bf16gemm_scalar_store_tile(
    void *out, size_t ldo, const void *c_tile, size_t row_start, size_t col_start, size_t n_block, size_t m_block,
    double alpha
) {
    marmot_bfloat16_t *c_dst = (marmot_bfloat16_t *)out + row_start * ldo + col_start;
    const float *c_src = (const float *)c_tile;
    for (size_t i = 0; i < n_block; ++i) {
        marmot_bfloat16_t *row_dst = c_dst + i * ldo;
        const float *row_src = c_src + i * BF16GEMM_SCALAR_TILE_M;
        for (size_t j = 0; j < m_block; ++j) {
            row_dst[j] = marmot_f32_to_bf16_ref((float)(alpha * row_src[j]));
        }
    }
}

static void bf16gemm_scalar_fallback_nt(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    double alpha, double beta
) {
    const marmot_bfloat16_t *a = (const marmot_bfloat16_t *)input;
    const marmot_bfloat16_t *b = (const marmot_bfloat16_t *)weight;
    marmot_bfloat16_t *c = (marmot_bfloat16_t *)out;
    for (size_t n = row_start; n < row_end && n < N; ++n) {
        const marmot_bfloat16_t *a_row = a + n * K;
        marmot_bfloat16_t *c_row = c + n * M;
        for (size_t m = 0; m < M; ++m) {
            const marmot_bfloat16_t *b_row = b + m * K;
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += marmot_bf16_to_f32_ref(a_row[k]) * marmot_bf16_to_f32_ref(b_row[k]);
            }
            const float prior = beta != 0.0 ? marmot_bf16_to_f32_ref(c_row[m]) : 0.0f;
            c_row[m] = marmot_f32_to_bf16_ref((float)(alpha * sum + beta * prior));
        }
    }
}

static void bf16gemm_scalar_fallback_nn(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    double alpha, double beta
) {
    const marmot_bfloat16_t *a = (const marmot_bfloat16_t *)input;
    const marmot_bfloat16_t *b = (const marmot_bfloat16_t *)weight;
    marmot_bfloat16_t *c = (marmot_bfloat16_t *)out;
    for (size_t n = row_start; n < row_end && n < N; ++n) {
        const marmot_bfloat16_t *a_row = a + n * K;
        marmot_bfloat16_t *c_row = c + n * M;
        for (size_t m = 0; m < M; ++m) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += marmot_bf16_to_f32_ref(a_row[k]) * marmot_bf16_to_f32_ref(b[k * M + m]);
            }
            const float prior = beta != 0.0 ? marmot_bf16_to_f32_ref(c_row[m]) : 0.0f;
            c_row[m] = marmot_f32_to_bf16_ref((float)(alpha * sum + beta * prior));
        }
    }
}

static const scalar_gemm_ops_t scalar_ops_bf16 = {
    .pack_a = bf16gemm_scalar_pack_a,
    .pack_b_nt = bf16gemm_scalar_pack_b_nt,
    .pack_b_nn = bf16gemm_scalar_pack_b_nn,
    .micro_kernel = bf16gemm_scalar_micro_kernel,
    .init_tile = bf16gemm_scalar_init_tile,
    .store_tile = bf16gemm_scalar_store_tile,
    .fallback_nt = bf16gemm_scalar_fallback_nt,
    .fallback_nn = bf16gemm_scalar_fallback_nn,
    .tile_m = BF16GEMM_SCALAR_TILE_M,
    .tile_n = BF16GEMM_SCALAR_TILE_N,
    .block_m = BF16GEMM_SCALAR_BLOCK_M,
    .block_n = BF16GEMM_SCALAR_BLOCK_N,
    .block_k = BF16GEMM_SCALAR_BLOCK_K,
    .pack_a_elem_size = sizeof(float),
    .pack_b_elem_size = sizeof(float),
    .accum_elem_size = sizeof(float),
    .prefetch_k = BF16GEMM_SCALAR_PREFETCH_K,
};

#if MARMOT_ENABLE_FP8
#define FP8_SCALAR_TILE_M 4
#define FP8_SCALAR_TILE_N 4
#define FP8_SCALAR_BLOCK_M 64
#define FP8_SCALAR_BLOCK_N 64
#define FP8_SCALAR_BLOCK_K 512
#define FP8_SCALAR_PREFETCH_K 6

static void fp8_e4m3_pack_a(
    const void *input, size_t lda, size_t row_start, size_t rows, size_t k_start, size_t k_block, size_t tile_n,
    size_t prefetch_k, void *packed
) {
    const marmot_float8_e4m3_t *a = (const marmot_float8_e4m3_t *)input;
    float *dst_base = (float *)packed;
    for (size_t k = 0; k < k_block; ++k) {
        float *dst = dst_base + k * tile_n;
        if (k + prefetch_k < k_block) {
            for (size_t r = 0; r < rows; ++r) {
                __builtin_prefetch(a + (row_start + r) * lda + k_start + k + prefetch_k, 0, 1);
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            const marmot_float8_e4m3_t *a_row = a + (row_start + r) * lda;
            dst[r] = marmot_fp8_e4m3_to_native(a_row[k_start + k]);
        }
        for (size_t r = rows; r < tile_n; ++r) {
            dst[r] = 0.0f;
        }
    }
}

static void fp8_e5m2_pack_a(
    const void *input, size_t lda, size_t row_start, size_t rows, size_t k_start, size_t k_block, size_t tile_n,
    size_t prefetch_k, void *packed
) {
    const marmot_float8_e5m2_t *a = (const marmot_float8_e5m2_t *)input;
    float *dst_base = (float *)packed;
    for (size_t k = 0; k < k_block; ++k) {
        float *dst = dst_base + k * tile_n;
        if (k + prefetch_k < k_block) {
            for (size_t r = 0; r < rows; ++r) {
                __builtin_prefetch(a + (row_start + r) * lda + k_start + k + prefetch_k, 0, 1);
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            const marmot_float8_e5m2_t *a_row = a + (row_start + r) * lda;
            dst[r] = marmot_fp8_e5m2_to_native(a_row[k_start + k]);
        }
        for (size_t r = rows; r < tile_n; ++r) {
            dst[r] = 0.0f;
        }
    }
}

static void fp8_e4m3_pack_b_nt(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, size_t tile_m,
    size_t prefetch_k, void *packed
) {
    const marmot_float8_e4m3_t *b = (const marmot_float8_e4m3_t *)weight;
    float *dst_base = (float *)packed;
    for (size_t k = 0; k < k_block; ++k) {
        if (k + prefetch_k < k_block) {
            __builtin_prefetch(b + col_start * ldw + k_start + k + prefetch_k, 0, 1);
        }
        float *dst = dst_base + k * tile_m;
        for (size_t c = 0; c < cols; ++c) {
            const marmot_float8_e4m3_t *b_row = b + (col_start + c) * ldw;
            dst[c] = marmot_fp8_e4m3_to_native(b_row[k_start + k]);
        }
        for (size_t c = cols; c < tile_m; ++c) {
            dst[c] = 0.0f;
        }
    }
}

static void fp8_e5m2_pack_b_nt(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, size_t tile_m,
    size_t prefetch_k, void *packed
) {
    const marmot_float8_e5m2_t *b = (const marmot_float8_e5m2_t *)weight;
    float *dst_base = (float *)packed;
    for (size_t k = 0; k < k_block; ++k) {
        if (k + prefetch_k < k_block) {
            __builtin_prefetch(b + col_start * ldw + k_start + k + prefetch_k, 0, 1);
        }
        float *dst = dst_base + k * tile_m;
        for (size_t c = 0; c < cols; ++c) {
            const marmot_float8_e5m2_t *b_row = b + (col_start + c) * ldw;
            dst[c] = marmot_fp8_e5m2_to_native(b_row[k_start + k]);
        }
        for (size_t c = cols; c < tile_m; ++c) {
            dst[c] = 0.0f;
        }
    }
}

static void fp8_e4m3_pack_b_nn(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, size_t tile_m,
    size_t prefetch_k, void *packed
) {
    (void)prefetch_k;
    const marmot_float8_e4m3_t *b = (const marmot_float8_e4m3_t *)weight;
    float *dst_base = (float *)packed;
    for (size_t k = 0; k < k_block; ++k) {
        const marmot_float8_e4m3_t *src_row = b + (k_start + k) * ldw + col_start;
        float *dst = dst_base + k * tile_m;
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = marmot_fp8_e4m3_to_native(src_row[c]);
        }
        for (size_t c = cols; c < tile_m; ++c) {
            dst[c] = 0.0f;
        }
    }
}

static void fp8_e5m2_pack_b_nn(
    const void *weight, size_t ldw, size_t col_start, size_t cols, size_t k_start, size_t k_block, size_t tile_m,
    size_t prefetch_k, void *packed
) {
    (void)prefetch_k;
    const marmot_float8_e5m2_t *b = (const marmot_float8_e5m2_t *)weight;
    float *dst_base = (float *)packed;
    for (size_t k = 0; k < k_block; ++k) {
        const marmot_float8_e5m2_t *src_row = b + (k_start + k) * ldw + col_start;
        float *dst = dst_base + k * tile_m;
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = marmot_fp8_e5m2_to_native(src_row[c]);
        }
        for (size_t c = cols; c < tile_m; ++c) {
            dst[c] = 0.0f;
        }
    }
}

static inline void fp8_scalar_micro_kernel(
    const void *packed_a, const void *packed_b, void *c_tile, size_t k_block, size_t rows, size_t cols
) {
    const float *a = (const float *)packed_a;
    const float *b = (const float *)packed_b;
    float *c = (float *)c_tile;
    float acc[FP8_SCALAR_TILE_N][FP8_SCALAR_TILE_M] = {{0.0f}};
    const size_t a_stride = FP8_SCALAR_TILE_N;
    const size_t b_stride = FP8_SCALAR_TILE_M;

    for (size_t k = 0; k < k_block; ++k) {
        const float *a_k = a + k * a_stride;
        const float *b_k = b + k * b_stride;
        for (size_t i = 0; i < rows; ++i) {
            const float a_val = a_k[i];
            for (size_t j = 0; j < cols; ++j) {
                acc[i][j] += a_val * b_k[j];
            }
        }
    }

    for (size_t i = 0; i < rows; ++i) {
        float *c_row = c + i * FP8_SCALAR_TILE_M;
        for (size_t j = 0; j < cols; ++j) {
            c_row[j] += acc[i][j];
        }
    }
}

static void fp8_f32_init_tile(
    const void *out, size_t ldo, size_t row_start, size_t col_start, size_t n_block, size_t m_block, double beta,
    void *dst
) {
    const float *c_src = (const float *)out + row_start * ldo + col_start;
    float *c_dst = (float *)dst;
    for (size_t i = 0; i < FP8_SCALAR_TILE_N; ++i) {
        float *row_dst = c_dst + i * FP8_SCALAR_TILE_M;
        if (i < n_block) {
            const float *row_src = c_src + i * ldo;
            for (size_t j = 0; j < FP8_SCALAR_TILE_M; ++j) {
                row_dst[j] = j < m_block ? (float)(beta * row_src[j]) : 0.0f;
            }
        } else {
            for (size_t j = 0; j < FP8_SCALAR_TILE_M; ++j) {
                row_dst[j] = 0.0f;
            }
        }
    }
}

static void fp8_f32_store_tile(
    void *out, size_t ldo, const void *c_tile, size_t row_start, size_t col_start, size_t n_block, size_t m_block,
    double alpha
) {
    float *c_dst = (float *)out + row_start * ldo + col_start;
    const float *c_src = (const float *)c_tile;
    for (size_t i = 0; i < n_block; ++i) {
        float *row_dst = c_dst + i * ldo;
        const float *row_src = c_src + i * FP8_SCALAR_TILE_M;
        for (size_t j = 0; j < m_block; ++j) {
            row_dst[j] = (float)(alpha * row_src[j]);
        }
    }
}

static void fp8_e4m3_fallback_nt(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    double alpha, double beta
) {
    const marmot_float8_e4m3_t *a = (const marmot_float8_e4m3_t *)input;
    const marmot_float8_e4m3_t *b = (const marmot_float8_e4m3_t *)weight;
    float *c = (float *)out;
    for (size_t n = row_start; n < row_end && n < N; ++n) {
        const marmot_float8_e4m3_t *a_row = a + n * K;
        float *c_row = c + n * M;
        for (size_t m = 0; m < M; ++m) {
            const marmot_float8_e4m3_t *b_row = b + m * K;
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += marmot_fp8_e4m3_to_native(a_row[k]) * marmot_fp8_e4m3_to_native(b_row[k]);
            }
            const float prior = beta != 0.0 ? (float)(beta * c_row[m]) : 0.0f;
            c_row[m] = (float)(alpha * sum + prior);
        }
    }
}

static void fp8_e4m3_fallback_nn(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    double alpha, double beta
) {
    const marmot_float8_e4m3_t *a = (const marmot_float8_e4m3_t *)input;
    const marmot_float8_e4m3_t *b = (const marmot_float8_e4m3_t *)weight;
    float *c = (float *)out;
    for (size_t n = row_start; n < row_end && n < N; ++n) {
        const marmot_float8_e4m3_t *a_row = a + n * K;
        float *c_row = c + n * M;
        for (size_t m = 0; m < M; ++m) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += marmot_fp8_e4m3_to_native(a_row[k]) * marmot_fp8_e4m3_to_native(b[k * M + m]);
            }
            const float prior = beta != 0.0 ? (float)(beta * c_row[m]) : 0.0f;
            c_row[m] = (float)(alpha * sum + prior);
        }
    }
}

static void fp8_e5m2_fallback_nt(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    double alpha, double beta
) {
    const marmot_float8_e5m2_t *a = (const marmot_float8_e5m2_t *)input;
    const marmot_float8_e5m2_t *b = (const marmot_float8_e5m2_t *)weight;
    float *c = (float *)out;
    for (size_t n = row_start; n < row_end && n < N; ++n) {
        const marmot_float8_e5m2_t *a_row = a + n * K;
        float *c_row = c + n * M;
        for (size_t m = 0; m < M; ++m) {
            const marmot_float8_e5m2_t *b_row = b + m * K;
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += marmot_fp8_e5m2_to_native(a_row[k]) * marmot_fp8_e5m2_to_native(b_row[k]);
            }
            const float prior = beta != 0.0 ? (float)(beta * c_row[m]) : 0.0f;
            c_row[m] = (float)(alpha * sum + prior);
        }
    }
}

static void fp8_e5m2_fallback_nn(
    const void *input, const void *weight, void *out, size_t N, size_t M, size_t K, size_t row_start, size_t row_end,
    double alpha, double beta
) {
    const marmot_float8_e5m2_t *a = (const marmot_float8_e5m2_t *)input;
    const marmot_float8_e5m2_t *b = (const marmot_float8_e5m2_t *)weight;
    float *c = (float *)out;
    for (size_t n = row_start; n < row_end && n < N; ++n) {
        const marmot_float8_e5m2_t *a_row = a + n * K;
        float *c_row = c + n * M;
        for (size_t m = 0; m < M; ++m) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += marmot_fp8_e5m2_to_native(a_row[k]) * marmot_fp8_e5m2_to_native(b[k * M + m]);
            }
            const float prior = beta != 0.0 ? (float)(beta * c_row[m]) : 0.0f;
            c_row[m] = (float)(alpha * sum + prior);
        }
    }
}

static const scalar_gemm_ops_t scalar_ops_fp8_e4m3 = {
    .pack_a = fp8_e4m3_pack_a,
    .pack_b_nt = fp8_e4m3_pack_b_nt,
    .pack_b_nn = fp8_e4m3_pack_b_nn,
    .micro_kernel = fp8_scalar_micro_kernel,
    .init_tile = fp8_f32_init_tile,
    .store_tile = fp8_f32_store_tile,
    .fallback_nt = fp8_e4m3_fallback_nt,
    .fallback_nn = fp8_e4m3_fallback_nn,
    .tile_m = FP8_SCALAR_TILE_M,
    .tile_n = FP8_SCALAR_TILE_N,
    .block_m = FP8_SCALAR_BLOCK_M,
    .block_n = FP8_SCALAR_BLOCK_N,
    .block_k = FP8_SCALAR_BLOCK_K,
    .pack_a_elem_size = sizeof(float),
    .pack_b_elem_size = sizeof(float),
    .accum_elem_size = sizeof(float),
    .prefetch_k = FP8_SCALAR_PREFETCH_K,
};

static const scalar_gemm_ops_t scalar_ops_fp8_e5m2 = {
    .pack_a = fp8_e5m2_pack_a,
    .pack_b_nt = fp8_e5m2_pack_b_nt,
    .pack_b_nn = fp8_e5m2_pack_b_nn,
    .micro_kernel = fp8_scalar_micro_kernel,
    .init_tile = fp8_f32_init_tile,
    .store_tile = fp8_f32_store_tile,
    .fallback_nt = fp8_e5m2_fallback_nt,
    .fallback_nn = fp8_e5m2_fallback_nn,
    .tile_m = FP8_SCALAR_TILE_M,
    .tile_n = FP8_SCALAR_TILE_N,
    .block_m = FP8_SCALAR_BLOCK_M,
    .block_n = FP8_SCALAR_BLOCK_N,
    .block_k = FP8_SCALAR_BLOCK_K,
    .pack_a_elem_size = sizeof(float),
    .pack_b_elem_size = sizeof(float),
    .accum_elem_size = sizeof(float),
    .prefetch_k = FP8_SCALAR_PREFETCH_K,
};
#endif

static marmot_error_t cpu_matmul_run_scalar_entry(
    const scalar_gemm_ops_t *ops, const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    size_t N, size_t K, size_t M, marmot_tensor_t *out, cpu_matmul_layout_t layout
) {
    cpu_context_t *ctx = (cpu_context_t *)device_ctx;
    cpu_matmul_scalar_run(ops, ctx, input->data, weight->data, out->data, N, K, M, layout, 1.0, 0.0);
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_matmul_f32_scalar(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    return cpu_matmul_run_scalar_entry(&scalar_ops_f32, device_ctx, input, weight, N, K, M, out, CPU_MATMUL_LAYOUT_NT);
}

marmot_error_t cpu_matmul_f32_scalar_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    return cpu_matmul_run_scalar_entry(&scalar_ops_f32, device_ctx, input, weight, N, K, M, out, CPU_MATMUL_LAYOUT_NN);
}

marmot_error_t cpu_matmul_f64_scalar(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    return cpu_matmul_run_scalar_entry(&scalar_ops_f64, device_ctx, input, weight, N, K, M, out, CPU_MATMUL_LAYOUT_NT);
}

marmot_error_t cpu_matmul_f64_scalar_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    return cpu_matmul_run_scalar_entry(&scalar_ops_f64, device_ctx, input, weight, N, K, M, out, CPU_MATMUL_LAYOUT_NN);
}

marmot_error_t cpu_matmul_f16_scalar(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    return cpu_matmul_run_scalar_entry(&scalar_ops_f16, device_ctx, input, weight, N, K, M, out, CPU_MATMUL_LAYOUT_NT);
}

marmot_error_t cpu_matmul_f16_scalar_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    return cpu_matmul_run_scalar_entry(&scalar_ops_f16, device_ctx, input, weight, N, K, M, out, CPU_MATMUL_LAYOUT_NN);
}

marmot_error_t cpu_matmul_bf16_scalar(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    return cpu_matmul_run_scalar_entry(&scalar_ops_bf16, device_ctx, input, weight, N, K, M, out, CPU_MATMUL_LAYOUT_NT);
}

marmot_error_t cpu_matmul_bf16_scalar_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    return cpu_matmul_run_scalar_entry(&scalar_ops_bf16, device_ctx, input, weight, N, K, M, out, CPU_MATMUL_LAYOUT_NN);
}

#if MARMOT_ENABLE_FP8
static marmot_error_t cpu_matmul_fp8_require_f32_output(
    const marmot_tensor_t *input, const marmot_tensor_t *weight, const marmot_tensor_t *out
) {
    if (input == nullptr || weight == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "FP8 matmul received null tensor");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (weight->dtype != input->dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 matmul requires matching input and weight dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (out->dtype != MARMOT_DTYPE_FLOAT32) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 matmul requires FLOAT32 output");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_matmul_fp8_e4m3_scalar(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    marmot_error_t status = cpu_matmul_fp8_require_f32_output(input, weight, out);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return cpu_matmul_run_scalar_entry(
        &scalar_ops_fp8_e4m3, device_ctx, input, weight, N, K, M, out, CPU_MATMUL_LAYOUT_NT
    );
}

marmot_error_t cpu_matmul_fp8_e5m2_scalar(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    marmot_error_t status = cpu_matmul_fp8_require_f32_output(input, weight, out);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return cpu_matmul_run_scalar_entry(
        &scalar_ops_fp8_e5m2, device_ctx, input, weight, N, K, M, out, CPU_MATMUL_LAYOUT_NT
    );
}

marmot_error_t cpu_matmul_fp8_e4m3_scalar_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    marmot_error_t status = cpu_matmul_fp8_require_f32_output(input, weight, out);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return cpu_matmul_run_scalar_entry(
        &scalar_ops_fp8_e4m3, device_ctx, input, weight, N, K, M, out, CPU_MATMUL_LAYOUT_NN
    );
}

marmot_error_t cpu_matmul_fp8_e5m2_scalar_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    marmot_error_t status = cpu_matmul_fp8_require_f32_output(input, weight, out);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return cpu_matmul_run_scalar_entry(
        &scalar_ops_fp8_e5m2, device_ctx, input, weight, N, K, M, out, CPU_MATMUL_LAYOUT_NN
    );
}
#else
marmot_error_t cpu_matmul_fp8_e4m3_scalar(
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
    marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 matmul not enabled");
    return MARMOT_ERROR_UNSUPPORTED_DTYPE;
}

marmot_error_t cpu_matmul_fp8_e5m2_scalar(
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
    marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 matmul not enabled");
    return MARMOT_ERROR_UNSUPPORTED_DTYPE;
}

marmot_error_t cpu_matmul_fp8_e4m3_scalar_nn(
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
    marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 matmul not enabled");
    return MARMOT_ERROR_UNSUPPORTED_DTYPE;
}

marmot_error_t cpu_matmul_fp8_e5m2_scalar_nn(
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
    marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 matmul not enabled");
    return MARMOT_ERROR_UNSUPPORTED_DTYPE;
}
#endif

marmot_error_t cpu_matmul_direct(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    if (device_ctx == nullptr || input == nullptr || weight == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input->shape.ndim != 2 || weight->shape.ndim != 2 || out->shape.ndim != 2) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Matmul requires 2D tensors");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    const size_t N = input->shape.shape[0];
    const size_t K = input->shape.shape[1];
    const size_t M = weight->shape.shape[0];

    marmot_error_t status = MARMOT_ERROR_UNSUPPORTED_DTYPE;
    switch (input->dtype) {
    case MARMOT_DTYPE_FLOAT32:
        status = cpu_matmul_f32_scalar(device_ctx, input, weight, N, K, M, out);
        break;
    case MARMOT_DTYPE_FLOAT64:
        status = cpu_matmul_f64_scalar(device_ctx, input, weight, N, K, M, out);
        break;
    case MARMOT_DTYPE_FLOAT16:
        status = cpu_matmul_f16_scalar(device_ctx, input, weight, N, K, M, out);
        break;
    case MARMOT_DTYPE_BFLOAT16:
        status = cpu_matmul_bf16_scalar(device_ctx, input, weight, N, K, M, out);
        break;
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
        status = cpu_matmul_fp8_e4m3_scalar(device_ctx, input, weight, N, K, M, out);
        break;
    case MARMOT_DTYPE_FLOAT8_E5M2:
        status = cpu_matmul_fp8_e5m2_scalar(device_ctx, input, weight, N, K, M, out);
        break;
#endif
    default:
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unsupported dtype for CPU matmul");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return cpu_matmul_apply_epilogue(device_ctx, out, epilogue);
}
