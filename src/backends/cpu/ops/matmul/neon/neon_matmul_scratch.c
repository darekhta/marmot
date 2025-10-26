#include "neon_matmul_scratch.h"

#include <stdlib.h>

#include <string.h>

#include "cpu_backend_internal.h"

static inline size_t scratch_min(size_t a, size_t b) {
    return a < b ? a : b;
}

void marmot_neon_scratch_pool_init(marmot_neon_scratch_pool_t *pool, size_t num_workers) {
    if (pool == nullptr || num_workers == 0) {
        return;
    }

    pool->num_workers = num_workers;
    pool->f32 = (marmot_neon_f32_scratch_t *)calloc(num_workers, sizeof(marmot_neon_f32_scratch_t));
    pool->f32_gen = (marmot_neon_f32_generic_scratch_t *)calloc(num_workers, sizeof(marmot_neon_f32_generic_scratch_t));
    pool->f64 = (marmot_neon_f64_scratch_t *)calloc(num_workers, sizeof(marmot_neon_f64_scratch_t));
}

void marmot_neon_scratch_pool_destroy(marmot_neon_scratch_pool_t *pool) {
    if (pool == nullptr) {
        return;
    }

    if (pool->f32 != nullptr) {
        for (size_t i = 0; i < pool->num_workers; ++i) {
            free(pool->f32[i].packed_a);
            free(pool->f32[i].packed_b);
            free(pool->f32[i].c_panel);
        }
        free(pool->f32);
        pool->f32 = nullptr;
    }

    if (pool->f32_gen != nullptr) {
        for (size_t i = 0; i < pool->num_workers; ++i) {
            free(pool->f32_gen[i].packed_a);
            free(pool->f32_gen[i].packed_b_cache);
            free(pool->f32_gen[i].packed_b_storage);
            free(pool->f32_gen[i].c_panel);
        }
        free(pool->f32_gen);
        pool->f32_gen = nullptr;
    }

    if (pool->f64 != nullptr) {
        for (size_t i = 0; i < pool->num_workers; ++i) {
            free(pool->f64[i].packed_a);
            free(pool->f64[i].packed_b0);
            free(pool->f64[i].packed_b1);
            free(pool->f64[i].c_panel);
        }
        free(pool->f64);
        pool->f64 = nullptr;
    }

    pool->num_workers = 0;
}

bool marmot_neon_f32_scratch_ensure(
    marmot_neon_f32_scratch_t *scratch, const marmot_neon_f32_params_t *params, size_t N, size_t M, size_t K
) {
    if (scratch == nullptr || params == nullptr) {
        return false;
    }

    const size_t block_n_cap = scratch_min(params->block_n, N);
    const size_t block_m_cap = scratch_min(params->block_m, M);
    const size_t n_tiles_cap = (block_n_cap + MARMOT_NEON_F32_TILE_N - 1) / MARMOT_NEON_F32_TILE_N;
    const size_t m_tiles_cap = (block_m_cap + MARMOT_NEON_F32_TILE_M - 1) / MARMOT_NEON_F32_TILE_M;

    const size_t pack_a_elems = params->block_k * MARMOT_NEON_F32_TILE_N;
    const size_t pack_b_elems = params->block_k * params->block_m;
    const size_t c_tile_elems = MARMOT_NEON_F32_TILE_N * MARMOT_NEON_F32_TILE_M;
    const size_t c_panel_elems = n_tiles_cap * m_tiles_cap * c_tile_elems;

    if (scratch->packed_a_cap < pack_a_elems) {
        free(scratch->packed_a);
        scratch->packed_a = (float *)marmot_aligned_alloc(64, pack_a_elems * sizeof(float));
        if (scratch->packed_a == nullptr) {
            scratch->packed_a_cap = 0;
            return false;
        }
        scratch->packed_a_cap = pack_a_elems;
    }

    if (scratch->packed_b_cap < pack_b_elems) {
        free(scratch->packed_b);
        scratch->packed_b = (float *)marmot_aligned_alloc(64, pack_b_elems * sizeof(float));
        if (scratch->packed_b == nullptr) {
            scratch->packed_b_cap = 0;
            return false;
        }
        scratch->packed_b_cap = pack_b_elems;
    }

    if (scratch->c_panel_cap < c_panel_elems) {
        free(scratch->c_panel);
        scratch->c_panel = (float *)marmot_aligned_alloc(64, c_panel_elems * sizeof(float));
        if (scratch->c_panel == nullptr) {
            scratch->c_panel_cap = 0;
            return false;
        }
        scratch->c_panel_cap = c_panel_elems;
    }

    return true;
}

bool marmot_neon_f32_generic_scratch_ensure(
    marmot_neon_f32_generic_scratch_t *scratch, const marmot_neon_f32_params_t *params, size_t N, size_t M, size_t K
) {
    if (scratch == nullptr || params == nullptr) {
        return false;
    }

    const size_t block_n_cap = scratch_min(params->block_n, N);
    const size_t n_tiles_cap = (block_n_cap + MARMOT_NEON_F32_TILE_N - 1) / MARMOT_NEON_F32_TILE_N;

    const size_t pack_a_elems = params->block_k * MARMOT_NEON_F32_TILE_N;
    const size_t max_k_blocks = (K + params->block_k - 1) / params->block_k;
    const size_t c_tile_elems = MARMOT_NEON_F32_TILE_N * MARMOT_NEON_F32_TILE_M;
    const size_t c_panel_elems = n_tiles_cap * c_tile_elems;
    const size_t packed_b_block_size = params->block_k * MARMOT_NEON_F32_TILE_M;
    const size_t total_packed_b_storage = max_k_blocks * packed_b_block_size;

    if (scratch->packed_a_cap < pack_a_elems) {
        free(scratch->packed_a);
        scratch->packed_a = (float *)marmot_aligned_alloc(64, pack_a_elems * sizeof(float));
        if (scratch->packed_a == nullptr) {
            scratch->packed_a_cap = 0;
            return false;
        }
        scratch->packed_a_cap = pack_a_elems;
    }

    if (scratch->packed_b_cache_cap < max_k_blocks) {
        free(scratch->packed_b_cache);
        scratch->packed_b_cache = (float **)malloc(max_k_blocks * sizeof(float *));
        if (scratch->packed_b_cache == nullptr) {
            scratch->packed_b_cache_cap = 0;
            return false;
        }
        memset(scratch->packed_b_cache, 0, max_k_blocks * sizeof(float *));
        scratch->packed_b_cache_cap = max_k_blocks;
    }

    if (scratch->packed_b_storage_cap < total_packed_b_storage) {
        free(scratch->packed_b_storage);
        scratch->packed_b_storage = (float *)marmot_aligned_alloc(64, total_packed_b_storage * sizeof(float));
        if (scratch->packed_b_storage == nullptr) {
            scratch->packed_b_storage_cap = 0;
            return false;
        }
        scratch->packed_b_storage_cap = total_packed_b_storage;

        for (size_t i = 0; i < max_k_blocks; ++i) {
            scratch->packed_b_cache[i] = scratch->packed_b_storage + i * packed_b_block_size;
        }
    }

    if (scratch->c_panel_cap < c_panel_elems) {
        free(scratch->c_panel);
        scratch->c_panel = (float *)marmot_aligned_alloc(64, c_panel_elems * sizeof(float));
        if (scratch->c_panel == nullptr) {
            scratch->c_panel_cap = 0;
            return false;
        }
        scratch->c_panel_cap = c_panel_elems;
    }

    return true;
}

bool marmot_neon_f64_scratch_ensure(
    marmot_neon_f64_scratch_t *scratch, size_t block_m, size_t block_n, size_t block_k, bool double_buffer
) {
    if (scratch == nullptr) {
        return false;
    }

    const size_t dgemm_tile_m = 4;
    const size_t dgemm_tile_n = 8;
    const size_t n_tiles_cap = (block_n + dgemm_tile_n - 1) / dgemm_tile_n;
    const size_t m_tiles_cap = (block_m + dgemm_tile_m - 1) / dgemm_tile_m;

    const size_t pack_a_elems = block_k * dgemm_tile_n;
    const size_t pack_b_elems = block_k * dgemm_tile_m;
    const size_t c_tile_elems = dgemm_tile_n * dgemm_tile_m;
    const size_t c_panel_elems = n_tiles_cap * m_tiles_cap * c_tile_elems;

    if (scratch->packed_a_cap < pack_a_elems) {
        free(scratch->packed_a);
        scratch->packed_a = (double *)marmot_aligned_alloc(64, pack_a_elems * sizeof(double));
        if (scratch->packed_a == nullptr) {
            scratch->packed_a_cap = 0;
            return false;
        }
        scratch->packed_a_cap = pack_a_elems;
    }

    if (scratch->packed_b_cap < pack_b_elems) {
        free(scratch->packed_b0);
        scratch->packed_b0 = (double *)marmot_aligned_alloc(64, pack_b_elems * sizeof(double));
        if (scratch->packed_b0 == nullptr) {
            scratch->packed_b_cap = 0;
            return false;
        }

        if (double_buffer) {
            free(scratch->packed_b1);
            scratch->packed_b1 = (double *)marmot_aligned_alloc(64, pack_b_elems * sizeof(double));
            if (scratch->packed_b1 == nullptr) {
                free(scratch->packed_b0);
                scratch->packed_b0 = nullptr;
                scratch->packed_b_cap = 0;
                return false;
            }
        }
        scratch->packed_b_cap = pack_b_elems;
    }

    if (scratch->c_panel_cap < c_panel_elems) {
        free(scratch->c_panel);
        scratch->c_panel = (double *)marmot_aligned_alloc(64, c_panel_elems * sizeof(double));
        if (scratch->c_panel == nullptr) {
            scratch->c_panel_cap = 0;
            return false;
        }
        scratch->c_panel_cap = c_panel_elems;
    }

    return true;
}
