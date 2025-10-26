#ifndef MARMOT_NEON_MATMUL_SCRATCH_H
#define MARMOT_NEON_MATMUL_SCRATCH_H

#include <stdbool.h>
#include <stddef.h>

#include "neon_matmul_params.h"

#ifdef __cplusplus
extern "C" {
#endif

// Per-worker scratch for standard f32 matmul (marmot_neon_f32_compute_rows)
typedef struct {
    float *packed_a;
    float *packed_b;
    float *c_panel;
    size_t packed_a_cap;
    size_t packed_b_cap;
    size_t c_panel_cap;
} marmot_neon_f32_scratch_t;

// Per-worker scratch for generic f32 matmul with packed_b cache
typedef struct {
    float *packed_a;
    float **packed_b_cache;
    float *packed_b_storage;
    float *c_panel;
    size_t packed_a_cap;
    size_t packed_b_cache_cap;
    size_t packed_b_storage_cap;
    size_t c_panel_cap;
} marmot_neon_f32_generic_scratch_t;

// Per-worker scratch for f64 matmul with double buffering
typedef struct {
    double *packed_a;
    double *packed_b0;
    double *packed_b1;
    double *c_panel;
    size_t packed_a_cap;
    size_t packed_b_cap;
    size_t c_panel_cap;
} marmot_neon_f64_scratch_t;

// Pool owned by cpu_context_t
typedef struct {
    marmot_neon_f32_scratch_t *f32;
    marmot_neon_f32_generic_scratch_t *f32_gen;
    marmot_neon_f64_scratch_t *f64;
    size_t num_workers;
} marmot_neon_scratch_pool_t;

// Initialize pool with given number of workers (allocates arrays, not buffers)
void marmot_neon_scratch_pool_init(marmot_neon_scratch_pool_t *pool, size_t num_workers);

// Destroy pool and free all allocated buffers
void marmot_neon_scratch_pool_destroy(marmot_neon_scratch_pool_t *pool);

// Ensure f32 scratch buffers are large enough for given dimensions
// Returns true on success, false on allocation failure
bool marmot_neon_f32_scratch_ensure(
    marmot_neon_f32_scratch_t *scratch, const marmot_neon_f32_params_t *params, size_t N, size_t M, size_t K
);

// Ensure generic f32 scratch buffers (with packed_b cache) are large enough
bool marmot_neon_f32_generic_scratch_ensure(
    marmot_neon_f32_generic_scratch_t *scratch, const marmot_neon_f32_params_t *params, size_t N, size_t M, size_t K
);

// Ensure f64 scratch buffers are large enough for given dimensions
bool marmot_neon_f64_scratch_ensure(
    marmot_neon_f64_scratch_t *scratch, size_t block_m, size_t block_n, size_t block_k, bool double_buffer
);

#ifdef __cplusplus
}
#endif

#endif
