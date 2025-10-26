#ifndef MARMOT_BENCH_GRID_H
#define MARMOT_BENCH_GRID_H

#include "bench_param_sweep.h"
#include "marmot/error.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// A single configuration in the parameter grid
typedef struct {
    size_t ctx_size;
    size_t batch_size;
    size_t n_threads;
    size_t n_prompt;
    size_t n_gen;
    size_t n_depth;
} marmot_bench_grid_config_t;

// Parameter grid for multi-configuration sweeps (Cartesian product)
typedef struct {
    marmot_bench_grid_config_t *configs;
    size_t count;
    size_t capacity;
} marmot_bench_grid_t;

// Initialize grid to empty
void marmot_bench_grid_init(marmot_bench_grid_t *grid);

// Build grid from parameter ranges (Cartesian product)
[[nodiscard]] marmot_error_t marmot_bench_grid_build(
    marmot_bench_grid_t *grid, const marmot_bench_param_range_t *ctx_sizes, const marmot_bench_param_range_t *n_prompts,
    const marmot_bench_param_range_t *n_gens, const marmot_bench_param_range_t *n_depths,
    const marmot_bench_param_range_t *batch_sizes, const marmot_bench_param_range_t *n_threads
);

// Free grid resources
void marmot_bench_grid_free(marmot_bench_grid_t *grid);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_BENCH_GRID_H
