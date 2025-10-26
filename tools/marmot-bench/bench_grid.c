#include "bench_grid.h"

#include <stdlib.h>

void marmot_bench_grid_init(marmot_bench_grid_t *grid) {
    grid->configs = nullptr;
    grid->count = 0;
    grid->capacity = 0;
}

static marmot_error_t grid_ensure_capacity(marmot_bench_grid_t *grid, size_t needed) {
    if (needed <= grid->capacity) {
        return MARMOT_SUCCESS;
    }

    size_t new_cap = grid->capacity == 0 ? 16 : grid->capacity * 2;
    while (new_cap < needed) {
        new_cap *= 2;
    }

    marmot_bench_grid_config_t *new_configs = realloc(grid->configs, new_cap * sizeof(marmot_bench_grid_config_t));
    if (new_configs == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    grid->configs = new_configs;
    grid->capacity = new_cap;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_bench_grid_build(
    marmot_bench_grid_t *grid, const marmot_bench_param_range_t *ctx_sizes, const marmot_bench_param_range_t *n_prompts,
    const marmot_bench_param_range_t *n_gens, const marmot_bench_param_range_t *n_depths,
    const marmot_bench_param_range_t *batch_sizes, const marmot_bench_param_range_t *n_threads
) {
    if (grid == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    // Free any existing configs
    marmot_bench_grid_free(grid);
    marmot_bench_grid_init(grid);

    // Calculate total configurations (Cartesian product)
    size_t n_ctx = ctx_sizes != nullptr && ctx_sizes->count > 0 ? ctx_sizes->count : 1;
    size_t n_pp = n_prompts != nullptr && n_prompts->count > 0 ? n_prompts->count : 1;
    size_t n_tg = n_gens != nullptr && n_gens->count > 0 ? n_gens->count : 1;
    size_t n_depth = n_depths != nullptr && n_depths->count > 0 ? n_depths->count : 1;
    size_t n_batch = batch_sizes != nullptr && batch_sizes->count > 0 ? batch_sizes->count : 1;
    size_t n_thread = n_threads != nullptr && n_threads->count > 0 ? n_threads->count : 1;

    size_t total = n_ctx * n_pp * n_tg * n_depth * n_batch * n_thread;

    marmot_error_t err = grid_ensure_capacity(grid, total);
    if (err != MARMOT_SUCCESS) {
        return err;
    }

    // Build Cartesian product
    for (size_t i_ctx = 0; i_ctx < n_ctx; i_ctx++) {
        for (size_t i_batch = 0; i_batch < n_batch; i_batch++) {
            for (size_t i_thread = 0; i_thread < n_thread; i_thread++) {
                for (size_t i_pp = 0; i_pp < n_pp; i_pp++) {
                    for (size_t i_tg = 0; i_tg < n_tg; i_tg++) {
                        for (size_t i_depth = 0; i_depth < n_depth; i_depth++) {
                            marmot_bench_grid_config_t *cfg = &grid->configs[grid->count++];

                            cfg->ctx_size = (ctx_sizes != nullptr && ctx_sizes->count > 0) ? ctx_sizes->values[i_ctx] : 4096;

                            cfg->batch_size =
                                (batch_sizes != nullptr && batch_sizes->count > 0) ? batch_sizes->values[i_batch] : 512;

                            cfg->n_threads =
                                (n_threads != nullptr && n_threads->count > 0) ? n_threads->values[i_thread] : 4;

                            cfg->n_prompt = (n_prompts != nullptr && n_prompts->count > 0) ? n_prompts->values[i_pp] : 512;

                            cfg->n_gen = (n_gens != nullptr && n_gens->count > 0) ? n_gens->values[i_tg] : 128;

                            cfg->n_depth = (n_depths != nullptr && n_depths->count > 0) ? n_depths->values[i_depth] : 0;
                        }
                    }
                }
            }
        }
    }

    return MARMOT_SUCCESS;
}

void marmot_bench_grid_free(marmot_bench_grid_t *grid) {
    if (grid == nullptr) {
        return;
    }

    free(grid->configs);
    grid->configs = nullptr;
    grid->count = 0;
    grid->capacity = 0;
}
