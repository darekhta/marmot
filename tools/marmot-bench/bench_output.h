#ifndef MARMOT_BENCH_OUTPUT_H
#define MARMOT_BENCH_OUTPUT_H

#include "bench_core.h"

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

marmot_error_t marmot_bench_output_console(
    FILE *f, const marmot_bench_config_t *config, const marmot_bench_result_t *results, size_t num_results
);

marmot_error_t marmot_bench_output_csv(
    FILE *f, const marmot_bench_config_t *config, const marmot_bench_result_t *results, size_t num_results
);

marmot_error_t marmot_bench_output_json(
    FILE *f, const marmot_bench_config_t *config, const marmot_bench_result_t *results, size_t num_results
);

marmot_error_t marmot_bench_output_markdown(
    FILE *f, const marmot_bench_config_t *config, const marmot_bench_result_t *results, size_t num_results
);

marmot_error_t marmot_bench_output_sql(
    FILE *f, const marmot_bench_config_t *config, const marmot_bench_result_t *results, size_t num_results
);

marmot_error_t marmot_bench_output_jsonl(
    FILE *f, const marmot_bench_config_t *config, const marmot_bench_result_t *results, size_t num_results
);

#ifdef __cplusplus
}
#endif

#endif
