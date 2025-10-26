#ifndef MARMOT_BENCH_CORE_H
#define MARMOT_BENCH_CORE_H

#include "marmot/marmot.h"
#include "marmot/graph/graph.h"
#include "marmot/graph/op_signature.h"
#include "marmot/device_caps.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MARMOT_BENCH_OUTPUT_CONSOLE = 0,
    MARMOT_BENCH_OUTPUT_JSON = 1,
    MARMOT_BENCH_OUTPUT_MARKDOWN = 2,
    MARMOT_BENCH_OUTPUT_CSV = 3,
    MARMOT_BENCH_OUTPUT_SQL = 4,
    MARMOT_BENCH_OUTPUT_JSONL = 5,
} marmot_bench_output_format_t;

typedef enum {
    MARMOT_BENCH_CATEGORY_MICRO = 1 << 0,
    MARMOT_BENCH_CATEGORY_COMPOSITE = 1 << 1,
    MARMOT_BENCH_CATEGORY_ALL = 0x3,
} marmot_bench_category_t;

typedef enum {
    MARMOT_BENCH_BACKEND_CPU = 0,
    MARMOT_BENCH_BACKEND_METAL = 1,
    MARMOT_BENCH_BACKEND_COMPARE = 2,
} marmot_bench_backend_mode_t;

typedef struct {
    marmot_bench_backend_mode_t backend_mode;
    uint32_t warmup_iterations;
    uint32_t measure_iterations;
    uint32_t batch_size;
    double min_time_seconds;
    double target_sample_time_us;
    double confidence_level;
    uint32_t category_mask;
    const char *workload_filter;
    marmot_bench_output_format_t output_format;
    const char *output_path;
    bool verbose;
} marmot_bench_config_t;

typedef struct {
    double min_us;
    double max_us;
    double mean_us;
    double stddev_us;
    double p50_us;
    double p95_us;
    double p99_us;
    double ci_low_us;
    double ci_high_us;
    size_t sample_count;
} marmot_bench_stats_t;

typedef struct {
    double achieved_tflops;
    double peak_tflops;
    double compute_efficiency;
    double achieved_gbps;
    double peak_gbps;
    double memory_efficiency;
    double arithmetic_intensity;
} marmot_bench_efficiency_t;

typedef struct {
    const char *workload_name;
    marmot_bench_category_t category;
    marmot_backend_type_t backend;
    marmot_bench_stats_t stats;
    marmot_bench_efficiency_t efficiency;
    double throughput_ops_per_sec;
    bool success;
    const char *error_message;
} marmot_bench_result_t;

typedef struct {
    const char *name;
    marmot_bench_category_t category;
    marmot_dtype_t primary_dtype;
    uint64_t flops;
    uint64_t bytes_read;
    uint64_t bytes_written;
    marmot_op_signature_t signature;
} marmot_bench_workload_desc_t;

typedef struct marmot_bench_workload marmot_bench_workload_t;

struct marmot_bench_workload {
    marmot_bench_workload_desc_t desc;
    void *user_data;

    marmot_error_t (*setup)(
        marmot_backend_type_t backend,
        marmot_context_t *ctx,
        marmot_graph_t **graph,
        marmot_tensor_t ***inputs,
        size_t *num_inputs,
        marmot_tensor_t ***outputs,
        size_t *num_outputs,
        void *user_data
    );

    marmot_error_t (*execute)(
        marmot_context_t *ctx,
        marmot_graph_t *graph,
        marmot_tensor_t **inputs,
        size_t num_inputs,
        marmot_tensor_t **outputs,
        size_t num_outputs
    );

    void (*teardown)(
        marmot_graph_t *graph,
        marmot_tensor_t **inputs,
        size_t num_inputs,
        marmot_tensor_t **outputs,
        size_t num_outputs,
        void *user_data
    );
};

typedef struct {
    const char *name;
    marmot_bench_workload_t **workloads;
    size_t num_workloads;
    size_t capacity;
} marmot_bench_suite_t;

void marmot_bench_config_defaults(marmot_bench_config_t *config);

marmot_bench_suite_t *marmot_bench_suite_create(const char *name);
void marmot_bench_suite_destroy(marmot_bench_suite_t *suite);
marmot_error_t marmot_bench_suite_add(marmot_bench_suite_t *suite, marmot_bench_workload_t *workload);

marmot_error_t marmot_bench_run(
    const marmot_bench_config_t *config,
    const marmot_bench_suite_t *suite,
    marmot_bench_result_t **results,
    size_t *num_results
);

void marmot_bench_results_free(marmot_bench_result_t *results, size_t num_results);

marmot_error_t marmot_bench_output(
    const marmot_bench_config_t *config,
    const marmot_bench_result_t *results,
    size_t num_results
);

void marmot_bench_get_device_caps(marmot_backend_type_t backend, marmot_device_caps_t *caps);

#ifdef __cplusplus
}
#endif

#endif
