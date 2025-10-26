#include "bench_output.h"

#include <stdio.h>
#include <time.h>

static const char *backend_str(marmot_backend_type_t backend) {
    switch (backend) {
    case MARMOT_BACKEND_CPU:
        return "cpu";
    case MARMOT_BACKEND_METAL:
        return "metal";
    default:
        return "unknown";
    }
}

static const char *category_str(marmot_bench_category_t category) {
    if (category & MARMOT_BENCH_CATEGORY_MICRO) {
        return "micro";
    }
    if (category & MARMOT_BENCH_CATEGORY_COMPOSITE) {
        return "composite";
    }
    return "unknown";
}

marmot_error_t marmot_bench_output_sql(
    FILE *f,
    const marmot_bench_config_t *config,
    const marmot_bench_result_t *results,
    size_t num_results
) {
    if (f == nullptr || config == nullptr || results == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    // Get current timestamp
    time_t now = time(nullptr);
    struct tm *tm_info = localtime(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_info);

    // Create table if not exists
    fprintf(f, "CREATE TABLE IF NOT EXISTS benchmarks (\n");
    fprintf(f, "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n");
    fprintf(f, "    timestamp TEXT NOT NULL,\n");
    fprintf(f, "    backend TEXT NOT NULL,\n");
    fprintf(f, "    category TEXT NOT NULL,\n");
    fprintf(f, "    workload TEXT NOT NULL,\n");
    fprintf(f, "    success INTEGER,\n");
    fprintf(f, "    samples INTEGER,\n");
    fprintf(f, "    min_us REAL,\n");
    fprintf(f, "    max_us REAL,\n");
    fprintf(f, "    mean_us REAL,\n");
    fprintf(f, "    p50_us REAL,\n");
    fprintf(f, "    p95_us REAL,\n");
    fprintf(f, "    p99_us REAL,\n");
    fprintf(f, "    stddev_us REAL,\n");
    fprintf(f, "    throughput_ops REAL,\n");
    fprintf(f, "    tflops REAL,\n");
    fprintf(f, "    gbps REAL,\n");
    fprintf(f, "    compute_efficiency REAL,\n");
    fprintf(f, "    memory_efficiency REAL,\n");
    fprintf(f, "    arithmetic_intensity REAL\n");
    fprintf(f, ");\n\n");

    // Insert results
    for (size_t i = 0; i < num_results; i++) {
        const marmot_bench_result_t *r = &results[i];

        fprintf(
            f,
            "INSERT INTO benchmarks (timestamp, backend, category, workload, "
            "success, samples, "
            "min_us, max_us, mean_us, p50_us, p95_us, p99_us, stddev_us, "
            "throughput_ops, tflops, gbps, compute_efficiency, memory_efficiency, "
            "arithmetic_intensity) "
            "VALUES ('%s', '%s', '%s', '%s', "
            "%d, %zu, "
            "%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, "
            "%.2f, %.6f, %.3f, %.4f, %.4f, %.3f);\n",
            timestamp,
            backend_str(r->backend),
            category_str(r->category),
            r->workload_name != nullptr ? r->workload_name : "unknown",
            r->success ? 1 : 0,
            r->stats.sample_count,
            r->stats.min_us,
            r->stats.max_us,
            r->stats.mean_us,
            r->stats.p50_us,
            r->stats.p95_us,
            r->stats.p99_us,
            r->stats.stddev_us,
            r->throughput_ops_per_sec,
            r->efficiency.achieved_tflops,
            r->efficiency.achieved_gbps,
            r->efficiency.compute_efficiency,
            r->efficiency.memory_efficiency,
            r->efficiency.arithmetic_intensity
        );
    }

    return MARMOT_SUCCESS;
}
