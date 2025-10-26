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

marmot_error_t marmot_bench_output_jsonl(
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
    struct tm *tm_info = gmtime(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", tm_info);

    // Output one JSON object per line
    for (size_t i = 0; i < num_results; i++) {
        const marmot_bench_result_t *r = &results[i];

        fprintf(
            f,
            "{\"timestamp\":\"%s\","
            "\"backend\":\"%s\","
            "\"category\":\"%s\","
            "\"workload\":\"%s\","
            "\"success\":%s,"
            "\"stats\":{\"samples\":%zu,\"min_us\":%.3f,\"max_us\":%.3f,\"mean_us\":%.3f,"
            "\"p50_us\":%.3f,\"p95_us\":%.3f,\"p99_us\":%.3f,\"stddev_us\":%.3f},"
            "\"throughput_ops\":%.2f,"
            "\"efficiency\":{\"tflops\":%.6f,\"gbps\":%.3f,"
            "\"compute_efficiency\":%.4f,\"memory_efficiency\":%.4f,"
            "\"arithmetic_intensity\":%.3f}}\n",
            timestamp,
            backend_str(r->backend),
            category_str(r->category),
            r->workload_name != nullptr ? r->workload_name : "unknown",
            r->success ? "true" : "false",
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
