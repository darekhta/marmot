#include "bench_output.h"

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

// Escape CSV field if it contains comma, quote, or newline
static void write_csv_field(FILE *f, const char *value) {
    bool needs_quote = false;
    for (const char *p = value; *p; p++) {
        if (*p == ',' || *p == '"' || *p == '\n' || *p == '\r') {
            needs_quote = true;
            break;
        }
    }

    if (!needs_quote) {
        fprintf(f, "%s", value);
        return;
    }

    fputc('"', f);
    for (const char *p = value; *p; p++) {
        if (*p == '"') {
            fputs("\"\"", f);
        } else {
            fputc(*p, f);
        }
    }
    fputc('"', f);
}

marmot_error_t marmot_bench_output_csv(
    FILE *f, const marmot_bench_config_t *config, const marmot_bench_result_t *results, size_t num_results
) {
    if (f == nullptr || config == nullptr || results == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    // Get timestamp
    time_t now = time(nullptr);
    struct tm *tm_info = gmtime(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", tm_info);

    // Write CSV header
    fprintf(
        f, "timestamp,backend,category,workload,success,"
           "samples,min_us,max_us,mean_us,p50_us,p95_us,p99_us,stddev_us,"
           "throughput_ops,tflops,gbps,compute_eff,memory_eff,arithmetic_intensity\n"
    );

    // Write data rows
    for (size_t i = 0; i < num_results; i++) {
        const marmot_bench_result_t *r = &results[i];

        fprintf(f, "%s,", timestamp);
        fprintf(f, "%s,", backend_str(r->backend));
        fprintf(f, "%s,", category_str(r->category));
        write_csv_field(f, r->workload_name != nullptr ? r->workload_name : "unknown");
        fprintf(f, ",");
        fprintf(f, "%s,", r->success ? "true" : "false");

        // Statistics
        fprintf(f, "%zu,", r->stats.sample_count);
        fprintf(f, "%.3f,", r->stats.min_us);
        fprintf(f, "%.3f,", r->stats.max_us);
        fprintf(f, "%.3f,", r->stats.mean_us);
        fprintf(f, "%.3f,", r->stats.p50_us);
        fprintf(f, "%.3f,", r->stats.p95_us);
        fprintf(f, "%.3f,", r->stats.p99_us);
        fprintf(f, "%.3f,", r->stats.stddev_us);

        // Throughput and efficiency
        fprintf(f, "%.2f,", r->throughput_ops_per_sec);
        fprintf(f, "%.6f,", r->efficiency.achieved_tflops);
        fprintf(f, "%.3f,", r->efficiency.achieved_gbps);
        fprintf(f, "%.4f,", r->efficiency.compute_efficiency);
        fprintf(f, "%.4f,", r->efficiency.memory_efficiency);
        fprintf(f, "%.3f\n", r->efficiency.arithmetic_intensity);
    }

    return MARMOT_SUCCESS;
}
