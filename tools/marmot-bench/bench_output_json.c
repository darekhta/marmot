#include "bench_output.h"

#include <time.h>

static void print_escaped_string(FILE *f, const char *s) {
    if (s == nullptr) {
        fprintf(f, "null");
        return;
    }
    fprintf(f, "\"");
    for (const char *p = s; *p; ++p) {
        switch (*p) {
        case '"':
            fprintf(f, "\\\"");
            break;
        case '\\':
            fprintf(f, "\\\\");
            break;
        case '\n':
            fprintf(f, "\\n");
            break;
        case '\r':
            fprintf(f, "\\r");
            break;
        case '\t':
            fprintf(f, "\\t");
            break;
        default:
            fputc(*p, f);
            break;
        }
    }
    fprintf(f, "\"");
}

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

marmot_error_t marmot_bench_output_json(
    FILE *f, const marmot_bench_config_t *config, const marmot_bench_result_t *results, size_t num_results
) {
    time_t now = time(nullptr);
    struct tm *tm_info = gmtime(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", tm_info);

    fprintf(f, "{\n");
    fprintf(f, "  \"version\": \"1.0\",\n");
    fprintf(f, "  \"timestamp\": \"%s\",\n", timestamp);
    fprintf(f, "  \"config\": {\n");
    fprintf(f, "    \"warmup_iterations\": %u,\n", config->warmup_iterations);
    fprintf(f, "    \"measure_iterations\": %u,\n", config->measure_iterations);
    fprintf(f, "    \"min_time_seconds\": %.2f,\n", config->min_time_seconds);
    fprintf(f, "    \"confidence_level\": %.2f\n", config->confidence_level);
    fprintf(f, "  },\n");

    fprintf(f, "  \"results\": [\n");

    for (size_t i = 0; i < num_results; ++i) {
        const marmot_bench_result_t *r = &results[i];

        fprintf(f, "    {\n");
        fprintf(f, "      \"workload\": ");
        print_escaped_string(f, r->workload_name);
        fprintf(f, ",\n");
        fprintf(f, "      \"backend\": \"%s\",\n", backend_str(r->backend));
        fprintf(f, "      \"success\": %s,\n", r->success ? "true" : "false");

        if (!r->success) {
            fprintf(f, "      \"error\": ");
            print_escaped_string(f, r->error_message);
            fprintf(f, "\n");
        } else {
            fprintf(f, "      \"latency_us\": {\n");
            fprintf(f, "        \"mean\": %.2f,\n", r->stats.mean_us);
            fprintf(f, "        \"std\": %.2f,\n", r->stats.stddev_us);
            fprintf(f, "        \"min\": %.2f,\n", r->stats.min_us);
            fprintf(f, "        \"max\": %.2f,\n", r->stats.max_us);
            fprintf(f, "        \"p50\": %.2f,\n", r->stats.p50_us);
            fprintf(f, "        \"p95\": %.2f,\n", r->stats.p95_us);
            fprintf(f, "        \"p99\": %.2f,\n", r->stats.p99_us);
            fprintf(f, "        \"ci_low\": %.2f,\n", r->stats.ci_low_us);
            fprintf(f, "        \"ci_high\": %.2f,\n", r->stats.ci_high_us);
            fprintf(f, "        \"samples\": %zu\n", r->stats.sample_count);
            fprintf(f, "      },\n");

            fprintf(f, "      \"throughput_ops_per_sec\": %.2f,\n", r->throughput_ops_per_sec);

            fprintf(f, "      \"efficiency\": {\n");
            fprintf(f, "        \"achieved_tflops\": %.4f,\n", r->efficiency.achieved_tflops);
            fprintf(f, "        \"peak_tflops\": %.2f,\n", r->efficiency.peak_tflops);
            fprintf(f, "        \"compute_efficiency\": %.4f,\n", r->efficiency.compute_efficiency);
            fprintf(f, "        \"achieved_gbps\": %.2f,\n", r->efficiency.achieved_gbps);
            fprintf(f, "        \"peak_gbps\": %.2f,\n", r->efficiency.peak_gbps);
            fprintf(f, "        \"memory_efficiency\": %.4f,\n", r->efficiency.memory_efficiency);
            fprintf(f, "        \"arithmetic_intensity\": %.2f\n", r->efficiency.arithmetic_intensity);
            fprintf(f, "      }\n");
        }

        fprintf(f, "    }%s\n", i + 1 < num_results ? "," : "");
    }

    fprintf(f, "  ]\n");
    fprintf(f, "}\n");

    return MARMOT_SUCCESS;
}
