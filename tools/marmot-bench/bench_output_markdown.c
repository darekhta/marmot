#include "bench_output.h"

#include <time.h>

static const char *backend_str(marmot_backend_type_t backend) {
    switch (backend) {
    case MARMOT_BACKEND_CPU:
        return "CPU";
    case MARMOT_BACKEND_METAL:
        return "Metal";
    default:
        return "Unknown";
    }
}

marmot_error_t marmot_bench_output_markdown(
    FILE *f, const marmot_bench_config_t *config, const marmot_bench_result_t *results, size_t num_results
) {
    time_t now = time(nullptr);
    struct tm *tm_info = localtime(&now);
    char date_str[64];
    strftime(date_str, sizeof(date_str), "%Y-%m-%d %H:%M:%S", tm_info);

    fprintf(f, "# Marmot Benchmark Results\n\n");
    fprintf(f, "**Date:** %s\n\n", date_str);

    fprintf(f, "## Configuration\n\n");
    fprintf(f, "| Parameter | Value |\n");
    fprintf(f, "|-----------|-------|\n");
    fprintf(f, "| Warmup Iterations | %u |\n", config->warmup_iterations);
    fprintf(f, "| Measure Iterations | %u |\n", config->measure_iterations);
    fprintf(f, "| Min Time (s) | %.2f |\n", config->min_time_seconds);
    fprintf(f, "| Confidence Level | %.0f%% |\n\n", config->confidence_level * 100);

    bool is_compare = config->backend_mode == MARMOT_BENCH_BACKEND_COMPARE;

    if (is_compare) {
        fprintf(f, "## Comparison Results\n\n");
        fprintf(f, "| Workload | CPU P50 (μs) | Metal P50 (μs) | Speedup |\n");
        fprintf(f, "|----------|-------------|----------------|--------|\n");

        for (size_t i = 0; i + 1 < num_results; i += 2) {
            const marmot_bench_result_t *cpu_r = &results[i];
            const marmot_bench_result_t *metal_r = &results[i + 1];

            if (!cpu_r->success || !metal_r->success)
                continue;

            double speedup = cpu_r->stats.p50_us / metal_r->stats.p50_us;
            fprintf(
                f, "| %s | %.1f | %.1f | %.1fx |\n", cpu_r->workload_name, cpu_r->stats.p50_us,
                metal_r->stats.p50_us, speedup
            );
        }
        fprintf(f, "\n");

        fprintf(f, "## Detailed Results\n\n");
    }

    fprintf(f, "### Latency Statistics\n\n");
    fprintf(f, "| Workload | Backend | Mean (μs) | P50 (μs) | P95 (μs) | P99 (μs) | Std (μs) |\n");
    fprintf(f, "|----------|---------|-----------|----------|----------|----------|----------|\n");

    for (size_t i = 0; i < num_results; ++i) {
        const marmot_bench_result_t *r = &results[i];
        if (!r->success)
            continue;

        fprintf(
            f, "| %s | %s | %.1f | %.1f | %.1f | %.1f | %.1f |\n", r->workload_name, backend_str(r->backend),
            r->stats.mean_us, r->stats.p50_us, r->stats.p95_us, r->stats.p99_us, r->stats.stddev_us
        );
    }
    fprintf(f, "\n");

    fprintf(f, "### Efficiency Metrics\n\n");
    fprintf(f, "| Workload | Backend | TFLOPS | Compute %% | GB/s | Memory %% | AI |\n");
    fprintf(f, "|----------|---------|--------|-----------|------|----------|----|\n");

    for (size_t i = 0; i < num_results; ++i) {
        const marmot_bench_result_t *r = &results[i];
        if (!r->success)
            continue;

        fprintf(
            f, "| %s | %s | %.2f | %.1f%% | %.1f | %.1f%% | %.1f |\n", r->workload_name, backend_str(r->backend),
            r->efficiency.achieved_tflops, r->efficiency.compute_efficiency * 100, r->efficiency.achieved_gbps,
            r->efficiency.memory_efficiency * 100, r->efficiency.arithmetic_intensity
        );
    }
    fprintf(f, "\n");

    return MARMOT_SUCCESS;
}
