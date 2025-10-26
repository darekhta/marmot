#include "bench_output.h"

#include <string.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

static void get_device_name(char *buf, size_t buf_sz) {
    buf[0] = '\0';
#if defined(__APPLE__)
    size_t sz = buf_sz;
    if (sysctlbyname("machdep.cpu.brand_string", buf, &sz, nullptr, 0) != 0) {
        strncpy(buf, "Apple Silicon", buf_sz - 1);
    }
#else
    strncpy(buf, "Unknown CPU", buf_sz - 1);
#endif
    buf[buf_sz - 1] = '\0';
}

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

static void print_header(FILE *f, const marmot_bench_config_t *config) {
    char device_name[128];
    get_device_name(device_name, sizeof(device_name));

    fprintf(f, "\n");
    fprintf(f, "%s\n", "═══════════════════════════════════════════════════════════════════════════════");

    if (config->backend_mode == MARMOT_BENCH_BACKEND_COMPARE) {
        fprintf(f, "  MARMOT BENCHMARK v1.0 | %s\n", device_name);
    } else {
        const char *be = config->backend_mode == MARMOT_BENCH_BACKEND_CPU ? "CPU" : "Metal";
        fprintf(f, "  MARMOT BENCHMARK v1.0 | %s | %s\n", device_name, be);
    }

    fprintf(f, "%s\n", "═══════════════════════════════════════════════════════════════════════════════");
    fprintf(f, "\n");
}

static void print_single_result(FILE *f, const marmot_bench_result_t *r) {
    if (!r->success) {
        fprintf(f, "[%s] FAILED: %s\n\n", r->workload_name, r->error_message ? r->error_message : "unknown error");
        return;
    }

    fprintf(f, "[%s] (%s)\n", r->workload_name, backend_str(r->backend));
    fprintf(f, "  Latency:  P50=%.1fus  P95=%.1fus  P99=%.1fus\n", r->stats.p50_us, r->stats.p95_us, r->stats.p99_us);
    fprintf(f, "  Throughput: %.0f ops/sec\n", r->throughput_ops_per_sec);

    if (r->efficiency.peak_gbps > 0) {
        fprintf(
            f, "  Memory BW: %.1f GB/s (%.1f%% of %.1f GB/s peak)\n", r->efficiency.achieved_gbps,
            r->efficiency.memory_efficiency * 100.0, r->efficiency.peak_gbps
        );
    }

    if (r->efficiency.peak_tflops > 0) {
        fprintf(
            f, "  Compute: %.2f TFLOPS (%.1f%% of %.1f TFLOPS peak)\n", r->efficiency.achieved_tflops,
            r->efficiency.compute_efficiency * 100.0, r->efficiency.peak_tflops
        );
    }

    fprintf(f, "\n");
}

static void print_compare_result(
    FILE *f, const marmot_bench_result_t *cpu_r, const marmot_bench_result_t *metal_r
) {
    const char *name = cpu_r ? cpu_r->workload_name : (metal_r ? metal_r->workload_name : "unknown");

    if (cpu_r && !cpu_r->success) {
        fprintf(f, "[%s] CPU FAILED: %s\n", name, cpu_r->error_message ? cpu_r->error_message : "unknown");
    }
    if (metal_r && !metal_r->success) {
        fprintf(f, "[%s] Metal FAILED: %s\n", name, metal_r->error_message ? metal_r->error_message : "unknown");
    }

    if ((cpu_r && !cpu_r->success) || (metal_r && !metal_r->success)) {
        fprintf(f, "\n");
        return;
    }

    if (!cpu_r || !metal_r) {
        if (cpu_r)
            print_single_result(f, cpu_r);
        if (metal_r)
            print_single_result(f, metal_r);
        return;
    }

    double speedup = cpu_r->stats.p50_us / metal_r->stats.p50_us;

    fprintf(f, "[%s]\n", name);
    fprintf(
        f, "  CPU:    %7.1fus (P50)  |  Metal: %7.1fus (P50)  |  %.1fx speedup\n", cpu_r->stats.p50_us,
        metal_r->stats.p50_us, speedup
    );

    if (cpu_r->efficiency.peak_gbps > 0 && metal_r->efficiency.peak_gbps > 0) {
        fprintf(
            f, "  Memory: %5.1f GB/s (%2.0f%%)  |         %5.1f GB/s (%2.0f%%)\n", cpu_r->efficiency.achieved_gbps,
            cpu_r->efficiency.memory_efficiency * 100.0, metal_r->efficiency.achieved_gbps,
            metal_r->efficiency.memory_efficiency * 100.0
        );
    }

    if (cpu_r->efficiency.peak_tflops > 0 && metal_r->efficiency.peak_tflops > 0) {
        fprintf(
            f, "  Compute: %4.2f TFLOPS      |         %4.2f TFLOPS (%2.0f%%)\n", cpu_r->efficiency.achieved_tflops,
            metal_r->efficiency.achieved_tflops, metal_r->efficiency.compute_efficiency * 100.0
        );
    }

    fprintf(f, "\n");
}

static void print_summary(FILE *f, const marmot_bench_result_t *results, size_t num_results, bool is_compare) {
    size_t passed = 0, failed = 0;
    double speedup_sum = 0.0;
    size_t speedup_count = 0;

    for (size_t i = 0; i < num_results; ++i) {
        const marmot_bench_result_t *r = &results[i];
        if (r->success) {
            passed++;
        } else {
            failed++;
        }
    }

    if (is_compare) {
        for (size_t i = 0; i + 1 < num_results; i += 2) {
            const marmot_bench_result_t *cpu_r = &results[i];
            const marmot_bench_result_t *metal_r = &results[i + 1];
            if (cpu_r->success && metal_r->success && cpu_r->backend == MARMOT_BACKEND_CPU &&
                metal_r->backend == MARMOT_BACKEND_METAL) {
                double sp = cpu_r->stats.p50_us / metal_r->stats.p50_us;
                speedup_sum += sp;
                speedup_count++;
            }
        }
    }

    fprintf(f, "%s\n", "═══════════════════════════════════════════════════════════════════════════════");

    if (is_compare && speedup_count > 0) {
        fprintf(
            f, "  SUMMARY: %zu workloads | Avg Metal speedup: %.1fx\n", speedup_count, speedup_sum / speedup_count
        );
    } else {
        fprintf(f, "  SUMMARY: %zu passed, %zu failed\n", passed, failed);
    }

    fprintf(f, "%s\n", "═══════════════════════════════════════════════════════════════════════════════");
}

marmot_error_t marmot_bench_output_console(
    FILE *f, const marmot_bench_config_t *config, const marmot_bench_result_t *results, size_t num_results
) {
    print_header(f, config);

    bool is_compare = config->backend_mode == MARMOT_BENCH_BACKEND_COMPARE;

    if (is_compare) {
        for (size_t i = 0; i + 1 < num_results; i += 2) {
            const marmot_bench_result_t *cpu_r = &results[i];
            const marmot_bench_result_t *metal_r = &results[i + 1];
            print_compare_result(f, cpu_r, metal_r);
        }
        if (num_results % 2 == 1) {
            print_single_result(f, &results[num_results - 1]);
        }
    } else {
        for (size_t i = 0; i < num_results; ++i) {
            print_single_result(f, &results[i]);
        }
    }

    print_summary(f, results, num_results, is_compare);

    return MARMOT_SUCCESS;
}
