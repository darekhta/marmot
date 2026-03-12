#include "bench_core.h"
#include "bench_stats.h"

#include "backends/cpu/cpu_caps.h"
#if MARMOT_ENABLE_METAL
marmot_device_caps_t marmot_metal_detect_default_caps(void);
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void marmot_bench_config_defaults(marmot_bench_config_t *config) {
    memset(config, 0, sizeof(*config));
    config->backend_mode = MARMOT_BENCH_BACKEND_COMPARE;
    config->warmup_iterations = 50;
    config->measure_iterations = 100;
    config->batch_size = 0;
    config->min_time_seconds = 1.0;
    config->target_sample_time_us = 10000.0;
    config->confidence_level = 0.95;
    config->category_mask = MARMOT_BENCH_CATEGORY_ALL;
    config->workload_filter = nullptr;
    config->output_format = MARMOT_BENCH_OUTPUT_CONSOLE;
    config->output_path = nullptr;
    config->verbose = false;
}

marmot_bench_suite_t *marmot_bench_suite_create(const char *name) {
    marmot_bench_suite_t *suite = calloc(1, sizeof(marmot_bench_suite_t));
    if (suite == nullptr)
        return nullptr;

    suite->name = name;
    suite->capacity = 64;
    suite->workloads = calloc(suite->capacity, sizeof(marmot_bench_workload_t *));
    if (suite->workloads == nullptr) {
        free(suite);
        return nullptr;
    }
    suite->num_workloads = 0;
    return suite;
}

void marmot_bench_suite_destroy(marmot_bench_suite_t *suite) {
    if (suite == nullptr)
        return;
    free(suite->workloads);
    free(suite);
}

marmot_error_t marmot_bench_suite_add(marmot_bench_suite_t *suite, marmot_bench_workload_t *workload) {
    if (suite == nullptr || workload == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (suite->num_workloads >= suite->capacity) {
        size_t new_cap = suite->capacity * 2;
        marmot_bench_workload_t **new_arr = realloc(suite->workloads, new_cap * sizeof(marmot_bench_workload_t *));
        if (new_arr == nullptr) {
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        suite->workloads = new_arr;
        suite->capacity = new_cap;
    }

    suite->workloads[suite->num_workloads++] = workload;
    return MARMOT_SUCCESS;
}

void marmot_bench_get_device_caps(marmot_backend_type_t backend, marmot_device_caps_t *caps) {
    memset(caps, 0, sizeof(*caps));
    if (backend == MARMOT_BACKEND_CPU) {
        *caps = marmot_cpu_detect_capabilities();
    }
#if MARMOT_ENABLE_METAL
    else if (backend == MARMOT_BACKEND_METAL) {
        *caps = marmot_metal_detect_default_caps();
    }
#endif
}

static marmot_error_t run_single_workload(
    marmot_backend_type_t backend,
    marmot_context_t *ctx,
    const marmot_bench_workload_t *workload,
    const marmot_bench_config_t *config,
    marmot_bench_result_t *out_result
) {
    memset(out_result, 0, sizeof(*out_result));
    out_result->workload_name = workload->desc.name;
    out_result->category = workload->desc.category;
    out_result->backend = backend;

    marmot_graph_t *graph = nullptr;
    marmot_tensor_t **inputs = nullptr;
    marmot_tensor_t **outputs = nullptr;
    size_t num_inputs = 0, num_outputs = 0;

    marmot_error_t err =
        workload->setup(backend, ctx, &graph, &inputs, &num_inputs, &outputs, &num_outputs, workload->user_data);
    if (err != MARMOT_SUCCESS) {
        out_result->success = false;
        out_result->error_message = marmot_get_last_error_detail();
        return err;
    }

    for (uint32_t i = 0; i < config->warmup_iterations; ++i) {
        workload->execute(ctx, graph, inputs, num_inputs, outputs, num_outputs);
    }
    (void)marmot_device_synchronize(ctx);

    uint32_t batch_size = config->batch_size;
    if (batch_size == 0) {
        const uint32_t calib_iters = 10;
        double calib_start = now_seconds();
        for (uint32_t i = 0; i < calib_iters; ++i) {
            workload->execute(ctx, graph, inputs, num_inputs, outputs, num_outputs);
        }
        (void)marmot_device_synchronize(ctx);
        double calib_end = now_seconds();
        double per_op_us = (calib_end - calib_start) * 1e6 / calib_iters;

        if (per_op_us > 0) {
            batch_size = (uint32_t)(config->target_sample_time_us / per_op_us);
            if (batch_size < 1) batch_size = 1;
            if (batch_size > 10000) batch_size = 10000;
        } else {
            batch_size = 100;
        }

        if (config->verbose) {
            fprintf(stderr, "(batch=%u, ~%.0f us/op) ", batch_size, per_op_us);
        }
    }

    size_t max_samples = config->measure_iterations * 2;
    double *samples = malloc(max_samples * sizeof(double));
    if (samples == nullptr) {
        workload->teardown(graph, inputs, num_inputs, outputs, num_outputs, workload->user_data);
        out_result->success = false;
        out_result->error_message = "Out of memory";
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    size_t sample_count = 0;
    double total_time = 0.0;

    while (sample_count < config->measure_iterations || total_time < config->min_time_seconds) {
        if (sample_count >= max_samples) {
            max_samples *= 2;
            double *new_samples = realloc(samples, max_samples * sizeof(double));
            if (new_samples == nullptr) {
                free(samples);
                workload->teardown(graph, inputs, num_inputs, outputs, num_outputs, workload->user_data);
                out_result->success = false;
                out_result->error_message = "Out of memory";
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            samples = new_samples;
        }

        double start = now_seconds();
        for (uint32_t b = 0; b < batch_size; ++b) {
            err = workload->execute(ctx, graph, inputs, num_inputs, outputs, num_outputs);
            if (err != MARMOT_SUCCESS) {
                free(samples);
                workload->teardown(graph, inputs, num_inputs, outputs, num_outputs, workload->user_data);
                out_result->success = false;
                out_result->error_message = marmot_get_last_error_detail();
                return err;
            }
        }
        (void)marmot_device_synchronize(ctx);
        double end = now_seconds();

        double elapsed_us = (end - start) * 1e6 / batch_size;
        samples[sample_count++] = elapsed_us;
        total_time += (end - start);
    }

    marmot_bench_compute_stats(samples, sample_count, config->confidence_level, &out_result->stats);

    marmot_device_caps_t caps;
    marmot_bench_get_device_caps(backend, &caps);

    double seconds = out_result->stats.mean_us * 1e-6;
    if (workload->desc.flops > 0 && seconds > 0) {
        out_result->efficiency.achieved_tflops = (double)workload->desc.flops / seconds / 1e12;
        if (workload->desc.primary_dtype == MARMOT_DTYPE_FLOAT16 ||
            workload->desc.primary_dtype == MARMOT_DTYPE_BFLOAT16) {
            out_result->efficiency.peak_tflops = caps.peak_flops_tflops_fp16;
        } else {
            out_result->efficiency.peak_tflops = caps.peak_flops_tflops_fp32;
        }
        if (out_result->efficiency.peak_tflops > 0) {
            out_result->efficiency.compute_efficiency =
                out_result->efficiency.achieved_tflops / out_result->efficiency.peak_tflops;
        }
    }

    uint64_t total_bytes = workload->desc.bytes_read + workload->desc.bytes_written;
    if (total_bytes > 0 && seconds > 0) {
        out_result->efficiency.achieved_gbps = (double)total_bytes / seconds / 1e9;
        out_result->efficiency.peak_gbps = caps.mem_bw_gbps;
        if (out_result->efficiency.peak_gbps > 0) {
            out_result->efficiency.memory_efficiency =
                out_result->efficiency.achieved_gbps / out_result->efficiency.peak_gbps;
        }
    }

    if (workload->desc.flops > 0 && total_bytes > 0) {
        out_result->efficiency.arithmetic_intensity = (double)workload->desc.flops / (double)total_bytes;
    }

    workload->teardown(graph, inputs, num_inputs, outputs, num_outputs, workload->user_data);
    free(samples);

    out_result->success = true;
    out_result->throughput_ops_per_sec = 1e6 / out_result->stats.mean_us;

    return MARMOT_SUCCESS;
}

static bool workload_matches_filter(const marmot_bench_workload_t *w, const marmot_bench_config_t *config) {
    if ((w->desc.category & config->category_mask) == 0)
        return false;
    if (config->workload_filter != nullptr && strstr(w->desc.name, config->workload_filter) == nullptr)
        return false;
    return true;
}

static size_t count_matching_workloads(const marmot_bench_config_t *config, const marmot_bench_suite_t *suite) {
    size_t count = 0;
    for (size_t i = 0; i < suite->num_workloads; ++i) {
        if (workload_matches_filter(suite->workloads[i], config))
            count++;
    }
    return count;
}

marmot_error_t marmot_bench_run(
    const marmot_bench_config_t *config,
    const marmot_bench_suite_t *suite,
    marmot_bench_result_t **results,
    size_t *num_results
) {
    size_t match_count = count_matching_workloads(config, suite);
    if (match_count == 0) {
        *results = nullptr;
        *num_results = 0;
        return MARMOT_SUCCESS;
    }

    bool run_cpu = config->backend_mode == MARMOT_BENCH_BACKEND_CPU ||
                   config->backend_mode == MARMOT_BENCH_BACKEND_COMPARE;
    bool run_metal = config->backend_mode == MARMOT_BENCH_BACKEND_METAL ||
                     config->backend_mode == MARMOT_BENCH_BACKEND_COMPARE;

#if !MARMOT_ENABLE_METAL
    if (run_metal && config->backend_mode == MARMOT_BENCH_BACKEND_METAL) {
        fprintf(stderr, "Metal backend not available\n");
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
    }
    run_metal = false;
#endif

    size_t backends_count = (run_cpu ? 1 : 0) + (run_metal ? 1 : 0);
    size_t total_results = match_count * backends_count;

    *results = calloc(total_results, sizeof(marmot_bench_result_t));
    if (*results == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    *num_results = 0;

    marmot_context_t *cpu_ctx = nullptr;
    marmot_context_t *metal_ctx = nullptr;

    if (run_cpu) {
        cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
        if (cpu_ctx == nullptr) {
            free(*results);
            *results = nullptr;
            return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
        }
    }

#if MARMOT_ENABLE_METAL
    if (run_metal) {
        metal_ctx = marmot_init(MARMOT_BACKEND_METAL);
        if (metal_ctx == nullptr) {
            if (config->backend_mode == MARMOT_BENCH_BACKEND_METAL) {
                if (cpu_ctx)
                    marmot_destroy(cpu_ctx);
                free(*results);
                *results = nullptr;
                return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
            }
            run_metal = false;
        }
    }
#endif

    for (size_t i = 0; i < suite->num_workloads; ++i) {
        const marmot_bench_workload_t *w = suite->workloads[i];
        if (!workload_matches_filter(w, config))
            continue;

        if (config->verbose) {
            fprintf(stderr, "Running: %s\n", w->desc.name);
        }

        if (run_cpu && cpu_ctx) {
            if (config->verbose) {
                fprintf(stderr, "  [CPU] ");
            }
            run_single_workload(MARMOT_BACKEND_CPU, cpu_ctx, w, config, &(*results)[*num_results]);
            (*num_results)++;
        }

#if MARMOT_ENABLE_METAL
        if (run_metal && metal_ctx) {
            if (config->verbose) {
                fprintf(stderr, "  [Metal] ");
            }
            run_single_workload(MARMOT_BACKEND_METAL, metal_ctx, w, config, &(*results)[*num_results]);
            (*num_results)++;
        }
#endif
    }

    if (cpu_ctx)
        marmot_destroy(cpu_ctx);
#if MARMOT_ENABLE_METAL
    if (metal_ctx)
        marmot_destroy(metal_ctx);
#endif

    return MARMOT_SUCCESS;
}

void marmot_bench_results_free(marmot_bench_result_t *results, size_t num_results) {
    (void)num_results;
    free(results);
}
