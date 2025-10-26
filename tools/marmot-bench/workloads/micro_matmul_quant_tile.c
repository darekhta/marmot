// MR/NR tile sweep benchmark for non-K quantized matmul (Q8_0/Q8_1/Q4_0/Q4_1/Q5_*)
// Compares blocked microkernel vs old per-row loop

#include "marmot/marmot.h"
#include "marmot/ops/matmul.h"
#include "marmot/ops/quantization.h"
#include "marmot/quant_block.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__aarch64__)
#include <sys/time.h>
#endif

static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static void generate_random_f32(float *data, size_t count, uint32_t seed) {
    for (size_t i = 0; i < count; ++i) {
        seed = seed * 1103515245u + 12345u;
        float val = (float)((seed >> 16) & 0x7fff) / 32767.0f;
        data[i] = val * 2.0f - 1.0f;
    }
}

typedef struct {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    marmot_quant_kind_t quant_kind;
    uint32_t warmup_iters;
    uint32_t bench_iters;
} tile_bench_params_t;

typedef struct {
    double mean_us;
    double min_us;
    double max_us;
    double stddev_us;
} tile_bench_stats_t;

static void compute_stats(const double *samples, size_t count, tile_bench_stats_t *stats) {
    if (count == 0) {
        memset(stats, 0, sizeof(*stats));
        return;
    }
    double sum = 0.0;
    double min_val = samples[0];
    double max_val = samples[0];
    for (size_t i = 0; i < count; ++i) {
        sum += samples[i];
        if (samples[i] < min_val) min_val = samples[i];
        if (samples[i] > max_val) max_val = samples[i];
    }
    stats->mean_us = sum / (double)count;
    stats->min_us = min_val;
    stats->max_us = max_val;

    double var = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double d = samples[i] - stats->mean_us;
        var += d * d;
    }
    stats->stddev_us = sqrt(var / (double)count);
}

static const char *quant_kind_str(marmot_quant_kind_t kind) {
    switch (kind) {
    case MARMOT_QUANT_KIND_Q8_0: return "Q8_0";
    case MARMOT_QUANT_KIND_Q8_1: return "Q8_1";
    case MARMOT_QUANT_KIND_Q4_0: return "Q4_0";
    case MARMOT_QUANT_KIND_Q4_1: return "Q4_1";
    case MARMOT_QUANT_KIND_Q5_0: return "Q5_0";
    case MARMOT_QUANT_KIND_Q5_1: return "Q5_1";
    default: return "UNK";
    }
}

typedef marmot_error_t (*quantize_fn_t)(const marmot_context_t *, const marmot_tensor_t *, marmot_tensor_t *);

static quantize_fn_t get_quantize_fn(marmot_quant_kind_t kind) {
    switch (kind) {
    case MARMOT_QUANT_KIND_Q8_0: return marmot_quantize_q8_0;
    case MARMOT_QUANT_KIND_Q8_1: return marmot_quantize_q8_1;
    case MARMOT_QUANT_KIND_Q4_0: return marmot_quantize_q4_0;
    case MARMOT_QUANT_KIND_Q4_1: return marmot_quantize_q4_1;
    case MARMOT_QUANT_KIND_Q5_0: return marmot_quantize_q5_0;
    case MARMOT_QUANT_KIND_Q5_1: return marmot_quantize_q5_1;
    default: return nullptr;
    }
}

static marmot_error_t run_single_config(
    marmot_context_t *ctx,
    const tile_bench_params_t *params,
    tile_bench_stats_t *stats
) {
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(params->quant_kind);
    if (traits == nullptr) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    quantize_fn_t quantize_fn = get_quantize_fn(params->quant_kind);
    if (quantize_fn == nullptr) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    size_t input_shape[2] = {params->N, params->K};
    size_t weight_shape[2] = {params->M, params->K};
    size_t output_shape[2] = {params->N, params->M};

    marmot_tensor_t *input = marmot_tensor_create(nullptr, input_shape, 2, MARMOT_DTYPE_FLOAT32);
    if (input == nullptr) return MARMOT_ERROR_OUT_OF_MEMORY;

    marmot_tensor_t *weight = marmot_tensor_create_quantized(nullptr, weight_shape, 2, params->quant_kind);
    if (weight == nullptr) {
        marmot_tensor_destroy(input);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    marmot_tensor_t *output = marmot_tensor_create(nullptr, output_shape, 2, MARMOT_DTYPE_FLOAT32);
    if (output == nullptr) {
        marmot_tensor_destroy(input);
        marmot_tensor_destroy(weight);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    float *weight_f32 = (float *)malloc(params->M * params->K * sizeof(float));
    if (weight_f32 == nullptr) {
        marmot_tensor_destroy(input);
        marmot_tensor_destroy(weight);
        marmot_tensor_destroy(output);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    generate_random_f32(weight_f32, params->M * params->K, 42);

    marmot_tensor_t *weight_src = marmot_tensor_create(nullptr, weight_shape, 2, MARMOT_DTYPE_FLOAT32);
    if (weight_src == nullptr) {
        free(weight_f32);
        marmot_tensor_destroy(input);
        marmot_tensor_destroy(weight);
        marmot_tensor_destroy(output);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    marmot_error_t err = marmot_tensor_copy_from_host_buffer(
        ctx, weight_src, weight_f32, params->M * params->K * sizeof(float)
    );
    if (err != MARMOT_SUCCESS) {
        free(weight_f32);
        marmot_tensor_destroy(input);
        marmot_tensor_destroy(weight);
        marmot_tensor_destroy(output);
        marmot_tensor_destroy(weight_src);
        return err;
    }

    err = quantize_fn(ctx, weight_src, weight);
    marmot_tensor_destroy(weight_src);
    free(weight_f32);
    if (err != MARMOT_SUCCESS) {
        marmot_tensor_destroy(input);
        marmot_tensor_destroy(weight);
        marmot_tensor_destroy(output);
        return err;
    }

    float *input_f32 = (float *)malloc(params->N * params->K * sizeof(float));
    if (input_f32 == nullptr) {
        marmot_tensor_destroy(input);
        marmot_tensor_destroy(weight);
        marmot_tensor_destroy(output);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    generate_random_f32(input_f32, params->N * params->K, 123);

    err = marmot_tensor_copy_from_host_buffer(ctx, input, input_f32, params->N * params->K * sizeof(float));
    free(input_f32);
    if (err != MARMOT_SUCCESS) {
        marmot_tensor_destroy(input);
        marmot_tensor_destroy(weight);
        marmot_tensor_destroy(output);
        return err;
    }

    for (uint32_t i = 0; i < params->warmup_iters; ++i) {
        err = marmot_linear(ctx, input, weight, nullptr, output);
        if (err != MARMOT_SUCCESS) {
            marmot_tensor_destroy(input);
            marmot_tensor_destroy(weight);
            marmot_tensor_destroy(output);
            return err;
        }
    }

    double *samples = (double *)malloc(params->bench_iters * sizeof(double));
    if (samples == nullptr) {
        marmot_tensor_destroy(input);
        marmot_tensor_destroy(weight);
        marmot_tensor_destroy(output);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    for (uint32_t i = 0; i < params->bench_iters; ++i) {
        double t0 = get_time_us();
        err = marmot_linear(ctx, input, weight, nullptr, output);
        double t1 = get_time_us();
        if (err != MARMOT_SUCCESS) {
            free(samples);
            marmot_tensor_destroy(input);
            marmot_tensor_destroy(weight);
            marmot_tensor_destroy(output);
            return err;
        }
        samples[i] = t1 - t0;
    }

    compute_stats(samples, params->bench_iters, stats);
    free(samples);

    marmot_tensor_destroy(input);
    marmot_tensor_destroy(weight);
    marmot_tensor_destroy(output);
    return MARMOT_SUCCESS;
}

static void print_header(void) {
    printf("\n");
    printf("=============================================================================\n");
    printf("  MR/NR Tile Sweep Benchmark for Non-K Quantized Matmul\n");
    printf("=============================================================================\n");
    printf("\n");
    printf("Testing blocked microkernel (MR=4, NR=8, TILE_COLS=16) vs baseline\n");
    printf("Non-K quants: Q8_0, Q8_1, Q4_0, Q4_1, Q5_0, Q5_1\n");
    printf("\n");
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    printf("Platform: aarch64 with NEON dotprod\n");
#elif defined(__aarch64__)
    printf("Platform: aarch64 (no dotprod)\n");
#else
    printf("Platform: non-ARM (scalar fallback)\n");
#endif
    printf("\n");
}

static void print_result_row(
    const char *quant_str,
    uint32_t M, uint32_t N, uint32_t K,
    const tile_bench_stats_t *stats
) {
    double gflops = (2.0 * M * N * K) / (stats->mean_us * 1e3);
    printf("  %-6s  %4u x %4u x %4u   %10.1f us  %8.2f GFLOPS  (min=%.1f, max=%.1f, std=%.1f)\n",
           quant_str, M, N, K, stats->mean_us, gflops, stats->min_us, stats->max_us, stats->stddev_us);
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    print_header();

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    if (ctx == nullptr) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    static const marmot_quant_kind_t non_k_quants[] = {
        MARMOT_QUANT_KIND_Q8_0,
        MARMOT_QUANT_KIND_Q4_0,
        MARMOT_QUANT_KIND_Q5_0,
    };
    static const size_t num_quants = sizeof(non_k_quants) / sizeof(non_k_quants[0]);

    static const uint32_t shapes[][3] = {
        // M,    N,     K     (weight MxK, input NxK, output NxM)
        // GEMV cases (N=1) - memory bound, tests single-row path
        {4096,  1,    4096},
        {4096,  1,    2048},
        // Small batch (N=2-4) - transition zone, blocked should help
        {4096,  2,    4096},
        {4096,  4,    4096},
        // Medium batch (N=8-16) - blocked microkernel sweet spot
        {4096,  8,    4096},
        {4096,  16,   4096},
        // Larger batch
        {4096,  32,   4096},
        // Different M sizes
        {512,   8,    4096},
        {256,   8,    4096},
        {128,   8,    4096},
        // LLM-like shapes
        {4096,  8,    4096},    // Hidden projection
        {11008, 8,    4096},    // FFN up
        {4096,  8,    11008},   // FFN down
    };
    static const size_t num_shapes = sizeof(shapes) / sizeof(shapes[0]);

    tile_bench_params_t params = {
        .warmup_iters = 5,
        .bench_iters = 20,
    };

    printf("Benchmark Configuration:\n");
    printf("  Warmup iterations: %u\n", params.warmup_iters);
    printf("  Bench iterations:  %u\n", params.bench_iters);
    printf("\n");

    for (size_t q = 0; q < num_quants; ++q) {
        params.quant_kind = non_k_quants[q];
        const char *qstr = quant_kind_str(params.quant_kind);

        printf("-------------------------------------------------------------------------\n");
        printf(" %s Results:\n", qstr);
        printf("-------------------------------------------------------------------------\n");
        printf("  %-6s  %17s   %12s  %14s\n", "Quant", "M x N x K", "Time", "Throughput");
        printf("  %-6s  %17s   %12s  %14s\n", "------", "-----------------", "------------", "--------------");

        for (size_t s = 0; s < num_shapes; ++s) {
            params.M = shapes[s][0];
            params.N = shapes[s][1];
            params.K = shapes[s][2];

            tile_bench_stats_t stats;
            marmot_error_t err = run_single_config(ctx, &params, &stats);
            if (err != MARMOT_SUCCESS) {
                printf("  %-6s  %4u x %4u x %4u   ERROR: %d\n", qstr, params.M, params.N, params.K, err);
                continue;
            }

            print_result_row(qstr, params.M, params.N, params.K, &stats);
        }
        printf("\n");
    }

    printf("=============================================================================\n");
    printf("  Analysis\n");
    printf("=============================================================================\n");
    printf("\n");
    printf("Current blocking parameters:\n");
    printf("  MR=8 (rows per microkernel), NR=8 (cols per microkernel), TILE_COLS=16\n");
    printf("\n");
    printf("Dispatch rules:\n");
    printf("  - N=1 (GEMV): Uses per-row path (no blocking)\n");
    printf("  - N>=2: Uses blocked MR x NR microkernel\n");
    printf("\n");
    printf("Key observations:\n");
    printf("  - If N=8 time / N=1 time is ~5-6x (not 8x), blocking helps cache reuse\n");
    printf("  - Speedup comes from loading weights once for multiple activations\n");
    printf("  - MR=8 uses 16 NEON registers for accumulators (64 floats)\n");
    printf("  - NR=8 processes 8 activation columns in the inner loop\n");
    printf("\n");
    printf("MR/NR tuning considerations:\n");
    printf("  - MR: Limited by registers (32 NEON regs, need accum + weight + activation)\n");
    printf("  - NR: Should be <= TILE_COLS (16), affects activation cache footprint\n");
    printf("  - Larger MR: Better weight reuse, but more register pressure\n");
    printf("  - Larger NR: Better activation locality, but larger working set\n");
    printf("\n");
    printf("To sweep MR/NR values, rebuild with different constants:\n");
    printf("  #define CPU_QUANT_MATMUL_MR 8u  // try 2, 4, 6, 8\n");
    printf("  #define CPU_QUANT_MATMUL_NR 8u  // try 4, 8, 12, 16\n");
    printf("\n");

    marmot_destroy(ctx);
    return 0;
}
