#include "marmot/ops/matmul.h"
#include "marmot/ops/quantization.h"

#include <stdlib.h>

#include <math.h>
#include <string.h>
#include <time.h>

#include "backend/test_backend_utils.h"

typedef struct {
    size_t N;
    size_t K;
    size_t M;
    marmot_quant_kind_t kind;
    const char *name;
} kquant_case_t;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_data(float *dst, size_t len, uint32_t seed) {
    uint32_t state = seed;
    for (size_t i = 0; i < len; ++i) {
        state = state * 1664525u + 1013904223u;
        uint32_t bits = 0x3f000000u | ((state >> 9) & 0x007fffffu);
        float val = 0.0f;
        memcpy(&val, &bits, sizeof(val));
        val -= 1.0f;
        dst[i] = val * 0.5f;
    }
}

static marmot_error_t quantize_weight(
    const marmot_context_t *ctx, marmot_quant_kind_t kind, const marmot_tensor_t *src, marmot_tensor_t *dst
) {
    switch (kind) {
    case MARMOT_QUANT_KIND_Q4_0:
        return marmot_quantize_q4_0(ctx, src, dst);
    case MARMOT_QUANT_KIND_Q4_1:
        return marmot_quantize_q4_1(ctx, src, dst);
    case MARMOT_QUANT_KIND_Q5_0:
        return marmot_quantize_q5_0(ctx, src, dst);
    case MARMOT_QUANT_KIND_Q5_1:
        return marmot_quantize_q5_1(ctx, src, dst);
    case MARMOT_QUANT_KIND_Q8_0:
        return marmot_quantize_q8_0(ctx, src, dst);
    case MARMOT_QUANT_KIND_Q4_K:
        return marmot_quantize_q4_k(ctx, src, dst);
    case MARMOT_QUANT_KIND_Q5_K:
        return marmot_quantize_q5_k(ctx, src, dst);
    case MARMOT_QUANT_KIND_Q6_K:
        return marmot_quantize_q6_k(ctx, src, dst);
    case MARMOT_QUANT_KIND_Q8_K:
        return marmot_quantize_q8_k(ctx, src, dst);
    default:
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
}

static marmot_error_t dequantize_weight(
    const marmot_context_t *ctx, marmot_quant_kind_t kind, const marmot_tensor_t *src, marmot_tensor_t *dst
) {
    switch (kind) {
    case MARMOT_QUANT_KIND_Q4_0:
        return marmot_dequantize_q4_0(ctx, src, dst);
    case MARMOT_QUANT_KIND_Q4_1:
        return marmot_dequantize_q4_1(ctx, src, dst);
    case MARMOT_QUANT_KIND_Q5_0:
        return marmot_dequantize_q5_0(ctx, src, dst);
    case MARMOT_QUANT_KIND_Q5_1:
        return marmot_dequantize_q5_1(ctx, src, dst);
    case MARMOT_QUANT_KIND_Q8_0:
        return marmot_dequantize_q8_0(ctx, src, dst);
    case MARMOT_QUANT_KIND_Q4_K:
        return marmot_dequantize_q4_k(ctx, src, dst);
    case MARMOT_QUANT_KIND_Q5_K:
        return marmot_dequantize_q5_k(ctx, src, dst);
    case MARMOT_QUANT_KIND_Q6_K:
        return marmot_dequantize_q6_k(ctx, src, dst);
    case MARMOT_QUANT_KIND_Q8_K:
        return marmot_dequantize_q8_k(ctx, src, dst);
    default:
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
}

static void run_kquant_case(marmot_test_env_t *env, marmot_context_t *cpu_ctx, const kquant_case_t *tc, bool log_perf) {
    size_t shape_input[] = {tc->N, tc->K};
    size_t shape_weight[] = {tc->M, tc->K};
    size_t shape_output[] = {tc->N, tc->M};

    const size_t input_elems = tc->N * tc->K;
    const size_t weight_elems = tc->M * tc->K;
    float *input_data = (float *)malloc(input_elems * sizeof(float));
    float *weight_data = (float *)malloc(weight_elems * sizeof(float));
    assert_non_null(input_data);
    assert_non_null(weight_data);
    fill_data(input_data, input_elems, 1234u);
    fill_data(weight_data, weight_elems, 42u);

    marmot_tensor_t *input_cpu = marmot_tensor_create(nullptr, shape_input, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *weight_src_cpu = marmot_tensor_create(nullptr, shape_weight, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *weight_q_cpu = marmot_tensor_create_quantized(nullptr, shape_weight, 2, tc->kind);
    marmot_tensor_t *weight_deq_cpu = marmot_tensor_create(nullptr, shape_weight, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output_cpu = marmot_tensor_create(nullptr, shape_output, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(input_cpu);
    assert_non_null(weight_src_cpu);
    assert_non_null(weight_q_cpu);
    assert_non_null(weight_deq_cpu);
    assert_non_null(output_cpu);

    const size_t input_bytes = input_elems * sizeof(float);
    const size_t weight_bytes = weight_elems * sizeof(float);
    assert_int_equal(marmot_tensor_copy_from_host_buffer(cpu_ctx, input_cpu, input_data, input_bytes), MARMOT_SUCCESS);
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(cpu_ctx, weight_src_cpu, weight_data, weight_bytes), MARMOT_SUCCESS
    );
    assert_int_equal(quantize_weight(cpu_ctx, tc->kind, weight_src_cpu, weight_q_cpu), MARMOT_SUCCESS);
    assert_int_equal(dequantize_weight(cpu_ctx, tc->kind, weight_q_cpu, weight_deq_cpu), MARMOT_SUCCESS);

    marmot_error_t err = marmot_linear(cpu_ctx, input_cpu, weight_deq_cpu, nullptr, output_cpu);
    assert_int_equal(err, MARMOT_SUCCESS);

    const size_t weight_q_bytes = marmot_tensor_size_bytes(weight_q_cpu);
    float *cpu_out = (float *)malloc(shape_output[0] * shape_output[1] * sizeof(float));
    void *weight_q_host = malloc(weight_q_bytes);
    assert_non_null(cpu_out);
    assert_non_null(weight_q_host);
    assert_int_equal(
        marmot_tensor_copy_to_host_buffer(
            cpu_ctx, output_cpu, cpu_out, shape_output[0] * shape_output[1] * sizeof(float)
        ),
        MARMOT_SUCCESS
    );
    assert_int_equal(
        marmot_tensor_copy_to_host_buffer(cpu_ctx, weight_q_cpu, weight_q_host, weight_q_bytes), MARMOT_SUCCESS
    );

    marmot_tensor_t *input_metal = marmot_tensor_create(env->ctx, shape_input, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *weight_metal = marmot_tensor_create_quantized(env->ctx, shape_weight, 2, tc->kind);
    marmot_tensor_t *output_metal = marmot_tensor_create(env->ctx, shape_output, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(input_metal);
    assert_non_null(weight_metal);
    assert_non_null(output_metal);

    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, input_metal, input_data, input_bytes), MARMOT_SUCCESS
    );
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, weight_metal, weight_q_host, weight_q_bytes), MARMOT_SUCCESS
    );

    // Warmup runs - small batches need more iterations to reach steady-state clocks.
    const int base_warmup_iters = (tc->N <= 8) ? 50 : 10;
    int warmup_iters = base_warmup_iters;
    if (tc->N <= 8) {
        warmup_iters = 200;
    } else if (tc->N <= 32) {
        warmup_iters = 50;
    }
    for (int i = 0; i < warmup_iters; i++) {
        err = marmot_linear(env->ctx, input_metal, weight_metal, nullptr, output_metal);
        assert_int_equal(err, MARMOT_SUCCESS);
    }
    // Sync after warmup to ensure GPU is fully warmed up
    assert_int_equal(marmot_device_synchronize(env->ctx), MARMOT_SUCCESS);

    // Scale iterations based on batch size to match llama.cpp methodology:
    // - llama.cpp runs 7000+ iterations for N=1, ~100 for N=512
    // - More iterations for small batches amortizes dispatch overhead
    int num_iters;
    if (tc->N <= 1) {
        num_iters = 2000; // Small MV ops need many iterations
    } else if (tc->N <= 8) {
        num_iters = 1000;
    } else if (tc->N <= 32) {
        num_iters = 500;
    } else {
        num_iters = 200;
    }
    const int total_iters = base_warmup_iters + num_iters;
    num_iters = total_iters - warmup_iters;

    // Timed runs - dispatch all work then sync once at end
    double t0 = now_seconds();
    for (int iter = 0; iter < num_iters; ++iter) {
        err = marmot_linear(env->ctx, input_metal, weight_metal, nullptr, output_metal);
        assert_int_equal(err, MARMOT_SUCCESS);
    }
    // Sync INSIDE timing window to measure actual GPU execution time
    assert_int_equal(marmot_device_synchronize(env->ctx), MARMOT_SUCCESS);
    double t1 = now_seconds();
    double avg_time = (t1 - t0) / num_iters;

    float *metal_out = (float *)malloc(shape_output[0] * shape_output[1] * sizeof(float));
    assert_non_null(metal_out);
    assert_int_equal(
        marmot_tensor_copy_to_host_buffer(
            env->ctx, output_metal, metal_out, shape_output[0] * shape_output[1] * sizeof(float)
        ),
        MARMOT_SUCCESS
    );

    float max_diff = 0.0f;
    float max_allowed = 0.0f;
    size_t max_idx = 0;
    float cpu_max = 0.0f;
    float metal_max = 0.0f;
    // Use tighter tolerance for K-quants (Q4_K, Q5_K, Q6_K, Q8_K), looser for basic quants
    const bool is_k_quant = (tc->kind >= MARMOT_QUANT_KIND_Q4_K && tc->kind <= MARMOT_QUANT_KIND_Q8_K);
    const float rel_tol = is_k_quant ? 1e-4f : 1e-3f;
    const float abs_tol = is_k_quant ? 1e-6f : 0.1f;
    for (size_t i = 0; i < shape_output[0] * shape_output[1]; ++i) {
        const float ref = cpu_out[i];
        const float got = metal_out[i];
        const float diff = fabsf(ref - got);
        const float allowed = fabsf(ref) * rel_tol + abs_tol;
        cpu_max = fmaxf(cpu_max, fabsf(ref));
        metal_max = fmaxf(metal_max, fabsf(got));
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
        if (allowed > max_allowed) {
            max_allowed = allowed;
        }
    }
    if (max_diff > max_allowed) {
        printf(
            "[metal kquant %s] tolerance exceeded: max_diff=%.6g allowed=%.6g N=%zu K=%zu M=%zu kind=%d\n", tc->name,
            max_diff, max_allowed, tc->N, tc->K, tc->M, (int)tc->kind
        );
        const size_t row = max_idx / tc->M;
        const size_t col = max_idx % tc->M;
        printf(
            "  worst idx (%zu,%zu): cpu=%.6g metal=%.6g diff=%.6g\n", row, col, cpu_out[max_idx], metal_out[max_idx],
            max_diff
        );
        printf("  cpu_max=%.6g metal_max=%.6g\n", cpu_max, metal_max);
    }
    assert_true(max_diff <= max_allowed);

    if (log_perf) {
        const double time_us = avg_time * 1e6;
        const double flops = 2.0 * (double)tc->N * (double)tc->K * (double)tc->M;
        const double gflops = flops / (avg_time * 1e9);
        const double tflops = gflops / 1000.0;
        printf(
            "[metal kquant %s] N=%zu M=%zu K=%zu | %.1f us | %.2f GFLOPS (%.3f TFLOPS) | err=%.3g\n", tc->name, tc->N,
            tc->M, tc->K, time_us, gflops, tflops, max_diff
        );
    }

    free(metal_out);
    marmot_tensor_destroy(output_metal);
    marmot_tensor_destroy(weight_metal);
    marmot_tensor_destroy(input_metal);
    free(weight_q_host);
    free(cpu_out);
    marmot_tensor_destroy(output_cpu);
    marmot_tensor_destroy(weight_deq_cpu);
    marmot_tensor_destroy(weight_q_cpu);
    marmot_tensor_destroy(weight_src_cpu);
    marmot_tensor_destroy(input_cpu);
    free(weight_data);
    free(input_data);
}

static void test_metal_quant_optimized(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    assert_non_null(env);
    if (env->backend != MARMOT_BACKEND_METAL) {
        skip();
        return;
    }

    marmot_context_t *cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
    if (cpu_ctx == nullptr) {
        skip();
        return;
    }

    marmot_context_set_quant_activation_mode(env->ctx, MARMOT_QUANT_ACTIVATION_FORCE_DIRECT);
    marmot_context_set_quant_activation_mode(cpu_ctx, MARMOT_QUANT_ACTIVATION_FORCE_DIRECT);

    const kquant_case_t accuracy_cases[] = {
        {.N = 8, .K = 1024, .M = 256, .kind = MARMOT_QUANT_KIND_Q4_K, .name = "accuracy_q4k"},
        {.N = 8, .K = 1024, .M = 256, .kind = MARMOT_QUANT_KIND_Q5_K, .name = "accuracy_q5k"},
        {.N = 8, .K = 1024, .M = 256, .kind = MARMOT_QUANT_KIND_Q6_K, .name = "accuracy_q6k"},
        {.N = 8, .K = 1024, .M = 256, .kind = MARMOT_QUANT_KIND_Q8_K, .name = "accuracy_q8k"},
        {.N = 4, .K = 128, .M = 64, .kind = MARMOT_QUANT_KIND_Q4_K, .name = "small_k_q4k"},
    };

    // Shapes aligned with llama.cpp test-backend-ops for direct comparison
    // llama.cpp: MUL_MAT(type_a=q4_K,type_b=f32,m=4096,n=X,k=14336)
    // marmot:    N=batch, K=input_features, M=output_features
    const kquant_case_t perf_cases[] = {
        // === FFN shapes (m=4096, k=14336) - All quant types at N=512 ===
        {.N = 512, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_0, .name = "ffn_n512_q4_0"},
        {.N = 512, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_1, .name = "ffn_n512_q4_1"},
        {.N = 512, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_0, .name = "ffn_n512_q5_0"},
        {.N = 512, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_1, .name = "ffn_n512_q5_1"},
        {.N = 512, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q8_0, .name = "ffn_n512_q8_0"},
        {.N = 512, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_K, .name = "ffn_n512_q4_k"},
        {.N = 512, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_K, .name = "ffn_n512_q5_k"},
        {.N = 512, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q6_K, .name = "ffn_n512_q6_k"},
        {.N = 512, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q8_K, .name = "ffn_n512_q8_k"},
        // === FFN shapes - All quant types at N=32 ===
        {.N = 32, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_0, .name = "ffn_n32_q4_0"},
        {.N = 32, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_1, .name = "ffn_n32_q4_1"},
        {.N = 32, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_0, .name = "ffn_n32_q5_0"},
        {.N = 32, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_1, .name = "ffn_n32_q5_1"},
        {.N = 32, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q8_0, .name = "ffn_n32_q8_0"},
        {.N = 32, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_K, .name = "ffn_n32_q4_k"},
        {.N = 32, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_K, .name = "ffn_n32_q5_k"},
        {.N = 32, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q6_K, .name = "ffn_n32_q6_k"},
        {.N = 32, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q8_K, .name = "ffn_n32_q8_k"},
        // === FFN shapes - MM16 tile (K-quants only) ===
        {.N = 16, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_K, .name = "ffn_n16_q4_k"},
        {.N = 16, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_K, .name = "ffn_n16_q5_k"},
        {.N = 16, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q6_K, .name = "ffn_n16_q6_k"},
        {.N = 16, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q8_K, .name = "ffn_n16_q8_k"},
        {.N = 9, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_K, .name = "ffn_n9_q4_k"},
        {.N = 9, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_K, .name = "ffn_n9_q5_k"},
        {.N = 9, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q6_K, .name = "ffn_n9_q6_k"},
        {.N = 9, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q8_K, .name = "ffn_n9_q8_k"},
        // === FFN shapes - All quant types at N=8 ===
        {.N = 8, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_0, .name = "ffn_n8_q4_0"},
        {.N = 8, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_1, .name = "ffn_n8_q4_1"},
        {.N = 8, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_0, .name = "ffn_n8_q5_0"},
        {.N = 8, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_1, .name = "ffn_n8_q5_1"},
        {.N = 8, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q8_0, .name = "ffn_n8_q8_0"},
        {.N = 8, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_K, .name = "ffn_n8_q4_k"},
        {.N = 8, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_K, .name = "ffn_n8_q5_k"},
        {.N = 8, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q6_K, .name = "ffn_n8_q6_k"},
        {.N = 8, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q8_K, .name = "ffn_n8_q8_k"},
        // === FFN shapes - All quant types at N=1 ===
        {.N = 1, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_0, .name = "ffn_n1_q4_0"},
        {.N = 1, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_1, .name = "ffn_n1_q4_1"},
        {.N = 1, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_0, .name = "ffn_n1_q5_0"},
        {.N = 1, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_1, .name = "ffn_n1_q5_1"},
        {.N = 1, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q8_0, .name = "ffn_n1_q8_0"},
        {.N = 1, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_K, .name = "ffn_n1_q4_k"},
        {.N = 1, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_K, .name = "ffn_n1_q5_k"},
        {.N = 1, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q6_K, .name = "ffn_n1_q6_k"},
        {.N = 1, .K = 14336, .M = 4096, .kind = MARMOT_QUANT_KIND_Q8_K, .name = "ffn_n1_q8_k"},
        // === QKV shapes (m=4096, k=4096) - All quant types at N=512 ===
        {.N = 512, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_0, .name = "qkv_n512_q4_0"},
        {.N = 512, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_1, .name = "qkv_n512_q4_1"},
        {.N = 512, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_0, .name = "qkv_n512_q5_0"},
        {.N = 512, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_1, .name = "qkv_n512_q5_1"},
        {.N = 512, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q8_0, .name = "qkv_n512_q8_0"},
        {.N = 512, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_K, .name = "qkv_n512_q4_k"},
        {.N = 512, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_K, .name = "qkv_n512_q5_k"},
        {.N = 512, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q6_K, .name = "qkv_n512_q6_k"},
        {.N = 512, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q8_K, .name = "qkv_n512_q8_k"},
        // === QKV shapes - All quant types at N=32 ===
        {.N = 32, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_0, .name = "qkv_n32_q4_0"},
        {.N = 32, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_1, .name = "qkv_n32_q4_1"},
        {.N = 32, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_0, .name = "qkv_n32_q5_0"},
        {.N = 32, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_1, .name = "qkv_n32_q5_1"},
        {.N = 32, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q8_0, .name = "qkv_n32_q8_0"},
        {.N = 32, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_K, .name = "qkv_n32_q4_k"},
        {.N = 32, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_K, .name = "qkv_n32_q5_k"},
        {.N = 32, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q6_K, .name = "qkv_n32_q6_k"},
        {.N = 32, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q8_K, .name = "qkv_n32_q8_k"},
        // === QKV shapes - MM16 tile (K-quants only) ===
        {.N = 16, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_K, .name = "qkv_n16_q4_k"},
        {.N = 16, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_K, .name = "qkv_n16_q5_k"},
        {.N = 16, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q6_K, .name = "qkv_n16_q6_k"},
        {.N = 16, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q8_K, .name = "qkv_n16_q8_k"},
        {.N = 9, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q4_K, .name = "qkv_n9_q4_k"},
        {.N = 9, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q5_K, .name = "qkv_n9_q5_k"},
        {.N = 9, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q6_K, .name = "qkv_n9_q6_k"},
        {.N = 9, .K = 4096, .M = 4096, .kind = MARMOT_QUANT_KIND_Q8_K, .name = "qkv_n9_q8_k"},
    };

    for (size_t i = 0; i < sizeof(accuracy_cases) / sizeof(accuracy_cases[0]); ++i) {
        run_kquant_case(env, cpu_ctx, &accuracy_cases[i], false);
    }

    const char *bench_env = getenv("MARMOT_RUN_METAL_QK_BENCH");
    if (bench_env != nullptr && bench_env[0] != '\0') {
        for (size_t i = 0; i < sizeof(perf_cases) / sizeof(perf_cases[0]); ++i) {
            run_kquant_case(env, cpu_ctx, &perf_cases[i], true);
        }
    }

    marmot_destroy(cpu_ctx);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(
            test_metal_quant_optimized, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
