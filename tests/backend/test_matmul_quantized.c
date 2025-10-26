// Test quantized matmul: Q4_0 weights × FP32 activations
#include "marmot/device.h"
#include "marmot/ops/unary.h"
#include "marmot/quant_block.h"

#include <stdbool.h>
#include <stdlib.h>

#include <assert.h>
#include <math.h>
#include <string.h>

#include "backend/golden_matmul_llama.h"
#include "backend/golden_quant_llama.h"
#include "backend/test_backend_utils.h"
#include "matmul_quantized_golden_cases.h"
#include "utils/dtype_ref.h"

static void matmul_quantized_expect_relu(float *data, size_t rows, size_t cols, const float *bias) {
    for (size_t r = 0; r < rows; ++r) {
        float *row = data + r * cols;
        for (size_t c = 0; c < cols; ++c) {
            const float val = row[c] + bias[c];
            row[c] = val > 0.0f ? val : 0.0f;
        }
    }
}

// Small test case for fast iteration
// Linear convention: input(N×K) @ weight(M×K).T = output(N×M)
// NOTE: K=64 (2 blocks) has limited statistical averaging
// Expected error: ~15-20% due to double quantization + limited averaging
#define TEST_SMALL_N 2
#define TEST_SMALL_K 64
#define TEST_SMALL_M 3

// Medium test case for development validation
// K=256 (8 blocks) provides moderate averaging
// Expected error: ~8-12% for Q4_0×Q8_0 double quantization
#define TEST_MEDIUM_N 32
#define TEST_MEDIUM_K 256
#define TEST_MEDIUM_M 32

// Production-scale test case for validating llama.cpp-level accuracy
// K=4096 (128 blocks) matches real LLM dimensions
// Expected error: ~3-4% converging to llama.cpp perplexity benchmarks
#define TEST_PROD_N 64
#define TEST_PROD_K 4096
#define TEST_PROD_M 128

static void generate_test_input(float *input, int N, int K) {
    // Simple deterministic pattern
    for (int i = 0; i < N * K; i++) {
        input[i] = (float)(i % 17) * 0.1f - 0.8f;
    }
}

static void generate_test_weight(float *weight, int M, int K) {
    // Varied pattern for weights
    for (int i = 0; i < M * K; i++) {
        int row = i / K;
        int col = i % K;
        weight[i] = (float)((row * 13 + col * 7) % 23) * 0.2f - 2.0f;
    }
}

static void run_matmul_golden_case(const marmot_test_env_t *env, const matmul_golden_case_t *tc, bool fp16_input) {
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(tc->kind);
    assert_non_null(traits);

    size_t shape_input[] = {tc->N, tc->K};
    size_t shape_weight[] = {tc->M, tc->K};
    size_t shape_output[] = {tc->N, tc->M};

    marmot_tensor_t *input = nullptr;
    if (fp16_input) {
        input = marmot_tensor_create(env->ctx, shape_input, 2, MARMOT_DTYPE_FLOAT16);
        assert_non_null(input);
        const size_t bytes = tc->N * tc->K * sizeof(uint16_t);
        assert_int_equal(marmot_tensor_copy_from_host_buffer(env->ctx, input, tc->input_f16, bytes), MARMOT_SUCCESS);
    } else {
        input = marmot_test_tensor_from_array(env, shape_input, 2, tc->input_f32);
    }

    marmot_tensor_t *weight = marmot_tensor_create_quantized(env->ctx, shape_weight, 2, tc->kind);
    assert_non_null(weight);

    const size_t expected_bytes = marmot_tensor_size_bytes(weight);
    assert_int_equal(expected_bytes, tc->weight_bytes);
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, weight, tc->weights, tc->weight_bytes), MARMOT_SUCCESS
    );

    marmot_tensor_t *output = marmot_tensor_create(env->ctx, shape_output, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(output);

    marmot_error_t err = marmot_linear(env->ctx, input, weight, nullptr, output);
    if (err != MARMOT_SUCCESS) {
        printf(
            "Matmul golden \"%s\" (%s input) failed: %s\n", tc->name, fp16_input ? "fp16" : "fp32",
            marmot_error_string(err)
        );
    }
    assert_int_equal(err, MARMOT_SUCCESS);

    float *result = (float *)malloc(tc->N * tc->M * sizeof(float));
    assert_non_null(result);
    assert_int_equal(
        marmot_tensor_copy_to_host_buffer(env->ctx, output, result, tc->N * tc->M * sizeof(float)), MARMOT_SUCCESS
    );

    const float *expected = fp16_input ? tc->expected_f16 : tc->expected_f32;
    const float abs_tol = 2e-5f;
    const float rel_tol = 5e-5f;
    for (size_t i = 0; i < tc->N * tc->M; ++i) {
        const float diff = fabsf(result[i] - expected[i]);
        const float allowed = fabsf(expected[i]) * rel_tol + abs_tol;
        if (diff > allowed) {
            printf(
                "Matmul golden mismatch [%s %s input] idx=%zu expected=%.9f got=%.9f diff=%.9g tol=%.9g\n", tc->name,
                fp16_input ? "fp16" : "fp32", i, expected[i], result[i], diff, allowed
            );
        }
        assert_true(diff <= allowed);
    }

    free(result);
    marmot_test_tensor_destroy_all(3, output, weight, input);
}

static void matmul_llama_goldens_suite(marmot_test_env_t *env) {
    for (size_t i = 0; i < sizeof(g_matmul_quant_goldens) / sizeof(g_matmul_quant_goldens[0]); ++i) {
        run_matmul_golden_case(env, &g_matmul_quant_goldens[i], false);
        run_matmul_golden_case(env, &g_matmul_quant_goldens[i], true);
    }
}

static void test_matmul_llama_goldens(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
#if MARMOT_TEST_HAS_CPU_INTERNALS
    if (env->backend != MARMOT_BACKEND_CPU) {
        marmot_test_run_with_cpu_scalar(env, matmul_llama_goldens_suite);
        return;
    }
#endif
    matmul_llama_goldens_suite(env);
}

static inline float rand_uniform_open(void) {
    // (0,1) to avoid log(0) in Box-Muller
    return ((float)(rand() + 1)) / ((float)RAND_MAX + 2.0f);
}

static void generate_gaussian(float *data, int len, float sigma) {
    for (int i = 0; i < len; i += 2) {
        float u1 = rand_uniform_open();
        float u2 = rand_uniform_open();
        float r = sqrtf(-2.0f * logf(u1));
        float t = 6.283185307179586f * u2;
        float z0 = r * cosf(t) * sigma;
        float z1 = r * sinf(t) * sigma;
        data[i] = z0;
        if (i + 1 < len)
            data[i + 1] = z1;
    }
}

static void generate_gaussian_input(float *input, int N, int K) {
    generate_gaussian(input, N * K, 1.0f);
}

static void generate_gaussian_weight(float *weight, int M, int K) {
    generate_gaussian(weight, M * K, 0.5f);
}

// llama.cpp-compatible reference quantizers (minimal subset used for parity tests)
static void quantize_row_q4_0_ref(const float *x, marmot_q4_0_block_t *y, int64_t k) {
    const int qk = (int)MARMOT_QUANT_BLOCK_SIZE;
    assert(k % qk == 0);
    const int nb = (int)(k / qk);
    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f;
        float vmax = 0.0f;
        for (int j = 0; j < qk; ++j) {
            const float v = x[i * qk + j];
            const float av = fabsf(v);
            if (av > amax) {
                amax = av;
                vmax = v;
            }
        }
        const float d = vmax / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].scale = marmot_f32_to_f16_ref(d);
        for (int j = 0; j < qk / 2; ++j) {
            const float x0 = x[i * qk + j] * id;
            const float x1 = x[i * qk + qk / 2 + j] * id;
            const uint8_t xi0 =
                (uint8_t)((int8_t)(x0 + 8.5f) < 0 ? 0 : ((int8_t)(x0 + 8.5f) > 15 ? 15 : (int8_t)(x0 + 8.5f)));
            const uint8_t xi1 =
                (uint8_t)((int8_t)(x1 + 8.5f) < 0 ? 0 : ((int8_t)(x1 + 8.5f) > 15 ? 15 : (int8_t)(x1 + 8.5f)));
            y[i].qs[j] = (uint8_t)(xi0 | (xi1 << 4));
        }
    }
}

static void quantize_row_q4_1_ref(const float *x, marmot_q4_1_block_t *y, int64_t k) {
    const int qk = (int)MARMOT_QUANT_BLOCK_SIZE;
    assert(k % qk == 0);
    const int nb = (int)(k / qk);
    for (int i = 0; i < nb; ++i) {
        float vmin = x[i * qk + 0];
        float vmax = x[i * qk + 0];
        for (int j = 1; j < qk; ++j) {
            const float v = x[i * qk + j];
            if (v < vmin)
                vmin = v;
            if (v > vmax)
                vmax = v;
        }
        const float d = (vmax - vmin) / 15.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].scale = marmot_f32_to_f16_ref(d);
        y[i].min = marmot_f32_to_f16_ref(vmin);
        for (int j = 0; j < qk / 2; ++j) {
            const float x0 = (x[i * qk + j] - vmin) * id;
            const float x1 = (x[i * qk + qk / 2 + j] - vmin) * id;
            const uint8_t xi0 =
                (uint8_t)((int8_t)(x0 + 0.5f) < 0 ? 0 : ((int8_t)(x0 + 0.5f) > 15 ? 15 : (int8_t)(x0 + 0.5f)));
            const uint8_t xi1 =
                (uint8_t)((int8_t)(x1 + 0.5f) < 0 ? 0 : ((int8_t)(x1 + 0.5f) > 15 ? 15 : (int8_t)(x1 + 0.5f)));
            y[i].qs[j] = (uint8_t)(xi0 | (xi1 << 4));
        }
    }
}

// Tail-friendly wrappers for K not divisible by block size
static void quantize_row_q4_0_ref_tail(const float *row, int K, marmot_q4_0_block_t *dst) {
    const int qk = (int)MARMOT_QUANT_BLOCK_SIZE;
    const int nb = (K + qk - 1) / qk;
    const int padded = nb * qk;
    float *tmp = (float *)malloc((size_t)padded * sizeof(float));
    assert_non_null(tmp);
    for (int i = 0; i < K; ++i)
        tmp[i] = row[i];
    for (int i = K; i < padded; ++i)
        tmp[i] = 0.0f;
    quantize_row_q4_0_ref(tmp, dst, (int64_t)padded);
    free(tmp);
}

static void assert_q4_0_blocks_match_reference(const float *weights, const marmot_q4_0_block_t *blocks, int M, int K) {
    const int row_blocks = (K + (int)MARMOT_QUANT_BLOCK_SIZE - 1) / (int)MARMOT_QUANT_BLOCK_SIZE;
    marmot_q4_0_block_t *ref = (marmot_q4_0_block_t *)malloc((size_t)row_blocks * sizeof(marmot_q4_0_block_t));
    assert_non_null(ref);
    for (int m = 0; m < M; ++m) {
        const float *row = weights + (size_t)m * (size_t)K;
        quantize_row_q4_0_ref_tail(row, K, ref);
        for (int b = 0; b < row_blocks; ++b) {
            const size_t idx = (size_t)m * (size_t)row_blocks + (size_t)b;
            const marmot_q4_0_block_t *blk = &blocks[idx];
            const float blk_scale = (float)marmot_float16_to_native(blk->scale);
            const float ref_scale = (float)marmot_float16_to_native(ref[b].scale);
            if (memcmp(blk->qs, ref[b].qs, sizeof(blk->qs)) != 0 || fabsf(blk_scale - ref_scale) > 5e-4f) {
                fail_msg("Q4_0 block mismatch at row %d block %d (idx=%zu)", m, b, idx);
            }
        }
    }
    free(ref);
}

static void quantize_row_q5_0_ref(const float *x, marmot_q5_0_block_t *y, int64_t k) {
    const int qk = (int)MARMOT_QUANT_BLOCK_SIZE;
    assert(k % qk == 0);
    const int nb = (int)(k / qk);
    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f, vmax = 0.0f;
        for (int j = 0; j < qk; ++j) {
            const float v = x[i * qk + j];
            const float av = fabsf(v);
            if (av > amax) {
                amax = av;
                vmax = v;
            }
        }
        const float d = vmax / -16.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].scale = marmot_f32_to_f16_ref(d);
        uint32_t qh = 0;
        for (int j = 0; j < qk / 2; ++j) {
            const float x0 = x[i * qk + j] * id;
            const float x1 = x[i * qk + qk / 2 + j] * id;
            int32_t xi0 = (int32_t)(x0 + 16.5f);
            int32_t xi1 = (int32_t)(x1 + 16.5f);
            if (xi0 < 0)
                xi0 = 0;
            else if (xi0 > 31)
                xi0 = 31;
            if (xi1 < 0)
                xi1 = 0;
            else if (xi1 > 31)
                xi1 = 31;
            y[i].qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
            qh |= (uint32_t)(((uint32_t)xi0 & 0x10u) >> 4) << (j + 0);
            qh |= (uint32_t)(((uint32_t)xi1 & 0x10u) >> 4) << (j + qk / 2);
        }
        memcpy(y[i].qh, &qh, sizeof(qh));
    }
}

static void quantize_row_q5_1_ref(const float *x, marmot_q5_1_block_t *y, int64_t k) {
    const int qk = (int)MARMOT_QUANT_BLOCK_SIZE;
    assert(k % qk == 0);
    const int nb = (int)(k / qk);
    for (int i = 0; i < nb; ++i) {
        float vmin = x[i * qk + 0];
        float vmax = x[i * qk + 0];
        for (int j = 1; j < qk; ++j) {
            const float v = x[i * qk + j];
            if (v < vmin)
                vmin = v;
            if (v > vmax)
                vmax = v;
        }
        const float d = (vmax - vmin) / 31.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].scale = marmot_f32_to_f16_ref(d);
        y[i].min = marmot_f32_to_f16_ref(vmin);
        uint32_t qh = 0;
        for (int j = 0; j < qk / 2; ++j) {
            const float x0 = (x[i * qk + j] - vmin) * id;
            const float x1 = (x[i * qk + qk / 2 + j] - vmin) * id;
            int32_t xi0 = (int32_t)roundf(x0);
            int32_t xi1 = (int32_t)roundf(x1);
            if (xi0 < 0)
                xi0 = 0;
            else if (xi0 > 31)
                xi0 = 31;
            if (xi1 < 0)
                xi1 = 0;
            else if (xi1 > 31)
                xi1 = 31;
            y[i].qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
            qh |= (uint32_t)(((uint32_t)xi0 & 0x10u) >> 4) << (j + 0);
            qh |= (uint32_t)(((uint32_t)xi1 & 0x10u) >> 4) << (j + qk / 2);
        }
        memcpy(y[i].qh, &qh, sizeof(qh));
    }
}

static void quantize_row_q8_0_ref(const float *x, marmot_q8_0_block_t *y, int64_t k) {
    const int qk = (int)MARMOT_QUANT_BLOCK_SIZE; // 32
    assert(k % qk == 0);
    const int nb = (int)(k / qk);
    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < qk; ++j) {
            const float v = x[i * qk + j];
            const float av = fabsf(v);
            if (av > amax)
                amax = av;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].scale = marmot_f32_to_f16_ref(d);
        for (int j = 0; j < qk; ++j) {
            int32_t q = (int32_t)roundf(x[i * qk + j] * id);
            if (q < -127)
                q = -127;
            else if (q > 127)
                q = 127;
            y[i].qs[j] = (int8_t)q;
        }
    }
}

static void dequantize_q5_0_blocks_to_f32(const marmot_q5_0_block_t *src, size_t len, float *dst) {
    const size_t qk = (size_t)MARMOT_QUANT_BLOCK_SIZE;
    size_t nb = (len + qk - 1) / qk;
    for (size_t b = 0; b < nb; ++b) {
        const float d = (float)marmot_float16_to_native(src[b].scale);
        uint32_t qh;
        memcpy(&qh, src[b].qh, sizeof(qh));
        const size_t base = b * qk;
        const size_t half = qk / 2;
        for (size_t j = 0; j < half; ++j) {
            const uint8_t packed = src[b].qs[j];
            const uint8_t xh0 = (uint8_t)(((qh >> (j + 0)) & 0x1u) << 4);
            const uint8_t xh1 = (uint8_t)(((qh >> (j + half)) & 0x1u) << 4);
            const int32_t x0 = ((packed & 0x0F) | xh0) - 16;
            const int32_t x1 = ((packed >> 4) | xh1) - 16;
            dst[base + j] = (float)x0 * d;
            dst[base + j + half] = (float)x1 * d;
        }
    }
}

static void dequantize_q5_1_blocks_to_f32(const marmot_q5_1_block_t *src, size_t len, float *dst) {
    const size_t qk = (size_t)MARMOT_QUANT_BLOCK_SIZE;
    size_t nb = (len + qk - 1) / qk;
    for (size_t b = 0; b < nb; ++b) {
        const float d = (float)marmot_float16_to_native(src[b].scale);
        const float m = (float)marmot_float16_to_native(src[b].min);
        uint32_t qh;
        memcpy(&qh, src[b].qh, sizeof(qh));
        const size_t base = b * qk;
        const size_t half = qk / 2;
        for (size_t j = 0; j < half; ++j) {
            const uint8_t packed = src[b].qs[j];
            const uint8_t xh0 = (uint8_t)(((qh >> (j + 0)) & 0x1u) << 4);
            const uint8_t xh1 = (uint8_t)(((qh >> (j + half)) & 0x1u) << 4);
            const int32_t x0 = (packed & 0x0F) | xh0;
            const int32_t x1 = (packed >> 4) | xh1;
            dst[base + j] = (float)x0 * d + m;
            dst[base + j + half] = (float)x1 * d + m;
        }
    }
}

static void dequantize_q8_0_blocks_to_f32(const marmot_q8_0_block_t *src, size_t len, float *dst) {
    const size_t qk = (size_t)MARMOT_QUANT_BLOCK_SIZE;
    size_t nb = (len + qk - 1) / qk;
    for (size_t b = 0; b < nb; ++b) {
        const float d = (float)marmot_float16_to_native(src[b].scale);
        const size_t base = b * qk;
        for (size_t j = 0; j < qk; ++j) {
            dst[base + j] = (float)src[b].qs[j] * d;
        }
    }
}

// Reference FP32 matmul: output = input @ weight.T
// Linear convention: input(N×K) @ weight(M×K).T = output(N×M)
static void reference_matmul_fp32(const float *input, const float *weight, float *output, int N, int K, int M) {
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += input[n * K + k] * weight[m * K + k];
            }
            output[n * M + m] = sum;
        }
    }
}

// Global relative error: sum |diff| / sum |ref|
static float compute_global_relative_error(const float *out, const float *ref, int count) {
    double sum_diff = 0.0;
    double sum_ref = 0.0;
    for (int i = 0; i < count; i++) {
        float diff = fabsf(out[i] - ref[i]);
        float ref_abs = fabsf(ref[i]);
        sum_diff += (double)diff;
        sum_ref += (double)ref_abs;
    }
    if (sum_ref < 1e-12) {
        return (float)sum_diff;
    }
    return (float)(sum_diff / sum_ref);
}

static void test_matmul_quantized_q4_0_sized(
    marmot_test_env_t *env, int N, int K, int M, float max_relative_error, float avg_relative_error,
    float max_absolute_error, int use_gaussian
) {
    // Allocate test data
    float *input = (float *)malloc(N * K * sizeof(float));
    float *weight_fp32 = (float *)malloc(M * K * sizeof(float));
    float *output_fp32 = (float *)malloc(N * M * sizeof(float));
    assert_non_null(input);
    assert_non_null(weight_fp32);
    assert_non_null(output_fp32);

    if (use_gaussian) {
        srand(42);
        generate_gaussian_input(input, N, K);
        generate_gaussian_weight(weight_fp32, M, K);
    } else {
        generate_test_input(input, N, K);
        generate_test_weight(weight_fp32, M, K);
    }

    // Compute reference FP32 matmul
    reference_matmul_fp32(input, weight_fp32, output_fp32, N, K, M);

    // Quantize weight to Q4_0
    const size_t num_blocks = (M * K + MARMOT_QUANT_BLOCK_SIZE - 1) / MARMOT_QUANT_BLOCK_SIZE;
    marmot_q4_0_block_t *weight_q4_0 = (marmot_q4_0_block_t *)malloc(num_blocks * sizeof(marmot_q4_0_block_t));
    assert_non_null(weight_q4_0);

    size_t shape_weight[] = {(size_t)M, (size_t)K};
    marmot_tensor_t *tensor_weight_fp32 = marmot_test_tensor_from_array(env, shape_weight, 2, weight_fp32);
    size_t q4_0_bytes = num_blocks * sizeof(marmot_q4_0_block_t);
    marmot_tensor_t *tensor_weight_q4_0 = marmot_tensor_create(env->ctx, &q4_0_bytes, 1, MARMOT_DTYPE_UINT8);
    assert_non_null(tensor_weight_fp32);
    assert_non_null(tensor_weight_q4_0);

    marmot_error_t err = marmot_quantize_q4_0(env->ctx, tensor_weight_fp32, tensor_weight_q4_0);
    assert_int_equal(err, MARMOT_SUCCESS);

    assert_int_equal(
        marmot_tensor_copy_to_host_buffer(env->ctx, tensor_weight_q4_0, weight_q4_0, q4_0_bytes), MARMOT_SUCCESS
    );

    const int row_blocks = (K + (int)MARMOT_QUANT_BLOCK_SIZE - 1) / (int)MARMOT_QUANT_BLOCK_SIZE;
    const size_t total_blocks = (size_t)row_blocks * (size_t)M;
    marmot_q4_0_block_t *weight_ref_blocks = (marmot_q4_0_block_t *)malloc(total_blocks * sizeof(marmot_q4_0_block_t));
    assert_non_null(weight_ref_blocks);
    for (int row = 0; row < M; ++row) {
        quantize_row_q4_0_ref_tail(
            weight_fp32 + (size_t)row * (size_t)K, K, weight_ref_blocks + (size_t)row * row_blocks
        );
    }
    assert_q4_0_blocks_match_reference(weight_fp32, weight_q4_0, M, K);

    // TODO: Call quantized matmul when implemented
    // For now, compute dequantized matmul as validation
    float *weight_dequant = (float *)malloc(M * K * sizeof(float));
    assert_non_null(weight_dequant);

    // Dequantize for reference
    for (size_t b = 0; b < num_blocks; b++) {
        const marmot_q4_0_block_t *block = &weight_q4_0[b];
        const float scale = (float)marmot_float16_to_native(block->scale);
        const size_t base_idx = b * MARMOT_QUANT_BLOCK_SIZE;

        for (size_t i = 0; i < MARMOT_Q4_PACKED_BYTES; i++) {
            const uint8_t packed = block->qs[i];
            const int lo = (int)(packed & 0x0f) - 8;
            const int hi = (int)(packed >> 4) - 8;
            weight_dequant[base_idx + i] = (float)lo * scale;
            weight_dequant[base_idx + i + MARMOT_QUANT_BLOCK_SIZE / 2] = (float)hi * scale;
        }
    }

    // Compute dequantized matmul
    float *output_dequant = (float *)malloc(N * M * sizeof(float));
    assert_non_null(output_dequant);
    reference_matmul_fp32(input, weight_dequant, output_dequant, N, K, M);

    float *weight_ref_dequant = (float *)malloc(M * K * sizeof(float));
    assert_non_null(weight_ref_dequant);
    for (size_t b = 0; b < total_blocks; ++b) {
        const marmot_q4_0_block_t *block = &weight_ref_blocks[b];
        const float scale = (float)marmot_float16_to_native(block->scale);
        const size_t base_idx = b * MARMOT_QUANT_BLOCK_SIZE;
        for (size_t i = 0; i < MARMOT_Q4_PACKED_BYTES; ++i) {
            const uint8_t packed = block->qs[i];
            const int lo = (packed & 0x0f) - 8;
            const int hi = (packed >> 4) - 8;
            weight_ref_dequant[base_idx + i] = (float)lo * scale;
            weight_ref_dequant[base_idx + i + MARMOT_QUANT_BLOCK_SIZE / 2] = (float)hi * scale;
        }
    }
    float *output_ref = (float *)malloc(N * M * sizeof(float));
    assert_non_null(output_ref);
    reference_matmul_fp32(input, weight_ref_dequant, output_ref, N, K, M);

    // Validate results - should be close to FP32 reference
    float avg_rel = 0.0f;
    for (int i = 0; i < N * M; i++) {
        float diff = fabsf(output_dequant[i] - output_fp32[i]);
        float ref_abs = fabsf(output_fp32[i]);
        float relative_error = ref_abs > 1e-6f ? diff / ref_abs : diff;

        if (relative_error > max_relative_error && diff > max_absolute_error) {
            printf(
                "  Element %d: FP32=%.4f, Q4_0=%.4f, diff=%.4f rel_err=%.2f%%\n", i, output_fp32[i], output_dequant[i],
                diff, relative_error * 100.0f
            );
        }
        if (!use_gaussian) {
            assert_true(relative_error <= max_relative_error || diff <= max_absolute_error);
        }
        avg_rel += relative_error;
    }
    const float avg_elem_rel = avg_rel / ((float)N * (float)M);
    float global_rel = compute_global_relative_error(output_dequant, output_fp32, N * M);
    float ref_global_rel = compute_global_relative_error(output_ref, output_fp32, N * M);
    printf(
        "Q4_0 global_rel=%.4f ref_global_rel=%.4f avg_elem_rel=%.4f (threshold=%.4f)\n", global_rel, ref_global_rel,
        avg_elem_rel, avg_relative_error
    );
    assert_true(global_rel <= avg_relative_error);

    free(output_ref);
    free(weight_ref_dequant);
    free(weight_ref_blocks);
    free(output_dequant);
    free(weight_dequant);
    marmot_test_tensor_destroy_all(2, tensor_weight_q4_0, tensor_weight_fp32);
    free(weight_q4_0);
    free(output_fp32);
    free(weight_fp32);
    free(input);
}

static void test_llama_ref_quant_parity_q4_0(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const int K = 64;
    const int M = 4;
    float *w = (float *)malloc((size_t)M * (size_t)K * sizeof(float));
    assert_non_null(w);
    srand(42);
    generate_gaussian_weight(w, M, K);
    // our quantizer
    size_t shape_q[] = {(size_t)M, (size_t)K};
    marmot_tensor_t *src = marmot_test_tensor_from_array(env, shape_q, 2, w);
    const size_t nblocks = (size_t)((M * K) / (int)MARMOT_QUANT_BLOCK_SIZE);
    const size_t qbytes = nblocks * sizeof(marmot_q4_0_block_t);
    // Q4_0 uses packed 4-bit codes -> UINT8 storage
    marmot_tensor_t *dst = marmot_tensor_create(env->ctx, &qbytes, 1, MARMOT_DTYPE_UINT8);
    assert_non_null(src);
    assert_non_null(dst);
    assert_int_equal(marmot_quantize_q4_0(env->ctx, src, dst), MARMOT_SUCCESS);
    marmot_q4_0_block_t *got = (marmot_q4_0_block_t *)malloc(qbytes);
    assert_non_null(got);
    assert_int_equal(marmot_tensor_copy_to_host_buffer(env->ctx, dst, got, qbytes), MARMOT_SUCCESS);
    // llama.cpp reference
    marmot_q4_0_block_t *ref = (marmot_q4_0_block_t *)malloc(qbytes);
    assert_non_null(ref);
    quantize_row_q4_0_ref(w, ref, (int64_t)M * (int64_t)K);
    // Compare via dequantized floats to avoid tiny rounding diffs
    float *deq_got = (float *)malloc((size_t)M * (size_t)K * sizeof(float));
    float *deq_ref = (float *)malloc((size_t)M * (size_t)K * sizeof(float));
    assert_non_null(deq_got);
    assert_non_null(deq_ref);
    // dequantize
    for (size_t b = 0; b < nblocks; ++b) {
        const marmot_q4_0_block_t *bg = &got[b];
        const marmot_q4_0_block_t *br = &ref[b];
        const float dg = (float)marmot_float16_to_native(bg->scale);
        const float dr = (float)marmot_float16_to_native(br->scale);
        const size_t base = b * MARMOT_QUANT_BLOCK_SIZE;
        for (size_t i = 0; i < MARMOT_Q4_PACKED_BYTES; ++i) {
            const uint8_t pg = bg->qs[i];
            const uint8_t pr = br->qs[i];
            const int lg0 = (pg & 0x0F) - 8;
            const int lg1 = (pg >> 4) - 8;
            const int lr0 = (pr & 0x0F) - 8;
            const int lr1 = (pr >> 4) - 8;
            deq_got[base + i] = (float)lg0 * dg;
            deq_got[base + i + MARMOT_QUANT_BLOCK_SIZE / 2] = (float)lg1 * dg;
            deq_ref[base + i] = (float)lr0 * dr;
            deq_ref[base + i + MARMOT_QUANT_BLOCK_SIZE / 2] = (float)lr1 * dr;
        }
    }
    float grel = compute_global_relative_error(deq_got, deq_ref, M * K);
    assert_true(grel <= 1e-3f);
    free(deq_ref);
    free(deq_got);
    free(ref);
    free(got);
    marmot_test_tensor_destroy_all(2, dst, src);
    free(w);
}

static void test_llama_ref_quant_parity_q4_1(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const int K = 64;
    const int M = 3;
    float *w = (float *)malloc((size_t)M * (size_t)K * sizeof(float));
    assert_non_null(w);
    srand(123);
    generate_gaussian_weight(w, M, K);
    size_t shape_q[] = {(size_t)M, (size_t)K};
    marmot_tensor_t *src = marmot_test_tensor_from_array(env, shape_q, 2, w);
    const size_t nblocks = (size_t)((M * K) / (int)MARMOT_QUANT_BLOCK_SIZE);
    const size_t qbytes = nblocks * sizeof(marmot_q4_1_block_t);
    // Q4_1 uses packed 4-bit codes -> UINT8 storage
    marmot_tensor_t *dst = marmot_tensor_create(env->ctx, &qbytes, 1, MARMOT_DTYPE_UINT8);
    assert_non_null(src);
    assert_non_null(dst);
    assert_int_equal(marmot_quantize_q4_1(env->ctx, src, dst), MARMOT_SUCCESS);
    marmot_q4_1_block_t *got = (marmot_q4_1_block_t *)malloc(qbytes);
    assert_non_null(got);
    assert_int_equal(marmot_tensor_copy_to_host_buffer(env->ctx, dst, got, qbytes), MARMOT_SUCCESS);
    marmot_q4_1_block_t *ref = (marmot_q4_1_block_t *)malloc(qbytes);
    assert_non_null(ref);
    quantize_row_q4_1_ref(w, ref, (int64_t)M * (int64_t)K);
    float *deq_got = (float *)malloc((size_t)M * (size_t)K * sizeof(float));
    float *deq_ref = (float *)malloc((size_t)M * (size_t)K * sizeof(float));
    assert_non_null(deq_got);
    assert_non_null(deq_ref);
    for (size_t b = 0; b < nblocks; ++b) {
        const marmot_q4_1_block_t *bg = &got[b];
        const marmot_q4_1_block_t *br = &ref[b];
        const float dg = (float)marmot_float16_to_native(bg->scale);
        const float mg = (float)marmot_float16_to_native(bg->min);
        const float dr = (float)marmot_float16_to_native(br->scale);
        const float mr = (float)marmot_float16_to_native(br->min);
        const size_t base = b * MARMOT_QUANT_BLOCK_SIZE;
        for (size_t i = 0; i < MARMOT_Q4_PACKED_BYTES; ++i) {
            const uint8_t pg = bg->qs[i];
            const uint8_t pr = br->qs[i];
            const int lg0 = (pg & 0x0F);
            const int lg1 = (pg >> 4);
            const int lr0 = (pr & 0x0F);
            const int lr1 = (pr >> 4);
            deq_got[base + i] = (float)lg0 * dg + mg;
            deq_got[base + i + MARMOT_QUANT_BLOCK_SIZE / 2] = (float)lg1 * dg + mg;
            deq_ref[base + i] = (float)lr0 * dr + mr;
            deq_ref[base + i + MARMOT_QUANT_BLOCK_SIZE / 2] = (float)lr1 * dr + mr;
        }
    }
    float grel = compute_global_relative_error(deq_got, deq_ref, M * K);
    assert_true(grel <= 1e-3f);
    free(deq_ref);
    free(deq_got);
    free(ref);
    free(got);
    marmot_test_tensor_destroy_all(2, dst, src);
    free(w);
}

static void test_llama_ref_quant_parity_q5_0(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const int seeds[] = {7, 42, 123, 777, 1337};
    const int M = 3, K = 64; // divisible by 32
    const size_t L = (size_t)M * (size_t)K;
    float *w = (float *)malloc(L * sizeof(float));
    assert_non_null(w);
    size_t shape_1d[] = {L};
    const size_t nblocks = L / (size_t)MARMOT_QUANT_BLOCK_SIZE;
    const size_t qbytes = nblocks * sizeof(marmot_q5_0_block_t);
    marmot_tensor_t *src = marmot_tensor_create(env->ctx, shape_1d, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *dst = marmot_tensor_create(env->ctx, &qbytes, 1, MARMOT_DTYPE_UINT8);
    assert_non_null(src);
    assert_non_null(dst);
    for (size_t si = 0; si < sizeof(seeds) / sizeof(seeds[0]); ++si) {
        srand(seeds[si]);
        generate_gaussian_weight(w, M, K);
        assert_int_equal(marmot_tensor_copy_from_host_buffer(env->ctx, src, w, L * sizeof(float)), MARMOT_SUCCESS);
        assert_int_equal(marmot_quantize_q5_0(env->ctx, src, dst), MARMOT_SUCCESS);
        marmot_q5_0_block_t *got = (marmot_q5_0_block_t *)malloc(qbytes);
        assert_non_null(got);
        assert_int_equal(marmot_tensor_copy_to_host_buffer(env->ctx, dst, got, qbytes), MARMOT_SUCCESS);
        marmot_q5_0_block_t *ref = (marmot_q5_0_block_t *)malloc(qbytes);
        assert_non_null(ref);
        quantize_row_q5_0_ref(w, ref, (int64_t)L);
        float *deq_got = (float *)malloc(L * sizeof(float));
        float *deq_ref = (float *)malloc(L * sizeof(float));
        assert_non_null(deq_got);
        assert_non_null(deq_ref);
        dequantize_q5_0_blocks_to_f32(got, L, deq_got);
        dequantize_q5_0_blocks_to_f32(ref, L, deq_ref);
        float grel = compute_global_relative_error(deq_got, deq_ref, (int)L);
        assert_true(grel <= 2e-3f);
        free(deq_ref);
        free(deq_got);
        free(ref);
        free(got);
    }
    marmot_test_tensor_destroy_all(2, dst, src);
    free(w);
}

static void test_llama_ref_quant_parity_q5_1(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const int seeds[] = {7, 42, 123, 777, 1337};
    const int M = 3, K = 64;
    const size_t L = (size_t)M * (size_t)K;
    float *w = (float *)malloc(L * sizeof(float));
    assert_non_null(w);
    size_t shape_1d[] = {L};
    const size_t nblocks = L / (size_t)MARMOT_QUANT_BLOCK_SIZE;
    const size_t qbytes = nblocks * sizeof(marmot_q5_1_block_t);
    marmot_tensor_t *src = marmot_tensor_create(env->ctx, shape_1d, 1, MARMOT_DTYPE_FLOAT32);
    // Q5_1 uses packed 5-bit codes -> UINT8 storage
    marmot_tensor_t *dst = marmot_tensor_create(env->ctx, &qbytes, 1, MARMOT_DTYPE_UINT8);
    assert_non_null(src);
    assert_non_null(dst);
    for (size_t si = 0; si < sizeof(seeds) / sizeof(seeds[0]); ++si) {
        srand(seeds[si]);
        generate_gaussian_weight(w, M, K);
        assert_int_equal(marmot_tensor_copy_from_host_buffer(env->ctx, src, w, L * sizeof(float)), MARMOT_SUCCESS);
        assert_int_equal(marmot_quantize_q5_1(env->ctx, src, dst), MARMOT_SUCCESS);
        marmot_q5_1_block_t *got = (marmot_q5_1_block_t *)malloc(qbytes);
        assert_non_null(got);
        assert_int_equal(marmot_tensor_copy_to_host_buffer(env->ctx, dst, got, qbytes), MARMOT_SUCCESS);
        marmot_q5_1_block_t *ref = (marmot_q5_1_block_t *)malloc(qbytes);
        assert_non_null(ref);
        quantize_row_q5_1_ref(w, ref, (int64_t)L);
        float *deq_got = (float *)malloc(L * sizeof(float));
        float *deq_ref = (float *)malloc(L * sizeof(float));
        assert_non_null(deq_got);
        assert_non_null(deq_ref);
        dequantize_q5_1_blocks_to_f32(got, L, deq_got);
        dequantize_q5_1_blocks_to_f32(ref, L, deq_ref);
        float grel = compute_global_relative_error(deq_got, deq_ref, (int)L);
        assert_true(grel <= 2e-3f);
        free(deq_ref);
        free(deq_got);
        free(ref);
        free(got);
    }
    marmot_test_tensor_destroy_all(2, dst, src);
    free(w);
}

static void test_llama_ref_quant_parity_q8_0(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const int seeds[] = {7, 42, 123, 777, 1337};
    const int M = 3, K = 64;
    const size_t L = (size_t)M * (size_t)K;
    float *w = (float *)malloc(L * sizeof(float));
    assert_non_null(w);
    size_t shape_1d[] = {L};
    const size_t nblocks = L / (size_t)MARMOT_QUANT_BLOCK_SIZE;
    const size_t qbytes = nblocks * sizeof(marmot_q8_0_block_t);
    marmot_tensor_t *src = marmot_tensor_create(env->ctx, shape_1d, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *dst = marmot_tensor_create(env->ctx, &qbytes, 1, MARMOT_DTYPE_INT8);
    assert_non_null(src);
    assert_non_null(dst);
    for (size_t si = 0; si < sizeof(seeds) / sizeof(seeds[0]); ++si) {
        srand(seeds[si]);
        generate_gaussian_weight(w, M, K);
        assert_int_equal(marmot_tensor_copy_from_host_buffer(env->ctx, src, w, L * sizeof(float)), MARMOT_SUCCESS);
        {
            printf(
                "  [q8_0] backend=%d dst_dtype=%d (INT8=%d) src_dtype=%d\n", (int)env->backend, (int)dst->dtype,
                (int)MARMOT_DTYPE_INT8, (int)src->dtype
            );
            marmot_error_t qerr = marmot_quantize_q8_0(env->ctx, src, dst);
            if (qerr != MARMOT_SUCCESS) {
                printf("  q8_0 quantize error: %s\n", marmot_error_string(qerr));
            }
            assert_int_equal(qerr, MARMOT_SUCCESS);
        }
        marmot_q8_0_block_t *got = (marmot_q8_0_block_t *)malloc(qbytes);
        assert_non_null(got);
        assert_int_equal(marmot_tensor_copy_to_host_buffer(env->ctx, dst, got, qbytes), MARMOT_SUCCESS);
        marmot_q8_0_block_t *ref = (marmot_q8_0_block_t *)malloc(qbytes);
        assert_non_null(ref);
        quantize_row_q8_0_ref(w, ref, (int64_t)L);
        float *deq_got = (float *)malloc(L * sizeof(float));
        float *deq_ref = (float *)malloc(L * sizeof(float));
        assert_non_null(deq_got);
        assert_non_null(deq_ref);
        dequantize_q8_0_blocks_to_f32(got, L, deq_got);
        dequantize_q8_0_blocks_to_f32(ref, L, deq_ref);
        float grel = compute_global_relative_error(deq_got, deq_ref, (int)L);
        assert_true(grel <= 1e-3f);
        free(deq_ref);
        free(deq_got);
        free(ref);
        free(got);
    }
    marmot_test_tensor_destroy_all(2, dst, src);
    free(w);
}

static void test_matmul_quantized_via_generic_api_q4_0_sized(
    marmot_test_env_t *env, int N, int K, int M, float max_relative_error, float avg_relative_error,
    float max_absolute_error, int use_gaussian
) {
    // Uses generic API path; supported on all backends

    // Linear convention: input(N×K) @ weight(M×K).T = output(N×M)
    float *weight_fp32 = (float *)malloc(M * K * sizeof(float));
    float *input = (float *)malloc(N * K * sizeof(float));
    float *output_fp32 = (float *)malloc(N * M * sizeof(float));
    assert_non_null(weight_fp32);
    assert_non_null(input);
    assert_non_null(output_fp32);

    if (use_gaussian) {
        srand(42);
        generate_gaussian_weight(weight_fp32, M, K);
        generate_gaussian_input(input, N, K);
    } else {
        generate_test_weight(weight_fp32, M, K);
        generate_test_input(input, N, K);
    }

    // Compute FP32 reference: input(N×K) @ weight(M×K).T
    reference_matmul_fp32(input, weight_fp32, output_fp32, N, K, M);

    size_t shape_weight[] = {(size_t)M, (size_t)K};
    marmot_tensor_t *tensor_weight_fp32 = marmot_test_tensor_from_array(env, shape_weight, 2, weight_fp32);
    assert_non_null(tensor_weight_fp32);

    const size_t blocks_per_row = (size_t)((K + (int)MARMOT_QUANT_BLOCK_SIZE - 1) / (int)MARMOT_QUANT_BLOCK_SIZE);
    const size_t num_blocks = (size_t)M * blocks_per_row;
    const size_t q4_0_bytes = num_blocks * sizeof(marmot_q4_0_block_t);
    marmot_q4_0_block_t *weight_q4_0 = (marmot_q4_0_block_t *)malloc(q4_0_bytes);
    assert_non_null(weight_q4_0);

    marmot_tensor_t *tensor_weight_q4_0 = marmot_tensor_create(env->ctx, &q4_0_bytes, 1, MARMOT_DTYPE_UINT8);
    assert_non_null(tensor_weight_q4_0);
    assert_int_equal(marmot_quantize_q4_0(env->ctx, tensor_weight_fp32, tensor_weight_q4_0), MARMOT_SUCCESS);
    assert_int_equal(
        marmot_tensor_copy_to_host_buffer(env->ctx, tensor_weight_q4_0, weight_q4_0, q4_0_bytes), MARMOT_SUCCESS
    );

    // Set up quantized tensor metadata for matmul
    tensor_weight_q4_0->quant_kind = MARMOT_QUANT_KIND_Q4_0;
    tensor_weight_q4_0->quant_layout = MARMOT_QUANT_LAYOUT_GGUF;
    tensor_weight_q4_0->shape.ndim = 2;
    tensor_weight_q4_0->shape.shape[0] = M;
    tensor_weight_q4_0->shape.shape[1] = K;
    tensor_weight_q4_0->shape.strides[0] = K;
    tensor_weight_q4_0->shape.strides[1] = 1;

    // Create input and output tensors
    size_t shape_input[] = {(size_t)N, (size_t)K};
    size_t shape_output[] = {(size_t)N, (size_t)M};
    marmot_tensor_t *tensor_input = marmot_test_tensor_from_array(env, shape_input, 2, input);
    marmot_tensor_t *tensor_output = marmot_tensor_create(env->ctx, shape_output, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(tensor_input);
    assert_non_null(tensor_output);

    // Call generic matmul - should auto-dispatch to quantized version
    marmot_error_t err = marmot_linear(env->ctx, tensor_input, tensor_weight_q4_0, nullptr, tensor_output);
    if (err != MARMOT_SUCCESS) {
        printf("  matmul failed with error %d: %s\n", err, marmot_error_string(err));
    }
    assert_int_equal(err, MARMOT_SUCCESS);

    // Copy output and validate
    float *output_quantized = (float *)malloc(N * M * sizeof(float));
    assert_non_null(output_quantized);

    // Copy from device memory (Metal) or host memory (CPU)
    assert_int_equal(
        marmot_tensor_copy_to_host_buffer(env->ctx, tensor_output, output_quantized, N * M * sizeof(float)),
        MARMOT_SUCCESS
    );

    float avg_rel = 0.0f;
    for (int i = 0; i < N * M; i++) {
        float diff = fabsf(output_quantized[i] - output_fp32[i]);
        float ref_abs = fabsf(output_fp32[i]);
        float relative_error = ref_abs > 1e-6f ? diff / ref_abs : diff;

        if (relative_error > max_relative_error && diff > max_absolute_error) {
            printf(
                "    [%d] FP32=%.4f Q4_0=%.4f diff=%.4f rel_err=%.2f%%\n", i, output_fp32[i], output_quantized[i], diff,
                relative_error * 100.0f
            );
        }
        if (!use_gaussian) {
            assert_true(relative_error <= max_relative_error || diff <= max_absolute_error);
        }
        avg_rel += relative_error;
    }
    const float avg_elem_rel = avg_rel / ((float)N * (float)M);
    float global_rel = compute_global_relative_error(output_quantized, output_fp32, N * M);
    printf(
        "Q4_0 matmul global_rel=%.4f avg_elem_rel=%.4f (threshold=%.4f)\n", global_rel, avg_elem_rel, avg_relative_error
    );
    assert_true(global_rel <= avg_relative_error);

    marmot_test_tensor_destroy_all(4, tensor_output, tensor_input, tensor_weight_q4_0, tensor_weight_fp32);
    free(output_quantized);
    free(weight_q4_0);
    free(output_fp32);
    free(input);
    free(weight_fp32);
}

static void test_matmul_quantized_q4_0_fp16_path(
    marmot_test_env_t *env, int N, int K, int M, marmot_dtype_t out_dtype, float max_relative_error,
    float global_relative_error, float max_absolute_error
) {
    float *input_fp32 = (float *)malloc(N * K * sizeof(float));
    float *weight_fp32 = (float *)malloc(M * K * sizeof(float));
    float *output_fp32 = (float *)malloc(N * M * sizeof(float));
    assert_non_null(input_fp32);
    assert_non_null(weight_fp32);
    assert_non_null(output_fp32);

    srand(42);
    generate_gaussian_input(input_fp32, N, K);
    generate_gaussian_weight(weight_fp32, M, K);

    reference_matmul_fp32(input_fp32, weight_fp32, output_fp32, N, K, M);

    const size_t blocks_per_row = (size_t)((K + (int)MARMOT_QUANT_BLOCK_SIZE - 1) / (int)MARMOT_QUANT_BLOCK_SIZE);
    const size_t num_blocks = (size_t)M * blocks_per_row;
    const size_t q4_0_bytes = num_blocks * sizeof(marmot_q4_0_block_t);
    marmot_q4_0_block_t *weight_q4_0 = (marmot_q4_0_block_t *)malloc(q4_0_bytes);
    assert_non_null(weight_q4_0);
    for (int m = 0; m < M; ++m) {
        const float *row = weight_fp32 + (size_t)m * (size_t)K;
        marmot_q4_0_block_t *row_blocks = weight_q4_0 + (size_t)m * blocks_per_row;
        quantize_row_q4_0_ref_tail(row, K, row_blocks);
    }

    marmot_tensor_t *tensor_weight_q4_0 = marmot_tensor_create(env->ctx, &q4_0_bytes, 1, MARMOT_DTYPE_UINT8);
    assert_non_null(tensor_weight_q4_0);
    memcpy(tensor_weight_q4_0->data, weight_q4_0, q4_0_bytes);
    assert_int_equal(marmot_tensor_to_device(env->ctx, tensor_weight_q4_0), MARMOT_SUCCESS);

    tensor_weight_q4_0->quant_kind = MARMOT_QUANT_KIND_Q4_0;
    tensor_weight_q4_0->quant_layout = MARMOT_QUANT_LAYOUT_GGUF;
    tensor_weight_q4_0->shape.ndim = 2;
    tensor_weight_q4_0->shape.shape[0] = (size_t)M;
    tensor_weight_q4_0->shape.shape[1] = (size_t)K;
    tensor_weight_q4_0->shape.strides[0] = (size_t)K;
    tensor_weight_q4_0->shape.strides[1] = 1;

    size_t shape_input[] = {(size_t)N, (size_t)K};
    size_t shape_output[] = {(size_t)N, (size_t)M};

    marmot_tensor_t *tensor_input = marmot_tensor_create(env->ctx, shape_input, 2, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *tensor_output = marmot_tensor_create(env->ctx, shape_output, 2, out_dtype);
    assert_non_null(tensor_input);
    assert_non_null(tensor_output);

    marmot_test_convert_f32_span(env, tensor_input, input_fp32, (size_t)(N * K));

    marmot_error_t err = marmot_linear(env->ctx, tensor_input, tensor_weight_q4_0, nullptr, tensor_output);
    if (err != MARMOT_SUCCESS) {
        printf("  FP16 activations matmul failed: %s\n", marmot_error_string(err));
    }
    assert_int_equal(err, MARMOT_SUCCESS);

    float *output_quantized = (float *)malloc(N * M * sizeof(float));
    assert_non_null(output_quantized);
    marmot_test_fetch_f32_span(env, output_quantized, tensor_output, (size_t)(N * M));

    double sum_rel = 0.0;
    for (int i = 0; i < N * M; ++i) {
        float diff = fabsf(output_quantized[i] - output_fp32[i]);
        float ref_abs = fabsf(output_fp32[i]);
        float rel_err = ref_abs > 1e-6f ? diff / ref_abs : diff;
        if (rel_err > max_relative_error && diff > max_absolute_error) {
            if (i < 4) {
                printf(
                    "  [FP16] idx=%d FP32=%.4f quant=%.4f diff=%.4f rel=%.2f%%\n", i, output_fp32[i],
                    output_quantized[i], diff, rel_err * 100.0f
                );
            }
        }
        sum_rel += rel_err;
    }

    float global_rel = compute_global_relative_error(output_quantized, output_fp32, N * M);
    printf(
        "  FP16 activations out=%s global_rel=%.2f%% (avg_elem=%.2f%%)\n",
        out_dtype == MARMOT_DTYPE_FLOAT16 ? "FP16" : "FP32", global_rel * 100.0f,
        (float)(sum_rel / (double)(N * M) * 100.0)
    );
    assert_true(global_rel <= global_relative_error);

    marmot_test_tensor_destroy_all(3, tensor_output, tensor_input, tensor_weight_q4_0);
    free(output_quantized);
    free(weight_q4_0);
    free(output_fp32);
    free(weight_fp32);
    free(input_fp32);
}

// --------------------------- Q4_1 Helpers ---------------------------
static void dequantize_q4_1_blocks_to_f32(const marmot_q4_1_block_t *blocks, size_t total, float *dst) {
    const size_t num_blocks = (total + MARMOT_QUANT_BLOCK_SIZE - 1) / MARMOT_QUANT_BLOCK_SIZE;
    const size_t half = MARMOT_QUANT_BLOCK_SIZE / 2;
    for (size_t b = 0; b < num_blocks; ++b) {
        const marmot_q4_1_block_t *blk = &blocks[b];
        const float scale = marmot_f16_to_f32_ref(blk->scale);
        const float minv = marmot_f16_to_f32_ref(blk->min);
        const size_t base = b * MARMOT_QUANT_BLOCK_SIZE;
        for (size_t j = 0; j < MARMOT_Q4_PACKED_BYTES; ++j) {
            const uint8_t packed = blk->qs[j];
            const uint8_t q0 = packed & 0x0F;
            const uint8_t q1 = packed >> 4;
            if (base + j < total) {
                dst[base + j] = (float)q0 * scale + minv;
            }
            if (base + j + half < total) {
                dst[base + j + half] = (float)q1 * scale + minv;
            }
        }
    }
}

static void test_matmul_quantized_q4_1_sized(
    marmot_test_env_t *env, int N, int K, int M, float max_relative_error, float global_relative_error,
    float max_absolute_error
) {
    // Enable for all backends; use device memcpy for host copies when needed
    float *input = (float *)malloc(N * K * sizeof(float));
    float *weight_fp32 = (float *)malloc(M * K * sizeof(float));
    float *output_fp32 = (float *)malloc(N * M * sizeof(float));
    assert_non_null(input);
    assert_non_null(weight_fp32);
    assert_non_null(output_fp32);

    srand(42);
    generate_gaussian_input(input, N, K);
    generate_gaussian_weight(weight_fp32, M, K);

    reference_matmul_fp32(input, weight_fp32, output_fp32, N, K, M);

    // Quantize weight to Q4_1
    const size_t num_blocks = (M * K + MARMOT_QUANT_BLOCK_SIZE - 1) / MARMOT_QUANT_BLOCK_SIZE;
    size_t shape_weight[] = {(size_t)M, (size_t)K};
    marmot_tensor_t *tensor_weight_fp32 = marmot_test_tensor_from_array(env, shape_weight, 2, weight_fp32);
    size_t q4_1_bytes = num_blocks * sizeof(marmot_q4_1_block_t);
    marmot_tensor_t *tensor_weight_q4_1 = marmot_tensor_create(env->ctx, &q4_1_bytes, 1, MARMOT_DTYPE_UINT8);
    assert_non_null(tensor_weight_fp32);
    assert_non_null(tensor_weight_q4_1);

    marmot_error_t err = marmot_quantize_q4_1(env->ctx, tensor_weight_fp32, tensor_weight_q4_1);
    assert_int_equal(err, MARMOT_SUCCESS);

    // Dequantize on host to validate dequantized matmul
    marmot_q4_1_block_t *weight_q4_1 = (marmot_q4_1_block_t *)malloc(q4_1_bytes);
    assert_non_null(weight_q4_1);
    assert_int_equal(
        marmot_tensor_copy_to_host_buffer(env->ctx, tensor_weight_q4_1, weight_q4_1, q4_1_bytes), MARMOT_SUCCESS
    );
    float *weight_dequant = (float *)malloc(M * K * sizeof(float));
    assert_non_null(weight_dequant);
    dequantize_q4_1_blocks_to_f32(weight_q4_1, (size_t)(M * K), weight_dequant);

    float *output_dequant = (float *)malloc(N * M * sizeof(float));
    assert_non_null(output_dequant);
    reference_matmul_fp32(input, weight_dequant, output_dequant, N, K, M);

    // Per-element checks are noisy for small refs; rely on global_rel plus abs guard
    for (int i = 0; i < N * M; ++i) {
        float diff = fabsf(output_dequant[i] - output_fp32[i]);
        float ref_abs = fabsf(output_fp32[i]);
        float relative_error = ref_abs > 1e-6f ? diff / ref_abs : diff;
        if (relative_error > max_relative_error && diff > max_absolute_error) {
            // Print a couple outliers for debuggability
            if (i < 4) {
                printf(
                    "  [Q4_1 deq] idx=%d FP32=%.4f Q4_1=%.4f diff=%.4f rel=%.2f%%\n", i, output_fp32[i],
                    output_dequant[i], diff, relative_error * 100.0f
                );
            }
        }
    }
    float global_rel = compute_global_relative_error(output_dequant, output_fp32, N * M);
    assert_true(global_rel <= global_relative_error);

    free(output_dequant);
    free(weight_dequant);
    free(weight_q4_1);
    marmot_test_tensor_destroy_all(2, tensor_weight_q4_1, tensor_weight_fp32);
    free(output_fp32);
    free(weight_fp32);
    free(input);
}

static void test_matmul_quantized_via_generic_api_q4_1_sized(
    marmot_test_env_t *env, int N, int K, int M, float max_relative_error, float global_relative_error,
    float max_absolute_error
) {
    // Uses generic API path; supported on all backends

    float *weight_fp32 = (float *)malloc(M * K * sizeof(float));
    float *input = (float *)malloc(N * K * sizeof(float));
    float *output_fp32 = (float *)malloc(N * M * sizeof(float));
    assert_non_null(weight_fp32);
    assert_non_null(input);
    assert_non_null(output_fp32);

    srand(42);
    generate_gaussian_weight(weight_fp32, M, K);
    generate_gaussian_input(input, N, K);

    reference_matmul_fp32(input, weight_fp32, output_fp32, N, K, M);

    const size_t num_blocks = (M * K + MARMOT_QUANT_BLOCK_SIZE - 1) / MARMOT_QUANT_BLOCK_SIZE;
    size_t shape_weight[] = {(size_t)M, (size_t)K};
    marmot_tensor_t *tensor_weight_fp32 = marmot_test_tensor_from_array(env, shape_weight, 2, weight_fp32);
    size_t q4_1_bytes = num_blocks * sizeof(marmot_q4_1_block_t);
    marmot_tensor_t *tensor_weight_q4_1 = marmot_tensor_create(env->ctx, &q4_1_bytes, 1, MARMOT_DTYPE_UINT8);
    assert_non_null(tensor_weight_fp32);
    assert_non_null(tensor_weight_q4_1);

    marmot_error_t err = marmot_quantize_q4_1(env->ctx, tensor_weight_fp32, tensor_weight_q4_1);
    assert_int_equal(err, MARMOT_SUCCESS);

    tensor_weight_q4_1->quant_kind = MARMOT_QUANT_KIND_Q4_1;
    tensor_weight_q4_1->quant_layout = MARMOT_QUANT_LAYOUT_GGUF;
    tensor_weight_q4_1->shape.ndim = 2;
    tensor_weight_q4_1->shape.shape[0] = M;
    tensor_weight_q4_1->shape.shape[1] = K;
    tensor_weight_q4_1->shape.strides[0] = K;
    tensor_weight_q4_1->shape.strides[1] = 1;

    size_t shape_input[] = {(size_t)N, (size_t)K};
    size_t shape_output[] = {(size_t)N, (size_t)M};
    marmot_tensor_t *tensor_input = marmot_test_tensor_from_array(env, shape_input, 2, input);
    marmot_tensor_t *tensor_output = marmot_tensor_create(env->ctx, shape_output, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(tensor_input);
    assert_non_null(tensor_output);

    err = marmot_linear(env->ctx, tensor_input, tensor_weight_q4_1, nullptr, tensor_output);
    assert_int_equal(err, MARMOT_SUCCESS);

    float *output_quantized = (float *)malloc(N * M * sizeof(float));
    assert_non_null(output_quantized);
    marmot_test_fetch_f32_span(env, output_quantized, tensor_output, (size_t)(N * M));

    for (int i = 0; i < N * M; ++i) {
        float diff = fabsf(output_quantized[i] - output_fp32[i]);
        float ref_abs = fabsf(output_fp32[i]);
        float relative_error = ref_abs > 1e-6f ? diff / ref_abs : diff;
        if (relative_error > max_relative_error && diff > max_absolute_error) {
            if (i < 4) {
                printf(
                    "    [Q4_1 mm] idx=%d FP32=%.4f Q4_1=%.4f diff=%.4f rel=%.2f%%\n", i, output_fp32[i],
                    output_quantized[i], diff, relative_error * 100.0f
                );
            }
        }
    }
    float global_rel = compute_global_relative_error(output_quantized, output_fp32, N * M);
    assert_true(global_rel <= global_relative_error);

    marmot_test_tensor_destroy_all(3, tensor_output, tensor_input, tensor_weight_q4_1);
    marmot_tensor_destroy(tensor_weight_fp32);
    free(output_quantized);
    free(output_fp32);
    free(input);
    free(weight_fp32);
}

// Q4_1 Gaussian tests
static void test_matmul_quantized_q4_1_small(marmot_test_env_t *env) {
    test_matmul_quantized_q4_1_sized(env, TEST_SMALL_N, TEST_SMALL_K, TEST_SMALL_M, 0.20f, 0.12f, 1.5f);
}
static void test_matmul_quantized_q4_1_medium(marmot_test_env_t *env) {
    test_matmul_quantized_q4_1_sized(env, TEST_MEDIUM_N, TEST_MEDIUM_K, TEST_MEDIUM_M, 0.12f, 0.08f, 0.6f);
}
static void test_matmul_quantized_q4_1_production(marmot_test_env_t *env) {
    test_matmul_quantized_q4_1_sized(env, TEST_PROD_N, TEST_PROD_K, TEST_PROD_M, 0.10f, 0.09f, 0.2f);
}

static void test_matmul_quantized_via_generic_api_q4_1_small(marmot_test_env_t *env) {
    test_matmul_quantized_via_generic_api_q4_1_sized(env, TEST_SMALL_N, TEST_SMALL_K, TEST_SMALL_M, 0.20f, 0.12f, 1.5f);
}
static void test_matmul_quantized_via_generic_api_q4_1_medium(marmot_test_env_t *env) {
    test_matmul_quantized_via_generic_api_q4_1_sized(
        env, TEST_MEDIUM_N, TEST_MEDIUM_K, TEST_MEDIUM_M, 0.12f, 0.08f, 0.6f
    );
}
static void test_matmul_quantized_via_generic_api_q4_1_production(marmot_test_env_t *env) {
    test_matmul_quantized_via_generic_api_q4_1_sized(env, TEST_PROD_N, TEST_PROD_K, TEST_PROD_M, 0.10f, 0.09f, 0.2f);
}

static void test_matmul_quantized_q4_0_fp16_fp32(marmot_test_env_t *env) {
    test_matmul_quantized_q4_0_fp16_path(
        env, TEST_MEDIUM_N, TEST_MEDIUM_K, TEST_MEDIUM_M, MARMOT_DTYPE_FLOAT32, 0.14f, 0.10f, 0.35f
    );
}

static void test_matmul_quantized_q4_0_fp16_fp16(marmot_test_env_t *env) {
    test_matmul_quantized_q4_0_fp16_path(
        env, TEST_MEDIUM_N, TEST_MEDIUM_K, TEST_MEDIUM_M, MARMOT_DTYPE_FLOAT16, 0.16f, 0.11f, 0.45f
    );
}

static void test_matmul_quantized_q4_0_fp16_small_batch(marmot_test_env_t *env) {
    const int small_n = 3;
    const int skinny_m = 17;
    test_matmul_quantized_q4_0_fp16_path(
        env, small_n, TEST_MEDIUM_K, skinny_m, MARMOT_DTYPE_FLOAT32, 0.18f, 0.13f, 0.50f
    );
}

static void test_matmul_quantized_qx_fp16_path(
    marmot_test_env_t *env, marmot_quant_kind_t kind, int N, int K, int M, marmot_dtype_t out_dtype,
    float max_relative_error, float global_relative_error, float max_absolute_error
) {
    float *input_fp32 = (float *)malloc(N * K * sizeof(float));
    float *weight_fp32 = (float *)malloc(M * K * sizeof(float));
    float *output_fp32 = (float *)malloc(N * M * sizeof(float));
    assert_non_null(input_fp32);
    assert_non_null(weight_fp32);
    assert_non_null(output_fp32);

    srand(42);
    generate_gaussian_input(input_fp32, N, K);
    generate_gaussian_weight(weight_fp32, M, K);

    reference_matmul_fp32(input_fp32, weight_fp32, output_fp32, N, K, M);

    // Create weight tensors
    size_t q_bytes = 0;
    marmot_dtype_t storage = MARMOT_DTYPE_UINT8;
    const size_t blocks_per_row = (size_t)((K + (int)MARMOT_QUANT_BLOCK_SIZE - 1) / (int)MARMOT_QUANT_BLOCK_SIZE);
    if (kind == MARMOT_QUANT_KIND_Q5_0) {
        q_bytes = (size_t)M * blocks_per_row * sizeof(marmot_q5_0_block_t);
        storage = MARMOT_DTYPE_UINT8;
    } else if (kind == MARMOT_QUANT_KIND_Q5_1) {
        q_bytes = (size_t)M * blocks_per_row * sizeof(marmot_q5_1_block_t);
        storage = MARMOT_DTYPE_UINT8;
    } else if (kind == MARMOT_QUANT_KIND_Q8_0) {
        q_bytes = (size_t)M * blocks_per_row * sizeof(marmot_q8_0_block_t);
        storage = MARMOT_DTYPE_INT8;
    } else {
        assert_true(0 && "unsupported kind");
    }

    marmot_tensor_t *tensor_weight_qx = marmot_tensor_create(env->ctx, &q_bytes, 1, storage);
    assert_non_null(tensor_weight_qx);

    // Row-wise quantization: build per-row temporary tensors and quantize
    for (int m = 0; m < M; ++m) {
        size_t row_shape[] = {(size_t)K};
        const float *row_src = weight_fp32 + (size_t)m * (size_t)K;
        marmot_tensor_t *row_fp32 = marmot_test_tensor_from_array(env, row_shape, 1, (float *)row_src);
        assert_non_null(row_fp32);

        size_t row_blocks = blocks_per_row;
        size_t row_bytes = 0;
        if (kind == MARMOT_QUANT_KIND_Q5_0) {
            row_bytes = row_blocks * sizeof(marmot_q5_0_block_t);
        } else if (kind == MARMOT_QUANT_KIND_Q5_1) {
            row_bytes = row_blocks * sizeof(marmot_q5_1_block_t);
        } else if (kind == MARMOT_QUANT_KIND_Q8_0) {
            row_bytes = row_blocks * sizeof(marmot_q8_0_block_t);
        }
        marmot_tensor_t *row_q = marmot_tensor_create(env->ctx, &row_bytes, 1, storage);
        assert_non_null(row_q);

        marmot_error_t qerr = MARMOT_ERROR_INVALID_ARGUMENT;
        if (kind == MARMOT_QUANT_KIND_Q5_0) {
            qerr = marmot_quantize_q5_0(env->ctx, row_fp32, row_q);
        } else if (kind == MARMOT_QUANT_KIND_Q5_1) {
            qerr = marmot_quantize_q5_1(env->ctx, row_fp32, row_q);
        } else if (kind == MARMOT_QUANT_KIND_Q8_0) {
            qerr = marmot_quantize_q8_0(env->ctx, row_fp32, row_q);
        }
        assert_int_equal(qerr, MARMOT_SUCCESS);

        // Copy row into the final tensor at row offset (row-major blocks)
        uint8_t *dst_bytes = (uint8_t *)tensor_weight_qx->data + (size_t)m * row_bytes;
        assert_int_equal(marmot_tensor_copy_to_host_buffer(env->ctx, row_q, dst_bytes, row_bytes), MARMOT_SUCCESS);
        marmot_tensor_destroy(row_q);
        marmot_tensor_destroy(row_fp32);
    }

    assert_int_equal(marmot_tensor_to_device(env->ctx, tensor_weight_qx), MARMOT_SUCCESS);

    tensor_weight_qx->quant_kind = kind;
    tensor_weight_qx->quant_layout = MARMOT_QUANT_LAYOUT_GGUF;
    tensor_weight_qx->shape.ndim = 2;
    tensor_weight_qx->shape.shape[0] = (size_t)M;
    tensor_weight_qx->shape.shape[1] = (size_t)K;
    tensor_weight_qx->shape.strides[0] = (size_t)K;
    tensor_weight_qx->shape.strides[1] = 1;

    size_t shape_input[] = {(size_t)N, (size_t)K};
    size_t shape_output[] = {(size_t)N, (size_t)M};

    marmot_tensor_t *tensor_input = marmot_tensor_create(env->ctx, shape_input, 2, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *tensor_output = marmot_tensor_create(env->ctx, shape_output, 2, out_dtype);
    assert_non_null(tensor_input);
    assert_non_null(tensor_output);

    marmot_test_convert_f32_span(env, tensor_input, input_fp32, (size_t)(N * K));

    marmot_error_t err = marmot_linear(env->ctx, tensor_input, tensor_weight_qx, nullptr, tensor_output);
    if (err != MARMOT_SUCCESS) {
        printf("  FP16 activations matmul failed: %s\n", marmot_error_string(err));
    }
    assert_int_equal(err, MARMOT_SUCCESS);

    float *output_quantized = (float *)malloc(N * M * sizeof(float));
    assert_non_null(output_quantized);
    marmot_test_fetch_f32_span(env, output_quantized, tensor_output, (size_t)(N * M));

    double sum_rel = 0.0;
    for (int i = 0; i < N * M; ++i) {
        float diff = fabsf(output_quantized[i] - output_fp32[i]);
        float ref_abs = fabsf(output_fp32[i]);
        float rel_err = ref_abs > 1e-6f ? diff / ref_abs : diff;
        if (rel_err > max_relative_error && diff > max_absolute_error) {
            if (i < 4) {
                printf(
                    "  [FP16] idx=%d FP32=%.4f quant=%.4f diff=%.4f rel=%.2f%%\n", i, output_fp32[i],
                    output_quantized[i], diff, rel_err * 100.0f
                );
            }
        }
        sum_rel += rel_err;
    }

    float global_rel = compute_global_relative_error(output_quantized, output_fp32, N * M);
    printf(
        "  kind=%d out=%s global_rel=%.2f%% (avg_elem=%.2f%%)\n", (int)kind,
        out_dtype == MARMOT_DTYPE_FLOAT16 ? "FP16" : "FP32", global_rel * 100.0f,
        (float)(sum_rel / (double)(N * M) * 100.0)
    );
    assert_true(global_rel <= global_relative_error);

    marmot_test_tensor_destroy_all(3, tensor_output, tensor_input, tensor_weight_qx);
    free(output_quantized);
    free(output_fp32);
    free(weight_fp32);
    free(input_fp32);
}

static void test_matmul_quantized_q5_0_fp16_fp32(marmot_test_env_t *env) {
    const float max_rel = 0.14f;
    const float glob = (env->backend == MARMOT_BACKEND_METAL) ? 0.12f : 0.10f;
    test_matmul_quantized_qx_fp16_path(
        env, MARMOT_QUANT_KIND_Q5_0, TEST_MEDIUM_N, TEST_MEDIUM_K, TEST_MEDIUM_M, MARMOT_DTYPE_FLOAT32, max_rel, glob,
        0.35f
    );
}
static void test_matmul_quantized_q5_0_fp16_fp16(marmot_test_env_t *env) {
    const float max_rel = 0.16f;
    const float glob = (env->backend == MARMOT_BACKEND_METAL) ? 0.13f : 0.11f;
    test_matmul_quantized_qx_fp16_path(
        env, MARMOT_QUANT_KIND_Q5_0, TEST_MEDIUM_N, TEST_MEDIUM_K, TEST_MEDIUM_M, MARMOT_DTYPE_FLOAT16, max_rel, glob,
        0.45f
    );
}
static void test_matmul_quantized_q5_1_fp16_fp32(marmot_test_env_t *env) {
    const float max_rel = 0.14f;
    const float glob = 0.10f;
    test_matmul_quantized_qx_fp16_path(
        env, MARMOT_QUANT_KIND_Q5_1, TEST_MEDIUM_N, TEST_MEDIUM_K, TEST_MEDIUM_M, MARMOT_DTYPE_FLOAT32, max_rel, glob,
        0.35f
    );
}
static void test_matmul_quantized_q5_1_fp16_fp16(marmot_test_env_t *env) {
    const float max_rel = 0.16f;
    const float glob = 0.11f;
    test_matmul_quantized_qx_fp16_path(
        env, MARMOT_QUANT_KIND_Q5_1, TEST_MEDIUM_N, TEST_MEDIUM_K, TEST_MEDIUM_M, MARMOT_DTYPE_FLOAT16, max_rel, glob,
        0.45f
    );
}
static void test_matmul_quantized_q8_0_fp16_fp32(marmot_test_env_t *env) {
    test_matmul_quantized_qx_fp16_path(
        env, MARMOT_QUANT_KIND_Q8_0, TEST_MEDIUM_N, TEST_MEDIUM_K, TEST_MEDIUM_M, MARMOT_DTYPE_FLOAT32, 0.12f, 0.10f,
        0.35f
    );
}
static void test_matmul_quantized_q8_0_fp16_fp16(marmot_test_env_t *env) {
    test_matmul_quantized_qx_fp16_path(
        env, MARMOT_QUANT_KIND_Q8_0, TEST_MEDIUM_N, TEST_MEDIUM_K, TEST_MEDIUM_M, MARMOT_DTYPE_FLOAT16, 0.14f, 0.11f,
        0.45f
    );
}

// Gaussian-based primary tests
static void test_matmul_quantized_q4_0_small(marmot_test_env_t *env) {
    test_matmul_quantized_q4_0_sized(env, TEST_SMALL_N, TEST_SMALL_K, TEST_SMALL_M, 0.40f, 0.25f, 1.5f, 1);
}
static void test_matmul_quantized_q4_0_medium(marmot_test_env_t *env) {
    test_matmul_quantized_q4_0_sized(env, TEST_MEDIUM_N, TEST_MEDIUM_K, TEST_MEDIUM_M, 0.20f, 0.12f, 0.6f, 1);
}
static void test_matmul_quantized_q4_0_production(marmot_test_env_t *env) {
    test_matmul_quantized_q4_0_sized(env, TEST_PROD_N, TEST_PROD_K, TEST_PROD_M, 0.12f, 0.10f, 0.2f, 1);
}

static void test_matmul_quantized_via_generic_api_q4_0_small(marmot_test_env_t *env) {
    test_matmul_quantized_via_generic_api_q4_0_sized(
        env, TEST_SMALL_N, TEST_SMALL_K, TEST_SMALL_M, 0.40f, 0.25f, 1.5f, 1
    );
}
static void test_matmul_quantized_via_generic_api_q4_0_medium(marmot_test_env_t *env) {
    test_matmul_quantized_via_generic_api_q4_0_sized(
        env, TEST_MEDIUM_N, TEST_MEDIUM_K, TEST_MEDIUM_M, 0.20f, 0.12f, 0.6f, 1
    );
}
static void test_matmul_quantized_via_generic_api_q4_0_production(marmot_test_env_t *env) {
    test_matmul_quantized_via_generic_api_q4_0_sized(env, TEST_PROD_N, TEST_PROD_K, TEST_PROD_M, 0.12f, 0.10f, 0.2f, 1);
}

static void run_matmul_quantized_suite(marmot_test_env_t *env) {
    printf("\n=== Q4_0 Dequantization (Gaussian, Small: K=64) ===\n");
    test_matmul_quantized_q4_0_small(env);
    printf("\n=== Q4_0 Dequantization (Gaussian, Medium: K=256) ===\n");
    test_matmul_quantized_q4_0_medium(env);
    printf("\n=== Q4_0 Dequantization (Gaussian, Production: K=4096) ===\n");
    test_matmul_quantized_q4_0_production(env);

    printf("\n=== Q4_1 Dequantization (Gaussian, Small: K=64) ===\n");
    test_matmul_quantized_q4_1_small(env);
    printf("\n=== Q4_1 Dequantization (Gaussian, Medium: K=256) ===\n");
    test_matmul_quantized_q4_1_medium(env);
    printf("\n=== Q4_1 Dequantization (Gaussian, Production: K=4096) ===\n");
    test_matmul_quantized_q4_1_production(env);
}

static void run_matmul_via_generic_api_suite(marmot_test_env_t *env) {
    printf("\n=== Q4_0×Q8_0 Matmul (Gaussian, Small: K=64) ===\n");
    test_matmul_quantized_via_generic_api_q4_0_small(env);
    printf("\n=== Q4_0×Q8_0 Matmul (Gaussian, Medium: K=256) ===\n");
    test_matmul_quantized_via_generic_api_q4_0_medium(env);
    printf("\n=== Q4_0×Q8_0 Matmul (Gaussian, Production: K=4096) ===\n");
    test_matmul_quantized_via_generic_api_q4_0_production(env);

    // Tail-K coverage (K % 32 != 0)
    printf("\n=== Q4_0×Q8_0 Matmul (Gaussian, Tail K=97) ===\n");
    // Slightly relaxed thresholds vs medium (reduced averaging). Metal tends to be ~0.16.
    const float tail_glob = (env->backend == MARMOT_BACKEND_METAL) ? 0.165f : 0.15f;
    test_matmul_quantized_via_generic_api_q4_0_sized(env, 7, 97, 11, 0.24f, tail_glob, 0.80f, 1);

    printf("\n=== Q4_1×Q8_0 Matmul (Gaussian, Small: K=64) ===\n");
    test_matmul_quantized_via_generic_api_q4_1_small(env);
    printf("\n=== Q4_1×Q8_0 Matmul (Gaussian, Medium: K=256) ===\n");
    test_matmul_quantized_via_generic_api_q4_1_medium(env);
    printf("\n=== Q4_1×Q8_0 Matmul (Gaussian, Production: K=4096) ===\n");
    test_matmul_quantized_via_generic_api_q4_1_production(env);

    printf("\n=== Q4_0×FP16 Matmul (Gaussian, Medium: K=256, out=FP32) ===\n");
    test_matmul_quantized_q4_0_fp16_fp32(env);
    printf("\n=== Q4_0×FP16 Matmul (Gaussian, Medium: K=256, out=FP16) ===\n");
    test_matmul_quantized_q4_0_fp16_fp16(env);
    printf("\n=== Q4_0×FP16 Matmul (Gaussian, Small Batch: N=3, M=17) ===\n");
    test_matmul_quantized_q4_0_fp16_small_batch(env);

    // Tail-K: FP16 activations (direct path), out=FP32 and out=FP16
    printf("\n=== Q4_0×FP16 Matmul (Gaussian, Tail K=97, out=FP32) ===\n");
    test_matmul_quantized_q4_0_fp16_path(env, 7, 97, 11, MARMOT_DTYPE_FLOAT32, 0.18f, 0.13f, 0.50f);
    printf("\n=== Q4_0×FP16 Matmul (Gaussian, Tail K=97, out=FP16) ===\n");
    test_matmul_quantized_q4_0_fp16_path(env, 7, 97, 11, MARMOT_DTYPE_FLOAT16, 0.20f, 0.14f, 0.55f);

    printf("\n=== Q5_0×FP16 Matmul (Gaussian, Medium: K=256, out=FP32) ===\n");
    test_matmul_quantized_q5_0_fp16_fp32(env);
    printf("\n=== Q5_0×FP16 Matmul (Gaussian, Medium: K=256, out=FP16) ===\n");
    test_matmul_quantized_q5_0_fp16_fp16(env);

    printf("\n=== Q5_1×FP16 Matmul (Gaussian, Medium: K=256, out=FP32) ===\n");
    test_matmul_quantized_q5_1_fp16_fp32(env);
    printf("\n=== Q5_1×FP16 Matmul (Gaussian, Medium: K=256, out=FP16) ===\n");
    test_matmul_quantized_q5_1_fp16_fp16(env);

    printf("\n=== Q8_0×FP16 Matmul (Gaussian, Medium: K=256, out=FP32) ===\n");
    test_matmul_quantized_q8_0_fp16_fp32(env);
    printf("\n=== Q8_0×FP16 Matmul (Gaussian, Medium: K=256, out=FP16) ===\n");
    test_matmul_quantized_q8_0_fp16_fp16(env);

    // Tail-K for other formats (spot checks)
    printf("\n=== Q5_0×FP16 Matmul (Gaussian, Tail K=97, out=FP32) ===\n");
    test_matmul_quantized_qx_fp16_path(
        env, MARMOT_QUANT_KIND_Q5_0, 5, 97, 13, MARMOT_DTYPE_FLOAT32, 0.18f, 0.12f, 0.55f
    );
    printf("\n=== Q8_0×FP16 Matmul (Gaussian, Tail K=97, out=FP32) ===\n");
    test_matmul_quantized_qx_fp16_path(
        env, MARMOT_QUANT_KIND_Q8_0, 5, 97, 13, MARMOT_DTYPE_FLOAT32, 0.16f, 0.11f, 0.45f
    );
}

static void test_matmul_quantized_default(void **state) {
    run_matmul_quantized_suite((marmot_test_env_t *)(*state));
}

static void test_matmul_generic_api(void **state) {
    run_matmul_via_generic_api_suite((marmot_test_env_t *)(*state));
}

static void test_matmul_quantized_epilogue_bias_relu_f32(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);

    const matmul_golden_case_t *tc = matmul_quantized_default_case();
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(tc->kind);
    assert_non_null(traits);

    size_t shape_input[] = {tc->N, tc->K};
    size_t shape_weight[] = {tc->M, tc->K};
    size_t shape_output[] = {tc->N, tc->M};

    marmot_tensor_t *input = marmot_test_tensor_from_array(env, shape_input, 2, tc->input_f32);
    marmot_tensor_t *weight = marmot_tensor_create_quantized(env->ctx, shape_weight, 2, tc->kind);
    assert_non_null(weight);
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, weight, tc->weights, tc->weight_bytes), MARMOT_SUCCESS
    );

    marmot_tensor_t *output = marmot_tensor_create(env->ctx, shape_output, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(output);

    float *bias_values = (float *)malloc(tc->M * sizeof(float));
    assert_non_null(bias_values);
    for (size_t i = 0; i < tc->M; ++i) {
        bias_values[i] = 0.05f * (float)i - 1.0f;
    }
    size_t shape_bias[] = {tc->M};
    marmot_tensor_t *bias = marmot_test_tensor_from_array(env, shape_bias, 1, bias_values);

    marmot_matmul_epilogue_t ep = {
        .bias = bias,
    };

    assert_int_equal(marmot_linear(env->ctx, input, weight, &ep, output), MARMOT_SUCCESS);
    marmot_tensor_t *relu_out = marmot_tensor_create(env->ctx, shape_output, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(relu_out);
    assert_int_equal(marmot_relu(env->ctx, output, relu_out), MARMOT_SUCCESS);

    const size_t elem_count = tc->N * tc->M;
    float *actual = (float *)malloc(elem_count * sizeof(float));
    assert_non_null(actual);
    assert_int_equal(
        marmot_tensor_copy_to_host_buffer(env->ctx, relu_out, actual, elem_count * sizeof(float)), MARMOT_SUCCESS
    );

    float *expected = (float *)malloc(elem_count * sizeof(float));
    assert_non_null(expected);
    const bool metal_basic_ep = env->backend == MARMOT_BACKEND_METAL &&
        (tc->kind == MARMOT_QUANT_KIND_Q4_0 || tc->kind == MARMOT_QUANT_KIND_Q4_1 ||
         tc->kind == MARMOT_QUANT_KIND_Q5_0 || tc->kind == MARMOT_QUANT_KIND_Q5_1 ||
         tc->kind == MARMOT_QUANT_KIND_Q8_0 || tc->kind == MARMOT_QUANT_KIND_Q8_1);
    memcpy(expected, metal_basic_ep ? tc->expected_f16 : tc->expected_f32, elem_count * sizeof(float));
    matmul_quantized_expect_relu(expected, tc->N, tc->M, bias_values);

    marmot_test_expect_close_array(actual, expected, elem_count, 5e-3f);

    free(expected);
    free(actual);
    free(bias_values);
    marmot_test_tensor_destroy_all(5, relu_out, output, bias, weight, input);
}

static void test_matmul_quantized_epilogue_bias_relu_fp16_out(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);

    const matmul_golden_case_t *tc = matmul_quantized_default_case();
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(tc->kind);
    assert_non_null(traits);

    size_t shape_input[] = {tc->N, tc->K};
    size_t shape_weight[] = {tc->M, tc->K};
    size_t shape_output[] = {tc->N, tc->M};

    marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape_input, 2, MARMOT_DTYPE_FLOAT16);
    assert_non_null(input);
    const size_t input_bytes = tc->N * tc->K * sizeof(uint16_t);
    assert_int_equal(marmot_tensor_copy_from_host_buffer(env->ctx, input, tc->input_f16, input_bytes), MARMOT_SUCCESS);

    marmot_tensor_t *weight = marmot_tensor_create_quantized(env->ctx, shape_weight, 2, tc->kind);
    assert_non_null(weight);
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, weight, tc->weights, tc->weight_bytes), MARMOT_SUCCESS
    );

    marmot_tensor_t *output = marmot_tensor_create(env->ctx, shape_output, 2, MARMOT_DTYPE_FLOAT16);
    assert_non_null(output);

    float *bias_values = (float *)malloc(tc->M * sizeof(float));
    assert_non_null(bias_values);
    for (size_t i = 0; i < tc->M; ++i) {
        bias_values[i] = -0.02f * (float)i + 0.5f;
    }
    size_t shape_bias[] = {tc->M};
    marmot_tensor_t *bias = marmot_test_tensor_from_array(env, shape_bias, 1, bias_values);

    marmot_matmul_epilogue_t ep = {
        .bias = bias,
    };

    assert_int_equal(marmot_linear(env->ctx, input, weight, &ep, output), MARMOT_SUCCESS);
    marmot_tensor_t *relu_out = marmot_tensor_create(env->ctx, shape_output, 2, MARMOT_DTYPE_FLOAT16);
    assert_non_null(relu_out);
    assert_int_equal(marmot_relu(env->ctx, output, relu_out), MARMOT_SUCCESS);

    const size_t elem_count = tc->N * tc->M;
    marmot_float16_t *result_bits = (marmot_float16_t *)malloc(elem_count * sizeof(marmot_float16_t));
    assert_non_null(result_bits);
    assert_int_equal(
        marmot_tensor_copy_to_host_buffer(env->ctx, relu_out, result_bits, elem_count * sizeof(marmot_float16_t)),
        MARMOT_SUCCESS
    );

    float *actual = (float *)malloc(elem_count * sizeof(float));
    assert_non_null(actual);
    for (size_t i = 0; i < elem_count; ++i) {
        actual[i] = (float)marmot_float16_to_native(result_bits[i]);
    }

    float *expected = (float *)malloc(elem_count * sizeof(float));
    assert_non_null(expected);
    memcpy(expected, tc->expected_f16, elem_count * sizeof(float));
    matmul_quantized_expect_relu(expected, tc->N, tc->M, bias_values);
    for (size_t i = 0; i < elem_count; ++i) {
        marmot_float16_t rounded = marmot_native_to_float16((_Float16)expected[i]);
        expected[i] = (float)marmot_float16_to_native(rounded);
    }

    marmot_test_expect_close_array(actual, expected, elem_count, 5e-2f);

    free(expected);
    free(actual);
    free(result_bits);
    free(bias_values);
    marmot_test_tensor_destroy_all(5, relu_out, output, bias, weight, input);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(
            test_matmul_quantized_default, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_generic_api, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_llama_goldens, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_llama_ref_quant_parity_q4_0, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_llama_ref_quant_parity_q4_1, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_llama_ref_quant_parity_q5_0, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_llama_ref_quant_parity_q5_1, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_llama_ref_quant_parity_q8_0, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_quantized_epilogue_bias_relu_f32, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_quantized_epilogue_bias_relu_fp16_out, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
