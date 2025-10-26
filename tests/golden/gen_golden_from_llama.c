// Generate golden quantization, embedding fixtures, and vec_dot data from llama.cpp reference
#include "marmot/quant_block.h"
#include "marmot/types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <assert.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <math.h>
#include <string.h>

#include "ggml-quants.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/marmot.h"
#include "marmot/tensor.h"
#ifdef __cplusplus
}
#endif

static const float test_input[96] = {
    -2.5f,  -1.0f,  -0.5f,  0.0f,   0.5f,    1.0f,    1.5f,    2.0f,    2.5f,   3.0f,   -3.0f,
    -1.5f,  0.25f,  0.75f,  1.25f,  1.75f,   -0.25f,  -0.75f,  -1.25f,  -1.75f, 2.25f,  2.75f,
    3.25f,  3.75f,  -2.25f, -2.75f, -3.25f,  -3.75f,  4.0f,    4.5f,    5.0f,   -4.0f,

    0.001f, 0.002f, 0.003f, 0.004f, -0.001f, -0.002f, -0.003f, -0.004f, 0.01f,  0.02f,  0.03f,
    0.04f,  -0.01f, -0.02f, -0.03f, -0.04f,  0.1f,    0.2f,    0.3f,    0.4f,   -0.1f,  -0.2f,
    -0.3f,  -0.4f,  0.05f,  0.15f,  0.25f,   0.35f,   -0.05f,  -0.15f,  -0.25f, -0.35f,

    10.0f,  20.0f,  30.0f,  40.0f,  -10.0f,  -20.0f,  -30.0f,  -40.0f,  15.0f,  25.0f,  35.0f,
    45.0f,  -15.0f, -25.0f, -35.0f, -45.0f,  12.5f,   17.5f,   22.5f,   27.5f,  -12.5f, -17.5f,
    -22.5f, -27.5f, 32.5f,  37.5f,  42.5f,   47.5f,   -32.5f,  -37.5f,  -42.5f, -47.5f,
};

static marmot_context_t *g_marmot_ctx = nullptr;

static void write_embedding_fixture_q4_0_ragged(const char *fname);
static void gen_matmul_goldens(FILE *fp);
static void gen_float_ops_goldens(FILE *fp);

static void print_float_array(FILE *fp, const char *name, const float *values, size_t count) {
    fprintf(fp, "static const float %s[%zu] = {\n", name, count);
    for (size_t i = 0; i < count; ++i) {
        if (i % 8 == 0) {
            fprintf(fp, "    ");
        }
        fprintf(fp, "% .9ff", values[i]);
        if (i + 1 < count) {
            fprintf(fp, ", ");
        }
        if ((i + 1) % 8 == 0) {
            fprintf(fp, "\n");
        }
    }
    if (count % 8 != 0) {
        fprintf(fp, "\n");
    }
    fprintf(fp, "};\n\n");
}

static void print_double_array(FILE *fp, const char *name, const double *values, size_t count) {
    fprintf(fp, "static const double %s[%zu] = {\n", name, count);
    for (size_t i = 0; i < count; ++i) {
        if (i % 6 == 0) {
            fprintf(fp, "    ");
        }
        fprintf(fp, "% .17lf", values[i]);
        if (i + 1 < count) {
            fprintf(fp, ", ");
        }
        if ((i + 1) % 6 == 0) {
            fprintf(fp, "\n");
        }
    }
    if (count % 6 != 0) {
        fprintf(fp, "\n");
    }
    fprintf(fp, "};\n\n");
}

static float vec_dot_f32(const float *lhs, const float *rhs, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += (double)lhs[i] * (double)rhs[i];
    }
    return (float)acc;
}

static void print_u8_array(FILE *fp, const char *name, const uint8_t *values, size_t count) {
    fprintf(fp, "static const uint8_t %s[%zu] = {\n", name, count);
    for (size_t i = 0; i < count; ++i) {
        if (i % 16 == 0) {
            fprintf(fp, "    ");
        }
        fprintf(fp, "0x%02x", values[i]);
        if (i + 1 < count) {
            fprintf(fp, ", ");
        }
        if ((i + 1) % 16 == 0) {
            fprintf(fp, "\n");
        }
    }
    if (count % 16 != 0) {
        fprintf(fp, "\n");
    }
    fprintf(fp, "};\n\n");
}

static void print_u16_array(FILE *fp, const char *name, const uint16_t *values, size_t count) {
    fprintf(fp, "static const uint16_t %s[%zu] = {\n", name, count);
    for (size_t i = 0; i < count; ++i) {
        if (i % 12 == 0) {
            fprintf(fp, "    ");
        }
        fprintf(fp, "0x%04x", values[i]);
        if (i + 1 < count) {
            fprintf(fp, ", ");
        }
        if ((i + 1) % 12 == 0) {
            fprintf(fp, "\n");
        }
    }
    if (count % 12 != 0) {
        fprintf(fp, "\n");
    }
    fprintf(fp, "};\n\n");
}

static void fill_activation_pattern(float *dst, int N, int K) {
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            int idx = (n * 29 + k * 17) % (int)(sizeof(test_input) / sizeof(test_input[0]));
            dst[(size_t)n * (size_t)K + (size_t)k] = test_input[idx];
        }
    }
}

static void fill_weight_pattern(float *dst, int M, int K) {
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            int idx = (m * 23 + k * 31 + 7) % (int)(sizeof(test_input) / sizeof(test_input[0]));
            dst[(size_t)m * (size_t)K + (size_t)k] = test_input[idx];
        }
    }
}

static void convert_fp32_to_fp16(const float *src, ggml_fp16_t *dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = ggml_fp32_to_fp16(src[i]);
    }
}

static void fill_rhs_pattern(float *dst, int K, int N) {
    float *tmp = (float *)malloc((size_t)N * (size_t)K * sizeof(float));
    if (tmp == nullptr) {
        fprintf(stderr, "Allocation failure in fill_rhs_pattern\n");
        exit(EXIT_FAILURE);
    }
    fill_activation_pattern(tmp, N, K);
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            dst[(size_t)k * (size_t)N + (size_t)n] = tmp[(size_t)n * (size_t)K + (size_t)k];
        }
    }
    free(tmp);
}

static void matmul_reference(
    const float *a /* (M,K) */, const float *b /* (K,N) */, int M, int K, int N, float *out /* (M,N) */
) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += (double)a[(size_t)m * (size_t)K + (size_t)k] * (double)b[(size_t)k * (size_t)N + (size_t)n];
            }
            out[(size_t)m * (size_t)N + (size_t)n] = (float)acc;
        }
    }
}

static void layernorm_reference(
    const float *x, const float *residual, const float *weight, const float *bias, size_t rows, size_t cols, float eps,
    float *out
) {
    float *buffer = (float *)malloc(cols * sizeof(float));
    if (buffer == nullptr) {
        fprintf(stderr, "Allocation failure in layernorm_reference\n");
        exit(EXIT_FAILURE);
    }

    for (size_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (size_t col = 0; col < cols; ++col) {
            float value = x[row * cols + col];
            if (residual != nullptr) {
                value += residual[row * cols + col];
            }
            buffer[col] = value;
            sum += (double)value;
        }
        const float mean = (float)(sum / (double)cols);

        double var_acc = 0.0;
        for (size_t col = 0; col < cols; ++col) {
            const double diff = (double)buffer[col] - (double)mean;
            var_acc += diff * diff;
        }
        const float variance = (float)(var_acc / (double)cols);
        const float inv_std = 1.0f / sqrtf(variance + eps);

        for (size_t col = 0; col < cols; ++col) {
            float normalized = (buffer[col] - mean) * inv_std;
            if (weight != nullptr) {
                normalized *= weight[col];
            }
            if (bias != nullptr) {
                normalized += bias[col];
            }
            out[row * cols + col] = normalized;
        }
    }

    free(buffer);
}

static void rmsnorm_reference(
    const float *x, const float *residual, const float *weight, size_t rows, size_t cols, float eps, float *out
) {
    for (size_t row = 0; row < rows; ++row) {
        double sum_sq = 0.0;
        for (size_t col = 0; col < cols; ++col) {
            float value = x[row * cols + col];
            if (residual != nullptr) {
                value += residual[row * cols + col];
            }
            sum_sq += (double)value * (double)value;
            out[row * cols + col] = value;
        }
        const float mean_sq = (float)(sum_sq / (double)cols);
        const float inv_rms = 1.0f / sqrtf(mean_sq + eps);
        for (size_t col = 0; col < cols; ++col) {
            float normalized = out[row * cols + col] * inv_rms;
            if (weight != nullptr) {
                normalized *= weight[col];
            }
            out[row * cols + col] = normalized;
        }
    }
}

static float gelu_erf(float x) {
    const float inv_sqrt2 = 0.7071067811865475f;
    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}

static float gelu_tanh_approx(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static float silu_fn(float x) {
    return x / (1.0f + expf(-x));
}

static float sigmoid_fn(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

static float mish_fn(float x) {
    float abs_x = fabsf(x);
    float softplus = log1pf(expf(-abs_x)) + fmaxf(x, 0.0f);
    return x * tanhf(softplus);
}

static float elu_fn(float x, float alpha) {
    return x >= 0.0f ? x : alpha * (expf(x) - 1.0f);
}

static float selu_fn(float x, float alpha, float lambda) {
    float inner = x >= 0.0f ? x : alpha * (expf(x) - 1.0f);
    return lambda * inner;
}

static float leaky_relu_fn(float x, float slope) {
    return x >= 0.0f ? x : slope * x;
}

static float prelu_fn(float x, float slope) {
    return x >= 0.0f ? x : slope * x;
}

static void quantize_weight_row(enum ggml_type type, const float *row, int K, uint8_t *dst_row) {
    if (type == GGML_TYPE_Q8_1) {
        quantize_row_q8_1_ref(row, (block_q8_1 *)dst_row, K);
    } else if (type == GGML_TYPE_Q8_K) {
        quantize_row_q8_K_ref(row, (block_q8_K *)dst_row, K);
    } else {
        ggml_quantize_chunk(type, row, dst_row, 0, 1, K, nullptr);
    }
}

static bool ensure_marmot_context(void) {
    if (g_marmot_ctx != nullptr) {
        return true;
    }
    g_marmot_ctx = marmot_init(MARMOT_BACKEND_CPU);
    if (g_marmot_ctx == nullptr) {
        fprintf(stderr, "Failed to initialise Marmot CPU context\n");
        return false;
    }
    return true;
}

typedef struct {
    const char *suffix;
    const char *upper;
    enum ggml_type ggml_type;
    marmot_quant_kind_t quant_kind;
    const char *quant_macro;
    int N;
    int K;
    int M;
} matmul_case_t;

static bool compute_matmul_reference(
    const matmul_case_t *tc, const uint8_t *weight_bytes, size_t row_size, const float *input_f32,
    const ggml_fp16_t *input_f16, float *out_from_f32, float *out_from_f16
) {
    if (!ensure_marmot_context()) {
        return false;
    }

    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(tc->quant_kind);
    if (traits == nullptr) {
        fprintf(stderr, "Unsupported quant kind for matmul golden %s\n", tc->suffix);
        return false;
    }

    const size_t input_elems = (size_t)tc->N * (size_t)tc->K;
    const size_t output_elems = (size_t)tc->N * (size_t)tc->M;
    const size_t weight_bytes_total = (size_t)tc->M * row_size;
    const size_t storage_elem_size = marmot_dtype_size(traits->storage_dtype);
    if (storage_elem_size == 0 || weight_bytes_total % storage_elem_size != 0) {
        fprintf(stderr, "Unexpected storage size for matmul golden %s\n", tc->suffix);
        return false;
    }

    size_t input_shape[2] = {(size_t)tc->N, (size_t)tc->K};
    size_t weight_storage_shape[1] = {weight_bytes_total / storage_elem_size};

    marmot_tensor_t *input_tensor_f32 = marmot_tensor_create(nullptr, input_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *input_tensor_f16 = marmot_tensor_create(nullptr, input_shape, 2, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *weight_tensor =
        marmot_tensor_create(weight_storage_shape, 1, traits->storage_dtype, MARMOT_BACKEND_CPU);
    marmot_tensor_t *output_tensor =
        marmot_tensor_create((size_t[]){(size_t)tc->N, (size_t)tc->M}, 2, MARMOT_DTYPE_FLOAT32, MARMOT_BACKEND_CPU);

    if (input_tensor_f32 == nullptr || input_tensor_f16 == nullptr || weight_tensor == nullptr ||
        output_tensor == nullptr) {
        fprintf(stderr, "Failed to allocate Marmot tensors for matmul golden %s\n", tc->suffix);
        marmot_tensor_destroy(output_tensor);
        marmot_tensor_destroy(weight_tensor);
        marmot_tensor_destroy(input_tensor_f16);
        marmot_tensor_destroy(input_tensor_f32);
        return false;
    }

    if (marmot_tensor_copy_from_host_buffer(g_marmot_ctx, input_tensor_f32, input_f32, input_elems * sizeof(float)) !=
            MARMOT_SUCCESS ||
        marmot_tensor_copy_from_host_buffer(
            g_marmot_ctx, input_tensor_f16, input_f16, input_elems * sizeof(uint16_t)
        ) != MARMOT_SUCCESS ||
        marmot_tensor_copy_from_host_buffer(g_marmot_ctx, weight_tensor, weight_bytes, weight_bytes_total) !=
            MARMOT_SUCCESS) {
        fprintf(stderr, "Failed to populate Marmot tensors for matmul golden %s\n", tc->suffix);
        marmot_tensor_destroy(output_tensor);
        marmot_tensor_destroy(weight_tensor);
        marmot_tensor_destroy(input_tensor_f16);
        marmot_tensor_destroy(input_tensor_f32);
        return false;
    }

    weight_tensor->quant_kind = tc->quant_kind;
    weight_tensor->quant_layout = MARMOT_QUANT_LAYOUT_GGUF;
    weight_tensor->shape.ndim = 2;
    weight_tensor->shape.shape[0] = (size_t)tc->M;
    weight_tensor->shape.shape[1] = (size_t)tc->K;
    weight_tensor->shape.strides[0] = (size_t)tc->K;
    weight_tensor->shape.strides[1] = 1;

    if (marmot_linear(g_marmot_ctx, input_tensor_f32, weight_tensor, nullptr, output_tensor) != MARMOT_SUCCESS) {
        fprintf(
            stderr, "marmot_matmul (fp32) failed for %s: %s\n", tc->suffix, marmot_error_string(marmot_get_last_error())
        );
        marmot_tensor_destroy(output_tensor);
        marmot_tensor_destroy(weight_tensor);
        marmot_tensor_destroy(input_tensor_f16);
        marmot_tensor_destroy(input_tensor_f32);
        return false;
    }
    if (marmot_tensor_copy_to_host_buffer(g_marmot_ctx, output_tensor, out_from_f32, output_elems * sizeof(float)) !=
        MARMOT_SUCCESS) {
        fprintf(stderr, "Failed to fetch fp32 output for %s\n", tc->suffix);
        marmot_tensor_destroy(output_tensor);
        marmot_tensor_destroy(weight_tensor);
        marmot_tensor_destroy(input_tensor_f16);
        marmot_tensor_destroy(input_tensor_f32);
        return false;
    }

    if (marmot_linear(g_marmot_ctx, input_tensor_f16, weight_tensor, nullptr, output_tensor) != MARMOT_SUCCESS) {
        fprintf(
            stderr, "marmot_matmul (fp16) failed for %s: %s\n", tc->suffix, marmot_error_string(marmot_get_last_error())
        );
        marmot_tensor_destroy(output_tensor);
        marmot_tensor_destroy(weight_tensor);
        marmot_tensor_destroy(input_tensor_f16);
        marmot_tensor_destroy(input_tensor_f32);
        return false;
    }
    if (marmot_tensor_copy_to_host_buffer(g_marmot_ctx, output_tensor, out_from_f16, output_elems * sizeof(float)) !=
        MARMOT_SUCCESS) {
        fprintf(stderr, "Failed to fetch fp16 output for %s\n", tc->suffix);
        marmot_tensor_destroy(output_tensor);
        marmot_tensor_destroy(weight_tensor);
        marmot_tensor_destroy(input_tensor_f16);
        marmot_tensor_destroy(input_tensor_f32);
        return false;
    }

    marmot_tensor_destroy(output_tensor);
    marmot_tensor_destroy(weight_tensor);
    marmot_tensor_destroy(input_tensor_f16);
    marmot_tensor_destroy(input_tensor_f32);
    return true;
}

static void dequantize_into(enum ggml_type type, const void *src, float *dst, int64_t count) {
    switch (type) {
    case GGML_TYPE_Q4_0:
        dequantize_row_q4_0((const block_q4_0 *)src, dst, count);
        return;
    case GGML_TYPE_Q4_1:
        dequantize_row_q4_1((const block_q4_1 *)src, dst, count);
        return;
    case GGML_TYPE_Q5_0:
        dequantize_row_q5_0((const block_q5_0 *)src, dst, count);
        return;
    case GGML_TYPE_Q5_1:
        dequantize_row_q5_1((const block_q5_1 *)src, dst, count);
        return;
    case GGML_TYPE_Q8_0:
        dequantize_row_q8_0((const block_q8_0 *)src, dst, count);
        return;
    case GGML_TYPE_Q8_1: {
        const block_q8_1 *blocks = (const block_q8_1 *)src;
        int64_t produced = 0;
        int64_t block_index = 0;
        while (produced < count) {
            const float d = ggml_fp16_to_fp32(blocks[block_index].d);
            const int64_t remaining = count - produced;
            const int64_t chunk = remaining < QK8_1 ? remaining : QK8_1;
            for (int64_t j = 0; j < chunk; ++j) {
                dst[produced + j] = d * (float)blocks[block_index].qs[j];
            }
            produced += chunk;
            ++block_index;
        }
        return;
    }
    case GGML_TYPE_Q2_K:
        dequantize_row_q2_K((const block_q2_K *)src, dst, count);
        return;
    case GGML_TYPE_Q3_K:
        dequantize_row_q3_K((const block_q3_K *)src, dst, count);
        return;
    case GGML_TYPE_Q4_K:
        dequantize_row_q4_K((const block_q4_K *)src, dst, count);
        return;
    case GGML_TYPE_Q5_K:
        dequantize_row_q5_K((const block_q5_K *)src, dst, count);
        return;
    case GGML_TYPE_Q6_K:
        dequantize_row_q6_K((const block_q6_K *)src, dst, count);
        return;
    case GGML_TYPE_Q8_K: {
        const marmot_q8_k_block_t *blocks = (const marmot_q8_k_block_t *)src;
        int64_t produced = 0;
        int64_t block_index = 0;
        while (produced < count) {
            const marmot_q8_k_block_t *block = &blocks[block_index];
            const float d = block->d;
            const int64_t remaining = count - produced;
            const int64_t chunk = remaining < (int64_t)MARMOT_QK_K_VALUES ? remaining : (int64_t)MARMOT_QK_K_VALUES;
            for (int64_t j = 0; j < chunk; ++j) {
                dst[produced + j] = d * (float)block->qs[j];
            }
            produced += chunk;
            ++block_index;
        }
        return;
    }
    default:
        fprintf(stderr, "dequantize_into: unsupported type %d\n", type);
        exit(EXIT_FAILURE);
    }
}

static void print_header(FILE *fp) {
    fprintf(fp, "// Generated from llama.cpp reference implementation\n");
    fprintf(fp, "// DO NOT EDIT - regenerate with: cd tests/golden && ./generate_from_llama.sh\n");
    fprintf(fp, "#pragma once\n");
    fprintf(fp, "#include \"marmot/quant_block.h\"\n\n");
}

static void print_input(FILE *fp) {
    fprintf(fp, "static const float g_llama_test_input[96] = {\n");
    for (int i = 0; i < 96; i++) {
        if (i % 8 == 0)
            fprintf(fp, "    ");
        fprintf(fp, "%6.3ff", test_input[i]);
        if (i < 95)
            fprintf(fp, ", ");
        if ((i + 1) % 8 == 0)
            fprintf(fp, "\n");
    }
    fprintf(fp, "};\n\n");
}

static void gen_q4_0(FILE *fp) {
    size_t qsize = ggml_row_size(GGML_TYPE_Q4_0, 96);
    void *qdata = malloc(qsize);

    ggml_quantize_chunk(GGML_TYPE_Q4_0, test_input, qdata, 0, 1, 96, nullptr);
    float dequant[96];
    dequantize_into(GGML_TYPE_Q4_0, qdata, dequant, 96);

    fprintf(fp, "static const marmot_q4_0_block_t g_llama_q4_0[3] = {\n");
    const uint8_t *bytes = (const uint8_t *)qdata;
    for (int b = 0; b < 3; b++) {
        const uint8_t *block = bytes + b * 18;
        uint16_t scale = *(uint16_t *)block;
        fprintf(fp, "    {.scale = {.bits = 0x%04x}, .qs = {", scale);
        for (int i = 0; i < 16; i++) {
            fprintf(fp, "0x%02x", block[2 + i]);
            if (i < 15)
                fprintf(fp, ", ");
        }
        fprintf(fp, "}},\n");
    }
    fprintf(fp, "};\n\n");
    print_float_array(fp, "g_llama_q4_0_dequant", dequant, 96);
    free(qdata);
}

static void gen_q4_1(FILE *fp) {
    size_t qsize = ggml_row_size(GGML_TYPE_Q4_1, 96);
    void *qdata = malloc(qsize);

    ggml_quantize_chunk(GGML_TYPE_Q4_1, test_input, qdata, 0, 1, 96, nullptr);
    float dequant[96];
    dequantize_into(GGML_TYPE_Q4_1, qdata, dequant, 96);

    fprintf(fp, "static const marmot_q4_1_block_t g_llama_q4_1[3] = {\n");
    const uint8_t *bytes = (const uint8_t *)qdata;
    for (int b = 0; b < 3; b++) {
        const uint8_t *block = bytes + b * 20;
        uint16_t scale = *(uint16_t *)block;
        uint16_t min = *(uint16_t *)(block + 2);
        fprintf(fp, "    {.scale = {.bits = 0x%04x}, .min = {.bits = 0x%04x}, .qs = {", scale, min);
        for (int i = 0; i < 16; i++) {
            fprintf(fp, "0x%02x", block[4 + i]);
            if (i < 15)
                fprintf(fp, ", ");
        }
        fprintf(fp, "}},\n");
    }
    fprintf(fp, "};\n\n");
    print_float_array(fp, "g_llama_q4_1_dequant", dequant, 96);
    free(qdata);
}

static void gen_q5_0(FILE *fp) {
    size_t qsize = ggml_row_size(GGML_TYPE_Q5_0, 96);
    void *qdata = malloc(qsize);

    ggml_quantize_chunk(GGML_TYPE_Q5_0, test_input, qdata, 0, 1, 96, nullptr);
    float dequant[96];
    dequantize_into(GGML_TYPE_Q5_0, qdata, dequant, 96);

    fprintf(fp, "static const marmot_q5_0_block_t g_llama_q5_0[3] = {\n");
    const uint8_t *bytes = (const uint8_t *)qdata;
    for (int b = 0; b < 3; b++) {
        const uint8_t *block = bytes + b * 22;
        uint16_t scale = *(uint16_t *)block;
        fprintf(fp, "    {.scale = {.bits = 0x%04x}, .qh = {", scale);
        for (int i = 0; i < 4; i++) {
            fprintf(fp, "0x%02x", block[2 + i]);
            if (i < 3)
                fprintf(fp, ", ");
        }
        fprintf(fp, "}, .qs = {");
        for (int i = 0; i < 16; i++) {
            fprintf(fp, "0x%02x", block[6 + i]);
            if (i < 15)
                fprintf(fp, ", ");
        }
        fprintf(fp, "}},\n");
    }
    fprintf(fp, "};\n\n");
    print_float_array(fp, "g_llama_q5_0_dequant", dequant, 96);
    free(qdata);
}

static void gen_q5_1(FILE *fp) {
    size_t qsize = ggml_row_size(GGML_TYPE_Q5_1, 96);
    void *qdata = malloc(qsize);

    ggml_quantize_chunk(GGML_TYPE_Q5_1, test_input, qdata, 0, 1, 96, nullptr);
    float dequant[96];
    dequantize_into(GGML_TYPE_Q5_1, qdata, dequant, 96);

    fprintf(fp, "static const marmot_q5_1_block_t g_llama_q5_1[3] = {\n");
    const uint8_t *bytes = (const uint8_t *)qdata;
    for (int b = 0; b < 3; b++) {
        const uint8_t *block = bytes + b * 24;
        uint16_t scale = *(uint16_t *)block;
        uint16_t min = *(uint16_t *)(block + 2);
        fprintf(fp, "    {.scale = {.bits = 0x%04x}, .min = {.bits = 0x%04x}, .qh = {", scale, min);
        for (int i = 0; i < 4; i++) {
            fprintf(fp, "0x%02x", block[4 + i]);
            if (i < 3)
                fprintf(fp, ", ");
        }
        fprintf(fp, "}, .qs = {");
        for (int i = 0; i < 16; i++) {
            fprintf(fp, "0x%02x", block[8 + i]);
            if (i < 15)
                fprintf(fp, ", ");
        }
        fprintf(fp, "}},\n");
    }
    fprintf(fp, "};\n\n");
    print_float_array(fp, "g_llama_q5_1_dequant", dequant, 96);
    free(qdata);
}

static void gen_q8_0(FILE *fp) {
    size_t qsize = ggml_row_size(GGML_TYPE_Q8_0, 96);
    void *qdata = malloc(qsize);

    ggml_quantize_chunk(GGML_TYPE_Q8_0, test_input, qdata, 0, 1, 96, nullptr);
    float dequant[96];
    dequantize_into(GGML_TYPE_Q8_0, qdata, dequant, 96);

    fprintf(fp, "static const marmot_q8_0_block_t g_llama_q8_0[3] = {\n");
    const uint8_t *bytes = (const uint8_t *)qdata;
    for (int b = 0; b < 3; b++) {
        const uint8_t *block = bytes + b * 34;
        uint16_t scale = *(uint16_t *)block;
        fprintf(fp, "    {.scale = {.bits = 0x%04x}, .qs = {", scale);
        const int8_t *qs = (int8_t *)(block + 2);
        for (int i = 0; i < 32; i++) {
            if (i % 16 == 0 && i > 0)
                fprintf(fp, "\n        ");
            fprintf(fp, "%4d", qs[i]);
            if (i < 31)
                fprintf(fp, ", ");
        }
        fprintf(fp, "}},\n");
    }
    fprintf(fp, "};\n\n");
    print_float_array(fp, "g_llama_q8_0_dequant", dequant, 96);
    free(qdata);
}

static void gen_q8_1(FILE *fp) {
    size_t qsize = ggml_row_size(GGML_TYPE_Q8_1, 96);
    void *qdata = malloc(qsize);

    // Use direct quantize function instead of ggml_quantize_chunk
    quantize_row_q8_1_ref(test_input, (block_q8_1 *)qdata, 96);

    fprintf(fp, "static const marmot_q8_1_block_t g_llama_q8_1[3] = {\n");
    const uint8_t *bytes = (const uint8_t *)qdata;
    for (int b = 0; b < 3; b++) {
        const uint8_t *block = bytes + b * 36; // Q8_1 is 36 bytes per block
        uint16_t scale = *(uint16_t *)block;
        uint16_t sum = *(uint16_t *)(block + 2);
        fprintf(fp, "    {.scale = {.bits = 0x%04x}, .sum = {.bits = 0x%04x}, .qs = {", scale, sum);
        const int8_t *qs = (int8_t *)(block + 4);
        for (int i = 0; i < 32; i++) {
            if (i % 16 == 0 && i > 0)
                fprintf(fp, "\n        ");
            fprintf(fp, "%4d", qs[i]);
            if (i < 31)
                fprintf(fp, ", ");
        }
        fprintf(fp, "}},\n");
    }
    fprintf(fp, "};\n\n");
    float dequant[96];
    dequantize_into(GGML_TYPE_Q8_1, qdata, dequant, 96);
    print_float_array(fp, "g_llama_q8_1_dequant", dequant, 96);
    free(qdata);
}

// K-quant generation functions (256 values per block)
static void gen_q2_k(FILE *fp) {
    size_t qsize = ggml_row_size(GGML_TYPE_Q2_K, 256);
    void *qdata = malloc(qsize);

    // Expand test_input to 256 values by repeating
    float expanded[256];
    for (int i = 0; i < 256; i++) {
        expanded[i] = test_input[i % 96];
    }

    ggml_quantize_chunk(GGML_TYPE_Q2_K, expanded, qdata, 0, 1, 256, nullptr);
    float dequant[256];
    dequantize_into(GGML_TYPE_Q2_K, qdata, dequant, 256);

    fprintf(fp, "// Q2_K: 84 bytes per 256-value block\n");
    fprintf(fp, "static const marmot_q2_k_block_t g_llama_q2_k[1] = {\n");
    const uint8_t *bytes = (const uint8_t *)qdata;
    fprintf(fp, "    {.scales = {");
    for (int i = 0; i < 16; i++) {
        fprintf(fp, "0x%02x%s", bytes[i], i < 15 ? ", " : "");
    }
    fprintf(fp, "},\n     .qs = {");
    for (int i = 0; i < 64; i++) {
        if (i % 16 == 0 && i > 0)
            fprintf(fp, "\n            ");
        fprintf(fp, "0x%02x%s", bytes[16 + i], i < 63 ? ", " : "");
    }
    uint16_t d = *(uint16_t *)(bytes + 80);
    uint16_t dmin = *(uint16_t *)(bytes + 82);
    fprintf(fp, "},\n     .d = {.bits = 0x%04x}, .dmin = {.bits = 0x%04x}},\n", d, dmin);
    fprintf(fp, "};\n\n");
    print_float_array(fp, "g_llama_q2_k_dequant", dequant, 256);
    free(qdata);
}

static void gen_q3_k(FILE *fp) {
    size_t qsize = ggml_row_size(GGML_TYPE_Q3_K, 256);
    void *qdata = malloc(qsize);

    float expanded[256];
    for (int i = 0; i < 256; i++) {
        expanded[i] = test_input[i % 96];
    }

    ggml_quantize_chunk(GGML_TYPE_Q3_K, expanded, qdata, 0, 1, 256, nullptr);
    float dequant[256];
    dequantize_into(GGML_TYPE_Q3_K, qdata, dequant, 256);

    fprintf(fp, "// Q3_K: 110 bytes per 256-value block\n");
    fprintf(fp, "static const marmot_q3_k_block_t g_llama_q3_k[1] = {\n");
    const uint8_t *bytes = (const uint8_t *)qdata;
    fprintf(fp, "    {.hmask = {");
    for (int i = 0; i < 32; i++) {
        if (i % 16 == 0 && i > 0)
            fprintf(fp, "\n              ");
        fprintf(fp, "0x%02x%s", bytes[i], i < 31 ? ", " : "");
    }
    fprintf(fp, "},\n     .qs = {");
    for (int i = 0; i < 64; i++) {
        if (i % 16 == 0 && i > 0)
            fprintf(fp, "\n            ");
        fprintf(fp, "0x%02x%s", bytes[32 + i], i < 63 ? ", " : "");
    }
    fprintf(fp, "},\n     .scales = {");
    for (int i = 0; i < 12; i++) {
        fprintf(fp, "0x%02x%s", bytes[96 + i], i < 11 ? ", " : "");
    }
    uint16_t d = *(uint16_t *)(bytes + 108);
    fprintf(fp, "},\n     .d = {.bits = 0x%04x}},\n", d);
    fprintf(fp, "};\n\n");
    print_float_array(fp, "g_llama_q3_k_dequant", dequant, 256);
    free(qdata);
}

static void gen_q4_k(FILE *fp) {
    size_t qsize = ggml_row_size(GGML_TYPE_Q4_K, 256);
    void *qdata = malloc(qsize);

    float expanded[256];
    for (int i = 0; i < 256; i++) {
        expanded[i] = test_input[i % 96];
    }

    ggml_quantize_chunk(GGML_TYPE_Q4_K, expanded, qdata, 0, 1, 256, nullptr);
    float dequant[256];
    dequantize_into(GGML_TYPE_Q4_K, qdata, dequant, 256);

    fprintf(fp, "// Q4_K: 144 bytes per 256-value block\n");
    fprintf(fp, "static const marmot_q4_k_block_t g_llama_q4_k[1] = {\n");
    const uint8_t *bytes = (const uint8_t *)qdata;
    uint16_t d = *(uint16_t *)bytes;
    uint16_t dmin = *(uint16_t *)(bytes + 2);
    fprintf(fp, "    {.d = {.bits = 0x%04x}, .dmin = {.bits = 0x%04x},\n", d, dmin);
    fprintf(fp, "     .scales = {");
    for (int i = 0; i < 12; i++) {
        fprintf(fp, "0x%02x%s", bytes[4 + i], i < 11 ? ", " : "");
    }
    fprintf(fp, "},\n     .qs = {");
    for (int i = 0; i < 128; i++) {
        if (i % 16 == 0 && i > 0)
            fprintf(fp, "\n            ");
        fprintf(fp, "0x%02x%s", bytes[16 + i], i < 127 ? ", " : "");
    }
    fprintf(fp, "}},\n");
    fprintf(fp, "};\n\n");
    print_float_array(fp, "g_llama_q4_k_dequant", dequant, 256);
    free(qdata);
}

static void gen_q5_k(FILE *fp) {
    size_t qsize = ggml_row_size(GGML_TYPE_Q5_K, 256);
    void *qdata = malloc(qsize);

    float expanded[256];
    for (int i = 0; i < 256; i++) {
        expanded[i] = test_input[i % 96];
    }

    ggml_quantize_chunk(GGML_TYPE_Q5_K, expanded, qdata, 0, 1, 256, nullptr);
    float dequant[256];
    dequantize_into(GGML_TYPE_Q5_K, qdata, dequant, 256);

    fprintf(fp, "// Q5_K: 176 bytes per 256-value block\n");
    fprintf(fp, "static const marmot_q5_k_block_t g_llama_q5_k[1] = {\n");
    const uint8_t *bytes = (const uint8_t *)qdata;
    uint16_t d = *(uint16_t *)bytes;
    uint16_t dmin = *(uint16_t *)(bytes + 2);
    fprintf(fp, "    {.d = {.bits = 0x%04x}, .dmin = {.bits = 0x%04x},\n", d, dmin);
    fprintf(fp, "     .scales = {");
    for (int i = 0; i < 12; i++) {
        fprintf(fp, "0x%02x%s", bytes[4 + i], i < 11 ? ", " : "");
    }
    fprintf(fp, "},\n     .qh = {");
    for (int i = 0; i < 32; i++) {
        if (i % 16 == 0 && i > 0)
            fprintf(fp, "\n             ");
        fprintf(fp, "0x%02x%s", bytes[16 + i], i < 31 ? ", " : "");
    }
    fprintf(fp, "},\n     .qs = {");
    for (int i = 0; i < 128; i++) {
        if (i % 16 == 0 && i > 0)
            fprintf(fp, "\n            ");
        fprintf(fp, "0x%02x%s", bytes[48 + i], i < 127 ? ", " : "");
    }
    fprintf(fp, "}},\n");
    fprintf(fp, "};\n\n");
    print_float_array(fp, "g_llama_q5_k_dequant", dequant, 256);
    free(qdata);
}

static void gen_q6_k(FILE *fp) {
    size_t qsize = ggml_row_size(GGML_TYPE_Q6_K, 256);
    void *qdata = malloc(qsize);

    float expanded[256];
    for (int i = 0; i < 256; i++) {
        expanded[i] = test_input[i % 96];
    }

    ggml_quantize_chunk(GGML_TYPE_Q6_K, expanded, qdata, 0, 1, 256, nullptr);
    float dequant[256];
    dequantize_into(GGML_TYPE_Q6_K, qdata, dequant, 256);

    fprintf(fp, "// Q6_K: 210 bytes per 256-value block\n");
    fprintf(fp, "static const marmot_q6_k_block_t g_llama_q6_k[1] = {\n");
    const uint8_t *bytes = (const uint8_t *)qdata;
    fprintf(fp, "    {.ql = {");
    for (int i = 0; i < 128; i++) {
        if (i % 16 == 0 && i > 0)
            fprintf(fp, "\n            ");
        fprintf(fp, "0x%02x%s", bytes[i], i < 127 ? ", " : "");
    }
    fprintf(fp, "},\n     .qh = {");
    for (int i = 0; i < 64; i++) {
        if (i % 16 == 0 && i > 0)
            fprintf(fp, "\n            ");
        fprintf(fp, "0x%02x%s", bytes[128 + i], i < 63 ? ", " : "");
    }
    fprintf(fp, "},\n     .scales = {");
    for (int i = 0; i < 16; i++) {
        fprintf(fp, "%d%s", (int8_t)bytes[192 + i], i < 15 ? ", " : "");
    }
    uint16_t d = *(uint16_t *)(bytes + 208);
    fprintf(fp, "},\n     .d = {.bits = 0x%04x}},\n", d);
    fprintf(fp, "};\n\n");
    print_float_array(fp, "g_llama_q6_k_dequant", dequant, 256);
    free(qdata);
}

static void gen_q8_k(FILE *fp) {
    size_t qsize = ggml_row_size(GGML_TYPE_Q8_K, 256);
    void *qdata = malloc(qsize);

    float expanded[256];
    for (int i = 0; i < 256; i++) {
        expanded[i] = test_input[i % 96];
    }

    quantize_row_q8_K_ref(expanded, (block_q8_K *)qdata, 256);
    float dequant[256];
    dequantize_into(GGML_TYPE_Q8_K, qdata, dequant, 256);

    fprintf(fp, "// Q8_K: 292 bytes per 256-value block\n");
    fprintf(fp, "static const marmot_q8_k_block_t g_llama_q8_k[1] = {\n");
    const uint8_t *bytes = (const uint8_t *)qdata;
    float d = *(float *)bytes;
    fprintf(fp, "    {.d = %.8ff,\n", d);
    fprintf(fp, "     .qs = {");
    for (int i = 0; i < 256; i++) {
        if (i % 16 == 0 && i > 0)
            fprintf(fp, "\n            ");
        fprintf(fp, "%d%s", (int8_t)bytes[4 + i], i < 255 ? ", " : "");
    }
    fprintf(fp, "},\n     .bsums = {");
    const int16_t *bsums = (const int16_t *)(bytes + 260);
    for (int i = 0; i < 16; i++) {
        fprintf(fp, "%d%s", bsums[i], i < 15 ? ", " : "");
    }
    fprintf(fp, "}},\n");
    fprintf(fp, "};\n\n");
    print_float_array(fp, "g_llama_q8_k_dequant", dequant, 256);
    free(qdata);
}

static void gen_vec_dot_data(FILE *fp) {
    const int length = 256;
    const int blocks32 = length / 32;
    const int kBlocks = length / 256;

    float test_vec1[length];
    float test_vec2[length];
    double test_vec1_f64[length];
    double test_vec2_f64[length];
    for (int i = 0; i < length; ++i) {
        const float a = (float)((i % 31) - 15);
        const float b = (float)((i % 23) - 11);
        test_vec1[i] = a * 0.55f + 0.12f * (float)((i % 5) - 2);
        test_vec2[i] = b * 0.60f + 0.07f * (float)((i % 7) - 3);
        test_vec1_f64[i] = (double)test_vec1[i];
        test_vec2_f64[i] = (double)test_vec2[i];
    }

    fprintf(fp, "static const float g_vec_dot_test_vec1[%d] = {\n    ", length);
    for (int i = 0; i < length; ++i) {
        fprintf(fp, "%.6ff%s", test_vec1[i], i < length - 1 ? ", " : "");
        if ((i + 1) % 8 == 0 && i < length - 1)
            fprintf(fp, "\n    ");
    }
    fprintf(fp, "\n};\n\n");

    print_double_array(fp, "g_vec_dot_test_vec1_f64", test_vec1_f64, length);
    print_double_array(fp, "g_vec_dot_test_vec2_f64", test_vec2_f64, length);

    fprintf(fp, "static const float g_vec_dot_test_vec2[%d] = {\n    ", length);
    for (int i = 0; i < length; ++i) {
        fprintf(fp, "%.6ff%s", test_vec2[i], i < length - 1 ? ", " : "");
        if ((i + 1) % 8 == 0 && i < length - 1)
            fprintf(fp, "\n    ");
    }
    fprintf(fp, "\n};\n\n");

    block_q8_0 act_blocks[blocks32];
    quantize_row_q8_0_ref(test_vec2, act_blocks, length);

    fprintf(fp, "static const marmot_q8_0_block_t g_vec_dot_q8_0_activations[%d] = {\n", blocks32);
    for (int b = 0; b < blocks32; ++b) {
        fprintf(fp, "    {.scale = {.bits = 0x%04x}, .qs = {", act_blocks[b].d);
        for (int i = 0; i < 32; ++i) {
            fprintf(fp, "%d%s", act_blocks[b].qs[i], i < 31 ? ", " : "");
            if ((i + 1) % 16 == 0 && i < 31)
                fprintf(fp, "\n        ");
        }
        fprintf(fp, "}},\n");
    }
    fprintf(fp, "};\n\n");

    const int blocks256 = length / MARMOT_QK_K_VALUES;
    block_q8_K act_blocks_q8k[blocks256];
    quantize_row_q8_K_ref(test_vec2, act_blocks_q8k, length);

    fprintf(fp, "static const marmot_q8_k_block_t g_vec_dot_q8_k_activations[%d] = {\n", blocks256);
    for (int b = 0; b < blocks256; ++b) {
        fprintf(fp, "    {.d = %.8ff, .qs = {", act_blocks_q8k[b].d);
        const int8_t *qs = act_blocks_q8k[b].qs;
        for (int i = 0; i < MARMOT_QK_K_VALUES; ++i) {
            if (i % 16 == 0 && i > 0) {
                fprintf(fp, "\n            ");
            }
            fprintf(fp, "%d%s", qs[i], i < MARMOT_QK_K_VALUES - 1 ? ", " : "");
        }
        fprintf(fp, "},\n     .bsums = {");
        const int16_t *bsums = act_blocks_q8k[b].bsums;
        for (int i = 0; i < MARMOT_QK_K_VALUES / 16; ++i) {
            fprintf(fp, "%d%s", bsums[i], i < (MARMOT_QK_K_VALUES / 16) - 1 ? ", " : "");
        }
        fprintf(fp, "}},\n");
    }
    fprintf(fp, "};\n\n");

    block_q4_0 q4_0_blocks[blocks32];
    block_q4_1 q4_1_blocks[blocks32];
    block_q5_0 q5_0_blocks[blocks32];
    block_q5_1 q5_1_blocks[blocks32];
    block_q8_0 q8_0_blocks[blocks32];
    block_q8_1 q8_1_blocks[blocks32];

    quantize_row_q4_0_ref(test_vec1, q4_0_blocks, length);
    quantize_row_q4_1_ref(test_vec1, q4_1_blocks, length);
    quantize_row_q5_0_ref(test_vec1, q5_0_blocks, length);
    quantize_row_q5_1_ref(test_vec1, q5_1_blocks, length);
    quantize_row_q8_0_ref(test_vec1, q8_0_blocks, length);
    quantize_row_q8_1_ref(test_vec1, q8_1_blocks, length);

    float activations_f32[length];
    dequantize_into(GGML_TYPE_Q8_0, act_blocks, activations_f32, length);
    float activations_f32_q8k[length];
    dequantize_into(GGML_TYPE_Q8_K, act_blocks_q8k, activations_f32_q8k, length);

    float weights_tmp[length];

    dequantize_into(GGML_TYPE_Q4_0, q4_0_blocks, weights_tmp, length);
    float dot_q4_0 = vec_dot_f32(weights_tmp, activations_f32, length);

    dequantize_into(GGML_TYPE_Q4_1, q4_1_blocks, weights_tmp, length);
    float dot_q4_1 = vec_dot_f32(weights_tmp, activations_f32, length);

    dequantize_into(GGML_TYPE_Q5_0, q5_0_blocks, weights_tmp, length);
    float dot_q5_0 = vec_dot_f32(weights_tmp, activations_f32, length);

    dequantize_into(GGML_TYPE_Q5_1, q5_1_blocks, weights_tmp, length);
    float dot_q5_1 = vec_dot_f32(weights_tmp, activations_f32, length);

    dequantize_into(GGML_TYPE_Q8_0, q8_0_blocks, weights_tmp, length);
    float dot_q8_0 = vec_dot_f32(weights_tmp, activations_f32, length);

    dequantize_into(GGML_TYPE_Q8_1, q8_1_blocks, weights_tmp, length);
    float dot_q8_1 = vec_dot_f32(weights_tmp, activations_f32, length);

    fprintf(fp, "static const marmot_q4_0_block_t g_vec_dot_q4_0_weights[%d] = {\n", blocks32);
    for (int b = 0; b < blocks32; ++b) {
        fprintf(fp, "    {.scale = {.bits = 0x%04x}, .qs = {", q4_0_blocks[b].d);
        for (int i = 0; i < 16; ++i) {
            fprintf(fp, "0x%02x%s", q4_0_blocks[b].qs[i], i < 15 ? ", " : "");
        }
        fprintf(fp, "}},\n");
    }
    fprintf(fp, "};\n\n");

    fprintf(fp, "static const marmot_q4_1_block_t g_vec_dot_q4_1_weights[%d] = {\n", blocks32);
    for (int b = 0; b < blocks32; ++b) {
        fprintf(
            fp, "    {.scale = {.bits = 0x%04x}, .min = {.bits = 0x%04x}, .qs = {", q4_1_blocks[b].d, q4_1_blocks[b].m
        );
        for (int i = 0; i < 16; ++i) {
            fprintf(fp, "0x%02x%s", q4_1_blocks[b].qs[i], i < 15 ? ", " : "");
        }
        fprintf(fp, "}},\n");
    }
    fprintf(fp, "};\n\n");

    fprintf(fp, "static const marmot_q5_0_block_t g_vec_dot_q5_0_weights[%d] = {\n", blocks32);
    for (int b = 0; b < blocks32; ++b) {
        fprintf(
            fp, "    {.scale = {.bits = 0x%04x}, .qh = {0x%02x, 0x%02x, 0x%02x, 0x%02x}, .qs = {", q5_0_blocks[b].d,
            q5_0_blocks[b].qh[0], q5_0_blocks[b].qh[1], q5_0_blocks[b].qh[2], q5_0_blocks[b].qh[3]
        );
        for (int i = 0; i < 16; ++i) {
            fprintf(fp, "0x%02x%s", (uint8_t)q5_0_blocks[b].qs[i], i < 15 ? ", " : "");
        }
        fprintf(fp, "}},\n");
    }
    fprintf(fp, "};\n\n");

    fprintf(fp, "static const marmot_q5_1_block_t g_vec_dot_q5_1_weights[%d] = {\n", blocks32);
    for (int b = 0; b < blocks32; ++b) {
        fprintf(
            fp,
            "    {.scale = {.bits = 0x%04x}, .min = {.bits = 0x%04x}, .qh = {0x%02x, 0x%02x, 0x%02x, 0x%02x}, .qs = {",
            q5_1_blocks[b].d, q5_1_blocks[b].m, q5_1_blocks[b].qh[0], q5_1_blocks[b].qh[1], q5_1_blocks[b].qh[2],
            q5_1_blocks[b].qh[3]
        );
        for (int i = 0; i < 16; ++i) {
            fprintf(fp, "0x%02x%s", (uint8_t)q5_1_blocks[b].qs[i], i < 15 ? ", " : "");
        }
        fprintf(fp, "}},\n");
    }
    fprintf(fp, "};\n\n");

    fprintf(fp, "static const marmot_q8_0_block_t g_vec_dot_q8_0_weights[%d] = {\n", blocks32);
    for (int b = 0; b < blocks32; ++b) {
        fprintf(fp, "    {.scale = {.bits = 0x%04x}, .qs = {", q8_0_blocks[b].d);
        for (int i = 0; i < 32; ++i) {
            fprintf(fp, "%d%s", q8_0_blocks[b].qs[i], i < 31 ? ", " : "");
            if ((i + 1) % 16 == 0 && i < 31)
                fprintf(fp, "\n        ");
        }
        fprintf(fp, "}},\n");
    }
    fprintf(fp, "};\n\n");

    fprintf(fp, "static const marmot_q8_1_block_t g_vec_dot_q8_1_weights[%d] = {\n", blocks32);
    for (int b = 0; b < blocks32; ++b) {
        const uint8_t *block = (const uint8_t *)&q8_1_blocks[b];
        uint16_t scale;
        uint16_t sum;
        memcpy(&scale, block, sizeof(scale));
        memcpy(&sum, block + 2, sizeof(sum));
        const int8_t *qs = (const int8_t *)(block + 4);
        fprintf(fp, "    {.scale = {.bits = 0x%04x}, .sum = {.bits = 0x%04x}, .qs = {", scale, sum);
        for (int i = 0; i < 32; ++i) {
            fprintf(fp, "%4d", qs[i]);
            if (i < 31)
                fprintf(fp, ",%s", (i + 1) % 16 == 0 ? "\n        " : " ");
        }
        fprintf(fp, "}},\n");
    }
    fprintf(fp, "};\n\n");

    block_q2_K q2_k_blocks[kBlocks];
    block_q3_K q3_k_blocks[kBlocks];
    block_q4_K q4_k_blocks[kBlocks];
    block_q5_K q5_k_blocks[kBlocks];
    block_q6_K q6_k_blocks[kBlocks];
    block_q8_K q8_k_blocks[kBlocks];

    quantize_row_q2_K_ref(test_vec1, q2_k_blocks, length);
    quantize_row_q3_K_ref(test_vec1, q3_k_blocks, length);
    quantize_row_q4_K_ref(test_vec1, q4_k_blocks, length);
    quantize_row_q5_K_ref(test_vec1, q5_k_blocks, length);
    quantize_row_q6_K_ref(test_vec1, q6_k_blocks, length);
    quantize_row_q8_K_ref(test_vec1, q8_k_blocks, length);

    dequantize_into(GGML_TYPE_Q2_K, q2_k_blocks, weights_tmp, length);
    float dot_q2_k = vec_dot_f32(weights_tmp, activations_f32_q8k, length);

    dequantize_into(GGML_TYPE_Q3_K, q3_k_blocks, weights_tmp, length);
    float dot_q3_k = vec_dot_f32(weights_tmp, activations_f32_q8k, length);

    dequantize_into(GGML_TYPE_Q4_K, q4_k_blocks, weights_tmp, length);
    float dot_q4_k = vec_dot_f32(weights_tmp, activations_f32_q8k, length);

    dequantize_into(GGML_TYPE_Q5_K, q5_k_blocks, weights_tmp, length);
    float dot_q5_k = vec_dot_f32(weights_tmp, activations_f32_q8k, length);

    dequantize_into(GGML_TYPE_Q6_K, q6_k_blocks, weights_tmp, length);
    float dot_q6_k = vec_dot_f32(weights_tmp, activations_f32_q8k, length);

    dequantize_into(GGML_TYPE_Q8_K, q8_k_blocks, weights_tmp, length);
    float dot_q8_k = vec_dot_f32(weights_tmp, activations_f32_q8k, length);

    fprintf(fp, "static const marmot_q2_k_block_t g_vec_dot_q2_k_weights[%d] = {\n", kBlocks);
    for (int b = 0; b < kBlocks; ++b) {
        const uint8_t *bytes = (const uint8_t *)&q2_k_blocks[b];
        fprintf(fp, "    {.scales = {");
        for (int i = 0; i < 16; ++i) {
            fprintf(fp, "0x%02x%s", bytes[i], i < 15 ? ", " : "");
        }
        fprintf(fp, "},\n     .qs = {");
        for (int i = 0; i < 64; ++i) {
            if (i % 16 == 0 && i > 0)
                fprintf(fp, "\n            ");
            fprintf(fp, "0x%02x%s", bytes[16 + i], i < 63 ? ", " : "");
        }
        uint16_t d;
        uint16_t dmin;
        memcpy(&d, bytes + 80, sizeof(d));
        memcpy(&dmin, bytes + 82, sizeof(dmin));
        fprintf(fp, "},\n     .d = {.bits = 0x%04x}, .dmin = {.bits = 0x%04x}},\n", d, dmin);
    }
    fprintf(fp, "};\n\n");

    fprintf(fp, "static const marmot_q3_k_block_t g_vec_dot_q3_k_weights[%d] = {\n", kBlocks);
    for (int b = 0; b < kBlocks; ++b) {
        const uint8_t *bytes = (const uint8_t *)&q3_k_blocks[b];
        fprintf(fp, "    {.hmask = {");
        for (int i = 0; i < 32; ++i) {
            if (i % 16 == 0 && i > 0)
                fprintf(fp, "\n              ");
            fprintf(fp, "0x%02x%s", bytes[i], i < 31 ? ", " : "");
        }
        fprintf(fp, "},\n     .qs = {");
        for (int i = 0; i < 64; ++i) {
            if (i % 16 == 0 && i > 0)
                fprintf(fp, "\n            ");
            fprintf(fp, "0x%02x%s", bytes[32 + i], i < 63 ? ", " : "");
        }
        fprintf(fp, "},\n     .scales = {");
        for (int i = 0; i < 12; ++i) {
            fprintf(fp, "0x%02x%s", bytes[96 + i], i < 11 ? ", " : "");
        }
        uint16_t d;
        memcpy(&d, bytes + 108, sizeof(d));
        fprintf(fp, "},\n     .d = {.bits = 0x%04x}},\n", d);
    }
    fprintf(fp, "};\n\n");

    fprintf(fp, "static const marmot_q4_k_block_t g_vec_dot_q4_k_weights[%d] = {\n", kBlocks);
    for (int b = 0; b < kBlocks; ++b) {
        const uint8_t *bytes = (const uint8_t *)&q4_k_blocks[b];
        uint16_t d;
        uint16_t dmin;
        memcpy(&d, bytes, sizeof(d));
        memcpy(&dmin, bytes + 2, sizeof(dmin));
        fprintf(fp, "    {.d = {.bits = 0x%04x}, .dmin = {.bits = 0x%04x}, .scales = {", d, dmin);
        for (int i = 0; i < 12; ++i) {
            fprintf(fp, "0x%02x%s", bytes[4 + i], i < 11 ? ", " : "");
        }
        fprintf(fp, "},\n     .qs = {");
        for (int i = 0; i < 128; ++i) {
            if (i % 16 == 0 && i > 0)
                fprintf(fp, "\n            ");
            fprintf(fp, "0x%02x%s", bytes[16 + i], i < 127 ? ", " : "");
        }
        fprintf(fp, "}},\n");
    }
    fprintf(fp, "};\n\n");

    fprintf(fp, "static const marmot_q5_k_block_t g_vec_dot_q5_k_weights[%d] = {\n", kBlocks);
    for (int b = 0; b < kBlocks; ++b) {
        const uint8_t *bytes = (const uint8_t *)&q5_k_blocks[b];
        uint16_t d;
        uint16_t dmin;
        memcpy(&d, bytes, sizeof(d));
        memcpy(&dmin, bytes + 2, sizeof(dmin));
        fprintf(fp, "    {.d = {.bits = 0x%04x}, .dmin = {.bits = 0x%04x}, .scales = {", d, dmin);
        for (int i = 0; i < 12; ++i) {
            fprintf(fp, "0x%02x%s", bytes[4 + i], i < 11 ? ", " : "");
        }
        fprintf(fp, "},\n     .qh = {");
        for (int i = 0; i < 32; ++i) {
            if (i % 16 == 0 && i > 0)
                fprintf(fp, "\n           ");
            fprintf(fp, "0x%02x%s", bytes[16 + i], i < 31 ? ", " : "");
        }
        fprintf(fp, "},\n     .qs = {");
        for (int i = 0; i < 128; ++i) {
            if (i % 16 == 0 && i > 0)
                fprintf(fp, "\n            ");
            fprintf(fp, "0x%02x%s", bytes[48 + i], i < 127 ? ", " : "");
        }
        fprintf(fp, "}},\n");
    }
    fprintf(fp, "};\n\n");

    fprintf(fp, "static const marmot_q6_k_block_t g_vec_dot_q6_k_weights[%d] = {\n", kBlocks);
    for (int b = 0; b < kBlocks; ++b) {
        const uint8_t *bytes = (const uint8_t *)&q6_k_blocks[b];
        fprintf(fp, "    {.ql = {");
        for (int i = 0; i < 128; ++i) {
            if (i % 16 == 0 && i > 0)
                fprintf(fp, "\n           ");
            fprintf(fp, "0x%02x%s", bytes[i], i < 127 ? ", " : "");
        }
        fprintf(fp, "},\n     .qh = {");
        for (int i = 0; i < 64; ++i) {
            if (i % 16 == 0 && i > 0)
                fprintf(fp, "\n           ");
            fprintf(fp, "0x%02x%s", bytes[128 + i], i < 63 ? ", " : "");
        }
        fprintf(fp, "},\n     .scales = {");
        for (int i = 0; i < 16; ++i) {
            fprintf(fp, "%d%s", (int8_t)bytes[192 + i], i < 15 ? ", " : "");
        }
        uint16_t d;
        memcpy(&d, bytes + 208, sizeof(d));
        fprintf(fp, "},\n     .d = {.bits = 0x%04x}},\n", d);
    }
    fprintf(fp, "};\n\n");

    fprintf(fp, "static const marmot_q8_k_block_t g_vec_dot_q8_k_weights[%d] = {\n", kBlocks);
    for (int b = 0; b < kBlocks; ++b) {
        const uint8_t *bytes = (const uint8_t *)&q8_k_blocks[b];
        float d;
        memcpy(&d, bytes, sizeof(d));
        fprintf(fp, "    {.d = %.8ff, .qs = {", d);
        const int8_t *qs = (const int8_t *)(bytes + 4);
        for (int i = 0; i < 256; ++i) {
            if (i % 16 == 0 && i > 0)
                fprintf(fp, "\n            ");
            fprintf(fp, "%d%s", qs[i], i < 255 ? ", " : "");
        }
        fprintf(fp, "},\n     .bsums = {");
        const int16_t *bsums = (const int16_t *)(bytes + 260);
        for (int i = 0; i < 16; ++i) {
            fprintf(fp, "%d%s", bsums[i], i < 15 ? ", " : "");
        }
        fprintf(fp, "}},\n");
    }
    fprintf(fp, "};\n\n");

    fprintf(fp, "static const float g_vec_dot_q4_0_q8_0_expected = %.8ff;\n", dot_q4_0);
    fprintf(fp, "static const float g_vec_dot_q4_1_q8_0_expected = %.8ff;\n", dot_q4_1);
    fprintf(fp, "static const float g_vec_dot_q5_0_q8_0_expected = %.8ff;\n", dot_q5_0);
    fprintf(fp, "static const float g_vec_dot_q5_1_q8_0_expected = %.8ff;\n", dot_q5_1);
    fprintf(fp, "static const float g_vec_dot_q8_0_q8_0_expected = %.8ff;\n", dot_q8_0);
    fprintf(fp, "static const float g_vec_dot_q8_1_q8_0_expected = %.8ff;\n", dot_q8_1);
    fprintf(fp, "static const float g_vec_dot_q2_k_q8_k_expected = %.8ff;\n", dot_q2_k);
    fprintf(fp, "static const float g_vec_dot_q3_k_q8_k_expected = %.8ff;\n", dot_q3_k);
    fprintf(fp, "static const float g_vec_dot_q4_k_q8_k_expected = %.8ff;\n", dot_q4_k);
    fprintf(fp, "static const float g_vec_dot_q5_k_q8_k_expected = %.8ff;\n", dot_q5_k);
    fprintf(fp, "static const float g_vec_dot_q6_k_q8_k_expected = %.8ff;\n", dot_q6_k);
    fprintf(fp, "static const float g_vec_dot_q8_k_q8_k_expected = %.8ff;\n", dot_q8_k);
}

// -----------------------------------------------------------------------------
// Embedding golden fixtures (Q4_0, Q4_1, Q8_0) as text files
// -----------------------------------------------------------------------------

static int read_llama_commit_sha(char *buf, size_t n) {
    if (buf == nullptr || n == 0)
        return -1;
    FILE *p = popen("git -C llama.cpp rev-parse HEAD", "r");
    if (!p)
        return -1;
    size_t off = 0;
    while (off + 1 < n) {
        int c = fgetc(p);
        if (c == EOF)
            break;
        if (c == '\n' || c == '\r')
            continue;
        buf[off++] = (char)c;
    }
    buf[off] = '\0';
    int rc = pclose(p);
    (void)rc;
    return (off > 0) ? 0 : -1;
}

static void write_embedding_fixture(const char *fname, int ggml_type, const char *kind_str, int vocab, int dim) {
    // Build a small FP32 weight matrix deterministically from test_input
    const int K = dim;
    float *weights = (float *)malloc((size_t)vocab * (size_t)K * sizeof(float));
    for (int r = 0; r < vocab; ++r) {
        for (int c = 0; c < K; ++c) {
            // Use cyclic pattern from test_input to avoid licensing issues
            float base = test_input[(r * 17 + c) % 96];
            weights[(size_t)r * (size_t)K + (size_t)c] = base;
        }
    }

    int blocks_per_row = (K + 31) / 32;
    size_t row_size = ggml_row_size((enum ggml_type)ggml_type, K);
    unsigned char *row_bytes = (unsigned char *)malloc(row_size);

    // Accumulate weights bytes (row-major blocks) and expected decoded floats for ids [0,2,3]
    int ids[3] = {0, 2, 3};
    float *expected = (float *)malloc((size_t)3 * (size_t)K * sizeof(float));
    unsigned char *weights_hex_bytes = (unsigned char *)malloc((size_t)vocab * row_size);

    // Quantize each row using ggml
    for (int r = 0; r < vocab; ++r) {
        if (ggml_type == GGML_TYPE_Q8_1) {
            quantize_row_q8_1_ref(weights + (size_t)r * (size_t)K, (block_q8_1 *)row_bytes, K);
        } else {
            ggml_quantize_chunk(
                (enum ggml_type)ggml_type, weights + (size_t)r * (size_t)K, row_bytes, 0, 1, K, nullptr
            );
        }
        memcpy(weights_hex_bytes + (size_t)r * row_size, row_bytes, row_size);
    }

    // Decode expected per selected ids
    for (int id_idx = 0; id_idx < 3; ++id_idx) {
        int r = ids[id_idx];
        unsigned char *rb = weights_hex_bytes + (size_t)r * row_size;
        size_t off = 0;
        float *dst = expected + (size_t)id_idx * (size_t)K;
        int written = 0;
        for (int b = 0; b < blocks_per_row; ++b) {
            if (ggml_type == GGML_TYPE_Q4_0) {
                uint16_t d = *(uint16_t *)(rb + off);
                off += 2;
                unsigned char *qs = rb + off;
                off += 16;
                float scale = *(const _Float16 *)&d; // rely on IEEE layout
                for (int i = 0; i < 32 && written < K; ++i, ++written) {
                    unsigned char packed = qs[i >> 1];
                    int q = (i & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
                    q -= 8;
                    dst[written] = (float)scale * (float)q;
                }
            } else if (ggml_type == GGML_TYPE_Q4_1) {
                uint16_t d = *(uint16_t *)(rb + off);
                off += 2;
                uint16_t m = *(uint16_t *)(rb + off);
                off += 2;
                unsigned char *qs = rb + off;
                off += 16;
                float scale = *(const _Float16 *)&d;
                float minv = *(const _Float16 *)&m;
                for (int i = 0; i < 32 && written < K; ++i, ++written) {
                    unsigned char packed = qs[i >> 1];
                    int q = (i & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
                    dst[written] = scale * (float)q + minv;
                }
            } else if (ggml_type == GGML_TYPE_Q5_0) {
                uint16_t d = *(uint16_t *)(rb + off);
                off += 2;
                unsigned char *qh = rb + off;
                off += 4;
                unsigned char *qs = rb + off;
                off += 16;
                float scale = *(const _Float16 *)&d;
                for (int i = 0; i < 32 && written < K; ++i, ++written) {
                    unsigned char packed = qs[i >> 1];
                    int q = (i & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
                    int high_bit = (qh[i / 8] >> (i % 8)) & 1;
                    q |= (high_bit << 4);
                    q -= 16;
                    dst[written] = scale * (float)q;
                }
            } else if (ggml_type == GGML_TYPE_Q5_1) {
                uint16_t d = *(uint16_t *)(rb + off);
                off += 2;
                uint16_t m = *(uint16_t *)(rb + off);
                off += 2;
                unsigned char *qh = rb + off;
                off += 4;
                unsigned char *qs = rb + off;
                off += 16;
                float scale = *(const _Float16 *)&d;
                float minv = *(const _Float16 *)&m;
                for (int i = 0; i < 32 && written < K; ++i, ++written) {
                    unsigned char packed = qs[i >> 1];
                    int q = (i & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
                    int high_bit = (qh[i / 8] >> (i % 8)) & 1;
                    q |= (high_bit << 4);
                    dst[written] = scale * (float)q + minv;
                }
            } else if (ggml_type == GGML_TYPE_Q8_0) {
                uint16_t d = *(uint16_t *)(rb + off);
                off += 2;
                signed char *qs = (signed char *)(rb + off);
                off += 32;
                float scale = *(const _Float16 *)&d;
                for (int i = 0; i < 32 && written < K; ++i, ++written) {
                    dst[written] = scale * (float)qs[i];
                }
            } else if (ggml_type == GGML_TYPE_Q8_1) {
                uint16_t d = *(uint16_t *)(rb + off);
                off += 2;
                uint16_t s = *(uint16_t *)(rb + off);
                (void)s;
                off += 2;
                signed char *qs = (signed char *)(rb + off);
                off += 32;
                float scale = *(const _Float16 *)&d;
                for (int i = 0; i < 32 && written < K; ++i, ++written) {
                    dst[written] = scale * (float)qs[i];
                }
            }
        }
    }

    // Write text file
    FILE *fp = fopen(fname, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open %s for writing\n", fname);
        free(weights);
        free(row_bytes);
        free(weights_hex_bytes);
        free(expected);
        return;
    }
    char sha[64] = {0};
    if (read_llama_commit_sha(sha, sizeof(sha)) != 0) {
        strcpy(sha, "unknown");
    }
    fprintf(fp, "# Generated by gen_golden_from_llama (llama.cpp commit %s)\n", sha);
    fprintf(fp, "quant_kind: %s\n", kind_str);
    fprintf(fp, "vocab: %d\n", vocab);
    fprintf(fp, "dim: %d\n", dim);
    fprintf(fp, "ids: %d %d %d\n", ids[0], ids[1], ids[2]);
    fprintf(fp, "weights_hex: ");
    for (size_t i = 0; i < (size_t)vocab * row_size; ++i)
        fprintf(fp, "%02x", weights_hex_bytes[i]);
    fprintf(fp, "\nexpected: ");
    for (size_t i = 0; i < (size_t)3 * (size_t)K; ++i) {
        fprintf(fp, i + 1 < (size_t)3 * (size_t)K ? "%0.8f " : "%0.8f", expected[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);

    free(weights);
    free(row_bytes);
    free(weights_hex_bytes);
    free(expected);
}

static void gen_embedding_fixtures(void) {
    // Write into tests/golden directory
    write_embedding_fixture("embedding_q4_0.txt", GGML_TYPE_Q4_0, "Q4_0", 4, 64);
    write_embedding_fixture("embedding_q4_1.txt", GGML_TYPE_Q4_1, "Q4_1", 4, 64);
    write_embedding_fixture("embedding_q5_0.txt", GGML_TYPE_Q5_0, "Q5_0", 4, 64);
    write_embedding_fixture("embedding_q5_1.txt", GGML_TYPE_Q5_1, "Q5_1", 4, 64);
    write_embedding_fixture("embedding_q8_0.txt", GGML_TYPE_Q8_0, "Q8_0", 4, 64);
    write_embedding_fixture("embedding_q8_1.txt", GGML_TYPE_Q8_1, "Q8_1", 4, 64);
    write_embedding_fixture_q4_0_ragged("embedding_q4_0_ragged.txt");
}

// Ragged variant for Q4_0 (adds row_offsets)
static void write_embedding_fixture_q4_0_ragged(const char *fname) {
    const int ggml_type = GGML_TYPE_Q4_0;
    const char *kind_str = "Q4_0";
    const int vocab = 4;
    const int dim = 64;

    // Deterministic FP32 weights (same scheme as write_embedding_fixture)
    float *weights = (float *)malloc((size_t)vocab * (size_t)dim * sizeof(float));
    for (int r = 0; r < vocab; ++r) {
        for (int c = 0; c < dim; ++c) {
            float base = test_input[(r * 17 + c) % 96];
            weights[(size_t)r * (size_t)dim + (size_t)c] = base;
        }
    }

    const int blocks_per_row = (dim + 31) / 32;
    const size_t row_size = ggml_row_size((enum ggml_type)ggml_type, dim);
    unsigned char *row_bytes = (unsigned char *)malloc(row_size);

    int ids[] = {0, 2, 1, 3, 0};
    const size_t ids_count = sizeof(ids) / sizeof(ids[0]);
    const int32_t row_offsets[] = {0, 2, 5};
    const size_t num_row_offsets = sizeof(row_offsets) / sizeof(row_offsets[0]);

    unsigned char *weights_hex_bytes = (unsigned char *)malloc((size_t)vocab * row_size);
    for (int r = 0; r < vocab; ++r) {
        ggml_quantize_chunk(
            (enum ggml_type)ggml_type, weights + (size_t)r * (size_t)dim, row_bytes, 0, 1, dim, nullptr
        );
        memcpy(weights_hex_bytes + (size_t)r * row_size, row_bytes, row_size);
    }

    float *expected = (float *)malloc(ids_count * (size_t)dim * sizeof(float));
    for (size_t id_idx = 0; id_idx < ids_count; ++id_idx) {
        int r = ids[id_idx];
        unsigned char *rb = weights_hex_bytes + (size_t)r * row_size;
        size_t off = 0;
        float *dst = expected + id_idx * (size_t)dim;
        int written = 0;
        for (int b = 0; b < blocks_per_row; ++b) {
            uint16_t d = *(uint16_t *)(rb + off);
            off += 2;
            unsigned char *qs = rb + off;
            off += 16;
            float scale = *(const _Float16 *)&d;
            for (int i = 0; i < 32 && written < dim; ++i, ++written) {
                unsigned char packed = qs[i >> 1];
                int q = (i & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
                q -= 8;
                dst[written] = (float)scale * (float)q;
            }
        }
    }

    FILE *fp = fopen(fname, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open %s for writing\n", fname);
        free(weights);
        free(row_bytes);
        free(weights_hex_bytes);
        free(expected);
        return;
    }
    char sha[64] = {0};
    if (read_llama_commit_sha(sha, sizeof(sha)) != 0) {
        strcpy(sha, "unknown");
    }
    fprintf(fp, "# Generated by gen_golden_from_llama (llama.cpp commit %s)\n", sha);
    fprintf(fp, "quant_kind: %s\n", kind_str);
    fprintf(fp, "vocab: %d\n", vocab);
    fprintf(fp, "dim: %d\n", dim);
    fprintf(fp, "ids:");
    for (size_t i = 0; i < ids_count; ++i) {
        fprintf(fp, i + 1 < ids_count ? " %d" : " %d\n", ids[i]);
    }
    fprintf(fp, "row_offsets:");
    for (size_t i = 0; i < num_row_offsets; ++i) {
        fprintf(fp, i + 1 < num_row_offsets ? " %d" : " %d\n", row_offsets[i]);
    }
    fprintf(fp, "weights_hex: ");
    for (size_t i = 0; i < (size_t)vocab * row_size; ++i)
        fprintf(fp, "%02x", weights_hex_bytes[i]);
    fprintf(fp, "\nexpected: ");
    for (size_t i = 0; i < ids_count * (size_t)dim; ++i) {
        fprintf(fp, (i + 1) < ids_count * (size_t)dim ? "%0.8f " : "%0.8f", expected[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);

    free(weights);
    free(row_bytes);
    free(weights_hex_bytes);
    free(expected);
}

int main(void) {
    FILE *quant_file = fopen("../backend/golden_quant_llama.h", "w");
    if (!quant_file) {
        fprintf(stderr, "Failed to open golden_quant_llama.h\n");
        return 1;
    }

    print_header(quant_file);
    print_input(quant_file);
    gen_q4_0(quant_file);
    gen_q4_1(quant_file);
    gen_q5_0(quant_file);
    gen_q5_1(quant_file);
    gen_q8_0(quant_file);
    gen_q8_1(quant_file);
    gen_q2_k(quant_file);
    gen_q3_k(quant_file);
    gen_q4_k(quant_file);
    gen_q5_k(quant_file);
    gen_q6_k(quant_file);
    gen_q8_k(quant_file);
    fclose(quant_file);

    FILE *vec_dot_file = fopen("../backend/golden_vec_dot_llama.h", "w");
    if (!vec_dot_file) {
        fprintf(stderr, "Failed to open golden_vec_dot_llama.h\n");
        return 1;
    }

    print_header(vec_dot_file);
    gen_vec_dot_data(vec_dot_file);
    fclose(vec_dot_file);

    FILE *matmul_file = fopen("../backend/golden_matmul_llama.h", "w");
    if (!matmul_file) {
        fprintf(stderr, "Failed to open golden_matmul_llama.h\n");
        return 1;
    }

    print_header(matmul_file);
    gen_matmul_goldens(matmul_file);
    fclose(matmul_file);

    FILE *float_ops_file = fopen("../backend/golden_float_ops_llama.h", "w");
    if (!float_ops_file) {
        fprintf(stderr, "Failed to open golden_float_ops_llama.h\n");
        return 1;
    }
    gen_float_ops_goldens(float_ops_file);
    fclose(float_ops_file);

    fprintf(stderr, "Generated golden data from llama.cpp\n");
    // Generate embedding fixtures (text-based) in current directory
    gen_embedding_fixtures();
    if (g_marmot_ctx != nullptr) {
        marmot_destroy(g_marmot_ctx);
        g_marmot_ctx = nullptr;
    }
    return 0;
}

static void gen_matmul_goldens(FILE *fp) {
    const matmul_case_t cases[] = {
        {"q4_0", "Q4_0", GGML_TYPE_Q4_0, MARMOT_QUANT_KIND_Q4_0, "MARMOT_QUANT_KIND_Q4_0", 3, 64, 4},
        {"q4_1", "Q4_1", GGML_TYPE_Q4_1, MARMOT_QUANT_KIND_Q4_1, "MARMOT_QUANT_KIND_Q4_1", 3, 64, 4},
        {"q5_0", "Q5_0", GGML_TYPE_Q5_0, MARMOT_QUANT_KIND_Q5_0, "MARMOT_QUANT_KIND_Q5_0", 3, 64, 4},
        {"q5_1", "Q5_1", GGML_TYPE_Q5_1, MARMOT_QUANT_KIND_Q5_1, "MARMOT_QUANT_KIND_Q5_1", 3, 64, 4},
        {"q8_0", "Q8_0", GGML_TYPE_Q8_0, MARMOT_QUANT_KIND_Q8_0, "MARMOT_QUANT_KIND_Q8_0", 3, 64, 4},
        {"q8_1", "Q8_1", GGML_TYPE_Q8_1, MARMOT_QUANT_KIND_Q8_1, "MARMOT_QUANT_KIND_Q8_1", 3, 64, 4},
        {"q2_k", "Q2_K", GGML_TYPE_Q2_K, MARMOT_QUANT_KIND_Q2_K, "MARMOT_QUANT_KIND_Q2_K", 3, 256, 4},
        {"q3_k", "Q3_K", GGML_TYPE_Q3_K, MARMOT_QUANT_KIND_Q3_K, "MARMOT_QUANT_KIND_Q3_K", 3, 256, 4},
        {"q4_k", "Q4_K", GGML_TYPE_Q4_K, MARMOT_QUANT_KIND_Q4_K, "MARMOT_QUANT_KIND_Q4_K", 3, 256, 4},
        {"q5_k", "Q5_K", GGML_TYPE_Q5_K, MARMOT_QUANT_KIND_Q5_K, "MARMOT_QUANT_KIND_Q5_K", 3, 256, 4},
        {"q6_k", "Q6_K", GGML_TYPE_Q6_K, MARMOT_QUANT_KIND_Q6_K, "MARMOT_QUANT_KIND_Q6_K", 3, 256, 4},
        {"q8_k", "Q8_K", GGML_TYPE_Q8_K, MARMOT_QUANT_KIND_Q8_K, "MARMOT_QUANT_KIND_Q8_K", 3, 256, 4},
    };

    fprintf(fp, "// Quantized matmul golden data generated from llama.cpp reference implementation\n\n");

    for (size_t case_idx = 0; case_idx < sizeof(cases) / sizeof(cases[0]); ++case_idx) {
        const matmul_case_t *tc = &cases[case_idx];
        const size_t input_elems = (size_t)tc->N * (size_t)tc->K;
        const size_t output_elems = (size_t)tc->N * (size_t)tc->M;
        const size_t weight_elems = (size_t)tc->M * (size_t)tc->K;
        const size_t row_size = ggml_row_size(tc->ggml_type, tc->K);
        const size_t weight_bytes_total = (size_t)tc->M * row_size;

        float *input_f32 = (float *)malloc(input_elems * sizeof(float));
        float *weight_f32 = (float *)malloc(weight_elems * sizeof(float));
        ggml_fp16_t *input_f16 = (ggml_fp16_t *)malloc(input_elems * sizeof(ggml_fp16_t));
        uint8_t *weight_bytes = (uint8_t *)malloc(weight_bytes_total);
        float *out_from_f32 = (float *)malloc(output_elems * sizeof(float));
        float *out_from_f16 = (float *)malloc(output_elems * sizeof(float));

        if (!input_f32 || !weight_f32 || !input_f16 || !weight_bytes || !out_from_f32 || !out_from_f16) {
            fprintf(stderr, "Out of memory generating matmul golden for %s\n", tc->suffix);
            free(input_f32);
            free(weight_f32);
            free(input_f16);
            free(weight_bytes);
            free(out_from_f32);
            free(out_from_f16);
            exit(1);
        }

        fill_activation_pattern(input_f32, tc->N, tc->K);
        fill_weight_pattern(weight_f32, tc->M, tc->K);
        convert_fp32_to_fp16(input_f32, input_f16, input_elems);

        for (int m = 0; m < tc->M; ++m) {
            uint8_t *dst = weight_bytes + (size_t)m * row_size;
            quantize_weight_row(tc->ggml_type, weight_f32 + (size_t)m * tc->K, tc->K, dst);
        }

        if (!compute_matmul_reference(tc, weight_bytes, row_size, input_f32, input_f16, out_from_f32, out_from_f16)) {
            fprintf(stderr, "Failed to compute matmul reference for %s\n", tc->suffix);
            free(input_f32);
            free(weight_f32);
            free(input_f16);
            free(weight_bytes);
            free(out_from_f32);
            free(out_from_f16);
            exit(1);
        }

        fprintf(fp, "// Matmul fixture for %s (N=%d, K=%d, M=%d)\n", tc->upper, tc->N, tc->K, tc->M);
        fprintf(fp, "#define MATMUL_%s_N %d\n", tc->upper, tc->N);
        fprintf(fp, "#define MATMUL_%s_K %d\n", tc->upper, tc->K);
        fprintf(fp, "#define MATMUL_%s_M %d\n", tc->upper, tc->M);
        fprintf(fp, "#define MATMUL_%s_WEIGHT_ROW_SIZE %zu\n", tc->upper, row_size);
        fprintf(fp, "#define MATMUL_%s_QUANT_KIND %s\n\n", tc->upper, tc->quant_macro);

        char array_name[128];
        snprintf(array_name, sizeof(array_name), "g_matmul_%s_input_f32", tc->suffix);
        print_float_array(fp, array_name, input_f32, input_elems);

        snprintf(array_name, sizeof(array_name), "g_matmul_%s_input_f16", tc->suffix);
        print_u16_array(fp, array_name, (const uint16_t *)input_f16, input_elems);

        snprintf(array_name, sizeof(array_name), "g_matmul_%s_weight", tc->suffix);
        print_u8_array(fp, array_name, weight_bytes, weight_bytes_total);

        snprintf(array_name, sizeof(array_name), "g_matmul_%s_output_from_f32", tc->suffix);
        print_float_array(fp, array_name, out_from_f32, output_elems);

        snprintf(array_name, sizeof(array_name), "g_matmul_%s_output_from_f16", tc->suffix);
        print_float_array(fp, array_name, out_from_f16, output_elems);

        free(input_f32);
        free(weight_f32);
        free(input_f16);
        free(weight_bytes);
        free(out_from_f32);
        free(out_from_f16);
    }
}

static void gen_float_ops_goldens(FILE *fp) {
    fprintf(fp, "// Generated from llama.cpp reference implementation\n");
    fprintf(fp, "// DO NOT EDIT - regenerate with: cd tests/golden && ./generate_from_llama.sh\n");
    fprintf(fp, "#pragma once\n");
    fprintf(fp, "#include <stddef.h>\n");
    fprintf(fp, "#include <stdbool.h>\n\n");

    fprintf(
        fp,
        "typedef struct {\n"
        "    const char *name;\n"
        "    size_t m;\n"
        "    size_t k;\n"
        "    size_t n;\n"
        "    const float *weight;\n"
        "    const float *rhs;\n"
        "    const float *expected;\n"
        "    const double *weight_f64;\n"
        "    const double *rhs_f64;\n"
        "    const double *expected_f64;\n"
        "} llama_matmul_case_t;\n\n"
    );

    fprintf(
        fp,
        "typedef struct {\n"
        "    const char *name;\n"
        "    size_t rows;\n"
        "    size_t cols;\n"
        "    float epsilon;\n"
        "    const float *input;\n"
        "    const float *residual;\n"
        "    const float *weight;\n"
        "    const float *bias;\n"
        "    const float *expected;\n"
        "    const double *input_f64;\n"
        "    const double *residual_f64;\n"
        "    const double *weight_f64;\n"
        "    const double *bias_f64;\n"
        "    const double *expected_f64;\n"
        "} llama_layernorm_case_t;\n\n"
    );

    fprintf(
        fp,
        "typedef struct {\n"
        "    const char *name;\n"
        "    size_t rows;\n"
        "    size_t cols;\n"
        "    float epsilon;\n"
        "    const float *input;\n"
        "    const float *residual;\n"
        "    const float *weight;\n"
        "    const float *expected;\n"
        "    const double *input_f64;\n"
        "    const double *residual_f64;\n"
        "    const double *weight_f64;\n"
        "    const double *expected_f64;\n"
        "} llama_rmsnorm_case_t;\n\n"
    );

    fprintf(
        fp,
        "typedef struct {\n"
        "    size_t length;\n"
        "    float elu_alpha;\n"
        "    float selu_alpha;\n"
        "    float selu_lambda;\n"
        "    float leaky_slope;\n"
        "    float prelu_slope;\n"
        "    const float *input;\n"
        "    const float *relu;\n"
        "    const float *gelu;\n"
        "    const float *gelu_tanh;\n"
        "    const float *silu;\n"
        "    const float *sigmoid;\n"
        "    const float *tanh_v;\n"
        "    const float *mish;\n"
        "    const float *elu;\n"
        "    const float *selu;\n"
        "    const float *leaky_relu;\n"
        "    const float *prelu;\n"
        "    const double *input_f64;\n"
        "    const double *relu_f64;\n"
        "    const double *gelu_f64;\n"
        "    const double *gelu_tanh_f64;\n"
        "    const double *silu_f64;\n"
        "    const double *sigmoid_f64;\n"
        "    const double *tanh_v_f64;\n"
        "    const double *mish_f64;\n"
        "    const double *elu_f64;\n"
        "    const double *selu_f64;\n"
        "    const double *leaky_relu_f64;\n"
        "    const double *prelu_f64;\n"
        "} llama_activation_golden_t;\n\n"
    );

    // ------------------------------------------------------------------
    // Matmul cases (float)
    // ------------------------------------------------------------------
    static const struct {
        const char *name;
        int M;
        int K;
        int N;
    } matmul_cfgs[] = {
        {"case0", 2, 3, 2},
        {"case1", 3, 4, 2},
        {"case2", 2, 4, 3},
        {"case3", 4, 5, 3},
    };

    fprintf(fp, "// Float matmul fixtures\n");

    for (size_t idx = 0; idx < sizeof(matmul_cfgs) / sizeof(matmul_cfgs[0]); ++idx) {
        const char *name = matmul_cfgs[idx].name;
        const int M = matmul_cfgs[idx].M;
        const int K = matmul_cfgs[idx].K;
        const int N = matmul_cfgs[idx].N;

        const size_t weight_elems = (size_t)M * (size_t)K;
        const size_t rhs_elems = (size_t)K * (size_t)N;
        const size_t expected_elems = (size_t)M * (size_t)N;

        float *weight = (float *)malloc(weight_elems * sizeof(float));
        float *rhs = (float *)malloc(rhs_elems * sizeof(float));
        float *expected = (float *)malloc(expected_elems * sizeof(float));
        double *weight_f64 = (double *)malloc(weight_elems * sizeof(double));
        double *rhs_f64 = (double *)malloc(rhs_elems * sizeof(double));
        double *expected_f64 = (double *)malloc(expected_elems * sizeof(double));
        if (weight == nullptr || rhs == nullptr || expected == nullptr || weight_f64 == nullptr || rhs_f64 == nullptr ||
            expected_f64 == nullptr) {
            fprintf(stderr, "Allocation failure generating matmul fixture %s\n", name);
            exit(EXIT_FAILURE);
        }

        fill_weight_pattern(weight, M, K);
        fill_rhs_pattern(rhs, K, N);
        matmul_reference(weight, rhs, M, K, N, expected);

        for (size_t i = 0; i < weight_elems; ++i) {
            weight_f64[i] = (double)weight[i];
        }
        for (size_t i = 0; i < rhs_elems; ++i) {
            rhs_f64[i] = (double)rhs[i];
        }
        for (size_t i = 0; i < expected_elems; ++i) {
            expected_f64[i] = (double)expected[i];
        }

        char symbol[128];
        snprintf(symbol, sizeof(symbol), "g_llama_matmul_%s_weight", name);
        print_float_array(fp, symbol, weight, weight_elems);
        snprintf(symbol, sizeof(symbol), "g_llama_matmul_%s_weight_f64", name);
        print_double_array(fp, symbol, weight_f64, weight_elems);

        snprintf(symbol, sizeof(symbol), "g_llama_matmul_%s_rhs", name);
        print_float_array(fp, symbol, rhs, rhs_elems);
        snprintf(symbol, sizeof(symbol), "g_llama_matmul_%s_rhs_f64", name);
        print_double_array(fp, symbol, rhs_f64, rhs_elems);

        snprintf(symbol, sizeof(symbol), "g_llama_matmul_%s_expected", name);
        print_float_array(fp, symbol, expected, expected_elems);
        snprintf(symbol, sizeof(symbol), "g_llama_matmul_%s_expected_f64", name);
        print_double_array(fp, symbol, expected_f64, expected_elems);

        free(weight);
        free(rhs);
        free(expected);
        free(weight_f64);
        free(rhs_f64);
        free(expected_f64);
    }

    fprintf(fp, "static const llama_matmul_case_t g_llama_matmul_cases[] = {\n");
    for (size_t idx = 0; idx < sizeof(matmul_cfgs) / sizeof(matmul_cfgs[0]); ++idx) {
        const char *name = matmul_cfgs[idx].name;
        fprintf(
            fp,
            "    {\"%s\", %d, %d, %d, g_llama_matmul_%s_weight, g_llama_matmul_%s_rhs, g_llama_matmul_%s_expected, "
            "g_llama_matmul_%s_weight_f64, g_llama_matmul_%s_rhs_f64, g_llama_matmul_%s_expected_f64},\n",
            name, matmul_cfgs[idx].M, matmul_cfgs[idx].K, matmul_cfgs[idx].N, name, name, name, name, name, name
        );
    }
    fprintf(fp, "};\n");
    fprintf(
        fp,
        "static const size_t g_llama_matmul_case_count = sizeof(g_llama_matmul_cases) / "
        "sizeof(g_llama_matmul_cases[0]);\n\n"
    );

    // ------------------------------------------------------------------
    // LayerNorm fixtures
    // ------------------------------------------------------------------
    static const struct {
        const char *name;
        size_t rows;
        size_t cols;
        float eps;
        bool with_residual;
        bool with_affine;
    } layernorm_cfgs[] = {
        {"ln_affine", 2, 8, 1e-5f, false, true},
        {"ln_residual_affine", 3, 6, 1e-5f, true, true},
        {"ln_no_affine", 2, 10, 1e-5f, false, false},
    };

    fprintf(fp, "// LayerNorm fixtures\n");

    for (size_t idx = 0; idx < sizeof(layernorm_cfgs) / sizeof(layernorm_cfgs[0]); ++idx) {
        const char *name = layernorm_cfgs[idx].name;
        const size_t rows = layernorm_cfgs[idx].rows;
        const size_t cols = layernorm_cfgs[idx].cols;
        const float eps = layernorm_cfgs[idx].eps;
        const bool with_residual = layernorm_cfgs[idx].with_residual;
        const bool with_affine = layernorm_cfgs[idx].with_affine;

        const size_t elem_count = rows * cols;
        float *input = (float *)malloc(elem_count * sizeof(float));
        float *residual = with_residual ? (float *)malloc(elem_count * sizeof(float)) : nullptr;
        float *weight = with_affine ? (float *)malloc(cols * sizeof(float)) : nullptr;
        float *bias = with_affine ? (float *)malloc(cols * sizeof(float)) : nullptr;
        float *expected = (float *)malloc(elem_count * sizeof(float));
        double *input_f64 = (double *)malloc(elem_count * sizeof(double));
        double *residual_f64 = with_residual ? (double *)malloc(elem_count * sizeof(double)) : nullptr;
        double *weight_f64 = with_affine ? (double *)malloc(cols * sizeof(double)) : nullptr;
        double *bias_f64 = with_affine ? (double *)malloc(cols * sizeof(double)) : nullptr;
        double *expected_f64 = (double *)malloc(elem_count * sizeof(double));
        if (input == nullptr || expected == nullptr || input_f64 == nullptr || expected_f64 == nullptr ||
            (with_affine && (weight == nullptr || bias == nullptr || weight_f64 == nullptr || bias_f64 == nullptr)) ||
            (with_residual && (residual == nullptr || residual_f64 == nullptr))) {
            fprintf(stderr, "Allocation failure generating layernorm fixture %s\n", name);
            exit(EXIT_FAILURE);
        }

        fill_activation_pattern(input, (int)rows, (int)cols);
        if (with_residual) {
            fill_weight_pattern(residual, (int)rows, (int)cols);
        }
        if (with_affine) {
            for (size_t c = 0; c < cols; ++c) {
                weight[c] = 0.5f + 0.1f * (float)c;
                bias[c] = -0.2f + 0.05f * (float)c;
            }
        }

        layernorm_reference(input, residual, weight, bias, rows, cols, eps, expected);

        for (size_t i = 0; i < elem_count; ++i) {
            input_f64[i] = (double)input[i];
            expected_f64[i] = (double)expected[i];
            if (with_residual) {
                residual_f64[i] = (double)residual[i];
            }
        }
        if (with_affine) {
            for (size_t c = 0; c < cols; ++c) {
                weight_f64[c] = (double)weight[c];
                bias_f64[c] = (double)bias[c];
            }
        }

        char symbol[128];
        snprintf(symbol, sizeof(symbol), "g_llama_layernorm_%s_input", name);
        print_float_array(fp, symbol, input, elem_count);
        snprintf(symbol, sizeof(symbol), "g_llama_layernorm_%s_input_f64", name);
        print_double_array(fp, symbol, input_f64, elem_count);
        if (with_residual) {
            snprintf(symbol, sizeof(symbol), "g_llama_layernorm_%s_residual", name);
            print_float_array(fp, symbol, residual, elem_count);
            snprintf(symbol, sizeof(symbol), "g_llama_layernorm_%s_residual_f64", name);
            print_double_array(fp, symbol, residual_f64, elem_count);
        } else {
            fprintf(fp, "static const float *g_llama_layernorm_%s_residual = nullptr;\n\n", name);
            fprintf(fp, "static const double *g_llama_layernorm_%s_residual_f64 = nullptr;\n\n", name);
        }
        if (with_affine) {
            snprintf(symbol, sizeof(symbol), "g_llama_layernorm_%s_weight", name);
            print_float_array(fp, symbol, weight, cols);
            snprintf(symbol, sizeof(symbol), "g_llama_layernorm_%s_weight_f64", name);
            print_double_array(fp, symbol, weight_f64, cols);
            snprintf(symbol, sizeof(symbol), "g_llama_layernorm_%s_bias", name);
            print_float_array(fp, symbol, bias, cols);
            snprintf(symbol, sizeof(symbol), "g_llama_layernorm_%s_bias_f64", name);
            print_double_array(fp, symbol, bias_f64, cols);
        } else {
            fprintf(fp, "static const float *g_llama_layernorm_%s_weight = nullptr;\n", name);
            fprintf(fp, "static const float *g_llama_layernorm_%s_bias = nullptr;\n\n", name);
            fprintf(fp, "static const double *g_llama_layernorm_%s_weight_f64 = nullptr;\n", name);
            fprintf(fp, "static const double *g_llama_layernorm_%s_bias_f64 = nullptr;\n\n", name);
        }
        snprintf(symbol, sizeof(symbol), "g_llama_layernorm_%s_expected", name);
        print_float_array(fp, symbol, expected, elem_count);
        snprintf(symbol, sizeof(symbol), "g_llama_layernorm_%s_expected_f64", name);
        print_double_array(fp, symbol, expected_f64, elem_count);

        free(input);
        free(expected);
        free(input_f64);
        free(expected_f64);
        if (residual) {
            free(residual);
        }
        if (residual_f64) {
            free(residual_f64);
        }
        if (weight) {
            free(weight);
        }
        if (bias) {
            free(bias);
        }
        if (weight_f64) {
            free(weight_f64);
        }
        if (bias_f64) {
            free(bias_f64);
        }
    }

    fprintf(fp, "static const llama_layernorm_case_t g_llama_layernorm_cases[] = {\n");
    for (size_t idx = 0; idx < sizeof(layernorm_cfgs) / sizeof(layernorm_cfgs[0]); ++idx) {
        const char *name = layernorm_cfgs[idx].name;
        char residual_symbol[160];
        char residual_symbol_f64[160];
        char weight_symbol[160];
        char weight_symbol_f64[160];
        char bias_symbol[160];
        char bias_symbol_f64[160];
        if (layernorm_cfgs[idx].with_residual) {
            snprintf(residual_symbol, sizeof(residual_symbol), "g_llama_layernorm_%s_residual", name);
            snprintf(residual_symbol_f64, sizeof(residual_symbol_f64), "g_llama_layernorm_%s_residual_f64", name);
        } else {
            snprintf(residual_symbol, sizeof(residual_symbol), "nullptr");
            snprintf(residual_symbol_f64, sizeof(residual_symbol_f64), "nullptr");
        }
        if (layernorm_cfgs[idx].with_affine) {
            snprintf(weight_symbol, sizeof(weight_symbol), "g_llama_layernorm_%s_weight", name);
            snprintf(weight_symbol_f64, sizeof(weight_symbol_f64), "g_llama_layernorm_%s_weight_f64", name);
            snprintf(bias_symbol, sizeof(bias_symbol), "g_llama_layernorm_%s_bias", name);
            snprintf(bias_symbol_f64, sizeof(bias_symbol_f64), "g_llama_layernorm_%s_bias_f64", name);
        } else {
            snprintf(weight_symbol, sizeof(weight_symbol), "nullptr");
            snprintf(weight_symbol_f64, sizeof(weight_symbol_f64), "nullptr");
            snprintf(bias_symbol, sizeof(bias_symbol), "nullptr");
            snprintf(bias_symbol_f64, sizeof(bias_symbol_f64), "nullptr");
        }
        fprintf(
            fp,
            "    {\"%s\", %zu, %zu, %.8ff, g_llama_layernorm_%s_input, %s, %s, %s, g_llama_layernorm_%s_expected, "
            "g_llama_layernorm_%s_input_f64, %s, %s, %s, g_llama_layernorm_%s_expected_f64},\n",
            name, layernorm_cfgs[idx].rows, layernorm_cfgs[idx].cols, layernorm_cfgs[idx].eps, name, residual_symbol,
            weight_symbol, bias_symbol, name, name, residual_symbol_f64, weight_symbol_f64, bias_symbol_f64, name
        );
    }
    fprintf(fp, "};\n");
    fprintf(
        fp,
        "static const size_t g_llama_layernorm_case_count = sizeof(g_llama_layernorm_cases) / "
        "sizeof(g_llama_layernorm_cases[0]);\n\n"
    );

    // ------------------------------------------------------------------
    // RMSNorm fixtures
    // ------------------------------------------------------------------
    static const struct {
        const char *name;
        size_t rows;
        size_t cols;
        float eps;
        bool with_residual;
        bool with_weight;
    } rms_cfgs[] = {
        {"rms_affine", 2, 8, 1e-6f, false, true},
        {"rms_residual", 3, 6, 1e-6f, true, true},
        {"rms_plain", 2, 12, 1e-6f, false, false},
    };

    fprintf(fp, "// RMSNorm fixtures\n");

    for (size_t idx = 0; idx < sizeof(rms_cfgs) / sizeof(rms_cfgs[0]); ++idx) {
        const char *name = rms_cfgs[idx].name;
        const size_t rows = rms_cfgs[idx].rows;
        const size_t cols = rms_cfgs[idx].cols;
        const float eps = rms_cfgs[idx].eps;
        const bool with_residual = rms_cfgs[idx].with_residual;
        const bool with_weight = rms_cfgs[idx].with_weight;

        const size_t elem_count = rows * cols;
        float *input = (float *)malloc(elem_count * sizeof(float));
        float *residual = with_residual ? (float *)malloc(elem_count * sizeof(float)) : nullptr;
        float *weight = with_weight ? (float *)malloc(cols * sizeof(float)) : nullptr;
        float *expected = (float *)malloc(elem_count * sizeof(float));
        double *input_f64 = (double *)malloc(elem_count * sizeof(double));
        double *residual_f64 = with_residual ? (double *)malloc(elem_count * sizeof(double)) : nullptr;
        double *weight_f64 = with_weight ? (double *)malloc(cols * sizeof(double)) : nullptr;
        double *expected_f64 = (double *)malloc(elem_count * sizeof(double));
        if (input == nullptr || expected == nullptr || input_f64 == nullptr || expected_f64 == nullptr ||
            (with_residual && (residual == nullptr || residual_f64 == nullptr)) ||
            (with_weight && (weight == nullptr || weight_f64 == nullptr))) {
            fprintf(stderr, "Allocation failure generating rmsnorm fixture %s\n", name);
            exit(EXIT_FAILURE);
        }

        fill_activation_pattern(input, (int)rows, (int)cols);
        if (with_residual) {
            fill_weight_pattern(residual, (int)rows, (int)cols);
        }
        if (with_weight) {
            for (size_t c = 0; c < cols; ++c) {
                weight[c] = 0.8f + 0.02f * (float)c;
            }
        }

        rmsnorm_reference(input, residual, weight, rows, cols, eps, expected);

        for (size_t i = 0; i < elem_count; ++i) {
            input_f64[i] = (double)input[i];
            expected_f64[i] = (double)expected[i];
            if (with_residual) {
                residual_f64[i] = (double)residual[i];
            }
        }
        if (with_weight) {
            for (size_t c = 0; c < cols; ++c) {
                weight_f64[c] = (double)weight[c];
            }
        }

        char symbol[128];
        snprintf(symbol, sizeof(symbol), "g_llama_rmsnorm_%s_input", name);
        print_float_array(fp, symbol, input, elem_count);
        snprintf(symbol, sizeof(symbol), "g_llama_rmsnorm_%s_input_f64", name);
        print_double_array(fp, symbol, input_f64, elem_count);
        if (with_residual) {
            snprintf(symbol, sizeof(symbol), "g_llama_rmsnorm_%s_residual", name);
            print_float_array(fp, symbol, residual, elem_count);
            snprintf(symbol, sizeof(symbol), "g_llama_rmsnorm_%s_residual_f64", name);
            print_double_array(fp, symbol, residual_f64, elem_count);
        } else {
            fprintf(fp, "static const float *g_llama_rmsnorm_%s_residual = nullptr;\n\n", name);
            fprintf(fp, "static const double *g_llama_rmsnorm_%s_residual_f64 = nullptr;\n\n", name);
        }
        if (with_weight) {
            snprintf(symbol, sizeof(symbol), "g_llama_rmsnorm_%s_weight", name);
            print_float_array(fp, symbol, weight, cols);
            snprintf(symbol, sizeof(symbol), "g_llama_rmsnorm_%s_weight_f64", name);
            print_double_array(fp, symbol, weight_f64, cols);
        } else {
            fprintf(fp, "static const float *g_llama_rmsnorm_%s_weight = nullptr;\n", name);
            fprintf(fp, "static const double *g_llama_rmsnorm_%s_weight_f64 = nullptr;\n", name);
        }
        snprintf(symbol, sizeof(symbol), "g_llama_rmsnorm_%s_expected", name);
        print_float_array(fp, symbol, expected, elem_count);
        snprintf(symbol, sizeof(symbol), "g_llama_rmsnorm_%s_expected_f64", name);
        print_double_array(fp, symbol, expected_f64, elem_count);

        free(input);
        free(expected);
        free(input_f64);
        free(expected_f64);
        if (residual) {
            free(residual);
        }
        if (residual_f64) {
            free(residual_f64);
        }
        if (weight) {
            free(weight);
        }
        if (weight_f64) {
            free(weight_f64);
        }
    }

    fprintf(fp, "static const llama_rmsnorm_case_t g_llama_rmsnorm_cases[] = {\n");
    for (size_t idx = 0; idx < sizeof(rms_cfgs) / sizeof(rms_cfgs[0]); ++idx) {
        const char *name = rms_cfgs[idx].name;
        char residual_symbol[160];
        char residual_symbol_f64[160];
        char weight_symbol[160];
        char weight_symbol_f64[160];
        if (rms_cfgs[idx].with_residual) {
            snprintf(residual_symbol, sizeof(residual_symbol), "g_llama_rmsnorm_%s_residual", name);
            snprintf(residual_symbol_f64, sizeof(residual_symbol_f64), "g_llama_rmsnorm_%s_residual_f64", name);
        } else {
            snprintf(residual_symbol, sizeof(residual_symbol), "nullptr");
            snprintf(residual_symbol_f64, sizeof(residual_symbol_f64), "nullptr");
        }
        if (rms_cfgs[idx].with_weight) {
            snprintf(weight_symbol, sizeof(weight_symbol), "g_llama_rmsnorm_%s_weight", name);
            snprintf(weight_symbol_f64, sizeof(weight_symbol_f64), "g_llama_rmsnorm_%s_weight_f64", name);
        } else {
            snprintf(weight_symbol, sizeof(weight_symbol), "nullptr");
            snprintf(weight_symbol_f64, sizeof(weight_symbol_f64), "nullptr");
        }
        fprintf(
            fp,
            "    {\"%s\", %zu, %zu, %.8ff, g_llama_rmsnorm_%s_input, %s, %s, g_llama_rmsnorm_%s_expected, "
            "g_llama_rmsnorm_%s_input_f64, %s, %s, g_llama_rmsnorm_%s_expected_f64},\n",
            name, rms_cfgs[idx].rows, rms_cfgs[idx].cols, rms_cfgs[idx].eps, name, residual_symbol, weight_symbol, name,
            name, residual_symbol_f64, weight_symbol_f64, name
        );
    }
    fprintf(fp, "};\n");
    fprintf(
        fp,
        "static const size_t g_llama_rmsnorm_case_count = sizeof(g_llama_rmsnorm_cases) / "
        "sizeof(g_llama_rmsnorm_cases[0]);\n\n"
    );

    // ------------------------------------------------------------------
    // Activation fixtures
    // ------------------------------------------------------------------
    const size_t activation_len = 16;
    float activation_input[activation_len];
    for (size_t i = 0; i < activation_len; ++i) {
        activation_input[i] = test_input[i] * 0.5f;
    }

    float activation_relu[activation_len];
    float activation_gelu[activation_len];
    float activation_gelu_tanh[activation_len];
    float activation_silu[activation_len];
    float activation_sigmoid[activation_len];
    float activation_tanh[activation_len];
    float activation_mish[activation_len];
    float activation_elu[activation_len];
    float activation_selu[activation_len];
    float activation_leaky[activation_len];
    float activation_prelu[activation_len];
    double activation_input_f64[activation_len];
    double activation_relu_f64[activation_len];
    double activation_gelu_f64[activation_len];
    double activation_gelu_tanh_f64[activation_len];
    double activation_silu_f64[activation_len];
    double activation_sigmoid_f64[activation_len];
    double activation_tanh_f64[activation_len];
    double activation_mish_f64[activation_len];
    double activation_elu_f64[activation_len];
    double activation_selu_f64[activation_len];
    double activation_leaky_f64[activation_len];
    double activation_prelu_f64[activation_len];

    const float elu_alpha = 1.1f;
    const float selu_alpha = 1.6732632423543772f;
    const float selu_lambda = 1.0507009873554804f;
    const float leaky_slope = 0.02f;
    const float prelu_slope = 0.25f;

    for (size_t i = 0; i < activation_len; ++i) {
        const float x = activation_input[i];
        activation_input_f64[i] = (double)x;
        activation_relu[i] = x > 0.0f ? x : 0.0f;
        activation_gelu[i] = gelu_erf(x);
        activation_gelu_tanh[i] = gelu_tanh_approx(x);
        activation_silu[i] = silu_fn(x);
        activation_sigmoid[i] = sigmoid_fn(x);
        activation_tanh[i] = tanhf(x);
        activation_mish[i] = mish_fn(x);
        activation_elu[i] = elu_fn(x, elu_alpha);
        activation_selu[i] = selu_fn(x, selu_alpha, selu_lambda);
        activation_leaky[i] = leaky_relu_fn(x, leaky_slope);
        activation_prelu[i] = prelu_fn(x, prelu_slope);
        activation_relu_f64[i] = (double)activation_relu[i];
        activation_gelu_f64[i] = (double)activation_gelu[i];
        activation_gelu_tanh_f64[i] = (double)activation_gelu_tanh[i];
        activation_silu_f64[i] = (double)activation_silu[i];
        activation_sigmoid_f64[i] = (double)activation_sigmoid[i];
        activation_tanh_f64[i] = (double)activation_tanh[i];
        activation_mish_f64[i] = (double)activation_mish[i];
        activation_elu_f64[i] = (double)activation_elu[i];
        activation_selu_f64[i] = (double)activation_selu[i];
        activation_leaky_f64[i] = (double)activation_leaky[i];
        activation_prelu_f64[i] = (double)activation_prelu[i];
    }

    print_float_array(fp, "g_llama_activation_input", activation_input, activation_len);
    print_float_array(fp, "g_llama_activation_relu", activation_relu, activation_len);
    print_float_array(fp, "g_llama_activation_gelu", activation_gelu, activation_len);
    print_float_array(fp, "g_llama_activation_gelu_tanh", activation_gelu_tanh, activation_len);
    print_float_array(fp, "g_llama_activation_silu", activation_silu, activation_len);
    print_float_array(fp, "g_llama_activation_sigmoid", activation_sigmoid, activation_len);
    print_float_array(fp, "g_llama_activation_tanh", activation_tanh, activation_len);
    print_float_array(fp, "g_llama_activation_mish", activation_mish, activation_len);
    print_float_array(fp, "g_llama_activation_elu", activation_elu, activation_len);
    print_float_array(fp, "g_llama_activation_selu", activation_selu, activation_len);
    print_float_array(fp, "g_llama_activation_leaky", activation_leaky, activation_len);
    print_float_array(fp, "g_llama_activation_prelu", activation_prelu, activation_len);
    print_double_array(fp, "g_llama_activation_input_f64", activation_input_f64, activation_len);
    print_double_array(fp, "g_llama_activation_relu_f64", activation_relu_f64, activation_len);
    print_double_array(fp, "g_llama_activation_gelu_f64", activation_gelu_f64, activation_len);
    print_double_array(fp, "g_llama_activation_gelu_tanh_f64", activation_gelu_tanh_f64, activation_len);
    print_double_array(fp, "g_llama_activation_silu_f64", activation_silu_f64, activation_len);
    print_double_array(fp, "g_llama_activation_sigmoid_f64", activation_sigmoid_f64, activation_len);
    print_double_array(fp, "g_llama_activation_tanh_f64", activation_tanh_f64, activation_len);
    print_double_array(fp, "g_llama_activation_mish_f64", activation_mish_f64, activation_len);
    print_double_array(fp, "g_llama_activation_elu_f64", activation_elu_f64, activation_len);
    print_double_array(fp, "g_llama_activation_selu_f64", activation_selu_f64, activation_len);
    print_double_array(fp, "g_llama_activation_leaky_f64", activation_leaky_f64, activation_len);
    print_double_array(fp, "g_llama_activation_prelu_f64", activation_prelu_f64, activation_len);

    fprintf(
        fp,
        "static const llama_activation_golden_t g_llama_activation_golden = {\n"
        "    .length = %zu,\n"
        "    .elu_alpha = %.9ff,\n"
        "    .selu_alpha = %.16ff,\n"
        "    .selu_lambda = %.16ff,\n"
        "    .leaky_slope = %.9ff,\n"
        "    .prelu_slope = %.9ff,\n"
        "    .input = g_llama_activation_input,\n"
        "    .relu = g_llama_activation_relu,\n"
        "    .gelu = g_llama_activation_gelu,\n"
        "    .gelu_tanh = g_llama_activation_gelu_tanh,\n"
        "    .silu = g_llama_activation_silu,\n"
        "    .sigmoid = g_llama_activation_sigmoid,\n"
        "    .tanh_v = g_llama_activation_tanh,\n"
        "    .mish = g_llama_activation_mish,\n"
        "    .elu = g_llama_activation_elu,\n"
        "    .selu = g_llama_activation_selu,\n"
        "    .leaky_relu = g_llama_activation_leaky,\n"
        "    .prelu = g_llama_activation_prelu,\n"
        "    .input_f64 = g_llama_activation_input_f64,\n"
        "    .relu_f64 = g_llama_activation_relu_f64,\n"
        "    .gelu_f64 = g_llama_activation_gelu_f64,\n"
        "    .gelu_tanh_f64 = g_llama_activation_gelu_tanh_f64,\n"
        "    .silu_f64 = g_llama_activation_silu_f64,\n"
        "    .sigmoid_f64 = g_llama_activation_sigmoid_f64,\n"
        "    .tanh_v_f64 = g_llama_activation_tanh_f64,\n"
        "    .mish_f64 = g_llama_activation_mish_f64,\n"
        "    .elu_f64 = g_llama_activation_elu_f64,\n"
        "    .selu_f64 = g_llama_activation_selu_f64,\n"
        "    .leaky_relu_f64 = g_llama_activation_leaky_f64,\n"
        "    .prelu_f64 = g_llama_activation_prelu_f64,\n"
        "};\n",
        activation_len, elu_alpha, selu_alpha, selu_lambda, leaky_slope, prelu_slope
    );
}
