// Benchmark for quantized matrix multiplication (Q4_0, Q4_K, Q8_0, Q8_K)
// Uses marmot_linear API which performs: input(N×K) @ weight(M×K).T = output(N×M)

#include "marmot/marmot.h"
#include "marmot/ops/matmul.h"
#include "marmot/ops/quantization.h"
#include "marmot/quant_block.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../bench_workloads.h"

static inline marmot_float16_t bench_f32_to_f16(float value) {
    union {
        float f;
        uint32_t u;
    } bits = {.f = value};
    uint32_t sign = (bits.u >> 16) & 0x8000;
    int32_t exp = ((bits.u >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = bits.u & 0x7fffff;
    if (exp <= 0) {
        return marmot_make_f16((uint16_t)sign);
    } else if (exp >= 31) {
        return marmot_make_f16((uint16_t)(sign | 0x7c00));
    }
    return marmot_make_f16((uint16_t)(sign | (exp << 10) | (mantissa >> 13)));
}

typedef struct {
    uint32_t N;
    uint32_t K;
    uint32_t M;
    marmot_quant_kind_t quant_kind;
    marmot_dtype_t input_dtype;
    char name[64];
} matmul_quant_params_t;

static void generate_random_weights(float *data, size_t count, uint32_t seed) {
    for (size_t i = 0; i < count; ++i) {
        seed = seed * 1103515245u + 12345u;
        float val = (float)((seed >> 16) & 0x7fff) / 32767.0f;
        data[i] = val * 2.0f - 1.0f;
    }
}

static void generate_random_input(float *data, size_t count, uint32_t seed) {
    for (size_t i = 0; i < count; ++i) {
        seed = seed * 1103515245u + 12345u;
        float val = (float)((seed >> 16) & 0x7fff) / 32767.0f;
        data[i] = val * 1.5f - 0.75f;
    }
}

typedef marmot_error_t (*quantize_fn_t)(const marmot_context_t *, const marmot_tensor_t *, marmot_tensor_t *);

static quantize_fn_t get_quantize_fn(marmot_quant_kind_t kind) {
    switch (kind) {
    case MARMOT_QUANT_KIND_Q4_0:
        return marmot_quantize_q4_0;
    case MARMOT_QUANT_KIND_Q4_K:
        return marmot_quantize_q4_k;
    case MARMOT_QUANT_KIND_Q8_0:
        return marmot_quantize_q8_0;
    case MARMOT_QUANT_KIND_Q8_K:
        return marmot_quantize_q8_k;
    default:
        return nullptr;
    }
}

static marmot_error_t matmul_quant_setup(
    marmot_backend_type_t backend, marmot_context_t *ctx, marmot_graph_t **graph, marmot_tensor_t ***inputs,
    size_t *num_inputs, marmot_tensor_t ***outputs, size_t *num_outputs, void *user_data
) {
    matmul_quant_params_t *params = (matmul_quant_params_t *)user_data;
    (void)graph;

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

    marmot_tensor_t *input = marmot_tensor_create(ctx, input_shape, 2, params->input_dtype);
    if (input == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    marmot_tensor_t *weight = marmot_tensor_create_quantized(ctx, weight_shape, 2, params->quant_kind);
    if (weight == nullptr) {
        marmot_tensor_destroy(input);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    marmot_tensor_t *output = marmot_tensor_create(ctx, output_shape, 2, MARMOT_DTYPE_FLOAT32);
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
    generate_random_weights(weight_f32, params->M * params->K, 42);

    marmot_tensor_t *weight_src = marmot_tensor_create(ctx, weight_shape, 2, MARMOT_DTYPE_FLOAT32);
    if (weight_src == nullptr) {
        free(weight_f32);
        marmot_tensor_destroy(input);
        marmot_tensor_destroy(weight);
        marmot_tensor_destroy(output);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    marmot_error_t err =
        marmot_tensor_copy_from_host_buffer(ctx, weight_src, weight_f32, params->M * params->K * sizeof(float));
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
    generate_random_input(input_f32, params->N * params->K, 123);

    if (params->input_dtype == MARMOT_DTYPE_FLOAT32) {
        err = marmot_tensor_copy_from_host_buffer(ctx, input, input_f32, params->N * params->K * sizeof(float));
    } else if (params->input_dtype == MARMOT_DTYPE_FLOAT16) {
        marmot_float16_t *input_f16 = (marmot_float16_t *)malloc(params->N * params->K * sizeof(marmot_float16_t));
        if (input_f16 == nullptr) {
            free(input_f32);
            marmot_tensor_destroy(input);
            marmot_tensor_destroy(weight);
            marmot_tensor_destroy(output);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        for (size_t i = 0; i < params->N * params->K; ++i) {
            input_f16[i] = bench_f32_to_f16(input_f32[i]);
        }
        err = marmot_tensor_copy_from_host_buffer(ctx, input, input_f16, params->N * params->K * sizeof(marmot_float16_t));
        free(input_f16);
    } else {
        err = MARMOT_ERROR_INVALID_ARGUMENT;
    }
    free(input_f32);

    if (err != MARMOT_SUCCESS) {
        marmot_tensor_destroy(input);
        marmot_tensor_destroy(weight);
        marmot_tensor_destroy(output);
        return err;
    }

    *inputs = malloc(2 * sizeof(marmot_tensor_t *));
    (*inputs)[0] = input;
    (*inputs)[1] = weight;
    *num_inputs = 2;

    *outputs = malloc(1 * sizeof(marmot_tensor_t *));
    (*outputs)[0] = output;
    *num_outputs = 1;

    return MARMOT_SUCCESS;
}

static marmot_error_t matmul_quant_execute(
    marmot_context_t *ctx, marmot_graph_t *graph, marmot_tensor_t **inputs, size_t num_inputs,
    marmot_tensor_t **outputs, size_t num_outputs
) {
    (void)graph;
    (void)num_inputs;
    (void)num_outputs;
    return marmot_linear(ctx, inputs[0], inputs[1], nullptr, outputs[0]);
}

static void matmul_quant_teardown(
    marmot_graph_t *graph, marmot_tensor_t **inputs, size_t num_inputs, marmot_tensor_t **outputs, size_t num_outputs,
    void *user_data
) {
    (void)graph;
    (void)user_data;

    for (size_t i = 0; i < num_inputs; ++i) {
        marmot_tensor_destroy(inputs[i]);
    }
    for (size_t i = 0; i < num_outputs; ++i) {
        marmot_tensor_destroy(outputs[i]);
    }
    free(inputs);
    free(outputs);
}

static const char *quant_kind_to_str(marmot_quant_kind_t kind) {
    switch (kind) {
    case MARMOT_QUANT_KIND_Q4_0:
        return "q4_0";
    case MARMOT_QUANT_KIND_Q4_K:
        return "q4_k";
    case MARMOT_QUANT_KIND_Q8_0:
        return "q8_0";
    case MARMOT_QUANT_KIND_Q8_K:
        return "q8_k";
    default:
        return "unk";
    }
}

static const char *dtype_to_str(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT16:
        return "f16";
    case MARMOT_DTYPE_FLOAT32:
        return "f32";
    default:
        return "unk";
    }
}

static uint32_t quant_bits_per_weight(marmot_quant_kind_t kind) {
    switch (kind) {
    case MARMOT_QUANT_KIND_Q4_0:
    case MARMOT_QUANT_KIND_Q4_K:
        return 4;
    case MARMOT_QUANT_KIND_Q8_0:
    case MARMOT_QUANT_KIND_Q8_K:
        return 8;
    default:
        return 8;
    }
}

static size_t dtype_size(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return 4;
    case MARMOT_DTYPE_FLOAT16:
        return 2;
    default:
        return 4;
    }
}

static marmot_bench_workload_t *create_matmul_quant_workload(
    uint32_t N, uint32_t K, uint32_t M, marmot_quant_kind_t quant_kind, marmot_dtype_t input_dtype
) {
    marmot_bench_workload_t *w = calloc(1, sizeof(marmot_bench_workload_t));
    if (w == nullptr) {
        return nullptr;
    }

    matmul_quant_params_t *params = calloc(1, sizeof(matmul_quant_params_t));
    if (params == nullptr) {
        free(w);
        return nullptr;
    }

    params->N = N;
    params->K = K;
    params->M = M;
    params->quant_kind = quant_kind;
    params->input_dtype = input_dtype;

    snprintf(
        params->name, sizeof(params->name), "linear_%s_%s_%ux%ux%u", quant_kind_to_str(quant_kind),
        dtype_to_str(input_dtype), N, M, K
    );

    uint32_t weight_bits = quant_bits_per_weight(quant_kind);
    uint64_t flops = 2ULL * N * K * M;
    uint64_t weight_bytes = (uint64_t)M * K * weight_bits / 8;
    uint64_t input_bytes = (uint64_t)N * K * dtype_size(input_dtype);
    uint64_t output_bytes = (uint64_t)N * M * sizeof(float);

    w->desc.name = params->name;
    w->desc.category = MARMOT_BENCH_CATEGORY_MICRO;
    w->desc.primary_dtype = input_dtype;
    w->desc.flops = flops;
    w->desc.bytes_read = weight_bytes + input_bytes;
    w->desc.bytes_written = output_bytes;

    memset(&w->desc.signature, 0, sizeof(w->desc.signature));
    w->desc.signature.op_id = MARMOT_OP_LINEAR;
    w->desc.signature.profile_id = MARMOT_PROFILE_INVALID;
    w->desc.signature.matmul_layout = MARMOT_MATMUL_LAYOUT_NT;
    w->desc.signature.input_dtype = input_dtype;
    w->desc.signature.weight_dtype = MARMOT_DTYPE_UINT8;
    w->desc.signature.output_dtype = MARMOT_DTYPE_FLOAT32;
    w->desc.signature.accum_dtype = MARMOT_DTYPE_FLOAT32;
    w->desc.signature.dims.matmul.N = N;
    w->desc.signature.dims.matmul.M = M;
    w->desc.signature.dims.matmul.K = K;

    w->user_data = params;
    w->setup = matmul_quant_setup;
    w->execute = matmul_quant_execute;
    w->teardown = matmul_quant_teardown;

    return w;
}

void marmot_bench_register_matmul_quant_workloads(marmot_bench_suite_t *suite) {
    // Matrix shapes for benchmarking:
    // - GEMV (batch=1): token generation, memory-bound
    // - Small batch: small prompt processing
    // - Medium batch: typical inference
    // - LLM-realistic: FFN and vocab projection shapes
    static const uint32_t shapes[][3] = {
        // N,    K,     M     (input NxK, weight MxK, output NxM)
        {1, 4096, 4096},      // GEMV - token generation
        {8, 4096, 4096},      // Small batch
        {32, 4096, 4096},     // Medium batch - typical inference
        {32, 4096, 11008},    // LLaMA FFN up-projection
        {32, 11008, 4096},    // LLaMA FFN down-projection
        {32, 4096, 32000},    // LLaMA vocab projection
        {512, 4096, 4096},    // Large prefill
    };
    static const size_t num_shapes = sizeof(shapes) / sizeof(shapes[0]);

    // Quantization types to benchmark
    static const marmot_quant_kind_t quant_kinds[] = {
        MARMOT_QUANT_KIND_Q4_0,
        MARMOT_QUANT_KIND_Q4_K,
        MARMOT_QUANT_KIND_Q8_0,
        MARMOT_QUANT_KIND_Q8_K,
    };
    static const size_t num_quant_kinds = sizeof(quant_kinds) / sizeof(quant_kinds[0]);

    // Input dtypes to benchmark
    static const marmot_dtype_t input_dtypes[] = {
        MARMOT_DTYPE_FLOAT32,
        MARMOT_DTYPE_FLOAT16,
    };
    static const size_t num_input_dtypes = sizeof(input_dtypes) / sizeof(input_dtypes[0]);

    for (size_t i = 0; i < num_shapes; ++i) {
        uint32_t N = shapes[i][0];
        uint32_t K = shapes[i][1];
        uint32_t M = shapes[i][2];

        for (size_t j = 0; j < num_quant_kinds; ++j) {
            for (size_t k = 0; k < num_input_dtypes; ++k) {
                marmot_bench_workload_t *w = create_matmul_quant_workload(N, K, M, quant_kinds[j], input_dtypes[k]);
                if (w) {
                    marmot_bench_suite_add(suite, w);
                }
            }
        }
    }
}
