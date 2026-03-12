#include "../bench_workloads.h"

#include "backends/cpu/ops/matmul/quantized/matmul_quant_internal.h"
#include "marmot/device.h"
#include "marmot/ops/matmul.h"
#include "marmot/ops/quantization.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    BENCH_LOGITS_MODE_DEFAULT = 0,
    BENCH_LOGITS_MODE_OUTPUT = 1,
    BENCH_LOGITS_MODE_OUTPUT_PREPACKED = 2,
    BENCH_LOGITS_MODE_OUTPUT_RAW = 3,
} bench_logits_mode_t;

typedef struct {
    uint32_t batch;
    uint32_t hidden;
    uint32_t vocab;
    marmot_quant_kind_t quant_kind;
    bench_logits_mode_t mode;
    char name[128];
} bench_logits_params_t;

typedef marmot_error_t (*bench_quantize_fn_t)(const marmot_context_t *, const marmot_tensor_t *, marmot_tensor_t *);

static void bench_logits_fill_f32(float *data, size_t count, uint32_t seed, float scale) {
    for (size_t i = 0; i < count; ++i) {
        seed = seed * 1103515245u + 12345u;
        const float value = (float)((seed >> 16) & 0x7fff) / 32767.0f;
        data[i] = (value * 2.0f - 1.0f) * scale;
    }
}

static bench_quantize_fn_t bench_logits_quantize_fn(marmot_quant_kind_t kind) {
    switch (kind) {
    case MARMOT_QUANT_KIND_Q4_K:
        return marmot_quantize_q4_k;
    case MARMOT_QUANT_KIND_Q6_K:
        return marmot_quantize_q6_k;
    default:
        return nullptr;
    }
}

static const char *bench_logits_quant_kind_str(marmot_quant_kind_t kind) {
    switch (kind) {
    case MARMOT_QUANT_KIND_Q4_K:
        return "q4_k";
    case MARMOT_QUANT_KIND_Q6_K:
        return "q6_k";
    default:
        return "quant";
    }
}

static const char *bench_logits_mode_str(bench_logits_mode_t mode) {
    switch (mode) {
    case BENCH_LOGITS_MODE_DEFAULT:
        return "default";
    case BENCH_LOGITS_MODE_OUTPUT:
        return "output";
    case BENCH_LOGITS_MODE_OUTPUT_PREPACKED:
        return "output_prepacked";
    case BENCH_LOGITS_MODE_OUTPUT_RAW:
        return "output_raw";
    default:
        return "mode";
    }
}

static marmot_error_t bench_logits_create_quantized_weight(
    marmot_context_t *ctx, size_t rows, size_t cols, marmot_quant_kind_t kind, uint32_t seed, marmot_tensor_t **out
) {
    if (ctx == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    bench_quantize_fn_t quantize_fn = bench_logits_quantize_fn(kind);
    if (quantize_fn == nullptr) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const size_t shape[] = {rows, cols};
    marmot_tensor_t *src = marmot_tensor_create(ctx, shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *dst = marmot_tensor_create_quantized(ctx, shape, 2, kind);
    if (src == nullptr || dst == nullptr) {
        marmot_tensor_destroy(dst);
        marmot_tensor_destroy(src);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    const size_t count = rows * cols;
    float *host = (float *)malloc(count * sizeof(float));
    if (host == nullptr) {
        marmot_tensor_destroy(dst);
        marmot_tensor_destroy(src);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    bench_logits_fill_f32(host, count, seed, 0.025f);
    marmot_error_t status = marmot_tensor_copy_from_host_buffer(ctx, src, host, count * sizeof(float));
    if (status == MARMOT_SUCCESS) {
        status = quantize_fn(ctx, src, dst);
    }
    free(host);
    marmot_tensor_destroy(src);
    if (status != MARMOT_SUCCESS) {
        marmot_tensor_destroy(dst);
        return status;
    }

    *out = dst;
    return MARMOT_SUCCESS;
}

static marmot_error_t bench_logits_setup(
    marmot_backend_type_t backend, marmot_context_t *ctx, marmot_graph_t **graph, marmot_tensor_t ***inputs,
    size_t *num_inputs, marmot_tensor_t ***outputs, size_t *num_outputs, void *user_data
) {
    (void)graph;

    bench_logits_params_t *params = (bench_logits_params_t *)user_data;
    const size_t input_shape[] = {params->batch, params->hidden};
    const size_t output_shape[] = {params->batch, params->vocab};
    const size_t input_count = (size_t)params->batch * (size_t)params->hidden;

    marmot_tensor_t *input = marmot_tensor_create(ctx, input_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *weight = nullptr;
    marmot_tensor_t *output = nullptr;
    if (input == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    marmot_error_t status = bench_logits_create_quantized_weight(
        ctx, params->vocab, params->hidden, params->quant_kind, 17u + params->batch, &weight
    );
    if (status != MARMOT_SUCCESS) {
        marmot_tensor_destroy(input);
        return status;
    }

    output = marmot_tensor_create(ctx, output_shape, 2, MARMOT_DTYPE_FLOAT32);
    if (output == nullptr) {
        marmot_tensor_destroy(weight);
        marmot_tensor_destroy(input);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    float *input_host = (float *)malloc(input_count * sizeof(float));
    if (input_host == nullptr) {
        marmot_tensor_destroy(output);
        marmot_tensor_destroy(weight);
        marmot_tensor_destroy(input);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    bench_logits_fill_f32(input_host, input_count, 101u + params->batch, 0.35f);
    status = marmot_tensor_copy_from_host_buffer(ctx, input, input_host, input_count * sizeof(float));
    free(input_host);
    if (status != MARMOT_SUCCESS) {
        marmot_tensor_destroy(output);
        marmot_tensor_destroy(weight);
        marmot_tensor_destroy(input);
        return status;
    }

    if (params->mode == BENCH_LOGITS_MODE_OUTPUT_PREPACKED && backend == MARMOT_BACKEND_CPU) {
        status = marmot_matmul_prepack_quant_weight(ctx, weight);
        if (status != MARMOT_SUCCESS) {
            marmot_tensor_destroy(output);
            marmot_tensor_destroy(weight);
            marmot_tensor_destroy(input);
            return status;
        }
    }

    *inputs = (marmot_tensor_t **)calloc(2, sizeof(marmot_tensor_t *));
    *outputs = (marmot_tensor_t **)calloc(1, sizeof(marmot_tensor_t *));
    if (*inputs == nullptr || *outputs == nullptr) {
        free(*outputs);
        free(*inputs);
        marmot_tensor_destroy(output);
        marmot_tensor_destroy(weight);
        marmot_tensor_destroy(input);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    (*inputs)[0] = input;
    (*inputs)[1] = weight;
    (*outputs)[0] = output;
    *num_inputs = 2;
    *num_outputs = 1;
    return MARMOT_SUCCESS;
}

static marmot_error_t bench_logits_execute(
    marmot_context_t *ctx, marmot_graph_t *graph, marmot_tensor_t **inputs, size_t num_inputs, marmot_tensor_t **outputs,
    size_t num_outputs
) {
    (void)graph;
    (void)num_inputs;
    (void)num_outputs;

    if (ctx != nullptr && ctx->backend_type == MARMOT_BACKEND_CPU) {
        return cpu_matmul_quantized_with_hints(
            ctx->device_ctx, inputs[0], inputs[1], nullptr, outputs[0], CPU_QUANT_MATMUL_HINT_OUTPUT_PROJECTION
        );
    }
    return marmot_linear(ctx, inputs[0], inputs[1], nullptr, outputs[0]);
}

static marmot_error_t bench_logits_execute_default(
    marmot_context_t *ctx, marmot_graph_t *graph, marmot_tensor_t **inputs, size_t num_inputs, marmot_tensor_t **outputs,
    size_t num_outputs
) {
    (void)graph;
    (void)num_inputs;
    (void)num_outputs;

    if (ctx != nullptr && ctx->backend_type == MARMOT_BACKEND_CPU) {
        return cpu_matmul_quantized_with_hints(ctx->device_ctx, inputs[0], inputs[1], nullptr, outputs[0], 0);
    }
    return marmot_linear(ctx, inputs[0], inputs[1], nullptr, outputs[0]);
}

static marmot_error_t bench_logits_execute_output_raw(
    marmot_context_t *ctx, marmot_graph_t *graph, marmot_tensor_t **inputs, size_t num_inputs, marmot_tensor_t **outputs,
    size_t num_outputs
) {
    (void)graph;
    (void)num_inputs;
    (void)num_outputs;

    if (ctx != nullptr && ctx->backend_type == MARMOT_BACKEND_CPU) {
        return cpu_matmul_quantized_with_hints(
            ctx->device_ctx, inputs[0], inputs[1], nullptr, outputs[0],
            CPU_QUANT_MATMUL_HINT_OUTPUT_PROJECTION | CPU_QUANT_MATMUL_HINT_PREFER_RAW
        );
    }
    return marmot_linear(ctx, inputs[0], inputs[1], nullptr, outputs[0]);
}

static void bench_logits_teardown(
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

static marmot_bench_workload_t *bench_create_logits_workload(
    uint32_t batch, uint32_t hidden, uint32_t vocab, marmot_quant_kind_t quant_kind, bench_logits_mode_t mode
) {
    marmot_bench_workload_t *workload = (marmot_bench_workload_t *)calloc(1, sizeof(marmot_bench_workload_t));
    bench_logits_params_t *params = (bench_logits_params_t *)calloc(1, sizeof(bench_logits_params_t));
    if (workload == nullptr || params == nullptr) {
        free(params);
        free(workload);
        return nullptr;
    }

    params->batch = batch;
    params->hidden = hidden;
    params->vocab = vocab;
    params->quant_kind = quant_kind;
    params->mode = mode;

    snprintf(
        params->name, sizeof(params->name), "logits_%s_%s_n%u_h%u_v%u", bench_logits_mode_str(mode),
        bench_logits_quant_kind_str(quant_kind), batch, hidden, vocab
    );

    const uint64_t flops = 2ULL * batch * hidden * vocab;
    const uint64_t weight_bits = quant_kind == MARMOT_QUANT_KIND_Q6_K ? 6ULL : 4ULL;
    const uint64_t weight_bytes = (uint64_t)hidden * vocab * weight_bits / 8ULL;
    const uint64_t input_bytes = (uint64_t)batch * hidden * sizeof(float);
    const uint64_t output_bytes = (uint64_t)batch * vocab * sizeof(float);

    workload->desc.name = params->name;
    workload->desc.category = MARMOT_BENCH_CATEGORY_MICRO;
    workload->desc.primary_dtype = MARMOT_DTYPE_FLOAT32;
    workload->desc.flops = flops;
    workload->desc.bytes_read = input_bytes + weight_bytes;
    workload->desc.bytes_written = output_bytes;
    memset(&workload->desc.signature, 0, sizeof(workload->desc.signature));
    workload->desc.signature.op_id = MARMOT_OP_LINEAR;
    workload->desc.signature.input_dtype = MARMOT_DTYPE_FLOAT32;
    workload->desc.signature.weight_dtype = MARMOT_DTYPE_UINT8;
    workload->desc.signature.output_dtype = MARMOT_DTYPE_FLOAT32;
    workload->desc.signature.accum_dtype = MARMOT_DTYPE_FLOAT32;
    workload->desc.signature.matmul_layout = MARMOT_MATMUL_LAYOUT_NT;
    workload->desc.signature.dims.matmul.N = batch;
    workload->desc.signature.dims.matmul.M = vocab;
    workload->desc.signature.dims.matmul.K = hidden;

    workload->user_data = params;
    workload->setup = bench_logits_setup;
    workload->execute = mode == BENCH_LOGITS_MODE_DEFAULT
        ? bench_logits_execute_default
        : (mode == BENCH_LOGITS_MODE_OUTPUT_RAW ? bench_logits_execute_output_raw : bench_logits_execute);
    workload->teardown = bench_logits_teardown;
    return workload;
}

void marmot_bench_register_logits_workloads(marmot_bench_suite_t *suite) {
    static const uint32_t batches[] = {1, 4};
    static const bench_logits_mode_t modes[] = {
        BENCH_LOGITS_MODE_DEFAULT,
        BENCH_LOGITS_MODE_OUTPUT,
        BENCH_LOGITS_MODE_OUTPUT_PREPACKED,
        BENCH_LOGITS_MODE_OUTPUT_RAW,
    };
    static const marmot_quant_kind_t quant_kinds[] = {
        MARMOT_QUANT_KIND_Q4_K,
        MARMOT_QUANT_KIND_Q6_K,
    };
    constexpr uint32_t hidden = 2048;
    constexpr uint32_t vocab = 151936;

    for (size_t i = 0; i < sizeof(batches) / sizeof(batches[0]); ++i) {
        for (size_t j = 0; j < sizeof(quant_kinds) / sizeof(quant_kinds[0]); ++j) {
            for (size_t k = 0; k < sizeof(modes) / sizeof(modes[0]); ++k) {
                marmot_bench_workload_t *workload =
                    bench_create_logits_workload(batches[i], hidden, vocab, quant_kinds[j], modes[k]);
                if (workload != nullptr) {
                    marmot_bench_suite_add(suite, workload);
                }
            }
        }
    }
}
