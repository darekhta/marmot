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
    BENCH_MOE_EXPERT_PROJ_GATE_UP = 0,
    BENCH_MOE_EXPERT_PROJ_DOWN = 1,
} bench_moe_expert_proj_t;

typedef enum {
    BENCH_MOE_EXPERT_MODE_DEFAULT = 0,
    BENCH_MOE_EXPERT_MODE_PREPACKED = 1,
    BENCH_MOE_EXPERT_MODE_PREFER_RAW = 2,
} bench_moe_expert_mode_t;

typedef struct {
    uint32_t batch;
    uint32_t hidden;
    uint32_t ff_length;
    marmot_quant_kind_t quant_kind;
    bench_moe_expert_proj_t proj;
    bench_moe_expert_mode_t mode;
    char name[96];
} bench_moe_expert_params_t;

static void bench_fill_f32(float *data, size_t count, uint32_t seed, float scale) {
    for (size_t i = 0; i < count; ++i) {
        seed = seed * 1103515245u + 12345u;
        const float value = (float)((seed >> 16) & 0x7fff) / 32767.0f;
        data[i] = (value * 2.0f - 1.0f) * scale;
    }
}

typedef marmot_error_t (*bench_quantize_fn_t)(const marmot_context_t *, const marmot_tensor_t *, marmot_tensor_t *);

static bench_quantize_fn_t bench_quantize_fn_for_kind(marmot_quant_kind_t kind) {
    switch (kind) {
    case MARMOT_QUANT_KIND_Q4_K:
        return marmot_quantize_q4_k;
    case MARMOT_QUANT_KIND_Q5_K:
        return marmot_quantize_q5_k;
    case MARMOT_QUANT_KIND_Q6_K:
        return marmot_quantize_q6_k;
    case MARMOT_QUANT_KIND_Q8_0:
        return marmot_quantize_q8_0;
    default:
        return nullptr;
    }
}

static const char *bench_quant_kind_str(marmot_quant_kind_t kind) {
    switch (kind) {
    case MARMOT_QUANT_KIND_Q4_K:
        return "q4_k";
    case MARMOT_QUANT_KIND_Q5_K:
        return "q5_k";
    case MARMOT_QUANT_KIND_Q6_K:
        return "q6_k";
    case MARMOT_QUANT_KIND_Q8_0:
        return "q8_0";
    default:
        return "quant";
    }
}

static const char *bench_proj_str(bench_moe_expert_proj_t proj) {
    switch (proj) {
    case BENCH_MOE_EXPERT_PROJ_GATE_UP:
        return "gateup";
    case BENCH_MOE_EXPERT_PROJ_DOWN:
        return "down";
    default:
        return "proj";
    }
}

static const char *bench_mode_str(bench_moe_expert_mode_t mode) {
    switch (mode) {
    case BENCH_MOE_EXPERT_MODE_DEFAULT:
        return "default";
    case BENCH_MOE_EXPERT_MODE_PREPACKED:
        return "prepacked";
    case BENCH_MOE_EXPERT_MODE_PREFER_RAW:
        return "raw";
    default:
        return "mode";
    }
}

static marmot_error_t bench_create_quantized_weight(
    marmot_context_t *ctx, size_t rows, size_t cols, marmot_quant_kind_t kind, uint32_t seed, marmot_tensor_t **out
) {
    if (ctx == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    bench_quantize_fn_t quantize_fn = bench_quantize_fn_for_kind(kind);
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

    bench_fill_f32(host, count, seed, 0.025f);
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

static marmot_error_t bench_moe_expert_setup(
    marmot_backend_type_t backend, marmot_context_t *ctx, marmot_graph_t **graph, marmot_tensor_t ***inputs,
    size_t *num_inputs, marmot_tensor_t ***outputs, size_t *num_outputs, void *user_data
) {
    (void)graph;

    bench_moe_expert_params_t *params = (bench_moe_expert_params_t *)user_data;
    const size_t input_cols =
        params->proj == BENCH_MOE_EXPERT_PROJ_GATE_UP ? (size_t)params->hidden : (size_t)params->ff_length;
    const size_t output_cols =
        params->proj == BENCH_MOE_EXPERT_PROJ_GATE_UP ? (size_t)params->ff_length : (size_t)params->hidden;
    const size_t input_shape[] = {params->batch, input_cols};
    const size_t output_shape[] = {params->batch, output_cols};
    const size_t input_count = (size_t)params->batch * input_cols;

    marmot_tensor_t *input = marmot_tensor_create(ctx, input_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *weight_a = nullptr;
    marmot_tensor_t *weight_b = nullptr;
    marmot_tensor_t *output_a = nullptr;
    marmot_tensor_t *output_b = nullptr;
    if (input == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    marmot_error_t status = bench_create_quantized_weight(
        ctx, output_cols, input_cols, params->quant_kind, 17u + params->batch, &weight_a
    );
    if (status == MARMOT_SUCCESS && params->proj == BENCH_MOE_EXPERT_PROJ_GATE_UP) {
        status = bench_create_quantized_weight(
            ctx, output_cols, input_cols, params->quant_kind, 53u + params->batch, &weight_b
        );
    }
    if (status != MARMOT_SUCCESS) {
        marmot_tensor_destroy(weight_b);
        marmot_tensor_destroy(weight_a);
        marmot_tensor_destroy(input);
        return status;
    }

    output_a = marmot_tensor_create(ctx, output_shape, 2, MARMOT_DTYPE_FLOAT32);
    if (output_a == nullptr) {
        marmot_tensor_destroy(weight_b);
        marmot_tensor_destroy(weight_a);
        marmot_tensor_destroy(input);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    if (params->proj == BENCH_MOE_EXPERT_PROJ_GATE_UP) {
        output_b = marmot_tensor_create(ctx, output_shape, 2, MARMOT_DTYPE_FLOAT32);
        if (output_b == nullptr) {
            marmot_tensor_destroy(output_a);
            marmot_tensor_destroy(weight_b);
            marmot_tensor_destroy(weight_a);
            marmot_tensor_destroy(input);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }

    float *input_host = (float *)malloc(input_count * sizeof(float));
    if (input_host == nullptr) {
        marmot_tensor_destroy(output_b);
        marmot_tensor_destroy(output_a);
        marmot_tensor_destroy(weight_b);
        marmot_tensor_destroy(weight_a);
        marmot_tensor_destroy(input);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    bench_fill_f32(input_host, input_count, 101u + params->batch, 0.35f);
    status = marmot_tensor_copy_from_host_buffer(ctx, input, input_host, input_count * sizeof(float));
    free(input_host);
    if (status != MARMOT_SUCCESS) {
        marmot_tensor_destroy(output_b);
        marmot_tensor_destroy(output_a);
        marmot_tensor_destroy(weight_b);
        marmot_tensor_destroy(weight_a);
        marmot_tensor_destroy(input);
        return status;
    }

    if (params->mode == BENCH_MOE_EXPERT_MODE_PREPACKED && backend == MARMOT_BACKEND_CPU) {
        status = marmot_matmul_prepack_quant_weight(ctx, weight_a);
        if (status == MARMOT_SUCCESS && weight_b != nullptr) {
            status = marmot_matmul_prepack_quant_weight(ctx, weight_b);
        }
        if (status != MARMOT_SUCCESS) {
            marmot_tensor_destroy(output_b);
            marmot_tensor_destroy(output_a);
            marmot_tensor_destroy(weight_b);
            marmot_tensor_destroy(weight_a);
            marmot_tensor_destroy(input);
            return status;
        }
    }

    *inputs = (marmot_tensor_t **)calloc(weight_b != nullptr ? 3 : 2, sizeof(marmot_tensor_t *));
    *outputs = (marmot_tensor_t **)calloc(output_b != nullptr ? 2 : 1, sizeof(marmot_tensor_t *));
    if (*inputs == nullptr || *outputs == nullptr) {
        free(*outputs);
        free(*inputs);
        marmot_tensor_destroy(output_b);
        marmot_tensor_destroy(output_a);
        marmot_tensor_destroy(weight_b);
        marmot_tensor_destroy(weight_a);
        marmot_tensor_destroy(input);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    (*inputs)[0] = input;
    (*inputs)[1] = weight_a;
    *num_inputs = 2;
    if (weight_b != nullptr) {
        (*inputs)[2] = weight_b;
        *num_inputs = 3;
    }

    (*outputs)[0] = output_a;
    *num_outputs = 1;
    if (output_b != nullptr) {
        (*outputs)[1] = output_b;
        *num_outputs = 2;
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t bench_moe_expert_execute(
    marmot_context_t *ctx, marmot_graph_t *graph, marmot_tensor_t **inputs, size_t num_inputs, marmot_tensor_t **outputs,
    size_t num_outputs
) {
    (void)graph;
    (void)num_inputs;
    (void)num_outputs;

    const bool is_cpu = ctx != nullptr && ctx->backend_type == MARMOT_BACKEND_CPU;
    if (is_cpu && num_inputs == 3 && num_outputs == 2) {
        return cpu_matmul_quantized_dual_output(ctx->device_ctx, inputs[0], inputs[1], inputs[2], outputs[0], outputs[1]);
    }
    if (is_cpu && num_inputs == 2 && num_outputs == 1) {
        return cpu_matmul_quantized_with_hints(
            ctx->device_ctx, inputs[0], inputs[1], nullptr, outputs[0], CPU_QUANT_MATMUL_HINT_PREFER_RAW
        );
    }
    if (num_inputs == 3 && num_outputs == 2) {
        marmot_error_t status = marmot_linear(ctx, inputs[0], inputs[1], nullptr, outputs[0]);
        if (status == MARMOT_SUCCESS) {
            status = marmot_linear(ctx, inputs[0], inputs[2], nullptr, outputs[1]);
        }
        return status;
    }
    return marmot_linear(ctx, inputs[0], inputs[1], nullptr, outputs[0]);
}

static marmot_error_t bench_moe_expert_execute_default(
    marmot_context_t *ctx, marmot_graph_t *graph, marmot_tensor_t **inputs, size_t num_inputs, marmot_tensor_t **outputs,
    size_t num_outputs
) {
    (void)graph;
    (void)num_inputs;
    (void)num_outputs;

    const bool is_cpu = ctx != nullptr && ctx->backend_type == MARMOT_BACKEND_CPU;
    if (is_cpu && num_inputs == 3 && num_outputs == 2) {
        return cpu_matmul_quantized_dual_output(ctx->device_ctx, inputs[0], inputs[1], inputs[2], outputs[0], outputs[1]);
    }
    if (is_cpu && num_inputs == 2 && num_outputs == 1) {
        return cpu_matmul_quantized_with_hints(ctx->device_ctx, inputs[0], inputs[1], nullptr, outputs[0], 0);
    }
    if (num_inputs == 3 && num_outputs == 2) {
        marmot_error_t status = marmot_linear(ctx, inputs[0], inputs[1], nullptr, outputs[0]);
        if (status == MARMOT_SUCCESS) {
            status = marmot_linear(ctx, inputs[0], inputs[2], nullptr, outputs[1]);
        }
        return status;
    }
    return marmot_linear(ctx, inputs[0], inputs[1], nullptr, outputs[0]);
}

static void bench_moe_expert_teardown(
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

static marmot_bench_workload_t *bench_create_moe_expert_workload(
    uint32_t batch, uint32_t hidden, uint32_t ff_length, marmot_quant_kind_t quant_kind, bench_moe_expert_proj_t proj,
    bench_moe_expert_mode_t mode
) {
    marmot_bench_workload_t *workload = (marmot_bench_workload_t *)calloc(1, sizeof(marmot_bench_workload_t));
    bench_moe_expert_params_t *params = (bench_moe_expert_params_t *)calloc(1, sizeof(bench_moe_expert_params_t));
    if (workload == nullptr || params == nullptr) {
        free(params);
        free(workload);
        return nullptr;
    }

    params->batch = batch;
    params->hidden = hidden;
    params->ff_length = ff_length;
    params->quant_kind = quant_kind;
    params->proj = proj;
    params->mode = mode;

    snprintf(
        params->name, sizeof(params->name), "moe_%s_%s_%s_n%u_h%u_f%u", bench_proj_str(proj), bench_mode_str(mode),
        bench_quant_kind_str(quant_kind), batch, hidden, ff_length
    );

    const uint64_t linear_flops = 2ULL * batch * hidden * ff_length;
    const uint64_t weight_bits = quant_kind == MARMOT_QUANT_KIND_Q8_0 ? 8ULL : 4ULL;
    const uint64_t weight_bytes = (uint64_t)hidden * ff_length * weight_bits / 8ULL;
    const uint64_t input_bytes =
        (uint64_t)batch * (proj == BENCH_MOE_EXPERT_PROJ_GATE_UP ? hidden : ff_length) * sizeof(float);
    const uint64_t output_bytes =
        (uint64_t)batch * (proj == BENCH_MOE_EXPERT_PROJ_GATE_UP ? ff_length : hidden) * sizeof(float);

    workload->desc.name = params->name;
    workload->desc.category = MARMOT_BENCH_CATEGORY_MICRO;
    workload->desc.primary_dtype = MARMOT_DTYPE_FLOAT32;
    workload->desc.flops = proj == BENCH_MOE_EXPERT_PROJ_GATE_UP ? linear_flops * 2ULL : linear_flops;
    workload->desc.bytes_read = input_bytes + weight_bytes * (proj == BENCH_MOE_EXPERT_PROJ_GATE_UP ? 2ULL : 1ULL);
    workload->desc.bytes_written = output_bytes * (proj == BENCH_MOE_EXPERT_PROJ_GATE_UP ? 2ULL : 1ULL);
    memset(&workload->desc.signature, 0, sizeof(workload->desc.signature));
    workload->desc.signature.op_id = MARMOT_OP_LINEAR;
    workload->desc.signature.input_dtype = MARMOT_DTYPE_FLOAT32;
    workload->desc.signature.weight_dtype = MARMOT_DTYPE_UINT8;
    workload->desc.signature.output_dtype = MARMOT_DTYPE_FLOAT32;
    workload->desc.signature.accum_dtype = MARMOT_DTYPE_FLOAT32;
    workload->desc.signature.matmul_layout = MARMOT_MATMUL_LAYOUT_NT;
    workload->desc.signature.dims.matmul.N = batch;
    workload->desc.signature.dims.matmul.M = proj == BENCH_MOE_EXPERT_PROJ_GATE_UP ? ff_length : hidden;
    workload->desc.signature.dims.matmul.K = proj == BENCH_MOE_EXPERT_PROJ_GATE_UP ? hidden : ff_length;

    workload->user_data = params;
    workload->setup = bench_moe_expert_setup;
    workload->execute = mode == BENCH_MOE_EXPERT_MODE_PREFER_RAW ? bench_moe_expert_execute : bench_moe_expert_execute_default;
    workload->teardown = bench_moe_expert_teardown;
    return workload;
}

void marmot_bench_register_moe_expert_workloads(marmot_bench_suite_t *suite) {
    static const uint32_t batches[] = {1, 2, 4, 6, 8, 12, 16, 24, 32, 47};
    static const bench_moe_expert_mode_t gate_up_modes[] = {
        BENCH_MOE_EXPERT_MODE_DEFAULT,
        BENCH_MOE_EXPERT_MODE_PREPACKED,
    };
    static const bench_moe_expert_mode_t down_modes[] = {
        BENCH_MOE_EXPERT_MODE_DEFAULT,
        BENCH_MOE_EXPERT_MODE_PREPACKED,
        BENCH_MOE_EXPERT_MODE_PREFER_RAW,
    };
    constexpr uint32_t hidden = 2048;
    constexpr uint32_t ff_length = 768;

    for (size_t i = 0; i < sizeof(batches) / sizeof(batches[0]); ++i) {
        for (size_t j = 0; j < sizeof(gate_up_modes) / sizeof(gate_up_modes[0]); ++j) {
            marmot_bench_workload_t *workload = bench_create_moe_expert_workload(
                batches[i], hidden, ff_length, MARMOT_QUANT_KIND_Q4_K, BENCH_MOE_EXPERT_PROJ_GATE_UP, gate_up_modes[j]
            );
            if (workload != nullptr) {
                marmot_bench_suite_add(suite, workload);
            }
        }
        for (size_t j = 0; j < sizeof(down_modes) / sizeof(down_modes[0]); ++j) {
            marmot_bench_workload_t *workload = bench_create_moe_expert_workload(
                batches[i], hidden, ff_length, MARMOT_QUANT_KIND_Q4_K, BENCH_MOE_EXPERT_PROJ_DOWN, down_modes[j]
            );
            if (workload != nullptr) {
                marmot_bench_suite_add(suite, workload);
            }
        }
    }
}
