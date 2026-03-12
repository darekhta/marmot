#include "../bench_workloads.h"

#include "marmot/ops/neural.h"
#include "marmot/ops/quantization.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    BENCH_MOE_ROUTE_UNIFORM = 0,
    BENCH_MOE_ROUTE_HOT = 1,
} bench_moe_route_pattern_t;

typedef struct {
    uint32_t tokens;
    uint32_t hidden;
    uint32_t ff_length;
    uint32_t experts;
    uint32_t topk;
    marmot_quant_kind_t quant_kind;
    marmot_ffn_type_t ffn_type;
    bench_moe_route_pattern_t route_pattern;
    float weights_scale;
    char name[96];
} moe_params_t;

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

static const char *bench_route_pattern_str(bench_moe_route_pattern_t pattern) {
    switch (pattern) {
    case BENCH_MOE_ROUTE_UNIFORM:
        return "uniform";
    case BENCH_MOE_ROUTE_HOT:
        return "hot";
    default:
        return "route";
    }
}

static marmot_qscheme_id_t bench_qscheme_id_for_kind(marmot_quant_kind_t kind) {
    switch (kind) {
    case MARMOT_QUANT_KIND_Q4_K:
        return MARMOT_QSCHEME_Q4_K;
    case MARMOT_QUANT_KIND_Q5_K:
        return MARMOT_QSCHEME_Q5_K;
    case MARMOT_QUANT_KIND_Q6_K:
        return MARMOT_QSCHEME_Q6_K;
    case MARMOT_QUANT_KIND_Q8_0:
        return MARMOT_QSCHEME_Q8_0;
    default:
        return MARMOT_QSCHEME_NONE;
    }
}

static float bench_moe_hidden_value(size_t token, size_t col) {
    uint32_t seed = (uint32_t)(token * 131u + col * 17u + 23u);
    seed ^= seed >> 13;
    seed *= 1103515245u;
    return ((float)(seed & 0x3ffu) / 512.0f - 1.0f) * 0.35f;
}

static float bench_moe_weight_value(size_t bank, size_t expert, size_t row, size_t col) {
    const float center = (float)((int)((row * 11 + col * 7 + expert * 5 + bank * 13) % 29) - 14);
    const float modulation = 1.0f + 0.03f * (float)(expert % 9);
    const float bank_scale = bank == 2 ? 0.010f : 0.014f;
    return center * bank_scale * modulation;
}

static marmot_error_t bench_create_quantized_expert_tensor(
    const marmot_context_t *dst_ctx, marmot_quant_kind_t kind, size_t rows, size_t cols, size_t experts, size_t bank,
    marmot_tensor_t **out_tensor
) {
    if (out_tensor == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(kind);
    if (traits == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t logical_shape[] = {rows, cols};
    marmot_tensor_t *probe = marmot_tensor_create_quantized(nullptr, logical_shape, 2, kind);
    if (probe == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    const size_t slice_bytes = marmot_tensor_size_bytes(probe);
    marmot_tensor_destroy(probe);
    const size_t total_bytes = slice_bytes * experts;

    size_t storage_shape[] = {total_bytes};
    marmot_tensor_t *tensor = marmot_tensor_create(dst_ctx, storage_shape, 1, traits->storage_dtype);
    if (tensor == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    tensor->shape.ndim = 3;
    tensor->shape.shape[0] = cols;
    tensor->shape.shape[1] = rows;
    tensor->shape.shape[2] = experts;
    tensor->shape.strides[2] = 1;
    tensor->shape.strides[1] = experts;
    tensor->shape.strides[0] = rows * experts;
    tensor->quant_kind = kind;
    tensor->quant_layout = MARMOT_QUANT_LAYOUT_GGUF;

    marmot_context_t *quant_ctx = marmot_init(MARMOT_BACKEND_CPU);
    if (quant_ctx == nullptr) {
        marmot_tensor_destroy(tensor);
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
    }

    float *expert_src = (float *)malloc(rows * cols * sizeof(float));
    uint8_t *packed = (uint8_t *)malloc(total_bytes);
    if (expert_src == nullptr || packed == nullptr) {
        free(packed);
        free(expert_src);
        marmot_destroy(quant_ctx);
        marmot_tensor_destroy(tensor);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    for (size_t expert = 0; expert < experts; ++expert) {
        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < cols; ++col) {
                expert_src[row * cols + col] = bench_moe_weight_value(bank, expert, row, col);
            }
        }
        marmot_tensor_t *src = marmot_tensor_create(quant_ctx, logical_shape, 2, MARMOT_DTYPE_FLOAT32);
        marmot_tensor_t *dst = marmot_tensor_create_quantized(quant_ctx, logical_shape, 2, kind);
        if (src == nullptr || dst == nullptr) {
            marmot_tensor_destroy(dst);
            marmot_tensor_destroy(src);
            free(packed);
            free(expert_src);
            marmot_destroy(quant_ctx);
            marmot_tensor_destroy(tensor);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }

        marmot_error_t status = marmot_tensor_copy_from_host_buffer(quant_ctx, src, expert_src, rows * cols * sizeof(float));
        if (status == MARMOT_SUCCESS) {
            switch (kind) {
            case MARMOT_QUANT_KIND_Q4_K:
                status = marmot_quantize_q4_k(quant_ctx, src, dst);
                break;
            case MARMOT_QUANT_KIND_Q5_K:
                status = marmot_quantize_q5_k(quant_ctx, src, dst);
                break;
            case MARMOT_QUANT_KIND_Q6_K:
                status = marmot_quantize_q6_k(quant_ctx, src, dst);
                break;
            case MARMOT_QUANT_KIND_Q8_0:
                status = marmot_quantize_q8_0(quant_ctx, src, dst);
                break;
            default:
                status = MARMOT_ERROR_NOT_IMPLEMENTED;
                break;
            }
        }
        if (status == MARMOT_SUCCESS) {
            status = marmot_tensor_copy_to_host_buffer(quant_ctx, dst, packed + expert * slice_bytes, slice_bytes);
        }
        marmot_tensor_destroy(dst);
        marmot_tensor_destroy(src);
        if (status != MARMOT_SUCCESS) {
            free(packed);
            free(expert_src);
            marmot_destroy(quant_ctx);
            marmot_tensor_destroy(tensor);
            return status;
        }
    }

    marmot_error_t copy_status = marmot_tensor_copy_from_host_buffer(dst_ctx, tensor, packed, total_bytes);
    free(packed);
    free(expert_src);
    marmot_destroy(quant_ctx);
    if (copy_status != MARMOT_SUCCESS) {
        marmot_tensor_destroy(tensor);
        return copy_status;
    }

    *out_tensor = tensor;
    return MARMOT_SUCCESS;
}

static void bench_fill_routes(
    moe_params_t *params, marmot_int32_t *topk_ids, float *topk_weights
) {
    float base_weights[16];
    float weight_sum = 0.0f;
    for (size_t slot = 0; slot < params->topk; ++slot) {
        base_weights[slot] = 1.0f / (float)(slot + 1);
        weight_sum += base_weights[slot];
    }
    for (size_t slot = 0; slot < params->topk; ++slot) {
        base_weights[slot] /= weight_sum;
    }

    const size_t hot_experts = params->experts < 8 ? params->experts : 8;
    for (size_t token = 0; token < params->tokens; ++token) {
        size_t base = 0;
        if (params->route_pattern == BENCH_MOE_ROUTE_UNIFORM) {
            base = (token * 11u) % params->experts;
            for (size_t slot = 0; slot < params->topk; ++slot) {
                topk_ids[token * params->topk + slot].value = (int32_t)((base + slot * 7u) % params->experts);
                topk_weights[token * params->topk + slot] = base_weights[slot];
            }
            continue;
        }

        base = (token * 3u) % hot_experts;
        for (size_t slot = 0; slot < params->topk; ++slot) {
            topk_ids[token * params->topk + slot].value = (int32_t)((base + slot) % hot_experts);
            topk_weights[token * params->topk + slot] = base_weights[slot];
        }
    }
}

static marmot_error_t moe_setup(
    marmot_backend_type_t backend, marmot_context_t *ctx, marmot_graph_t **graph, marmot_tensor_t ***inputs,
    size_t *num_inputs, marmot_tensor_t ***outputs, size_t *num_outputs, void *user_data
) {
    (void)backend;
    (void)graph;

    moe_params_t *params = (moe_params_t *)user_data;
    size_t hidden_shape[] = {params->tokens, params->hidden};
    size_t route_shape[] = {params->tokens, params->topk};

    marmot_tensor_t *hidden = marmot_tensor_create(ctx, hidden_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *topk_ids = marmot_tensor_create(ctx, route_shape, 2, MARMOT_DTYPE_INT32);
    marmot_tensor_t *topk_weights = marmot_tensor_create(ctx, route_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(ctx, hidden_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *gate_exps = nullptr;
    marmot_tensor_t *up_exps = nullptr;
    marmot_tensor_t *down_exps = nullptr;
    if (hidden == nullptr || topk_ids == nullptr || topk_weights == nullptr || out == nullptr) {
        marmot_tensor_destroy(out);
        marmot_tensor_destroy(topk_weights);
        marmot_tensor_destroy(topk_ids);
        marmot_tensor_destroy(hidden);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    marmot_error_t status = bench_create_quantized_expert_tensor(
        ctx, params->quant_kind, params->ff_length, params->hidden, params->experts, 0, &gate_exps
    );
    if (status == MARMOT_SUCCESS) {
        status = bench_create_quantized_expert_tensor(
            ctx, params->quant_kind, params->ff_length, params->hidden, params->experts, 1, &up_exps
        );
    }
    if (status == MARMOT_SUCCESS) {
        status = bench_create_quantized_expert_tensor(
            ctx, params->quant_kind, params->hidden, params->ff_length, params->experts, 2, &down_exps
        );
    }
    if (status != MARMOT_SUCCESS) {
        marmot_tensor_destroy(gate_exps);
        marmot_tensor_destroy(up_exps);
        marmot_tensor_destroy(down_exps);
        marmot_tensor_destroy(out);
        marmot_tensor_destroy(topk_weights);
        marmot_tensor_destroy(topk_ids);
        marmot_tensor_destroy(hidden);
        return status;
    }

    float *hidden_host = (float *)malloc(params->tokens * params->hidden * sizeof(float));
    marmot_int32_t *topk_ids_host = (marmot_int32_t *)malloc(params->tokens * params->topk * sizeof(marmot_int32_t));
    float *topk_weights_host = (float *)malloc(params->tokens * params->topk * sizeof(float));
    if (hidden_host == nullptr || topk_ids_host == nullptr || topk_weights_host == nullptr) {
        free(topk_weights_host);
        free(topk_ids_host);
        free(hidden_host);
        marmot_tensor_destroy(gate_exps);
        marmot_tensor_destroy(up_exps);
        marmot_tensor_destroy(down_exps);
        marmot_tensor_destroy(out);
        marmot_tensor_destroy(topk_weights);
        marmot_tensor_destroy(topk_ids);
        marmot_tensor_destroy(hidden);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    for (size_t token = 0; token < params->tokens; ++token) {
        for (size_t col = 0; col < params->hidden; ++col) {
            hidden_host[token * params->hidden + col] = bench_moe_hidden_value(token, col);
        }
    }
    bench_fill_routes(params, topk_ids_host, topk_weights_host);

    status = marmot_tensor_copy_from_host_buffer(ctx, hidden, hidden_host, params->tokens * params->hidden * sizeof(float));
    if (status == MARMOT_SUCCESS) {
        status =
            marmot_tensor_copy_from_host_buffer(ctx, topk_ids, topk_ids_host, params->tokens * params->topk * sizeof(marmot_int32_t));
    }
    if (status == MARMOT_SUCCESS) {
        status = marmot_tensor_copy_from_host_buffer(
            ctx, topk_weights, topk_weights_host, params->tokens * params->topk * sizeof(float)
        );
    }

    free(topk_weights_host);
    free(topk_ids_host);
    free(hidden_host);
    if (status != MARMOT_SUCCESS) {
        marmot_tensor_destroy(gate_exps);
        marmot_tensor_destroy(up_exps);
        marmot_tensor_destroy(down_exps);
        marmot_tensor_destroy(out);
        marmot_tensor_destroy(topk_weights);
        marmot_tensor_destroy(topk_ids);
        marmot_tensor_destroy(hidden);
        return status;
    }

    *inputs = (marmot_tensor_t **)malloc(6 * sizeof(marmot_tensor_t *));
    *outputs = (marmot_tensor_t **)malloc(sizeof(marmot_tensor_t *));
    if (*inputs == nullptr || *outputs == nullptr) {
        free(*outputs);
        free(*inputs);
        marmot_tensor_destroy(gate_exps);
        marmot_tensor_destroy(up_exps);
        marmot_tensor_destroy(down_exps);
        marmot_tensor_destroy(out);
        marmot_tensor_destroy(topk_weights);
        marmot_tensor_destroy(topk_ids);
        marmot_tensor_destroy(hidden);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    (*inputs)[0] = hidden;
    (*inputs)[1] = gate_exps;
    (*inputs)[2] = up_exps;
    (*inputs)[3] = down_exps;
    (*inputs)[4] = topk_ids;
    (*inputs)[5] = topk_weights;
    (*outputs)[0] = out;
    *num_inputs = 6;
    *num_outputs = 1;
    return MARMOT_SUCCESS;
}

static marmot_error_t moe_execute(
    marmot_context_t *ctx, marmot_graph_t *graph, marmot_tensor_t **inputs, size_t num_inputs, marmot_tensor_t **outputs,
    size_t num_outputs
) {
    (void)graph;
    (void)num_inputs;
    (void)num_outputs;
    return marmot_moe_experts(
        ctx,
        &(marmot_moe_experts_desc_t){
            .hidden_states = inputs[0],
            .gate_exps = inputs[1],
            .up_exps = inputs[2],
            .down_exps = inputs[3],
            .topk_ids = inputs[4],
            .topk_weights = inputs[5],
            .out = outputs[0],
            .ffn_type = MARMOT_FFN_SWIGLU,
            .weights_scale = 1.0f,
            .router_weight_policy = MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED,
        }
    );
}

static void moe_teardown(
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

static marmot_bench_workload_t *create_moe_workload(
    uint32_t tokens, uint32_t hidden, uint32_t ff_length, uint32_t experts, uint32_t topk, marmot_quant_kind_t quant_kind,
    bench_moe_route_pattern_t route_pattern
) {
    marmot_bench_workload_t *w = calloc(1, sizeof(*w));
    moe_params_t *params = calloc(1, sizeof(*params));
    if (w == nullptr || params == nullptr) {
        free(params);
        free(w);
        return nullptr;
    }

    *params = (moe_params_t){
        .tokens = tokens,
        .hidden = hidden,
        .ff_length = ff_length,
        .experts = experts,
        .topk = topk,
        .quant_kind = quant_kind,
        .ffn_type = MARMOT_FFN_SWIGLU,
        .route_pattern = route_pattern,
        .weights_scale = 1.0f,
    };
    snprintf(
        params->name, sizeof(params->name), "moe_%s_%s_t%u_h%u_f%u_e%u_k%u", bench_route_pattern_str(route_pattern),
        bench_quant_kind_str(quant_kind), tokens, hidden, ff_length, experts, topk
    );

    const uint64_t routes = (uint64_t)tokens * topk;
    const uint64_t flops = routes * (6ULL * hidden * ff_length + 8ULL * ff_length);
    const uint64_t quant_bits = quant_kind == MARMOT_QUANT_KIND_Q8_0 ? 8ULL : 5ULL;
    const uint64_t hidden_bytes = (uint64_t)tokens * hidden * sizeof(float);
    const uint64_t output_bytes = hidden_bytes;
    const uint64_t route_bytes = (uint64_t)tokens * topk * (sizeof(float) + sizeof(marmot_int32_t));
    const uint64_t weight_bytes =
        ((uint64_t)experts * hidden * ff_length * quant_bits / 8ULL) * 3ULL;

    w->desc.name = params->name;
    w->desc.category = MARMOT_BENCH_CATEGORY_COMPOSITE;
    w->desc.primary_dtype = MARMOT_DTYPE_FLOAT32;
    w->desc.flops = flops;
    w->desc.bytes_read = hidden_bytes + route_bytes + weight_bytes;
    w->desc.bytes_written = output_bytes;
    memset(&w->desc.signature, 0, sizeof(w->desc.signature));
    w->desc.signature.op_id = MARMOT_OP_MOE_EXPERTS;
    w->desc.signature.input_dtype = MARMOT_DTYPE_FLOAT32;
    w->desc.signature.output_dtype = MARMOT_DTYPE_FLOAT32;
    w->desc.signature.accum_dtype = MARMOT_DTYPE_FLOAT32;
    w->desc.signature.qscheme_id = bench_qscheme_id_for_kind(quant_kind);

    w->user_data = params;
    w->setup = moe_setup;
    w->execute = moe_execute;
    w->teardown = moe_teardown;
    return w;
}

void marmot_bench_register_moe_workloads(marmot_bench_suite_t *suite) {
    const struct {
        uint32_t tokens;
        bench_moe_route_pattern_t pattern;
    } cases[] = {
        {64, BENCH_MOE_ROUTE_UNIFORM},
        {64, BENCH_MOE_ROUTE_HOT},
        {1, BENCH_MOE_ROUTE_UNIFORM},
        {1, BENCH_MOE_ROUTE_HOT},
        {2, BENCH_MOE_ROUTE_UNIFORM},
        {2, BENCH_MOE_ROUTE_HOT},
        {4, BENCH_MOE_ROUTE_UNIFORM},
        {4, BENCH_MOE_ROUTE_HOT},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        marmot_bench_workload_t *w =
            create_moe_workload(cases[i].tokens, 512, 1536, 32, 8, MARMOT_QUANT_KIND_Q4_K, cases[i].pattern);
        if (w != nullptr) {
            marmot_bench_suite_add(suite, w);
        }
    }
}
