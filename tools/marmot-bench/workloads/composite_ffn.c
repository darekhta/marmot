#include "../bench_workloads.h"

#include "marmot/graph/graph.h"
#include "marmot/graph/graph_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t batch;
    uint32_t seq_len;
    uint32_t hidden;
    uint32_t intermediate;
    marmot_dtype_t dtype;
    char name[64];
} ffn_params_t;

static void init_tensor_desc_2d(marmot_graph_tensor_desc_t *desc, size_t dim0, size_t dim1, marmot_dtype_t dtype) {
    memset(desc, 0, sizeof(*desc));
    desc->dtype = dtype;
    desc->ndim = 2;
    desc->shape[0] = dim0;
    desc->shape[1] = dim1;
    desc->strides[1] = 1;
    desc->strides[0] = dim1;
}

static marmot_error_t ffn_setup(
    marmot_backend_type_t backend,
    marmot_context_t *ctx,
    marmot_graph_t **graph,
    marmot_tensor_t ***inputs,
    size_t *num_inputs,
    marmot_tensor_t ***outputs,
    size_t *num_outputs,
    void *user_data
) {
    ffn_params_t *params = (ffn_params_t *)user_data;
    (void)ctx;

    *graph = marmot_graph_create();
    if (*graph == nullptr)
        return MARMOT_ERROR_OUT_OF_MEMORY;

    size_t tokens = params->batch * params->seq_len;

    marmot_graph_tensor_desc_t x_desc, up_w_desc, down_w_desc, inter_desc, out_desc;
    init_tensor_desc_2d(&x_desc, tokens, params->hidden, params->dtype);
    init_tensor_desc_2d(&up_w_desc, params->hidden, params->intermediate, params->dtype);
    init_tensor_desc_2d(&down_w_desc, params->intermediate, params->hidden, params->dtype);
    init_tensor_desc_2d(&inter_desc, tokens, params->intermediate, params->dtype);
    init_tensor_desc_2d(&out_desc, tokens, params->hidden, params->dtype);

    marmot_value_id_t x_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t up_w_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t down_w_id = MARMOT_VALUE_ID_INVALID;

    marmot_error_t err = marmot_graph_add_input(*graph, &x_desc, &x_id);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }
    err = marmot_graph_add_input(*graph, &up_w_desc, &up_w_id);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }
    err = marmot_graph_add_input(*graph, &down_w_desc, &down_w_id);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }

    marmot_value_id_t up_out_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t up_inputs[2] = {x_id, up_w_id};
    err = marmot_graph_add_op(*graph, "matmul", nullptr, up_inputs, 2, &inter_desc, 1, &up_out_id);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }

    marmot_value_id_t gelu_out_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t gelu_inputs[1] = {up_out_id};
    err = marmot_graph_add_op(*graph, "gelu", nullptr, gelu_inputs, 1, &inter_desc, 1, &gelu_out_id);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }

    marmot_value_id_t down_out_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t down_inputs[2] = {gelu_out_id, down_w_id};
    err = marmot_graph_add_op(*graph, "matmul", nullptr, down_inputs, 2, &out_desc, 1, &down_out_id);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }

    err = marmot_graph_finalize(*graph, backend);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }

    size_t x_shape[2] = {tokens, params->hidden};
    size_t up_w_shape[2] = {params->hidden, params->intermediate};
    size_t down_w_shape[2] = {params->intermediate, params->hidden};
    size_t out_shape[2] = {tokens, params->hidden};

    *inputs = malloc(3 * sizeof(marmot_tensor_t *));
    (*inputs)[0] = marmot_tensor_create(ctx, x_shape, 2, params->dtype);
    (*inputs)[1] = marmot_tensor_create(ctx, up_w_shape, 2, params->dtype);
    (*inputs)[2] = marmot_tensor_create(ctx, down_w_shape, 2, params->dtype);
    *num_inputs = 3;

    *outputs = malloc(1 * sizeof(marmot_tensor_t *));
    (*outputs)[0] = marmot_tensor_create(ctx, out_shape, 2, params->dtype);
    *num_outputs = 1;

    return MARMOT_SUCCESS;
}

static marmot_error_t ffn_execute(
    marmot_context_t *ctx,
    marmot_graph_t *graph,
    marmot_tensor_t **inputs,
    size_t num_inputs,
    marmot_tensor_t **outputs,
    size_t num_outputs
) {
    return marmot_graph_execute(graph, ctx, (const marmot_tensor_t **)inputs, num_inputs, outputs, num_outputs);
}

static void ffn_teardown(
    marmot_graph_t *graph,
    marmot_tensor_t **inputs,
    size_t num_inputs,
    marmot_tensor_t **outputs,
    size_t num_outputs,
    void *user_data
) {
    (void)user_data;

    for (size_t i = 0; i < num_inputs; ++i) {
        marmot_tensor_destroy(inputs[i]);
    }
    for (size_t i = 0; i < num_outputs; ++i) {
        marmot_tensor_destroy(outputs[i]);
    }
    free(inputs);
    free(outputs);
    marmot_graph_destroy(graph);
}

static marmot_bench_workload_t *
create_ffn_workload(uint32_t batch, uint32_t seq_len, uint32_t hidden, uint32_t intermediate, marmot_dtype_t dtype) {
    marmot_bench_workload_t *w = calloc(1, sizeof(marmot_bench_workload_t));
    if (w == nullptr)
        return nullptr;

    ffn_params_t *params = calloc(1, sizeof(ffn_params_t));
    if (params == nullptr) {
        free(w);
        return nullptr;
    }

    params->batch = batch;
    params->seq_len = seq_len;
    params->hidden = hidden;
    params->intermediate = intermediate;
    params->dtype = dtype;

    const char *dtype_str = dtype == MARMOT_DTYPE_FLOAT16 ? "f16" : "f32";
    snprintf(params->name, sizeof(params->name), "ffn_%s_h%u_i%u", dtype_str, hidden, intermediate);

    size_t elem_size = dtype == MARMOT_DTYPE_FLOAT16 ? 2 : 4;
    size_t tokens = batch * seq_len;

    uint64_t up_flops = 2ULL * tokens * hidden * intermediate;
    uint64_t gelu_flops = 8ULL * tokens * intermediate;
    uint64_t down_flops = 2ULL * tokens * intermediate * hidden;
    uint64_t flops = up_flops + gelu_flops + down_flops;

    uint64_t bytes_read = (tokens * hidden + hidden * intermediate + intermediate * hidden) * elem_size;
    uint64_t bytes_written = tokens * hidden * elem_size;

    w->desc.name = params->name;
    w->desc.category = MARMOT_BENCH_CATEGORY_COMPOSITE;
    w->desc.primary_dtype = dtype;
    w->desc.flops = flops;
    w->desc.bytes_read = bytes_read;
    w->desc.bytes_written = bytes_written;

    memset(&w->desc.signature, 0, sizeof(w->desc.signature));
    w->desc.signature.op_id = MARMOT_OP_MATMUL;
    w->desc.signature.input_dtype = dtype;
    w->desc.signature.output_dtype = dtype;
    w->desc.signature.accum_dtype = MARMOT_DTYPE_FLOAT32;
    w->desc.signature.dims.matmul.M = tokens;
    w->desc.signature.dims.matmul.N = intermediate;
    w->desc.signature.dims.matmul.K = hidden;

    w->user_data = params;
    w->setup = ffn_setup;
    w->execute = ffn_execute;
    w->teardown = ffn_teardown;

    return w;
}

void marmot_bench_register_ffn_workloads(marmot_bench_suite_t *suite) {
    uint32_t batch = 1;
    uint32_t seq_len = 512;
    uint32_t hidden = 4096;
    uint32_t intermediate = 11008;

    marmot_bench_workload_t *w = create_ffn_workload(batch, seq_len, hidden, intermediate, MARMOT_DTYPE_FLOAT16);
    if (w)
        marmot_bench_suite_add(suite, w);
}
