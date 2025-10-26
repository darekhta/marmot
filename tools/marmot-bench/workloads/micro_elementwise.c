#include "../bench_workloads.h"

#include "marmot/graph/graph.h"
#include "marmot/graph/graph_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t n_elems;
    marmot_op_id_t op_id;
    const char *op_name;
    marmot_dtype_t dtype;
    char name[64];
    bool is_unary;
} elementwise_params_t;

static void init_tensor_desc_1d(marmot_graph_tensor_desc_t *desc, size_t dim0, marmot_dtype_t dtype) {
    memset(desc, 0, sizeof(*desc));
    desc->dtype = dtype;
    desc->ndim = 1;
    desc->shape[0] = dim0;
    desc->strides[0] = 1;
}

static marmot_error_t elementwise_setup(
    marmot_backend_type_t backend,
    marmot_context_t *ctx,
    marmot_graph_t **graph,
    marmot_tensor_t ***inputs,
    size_t *num_inputs,
    marmot_tensor_t ***outputs,
    size_t *num_outputs,
    void *user_data
) {
    elementwise_params_t *params = (elementwise_params_t *)user_data;
    (void)ctx;

    *graph = marmot_graph_create();
    if (*graph == nullptr)
        return MARMOT_ERROR_OUT_OF_MEMORY;

    marmot_graph_tensor_desc_t desc;
    init_tensor_desc_1d(&desc, params->n_elems, params->dtype);

    marmot_value_id_t a_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t b_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t out_id = MARMOT_VALUE_ID_INVALID;

    marmot_error_t err = marmot_graph_add_input(*graph, &desc, &a_id);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }

    size_t n_op_inputs = 1;
    marmot_value_id_t op_inputs[2] = {a_id, MARMOT_VALUE_ID_INVALID};

    if (!params->is_unary) {
        err = marmot_graph_add_input(*graph, &desc, &b_id);
        if (err != MARMOT_SUCCESS) {
            marmot_graph_destroy(*graph);
            return err;
        }
        op_inputs[1] = b_id;
        n_op_inputs = 2;
    }

    err = marmot_graph_add_op(*graph, params->op_name, nullptr, op_inputs, n_op_inputs, &desc, 1, &out_id);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }

    err = marmot_graph_finalize(*graph, backend);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }

    size_t shape[1] = {params->n_elems};

    if (params->is_unary) {
        *inputs = malloc(1 * sizeof(marmot_tensor_t *));
        (*inputs)[0] = marmot_tensor_create(ctx, shape, 1, params->dtype);
        *num_inputs = 1;
    } else {
        *inputs = malloc(2 * sizeof(marmot_tensor_t *));
        (*inputs)[0] = marmot_tensor_create(ctx, shape, 1, params->dtype);
        (*inputs)[1] = marmot_tensor_create(ctx, shape, 1, params->dtype);
        *num_inputs = 2;
    }

    *outputs = malloc(1 * sizeof(marmot_tensor_t *));
    (*outputs)[0] = marmot_tensor_create(ctx, shape, 1, params->dtype);
    *num_outputs = 1;

    return MARMOT_SUCCESS;
}

static marmot_error_t elementwise_execute(
    marmot_context_t *ctx,
    marmot_graph_t *graph,
    marmot_tensor_t **inputs,
    size_t num_inputs,
    marmot_tensor_t **outputs,
    size_t num_outputs
) {
    return marmot_graph_execute(graph, ctx, (const marmot_tensor_t **)inputs, num_inputs, outputs, num_outputs);
}

static void elementwise_teardown(
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

static marmot_bench_workload_t *create_elementwise_workload(
    uint32_t n_elems, marmot_op_id_t op_id, const char *op_name, bool is_unary, marmot_dtype_t dtype
) {
    marmot_bench_workload_t *w = calloc(1, sizeof(marmot_bench_workload_t));
    if (w == nullptr)
        return nullptr;

    elementwise_params_t *params = calloc(1, sizeof(elementwise_params_t));
    if (params == nullptr) {
        free(w);
        return nullptr;
    }

    params->n_elems = n_elems;
    params->op_id = op_id;
    params->op_name = op_name;
    params->dtype = dtype;
    params->is_unary = is_unary;

    const char *dtype_str = dtype == MARMOT_DTYPE_FLOAT16 ? "f16" : "f32";
    if (n_elems >= 1000000) {
        snprintf(params->name, sizeof(params->name), "%s_%s_%uM", op_name, dtype_str, n_elems / 1000000);
    } else if (n_elems >= 1000) {
        snprintf(params->name, sizeof(params->name), "%s_%s_%uK", op_name, dtype_str, n_elems / 1000);
    } else {
        snprintf(params->name, sizeof(params->name), "%s_%s_%u", op_name, dtype_str, n_elems);
    }

    size_t elem_size = dtype == MARMOT_DTYPE_FLOAT16 ? 2 : 4;
    uint64_t bytes_read = is_unary ? (uint64_t)n_elems * elem_size : 2ULL * n_elems * elem_size;
    uint64_t bytes_written = (uint64_t)n_elems * elem_size;
    uint64_t flops = (uint64_t)n_elems;
    if (op_id == MARMOT_OP_GELU || op_id == MARMOT_OP_SILU) {
        flops = 8ULL * n_elems;
    }

    w->desc.name = params->name;
    w->desc.category = MARMOT_BENCH_CATEGORY_MICRO;
    w->desc.primary_dtype = dtype;
    w->desc.flops = flops;
    w->desc.bytes_read = bytes_read;
    w->desc.bytes_written = bytes_written;

    memset(&w->desc.signature, 0, sizeof(w->desc.signature));
    w->desc.signature.op_id = op_id;
    w->desc.signature.profile_id = MARMOT_PROFILE_SCALAR;
    w->desc.signature.input_dtype = dtype;
    w->desc.signature.weight_dtype = dtype;
    w->desc.signature.output_dtype = dtype;
    w->desc.signature.accum_dtype = dtype;
    w->desc.signature.dims.elementwise.n_elems = n_elems;

    w->user_data = params;
    w->setup = elementwise_setup;
    w->execute = elementwise_execute;
    w->teardown = elementwise_teardown;

    return w;
}

void marmot_bench_register_elementwise_workloads(marmot_bench_suite_t *suite) {
    static const uint32_t sizes[] = {1024 * 1024, 16 * 1024 * 1024};
    static const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; ++i) {
        uint32_t sz = sizes[i];

        marmot_bench_workload_t *w_add = create_elementwise_workload(sz, MARMOT_OP_ADD, "add", false, MARMOT_DTYPE_FLOAT32);
        if (w_add)
            marmot_bench_suite_add(suite, w_add);

        marmot_bench_workload_t *w_mul = create_elementwise_workload(sz, MARMOT_OP_MUL, "mul", false, MARMOT_DTYPE_FLOAT32);
        if (w_mul)
            marmot_bench_suite_add(suite, w_mul);

        marmot_bench_workload_t *w_gelu = create_elementwise_workload(sz, MARMOT_OP_GELU, "gelu", true, MARMOT_DTYPE_FLOAT32);
        if (w_gelu)
            marmot_bench_suite_add(suite, w_gelu);

        marmot_bench_workload_t *w_silu = create_elementwise_workload(sz, MARMOT_OP_SILU, "silu", true, MARMOT_DTYPE_FLOAT32);
        if (w_silu)
            marmot_bench_suite_add(suite, w_silu);
    }
}
