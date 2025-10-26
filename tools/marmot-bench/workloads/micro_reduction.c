#include "../bench_workloads.h"

#include "marmot/graph/graph.h"
#include "marmot/graph/graph_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t batch;
    uint32_t dim;
    marmot_op_id_t op_id;
    const char *op_name;
    marmot_dtype_t dtype;
    char name[64];
} reduction_params_t;

static void init_tensor_desc_2d(marmot_graph_tensor_desc_t *desc, size_t dim0, size_t dim1, marmot_dtype_t dtype) {
    memset(desc, 0, sizeof(*desc));
    desc->dtype = dtype;
    desc->ndim = 2;
    desc->shape[0] = dim0;
    desc->shape[1] = dim1;
    desc->strides[1] = 1;
    desc->strides[0] = dim1;
}

static marmot_error_t reduction_setup(
    marmot_backend_type_t backend,
    marmot_context_t *ctx,
    marmot_graph_t **graph,
    marmot_tensor_t ***inputs,
    size_t *num_inputs,
    marmot_tensor_t ***outputs,
    size_t *num_outputs,
    void *user_data
) {
    reduction_params_t *params = (reduction_params_t *)user_data;
    (void)ctx;

    *graph = marmot_graph_create();
    if (*graph == nullptr)
        return MARMOT_ERROR_OUT_OF_MEMORY;

    marmot_graph_tensor_desc_t in_desc, out_desc;
    init_tensor_desc_2d(&in_desc, params->batch, params->dim, params->dtype);

    if (params->op_id == MARMOT_OP_SOFTMAX || params->op_id == MARMOT_OP_LAYERNORM ||
        params->op_id == MARMOT_OP_RMS_NORM) {
        init_tensor_desc_2d(&out_desc, params->batch, params->dim, params->dtype);
    } else {
        memset(&out_desc, 0, sizeof(out_desc));
        out_desc.dtype = params->dtype;
        out_desc.ndim = 1;
        out_desc.shape[0] = params->batch;
        out_desc.strides[0] = 1;
    }

    marmot_value_id_t in_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t out_id = MARMOT_VALUE_ID_INVALID;

    marmot_error_t err = marmot_graph_add_input(*graph, &in_desc, &in_id);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }

    marmot_value_id_t op_inputs[1] = {in_id};
    err = marmot_graph_add_op(*graph, params->op_name, nullptr, op_inputs, 1, &out_desc, 1, &out_id);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }

    err = marmot_graph_finalize(*graph, backend);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }

    size_t in_shape[2] = {params->batch, params->dim};
    size_t out_shape[2] = {params->batch, params->dim};
    size_t out_ndim = 2;
    if (params->op_id != MARMOT_OP_SOFTMAX && params->op_id != MARMOT_OP_LAYERNORM &&
        params->op_id != MARMOT_OP_RMS_NORM) {
        out_shape[0] = params->batch;
        out_ndim = 1;
    }

    *inputs = malloc(1 * sizeof(marmot_tensor_t *));
    (*inputs)[0] = marmot_tensor_create(ctx, in_shape, 2, params->dtype);
    *num_inputs = 1;

    *outputs = malloc(1 * sizeof(marmot_tensor_t *));
    (*outputs)[0] = marmot_tensor_create(ctx, out_shape, out_ndim, params->dtype);
    *num_outputs = 1;

    return MARMOT_SUCCESS;
}

static marmot_error_t reduction_execute(
    marmot_context_t *ctx,
    marmot_graph_t *graph,
    marmot_tensor_t **inputs,
    size_t num_inputs,
    marmot_tensor_t **outputs,
    size_t num_outputs
) {
    return marmot_graph_execute(graph, ctx, (const marmot_tensor_t **)inputs, num_inputs, outputs, num_outputs);
}

static void reduction_teardown(
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
create_reduction_workload(uint32_t batch, uint32_t dim, marmot_op_id_t op_id, const char *op_name, marmot_dtype_t dtype) {
    marmot_bench_workload_t *w = calloc(1, sizeof(marmot_bench_workload_t));
    if (w == nullptr)
        return nullptr;

    reduction_params_t *params = calloc(1, sizeof(reduction_params_t));
    if (params == nullptr) {
        free(w);
        return nullptr;
    }

    params->batch = batch;
    params->dim = dim;
    params->op_id = op_id;
    params->op_name = op_name;
    params->dtype = dtype;

    const char *dtype_str = dtype == MARMOT_DTYPE_FLOAT16 ? "f16" : "f32";
    snprintf(params->name, sizeof(params->name), "%s_%s_%ux%u", op_name, dtype_str, batch, dim);

    size_t elem_size = dtype == MARMOT_DTYPE_FLOAT16 ? 2 : 4;
    uint64_t n_elems = (uint64_t)batch * dim;
    uint64_t bytes_read = n_elems * elem_size;
    uint64_t bytes_written;
    uint64_t flops;

    if (op_id == MARMOT_OP_SOFTMAX) {
        bytes_written = n_elems * elem_size;
        flops = 5ULL * n_elems;
    } else if (op_id == MARMOT_OP_LAYERNORM || op_id == MARMOT_OP_RMS_NORM) {
        bytes_written = n_elems * elem_size;
        flops = 8ULL * n_elems;
    } else {
        bytes_written = batch * elem_size;
        flops = n_elems;
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
    w->desc.signature.accum_dtype = MARMOT_DTYPE_FLOAT32;

    w->user_data = params;
    w->setup = reduction_setup;
    w->execute = reduction_execute;
    w->teardown = reduction_teardown;

    return w;
}

void marmot_bench_register_reduction_workloads(marmot_bench_suite_t *suite) {
    uint32_t batch = 512;
    uint32_t dim = 4096;

    marmot_bench_workload_t *w_softmax = create_reduction_workload(batch, dim, MARMOT_OP_SOFTMAX, "softmax", MARMOT_DTYPE_FLOAT32);
    if (w_softmax)
        marmot_bench_suite_add(suite, w_softmax);

    marmot_bench_workload_t *w_layernorm = create_reduction_workload(batch, dim, MARMOT_OP_LAYERNORM, "layernorm", MARMOT_DTYPE_FLOAT32);
    if (w_layernorm)
        marmot_bench_suite_add(suite, w_layernorm);

    marmot_bench_workload_t *w_rmsnorm = create_reduction_workload(batch, dim, MARMOT_OP_RMS_NORM, "rms_norm", MARMOT_DTYPE_FLOAT32);
    if (w_rmsnorm)
        marmot_bench_suite_add(suite, w_rmsnorm);

    marmot_bench_workload_t *w_sum = create_reduction_workload(batch, dim, MARMOT_OP_REDUCTION_SUM, "reduction_sum", MARMOT_DTYPE_FLOAT32);
    if (w_sum)
        marmot_bench_suite_add(suite, w_sum);

    marmot_bench_workload_t *w_max = create_reduction_workload(batch, dim, MARMOT_OP_REDUCTION_MAX, "reduction_max", MARMOT_DTYPE_FLOAT32);
    if (w_max)
        marmot_bench_suite_add(suite, w_max);
}
