#include "marmot/graph/graph.h"
#include "marmot/graph/graph_types.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../bench_workloads.h"

typedef struct {
    uint32_t n_elems;
    marmot_dtype_t dtype;
    bool fused;
    char name[64];
} fusion_params_t;

static void init_tensor_desc_1d(marmot_graph_tensor_desc_t *desc, size_t dim0, marmot_dtype_t dtype) {
    memset(desc, 0, sizeof(*desc));
    desc->dtype = dtype;
    desc->ndim = 1;
    desc->shape[0] = dim0;
    desc->strides[0] = 1;
}

static marmot_error_t fusion_setup(
    marmot_backend_type_t backend,
    marmot_context_t *ctx,
    marmot_graph_t **graph,
    marmot_tensor_t ***inputs,
    size_t *num_inputs,
    marmot_tensor_t ***outputs,
    size_t *num_outputs,
    void *user_data
) {
    fusion_params_t *params = (fusion_params_t *)user_data;
    *graph = marmot_graph_create();
    if (*graph == nullptr)
        return MARMOT_ERROR_OUT_OF_MEMORY;

    marmot_graph_tensor_desc_t desc;
    init_tensor_desc_1d(&desc, params->n_elems, params->dtype);

    marmot_value_id_t a_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t b_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t c_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t tmp_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t out_id = MARMOT_VALUE_ID_INVALID;

    marmot_error_t err = marmot_graph_add_input(*graph, &desc, &a_id);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }

    err = marmot_graph_add_input(*graph, &desc, &b_id);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }

    err = marmot_graph_add_input(*graph, &desc, &c_id);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }

    if (params->fused) {
        marmot_value_id_t op_inputs[3] = {a_id, b_id, c_id};
        err = marmot_graph_add_op(*graph, "fma", nullptr, op_inputs, 3, &desc, 1, &out_id);
        if (err != MARMOT_SUCCESS) {
            marmot_graph_destroy(*graph);
            return err;
        }
    } else {
        marmot_value_id_t mul_inputs[2] = {a_id, b_id};
        err = marmot_graph_add_op(*graph, "mul", nullptr, mul_inputs, 2, &desc, 1, &tmp_id);
        if (err != MARMOT_SUCCESS) {
            marmot_graph_destroy(*graph);
            return err;
        }
        marmot_value_id_t add_inputs[2] = {tmp_id, c_id};
        err = marmot_graph_add_op(*graph, "add", nullptr, add_inputs, 2, &desc, 1, &out_id);
        if (err != MARMOT_SUCCESS) {
            marmot_graph_destroy(*graph);
            return err;
        }
    }

    err = marmot_graph_finalize(*graph, backend);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }

    size_t shape[1] = {params->n_elems};

    *inputs = malloc(3 * sizeof(marmot_tensor_t *));
    if (*inputs == nullptr) {
        marmot_graph_destroy(*graph);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    *outputs = malloc(sizeof(marmot_tensor_t *));
    if (*outputs == nullptr) {
        free(*inputs);
        marmot_graph_destroy(*graph);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    (*inputs)[0] = marmot_tensor_create(ctx, shape, 1, params->dtype);
    (*inputs)[1] = marmot_tensor_create(ctx, shape, 1, params->dtype);
    (*inputs)[2] = marmot_tensor_create(ctx, shape, 1, params->dtype);
    (*outputs)[0] = marmot_tensor_create(ctx, shape, 1, params->dtype);
    *num_inputs = 3;
    *num_outputs = 1;

    if ((*inputs)[0] == nullptr || (*inputs)[1] == nullptr || (*inputs)[2] == nullptr || (*outputs)[0] == nullptr) {
        marmot_tensor_destroy((*inputs)[0]);
        marmot_tensor_destroy((*inputs)[1]);
        marmot_tensor_destroy((*inputs)[2]);
        marmot_tensor_destroy((*outputs)[0]);
        free(*inputs);
        free(*outputs);
        marmot_graph_destroy(*graph);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t fusion_execute(
    marmot_context_t *ctx,
    marmot_graph_t *graph,
    marmot_tensor_t **inputs,
    size_t num_inputs,
    marmot_tensor_t **outputs,
    size_t num_outputs
) {
    return marmot_graph_execute(graph, ctx, (const marmot_tensor_t **)inputs, num_inputs, outputs, num_outputs);
}

static void fusion_teardown(
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

static size_t dtype_size(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT64:
        return 8;
    case MARMOT_DTYPE_FLOAT32:
        return 4;
    case MARMOT_DTYPE_FLOAT16:
    case MARMOT_DTYPE_BFLOAT16:
        return 2;
    default:
        return 4;
    }
}

static const char *dtype_to_str(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT16:
        return "f16";
    case MARMOT_DTYPE_BFLOAT16:
        return "bf16";
    case MARMOT_DTYPE_FLOAT32:
        return "f32";
    case MARMOT_DTYPE_FLOAT64:
        return "f64";
    default:
        return "unk";
    }
}

static marmot_bench_workload_t *create_fusion_workload(uint32_t n_elems, marmot_dtype_t dtype, bool fused) {
    marmot_bench_workload_t *w = calloc(1, sizeof(marmot_bench_workload_t));
    if (w == nullptr)
        return nullptr;

    fusion_params_t *params = calloc(1, sizeof(fusion_params_t));
    if (params == nullptr) {
        free(w);
        return nullptr;
    }

    params->n_elems = n_elems;
    params->dtype = dtype;
    params->fused = fused;

    const char *op_name = fused ? "fma" : "mul_add";
    const char *dtype_str = dtype_to_str(dtype);
    if (n_elems >= 1000000) {
        snprintf(params->name, sizeof(params->name), "%s_%s_%uM", op_name, dtype_str, n_elems / 1000000);
    } else if (n_elems >= 1000) {
        snprintf(params->name, sizeof(params->name), "%s_%s_%uK", op_name, dtype_str, n_elems / 1000);
    } else {
        snprintf(params->name, sizeof(params->name), "%s_%s_%u", op_name, dtype_str, n_elems);
    }

    size_t elem_size = dtype_size(dtype);
    uint64_t flops = 2ULL * n_elems;
    uint64_t bytes_read = fused ? 3ULL * n_elems * elem_size : 4ULL * n_elems * elem_size;
    uint64_t bytes_written = fused ? 1ULL * n_elems * elem_size : 2ULL * n_elems * elem_size;

    w->desc.name = params->name;
    w->desc.category = MARMOT_BENCH_CATEGORY_MICRO;
    w->desc.primary_dtype = dtype;
    w->desc.flops = flops;
    w->desc.bytes_read = bytes_read;
    w->desc.bytes_written = bytes_written;

    memset(&w->desc.signature, 0, sizeof(w->desc.signature));
    w->desc.signature.op_id = fused ? MARMOT_OP_FMA : MARMOT_OP_ADD;
    w->desc.signature.profile_id = MARMOT_PROFILE_SCALAR;
    w->desc.signature.input_dtype = dtype;
    w->desc.signature.weight_dtype = dtype;
    w->desc.signature.output_dtype = dtype;
    w->desc.signature.accum_dtype = MARMOT_DTYPE_FLOAT32;
    w->desc.signature.dims.elementwise.n_elems = n_elems;

    w->user_data = params;
    w->setup = fusion_setup;
    w->execute = fusion_execute;
    w->teardown = fusion_teardown;

    return w;
}

void marmot_bench_register_fusion_workloads(marmot_bench_suite_t *suite) {
    static const uint32_t sizes[] = {1024 * 1024, 16 * 1024 * 1024};
    static const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    static const marmot_dtype_t dtypes[] = {MARMOT_DTYPE_FLOAT32};
    static const size_t num_dtypes = sizeof(dtypes) / sizeof(dtypes[0]);

    for (size_t i = 0; i < num_sizes; ++i) {
        uint32_t sz = sizes[i];
        for (size_t j = 0; j < num_dtypes; ++j) {
            marmot_bench_workload_t *fused = create_fusion_workload(sz, dtypes[j], true);
            if (fused)
                marmot_bench_suite_add(suite, fused);

            marmot_bench_workload_t *unfused = create_fusion_workload(sz, dtypes[j], false);
            if (unfused)
                marmot_bench_suite_add(suite, unfused);
        }
    }
}
