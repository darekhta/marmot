#include "marmot/graph/graph.h"
#include "marmot/graph/graph_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <string.h>

#include "../bench_workloads.h"

typedef struct {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    marmot_dtype_t dtype;
    char name[64];
} matmul_params_t;

static void init_tensor_desc_2d(marmot_graph_tensor_desc_t *desc, size_t dim0, size_t dim1, marmot_dtype_t dtype) {
    memset(desc, 0, sizeof(*desc));
    desc->dtype = dtype;
    desc->ndim = 2;
    desc->shape[0] = dim0;
    desc->shape[1] = dim1;
    desc->strides[1] = 1;
    desc->strides[0] = dim1;
}

static marmot_error_t matmul_setup(
    marmot_backend_type_t backend, marmot_context_t *ctx, marmot_graph_t **graph, marmot_tensor_t ***inputs,
    size_t *num_inputs, marmot_tensor_t ***outputs, size_t *num_outputs, void *user_data
) {
    matmul_params_t *params = (matmul_params_t *)user_data;
    *graph = marmot_graph_create();
    if (*graph == nullptr)
        return MARMOT_ERROR_OUT_OF_MEMORY;

    marmot_graph_tensor_desc_t a_desc, b_desc, c_desc;
    init_tensor_desc_2d(&a_desc, params->M, params->K, params->dtype);
    init_tensor_desc_2d(&b_desc, params->K, params->N, params->dtype);
    init_tensor_desc_2d(&c_desc, params->M, params->N, params->dtype);

    marmot_value_id_t a_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t b_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t c_id = MARMOT_VALUE_ID_INVALID;

    marmot_error_t err = marmot_graph_add_input(*graph, &a_desc, &a_id);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }

    err = marmot_graph_add_input(*graph, &b_desc, &b_id);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }

    marmot_value_id_t op_inputs[2] = {a_id, b_id};
    marmot_dtype_t accum = (params->dtype == MARMOT_DTYPE_FLOAT64) ? MARMOT_DTYPE_FLOAT64 : MARMOT_DTYPE_FLOAT32;
    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_MATMUL,
        .profile_id = ctx != nullptr ? ctx->best_profile : MARMOT_PROFILE_INVALID,
        .matmul_layout = MARMOT_MATMUL_LAYOUT_NN,
        .input_dtype = params->dtype,
        .weight_dtype = params->dtype,
        .output_dtype = params->dtype,
        .accum_dtype = accum,
    };
    err = marmot_graph_add_op(*graph, "matmul", &sig, op_inputs, 2, &c_desc, 1, &c_id);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }

    err = marmot_graph_finalize(*graph, backend);
    if (err != MARMOT_SUCCESS) {
        marmot_graph_destroy(*graph);
        return err;
    }

    size_t a_shape[2] = {params->M, params->K};
    size_t b_shape[2] = {params->K, params->N};
    size_t c_shape[2] = {params->M, params->N};

    *inputs = malloc(2 * sizeof(marmot_tensor_t *));
    (*inputs)[0] = marmot_tensor_create(ctx, a_shape, 2, params->dtype);
    (*inputs)[1] = marmot_tensor_create(ctx, b_shape, 2, params->dtype);
    *num_inputs = 2;

    *outputs = malloc(1 * sizeof(marmot_tensor_t *));
    (*outputs)[0] = marmot_tensor_create(ctx, c_shape, 2, params->dtype);
    *num_outputs = 1;

    if ((*inputs)[0] == nullptr || (*inputs)[1] == nullptr || (*outputs)[0] == nullptr) {
        marmot_tensor_destroy((*inputs)[0]);
        marmot_tensor_destroy((*inputs)[1]);
        marmot_tensor_destroy((*outputs)[0]);
        free(*inputs);
        free(*outputs);
        marmot_graph_destroy(*graph);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t matmul_execute(
    marmot_context_t *ctx, marmot_graph_t *graph, marmot_tensor_t **inputs, size_t num_inputs,
    marmot_tensor_t **outputs, size_t num_outputs
) {
    return marmot_graph_execute(graph, ctx, (const marmot_tensor_t **)inputs, num_inputs, outputs, num_outputs);
}

static void matmul_teardown(
    marmot_graph_t *graph, marmot_tensor_t **inputs, size_t num_inputs, marmot_tensor_t **outputs, size_t num_outputs,
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

static marmot_bench_workload_t *create_matmul_workload(uint32_t M, uint32_t N, uint32_t K, marmot_dtype_t dtype) {
    marmot_bench_workload_t *w = calloc(1, sizeof(marmot_bench_workload_t));
    if (w == nullptr)
        return nullptr;

    matmul_params_t *params = calloc(1, sizeof(matmul_params_t));
    if (params == nullptr) {
        free(w);
        return nullptr;
    }

    params->M = M;
    params->N = N;
    params->K = K;
    params->dtype = dtype;

    snprintf(params->name, sizeof(params->name), "matmul_%s_%ux%ux%u", dtype_to_str(dtype), M, N, K);

    size_t elem_size = dtype_size(dtype);
    uint64_t flops = 2ULL * M * N * K;
    uint64_t bytes_read = (uint64_t)M * K * elem_size + (uint64_t)K * N * elem_size;
    uint64_t bytes_written = (uint64_t)M * N * elem_size;

    w->desc.name = params->name;
    w->desc.category = MARMOT_BENCH_CATEGORY_MICRO;
    w->desc.primary_dtype = dtype;
    w->desc.flops = flops;
    w->desc.bytes_read = bytes_read;
    w->desc.bytes_written = bytes_written;

    memset(&w->desc.signature, 0, sizeof(w->desc.signature));
    w->desc.signature.op_id = MARMOT_OP_MATMUL;
    w->desc.signature.profile_id = MARMOT_PROFILE_INVALID;
    w->desc.signature.matmul_layout = MARMOT_MATMUL_LAYOUT_NN;
    w->desc.signature.input_dtype = dtype;
    w->desc.signature.weight_dtype = dtype;
    w->desc.signature.output_dtype = dtype;
    w->desc.signature.accum_dtype = MARMOT_DTYPE_FLOAT32;
    w->desc.signature.dims.matmul.N = M;
    w->desc.signature.dims.matmul.M = N;
    w->desc.signature.dims.matmul.K = K;

    w->user_data = params;
    w->setup = matmul_setup;
    w->execute = matmul_execute;
    w->teardown = matmul_teardown;

    return w;
}

void marmot_bench_register_matmul_workloads(marmot_bench_suite_t *suite) {
    static const uint32_t sizes[][3] = {
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
    };
    static const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    static const marmot_dtype_t dtypes[] = {
        MARMOT_DTYPE_FLOAT16,
        MARMOT_DTYPE_BFLOAT16,
        MARMOT_DTYPE_FLOAT32,
        MARMOT_DTYPE_FLOAT64,
    };
    static const size_t num_dtypes = sizeof(dtypes) / sizeof(dtypes[0]);

    for (size_t i = 0; i < num_sizes; ++i) {
        uint32_t M = sizes[i][0];
        uint32_t N = sizes[i][1];
        uint32_t K = sizes[i][2];
        for (size_t j = 0; j < num_dtypes; ++j) {
            marmot_bench_workload_t *w = create_matmul_workload(M, N, K, dtypes[j]);
            if (w)
                marmot_bench_suite_add(suite, w);
        }
    }
}
