#include "../bench_workloads.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t batch;
    uint32_t seq_len;
    uint32_t dim;
    marmot_dtype_t dtype;
    marmot_dtype_t positions_dtype;
    marmot_rope_type_t rope_type;
    char name[64];
} rope_params_t;

static void fill_input(float *data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        data[i] = 0.01f * (float)i;
    }
}

static void fill_positions_f32(float *data, size_t batch, size_t seq_len) {
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            data[b * seq_len + s] = (float)s;
        }
    }
}

static void fill_positions_i32(int32_t *data, size_t batch, size_t seq_len) {
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            data[b * seq_len + s] = (int32_t)s;
        }
    }
}

static marmot_error_t rope_setup(
    marmot_backend_type_t backend,
    marmot_context_t *ctx,
    marmot_graph_t **graph,
    marmot_tensor_t ***inputs,
    size_t *num_inputs,
    marmot_tensor_t ***outputs,
    size_t *num_outputs,
    void *user_data
) {
    rope_params_t *params = (rope_params_t *)user_data;

    *graph = nullptr;

    const size_t shape[3] = {params->batch, params->seq_len, params->dim};
    const size_t positions_shape[2] = {params->batch, params->seq_len};

    *inputs = malloc(2 * sizeof(marmot_tensor_t *));
    *outputs = malloc(1 * sizeof(marmot_tensor_t *));
    if (*inputs == nullptr || *outputs == nullptr) {
        free(*inputs);
        free(*outputs);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    (*inputs)[0] = marmot_tensor_create(ctx, shape, 3, params->dtype);
    (*inputs)[1] = marmot_tensor_create(ctx, positions_shape, 2, params->positions_dtype);
    (*outputs)[0] = marmot_tensor_create(ctx, shape, 3, params->dtype);
    *num_inputs = 2;
    *num_outputs = 1;

    if ((*inputs)[0] == nullptr || (*inputs)[1] == nullptr || (*outputs)[0] == nullptr) {
        marmot_tensor_destroy((*inputs)[0]);
        marmot_tensor_destroy((*inputs)[1]);
        marmot_tensor_destroy((*outputs)[0]);
        free(*inputs);
        free(*outputs);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    const size_t element_count = (size_t)params->batch * params->seq_len * params->dim;
    const size_t positions_count = (size_t)params->batch * params->seq_len;

    float *input_data = malloc(element_count * sizeof(float));
    if (input_data == nullptr) {
        marmot_tensor_destroy((*inputs)[0]);
        marmot_tensor_destroy((*inputs)[1]);
        marmot_tensor_destroy((*outputs)[0]);
        free(*inputs);
        free(*outputs);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    fill_input(input_data, element_count);
    marmot_error_t err = marmot_tensor_copy_from_host_buffer(
        ctx, (*inputs)[0], input_data, element_count * sizeof(float)
    );
    free(input_data);
    if (err != MARMOT_SUCCESS) {
        marmot_tensor_destroy((*inputs)[0]);
        marmot_tensor_destroy((*inputs)[1]);
        marmot_tensor_destroy((*outputs)[0]);
        free(*inputs);
        free(*outputs);
        return err;
    }

    if (params->positions_dtype == MARMOT_DTYPE_FLOAT32) {
        float *positions = malloc(positions_count * sizeof(float));
        if (positions == nullptr) {
            marmot_tensor_destroy((*inputs)[0]);
            marmot_tensor_destroy((*inputs)[1]);
            marmot_tensor_destroy((*outputs)[0]);
            free(*inputs);
            free(*outputs);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        fill_positions_f32(positions, params->batch, params->seq_len);
        err = marmot_tensor_copy_from_host_buffer(
            ctx, (*inputs)[1], positions, positions_count * sizeof(float)
        );
        free(positions);
        if (err != MARMOT_SUCCESS) {
            marmot_tensor_destroy((*inputs)[0]);
            marmot_tensor_destroy((*inputs)[1]);
            marmot_tensor_destroy((*outputs)[0]);
            free(*inputs);
            free(*outputs);
            return err;
        }
    } else if (params->positions_dtype == MARMOT_DTYPE_INT32) {
        int32_t *positions = malloc(positions_count * sizeof(int32_t));
        if (positions == nullptr) {
            marmot_tensor_destroy((*inputs)[0]);
            marmot_tensor_destroy((*inputs)[1]);
            marmot_tensor_destroy((*outputs)[0]);
            free(*inputs);
            free(*outputs);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        fill_positions_i32(positions, params->batch, params->seq_len);
        err = marmot_tensor_copy_from_host_buffer(
            ctx, (*inputs)[1], positions, positions_count * sizeof(int32_t)
        );
        free(positions);
        if (err != MARMOT_SUCCESS) {
            marmot_tensor_destroy((*inputs)[0]);
            marmot_tensor_destroy((*inputs)[1]);
            marmot_tensor_destroy((*outputs)[0]);
            free(*inputs);
            free(*outputs);
            return err;
        }
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t rope_execute_norm(
    marmot_context_t *ctx,
    marmot_graph_t *graph,
    marmot_tensor_t **inputs,
    size_t num_inputs,
    marmot_tensor_t **outputs,
    size_t num_outputs
) {
    (void)graph;
    (void)num_inputs;
    (void)num_outputs;

    marmot_rope_params_t params = marmot_rope_params_default();
    params.positions = inputs[1];
    params.rope_type = MARMOT_ROPE_TYPE_NORM;
    return marmot_rope(ctx, inputs[0], &params, outputs[0]);
}

static marmot_error_t rope_execute_neox(
    marmot_context_t *ctx,
    marmot_graph_t *graph,
    marmot_tensor_t **inputs,
    size_t num_inputs,
    marmot_tensor_t **outputs,
    size_t num_outputs
) {
    (void)graph;
    (void)num_inputs;
    (void)num_outputs;

    marmot_rope_params_t params = marmot_rope_params_default();
    params.positions = inputs[1];
    params.rope_type = MARMOT_ROPE_TYPE_NEOX;
    return marmot_rope(ctx, inputs[0], &params, outputs[0]);
}

static void rope_teardown(
    marmot_graph_t *graph,
    marmot_tensor_t **inputs,
    size_t num_inputs,
    marmot_tensor_t **outputs,
    size_t num_outputs,
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

static const char *rope_type_to_str(marmot_rope_type_t rope_type) {
    return rope_type == MARMOT_ROPE_TYPE_NEOX ? "neox" : "norm";
}

static const char *positions_dtype_to_str(marmot_dtype_t dtype) {
    return dtype == MARMOT_DTYPE_INT32 ? "cache" : "no_cache";
}

static marmot_bench_workload_t *create_rope_workload(
    uint32_t batch,
    uint32_t seq_len,
    uint32_t dim,
    marmot_dtype_t dtype,
    marmot_dtype_t positions_dtype,
    marmot_rope_type_t rope_type
) {
    marmot_bench_workload_t *w = calloc(1, sizeof(marmot_bench_workload_t));
    if (w == nullptr)
        return nullptr;

    rope_params_t *params = calloc(1, sizeof(rope_params_t));
    if (params == nullptr) {
        free(w);
        return nullptr;
    }

    params->batch = batch;
    params->seq_len = seq_len;
    params->dim = dim;
    params->dtype = dtype;
    params->positions_dtype = positions_dtype;
    params->rope_type = rope_type;

    snprintf(
        params->name,
        sizeof(params->name),
        "rope_%s_f32_%s_%ux%ux%u",
        rope_type_to_str(rope_type),
        positions_dtype_to_str(positions_dtype),
        batch,
        seq_len,
        dim
    );

    const uint64_t token_count = (uint64_t)batch * seq_len;
    const uint64_t pair_count = dim / 2;
    const uint64_t flops = token_count * pair_count * 6ULL;
    const uint64_t input_bytes = token_count * dim * 4ULL;
    const uint64_t output_bytes = input_bytes;
    const uint64_t positions_bytes =
        token_count * (positions_dtype == MARMOT_DTYPE_INT32 ? sizeof(int32_t) : sizeof(float));

    w->desc.name = params->name;
    w->desc.category = MARMOT_BENCH_CATEGORY_MICRO;
    w->desc.primary_dtype = dtype;
    w->desc.flops = flops;
    w->desc.bytes_read = input_bytes + positions_bytes;
    w->desc.bytes_written = output_bytes;

    memset(&w->desc.signature, 0, sizeof(w->desc.signature));
    w->desc.signature.op_id = MARMOT_OP_ROPE;
    w->desc.signature.profile_id = MARMOT_PROFILE_SCALAR;
    w->desc.signature.input_dtype = dtype;
    w->desc.signature.weight_dtype = dtype;
    w->desc.signature.output_dtype = dtype;
    w->desc.signature.accum_dtype = MARMOT_DTYPE_FLOAT32;
    w->desc.signature.dims.rope.seq_len = seq_len;
    w->desc.signature.dims.rope.n_rot = dim;

    w->user_data = params;
    w->setup = rope_setup;
    w->execute = rope_type == MARMOT_ROPE_TYPE_NEOX ? rope_execute_neox : rope_execute_norm;
    w->teardown = rope_teardown;

    return w;
}

void marmot_bench_register_rope_workloads(marmot_bench_suite_t *suite) {
    const uint32_t batch = 1;
    const uint32_t seq_len = 4096;
    const uint32_t dim = 128;

    marmot_bench_workload_t *w_norm_cache = create_rope_workload(
        batch, seq_len, dim, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_INT32, MARMOT_ROPE_TYPE_NORM
    );
    if (w_norm_cache)
        marmot_bench_suite_add(suite, w_norm_cache);

    marmot_bench_workload_t *w_norm_no_cache = create_rope_workload(
        batch, seq_len, dim, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT32, MARMOT_ROPE_TYPE_NORM
    );
    if (w_norm_no_cache)
        marmot_bench_suite_add(suite, w_norm_no_cache);

    marmot_bench_workload_t *w_neox_cache = create_rope_workload(
        batch, seq_len, dim, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_INT32, MARMOT_ROPE_TYPE_NEOX
    );
    if (w_neox_cache)
        marmot_bench_suite_add(suite, w_neox_cache);

    marmot_bench_workload_t *w_neox_no_cache = create_rope_workload(
        batch, seq_len, dim, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT32, MARMOT_ROPE_TYPE_NEOX
    );
    if (w_neox_no_cache)
        marmot_bench_suite_add(suite, w_neox_no_cache);
}
