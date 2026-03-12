/* clang-format off */
#include "marmot/marmot.h"
#include "core/dispatch/fusion_flags.h"

#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include <cmocka.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
/* clang-format on */

#include "backend/golden_data.h"
#include "backend/golden_float_ops_llama.h"
#include "graph/golden_graph_numpy.h"

static void init_tensor_desc(marmot_graph_tensor_desc_t *desc, size_t dim0, size_t dim1, marmot_dtype_t dtype) {
    memset(desc, 0, sizeof(*desc));
    desc->dtype = dtype;
    desc->ndim = 2;
    desc->shape[0] = dim0;
    desc->shape[1] = dim1;
    desc->strides[1] = 1;
    desc->strides[0] = desc->shape[1];
}

static void fill_tensor_f32(marmot_tensor_t *tensor, const float *values, size_t count) {
    assert_non_null(tensor);
    assert_int_equal(tensor->dtype, MARMOT_DTYPE_FLOAT32);
    assert_true(tensor->capacity_bytes >= count * sizeof(float));
    memcpy(tensor->data, values, count * sizeof(float));
}

static void init_tensor_desc_1d(marmot_graph_tensor_desc_t *desc, size_t dim0, marmot_dtype_t dtype) {
    memset(desc, 0, sizeof(*desc));
    desc->dtype = dtype;
    desc->ndim = 1;
    desc->shape[0] = dim0;
    desc->strides[0] = 1;
}

static marmot_op_signature_t make_qkv_signature(uint32_t epilogue_flags) {
    marmot_op_signature_t sig = {0};
    sig.op_id = MARMOT_OP_QKV_SHARED_INPUT;
    sig.profile_id = MARMOT_PROFILE_INVALID;
    sig.matmul_layout = MARMOT_MATMUL_LAYOUT_NT;
    sig.input_dtype = MARMOT_DTYPE_FLOAT32;
    sig.weight_dtype = MARMOT_DTYPE_FLOAT32;
    sig.output_dtype = MARMOT_DTYPE_FLOAT32;
    sig.accum_dtype = MARMOT_DTYPE_FLOAT32;
    sig.qscheme_id = MARMOT_QSCHEME_NONE;
    sig.weight_layout = MARMOT_WEIGHT_LAYOUT_SEPARATE;
    sig.epilogue_flags = epilogue_flags;
    sig.activation = MARMOT_DEVICE_UNARY_IDENTITY;
    sig.variant_flags = MARMOT_FUSION_NONE;
    return sig;
}

static void split_qkv_weights(const float *fused, size_t m, size_t k, float *wq, float *wk, float *wv) {
    const size_t block = m * k;
    memcpy(wq, fused, block * sizeof(float));
    memcpy(wk, fused + block, block * sizeof(float));
    memcpy(wv, fused + 2 * block, block * sizeof(float));
}

static void split_qkv_bias(const float *fused, size_t m, float *bq, float *bk, float *bv) {
    memcpy(bq, fused, m * sizeof(float));
    memcpy(bk, fused + m, m * sizeof(float));
    memcpy(bv, fused + 2 * m, m * sizeof(float));
}

static float test_moe_silu(float x) {
    return x / (1.0f + expf(-x));
}

static void matmul_row_vector(const float *x, size_t in_dim, const float *weight, size_t out_dim, float *out) {
    for (size_t row = 0; row < out_dim; ++row) {
        float acc = 0.0f;
        for (size_t col = 0; col < in_dim; ++col) {
            acc += x[col] * weight[row * in_dim + col];
        }
        out[row] = acc;
    }
}

static void test_graph_build_and_finalize(void **state) {
    (void)state;

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t input_desc;
    init_tensor_desc(&input_desc, 4, 8, MARMOT_DTYPE_FLOAT16);

    marmot_value_id_t input_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &input_id), MARMOT_SUCCESS);

    marmot_graph_tensor_desc_t weight_desc;
    init_tensor_desc(&weight_desc, 16, 8, MARMOT_DTYPE_FLOAT16);

    marmot_value_id_t weight_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &weight_desc, &weight_id), MARMOT_SUCCESS);

    marmot_graph_tensor_desc_t output_desc;
    init_tensor_desc(&output_desc, 4, 16, MARMOT_DTYPE_FLOAT16);

    marmot_value_id_t output_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t matmul_inputs[2] = {input_id, weight_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "matmul", nullptr, matmul_inputs, 2, &output_desc, 1, &output_id), MARMOT_SUCCESS
    );

    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);

    marmot_graph_destroy(graph);
}

static void test_graph_rejects_unknown_op(void **state) {
    (void)state;

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t input_desc;
    init_tensor_desc(&input_desc, 2, 2, MARMOT_DTYPE_FLOAT16);

    marmot_value_id_t input_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &input_id), MARMOT_SUCCESS);

    marmot_graph_tensor_desc_t weight_desc;
    init_tensor_desc(&weight_desc, 2, 2, MARMOT_DTYPE_FLOAT16);

    marmot_value_id_t weight_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &weight_desc, &weight_id), MARMOT_SUCCESS);

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_INVALID,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = MARMOT_DTYPE_FLOAT16,
        .weight_dtype = MARMOT_DTYPE_FLOAT16,
        .output_dtype = MARMOT_DTYPE_FLOAT16,
        .accum_dtype = MARMOT_DTYPE_FLOAT32,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_SEPARATE,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_IDENTITY,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    marmot_graph_tensor_desc_t output_desc;
    init_tensor_desc(&output_desc, 2, 2, MARMOT_DTYPE_FLOAT16);

    marmot_value_id_t output_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t op_inputs[2] = {input_id, weight_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "unknown_op", &sig, op_inputs, 2, &output_desc, 1, &output_id),
        MARMOT_ERROR_INVALID_ARGUMENT
    );

    marmot_graph_destroy(graph);
}

static void test_graph_finalize_prevents_mutation(void **state) {
    (void)state;

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t input_desc;
    init_tensor_desc(&input_desc, 4, 8, MARMOT_DTYPE_FLOAT16);

    marmot_value_id_t input_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &input_id), MARMOT_SUCCESS);

    marmot_graph_tensor_desc_t weight_desc;
    init_tensor_desc(&weight_desc, 16, 8, MARMOT_DTYPE_FLOAT16);
    marmot_value_id_t weight_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &weight_desc, &weight_id), MARMOT_SUCCESS);

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_INVALID,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = MARMOT_DTYPE_FLOAT16,
        .weight_dtype = MARMOT_DTYPE_FLOAT16,
        .output_dtype = MARMOT_DTYPE_FLOAT16,
        .accum_dtype = MARMOT_DTYPE_FLOAT32,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_SEPARATE,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_IDENTITY,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    marmot_graph_tensor_desc_t output_desc;
    init_tensor_desc(&output_desc, 4, 16, MARMOT_DTYPE_FLOAT16);

    marmot_value_id_t output_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t op_inputs[2] = {input_id, weight_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "matmul", &sig, op_inputs, 2, &output_desc, 1, &output_id), MARMOT_SUCCESS
    );

    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);

    marmot_graph_tensor_desc_t extra_desc;
    init_tensor_desc(&extra_desc, 1, 1, MARMOT_DTYPE_FLOAT16);
    marmot_value_id_t extra_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &extra_desc, &extra_id), MARMOT_ERROR_INVALID_OPERATION);
    assert_int_equal(
        marmot_graph_add_op(graph, "matmul", &sig, op_inputs, 2, &output_desc, 1, &extra_id),
        MARMOT_ERROR_INVALID_OPERATION
    );

    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_ERROR_INVALID_OPERATION);

    marmot_graph_destroy(graph);
}

static void test_graph_finalize_requires_supported_kernel(void **state) {
    (void)state;

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t input_desc;
    init_tensor_desc(&input_desc, 2, 2, MARMOT_DTYPE_INT4);
    marmot_value_id_t input_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &input_id), MARMOT_SUCCESS);

    marmot_graph_tensor_desc_t weight_desc;
    init_tensor_desc(&weight_desc, 2, 2, MARMOT_DTYPE_INT4);
    marmot_value_id_t weight_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &weight_desc, &weight_id), MARMOT_SUCCESS);

    // Use TRANSPOSE with an unsupported dtype to force a missing kernel
    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_TRANSPOSE,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = MARMOT_DTYPE_INT4,
        .weight_dtype = MARMOT_DTYPE_INT4,
        .output_dtype = MARMOT_DTYPE_INT4,
        .accum_dtype = MARMOT_DTYPE_INT4,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_SEPARATE,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_IDENTITY,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    marmot_graph_tensor_desc_t output_desc;
    init_tensor_desc(&output_desc, 2, 2, MARMOT_DTYPE_INT4);
    marmot_value_id_t output_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t op_inputs[2] = {input_id, weight_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "transpose", &sig, op_inputs, 2, &output_desc, 1, &output_id), MARMOT_SUCCESS
    );

    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_ERROR_NOT_IMPLEMENTED);

    marmot_graph_destroy(graph);
}

static void test_graph_finalize_requires_rank2_inputs(void **state) {
    (void)state;

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t input_desc;
    memset(&input_desc, 0, sizeof(input_desc));
    input_desc.dtype = MARMOT_DTYPE_FLOAT16;
    input_desc.ndim = 1;
    input_desc.shape[0] = 8;

    marmot_value_id_t input_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &input_id), MARMOT_SUCCESS);

    marmot_graph_tensor_desc_t weight_desc;
    init_tensor_desc(&weight_desc, 8, 8, MARMOT_DTYPE_FLOAT16);
    marmot_value_id_t weight_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &weight_desc, &weight_id), MARMOT_SUCCESS);

    marmot_graph_tensor_desc_t output_desc;
    init_tensor_desc(&output_desc, 1, 8, MARMOT_DTYPE_FLOAT16);

    marmot_value_id_t output_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t op_inputs[2] = {input_id, weight_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "matmul", nullptr, op_inputs, 2, &output_desc, 1, &output_id), MARMOT_SUCCESS
    );

    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_ERROR_INVALID_ARGUMENT);

    marmot_graph_destroy(graph);
}

static void test_graph_execute_matmul_dispatch(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    const graph_matmul_case_t *golden = &g_graph_matmul_single_case;
    marmot_graph_tensor_desc_t input_desc;
    init_tensor_desc(&input_desc, golden->rows, golden->k, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t weight_desc;
    init_tensor_desc(&weight_desc, golden->cols, golden->k, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t output_desc;
    init_tensor_desc(&output_desc, golden->rows, golden->cols, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t input_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t weight_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &input_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &weight_desc, &weight_id), MARMOT_SUCCESS);

    marmot_value_id_t output_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t op_inputs[2] = {input_id, weight_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "matmul", nullptr, op_inputs, 2, &output_desc, 1, &output_id), MARMOT_SUCCESS
    );

    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);

    size_t input_shape[2] = {golden->rows, golden->k};
    size_t weight_shape[2] = {golden->cols, golden->k};
    size_t output_shape[2] = {golden->rows, golden->cols};
    marmot_tensor_t *input_tensor = marmot_tensor_create(nullptr, input_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *weight_tensor = marmot_tensor_create(nullptr, weight_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output_tensor = marmot_tensor_create(nullptr, output_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(input_tensor);
    assert_non_null(weight_tensor);
    assert_non_null(output_tensor);

    fill_tensor_f32(input_tensor, golden->input, golden->rows * golden->k);
    fill_tensor_f32(weight_tensor, golden->weight, golden->cols * golden->k);

    const marmot_tensor_t *graph_inputs[] = {input_tensor, weight_tensor};
    marmot_tensor_t *graph_outputs[] = {output_tensor};

    assert_int_equal(marmot_graph_execute(graph, ctx, graph_inputs, 2, graph_outputs, 1), MARMOT_SUCCESS);

    const float *actual = (const float *)output_tensor->data;
    const size_t elems = golden->rows * golden->cols;
    for (size_t i = 0; i < elems; ++i) {
        assert_float_equal(actual[i], golden->expected[i], 1e-5f);
    }

    marmot_tensor_destroy(output_tensor);
    marmot_tensor_destroy(weight_tensor);
    marmot_tensor_destroy(input_tensor);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
}

static void test_graph_execute_gather_rows(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t input_desc;
    init_tensor_desc(&input_desc, 4, 3, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t indices_desc;
    init_tensor_desc_1d(&indices_desc, 2, MARMOT_DTYPE_UINT32);
    marmot_graph_tensor_desc_t output_desc;
    init_tensor_desc(&output_desc, 2, 3, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t input_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t indices_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &input_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &indices_desc, &indices_id), MARMOT_SUCCESS);

    marmot_value_id_t output_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t op_inputs[2] = {input_id, indices_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "gather_rows", nullptr, op_inputs, 2, &output_desc, 1, &output_id), MARMOT_SUCCESS
    );

    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);

    const size_t input_shape[2] = {4, 3};
    const size_t indices_shape[1] = {2};
    const size_t output_shape[2] = {2, 3};
    marmot_tensor_t *input_tensor = marmot_tensor_create(nullptr, input_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *indices_tensor = marmot_tensor_create(nullptr, indices_shape, 1, MARMOT_DTYPE_UINT32);
    marmot_tensor_t *output_tensor = marmot_tensor_create(nullptr, output_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(input_tensor);
    assert_non_null(indices_tensor);
    assert_non_null(output_tensor);

    const float input_values[12] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
    };
    fill_tensor_f32(input_tensor, input_values, 12);

    marmot_uint32_t idx_values[2];
    idx_values[0].value = 2;
    idx_values[1].value = 0;
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(ctx, indices_tensor, idx_values, sizeof(idx_values)), MARMOT_SUCCESS
    );

    const marmot_tensor_t *graph_inputs[] = {input_tensor, indices_tensor};
    marmot_tensor_t *graph_outputs[] = {output_tensor};
    assert_int_equal(marmot_graph_execute(graph, ctx, graph_inputs, 2, graph_outputs, 1), MARMOT_SUCCESS);

    const float expected[6] = {7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f};
    const float *actual = (const float *)output_tensor->data;
    for (size_t i = 0; i < 6; ++i) {
        assert_float_equal(actual[i], expected[i], 1e-6f);
    }

    marmot_tensor_destroy(output_tensor);
    marmot_tensor_destroy(indices_tensor);
    marmot_tensor_destroy(input_tensor);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
}

static void test_graph_execute_topk_dispatch(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t input_desc;
    init_tensor_desc(&input_desc, 2, 5, MARMOT_DTYPE_FLOAT32);

    marmot_graph_tensor_desc_t values_desc;
    init_tensor_desc(&values_desc, 2, 3, MARMOT_DTYPE_FLOAT32);

    marmot_graph_tensor_desc_t indices_desc;
    init_tensor_desc(&indices_desc, 2, 3, MARMOT_DTYPE_INT32);

    marmot_value_id_t input_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &input_id), MARMOT_SUCCESS);

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_TOPK,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = MARMOT_DTYPE_FLOAT32,
        .weight_dtype = MARMOT_DTYPE_INT32,
        .output_dtype = MARMOT_DTYPE_FLOAT32,
        .accum_dtype = MARMOT_DTYPE_FLOAT32,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .stride_mode = MARMOT_STRIDE_MODE_STRIDED,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    marmot_graph_tensor_desc_t output_descs[2] = {values_desc, indices_desc};
    marmot_value_id_t output_ids[2] = {MARMOT_VALUE_ID_INVALID, MARMOT_VALUE_ID_INVALID};
    assert_int_equal(
        marmot_graph_add_op(graph, "topk", &sig, &input_id, 1, output_descs, 2, output_ids), MARMOT_SUCCESS
    );
    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);

    const size_t input_shape[2] = {2, 5};
    const size_t output_shape[2] = {2, 3};
    const float input_data[] = {
        1.0f, 4.0f, 4.0f, 2.0f, -1.0f, 0.0f, -3.0f, 5.0f, 5.0f, 1.0f,
    };
    const float expected_values[] = {
        4.0f, 4.0f, 2.0f, 5.0f, 5.0f, 1.0f,
    };
    const int32_t expected_indices[] = {
        1, 2, 3, 2, 3, 4,
    };

    marmot_tensor_t *input_tensor = marmot_tensor_create(nullptr, input_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *values_tensor = marmot_tensor_create(nullptr, output_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *indices_tensor = marmot_tensor_create(nullptr, output_shape, 2, MARMOT_DTYPE_INT32);
    assert_non_null(input_tensor);
    assert_non_null(values_tensor);
    assert_non_null(indices_tensor);
    fill_tensor_f32(input_tensor, input_data, 10);

    const marmot_tensor_t *graph_inputs[] = {input_tensor};
    marmot_tensor_t *graph_outputs[] = {values_tensor, indices_tensor};
    assert_int_equal(marmot_graph_execute(graph, ctx, graph_inputs, 1, graph_outputs, 2), MARMOT_SUCCESS);

    const float *actual_values = (const float *)values_tensor->data;
    for (size_t i = 0; i < 6; ++i) {
        assert_float_equal(actual_values[i], expected_values[i], 1e-6f);
    }

    const marmot_int32_t *actual_indices = marmot_tensor_data_i32(ctx, indices_tensor);
    assert_non_null(actual_indices);
    for (size_t i = 0; i < 6; ++i) {
        assert_int_equal(actual_indices[i].value, expected_indices[i]);
    }

    marmot_tensor_destroy(indices_tensor);
    marmot_tensor_destroy(values_tensor);
    marmot_tensor_destroy(input_tensor);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
}

static void test_graph_execute_moe_experts_dispatch(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t hidden_desc;
    init_tensor_desc(&hidden_desc, 2, 2, MARMOT_DTYPE_FLOAT32);

    marmot_graph_tensor_desc_t expert_desc;
    memset(&expert_desc, 0, sizeof(expert_desc));
    expert_desc.dtype = MARMOT_DTYPE_FLOAT32;
    expert_desc.ndim = 3;
    expert_desc.shape[0] = 2;
    expert_desc.shape[1] = 2;
    expert_desc.shape[2] = 2;
    expert_desc.strides[2] = 1;
    expert_desc.strides[1] = expert_desc.shape[2];
    expert_desc.strides[0] = expert_desc.shape[1] * expert_desc.shape[2];

    marmot_graph_tensor_desc_t topk_ids_desc;
    init_tensor_desc(&topk_ids_desc, 2, 2, MARMOT_DTYPE_INT32);

    marmot_graph_tensor_desc_t topk_weights_desc;
    init_tensor_desc(&topk_weights_desc, 2, 2, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t hidden_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t gate_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t up_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t down_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t topk_ids_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t topk_weights_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &hidden_desc, &hidden_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &expert_desc, &gate_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &expert_desc, &up_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &expert_desc, &down_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &topk_ids_desc, &topk_ids_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &topk_weights_desc, &topk_weights_id), MARMOT_SUCCESS);

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_MOE_EXPERTS,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = MARMOT_DTYPE_FLOAT32,
        .weight_dtype = MARMOT_DTYPE_FLOAT32,
        .output_dtype = MARMOT_DTYPE_FLOAT32,
        .accum_dtype = MARMOT_DTYPE_FLOAT32,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .stride_mode = MARMOT_STRIDE_MODE_STRIDED,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    marmot_value_id_t op_inputs[6] = {hidden_id, gate_id, up_id, down_id, topk_ids_id, topk_weights_id};
    marmot_value_id_t output_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(
        marmot_graph_add_op(graph, "moe_experts", &sig, op_inputs, 6, &hidden_desc, 1, &output_id), MARMOT_SUCCESS
    );
    assert_int_equal(marmot_graph_set_last_node_moe_params(graph, MARMOT_FFN_SWIGLU, 1.5f), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);

    const size_t hidden_shape[2] = {2, 2};
    const size_t expert_shape[3] = {2, 2, 2};
    const size_t topk_shape[2] = {2, 2};
    const float hidden_data[] = {
        1.0f,
        -0.5f,
        0.25f,
        0.75f,
    };
    const float gate_expert0[] = {
        1.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    const float gate_expert1[] = {
        0.2f,
        -0.4f,
        0.7f,
        0.1f,
    };
    const float up_expert0[] = {
        0.5f,
        0.0f,
        0.0f,
        2.0f,
    };
    const float up_expert1[] = {
        1.2f,
        0.3f,
        -0.6f,
        0.8f,
    };
    const float down_expert0[] = {
        1.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    const float down_expert1[] = {
        0.6f,
        -0.2f,
        0.1f,
        0.9f,
    };
    const float gate_exps_storage[] = {
        1.0f, 0.0f, 0.0f, 1.0f, 0.2f, -0.4f, 0.7f, 0.1f,
    };
    const float up_exps_storage[] = {
        0.5f, 0.0f, 0.0f, 2.0f, 1.2f, 0.3f, -0.6f, 0.8f,
    };
    const float down_exps_storage[] = {
        1.0f, 0.0f, 0.0f, 1.0f, 0.6f, -0.2f, 0.1f, 0.9f,
    };
    const marmot_int32_t topk_ids_data[] = {
        MARMOT_I32(0),
        MARMOT_I32(1),
        MARMOT_I32(1),
        MARMOT_I32(0),
    };
    const float topk_weights_data[] = {
        0.75f,
        0.25f,
        1.0f,
        0.5f,
    };

    marmot_tensor_t *hidden_tensor = marmot_tensor_create(ctx, hidden_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *gate_tensor = marmot_tensor_create(ctx, expert_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *up_tensor = marmot_tensor_create(ctx, expert_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *down_tensor = marmot_tensor_create(ctx, expert_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *topk_ids_tensor = marmot_tensor_create(ctx, topk_shape, 2, MARMOT_DTYPE_INT32);
    marmot_tensor_t *topk_weights_tensor = marmot_tensor_create(ctx, topk_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output_tensor = marmot_tensor_create(ctx, hidden_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(hidden_tensor);
    assert_non_null(gate_tensor);
    assert_non_null(up_tensor);
    assert_non_null(down_tensor);
    assert_non_null(topk_ids_tensor);
    assert_non_null(topk_weights_tensor);
    assert_non_null(output_tensor);

    fill_tensor_f32(hidden_tensor, hidden_data, 4);
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(ctx, gate_tensor, gate_exps_storage, sizeof(gate_exps_storage)),
        MARMOT_SUCCESS
    );
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(ctx, up_tensor, up_exps_storage, sizeof(up_exps_storage)), MARMOT_SUCCESS
    );
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(ctx, down_tensor, down_exps_storage, sizeof(down_exps_storage)),
        MARMOT_SUCCESS
    );
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(ctx, topk_ids_tensor, topk_ids_data, sizeof(topk_ids_data)), MARMOT_SUCCESS
    );
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(ctx, topk_weights_tensor, topk_weights_data, sizeof(topk_weights_data)),
        MARMOT_SUCCESS
    );

    const marmot_tensor_t *graph_inputs[] = {
        hidden_tensor, gate_tensor, up_tensor, down_tensor, topk_ids_tensor, topk_weights_tensor,
    };
    marmot_tensor_t *graph_outputs[] = {output_tensor};
    assert_int_equal(marmot_graph_execute(graph, ctx, graph_inputs, 6, graph_outputs, 1), MARMOT_SUCCESS);

    float expected[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float *expert_gate[2] = {gate_expert0, gate_expert1};
    const float *expert_up[2] = {up_expert0, up_expert1};
    const float *expert_down[2] = {down_expert0, down_expert1};
    for (size_t token = 0; token < 2; ++token) {
        const float *x = hidden_data + token * 2;
        float *out_row = expected + token * 2;
        for (size_t slot = 0; slot < 2; ++slot) {
            const int32_t expert_idx = topk_ids_data[token * 2 + slot].value;
            float gate_vals[2];
            float up_vals[2];
            float fused[2];
            float down_vals[2];
            matmul_row_vector(x, 2, expert_gate[expert_idx], 2, gate_vals);
            matmul_row_vector(x, 2, expert_up[expert_idx], 2, up_vals);
            for (size_t i = 0; i < 2; ++i) {
                fused[i] = test_moe_silu(gate_vals[i]) * up_vals[i];
            }
            matmul_row_vector(fused, 2, expert_down[expert_idx], 2, down_vals);
            const float weight = topk_weights_data[token * 2 + slot] * 1.5f;
            for (size_t i = 0; i < 2; ++i) {
                out_row[i] += weight * down_vals[i];
            }
        }
    }

    const float *actual = (const float *)output_tensor->data;
    for (size_t i = 0; i < 4; ++i) {
        assert_float_equal(actual[i], expected[i], 1e-5f);
    }

    marmot_tensor_destroy(output_tensor);
    marmot_tensor_destroy(topk_weights_tensor);
    marmot_tensor_destroy(topk_ids_tensor);
    marmot_tensor_destroy(down_tensor);
    marmot_tensor_destroy(up_tensor);
    marmot_tensor_destroy(gate_tensor);
    marmot_tensor_destroy(hidden_tensor);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
}

static void test_graph_execute_qkv_dispatch(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    for (size_t case_idx = 0; case_idx < g_matmul_qkv_case_count; ++case_idx) {
        const typeof(g_matmul_qkv_cases[0]) *tc = &g_matmul_qkv_cases[case_idx];
        const size_t weight_elems = tc->m * tc->k;
        const size_t out_elems = tc->n * tc->m;

        float *wq_host = malloc(weight_elems * sizeof(float));
        float *wk_host = malloc(weight_elems * sizeof(float));
        float *wv_host = malloc(weight_elems * sizeof(float));
        assert_non_null(wq_host);
        assert_non_null(wk_host);
        assert_non_null(wv_host);
        split_qkv_weights(tc->weight, tc->m, tc->k, wq_host, wk_host, wv_host);

        float *bq_host = nullptr;
        float *bk_host = nullptr;
        float *bv_host = nullptr;
        if (tc->has_bias) {
            bq_host = malloc(tc->m * sizeof(float));
            bk_host = malloc(tc->m * sizeof(float));
            bv_host = malloc(tc->m * sizeof(float));
            assert_non_null(bq_host);
            assert_non_null(bk_host);
            assert_non_null(bv_host);
            split_qkv_bias(tc->bias, tc->m, bq_host, bk_host, bv_host);
        }

        marmot_graph_t *graph = marmot_graph_create();
        assert_non_null(graph);

        marmot_graph_tensor_desc_t input_desc;
        marmot_graph_tensor_desc_t weight_desc;
        marmot_graph_tensor_desc_t bias_desc;
        marmot_graph_tensor_desc_t output_desc;
        init_tensor_desc(&input_desc, tc->n, tc->k, MARMOT_DTYPE_FLOAT32);
        init_tensor_desc(&weight_desc, tc->m, tc->k, MARMOT_DTYPE_FLOAT32);
        if (tc->has_bias) {
            init_tensor_desc_1d(&bias_desc, tc->m, MARMOT_DTYPE_FLOAT32);
        }
        init_tensor_desc(&output_desc, tc->n, tc->m, MARMOT_DTYPE_FLOAT32);

        marmot_value_id_t input_id = MARMOT_VALUE_ID_INVALID;
        marmot_value_id_t wq_id = MARMOT_VALUE_ID_INVALID;
        marmot_value_id_t wk_id = MARMOT_VALUE_ID_INVALID;
        marmot_value_id_t wv_id = MARMOT_VALUE_ID_INVALID;
        marmot_value_id_t bq_id = MARMOT_VALUE_ID_INVALID;
        marmot_value_id_t bk_id = MARMOT_VALUE_ID_INVALID;
        marmot_value_id_t bv_id = MARMOT_VALUE_ID_INVALID;

        assert_int_equal(marmot_graph_add_input(graph, &input_desc, &input_id), MARMOT_SUCCESS);
        assert_int_equal(marmot_graph_add_input(graph, &weight_desc, &wq_id), MARMOT_SUCCESS);
        assert_int_equal(marmot_graph_add_input(graph, &weight_desc, &wk_id), MARMOT_SUCCESS);
        assert_int_equal(marmot_graph_add_input(graph, &weight_desc, &wv_id), MARMOT_SUCCESS);
        if (tc->has_bias) {
            assert_int_equal(marmot_graph_add_input(graph, &bias_desc, &bq_id), MARMOT_SUCCESS);
            assert_int_equal(marmot_graph_add_input(graph, &bias_desc, &bk_id), MARMOT_SUCCESS);
            assert_int_equal(marmot_graph_add_input(graph, &bias_desc, &bv_id), MARMOT_SUCCESS);
        }

        marmot_value_id_t op_inputs[7];
        size_t num_inputs = 0;
        op_inputs[num_inputs++] = input_id;
        op_inputs[num_inputs++] = wq_id;
        op_inputs[num_inputs++] = wk_id;
        op_inputs[num_inputs++] = wv_id;
        if (tc->has_bias) {
            op_inputs[num_inputs++] = bq_id;
            op_inputs[num_inputs++] = bk_id;
            op_inputs[num_inputs++] = bv_id;
        }

        marmot_op_signature_t sig = make_qkv_signature(tc->has_bias ? MARMOT_EPILOGUE_BIAS : MARMOT_EPILOGUE_NONE);
        marmot_graph_tensor_desc_t output_descs[3] = {output_desc, output_desc, output_desc};
        marmot_value_id_t output_ids[3] = {MARMOT_VALUE_ID_INVALID, MARMOT_VALUE_ID_INVALID, MARMOT_VALUE_ID_INVALID};

        assert_int_equal(
            marmot_graph_add_op(graph, "qkv", &sig, op_inputs, num_inputs, output_descs, 3, output_ids), MARMOT_SUCCESS
        );
        assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);

        size_t input_shape[2] = {tc->n, tc->k};
        size_t weight_shape[2] = {tc->m, tc->k};
        size_t output_shape[2] = {tc->n, tc->m};

        marmot_tensor_t *input_tensor = marmot_tensor_create(nullptr, input_shape, 2, MARMOT_DTYPE_FLOAT32);
        marmot_tensor_t *wq_tensor = marmot_tensor_create(nullptr, weight_shape, 2, MARMOT_DTYPE_FLOAT32);
        marmot_tensor_t *wk_tensor = marmot_tensor_create(nullptr, weight_shape, 2, MARMOT_DTYPE_FLOAT32);
        marmot_tensor_t *wv_tensor = marmot_tensor_create(nullptr, weight_shape, 2, MARMOT_DTYPE_FLOAT32);
        marmot_tensor_t *bq_tensor = nullptr;
        marmot_tensor_t *bk_tensor = nullptr;
        marmot_tensor_t *bv_tensor = nullptr;
        if (tc->has_bias) {
            bq_tensor = marmot_tensor_create(nullptr, (const size_t[]){tc->m}, 1, MARMOT_DTYPE_FLOAT32);
            bk_tensor = marmot_tensor_create(nullptr, (const size_t[]){tc->m}, 1, MARMOT_DTYPE_FLOAT32);
            bv_tensor = marmot_tensor_create(nullptr, (const size_t[]){tc->m}, 1, MARMOT_DTYPE_FLOAT32);
        }
        marmot_tensor_t *out_q = marmot_tensor_create(nullptr, output_shape, 2, MARMOT_DTYPE_FLOAT32);
        marmot_tensor_t *out_k = marmot_tensor_create(nullptr, output_shape, 2, MARMOT_DTYPE_FLOAT32);
        marmot_tensor_t *out_v = marmot_tensor_create(nullptr, output_shape, 2, MARMOT_DTYPE_FLOAT32);

        assert_non_null(input_tensor);
        assert_non_null(wq_tensor);
        assert_non_null(wk_tensor);
        assert_non_null(wv_tensor);
        assert_non_null(out_q);
        assert_non_null(out_k);
        assert_non_null(out_v);
        if (tc->has_bias) {
            assert_non_null(bq_tensor);
            assert_non_null(bk_tensor);
            assert_non_null(bv_tensor);
        }

        fill_tensor_f32(input_tensor, tc->input, tc->n * tc->k);
        fill_tensor_f32(wq_tensor, wq_host, weight_elems);
        fill_tensor_f32(wk_tensor, wk_host, weight_elems);
        fill_tensor_f32(wv_tensor, wv_host, weight_elems);
        if (tc->has_bias) {
            fill_tensor_f32(bq_tensor, bq_host, tc->m);
            fill_tensor_f32(bk_tensor, bk_host, tc->m);
            fill_tensor_f32(bv_tensor, bv_host, tc->m);
        }

        const marmot_tensor_t *graph_inputs[] = {input_tensor, wq_tensor, wk_tensor, wv_tensor,
                                                 bq_tensor,    bk_tensor, bv_tensor};
        marmot_tensor_t *graph_outputs[] = {out_q, out_k, out_v};
        const size_t input_count = tc->has_bias ? 7 : 4;

        marmot_error_t exec_err = marmot_graph_execute(graph, ctx, graph_inputs, input_count, graph_outputs, 3);
        if (exec_err != MARMOT_SUCCESS) {
            fail_msg(
                "QKV graph_execute failed for case %s: %d (%s)", tc->name, exec_err, marmot_get_last_error_detail()
            );
        }
        assert_int_equal(exec_err, MARMOT_SUCCESS);

        const float *actual_q = (const float *)out_q->data;
        const float *actual_k = (const float *)out_k->data;
        const float *actual_v = (const float *)out_v->data;

        for (size_t i = 0; i < out_elems; ++i) {
            assert_float_equal(actual_q[i], tc->expected_q[i], 1e-5f);
            assert_float_equal(actual_k[i], tc->expected_k[i], 1e-5f);
            assert_float_equal(actual_v[i], tc->expected_v[i], 1e-5f);
        }

        marmot_tensor_destroy(out_v);
        marmot_tensor_destroy(out_k);
        marmot_tensor_destroy(out_q);
        if (tc->has_bias) {
            marmot_tensor_destroy(bv_tensor);
            marmot_tensor_destroy(bk_tensor);
            marmot_tensor_destroy(bq_tensor);
        }
        marmot_tensor_destroy(wv_tensor);
        marmot_tensor_destroy(wk_tensor);
        marmot_tensor_destroy(wq_tensor);
        marmot_tensor_destroy(input_tensor);
        marmot_graph_destroy(graph);

        free(wv_host);
        free(wk_host);
        free(wq_host);
        free(bv_host);
        free(bk_host);
        free(bq_host);
    }

    marmot_destroy(ctx);
}

static void test_graph_execute_two_stage_matmul(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    const graph_matmul_chain_case_t *golden = &g_graph_matmul_chain_case;
    marmot_graph_tensor_desc_t input_desc;
    init_tensor_desc(&input_desc, golden->rows, golden->k1, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t weight1_desc;
    init_tensor_desc(&weight1_desc, golden->m1, golden->k1, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t weight2_desc;
    init_tensor_desc(&weight2_desc, golden->m2, golden->m1, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t mid_desc;
    init_tensor_desc(&mid_desc, golden->rows, golden->m1, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t output_desc;
    init_tensor_desc(&output_desc, golden->rows, golden->m2, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t input_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t weight1_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t weight2_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &input_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &weight1_desc, &weight1_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &weight2_desc, &weight2_id), MARMOT_SUCCESS);

    marmot_value_id_t mid_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t matmul1_inputs[2] = {input_id, weight1_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "matmul", nullptr, matmul1_inputs, 2, &mid_desc, 1, &mid_id), MARMOT_SUCCESS
    );

    marmot_value_id_t output_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t matmul2_inputs[2] = {mid_id, weight2_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "matmul", nullptr, matmul2_inputs, 2, &output_desc, 1, &output_id), MARMOT_SUCCESS
    );

    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);

    size_t input_shape[2] = {golden->rows, golden->k1};
    size_t weight1_shape[2] = {golden->m1, golden->k1};
    size_t weight2_shape[2] = {golden->m2, golden->m1};
    size_t output_shape[2] = {golden->rows, golden->m2};

    marmot_tensor_t *input_tensor = marmot_tensor_create(nullptr, input_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *weight1_tensor = marmot_tensor_create(nullptr, weight1_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *weight2_tensor = marmot_tensor_create(nullptr, weight2_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output_tensor = marmot_tensor_create(nullptr, output_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(input_tensor);
    assert_non_null(weight1_tensor);
    assert_non_null(weight2_tensor);
    assert_non_null(output_tensor);

    fill_tensor_f32(input_tensor, golden->input, golden->rows * golden->k1);
    fill_tensor_f32(weight1_tensor, golden->weight1, golden->m1 * golden->k1);
    fill_tensor_f32(weight2_tensor, golden->weight2, golden->m2 * golden->m1);

    const marmot_tensor_t *graph_inputs[] = {input_tensor, weight1_tensor, weight2_tensor};
    marmot_tensor_t *graph_outputs[] = {output_tensor};

    assert_int_equal(marmot_graph_execute(graph, ctx, graph_inputs, 3, graph_outputs, 1), MARMOT_SUCCESS);

    const float *actual = (const float *)output_tensor->data;
    const size_t elems = golden->rows * golden->m2;
    for (size_t i = 0; i < elems; ++i) {
        assert_float_equal(actual[i], golden->expected[i], 1e-4f);
    }

    marmot_tensor_destroy(output_tensor);
    marmot_tensor_destroy(weight2_tensor);
    marmot_tensor_destroy(weight1_tensor);
    marmot_tensor_destroy(input_tensor);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
}

static void test_graph_execute_llama_matmul_cases(void **state) {
    (void)state;

    for (size_t case_idx = 0; case_idx < g_llama_matmul_case_count; ++case_idx) {
        const llama_matmul_case_t *tc = &g_llama_matmul_cases[case_idx];
        marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
        assert_non_null(ctx);
        marmot_graph_t *graph = marmot_graph_create();
        assert_non_null(graph);

        marmot_graph_tensor_desc_t input_desc;
        init_tensor_desc(&input_desc, tc->n, tc->k, MARMOT_DTYPE_FLOAT32);
        marmot_graph_tensor_desc_t weight_desc;
        init_tensor_desc(&weight_desc, tc->m, tc->k, MARMOT_DTYPE_FLOAT32);
        marmot_graph_tensor_desc_t output_desc;
        init_tensor_desc(&output_desc, tc->n, tc->m, MARMOT_DTYPE_FLOAT32);

        marmot_value_id_t input_id = MARMOT_VALUE_ID_INVALID;
        marmot_value_id_t weight_id = MARMOT_VALUE_ID_INVALID;
        assert_int_equal(marmot_graph_add_input(graph, &input_desc, &input_id), MARMOT_SUCCESS);
        assert_int_equal(marmot_graph_add_input(graph, &weight_desc, &weight_id), MARMOT_SUCCESS);

        marmot_value_id_t output_id = MARMOT_VALUE_ID_INVALID;
        marmot_value_id_t op_inputs[2] = {input_id, weight_id};
        assert_int_equal(
            marmot_graph_add_op(graph, "matmul", nullptr, op_inputs, 2, &output_desc, 1, &output_id), MARMOT_SUCCESS
        );

        assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);

        size_t input_shape[2] = {tc->n, tc->k};
        size_t weight_shape[2] = {tc->m, tc->k};
        size_t output_shape[2] = {tc->n, tc->m};

        marmot_tensor_t *input_tensor = marmot_tensor_create(nullptr, input_shape, 2, MARMOT_DTYPE_FLOAT32);
        marmot_tensor_t *weight_tensor = marmot_tensor_create(nullptr, weight_shape, 2, MARMOT_DTYPE_FLOAT32);
        marmot_tensor_t *output_tensor = marmot_tensor_create(nullptr, output_shape, 2, MARMOT_DTYPE_FLOAT32);
        assert_non_null(input_tensor);
        assert_non_null(weight_tensor);
        assert_non_null(output_tensor);

        const size_t elems_input = tc->n * tc->k;
        const size_t elems_weight = tc->m * tc->k;
        const size_t elems_out = tc->n * tc->m;
        float *input_host = malloc(elems_input * sizeof(float));
        float *expected_host = malloc(elems_out * sizeof(float));
        assert_non_null(input_host);
        assert_non_null(expected_host);

        // Transpose RHS (K×N) -> input (N×K)
        for (size_t n = 0; n < tc->n; ++n) {
            for (size_t k = 0; k < tc->k; ++k) {
                input_host[n * tc->k + k] = tc->rhs[k * tc->n + n];
            }
        }
        // Transpose expected (M×N) -> (N×M)
        for (size_t n = 0; n < tc->n; ++n) {
            for (size_t m = 0; m < tc->m; ++m) {
                expected_host[n * tc->m + m] = tc->expected[m * tc->n + n];
            }
        }

        fill_tensor_f32(input_tensor, input_host, elems_input);
        fill_tensor_f32(weight_tensor, tc->weight, elems_weight);

        const marmot_tensor_t *graph_inputs[] = {input_tensor, weight_tensor};
        marmot_tensor_t *graph_outputs[] = {output_tensor};
        assert_int_equal(marmot_graph_execute(graph, ctx, graph_inputs, 2, graph_outputs, 1), MARMOT_SUCCESS);

        const float *actual = (const float *)output_tensor->data;
        for (size_t i = 0; i < elems_out; ++i) {
            assert_float_equal(actual[i], expected_host[i], 5e-4f);
        }

        free(expected_host);
        free(input_host);
        marmot_tensor_destroy(output_tensor);
        marmot_tensor_destroy(weight_tensor);
        marmot_tensor_destroy(input_tensor);
        marmot_graph_destroy(graph);
        marmot_destroy(ctx);
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_graph_build_and_finalize),
        cmocka_unit_test(test_graph_rejects_unknown_op),
        cmocka_unit_test(test_graph_finalize_prevents_mutation),
        cmocka_unit_test(test_graph_finalize_requires_supported_kernel),
        cmocka_unit_test(test_graph_finalize_requires_rank2_inputs),
        cmocka_unit_test(test_graph_execute_matmul_dispatch),
        cmocka_unit_test(test_graph_execute_gather_rows),
        cmocka_unit_test(test_graph_execute_topk_dispatch),
        cmocka_unit_test(test_graph_execute_moe_experts_dispatch),
        cmocka_unit_test(test_graph_execute_qkv_dispatch),
        cmocka_unit_test(test_graph_execute_two_stage_matmul),
        cmocka_unit_test(test_graph_execute_llama_matmul_cases),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
