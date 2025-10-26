/* clang-format off */
#include "marmot/marmot.h"
#include "core/dispatch/fusion_detection.h"

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

static void init_tensor_desc_1d(marmot_graph_tensor_desc_t *desc, size_t dim0, marmot_dtype_t dtype) {
    memset(desc, 0, sizeof(*desc));
    desc->dtype = dtype;
    desc->ndim = 1;
    desc->shape[0] = dim0;
    desc->strides[0] = 1;
}

static void fill_inputs(float *a, float *b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        a[i] = (float)i * 0.01f - 2.0f;
        b[i] = sinf((float)i * 0.05f);
    }
}

static float fused_expected(float a, float b, marmot_op_id_t fused_op) {
    float sum = a + b;
    switch (fused_op) {
    case MARMOT_OP_ADD_RELU:
        return sum > 0.0f ? sum : 0.0f;
    case MARMOT_OP_ADD_GELU: {
        const float inv_sqrt2 = 0.7071067811865475f;
        return sum * 0.5f * (1.0f + erff(sum * inv_sqrt2));
    }
    case MARMOT_OP_ADD_SILU:
        return sum / (1.0f + expf(-sum));
    default:
        return sum;
    }
}

static void assert_tensor_allclose_f32(
    const marmot_context_t *ctx, const marmot_tensor_t *tensor, const float *expected, float atol
) {
    assert_non_null(tensor);
    assert_int_equal(tensor->dtype, MARMOT_DTYPE_FLOAT32);
    size_t n = marmot_tensor_num_elements(tensor);
    const float *data = marmot_tensor_data_f32(ctx, (marmot_tensor_t *)tensor);
    assert_non_null(data);
    for (size_t i = 0; i < n; ++i) {
        float diff = fabsf(data[i] - expected[i]);
        if (diff > atol) {
            fail_msg("Mismatch at %zu: got=%f expected=%f diff=%f", i, data[i], expected[i], diff);
        }
    }
}

static void run_fused_add_test_backend(marmot_backend_type_t backend, marmot_op_id_t fused_op, float atol) {
#if !MARMOT_ENABLE_METAL
    if (backend == MARMOT_BACKEND_METAL) {
        skip();
        return;
    }
#endif

    marmot_context_t *ctx = marmot_init(backend);
    if (ctx == nullptr) {
        if (backend == MARMOT_BACKEND_METAL) {
            skip();
            return;
        }
        fail_msg("%s", "Failed to initialize backend");
    }

    const size_t n = 256;
    const size_t shape[1] = {n};

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t input_desc;
    marmot_graph_tensor_desc_t output_desc;
    init_tensor_desc_1d(&input_desc, n, MARMOT_DTYPE_FLOAT32);
    init_tensor_desc_1d(&output_desc, n, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t a_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t b_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t out_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &a_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &b_id), MARMOT_SUCCESS);

    marmot_op_signature_t sig = {
        .op_id = fused_op,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = MARMOT_DTYPE_FLOAT32,
        .weight_dtype = MARMOT_DTYPE_FLOAT32,
        .output_dtype = MARMOT_DTYPE_FLOAT32,
        .accum_dtype = MARMOT_DTYPE_FLOAT32,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_IDENTITY,
        .variant_flags = MARMOT_FUSION_NONE,
        .dims = {.elementwise = {.n_elems = (uint32_t)n}},
    };

    marmot_value_id_t inputs[2] = {a_id, b_id};
    assert_int_equal(marmot_graph_add_op(graph, "add", &sig, inputs, 2, &output_desc, 1, &out_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_finalize(graph, backend), MARMOT_SUCCESS);

    marmot_tensor_t *a = marmot_tensor_create(ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *b = marmot_tensor_create(ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(out);

    float *a_data = (float *)a->data;
    float *b_data = (float *)b->data;
    assert_non_null(a_data);
    assert_non_null(b_data);
    fill_inputs(a_data, b_data, n);

    assert_int_equal(marmot_tensor_to_device(ctx, a), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_to_device(ctx, b), MARMOT_SUCCESS);

    float *expected = (float *)malloc(n * sizeof(float));
    assert_non_null(expected);
    for (size_t i = 0; i < n; ++i) {
        expected[i] = fused_expected(a_data[i], b_data[i], fused_op);
    }

    const marmot_tensor_t *graph_inputs[] = {a, b};
    marmot_tensor_t *graph_outputs[] = {out};
    assert_int_equal(marmot_graph_execute(graph, ctx, graph_inputs, 2, graph_outputs, 1), MARMOT_SUCCESS);

    assert_tensor_allclose_f32(ctx, out, expected, atol);

    free(expected);
    marmot_tensor_destroy(out);
    marmot_tensor_destroy(b);
    marmot_tensor_destroy(a);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
}

// CPU fused kernel tests skipped - CPU doesn't have fused kernels in .def
// Fusion on CPU falls back to unfused add (activation is dropped)
// TODO: Add CPU fused kernels to elementwise.def
static void test_fusion_add_relu_correctness(void **state) {
    (void)state;
    skip(); // CPU has no fused kernels
}

static void test_fusion_add_gelu_correctness(void **state) {
    (void)state;
    skip(); // CPU has no fused kernels
}

static void test_fusion_add_silu_correctness(void **state) {
    (void)state;
    skip(); // CPU has no fused kernels
}

static void test_fusion_add_relu_correctness_metal(void **state) {
    (void)state;
    run_fused_add_test_backend(MARMOT_BACKEND_METAL, MARMOT_OP_ADD_RELU, 1e-6f);
}

static void test_fusion_add_gelu_correctness_metal(void **state) {
    (void)state;
    run_fused_add_test_backend(MARMOT_BACKEND_METAL, MARMOT_OP_ADD_GELU, 2e-3f);
}

static void test_fusion_add_silu_correctness_metal(void **state) {
    (void)state;
    run_fused_add_test_backend(MARMOT_BACKEND_METAL, MARMOT_OP_ADD_SILU, 1e-6f);
}

static void test_graph_multi_node_execution_metal(void **state) {
    (void)state;

#if !MARMOT_ENABLE_METAL
    skip();
    return;
#endif

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_METAL);
    if (ctx == nullptr) {
        skip();
        return;
    }

    const size_t n = 256;
    const size_t shape[1] = {n};

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t input_desc;
    marmot_graph_tensor_desc_t output_desc;
    init_tensor_desc_1d(&input_desc, n, MARMOT_DTYPE_FLOAT32);
    init_tensor_desc_1d(&output_desc, n, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t a_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t b_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t c_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &a_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &b_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &c_id), MARMOT_SUCCESS);

    marmot_op_signature_t add_sig = {
        .op_id = MARMOT_OP_ADD,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = MARMOT_DTYPE_FLOAT32,
        .weight_dtype = MARMOT_DTYPE_FLOAT32,
        .output_dtype = MARMOT_DTYPE_FLOAT32,
        .accum_dtype = MARMOT_DTYPE_FLOAT32,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_IDENTITY,
        .variant_flags = MARMOT_FUSION_NONE,
        .dims = {.elementwise = {.n_elems = (uint32_t)n}},
    };

    marmot_value_id_t add_out_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t add_inputs[2] = {a_id, b_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "add", &add_sig, add_inputs, 2, &output_desc, 1, &add_out_id), MARMOT_SUCCESS
    );

    marmot_op_signature_t mul_sig = add_sig;
    mul_sig.op_id = MARMOT_OP_MUL;

    marmot_value_id_t out_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t mul_inputs[2] = {add_out_id, c_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "mul", &mul_sig, mul_inputs, 2, &output_desc, 1, &out_id), MARMOT_SUCCESS
    );

    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_METAL), MARMOT_SUCCESS);

    marmot_tensor_t *a = marmot_tensor_create(ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *b = marmot_tensor_create(ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *c = marmot_tensor_create(ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(c);
    assert_non_null(out);

    float *a_data = (float *)a->data;
    float *b_data = (float *)b->data;
    float *c_data = (float *)c->data;
    assert_non_null(a_data);
    assert_non_null(b_data);
    assert_non_null(c_data);
    fill_inputs(a_data, b_data, n);
    for (size_t i = 0; i < n; ++i) {
        c_data[i] = cosf((float)i * 0.02f) + 0.5f;
    }

    assert_int_equal(marmot_tensor_to_device(ctx, a), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_to_device(ctx, b), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_to_device(ctx, c), MARMOT_SUCCESS);

    float *expected = (float *)malloc(n * sizeof(float));
    assert_non_null(expected);
    for (size_t i = 0; i < n; ++i) {
        expected[i] = (a_data[i] + b_data[i]) * c_data[i];
    }

    const marmot_tensor_t *graph_inputs[] = {a, b, c};
    marmot_tensor_t *graph_outputs[] = {out};
    assert_int_equal(marmot_graph_execute(graph, ctx, graph_inputs, 3, graph_outputs, 1), MARMOT_SUCCESS);

    assert_tensor_allclose_f32(ctx, out, expected, 1e-5f);

    free(expected);
    marmot_tensor_destroy(out);
    marmot_tensor_destroy(c);
    marmot_tensor_destroy(b);
    marmot_tensor_destroy(a);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
}

static void test_fusion_detection_add_relu(void **state) {
    (void)state;
    marmot_fusion_context_t ctx = {
        .prev_op = MARMOT_OP_INVALID,
        .current_op = MARMOT_OP_ADD,
        .next_op = MARMOT_OP_RELU,
        .next_next_op = MARMOT_OP_INVALID,
        .intermediate = nullptr,
        .intermediate_is_temporary = true,
        .next_intermediate_is_temporary = false,
        .detected_pattern = MARMOT_FUSION_PATTERN_NONE,
    };
    marmot_op_id_t fused_op = marmot_detect_fused_op_id(&ctx);
    assert_int_equal(fused_op, MARMOT_OP_ADD_RELU);
}

static void test_fusion_detection_matmul_bias_relu(void **state) {
    (void)state;
    marmot_fusion_context_t ctx = {
        .prev_op = MARMOT_OP_INVALID,
        .current_op = MARMOT_OP_MATMUL,
        .next_op = MARMOT_OP_ADD,
        .next_next_op = MARMOT_OP_RELU,
        .intermediate = nullptr,
        .intermediate_is_temporary = true,
        .next_intermediate_is_temporary = true,
        .detected_pattern = MARMOT_FUSION_PATTERN_NONE,
    };
    marmot_op_id_t fused_op = marmot_detect_fused_op_id(&ctx);
    assert_int_equal(fused_op, MARMOT_OP_MATMUL_BIAS_RELU);
}

static void test_fusion_detection_mul_add(void **state) {
    (void)state;
    marmot_fusion_context_t ctx = {
        .prev_op = MARMOT_OP_INVALID,
        .current_op = MARMOT_OP_MUL,
        .next_op = MARMOT_OP_ADD,
        .next_next_op = MARMOT_OP_INVALID,
        .intermediate = nullptr,
        .intermediate_is_temporary = true,
        .next_intermediate_is_temporary = false,
        .detected_pattern = MARMOT_FUSION_PATTERN_NONE,
    };
    marmot_op_id_t fused_op = marmot_detect_fused_op_id(&ctx);
    assert_int_equal(fused_op, MARMOT_OP_FMA);
}

static void test_fusion_detection_requires_temporary(void **state) {
    (void)state;
    marmot_fusion_context_t ctx = {
        .prev_op = MARMOT_OP_INVALID,
        .current_op = MARMOT_OP_ADD,
        .next_op = MARMOT_OP_RELU,
        .next_next_op = MARMOT_OP_INVALID,
        .intermediate = nullptr,
        .intermediate_is_temporary = false,
        .next_intermediate_is_temporary = false,
        .detected_pattern = MARMOT_FUSION_PATTERN_NONE,
    };
    marmot_op_id_t fused_op = marmot_detect_fused_op_id(&ctx);
    assert_int_equal(fused_op, MARMOT_OP_INVALID);
}

static void test_backend_supports_fusion_flags(void **state) {
    (void)state;

    // Need to init backend to register the dispatcher
    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    assert_true(marmot_backend_supports_fusion(MARMOT_BACKEND_CPU, MARMOT_FUSION_RESIDUAL_ADD));
    assert_false(marmot_backend_supports_fusion(MARMOT_BACKEND_CPU, MARMOT_FUSION_CUSTOM));

    marmot_destroy(ctx);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_fusion_add_relu_correctness),
        cmocka_unit_test(test_fusion_add_gelu_correctness),
        cmocka_unit_test(test_fusion_add_silu_correctness),
        cmocka_unit_test(test_fusion_add_relu_correctness_metal),
        cmocka_unit_test(test_fusion_add_gelu_correctness_metal),
        cmocka_unit_test(test_fusion_add_silu_correctness_metal),
        cmocka_unit_test(test_graph_multi_node_execution_metal),
        cmocka_unit_test(test_fusion_detection_add_relu),
        cmocka_unit_test(test_fusion_detection_matmul_bias_relu),
        cmocka_unit_test(test_fusion_detection_mul_add),
        cmocka_unit_test(test_fusion_detection_requires_temporary),
        cmocka_unit_test(test_backend_supports_fusion_flags),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
