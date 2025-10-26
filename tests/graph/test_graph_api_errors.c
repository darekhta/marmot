/* clang-format off */
#include "marmot/marmot.h"
#include "marmot/graph/gguf_loader.h"

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include <cmocka.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
/* clang-format on */

static void init_tensor_desc(marmot_graph_tensor_desc_t *desc, size_t dim0, size_t dim1, marmot_dtype_t dtype) {
    memset(desc, 0, sizeof(*desc));
    desc->dtype = dtype;
    desc->ndim = 2;
    desc->shape[0] = dim0;
    desc->shape[1] = dim1;
    desc->strides[1] = 1;
    desc->strides[0] = desc->shape[1];
}

static void test_add_input_null_guard(void **state) {
    (void)state;
    marmot_value_id_t id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(nullptr, nullptr, &id), MARMOT_ERROR_INVALID_ARGUMENT);
}

static void test_finalize_null_guard(void **state) {
    (void)state;
    assert_int_equal(marmot_graph_finalize(nullptr, MARMOT_BACKEND_CPU), MARMOT_ERROR_INVALID_ARGUMENT);
}

static void test_graph_get_backend_null_guard(void **state) {
    (void)state;
    assert_int_equal(marmot_graph_get_backend(nullptr), MARMOT_BACKEND_CPU);
}

static void test_finalize_auto_null_guard(void **state) {
    (void)state;
    marmot_backend_type_t backend = MARMOT_BACKEND_CPU;
    assert_int_equal(marmot_graph_finalize_auto(nullptr, &backend), MARMOT_ERROR_INVALID_ARGUMENT);
}

static void test_finalize_with_options_null_guard(void **state) {
    (void)state;
    marmot_graph_finalize_options_t opts;
    assert_int_equal(marmot_graph_finalize_options_init(&opts), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_finalize_with_options(nullptr, &opts, nullptr), MARMOT_ERROR_INVALID_ARGUMENT);
}

static marmot_graph_t *create_matmul_graph(void) {
    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t input_desc;
    marmot_graph_tensor_desc_t weight_desc;
    marmot_graph_tensor_desc_t output_desc;
    init_tensor_desc(&input_desc, 512, 512, MARMOT_DTYPE_FLOAT16);
    init_tensor_desc(&weight_desc, 512, 512, MARMOT_DTYPE_FLOAT16);
    init_tensor_desc(&output_desc, 512, 512, MARMOT_DTYPE_FLOAT16);

    marmot_value_id_t input_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t weight_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t output_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &input_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &weight_desc, &weight_id), MARMOT_SUCCESS);

    marmot_value_id_t op_inputs[2] = {input_id, weight_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "matmul", nullptr, op_inputs, 2, &output_desc, 1, &output_id), MARMOT_SUCCESS
    );

    return graph;
}

static void test_finalize_auto_prefers_gpu_backend(void **state) {
    (void)state;

    // Check if CPU backend works
    marmot_graph_t *cpu_graph = create_matmul_graph();
    bool cpu_ok = (marmot_graph_finalize(cpu_graph, MARMOT_BACKEND_CPU) == MARMOT_SUCCESS);
    marmot_graph_destroy(cpu_graph);

    // Check if Metal backend works
    bool metal_ok = false;
#if MARMOT_ENABLE_METAL
    marmot_graph_t *metal_graph = create_matmul_graph();
    metal_ok = (marmot_graph_finalize(metal_graph, MARMOT_BACKEND_METAL) == MARMOT_SUCCESS);
    marmot_graph_destroy(metal_graph);
#endif

    // Auto selection should prefer Metal over CPU when both work
    marmot_graph_t *auto_graph = create_matmul_graph();
    marmot_backend_type_t chosen_backend = MARMOT_BACKEND_CPU;
    assert_int_equal(marmot_graph_finalize_auto(auto_graph, &chosen_backend), MARMOT_SUCCESS);

    if (metal_ok) {
        // Metal is preferred when available
        assert_int_equal(chosen_backend, MARMOT_BACKEND_METAL);
    } else if (cpu_ok) {
        // Fallback to CPU if Metal unavailable
        assert_int_equal(chosen_backend, MARMOT_BACKEND_CPU);
    }

    marmot_graph_destroy(auto_graph);
}

static void test_finalize_with_options_always_cpu(void **state) {
    (void)state;

    marmot_graph_t *graph = create_matmul_graph();

    marmot_graph_finalize_options_t opts;
    assert_int_equal(marmot_graph_finalize_options_init(&opts), MARMOT_SUCCESS);
    opts.flags |= MARMOT_GRAPH_FINALIZE_FLAG_AUTO_BACKEND;
    opts.routing_policy = MARMOT_ROUTING_ALWAYS_CPU;

    marmot_backend_type_t chosen_backend = MARMOT_BACKEND_METAL;
    assert_int_equal(marmot_graph_finalize_with_options(graph, &opts, &chosen_backend), MARMOT_SUCCESS);
    assert_int_equal(chosen_backend, MARMOT_BACKEND_CPU);
    assert_int_equal(marmot_graph_get_backend(graph), MARMOT_BACKEND_CPU);

    marmot_graph_destroy(graph);
}

static void test_finalize_with_options_always_gpu(void **state) {
    (void)state;

    marmot_graph_t *graph = create_matmul_graph();

    marmot_graph_finalize_options_t opts;
    assert_int_equal(marmot_graph_finalize_options_init(&opts), MARMOT_SUCCESS);
    opts.flags |= MARMOT_GRAPH_FINALIZE_FLAG_AUTO_BACKEND;
    opts.routing_policy = MARMOT_ROUTING_ALWAYS_GPU;

    marmot_backend_type_t chosen_backend = MARMOT_BACKEND_CPU;
    marmot_error_t status = marmot_graph_finalize_with_options(graph, &opts, &chosen_backend);

#if MARMOT_ENABLE_METAL
    if (status == MARMOT_SUCCESS) {
        assert_int_equal(chosen_backend, MARMOT_BACKEND_METAL);
        assert_int_equal(marmot_graph_get_backend(graph), MARMOT_BACKEND_METAL);
    } else {
        assert_int_not_equal(status, MARMOT_SUCCESS);
    }
#else
    assert_int_equal(status, MARMOT_ERROR_DEVICE_NOT_AVAILABLE);
#endif

    marmot_graph_destroy(graph);
}

static void test_execute_count_mismatch(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t input_desc;
    marmot_graph_tensor_desc_t weight_desc;
    marmot_graph_tensor_desc_t output_desc;
    init_tensor_desc(&input_desc, 2, 4, MARMOT_DTYPE_FLOAT32);
    init_tensor_desc(&weight_desc, 3, 4, MARMOT_DTYPE_FLOAT32);
    init_tensor_desc(&output_desc, 2, 3, MARMOT_DTYPE_FLOAT32);

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

    size_t input_shape[2] = {2, 4};
    size_t weight_shape[2] = {3, 4};
    size_t output_shape[2] = {2, 3};
    marmot_tensor_t *input_tensor = marmot_tensor_create(nullptr, input_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *weight_tensor = marmot_tensor_create(nullptr, weight_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output_tensor = marmot_tensor_create(nullptr, output_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(input_tensor);
    assert_non_null(weight_tensor);
    assert_non_null(output_tensor);

    const marmot_tensor_t *inputs[] = {input_tensor, weight_tensor};
    /* Intentionally pass zero outputs to hit count mismatch */
    assert_int_equal(marmot_graph_execute(graph, ctx, inputs, 2, nullptr, 0), MARMOT_ERROR_INVALID_ARGUMENT);

    marmot_tensor_destroy(output_tensor);
    marmot_tensor_destroy(weight_tensor);
    marmot_tensor_destroy(input_tensor);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
}

static void test_gguf_invalid_file(void **state) {
    (void)state;

    char tmpl[] = "/tmp/marmot_gguf_invalidXXXXXX";
    int fd = mkstemp(tmpl);
    assert_true(fd >= 0);
    uint8_t zeros[16] = {0};
    assert_true(write(fd, zeros, sizeof(zeros)) == (ssize_t)sizeof(zeros));
    close(fd);

    marmot_gguf_t *gguf = marmot_gguf_load(tmpl);
    assert_null(gguf);
    assert_int_equal(marmot_get_last_error(), MARMOT_ERROR_INVALID_ARGUMENT);

    unlink(tmpl);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_add_input_null_guard),
        cmocka_unit_test(test_finalize_null_guard),
        cmocka_unit_test(test_graph_get_backend_null_guard),
        cmocka_unit_test(test_finalize_auto_null_guard),
        cmocka_unit_test(test_finalize_with_options_null_guard),
        cmocka_unit_test(test_finalize_auto_prefers_gpu_backend),
        cmocka_unit_test(test_finalize_with_options_always_cpu),
        cmocka_unit_test(test_finalize_with_options_always_gpu),
        cmocka_unit_test(test_execute_count_mismatch),
        cmocka_unit_test(test_gguf_invalid_file),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
