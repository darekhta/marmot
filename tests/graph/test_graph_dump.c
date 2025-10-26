/* clang-format off */
#include "marmot/marmot.h"
#include "core/dispatch/fusion_flags.h"

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
#include "yyjson.h"
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

static void test_graph_dump_json_basic(void **state) {
    (void)state;

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t input_desc;
    marmot_graph_tensor_desc_t weight_desc;
    marmot_graph_tensor_desc_t output_desc;
    init_tensor_desc(&input_desc, 2, 4, MARMOT_DTYPE_FLOAT16);
    init_tensor_desc(&weight_desc, 3, 4, MARMOT_DTYPE_FLOAT16);
    init_tensor_desc(&output_desc, 2, 3, MARMOT_DTYPE_FLOAT16);

    marmot_value_id_t input_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t weight_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t output_id = MARMOT_VALUE_ID_INVALID;

    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &input_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &weight_desc, &weight_id), MARMOT_SUCCESS);

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_MATMUL,
        .profile_id = MARMOT_PROFILE_INVALID,
        .matmul_layout = MARMOT_MATMUL_LAYOUT_NT,
        .input_dtype = MARMOT_DTYPE_FLOAT16,
        .weight_dtype = MARMOT_DTYPE_FLOAT16,
        .output_dtype = MARMOT_DTYPE_FLOAT16,
        .accum_dtype = MARMOT_DTYPE_FLOAT32,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_IDENTITY,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    marmot_value_id_t matmul_inputs[2] = {input_id, weight_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "matmul", &sig, matmul_inputs, 2, &output_desc, 1, &output_id), MARMOT_SUCCESS
    );

    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);

    char path[] = "/tmp/marmot_graph_dumpXXXXXX";
    int fd = mkstemp(path);
    assert_true(fd >= 0);
    close(fd);

    assert_int_equal(marmot_graph_dump_json(graph, path), MARMOT_SUCCESS);

    FILE *f = fopen(path, "rb");
    assert_non_null(f);
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    assert_true(len > 0);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)len);
    assert_non_null(buf);
    assert_int_equal((size_t)len, fread(buf, 1, (size_t)len, f));
    fclose(f);

    yyjson_doc *doc = yyjson_read(buf, (size_t)len, 0);
    assert_non_null(doc);
    yyjson_val *root = yyjson_doc_get_root(doc);
    assert_non_null(root);

    yyjson_val *backend = yyjson_obj_get(root, "backend");
    assert_non_null(backend);
    assert_string_equal(yyjson_get_str(backend), "cpu");

    yyjson_val *nodes = yyjson_obj_get(root, "nodes");
    assert_non_null(nodes);
    assert_true(yyjson_is_arr(nodes));
    assert_int_equal(yyjson_arr_size(nodes), 1);

    yyjson_val *node0 = yyjson_arr_get(nodes, 0);
    assert_non_null(node0);
    // Kernel ID now includes platform (scalar/neon/accelerate) - just check it's valid
    assert_true(yyjson_get_uint(yyjson_obj_get(node0, "kernel_id")) > 0);
    yyjson_val *sig_obj = yyjson_obj_get(node0, "signature");
    assert_non_null(sig_obj);
    yyjson_val *dims = yyjson_obj_get(sig_obj, "dims");
    assert_non_null(dims);
    assert_int_equal(yyjson_get_uint(yyjson_obj_get(dims, "n")), 2);
    assert_int_equal(yyjson_get_uint(yyjson_obj_get(dims, "m")), 3);
    assert_int_equal(yyjson_get_uint(yyjson_obj_get(dims, "k")), 4);
    yyjson_val *activation = yyjson_obj_get(sig_obj, "activation");
    assert_non_null(activation);
    assert_string_equal(yyjson_get_str(activation), "identity");
    yyjson_val *epilogue = yyjson_obj_get(node0, "epilogue");
    assert_non_null(epilogue);
    yyjson_val *ep_flags = yyjson_obj_get(epilogue, "flags");
    assert_non_null(ep_flags);
    assert_true(yyjson_is_arr(ep_flags));
    assert_int_equal(yyjson_arr_size(ep_flags), 0);

    yyjson_val *values = yyjson_obj_get(root, "values");
    assert_non_null(values);
    assert_true(yyjson_is_arr(values));
    assert_int_equal(yyjson_arr_size(values), 3);
    yyjson_val *val0 = yyjson_arr_get(values, 0);
    assert_non_null(val0);
    yyjson_val *strides = yyjson_obj_get(val0, "strides");
    assert_non_null(strides);
    assert_true(yyjson_is_arr(strides));

    yyjson_val *tensors = yyjson_obj_get(root, "tensors");
    assert_non_null(tensors);
    assert_true(yyjson_is_arr(tensors));
    assert_int_equal(yyjson_arr_size(tensors), 3);

    yyjson_val *inputs = yyjson_obj_get(root, "inputs");
    yyjson_val *outputs = yyjson_obj_get(root, "outputs");
    assert_non_null(inputs);
    assert_non_null(outputs);
    assert_true(yyjson_is_arr(inputs));
    assert_true(yyjson_is_arr(outputs));
    assert_int_equal(yyjson_arr_size(inputs), 2);
    assert_int_equal(yyjson_arr_size(outputs), 1);

    yyjson_doc_free(doc);
    free(buf);
    unlink(path);

    marmot_graph_destroy(graph);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_graph_dump_json_basic),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
