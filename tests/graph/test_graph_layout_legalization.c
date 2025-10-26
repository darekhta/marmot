/* clang-format off */
#include "marmot/marmot.h"
#include "core/dispatch/fusion_flags.h"

#include <math.h>
#include <stdio.h>
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

static void init_tensor_desc_1d(marmot_graph_tensor_desc_t *desc, size_t dim0, marmot_dtype_t dtype) {
    memset(desc, 0, sizeof(*desc));
    desc->dtype = dtype;
    desc->ndim = 1;
    desc->shape[0] = dim0;
    desc->strides[0] = 1;
}

static void init_row_strided_desc(
    marmot_graph_tensor_desc_t *desc, size_t dim0, size_t dim1, size_t row_stride, marmot_dtype_t dtype
) {
    init_tensor_desc(desc, dim0, dim1, dtype);
    desc->strides[0] = row_stride;
}

static size_t json_count_ops(const char *path, const char *op_name) {
    FILE *f = fopen(path, "rb");
    if (f == nullptr) {
        return 0;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len <= 0) {
        fclose(f);
        return 0;
    }
    char *buf = malloc((size_t)len);
    if (buf == nullptr) {
        fclose(f);
        return 0;
    }
    if (fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf);
        fclose(f);
        return 0;
    }
    fclose(f);

    yyjson_doc *doc = yyjson_read(buf, (size_t)len, 0);
    free(buf);
    if (doc == nullptr) {
        return 0;
    }
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *nodes = yyjson_obj_get(root, "nodes");
    if (nodes == nullptr || !yyjson_is_arr(nodes)) {
        yyjson_doc_free(doc);
        return 0;
    }
    size_t found = 0;
    size_t idx = 0;
    size_t max = 0;
    yyjson_val *node = nullptr;
    yyjson_arr_foreach(nodes, idx, max, node) {
        yyjson_val *op = yyjson_obj_get(node, "op");
        if (op == nullptr) {
            continue;
        }
        const char *node_op_name = yyjson_get_str(op);
        if (node_op_name != nullptr && strcmp(node_op_name, op_name) == 0) {
            found++;
        }
    }
    yyjson_doc_free(doc);
    return found;
}

static bool json_has_contiguous_op(const char *path) {
    return json_count_ops(path, "contiguous") > 0;
}

static void test_graph_legalizes_strided_softmax(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t input_desc;
    init_tensor_desc(&input_desc, 2, 4, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t input_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &input_id), MARMOT_SUCCESS);

    marmot_graph_tensor_desc_t view_desc;
    init_tensor_desc(&view_desc, 2, 2, MARMOT_DTYPE_FLOAT32);
    view_desc.strides[0] = 4;
    view_desc.strides[1] = 1;

    marmot_op_signature_t view_sig = {0};
    view_sig.op_id = MARMOT_OP_VIEW;
    marmot_value_id_t view_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t view_inputs[1] = {input_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "view", &view_sig, view_inputs, 1, &view_desc, 1, &view_id), MARMOT_SUCCESS
    );

    marmot_graph_tensor_desc_t output_desc;
    init_tensor_desc(&output_desc, 2, 2, MARMOT_DTYPE_FLOAT32);

    marmot_op_signature_t softmax_sig = {0};
    softmax_sig.op_id = MARMOT_OP_SOFTMAX;
    marmot_value_id_t output_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t softmax_inputs[1] = {view_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "softmax", &softmax_sig, softmax_inputs, 1, &output_desc, 1, &output_id),
        MARMOT_SUCCESS
    );

    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);

    char path[] = "/tmp/marmot_graph_legalizeXXXXXX";
    int fd = mkstemp(path);
    assert_true(fd >= 0);
    close(fd);
    assert_int_equal(marmot_graph_dump_json(graph, path), MARMOT_SUCCESS);
    assert_true(json_has_contiguous_op(path));
    unlink(path);

    marmot_tensor_t *input = marmot_tensor_create(ctx, (size_t[]){2, 4}, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output = marmot_tensor_create(ctx, (size_t[]){2, 2}, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(input);
    assert_non_null(output);

    float input_vals[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    memcpy(input->data, input_vals, sizeof(input_vals));

    const marmot_tensor_t *inputs[1] = {input};
    marmot_tensor_t *outputs[1] = {output};
    marmot_error_t exec_err = marmot_graph_execute(graph, ctx, inputs, 1, outputs, 1);
    if (exec_err != MARMOT_SUCCESS) {
        fail_msg("graph_execute failed: %s", marmot_get_last_error_detail());
    }

    const float *out_data = marmot_tensor_data_f32(ctx, output);
    assert_non_null(out_data);

    const float exp1 = expf(1.0f);
    const float exp2 = expf(2.0f);
    const float row0_sum = exp1 + exp2;
    const float row0_0 = exp1 / row0_sum;
    const float row0_1 = exp2 / row0_sum;

    const float exp5 = expf(5.0f);
    const float exp6 = expf(6.0f);
    const float row1_sum = exp5 + exp6;
    const float row1_0 = exp5 / row1_sum;
    const float row1_1 = exp6 / row1_sum;

    assert_true(fabsf(out_data[0] - row0_0) < 1e-4f);
    assert_true(fabsf(out_data[1] - row0_1) < 1e-4f);
    assert_true(fabsf(out_data[2] - row1_0) < 1e-4f);
    assert_true(fabsf(out_data[3] - row1_1) < 1e-4f);

    marmot_tensor_destroy(output);
    marmot_tensor_destroy(input);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
}

static void test_graph_keeps_row_strided_add(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t a_desc;
    marmot_graph_tensor_desc_t b_desc;
    marmot_graph_tensor_desc_t out_desc;
    init_row_strided_desc(&a_desc, 2, 3, 4, MARMOT_DTYPE_FLOAT32);
    init_row_strided_desc(&b_desc, 2, 3, 4, MARMOT_DTYPE_FLOAT32);
    init_row_strided_desc(&out_desc, 2, 3, 4, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t a_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t b_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &a_desc, &a_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &b_desc, &b_id), MARMOT_SUCCESS);

    marmot_op_signature_t add_sig = {0};
    add_sig.op_id = MARMOT_OP_ADD;
    add_sig.profile_id = MARMOT_PROFILE_INVALID;
    add_sig.input_dtype = MARMOT_DTYPE_COUNT;
    add_sig.weight_dtype = MARMOT_DTYPE_COUNT;
    add_sig.output_dtype = MARMOT_DTYPE_COUNT;
    add_sig.accum_dtype = MARMOT_DTYPE_COUNT;
    add_sig.qscheme_id = MARMOT_QSCHEME_NONE;
    add_sig.weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID;
    add_sig.epilogue_flags = MARMOT_EPILOGUE_NONE;
    add_sig.activation = MARMOT_DEVICE_UNARY_IDENTITY;
    add_sig.variant_flags = MARMOT_FUSION_NONE;

    marmot_value_id_t out_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t add_inputs[2] = {a_id, b_id};
    assert_int_equal(marmot_graph_add_op(graph, "add", &add_sig, add_inputs, 2, &out_desc, 1, &out_id), MARMOT_SUCCESS);

    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);

    char path[] = "/tmp/marmot_graph_row_stridedXXXXXX";
    int fd = mkstemp(path);
    assert_true(fd >= 0);
    close(fd);
    assert_int_equal(marmot_graph_dump_json(graph, path), MARMOT_SUCCESS);
    assert_false(json_has_contiguous_op(path));
    unlink(path);

    marmot_tensor_t *a_base = marmot_tensor_create(ctx, (size_t[]){2, 4}, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *b_base = marmot_tensor_create(ctx, (size_t[]){2, 4}, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_base = marmot_tensor_create(ctx, (size_t[]){2, 4}, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *a = marmot_tensor_create(ctx, (size_t[]){2, 3}, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *b = marmot_tensor_create(ctx, (size_t[]){2, 3}, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(ctx, (size_t[]){2, 3}, 2, MARMOT_DTYPE_FLOAT32);

    assert_non_null(a_base);
    assert_non_null(b_base);
    assert_non_null(out_base);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(out);

    assert_int_equal(marmot_view(ctx, a_base, a, 0), MARMOT_SUCCESS);
    assert_int_equal(marmot_view(ctx, b_base, b, 0), MARMOT_SUCCESS);
    assert_int_equal(marmot_view(ctx, out_base, out, 0), MARMOT_SUCCESS);

    a->shape.strides[0] = a_base->shape.strides[0];
    a->shape.strides[1] = a_base->shape.strides[1];
    b->shape.strides[0] = b_base->shape.strides[0];
    b->shape.strides[1] = b_base->shape.strides[1];
    out->shape.strides[0] = out_base->shape.strides[0];
    out->shape.strides[1] = out_base->shape.strides[1];

    float a_vals[8] = {1.0f, 2.0f, 3.0f, 0.0f, 4.0f, 5.0f, 6.0f, 0.0f};
    float b_vals[8] = {10.0f, 20.0f, 30.0f, 0.0f, 40.0f, 50.0f, 60.0f, 0.0f};
    memcpy(a_base->data, a_vals, sizeof(a_vals));
    memcpy(b_base->data, b_vals, sizeof(b_vals));

    const marmot_tensor_t *inputs[2] = {a, b};
    marmot_tensor_t *outputs[1] = {out};
    marmot_error_t exec_err = marmot_graph_execute(graph, ctx, inputs, 2, outputs, 1);
    if (exec_err != MARMOT_SUCCESS) {
        fail_msg("graph_execute failed: %s", marmot_get_last_error_detail());
    }

    const float *out_data = marmot_tensor_data_f32(ctx, out);
    assert_non_null(out_data);
    assert_true(fabsf(out_data[0] - 11.0f) < 1e-4f);
    assert_true(fabsf(out_data[1] - 22.0f) < 1e-4f);
    assert_true(fabsf(out_data[2] - 33.0f) < 1e-4f);
    assert_true(fabsf(out_data[4] - 44.0f) < 1e-4f);
    assert_true(fabsf(out_data[5] - 55.0f) < 1e-4f);
    assert_true(fabsf(out_data[6] - 66.0f) < 1e-4f);

    marmot_tensor_destroy(out);
    marmot_tensor_destroy(b);
    marmot_tensor_destroy(a);
    marmot_tensor_destroy(out_base);
    marmot_tensor_destroy(b_base);
    marmot_tensor_destroy(a_base);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
}

static void test_graph_keeps_row_strided_gather_rows(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t input_desc;
    marmot_graph_tensor_desc_t indices_desc;
    marmot_graph_tensor_desc_t output_desc;
    init_row_strided_desc(&input_desc, 4, 3, 4, MARMOT_DTYPE_FLOAT32);
    init_tensor_desc_1d(&indices_desc, 2, MARMOT_DTYPE_UINT32);
    init_row_strided_desc(&output_desc, 2, 3, 4, MARMOT_DTYPE_FLOAT32);

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

    char path[] = "/tmp/marmot_graph_row_strided_gatherXXXXXX";
    int fd = mkstemp(path);
    assert_true(fd >= 0);
    close(fd);
    assert_int_equal(marmot_graph_dump_json(graph, path), MARMOT_SUCCESS);
    assert_int_equal(json_count_ops(path, "contiguous"), 0);
    unlink(path);

    marmot_tensor_t *input_base = marmot_tensor_create(ctx, (size_t[]){4, 4}, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output_base = marmot_tensor_create(ctx, (size_t[]){2, 4}, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *input = marmot_tensor_create(ctx, (size_t[]){4, 3}, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output = marmot_tensor_create(ctx, (size_t[]){2, 3}, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *indices = marmot_tensor_create(ctx, (size_t[]){2}, 1, MARMOT_DTYPE_UINT32);
    assert_non_null(input_base);
    assert_non_null(output_base);
    assert_non_null(input);
    assert_non_null(output);
    assert_non_null(indices);

    assert_int_equal(marmot_view(ctx, input_base, input, 0), MARMOT_SUCCESS);
    assert_int_equal(marmot_view(ctx, output_base, output, 0), MARMOT_SUCCESS);
    input->shape.strides[0] = input_base->shape.strides[0];
    input->shape.strides[1] = input_base->shape.strides[1];
    output->shape.strides[0] = output_base->shape.strides[0];
    output->shape.strides[1] = output_base->shape.strides[1];

    float in_vals[16] = {
        1.0f, 2.0f, 3.0f, 0.0f, 4.0f, 5.0f, 6.0f, 0.0f, 7.0f, 8.0f, 9.0f, 0.0f, 10.0f, 11.0f, 12.0f, 0.0f,
    };
    memcpy(input_base->data, in_vals, sizeof(in_vals));

    marmot_uint32_t idx_values[2];
    idx_values[0].value = 2;
    idx_values[1].value = 0;
    assert_int_equal(marmot_tensor_copy_from_host_buffer(ctx, indices, idx_values, sizeof(idx_values)), MARMOT_SUCCESS);

    const marmot_tensor_t *inputs[2] = {input, indices};
    marmot_tensor_t *outputs[1] = {output};
    marmot_error_t exec_err = marmot_graph_execute(graph, ctx, inputs, 2, outputs, 1);
    if (exec_err != MARMOT_SUCCESS) {
        fail_msg("graph_execute failed: %s", marmot_get_last_error_detail());
    }

    const float *out_data = marmot_tensor_data_f32(ctx, output);
    assert_non_null(out_data);
    assert_true(fabsf(out_data[0] - 7.0f) < 1e-4f);
    assert_true(fabsf(out_data[1] - 8.0f) < 1e-4f);
    assert_true(fabsf(out_data[2] - 9.0f) < 1e-4f);
    assert_true(fabsf(out_data[4] - 1.0f) < 1e-4f);
    assert_true(fabsf(out_data[5] - 2.0f) < 1e-4f);
    assert_true(fabsf(out_data[6] - 3.0f) < 1e-4f);

    marmot_tensor_destroy(indices);
    marmot_tensor_destroy(output);
    marmot_tensor_destroy(input);
    marmot_tensor_destroy(output_base);
    marmot_tensor_destroy(input_base);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
}

static void test_graph_keeps_row_strided_rope(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t input_desc;
    marmot_graph_tensor_desc_t positions_desc;
    marmot_graph_tensor_desc_t output_desc;
    init_row_strided_desc(&input_desc, 2, 4, 6, MARMOT_DTYPE_FLOAT32);
    init_tensor_desc_1d(&positions_desc, 2, MARMOT_DTYPE_FLOAT32);
    init_row_strided_desc(&output_desc, 2, 4, 6, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t input_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t positions_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &input_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &positions_desc, &positions_id), MARMOT_SUCCESS);

    marmot_op_signature_t rope_sig = {0};
    rope_sig.op_id = MARMOT_OP_ROPE;
    marmot_value_id_t output_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t op_inputs[2] = {input_id, positions_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "rope", &rope_sig, op_inputs, 2, &output_desc, 1, &output_id), MARMOT_SUCCESS
    );

    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);

    char path[] = "/tmp/marmot_graph_row_strided_ropeXXXXXX";
    int fd = mkstemp(path);
    assert_true(fd >= 0);
    close(fd);
    assert_int_equal(marmot_graph_dump_json(graph, path), MARMOT_SUCCESS);
    assert_int_equal(json_count_ops(path, "contiguous"), 0);
    unlink(path);

    marmot_tensor_t *input_base = marmot_tensor_create(ctx, (size_t[]){2, 6}, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output_base = marmot_tensor_create(ctx, (size_t[]){2, 6}, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *input = marmot_tensor_create(ctx, (size_t[]){2, 4}, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output = marmot_tensor_create(ctx, (size_t[]){2, 4}, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *positions = marmot_tensor_create(ctx, (size_t[]){2}, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(input_base);
    assert_non_null(output_base);
    assert_non_null(input);
    assert_non_null(output);
    assert_non_null(positions);

    assert_int_equal(marmot_view(ctx, input_base, input, 0), MARMOT_SUCCESS);
    assert_int_equal(marmot_view(ctx, output_base, output, 0), MARMOT_SUCCESS);
    input->shape.strides[0] = input_base->shape.strides[0];
    input->shape.strides[1] = input_base->shape.strides[1];
    output->shape.strides[0] = output_base->shape.strides[0];
    output->shape.strides[1] = output_base->shape.strides[1];

    float in_vals[12] = {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 5.0f, 6.0f, 7.0f, 8.0f, 0.0f, 0.0f};
    memcpy(input_base->data, in_vals, sizeof(in_vals));

    float pos_vals[2] = {0.0f, 0.0f};
    assert_int_equal(marmot_tensor_copy_from_host_buffer(ctx, positions, pos_vals, sizeof(pos_vals)), MARMOT_SUCCESS);

    const marmot_tensor_t *inputs[2] = {input, positions};
    marmot_tensor_t *outputs[1] = {output};
    marmot_error_t exec_err = marmot_graph_execute(graph, ctx, inputs, 2, outputs, 1);
    if (exec_err != MARMOT_SUCCESS) {
        fail_msg("graph_execute failed: %s", marmot_get_last_error_detail());
    }

    const float *out_data = marmot_tensor_data_f32(ctx, output);
    assert_non_null(out_data);
    assert_true(fabsf(out_data[0] - 1.0f) < 1e-4f);
    assert_true(fabsf(out_data[1] - 2.0f) < 1e-4f);
    assert_true(fabsf(out_data[2] - 3.0f) < 1e-4f);
    assert_true(fabsf(out_data[3] - 4.0f) < 1e-4f);
    assert_true(fabsf(out_data[6] - 5.0f) < 1e-4f);
    assert_true(fabsf(out_data[7] - 6.0f) < 1e-4f);
    assert_true(fabsf(out_data[8] - 7.0f) < 1e-4f);
    assert_true(fabsf(out_data[9] - 8.0f) < 1e-4f);

    marmot_tensor_destroy(positions);
    marmot_tensor_destroy(output);
    marmot_tensor_destroy(input);
    marmot_tensor_destroy(output_base);
    marmot_tensor_destroy(input_base);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
}

static void test_graph_reuses_contiguous_copy(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t input_desc;
    init_tensor_desc(&input_desc, 2, 4, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t input_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &input_id), MARMOT_SUCCESS);

    marmot_graph_tensor_desc_t view_desc;
    init_tensor_desc(&view_desc, 2, 2, MARMOT_DTYPE_FLOAT32);
    view_desc.strides[0] = 4;
    view_desc.strides[1] = 1;

    marmot_op_signature_t view_sig = {0};
    view_sig.op_id = MARMOT_OP_VIEW;
    marmot_value_id_t view_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t view_inputs[1] = {input_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "view", &view_sig, view_inputs, 1, &view_desc, 1, &view_id), MARMOT_SUCCESS
    );

    marmot_graph_tensor_desc_t output_desc;
    init_tensor_desc(&output_desc, 2, 2, MARMOT_DTYPE_FLOAT32);

    marmot_op_signature_t softmax_sig = {0};
    softmax_sig.op_id = MARMOT_OP_SOFTMAX;
    marmot_value_id_t softmax_inputs[1] = {view_id};
    marmot_value_id_t output_id_a = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t output_id_b = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(
        marmot_graph_add_op(graph, "softmax_a", &softmax_sig, softmax_inputs, 1, &output_desc, 1, &output_id_a),
        MARMOT_SUCCESS
    );
    assert_int_equal(
        marmot_graph_add_op(graph, "softmax_b", &softmax_sig, softmax_inputs, 1, &output_desc, 1, &output_id_b),
        MARMOT_SUCCESS
    );

    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);

    char path[] = "/tmp/marmot_graph_reuseXXXXXX";
    int fd = mkstemp(path);
    assert_true(fd >= 0);
    close(fd);
    assert_int_equal(marmot_graph_dump_json(graph, path), MARMOT_SUCCESS);
    assert_int_equal(json_count_ops(path, "contiguous"), 1);
    unlink(path);

    marmot_tensor_t *input = marmot_tensor_create(ctx, (size_t[]){2, 4}, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output_a = marmot_tensor_create(ctx, (size_t[]){2, 2}, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output_b = marmot_tensor_create(ctx, (size_t[]){2, 2}, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(input);
    assert_non_null(output_a);
    assert_non_null(output_b);

    float input_vals[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    memcpy(input->data, input_vals, sizeof(input_vals));

    const marmot_tensor_t *inputs[1] = {input};
    marmot_tensor_t *outputs[2] = {output_a, output_b};
    marmot_error_t exec_err = marmot_graph_execute(graph, ctx, inputs, 1, outputs, 2);
    if (exec_err != MARMOT_SUCCESS) {
        fail_msg("graph_execute failed: %s", marmot_get_last_error_detail());
    }

    const float *out_a = marmot_tensor_data_f32(ctx, output_a);
    const float *out_b = marmot_tensor_data_f32(ctx, output_b);
    assert_non_null(out_a);
    assert_non_null(out_b);

    const float exp1 = expf(1.0f);
    const float exp2 = expf(2.0f);
    const float row0_sum = exp1 + exp2;
    const float row0_0 = exp1 / row0_sum;
    const float row0_1 = exp2 / row0_sum;

    const float exp5 = expf(5.0f);
    const float exp6 = expf(6.0f);
    const float row1_sum = exp5 + exp6;
    const float row1_0 = exp5 / row1_sum;
    const float row1_1 = exp6 / row1_sum;

    assert_true(fabsf(out_a[0] - row0_0) < 1e-4f);
    assert_true(fabsf(out_a[1] - row0_1) < 1e-4f);
    assert_true(fabsf(out_a[2] - row1_0) < 1e-4f);
    assert_true(fabsf(out_a[3] - row1_1) < 1e-4f);

    assert_true(fabsf(out_b[0] - row0_0) < 1e-4f);
    assert_true(fabsf(out_b[1] - row0_1) < 1e-4f);
    assert_true(fabsf(out_b[2] - row1_0) < 1e-4f);
    assert_true(fabsf(out_b[3] - row1_1) < 1e-4f);

    marmot_tensor_destroy(output_b);
    marmot_tensor_destroy(output_a);
    marmot_tensor_destroy(input);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
}

static void test_graph_legalizes_output_strided_softmax(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t input_desc;
    init_tensor_desc(&input_desc, 2, 2, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t output_desc;
    init_row_strided_desc(&output_desc, 2, 2, 4, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t input_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &input_desc, &input_id), MARMOT_SUCCESS);

    marmot_op_signature_t softmax_sig = {0};
    softmax_sig.op_id = MARMOT_OP_SOFTMAX;
    marmot_value_id_t output_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t softmax_inputs[1] = {input_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "softmax", &softmax_sig, softmax_inputs, 1, &output_desc, 1, &output_id),
        MARMOT_SUCCESS
    );

    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);

    char path[] = "/tmp/marmot_graph_output_stridedXXXXXX";
    int fd = mkstemp(path);
    assert_true(fd >= 0);
    close(fd);
    assert_int_equal(marmot_graph_dump_json(graph, path), MARMOT_SUCCESS);
    assert_int_equal(json_count_ops(path, "contiguous"), 1);
    unlink(path);

    marmot_tensor_t *input = marmot_tensor_create(ctx, (size_t[]){2, 2}, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output_base = marmot_tensor_create(ctx, (size_t[]){2, 4}, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output = marmot_tensor_create(ctx, (size_t[]){2, 2}, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(input);
    assert_non_null(output_base);
    assert_non_null(output);

    assert_int_equal(marmot_view(ctx, output_base, output, 0), MARMOT_SUCCESS);
    output->shape.strides[0] = output_base->shape.strides[0];
    output->shape.strides[1] = output_base->shape.strides[1];

    float input_vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    memcpy(input->data, input_vals, sizeof(input_vals));

    const marmot_tensor_t *inputs[1] = {input};
    marmot_tensor_t *outputs[1] = {output};
    marmot_error_t exec_err = marmot_graph_execute(graph, ctx, inputs, 1, outputs, 1);
    if (exec_err != MARMOT_SUCCESS) {
        fail_msg("graph_execute failed: %s", marmot_get_last_error_detail());
    }

    const float *out_data = marmot_tensor_data_f32(ctx, output);
    assert_non_null(out_data);
    const float exp1 = expf(1.0f);
    const float exp2 = expf(2.0f);
    const float row0_sum = exp1 + exp2;
    const float row0_0 = exp1 / row0_sum;
    const float row0_1 = exp2 / row0_sum;

    const float exp3 = expf(3.0f);
    const float exp4 = expf(4.0f);
    const float row1_sum = exp3 + exp4;
    const float row1_0 = exp3 / row1_sum;
    const float row1_1 = exp4 / row1_sum;

    assert_true(fabsf(out_data[0] - row0_0) < 1e-4f);
    assert_true(fabsf(out_data[1] - row0_1) < 1e-4f);
    assert_true(fabsf(out_data[4] - row1_0) < 1e-4f);
    assert_true(fabsf(out_data[5] - row1_1) < 1e-4f);

    marmot_tensor_destroy(output);
    marmot_tensor_destroy(output_base);
    marmot_tensor_destroy(input);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_graph_legalizes_strided_softmax),
        cmocka_unit_test(test_graph_keeps_row_strided_add),
        cmocka_unit_test(test_graph_keeps_row_strided_gather_rows),
        cmocka_unit_test(test_graph_keeps_row_strided_rope),
        cmocka_unit_test(test_graph_reuses_contiguous_copy),
        cmocka_unit_test(test_graph_legalizes_output_strided_softmax),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
