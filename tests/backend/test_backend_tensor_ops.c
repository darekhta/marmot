#include "backend/golden_data.h"
#include "backend/test_backend_utils.h"

static const char *backend_name_for_test(marmot_backend_type_t backend) {
    switch (backend) {
    case MARMOT_BACKEND_CPU:
        return "cpu";
    case MARMOT_BACKEND_METAL:
        return "metal";
    case MARMOT_BACKEND_CUDA:
        return "cuda";
    default:
        return "unknown";
    }
}

static void fetch_tensor_as_f32(const marmot_test_env_t *env, const marmot_tensor_t *tensor, float *dst, size_t count) {
    assert_non_null(env);
    assert_non_null(tensor);
    assert_non_null(dst);
    const size_t bytes = count * sizeof(float);
    assert_int_equal(marmot_tensor_copy_to_host_buffer(env->ctx, tensor, dst, bytes), MARMOT_SUCCESS);
}

static void run_tensor_ops_checks(marmot_test_env_t *env) {
    assert_non_null(env);
    marmot_context_t *ctx = env->ctx;
    assert_non_null(ctx);
    assert_int_equal(marmot_context_get_backend(ctx), env->backend);
    const char *backend_name = marmot_context_get_backend_name(ctx);
    assert_non_null(backend_name);
    assert_string_equal(backend_name, backend_name_for_test(env->backend));
    assert_true(marmot_context_supports_dtype(ctx, MARMOT_DTYPE_FLOAT32));

    // ---------------------------------------------------------------------
    // Reshape success
    // ---------------------------------------------------------------------
    const size_t shape_2x3[] = {g_tensor_2d.shape_src[0], g_tensor_2d.shape_src[1]};
    const size_t shape_3x2[] = {g_tensor_2d.shape_dst[0], g_tensor_2d.shape_dst[1]};
    marmot_tensor_t *src = marmot_tensor_create(env->ctx, shape_2x3, 2, MARMOT_DTYPE_FLOAT32);
    marmot_test_convert_f32_span(env, src, g_tensor_2d.src, g_tensor_2d.shape_src[0] * g_tensor_2d.shape_src[1]);
    assert_int_equal(marmot_tensor_ndim(src), 2);
    assert_int_equal(marmot_tensor_shape_at(src, 0), shape_2x3[0]);
    assert_int_equal(marmot_tensor_shape_at(src, 1), shape_2x3[1]);
    assert_int_equal(marmot_tensor_stride_at(src, 0), shape_2x3[1]);
    assert_int_equal(marmot_tensor_stride_at(src, 1), 1);
    assert_int_equal(marmot_tensor_numel(src), shape_2x3[0] * shape_2x3[1]);
    marmot_tensor_t *reshaped = marmot_tensor_create(env->ctx, shape_3x2, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(reshaped);

    marmot_error_t err = marmot_reshape(ctx, src, reshaped, shape_3x2, 2);
    assert_int_equal(err, MARMOT_SUCCESS);
    float check_buf[24];
    fetch_tensor_as_f32(env, reshaped, check_buf, 6);
    marmot_test_expect_close_array(check_buf, g_tensor_2d.src, 6, 1e-6f);

    // ---------------------------------------------------------------------
    // Reshape mismatch should error
    // ---------------------------------------------------------------------
    const size_t bad_shape[] = {4, 2};
    marmot_tensor_t *bad_out = marmot_tensor_create(env->ctx, bad_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(bad_out);
    err = marmot_reshape(ctx, src, bad_out, bad_shape, 2);
    assert_int_equal(err, MARMOT_ERROR_DIMENSION_MISMATCH);

    // ---------------------------------------------------------------------
    // Transpose with default permutation (nullptr)
    // ---------------------------------------------------------------------
    marmot_tensor_t *transpose_default = marmot_tensor_create(env->ctx, shape_3x2, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(transpose_default);
    err = marmot_transpose(ctx, src, transpose_default, nullptr);
    assert_int_equal(err, MARMOT_SUCCESS);
    fetch_tensor_as_f32(env, transpose_default, check_buf, 6);
    marmot_test_expect_close_array(check_buf, g_tensor_2d.transpose_default, 6, 1e-6f);

    // Explicit identity permutation should copy
    int perm_identity[] = {0, 1};
    marmot_tensor_t *transpose_identity = marmot_tensor_create(env->ctx, shape_2x3, 2, MARMOT_DTYPE_FLOAT32);
    err = marmot_transpose(ctx, src, transpose_identity, perm_identity);
    assert_int_equal(err, MARMOT_SUCCESS);
    fetch_tensor_as_f32(env, transpose_identity, check_buf, 6);
    marmot_test_expect_close_array(check_buf, g_tensor_2d.transpose_identity, 6, 1e-6f);

    // Invalid permutation must fail
    int perm_invalid[] = {0, 2};
    marmot_tensor_t *transpose_invalid = marmot_tensor_create(env->ctx, shape_2x3, 2, MARMOT_DTYPE_FLOAT32);
    err = marmot_transpose(ctx, src, transpose_invalid, perm_invalid);
    assert_int_equal(err, MARMOT_ERROR_INVALID_ARGUMENT);

    // ---------------------------------------------------------------------
    // Concatenation in 1D
    // ---------------------------------------------------------------------
    const size_t vec_shape[] = {g_tensor_concat1d.length};
    marmot_tensor_t *left = marmot_test_tensor_from_array(env, vec_shape, 1, g_tensor_concat1d.left);
    marmot_tensor_t *right = marmot_test_tensor_from_array(env, vec_shape, 1, g_tensor_concat1d.right);
    const marmot_tensor_t *parts[] = {left, right};
    const size_t concat_shape[] = {g_tensor_concat1d.length * 2};
    marmot_tensor_t *concat_out = marmot_tensor_create(env->ctx, concat_shape, 1, MARMOT_DTYPE_FLOAT32);
    err = marmot_concat(ctx, parts, 2, concat_out, 0);
    assert_int_equal(err, MARMOT_SUCCESS);
    fetch_tensor_as_f32(env, concat_out, check_buf, concat_shape[0]);
    marmot_test_expect_close_array(check_buf, g_tensor_concat1d.concat_axis0, concat_shape[0], 1e-6f);

    // ---------------------------------------------------------------------
    // Slice 1D
    // ---------------------------------------------------------------------
    size_t starts[] = {2};
    size_t sizes[] = {g_tensor_concat1d.length - 2};
    marmot_tensor_t *slice_out = marmot_tensor_create(env->ctx, (size_t[]){sizes[0]}, 1, MARMOT_DTYPE_FLOAT32);
    err = marmot_slice(ctx, concat_out, slice_out, starts, sizes);
    assert_int_equal(err, MARMOT_SUCCESS);
    fetch_tensor_as_f32(env, slice_out, check_buf, sizes[0]);
    marmot_test_expect_close_array(check_buf, g_tensor_concat1d.slice_result, sizes[0], 1e-6f);

    // ---------------------------------------------------------------------
    // Gather rows
    // ---------------------------------------------------------------------
    const size_t gather_shape_in[] = {4, 3};
    const float gather_in[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
    };
    marmot_tensor_t *gather_src = marmot_tensor_create(env->ctx, gather_shape_in, 2, MARMOT_DTYPE_FLOAT32);
    marmot_test_convert_f32_span(env, gather_src, gather_in, 12);

    const size_t gather_idx_shape[] = {2};
    marmot_tensor_t *gather_indices = marmot_tensor_create(env->ctx, gather_idx_shape, 1, MARMOT_DTYPE_UINT32);
    marmot_uint32_t *idx_ptr = marmot_tensor_data_u32_mut(ctx, gather_indices);
    assert_non_null(idx_ptr);
    idx_ptr[0].value = 2;
    idx_ptr[1].value = 0;

    const size_t gather_out_shape[] = {2, 3};
    marmot_tensor_t *gather_out = marmot_tensor_create(env->ctx, gather_out_shape, 2, MARMOT_DTYPE_FLOAT32);
    err = marmot_gather_rows(ctx, gather_src, gather_indices, gather_out);
    assert_int_equal(err, MARMOT_SUCCESS);

    fetch_tensor_as_f32(env, gather_out, check_buf, 6);
    const float gather_expected[] = {7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f};
    marmot_test_expect_close_array(check_buf, gather_expected, 6, 1e-6f);

    // ---------------------------------------------------------------------
    // View with byte offset
    // ---------------------------------------------------------------------
    const size_t view_src_shape[] = {8};
    marmot_tensor_t *view_src = marmot_tensor_create(env->ctx, view_src_shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(view_src);
    const float view_src_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    marmot_test_convert_f32_span(env, view_src, view_src_data, 8);

    const size_t view_out_shape[] = {6};
    marmot_tensor_t *view_out = marmot_tensor_create(env->ctx, view_out_shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(view_out);
    const size_t view_offset = 2 * sizeof(float);
    err = marmot_view(ctx, view_src, view_out, view_offset);
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_ptr_equal(view_out->data, (uint8_t *)view_src->data + view_offset);
    assert_false(view_out->owns_data);
    fetch_tensor_as_f32(env, view_out, check_buf, view_out_shape[0]);
    marmot_test_expect_close_array(check_buf, view_src_data + 2, view_out_shape[0], 1e-6f);

    // ---------------------------------------------------------------------
    // Quantized view metadata aliasing
    // ---------------------------------------------------------------------
    const size_t quant_shape[] = {8};
    marmot_tensor_t *quant_src = marmot_tensor_create(env->ctx, quant_shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(quant_src);
    const float quant_src_data[] = {-1.0f, 0.0f, 1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f};
    marmot_test_convert_f32_span(env, quant_src, quant_src_data, quant_shape[0]);

    marmot_quant_params_t quant_params = {.scale = 0.5f, .zero_point = 0.0f, .block_size = 0};
    marmot_tensor_t *quant_src_q = marmot_tensor_create(env->ctx, quant_shape, 1, MARMOT_DTYPE_INT8);
    assert_non_null(quant_src_q);
    err = marmot_quantize(ctx, quant_src, &quant_params, quant_src_q);
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_non_null(quant_src_q->quant_params);

    const size_t quant_view_shape[] = {6};
    marmot_tensor_t *quant_view = marmot_tensor_create(env->ctx, quant_view_shape, 1, MARMOT_DTYPE_INT8);
    assert_non_null(quant_view);
    const size_t quant_view_offset = 2 * sizeof(marmot_int8_t);
    err = marmot_view(ctx, quant_src_q, quant_view, quant_view_offset);
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_ptr_equal(quant_view->data, (uint8_t *)quant_src_q->data + quant_view_offset);
    assert_false(quant_view->owns_data);
    assert_ptr_equal(quant_view->quant_params, quant_src_q->quant_params);
    assert_int_equal(quant_view->quant_kind, quant_src_q->quant_kind);
    assert_int_equal(quant_view->quant_layout, quant_src_q->quant_layout);

    // ---------------------------------------------------------------------
    // 3D transpose with custom permutation
    // ---------------------------------------------------------------------
    const size_t shape3[] = {
        g_tensor_transpose3d.shape_in[0], g_tensor_transpose3d.shape_in[1], g_tensor_transpose3d.shape_in[2]
    };
    const size_t out_shape3[] = {
        g_tensor_transpose3d.shape_in[g_tensor_transpose3d.perm[0]],
        g_tensor_transpose3d.shape_in[g_tensor_transpose3d.perm[1]],
        g_tensor_transpose3d.shape_in[g_tensor_transpose3d.perm[2]],
    };
    marmot_tensor_t *t3 = marmot_tensor_create(env->ctx, shape3, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *t3_out = marmot_tensor_create(env->ctx, out_shape3, 3, MARMOT_DTYPE_FLOAT32);
    marmot_test_convert_f32_span(env, t3, g_tensor_transpose3d.input, 24);
    int perm3[] = {g_tensor_transpose3d.perm[0], g_tensor_transpose3d.perm[1], g_tensor_transpose3d.perm[2]};
    err = marmot_transpose(ctx, t3, t3_out, perm3);
    assert_int_equal(err, MARMOT_SUCCESS);

    fetch_tensor_as_f32(env, t3_out, check_buf, 24);
    marmot_test_expect_close_array(check_buf, g_tensor_transpose3d.expected, 24, 1e-6f);

    // ---------------------------------------------------------------------
    // Concat along axis 1 for 3D tensors
    // ---------------------------------------------------------------------
    const size_t concat_shape_a[] = {
        g_tensor_concat3d.shape_a[0], g_tensor_concat3d.shape_a[1], g_tensor_concat3d.shape_a[2]
    };
    const size_t concat_shape_b[] = {
        g_tensor_concat3d.shape_b[0], g_tensor_concat3d.shape_b[1], g_tensor_concat3d.shape_b[2]
    };
    const size_t concat_out_shape3[] = {
        concat_shape_a[0],
        concat_shape_a[1] + concat_shape_b[1],
        concat_shape_a[2],
    };
    marmot_tensor_t *concat_a = marmot_tensor_create(env->ctx, concat_shape_a, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *concat_b = marmot_tensor_create(env->ctx, concat_shape_b, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *concat_out3 = marmot_tensor_create(env->ctx, concat_out_shape3, 3, MARMOT_DTYPE_FLOAT32);
    marmot_test_convert_f32_span(
        env, concat_a, g_tensor_concat3d.a, concat_shape_a[0] * concat_shape_a[1] * concat_shape_a[2]
    );
    marmot_test_convert_f32_span(
        env, concat_b, g_tensor_concat3d.b, concat_shape_b[0] * concat_shape_b[1] * concat_shape_b[2]
    );
    const marmot_tensor_t *concat_inputs3[] = {concat_a, concat_b};
    err = marmot_concat(ctx, concat_inputs3, 2, concat_out3, 1);
    assert_int_equal(err, MARMOT_SUCCESS);

    size_t concat_total = concat_out_shape3[0] * concat_out_shape3[1] * concat_out_shape3[2];
    fetch_tensor_as_f32(env, concat_out3, check_buf, concat_total);
    marmot_test_expect_close_array(check_buf, g_tensor_concat3d.expected, concat_total, 1e-6f);

    // ---------------------------------------------------------------------
    // 3D slice with offsets on multiple axes
    // ---------------------------------------------------------------------
    const size_t slice_src_shape[] = {
        g_tensor_slice3d.shape_src[0],
        g_tensor_slice3d.shape_src[1],
        g_tensor_slice3d.shape_src[2],
    };
    marmot_tensor_t *slice_src3 = marmot_tensor_create(env->ctx, slice_src_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_test_convert_f32_span(
        env, slice_src3, g_tensor_slice3d.src, slice_src_shape[0] * slice_src_shape[1] * slice_src_shape[2]
    );
    size_t slice_starts[] = {
        g_tensor_slice3d.starts[0],
        g_tensor_slice3d.starts[1],
        g_tensor_slice3d.starts[2],
    };
    size_t slice_sizes3[] = {
        g_tensor_slice3d.sizes[0],
        g_tensor_slice3d.sizes[1],
        g_tensor_slice3d.sizes[2],
    };
    const size_t slice_out_shape[] = {
        g_tensor_slice3d.sizes[0],
        g_tensor_slice3d.sizes[1],
        g_tensor_slice3d.sizes[2],
    };
    marmot_tensor_t *slice_dst3 = marmot_tensor_create(env->ctx, slice_out_shape, 3, MARMOT_DTYPE_FLOAT32);
    err = marmot_slice(ctx, slice_src3, slice_dst3, slice_starts, slice_sizes3);
    assert_int_equal(err, MARMOT_SUCCESS);

    size_t slice_total = slice_out_shape[0] * slice_out_shape[1] * slice_out_shape[2];
    fetch_tensor_as_f32(env, slice_dst3, check_buf, slice_total);
    marmot_test_expect_close_array(check_buf, g_tensor_slice3d.expected, slice_total, 1e-6f);

    // Avoid freeing shared quant params twice when destroying the view.
    quant_view->quant_params = nullptr;
    marmot_test_tensor_destroy_all(
        22, gather_out, gather_indices, gather_src, slice_dst3, slice_src3, concat_out3, concat_b, concat_a, t3_out, t3,
        quant_view, quant_src_q, quant_src, view_out, view_src, slice_out, concat_out, right, left, transpose_invalid,
        transpose_identity, transpose_default
    );
    marmot_test_tensor_destroy_all(3, bad_out, reshaped, src);
}

static void test_tensor_ops_default(void **state) {
    run_tensor_ops_checks((marmot_test_env_t *)(*state));
}

#if MARMOT_TEST_HAS_CPU_SCALAR_SUITE
static void test_tensor_ops_scalar(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_CPU) {
        skip();
        return;
    }
    run_tensor_ops_checks(env);
}
#endif

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(
            test_tensor_ops_default, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
#if MARMOT_TEST_HAS_CPU_SCALAR_SUITE
        cmocka_unit_test_setup_teardown(
            test_tensor_ops_scalar, marmot_test_backend_setup_scalar, marmot_test_backend_teardown
        ),
#endif
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
