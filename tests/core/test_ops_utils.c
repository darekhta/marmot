#include "marmot/error.h"
#include "marmot/ops_utils.h"

// clang-format off
#include <setjmp.h>
#include <stdarg.h>
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include <cmocka.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
// clang-format on

#include <stddef.h>

static marmot_shape_t make_shape(size_t ndim, const size_t *dims) {
    marmot_shape_t shape = {0};
    shape.ndim = ndim;
    for (size_t i = 0; i < ndim; ++i) {
        shape.shape[i] = dims[i];
    }
    if (ndim > 0) {
        shape.strides[ndim - 1] = 1;
        for (size_t i = ndim - 1; i > 0; --i) {
            shape.strides[i - 1] = shape.strides[i] * shape.shape[i];
        }
    }
    return shape;
}

static void assert_shape_dims(const marmot_shape_t *shape, size_t ndim, const size_t *dims) {
    assert_non_null(shape);
    assert_int_equal(shape->ndim, ndim);
    for (size_t i = 0; i < ndim; ++i) {
        assert_int_equal(shape->shape[i], dims[i]);
    }
}

static void assert_contiguous_strides(const marmot_shape_t *shape) {
    assert_non_null(shape);
    if (shape->ndim == 0) {
        return;
    }
    size_t expected = 1;
    for (size_t i = shape->ndim; i > 0; --i) {
        const size_t idx = i - 1;
        assert_int_equal(shape->strides[idx], expected);
        expected *= shape->shape[idx];
    }
}

static void test_unary_shape_copy(void **state) {
    (void)state;
    const size_t dims[3] = {2, 3, 4};
    marmot_shape_t input = make_shape(3, dims);
    marmot_shape_t out = {0};

    marmot_error_t err = marmot_infer_unary_output_shape(&input, &out);
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_shape_dims(&out, 3, dims);
    assert_contiguous_strides(&out);
}

static void test_binary_shape_mismatch(void **state) {
    (void)state;
    const size_t dims_a[2] = {2, 3};
    const size_t dims_b[2] = {2, 4};
    marmot_shape_t lhs = make_shape(2, dims_a);
    marmot_shape_t rhs = make_shape(2, dims_b);
    marmot_shape_t out = {0};

    marmot_error_t err = marmot_infer_binary_output_shape(&lhs, &rhs, &out);
    assert_int_equal(err, MARMOT_ERROR_DIMENSION_MISMATCH);
}

static void test_matmul_shape_basic(void **state) {
    (void)state;
    const size_t dims_a[2] = {2, 3};
    const size_t dims_b[2] = {3, 4};
    const size_t dims_out[2] = {2, 4};
    marmot_shape_t a = make_shape(2, dims_a);
    marmot_shape_t b = make_shape(2, dims_b);
    marmot_shape_t out = {0};

    marmot_error_t err = marmot_infer_matmul_output_shape(&a, &b, &out);
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_shape_dims(&out, 2, dims_out);
    assert_contiguous_strides(&out);
}

static void test_matmul_shape_broadcast(void **state) {
    (void)state;
    const size_t dims_a[3] = {2, 3, 4};
    const size_t dims_b[2] = {4, 5};
    const size_t dims_out[3] = {2, 3, 5};
    marmot_shape_t a = make_shape(3, dims_a);
    marmot_shape_t b = make_shape(2, dims_b);
    marmot_shape_t out = {0};

    marmot_error_t err = marmot_infer_matmul_output_shape(&a, &b, &out);
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_shape_dims(&out, 3, dims_out);
    assert_contiguous_strides(&out);
}

static void test_reduction_shape_keepdims(void **state) {
    (void)state;
    const size_t dims_in[3] = {2, 3, 4};
    const size_t dims_out[3] = {2, 1, 4};
    marmot_shape_t input = make_shape(3, dims_in);
    marmot_shape_t out = {0};
    int32_t axes[1] = {1};

    marmot_error_t err = marmot_infer_reduction_output_shape(&input, axes, 1, true, &out);
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_shape_dims(&out, 3, dims_out);
    assert_contiguous_strides(&out);
}

static void test_reduction_shape_dropdims(void **state) {
    (void)state;
    const size_t dims_in[3] = {2, 3, 4};
    const size_t dims_out[2] = {2, 4};
    marmot_shape_t input = make_shape(3, dims_in);
    marmot_shape_t out = {0};
    int32_t axes[1] = {1};

    marmot_error_t err = marmot_infer_reduction_output_shape(&input, axes, 1, false, &out);
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_shape_dims(&out, 2, dims_out);
    assert_contiguous_strides(&out);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_unary_shape_copy),         cmocka_unit_test(test_binary_shape_mismatch),
        cmocka_unit_test(test_matmul_shape_basic),       cmocka_unit_test(test_matmul_shape_broadcast),
        cmocka_unit_test(test_reduction_shape_keepdims), cmocka_unit_test(test_reduction_shape_dropdims),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
