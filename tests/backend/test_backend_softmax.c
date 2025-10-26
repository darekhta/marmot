#include <math.h>

#include "backend/golden_data.h"
#include "backend/test_backend_utils.h"

static void exercise_softmax(marmot_test_env_t *env, marmot_dtype_t dtype, float tolerance) {
    const bool use_f64 = (dtype == MARMOT_DTYPE_FLOAT64);
    const marmot_dtype_t golden_dtype = use_f64 ? MARMOT_DTYPE_FLOAT64 : MARMOT_DTYPE_FLOAT32;
    const size_t elem_count = g_softmax.shape[0] * g_softmax.shape[1];
    const size_t shape[] = {g_softmax.shape[0], g_softmax.shape[1]};

    marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape, 2, dtype);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 2, dtype);
    assert_non_null(input);
    assert_non_null(out);

    const void *input_src = use_f64 ? (const void *)g_softmax.input_f64 : (const void *)g_softmax.input;
    marmot_test_convert_span(env, input, golden_dtype, input_src, elem_count);

    marmot_error_t err = marmot_softmax(env->ctx, &(marmot_softmax_desc_t){.x = input, .out = out, .axis = -1});
    assert_int_equal(err, MARMOT_SUCCESS);

    double actual[elem_count];
    marmot_test_fetch_span(env, actual, MARMOT_DTYPE_FLOAT64, out, elem_count);
    for (size_t i = 0; i < elem_count; ++i) {
        double expected = g_softmax.expected_axis_last_f64[i];
        double diff = fabs(actual[i] - expected);
        assert_true(diff <= (double)tolerance);
    }

    err = marmot_softmax(
        env->ctx, &(marmot_softmax_desc_t){.x = input, .out = out, .axis = (int32_t)(input->shape.ndim - 1)}
    );
    assert_int_equal(err, MARMOT_SUCCESS);

    err = marmot_softmax(env->ctx, &(marmot_softmax_desc_t){.x = input, .out = out, .axis = 0});
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual, MARMOT_DTYPE_FLOAT64, out, elem_count);
    for (size_t i = 0; i < elem_count; ++i) {
        double expected = g_softmax.expected_axis0_f64[i];
        double diff = fabs(actual[i] - expected);
        assert_true(diff <= (double)tolerance * 2.0);
    }

    marmot_test_tensor_destroy_all(2, out, input);
}

static void run_softmax_suite(marmot_test_env_t *env) {
    if (env->backend != MARMOT_BACKEND_METAL) {
        exercise_softmax(env, MARMOT_DTYPE_FLOAT64, 1e-6f); // Metal does not support F64
    }
    exercise_softmax(env, MARMOT_DTYPE_FLOAT32, 1e-6f);
    exercise_softmax(env, MARMOT_DTYPE_FLOAT16, 5e-3f);
    exercise_softmax(env, MARMOT_DTYPE_BFLOAT16, 1.2e-2f);
#if MARMOT_ENABLE_FP8
    if (env->backend != MARMOT_BACKEND_METAL) {
        exercise_softmax(env, MARMOT_DTYPE_FLOAT8_E4M3, 4.0e-1f);
        exercise_softmax(env, MARMOT_DTYPE_FLOAT8_E5M2, 5.0e-1f);
    }
#endif
}

static void test_softmax_default(void **state) {
    run_softmax_suite((marmot_test_env_t *)(*state));
}

#if MARMOT_TEST_HAS_CPU_SCALAR_SUITE
static void test_softmax_scalar(void **state) {
    marmot_test_run_with_cpu_scalar((marmot_test_env_t *)(*state), run_softmax_suite);
}
#endif

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_softmax_default, marmot_test_backend_setup, marmot_test_backend_teardown),
#if MARMOT_TEST_HAS_CPU_SCALAR_SUITE
        cmocka_unit_test_setup_teardown(
            test_softmax_scalar, marmot_test_backend_setup_scalar, marmot_test_backend_teardown
        ),
#endif
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
