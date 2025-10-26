#include <math.h>

#include "backend/test_backend_utils.h"

static void rope_reference_f32(
    const float *input, const float *positions, size_t batch, size_t seq_len, size_t dim, float theta,
    marmot_rope_type_t rope_type, float *out
) {
    const size_t token_count = batch * seq_len;
    const size_t pair_count = dim / 2;
    if (pair_count == 0) {
        memcpy(out, input, token_count * dim * sizeof(float));
        return;
    }

    float *freqs = (float *)malloc(pair_count * sizeof(float));
    assert_non_null(freqs);
    for (size_t i = 0; i < pair_count; ++i) {
        freqs[i] = powf(theta, -((float)(2 * i) / (float)dim));
    }

    for (size_t token = 0; token < token_count; ++token) {
        const float pos = positions[token];
        const size_t base = token * dim;
        const bool is_neox = rope_type == MARMOT_ROPE_TYPE_NEOX;
        const size_t half_dim = dim / 2;

        for (size_t i = 0; i < pair_count; ++i) {
            const float angle = pos * freqs[i];
            const float cos_theta = cosf(angle);
            const float sin_theta = sinf(angle);
            const size_t even_index = base + (is_neox ? i : 2 * i);
            const size_t odd_index = base + (is_neox ? (i + half_dim) : (2 * i + 1));
            const float even = input[even_index];
            const float odd = input[odd_index];
            out[even_index] = even * cos_theta - odd * sin_theta;
            out[odd_index] = even * sin_theta + odd * cos_theta;
        }
    }

    free(freqs);
}

static void exercise_rope_case(
    marmot_test_env_t *env, marmot_dtype_t dtype, bool positions_int32, marmot_rope_type_t rope_type, float theta,
    float tolerance
) {
    const size_t batch = 2;
    const size_t seq_len = 5;
    const size_t dim = 8;
    const size_t shape[] = {batch, seq_len, dim};
    const size_t positions_shape[] = {batch, seq_len};
    const size_t token_count = batch * seq_len;
    const size_t element_count = token_count * dim;

    float *input = (float *)malloc(element_count * sizeof(float));
    float *positions_f32 = (float *)malloc(token_count * sizeof(float));
    float *expected = (float *)malloc(element_count * sizeof(float));
    float *actual = (float *)malloc(element_count * sizeof(float));
    assert_non_null(input);
    assert_non_null(positions_f32);
    assert_non_null(expected);
    assert_non_null(actual);

    for (size_t i = 0; i < element_count; ++i) {
        input[i] = sinf((float)i * 0.137f) + 0.01f * (float)i;
    }
    for (size_t token = 0; token < token_count; ++token) {
        const size_t seq_index = token % seq_len;
        positions_f32[token] = (float)(seq_index) + 0.125f * (float)(token % 3);
    }

    marmot_tensor_t *x = marmot_tensor_create(env->ctx, shape, 3, dtype);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 3, dtype);
    marmot_tensor_t *positions_tensor =
        marmot_tensor_create(env->ctx, positions_shape, 2, positions_int32 ? MARMOT_DTYPE_INT32 : MARMOT_DTYPE_FLOAT32);
    assert_non_null(x);
    assert_non_null(out);
    assert_non_null(positions_tensor);

    marmot_test_convert_f32_span(env, x, input, element_count);
    if (positions_int32) {
        int32_t *positions_i32 = (int32_t *)malloc(token_count * sizeof(int32_t));
        assert_non_null(positions_i32);
        for (size_t i = 0; i < token_count; ++i) {
            positions_i32[i] = (int32_t)lrintf(positions_f32[i]);
            positions_f32[i] = (float)positions_i32[i];
        }
        rope_reference_f32(input, positions_f32, batch, seq_len, dim, theta, rope_type, expected);
        assert_int_equal(
            marmot_tensor_copy_from_host_buffer(
                env->ctx, positions_tensor, positions_i32, token_count * sizeof(int32_t)
            ),
            MARMOT_SUCCESS
        );
        free(positions_i32);
    } else {
        rope_reference_f32(input, positions_f32, batch, seq_len, dim, theta, rope_type, expected);
        assert_int_equal(
            marmot_tensor_copy_from_host_buffer(env->ctx, positions_tensor, positions_f32, token_count * sizeof(float)),
            MARMOT_SUCCESS
        );
    }

    marmot_rope_params_t rope_params = marmot_rope_params_default();
    rope_params.positions = positions_tensor;
    rope_params.theta = theta;
    rope_params.rope_type = rope_type;
    marmot_error_t err = marmot_rope(env->ctx, x, &rope_params, out);
    assert_int_equal(err, MARMOT_SUCCESS);

    marmot_test_fetch_f32_span(env, actual, out, element_count);
    marmot_test_expect_close_array(actual, expected, element_count, tolerance);

    marmot_test_tensor_destroy_all(3, positions_tensor, out, x);
    free(actual);
    free(expected);
    free(positions_f32);
    free(input);
}

static void test_rope_f32_matches_reference(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    exercise_rope_case(env, MARMOT_DTYPE_FLOAT32, false, MARMOT_ROPE_TYPE_NORM, 10000.0f, 1.0e-5f);
}

static void test_rope_f64_matches_reference(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_CPU) {
        marmot_context_t *cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
        if (cpu_ctx == nullptr) {
            skip();
            return;
        }
        marmot_test_env_t cpu_env = {
            .backend = MARMOT_BACKEND_CPU,
            .ctx = cpu_ctx,
        };
        exercise_rope_case(&cpu_env, MARMOT_DTYPE_FLOAT64, false, MARMOT_ROPE_TYPE_NORM, 10000.0f, 1.0e-6f);
        marmot_destroy(cpu_ctx);
        return;
    }
    exercise_rope_case(env, MARMOT_DTYPE_FLOAT64, false, MARMOT_ROPE_TYPE_NORM, 10000.0f, 1.0e-6f);
}

static void test_rope_f16_matches_reference(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    exercise_rope_case(env, MARMOT_DTYPE_FLOAT16, false, MARMOT_ROPE_TYPE_NORM, 10000.0f, 4.0e-3f);
}

static void test_rope_bf16_matches_reference(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    exercise_rope_case(env, MARMOT_DTYPE_BFLOAT16, false, MARMOT_ROPE_TYPE_NORM, 10000.0f, 1.5e-2f);
}

#if MARMOT_ENABLE_FP8
static void test_rope_fp8_e4m3_matches_reference(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend == MARMOT_BACKEND_METAL) {
        skip();
        return;
    }
    exercise_rope_case(env, MARMOT_DTYPE_FLOAT8_E4M3, false, MARMOT_ROPE_TYPE_NORM, 10000.0f, 6.0e-1f);
}

static void test_rope_fp8_e5m2_matches_reference(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend == MARMOT_BACKEND_METAL) {
        skip();
        return;
    }
    exercise_rope_case(env, MARMOT_DTYPE_FLOAT8_E5M2, false, MARMOT_ROPE_TYPE_NORM, 10000.0f, 6.5e-1f);
}
#endif

static void test_rope_positions_int32(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    exercise_rope_case(env, MARMOT_DTYPE_FLOAT32, true, MARMOT_ROPE_TYPE_NORM, 10000.0f, 1.0e-5f);
}

static void test_rope_neox_f32_matches_reference(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    exercise_rope_case(env, MARMOT_DTYPE_FLOAT32, false, MARMOT_ROPE_TYPE_NEOX, 10000.0f, 1.0e-5f);
}

static void test_rope_identity_zero_positions(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const size_t batch = 1;
    const size_t seq_len = 4;
    const size_t dim = 8;
    const size_t shape[] = {batch, seq_len, dim};
    const size_t positions_shape[] = {batch, seq_len};
    const size_t token_count = batch * seq_len;
    const size_t element_count = token_count * dim;

    float *input = (float *)malloc(element_count * sizeof(float));
    float *actual = (float *)malloc(element_count * sizeof(float));
    assert_non_null(input);
    assert_non_null(actual);

    for (size_t i = 0; i < element_count; ++i) {
        input[i] = cosf((float)i * 0.09f);
    }

    float *zero_positions = (float *)calloc(token_count, sizeof(float));
    assert_non_null(zero_positions);

    marmot_tensor_t *x = marmot_tensor_create(env->ctx, shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *positions_tensor = marmot_tensor_create(env->ctx, positions_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(x);
    assert_non_null(out);
    assert_non_null(positions_tensor);

    marmot_test_convert_f32_span(env, x, input, element_count);
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, positions_tensor, zero_positions, token_count * sizeof(float)),
        MARMOT_SUCCESS
    );

    marmot_rope_params_t rope_params = marmot_rope_params_default();
    rope_params.positions = positions_tensor;
    rope_params.theta = 10000.0f;
    marmot_error_t err = marmot_rope(env->ctx, x, &rope_params, out);
    assert_int_equal(err, MARMOT_SUCCESS);

    marmot_test_fetch_f32_span(env, actual, out, element_count);
    marmot_test_expect_close_array(actual, input, element_count, 1.0e-5f);

    marmot_test_tensor_destroy_all(3, positions_tensor, out, x);
    free(zero_positions);
    free(actual);
    free(input);
}

static void test_rope_batched_thetas_use_correct_scalars(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const size_t batch = 2;
    const size_t seq_len = 5;
    const size_t dim = 32;
    const size_t shape[] = {batch, seq_len, dim};
    const size_t positions_shape[] = {batch, seq_len};
    const size_t token_count = batch * seq_len;
    const size_t element_count = token_count * dim;
    const float theta_a = 10000.0f;
    const float theta_b = 1000.0f;

    float *input = (float *)malloc(element_count * sizeof(float));
    float *positions_f32 = (float *)malloc(token_count * sizeof(float));
    float *expected_a = (float *)malloc(element_count * sizeof(float));
    float *expected_b = (float *)malloc(element_count * sizeof(float));
    float *actual_a = (float *)malloc(element_count * sizeof(float));
    float *actual_b = (float *)malloc(element_count * sizeof(float));
    assert_non_null(input);
    assert_non_null(positions_f32);
    assert_non_null(expected_a);
    assert_non_null(expected_b);
    assert_non_null(actual_a);
    assert_non_null(actual_b);

    for (size_t i = 0; i < element_count; ++i) {
        input[i] = sinf((float)i * 0.137f) + 0.01f * (float)i;
    }
    for (size_t token = 0; token < token_count; ++token) {
        const size_t seq_index = token % seq_len;
        positions_f32[token] = (float)(seq_index) + 0.125f * (float)(token % 3);
    }

    rope_reference_f32(input, positions_f32, batch, seq_len, dim, theta_a, MARMOT_ROPE_TYPE_NORM, expected_a);
    rope_reference_f32(input, positions_f32, batch, seq_len, dim, theta_b, MARMOT_ROPE_TYPE_NORM, expected_b);

    float max_expected_delta = 0.0f;
    for (size_t i = 0; i < element_count; ++i) {
        float diff = fabsf(expected_a[i] - expected_b[i]);
        if (diff > max_expected_delta) {
            max_expected_delta = diff;
        }
    }
    assert_true(max_expected_delta > 1.0e-4f);

    marmot_tensor_t *x = marmot_tensor_create(env->ctx, shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_a = marmot_tensor_create(env->ctx, shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_b = marmot_tensor_create(env->ctx, shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *positions_tensor = marmot_tensor_create(env->ctx, positions_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(x);
    assert_non_null(out_a);
    assert_non_null(out_b);
    assert_non_null(positions_tensor);

    marmot_test_convert_f32_span(env, x, input, element_count);
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, positions_tensor, positions_f32, token_count * sizeof(float)),
        MARMOT_SUCCESS
    );

    assert_int_equal(marmot_graph_batch_begin(env->ctx), MARMOT_SUCCESS);
    marmot_rope_params_t rope_params_a = marmot_rope_params_default();
    rope_params_a.positions = positions_tensor;
    rope_params_a.theta = theta_a;
    marmot_rope_params_t rope_params_b = marmot_rope_params_default();
    rope_params_b.positions = positions_tensor;
    rope_params_b.theta = theta_b;
    marmot_error_t err_a = marmot_rope(env->ctx, x, &rope_params_a, out_a);
    marmot_error_t err_b = marmot_rope(env->ctx, x, &rope_params_b, out_b);
    marmot_error_t batch_end = marmot_graph_batch_end(env->ctx, true);
    marmot_error_t sync_err = marmot_device_synchronize(env->ctx);

    assert_int_equal(err_a, MARMOT_SUCCESS);
    assert_int_equal(err_b, MARMOT_SUCCESS);
    assert_int_equal(batch_end, MARMOT_SUCCESS);
    assert_int_equal(sync_err, MARMOT_SUCCESS);

    marmot_test_fetch_f32_span(env, actual_a, out_a, element_count);
    marmot_test_fetch_f32_span(env, actual_b, out_b, element_count);
    marmot_test_expect_close_array(actual_a, expected_a, element_count, 1.0e-5f);
    marmot_test_expect_close_array(actual_b, expected_b, element_count, 1.0e-5f);

    marmot_test_tensor_destroy_all(4, positions_tensor, out_b, out_a, x);
    free(actual_b);
    free(actual_a);
    free(expected_b);
    free(expected_a);
    free(positions_f32);
    free(input);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(
            test_rope_f32_matches_reference, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_rope_neox_f32_matches_reference, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_rope_f64_matches_reference, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_rope_f16_matches_reference, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_rope_bf16_matches_reference, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
#if MARMOT_ENABLE_FP8
        cmocka_unit_test_setup_teardown(
            test_rope_fp8_e4m3_matches_reference, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_rope_fp8_e5m2_matches_reference, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
#endif
        cmocka_unit_test_setup_teardown(
            test_rope_positions_int32, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_rope_identity_zero_positions, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_rope_batched_thetas_use_correct_scalars, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
