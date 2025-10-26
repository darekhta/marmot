#include "marmot/config.h"

#include <stdbool.h>

#include <math.h>
#include <string.h>

#include "backend/golden_data.h"
#include "backend/golden_unary_numpy.h"
#include "backend/test_backend_utils.h"
#include "core/api/ops_internal.h"

static void check_activation_output(
    const char *name, marmot_dtype_t golden_dtype, const void *expected, const void *actual, size_t count,
    double tolerance, marmot_dtype_t tensor_dtype
) {
    if (golden_dtype == MARMOT_DTYPE_FLOAT64) {
        const double *exp = (const double *)expected;
        const double *act = (const double *)actual;
        for (size_t idx = 0; idx < count; ++idx) {
            double diff = fabs(act[idx] - exp[idx]);
            if (diff > tolerance) {
                fprintf(
                    stderr, "Activation %s (dtype=%d) idx=%zu expected=%.*g got=%.*g diff=%g tol=%g\n", name,
                    (int)tensor_dtype, idx, 17, exp[idx], 17, act[idx], diff, tolerance
                );
                assert_true(diff <= tolerance);
            }
        }
        return;
    }

    const float *exp = (const float *)expected;
    const float *act = (const float *)actual;
    for (size_t idx = 0; idx < count; ++idx) {
        float diff = fabsf(act[idx] - exp[idx]);
        if (diff > (float)tolerance) {
            fprintf(
                stderr, "Activation %s (dtype=%d) idx=%zu expected=%f got=%f diff=%f tol=%f\n", name, (int)tensor_dtype,
                idx, exp[idx], act[idx], diff, (float)tolerance
            );
            assert_true(diff <= (float)tolerance);
        }
    }
}

static void
exercise_activation_suite(const marmot_test_env_t *env, marmot_dtype_t dtype, double gelu_tol, double silu_tol) {
    const numpy_activation_golden_t *golden = &g_numpy_activation_golden;
    const size_t elem_count = golden->length;
    const size_t shape[] = {elem_count};
    const bool use_f64 = (dtype == MARMOT_DTYPE_FLOAT64);
    const marmot_dtype_t golden_dtype = use_f64 ? MARMOT_DTYPE_FLOAT64 : MARMOT_DTYPE_FLOAT32;
    const size_t golden_bytes = marmot_get_dtype_traits(golden_dtype)->storage_bytes * elem_count;

    marmot_tensor_t *x = marmot_tensor_create(env->ctx, shape, 1, dtype);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 1, dtype);
    assert_non_null(x);
    assert_non_null(out);

    const void *input_src = use_f64 ? (const void *)golden->input_f64 : (const void *)golden->input;
    marmot_test_convert_span(env, x, golden_dtype, input_src, elem_count);

    void *actual_host = malloc(golden_bytes);
    assert_non_null(actual_host);

    // ReLU
    marmot_error_t err = marmot_relu(env->ctx, x, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual_host, golden_dtype, out, elem_count);
    const void *relu_expected = use_f64 ? (const void *)golden->relu_f64 : (const void *)golden->relu;
    check_activation_output("relu", golden_dtype, relu_expected, actual_host, elem_count, 5e-4, dtype);

    // GELU
    err = marmot_gelu(env->ctx, x, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual_host, golden_dtype, out, elem_count);
    const void *gelu_expected = use_f64 ? (const void *)golden->gelu_f64 : (const void *)golden->gelu;
    check_activation_output("gelu", golden_dtype, gelu_expected, actual_host, elem_count, gelu_tol, dtype);

    // SiLU
    err = marmot_silu(env->ctx, x, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual_host, golden_dtype, out, elem_count);
    const void *silu_expected = use_f64 ? (const void *)golden->silu_f64 : (const void *)golden->silu;
    check_activation_output("silu", golden_dtype, silu_expected, actual_host, elem_count, silu_tol, dtype);

    // Swish alias
    err = marmot_swish(env->ctx, x, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual_host, golden_dtype, out, elem_count);
    check_activation_output("swish", golden_dtype, silu_expected, actual_host, elem_count, silu_tol, dtype);

    // GELU tanh approximation
    err = marmot_gelu_tanh(env->ctx, x, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual_host, golden_dtype, out, elem_count);
    const void *gelu_tanh_expected = use_f64 ? (const void *)golden->gelu_tanh_f64 : (const void *)golden->gelu_tanh;
    check_activation_output("gelu_tanh", golden_dtype, gelu_tanh_expected, actual_host, elem_count, gelu_tol, dtype);

    // Sigmoid
    err = marmot_sigmoid(env->ctx, x, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual_host, golden_dtype, out, elem_count);
    const void *sigmoid_expected = use_f64 ? (const void *)golden->sigmoid_f64 : (const void *)golden->sigmoid;
    check_activation_output("sigmoid", golden_dtype, sigmoid_expected, actual_host, elem_count, silu_tol, dtype);

    // Tanh
    err = marmot_tanh(env->ctx, x, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual_host, golden_dtype, out, elem_count);
    const void *tanh_expected = use_f64 ? (const void *)golden->tanh_v_f64 : (const void *)golden->tanh_v;
    check_activation_output("tanh", golden_dtype, tanh_expected, actual_host, elem_count, silu_tol, dtype);

    // Mish
    err = marmot_mish(env->ctx, x, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual_host, golden_dtype, out, elem_count);
    const void *mish_expected = use_f64 ? (const void *)golden->mish_f64 : (const void *)golden->mish;
    check_activation_output("mish", golden_dtype, mish_expected, actual_host, elem_count, silu_tol, dtype);

    // ELU
    const float elu_alpha = golden->elu_alpha;
    err = marmot_elu(env->ctx, x, out, elu_alpha);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual_host, golden_dtype, out, elem_count);
    const void *elu_expected = use_f64 ? (const void *)golden->elu_f64 : (const void *)golden->elu;
    check_activation_output("elu", golden_dtype, elu_expected, actual_host, elem_count, silu_tol, dtype);

    // SELU
    const float selu_alpha = golden->selu_alpha;
    const float selu_lambda = golden->selu_lambda;
    err = marmot_selu(env->ctx, x, out, selu_alpha, selu_lambda);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual_host, golden_dtype, out, elem_count);
    const void *selu_expected = use_f64 ? (const void *)golden->selu_f64 : (const void *)golden->selu;
    check_activation_output("selu", golden_dtype, selu_expected, actual_host, elem_count, silu_tol, dtype);

    // Leaky ReLU
    const float leaky_slope = golden->leaky_slope;
    err = marmot_leaky_relu(env->ctx, x, out, leaky_slope);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual_host, golden_dtype, out, elem_count);
    const void *leaky_expected = use_f64 ? (const void *)golden->leaky_relu_f64 : (const void *)golden->leaky_relu;
    check_activation_output("leaky_relu", golden_dtype, leaky_expected, actual_host, elem_count, silu_tol, dtype);

    // PReLU
    const float prelu_slope = golden->prelu_slope;
    err = marmot_prelu(env->ctx, x, out, prelu_slope);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual_host, golden_dtype, out, elem_count);
    const void *prelu_expected = use_f64 ? (const void *)golden->prelu_f64 : (const void *)golden->prelu;
    check_activation_output("prelu", golden_dtype, prelu_expected, actual_host, elem_count, silu_tol, dtype);

    marmot_test_tensor_destroy_all(2, out, x);
    free(actual_host);
}

static void run_activation_bias_variant(
    const marmot_test_env_t *env, marmot_dtype_t dtype, marmot_tensor_t *x, marmot_tensor_t *out,
    marmot_tensor_t *bias_tensor, marmot_device_unary_op_t op, marmot_op_id_t op_id, const char *name, float alpha,
    float beta, const void *expected, size_t elem_count, double tol, marmot_dtype_t golden_dtype, void *actual_host
) {
    marmot_activation_params_t params = {
        .parameter_tensor = nullptr,
        .bias = bias_tensor,
        .alpha = alpha,
        .beta = beta,
        .gamma = 0.0f,
    };
    marmot_error_t err = marmot_dispatch_unary_uniform(env->ctx, op, op_id, x, &params, out, name);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual_host, golden_dtype, out, elem_count);
    check_activation_output(name, golden_dtype, expected, actual_host, elem_count, tol, dtype);
}

static void exercise_activation_bias_suite(const marmot_test_env_t *env, marmot_dtype_t dtype, double tol) {
    const numpy_activation_golden_t *golden = &g_numpy_activation_golden;
    const size_t elem_count = golden->length;
    const size_t shape[] = {elem_count};
    const bool use_f64 = (dtype == MARMOT_DTYPE_FLOAT64);
    const marmot_dtype_t golden_dtype = use_f64 ? MARMOT_DTYPE_FLOAT64 : MARMOT_DTYPE_FLOAT32;
    const size_t golden_bytes = marmot_get_dtype_traits(golden_dtype)->storage_bytes * elem_count;

    marmot_tensor_t *x = marmot_tensor_create(env->ctx, shape, 1, dtype);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 1, dtype);
    assert_non_null(x);
    assert_non_null(out);
    const void *input_src = use_f64 ? (const void *)golden->input_f64 : (const void *)golden->input;
    marmot_test_convert_span(env, x, golden_dtype, input_src, elem_count);

    const size_t scalar_shape[] = {1};
    marmot_tensor_t *bias_scalar = marmot_tensor_create(env->ctx, scalar_shape, 1, dtype);
    marmot_tensor_t *bias_vector = marmot_tensor_create(env->ctx, shape, 1, dtype);
    assert_non_null(bias_scalar);
    assert_non_null(bias_vector);

    const void *bias_scalar_src = use_f64 ? (const void *)&golden->bias_scalar_f64 : (const void *)&golden->bias_scalar;
    marmot_test_convert_span(env, bias_scalar, golden_dtype, bias_scalar_src, 1);
    const void *bias_vector_src = use_f64 ? (const void *)golden->bias_vector_f64 : (const void *)golden->bias_vector;
    marmot_test_convert_span(env, bias_vector, golden_dtype, bias_vector_src, elem_count);

    void *actual_host = malloc(golden_bytes);
    assert_non_null(actual_host);

    const struct {
        marmot_device_unary_op_t op;
        marmot_op_id_t op_id;
        const char *name;
        float alpha;
        float beta;
        const float *scalar32;
        const float *vector32;
        const double *scalar64;
        const double *vector64;
    } cases[] = {
        {
            MARMOT_DEVICE_UNARY_RELU,
            MARMOT_OP_RELU,
            "relu+bias",
            0.0f,
            0.0f,
            golden->relu_bias_scalar,
            golden->relu_bias_vector,
            golden->relu_bias_scalar_f64,
            golden->relu_bias_vector_f64,
        },
        {
            MARMOT_DEVICE_UNARY_GELU,
            MARMOT_OP_GELU,
            "gelu+bias",
            0.0f,
            0.0f,
            golden->gelu_bias_scalar,
            golden->gelu_bias_vector,
            golden->gelu_bias_scalar_f64,
            golden->gelu_bias_vector_f64,
        },
        {
            MARMOT_DEVICE_UNARY_GELU_TANH,
            MARMOT_OP_GELU_TANH,
            "gelu_tanh+bias",
            0.0f,
            0.0f,
            golden->gelu_tanh_bias_scalar,
            golden->gelu_tanh_bias_vector,
            golden->gelu_tanh_bias_scalar_f64,
            golden->gelu_tanh_bias_vector_f64,
        },
        {
            MARMOT_DEVICE_UNARY_SILU,
            MARMOT_OP_SILU,
            "silu+bias",
            0.0f,
            0.0f,
            golden->silu_bias_scalar,
            golden->silu_bias_vector,
            golden->silu_bias_scalar_f64,
            golden->silu_bias_vector_f64,
        },
        {
            MARMOT_DEVICE_UNARY_SIGMOID,
            MARMOT_OP_SIGMOID,
            "sigmoid+bias",
            0.0f,
            0.0f,
            golden->sigmoid_bias_scalar,
            golden->sigmoid_bias_vector,
            golden->sigmoid_bias_scalar_f64,
            golden->sigmoid_bias_vector_f64,
        },
        {
            MARMOT_DEVICE_UNARY_TANH,
            MARMOT_OP_TANH,
            "tanh+bias",
            0.0f,
            0.0f,
            golden->tanh_v_bias_scalar,
            golden->tanh_v_bias_vector,
            golden->tanh_v_bias_scalar_f64,
            golden->tanh_v_bias_vector_f64,
        },
        {
            MARMOT_DEVICE_UNARY_MISH,
            MARMOT_OP_MISH,
            "mish+bias",
            0.0f,
            0.0f,
            golden->mish_bias_scalar,
            golden->mish_bias_vector,
            golden->mish_bias_scalar_f64,
            golden->mish_bias_vector_f64,
        },
        {
            MARMOT_DEVICE_UNARY_ELU,
            MARMOT_OP_ELU,
            "elu+bias",
            golden->elu_alpha,
            0.0f,
            golden->elu_bias_scalar,
            golden->elu_bias_vector,
            golden->elu_bias_scalar_f64,
            golden->elu_bias_vector_f64,
        },
        {
            MARMOT_DEVICE_UNARY_SELU,
            MARMOT_OP_SELU,
            "selu+bias",
            golden->selu_alpha,
            golden->selu_lambda,
            golden->selu_bias_scalar,
            golden->selu_bias_vector,
            golden->selu_bias_scalar_f64,
            golden->selu_bias_vector_f64,
        },
        {
            MARMOT_DEVICE_UNARY_LEAKY_RELU,
            MARMOT_OP_LEAKY_RELU,
            "leaky_relu+bias",
            golden->leaky_slope,
            0.0f,
            golden->leaky_relu_bias_scalar,
            golden->leaky_relu_bias_vector,
            golden->leaky_relu_bias_scalar_f64,
            golden->leaky_relu_bias_vector_f64,
        },
        {
            MARMOT_DEVICE_UNARY_PRELU,
            MARMOT_OP_PRELU,
            "prelu+bias",
            golden->prelu_slope,
            0.0f,
            golden->prelu_bias_scalar,
            golden->prelu_bias_vector,
            golden->prelu_bias_scalar_f64,
            golden->prelu_bias_vector_f64,
        },
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const void *expected_scalar = use_f64 ? (const void *)cases[i].scalar64 : (const void *)cases[i].scalar32;
        const void *expected_vector = use_f64 ? (const void *)cases[i].vector64 : (const void *)cases[i].vector32;
        run_activation_bias_variant(
            env, dtype, x, out, bias_scalar, cases[i].op, cases[i].op_id, cases[i].name, cases[i].alpha, cases[i].beta,
            expected_scalar, elem_count, tol, golden_dtype, actual_host
        );
        run_activation_bias_variant(
            env, dtype, x, out, bias_vector, cases[i].op, cases[i].op_id, cases[i].name, cases[i].alpha, cases[i].beta,
            expected_vector, elem_count, tol, golden_dtype, actual_host
        );
    }

    free(actual_host);
    marmot_test_tensor_destroy_all(3, bias_vector, bias_scalar, out);
    marmot_tensor_destroy(x);
}

static void check_unary_float(
    const marmot_test_env_t *env, marmot_dtype_t dtype, const llama_unary_float_golden_t *golden, float affine_tol,
    float math_tol
) {
    const size_t affine_len = golden->len_affine;
    const size_t affine_shape[] = {affine_len};
    marmot_tensor_t *input = marmot_tensor_create(env->ctx, affine_shape, 1, dtype);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, affine_shape, 1, dtype);
    assert_non_null(input);
    assert_non_null(out);
    marmot_test_convert_span(env, input, MARMOT_DTYPE_FLOAT32, golden->values_affine, affine_len);

    marmot_error_t err = marmot_abs(env->ctx, input, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    float buffer[8];
    marmot_test_fetch_span(env, buffer, MARMOT_DTYPE_FLOAT32, out, affine_len);
    for (size_t i = 0; i < affine_len; ++i) {
        float diff = fabsf(buffer[i] - golden->absv[i]);
        assert_true(diff <= affine_tol);
    }

    err = marmot_neg(env->ctx, input, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, buffer, MARMOT_DTYPE_FLOAT32, out, affine_len);
    for (size_t i = 0; i < affine_len; ++i) {
        float diff = fabsf(buffer[i] - golden->negv[i]);
        assert_true(diff <= affine_tol);
    }

    err = marmot_sign(env->ctx, input, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, buffer, MARMOT_DTYPE_FLOAT32, out, affine_len);
    for (size_t i = 0; i < affine_len; ++i) {
        float diff = fabsf(buffer[i] - golden->signv[i]);
        assert_true(diff <= affine_tol);
    }

    marmot_test_tensor_destroy_all(2, out, input);

    const size_t math_len = golden->len_math;
    const size_t math_shape[] = {math_len};
    marmot_tensor_t *math_input = marmot_tensor_create(env->ctx, math_shape, 1, dtype);
    marmot_tensor_t *math_out = marmot_tensor_create(env->ctx, math_shape, 1, dtype);
    assert_non_null(math_input);
    assert_non_null(math_out);
    marmot_test_convert_span(env, math_input, MARMOT_DTYPE_FLOAT32, golden->values_math, math_len);

    err = marmot_sqrt(env->ctx, math_input, math_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, buffer, MARMOT_DTYPE_FLOAT32, math_out, math_len);
    for (size_t i = 0; i < math_len; ++i) {
        float diff = fabsf(buffer[i] - golden->sqrtv[i]);
        assert_true(diff <= math_tol);
    }

    err = marmot_exp(env->ctx, math_input, math_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, buffer, MARMOT_DTYPE_FLOAT32, math_out, math_len);
    for (size_t i = 0; i < math_len; ++i) {
        float diff = fabsf(buffer[i] - golden->expv[i]);
        assert_true(diff <= math_tol);
    }

    err = marmot_log(env->ctx, math_input, math_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, buffer, MARMOT_DTYPE_FLOAT32, math_out, math_len);
    for (size_t i = 0; i < math_len; ++i) {
        float diff = fabsf(buffer[i] - golden->logv[i]);
        assert_true(diff <= math_tol);
    }

    marmot_test_tensor_destroy_all(2, math_out, math_input);
}

static void check_unary_i32(const marmot_test_env_t *env) {
    const size_t len = g_unary_i32.length;
    const size_t shape[] = {len};
    marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(input);
    assert_non_null(out);
    for (size_t i = 0; i < len; ++i) {
        ((marmot_int32_t *)input->data)[i] = MARMOT_I32(g_unary_i32.values[i]);
    }
    marmot_test_commit_tensor(env, input);

    marmot_error_t err = marmot_abs(env->ctx, input, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    const marmot_int32_t *out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(out_data[i].value, g_unary_i32.absv[i]);
    }

    err = marmot_neg(env->ctx, input, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(out_data[i].value, g_unary_i32.negv[i]);
    }

    err = marmot_sign(env->ctx, input, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(out_data[i].value, g_unary_i32.signv[i]);
    }

    err = marmot_bitwise_not(env->ctx, input, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(out_data[i].value, g_unary_i32.bitwise_not[i]);
    }

    marmot_test_tensor_destroy_all(2, out, input);
}

static void check_unary_u32(const marmot_test_env_t *env) {
    const size_t len = g_unary_u32.length;
    const size_t shape[] = {len};
    marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_UINT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_UINT32);
    assert_non_null(input);
    assert_non_null(out);
    for (size_t i = 0; i < len; ++i) {
        ((marmot_uint32_t *)input->data)[i] = MARMOT_U32(g_unary_u32.values[i]);
    }
    marmot_test_commit_tensor(env, input);

    marmot_error_t err = marmot_sign(env->ctx, input, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    const marmot_uint32_t *out_data = (const marmot_uint32_t *)marmot_test_tensor_data(env, out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(out_data[i].value, g_unary_u32.signv[i]);
    }

    err = marmot_bitwise_not(env->ctx, input, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    out_data = (const marmot_uint32_t *)marmot_test_tensor_data(env, out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(out_data[i].value, g_unary_u32.bitwise_not[i]);
    }

    marmot_test_tensor_destroy_all(2, out, input);
}

static void run_activation_checks(marmot_test_env_t *env) {
    exercise_activation_suite(env, MARMOT_DTYPE_FLOAT32, 1e-6f, 1e-6f);
    if (env->backend == MARMOT_BACKEND_CPU) {
        exercise_activation_suite(env, MARMOT_DTYPE_FLOAT64, 1e-6f, 1e-6f);
    }
    exercise_activation_suite(env, MARMOT_DTYPE_FLOAT16, 2e-2f, 2e-2f);
    exercise_activation_suite(env, MARMOT_DTYPE_BFLOAT16, 4e-1f, 4e-1f);
    exercise_activation_bias_suite(env, MARMOT_DTYPE_FLOAT32, 1e-6f);
    if (env->backend == MARMOT_BACKEND_CPU) {
        exercise_activation_bias_suite(env, MARMOT_DTYPE_FLOAT64, 1e-6f);
    }
    exercise_activation_bias_suite(env, MARMOT_DTYPE_FLOAT16, 2e-2f);
    exercise_activation_bias_suite(env, MARMOT_DTYPE_BFLOAT16, 4e-1f);
    check_unary_float(env, MARMOT_DTYPE_FLOAT32, &g_unary_fp32, 1e-6f, 1e-6f);
    if (env->backend == MARMOT_BACKEND_CPU) {
        check_unary_float(env, MARMOT_DTYPE_FLOAT64, &g_unary_fp32, 1e-6f, 1e-6f);
    }
    check_unary_float(env, MARMOT_DTYPE_FLOAT16, &g_unary_fp16, 2e-2f, 2e-2f);
    check_unary_float(env, MARMOT_DTYPE_BFLOAT16, &g_unary_bf16, 4e-1f, 4e-1f);
    check_unary_i32(env);
    check_unary_u32(env);
#if MARMOT_ENABLE_FP8
    const bool run_fp8 = env->backend != MARMOT_BACKEND_CPU && env->backend != MARMOT_BACKEND_METAL;
    if (run_fp8) {
        exercise_activation_suite(env, MARMOT_DTYPE_FLOAT8_E4M3, 3.5e-1f, 3.5e-1f);
        exercise_activation_suite(env, MARMOT_DTYPE_FLOAT8_E5M2, 4.0e-1f, 4.0e-1f);
        check_unary_float(env, MARMOT_DTYPE_FLOAT8_E4M3, &g_unary_fp8, 4.0e-1f, 4.0e-1f);
        check_unary_float(env, MARMOT_DTYPE_FLOAT8_E5M2, &g_unary_fp8, 4.0e-1f, 4.0e-1f);
    }
#endif
}

static void test_activations_default(void **state) {
    run_activation_checks((marmot_test_env_t *)(*state));
}

static void test_accelerate_unary_impl(void **state) {
#if !MARMOT_TEST_HAS_CPU_INTERNALS
    (void)state;
    skip();
#else
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    assert_non_null(env);
    marmot_context_t *ctx = env->ctx;
    bool destroy_ctx = false;
    if (env->backend != MARMOT_BACKEND_CPU) {
        ctx = marmot_init(MARMOT_BACKEND_CPU);
        if (ctx == nullptr) {
            skip();
            return;
        }
        destroy_ctx = true;
    }
    cpu_context_t *cpu_ctx = (cpu_context_t *)ctx->device_ctx;
    assert_non_null(cpu_ctx);
    const bool expect_accelerate = cpu_ctx->runtime_caps.has_accelerate;
    assert_int_equal(ctx->best_profile == MARMOT_PROFILE_ACCELERATE, expect_accelerate);
    if (destroy_ctx) {
        marmot_destroy(ctx);
    }
#endif
}

static void test_activation_bias_dimension_mismatch(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const size_t shape[] = {2, 4};
    const size_t bad_bias_shape[] = {3};
    marmot_tensor_t *x = marmot_tensor_create(env->ctx, shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *bias = marmot_tensor_create(env->ctx, bad_bias_shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(x);
    assert_non_null(out);
    assert_non_null(bias);

    marmot_activation_params_t params = {
        .bias = bias,
        .alpha = 0.0f,
        .beta = 0.0f,
        .gamma = 0.0f,
    };
    marmot_error_t err =
        marmot_dispatch_unary_uniform(env->ctx, MARMOT_DEVICE_UNARY_RELU, MARMOT_OP_RELU, x, &params, out, "relu+bias");
    assert_int_equal(err, MARMOT_ERROR_DIMENSION_MISMATCH);

    marmot_test_tensor_destroy_all(3, bias, out, x);
}

static void test_activation_bias_invalid_op(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const size_t shape[] = {8};
    marmot_tensor_t *x = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *bias = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(x);
    assert_non_null(out);
    assert_non_null(bias);

    marmot_activation_params_t params = {
        .bias = bias,
        .alpha = 0.0f,
        .beta = 0.0f,
        .gamma = 0.0f,
    };
    marmot_error_t err =
        marmot_dispatch_unary_uniform(env->ctx, MARMOT_DEVICE_UNARY_LOG, MARMOT_OP_LOG, x, &params, out, "log+bias");
    assert_int_equal(err, MARMOT_ERROR_INVALID_ARGUMENT);

    marmot_test_tensor_destroy_all(3, bias, out, x);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(
            test_activations_default, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_accelerate_unary_impl, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_activation_bias_dimension_mismatch, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_activation_bias_invalid_op, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
