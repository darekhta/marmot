#include <math.h>

#include "backend/golden_data.h"
#include "backend/test_backend_utils.h"

static void expect_close_f64(const double *actual, const double *expected, size_t count, double tol) {
    for (size_t i = 0; i < count; ++i) {
        double diff = fabs(actual[i] - expected[i]);
        assert_true(diff <= tol);
    }
}

static float test_silu(float x) {
    return x / (1.0f + expf(-x));
}

static float test_gelu_tanh(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static void check_geglu_float32_wide(const marmot_test_env_t *env) {
    const float gate_vals[] = {-15.0f, -10.0f, -5.0f, -1.0f, 0.0f, 1.0f, 5.0f, 10.0f, 12.0f, 15.0f};
    const float up_vals[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const size_t len = sizeof(gate_vals) / sizeof(gate_vals[0]);
    const size_t shape[] = {len};

    marmot_tensor_t *gate = marmot_test_tensor_from_array(env, shape, 1, gate_vals);
    marmot_tensor_t *up = marmot_test_tensor_from_array(env, shape, 1, up_vals);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(out);

    marmot_error_t err = marmot_geglu(env->ctx, gate, up, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    const float *geglu_out = marmot_test_tensor_f32_data(env, out);
    for (size_t i = 0; i < len; ++i) {
        assert_true(isfinite(geglu_out[i]));
        float expected = test_gelu_tanh(gate_vals[i]) * up_vals[i];
        assert_true(fabsf(geglu_out[i] - expected) <= 1e-5f);
    }

    marmot_test_tensor_destroy_all(3, out, up, gate);
}

static void check_glu_float32(const marmot_test_env_t *env) {
    const float gate_vals[] = {-1.0f, -0.25f, 0.0f, 0.75f, 1.5f};
    const float up_vals[] = {0.5f, -1.25f, 2.0f, -0.25f, 0.75f};
    const size_t len = sizeof(gate_vals) / sizeof(gate_vals[0]);
    const size_t shape[] = {len};

    marmot_tensor_t *gate = marmot_test_tensor_from_array(env, shape, 1, gate_vals);
    marmot_tensor_t *up = marmot_test_tensor_from_array(env, shape, 1, up_vals);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(out);

    marmot_error_t err = marmot_swiglu(env->ctx, gate, up, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    const float *swiglu_out = marmot_test_tensor_f32_data(env, out);
    for (size_t i = 0; i < len; ++i) {
        float expected = test_silu(gate_vals[i]) * up_vals[i];
        assert_true(fabsf(swiglu_out[i] - expected) <= 1e-6f);
    }

    err = marmot_geglu(env->ctx, gate, up, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    const float *geglu_out = marmot_test_tensor_f32_data(env, out);
    for (size_t i = 0; i < len; ++i) {
        float expected = test_gelu_tanh(gate_vals[i]) * up_vals[i];
        assert_true(fabsf(geglu_out[i] - expected) <= 1e-6f);
    }

    marmot_test_tensor_destroy_all(3, out, up, gate);
}

static void check_add_mul_float32(const marmot_test_env_t *env) {
    const size_t len = g_elementwise_fp32.length;
    const size_t shape[] = {len};
    marmot_tensor_t *a = marmot_test_tensor_from_array(env, shape, 1, g_elementwise_fp32.a);
    marmot_tensor_t *b = marmot_test_tensor_from_array(env, shape, 1, g_elementwise_fp32.b);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(out);

    marmot_error_t err = marmot_add(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, out), g_elementwise_fp32.add, len, 1e-6f);

    err = marmot_sub(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, out), g_elementwise_fp32.sub, len, 1e-6f);

    err = marmot_mul(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, out), g_elementwise_fp32.mul, len, 1e-6f);

    err = marmot_div(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, out), g_elementwise_fp32.div, len, 1e-6f);

    err = marmot_min(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, out), g_elementwise_fp32.minv, len, 1e-6f);

    err = marmot_max(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, out), g_elementwise_fp32.maxv, len, 1e-6f);

    marmot_test_tensor_destroy_all(3, out, b, a);
}

static void check_add_mul_float32_wide(const marmot_test_env_t *env) {
    const size_t len = g_elementwise_fp32_wide.length;
    const size_t shape[] = {len};
    marmot_tensor_t *a = marmot_test_tensor_from_array(env, shape, 1, g_elementwise_fp32_wide.a);
    marmot_tensor_t *b = marmot_test_tensor_from_array(env, shape, 1, g_elementwise_fp32_wide.b);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(out);

    marmot_error_t err = marmot_add(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, out), g_elementwise_fp32_wide.add, len, 1e-6f);

    err = marmot_sub(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, out), g_elementwise_fp32_wide.sub, len, 1e-6f);

    err = marmot_mul(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, out), g_elementwise_fp32_wide.mul, len, 1e-6f);

    err = marmot_div(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, out), g_elementwise_fp32_wide.div, len, 1e-6f);

    err = marmot_min(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, out), g_elementwise_fp32_wide.minv, len, 1e-6f);

    err = marmot_max(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, out), g_elementwise_fp32_wide.maxv, len, 1e-6f);

    marmot_test_tensor_destroy_all(3, out, b, a);
}

static void check_add_mul_float64(const marmot_test_env_t *env) {
    const size_t len = g_elementwise_fp32.length;
    const size_t shape[] = {len};
    marmot_tensor_t *a = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT64);
    marmot_tensor_t *b = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT64);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT64);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(out);

    marmot_test_convert_span(env, a, MARMOT_DTYPE_FLOAT64, g_elementwise_fp32.a_f64, len);
    marmot_test_convert_span(env, b, MARMOT_DTYPE_FLOAT64, g_elementwise_fp32.b_f64, len);

    marmot_error_t err = marmot_add(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    double actual[4];
    marmot_test_fetch_span(env, actual, MARMOT_DTYPE_FLOAT64, out, len);
    expect_close_f64(actual, g_elementwise_fp32.add_f64, len, 1e-12);

    err = marmot_sub(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual, MARMOT_DTYPE_FLOAT64, out, len);
    expect_close_f64(actual, g_elementwise_fp32.sub_f64, len, 1e-12);

    err = marmot_mul(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual, MARMOT_DTYPE_FLOAT64, out, len);
    expect_close_f64(actual, g_elementwise_fp32.mul_f64, len, 1e-12);

    err = marmot_div(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual, MARMOT_DTYPE_FLOAT64, out, len);
    expect_close_f64(actual, g_elementwise_fp32.div_f64, len, 1e-12);

    err = marmot_min(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual, MARMOT_DTYPE_FLOAT64, out, len);
    expect_close_f64(actual, g_elementwise_fp32.minv_f64, len, 1e-12);

    err = marmot_max(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual, MARMOT_DTYPE_FLOAT64, out, len);
    expect_close_f64(actual, g_elementwise_fp32.maxv_f64, len, 1e-12);

    marmot_test_tensor_destroy_all(3, out, b, a);
}

static void check_add_mul_float16(const marmot_test_env_t *env) {
    const size_t len = g_elementwise_fp16.length;
    const size_t shape[] = {len};
    marmot_tensor_t *a = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *b = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT16);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(out);

    marmot_test_convert_span(env, a, MARMOT_DTYPE_FLOAT32, g_elementwise_fp16.a, len);
    marmot_test_convert_span(env, b, MARMOT_DTYPE_FLOAT32, g_elementwise_fp16.b, len);
    marmot_error_t err = marmot_add(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    float converted[8];
    marmot_test_fetch_f32_span(env, converted, out, len);
    for (size_t i = 0; i < len; ++i) {
        float diff = fabsf(converted[i] - g_elementwise_fp16.add[i]);
        assert_true(diff <= 5e-3f);
    }

    err = marmot_sub(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_f32_span(env, converted, out, len);
    for (size_t i = 0; i < len; ++i) {
        float diff = fabsf(converted[i] - g_elementwise_fp16.sub[i]);
        assert_true(diff <= 6e-3f);
    }

    err = marmot_mul(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_f32_span(env, converted, out, len);
    for (size_t i = 0; i < len; ++i) {
        float diff = fabsf(converted[i] - g_elementwise_fp16.mul[i]);
        assert_true(diff <= 5e-3f);
    }

    err = marmot_div(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_f32_span(env, converted, out, len);
    for (size_t i = 0; i < len; ++i) {
        float diff = fabsf(converted[i] - g_elementwise_fp16.div[i]);
        assert_true(diff <= 7e-3f);
    }

    err = marmot_min(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_f32_span(env, converted, out, len);
    for (size_t i = 0; i < len; ++i) {
        float diff = fabsf(converted[i] - g_elementwise_fp16.minv[i]);
        assert_true(diff <= 7e-3f);
    }

    err = marmot_max(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_f32_span(env, converted, out, len);
    for (size_t i = 0; i < len; ++i) {
        float diff = fabsf(converted[i] - g_elementwise_fp16.maxv[i]);
        assert_true(diff <= 7e-3f);
    }

    marmot_test_tensor_destroy_all(3, out, b, a);
}

static void check_add_mul_bfloat16(const marmot_test_env_t *env) {
    const size_t len = g_elementwise_bf16.length;
    const size_t shape[] = {len};
    marmot_tensor_t *a = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_BFLOAT16);
    marmot_tensor_t *b = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_BFLOAT16);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_BFLOAT16);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(out);

    marmot_test_convert_span(env, a, MARMOT_DTYPE_FLOAT32, g_elementwise_bf16.a, len);
    marmot_test_convert_span(env, b, MARMOT_DTYPE_FLOAT32, g_elementwise_bf16.b, len);

    marmot_error_t err = marmot_add(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    float converted_bf16[4];
    marmot_test_fetch_f32_span(env, converted_bf16, out, len);
    for (size_t i = 0; i < len; ++i) {
        float diff = fabsf(converted_bf16[i] - g_elementwise_bf16.add[i]);
        assert_true(diff <= 1e-2f);
    }

    err = marmot_sub(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_f32_span(env, converted_bf16, out, len);
    for (size_t i = 0; i < len; ++i) {
        float diff = fabsf(converted_bf16[i] - g_elementwise_bf16.sub[i]);
        assert_true(diff <= 1.5e-2f);
    }

    err = marmot_mul(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_f32_span(env, converted_bf16, out, len);
    for (size_t i = 0; i < len; ++i) {
        float diff = fabsf(converted_bf16[i] - g_elementwise_bf16.mul[i]);
        assert_true(diff <= 1e-2f);
    }

    err = marmot_div(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_f32_span(env, converted_bf16, out, len);
    for (size_t i = 0; i < len; ++i) {
        float diff = fabsf(converted_bf16[i] - g_elementwise_bf16.div[i]);
        assert_true(diff <= 2e-2f);
    }

    err = marmot_min(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_f32_span(env, converted_bf16, out, len);
    for (size_t i = 0; i < len; ++i) {
        float diff = fabsf(converted_bf16[i] - g_elementwise_bf16.minv[i]);
        assert_true(diff <= 2e-2f);
    }

    err = marmot_max(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_f32_span(env, converted_bf16, out, len);
    for (size_t i = 0; i < len; ++i) {
        float diff = fabsf(converted_bf16[i] - g_elementwise_bf16.maxv[i]);
        assert_true(diff <= 2e-2f);
    }

    marmot_test_tensor_destroy_all(3, out, b, a);
}

static void check_pow_mod_float32(const marmot_test_env_t *env) {
    const size_t len = g_elementwise_powmod_fp32.length;
    const size_t shape[] = {len};
    marmot_tensor_t *base = marmot_test_tensor_from_array(env, shape, 1, g_elementwise_powmod_fp32.base);
    marmot_tensor_t *exponent = marmot_test_tensor_from_array(env, shape, 1, g_elementwise_powmod_fp32.exponent);
    marmot_tensor_t *divisor = marmot_test_tensor_from_array(env, shape, 1, g_elementwise_powmod_fp32.divisor);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(out);

    marmot_error_t err = marmot_pow(env->ctx, base, exponent, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, out), g_elementwise_powmod_fp32.powv, len, 1e-6f);

    err = marmot_mod(env->ctx, base, divisor, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, out), g_elementwise_powmod_fp32.modv, len, 1e-6f);

    marmot_test_tensor_destroy_all(4, out, divisor, exponent, base);
}

static void check_pow_mod_float64(const marmot_test_env_t *env) {
    const size_t len = g_elementwise_powmod_fp32.length;
    const size_t shape[] = {len};
    marmot_tensor_t *base = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT64);
    marmot_tensor_t *exp = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT64);
    marmot_tensor_t *div = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT64);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT64);
    assert_non_null(base);
    assert_non_null(exp);
    assert_non_null(div);
    assert_non_null(out);

    marmot_test_convert_span(env, base, MARMOT_DTYPE_FLOAT64, g_elementwise_powmod_fp32.base_f64, len);
    marmot_test_convert_span(env, exp, MARMOT_DTYPE_FLOAT64, g_elementwise_powmod_fp32.exponent_f64, len);
    marmot_test_convert_span(env, div, MARMOT_DTYPE_FLOAT64, g_elementwise_powmod_fp32.divisor_f64, len);

    marmot_error_t err = marmot_pow(env->ctx, base, exp, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    double actual[4];
    marmot_test_fetch_span(env, actual, MARMOT_DTYPE_FLOAT64, out, len);
    expect_close_f64(actual, g_elementwise_powmod_fp32.powv_f64, len, 1e-12);

    err = marmot_mod(env->ctx, base, div, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, actual, MARMOT_DTYPE_FLOAT64, out, len);
    expect_close_f64(actual, g_elementwise_powmod_fp32.modv_f64, len, 1e-12);

    marmot_test_tensor_destroy_all(4, out, div, exp, base);
}

static void check_pow_mod_float16(const marmot_test_env_t *env) {
    const size_t len = g_elementwise_powmod_fp16.length;
    const size_t shape[] = {len};
    marmot_tensor_t *base = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *exp = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *div = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT16);
    assert_non_null(base);
    assert_non_null(exp);
    assert_non_null(div);
    assert_non_null(out);

    marmot_test_convert_span(env, base, MARMOT_DTYPE_FLOAT32, g_elementwise_powmod_fp16.base, len);
    marmot_test_convert_span(env, exp, MARMOT_DTYPE_FLOAT32, g_elementwise_powmod_fp16.exponent, len);
    marmot_test_convert_span(env, div, MARMOT_DTYPE_FLOAT32, g_elementwise_powmod_fp16.divisor, len);

    marmot_error_t err = marmot_pow(env->ctx, base, exp, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    float converted[4];
    marmot_test_fetch_f32_span(env, converted, out, len);
    for (size_t i = 0; i < len; ++i) {
        float diff = fabsf(converted[i] - g_elementwise_powmod_fp16.powv[i]);
        assert_true(diff <= 6e-3f);
    }

    err = marmot_mod(env->ctx, base, div, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_f32_span(env, converted, out, len);
    for (size_t i = 0; i < len; ++i) {
        float diff = fabsf(converted[i] - g_elementwise_powmod_fp16.modv[i]);
        assert_true(diff <= 6e-3f);
    }

    marmot_test_tensor_destroy_all(4, out, div, exp, base);
}

static void check_pow_mod_bfloat16(const marmot_test_env_t *env) {
    const size_t len = g_elementwise_powmod_bf16.length;
    const size_t shape[] = {len};
    marmot_tensor_t *base = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_BFLOAT16);
    marmot_tensor_t *exp = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_BFLOAT16);
    marmot_tensor_t *div = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_BFLOAT16);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_BFLOAT16);
    assert_non_null(base);
    assert_non_null(exp);
    assert_non_null(div);
    assert_non_null(out);

    marmot_test_convert_span(env, base, MARMOT_DTYPE_FLOAT32, g_elementwise_powmod_bf16.base, len);
    marmot_test_convert_span(env, exp, MARMOT_DTYPE_FLOAT32, g_elementwise_powmod_bf16.exponent, len);
    marmot_test_convert_span(env, div, MARMOT_DTYPE_FLOAT32, g_elementwise_powmod_bf16.divisor, len);

    marmot_error_t err = marmot_pow(env->ctx, base, exp, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    float converted[3];
    marmot_test_fetch_f32_span(env, converted, out, len);
    for (size_t i = 0; i < len; ++i) {
        float diff = fabsf(converted[i] - g_elementwise_powmod_bf16.powv[i]);
        assert_true(diff <= 2e-2f);
    }

    err = marmot_mod(env->ctx, base, div, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_f32_span(env, converted, out, len);
    for (size_t i = 0; i < len; ++i) {
        float diff = fabsf(converted[i] - g_elementwise_powmod_bf16.modv[i]);
        assert_true(diff <= 2e-2f);
    }

    marmot_test_tensor_destroy_all(4, out, div, exp, base);
}

static void check_integer_ext_ops(const marmot_test_env_t *env) {
    const size_t len = g_elementwise_i32_ext.length;
    const size_t shape[] = {len};

    marmot_tensor_t *pow_base = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    marmot_tensor_t *pow_exp = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    marmot_tensor_t *pow_out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(pow_base);
    assert_non_null(pow_exp);
    assert_non_null(pow_out);
    for (size_t i = 0; i < len; ++i) {
        ((marmot_int32_t *)pow_base->data)[i] = MARMOT_I32(g_elementwise_i32_ext.pow_base[i]);
        ((marmot_int32_t *)pow_exp->data)[i] = MARMOT_I32(g_elementwise_i32_ext.pow_exp[i]);
    }
    marmot_test_commit_tensor(env, pow_base);
    marmot_test_commit_tensor(env, pow_exp);
    marmot_error_t err = marmot_pow(env->ctx, pow_base, pow_exp, pow_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    const marmot_int32_t *pow_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, pow_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(pow_out_data[i].value, g_elementwise_i32_ext.powv[i]);
    }
    marmot_test_tensor_destroy_all(3, pow_out, pow_exp, pow_base);

    marmot_tensor_t *mod_lhs = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    marmot_tensor_t *mod_rhs = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    marmot_tensor_t *mod_out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(mod_lhs);
    assert_non_null(mod_rhs);
    assert_non_null(mod_out);
    for (size_t i = 0; i < len; ++i) {
        ((marmot_int32_t *)mod_lhs->data)[i] = MARMOT_I32(g_elementwise_i32_ext.mod_lhs[i]);
        ((marmot_int32_t *)mod_rhs->data)[i] = MARMOT_I32(g_elementwise_i32_ext.mod_rhs[i]);
    }
    marmot_test_commit_tensor(env, mod_lhs);
    marmot_test_commit_tensor(env, mod_rhs);
    err = marmot_mod(env->ctx, mod_lhs, mod_rhs, mod_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    const marmot_int32_t *mod_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, mod_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(mod_out_data[i].value, g_elementwise_i32_ext.modv[i]);
    }
    marmot_test_tensor_destroy_all(3, mod_out, mod_rhs, mod_lhs);

    marmot_tensor_t *shift_src = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    marmot_tensor_t *shift_amt = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    marmot_tensor_t *shift_out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(shift_src);
    assert_non_null(shift_amt);
    assert_non_null(shift_out);
    for (size_t i = 0; i < len; ++i) {
        ((marmot_int32_t *)shift_src->data)[i] = MARMOT_I32(g_elementwise_i32_ext.shift_lhs[i]);
        ((marmot_int32_t *)shift_amt->data)[i] = MARMOT_I32(g_elementwise_i32_ext.shift_amt[i]);
    }
    marmot_test_commit_tensor(env, shift_src);
    marmot_test_commit_tensor(env, shift_amt);
    err = marmot_bitwise_shift_left(env->ctx, shift_src, shift_amt, shift_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    const marmot_int32_t *shift_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, shift_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(shift_out_data[i].value, g_elementwise_i32_ext.shift_left[i]);
    }

    err = marmot_bitwise_shift_right(env->ctx, shift_src, shift_amt, shift_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    shift_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, shift_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(shift_out_data[i].value, g_elementwise_i32_ext.shift_right[i]);
    }

    err = marmot_bitwise_shift_right_logical(env->ctx, shift_src, shift_amt, shift_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    shift_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, shift_out);
    for (size_t i = 0; i < len; ++i) {
        int32_t expected = (int32_t)g_elementwise_i32_ext.shift_right_logical[i];
        assert_int_equal(shift_out_data[i].value, expected);
    }
    marmot_test_tensor_destroy_all(3, shift_out, shift_amt, shift_src);
}

static void check_integer_wide_ops(const marmot_test_env_t *env) {
    const size_t len = g_elementwise_i32_wide.length;
    const size_t shape[] = {len};
    marmot_tensor_t *i32_a = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    marmot_tensor_t *i32_b = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    marmot_tensor_t *i32_out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    marmot_tensor_t *shift_amt = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(i32_a);
    assert_non_null(i32_b);
    assert_non_null(i32_out);
    assert_non_null(shift_amt);

    for (size_t i = 0; i < len; ++i) {
        ((marmot_int32_t *)i32_a->data)[i] = MARMOT_I32(g_elementwise_i32_wide.a[i]);
        ((marmot_int32_t *)i32_b->data)[i] = MARMOT_I32(g_elementwise_i32_wide.b[i]);
        ((marmot_int32_t *)shift_amt->data)[i] = MARMOT_I32((int32_t)g_elementwise_i32_wide.shift_amt[i]);
    }
    marmot_test_commit_tensor(env, i32_a);
    marmot_test_commit_tensor(env, i32_b);
    marmot_test_commit_tensor(env, shift_amt);

    marmot_error_t err = marmot_add(env->ctx, i32_a, i32_b, i32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    const marmot_int32_t *i32_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, i32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(i32_out_data[i].value, g_elementwise_i32_wide.add[i]);
    }

    err = marmot_sub(env->ctx, i32_a, i32_b, i32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    i32_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, i32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(i32_out_data[i].value, g_elementwise_i32_wide.sub[i]);
    }

    err = marmot_mul(env->ctx, i32_a, i32_b, i32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    i32_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, i32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(i32_out_data[i].value, g_elementwise_i32_wide.mul[i]);
    }

    err = marmot_div(env->ctx, i32_a, i32_b, i32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    i32_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, i32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(i32_out_data[i].value, g_elementwise_i32_wide.div[i]);
    }

    err = marmot_min(env->ctx, i32_a, i32_b, i32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    i32_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, i32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(i32_out_data[i].value, g_elementwise_i32_wide.minv[i]);
    }

    err = marmot_max(env->ctx, i32_a, i32_b, i32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    i32_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, i32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(i32_out_data[i].value, g_elementwise_i32_wide.maxv[i]);
    }

    err = marmot_bitwise_shift_left(env->ctx, i32_a, shift_amt, i32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    i32_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, i32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(i32_out_data[i].value, g_elementwise_i32_wide.shift_left[i]);
    }

    err = marmot_bitwise_shift_right(env->ctx, i32_a, shift_amt, i32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    i32_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, i32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(i32_out_data[i].value, g_elementwise_i32_wide.shift_right[i]);
    }

    err = marmot_bitwise_shift_right_logical(env->ctx, i32_a, shift_amt, i32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    i32_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, i32_out);
    for (size_t i = 0; i < len; ++i) {
        uint32_t value = (uint32_t)i32_out_data[i].value;
        assert_int_equal(value, g_elementwise_i32_wide.shift_right_logical[i]);
    }

    marmot_test_tensor_destroy_all(4, shift_amt, i32_out, i32_b, i32_a);
}

static void check_integer_ops(const marmot_test_env_t *env) {
    const size_t len = g_elementwise_i32.length;
    const size_t shape[] = {len};

    marmot_tensor_t *i32_a = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    marmot_tensor_t *i32_b = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    marmot_tensor_t *i32_out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(i32_a);
    assert_non_null(i32_b);
    assert_non_null(i32_out);

    for (size_t i = 0; i < len; ++i) {
        ((marmot_int32_t *)i32_a->data)[i] = MARMOT_I32(g_elementwise_i32.a[i]);
        ((marmot_int32_t *)i32_b->data)[i] = MARMOT_I32(g_elementwise_i32.b[i]);
    }
    marmot_test_commit_tensor(env, i32_a);
    marmot_test_commit_tensor(env, i32_b);

    marmot_error_t err = marmot_add(env->ctx, i32_a, i32_b, i32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    const marmot_int32_t *i32_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, i32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(i32_out_data[i].value, g_elementwise_i32.add[i]);
    }

    err = marmot_sub(env->ctx, i32_a, i32_b, i32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    i32_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, i32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(i32_out_data[i].value, g_elementwise_i32.sub[i]);
    }

    err = marmot_mul(env->ctx, i32_a, i32_b, i32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    i32_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, i32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(i32_out_data[i].value, g_elementwise_i32.mul[i]);
    }

    err = marmot_div(env->ctx, i32_a, i32_b, i32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    i32_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, i32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(i32_out_data[i].value, g_elementwise_i32.div[i]);
    }

    err = marmot_min(env->ctx, i32_a, i32_b, i32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    i32_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, i32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(i32_out_data[i].value, g_elementwise_i32.minv[i]);
    }

    err = marmot_max(env->ctx, i32_a, i32_b, i32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    i32_out_data = (const marmot_int32_t *)marmot_test_tensor_data(env, i32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(i32_out_data[i].value, g_elementwise_i32.maxv[i]);
    }

    marmot_test_tensor_destroy_all(3, i32_out, i32_b, i32_a);

    // Unsigned behaviour + bitwise
    marmot_tensor_t *u32_a = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_UINT32);
    marmot_tensor_t *u32_b = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_UINT32);
    marmot_tensor_t *u32_out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_UINT32);
    assert_non_null(u32_a);
    assert_non_null(u32_b);
    assert_non_null(u32_out);

    for (size_t i = 0; i < len; ++i) {
        ((marmot_uint32_t *)u32_a->data)[i] = MARMOT_U32(g_elementwise_bitwise_u32.a[i]);
        ((marmot_uint32_t *)u32_b->data)[i] = MARMOT_U32(g_elementwise_bitwise_u32.b[i]);
    }
    marmot_test_commit_tensor(env, u32_a);
    marmot_test_commit_tensor(env, u32_b);

    err = marmot_bitwise_and(env->ctx, u32_a, u32_b, u32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    const marmot_uint32_t *u32_out_data = (const marmot_uint32_t *)marmot_test_tensor_data(env, u32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(u32_out_data[i].value, g_elementwise_bitwise_u32.band[i]);
    }

    err = marmot_bitwise_or(env->ctx, u32_a, u32_b, u32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    u32_out_data = (const marmot_uint32_t *)marmot_test_tensor_data(env, u32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(u32_out_data[i].value, g_elementwise_bitwise_u32.bor[i]);
    }

    err = marmot_bitwise_xor(env->ctx, u32_a, u32_b, u32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    u32_out_data = (const marmot_uint32_t *)marmot_test_tensor_data(env, u32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(u32_out_data[i].value, g_elementwise_bitwise_u32.bxor[i]);
    }

    marmot_tensor_t *shift_amt_u32 = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_UINT32);
    assert_non_null(shift_amt_u32);
    for (size_t i = 0; i < len; ++i) {
        ((marmot_uint32_t *)shift_amt_u32->data)[i] = MARMOT_U32(g_elementwise_bitwise_u32.shift_amt[i]);
    }
    marmot_test_commit_tensor(env, shift_amt_u32);

    err = marmot_bitwise_shift_left(env->ctx, u32_a, shift_amt_u32, u32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    u32_out_data = (const marmot_uint32_t *)marmot_test_tensor_data(env, u32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(u32_out_data[i].value, g_elementwise_bitwise_u32.shl[i]);
    }

    err = marmot_bitwise_shift_right(env->ctx, u32_a, shift_amt_u32, u32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    u32_out_data = (const marmot_uint32_t *)marmot_test_tensor_data(env, u32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(u32_out_data[i].value, g_elementwise_bitwise_u32.shr[i]);
    }

    err = marmot_bitwise_shift_right_logical(env->ctx, u32_a, shift_amt_u32, u32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    u32_out_data = (const marmot_uint32_t *)marmot_test_tensor_data(env, u32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(u32_out_data[i].value, g_elementwise_bitwise_u32.shr[i]);
    }

    marmot_test_tensor_destroy_all(4, u32_out, shift_amt_u32, u32_b, u32_a);

    // Unsigned arithmetic wrap-around
    marmot_tensor_t *u16_a = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_UINT16);
    marmot_tensor_t *u16_b = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_UINT16);
    marmot_tensor_t *u16_out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_UINT16);
    assert_non_null(u16_a);
    assert_non_null(u16_b);
    assert_non_null(u16_out);

    for (size_t i = 0; i < len; ++i) {
        ((marmot_uint16_t *)u16_a->data)[i] = MARMOT_U16(g_elementwise_u16.a[i]);
        ((marmot_uint16_t *)u16_b->data)[i] = MARMOT_U16(g_elementwise_u16.b[i]);
    }
    marmot_test_commit_tensor(env, u16_a);
    marmot_test_commit_tensor(env, u16_b);

    err = marmot_add(env->ctx, u16_a, u16_b, u16_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    const marmot_uint16_t *u16_out_data = (const marmot_uint16_t *)marmot_test_tensor_data(env, u16_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(u16_out_data[i].value, g_elementwise_u16.add[i]);
    }

    err = marmot_sub(env->ctx, u16_a, u16_b, u16_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    u16_out_data = (const marmot_uint16_t *)marmot_test_tensor_data(env, u16_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(u16_out_data[i].value, g_elementwise_u16.sub[i]);
    }

    err = marmot_div(env->ctx, u16_a, u16_b, u16_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    u16_out_data = (const marmot_uint16_t *)marmot_test_tensor_data(env, u16_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(u16_out_data[i].value, g_elementwise_u16.div[i]);
    }

    marmot_test_tensor_destroy_all(3, u16_out, u16_b, u16_a);
}

static void check_bitwise_u32_wide(const marmot_test_env_t *env) {
    const size_t len = g_elementwise_bitwise_u32_wide.length;
    const size_t shape[] = {len};
    marmot_tensor_t *u32_a = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_UINT32);
    marmot_tensor_t *u32_b = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_UINT32);
    marmot_tensor_t *u32_out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_UINT32);
    marmot_tensor_t *shift_amt = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_UINT32);
    assert_non_null(u32_a);
    assert_non_null(u32_b);
    assert_non_null(u32_out);
    assert_non_null(shift_amt);

    for (size_t i = 0; i < len; ++i) {
        ((marmot_uint32_t *)u32_a->data)[i] = MARMOT_U32(g_elementwise_bitwise_u32_wide.a[i]);
        ((marmot_uint32_t *)u32_b->data)[i] = MARMOT_U32(g_elementwise_bitwise_u32_wide.b[i]);
        ((marmot_uint32_t *)shift_amt->data)[i] = MARMOT_U32(g_elementwise_bitwise_u32_wide.shift_amt[i]);
    }
    marmot_test_commit_tensor(env, u32_a);
    marmot_test_commit_tensor(env, u32_b);
    marmot_test_commit_tensor(env, shift_amt);

    marmot_error_t err = marmot_bitwise_and(env->ctx, u32_a, u32_b, u32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    const marmot_uint32_t *u32_out_data = (const marmot_uint32_t *)marmot_test_tensor_data(env, u32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(u32_out_data[i].value, g_elementwise_bitwise_u32_wide.band[i]);
    }

    err = marmot_bitwise_or(env->ctx, u32_a, u32_b, u32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    u32_out_data = (const marmot_uint32_t *)marmot_test_tensor_data(env, u32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(u32_out_data[i].value, g_elementwise_bitwise_u32_wide.bor[i]);
    }

    err = marmot_bitwise_xor(env->ctx, u32_a, u32_b, u32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    u32_out_data = (const marmot_uint32_t *)marmot_test_tensor_data(env, u32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(u32_out_data[i].value, g_elementwise_bitwise_u32_wide.bxor[i]);
    }

    err = marmot_bitwise_shift_left(env->ctx, u32_a, shift_amt, u32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    u32_out_data = (const marmot_uint32_t *)marmot_test_tensor_data(env, u32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(u32_out_data[i].value, g_elementwise_bitwise_u32_wide.shl[i]);
    }

    err = marmot_bitwise_shift_right(env->ctx, u32_a, shift_amt, u32_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    u32_out_data = (const marmot_uint32_t *)marmot_test_tensor_data(env, u32_out);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(u32_out_data[i].value, g_elementwise_bitwise_u32_wide.shr[i]);
    }

    marmot_test_tensor_destroy_all(4, u32_out, shift_amt, u32_b, u32_a);
}

static void check_unary_float_ops(const marmot_test_env_t *env) {
    const float signed_vals[] = {-4.0f, -1.5f, -0.0f, 2.0f, 5.5f};
    const size_t signed_len = sizeof(signed_vals) / sizeof(float);
    const size_t signed_shape[] = {signed_len};
    marmot_tensor_t *signed_tensor = marmot_test_tensor_from_array(env, signed_shape, 1, signed_vals);
    marmot_tensor_t *unary_out = marmot_tensor_create(env->ctx, signed_shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(signed_tensor);
    assert_non_null(unary_out);

    marmot_error_t err = marmot_neg(env->ctx, signed_tensor, unary_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    float expected[signed_len];
    for (size_t i = 0; i < signed_len; ++i) {
        expected[i] = -signed_vals[i];
    }
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, unary_out), expected, signed_len, 1e-6f);

    err = marmot_abs(env->ctx, signed_tensor, unary_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    for (size_t i = 0; i < signed_len; ++i) {
        expected[i] = fabsf(signed_vals[i]);
    }
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, unary_out), expected, signed_len, 1e-6f);

    const float positive_vals[] = {0.25f, 1.0f, 2.5f, 7.5f};
    const size_t pos_len = sizeof(positive_vals) / sizeof(float);
    const size_t pos_shape[] = {pos_len};
    marmot_tensor_t *pos_tensor = marmot_test_tensor_from_array(env, pos_shape, 1, positive_vals);
    marmot_tensor_t *pos_out = marmot_tensor_create(env->ctx, pos_shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(pos_tensor);
    assert_non_null(pos_out);

    err = marmot_sqrt(env->ctx, pos_tensor, pos_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    for (size_t i = 0; i < pos_len; ++i) {
        expected[i] = sqrtf(positive_vals[i]);
    }
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, pos_out), expected, pos_len, 5e-7f);

    err = marmot_exp(env->ctx, pos_tensor, pos_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    for (size_t i = 0; i < pos_len; ++i) {
        expected[i] = expf(positive_vals[i]);
    }
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, pos_out), expected, pos_len, 1e-3f);

    err = marmot_log(env->ctx, pos_tensor, pos_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    for (size_t i = 0; i < pos_len; ++i) {
        expected[i] = logf(positive_vals[i]);
    }
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, pos_out), expected, pos_len, 1e-6f);

    marmot_test_tensor_destroy_all(4, pos_out, unary_out, pos_tensor, signed_tensor);
}

static void check_bitwise_not_i32(const marmot_test_env_t *env) {
    const size_t len = 6;
    const size_t shape[] = {len};
    const int32_t values[] = {0, 1, -1, 0x12345678, -2147483647, 0x7FFFFFFF};
    marmot_tensor_t *src = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    marmot_tensor_t *dst = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(src);
    assert_non_null(dst);

    marmot_int32_t *src_data = (marmot_int32_t *)src->data;
    for (size_t i = 0; i < len; ++i) {
        src_data[i].value = values[i];
    }
    marmot_test_commit_tensor(env, src);

    marmot_error_t err = marmot_bitwise_not(env->ctx, src, dst);
    assert_int_equal(err, MARMOT_SUCCESS);
    const marmot_int32_t *dst_data = (const marmot_int32_t *)marmot_test_tensor_data(env, dst);
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(dst_data[i].value, ~values[i]);
    }

    marmot_test_tensor_destroy_all(2, dst, src);
}

static void check_where_float32(const marmot_test_env_t *env) {
    const size_t len = g_elementwise_fp32.length;
    const size_t shape[] = {len};
    uint8_t mask_data[] = {1, 0, 1, 0};
    float expected[4];
    for (size_t i = 0; i < len; ++i) {
        expected[i] = mask_data[i] ? g_elementwise_fp32.a[i] : g_elementwise_fp32.b[i];
    }

    marmot_tensor_t *mask = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_UINT8);
    marmot_tensor_t *a = marmot_test_tensor_from_array(env, shape, 1, g_elementwise_fp32.a);
    marmot_tensor_t *b = marmot_test_tensor_from_array(env, shape, 1, g_elementwise_fp32.b);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(mask);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(out);

    for (size_t i = 0; i < len; ++i) {
        ((marmot_uint8_t *)mask->data)[i].value = mask_data[i];
    }
    marmot_test_commit_tensor(env, mask);

    marmot_error_t err = marmot_where(env->ctx, mask, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, out), expected, len, 1e-6f);

    marmot_test_tensor_destroy_all(4, out, b, a, mask);
}

static void check_where_float32_wide(const marmot_test_env_t *env) {
    const size_t len = g_elementwise_fp32_wide.length;
    const size_t shape[] = {len};
    uint8_t mask_data[17];
    float expected[17];
    for (size_t i = 0; i < len; ++i) {
        mask_data[i] = (uint8_t)(((i * 3) & 1U) ? 1U : 0U);
        expected[i] = mask_data[i] ? g_elementwise_fp32_wide.a[i] : g_elementwise_fp32_wide.b[i];
    }

    marmot_tensor_t *mask = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_UINT8);
    marmot_tensor_t *a = marmot_test_tensor_from_array(env, shape, 1, g_elementwise_fp32_wide.a);
    marmot_tensor_t *b = marmot_test_tensor_from_array(env, shape, 1, g_elementwise_fp32_wide.b);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(mask);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(out);

    for (size_t i = 0; i < len; ++i) {
        ((marmot_uint8_t *)mask->data)[i].value = mask_data[i];
    }
    marmot_test_commit_tensor(env, mask);

    marmot_error_t err = marmot_where(env->ctx, mask, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, out), expected, len, 1e-6f);

    marmot_test_tensor_destroy_all(4, out, b, a, mask);
}

static unsigned normalize_shift_amount64(int64_t raw, unsigned bits) {
    if (raw < 0) {
        return 0;
    }
    if ((uint64_t)raw >= bits) {
        return bits;
    }
    return (unsigned)raw;
}

static int64_t test_int_pow_i64(int64_t base, int64_t exp) {
    int64_t e = exp;
    if (e < 0) {
        if (base == 1) {
            return 1;
        }
        if (base == -1) {
            return (e & 1LL) ? -1 : 1;
        }
        return 0;
    }
    int64_t result = 1;
    int64_t factor = base;
    while (e > 0) {
        if (e & 1LL) {
            result = result * factor;
        }
        e >>= 1LL;
        if (e != 0) {
            factor = factor * factor;
        }
    }
    return result;
}

static void check_integer_ops_i64(const marmot_test_env_t *env) {
    const size_t len = g_elementwise_i32.length;
    const size_t shape[] = {len};

    marmot_tensor_t *a = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT64);
    marmot_tensor_t *b = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT64);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT64);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(out);

    for (size_t i = 0; i < len; ++i) {
        ((marmot_int64_t *)a->data)[i] = MARMOT_I64((int64_t)g_elementwise_i32.a[i]);
        ((marmot_int64_t *)b->data)[i] = MARMOT_I64((int64_t)g_elementwise_i32.b[i]);
    }
    marmot_test_commit_tensor(env, a);
    marmot_test_commit_tensor(env, b);

    marmot_error_t err = marmot_add(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    const marmot_int64_t *out_data = (const marmot_int64_t *)marmot_test_tensor_data(env, out);
    for (size_t i = 0; i < len; ++i) {
        int64_t expected = (int64_t)g_elementwise_i32.add[i];
        assert_int_equal(out_data[i].value, expected);
    }

    err = marmot_sub(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    out_data = (const marmot_int64_t *)marmot_test_tensor_data(env, out);
    for (size_t i = 0; i < len; ++i) {
        int64_t expected = (int64_t)g_elementwise_i32.sub[i];
        assert_int_equal(out_data[i].value, expected);
    }

    err = marmot_mul(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    out_data = (const marmot_int64_t *)marmot_test_tensor_data(env, out);
    for (size_t i = 0; i < len; ++i) {
        int64_t expected = (int64_t)g_elementwise_i32.mul[i];
        assert_int_equal(out_data[i].value, expected);
    }

    err = marmot_div(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    out_data = (const marmot_int64_t *)marmot_test_tensor_data(env, out);
    for (size_t i = 0; i < len; ++i) {
        int64_t expected = (int64_t)g_elementwise_i32.div[i];
        assert_int_equal(out_data[i].value, expected);
    }

    err = marmot_min(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    out_data = (const marmot_int64_t *)marmot_test_tensor_data(env, out);
    for (size_t i = 0; i < len; ++i) {
        int64_t expected = (int64_t)g_elementwise_i32.minv[i];
        assert_int_equal(out_data[i].value, expected);
    }

    err = marmot_max(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_SUCCESS);
    out_data = (const marmot_int64_t *)marmot_test_tensor_data(env, out);
    for (size_t i = 0; i < len; ++i) {
        int64_t expected = (int64_t)g_elementwise_i32.maxv[i];
        assert_int_equal(out_data[i].value, expected);
    }

    marmot_test_tensor_destroy_all(3, out, b, a);

    const size_t ext_len = g_elementwise_i32_ext.length;
    const size_t ext_shape[] = {ext_len};
    marmot_tensor_t *pow_base = marmot_tensor_create(env->ctx, ext_shape, 1, MARMOT_DTYPE_INT64);
    marmot_tensor_t *pow_exp = marmot_tensor_create(env->ctx, ext_shape, 1, MARMOT_DTYPE_INT64);
    marmot_tensor_t *pow_out = marmot_tensor_create(env->ctx, ext_shape, 1, MARMOT_DTYPE_INT64);
    marmot_tensor_t *mod_lhs = marmot_tensor_create(env->ctx, ext_shape, 1, MARMOT_DTYPE_INT64);
    marmot_tensor_t *mod_rhs = marmot_tensor_create(env->ctx, ext_shape, 1, MARMOT_DTYPE_INT64);
    marmot_tensor_t *mod_out = marmot_tensor_create(env->ctx, ext_shape, 1, MARMOT_DTYPE_INT64);
    assert_non_null(pow_base);
    assert_non_null(pow_exp);
    assert_non_null(pow_out);
    assert_non_null(mod_lhs);
    assert_non_null(mod_rhs);
    assert_non_null(mod_out);

    for (size_t i = 0; i < ext_len; ++i) {
        ((marmot_int64_t *)pow_base->data)[i] = MARMOT_I64((int64_t)g_elementwise_i32_ext.pow_base[i]);
        ((marmot_int64_t *)pow_exp->data)[i] = MARMOT_I64((int64_t)g_elementwise_i32_ext.pow_exp[i]);
        ((marmot_int64_t *)mod_lhs->data)[i] = MARMOT_I64((int64_t)g_elementwise_i32_ext.mod_lhs[i]);
        ((marmot_int64_t *)mod_rhs->data)[i] = MARMOT_I64((int64_t)g_elementwise_i32_ext.mod_rhs[i]);
    }
    marmot_test_commit_tensor(env, pow_base);
    marmot_test_commit_tensor(env, pow_exp);
    marmot_test_commit_tensor(env, mod_lhs);
    marmot_test_commit_tensor(env, mod_rhs);

    err = marmot_pow(env->ctx, pow_base, pow_exp, pow_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    const marmot_int64_t *pow_out_data = (const marmot_int64_t *)marmot_test_tensor_data(env, pow_out);
    for (size_t i = 0; i < ext_len; ++i) {
        int64_t base_v = (int64_t)g_elementwise_i32_ext.pow_base[i];
        int64_t exp_v = (int64_t)g_elementwise_i32_ext.pow_exp[i];
        int64_t expected = test_int_pow_i64(base_v, exp_v);
        assert_int_equal(pow_out_data[i].value, expected);
    }

    err = marmot_mod(env->ctx, mod_lhs, mod_rhs, mod_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    const marmot_int64_t *mod_out_data = (const marmot_int64_t *)marmot_test_tensor_data(env, mod_out);
    for (size_t i = 0; i < ext_len; ++i) {
        int64_t lhs = (int64_t)g_elementwise_i32_ext.mod_lhs[i];
        int64_t rhs = (int64_t)g_elementwise_i32_ext.mod_rhs[i];
        int64_t expected = lhs % rhs;
        assert_int_equal(mod_out_data[i].value, expected);
    }

    marmot_test_tensor_destroy_all(6, mod_out, mod_rhs, mod_lhs, pow_out, pow_exp, pow_base);

    marmot_tensor_t *shift_src = marmot_tensor_create(env->ctx, ext_shape, 1, MARMOT_DTYPE_INT64);
    marmot_tensor_t *shift_amt = marmot_tensor_create(env->ctx, ext_shape, 1, MARMOT_DTYPE_INT64);
    marmot_tensor_t *shift_out = marmot_tensor_create(env->ctx, ext_shape, 1, MARMOT_DTYPE_INT64);
    assert_non_null(shift_src);
    assert_non_null(shift_amt);
    assert_non_null(shift_out);

    for (size_t i = 0; i < ext_len; ++i) {
        ((marmot_int64_t *)shift_src->data)[i] = MARMOT_I64((int64_t)g_elementwise_i32_ext.shift_lhs[i]);
        ((marmot_int64_t *)shift_amt->data)[i] = MARMOT_I64((int64_t)g_elementwise_i32_ext.shift_amt[i]);
    }
    marmot_test_commit_tensor(env, shift_src);
    marmot_test_commit_tensor(env, shift_amt);

    err = marmot_bitwise_shift_left(env->ctx, shift_src, shift_amt, shift_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    const marmot_int64_t *shift_out_data = (const marmot_int64_t *)marmot_test_tensor_data(env, shift_out);
    for (size_t i = 0; i < ext_len; ++i) {
        int64_t lhs = (int64_t)g_elementwise_i32_ext.shift_lhs[i];
        unsigned amount = normalize_shift_amount64((int64_t)g_elementwise_i32_ext.shift_amt[i], 64);
        int64_t expected = 0;
        if (amount < 64) {
            uint64_t shifted = ((uint64_t)lhs) << amount;
            expected = (int64_t)shifted;
        }
        assert_int_equal(shift_out_data[i].value, expected);
    }

    err = marmot_bitwise_shift_right(env->ctx, shift_src, shift_amt, shift_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    shift_out_data = (const marmot_int64_t *)marmot_test_tensor_data(env, shift_out);
    for (size_t i = 0; i < ext_len; ++i) {
        int64_t lhs = (int64_t)g_elementwise_i32_ext.shift_lhs[i];
        unsigned amount = normalize_shift_amount64((int64_t)g_elementwise_i32_ext.shift_amt[i], 64);
        int64_t expected = (amount >= 64) ? (lhs < 0 ? -1 : 0) : (lhs >> amount);
        assert_int_equal(shift_out_data[i].value, expected);
    }

    err = marmot_bitwise_shift_right_logical(env->ctx, shift_src, shift_amt, shift_out);
    assert_int_equal(err, MARMOT_SUCCESS);
    shift_out_data = (const marmot_int64_t *)marmot_test_tensor_data(env, shift_out);
    for (size_t i = 0; i < ext_len; ++i) {
        int64_t lhs = (int64_t)g_elementwise_i32_ext.shift_lhs[i];
        unsigned amount = normalize_shift_amount64((int64_t)g_elementwise_i32_ext.shift_amt[i], 64);
        int64_t expected = (amount >= 64) ? 0 : (int64_t)(((uint64_t)lhs) >> amount);
        assert_int_equal(shift_out_data[i].value, expected);
    }

    marmot_test_tensor_destroy_all(3, shift_out, shift_amt, shift_src);
}

static void check_shape_mismatch(const marmot_test_env_t *env) {
    size_t a_shape[] = {4};
    size_t b_shape[] = {2};
    marmot_tensor_t *a = marmot_test_tensor_from_array(env, a_shape, 1, (float[]){0.f, 1.f, 2.f, 3.f});
    marmot_tensor_t *b = marmot_test_tensor_from_array(env, b_shape, 1, (float[]){0.f, 1.f});
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, a_shape, 1, MARMOT_DTYPE_FLOAT32);
    marmot_error_t err = marmot_add(env->ctx, a, b, out);
    assert_int_equal(err, MARMOT_ERROR_DIMENSION_MISMATCH);
    marmot_test_tensor_destroy_all(3, out, b, a);
}

static void check_comparisons(const marmot_test_env_t *env) {
    const size_t len = g_elementwise_cmp_float.length;
    const size_t shape[] = {len};

    marmot_tensor_t *fa = marmot_test_tensor_from_array(env, shape, 1, g_elementwise_cmp_float.a);
    marmot_tensor_t *fb = marmot_test_tensor_from_array(env, shape, 1, g_elementwise_cmp_float.b);
    marmot_tensor_t *mask = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_UINT8);
    assert_non_null(mask);
    marmot_error_t err = marmot_compare_eq(env->ctx, fa, fb, mask);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_uint8_t mask_buf[len];
    marmot_test_fetch_span(env, mask_buf, MARMOT_DTYPE_UINT8, mask, len);
    const marmot_uint8_t *mask_data = mask_buf;
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(mask_data[i].value, g_elementwise_cmp_float.eq[i]);
    }

    err = marmot_compare_lt(env->ctx, fa, fb, mask);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, mask_buf, MARMOT_DTYPE_UINT8, mask, len);
    mask_data = mask_buf;
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(mask_data[i].value, g_elementwise_cmp_float.lt[i]);
    }

    err = marmot_compare_ge(env->ctx, fa, fb, mask);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, mask_buf, MARMOT_DTYPE_UINT8, mask, len);
    mask_data = mask_buf;
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(mask_data[i].value, g_elementwise_cmp_float.ge[i]);
    }

    marmot_test_tensor_destroy_all(2, fb, fa);

    // FLOAT16 comparisons
    marmot_tensor_t *ha = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *hb = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT16);
    assert_non_null(ha);
    assert_non_null(hb);
    assert_int_equal(
        marmot_convert_f32_to_f16(env->ctx, (marmot_float16_t *)ha->data, g_elementwise_cmp_float.a, len),
        MARMOT_SUCCESS
    );
    assert_int_equal(
        marmot_convert_f32_to_f16(env->ctx, (marmot_float16_t *)hb->data, g_elementwise_cmp_float.b, len),
        MARMOT_SUCCESS
    );

    err = marmot_compare_eq(env->ctx, ha, hb, mask);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, mask_buf, MARMOT_DTYPE_UINT8, mask, len);
    mask_data = mask_buf;
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(mask_data[i].value, g_elementwise_cmp_float.eq[i]);
    }

    err = marmot_compare_lt(env->ctx, ha, hb, mask);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, mask_buf, MARMOT_DTYPE_UINT8, mask, len);
    mask_data = mask_buf;
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(mask_data[i].value, g_elementwise_cmp_float.lt[i]);
    }

    err = marmot_compare_ge(env->ctx, ha, hb, mask);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, mask_buf, MARMOT_DTYPE_UINT8, mask, len);
    mask_data = mask_buf;
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(mask_data[i].value, g_elementwise_cmp_float.ge[i]);
    }
    marmot_test_tensor_destroy_all(2, hb, ha);

    // BFLOAT16 comparisons
    marmot_tensor_t *ba = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_BFLOAT16);
    marmot_tensor_t *bb = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_BFLOAT16);
    assert_non_null(ba);
    assert_non_null(bb);
    assert_int_equal(
        marmot_convert_f32_to_bf16(env->ctx, (marmot_bfloat16_t *)ba->data, g_elementwise_cmp_float.a, len),
        MARMOT_SUCCESS
    );
    assert_int_equal(
        marmot_convert_f32_to_bf16(env->ctx, (marmot_bfloat16_t *)bb->data, g_elementwise_cmp_float.b, len),
        MARMOT_SUCCESS
    );

    err = marmot_compare_eq(env->ctx, ba, bb, mask);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, mask_buf, MARMOT_DTYPE_UINT8, mask, len);
    mask_data = mask_buf;
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(mask_data[i].value, g_elementwise_cmp_float.eq[i]);
    }

    err = marmot_compare_lt(env->ctx, ba, bb, mask);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, mask_buf, MARMOT_DTYPE_UINT8, mask, len);
    mask_data = mask_buf;
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(mask_data[i].value, g_elementwise_cmp_float.lt[i]);
    }

    err = marmot_compare_ge(env->ctx, ba, bb, mask);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, mask_buf, MARMOT_DTYPE_UINT8, mask, len);
    mask_data = mask_buf;
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(mask_data[i].value, g_elementwise_cmp_float.ge[i]);
    }
    marmot_test_tensor_destroy_all(2, bb, ba);

#if MARMOT_ENABLE_FP8
    if (env->backend != MARMOT_BACKEND_METAL) {
        const float fp8_a[] = {1.0f, -2.0f, 3.0f, 0.0f};
        const float fp8_b[] = {1.0f, -3.0f, 2.0f, 0.5f};
        const uint8_t fp8_eq[] = {1u, 0u, 0u, 0u};
        const uint8_t fp8_lt[] = {0u, 0u, 0u, 1u};
        const uint8_t fp8_ge[] = {1u, 1u, 1u, 0u};

        marmot_tensor_t *pa = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT8_E4M3);
        marmot_tensor_t *pb = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT8_E4M3);
        assert_non_null(pa);
        assert_non_null(pb);
        assert_int_equal(
            marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT8_E4M3, pa->data, MARMOT_DTYPE_FLOAT32, fp8_a, len),
            MARMOT_SUCCESS
        );
        assert_int_equal(
            marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT8_E4M3, pb->data, MARMOT_DTYPE_FLOAT32, fp8_b, len),
            MARMOT_SUCCESS
        );

        err = marmot_compare_eq(env->ctx, pa, pb, mask);
        assert_int_equal(err, MARMOT_SUCCESS);
        marmot_test_fetch_span(env, mask_buf, MARMOT_DTYPE_UINT8, mask, len);
        mask_data = mask_buf;
        for (size_t i = 0; i < len; ++i) {
            assert_int_equal(mask_data[i].value, fp8_eq[i]);
        }

        err = marmot_compare_lt(env->ctx, pa, pb, mask);
        assert_int_equal(err, MARMOT_SUCCESS);
        marmot_test_fetch_span(env, mask_buf, MARMOT_DTYPE_UINT8, mask, len);
        mask_data = mask_buf;
        for (size_t i = 0; i < len; ++i) {
            assert_int_equal(mask_data[i].value, fp8_lt[i]);
        }

        err = marmot_compare_ge(env->ctx, pa, pb, mask);
        assert_int_equal(err, MARMOT_SUCCESS);
        marmot_test_fetch_span(env, mask_buf, MARMOT_DTYPE_UINT8, mask, len);
        mask_data = mask_buf;
        for (size_t i = 0; i < len; ++i) {
            assert_int_equal(mask_data[i].value, fp8_ge[i]);
        }
        marmot_test_tensor_destroy_all(2, pb, pa);

        marmot_tensor_t *qa = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT8_E5M2);
        marmot_tensor_t *qb = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT8_E5M2);
        assert_non_null(qa);
        assert_non_null(qb);
        assert_int_equal(
            marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT8_E5M2, qa->data, MARMOT_DTYPE_FLOAT32, fp8_a, len),
            MARMOT_SUCCESS
        );
        assert_int_equal(
            marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT8_E5M2, qb->data, MARMOT_DTYPE_FLOAT32, fp8_b, len),
            MARMOT_SUCCESS
        );

        err = marmot_compare_eq(env->ctx, qa, qb, mask);
        assert_int_equal(err, MARMOT_SUCCESS);
        marmot_test_fetch_span(env, mask_buf, MARMOT_DTYPE_UINT8, mask, len);
        mask_data = mask_buf;
        for (size_t i = 0; i < len; ++i) {
            assert_int_equal(mask_data[i].value, fp8_eq[i]);
        }

        err = marmot_compare_lt(env->ctx, qa, qb, mask);
        assert_int_equal(err, MARMOT_SUCCESS);
        marmot_test_fetch_span(env, mask_buf, MARMOT_DTYPE_UINT8, mask, len);
        mask_data = mask_buf;
        for (size_t i = 0; i < len; ++i) {
            assert_int_equal(mask_data[i].value, fp8_lt[i]);
        }

        err = marmot_compare_ge(env->ctx, qa, qb, mask);
        assert_int_equal(err, MARMOT_SUCCESS);
        marmot_test_fetch_span(env, mask_buf, MARMOT_DTYPE_UINT8, mask, len);
        mask_data = mask_buf;
        for (size_t i = 0; i < len; ++i) {
            assert_int_equal(mask_data[i].value, fp8_ge[i]);
        }
        marmot_test_tensor_destroy_all(2, qb, qa);
    }
#endif

    if (env->backend == MARMOT_BACKEND_CPU) {
        marmot_tensor_t *fa64 = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT64);
        marmot_tensor_t *fb64 = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT64);
        assert_non_null(fa64);
        assert_non_null(fb64);

        double *fa64_data = (double *)fa64->data;
        double *fb64_data = (double *)fb64->data;
        for (size_t i = 0; i < len; ++i) {
            fa64_data[i] = (double)g_elementwise_cmp_float.a[i];
            fb64_data[i] = (double)g_elementwise_cmp_float.b[i];
        }

        err = marmot_compare_eq(env->ctx, fa64, fb64, mask);
        assert_int_equal(err, MARMOT_SUCCESS);
        marmot_test_fetch_span(env, mask_buf, MARMOT_DTYPE_UINT8, mask, len);
        mask_data = mask_buf;
        for (size_t i = 0; i < len; ++i) {
            assert_int_equal(mask_data[i].value, g_elementwise_cmp_float.eq[i]);
        }

        err = marmot_compare_lt(env->ctx, fa64, fb64, mask);
        assert_int_equal(err, MARMOT_SUCCESS);
        marmot_test_fetch_span(env, mask_buf, MARMOT_DTYPE_UINT8, mask, len);
        mask_data = mask_buf;
        for (size_t i = 0; i < len; ++i) {
            assert_int_equal(mask_data[i].value, g_elementwise_cmp_float.lt[i]);
        }

        err = marmot_compare_ge(env->ctx, fa64, fb64, mask);
        assert_int_equal(err, MARMOT_SUCCESS);
        marmot_test_fetch_span(env, mask_buf, MARMOT_DTYPE_UINT8, mask, len);
        mask_data = mask_buf;
        for (size_t i = 0; i < len; ++i) {
            assert_int_equal(mask_data[i].value, g_elementwise_cmp_float.ge[i]);
        }

        marmot_test_tensor_destroy_all(2, fb64, fa64);
    }
    marmot_tensor_destroy(mask);

    marmot_tensor_t *ia = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    marmot_tensor_t *ib = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT32);
    marmot_tensor_t *out_uint8 = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_UINT8);
    assert_non_null(ia);
    assert_non_null(ib);
    assert_non_null(out_uint8);

    for (size_t i = 0; i < len; ++i) {
        ((marmot_int32_t *)ia->data)[i] = MARMOT_I32(g_elementwise_cmp_int.a[i]);
        ((marmot_int32_t *)ib->data)[i] = MARMOT_I32(g_elementwise_cmp_int.b[i]);
    }
    marmot_test_commit_tensor(env, ia);
    marmot_test_commit_tensor(env, ib);

    err = marmot_compare_gt(env->ctx, ia, ib, out_uint8);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_uint8_t out_uint8_buf[len];
    marmot_test_fetch_span(env, out_uint8_buf, MARMOT_DTYPE_UINT8, out_uint8, len);
    const marmot_uint8_t *out_uint8_data = out_uint8_buf;
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(out_uint8_data[i].value, g_elementwise_cmp_int.gt[i]);
    }

    err = marmot_compare_ne(env->ctx, ia, ib, out_uint8);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_fetch_span(env, out_uint8_buf, MARMOT_DTYPE_UINT8, out_uint8, len);
    out_uint8_data = out_uint8_buf;
    for (size_t i = 0; i < len; ++i) {
        assert_int_equal(out_uint8_data[i].value, g_elementwise_cmp_int.ne[i]);
    }

    marmot_test_tensor_destroy_all(2, ib, ia);

    if (env->backend == MARMOT_BACKEND_CPU) {
        marmot_tensor_t *ia64 = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT64);
        marmot_tensor_t *ib64 = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT64);
        assert_non_null(ia64);
        assert_non_null(ib64);

        for (size_t i = 0; i < len; ++i) {
            ((marmot_int64_t *)ia64->data)[i] = MARMOT_I64((int64_t)g_elementwise_cmp_int.a[i]);
            ((marmot_int64_t *)ib64->data)[i] = MARMOT_I64((int64_t)g_elementwise_cmp_int.b[i]);
        }
        marmot_test_commit_tensor(env, ia64);
        marmot_test_commit_tensor(env, ib64);

        err = marmot_compare_gt(env->ctx, ia64, ib64, out_uint8);
        assert_int_equal(err, MARMOT_SUCCESS);
        marmot_test_fetch_span(env, out_uint8_buf, MARMOT_DTYPE_UINT8, out_uint8, len);
        out_uint8_data = out_uint8_buf;
        for (size_t i = 0; i < len; ++i) {
            assert_int_equal(out_uint8_data[i].value, g_elementwise_cmp_int.gt[i]);
        }

        err = marmot_compare_ne(env->ctx, ia64, ib64, out_uint8);
        assert_int_equal(err, MARMOT_SUCCESS);
        marmot_test_fetch_span(env, out_uint8_buf, MARMOT_DTYPE_UINT8, out_uint8, len);
        out_uint8_data = out_uint8_buf;
        for (size_t i = 0; i < len; ++i) {
            assert_int_equal(out_uint8_data[i].value, g_elementwise_cmp_int.ne[i]);
        }

        marmot_test_tensor_destroy_all(2, ib64, ia64);
    }
    marmot_tensor_destroy(out_uint8);
}

static void check_fma(const marmot_test_env_t *env) {
    const size_t len = g_elementwise_fma.length;
    const size_t shape[] = {len};

    marmot_tensor_t *fa = marmot_test_tensor_from_array(env, shape, 1, g_elementwise_fma.a);
    marmot_tensor_t *fb = marmot_test_tensor_from_array(env, shape, 1, g_elementwise_fma.b);
    marmot_tensor_t *fc = marmot_test_tensor_from_array(env, shape, 1, g_elementwise_fma.c);
    marmot_tensor_t *fout = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(fout);

    marmot_error_t err = marmot_fma(env->ctx, fa, fb, fc, fout);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_test_expect_close_array(marmot_test_tensor_f32_data(env, fout), g_elementwise_fma.expected, len, 1e-6f);

    marmot_test_tensor_destroy_all(4, fout, fc, fb, fa);

    // FLOAT16
    marmot_tensor_t *ha = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *hb = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *hc = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *hout = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT16);
    assert_non_null(ha);
    assert_non_null(hb);
    assert_non_null(hc);
    assert_non_null(hout);

    marmot_test_convert_span(env, ha, MARMOT_DTYPE_FLOAT32, g_elementwise_fma.a, len);
    marmot_test_convert_span(env, hb, MARMOT_DTYPE_FLOAT32, g_elementwise_fma.b, len);
    marmot_test_convert_span(env, hc, MARMOT_DTYPE_FLOAT32, g_elementwise_fma.c, len);

    err = marmot_fma(env->ctx, ha, hb, hc, hout);
    assert_int_equal(err, MARMOT_SUCCESS);
    float converted[8];
    marmot_test_fetch_f32_span(env, converted, hout, len);
    for (size_t i = 0; i < len; ++i) {
        float diff = fabsf(converted[i] - g_elementwise_fma.expected[i]);
        assert_true(diff <= 6e-3f);
    }
    marmot_test_tensor_destroy_all(4, hout, hc, hb, ha);

    // BF16
    marmot_tensor_t *ba = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_BFLOAT16);
    marmot_tensor_t *bb = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_BFLOAT16);
    marmot_tensor_t *bc = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_BFLOAT16);
    marmot_tensor_t *bout = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_BFLOAT16);
    assert_non_null(ba);
    assert_non_null(bb);
    assert_non_null(bc);
    assert_non_null(bout);

    marmot_test_convert_span(env, ba, MARMOT_DTYPE_FLOAT32, g_elementwise_fma.a, len);
    marmot_test_convert_span(env, bb, MARMOT_DTYPE_FLOAT32, g_elementwise_fma.b, len);
    marmot_test_convert_span(env, bc, MARMOT_DTYPE_FLOAT32, g_elementwise_fma.c, len);

    err = marmot_fma(env->ctx, ba, bb, bc, bout);
    assert_int_equal(err, MARMOT_SUCCESS);
    float converted_bf16[8];
    marmot_test_fetch_f32_span(env, converted_bf16, bout, len);
    for (size_t i = 0; i < len; ++i) {
        float diff = fabsf(converted_bf16[i] - g_elementwise_fma.expected[i]);
        assert_true(diff <= 2e-2f);
    }
    marmot_test_tensor_destroy_all(4, bout, bc, bb, ba);

#if MARMOT_ENABLE_FP8 && MARMOT_TEST_HAS_CPU_INTERNALS
    if (env->backend == MARMOT_BACKEND_CPU) {
        marmot_tensor_t *pa = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT8_E4M3);
        marmot_tensor_t *pb = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT8_E4M3);
        marmot_tensor_t *pc = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT8_E4M3);
        marmot_tensor_t *pout = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT8_E4M3);
        assert_non_null(pa);
        assert_non_null(pb);
        assert_non_null(pc);
        assert_non_null(pout);

        cpu_convert_f32_to_fp8_e4m3(env->ctx->device_ctx, (marmot_float8_e4m3_t *)pa->data, g_elementwise_fma.a, len);
        cpu_convert_f32_to_fp8_e4m3(env->ctx->device_ctx, (marmot_float8_e4m3_t *)pb->data, g_elementwise_fma.b, len);
        cpu_convert_f32_to_fp8_e4m3(env->ctx->device_ctx, (marmot_float8_e4m3_t *)pc->data, g_elementwise_fma.c, len);

        err = marmot_fma(env->ctx, pa, pb, pc, pout);
        assert_int_equal(err, MARMOT_SUCCESS);
        float converted_fp8[8];
        cpu_convert_fp8_e4m3_to_f32(
            env->ctx->device_ctx, converted_fp8, (const marmot_float8_e4m3_t *)marmot_test_tensor_data(env, pout), len
        );
        for (size_t i = 0; i < len; ++i) {
            float diff = fabsf(converted_fp8[i] - g_elementwise_fma.expected[i]);
            assert_true(diff <= 5e-1f);
        }
        marmot_test_tensor_destroy_all(4, pout, pc, pb, pa);
    }
#endif
}

static void run_elementwise_suite(marmot_test_env_t *env) {
    check_add_mul_float32(env);
    check_glu_float32(env);
    check_geglu_float32_wide(env);
    check_add_mul_float32_wide(env);
    if (env->backend == MARMOT_BACKEND_CPU) {
        check_add_mul_float64(env);
    }
    check_add_mul_float16(env);
    check_add_mul_bfloat16(env);
    check_pow_mod_float32(env);
    if (env->backend == MARMOT_BACKEND_CPU) {
        check_pow_mod_float64(env);
        check_integer_ops_i64(env);
    }
    check_pow_mod_float16(env);
    check_pow_mod_bfloat16(env);
    check_integer_ext_ops(env);
    check_integer_wide_ops(env);
    check_integer_ops(env);
    check_comparisons(env);
    check_fma(env);
    check_bitwise_u32_wide(env);
    check_unary_float_ops(env);
    check_bitwise_not_i32(env);
    check_where_float32(env);
    check_where_float32_wide(env);
    check_shape_mismatch(env);
}

static void test_elementwise_default(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    run_elementwise_suite(env);
}

#if MARMOT_TEST_HAS_CPU_SCALAR_SUITE
static void test_elementwise_scalar(void **state) {
    marmot_test_run_with_cpu_scalar((marmot_test_env_t *)(*state), run_elementwise_suite);
}
#endif

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(
            test_elementwise_default, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
#if MARMOT_TEST_HAS_CPU_SCALAR_SUITE
        cmocka_unit_test_setup_teardown(
            test_elementwise_scalar, marmot_test_backend_setup_scalar, marmot_test_backend_teardown
        ),
#endif
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
