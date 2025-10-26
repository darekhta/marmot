#include "marmot/quant_block.h"

#include <float.h>
#include <math.h>
#include <string.h>

#include "backend/golden_data.h"
#include "backend/golden_quant_llama.h"
#include "backend/test_backend_utils.h"
#include "utils/dtype_ref.h"
#include "utils/quant_test_utils.h"

// #define DEBUG_METAL_QUANT 1

static void test_compute_quant_params(marmot_test_env_t *env) {
    size_t shape[] = {g_quant_params.length};
    marmot_tensor_t *tensor = marmot_test_tensor_from_array(env, shape, 1, g_quant_params.values);
    marmot_quant_params_t params;

    marmot_error_t err = marmot_compute_quant_params(env->ctx, tensor, MARMOT_DTYPE_INT8, 0, &params);
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_true(fabsf(params.scale - g_quant_params.scale_int8) <= 1e-6f);
    assert_true(fabsf(params.zero_point - g_quant_params.zero_int8) <= 1e-4f);

    err = marmot_compute_quant_params(env->ctx, tensor, MARMOT_DTYPE_UINT8, 0, &params);
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_true(fabsf(params.scale - g_quant_params.scale_uint8) <= 1e-6f);
    assert_true(fabsf(params.zero_point - g_quant_params.zero_uint8) <= 1e-3f);

    marmot_tensor_destroy(tensor);
}

static void test_int8_quantize_dequantize(marmot_test_env_t *env) {
    size_t shape[] = {g_quant_int8.length};
    marmot_tensor_t *src = marmot_test_tensor_from_array(env, shape, 1, g_quant_int8.input);
    marmot_quant_params_t params;
    params.scale = g_quant_int8.scale;
    params.zero_point = g_quant_int8.zero_point;
    params.block_size = 0;

    marmot_tensor_t *quant = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_INT8);
    marmot_tensor_t *dequant = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(quant);
    assert_non_null(dequant);

    marmot_error_t err = marmot_quantize(env->ctx, src, &params, quant);
    assert_int_equal(err, MARMOT_SUCCESS);
    err = marmot_dequantize(env->ctx, quant, dequant);
    assert_int_equal(err, MARMOT_SUCCESS);

    if (env->backend == MARMOT_BACKEND_CPU) {
        const marmot_int8_t *quant_data = (const marmot_int8_t *)marmot_test_tensor_data(env, quant);
        for (size_t i = 0; i < shape[0]; ++i) {
            assert_int_equal(quant_data[i].value, (int8_t)g_quant_int8.quantized[i]);
        }
    }

    float *roundtrip = (float *)malloc(shape[0] * sizeof(float));
    assert_non_null(roundtrip);
    marmot_test_fetch_f32_span(env, roundtrip, dequant, shape[0]);
    for (size_t i = 0; i < shape[0]; ++i) {
        float diff = fabsf(roundtrip[i] - g_quant_int8.dequantized[i]);
        assert_true(diff <= params.scale * 2.5f);
    }
    free(roundtrip);

    marmot_test_tensor_destroy_all(3, dequant, quant, src);
}

static void test_uint8_quantize_dequantize(marmot_test_env_t *env) {
    size_t shape[] = {g_quant_uint8.length};
    marmot_tensor_t *src = marmot_test_tensor_from_array(env, shape, 1, g_quant_uint8.input);
    marmot_quant_params_t params;
    params.scale = g_quant_uint8.scale;
    params.zero_point = g_quant_uint8.zero_point;
    params.block_size = 0;

    marmot_tensor_t *quant = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_UINT8);
    marmot_tensor_t *dequant = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(quant);
    assert_non_null(dequant);

    marmot_error_t err = marmot_quantize(env->ctx, src, &params, quant);
    assert_int_equal(err, MARMOT_SUCCESS);
    err = marmot_dequantize(env->ctx, quant, dequant);
    assert_int_equal(err, MARMOT_SUCCESS);

    if (env->backend == MARMOT_BACKEND_CPU) {
        const marmot_uint8_t *quant_data = (const marmot_uint8_t *)marmot_test_tensor_data(env, quant);
        for (size_t i = 0; i < shape[0]; ++i) {
            assert_int_equal(quant_data[i].value, (uint8_t)g_quant_uint8.quantized[i]);
        }
    }

    float *roundtrip = (float *)malloc(shape[0] * sizeof(float));
    assert_non_null(roundtrip);
    marmot_test_fetch_f32_span(env, roundtrip, dequant, shape[0]);
    for (size_t i = 0; i < shape[0]; ++i) {
        float diff = fabsf(roundtrip[i] - g_quant_uint8.dequantized[i]);
        assert_true(diff <= params.scale * 2.5f);
    }
    free(roundtrip);

    marmot_test_tensor_destroy_all(3, dequant, quant, src);
}

static void test_q4_quantization(marmot_test_env_t *env) {
    const size_t elements = 96;
    const size_t num_blocks = 3;
    size_t shape[] = {elements};
    marmot_tensor_t *src = marmot_test_tensor_from_array(env, shape, 1, g_llama_test_input);

    size_t q4_0_bytes = num_blocks * sizeof(marmot_q4_0_block_t);
    size_t q4_1_bytes = num_blocks * sizeof(marmot_q4_1_block_t);
    marmot_tensor_t *q4_0 = marmot_tensor_create(env->ctx, &q4_0_bytes, 1, MARMOT_DTYPE_UINT8);
    marmot_tensor_t *q4_1 = marmot_tensor_create(env->ctx, &q4_1_bytes, 1, MARMOT_DTYPE_UINT8);
    marmot_tensor_t *deq = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(q4_0);
    assert_non_null(q4_1);
    assert_non_null(deq);

    marmot_error_t err = marmot_quantize_q4_0(env->ctx, src, q4_0);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_quant_expect_quant_bytes(env, q4_0, g_llama_q4_0, sizeof(g_llama_q4_0), "Q4_0");
    marmot_quant_expect_dequant_golden(
        env, q4_0, g_llama_q4_0, sizeof(g_llama_q4_0), deq, marmot_dequantize_q4_0, g_llama_q4_0_dequant, elements,
        5e-4f, MARMOT_QUANT_KIND_Q4_0
    );

    err = marmot_quantize_q4_1(env->ctx, src, q4_1);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_quant_expect_quant_bytes(env, q4_1, g_llama_q4_1, sizeof(g_llama_q4_1), "Q4_1");
    marmot_quant_expect_dequant_golden(
        env, q4_1, g_llama_q4_1, sizeof(g_llama_q4_1), deq, marmot_dequantize_q4_1, g_llama_q4_1_dequant, elements,
        5e-4f, MARMOT_QUANT_KIND_Q4_1
    );

    marmot_test_tensor_destroy_all(4, deq, q4_1, q4_0, src);
}

static void test_q5_q8_quantization(marmot_test_env_t *env) {
    const size_t elements = 96;
    const size_t num_blocks = 3;
    size_t shape[] = {elements};

    marmot_tensor_t *src = marmot_test_tensor_from_array(env, shape, 1, g_llama_test_input);
    assert_non_null(src);

    size_t q5_0_bytes = num_blocks * sizeof(marmot_q5_0_block_t);
    size_t q5_1_bytes = num_blocks * sizeof(marmot_q5_1_block_t);
    size_t q8_0_bytes = num_blocks * sizeof(marmot_q8_0_block_t);
    size_t q8_1_bytes = num_blocks * sizeof(marmot_q8_1_block_t);

    marmot_tensor_t *q5_0 = marmot_tensor_create(env->ctx, &q5_0_bytes, 1, MARMOT_DTYPE_UINT8);
    marmot_tensor_t *q5_1 = marmot_tensor_create(env->ctx, &q5_1_bytes, 1, MARMOT_DTYPE_UINT8);
    marmot_tensor_t *q8_0 = marmot_tensor_create(env->ctx, &q8_0_bytes, 1, MARMOT_DTYPE_INT8);
    marmot_tensor_t *q8_1 = marmot_tensor_create(env->ctx, &q8_1_bytes, 1, MARMOT_DTYPE_INT8);
    marmot_tensor_t *deq = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(q5_0);
    assert_non_null(q5_1);
    assert_non_null(q8_0);
    assert_non_null(q8_1);
    assert_non_null(deq);

    // Q5_0: quantize + bit-exact validation + dequantize + tolerance check
    marmot_error_t err = marmot_quantize_q5_0(env->ctx, src, q5_0);
    assert_int_equal(err, MARMOT_SUCCESS);

    marmot_quant_expect_quant_bytes(env, q5_0, g_llama_q5_0, sizeof(g_llama_q5_0), "Q5_0");

    marmot_quant_expect_dequant_golden(
        env, q5_0, g_llama_q5_0, sizeof(g_llama_q5_0), deq, marmot_dequantize_q5_0, g_llama_q5_0_dequant, elements,
        5e-4f, MARMOT_QUANT_KIND_Q5_0
    );

    // Q5_1: quantize + bit-exact validation + dequantize + tolerance check
    err = marmot_quantize_q5_1(env->ctx, src, q5_1);
    assert_int_equal(err, MARMOT_SUCCESS);

    marmot_quant_expect_quant_bytes(env, q5_1, g_llama_q5_1, sizeof(g_llama_q5_1), "Q5_1");

    marmot_quant_expect_dequant_golden(
        env, q5_1, g_llama_q5_1, sizeof(g_llama_q5_1), deq, marmot_dequantize_q5_1, g_llama_q5_1_dequant, elements,
        6e-4f, MARMOT_QUANT_KIND_Q5_1
    );

    // Q8_0: quantize + bit-exact validation + dequantize + tolerance check
    err = marmot_quantize_q8_0(env->ctx, src, q8_0);
    assert_int_equal(err, MARMOT_SUCCESS);

    marmot_quant_expect_quant_bytes(env, q8_0, g_llama_q8_0, sizeof(g_llama_q8_0), "Q8_0");

    marmot_quant_expect_dequant_golden(
        env, q8_0, g_llama_q8_0, sizeof(g_llama_q8_0), deq, marmot_dequantize_q8_0, g_llama_q8_0_dequant, elements,
        5e-4f, MARMOT_QUANT_KIND_Q8_0
    );

    // Q8_1 golden check
    err = marmot_quantize_q8_1(env->ctx, src, q8_1);
    assert_int_equal(err, MARMOT_SUCCESS);
    marmot_quant_expect_quant_bytes(env, q8_1, g_llama_q8_1, sizeof(g_llama_q8_1), "Q8_1");
    marmot_quant_expect_dequant_golden(
        env, q8_1, g_llama_q8_1, sizeof(g_llama_q8_1), deq, marmot_dequantize_q8_1, g_llama_q8_1_dequant, elements,
        5e-4f, MARMOT_QUANT_KIND_Q8_1
    );
    marmot_test_tensor_destroy_all(6, q8_1, q8_0, q5_1, q5_0, deq, src);
}

static void test_k_quants_roundtrip(marmot_test_env_t *env) {
    // K-quants use 256-value super-blocks, so create 256 test values
    const size_t elements = 256;
    float test_data[256];
    for (size_t i = 0; i < 256; i++) {
        test_data[i] = g_llama_test_input[i % 96];
    }

    size_t shape[] = {elements};
    marmot_tensor_t *src = marmot_test_tensor_from_array(env, shape, 1, test_data);
    assert_non_null(src);

    marmot_tensor_t *deq = marmot_tensor_create(env->ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(deq);

    // Test Q2_K (2-bit quantization)
    {
        size_t q2_k_bytes = 1 * sizeof(marmot_q2_k_block_t);
        marmot_tensor_t *q2_k = marmot_tensor_create(env->ctx, &q2_k_bytes, 1, MARMOT_DTYPE_UINT8);
        assert_non_null(q2_k);

        marmot_error_t err = marmot_quantize_q2_k(env->ctx, src, q2_k);
        assert_int_equal(err, MARMOT_SUCCESS);

        marmot_quant_expect_quant_bytes(env, q2_k, g_llama_q2_k, sizeof(g_llama_q2_k), "Q2_K");

        marmot_quant_expect_dequant_golden(
            env, q2_k, g_llama_q2_k, sizeof(g_llama_q2_k), deq, marmot_dequantize_q2_k, g_llama_q2_k_dequant, elements,
            7e-4f, MARMOT_QUANT_KIND_Q2_K
        );

        marmot_tensor_destroy(q2_k);
    }

    // Test Q3_K (3-bit quantization)
    {
        size_t q3_k_bytes = 1 * sizeof(marmot_q3_k_block_t);
        marmot_tensor_t *q3_k = marmot_tensor_create(env->ctx, &q3_k_bytes, 1, MARMOT_DTYPE_UINT8);
        assert_non_null(q3_k);

        marmot_error_t err = marmot_quantize_q3_k(env->ctx, src, q3_k);
        assert_int_equal(err, MARMOT_SUCCESS);

        marmot_quant_expect_quant_bytes(env, q3_k, g_llama_q3_k, sizeof(g_llama_q3_k), "Q3_K");

        marmot_quant_expect_dequant_golden(
            env, q3_k, g_llama_q3_k, sizeof(g_llama_q3_k), deq, marmot_dequantize_q3_k, g_llama_q3_k_dequant, elements,
            6e-4f, MARMOT_QUANT_KIND_Q3_K
        );

        marmot_tensor_destroy(q3_k);
    }

    // Test Q4_K (4-bit quantization)
    {
        size_t q4_k_bytes = 1 * sizeof(marmot_q4_k_block_t);
        marmot_tensor_t *q4_k = marmot_tensor_create(env->ctx, &q4_k_bytes, 1, MARMOT_DTYPE_UINT8);
        assert_non_null(q4_k);

        marmot_error_t err = marmot_quantize_q4_k(env->ctx, src, q4_k);
        assert_int_equal(err, MARMOT_SUCCESS);

        marmot_quant_expect_quant_bytes(env, q4_k, g_llama_q4_k, sizeof(g_llama_q4_k), "Q4_K");

        marmot_quant_expect_dequant_golden(
            env, q4_k, g_llama_q4_k, sizeof(g_llama_q4_k), deq, marmot_dequantize_q4_k, g_llama_q4_k_dequant, elements,
            5e-4f, MARMOT_QUANT_KIND_Q4_K
        );

        marmot_tensor_destroy(q4_k);
    }

    // Test Q5_K (5-bit quantization)
    {
        size_t q5_k_bytes = 1 * sizeof(marmot_q5_k_block_t);
        marmot_tensor_t *q5_k = marmot_tensor_create(env->ctx, &q5_k_bytes, 1, MARMOT_DTYPE_UINT8);
        assert_non_null(q5_k);

        marmot_error_t err = marmot_quantize_q5_k(env->ctx, src, q5_k);
        assert_int_equal(err, MARMOT_SUCCESS);

        marmot_quant_expect_quant_bytes(env, q5_k, g_llama_q5_k, sizeof(g_llama_q5_k), "Q5_K");

        marmot_quant_expect_dequant_golden(
            env, q5_k, g_llama_q5_k, sizeof(g_llama_q5_k), deq, marmot_dequantize_q5_k, g_llama_q5_k_dequant, elements,
            5e-4f, MARMOT_QUANT_KIND_Q5_K
        );

        marmot_tensor_destroy(q5_k);
    }

    // Test Q6_K (6-bit quantization)
    {
        size_t q6_k_bytes = 1 * sizeof(marmot_q6_k_block_t);
        marmot_tensor_t *q6_k = marmot_tensor_create(env->ctx, &q6_k_bytes, 1, MARMOT_DTYPE_UINT8);
        assert_non_null(q6_k);

        marmot_error_t err = marmot_quantize_q6_k(env->ctx, src, q6_k);
        assert_int_equal(err, MARMOT_SUCCESS);

        marmot_quant_expect_quant_bytes(env, q6_k, g_llama_q6_k, sizeof(g_llama_q6_k), "Q6_K");

        marmot_quant_expect_dequant_golden(
            env, q6_k, g_llama_q6_k, sizeof(g_llama_q6_k), deq, marmot_dequantize_q6_k, g_llama_q6_k_dequant, elements,
            5e-4f, MARMOT_QUANT_KIND_Q6_K
        );

        marmot_tensor_destroy(q6_k);
    }

    // Test Q8_K (8-bit quantization with block sums)
    {
        size_t q8_k_bytes = 1 * sizeof(marmot_q8_k_block_t);
        marmot_tensor_t *q8_k = marmot_tensor_create(env->ctx, &q8_k_bytes, 1, MARMOT_DTYPE_INT8);
        assert_non_null(q8_k);

        marmot_error_t err = marmot_quantize_q8_k(env->ctx, src, q8_k);
        assert_int_equal(err, MARMOT_SUCCESS);

        marmot_quant_expect_quant_bytes(env, q8_k, g_llama_q8_k, sizeof(g_llama_q8_k), "Q8_K");

        marmot_quant_expect_dequant_golden(
            env, q8_k, g_llama_q8_k, sizeof(g_llama_q8_k), deq, marmot_dequantize_q8_k, g_llama_q8_k_dequant, elements,
            5e-4f, MARMOT_QUANT_KIND_Q8_K
        );

        marmot_tensor_destroy(q8_k);
    }

    marmot_test_tensor_destroy_all(2, deq, src);
}

static void run_quantization_suite(marmot_test_env_t *env) {
    test_compute_quant_params(env);
    test_int8_quantize_dequantize(env);
    test_uint8_quantize_dequantize(env);
    test_q4_quantization(env);
    test_q5_q8_quantization(env);
    test_k_quants_roundtrip(env);
}

static void test_quantization_default(void **state) {
    run_quantization_suite((marmot_test_env_t *)(*state));
}

#if MARMOT_TEST_HAS_CPU_SCALAR_SUITE
static void test_quantization_scalar(void **state) {
    run_quantization_suite((marmot_test_env_t *)(*state));
}
#endif

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(
            test_quantization_default, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
#if MARMOT_TEST_HAS_CPU_SCALAR_SUITE
        cmocka_unit_test_setup_teardown(
            test_quantization_scalar, marmot_test_backend_setup_scalar, marmot_test_backend_teardown
        ),
#endif
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
