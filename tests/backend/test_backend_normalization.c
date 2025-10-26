#include <stdbool.h>

#include <math.h>
#include <string.h>

#include "backend/golden_float_ops_llama.h"
#include "backend/test_backend_utils.h"

static float dtype_tolerance(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return 2e-5f;
    case MARMOT_DTYPE_FLOAT64:
        return 2e-5f;
    case MARMOT_DTYPE_FLOAT16:
        return 4.0f;
    case MARMOT_DTYPE_BFLOAT16:
        return 4.0f;
    default:
        return 1e-2f;
    }
}

static void
run_layernorm_case(marmot_test_env_t *env, const llama_layernorm_case_t *tc, marmot_dtype_t dtype, float tolerance) {
    const size_t elem_count = tc->rows * tc->cols;
    const size_t shape[] = {tc->rows, tc->cols};
    const bool use_f64 = (dtype == MARMOT_DTYPE_FLOAT64);
    const marmot_dtype_t golden_dtype = use_f64 ? MARMOT_DTYPE_FLOAT64 : MARMOT_DTYPE_FLOAT32;
    const size_t golden_bytes = marmot_get_dtype_traits(golden_dtype)->storage_bytes;

    marmot_tensor_t *x = marmot_tensor_create(env->ctx, shape, 2, dtype);
    marmot_tensor_t *residual = tc->residual != nullptr ? marmot_tensor_create(env->ctx, shape, 2, dtype) : nullptr;
    marmot_tensor_t *weight = tc->weight != nullptr ? marmot_tensor_create(env->ctx, &tc->cols, 1, dtype) : nullptr;
    marmot_tensor_t *bias = tc->bias != nullptr ? marmot_tensor_create(env->ctx, &tc->cols, 1, dtype) : nullptr;
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 2, dtype);

    assert_non_null(x);
    assert_non_null(out);

    const void *input_src = use_f64 ? (const void *)tc->input_f64 : (const void *)tc->input;
    marmot_test_convert_span(env, x, golden_dtype, input_src, elem_count);
    if (residual != nullptr) {
        const void *res_src = use_f64 ? (const void *)tc->residual_f64 : (const void *)tc->residual;
        marmot_test_convert_span(env, residual, golden_dtype, res_src, elem_count);
    }
    if (weight != nullptr) {
        const void *weight_src = use_f64 ? (const void *)tc->weight_f64 : (const void *)tc->weight;
        marmot_test_convert_span(env, weight, golden_dtype, weight_src, tc->cols);
    }
    if (bias != nullptr) {
        const void *bias_src = use_f64 ? (const void *)tc->bias_f64 : (const void *)tc->bias;
        marmot_test_convert_span(env, bias, golden_dtype, bias_src, tc->cols);
    }

    marmot_error_t err = marmot_layernorm(
        env->ctx,
        &(
            marmot_layernorm_desc_t
        ){.x = x, .residual = residual, .weight = weight, .bias = bias, .out = out, .eps = tc->epsilon}
    );
    assert_int_equal(err, MARMOT_SUCCESS);

    void *actual_host = malloc(elem_count * golden_bytes);
    assert_non_null(actual_host);
    marmot_test_fetch_span(env, actual_host, golden_dtype, out, elem_count);
    const void *expected_src = use_f64 ? (const void *)tc->expected_f64 : (const void *)tc->expected;

    if (use_f64) {
        const double *expected = (const double *)expected_src;
        const double *actual = (const double *)actual_host;
        for (size_t i = 0; i < elem_count; ++i) {
            double diff = fabs(actual[i] - expected[i]);
            if (diff > (double)tolerance) {
                fail_msg(
                    "LayerNorm case %s diff=%f exceeds tolerance %f (dtype=%d, index=%zu)", tc->name, diff, tolerance,
                    (int)dtype, i
                );
            }
        }
    } else {
        const float *expected = (const float *)expected_src;
        const float *actual = (const float *)actual_host;
        for (size_t i = 0; i < elem_count; ++i) {
            float diff = fabsf(actual[i] - expected[i]);
            if (diff > tolerance) {
                fail_msg(
                    "LayerNorm case %s diff=%f exceeds tolerance %f (dtype=%d, index=%zu)", tc->name, diff, tolerance,
                    (int)dtype, i
                );
            }
        }
    }

    marmot_test_tensor_destroy_all(5, out, bias, weight, residual, x);
    free(actual_host);
}

static void
run_rmsnorm_case(marmot_test_env_t *env, const llama_rmsnorm_case_t *tc, marmot_dtype_t dtype, float tolerance) {
    const size_t elem_count = tc->rows * tc->cols;
    const size_t shape[] = {tc->rows, tc->cols};
    const bool use_f64 = (dtype == MARMOT_DTYPE_FLOAT64);
    const marmot_dtype_t golden_dtype = use_f64 ? MARMOT_DTYPE_FLOAT64 : MARMOT_DTYPE_FLOAT32;
    const size_t golden_bytes = marmot_get_dtype_traits(golden_dtype)->storage_bytes;

    marmot_tensor_t *x = marmot_tensor_create(env->ctx, shape, 2, dtype);
    marmot_tensor_t *residual = tc->residual != nullptr ? marmot_tensor_create(env->ctx, shape, 2, dtype) : nullptr;
    marmot_tensor_t *weight = tc->weight != nullptr ? marmot_tensor_create(env->ctx, &tc->cols, 1, dtype) : nullptr;
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 2, dtype);

    assert_non_null(x);
    assert_non_null(out);

    const void *input_src = use_f64 ? (const void *)tc->input_f64 : (const void *)tc->input;
    marmot_test_convert_span(env, x, golden_dtype, input_src, elem_count);
    if (residual != nullptr) {
        const void *res_src = use_f64 ? (const void *)tc->residual_f64 : (const void *)tc->residual;
        marmot_test_convert_span(env, residual, golden_dtype, res_src, elem_count);
    }
    if (weight != nullptr) {
        const void *weight_src = use_f64 ? (const void *)tc->weight_f64 : (const void *)tc->weight;
        marmot_test_convert_span(env, weight, golden_dtype, weight_src, tc->cols);
    }

    marmot_error_t err = marmot_rmsnorm(
        env->ctx,
        &(marmot_rmsnorm_desc_t){.x = x, .residual = residual, .weight = weight, .out = out, .eps = tc->epsilon}
    );
    assert_int_equal(err, MARMOT_SUCCESS);

    void *actual_host = malloc(elem_count * golden_bytes);
    assert_non_null(actual_host);
    marmot_test_fetch_span(env, actual_host, golden_dtype, out, elem_count);
    const void *expected_src = use_f64 ? (const void *)tc->expected_f64 : (const void *)tc->expected;

    if (use_f64) {
        const double *expected = (const double *)expected_src;
        const double *actual = (const double *)actual_host;
        for (size_t i = 0; i < elem_count; ++i) {
            double diff = fabs(actual[i] - expected[i]);
            if (diff > (double)tolerance) {
                fail_msg(
                    "RMSNorm case %s diff=%f exceeds tolerance %f (dtype=%d, index=%zu)", tc->name, diff, tolerance,
                    (int)dtype, i
                );
            }
        }
    } else {
        const float *expected = (const float *)expected_src;
        const float *actual = (const float *)actual_host;
        for (size_t i = 0; i < elem_count; ++i) {
            float diff = fabsf(actual[i] - expected[i]);
            if (diff > tolerance) {
                fail_msg(
                    "RMSNorm case %s diff=%f exceeds tolerance %f (dtype=%d, index=%zu)", tc->name, diff, tolerance,
                    (int)dtype, i
                );
            }
        }
    }

    marmot_test_tensor_destroy_all(4, out, weight, residual, x);
    free(actual_host);
}

static void compute_rmsnorm_gemma_expected_f32(
    const float *input, const float *residual, const float *weight, size_t rows, size_t cols, float eps, float *out
) {
    for (size_t row = 0; row < rows; ++row) {
        float sum_sq = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            const size_t idx = row * cols + col;
            float v = input[idx] + (residual != nullptr ? residual[idx] : 0.0f);
            sum_sq += v * v;
        }
        float inv_rms = 1.0f / sqrtf(sum_sq / (float)cols + eps);
        for (size_t col = 0; col < cols; ++col) {
            const size_t idx = row * cols + col;
            float v = input[idx] + (residual != nullptr ? residual[idx] : 0.0f);
            float scale = weight != nullptr ? weight[col] + 1.0f : 1.0f;
            out[idx] = v * inv_rms * scale;
        }
    }
}

static void compute_rmsnorm_gemma_expected_f64(
    const double *input, const double *residual, const double *weight, size_t rows, size_t cols, double eps, double *out
) {
    for (size_t row = 0; row < rows; ++row) {
        double sum_sq = 0.0;
        for (size_t col = 0; col < cols; ++col) {
            const size_t idx = row * cols + col;
            double v = input[idx] + (residual != nullptr ? residual[idx] : 0.0);
            sum_sq += v * v;
        }
        double inv_rms = 1.0 / sqrt(sum_sq / (double)cols + eps);
        for (size_t col = 0; col < cols; ++col) {
            const size_t idx = row * cols + col;
            double v = input[idx] + (residual != nullptr ? residual[idx] : 0.0);
            double scale = weight != nullptr ? weight[col] + 1.0 : 1.0;
            out[idx] = v * inv_rms * scale;
        }
    }
}

static void
run_rmsnorm_gemma_case(marmot_test_env_t *env, const llama_rmsnorm_case_t *tc, marmot_dtype_t dtype, float tolerance) {
    const size_t elem_count = tc->rows * tc->cols;
    const size_t shape[] = {tc->rows, tc->cols};
    const bool use_f64 = (dtype == MARMOT_DTYPE_FLOAT64);
    const marmot_dtype_t golden_dtype = use_f64 ? MARMOT_DTYPE_FLOAT64 : MARMOT_DTYPE_FLOAT32;
    const size_t golden_bytes = marmot_get_dtype_traits(golden_dtype)->storage_bytes;

    marmot_tensor_t *x = marmot_tensor_create(env->ctx, shape, 2, dtype);
    marmot_tensor_t *residual = tc->residual != nullptr ? marmot_tensor_create(env->ctx, shape, 2, dtype) : nullptr;
    marmot_tensor_t *weight = tc->weight != nullptr ? marmot_tensor_create(env->ctx, &tc->cols, 1, dtype) : nullptr;
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 2, dtype);

    assert_non_null(x);
    assert_non_null(out);

    const void *input_src = use_f64 ? (const void *)tc->input_f64 : (const void *)tc->input;
    marmot_test_convert_span(env, x, golden_dtype, input_src, elem_count);
    if (residual != nullptr) {
        const void *res_src = use_f64 ? (const void *)tc->residual_f64 : (const void *)tc->residual;
        marmot_test_convert_span(env, residual, golden_dtype, res_src, elem_count);
    }
    if (weight != nullptr) {
        const void *weight_src = use_f64 ? (const void *)tc->weight_f64 : (const void *)tc->weight;
        marmot_test_convert_span(env, weight, golden_dtype, weight_src, tc->cols);
    }

    marmot_error_t err = marmot_rmsnorm_gemma(
        env->ctx,
        &(marmot_rmsnorm_desc_t){.x = x, .residual = residual, .weight = weight, .out = out, .eps = tc->epsilon}
    );
    assert_int_equal(err, MARMOT_SUCCESS);

    void *actual_host = malloc(elem_count * golden_bytes);
    void *expected_host = malloc(elem_count * golden_bytes);
    assert_non_null(actual_host);
    assert_non_null(expected_host);
    marmot_test_fetch_span(env, actual_host, golden_dtype, out, elem_count);

    if (use_f64) {
        compute_rmsnorm_gemma_expected_f64(
            tc->input_f64, tc->residual_f64, tc->weight_f64, tc->rows, tc->cols, (double)tc->epsilon,
            (double *)expected_host
        );
        const double *expected = (const double *)expected_host;
        const double *actual = (const double *)actual_host;
        for (size_t i = 0; i < elem_count; ++i) {
            double diff = fabs(actual[i] - expected[i]);
            if (diff > (double)tolerance) {
                fail_msg(
                    "Gemma RMSNorm case %s diff=%f exceeds tolerance %f (dtype=%d, index=%zu)", tc->name, diff,
                    tolerance, (int)dtype, i
                );
            }
        }
    } else {
        compute_rmsnorm_gemma_expected_f32(
            tc->input, tc->residual, tc->weight, tc->rows, tc->cols, tc->epsilon, (float *)expected_host
        );
        const float *expected = (const float *)expected_host;
        const float *actual = (const float *)actual_host;
        for (size_t i = 0; i < elem_count; ++i) {
            float diff = fabsf(actual[i] - expected[i]);
            if (diff > tolerance) {
                fail_msg(
                    "Gemma RMSNorm case %s diff=%f exceeds tolerance %f (dtype=%d, index=%zu)", tc->name, diff,
                    tolerance, (int)dtype, i
                );
            }
        }
    }

    marmot_test_tensor_destroy_all(4, out, weight, residual, x);
    free(expected_host);
    free(actual_host);
}

static void run_layernorm_case_mixed_vector_f32(
    marmot_test_env_t *env, const llama_layernorm_case_t *tc, marmot_dtype_t activation_dtype, float tolerance
) {
    const size_t elem_count = tc->rows * tc->cols;
    const size_t shape[] = {tc->rows, tc->cols};

    marmot_tensor_t *x = marmot_tensor_create(env->ctx, shape, 2, activation_dtype);
    marmot_tensor_t *residual =
        tc->residual != nullptr ? marmot_tensor_create(env->ctx, shape, 2, activation_dtype) : nullptr;
    marmot_tensor_t *weight =
        tc->weight != nullptr ? marmot_tensor_create(env->ctx, &tc->cols, 1, MARMOT_DTYPE_FLOAT32) : nullptr;
    marmot_tensor_t *bias =
        tc->bias != nullptr ? marmot_tensor_create(env->ctx, &tc->cols, 1, MARMOT_DTYPE_FLOAT32) : nullptr;
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 2, activation_dtype);

    assert_non_null(x);
    assert_non_null(out);

    marmot_test_convert_span(env, x, MARMOT_DTYPE_FLOAT32, (const void *)tc->input, elem_count);
    if (residual != nullptr) {
        marmot_test_convert_span(env, residual, MARMOT_DTYPE_FLOAT32, (const void *)tc->residual, elem_count);
    }
    if (weight != nullptr) {
        marmot_test_convert_span(env, weight, MARMOT_DTYPE_FLOAT32, (const void *)tc->weight, tc->cols);
    }
    if (bias != nullptr) {
        marmot_test_convert_span(env, bias, MARMOT_DTYPE_FLOAT32, (const void *)tc->bias, tc->cols);
    }

    marmot_error_t err = marmot_layernorm(
        env->ctx,
        &(
            marmot_layernorm_desc_t
        ){.x = x, .residual = residual, .weight = weight, .bias = bias, .out = out, .eps = tc->epsilon}
    );
    assert_int_equal(err, MARMOT_SUCCESS);

    float *actual_host = malloc(elem_count * sizeof(*actual_host));
    assert_non_null(actual_host);
    marmot_test_fetch_span(env, actual_host, MARMOT_DTYPE_FLOAT32, out, elem_count);

    for (size_t i = 0; i < elem_count; ++i) {
        float diff = fabsf(actual_host[i] - tc->expected[i]);
        if (diff > tolerance) {
            fail_msg(
                "LayerNorm mixed-vector case %s diff=%f exceeds tolerance %f (dtype=%d, index=%zu)", tc->name, diff,
                tolerance, (int)activation_dtype, i
            );
        }
    }

    marmot_test_tensor_destroy_all(5, out, bias, weight, residual, x);
    free(actual_host);
}

static void run_rmsnorm_case_mixed_vector_f32(
    marmot_test_env_t *env, const llama_rmsnorm_case_t *tc, marmot_dtype_t activation_dtype, float tolerance
) {
    const size_t elem_count = tc->rows * tc->cols;
    const size_t shape[] = {tc->rows, tc->cols};

    marmot_tensor_t *x = marmot_tensor_create(env->ctx, shape, 2, activation_dtype);
    marmot_tensor_t *residual =
        tc->residual != nullptr ? marmot_tensor_create(env->ctx, shape, 2, activation_dtype) : nullptr;
    marmot_tensor_t *weight =
        tc->weight != nullptr ? marmot_tensor_create(env->ctx, &tc->cols, 1, MARMOT_DTYPE_FLOAT32) : nullptr;
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 2, activation_dtype);

    assert_non_null(x);
    assert_non_null(out);

    marmot_test_convert_span(env, x, MARMOT_DTYPE_FLOAT32, (const void *)tc->input, elem_count);
    if (residual != nullptr) {
        marmot_test_convert_span(env, residual, MARMOT_DTYPE_FLOAT32, (const void *)tc->residual, elem_count);
    }
    if (weight != nullptr) {
        marmot_test_convert_span(env, weight, MARMOT_DTYPE_FLOAT32, (const void *)tc->weight, tc->cols);
    }

    marmot_error_t err = marmot_rmsnorm(
        env->ctx,
        &(marmot_rmsnorm_desc_t){.x = x, .residual = residual, .weight = weight, .out = out, .eps = tc->epsilon}
    );
    assert_int_equal(err, MARMOT_SUCCESS);

    float *actual_host = malloc(elem_count * sizeof(*actual_host));
    assert_non_null(actual_host);
    marmot_test_fetch_span(env, actual_host, MARMOT_DTYPE_FLOAT32, out, elem_count);

    for (size_t i = 0; i < elem_count; ++i) {
        float diff = fabsf(actual_host[i] - tc->expected[i]);
        if (diff > tolerance) {
            fail_msg(
                "RMSNorm mixed-vector case %s diff=%f exceeds tolerance %f (dtype=%d, index=%zu)", tc->name, diff,
                tolerance, (int)activation_dtype, i
            );
        }
    }

    marmot_test_tensor_destroy_all(4, out, weight, residual, x);
    free(actual_host);
}

static void run_rmsnorm_gemma_case_mixed_vector_f32(
    marmot_test_env_t *env, const llama_rmsnorm_case_t *tc, marmot_dtype_t activation_dtype, float tolerance
) {
    const size_t elem_count = tc->rows * tc->cols;
    const size_t shape[] = {tc->rows, tc->cols};

    marmot_tensor_t *x = marmot_tensor_create(env->ctx, shape, 2, activation_dtype);
    marmot_tensor_t *residual =
        tc->residual != nullptr ? marmot_tensor_create(env->ctx, shape, 2, activation_dtype) : nullptr;
    marmot_tensor_t *weight =
        tc->weight != nullptr ? marmot_tensor_create(env->ctx, &tc->cols, 1, MARMOT_DTYPE_FLOAT32) : nullptr;
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape, 2, activation_dtype);

    assert_non_null(x);
    assert_non_null(out);

    marmot_test_convert_span(env, x, MARMOT_DTYPE_FLOAT32, (const void *)tc->input, elem_count);
    if (residual != nullptr) {
        marmot_test_convert_span(env, residual, MARMOT_DTYPE_FLOAT32, (const void *)tc->residual, elem_count);
    }
    if (weight != nullptr) {
        marmot_test_convert_span(env, weight, MARMOT_DTYPE_FLOAT32, (const void *)tc->weight, tc->cols);
    }

    marmot_error_t err = marmot_rmsnorm_gemma(
        env->ctx,
        &(marmot_rmsnorm_desc_t){.x = x, .residual = residual, .weight = weight, .out = out, .eps = tc->epsilon}
    );
    assert_int_equal(err, MARMOT_SUCCESS);

    float *actual_host = malloc(elem_count * sizeof(float));
    float *expected_host = malloc(elem_count * sizeof(float));
    assert_non_null(actual_host);
    assert_non_null(expected_host);
    marmot_test_fetch_span(env, actual_host, MARMOT_DTYPE_FLOAT32, out, elem_count);
    compute_rmsnorm_gemma_expected_f32(
        tc->input, tc->residual, tc->weight, tc->rows, tc->cols, tc->epsilon, expected_host
    );

    for (size_t i = 0; i < elem_count; ++i) {
        float diff = fabsf(actual_host[i] - expected_host[i]);
        if (diff > tolerance) {
            fail_msg(
                "Gemma RMSNorm mixed-vector case %s diff=%f exceeds tolerance %f (dtype=%d, index=%zu)", tc->name, diff,
                tolerance, (int)activation_dtype, i
            );
        }
    }

    marmot_test_tensor_destroy_all(4, out, weight, residual, x);
    free(expected_host);
    free(actual_host);
}

static void test_layernorm_llama_goldens(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const marmot_dtype_t dtypes[] = {
        MARMOT_DTYPE_FLOAT32,
        MARMOT_DTYPE_FLOAT64,
        MARMOT_DTYPE_FLOAT16,
        MARMOT_DTYPE_BFLOAT16,
    };

    for (size_t d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
        const marmot_dtype_t dtype = dtypes[d];
        if (dtype == MARMOT_DTYPE_FLOAT64 && env->backend == MARMOT_BACKEND_METAL) {
            continue; // Metal does not support F64
        }
        const float tol = dtype_tolerance(dtype);
        for (size_t i = 0; i < g_llama_layernorm_case_count; ++i) {
            run_layernorm_case(env, &g_llama_layernorm_cases[i], dtype, tol);
        }
    }
}

static void test_rmsnorm_llama_goldens(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const marmot_dtype_t dtypes[] = {
        MARMOT_DTYPE_FLOAT32,
        MARMOT_DTYPE_FLOAT64,
        MARMOT_DTYPE_FLOAT16,
        MARMOT_DTYPE_BFLOAT16,
    };

    for (size_t d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
        const marmot_dtype_t dtype = dtypes[d];
        if (dtype == MARMOT_DTYPE_FLOAT64 && env->backend == MARMOT_BACKEND_METAL) {
            continue; // Metal does not support F64
        }
        const float tol = dtype_tolerance(dtype);
        for (size_t i = 0; i < g_llama_rmsnorm_case_count; ++i) {
            run_rmsnorm_case(env, &g_llama_rmsnorm_cases[i], dtype, tol);
        }
    }
}

static void test_rmsnorm_gemma_goldens(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const marmot_dtype_t dtypes[] = {
        MARMOT_DTYPE_FLOAT32,
        MARMOT_DTYPE_FLOAT64,
        MARMOT_DTYPE_FLOAT16,
        MARMOT_DTYPE_BFLOAT16,
    };

    for (size_t d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
        const marmot_dtype_t dtype = dtypes[d];
        if (dtype == MARMOT_DTYPE_FLOAT64 && env->backend == MARMOT_BACKEND_METAL) {
            continue; // Metal does not support F64
        }
        const float tol = dtype_tolerance(dtype);
        for (size_t i = 0; i < g_llama_rmsnorm_case_count; ++i) {
            run_rmsnorm_gemma_case(env, &g_llama_rmsnorm_cases[i], dtype, tol);
        }
    }
}

static void test_layernorm_llama_mixed_vector_f32_goldens(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const marmot_dtype_t dtypes[] = {
        MARMOT_DTYPE_FLOAT16,
        MARMOT_DTYPE_BFLOAT16,
    };

    for (size_t d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
        const marmot_dtype_t dtype = dtypes[d];
        const float tol = dtype_tolerance(dtype);
        for (size_t i = 0; i < g_llama_layernorm_case_count; ++i) {
            run_layernorm_case_mixed_vector_f32(env, &g_llama_layernorm_cases[i], dtype, tol);
        }
    }
}

static void test_rmsnorm_llama_mixed_vector_f32_goldens(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const marmot_dtype_t dtypes[] = {
        MARMOT_DTYPE_FLOAT16,
        MARMOT_DTYPE_BFLOAT16,
    };

    for (size_t d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
        const marmot_dtype_t dtype = dtypes[d];
        const float tol = dtype_tolerance(dtype);
        for (size_t i = 0; i < g_llama_rmsnorm_case_count; ++i) {
            run_rmsnorm_case_mixed_vector_f32(env, &g_llama_rmsnorm_cases[i], dtype, tol);
        }
    }
}

static void test_rmsnorm_gemma_mixed_vector_f32_goldens(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const marmot_dtype_t dtypes[] = {
        MARMOT_DTYPE_FLOAT16,
        MARMOT_DTYPE_BFLOAT16,
    };

    for (size_t d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
        const marmot_dtype_t dtype = dtypes[d];
        const float tol = dtype_tolerance(dtype);
        for (size_t i = 0; i < g_llama_rmsnorm_case_count; ++i) {
            run_rmsnorm_gemma_case_mixed_vector_f32(env, &g_llama_rmsnorm_cases[i], dtype, tol);
        }
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(
            test_layernorm_llama_goldens, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_rmsnorm_llama_goldens, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_rmsnorm_gemma_goldens, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_layernorm_llama_mixed_vector_f32_goldens, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_rmsnorm_llama_mixed_vector_f32_goldens, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_rmsnorm_gemma_mixed_vector_f32_goldens, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
