#include "marmot/ops/elementwise.h"
#include "marmot/ops/matmul.h"
#include "marmot/ops/unary.h"
#include "marmot/quant_block.h"

#include <stdbool.h>
#include <stdlib.h>

#include <math.h>
#include <string.h>

#include "backend/golden_data.h"
#include "backend/golden_float_ops_llama.h"
#include "backend/test_backend_utils.h"
#include "matmul_quantized_golden_cases.h"

static const matmul_golden_case_t *matmul_quantized_case_for_kind(marmot_quant_kind_t kind) {
    const size_t count = sizeof(g_matmul_quant_goldens) / sizeof(g_matmul_quant_goldens[0]);
    for (size_t i = 0; i < count; ++i) {
        if (g_matmul_quant_goldens[i].kind == kind) {
            return &g_matmul_quant_goldens[i];
        }
    }
    return nullptr;
}

#ifdef __APPLE__
static void test_restore_env(const char *name, const char *saved) {
    if (name == nullptr) {
        return;
    }
    if (saved == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, saved, 1);
    }
}

static void fill_span(float *dst, size_t count, float scale, float offset) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = offset + scale * (float)i;
    }
}

static void
reference_rope_rotate(float *dst, const float *src, size_t rows, size_t dim, const float *positions, float theta) {
    const size_t pairs = dim / 2;
    const size_t total = rows * dim;
    if (pairs == 0 || dst == nullptr || src == nullptr) {
        if (dst != nullptr && src != nullptr && total > 0) {
            memcpy(dst, src, total * sizeof(float));
        }
        return;
    }
    float *freqs = (float *)malloc(pairs * sizeof(float));
    assert_non_null(freqs);
    for (size_t i = 0; i < pairs; ++i) {
        freqs[i] = powf(theta, -((float)(2 * i) / (float)dim));
    }
    for (size_t row = 0; row < rows; ++row) {
        const float pos = positions[row];
        for (size_t pair = 0; pair < pairs; ++pair) {
            const size_t even_idx = row * dim + pair * 2;
            const size_t odd_idx = even_idx + 1;
            const float angle = pos * freqs[pair];
            const float c = cosf(angle);
            const float s = sinf(angle);
            const float even = src[even_idx];
            const float odd = src[odd_idx];
            dst[even_idx] = even * c - odd * s;
            dst[odd_idx] = even * s + odd * c;
        }
    }
    free(freqs);
}

static bool run_metal_qkv_case(
    const marmot_test_env_t *env, bool enable_packed, float *out_q, float *out_k, float *out_v, size_t N, size_t M,
    size_t K
) {
    if (env == nullptr || env->ctx == nullptr || env->backend != MARMOT_BACKEND_METAL) {
        return false;
    }
    const size_t elems_input = N * K;
    const size_t elems_weight = 3 * M * K;
    const size_t elems_bias = 3 * M;
    const size_t elems_out = N * M;
    const char *packed_env_name = "MARMOT_METAL_ENABLE_PACKED_WEIGHTS";
    const char *min_dim_env_name = "MARMOT_METAL_PACKED_MIN_DIM";
    const char *min_elements_env_name = "MARMOT_METAL_PACKED_MIN_ELEMENTS";
    const char *old_packed = getenv(packed_env_name);
    const char *old_min_dim = getenv(min_dim_env_name);
    const char *old_min_elements = getenv(min_elements_env_name);
    setenv(packed_env_name, enable_packed ? "1" : "0", 1);
    if (!enable_packed) {
        setenv(min_dim_env_name, "8192", 1);
        setenv(min_elements_env_name, "67108864", 1);
    } else {
        unsetenv(min_dim_env_name);
        unsetenv(min_elements_env_name);
    }
    marmot_context_t *ctx = env->ctx;

    size_t shape_input[] = {N, K};
    size_t shape_weight[] = {3 * M, K};
    size_t shape_out[] = {N, M};
    size_t shape_bias[] = {3 * M};

    marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape_input, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *weight = marmot_tensor_create(env->ctx, shape_weight, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *bias = marmot_tensor_create(env->ctx, shape_bias, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_q_tensor = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_k_tensor = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_v_tensor = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    if (input == nullptr || weight == nullptr || bias == nullptr || out_q_tensor == nullptr ||
        out_k_tensor == nullptr || out_v_tensor == nullptr) {
        marmot_test_tensor_destroy_all(6, out_v_tensor, out_k_tensor, out_q_tensor, bias, weight, input);
        test_restore_env(packed_env_name, old_packed);
        test_restore_env(min_dim_env_name, old_min_dim);
        test_restore_env(min_elements_env_name, old_min_elements);
        return false;
    }

    float *host_input = (float *)malloc(elems_input * sizeof(float));
    float *host_weight = (float *)malloc(elems_weight * sizeof(float));
    float *host_bias = (float *)malloc(elems_bias * sizeof(float));
    bool ok = host_input != nullptr && host_weight != nullptr && host_bias != nullptr;
    if (ok) {
        fill_span(host_input, elems_input, 0.01f, 0.1f);
        fill_span(host_weight, elems_weight, 0.001f, -0.05f);
        fill_span(host_bias, elems_bias, 0.02f, 0.0f);
        size_t bytes_input = elems_input * sizeof(float);
        size_t bytes_weight = elems_weight * sizeof(float);
        size_t bytes_bias = elems_bias * sizeof(float);
        marmot_error_t err = marmot_tensor_copy_from_host_buffer(ctx, input, host_input, bytes_input);
        err = err == MARMOT_SUCCESS ? marmot_tensor_copy_from_host_buffer(ctx, weight, host_weight, bytes_weight) : err;
        err = err == MARMOT_SUCCESS ? marmot_tensor_copy_from_host_buffer(ctx, bias, host_bias, bytes_bias) : err;
        if (err != MARMOT_SUCCESS) {
            ok = false;
        }
    }
    free(host_input);
    free(host_weight);
    free(host_bias);
    if (!ok) {
        marmot_test_tensor_destroy_all(6, out_v_tensor, out_k_tensor, out_q_tensor, bias, weight, input);
        test_restore_env(packed_env_name, old_packed);
        test_restore_env(min_dim_env_name, old_min_dim);
        test_restore_env(min_elements_env_name, old_min_elements);
        return false;
    }

    marmot_matmul_qkv_desc_t desc = {
        .input = input,
        .layout = MARMOT_QKV_LAYOUT_FUSED,
        .fused =
            {
                .weight = weight,
                .bias = bias,
            },
        .epilogue = nullptr,
        .out_q = out_q_tensor,
        .out_k = out_k_tensor,
        .out_v = out_v_tensor,
    };
    marmot_error_t status = marmot_matmul_qkv(ctx, &desc);
    if (status == MARMOT_SUCCESS) {
        size_t bytes_out = elems_out * sizeof(float);
        status = marmot_tensor_copy_to_host_buffer(ctx, out_q_tensor, out_q, bytes_out);
        status =
            status == MARMOT_SUCCESS ? marmot_tensor_copy_to_host_buffer(ctx, out_k_tensor, out_k, bytes_out) : status;
        status =
            status == MARMOT_SUCCESS ? marmot_tensor_copy_to_host_buffer(ctx, out_v_tensor, out_v, bytes_out) : status;
    }

    marmot_test_tensor_destroy_all(6, out_v_tensor, out_k_tensor, out_q_tensor, bias, weight, input);
    test_restore_env(packed_env_name, old_packed);
    test_restore_env(min_dim_env_name, old_min_dim);
    test_restore_env(min_elements_env_name, old_min_elements);
    return status == MARMOT_SUCCESS;
}
#endif

static double matmul_epilogue_tolerance(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT64:
        return 5e-7;
    case MARMOT_DTYPE_FLOAT32:
        return 5e-6;
    case MARMOT_DTYPE_FLOAT16:
    case MARMOT_DTYPE_BFLOAT16:
        return 2e-2;
    default:
        return 1e-5;
    }
}

static void exercise_linear_case(marmot_test_env_t *env, marmot_dtype_t dtype, int case_idx, float tolerance) {
    assert_true(case_idx >= 0 && (size_t)case_idx < g_llama_matmul_case_count);
    const llama_matmul_case_t *tc = &g_llama_matmul_cases[case_idx];
    const bool use_f64 = (dtype == MARMOT_DTYPE_FLOAT64);
    const marmot_dtype_t golden_dtype = use_f64 ? MARMOT_DTYPE_FLOAT64 : MARMOT_DTYPE_FLOAT32;
    const void *a_vals = use_f64 ? (const void *)tc->weight_f64 : (const void *)tc->weight;
    const void *b_vals = use_f64 ? (const void *)tc->rhs_f64 : (const void *)tc->rhs;
    const void *expected = use_f64 ? (const void *)tc->expected_f64 : (const void *)tc->expected;
    const size_t M = tc->m;
    const size_t K = tc->k;
    const size_t N = tc->n;

    // Linear convention: input(N×K) @ weight(M×K).T = output(N×M)
    // Golden data: A(M×K) @ B(K×N) = C(M×N)
    // Mapping: weight=A(M×K), input=B.T(N×K), output=C.T(N×M)
    size_t shape_weight[] = {M, K};
    size_t shape_input[] = {N, K};
    size_t shape_out[] = {N, M};

    size_t elems_weight = M * K;
    size_t elems_input = N * K;
    size_t elems_out = N * M;

    const size_t golden_bytes = marmot_get_dtype_traits(golden_dtype)->storage_bytes;
    void *input_transposed = malloc(elems_input * golden_bytes);
    void *expected_transposed = malloc(elems_out * golden_bytes);
    assert_non_null(input_transposed);
    assert_non_null(expected_transposed);
    if (use_f64) {
        const double *rhs = (const double *)b_vals;
        double *dst = (double *)input_transposed;
        for (size_t n = 0; n < N; ++n) {
            for (size_t k = 0; k < K; ++k) {
                dst[n * K + k] = rhs[k * N + n];
            }
        }
        const double *expected_src = (const double *)expected;
        double *expected_dst = (double *)expected_transposed;
        for (size_t n = 0; n < N; ++n) {
            for (size_t m = 0; m < M; ++m) {
                expected_dst[n * M + m] = expected_src[m * N + n];
            }
        }
    } else {
        const float *rhs = (const float *)b_vals;
        float *dst = (float *)input_transposed;
        for (size_t n = 0; n < N; ++n) {
            for (size_t k = 0; k < K; ++k) {
                dst[n * K + k] = rhs[k * N + n];
            }
        }
        const float *expected_src = (const float *)expected;
        float *expected_dst = (float *)expected_transposed;
        for (size_t n = 0; n < N; ++n) {
            for (size_t m = 0; m < M; ++m) {
                expected_dst[n * M + m] = expected_src[m * N + n];
            }
        }
    }

    marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape_input, 2, dtype);
    marmot_tensor_t *weight = marmot_tensor_create(env->ctx, shape_weight, 2, dtype);
    marmot_tensor_t *OUT = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
    assert_non_null(input);
    assert_non_null(weight);
    assert_non_null(OUT);

    marmot_test_convert_span(env, input, golden_dtype, input_transposed, elems_input);
    marmot_test_convert_span(env, weight, golden_dtype, a_vals, elems_weight);

    marmot_error_t err = marmot_linear(env->ctx, input, weight, nullptr, OUT);
    assert_int_equal(err, MARMOT_SUCCESS);

    void *actual_host = malloc(elems_out * golden_bytes);
    assert_non_null(actual_host);
    marmot_test_fetch_span(env, actual_host, golden_dtype, OUT, elems_out);
    if (use_f64) {
        const double *expected_host = (const double *)expected_transposed;
        const double *actual = (const double *)actual_host;
        for (size_t i = 0; i < elems_out; ++i) {
            double diff = fabs(actual[i] - expected_host[i]);
            assert_true(diff <= (double)tolerance);
        }
    } else {
        const float *expected_host = (const float *)expected_transposed;
        const float *actual = (const float *)actual_host;
        for (size_t i = 0; i < elems_out; ++i) {
            float diff = fabsf(actual[i] - expected_host[i]);
            assert_true(diff <= tolerance);
        }
    }

    marmot_test_tensor_destroy_all(3, OUT, weight, input);
    free(actual_host);
    free(expected_transposed);
    free(input_transposed);
}

static float
apply_activation_ref(marmot_device_unary_op_t activation, float value, const marmot_activation_params_t *params) {
    const float alpha = params != nullptr ? params->alpha : 0.0f;
    const float beta = params != nullptr ? params->beta : 0.0f;
    switch (activation) {
    case MARMOT_DEVICE_UNARY_IDENTITY:
        return value;
    case MARMOT_DEVICE_UNARY_RELU:
        return value > 0.0f ? value : 0.0f;
    case MARMOT_DEVICE_UNARY_GELU: {
        const float inv_sqrt2 = 0.70710678118f;
        float erf_term = erff(value * inv_sqrt2);
        return 0.5f * value * (1.0f + erf_term);
    }
    case MARMOT_DEVICE_UNARY_GELU_TANH: {
        const float k0 = 0.7978845608f;
        const float k1 = 0.044715f;
        float inner = k0 * (value + k1 * value * value * value);
        return 0.5f * value * (1.0f + tanhf(inner));
    }
    case MARMOT_DEVICE_UNARY_SILU:
        return value / (1.0f + expf(-value));
    case MARMOT_DEVICE_UNARY_SIGMOID:
        if (value >= 0.0f) {
            float z = expf(-value);
            return 1.0f / (1.0f + z);
        } else {
            float z = expf(value);
            return z / (1.0f + z);
        }
    case MARMOT_DEVICE_UNARY_TANH:
        return tanhf(value);
    case MARMOT_DEVICE_UNARY_MISH: {
        float abs_x = fabsf(value);
        float softplus = log1pf(expf(-abs_x)) + fmaxf(value, 0.0f);
        return value * tanhf(softplus);
    }
    case MARMOT_DEVICE_UNARY_ELU:
        return value >= 0.0f ? value : alpha * (expf(value) - 1.0f);
    case MARMOT_DEVICE_UNARY_SELU: {
        float inner = value >= 0.0f ? value : alpha * (expf(value) - 1.0f);
        return beta * inner;
    }
    case MARMOT_DEVICE_UNARY_LEAKY_RELU:
        return value >= 0.0f ? value : alpha * value;
    case MARMOT_DEVICE_UNARY_PRELU:
        return value >= 0.0f ? value : alpha * value;
    default:
        return value;
    }
}

static marmot_error_t apply_activation_op(
    const marmot_context_t *ctx, marmot_device_unary_op_t activation, const marmot_tensor_t *input,
    marmot_tensor_t *out, const marmot_activation_params_t *params
) {
    switch (activation) {
    case MARMOT_DEVICE_UNARY_IDENTITY:
    case MARMOT_DEVICE_UNARY_COUNT:
        return marmot_tensor_copy(out, input);
    case MARMOT_DEVICE_UNARY_RELU:
        return marmot_relu(ctx, input, out);
    case MARMOT_DEVICE_UNARY_GELU:
        return marmot_gelu(ctx, input, out);
    case MARMOT_DEVICE_UNARY_GELU_TANH:
        return marmot_gelu_tanh(ctx, input, out);
    case MARMOT_DEVICE_UNARY_SILU:
        return marmot_silu(ctx, input, out);
    case MARMOT_DEVICE_UNARY_SIGMOID:
        return marmot_sigmoid(ctx, input, out);
    case MARMOT_DEVICE_UNARY_TANH:
        return marmot_tanh(ctx, input, out);
    case MARMOT_DEVICE_UNARY_MISH:
        return marmot_mish(ctx, input, out);
    case MARMOT_DEVICE_UNARY_ELU:
        if (params == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        return marmot_elu(ctx, input, out, params->alpha);
    case MARMOT_DEVICE_UNARY_SELU:
        if (params == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        return marmot_selu(ctx, input, out, params->alpha, params->beta);
    case MARMOT_DEVICE_UNARY_LEAKY_RELU:
        if (params == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        return marmot_leaky_relu(ctx, input, out, params->alpha);
    case MARMOT_DEVICE_UNARY_PRELU:
        if (params == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        return marmot_prelu(ctx, input, out, params->alpha);
    default:
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
}

static void exercise_matmul_fused_case(
    marmot_test_env_t *env, marmot_dtype_t dtype, marmot_device_unary_op_t activation, float tolerance
) {
    assert_true(g_llama_matmul_case_count > 0);
    const llama_matmul_case_t *tc = &g_llama_matmul_cases[0];
    const bool use_f64 = (dtype == MARMOT_DTYPE_FLOAT64);
    const marmot_dtype_t golden_dtype = use_f64 ? MARMOT_DTYPE_FLOAT64 : MARMOT_DTYPE_FLOAT32;
    const void *a_vals = use_f64 ? (const void *)tc->weight_f64 : (const void *)tc->weight;
    const void *b_vals = use_f64 ? (const void *)tc->rhs_f64 : (const void *)tc->rhs;
    const void *expected_base = use_f64 ? (const void *)tc->expected_f64 : (const void *)tc->expected;
    const size_t M = tc->m;
    const size_t K = tc->k;
    const size_t N = tc->n;

    // Linear convention: input(N×K) @ weight(M×K).T = output(N×M)
    // Bias shape is (M,) in PyTorch (broadcasted over output features)
    size_t shape_weight[] = {M, K};
    size_t shape_input[] = {N, K};
    size_t shape_out[] = {N, M};
    size_t shape_bias[] = {M};

    size_t elems_weight = M * K;
    size_t elems_input = N * K;
    size_t elems_out = N * M;

    const size_t golden_bytes = marmot_get_dtype_traits(golden_dtype)->storage_bytes;
    void *input_transposed = malloc(elems_input * golden_bytes);
    void *bias_vals = malloc(M * golden_bytes);
    void *expected_out = malloc(elems_out * golden_bytes);
    assert_non_null(input_transposed);
    assert_non_null(bias_vals);
    assert_non_null(expected_out);

    if (use_f64) {
        const double *rhs = (const double *)b_vals;
        double *dst = (double *)input_transposed;
        for (size_t n = 0; n < N; ++n) {
            for (size_t k = 0; k < K; ++k) {
                dst[n * K + k] = rhs[k * N + n];
            }
        }
        double *bias = (double *)bias_vals;
        for (size_t m = 0; m < M; ++m) {
            bias[m] = 0.1 * (double)(m + 1);
        }
        const double *expected_src = (const double *)expected_base;
        double *expected_dst = (double *)expected_out;
        for (size_t n = 0; n < N; ++n) {
            for (size_t m = 0; m < M; ++m) {
                double base_val = expected_src[m * N + n];
                double with_bias = base_val + bias[m];
                expected_dst[n * M + m] = (double)apply_activation_ref(activation, (float)with_bias, nullptr);
            }
        }
    } else {
        const float *rhs = (const float *)b_vals;
        float *dst = (float *)input_transposed;
        for (size_t n = 0; n < N; ++n) {
            for (size_t k = 0; k < K; ++k) {
                dst[n * K + k] = rhs[k * N + n];
            }
        }
        float *bias = (float *)bias_vals;
        for (size_t m = 0; m < M; ++m) {
            bias[m] = 0.1f * (float)(m + 1);
        }
        const float *expected_src = (const float *)expected_base;
        float *expected_dst = (float *)expected_out;
        for (size_t n = 0; n < N; ++n) {
            for (size_t m = 0; m < M; ++m) {
                float base_val = expected_src[m * N + n];
                float with_bias = base_val + bias[m];
                expected_dst[n * M + m] = apply_activation_ref(activation, with_bias, nullptr);
            }
        }
    }

    marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape_input, 2, dtype);
    marmot_tensor_t *weight = marmot_tensor_create(env->ctx, shape_weight, 2, dtype);
    marmot_tensor_t *BIAS = marmot_tensor_create(env->ctx, shape_bias, 1, dtype);
    marmot_tensor_t *OUT = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
    assert_non_null(input);
    assert_non_null(weight);
    assert_non_null(BIAS);
    assert_non_null(OUT);

    marmot_test_convert_span(env, input, golden_dtype, input_transposed, elems_input);
    marmot_test_convert_span(env, weight, golden_dtype, a_vals, elems_weight);
    marmot_test_convert_span(env, BIAS, golden_dtype, bias_vals, M);

    marmot_matmul_epilogue_t epilogue = {
        .bias = BIAS,
    };

    marmot_error_t err = marmot_linear(env->ctx, input, weight, &epilogue, OUT);
    assert_int_equal(err, MARMOT_SUCCESS);

    marmot_tensor_t *activated = nullptr;
    marmot_tensor_t *result = OUT;
    if (activation != MARMOT_DEVICE_UNARY_IDENTITY) {
        activated = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
        assert_non_null(activated);
        assert_int_equal(apply_activation_op(env->ctx, activation, OUT, activated, nullptr), MARMOT_SUCCESS);
        result = activated;
    }

    void *actual_host = malloc(elems_out * golden_bytes);
    assert_non_null(actual_host);
    marmot_test_fetch_span(env, actual_host, golden_dtype, result, elems_out);

    if (use_f64) {
        const double *expected_host = (const double *)expected_out;
        const double *actual = (const double *)actual_host;
        for (size_t i = 0; i < elems_out; ++i) {
            double diff = fabs(actual[i] - expected_host[i]);
            assert_true(diff <= (double)tolerance);
        }
    } else {
        const float *expected_host = (const float *)expected_out;
        const float *actual = (const float *)actual_host;
        for (size_t i = 0; i < elems_out; ++i) {
            float diff = fabsf(actual[i] - expected_host[i]);
            assert_true(diff <= tolerance);
        }
    }

    marmot_test_tensor_destroy_all(5, activated, OUT, BIAS, weight, input);
    free(input_transposed);
    free(bias_vals);
    free(expected_out);
    free(actual_host);
}

static void test_matmul_fused_bias_identity(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    exercise_matmul_fused_case(env, MARMOT_DTYPE_FLOAT32, MARMOT_DEVICE_UNARY_IDENTITY, 5e-6f);
}

static void test_matmul_fused_bias_identity_f64(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend == MARMOT_BACKEND_METAL) {
        marmot_context_t *cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
        if (cpu_ctx == nullptr) {
            skip();
            return;
        }
        marmot_test_env_t cpu_env = {
            .backend = MARMOT_BACKEND_CPU,
            .ctx = cpu_ctx,
        };
        exercise_matmul_fused_case(&cpu_env, MARMOT_DTYPE_FLOAT64, MARMOT_DEVICE_UNARY_IDENTITY, 5e-6f);
        marmot_destroy(cpu_ctx);
        return;
    }
    exercise_matmul_fused_case(env, MARMOT_DTYPE_FLOAT64, MARMOT_DEVICE_UNARY_IDENTITY, 5e-6f);
}

static void test_matmul_fused_bias_gelu(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_METAL) {
        marmot_context_t *metal_ctx = marmot_init(MARMOT_BACKEND_METAL);
        if (metal_ctx == nullptr) {
            skip();
            return;
        }
        marmot_test_env_t metal_env = {
            .backend = MARMOT_BACKEND_METAL,
            .ctx = metal_ctx,
        };
        exercise_matmul_fused_case(&metal_env, MARMOT_DTYPE_FLOAT32, MARMOT_DEVICE_UNARY_GELU, 2e-4f);
        marmot_destroy(metal_ctx);
        return;
    }
    exercise_matmul_fused_case(env, MARMOT_DTYPE_FLOAT32, MARMOT_DEVICE_UNARY_GELU, 2e-4f);
}

static void test_matmul_fused_bias_relu_f16(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_METAL) {
        marmot_context_t *metal_ctx = marmot_init(MARMOT_BACKEND_METAL);
        if (metal_ctx == nullptr) {
            skip();
            return;
        }
        marmot_test_env_t metal_env = {
            .backend = MARMOT_BACKEND_METAL,
            .ctx = metal_ctx,
        };
        exercise_matmul_fused_case(&metal_env, MARMOT_DTYPE_FLOAT16, MARMOT_DEVICE_UNARY_RELU, 2e-2f);
        marmot_destroy(metal_ctx);
        return;
    }
    exercise_matmul_fused_case(env, MARMOT_DTYPE_FLOAT16, MARMOT_DEVICE_UNARY_RELU, 2e-2f);
}

static void test_matmul_fused_goldens(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    marmot_test_env_t test_env = *env;
    marmot_context_t *metal_ctx = nullptr;
    if (env->backend != MARMOT_BACKEND_METAL) {
        metal_ctx = marmot_init(MARMOT_BACKEND_METAL);
        if (metal_ctx == nullptr) {
            skip();
            return;
        }
        test_env.backend = MARMOT_BACKEND_METAL;
        test_env.ctx = metal_ctx;
    }
    const marmot_dtype_t dtypes[] = {
        MARMOT_DTYPE_FLOAT32,
        MARMOT_DTYPE_FLOAT16,
        MARMOT_DTYPE_BFLOAT16,
        MARMOT_DTYPE_FLOAT64,
    };

    bool ran_any = false;
    for (size_t d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
        const marmot_dtype_t dtype = dtypes[d];
        if (test_env.backend == MARMOT_BACKEND_METAL && dtype == MARMOT_DTYPE_FLOAT64) {
            continue;
        }
        const bool use_f64 = (dtype == MARMOT_DTYPE_FLOAT64);
        const marmot_dtype_t golden_dtype = use_f64 ? MARMOT_DTYPE_FLOAT64 : MARMOT_DTYPE_FLOAT32;
        const double tolerance = matmul_epilogue_tolerance(dtype);

        for (size_t i = 0; i < g_matmul_fused_case_count; ++i) {
            const typeof(g_matmul_fused_cases[0]) *tc = &g_matmul_fused_cases[i];
            const size_t shape_input[2] = {tc->n, tc->k};
            const size_t shape_weight[2] = {tc->m, tc->k};
            const size_t shape_out[2] = {tc->n, tc->m};
            const size_t elem_count = tc->n * tc->m;

            marmot_tensor_t *input = marmot_tensor_create(test_env.ctx, shape_input, 2, dtype);
            marmot_tensor_t *weight = marmot_tensor_create(test_env.ctx, shape_weight, 2, dtype);
            marmot_tensor_t *out = marmot_tensor_create(test_env.ctx, shape_out, 2, dtype);
            marmot_tensor_t *bias = nullptr;
            marmot_tensor_t *residual = nullptr;

            assert_non_null(input);
            assert_non_null(weight);
            assert_non_null(out);

            marmot_test_convert_span(
                &test_env, input, golden_dtype, use_f64 ? (const void *)tc->input_f64 : (const void *)tc->input,
                tc->n * tc->k
            );
            marmot_test_convert_span(
                &test_env, weight, golden_dtype, use_f64 ? (const void *)tc->weight_f64 : (const void *)tc->weight,
                tc->m * tc->k
            );

            if (tc->has_bias) {
                bias = marmot_tensor_create(test_env.ctx, &tc->m, 1, dtype);
                assert_non_null(bias);
                marmot_test_convert_span(
                    &test_env, bias, golden_dtype, use_f64 ? (const void *)tc->bias_f64 : (const void *)tc->bias, tc->m
                );
            }
            if (tc->has_residual) {
                residual = marmot_tensor_create(test_env.ctx, shape_out, 2, dtype);
                assert_non_null(residual);
                marmot_test_convert_span(
                    &test_env, residual, golden_dtype,
                    use_f64 ? (const void *)tc->residual_f64 : (const void *)tc->residual, elem_count
                );
            }

            marmot_matmul_epilogue_t ep = {
                .bias = bias,
            };

            const marmot_matmul_epilogue_t *ep_ptr = bias != nullptr ? &ep : nullptr;
            marmot_error_t err = marmot_linear(test_env.ctx, input, weight, ep_ptr, out);
            if (err == MARMOT_ERROR_NOT_IMPLEMENTED || err == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
                marmot_test_tensor_destroy_all(5, out, residual, bias, weight, input);
                continue;
            }
            assert_int_equal(err, MARMOT_SUCCESS);

            marmot_tensor_t *activated = nullptr;
            marmot_tensor_t *residual_out = nullptr;
            marmot_tensor_t *result = out;
            if (tc->activation != MARMOT_DEVICE_UNARY_IDENTITY) {
                activated = marmot_tensor_create(test_env.ctx, shape_out, 2, dtype);
                assert_non_null(activated);
                marmot_error_t act_status =
                    apply_activation_op(test_env.ctx, tc->activation, result, activated, nullptr);
                if (act_status == MARMOT_ERROR_NOT_IMPLEMENTED || act_status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
                    marmot_test_tensor_destroy_all(7, activated, residual_out, out, residual, bias, weight, input);
                    continue;
                }
                assert_int_equal(act_status, MARMOT_SUCCESS);
                result = activated;
            }
            if (residual != nullptr) {
                residual_out = marmot_tensor_create(test_env.ctx, shape_out, 2, dtype);
                assert_non_null(residual_out);
                marmot_error_t add_status = marmot_add(test_env.ctx, result, residual, residual_out);
                if (add_status == MARMOT_ERROR_NOT_IMPLEMENTED || add_status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
                    marmot_test_tensor_destroy_all(7, activated, residual_out, out, residual, bias, weight, input);
                    continue;
                }
                assert_int_equal(add_status, MARMOT_SUCCESS);
                result = residual_out;
            }
            ran_any = true;

            const size_t elem_bytes = marmot_get_dtype_traits(golden_dtype)->storage_bytes * elem_count;
            void *actual = malloc(elem_bytes);
            assert_non_null(actual);
            marmot_test_fetch_span(&test_env, actual, golden_dtype, result, elem_count);
            const void *expected = use_f64 ? (const void *)tc->expected_f64 : (const void *)tc->expected;

            if (use_f64) {
                const double *act = (const double *)actual;
                const double *exp = (const double *)expected;
                for (size_t idx = 0; idx < elem_count; ++idx) {
                    double diff = fabs(act[idx] - exp[idx]);
                    if (diff > tolerance) {
                        fail_msg(
                            "Matmul fused golden %s (dtype=%d) diff=%f exceeds tolerance %f at index %zu", tc->name,
                            (int)dtype, diff, tolerance, idx
                        );
                    }
                }
            } else {
                const float *act = (const float *)actual;
                const float *exp = (const float *)expected;
                for (size_t idx = 0; idx < elem_count; ++idx) {
                    float diff = fabsf(act[idx] - exp[idx]);
                    if (diff > (float)tolerance) {
                        fail_msg(
                            "Matmul fused golden %s (dtype=%d) diff=%f exceeds tolerance %f at index %zu", tc->name,
                            (int)dtype, diff, tolerance, idx
                        );
                    }
                }
            }

            free(actual);
            marmot_test_tensor_destroy_all(7, activated, residual_out, out, residual, bias, weight, input);
        }
    }
    if (metal_ctx != nullptr) {
        marmot_destroy(metal_ctx);
    }
    if (!ran_any) {
        skip();
    }
}

static void test_matmul_rope_epilogue_matches_reference(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const llama_matmul_case_t *tc = &g_llama_matmul_cases[0];
    const size_t shape_input[2] = {tc->n, tc->k};
    const size_t shape_weight[2] = {tc->m, tc->k};
    const size_t shape_out[2] = {tc->n, tc->m};

    marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape_input, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *weight = marmot_tensor_create(env->ctx, shape_weight, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_fused = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_ref = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(input);
    assert_non_null(weight);
    assert_non_null(out_fused);
    assert_non_null(out_ref);

    marmot_test_convert_span(env, input, MARMOT_DTYPE_FLOAT32, tc->rhs, tc->n * tc->k);
    marmot_test_convert_span(env, weight, MARMOT_DTYPE_FLOAT32, tc->weight, tc->m * tc->k);

    size_t positions_shape[1] = {tc->n};
    marmot_tensor_t *positions = marmot_tensor_create(nullptr, positions_shape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(positions);
    int32_t *pos_data = (int32_t *)positions->data;
    for (size_t i = 0; i < tc->n; ++i) {
        pos_data[i] = (int32_t)i;
    }

    marmot_rope_params_t rope = marmot_rope_params_default();
    rope.positions = positions;
    rope.theta = 10000.0f;
    rope.apply_to_q = true;
    rope.apply_to_k = true;

    assert_int_equal(marmot_linear(env->ctx, input, weight, nullptr, out_ref), MARMOT_SUCCESS);
    assert_int_equal(marmot_rope(env->ctx, out_ref, &rope, out_fused), MARMOT_SUCCESS);

    float *base_host = (float *)malloc(tc->n * tc->m * sizeof(float));
    float *fused_host = (float *)malloc(tc->n * tc->m * sizeof(float));
    assert_non_null(base_host);
    assert_non_null(fused_host);
    marmot_test_fetch_f32_span(env, base_host, out_ref, tc->n * tc->m);
    marmot_test_fetch_f32_span(env, fused_host, out_fused, tc->n * tc->m);

    float positions_f32[tc->n];
    for (size_t i = 0; i < tc->n; ++i) {
        positions_f32[i] = (float)pos_data[i];
    }
    float *expected = (float *)malloc(tc->n * tc->m * sizeof(float));
    assert_non_null(expected);
    reference_rope_rotate(expected, base_host, tc->n, tc->m, positions_f32, rope.theta);

    const float tol = 5e-5f;
    for (size_t i = 0; i < tc->n * tc->m; ++i) {
        float diff = fabsf(fused_host[i] - expected[i]);
        assert_true(diff <= tol);
    }

    free(expected);
    free(base_host);
    free(fused_host);
    marmot_test_tensor_destroy_all(4, out_ref, out_fused, weight, input);
    marmot_tensor_destroy(positions);
}

static void test_matmul_quantized_rope_epilogue(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const matmul_golden_case_t *tc = matmul_quantized_default_case();
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(tc->kind);
    assert_non_null(traits);

    size_t shape_input[2] = {tc->N, tc->K};
    size_t shape_weight[2] = {tc->M, tc->K};
    size_t shape_out[2] = {tc->N, tc->M};

    marmot_tensor_t *input = marmot_test_tensor_from_array(env, shape_input, 2, tc->input_f32);
    marmot_tensor_t *weight = marmot_tensor_create_quantized(env->ctx, shape_weight, 2, tc->kind);
    assert_non_null(weight);
    assert_int_equal(marmot_tensor_size_bytes(weight), tc->weight_bytes);
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, weight, tc->weights, tc->weight_bytes), MARMOT_SUCCESS
    );
    marmot_tensor_t *out_fused = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_ref = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(out_fused);
    assert_non_null(out_ref);

    size_t positions_shape[1] = {tc->N};
    marmot_tensor_t *positions = marmot_tensor_create(nullptr, positions_shape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(positions);
    int32_t *pos_data = (int32_t *)positions->data;
    for (size_t i = 0; i < tc->N; ++i) {
        pos_data[i] = (int32_t)i;
    }

    marmot_rope_params_t rope = marmot_rope_params_default();
    rope.positions = positions;
    rope.theta = 10000.0f;
    rope.apply_to_q = true;
    rope.apply_to_k = true;

    assert_int_equal(marmot_linear(env->ctx, input, weight, nullptr, out_ref), MARMOT_SUCCESS);
    assert_int_equal(marmot_rope(env->ctx, out_ref, &rope, out_fused), MARMOT_SUCCESS);

    float *base_host = (float *)malloc(tc->N * tc->M * sizeof(float));
    float *fused_host = (float *)malloc(tc->N * tc->M * sizeof(float));
    assert_non_null(base_host);
    assert_non_null(fused_host);
    marmot_test_fetch_f32_span(env, base_host, out_ref, tc->N * tc->M);
    marmot_test_fetch_f32_span(env, fused_host, out_fused, tc->N * tc->M);

    float positions_f32[tc->N];
    for (size_t i = 0; i < tc->N; ++i) {
        positions_f32[i] = (float)pos_data[i];
    }
    float *expected = (float *)malloc(tc->N * tc->M * sizeof(float));
    assert_non_null(expected);
    reference_rope_rotate(expected, base_host, tc->N, tc->M, positions_f32, rope.theta);

    const float abs_tol = 2e-5f;
    const float rel_tol = 5e-5f;
    for (size_t i = 0; i < tc->N * tc->M; ++i) {
        const float diff = fabsf(fused_host[i] - expected[i]);
        const float allowed = fabsf(expected[i]) * rel_tol + abs_tol;
        if (diff > allowed) {
            fail_msg(
                "Quantized RoPE mismatch idx=%zu diff=%e allowed=%e expected=%e fused=%e", i, diff, allowed,
                expected[i], fused_host[i]
            );
        }
    }

    free(expected);
    free(base_host);
    free(fused_host);
    marmot_test_tensor_destroy_all(4, out_ref, out_fused, weight, input);
    marmot_tensor_destroy(positions);
}

static void test_rope_rotation_pattern_matches_reference(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_CPU) {
        return;
    }
    const size_t rows = 4;
    const size_t dim = 8;
    size_t shape[2] = {rows, dim};
    marmot_tensor_t *tensor = marmot_tensor_create(env->ctx, shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(tensor);
    float host_input[rows * dim];
    for (size_t i = 0; i < rows * dim; ++i) {
        host_input[i] = 0.01f * (float)(i + 1);
    }
    marmot_test_convert_span(env, tensor, MARMOT_DTYPE_FLOAT32, host_input, rows * dim);

    size_t pos_shape[1] = {rows};
    marmot_tensor_t *positions = marmot_tensor_create(nullptr, pos_shape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(positions);
    int32_t *pos_data = (int32_t *)positions->data;
    for (size_t i = 0; i < rows; ++i) {
        pos_data[i] = (int32_t)(i * 3);
    }

    marmot_rope_params_t rope = marmot_rope_params_default();
    rope.positions = positions;
    rope.theta = 10000.0f;
    rope.apply_to_q = true;
    rope.apply_to_k = false;

    assert_int_equal(marmot_rope(env->ctx, tensor, &rope, tensor), MARMOT_SUCCESS);

    float actual[rows * dim];
    marmot_test_fetch_f32_span(env, actual, tensor, rows * dim);

    float positions_f32[rows];
    for (size_t i = 0; i < rows; ++i) {
        positions_f32[i] = (float)pos_data[i];
    }
    float expected[rows * dim];
    reference_rope_rotate(expected, host_input, rows, dim, positions_f32, rope.theta);

    const float tol = 5e-6f;
    for (size_t i = 0; i < rows * dim; ++i) {
        float diff = fabsf(actual[i] - expected[i]);
        assert_true(diff <= tol);
    }

    marmot_tensor_destroy(positions);
    marmot_test_tensor_destroy_all(1, tensor);
}

static void test_matmul_qkv_quantized_separate_backend(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_CPU && env->backend != MARMOT_BACKEND_METAL) {
        skip();
        return;
    }

    const matmul_golden_case_t *tc = matmul_quantized_default_case();
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(tc->kind);
    assert_non_null(traits);

    size_t shape_input[2] = {tc->N, tc->K};
    size_t shape_weight[2] = {tc->M, tc->K};
    size_t shape_out[2] = {tc->N, tc->M};

    marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape_input, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *wq = marmot_tensor_create_quantized(env->ctx, shape_weight, 2, tc->kind);
    marmot_tensor_t *wk = marmot_tensor_create_quantized(env->ctx, shape_weight, 2, tc->kind);
    marmot_tensor_t *wv = marmot_tensor_create_quantized(env->ctx, shape_weight, 2, tc->kind);
    marmot_tensor_t *out_q = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_k = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_v = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(input);
    assert_non_null(wq);
    assert_non_null(wk);
    assert_non_null(wv);
    assert_non_null(out_q);
    assert_non_null(out_k);
    assert_non_null(out_v);

    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, input, tc->input_f32, tc->N * tc->K * sizeof(float)),
        MARMOT_SUCCESS
    );
    marmot_tensor_t *weights[3] = {wq, wk, wv};
    for (size_t i = 0; i < 3; ++i) {
        assert_int_equal(marmot_tensor_size_bytes(weights[i]), tc->weight_bytes);
        assert_int_equal(
            marmot_tensor_copy_from_host_buffer(env->ctx, weights[i], tc->weights, tc->weight_bytes), MARMOT_SUCCESS
        );
    }

    marmot_matmul_qkv_desc_t desc = marmot_matmul_qkv_desc_default();
    desc.input = input;
    desc.layout = MARMOT_QKV_LAYOUT_SEPARATE;
    desc.separate.wq = wq;
    desc.separate.wk = wk;
    desc.separate.wv = wv;
    desc.out_q = out_q;
    desc.out_k = out_k;
    desc.out_v = out_v;

    marmot_error_t qkv_status = marmot_matmul_qkv(env->ctx, &desc);
    if (qkv_status == MARMOT_ERROR_NOT_IMPLEMENTED) {
        marmot_test_tensor_destroy_all(7, out_v, out_k, out_q, wv, wk, wq, input);
        skip();
        return;
    }
    assert_int_equal(qkv_status, MARMOT_SUCCESS);

    const size_t out_elems = tc->N * tc->M;
    float *host_q = (float *)malloc(out_elems * sizeof(float));
    float *host_k = (float *)malloc(out_elems * sizeof(float));
    float *host_v = (float *)malloc(out_elems * sizeof(float));
    assert_non_null(host_q);
    assert_non_null(host_k);
    assert_non_null(host_v);
    assert_int_equal(
        marmot_tensor_copy_to_host_buffer(env->ctx, out_q, host_q, out_elems * sizeof(float)), MARMOT_SUCCESS
    );
    assert_int_equal(
        marmot_tensor_copy_to_host_buffer(env->ctx, out_k, host_k, out_elems * sizeof(float)), MARMOT_SUCCESS
    );
    assert_int_equal(
        marmot_tensor_copy_to_host_buffer(env->ctx, out_v, host_v, out_elems * sizeof(float)), MARMOT_SUCCESS
    );

    const bool metal_basic_qkv = env->backend == MARMOT_BACKEND_METAL &&
        (tc->kind == MARMOT_QUANT_KIND_Q4_0 || tc->kind == MARMOT_QUANT_KIND_Q4_1 ||
         tc->kind == MARMOT_QUANT_KIND_Q5_0 || tc->kind == MARMOT_QUANT_KIND_Q5_1 ||
         tc->kind == MARMOT_QUANT_KIND_Q8_0 || tc->kind == MARMOT_QUANT_KIND_Q8_1);
    const float *expected = metal_basic_qkv ? tc->expected_f16 : tc->expected_f32;
    const float tol = metal_basic_qkv ? 1e-2f : 5e-5f;
    for (size_t i = 0; i < out_elems; ++i) {
        assert_float_equal(host_q[i], expected[i], tol);
        assert_float_equal(host_k[i], expected[i], tol);
        assert_float_equal(host_v[i], expected[i], tol);
    }

    free(host_q);
    free(host_k);
    free(host_v);
    marmot_test_tensor_destroy_all(7, out_v, out_k, out_q, wv, wk, wq, input);
}

static void test_matmul_rope_epilogue_matches_reference_backend(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_CPU && env->backend != MARMOT_BACKEND_METAL) {
        skip();
        return;
    }

    marmot_context_t *cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(cpu_ctx);

    const size_t N = 3;
    const size_t K = 5;
    const size_t M = 6;
    size_t shape_input[2] = {N, K};
    size_t shape_weight[2] = {M, K};
    size_t shape_out[2] = {N, M};

    float host_input[N * K];
    float host_weight[M * K];
    for (size_t i = 0; i < N * K; ++i) {
        host_input[i] = 0.01f * (float)(i + 1);
    }
    for (size_t i = 0; i < M * K; ++i) {
        host_weight[i] = 0.02f * (float)(i + 2);
    }

    marmot_tensor_t *backend_input = marmot_tensor_create(env->ctx, shape_input, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *backend_weight = marmot_tensor_create(env->ctx, shape_weight, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *backend_out = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(backend_input);
    assert_non_null(backend_weight);
    assert_non_null(backend_out);
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, backend_input, host_input, sizeof(host_input)), MARMOT_SUCCESS
    );
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, backend_weight, host_weight, sizeof(host_weight)), MARMOT_SUCCESS
    );

    marmot_tensor_t *cpu_input = marmot_tensor_create(nullptr, shape_input, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *cpu_weight = marmot_tensor_create(nullptr, shape_weight, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *cpu_out = marmot_tensor_create(nullptr, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(cpu_input);
    assert_non_null(cpu_weight);
    assert_non_null(cpu_out);
    memcpy(cpu_input->data, host_input, sizeof(host_input));
    memcpy(cpu_weight->data, host_weight, sizeof(host_weight));

    size_t pos_shape[1] = {N};
    marmot_tensor_t *positions = marmot_tensor_create(nullptr, pos_shape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(positions);
    int32_t *pos_data = (int32_t *)positions->data;
    for (size_t i = 0; i < N; ++i) {
        pos_data[i] = (int32_t)(i * 2);
    }

    marmot_rope_params_t rope = marmot_rope_params_default();
    rope.positions = positions;
    rope.theta = 10000.0f;
    rope.apply_to_q = true;
    rope.apply_to_k = true;

    marmot_error_t backend_status = marmot_linear(env->ctx, backend_input, backend_weight, nullptr, backend_out);
    if (backend_status == MARMOT_ERROR_NOT_IMPLEMENTED) {
        marmot_tensor_destroy(positions);
        marmot_test_tensor_destroy_all(3, cpu_out, cpu_weight, cpu_input);
        marmot_test_tensor_destroy_all(3, backend_out, backend_weight, backend_input);
        marmot_destroy(cpu_ctx);
        skip();
        return;
    }
    assert_int_equal(backend_status, MARMOT_SUCCESS);
    assert_int_equal(marmot_linear(cpu_ctx, cpu_input, cpu_weight, nullptr, cpu_out), MARMOT_SUCCESS);

    marmot_error_t backend_rope_status = marmot_rope(env->ctx, backend_out, &rope, backend_out);
    if (backend_rope_status == MARMOT_ERROR_NOT_IMPLEMENTED || backend_rope_status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        marmot_tensor_destroy(positions);
        marmot_test_tensor_destroy_all(3, cpu_out, cpu_weight, cpu_input);
        marmot_test_tensor_destroy_all(3, backend_out, backend_weight, backend_input);
        marmot_destroy(cpu_ctx);
        skip();
        return;
    }
    assert_int_equal(backend_rope_status, MARMOT_SUCCESS);
    marmot_error_t cpu_rope_status = marmot_rope(cpu_ctx, cpu_out, &rope, cpu_out);
    if (cpu_rope_status == MARMOT_ERROR_NOT_IMPLEMENTED || cpu_rope_status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        marmot_tensor_destroy(positions);
        marmot_test_tensor_destroy_all(3, cpu_out, cpu_weight, cpu_input);
        marmot_test_tensor_destroy_all(3, backend_out, backend_weight, backend_input);
        marmot_destroy(cpu_ctx);
        skip();
        return;
    }
    assert_int_equal(cpu_rope_status, MARMOT_SUCCESS);

    float backend_host[N * M];
    float cpu_host[N * M];
    assert_int_equal(
        marmot_tensor_copy_to_host_buffer(env->ctx, backend_out, backend_host, sizeof(backend_host)), MARMOT_SUCCESS
    );
    memcpy(cpu_host, cpu_out->data, sizeof(cpu_host));

    const float tol = 1e-4f;
    for (size_t i = 0; i < N * M; ++i) {
        float diff = fabsf(backend_host[i] - cpu_host[i]);
        assert_true(diff <= tol);
    }

    marmot_tensor_destroy(positions);
    marmot_test_tensor_destroy_all(3, cpu_out, cpu_weight, cpu_input);
    marmot_test_tensor_destroy_all(3, backend_out, backend_weight, backend_input);
    marmot_destroy(cpu_ctx);
}

static void run_matmul_qkv_reference_case(marmot_test_env_t *env, size_t N, size_t K, size_t M, bool include_bias) {
    const marmot_dtype_t dtype = MARMOT_DTYPE_FLOAT32;
    const marmot_dtype_t host_dtype = MARMOT_DTYPE_FLOAT32;
    const size_t shape_input[2] = {N, K};
    const size_t shape_weight[2] = {3 * M, K};
    const size_t fused_rows = 3 * M;
    const size_t shape_out[2] = {N, M};
    const size_t shape_bias[1] = {fused_rows};
    const size_t elem_count = N * M;

    marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape_input, 2, dtype);
    marmot_tensor_t *weight = marmot_tensor_create(env->ctx, shape_weight, 2, dtype);
    marmot_tensor_t *bias = include_bias ? marmot_tensor_create(env->ctx, shape_bias, 1, dtype) : nullptr;
    marmot_tensor_t *out_q = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
    marmot_tensor_t *out_k = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
    marmot_tensor_t *out_v = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
    marmot_tensor_t *ref_q = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
    marmot_tensor_t *ref_k = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
    marmot_tensor_t *ref_v = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
    assert_non_null(input);
    assert_non_null(weight);
    assert_non_null(out_q);
    assert_non_null(out_k);
    assert_non_null(out_v);
    assert_non_null(ref_q);
    assert_non_null(ref_k);
    assert_non_null(ref_v);
    if (include_bias) {
        assert_non_null(bias);
    }

    float *host_input = (float *)malloc(N * K * sizeof(float));
    float *host_weight = (float *)malloc(3 * M * K * sizeof(float));
    float *host_bias = include_bias ? (float *)malloc(3 * M * sizeof(float)) : nullptr;
    assert_non_null(host_input);
    assert_non_null(host_weight);
    if (include_bias) {
        assert_non_null(host_bias);
    }

    for (size_t i = 0; i < N * K; ++i) {
        host_input[i] = sinf((float)(i + 1) * 0.013f) + 0.25f * cosf((float)(i + 3) * 0.007f);
    }
    for (size_t j = 0; j < 3 * M * K; ++j) {
        host_weight[j] = cosf((float)(j + 5) * 0.011f) - 0.5f * sinf((float)(j + 2) * 0.017f);
    }
    if (include_bias) {
        for (size_t b = 0; b < 3 * M; ++b) {
            host_bias[b] = 0.05f * sinf((float)(b + 1) * 0.1f);
        }
    }

    marmot_test_convert_span(env, input, host_dtype, host_input, N * K);
    marmot_test_convert_span(env, weight, host_dtype, host_weight, 3 * M * K);
    if (include_bias) {
        marmot_test_convert_span(env, bias, host_dtype, host_bias, 3 * M);
    }

    marmot_matmul_qkv_desc_t desc = {
        .input = input,
        .layout = MARMOT_QKV_LAYOUT_FUSED,
        .fused =
            {
                .weight = weight,
                .bias = bias,
            },
        .out_q = out_q,
        .out_k = out_k,
        .out_v = out_v,
    };

    assert_int_equal(marmot_matmul_qkv(env->ctx, &desc), MARMOT_SUCCESS);

    marmot_tensor_t *ref_outputs[3] = {ref_q, ref_k, ref_v};
    const size_t row_stride = weight->shape.strides[0];
    const size_t bias_stride = include_bias ? bias->shape.strides[0] : 0;
    const size_t element_size = marmot_dtype_size(dtype);
    const size_t bias_element_size = include_bias ? marmot_dtype_size(dtype) : 0;
    const char *weight_bytes = (const char *)weight->data;
    const char *bias_bytes = include_bias ? (const char *)bias->data : nullptr;

    for (size_t slice = 0; slice < 3; ++slice) {
        marmot_tensor_t weight_view = *weight;
        weight_view.shape.shape[0] = M;
        weight_view.shape.shape[1] = K;
        weight_view.data = (void *)(weight_bytes + slice * M * row_stride * element_size);

        marmot_matmul_epilogue_t ep = {
            .bias = nullptr,
        };

        marmot_tensor_t bias_view = {0};
        if (include_bias) {
            bias_view = *bias;
            bias_view.shape.shape[0] = M;
            bias_view.data = (void *)(bias_bytes + slice * M * bias_stride * bias_element_size);
            ep.bias = &bias_view;
        }

        assert_int_equal(
            marmot_linear(env->ctx, input, &weight_view, ep.bias != nullptr ? &ep : nullptr, ref_outputs[slice]),
            MARMOT_SUCCESS
        );
    }

    float *fused_q = (float *)malloc(elem_count * sizeof(float));
    float *fused_k = (float *)malloc(elem_count * sizeof(float));
    float *fused_v = (float *)malloc(elem_count * sizeof(float));
    float *ref_q_host = (float *)malloc(elem_count * sizeof(float));
    float *ref_k_host = (float *)malloc(elem_count * sizeof(float));
    float *ref_v_host = (float *)malloc(elem_count * sizeof(float));
    assert_non_null(fused_q);
    assert_non_null(fused_k);
    assert_non_null(fused_v);
    assert_non_null(ref_q_host);
    assert_non_null(ref_k_host);
    assert_non_null(ref_v_host);

    marmot_test_fetch_span(env, fused_q, host_dtype, out_q, elem_count);
    marmot_test_fetch_span(env, fused_k, host_dtype, out_k, elem_count);
    marmot_test_fetch_span(env, fused_v, host_dtype, out_v, elem_count);
    marmot_test_fetch_span(env, ref_q_host, host_dtype, ref_q, elem_count);
    marmot_test_fetch_span(env, ref_k_host, host_dtype, ref_k, elem_count);
    marmot_test_fetch_span(env, ref_v_host, host_dtype, ref_v, elem_count);

    const float tolerance = 5e-5f;
    for (size_t idx = 0; idx < elem_count; ++idx) {
        float diff_q = fabsf(fused_q[idx] - ref_q_host[idx]);
        float diff_k = fabsf(fused_k[idx] - ref_k_host[idx]);
        float diff_v = fabsf(fused_v[idx] - ref_v_host[idx]);
        assert_true(diff_q <= tolerance);
        assert_true(diff_k <= tolerance);
        assert_true(diff_v <= tolerance);
    }

    free(ref_v_host);
    free(ref_k_host);
    free(ref_q_host);
    free(fused_v);
    free(fused_k);
    free(fused_q);
    free(host_bias);
    free(host_weight);
    free(host_input);
    marmot_test_tensor_destroy_all(9, ref_v, ref_k, ref_q, out_v, out_k, out_q, bias, weight, input);
}

static void compute_qkv_expected_float(
    float *out_q, float *out_k, float *out_v, const float *input, const float *weight, const float *bias,
    const float *residual, size_t N, size_t K, size_t M, marmot_device_unary_op_t activation,
    const marmot_activation_params_t *act_params
) {
    for (size_t n = 0; n < N; ++n) {
        for (size_t m = 0; m < M; ++m) {
            float acc_q = 0.0f;
            float acc_k = 0.0f;
            float acc_v = 0.0f;
            size_t base_q = m * K;
            size_t base_k = (M + m) * K;
            size_t base_v = (2 * M + m) * K;
            for (size_t kk = 0; kk < K; ++kk) {
                float a = input[n * K + kk];
                acc_q += a * weight[base_q + kk];
                acc_k += a * weight[base_k + kk];
                acc_v += a * weight[base_v + kk];
            }
            if (bias != nullptr) {
                acc_q += bias[m];
                acc_k += bias[M + m];
                acc_v += bias[(2 * M) + m];
            }
            size_t idx = n * M + m;
            float value_q = apply_activation_ref(activation, acc_q, act_params);
            float value_k = apply_activation_ref(activation, acc_k, act_params);
            float value_v = apply_activation_ref(activation, acc_v, act_params);
            if (residual != nullptr) {
                float r = residual[n * M + m];
                value_q += r;
                value_k += r;
                value_v += r;
            }
            out_q[idx] = value_q;
            out_k[idx] = value_k;
            out_v[idx] = value_v;
        }
    }
}

static void
run_matmul_qkv_dtype_smoke(marmot_test_env_t *env, marmot_dtype_t dtype, float tolerance, bool include_bias) {
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
        run_matmul_qkv_dtype_smoke(&cpu_env, dtype, tolerance, include_bias);
        marmot_destroy(cpu_ctx);
        return;
    }
    const size_t N = 2;
    const size_t K = 5;
    const size_t M = 4;
    const size_t shape_input[2] = {N, K};
    const size_t shape_weight[2] = {3 * M, K};
    const size_t shape_out[2] = {N, M};
    const size_t shape_bias[1] = {3 * M};

    float host_input[N * K];
    float host_weight[3 * M * K];
    float host_bias[3 * M];
    for (size_t i = 0; i < N * K; ++i) {
        host_input[i] = 0.25f * sinf((float)(i + 1) * 0.31f);
    }
    for (size_t i = 0; i < 3 * M * K; ++i) {
        host_weight[i] = 0.3f * cosf((float)(i + 2) * 0.17f);
    }
    if (include_bias) {
        for (size_t i = 0; i < 3 * M; ++i) {
            host_bias[i] = 0.02f * (float)(i + 1);
        }
    }

    float expected_q[N * M];
    float expected_k[N * M];
    float expected_v[N * M];
    compute_qkv_expected_float(
        expected_q, expected_k, expected_v, host_input, host_weight, include_bias ? host_bias : nullptr, nullptr, N, K,
        M, MARMOT_DEVICE_UNARY_IDENTITY, nullptr
    );

    marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape_input, 2, dtype);
    marmot_tensor_t *weight = marmot_tensor_create(env->ctx, shape_weight, 2, dtype);
    marmot_tensor_t *bias = include_bias ? marmot_tensor_create(env->ctx, shape_bias, 1, dtype) : nullptr;
    marmot_tensor_t *out_q = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
    marmot_tensor_t *out_k = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
    marmot_tensor_t *out_v = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
    assert_non_null(input);
    assert_non_null(weight);
    assert_non_null(out_q);
    assert_non_null(out_k);
    assert_non_null(out_v);
    if (include_bias) {
        assert_non_null(bias);
    }

    marmot_test_convert_span(env, input, MARMOT_DTYPE_FLOAT32, host_input, N * K);
    marmot_test_convert_span(env, weight, MARMOT_DTYPE_FLOAT32, host_weight, 3 * M * K);
    if (include_bias) {
        marmot_test_convert_span(env, bias, MARMOT_DTYPE_FLOAT32, host_bias, 3 * M);
    }

    marmot_matmul_qkv_desc_t desc = {
        .input = input,
        .layout = MARMOT_QKV_LAYOUT_FUSED,
        .fused =
            {
                .weight = weight,
                .bias = bias,
            },
        .out_q = out_q,
        .out_k = out_k,
        .out_v = out_v,
    };
    assert_int_equal(marmot_matmul_qkv(env->ctx, &desc), MARMOT_SUCCESS);

    float actual_q[N * M];
    float actual_k[N * M];
    float actual_v[N * M];
    marmot_test_fetch_span(env, actual_q, MARMOT_DTYPE_FLOAT32, out_q, N * M);
    marmot_test_fetch_span(env, actual_k, MARMOT_DTYPE_FLOAT32, out_k, N * M);
    marmot_test_fetch_span(env, actual_v, MARMOT_DTYPE_FLOAT32, out_v, N * M);

    for (size_t i = 0; i < N * M; ++i) {
        assert_float_equal(actual_q[i], expected_q[i], tolerance);
        assert_float_equal(actual_k[i], expected_k[i], tolerance);
        assert_float_equal(actual_v[i], expected_v[i], tolerance);
    }

    marmot_test_tensor_destroy_all(6, out_v, out_k, out_q, bias, weight, input);
}

static void test_matmul_qkv_matches_reference(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    run_matmul_qkv_reference_case(env, 4, 8, 6, false);
    run_matmul_qkv_reference_case(env, 5, 7, 5, true);
}

static void test_matmul_qkv_post_residual_activation(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    marmot_test_env_t test_env = *env;
    marmot_context_t *metal_ctx = nullptr;
    if (env->backend != MARMOT_BACKEND_METAL) {
        metal_ctx = marmot_init(MARMOT_BACKEND_METAL);
        if (metal_ctx == nullptr) {
            skip();
            return;
        }
        test_env.backend = MARMOT_BACKEND_METAL;
        test_env.ctx = metal_ctx;
    }
    const size_t N = 2;
    const size_t K = 3;
    const size_t M = 4;
    const marmot_dtype_t dtype = MARMOT_DTYPE_FLOAT32;
    const marmot_device_unary_op_t activation = MARMOT_DEVICE_UNARY_RELU;
    const size_t shape_input[2] = {N, K};
    const size_t shape_weight[2] = {3 * M, K};
    const size_t shape_out[2] = {N, M};

    float host_input[N * K];
    float host_weight[3 * M * K];
    float host_bias[3 * M];
    float host_residual[N * M];
    for (size_t i = 0; i < N * K; ++i) {
        host_input[i] = sinf((float)(i + 1) * 0.17f);
    }
    for (size_t i = 0; i < 3 * M * K; ++i) {
        host_weight[i] = cosf((float)(i + 2) * 0.11f);
    }
    for (size_t i = 0; i < 3 * M; ++i) {
        host_bias[i] = 0.05f * (float)(i + 1);
    }
    for (size_t i = 0; i < N * M; ++i) {
        host_residual[i] = -0.03f * (float)(i + 1);
    }

    float expected_q[N * M];
    float expected_k[N * M];
    float expected_v[N * M];
    float actual_q[N * M];
    float actual_k[N * M];
    float actual_v[N * M];
    compute_qkv_expected_float(
        expected_q, expected_k, expected_v, host_input, host_weight, host_bias, host_residual, N, K, M, activation,
        nullptr
    );

    marmot_tensor_t *input = marmot_tensor_create(test_env.ctx, shape_input, 2, dtype);
    marmot_tensor_t *weight = marmot_tensor_create(test_env.ctx, shape_weight, 2, dtype);
    marmot_tensor_t *bias = marmot_tensor_create(test_env.ctx, (const size_t[]){3 * M}, 1, dtype);
    marmot_tensor_t *residual = marmot_tensor_create(test_env.ctx, shape_out, 2, dtype);
    marmot_tensor_t *out_q = marmot_tensor_create(test_env.ctx, shape_out, 2, dtype);
    marmot_tensor_t *out_k = marmot_tensor_create(test_env.ctx, shape_out, 2, dtype);
    marmot_tensor_t *out_v = marmot_tensor_create(test_env.ctx, shape_out, 2, dtype);
    marmot_tensor_t *sum_q = nullptr;
    marmot_tensor_t *sum_k = nullptr;
    marmot_tensor_t *sum_v = nullptr;
    marmot_tensor_t *act_q = nullptr;
    marmot_tensor_t *act_k = nullptr;
    marmot_tensor_t *act_v = nullptr;
    assert_non_null(input);
    assert_non_null(weight);
    assert_non_null(bias);
    assert_non_null(residual);
    assert_non_null(out_q);
    assert_non_null(out_k);
    assert_non_null(out_v);

    marmot_test_convert_span(&test_env, input, MARMOT_DTYPE_FLOAT32, host_input, N * K);
    marmot_test_convert_span(&test_env, weight, MARMOT_DTYPE_FLOAT32, host_weight, 3 * M * K);
    marmot_test_convert_span(&test_env, bias, MARMOT_DTYPE_FLOAT32, host_bias, 3 * M);
    marmot_test_convert_span(&test_env, residual, MARMOT_DTYPE_FLOAT32, host_residual, N * M);

    marmot_matmul_qkv_desc_t desc = {
        .input = input,
        .layout = MARMOT_QKV_LAYOUT_FUSED,
        .fused =
            {
                .weight = weight,
                .bias = bias,
            },
        .out_q = out_q,
        .out_k = out_k,
        .out_v = out_v,
    };

    bool should_skip = false;
    marmot_error_t status = marmot_matmul_qkv(test_env.ctx, &desc);
    if (status == MARMOT_ERROR_NOT_IMPLEMENTED || status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        should_skip = true;
        goto cleanup_qkv_post_activation;
    }
    assert_int_equal(status, MARMOT_SUCCESS);

    sum_q = marmot_tensor_create(test_env.ctx, shape_out, 2, dtype);
    sum_k = marmot_tensor_create(test_env.ctx, shape_out, 2, dtype);
    sum_v = marmot_tensor_create(test_env.ctx, shape_out, 2, dtype);
    act_q = marmot_tensor_create(test_env.ctx, shape_out, 2, dtype);
    act_k = marmot_tensor_create(test_env.ctx, shape_out, 2, dtype);
    act_v = marmot_tensor_create(test_env.ctx, shape_out, 2, dtype);
    assert_non_null(sum_q);
    assert_non_null(sum_k);
    assert_non_null(sum_v);
    assert_non_null(act_q);
    assert_non_null(act_k);
    assert_non_null(act_v);

    marmot_error_t act_status = apply_activation_op(test_env.ctx, activation, out_q, act_q, nullptr);
    if (act_status == MARMOT_ERROR_NOT_IMPLEMENTED || act_status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        should_skip = true;
        goto cleanup_qkv_post_activation;
    }
    assert_int_equal(act_status, MARMOT_SUCCESS);
    act_status = apply_activation_op(test_env.ctx, activation, out_k, act_k, nullptr);
    if (act_status == MARMOT_ERROR_NOT_IMPLEMENTED || act_status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        should_skip = true;
        goto cleanup_qkv_post_activation;
    }
    assert_int_equal(act_status, MARMOT_SUCCESS);
    act_status = apply_activation_op(test_env.ctx, activation, out_v, act_v, nullptr);
    if (act_status == MARMOT_ERROR_NOT_IMPLEMENTED || act_status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        should_skip = true;
        goto cleanup_qkv_post_activation;
    }
    assert_int_equal(act_status, MARMOT_SUCCESS);

    marmot_error_t add_status = marmot_add(test_env.ctx, act_q, residual, sum_q);
    if (add_status == MARMOT_ERROR_NOT_IMPLEMENTED || add_status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        should_skip = true;
        goto cleanup_qkv_post_activation;
    }
    assert_int_equal(add_status, MARMOT_SUCCESS);
    add_status = marmot_add(test_env.ctx, act_k, residual, sum_k);
    if (add_status == MARMOT_ERROR_NOT_IMPLEMENTED || add_status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        should_skip = true;
        goto cleanup_qkv_post_activation;
    }
    assert_int_equal(add_status, MARMOT_SUCCESS);
    add_status = marmot_add(test_env.ctx, act_v, residual, sum_v);
    if (add_status == MARMOT_ERROR_NOT_IMPLEMENTED || add_status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        should_skip = true;
        goto cleanup_qkv_post_activation;
    }
    assert_int_equal(add_status, MARMOT_SUCCESS);

    marmot_test_fetch_span(&test_env, actual_q, MARMOT_DTYPE_FLOAT32, sum_q, N * M);
    marmot_test_fetch_span(&test_env, actual_k, MARMOT_DTYPE_FLOAT32, sum_k, N * M);
    marmot_test_fetch_span(&test_env, actual_v, MARMOT_DTYPE_FLOAT32, sum_v, N * M);

    const double tol = matmul_epilogue_tolerance(dtype);
    for (size_t i = 0; i < N * M; ++i) {
        assert_true(fabs(actual_q[i] - expected_q[i]) <= tol);
        assert_true(fabs(actual_k[i] - expected_k[i]) <= tol);
        assert_true(fabs(actual_v[i] - expected_v[i]) <= tol);
    }

cleanup_qkv_post_activation:
    marmot_test_tensor_destroy_all(
        13, act_v, act_k, act_q, sum_v, sum_k, sum_q, out_v, out_k, out_q, residual, bias, weight, input
    );
    if (metal_ctx != nullptr) {
        marmot_destroy(metal_ctx);
    }
    if (should_skip) {
        skip();
    }
}

static void test_matmul_qkv_separate_weights(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const size_t N = 2;
    const size_t K = 3;
    const size_t M = 4;
    const marmot_dtype_t dtype = MARMOT_DTYPE_FLOAT32;
    const double tol = matmul_epilogue_tolerance(dtype);
    const size_t shape_input[2] = {N, K};
    const size_t shape_proj[2] = {M, K};
    const size_t shape_bias[1] = {M};
    const size_t shape_out[2] = {N, M};

    float host_input[N * K];
    float host_weight[3 * M * K];
    float host_bias[3 * M];
    for (size_t i = 0; i < N * K; ++i) {
        host_input[i] = 0.13f * sinf((float)(i + 3) * 0.19f);
    }
    for (size_t i = 0; i < 3 * M * K; ++i) {
        host_weight[i] = 0.07f * cosf((float)(i + 5) * 0.23f);
    }
    for (size_t i = 0; i < 3 * M; ++i) {
        host_bias[i] = 0.01f * (float)(i + 1);
    }

    float host_wq[M * K];
    float host_wk[M * K];
    float host_wv[M * K];
    float host_bq[M];
    float host_bk[M];
    float host_bv[M];
    memcpy(host_wq, host_weight, M * K * sizeof(float));
    memcpy(host_wk, host_weight + M * K, M * K * sizeof(float));
    memcpy(host_wv, host_weight + 2 * M * K, M * K * sizeof(float));
    memcpy(host_bq, host_bias, M * sizeof(float));
    memcpy(host_bk, host_bias + M, M * sizeof(float));
    memcpy(host_bv, host_bias + 2 * M, M * sizeof(float));

    float expected_q[N * M];
    float expected_k[N * M];
    float expected_v[N * M];
    compute_qkv_expected_float(
        expected_q, expected_k, expected_v, host_input, host_weight, host_bias, nullptr, N, K, M,
        MARMOT_DEVICE_UNARY_IDENTITY, nullptr
    );

    marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape_input, 2, dtype);
    marmot_tensor_t *wq = marmot_tensor_create(env->ctx, shape_proj, 2, dtype);
    marmot_tensor_t *wk = marmot_tensor_create(env->ctx, shape_proj, 2, dtype);
    marmot_tensor_t *wv = marmot_tensor_create(env->ctx, shape_proj, 2, dtype);
    marmot_tensor_t *bq = marmot_tensor_create(env->ctx, shape_bias, 1, dtype);
    marmot_tensor_t *bk = marmot_tensor_create(env->ctx, shape_bias, 1, dtype);
    marmot_tensor_t *bv = marmot_tensor_create(env->ctx, shape_bias, 1, dtype);
    marmot_tensor_t *out_q = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
    marmot_tensor_t *out_k = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
    marmot_tensor_t *out_v = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
    assert_non_null(input);
    assert_non_null(wq);
    assert_non_null(wk);
    assert_non_null(wv);
    assert_non_null(bq);
    assert_non_null(bk);
    assert_non_null(bv);
    assert_non_null(out_q);
    assert_non_null(out_k);
    assert_non_null(out_v);

    marmot_test_convert_span(env, input, MARMOT_DTYPE_FLOAT32, host_input, N * K);
    marmot_test_convert_span(env, wq, MARMOT_DTYPE_FLOAT32, host_wq, M * K);
    marmot_test_convert_span(env, wk, MARMOT_DTYPE_FLOAT32, host_wk, M * K);
    marmot_test_convert_span(env, wv, MARMOT_DTYPE_FLOAT32, host_wv, M * K);
    marmot_test_convert_span(env, bq, MARMOT_DTYPE_FLOAT32, host_bq, M);
    marmot_test_convert_span(env, bk, MARMOT_DTYPE_FLOAT32, host_bk, M);
    marmot_test_convert_span(env, bv, MARMOT_DTYPE_FLOAT32, host_bv, M);

    marmot_matmul_qkv_desc_t desc = {
        .input = input,
        .layout = MARMOT_QKV_LAYOUT_SEPARATE,
        .separate =
            {
                .wq = wq,
                .wk = wk,
                .wv = wv,
                .bq = bq,
                .bk = bk,
                .bv = bv,
            },
        .out_q = out_q,
        .out_k = out_k,
        .out_v = out_v,
    };

    assert_int_equal(marmot_matmul_qkv(env->ctx, &desc), MARMOT_SUCCESS);

    float actual_q[N * M];
    float actual_k[N * M];
    float actual_v[N * M];
    marmot_test_fetch_span(env, actual_q, MARMOT_DTYPE_FLOAT32, out_q, N * M);
    marmot_test_fetch_span(env, actual_k, MARMOT_DTYPE_FLOAT32, out_k, N * M);
    marmot_test_fetch_span(env, actual_v, MARMOT_DTYPE_FLOAT32, out_v, N * M);
    for (size_t i = 0; i < N * M; ++i) {
        assert_true(fabs(actual_q[i] - expected_q[i]) <= tol);
        assert_true(fabs(actual_k[i] - expected_k[i]) <= tol);
        assert_true(fabs(actual_v[i] - expected_v[i]) <= tol);
    }

    marmot_test_tensor_destroy_all(10, out_v, out_k, out_q, bv, bk, bq, wv, wk, wq, input);
}

static void test_matmul_qkv_separate_rope_q4km(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_CPU) {
        return;
    }
    const matmul_golden_case_t *tc = matmul_quantized_case_for_kind(MARMOT_QUANT_KIND_Q4_K);
    assert_non_null(tc);
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(tc->kind);
    assert_non_null(traits);

    size_t shape_input[2] = {tc->N, tc->K};
    size_t shape_weight[2] = {tc->M, tc->K};
    size_t shape_out[2] = {tc->N, tc->M};

    marmot_tensor_t *input = marmot_test_tensor_from_array(env, shape_input, 2, tc->input_f32);
    marmot_tensor_t *wq = marmot_tensor_create_quantized(env->ctx, shape_weight, 2, tc->kind);
    marmot_tensor_t *wk = marmot_tensor_create_quantized(env->ctx, shape_weight, 2, tc->kind);
    marmot_tensor_t *wv = marmot_tensor_create_quantized(env->ctx, shape_weight, 2, tc->kind);
    assert_non_null(input);
    assert_non_null(wq);
    assert_non_null(wk);
    assert_non_null(wv);

    marmot_tensor_t *out_q = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_k = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_v = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *ref_q = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *ref_k = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *ref_v = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(out_q);
    assert_non_null(out_k);
    assert_non_null(out_v);
    assert_non_null(ref_q);
    assert_non_null(ref_k);
    assert_non_null(ref_v);

    marmot_tensor_t *weights[3] = {wq, wk, wv};
    for (size_t slice = 0; slice < 3; ++slice) {
        assert_int_equal(marmot_tensor_size_bytes(weights[slice]), tc->weight_bytes);
        assert_int_equal(
            marmot_tensor_copy_from_host_buffer(env->ctx, weights[slice], tc->weights, tc->weight_bytes), MARMOT_SUCCESS
        );
    }

    size_t pos_shape[1] = {tc->N};
    marmot_tensor_t *positions = marmot_tensor_create(nullptr, pos_shape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(positions);
    int32_t *pos_data = (int32_t *)positions->data;
    for (size_t i = 0; i < tc->N; ++i) {
        pos_data[i] = (int32_t)(i * 3);
    }
    marmot_rope_params_t rope = marmot_rope_params_default();
    rope.positions = positions;
    rope.theta = 10000.0f;
    rope.apply_to_q = true;
    rope.apply_to_k = true;

    marmot_matmul_qkv_desc_t rope_desc = {
        .input = input,
        .layout = MARMOT_QKV_LAYOUT_SEPARATE,
        .separate =
            {
                .wq = wq,
                .wk = wk,
                .wv = wv,
                .bq = nullptr,
                .bk = nullptr,
                .bv = nullptr,
            },
        .epilogue = nullptr,
        .rope_params = &rope,
        .out_q = out_q,
        .out_k = out_k,
        .out_v = out_v,
    };
    assert_int_equal(marmot_matmul_qkv(env->ctx, &rope_desc), MARMOT_SUCCESS);

    marmot_matmul_qkv_desc_t ref_desc = rope_desc;
    ref_desc.rope_params = nullptr;
    ref_desc.out_q = ref_q;
    ref_desc.out_k = ref_k;
    ref_desc.out_v = ref_v;
    assert_int_equal(marmot_matmul_qkv(env->ctx, &ref_desc), MARMOT_SUCCESS);
    assert_int_equal(marmot_rope(env->ctx, ref_q, &rope, ref_q), MARMOT_SUCCESS);
    assert_int_equal(marmot_rope(env->ctx, ref_k, &rope, ref_k), MARMOT_SUCCESS);

    const size_t elem_count = tc->N * tc->M;
    float host_q[elem_count];
    float host_k[elem_count];
    float host_v[elem_count];
    float ref_host_q[elem_count];
    float ref_host_k[elem_count];
    float ref_host_v[elem_count];
    marmot_test_fetch_f32_span(env, host_q, out_q, elem_count);
    marmot_test_fetch_f32_span(env, host_k, out_k, elem_count);
    marmot_test_fetch_f32_span(env, host_v, out_v, elem_count);
    marmot_test_fetch_f32_span(env, ref_host_q, ref_q, elem_count);
    marmot_test_fetch_f32_span(env, ref_host_k, ref_k, elem_count);
    marmot_test_fetch_f32_span(env, ref_host_v, ref_v, elem_count);

    const float tol = 2.5e-3f;
    for (size_t i = 0; i < elem_count; ++i) {
        assert_true(fabsf(host_q[i] - ref_host_q[i]) <= tol);
        assert_true(fabsf(host_k[i] - ref_host_k[i]) <= tol);
        assert_true(fabsf(host_v[i] - ref_host_v[i]) <= tol);
    }

    marmot_tensor_destroy(positions);
    marmot_test_tensor_destroy_all(10, ref_v, ref_k, ref_q, out_v, out_k, out_q, wv, wk, wq, input);
}

static void test_matmul_qkv_separate_rope_q4_0(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_CPU && env->backend != MARMOT_BACKEND_METAL) {
        skip();
        return;
    }
    const matmul_golden_case_t *tc = matmul_quantized_case_for_kind(MARMOT_QUANT_KIND_Q4_0);
    assert_non_null(tc);
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(tc->kind);
    assert_non_null(traits);

    size_t shape_input[2] = {tc->N, tc->K};
    size_t shape_weight[2] = {tc->M, tc->K};
    size_t shape_out[2] = {tc->N, tc->M};

    marmot_tensor_t *input = marmot_test_tensor_from_array(env, shape_input, 2, tc->input_f32);
    marmot_tensor_t *wq = marmot_tensor_create_quantized(env->ctx, shape_weight, 2, tc->kind);
    marmot_tensor_t *wk = marmot_tensor_create_quantized(env->ctx, shape_weight, 2, tc->kind);
    marmot_tensor_t *wv = marmot_tensor_create_quantized(env->ctx, shape_weight, 2, tc->kind);
    assert_non_null(input);
    assert_non_null(wq);
    assert_non_null(wk);
    assert_non_null(wv);

    marmot_tensor_t *out_q = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_k = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_v = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *ref_q = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *ref_k = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *ref_v = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(out_q);
    assert_non_null(out_k);
    assert_non_null(out_v);
    assert_non_null(ref_q);
    assert_non_null(ref_k);
    assert_non_null(ref_v);

    marmot_tensor_t *weights[3] = {wq, wk, wv};
    for (size_t slice = 0; slice < 3; ++slice) {
        assert_int_equal(marmot_tensor_size_bytes(weights[slice]), tc->weight_bytes);
        assert_int_equal(
            marmot_tensor_copy_from_host_buffer(env->ctx, weights[slice], tc->weights, tc->weight_bytes), MARMOT_SUCCESS
        );
    }

    size_t pos_shape[1] = {tc->N};
    marmot_tensor_t *positions = marmot_tensor_create(nullptr, pos_shape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(positions);
    int32_t *pos_data = (int32_t *)positions->data;
    for (size_t i = 0; i < tc->N; ++i) {
        pos_data[i] = (int32_t)(i * 3);
    }
    marmot_rope_params_t rope = marmot_rope_params_default();
    rope.positions = positions;
    rope.theta = 10000.0f;
    rope.apply_to_q = true;
    rope.apply_to_k = true;

    marmot_matmul_qkv_desc_t rope_desc = {
        .input = input,
        .layout = MARMOT_QKV_LAYOUT_SEPARATE,
        .separate =
            {
                .wq = wq,
                .wk = wk,
                .wv = wv,
                .bq = nullptr,
                .bk = nullptr,
                .bv = nullptr,
            },
        .epilogue = nullptr,
        .rope_params = &rope,
        .out_q = out_q,
        .out_k = out_k,
        .out_v = out_v,
    };
    marmot_error_t qkv_status = marmot_matmul_qkv(env->ctx, &rope_desc);
    if (qkv_status == MARMOT_ERROR_NOT_IMPLEMENTED) {
        marmot_tensor_destroy(positions);
        marmot_test_tensor_destroy_all(10, ref_v, ref_k, ref_q, out_v, out_k, out_q, wv, wk, wq, input);
        skip();
        return;
    }
    assert_int_equal(qkv_status, MARMOT_SUCCESS);

    marmot_matmul_qkv_desc_t ref_desc = rope_desc;
    ref_desc.rope_params = nullptr;
    ref_desc.out_q = ref_q;
    ref_desc.out_k = ref_k;
    ref_desc.out_v = ref_v;
    assert_int_equal(marmot_matmul_qkv(env->ctx, &ref_desc), MARMOT_SUCCESS);
    assert_int_equal(marmot_rope(env->ctx, ref_q, &rope, ref_q), MARMOT_SUCCESS);
    assert_int_equal(marmot_rope(env->ctx, ref_k, &rope, ref_k), MARMOT_SUCCESS);

    const size_t elem_count = tc->N * tc->M;
    float host_q[elem_count];
    float host_k[elem_count];
    float host_v[elem_count];
    float ref_host_q[elem_count];
    float ref_host_k[elem_count];
    float ref_host_v[elem_count];
    marmot_test_fetch_f32_span(env, host_q, out_q, elem_count);
    marmot_test_fetch_f32_span(env, host_k, out_k, elem_count);
    marmot_test_fetch_f32_span(env, host_v, out_v, elem_count);
    marmot_test_fetch_f32_span(env, ref_host_q, ref_q, elem_count);
    marmot_test_fetch_f32_span(env, ref_host_k, ref_k, elem_count);
    marmot_test_fetch_f32_span(env, ref_host_v, ref_v, elem_count);

    const float tol = 2.5e-3f;
    for (size_t i = 0; i < elem_count; ++i) {
        assert_true(fabsf(host_q[i] - ref_host_q[i]) <= tol);
        assert_true(fabsf(host_k[i] - ref_host_k[i]) <= tol);
        assert_true(fabsf(host_v[i] - ref_host_v[i]) <= tol);
    }

    marmot_tensor_destroy(positions);
    marmot_test_tensor_destroy_all(10, ref_v, ref_k, ref_q, out_v, out_k, out_q, wv, wk, wq, input);
}

static void test_matmul_qkv_separate_rope_q8_0(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_CPU && env->backend != MARMOT_BACKEND_METAL) {
        skip();
        return;
    }
    const matmul_golden_case_t *tc = matmul_quantized_case_for_kind(MARMOT_QUANT_KIND_Q8_0);
    assert_non_null(tc);
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(tc->kind);
    assert_non_null(traits);

    size_t shape_input[2] = {tc->N, tc->K};
    size_t shape_weight[2] = {tc->M, tc->K};
    size_t shape_out[2] = {tc->N, tc->M};

    marmot_tensor_t *input = marmot_test_tensor_from_array(env, shape_input, 2, tc->input_f32);
    marmot_tensor_t *wq = marmot_tensor_create_quantized(env->ctx, shape_weight, 2, tc->kind);
    marmot_tensor_t *wk = marmot_tensor_create_quantized(env->ctx, shape_weight, 2, tc->kind);
    marmot_tensor_t *wv = marmot_tensor_create_quantized(env->ctx, shape_weight, 2, tc->kind);
    assert_non_null(input);
    assert_non_null(wq);
    assert_non_null(wk);
    assert_non_null(wv);

    marmot_tensor_t *out_q = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_k = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_v = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *ref_q = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *ref_k = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *ref_v = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(out_q);
    assert_non_null(out_k);
    assert_non_null(out_v);
    assert_non_null(ref_q);
    assert_non_null(ref_k);
    assert_non_null(ref_v);

    marmot_tensor_t *weights[3] = {wq, wk, wv};
    for (size_t slice = 0; slice < 3; ++slice) {
        assert_int_equal(marmot_tensor_size_bytes(weights[slice]), tc->weight_bytes);
        assert_int_equal(
            marmot_tensor_copy_from_host_buffer(env->ctx, weights[slice], tc->weights, tc->weight_bytes), MARMOT_SUCCESS
        );
    }

    size_t pos_shape[1] = {tc->N};
    marmot_tensor_t *positions = marmot_tensor_create(nullptr, pos_shape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(positions);
    int32_t *pos_data = (int32_t *)positions->data;
    for (size_t i = 0; i < tc->N; ++i) {
        pos_data[i] = (int32_t)(i * 3);
    }
    marmot_rope_params_t rope = marmot_rope_params_default();
    rope.positions = positions;
    rope.theta = 10000.0f;
    rope.apply_to_q = true;
    rope.apply_to_k = true;

    marmot_matmul_qkv_desc_t rope_desc = {
        .input = input,
        .layout = MARMOT_QKV_LAYOUT_SEPARATE,
        .separate =
            {
                .wq = wq,
                .wk = wk,
                .wv = wv,
                .bq = nullptr,
                .bk = nullptr,
                .bv = nullptr,
            },
        .epilogue = nullptr,
        .rope_params = &rope,
        .out_q = out_q,
        .out_k = out_k,
        .out_v = out_v,
    };
    marmot_error_t qkv_status = marmot_matmul_qkv(env->ctx, &rope_desc);
    if (qkv_status == MARMOT_ERROR_NOT_IMPLEMENTED) {
        marmot_tensor_destroy(positions);
        marmot_test_tensor_destroy_all(10, ref_v, ref_k, ref_q, out_v, out_k, out_q, wv, wk, wq, input);
        skip();
        return;
    }
    assert_int_equal(qkv_status, MARMOT_SUCCESS);

    marmot_matmul_qkv_desc_t ref_desc = rope_desc;
    ref_desc.rope_params = nullptr;
    ref_desc.out_q = ref_q;
    ref_desc.out_k = ref_k;
    ref_desc.out_v = ref_v;
    assert_int_equal(marmot_matmul_qkv(env->ctx, &ref_desc), MARMOT_SUCCESS);
    assert_int_equal(marmot_rope(env->ctx, ref_q, &rope, ref_q), MARMOT_SUCCESS);
    assert_int_equal(marmot_rope(env->ctx, ref_k, &rope, ref_k), MARMOT_SUCCESS);

    const size_t elem_count = tc->N * tc->M;
    float host_q[elem_count];
    float host_k[elem_count];
    float host_v[elem_count];
    float ref_host_q[elem_count];
    float ref_host_k[elem_count];
    float ref_host_v[elem_count];
    marmot_test_fetch_f32_span(env, host_q, out_q, elem_count);
    marmot_test_fetch_f32_span(env, host_k, out_k, elem_count);
    marmot_test_fetch_f32_span(env, host_v, out_v, elem_count);
    marmot_test_fetch_f32_span(env, ref_host_q, ref_q, elem_count);
    marmot_test_fetch_f32_span(env, ref_host_k, ref_k, elem_count);
    marmot_test_fetch_f32_span(env, ref_host_v, ref_v, elem_count);

    const float tol = 2.5e-3f;
    for (size_t i = 0; i < elem_count; ++i) {
        assert_true(fabsf(host_q[i] - ref_host_q[i]) <= tol);
        assert_true(fabsf(host_k[i] - ref_host_k[i]) <= tol);
        assert_true(fabsf(host_v[i] - ref_host_v[i]) <= tol);
    }

    marmot_tensor_destroy(positions);
    marmot_test_tensor_destroy_all(10, ref_v, ref_k, ref_q, out_v, out_k, out_q, wv, wk, wq, input);
}

static void test_matmul_qkv_separate_rope_f16(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_CPU) {
        return;
    }
    const size_t N = 3;
    const size_t K = 6;
    const size_t M = 4;
    const double tol = matmul_epilogue_tolerance(MARMOT_DTYPE_FLOAT16);
    const size_t shape_input[2] = {N, K};
    const size_t shape_proj[2] = {M, K};
    const size_t shape_bias[1] = {M};
    const size_t shape_out[2] = {N, M};

    float host_input[N * K];
    float host_wq[M * K];
    float host_wk[M * K];
    float host_wv[M * K];
    float host_bq[M];
    float host_bk[M];
    float host_bv[M];

    for (size_t i = 0; i < N * K; ++i) {
        host_input[i] = 0.031f * (float)cos((double)(i + 3) * 0.17);
    }
    for (size_t i = 0; i < M * K; ++i) {
        host_wq[i] = 0.021f * (float)sin((double)(i + 5) * 0.11);
        host_wk[i] = 0.017f * (float)cos((double)(i + 7) * 0.09);
        host_wv[i] = 0.019f * (float)sin((double)(i + 11) * 0.05);
    }
    for (size_t i = 0; i < M; ++i) {
        host_bq[i] = -0.002f * (float)(i + 1);
        host_bk[i] = 0.001f * (float)(i + 3);
        host_bv[i] = -0.003f * (float)(i + 5);
    }

    marmot_tensor_t *input_f16 = marmot_tensor_create(env->ctx, shape_input, 2, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *wq_f16 = marmot_tensor_create(env->ctx, shape_proj, 2, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *wk_f16 = marmot_tensor_create(env->ctx, shape_proj, 2, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *wv_f16 = marmot_tensor_create(env->ctx, shape_proj, 2, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *bq_f16 = marmot_tensor_create(env->ctx, shape_bias, 1, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *bk_f16 = marmot_tensor_create(env->ctx, shape_bias, 1, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *bv_f16 = marmot_tensor_create(env->ctx, shape_bias, 1, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *out_q_f16 = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *out_k_f16 = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *out_v_f16 = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *input_f32 = marmot_tensor_create(env->ctx, shape_input, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *wq_f32 = marmot_tensor_create(env->ctx, shape_proj, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *wk_f32 = marmot_tensor_create(env->ctx, shape_proj, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *wv_f32 = marmot_tensor_create(env->ctx, shape_proj, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *bq_f32 = marmot_tensor_create(env->ctx, shape_bias, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *bk_f32 = marmot_tensor_create(env->ctx, shape_bias, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *bv_f32 = marmot_tensor_create(env->ctx, shape_bias, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_q_f32 = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_k_f32 = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_v_f32 = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(input_f16);
    assert_non_null(wq_f16);
    assert_non_null(wk_f16);
    assert_non_null(wv_f16);
    assert_non_null(bq_f16);
    assert_non_null(bk_f16);
    assert_non_null(bv_f16);
    assert_non_null(out_q_f16);
    assert_non_null(out_k_f16);
    assert_non_null(out_v_f16);
    assert_non_null(input_f32);
    assert_non_null(wq_f32);
    assert_non_null(wk_f32);
    assert_non_null(wv_f32);
    assert_non_null(bq_f32);
    assert_non_null(bk_f32);
    assert_non_null(bv_f32);
    assert_non_null(out_q_f32);
    assert_non_null(out_k_f32);
    assert_non_null(out_v_f32);

    marmot_test_convert_span(env, input_f16, MARMOT_DTYPE_FLOAT32, host_input, N * K);
    marmot_test_convert_span(env, wq_f16, MARMOT_DTYPE_FLOAT32, host_wq, M * K);
    marmot_test_convert_span(env, wk_f16, MARMOT_DTYPE_FLOAT32, host_wk, M * K);
    marmot_test_convert_span(env, wv_f16, MARMOT_DTYPE_FLOAT32, host_wv, M * K);
    marmot_test_convert_span(env, bq_f16, MARMOT_DTYPE_FLOAT32, host_bq, M);
    marmot_test_convert_span(env, bk_f16, MARMOT_DTYPE_FLOAT32, host_bk, M);
    marmot_test_convert_span(env, bv_f16, MARMOT_DTYPE_FLOAT32, host_bv, M);
    marmot_test_convert_span(env, input_f32, MARMOT_DTYPE_FLOAT32, host_input, N * K);
    marmot_test_convert_span(env, wq_f32, MARMOT_DTYPE_FLOAT32, host_wq, M * K);
    marmot_test_convert_span(env, wk_f32, MARMOT_DTYPE_FLOAT32, host_wk, M * K);
    marmot_test_convert_span(env, wv_f32, MARMOT_DTYPE_FLOAT32, host_wv, M * K);
    marmot_test_convert_span(env, bq_f32, MARMOT_DTYPE_FLOAT32, host_bq, M);
    marmot_test_convert_span(env, bk_f32, MARMOT_DTYPE_FLOAT32, host_bk, M);
    marmot_test_convert_span(env, bv_f32, MARMOT_DTYPE_FLOAT32, host_bv, M);

    size_t pos_shape[1] = {N};
    marmot_tensor_t *positions = marmot_tensor_create(nullptr, pos_shape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(positions);
    for (size_t i = 0; i < N; ++i) {
        ((int32_t *)positions->data)[i] = (int32_t)i;
    }
    marmot_rope_params_t rope = marmot_rope_params_default();
    rope.positions = positions;
    rope.theta = 10000.0f;
    rope.apply_to_q = true;
    rope.apply_to_k = true;

    marmot_matmul_qkv_desc_t desc_f16 = {
        .input = input_f16,
        .layout = MARMOT_QKV_LAYOUT_SEPARATE,
        .separate =
            {
                .wq = wq_f16,
                .wk = wk_f16,
                .wv = wv_f16,
                .bq = bq_f16,
                .bk = bk_f16,
                .bv = bv_f16,
            },
        .rope_params = &rope,
        .out_q = out_q_f16,
        .out_k = out_k_f16,
        .out_v = out_v_f16,
    };

    marmot_matmul_qkv_desc_t desc_f32 = desc_f16;
    desc_f32.input = input_f32;
    desc_f32.separate.wq = wq_f32;
    desc_f32.separate.wk = wk_f32;
    desc_f32.separate.wv = wv_f32;
    desc_f32.separate.bq = bq_f32;
    desc_f32.separate.bk = bk_f32;
    desc_f32.separate.bv = bv_f32;
    desc_f32.out_q = out_q_f32;
    desc_f32.out_k = out_k_f32;
    desc_f32.out_v = out_v_f32;

    assert_int_equal(marmot_matmul_qkv(env->ctx, &desc_f16), MARMOT_SUCCESS);
    assert_int_equal(marmot_matmul_qkv(env->ctx, &desc_f32), MARMOT_SUCCESS);

    float actual_q[N * M];
    float actual_k[N * M];
    float actual_v[N * M];
    float ref_q[N * M];
    float ref_k[N * M];
    float ref_v[N * M];
    marmot_test_fetch_span(env, actual_q, MARMOT_DTYPE_FLOAT32, out_q_f16, N * M);
    marmot_test_fetch_span(env, actual_k, MARMOT_DTYPE_FLOAT32, out_k_f16, N * M);
    marmot_test_fetch_span(env, actual_v, MARMOT_DTYPE_FLOAT32, out_v_f16, N * M);
    marmot_test_fetch_span(env, ref_q, MARMOT_DTYPE_FLOAT32, out_q_f32, N * M);
    marmot_test_fetch_span(env, ref_k, MARMOT_DTYPE_FLOAT32, out_k_f32, N * M);
    marmot_test_fetch_span(env, ref_v, MARMOT_DTYPE_FLOAT32, out_v_f32, N * M);

    for (size_t i = 0; i < N * M; ++i) {
        assert_true(fabs(actual_q[i] - ref_q[i]) <= tol);
        assert_true(fabs(actual_k[i] - ref_k[i]) <= tol);
        assert_true(fabs(actual_v[i] - ref_v[i]) <= tol);
    }

    marmot_tensor_destroy(positions);
    marmot_test_tensor_destroy_all(
        20, out_v_f32, out_k_f32, out_q_f32, bv_f32, bk_f32, bq_f32, wv_f32, wk_f32, wq_f32, input_f32, out_v_f16,
        out_k_f16, out_q_f16, bv_f16, bk_f16, bq_f16, wv_f16, wk_f16, wq_f16, input_f16
    );
}

static void test_matmul_qkv_fused_rope_f16(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_CPU) {
        return;
    }
    const size_t N = 3;
    const size_t K = 6;
    const size_t M = 4;
    const double tol = matmul_epilogue_tolerance(MARMOT_DTYPE_FLOAT16);
    const size_t shape_input[2] = {N, K};
    const size_t shape_weight[2] = {3 * M, K};
    const size_t shape_bias[1] = {3 * M};
    const size_t shape_out[2] = {N, M};

    float host_input[N * K];
    float host_wq[M * K];
    float host_wk[M * K];
    float host_wv[M * K];
    float host_weight[3 * M * K];
    float host_bias[3 * M];

    for (size_t i = 0; i < N * K; ++i) {
        host_input[i] = 0.015f * (float)cos((double)(i + 2) * 0.21);
    }
    for (size_t i = 0; i < M * K; ++i) {
        host_wq[i] = 0.019f * (float)sin((double)(i + 13) * 0.07);
        host_wk[i] = 0.017f * (float)cos((double)(i + 17) * 0.05);
        host_wv[i] = 0.011f * (float)sin((double)(i + 23) * 0.09);
    }
    for (size_t row = 0; row < M; ++row) {
        for (size_t col = 0; col < K; ++col) {
            const size_t idx = row * K + col;
            host_weight[idx] = host_wq[idx];
            host_weight[(M + row) * K + col] = host_wk[idx];
            host_weight[(2 * M + row) * K + col] = host_wv[idx];
        }
        host_bias[row] = -0.0025f * (float)(row + 1);
        host_bias[M + row] = 0.0015f * (float)(row + 2);
        host_bias[2 * M + row] = -0.0021f * (float)(row + 3);
    }

    marmot_tensor_t *input_f16 = marmot_tensor_create(env->ctx, shape_input, 2, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *weight_f16 = marmot_tensor_create(env->ctx, shape_weight, 2, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *bias_f16 = marmot_tensor_create(env->ctx, shape_bias, 1, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *out_q_f16 = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *out_k_f16 = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *out_v_f16 = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *input_f32 = marmot_tensor_create(env->ctx, shape_input, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *weight_f32 = marmot_tensor_create(env->ctx, shape_weight, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *bias_f32 = marmot_tensor_create(env->ctx, shape_bias, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_q_f32 = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_k_f32 = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_v_f32 = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(input_f16);
    assert_non_null(weight_f16);
    assert_non_null(bias_f16);
    assert_non_null(out_q_f16);
    assert_non_null(out_k_f16);
    assert_non_null(out_v_f16);
    assert_non_null(input_f32);
    assert_non_null(weight_f32);
    assert_non_null(bias_f32);
    assert_non_null(out_q_f32);
    assert_non_null(out_k_f32);
    assert_non_null(out_v_f32);

    marmot_test_convert_span(env, input_f16, MARMOT_DTYPE_FLOAT32, host_input, N * K);
    marmot_test_convert_span(env, input_f32, MARMOT_DTYPE_FLOAT32, host_input, N * K);
    marmot_test_convert_span(env, weight_f16, MARMOT_DTYPE_FLOAT32, host_weight, 3 * M * K);
    marmot_test_convert_span(env, weight_f32, MARMOT_DTYPE_FLOAT32, host_weight, 3 * M * K);
    marmot_test_convert_span(env, bias_f16, MARMOT_DTYPE_FLOAT32, host_bias, 3 * M);
    marmot_test_convert_span(env, bias_f32, MARMOT_DTYPE_FLOAT32, host_bias, 3 * M);

    size_t pos_shape[1] = {N};
    marmot_tensor_t *positions = marmot_tensor_create(nullptr, pos_shape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(positions);
    for (size_t i = 0; i < N; ++i) {
        ((int32_t *)positions->data)[i] = (int32_t)(i * 2);
    }
    marmot_rope_params_t rope = marmot_rope_params_default();
    rope.positions = positions;
    rope.theta = 10000.0f;
    rope.apply_to_q = true;
    rope.apply_to_k = true;

    marmot_matmul_qkv_desc_t desc_f16 = {
        .input = input_f16,
        .layout = MARMOT_QKV_LAYOUT_FUSED,
        .fused =
            {
                .weight = weight_f16,
                .bias = bias_f16,
            },
        .rope_params = &rope,
        .out_q = out_q_f16,
        .out_k = out_k_f16,
        .out_v = out_v_f16,
    };

    marmot_matmul_qkv_desc_t desc_f32 = desc_f16;
    desc_f32.input = input_f32;
    desc_f32.fused.weight = weight_f32;
    desc_f32.fused.bias = bias_f32;
    desc_f32.out_q = out_q_f32;
    desc_f32.out_k = out_k_f32;
    desc_f32.out_v = out_v_f32;

    assert_int_equal(marmot_matmul_qkv(env->ctx, &desc_f16), MARMOT_SUCCESS);
    assert_int_equal(marmot_matmul_qkv(env->ctx, &desc_f32), MARMOT_SUCCESS);

    float actual_q[N * M];
    float actual_k[N * M];
    float actual_v[N * M];
    float ref_q[N * M];
    float ref_k[N * M];
    float ref_v[N * M];
    marmot_test_fetch_span(env, actual_q, MARMOT_DTYPE_FLOAT32, out_q_f16, N * M);
    marmot_test_fetch_span(env, actual_k, MARMOT_DTYPE_FLOAT32, out_k_f16, N * M);
    marmot_test_fetch_span(env, actual_v, MARMOT_DTYPE_FLOAT32, out_v_f16, N * M);
    marmot_test_fetch_span(env, ref_q, MARMOT_DTYPE_FLOAT32, out_q_f32, N * M);
    marmot_test_fetch_span(env, ref_k, MARMOT_DTYPE_FLOAT32, out_k_f32, N * M);
    marmot_test_fetch_span(env, ref_v, MARMOT_DTYPE_FLOAT32, out_v_f32, N * M);

    for (size_t i = 0; i < N * M; ++i) {
        assert_true(fabs(actual_q[i] - ref_q[i]) <= tol);
        assert_true(fabs(actual_k[i] - ref_k[i]) <= tol);
        assert_true(fabs(actual_v[i] - ref_v[i]) <= tol);
    }

    marmot_tensor_destroy(positions);
    marmot_test_tensor_destroy_all(
        12, out_v_f32, out_k_f32, out_q_f32, bias_f32, weight_f32, input_f32, out_v_f16, out_k_f16, out_q_f16, bias_f16,
        weight_f16, input_f16
    );
}

static void test_matmul_qkv_separate_weights_f16(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const size_t N = 2;
    const size_t K = 3;
    const size_t M = 4;
    const marmot_dtype_t dtype = MARMOT_DTYPE_FLOAT16;
    const double tol = matmul_epilogue_tolerance(dtype);
    const size_t shape_input[2] = {N, K};
    const size_t shape_proj[2] = {M, K};
    const size_t shape_bias[1] = {M};
    const size_t shape_out[2] = {N, M};

    float host_input[N * K];
    float host_weight[3 * M * K];
    float host_bias[3 * M];
    for (size_t i = 0; i < N * K; ++i) {
        host_input[i] = 0.07f * cosf((float)(i + 2) * 0.11f);
    }
    for (size_t i = 0; i < 3 * M * K; ++i) {
        host_weight[i] = 0.05f * sinf((float)(i + 5) * 0.13f);
    }
    for (size_t i = 0; i < 3 * M; ++i) {
        host_bias[i] = -0.01f * (float)(i + 1);
    }

    float expected_q[N * M];
    float expected_k[N * M];
    float expected_v[N * M];
    compute_qkv_expected_float(
        expected_q, expected_k, expected_v, host_input, host_weight, host_bias, nullptr, N, K, M,
        MARMOT_DEVICE_UNARY_IDENTITY, nullptr
    );

    float host_wq[M * K];
    float host_wk[M * K];
    float host_wv[M * K];
    float host_bq[M];
    float host_bk[M];
    float host_bv[M];
    memcpy(host_wq, host_weight, M * K * sizeof(float));
    memcpy(host_wk, host_weight + M * K, M * K * sizeof(float));
    memcpy(host_wv, host_weight + 2 * M * K, M * K * sizeof(float));
    memcpy(host_bq, host_bias, M * sizeof(float));
    memcpy(host_bk, host_bias + M, M * sizeof(float));
    memcpy(host_bv, host_bias + 2 * M, M * sizeof(float));

    marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape_input, 2, dtype);
    marmot_tensor_t *wq = marmot_tensor_create(env->ctx, shape_proj, 2, dtype);
    marmot_tensor_t *wk = marmot_tensor_create(env->ctx, shape_proj, 2, dtype);
    marmot_tensor_t *wv = marmot_tensor_create(env->ctx, shape_proj, 2, dtype);
    marmot_tensor_t *bq = marmot_tensor_create(env->ctx, shape_bias, 1, dtype);
    marmot_tensor_t *bk = marmot_tensor_create(env->ctx, shape_bias, 1, dtype);
    marmot_tensor_t *bv = marmot_tensor_create(env->ctx, shape_bias, 1, dtype);
    marmot_tensor_t *out_q = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
    marmot_tensor_t *out_k = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
    marmot_tensor_t *out_v = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
    assert_non_null(input);
    assert_non_null(wq);
    assert_non_null(wk);
    assert_non_null(wv);
    assert_non_null(bq);
    assert_non_null(bk);
    assert_non_null(bv);
    assert_non_null(out_q);
    assert_non_null(out_k);
    assert_non_null(out_v);

    marmot_test_convert_span(env, input, MARMOT_DTYPE_FLOAT32, host_input, N * K);
    marmot_test_convert_span(env, wq, MARMOT_DTYPE_FLOAT32, host_wq, M * K);
    marmot_test_convert_span(env, wk, MARMOT_DTYPE_FLOAT32, host_wk, M * K);
    marmot_test_convert_span(env, wv, MARMOT_DTYPE_FLOAT32, host_wv, M * K);
    marmot_test_convert_span(env, bq, MARMOT_DTYPE_FLOAT32, host_bq, M);
    marmot_test_convert_span(env, bk, MARMOT_DTYPE_FLOAT32, host_bk, M);
    marmot_test_convert_span(env, bv, MARMOT_DTYPE_FLOAT32, host_bv, M);

    marmot_matmul_qkv_desc_t desc = {
        .input = input,
        .layout = MARMOT_QKV_LAYOUT_SEPARATE,
        .separate =
            {
                .wq = wq,
                .wk = wk,
                .wv = wv,
                .bq = bq,
                .bk = bk,
                .bv = bv,
            },
        .out_q = out_q,
        .out_k = out_k,
        .out_v = out_v,
    };

    assert_int_equal(marmot_matmul_qkv(env->ctx, &desc), MARMOT_SUCCESS);

    float actual_q[N * M];
    float actual_k[N * M];
    float actual_v[N * M];
    marmot_test_fetch_span(env, actual_q, MARMOT_DTYPE_FLOAT32, out_q, N * M);
    marmot_test_fetch_span(env, actual_k, MARMOT_DTYPE_FLOAT32, out_k, N * M);
    marmot_test_fetch_span(env, actual_v, MARMOT_DTYPE_FLOAT32, out_v, N * M);
    for (size_t i = 0; i < N * M; ++i) {
        assert_true(fabs(actual_q[i] - expected_q[i]) <= tol);
        assert_true(fabs(actual_k[i] - expected_k[i]) <= tol);
        assert_true(fabs(actual_v[i] - expected_v[i]) <= tol);
    }

    marmot_test_tensor_destroy_all(10, out_v, out_k, out_q, bv, bk, bq, wv, wk, wq, input);
}

static void test_matmul_qkv_goldens(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const marmot_dtype_t dtypes[] = {
        MARMOT_DTYPE_FLOAT32,
        MARMOT_DTYPE_FLOAT16,
        MARMOT_DTYPE_BFLOAT16,
        MARMOT_DTYPE_FLOAT64,
    };

    for (size_t d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
        const marmot_dtype_t dtype = dtypes[d];
        if (env->backend == MARMOT_BACKEND_METAL && dtype == MARMOT_DTYPE_FLOAT64) {
            continue; // Metal does not support float64 matmul kernels
        }
        const bool use_f64 = (dtype == MARMOT_DTYPE_FLOAT64);
        const marmot_dtype_t golden_dtype = use_f64 ? MARMOT_DTYPE_FLOAT64 : MARMOT_DTYPE_FLOAT32;
        const double tolerance = matmul_epilogue_tolerance(dtype);

        for (size_t i = 0; i < g_matmul_qkv_case_count; ++i) {
            const typeof(g_matmul_qkv_cases[0]) *tc = &g_matmul_qkv_cases[i];
            const size_t shape_input[2] = {tc->n, tc->k};
            const size_t shape_weight[2] = {3 * tc->m, tc->k};
            const size_t shape_out[2] = {tc->n, tc->m};
            const size_t elem_count = tc->n * tc->m;

            marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape_input, 2, dtype);
            marmot_tensor_t *weight = marmot_tensor_create(env->ctx, shape_weight, 2, dtype);
            marmot_tensor_t *bias = nullptr;
            marmot_tensor_t *out_q = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
            marmot_tensor_t *out_k = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
            marmot_tensor_t *out_v = marmot_tensor_create(env->ctx, shape_out, 2, dtype);
            assert_non_null(input);
            assert_non_null(weight);
            assert_non_null(out_q);
            assert_non_null(out_k);
            assert_non_null(out_v);

            marmot_test_convert_span(
                env, input, golden_dtype, use_f64 ? (const void *)tc->input_f64 : (const void *)tc->input, tc->n * tc->k
            );
            marmot_test_convert_span(
                env, weight, golden_dtype, use_f64 ? (const void *)tc->weight_f64 : (const void *)tc->weight,
                3 * tc->m * tc->k
            );

            if (tc->has_bias) {
                bias = marmot_tensor_create(env->ctx, (const size_t[]){3 * tc->m}, 1, dtype);
                assert_non_null(bias);
                marmot_test_convert_span(
                    env, bias, golden_dtype, use_f64 ? (const void *)tc->bias_f64 : (const void *)tc->bias, 3 * tc->m
                );
            }

            marmot_matmul_qkv_desc_t desc = {
                .input = input,
                .layout = MARMOT_QKV_LAYOUT_FUSED,
                .fused =
                    {
                        .weight = weight,
                        .bias = bias,
                    },
                .out_q = out_q,
                .out_k = out_k,
                .out_v = out_v,
            };

            marmot_error_t err = marmot_matmul_qkv(env->ctx, &desc);
            assert_int_equal(err, MARMOT_SUCCESS);

            const size_t elem_bytes = marmot_get_dtype_traits(golden_dtype)->storage_bytes * elem_count;
            void *actual_q = malloc(elem_bytes);
            void *actual_k = malloc(elem_bytes);
            void *actual_v = malloc(elem_bytes);
            assert_non_null(actual_q);
            assert_non_null(actual_k);
            assert_non_null(actual_v);

            marmot_test_fetch_span(env, actual_q, golden_dtype, out_q, elem_count);
            marmot_test_fetch_span(env, actual_k, golden_dtype, out_k, elem_count);
            marmot_test_fetch_span(env, actual_v, golden_dtype, out_v, elem_count);

            const void *expected_q = use_f64 ? (const void *)tc->expected_q_f64 : (const void *)tc->expected_q;
            const void *expected_k = use_f64 ? (const void *)tc->expected_k_f64 : (const void *)tc->expected_k;
            const void *expected_v = use_f64 ? (const void *)tc->expected_v_f64 : (const void *)tc->expected_v;

            if (use_f64) {
                const double *aq = (const double *)actual_q;
                const double *ak = (const double *)actual_k;
                const double *av = (const double *)actual_v;
                const double *eq = (const double *)expected_q;
                const double *ek = (const double *)expected_k;
                const double *ev = (const double *)expected_v;
                for (size_t idx = 0; idx < elem_count; ++idx) {
                    if (fabs(aq[idx] - eq[idx]) > tolerance) {
                        fail_msg(
                            "Matmul QKV golden %s (dtype=%d, slice=q) diff exceeds tolerance at index %zu", tc->name,
                            (int)dtype, idx
                        );
                    }
                    if (fabs(ak[idx] - ek[idx]) > tolerance) {
                        fail_msg(
                            "Matmul QKV golden %s (dtype=%d, slice=k) diff exceeds tolerance at index %zu", tc->name,
                            (int)dtype, idx
                        );
                    }
                    if (fabs(av[idx] - ev[idx]) > tolerance) {
                        fail_msg(
                            "Matmul QKV golden %s (dtype=%d, slice=v) diff exceeds tolerance at index %zu", tc->name,
                            (int)dtype, idx
                        );
                    }
                }
            } else {
                const float *aq = (const float *)actual_q;
                const float *ak = (const float *)actual_k;
                const float *av = (const float *)actual_v;
                const float *eq = (const float *)expected_q;
                const float *ek = (const float *)expected_k;
                const float *ev = (const float *)expected_v;
                for (size_t idx = 0; idx < elem_count; ++idx) {
                    if (fabsf(aq[idx] - eq[idx]) > (float)tolerance) {
                        fail_msg(
                            "Matmul QKV golden %s (dtype=%d, slice=q) diff exceeds tolerance at index %zu", tc->name,
                            (int)dtype, idx
                        );
                    }
                    if (fabsf(ak[idx] - ek[idx]) > (float)tolerance) {
                        fail_msg(
                            "Matmul QKV golden %s (dtype=%d, slice=k) diff exceeds tolerance at index %zu", tc->name,
                            (int)dtype, idx
                        );
                    }
                    if (fabsf(av[idx] - ev[idx]) > (float)tolerance) {
                        fail_msg(
                            "Matmul QKV golden %s (dtype=%d, slice=v) diff exceeds tolerance at index %zu", tc->name,
                            (int)dtype, idx
                        );
                    }
                }
            }

            free(actual_v);
            free(actual_k);
            free(actual_q);
            marmot_test_tensor_destroy_all(6, out_v, out_k, out_q, bias, weight, input);
        }
    }
}

static void test_linear_default(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    // Tolerance relaxed from 5e-6f to 1e-5f to account for FMA contraction in release builds
    exercise_linear_case(env, MARMOT_DTYPE_FLOAT32, 0, 1e-5f);
    if (env->backend != MARMOT_BACKEND_METAL) {
        exercise_linear_case(env, MARMOT_DTYPE_FLOAT64, 0, 1e-5f);
    }
    exercise_linear_case(env, MARMOT_DTYPE_FLOAT16, 1, 2e-2f);
    exercise_linear_case(env, MARMOT_DTYPE_BFLOAT16, 2, 3.0f);
}

#if MARMOT_ENABLE_FP8
static void test_matmul_fp8_e4m3_golden(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);

    // FP8 matmul is only implemented on CPU backend
    if (env->backend != MARMOT_BACKEND_CPU) {
        skip();
    }

    const float tolerance = 1e-5f;

    for (size_t i = 0; i < g_matmul_fp8_e4m3_case_count; ++i) {
        const matmul_fp8_e4m3_case_t *tc = &g_matmul_fp8_e4m3_cases[i];
        const size_t N = tc->n;
        const size_t K = tc->k;
        const size_t M = tc->m;

        size_t shape_input[2] = {N, K};
        size_t shape_weight[2] = {M, K};
        size_t shape_out[2] = {N, M};

        marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape_input, 2, MARMOT_DTYPE_FLOAT8_E4M3);
        marmot_tensor_t *weight = marmot_tensor_create(env->ctx, shape_weight, 2, MARMOT_DTYPE_FLOAT8_E4M3);
        marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
        assert_non_null(input);
        assert_non_null(weight);
        assert_non_null(out);

        marmot_error_t err = marmot_tensor_copy_from_host_buffer(env->ctx, input, tc->input, N * K * sizeof(uint8_t));
        assert_int_equal(err, MARMOT_SUCCESS);
        err = marmot_tensor_copy_from_host_buffer(env->ctx, weight, tc->weight, M * K * sizeof(uint8_t));
        assert_int_equal(err, MARMOT_SUCCESS);

        err = marmot_linear(env->ctx, input, weight, nullptr, out);
        assert_int_equal(err, MARMOT_SUCCESS);

        float *actual = (float *)malloc(N * M * sizeof(float));
        assert_non_null(actual);
        err = marmot_tensor_copy_to_host_buffer(env->ctx, out, actual, N * M * sizeof(float));
        assert_int_equal(err, MARMOT_SUCCESS);

        for (size_t idx = 0; idx < N * M; ++idx) {
            float diff = fabsf(actual[idx] - tc->expected[idx]);
            if (diff > tolerance) {
                fail_msg(
                    "FP8 E4M3 matmul golden case %zu diff at index %zu: expected %f, got %f (diff=%f)", i, idx,
                    tc->expected[idx], actual[idx], diff
                );
            }
        }

        free(actual);
        marmot_tensor_destroy(out);
        marmot_tensor_destroy(weight);
        marmot_tensor_destroy(input);
    }
}

static void test_matmul_fp8_e5m2_golden(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);

    // FP8 matmul is only implemented on CPU backend
    if (env->backend != MARMOT_BACKEND_CPU) {
        skip();
    }

    const float tolerance = 1e-5f;

    for (size_t i = 0; i < g_matmul_fp8_e5m2_case_count; ++i) {
        const matmul_fp8_e5m2_case_t *tc = &g_matmul_fp8_e5m2_cases[i];
        const size_t N = tc->n;
        const size_t K = tc->k;
        const size_t M = tc->m;

        size_t shape_input[2] = {N, K};
        size_t shape_weight[2] = {M, K};
        size_t shape_out[2] = {N, M};

        marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape_input, 2, MARMOT_DTYPE_FLOAT8_E5M2);
        marmot_tensor_t *weight = marmot_tensor_create(env->ctx, shape_weight, 2, MARMOT_DTYPE_FLOAT8_E5M2);
        marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
        assert_non_null(input);
        assert_non_null(weight);
        assert_non_null(out);

        marmot_error_t err = marmot_tensor_copy_from_host_buffer(env->ctx, input, tc->input, N * K * sizeof(uint8_t));
        assert_int_equal(err, MARMOT_SUCCESS);
        err = marmot_tensor_copy_from_host_buffer(env->ctx, weight, tc->weight, M * K * sizeof(uint8_t));
        assert_int_equal(err, MARMOT_SUCCESS);

        err = marmot_linear(env->ctx, input, weight, nullptr, out);
        assert_int_equal(err, MARMOT_SUCCESS);

        float *actual = (float *)malloc(N * M * sizeof(float));
        assert_non_null(actual);
        err = marmot_tensor_copy_to_host_buffer(env->ctx, out, actual, N * M * sizeof(float));
        assert_int_equal(err, MARMOT_SUCCESS);

        for (size_t idx = 0; idx < N * M; ++idx) {
            float diff = fabsf(actual[idx] - tc->expected[idx]);
            if (diff > tolerance) {
                fail_msg(
                    "FP8 E5M2 matmul golden case %zu diff at index %zu: expected %f, got %f (diff=%f)", i, idx,
                    tc->expected[idx], actual[idx], diff
                );
            }
        }

        free(actual);
        marmot_tensor_destroy(out);
        marmot_tensor_destroy(weight);
        marmot_tensor_destroy(input);
    }
}
#endif // MARMOT_ENABLE_FP8

static void test_linear_followed_by_sum(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);

    marmot_context_t *cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
    if (cpu_ctx == nullptr) {
        skip();
    }

    // Linear convention: input(N×K) @ weight(M×K).T = output(N×M)
    const size_t N = 8;
    const size_t K = 16;
    const size_t M = 16;
    const size_t elems_input = N * K;
    const size_t elems_weight = M * K;
    float *host_input = (float *)malloc(elems_input * sizeof(float));
    float *host_weight = (float *)malloc(elems_weight * sizeof(float));
    assert_non_null(host_input);
    assert_non_null(host_weight);

    for (size_t i = 0; i < elems_input; ++i) {
        host_input[i] = sinf((float)i * 0.013f);
    }
    for (size_t i = 0; i < elems_weight; ++i) {
        host_weight[i] = cosf((float)i * 0.007f);
    }

    size_t shape_input[2] = {N, K};
    size_t shape_weight[2] = {M, K};
    size_t shape_out[2] = {N, M};
    size_t shape_scalar[1] = {1};

    marmot_tensor_t *cpu_input = marmot_tensor_create(nullptr, shape_input, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *cpu_weight = marmot_tensor_create(nullptr, shape_weight, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *cpu_out = marmot_tensor_create(nullptr, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *cpu_sum = marmot_tensor_create(nullptr, shape_scalar, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(cpu_input);
    assert_non_null(cpu_weight);
    assert_non_null(cpu_out);
    assert_non_null(cpu_sum);

    memcpy(cpu_input->data, host_input, elems_input * sizeof(float));
    memcpy(cpu_weight->data, host_weight, elems_weight * sizeof(float));
    assert_int_equal(marmot_linear(cpu_ctx, cpu_input, cpu_weight, nullptr, cpu_out), MARMOT_SUCCESS);
    assert_int_equal(
        marmot_reduce_sum(
            cpu_ctx,
            &(
                marmot_reduction_desc_t
            ){.input = cpu_out, .out = cpu_sum, .axes = nullptr, .num_axes = 0, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );

    marmot_tensor_t *dev_input = marmot_tensor_create(env->ctx, shape_input, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *dev_weight = marmot_tensor_create(env->ctx, shape_weight, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *dev_out = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *dev_sum = marmot_tensor_create(env->ctx, shape_scalar, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(dev_input);
    assert_non_null(dev_weight);
    assert_non_null(dev_out);
    assert_non_null(dev_sum);

    marmot_test_convert_span(env, dev_input, MARMOT_DTYPE_FLOAT32, host_input, elems_input);
    marmot_test_convert_span(env, dev_weight, MARMOT_DTYPE_FLOAT32, host_weight, elems_weight);

    // Debug: verify input data uploaded correctly
    const float *dev_input_data = (const float *)marmot_tensor_data_f32(env->ctx, dev_input);
    printf(
        "  host_input[0:4] = [%.6f, %.6f, %.6f, %.6f]\n", host_input[0], host_input[1], host_input[2], host_input[3]
    );
    printf(
        "  dev_input[0:4]  = [%.6f, %.6f, %.6f, %.6f]\n", dev_input_data[0], dev_input_data[1], dev_input_data[2],
        dev_input_data[3]
    );
    assert_int_equal(marmot_linear(env->ctx, dev_input, dev_weight, nullptr, dev_out), MARMOT_SUCCESS);
    assert_int_equal(
        marmot_reduce_sum(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = dev_out, .out = dev_sum, .axes = nullptr, .num_axes = 0, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );

    const float *cpu_sum_data = marmot_tensor_data_f32(cpu_ctx, cpu_sum);
    const float *dev_sum_data = marmot_tensor_data_f32(env->ctx, dev_sum);
    assert_non_null(cpu_sum_data);
    assert_non_null(dev_sum_data);
    // Debug: check matmul output before sum
    const float *cpu_out_data = (const float *)marmot_tensor_data_f32(cpu_ctx, cpu_out);
    const float *dev_out_data = (const float *)marmot_tensor_data_f32(env->ctx, dev_out);
    float max_matmul_diff = 0.0f;
    size_t max_diff_idx = 0;
    for (size_t i = 0; i < N * M; ++i) {
        float d = fabsf(cpu_out_data[i] - dev_out_data[i]);
        if (d > max_matmul_diff) {
            max_matmul_diff = d;
            max_diff_idx = i;
        }
    }
    printf(
        "  linear_followed_by_sum: max_matmul_diff=%.6f at idx=%zu (cpu=%.6f dev=%.6f)\n", max_matmul_diff,
        max_diff_idx, cpu_out_data[max_diff_idx], dev_out_data[max_diff_idx]
    );
    // Print first few values and check sparsity
    printf(
        "  cpu_out[0:4] = [%.6f, %.6f, %.6f, %.6f]\n", cpu_out_data[0], cpu_out_data[1], cpu_out_data[2],
        cpu_out_data[3]
    );
    printf(
        "  dev_out[0:4] = [%.6f, %.6f, %.6f, %.6f]\n", dev_out_data[0], dev_out_data[1], dev_out_data[2],
        dev_out_data[3]
    );
    size_t cpu_nonzero = 0, dev_nonzero = 0;
    for (size_t i = 0; i < N * M; ++i) {
        if (cpu_out_data[i] != 0.0f)
            cpu_nonzero++;
        if (dev_out_data[i] != 0.0f)
            dev_nonzero++;
    }
    printf("  cpu_nonzero=%zu/%zu, dev_nonzero=%zu/%zu\n", cpu_nonzero, N * M, dev_nonzero, N * M);

    float cpu_result = cpu_sum_data[0];
    float backend_result = dev_sum_data[0];
    float diff = fabsf(cpu_result - backend_result);
    printf("  linear_followed_by_sum: cpu_sum=%.6f backend_sum=%.6f diff=%.6f\n", cpu_result, backend_result, diff);
    assert_true(diff <= 1e-3f);

    marmot_test_tensor_destroy_all(8, dev_sum, dev_out, dev_weight, dev_input, cpu_sum, cpu_out, cpu_weight, cpu_input);
    marmot_destroy(cpu_ctx);
    free(host_weight);
    free(host_input);
}

static void test_matmul_pytorch_basic(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const size_t shape_a[2] = {2, 3};
    const size_t shape_b[2] = {3, 2};
    const size_t shape_out[2] = {2, 2};
    const size_t shape_bias[1] = {2};

    const float a_vals[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float b_vals[6] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    const float bias_vals[2] = {0.5f, -1.0f};
    const float expected[4] = {58.5f, 63.0f, 139.5f, 153.0f};

    marmot_tensor_t *a = marmot_tensor_create(env->ctx, shape_a, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *b = marmot_tensor_create(env->ctx, shape_b, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *bias = marmot_tensor_create(env->ctx, shape_bias, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(bias);
    assert_non_null(out);

    assert_int_equal(marmot_tensor_copy_from_host_buffer(env->ctx, a, a_vals, sizeof(a_vals)), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_copy_from_host_buffer(env->ctx, b, b_vals, sizeof(b_vals)), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_copy_from_host_buffer(env->ctx, bias, bias_vals, sizeof(bias_vals)), MARMOT_SUCCESS);

    marmot_matmul_epilogue_t ep = {.bias = bias};
    assert_int_equal(marmot_matmul_bias(env->ctx, a, b, &ep, out), MARMOT_SUCCESS);

    float result[4] = {0.0f};
    assert_int_equal(marmot_tensor_copy_to_host_buffer(env->ctx, out, result, sizeof(result)), MARMOT_SUCCESS);
    const float tol = 1e-5f;
    for (size_t i = 0; i < 4; ++i) {
        assert_true(fabsf(result[i] - expected[i]) <= tol);
    }

    marmot_test_tensor_destroy_all(4, out, bias, b, a);
}

static void matmul_reference(
    const float *a, const float *b, float *out, size_t batch, size_t M, size_t K, size_t N, bool b_broadcast
) {
    const size_t a_batch_stride = M * K;
    const size_t b_batch_stride = b_broadcast ? 0 : K * N;
    const size_t out_batch_stride = M * N;
    for (size_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        const float *a_ptr = a + batch_idx * a_batch_stride;
        const float *b_ptr = b + batch_idx * b_batch_stride;
        float *out_ptr = out + batch_idx * out_batch_stride;
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    acc += a_ptr[m * K + k] * b_ptr[k * N + n];
                }
                out_ptr[m * N + n] = acc;
            }
        }
    }
}

static void test_matmul_batched_basic(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const size_t batch = 2;
    const size_t M = 2;
    const size_t K = 3;
    const size_t N = 4;

    size_t a_shape[3] = {batch, M, K};
    size_t b_shape[3] = {batch, K, N};
    size_t out_shape[3] = {batch, M, N};

    marmot_tensor_t *a = marmot_tensor_create(env->ctx, a_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *b = marmot_tensor_create(env->ctx, b_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 3, MARMOT_DTYPE_FLOAT32);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(out);

    float host_a[batch * M * K];
    float host_b[batch * K * N];
    for (size_t i = 0; i < batch * M * K; ++i) {
        host_a[i] = (float)(i + 1) * 0.1f;
    }
    for (size_t i = 0; i < batch * K * N; ++i) {
        host_b[i] = (float)(i + 1) * 0.05f;
    }

    marmot_test_convert_f32_span(env, a, host_a, batch * M * K);
    marmot_test_convert_f32_span(env, b, host_b, batch * K * N);

    marmot_error_t status = marmot_matmul(env->ctx, a, b, nullptr, out);
    assert_int_equal(status, MARMOT_SUCCESS);

    float got[batch * M * N];
    marmot_test_fetch_f32_span(env, got, out, batch * M * N);

    float expected[batch * M * N];
    matmul_reference(host_a, host_b, expected, batch, M, K, N, false);

    for (size_t i = 0; i < batch * M * N; ++i) {
        assert_true(fabsf(got[i] - expected[i]) <= 1e-3f);
    }

    marmot_tensor_destroy(a);
    marmot_tensor_destroy(b);
    marmot_tensor_destroy(out);
}

static void test_matmul_batched_broadcast_rhs(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const size_t batch = 3;
    const size_t M = 2;
    constexpr size_t K = 3;
    constexpr size_t N = 2;

    size_t a_shape[3] = {batch, M, K};
    size_t b_shape[2] = {K, N};
    size_t out_shape[3] = {batch, M, N};

    marmot_tensor_t *a = marmot_tensor_create(env->ctx, a_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *b = marmot_tensor_create(env->ctx, b_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 3, MARMOT_DTYPE_FLOAT32);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(out);

    float host_a[batch * M * K];
    float host_b[K * N];
    for (size_t i = 0; i < batch * M * K; ++i) {
        host_a[i] = (float)(i + 1) * 0.01f;
    }
    for (size_t i = 0; i < K * N; ++i) {
        host_b[i] = (float)(i + 1) * 0.02f;
    }

    marmot_test_convert_f32_span(env, a, host_a, batch * M * K);
    marmot_test_convert_f32_span(env, b, host_b, K * N);

    marmot_error_t status = marmot_matmul(env->ctx, a, b, nullptr, out);
    assert_int_equal(status, MARMOT_SUCCESS);

    float got[batch * M * N];
    marmot_test_fetch_f32_span(env, got, out, batch * M * N);

    float expected[batch * M * N];
    matmul_reference(host_a, host_b, expected, batch, M, K, N, true);
    for (size_t i = 0; i < batch * M * N; ++i) {
        assert_true(fabsf(got[i] - expected[i]) <= 1e-3f);
    }

    marmot_tensor_destroy(a);
    marmot_tensor_destroy(b);
    marmot_tensor_destroy(out);
}

static void test_matmul_vector_matrix(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    constexpr size_t K = 3;
    constexpr size_t N = 2;

    size_t a_shape[1] = {K};
    size_t b_shape[2] = {K, N};
    size_t out_shape[1] = {N};

    marmot_tensor_t *a = marmot_tensor_create(env->ctx, a_shape, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *b = marmot_tensor_create(env->ctx, b_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(out);

    float host_a[K] = {0.1f, -0.2f, 0.3f};
    float host_b[K * N] = {1.0f, 2.0f, -1.0f, 0.5f, 0.25f, 0.75f};

    marmot_test_convert_f32_span(env, a, host_a, K);
    marmot_test_convert_f32_span(env, b, host_b, K * N);
    assert_int_equal(marmot_matmul(env->ctx, a, b, nullptr, out), MARMOT_SUCCESS);

    float got[N];
    marmot_test_fetch_f32_span(env, got, out, N);
    float expected[N];
    float tmp_a[1 * K];
    float tmp_b[1 * K * N];
    memcpy(tmp_a, host_a, sizeof(host_a));
    memcpy(tmp_b, host_b, sizeof(host_b));
    matmul_reference(tmp_a, tmp_b, expected, 1, 1, K, N, false);
    for (size_t i = 0; i < N; ++i) {
        assert_true(fabsf(got[i] - expected[i]) <= 1e-5f);
    }

    marmot_tensor_destroy(a);
    marmot_tensor_destroy(b);
    marmot_tensor_destroy(out);
}

static void test_matmul_matrix_vector(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    constexpr size_t M = 3;
    constexpr size_t K = 2;

    size_t a_shape[2] = {M, K};
    size_t b_shape[1] = {K};
    size_t out_shape[1] = {M};

    marmot_tensor_t *a = marmot_tensor_create(env->ctx, a_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *b = marmot_tensor_create(env->ctx, b_shape, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(out);

    float host_a[M * K] = {1.0f, 2.0f, -1.0f, 0.5f, 0.25f, 0.75f};
    float host_b[K] = {0.4f, -0.6f};

    marmot_test_convert_f32_span(env, a, host_a, M * K);
    marmot_test_convert_f32_span(env, b, host_b, K);
    assert_int_equal(marmot_matmul(env->ctx, a, b, nullptr, out), MARMOT_SUCCESS);

    float got[M];
    marmot_test_fetch_f32_span(env, got, out, M);
    float expected[M];
    float tmp_b[K * 1];
    memcpy(tmp_b, host_b, sizeof(host_b));
    matmul_reference(host_a, tmp_b, expected, 1, M, K, 1, true);
    for (size_t i = 0; i < M; ++i) {
        assert_true(fabsf(got[i] - expected[i]) <= 1e-5f);
    }

    marmot_tensor_destroy(a);
    marmot_tensor_destroy(b);
    marmot_tensor_destroy(out);
}

static void test_matmul_vector_dot(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    constexpr size_t K = 4;

    size_t a_shape[1] = {K};
    size_t b_shape[1] = {K};
    size_t out_shape[1] = {1};

    marmot_tensor_t *a = marmot_tensor_create(env->ctx, a_shape, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *b = marmot_tensor_create(env->ctx, b_shape, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(out);

    float host_a[K] = {0.5f, -0.5f, 1.0f, 2.0f};
    float host_b[K] = {1.0f, 2.0f, -1.0f, 0.25f};

    marmot_test_convert_f32_span(env, a, host_a, K);
    marmot_test_convert_f32_span(env, b, host_b, K);
    assert_int_equal(marmot_matmul(env->ctx, a, b, nullptr, out), MARMOT_SUCCESS);

    float got[1] = {0.0f};
    marmot_test_fetch_f32_span(env, got, out, 1);
    float expected[1];
    matmul_reference(host_a, host_b, expected, 1, 1, K, 1, false);
    assert_true(fabsf(got[0] - expected[0]) <= 1e-5f);

    marmot_tensor_destroy(a);
    marmot_tensor_destroy(b);
    marmot_tensor_destroy(out);
}

static void test_matmul_shape_mismatch(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    size_t shape_a[2] = {2, 3};
    size_t shape_b[2] = {2, 2};
    size_t shape_out[2] = {2, 2};

    marmot_tensor_t *a = marmot_tensor_create(env->ctx, shape_a, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *b = marmot_tensor_create(env->ctx, shape_b, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(out);

    marmot_error_t err = marmot_matmul(env->ctx, a, b, nullptr, out);
    assert_int_not_equal(err, MARMOT_SUCCESS);

    marmot_test_tensor_destroy_all(3, out, b, a);
}

static void test_matmul_qkv_packed_weight_correctness(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
#ifdef __APPLE__
    if (env == nullptr || env->backend != MARMOT_BACKEND_METAL) {
        skip();
        return;
    }
    const size_t N = 2;
    const size_t M = 64;
    const size_t K = 96;
    const size_t elems_out = N * M;
    float *ref_q = (float *)malloc(elems_out * sizeof(float));
    float *ref_k = (float *)malloc(elems_out * sizeof(float));
    float *ref_v = (float *)malloc(elems_out * sizeof(float));
    float *packed_q = (float *)malloc(elems_out * sizeof(float));
    float *packed_k = (float *)malloc(elems_out * sizeof(float));
    float *packed_v = (float *)malloc(elems_out * sizeof(float));
    if (ref_q == nullptr || ref_k == nullptr || ref_v == nullptr || packed_q == nullptr || packed_k == nullptr ||
        packed_v == nullptr) {
        free(ref_q);
        free(ref_k);
        free(ref_v);
        free(packed_q);
        free(packed_k);
        free(packed_v);
        skip();
        return;
    }
    bool ran_baseline = run_metal_qkv_case(env, false, ref_q, ref_k, ref_v, N, M, K);
    bool ran_packed = ran_baseline && run_metal_qkv_case(env, true, packed_q, packed_k, packed_v, N, M, K);
    if (!ran_packed) {
        free(ref_q);
        free(ref_k);
        free(ref_v);
        free(packed_q);
        free(packed_k);
        free(packed_v);
        skip();
        return;
    }
    for (size_t i = 0; i < elems_out; ++i) {
        float diff_q = fabsf(ref_q[i] - packed_q[i]);
        float diff_k = fabsf(ref_k[i] - packed_k[i]);
        float diff_v = fabsf(ref_v[i] - packed_v[i]);
        assert_true(diff_q <= 1e-5f);
        assert_true(diff_k <= 1e-5f);
        assert_true(diff_v <= 1e-5f);
    }
    free(ref_q);
    free(ref_k);
    free(ref_v);
    free(packed_q);
    free(packed_k);
    free(packed_v);
#else
    (void)env;
    skip();
#endif
}

static void test_matmul_qkv_threshold_toggle(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
#ifdef __APPLE__
    if (env == nullptr || env->backend != MARMOT_BACKEND_METAL) {
        skip();
        return;
    }
    const size_t N = 1;
    const size_t M = 48;
    const size_t K = 64;
    const size_t elems_out = N * M;
    float *ref_q = (float *)malloc(elems_out * sizeof(float));
    float *ref_k = (float *)malloc(elems_out * sizeof(float));
    float *ref_v = (float *)malloc(elems_out * sizeof(float));
    float *packed_q = (float *)malloc(elems_out * sizeof(float));
    float *packed_k = (float *)malloc(elems_out * sizeof(float));
    float *packed_v = (float *)malloc(elems_out * sizeof(float));
    if (ref_q == nullptr || ref_k == nullptr || ref_v == nullptr || packed_q == nullptr || packed_k == nullptr ||
        packed_v == nullptr) {
        free(ref_q);
        free(ref_k);
        free(ref_v);
        free(packed_q);
        free(packed_k);
        free(packed_v);
        skip();
        return;
    }
    bool ran_baseline = run_metal_qkv_case(env, false, ref_q, ref_k, ref_v, N, M, K);
    bool ran_packed = ran_baseline && run_metal_qkv_case(env, true, packed_q, packed_k, packed_v, N, M, K);
    if (!ran_packed) {
        free(ref_q);
        free(ref_k);
        free(ref_v);
        free(packed_q);
        free(packed_k);
        free(packed_v);
        skip();
        return;
    }
    for (size_t i = 0; i < elems_out; ++i) {
        float diff_q = fabsf(ref_q[i] - packed_q[i]);
        float diff_k = fabsf(ref_k[i] - packed_k[i]);
        float diff_v = fabsf(ref_v[i] - packed_v[i]);
        assert_true(diff_q <= 1e-5f);
        assert_true(diff_k <= 1e-5f);
        assert_true(diff_v <= 1e-5f);
    }
    free(ref_q);
    free(ref_k);
    free(ref_v);
    free(packed_q);
    free(packed_k);
    free(packed_v);
#else
    (void)env;
    skip();
#endif
}

static void test_matmul_qkv_separate_rope_metal(void **state) {
    (void)state;
#ifdef __APPLE__
    const size_t N = 3;
    const size_t M = 32;
    const size_t K = 64;
    const size_t elems_input = N * K;
    const size_t elems_proj = M * K;
    const size_t elems_out = N * M;
    float host_input[elems_input];
    float host_wq[elems_proj];
    float host_wk[elems_proj];
    float host_wv[elems_proj];
    float host_bq[M];
    float host_bk[M];
    float host_bv[M];
    float fused_q[elems_out];
    float fused_k[elems_out];
    float fused_v[elems_out];
    float base_q[elems_out];
    float base_k[elems_out];
    float base_v[elems_out];
    float expected_q[elems_out];
    float expected_k[elems_out];

    for (size_t i = 0; i < elems_input; ++i) {
        host_input[i] = 0.05f * sinf((float)(i + 1) * 0.17f);
    }
    for (size_t i = 0; i < elems_proj; ++i) {
        host_wq[i] = 0.02f * cosf((float)(i + 3) * 0.13f);
        host_wk[i] = -0.015f * sinf((float)(i + 5) * 0.07f);
        host_wv[i] = 0.01f * cosf((float)(i + 7) * 0.09f);
    }
    for (size_t i = 0; i < M; ++i) {
        host_bq[i] = 0.003f * (float)(i + 1);
        host_bk[i] = -0.004f * (float)(i + 3);
        host_bv[i] = 0.002f * (float)(i + 5);
    }

    const size_t shape_input[2] = {N, K};
    const size_t shape_proj[2] = {M, K};
    const size_t shape_bias[1] = {M};
    const size_t shape_out[2] = {N, M};
    const size_t shape_pos[1] = {N};

    bool should_skip = false;
    marmot_context_t *ctx = nullptr;
    marmot_tensor_t *positions = nullptr;
    marmot_tensor_t *input = nullptr;
    marmot_tensor_t *wq = nullptr;
    marmot_tensor_t *wk = nullptr;
    marmot_tensor_t *wv = nullptr;
    marmot_tensor_t *bq = nullptr;
    marmot_tensor_t *bk = nullptr;
    marmot_tensor_t *bv = nullptr;
    marmot_tensor_t *out_q = nullptr;
    marmot_tensor_t *out_k = nullptr;
    marmot_tensor_t *out_v = nullptr;

    positions = marmot_tensor_create(nullptr, shape_pos, 1, MARMOT_DTYPE_FLOAT32);
    ctx = marmot_init(MARMOT_BACKEND_METAL);
    if (positions == nullptr || ctx == nullptr) {
        should_skip = true;
        goto cleanup;
    }
    float *pos_data = (float *)positions->data;
    for (size_t i = 0; i < N; ++i) {
        pos_data[i] = (float)(i * 2 + 1);
    }

    input = marmot_tensor_create(ctx, shape_input, 2, MARMOT_DTYPE_FLOAT32);
    wq = marmot_tensor_create(ctx, shape_proj, 2, MARMOT_DTYPE_FLOAT32);
    wk = marmot_tensor_create(ctx, shape_proj, 2, MARMOT_DTYPE_FLOAT32);
    wv = marmot_tensor_create(ctx, shape_proj, 2, MARMOT_DTYPE_FLOAT32);
    bq = marmot_tensor_create(ctx, shape_bias, 1, MARMOT_DTYPE_FLOAT32);
    bk = marmot_tensor_create(ctx, shape_bias, 1, MARMOT_DTYPE_FLOAT32);
    bv = marmot_tensor_create(ctx, shape_bias, 1, MARMOT_DTYPE_FLOAT32);
    out_q = marmot_tensor_create(ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    out_k = marmot_tensor_create(ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    out_v = marmot_tensor_create(ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    if (input == nullptr || wq == nullptr || wk == nullptr || wv == nullptr || bq == nullptr || bk == nullptr ||
        bv == nullptr || out_q == nullptr || out_k == nullptr || out_v == nullptr) {
        should_skip = true;
        goto cleanup;
    }

    size_t bytes_input = elems_input * sizeof(float);
    size_t bytes_proj = elems_proj * sizeof(float);
    size_t bytes_bias = M * sizeof(float);
    size_t bytes_out = elems_out * sizeof(float);
    assert_int_equal(marmot_tensor_copy_from_host_buffer(ctx, input, host_input, bytes_input), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_copy_from_host_buffer(ctx, wq, host_wq, bytes_proj), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_copy_from_host_buffer(ctx, wk, host_wk, bytes_proj), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_copy_from_host_buffer(ctx, wv, host_wv, bytes_proj), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_copy_from_host_buffer(ctx, bq, host_bq, bytes_bias), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_copy_from_host_buffer(ctx, bk, host_bk, bytes_bias), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_copy_from_host_buffer(ctx, bv, host_bv, bytes_bias), MARMOT_SUCCESS);

    marmot_rope_params_t rope = marmot_rope_params_default();
    rope.positions = positions;
    rope.theta = 5000.0f;
    rope.apply_to_q = true;
    rope.apply_to_k = true;

    marmot_matmul_qkv_desc_t desc = {
        .input = input,
        .layout = MARMOT_QKV_LAYOUT_SEPARATE,
        .separate =
            {
                .wq = wq,
                .wk = wk,
                .wv = wv,
                .bq = bq,
                .bk = bk,
                .bv = bv,
            },
        .epilogue = nullptr,
        .out_q = out_q,
        .out_k = out_k,
        .out_v = out_v,
        .rope_params = &rope,
    };

    assert_int_equal(marmot_matmul_qkv(ctx, &desc), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_copy_to_host_buffer(ctx, out_q, fused_q, bytes_out), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_copy_to_host_buffer(ctx, out_k, fused_k, bytes_out), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_copy_to_host_buffer(ctx, out_v, fused_v, bytes_out), MARMOT_SUCCESS);

    desc.rope_params = nullptr;
    assert_int_equal(marmot_matmul_qkv(ctx, &desc), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_copy_to_host_buffer(ctx, out_q, base_q, bytes_out), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_copy_to_host_buffer(ctx, out_k, base_k, bytes_out), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_copy_to_host_buffer(ctx, out_v, base_v, bytes_out), MARMOT_SUCCESS);

    reference_rope_rotate(expected_q, base_q, N, M, pos_data, rope.theta);
    reference_rope_rotate(expected_k, base_k, N, M, pos_data, rope.theta);

    const float tol = 1e-3f;
    for (size_t i = 0; i < elems_out; ++i) {
        assert_float_equal(fused_q[i], expected_q[i], tol);
        assert_float_equal(fused_k[i], expected_k[i], tol);
        assert_float_equal(fused_v[i], base_v[i], tol);
    }

cleanup:
    marmot_test_tensor_destroy_all(10, out_v, out_k, out_q, bv, bk, bq, wv, wk, wq, input);
    if (positions != nullptr) {
        marmot_tensor_destroy(positions);
    }
    if (ctx != nullptr) {
        marmot_destroy(ctx);
    }
    if (should_skip) {
        skip();
    }
#else
    (void)state;
    skip();
#endif
}

static void test_matmul_qkv_post_residual_gelu_metal(void **state) {
    (void)state;
#ifdef __APPLE__
    const size_t N = 2;
    const size_t M = 48;
    const size_t K = 64;
    const size_t elems_input = N * K;
    const size_t elems_weight = 3 * M * K;
    const size_t elems_bias = 3 * M;
    const size_t elems_out = N * M;
    float *host_input = (float *)malloc(elems_input * sizeof(float));
    float *host_weight = (float *)malloc(elems_weight * sizeof(float));
    float *host_bias = (float *)malloc(elems_bias * sizeof(float));
    float *host_residual = (float *)malloc(elems_out * sizeof(float));
    float *expected_q = (float *)malloc(elems_out * sizeof(float));
    float *expected_k = (float *)malloc(elems_out * sizeof(float));
    float *expected_v = (float *)malloc(elems_out * sizeof(float));
    float *gpu_q = (float *)malloc(elems_out * sizeof(float));
    float *gpu_k = (float *)malloc(elems_out * sizeof(float));
    float *gpu_v = (float *)malloc(elems_out * sizeof(float));
    if (host_input == nullptr || host_weight == nullptr || host_bias == nullptr || host_residual == nullptr ||
        expected_q == nullptr || expected_k == nullptr || expected_v == nullptr || gpu_q == nullptr ||
        gpu_k == nullptr || gpu_v == nullptr) {
        free(host_input);
        free(host_weight);
        free(host_bias);
        free(host_residual);
        free(expected_q);
        free(expected_k);
        free(expected_v);
        free(gpu_q);
        free(gpu_k);
        free(gpu_v);
        skip();
        return;
    }
    fill_span(host_input, elems_input, 0.01f, -0.2f);
    fill_span(host_weight, elems_weight, 0.002f, 0.05f);
    fill_span(host_bias, elems_bias, 0.03f, -0.1f);
    fill_span(host_residual, elems_out, 0.015f, 0.02f);

    compute_qkv_expected_float(
        expected_q, expected_k, expected_v, host_input, host_weight, host_bias, host_residual, N, K, M,
        MARMOT_DEVICE_UNARY_GELU, nullptr
    );

    marmot_context_t *metal_ctx = marmot_init(MARMOT_BACKEND_METAL);
    if (metal_ctx == nullptr) {
        free(host_input);
        free(host_weight);
        free(host_bias);
        free(host_residual);
        free(expected_q);
        free(expected_k);
        free(expected_v);
        free(gpu_q);
        free(gpu_k);
        free(gpu_v);
        skip();
        return;
    }
    size_t shape_input[] = {N, K};
    size_t shape_weight[] = {3 * M, K};
    size_t shape_out[] = {N, M};
    size_t shape_bias[] = {3 * M};

    marmot_tensor_t *metal_input = marmot_tensor_create(metal_ctx, shape_input, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *metal_weight = marmot_tensor_create(metal_ctx, shape_weight, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *metal_bias = marmot_tensor_create(metal_ctx, shape_bias, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *metal_residual = marmot_tensor_create(metal_ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *metal_q_tensor = marmot_tensor_create(metal_ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *metal_k_tensor = marmot_tensor_create(metal_ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *metal_v_tensor = marmot_tensor_create(metal_ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *metal_sum_q = nullptr;
    marmot_tensor_t *metal_sum_k = nullptr;
    marmot_tensor_t *metal_sum_v = nullptr;
    marmot_tensor_t *metal_act_q = nullptr;
    marmot_tensor_t *metal_act_k = nullptr;
    marmot_tensor_t *metal_act_v = nullptr;
    if (metal_input == nullptr || metal_weight == nullptr || metal_bias == nullptr || metal_residual == nullptr ||
        metal_q_tensor == nullptr || metal_k_tensor == nullptr || metal_v_tensor == nullptr) {
        marmot_test_tensor_destroy_all(
            13, metal_act_v, metal_act_k, metal_act_q, metal_sum_v, metal_sum_k, metal_sum_q, metal_v_tensor,
            metal_k_tensor, metal_q_tensor, metal_residual, metal_bias, metal_weight, metal_input
        );
        marmot_destroy(metal_ctx);
        free(host_input);
        free(host_weight);
        free(host_bias);
        free(host_residual);
        free(expected_q);
        free(expected_k);
        free(expected_v);
        free(gpu_q);
        free(gpu_k);
        free(gpu_v);
        skip();
        return;
    }
    size_t bytes_input = elems_input * sizeof(float);
    size_t bytes_weight = elems_weight * sizeof(float);
    size_t bytes_bias = elems_bias * sizeof(float);
    size_t bytes_residual = elems_out * sizeof(float);
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(metal_ctx, metal_input, host_input, bytes_input), MARMOT_SUCCESS
    );
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(metal_ctx, metal_weight, host_weight, bytes_weight), MARMOT_SUCCESS
    );
    assert_int_equal(marmot_tensor_copy_from_host_buffer(metal_ctx, metal_bias, host_bias, bytes_bias), MARMOT_SUCCESS);
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(metal_ctx, metal_residual, host_residual, bytes_residual), MARMOT_SUCCESS
    );
    marmot_matmul_qkv_desc_t metal_desc = {
        .input = metal_input,
        .layout = MARMOT_QKV_LAYOUT_FUSED,
        .fused =
            {
                .weight = metal_weight,
                .bias = metal_bias,
            },
        .out_q = metal_q_tensor,
        .out_k = metal_k_tensor,
        .out_v = metal_v_tensor,
    };
    marmot_error_t status = marmot_matmul_qkv(metal_ctx, &metal_desc);
    if (status == MARMOT_ERROR_NOT_IMPLEMENTED) {
        marmot_test_tensor_destroy_all(
            13, metal_act_v, metal_act_k, metal_act_q, metal_sum_v, metal_sum_k, metal_sum_q, metal_v_tensor,
            metal_k_tensor, metal_q_tensor, metal_residual, metal_bias, metal_weight, metal_input
        );
        marmot_destroy(metal_ctx);
        free(host_input);
        free(host_weight);
        free(host_bias);
        free(host_residual);
        free(expected_q);
        free(expected_k);
        free(expected_v);
        free(gpu_q);
        free(gpu_k);
        free(gpu_v);
        skip();
        return;
    }
    assert_int_equal(status, MARMOT_SUCCESS);

    metal_sum_q = marmot_tensor_create(metal_ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    metal_sum_k = marmot_tensor_create(metal_ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    metal_sum_v = marmot_tensor_create(metal_ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    metal_act_q = marmot_tensor_create(metal_ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    metal_act_k = marmot_tensor_create(metal_ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    metal_act_v = marmot_tensor_create(metal_ctx, shape_out, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(metal_sum_q);
    assert_non_null(metal_sum_k);
    assert_non_null(metal_sum_v);
    assert_non_null(metal_act_q);
    assert_non_null(metal_act_k);
    assert_non_null(metal_act_v);

    marmot_error_t act_status =
        apply_activation_op(metal_ctx, MARMOT_DEVICE_UNARY_GELU, metal_q_tensor, metal_act_q, nullptr);
    if (act_status == MARMOT_ERROR_NOT_IMPLEMENTED || act_status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        marmot_test_tensor_destroy_all(
            13, metal_act_v, metal_act_k, metal_act_q, metal_sum_v, metal_sum_k, metal_sum_q, metal_v_tensor,
            metal_k_tensor, metal_q_tensor, metal_residual, metal_bias, metal_weight, metal_input
        );
        marmot_destroy(metal_ctx);
        free(host_input);
        free(host_weight);
        free(host_bias);
        free(host_residual);
        free(expected_q);
        free(expected_k);
        free(expected_v);
        free(gpu_q);
        free(gpu_k);
        free(gpu_v);
        skip();
        return;
    }
    assert_int_equal(act_status, MARMOT_SUCCESS);
    act_status = apply_activation_op(metal_ctx, MARMOT_DEVICE_UNARY_GELU, metal_k_tensor, metal_act_k, nullptr);
    if (act_status == MARMOT_ERROR_NOT_IMPLEMENTED || act_status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        marmot_test_tensor_destroy_all(
            13, metal_act_v, metal_act_k, metal_act_q, metal_sum_v, metal_sum_k, metal_sum_q, metal_v_tensor,
            metal_k_tensor, metal_q_tensor, metal_residual, metal_bias, metal_weight, metal_input
        );
        marmot_destroy(metal_ctx);
        free(host_input);
        free(host_weight);
        free(host_bias);
        free(host_residual);
        free(expected_q);
        free(expected_k);
        free(expected_v);
        free(gpu_q);
        free(gpu_k);
        free(gpu_v);
        skip();
        return;
    }
    assert_int_equal(act_status, MARMOT_SUCCESS);
    act_status = apply_activation_op(metal_ctx, MARMOT_DEVICE_UNARY_GELU, metal_v_tensor, metal_act_v, nullptr);
    if (act_status == MARMOT_ERROR_NOT_IMPLEMENTED || act_status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        marmot_test_tensor_destroy_all(
            13, metal_act_v, metal_act_k, metal_act_q, metal_sum_v, metal_sum_k, metal_sum_q, metal_v_tensor,
            metal_k_tensor, metal_q_tensor, metal_residual, metal_bias, metal_weight, metal_input
        );
        marmot_destroy(metal_ctx);
        free(host_input);
        free(host_weight);
        free(host_bias);
        free(host_residual);
        free(expected_q);
        free(expected_k);
        free(expected_v);
        free(gpu_q);
        free(gpu_k);
        free(gpu_v);
        skip();
        return;
    }
    assert_int_equal(act_status, MARMOT_SUCCESS);

    marmot_error_t add_status = marmot_add(metal_ctx, metal_act_q, metal_residual, metal_sum_q);
    if (add_status == MARMOT_ERROR_NOT_IMPLEMENTED || add_status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        marmot_test_tensor_destroy_all(
            13, metal_act_v, metal_act_k, metal_act_q, metal_sum_v, metal_sum_k, metal_sum_q, metal_v_tensor,
            metal_k_tensor, metal_q_tensor, metal_residual, metal_bias, metal_weight, metal_input
        );
        marmot_destroy(metal_ctx);
        free(host_input);
        free(host_weight);
        free(host_bias);
        free(host_residual);
        free(expected_q);
        free(expected_k);
        free(expected_v);
        free(gpu_q);
        free(gpu_k);
        free(gpu_v);
        skip();
        return;
    }
    assert_int_equal(add_status, MARMOT_SUCCESS);
    add_status = marmot_add(metal_ctx, metal_act_k, metal_residual, metal_sum_k);
    if (add_status == MARMOT_ERROR_NOT_IMPLEMENTED || add_status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        marmot_test_tensor_destroy_all(
            13, metal_act_v, metal_act_k, metal_act_q, metal_sum_v, metal_sum_k, metal_sum_q, metal_v_tensor,
            metal_k_tensor, metal_q_tensor, metal_residual, metal_bias, metal_weight, metal_input
        );
        marmot_destroy(metal_ctx);
        free(host_input);
        free(host_weight);
        free(host_bias);
        free(host_residual);
        free(expected_q);
        free(expected_k);
        free(expected_v);
        free(gpu_q);
        free(gpu_k);
        free(gpu_v);
        skip();
        return;
    }
    assert_int_equal(add_status, MARMOT_SUCCESS);
    add_status = marmot_add(metal_ctx, metal_act_v, metal_residual, metal_sum_v);
    if (add_status == MARMOT_ERROR_NOT_IMPLEMENTED || add_status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        marmot_test_tensor_destroy_all(
            13, metal_act_v, metal_act_k, metal_act_q, metal_sum_v, metal_sum_k, metal_sum_q, metal_v_tensor,
            metal_k_tensor, metal_q_tensor, metal_residual, metal_bias, metal_weight, metal_input
        );
        marmot_destroy(metal_ctx);
        free(host_input);
        free(host_weight);
        free(host_bias);
        free(host_residual);
        free(expected_q);
        free(expected_k);
        free(expected_v);
        free(gpu_q);
        free(gpu_k);
        free(gpu_v);
        skip();
        return;
    }
    assert_int_equal(add_status, MARMOT_SUCCESS);

    size_t bytes_out = elems_out * sizeof(float);
    assert_int_equal(marmot_tensor_copy_to_host_buffer(metal_ctx, metal_sum_q, gpu_q, bytes_out), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_copy_to_host_buffer(metal_ctx, metal_sum_k, gpu_k, bytes_out), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_copy_to_host_buffer(metal_ctx, metal_sum_v, gpu_v, bytes_out), MARMOT_SUCCESS);

    for (size_t i = 0; i < elems_out; ++i) {
        assert_true(fabsf(expected_q[i] - gpu_q[i]) <= 5e-4f);
        assert_true(fabsf(expected_k[i] - gpu_k[i]) <= 5e-4f);
        assert_true(fabsf(expected_v[i] - gpu_v[i]) <= 5e-4f);
    }

    marmot_test_tensor_destroy_all(
        13, metal_act_v, metal_act_k, metal_act_q, metal_sum_v, metal_sum_k, metal_sum_q, metal_v_tensor,
        metal_k_tensor, metal_q_tensor, metal_residual, metal_bias, metal_weight, metal_input
    );
    marmot_destroy(metal_ctx);
    free(host_input);
    free(host_weight);
    free(host_bias);
    free(host_residual);
    free(expected_q);
    free(expected_k);
    free(expected_v);
    free(gpu_q);
    free(gpu_k);
    free(gpu_v);
#else
    skip();
#endif
}

static void test_matmul_qkv_post_residual_leaky_relu_metal(void **state) {
#ifdef __APPLE__
    (void)state;
    const size_t N = 2;
    const size_t K = 4;
    const size_t M = 5;
    const marmot_dtype_t dtype = MARMOT_DTYPE_FLOAT32;
    float host_input[N * K];
    float host_weight[3 * M * K];
    float host_bias[3 * M];
    float host_residual[N * M];
    for (size_t i = 0; i < N * K; ++i) {
        host_input[i] = 0.15f * sinf((float)(i + 1));
    }
    for (size_t i = 0; i < 3 * M * K; ++i) {
        host_weight[i] = 0.07f * cosf((float)(i + 3));
    }
    for (size_t i = 0; i < 3 * M; ++i) {
        host_bias[i] = 0.01f * (float)(i + 1);
    }
    for (size_t i = 0; i < N * M; ++i) {
        host_residual[i] = -0.02f * (float)(i + 2);
    }

    marmot_activation_params_t act_params = {
        .parameter_tensor = nullptr,
        .bias = nullptr,
        .alpha = 0.2f,
        .beta = 0.0f,
        .gamma = 0.0f,
    };

    float expected_q[N * M];
    float expected_k[N * M];
    float expected_v[N * M];
    compute_qkv_expected_float(
        expected_q, expected_k, expected_v, host_input, host_weight, host_bias, host_residual, N, K, M,
        MARMOT_DEVICE_UNARY_LEAKY_RELU, &act_params
    );

    marmot_context_t *metal_ctx = marmot_init(MARMOT_BACKEND_METAL);
    if (metal_ctx == nullptr) {
        skip();
        return;
    }
    size_t shape_input[] = {N, K};
    size_t shape_weight[] = {3 * M, K};
    size_t shape_bias[] = {3 * M};
    size_t shape_out[] = {N, M};
    marmot_tensor_t *metal_input = marmot_tensor_create(metal_ctx, shape_input, 2, dtype);
    marmot_tensor_t *metal_weight = marmot_tensor_create(metal_ctx, shape_weight, 2, dtype);
    marmot_tensor_t *metal_bias = marmot_tensor_create(metal_ctx, shape_bias, 1, dtype);
    marmot_tensor_t *metal_residual = marmot_tensor_create(metal_ctx, shape_out, 2, dtype);
    marmot_tensor_t *metal_q = marmot_tensor_create(metal_ctx, shape_out, 2, dtype);
    marmot_tensor_t *metal_k = marmot_tensor_create(metal_ctx, shape_out, 2, dtype);
    marmot_tensor_t *metal_v = marmot_tensor_create(metal_ctx, shape_out, 2, dtype);
    marmot_tensor_t *metal_sum_q = nullptr;
    marmot_tensor_t *metal_sum_k = nullptr;
    marmot_tensor_t *metal_sum_v = nullptr;
    marmot_tensor_t *metal_act_q = nullptr;
    marmot_tensor_t *metal_act_k = nullptr;
    marmot_tensor_t *metal_act_v = nullptr;
    assert_non_null(metal_input);
    assert_non_null(metal_weight);
    assert_non_null(metal_bias);
    assert_non_null(metal_residual);
    marmot_test_convert_span(
        &(marmot_test_env_t){.backend = MARMOT_BACKEND_METAL, .ctx = metal_ctx}, metal_input, dtype, host_input, N * K
    );
    marmot_test_convert_span(
        &(marmot_test_env_t){.backend = MARMOT_BACKEND_METAL, .ctx = metal_ctx}, metal_weight, dtype, host_weight,
        3 * M * K
    );
    marmot_test_convert_span(
        &(marmot_test_env_t){.backend = MARMOT_BACKEND_METAL, .ctx = metal_ctx}, metal_bias, dtype, host_bias, 3 * M
    );
    marmot_test_convert_span(
        &(marmot_test_env_t){.backend = MARMOT_BACKEND_METAL, .ctx = metal_ctx}, metal_residual, dtype, host_residual,
        N * M
    );

    marmot_matmul_qkv_desc_t metal_desc = {
        .input = metal_input,
        .layout = MARMOT_QKV_LAYOUT_FUSED,
        .fused =
            {
                .weight = metal_weight,
                .bias = metal_bias,
            },
        .out_q = metal_q,
        .out_k = metal_k,
        .out_v = metal_v,
    };
    marmot_clear_error();
    marmot_error_t metal_status = marmot_matmul_qkv(metal_ctx, &metal_desc);
    if (metal_status != MARMOT_SUCCESS) {
        const char *detail = marmot_get_last_error_detail();
        const marmot_error_info_t *info = marmot_get_last_error_info();
        fail_msg(
            "Metal inline leaky ReLU QKV failed: %s (err=%d, at %s:%d)", detail != nullptr ? detail : "no detail",
            (int)metal_status, info != nullptr ? info->file : "unknown", info != nullptr ? info->line : 0
        );
    }
    assert_int_equal(metal_status, MARMOT_SUCCESS);

    metal_sum_q = marmot_tensor_create(metal_ctx, shape_out, 2, dtype);
    metal_sum_k = marmot_tensor_create(metal_ctx, shape_out, 2, dtype);
    metal_sum_v = marmot_tensor_create(metal_ctx, shape_out, 2, dtype);
    metal_act_q = marmot_tensor_create(metal_ctx, shape_out, 2, dtype);
    metal_act_k = marmot_tensor_create(metal_ctx, shape_out, 2, dtype);
    metal_act_v = marmot_tensor_create(metal_ctx, shape_out, 2, dtype);
    assert_non_null(metal_sum_q);
    assert_non_null(metal_sum_k);
    assert_non_null(metal_sum_v);
    assert_non_null(metal_act_q);
    assert_non_null(metal_act_k);
    assert_non_null(metal_act_v);

    marmot_error_t act_status =
        apply_activation_op(metal_ctx, MARMOT_DEVICE_UNARY_LEAKY_RELU, metal_q, metal_act_q, &act_params);
    assert_int_equal(act_status, MARMOT_SUCCESS);
    act_status = apply_activation_op(metal_ctx, MARMOT_DEVICE_UNARY_LEAKY_RELU, metal_k, metal_act_k, &act_params);
    assert_int_equal(act_status, MARMOT_SUCCESS);
    act_status = apply_activation_op(metal_ctx, MARMOT_DEVICE_UNARY_LEAKY_RELU, metal_v, metal_act_v, &act_params);
    assert_int_equal(act_status, MARMOT_SUCCESS);

    marmot_error_t add_status = marmot_add(metal_ctx, metal_act_q, metal_residual, metal_sum_q);
    assert_int_equal(add_status, MARMOT_SUCCESS);
    add_status = marmot_add(metal_ctx, metal_act_k, metal_residual, metal_sum_k);
    assert_int_equal(add_status, MARMOT_SUCCESS);
    add_status = marmot_add(metal_ctx, metal_act_v, metal_residual, metal_sum_v);
    assert_int_equal(add_status, MARMOT_SUCCESS);

    float gpu_q[N * M];
    float gpu_k[N * M];
    float gpu_v[N * M];
    marmot_test_fetch_span(
        &(marmot_test_env_t){.backend = MARMOT_BACKEND_METAL, .ctx = metal_ctx}, gpu_q, dtype, metal_sum_q, N * M
    );
    marmot_test_fetch_span(
        &(marmot_test_env_t){.backend = MARMOT_BACKEND_METAL, .ctx = metal_ctx}, gpu_k, dtype, metal_sum_k, N * M
    );
    marmot_test_fetch_span(
        &(marmot_test_env_t){.backend = MARMOT_BACKEND_METAL, .ctx = metal_ctx}, gpu_v, dtype, metal_sum_v, N * M
    );

    for (size_t i = 0; i < N * M; ++i) {
        assert_float_equal(gpu_q[i], expected_q[i], 5e-4f);
        assert_float_equal(gpu_k[i], expected_k[i], 5e-4f);
        assert_float_equal(gpu_v[i], expected_v[i], 5e-4f);
    }

    marmot_test_tensor_destroy_all(
        13, metal_act_v, metal_act_k, metal_act_q, metal_sum_v, metal_sum_k, metal_sum_q, metal_v, metal_k, metal_q,
        metal_residual, metal_bias, metal_weight, metal_input
    );
    marmot_destroy(metal_ctx);
#else
    skip();
#endif
}

static void test_matmul_qkv_cpu_f16_scalar(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    run_matmul_qkv_dtype_smoke(env, MARMOT_DTYPE_FLOAT16, 1e-2f, true);
}

static void test_matmul_qkv_cpu_bf16_scalar(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    run_matmul_qkv_dtype_smoke(env, MARMOT_DTYPE_BFLOAT16, 5e-2f, true);
}

// -----------------------------------------------------------------------------
// DGEMM edge case tests: non-multiple-of-4 dimensions exercise cleanup kernels
// -----------------------------------------------------------------------------

static void matmul_reference_f64(const double *a, const double *b, double *out, size_t M, size_t K, size_t N) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (size_t k = 0; k < K; ++k) {
                acc += a[m * K + k] * b[k * N + n];
            }
            out[m * N + n] = acc;
        }
    }
}

static void run_dgemm_edge_case(marmot_test_env_t *env, size_t M, size_t K, size_t N, double tol) {
    if (env->backend == MARMOT_BACKEND_METAL) {
        skip(); // Metal doesn't support f64
    }

    size_t a_shape[2] = {M, K};
    size_t b_shape[2] = {K, N};
    size_t out_shape[2] = {M, N};

    marmot_tensor_t *a = marmot_tensor_create(env->ctx, a_shape, 2, MARMOT_DTYPE_FLOAT64);
    marmot_tensor_t *b = marmot_tensor_create(env->ctx, b_shape, 2, MARMOT_DTYPE_FLOAT64);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 2, MARMOT_DTYPE_FLOAT64);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(out);

    double *host_a = (double *)malloc(M * K * sizeof(double));
    double *host_b = (double *)malloc(K * N * sizeof(double));
    double *expected = (double *)malloc(M * N * sizeof(double));
    double *got = (double *)malloc(M * N * sizeof(double));
    assert_non_null(host_a);
    assert_non_null(host_b);
    assert_non_null(expected);
    assert_non_null(got);

    for (size_t i = 0; i < M * K; ++i) {
        host_a[i] = ((double)(i % 17) - 8.0) * 0.1;
    }
    for (size_t i = 0; i < K * N; ++i) {
        host_b[i] = ((double)(i % 13) - 6.0) * 0.15;
    }

    assert_int_equal(marmot_tensor_copy_from_host_buffer(env->ctx, a, host_a, M * K * sizeof(double)), MARMOT_SUCCESS);
    assert_int_equal(marmot_tensor_copy_from_host_buffer(env->ctx, b, host_b, K * N * sizeof(double)), MARMOT_SUCCESS);

    marmot_error_t err = marmot_matmul(env->ctx, a, b, nullptr, out);
    if (err != MARMOT_SUCCESS) {
        const char *detail = marmot_get_last_error_detail();
        fail_msg("DGEMM %zux%zux%zu failed: %s (%s)", M, K, N, marmot_error_string(err), detail ? detail : "");
    }

    assert_int_equal(marmot_tensor_copy_to_host_buffer(env->ctx, out, got, M * N * sizeof(double)), MARMOT_SUCCESS);

    matmul_reference_f64(host_a, host_b, expected, M, K, N);

    for (size_t i = 0; i < M * N; ++i) {
        double diff = fabs(got[i] - expected[i]);
        if (diff > tol) {
            fail_msg(
                "DGEMM %zux%zux%zu mismatch at [%zu]: expected=%.12f got=%.12f diff=%.12g", M, K, N, i, expected[i],
                got[i], diff
            );
        }
    }

    free(got);
    free(expected);
    free(host_b);
    free(host_a);
    marmot_test_tensor_destroy_all(3, out, b, a);
}

static void test_dgemm_non_multiple_of_4(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const double tol = 1e-10;

    // DGEMM uses 8x4 tiles, so test various non-aligned sizes
    // Note: Sizes must be > 1 to avoid vector case (which changes tensor rank)
    run_dgemm_edge_case(env, 5, 9, 7, tol);    // All non-multiples of 4
    run_dgemm_edge_case(env, 2, 17, 3, tol);   // Small M (was 1, but 1 becomes vector)
    run_dgemm_edge_case(env, 9, 5, 2, tol);    // Small N (was 1, but 1 becomes vector)
    run_dgemm_edge_case(env, 13, 21, 11, tol); // Larger odd sizes
    run_dgemm_edge_case(env, 6, 10, 14, tol);  // Multiples of 2 but not 4
    run_dgemm_edge_case(env, 3, 7, 5, tol);    // Very small odd
}

static void test_dgemm_cleanup_kernel_sizes(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    const double tol = 1e-10;

    // Test sizes that specifically exercise cleanup kernels
    // DGEMM tile is 8x4, so remainders trigger cleanup paths
    run_dgemm_edge_case(env, 12, 16, 4, tol); // Exactly 4 cols (no N cleanup)
    run_dgemm_edge_case(env, 12, 16, 5, tol); // 4+1 cols
    run_dgemm_edge_case(env, 12, 16, 6, tol); // 4+2 cols
    run_dgemm_edge_case(env, 12, 16, 7, tol); // 4+3 cols
    run_dgemm_edge_case(env, 8, 16, 8, tol);  // Exactly 8 rows (no M cleanup)
    run_dgemm_edge_case(env, 9, 16, 8, tol);  // 8+1 rows
    run_dgemm_edge_case(env, 10, 16, 8, tol); // 8+2 rows
    run_dgemm_edge_case(env, 15, 16, 8, tol); // 8+7 rows
}

// -----------------------------------------------------------------------------
// NN layout tests for BF16 and F16
// -----------------------------------------------------------------------------

static void test_matmul_nn_layout_bf16(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);

    // marmot_matmul uses NN layout: A(M×K) @ B(K×N) = C(M×N)
    const size_t M = 16;
    const size_t K = 32;
    const size_t N = 24;

    size_t a_shape[2] = {M, K};
    size_t b_shape[2] = {K, N};
    size_t out_shape[2] = {M, N};

    marmot_tensor_t *a = marmot_tensor_create(env->ctx, a_shape, 2, MARMOT_DTYPE_BFLOAT16);
    marmot_tensor_t *b = marmot_tensor_create(env->ctx, b_shape, 2, MARMOT_DTYPE_BFLOAT16);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 2, MARMOT_DTYPE_BFLOAT16);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(out);

    float *host_a = (float *)malloc(M * K * sizeof(float));
    float *host_b = (float *)malloc(K * N * sizeof(float));
    float *expected = (float *)malloc(M * N * sizeof(float));
    float *got = (float *)malloc(M * N * sizeof(float));
    assert_non_null(host_a);
    assert_non_null(host_b);
    assert_non_null(expected);
    assert_non_null(got);

    for (size_t i = 0; i < M * K; ++i) {
        host_a[i] = ((float)(i % 11) - 5.0f) * 0.2f;
    }
    for (size_t i = 0; i < K * N; ++i) {
        host_b[i] = ((float)(i % 7) - 3.0f) * 0.15f;
    }

    marmot_test_convert_span(env, a, MARMOT_DTYPE_FLOAT32, host_a, M * K);
    marmot_test_convert_span(env, b, MARMOT_DTYPE_FLOAT32, host_b, K * N);

    marmot_error_t err = marmot_matmul(env->ctx, a, b, nullptr, out);
    assert_int_equal(err, MARMOT_SUCCESS);

    marmot_test_fetch_span(env, got, MARMOT_DTYPE_FLOAT32, out, M * N);

    matmul_reference(host_a, host_b, expected, 1, M, K, N, false);

    // BF16 has limited precision, use relaxed tolerance
    const float tol = 0.5f;
    size_t max_diff_idx = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < M * N; ++i) {
        float diff = fabsf(got[i] - expected[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    if (max_diff > tol) {
        fail_msg(
            "BF16 NN matmul max_diff=%.6f at [%zu]: expected=%.6f got=%.6f", max_diff, max_diff_idx,
            expected[max_diff_idx], got[max_diff_idx]
        );
    }

    free(got);
    free(expected);
    free(host_b);
    free(host_a);
    marmot_test_tensor_destroy_all(3, out, b, a);
}

static void test_matmul_nn_layout_f16(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);

    const size_t M = 16;
    const size_t K = 32;
    const size_t N = 24;

    size_t a_shape[2] = {M, K};
    size_t b_shape[2] = {K, N};
    size_t out_shape[2] = {M, N};

    marmot_tensor_t *a = marmot_tensor_create(env->ctx, a_shape, 2, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *b = marmot_tensor_create(env->ctx, b_shape, 2, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 2, MARMOT_DTYPE_FLOAT16);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(out);

    float *host_a = (float *)malloc(M * K * sizeof(float));
    float *host_b = (float *)malloc(K * N * sizeof(float));
    float *expected = (float *)malloc(M * N * sizeof(float));
    float *got = (float *)malloc(M * N * sizeof(float));
    assert_non_null(host_a);
    assert_non_null(host_b);
    assert_non_null(expected);
    assert_non_null(got);

    for (size_t i = 0; i < M * K; ++i) {
        host_a[i] = ((float)(i % 11) - 5.0f) * 0.2f;
    }
    for (size_t i = 0; i < K * N; ++i) {
        host_b[i] = ((float)(i % 7) - 3.0f) * 0.15f;
    }

    marmot_test_convert_span(env, a, MARMOT_DTYPE_FLOAT32, host_a, M * K);
    marmot_test_convert_span(env, b, MARMOT_DTYPE_FLOAT32, host_b, K * N);

    marmot_error_t err = marmot_matmul(env->ctx, a, b, nullptr, out);
    assert_int_equal(err, MARMOT_SUCCESS);

    marmot_test_fetch_span(env, got, MARMOT_DTYPE_FLOAT32, out, M * N);

    matmul_reference(host_a, host_b, expected, 1, M, K, N, false);

    const float tol = 0.1f;
    size_t max_diff_idx = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < M * N; ++i) {
        float diff = fabsf(got[i] - expected[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    if (max_diff > tol) {
        fail_msg(
            "F16 NN matmul max_diff=%.6f at [%zu]: expected=%.6f got=%.6f", max_diff, max_diff_idx,
            expected[max_diff_idx], got[max_diff_idx]
        );
    }

    free(got);
    free(expected);
    free(host_b);
    free(host_a);
    marmot_test_tensor_destroy_all(3, out, b, a);
}

static void test_matmul_nn_layout_bf16_edge_sizes(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);

    // Test non-multiple-of-8 sizes for BF16 NN layout
    const size_t sizes[][3] = {
        {7, 11, 5},   // All odd
        {9, 17, 13},  // Larger odd
        {15, 23, 19}, // Odd, exercises cleanup kernels
    };

    for (size_t t = 0; t < sizeof(sizes) / sizeof(sizes[0]); ++t) {
        const size_t M = sizes[t][0];
        const size_t K = sizes[t][1];
        const size_t N = sizes[t][2];

        size_t a_shape[2] = {M, K};
        size_t b_shape[2] = {K, N};
        size_t out_shape[2] = {M, N};

        marmot_tensor_t *a = marmot_tensor_create(env->ctx, a_shape, 2, MARMOT_DTYPE_BFLOAT16);
        marmot_tensor_t *b = marmot_tensor_create(env->ctx, b_shape, 2, MARMOT_DTYPE_BFLOAT16);
        marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 2, MARMOT_DTYPE_BFLOAT16);
        assert_non_null(a);
        assert_non_null(b);
        assert_non_null(out);

        float *host_a = (float *)malloc(M * K * sizeof(float));
        float *host_b = (float *)malloc(K * N * sizeof(float));
        float *expected = (float *)malloc(M * N * sizeof(float));
        float *got = (float *)malloc(M * N * sizeof(float));

        for (size_t i = 0; i < M * K; ++i) {
            host_a[i] = ((float)(i % 9) - 4.0f) * 0.1f;
        }
        for (size_t i = 0; i < K * N; ++i) {
            host_b[i] = ((float)(i % 5) - 2.0f) * 0.1f;
        }

        marmot_test_convert_span(env, a, MARMOT_DTYPE_FLOAT32, host_a, M * K);
        marmot_test_convert_span(env, b, MARMOT_DTYPE_FLOAT32, host_b, K * N);

        marmot_error_t err = marmot_matmul(env->ctx, a, b, nullptr, out);
        assert_int_equal(err, MARMOT_SUCCESS);

        marmot_test_fetch_span(env, got, MARMOT_DTYPE_FLOAT32, out, M * N);
        matmul_reference(host_a, host_b, expected, 1, M, K, N, false);

        const float tol = 0.3f;
        for (size_t i = 0; i < M * N; ++i) {
            float diff = fabsf(got[i] - expected[i]);
            if (diff > tol) {
                fail_msg(
                    "BF16 NN edge %zux%zux%zu mismatch at [%zu]: expected=%.6f got=%.6f", M, K, N, i, expected[i],
                    got[i]
                );
            }
        }

        free(got);
        free(expected);
        free(host_b);
        free(host_a);
        marmot_test_tensor_destroy_all(3, out, b, a);
    }
}

// -----------------------------------------------------------------------------
// Small matrix fast-path tests (sizes ≤32×32×64)
// -----------------------------------------------------------------------------

static void test_sgemm_small_matrix_fast_path(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);

    // These sizes should hit the sgemm_small_direct path in NEON kernel
    const size_t sizes[][3] = {
        {4, 4, 4},    // Tiny
        {8, 8, 8},    // Single tile
        {16, 16, 16}, // 2x2 tiles
        {32, 32, 32}, // Boundary case
        {32, 32, 64}, // Max small path K
        {24, 28, 48}, // Non-power-of-2 within small range
    };

    for (size_t t = 0; t < sizeof(sizes) / sizeof(sizes[0]); ++t) {
        const size_t M = sizes[t][0];
        const size_t K = sizes[t][1];
        const size_t N = sizes[t][2];

        size_t a_shape[2] = {M, K};
        size_t b_shape[2] = {K, N};
        size_t out_shape[2] = {M, N};

        marmot_tensor_t *a = marmot_tensor_create(env->ctx, a_shape, 2, MARMOT_DTYPE_FLOAT32);
        marmot_tensor_t *b = marmot_tensor_create(env->ctx, b_shape, 2, MARMOT_DTYPE_FLOAT32);
        marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 2, MARMOT_DTYPE_FLOAT32);
        assert_non_null(a);
        assert_non_null(b);
        assert_non_null(out);

        float *host_a = (float *)malloc(M * K * sizeof(float));
        float *host_b = (float *)malloc(K * N * sizeof(float));
        float *expected = (float *)malloc(M * N * sizeof(float));
        float *got = (float *)malloc(M * N * sizeof(float));

        for (size_t i = 0; i < M * K; ++i) {
            host_a[i] = ((float)(i % 23) - 11.0f) * 0.05f;
        }
        for (size_t i = 0; i < K * N; ++i) {
            host_b[i] = ((float)(i % 19) - 9.0f) * 0.07f;
        }

        marmot_test_convert_f32_span(env, a, host_a, M * K);
        marmot_test_convert_f32_span(env, b, host_b, K * N);

        marmot_error_t err = marmot_matmul(env->ctx, a, b, nullptr, out);
        assert_int_equal(err, MARMOT_SUCCESS);

        marmot_test_fetch_f32_span(env, got, out, M * N);
        matmul_reference(host_a, host_b, expected, 1, M, K, N, false);

        const float tol = 1e-4f;
        for (size_t i = 0; i < M * N; ++i) {
            float diff = fabsf(got[i] - expected[i]);
            if (diff > tol) {
                fail_msg(
                    "Small matmul %zux%zux%zu mismatch at [%zu]: expected=%.6f got=%.6f diff=%.6g", M, K, N, i,
                    expected[i], got[i], diff
                );
            }
        }

        free(got);
        free(expected);
        free(host_b);
        free(host_a);
        marmot_test_tensor_destroy_all(3, out, b, a);
    }
}

// -----------------------------------------------------------------------------
// SGEMM cleanup kernel tests (sizes that exercise 4×8, 8×4, etc. remainders)
// -----------------------------------------------------------------------------

static void test_sgemm_cleanup_kernel_sizes(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);

    // SGEMM uses 8×8 tiles. Test sizes that leave various remainders.
    const size_t sizes[][3] = {
        // M remainder tests (rows)
        {9, 32, 16},  // 8+1 rows
        {10, 32, 16}, // 8+2 rows
        {12, 32, 16}, // 8+4 rows
        {15, 32, 16}, // 8+7 rows
        {17, 32, 16}, // 2×8+1 rows
        // N remainder tests (columns)
        {16, 32, 9},  // 8+1 cols
        {16, 32, 12}, // 8+4 cols
        {16, 32, 15}, // 8+7 cols
        {16, 32, 20}, // 2×8+4 cols
        // Both M and N remainders
        {11, 32, 13}, // 8+3 rows, 8+5 cols
        {14, 32, 10}, // 8+6 rows, 8+2 cols
        {19, 32, 21}, // 2×8+3 rows, 2×8+5 cols
        // K remainder (affects inner loop)
        {16, 33, 16}, // Odd K
        {16, 35, 16}, // K not multiple of 4
    };

    for (size_t t = 0; t < sizeof(sizes) / sizeof(sizes[0]); ++t) {
        const size_t M = sizes[t][0];
        const size_t K = sizes[t][1];
        const size_t N = sizes[t][2];

        size_t a_shape[2] = {M, K};
        size_t b_shape[2] = {K, N};
        size_t out_shape[2] = {M, N};

        marmot_tensor_t *a = marmot_tensor_create(env->ctx, a_shape, 2, MARMOT_DTYPE_FLOAT32);
        marmot_tensor_t *b = marmot_tensor_create(env->ctx, b_shape, 2, MARMOT_DTYPE_FLOAT32);
        marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 2, MARMOT_DTYPE_FLOAT32);
        assert_non_null(a);
        assert_non_null(b);
        assert_non_null(out);

        float *host_a = (float *)malloc(M * K * sizeof(float));
        float *host_b = (float *)malloc(K * N * sizeof(float));
        float *expected = (float *)malloc(M * N * sizeof(float));
        float *got = (float *)malloc(M * N * sizeof(float));

        for (size_t i = 0; i < M * K; ++i) {
            host_a[i] = ((float)(i % 31) - 15.0f) * 0.03f;
        }
        for (size_t i = 0; i < K * N; ++i) {
            host_b[i] = ((float)(i % 29) - 14.0f) * 0.04f;
        }

        marmot_test_convert_f32_span(env, a, host_a, M * K);
        marmot_test_convert_f32_span(env, b, host_b, K * N);

        marmot_error_t err = marmot_matmul(env->ctx, a, b, nullptr, out);
        assert_int_equal(err, MARMOT_SUCCESS);

        marmot_test_fetch_f32_span(env, got, out, M * N);
        matmul_reference(host_a, host_b, expected, 1, M, K, N, false);

        const float tol = 1e-4f;
        for (size_t i = 0; i < M * N; ++i) {
            float diff = fabsf(got[i] - expected[i]);
            if (diff > tol) {
                fail_msg(
                    "SGEMM cleanup %zux%zux%zu mismatch at [%zu]: expected=%.6f got=%.6f diff=%.6g", M, K, N, i,
                    expected[i], got[i], diff
                );
            }
        }

        free(got);
        free(expected);
        free(host_b);
        free(host_a);
        marmot_test_tensor_destroy_all(3, out, b, a);
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_linear_default, marmot_test_backend_setup, marmot_test_backend_teardown),
#if MARMOT_ENABLE_FP8
        cmocka_unit_test_setup_teardown(
            test_matmul_fp8_e4m3_golden, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_fp8_e5m2_golden, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
#endif
        cmocka_unit_test_setup_teardown(
            test_matmul_fused_bias_identity, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_fused_bias_identity_f64, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_fused_bias_gelu, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_fused_bias_relu_f16, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_fused_goldens, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_rope_epilogue_matches_reference, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_quantized_rope_epilogue, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_rope_rotation_pattern_matches_reference, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_qkv_quantized_separate_backend, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_rope_epilogue_matches_reference_backend, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_qkv_matches_reference, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_qkv_post_residual_activation, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_qkv_separate_weights, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_qkv_separate_weights_f16, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_qkv_separate_rope_f16, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_qkv_fused_rope_f16, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_qkv_separate_rope_q4km, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_qkv_separate_rope_q4_0, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_qkv_separate_rope_q8_0, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_qkv_goldens, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_linear_followed_by_sum, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_pytorch_basic, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_shape_mismatch, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_batched_basic, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_batched_broadcast_rhs, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_vector_matrix, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_matrix_vector, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_vector_dot, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test(test_matmul_qkv_packed_weight_correctness),
        cmocka_unit_test(test_matmul_qkv_threshold_toggle),
        cmocka_unit_test(test_matmul_qkv_separate_rope_metal),
        cmocka_unit_test(test_matmul_qkv_post_residual_gelu_metal),
        cmocka_unit_test(test_matmul_qkv_post_residual_leaky_relu_metal),
        cmocka_unit_test_setup_teardown(
            test_matmul_qkv_cpu_f16_scalar, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_qkv_cpu_bf16_scalar, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        // DGEMM edge case tests
        cmocka_unit_test_setup_teardown(
            test_dgemm_non_multiple_of_4, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_dgemm_cleanup_kernel_sizes, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        // NN layout tests for BF16/F16
        cmocka_unit_test_setup_teardown(
            test_matmul_nn_layout_bf16, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_nn_layout_f16, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_matmul_nn_layout_bf16_edge_sizes, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        // Small matrix and cleanup kernel tests
        cmocka_unit_test_setup_teardown(
            test_sgemm_small_matrix_fast_path, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_sgemm_cleanup_kernel_sizes, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
