#include "marmot/ops/quantization.h"
#include "marmot/quant_block.h"
#include "marmot/tensor.h"

#include <math.h>
#include <string.h>

#include "backend/golden_quant_llama.h"
#include "backend/golden_vec_dot_llama.h"
#include "backend/test_backend_utils.h"
#include "quant_decode_helpers.h"

typedef struct {
    const char *name;
    marmot_quant_kind_t weight_kind;
    const void *weights;
    size_t blocks;
    const void *activations;
    marmot_quant_kind_t activation_kind;
    float expected;
} vec_dot_golden_case_t;

static const vec_dot_golden_case_t g_vec_dot_goldens[] = {
    {
        .name = "Q4_0×Q8_0",
        .weight_kind = MARMOT_QUANT_KIND_Q4_0,
        .weights = g_vec_dot_q4_0_weights,
        .blocks = sizeof(g_vec_dot_q4_0_weights) / sizeof(g_vec_dot_q4_0_weights[0]),
        .activations = g_vec_dot_q8_0_activations,
        .activation_kind = MARMOT_QUANT_KIND_Q8_0,
        .expected = g_vec_dot_q4_0_q8_0_expected,
    },
    {
        .name = "Q4_1×Q8_0",
        .weight_kind = MARMOT_QUANT_KIND_Q4_1,
        .weights = g_vec_dot_q4_1_weights,
        .blocks = sizeof(g_vec_dot_q4_1_weights) / sizeof(g_vec_dot_q4_1_weights[0]),
        .activations = g_vec_dot_q8_0_activations,
        .activation_kind = MARMOT_QUANT_KIND_Q8_0,
        .expected = g_vec_dot_q4_1_q8_0_expected,
    },
    {
        .name = "Q5_0×Q8_0",
        .weight_kind = MARMOT_QUANT_KIND_Q5_0,
        .weights = g_vec_dot_q5_0_weights,
        .blocks = sizeof(g_vec_dot_q5_0_weights) / sizeof(g_vec_dot_q5_0_weights[0]),
        .activations = g_vec_dot_q8_0_activations,
        .activation_kind = MARMOT_QUANT_KIND_Q8_0,
        .expected = g_vec_dot_q5_0_q8_0_expected,
    },
    {
        .name = "Q5_1×Q8_0",
        .weight_kind = MARMOT_QUANT_KIND_Q5_1,
        .weights = g_vec_dot_q5_1_weights,
        .blocks = sizeof(g_vec_dot_q5_1_weights) / sizeof(g_vec_dot_q5_1_weights[0]),
        .activations = g_vec_dot_q8_0_activations,
        .activation_kind = MARMOT_QUANT_KIND_Q8_0,
        .expected = g_vec_dot_q5_1_q8_0_expected,
    },
    {
        .name = "Q8_0×Q8_0",
        .weight_kind = MARMOT_QUANT_KIND_Q8_0,
        .weights = g_vec_dot_q8_0_weights,
        .blocks = sizeof(g_vec_dot_q8_0_weights) / sizeof(g_vec_dot_q8_0_weights[0]),
        .activations = g_vec_dot_q8_0_activations,
        .activation_kind = MARMOT_QUANT_KIND_Q8_0,
        .expected = g_vec_dot_q8_0_q8_0_expected,
    },
    {
        .name = "Q8_1×Q8_0",
        .weight_kind = MARMOT_QUANT_KIND_Q8_1,
        .weights = g_vec_dot_q8_1_weights,
        .blocks = sizeof(g_vec_dot_q8_1_weights) / sizeof(g_vec_dot_q8_1_weights[0]),
        .activations = g_vec_dot_q8_0_activations,
        .activation_kind = MARMOT_QUANT_KIND_Q8_0,
        .expected = g_vec_dot_q8_1_q8_0_expected,
    },
    {
        .name = "Q2_K×Q8_K",
        .weight_kind = MARMOT_QUANT_KIND_Q2_K,
        .weights = g_vec_dot_q2_k_weights,
        .blocks = sizeof(g_vec_dot_q2_k_weights) / sizeof(g_vec_dot_q2_k_weights[0]),
        .activations = g_vec_dot_q8_k_activations,
        .activation_kind = MARMOT_QUANT_KIND_Q8_K,
        .expected = g_vec_dot_q2_k_q8_k_expected,
    },
    {
        .name = "Q3_K×Q8_K",
        .weight_kind = MARMOT_QUANT_KIND_Q3_K,
        .weights = g_vec_dot_q3_k_weights,
        .blocks = sizeof(g_vec_dot_q3_k_weights) / sizeof(g_vec_dot_q3_k_weights[0]),
        .activations = g_vec_dot_q8_k_activations,
        .activation_kind = MARMOT_QUANT_KIND_Q8_K,
        .expected = g_vec_dot_q3_k_q8_k_expected,
    },
    {
        .name = "Q4_K×Q8_K",
        .weight_kind = MARMOT_QUANT_KIND_Q4_K,
        .weights = g_vec_dot_q4_k_weights,
        .blocks = sizeof(g_vec_dot_q4_k_weights) / sizeof(g_vec_dot_q4_k_weights[0]),
        .activations = g_vec_dot_q8_k_activations,
        .activation_kind = MARMOT_QUANT_KIND_Q8_K,
        .expected = g_vec_dot_q4_k_q8_k_expected,
    },
    {
        .name = "Q5_K×Q8_K",
        .weight_kind = MARMOT_QUANT_KIND_Q5_K,
        .weights = g_vec_dot_q5_k_weights,
        .blocks = sizeof(g_vec_dot_q5_k_weights) / sizeof(g_vec_dot_q5_k_weights[0]),
        .activations = g_vec_dot_q8_k_activations,
        .activation_kind = MARMOT_QUANT_KIND_Q8_K,
        .expected = g_vec_dot_q5_k_q8_k_expected,
    },
    {
        .name = "Q6_K×Q8_K",
        .weight_kind = MARMOT_QUANT_KIND_Q6_K,
        .weights = g_vec_dot_q6_k_weights,
        .blocks = sizeof(g_vec_dot_q6_k_weights) / sizeof(g_vec_dot_q6_k_weights[0]),
        .activations = g_vec_dot_q8_k_activations,
        .activation_kind = MARMOT_QUANT_KIND_Q8_K,
        .expected = g_vec_dot_q6_k_q8_k_expected,
    },
    {
        .name = "Q8_K×Q8_K",
        .weight_kind = MARMOT_QUANT_KIND_Q8_K,
        .weights = g_vec_dot_q8_k_weights,
        .blocks = sizeof(g_vec_dot_q8_k_weights) / sizeof(g_vec_dot_q8_k_weights[0]),
        .activations = g_vec_dot_q8_k_activations,
        .activation_kind = MARMOT_QUANT_KIND_Q8_K,
        .expected = g_vec_dot_q8_k_q8_k_expected,
    },
};

static void test_vec_dot_llama_goldens(marmot_test_env_t *env) {
    const float tol_abs = 1e-5f;
    const float tol_rel = 2e-5f;
    for (size_t i = 0; i < sizeof(g_vec_dot_goldens) / sizeof(g_vec_dot_goldens[0]); ++i) {
        const vec_dot_golden_case_t *tc = &g_vec_dot_goldens[i];
        marmot_vec_dot_descriptor_t desc = {
            .weights = tc->weights,
            .activations = tc->activations,
            .num_blocks = tc->blocks,
            .weight_kind = tc->weight_kind,
            .activation_kind = tc->activation_kind,
            .layout = MARMOT_QUANT_LAYOUT_GGUF,
        };

        float result = 0.0f;
        marmot_error_t err = marmot_vec_dot(env->ctx, &desc, &result);
        if (err != MARMOT_SUCCESS) {
            printf("Vec dot \"%s\" returned %s\n", tc->name, marmot_error_string(err));
        }
        assert_int_equal(err, MARMOT_SUCCESS);

        const float diff = fabsf(result - tc->expected);
        const float allowed = fabsf(tc->expected) * tol_rel + tol_abs;
        if (diff > allowed) {
            printf(
                "Golden vec_dot mismatch for %s: expected %.8f, got %.8f (diff %.6g, tol %.6g)\n", tc->name,
                tc->expected, result, diff, allowed
            );
        }
        assert_true(diff <= allowed);
    }
}

static void test_vec_dot_q4_0_q8_0_trivial_zero(marmot_test_env_t *env) {
    marmot_q4_0_block_t weights = {
        .scale = marmot_make_f16(0x0000),
        .qs = {0},
    };
    marmot_q8_0_block_t activations = {
        .scale = marmot_make_f16(0x0000),
        .qs = {0},
    };

    marmot_vec_dot_descriptor_t desc = {
        .weights = &weights,
        .activations = &activations,
        .num_blocks = 1,
        .weight_kind = MARMOT_QUANT_KIND_Q4_0,
        .activation_kind = MARMOT_QUANT_KIND_Q8_0,
        .layout = MARMOT_QUANT_LAYOUT_GGUF,
    };

    float result = -1.0f;
    marmot_error_t err = marmot_vec_dot(env->ctx, &desc, &result);
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_true(fabsf(result) <= 1e-7f);

    float decoded_weights[MARMOT_QUANT_BLOCK_SIZE] = {0};
    float decoded_activations[MARMOT_QUANT_BLOCK_SIZE] = {0};
    marmot_test_unpack_q4_0_block(&weights, decoded_weights);
    marmot_test_unpack_q8_0_block(&activations, decoded_activations);
    for (size_t i = 0; i < MARMOT_QUANT_BLOCK_SIZE; ++i) {
        assert_true(fabsf(decoded_weights[i]) <= 1e-7f);
        assert_true(fabsf(decoded_activations[i]) <= 1e-7f);
    }
}

static void run_vec_dot_suite(marmot_test_env_t *env) {
    test_vec_dot_llama_goldens(env);
    test_vec_dot_q4_0_q8_0_trivial_zero(env);
}

static void test_vec_dot_default(void **state) {
    run_vec_dot_suite((marmot_test_env_t *)(*state));
}

#if MARMOT_TEST_HAS_CPU_SCALAR_SUITE
static void test_vec_dot_scalar(void **state) {
    run_vec_dot_suite((marmot_test_env_t *)(*state));
}
#endif

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_vec_dot_default, marmot_test_backend_setup, marmot_test_backend_teardown),
#if MARMOT_TEST_HAS_CPU_SCALAR_SUITE
        cmocka_unit_test_setup_teardown(
            test_vec_dot_scalar, marmot_test_backend_setup_scalar, marmot_test_backend_teardown
        ),
#endif
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
