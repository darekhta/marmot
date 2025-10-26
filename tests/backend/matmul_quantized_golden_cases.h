#ifndef MARMOT_TEST_MATMUL_QUANTIZED_GOLDEN_CASES_H
#define MARMOT_TEST_MATMUL_QUANTIZED_GOLDEN_CASES_H

#include "backend/golden_matmul_llama.h"
#include "backend/golden_quant_llama.h"

typedef struct {
    const char *name;
    marmot_quant_kind_t kind;
    size_t N;
    size_t K;
    size_t M;
    const uint8_t *weights;
    size_t weight_bytes;
    const float *input_f32;
    const uint16_t *input_f16;
    const float *expected_f32;
    const float *expected_f16;
} matmul_golden_case_t;

#define MATMUL_GOLDEN_CASE(UPPER, lower)                                                                               \
    {                                                                                                                  \
        .name = #UPPER,                                                                                                \
        .kind = MATMUL_##UPPER##_QUANT_KIND,                                                                           \
        .N = MATMUL_##UPPER##_N,                                                                                       \
        .K = MATMUL_##UPPER##_K,                                                                                       \
        .M = MATMUL_##UPPER##_M,                                                                                       \
        .weights = g_matmul_##lower##_weight,                                                                          \
        .weight_bytes = sizeof(g_matmul_##lower##_weight),                                                             \
        .input_f32 = g_matmul_##lower##_input_f32,                                                                     \
        .input_f16 = g_matmul_##lower##_input_f16,                                                                     \
        .expected_f32 = g_matmul_##lower##_output_from_f32,                                                            \
        .expected_f16 = g_matmul_##lower##_output_from_f16,                                                            \
    }

static const matmul_golden_case_t g_matmul_quant_goldens[] = {
    MATMUL_GOLDEN_CASE(Q4_0, q4_0), MATMUL_GOLDEN_CASE(Q4_1, q4_1), MATMUL_GOLDEN_CASE(Q5_0, q5_0),
    MATMUL_GOLDEN_CASE(Q5_1, q5_1), MATMUL_GOLDEN_CASE(Q8_0, q8_0), MATMUL_GOLDEN_CASE(Q8_1, q8_1),
    MATMUL_GOLDEN_CASE(Q2_K, q2_k), MATMUL_GOLDEN_CASE(Q3_K, q3_k), MATMUL_GOLDEN_CASE(Q4_K, q4_k),
    MATMUL_GOLDEN_CASE(Q5_K, q5_k), MATMUL_GOLDEN_CASE(Q6_K, q6_k), MATMUL_GOLDEN_CASE(Q8_K, q8_k),
};

#undef MATMUL_GOLDEN_CASE

static inline const matmul_golden_case_t *matmul_quantized_default_case(void) {
    return &g_matmul_quant_goldens[0];
}

#endif // MARMOT_TEST_MATMUL_QUANTIZED_GOLDEN_CASES_H
