#include <string.h>

#include "ops/matmul/quantized/internal/vec_dot.h"
#include "ops/matmul/quantized/matmul_quant_kernels.h"

static cpu_matmul_quant_kernel_t cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_COUNT];
static bool cpu_matmul_quant_kernels_avx2_initialized = false;

static void cpu_matmul_quant_kernels_avx2_init(void) {
    if (cpu_matmul_quant_kernels_avx2_initialized) {
        return;
    }
    const cpu_matmul_quant_kernel_t *base = cpu_matmul_quant_kernels_scalar();
    memcpy(cpu_matmul_quant_kernels_avx2_table, base, sizeof(cpu_matmul_quant_kernels_avx2_table));

    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q4_0].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q4_0_q8_0_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q4_0].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q4_0_f16_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q4_0].impl_name = "avx2:q4_0";

    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q4_1].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q4_1_q8_0_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q4_1].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q4_1_f16_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q4_1].impl_name = "avx2:q4_1";

    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q5_0].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q5_0_q8_0_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q5_0].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q5_0_f16_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q5_0].impl_name = "avx2:q5_0";

    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q5_1].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q5_1_q8_0_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q5_1].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q5_1_f16_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q5_1].impl_name = "avx2:q5_1";

    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q8_0].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q8_0_q8_0_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q8_0].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q8_0_f16_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q8_0].impl_name = "avx2:q8_0";

    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q8_1].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q8_1_q8_0_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q8_1].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q8_1_f16_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q8_1].impl_name = "avx2:q8_1";

    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q2_K].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q2_k_f16_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q2_K].ops.dot_q8_k =
        (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q2_k_q8_k_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q2_K].impl_name = "avx2:q2_k";

    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q3_K].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q3_k_f16_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q3_K].ops.dot_q8_k =
        (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q3_k_q8_k_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q3_K].impl_name = "avx2:q3_k";

    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q4_K].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q4_k_f16_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q4_K].ops.dot_q8_k =
        (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q4_k_q8_k_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q4_K].impl_name = "avx2:q4_k";

    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q5_K].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q5_k_f16_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q5_K].ops.dot_q8_k =
        (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q5_k_q8_k_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q5_K].impl_name = "avx2:q5_k";

    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q6_K].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q6_k_f16_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q6_K].ops.dot_q8_k =
        (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q6_k_q8_k_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q6_K].impl_name = "avx2:q6_k";

    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q8_K].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q8_k_f16_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q8_K].ops.dot_q8_k =
        (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q8_k_q8_k_avx2;
    cpu_matmul_quant_kernels_avx2_table[MARMOT_QUANT_KIND_Q8_K].impl_name = "avx2:q8_k";

    cpu_matmul_quant_kernels_avx2_initialized = true;
}

const cpu_matmul_quant_kernel_t *cpu_matmul_quant_kernels_avx2(void) {
    cpu_matmul_quant_kernels_avx2_init();
    return cpu_matmul_quant_kernels_avx2_table;
}
