#include <string.h>

#include "cpu_caps.h"
#include "ops/matmul/quantized/internal/vec_dot.h"
#include "ops/matmul/quantized/matmul_quant_kernels.h"

static cpu_matmul_quant_kernel_t cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_COUNT];
static cpu_matmul_quant_kernel_t cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_COUNT];
static cpu_matmul_quant_kernel_t cpu_matmul_quant_kernels_neon_i8mm_table[MARMOT_QUANT_KIND_COUNT];
static bool cpu_matmul_quant_kernels_neon_initialized = false;
static bool cpu_matmul_quant_kernels_neon_dotprod_initialized = false;
static bool cpu_matmul_quant_kernels_neon_i8mm_initialized = false;

static void cpu_matmul_quant_kernels_neon_init(void) {
    if (cpu_matmul_quant_kernels_neon_initialized) {
        return;
    }
    const cpu_matmul_quant_kernel_t *base = cpu_matmul_quant_kernels_scalar();
    memcpy(cpu_matmul_quant_kernels_neon_table, base, sizeof(cpu_matmul_quant_kernels_neon_table));

    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q4_0].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q4_0_q8_0_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q4_0].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q4_0_f16_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q4_0].impl_name = "neon:q4_0";

    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q4_1].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q4_1_q8_0_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q4_1].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q4_1_f16_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q4_1].impl_name = "neon:q4_1";

    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q5_0].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q5_0_q8_0_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q5_0].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q5_0_f16_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q5_0].impl_name = "neon:q5_0";

    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q5_1].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q5_1_q8_0_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q5_1].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q5_1_f16_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q5_1].impl_name = "neon:q5_1";

    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q8_0].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q8_0_q8_0_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q8_0].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q8_0_f16_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q8_0].impl_name = "neon:q8_0";

    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q8_1].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q8_1_q8_0_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q8_1].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q8_1_f16_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q8_1].impl_name = "neon:q8_1";

    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q2_K].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q2_k_f16_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q2_K].ops.dot_q8_k =
        (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q2_k_q8_k_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q2_K].impl_name = "neon:q2_k";

    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q3_K].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q3_k_f16_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q3_K].ops.dot_q8_k =
        (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q3_k_q8_k_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q3_K].impl_name = "neon:q3_k";

    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q4_K].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q4_k_f16_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q4_K].ops.dot_q8_k =
        (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q4_k_q8_k_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q4_K].impl_name = "neon:q4_k";

    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q5_K].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q5_k_f16_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q5_K].ops.dot_q8_k =
        (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q5_k_q8_k_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q5_K].impl_name = "neon:q5_k";

    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q6_K].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q6_k_f16_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q6_K].ops.dot_q8_k =
        (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q6_k_q8_k_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q6_K].impl_name = "neon:q6_k";

    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q8_K].ops.dot_fp16 =
        (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q8_k_f16_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q8_K].ops.dot_q8_k =
        (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q8_k_q8_k_neon;
    cpu_matmul_quant_kernels_neon_table[MARMOT_QUANT_KIND_Q8_K].impl_name = "neon:q8_k";

    cpu_matmul_quant_kernels_neon_initialized = true;
}

static void cpu_matmul_quant_kernels_neon_dotprod_init(void) {
    if (cpu_matmul_quant_kernels_neon_dotprod_initialized) {
        return;
    }

    cpu_matmul_quant_kernels_neon_init();
    memcpy(
        cpu_matmul_quant_kernels_neon_dotprod_table, cpu_matmul_quant_kernels_neon_table,
        sizeof(cpu_matmul_quant_kernels_neon_dotprod_table)
    );

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q8_0].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q8_0_q8_0_neon_dotprod;
    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q8_0].impl_name = "neon:q8_0:dotprod";

    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q8_1].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q8_1_q8_0_neon_dotprod;
    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q8_1].impl_name = "neon:q8_1:dotprod";

    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q4_0].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q4_0_q8_0_neon_dotprod;
    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q4_0].impl_name = "neon:q4_0:dotprod";

    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q4_1].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q4_1_q8_0_neon_dotprod;
    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q4_1].impl_name = "neon:q4_1:dotprod";

    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q5_0].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q5_0_q8_0_neon_dotprod;
    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q5_0].impl_name = "neon:q5_0:dotprod";

    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q5_1].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q5_1_q8_0_neon_dotprod;
    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q5_1].impl_name = "neon:q5_1:dotprod";

    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q4_K].ops.dot_q8_k =
        (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q4_k_q8_k_neon_dotprod;
    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q4_K].impl_name = "neon:q4_k:dotprod";

    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q5_K].ops.dot_q8_k =
        (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q5_k_q8_k_neon_dotprod;
    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q5_K].impl_name = "neon:q5_k:dotprod";

    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q3_K].ops.dot_q8_k =
        (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q3_k_q8_k_neon_dotprod;
    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q3_K].impl_name = "neon:q3_k:dotprod";

    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q2_K].ops.dot_q8_k =
        (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q2_k_q8_k_neon_dotprod;
    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q2_K].impl_name = "neon:q2_k:dotprod";

    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q6_K].ops.dot_q8_k =
        (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q6_k_q8_k_neon_dotprod;
    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q6_K].impl_name = "neon:q6_k:dotprod";

    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q8_K].ops.dot_q8_k =
        (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q8_k_q8_k_neon_dotprod;
    cpu_matmul_quant_kernels_neon_dotprod_table[MARMOT_QUANT_KIND_Q8_K].impl_name = "neon:q8_k:dotprod";
#endif

    cpu_matmul_quant_kernels_neon_dotprod_initialized = true;
}

static void cpu_matmul_quant_kernels_neon_i8mm_init(void) {
    if (cpu_matmul_quant_kernels_neon_i8mm_initialized) {
        return;
    }

    cpu_matmul_quant_kernels_neon_dotprod_init();
    memcpy(
        cpu_matmul_quant_kernels_neon_i8mm_table, cpu_matmul_quant_kernels_neon_dotprod_table,
        sizeof(cpu_matmul_quant_kernels_neon_i8mm_table)
    );

#if defined(__aarch64__) && (defined(__ARM_FEATURE_I8MM) || defined(__ARM_FEATURE_MATMUL_INT8))
    cpu_matmul_quant_kernels_neon_i8mm_table[MARMOT_QUANT_KIND_Q8_0].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q8_0_q8_0_neon_i8mm;
    cpu_matmul_quant_kernels_neon_i8mm_table[MARMOT_QUANT_KIND_Q8_0].impl_name = "neon:q8_0:i8mm";

    cpu_matmul_quant_kernels_neon_i8mm_table[MARMOT_QUANT_KIND_Q8_1].ops.dot_q8_0 =
        (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q8_1_q8_0_neon_i8mm;
    cpu_matmul_quant_kernels_neon_i8mm_table[MARMOT_QUANT_KIND_Q8_1].impl_name = "neon:q8_1:i8mm";

    cpu_matmul_quant_kernels_neon_i8mm_table[MARMOT_QUANT_KIND_Q8_K].ops.dot_q8_k =
        (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q8_k_q8_k_neon_i8mm;
    cpu_matmul_quant_kernels_neon_i8mm_table[MARMOT_QUANT_KIND_Q8_K].impl_name = "neon:q8_k:i8mm";
#endif

    cpu_matmul_quant_kernels_neon_i8mm_initialized = true;
}

static bool cpu_matmul_quant_neon_has_dotprod(void) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    static bool checked = false;
    static bool supported = false;
    if (!checked) {
        supported = marmot_cpu_has_arm_dotprod();
        checked = true;
    }
    return supported;
#else
    return false;
#endif
}

static bool cpu_matmul_quant_neon_has_i8mm(void) {
#if defined(__aarch64__) && (defined(__ARM_FEATURE_I8MM) || defined(__ARM_FEATURE_MATMUL_INT8))
    static bool checked = false;
    static bool supported = false;
    if (!checked) {
        supported = marmot_cpu_has_arm_i8mm();
        checked = true;
    }
    return supported;
#else
    return false;
#endif
}

const cpu_matmul_quant_kernel_t *cpu_matmul_quant_kernels_neon(void) {
    if (cpu_matmul_quant_neon_has_i8mm()) {
        cpu_matmul_quant_kernels_neon_i8mm_init();
        return cpu_matmul_quant_kernels_neon_i8mm_table;
    }
    if (cpu_matmul_quant_neon_has_dotprod()) {
        cpu_matmul_quant_kernels_neon_dotprod_init();
        return cpu_matmul_quant_kernels_neon_dotprod_table;
    }

    cpu_matmul_quant_kernels_neon_init();
    return cpu_matmul_quant_kernels_neon_table;
}
