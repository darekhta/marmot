#include "ops/matmul/quantized/internal/vec_dot.h"
#include "ops/matmul/quantized/matmul_activation_packers_cpu.h"
#include "ops/matmul/quantized/matmul_quant_kernels.h"
#include "quantization/format_metadata.h"

static const cpu_matmul_quant_kernel_t cpu_matmul_quant_kernels_scalar_table[MARMOT_QUANT_KIND_COUNT] = {
    [MARMOT_QUANT_KIND_Q4_0] =
        {
            .format = CPU_QUANT_FORMAT_INFO(MARMOT_QUANT_KIND_Q4_0),
            .ops.dot_q8_0 = (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q4_0_q8_0_scalar,
            .ops.dot_fp16 = (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q4_0_f16_scalar,
            .ops.dot_q8_k = nullptr,
            .ops.pack_activations_f32 = CPU_ACT_PACK_F32(MARMOT_QUANT_KIND_Q4_0),
            .ops.pack_activations_f16 = CPU_ACT_PACK_F16(MARMOT_QUANT_KIND_Q4_0),
            .supports_fp16_input = true,
            .impl_name = "scalar:q4_0",
        },
    [MARMOT_QUANT_KIND_Q4_1] =
        {
            .format = CPU_QUANT_FORMAT_INFO(MARMOT_QUANT_KIND_Q4_1),
            .ops.dot_q8_0 = (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q4_1_q8_0_scalar,
            .ops.dot_fp16 = (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q4_1_f16_scalar,
            .ops.dot_q8_k = nullptr,
            .ops.pack_activations_f32 = CPU_ACT_PACK_F32(MARMOT_QUANT_KIND_Q4_1),
            .ops.pack_activations_f16 = CPU_ACT_PACK_F16(MARMOT_QUANT_KIND_Q4_1),
            .supports_fp16_input = true,
            .impl_name = "scalar:q4_1",
        },
    [MARMOT_QUANT_KIND_Q5_0] =
        {
            .format = CPU_QUANT_FORMAT_INFO(MARMOT_QUANT_KIND_Q5_0),
            .ops.dot_q8_0 = (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q5_0_q8_0_scalar,
            .ops.dot_fp16 = (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q5_0_f16_scalar,
            .ops.dot_q8_k = nullptr,
            .ops.pack_activations_f32 = CPU_ACT_PACK_F32(MARMOT_QUANT_KIND_Q5_0),
            .ops.pack_activations_f16 = CPU_ACT_PACK_F16(MARMOT_QUANT_KIND_Q5_0),
            .supports_fp16_input = true,
            .impl_name = "scalar:q5_0",
        },
    [MARMOT_QUANT_KIND_Q5_1] =
        {
            .format = CPU_QUANT_FORMAT_INFO(MARMOT_QUANT_KIND_Q5_1),
            .ops.dot_q8_0 = (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q5_1_q8_0_scalar,
            .ops.dot_fp16 = (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q5_1_f16_scalar,
            .ops.dot_q8_k = nullptr,
            .ops.pack_activations_f32 = CPU_ACT_PACK_F32(MARMOT_QUANT_KIND_Q5_1),
            .ops.pack_activations_f16 = CPU_ACT_PACK_F16(MARMOT_QUANT_KIND_Q5_1),
            .supports_fp16_input = true,
            .impl_name = "scalar:q5_1",
        },
    [MARMOT_QUANT_KIND_Q8_0] =
        {
            .format = CPU_QUANT_FORMAT_INFO(MARMOT_QUANT_KIND_Q8_0),
            .ops.dot_q8_0 = (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q8_0_q8_0_scalar,
            .ops.dot_fp16 = (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q8_0_f16_scalar,
            .ops.dot_q8_k = nullptr,
            .ops.pack_activations_f32 = CPU_ACT_PACK_F32(MARMOT_QUANT_KIND_Q8_0),
            .ops.pack_activations_f16 = CPU_ACT_PACK_F16(MARMOT_QUANT_KIND_Q8_0),
            .supports_fp16_input = true,
            .impl_name = "scalar:q8_0",
        },
    [MARMOT_QUANT_KIND_Q8_1] =
        {
            .format = CPU_QUANT_FORMAT_INFO(MARMOT_QUANT_KIND_Q8_1),
            .ops.dot_q8_0 = (cpu_matmul_quant_vec_dot_fn)cpu_vec_dot_q8_1_q8_0_scalar,
            .ops.dot_fp16 = (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q8_1_f16_scalar,
            .ops.dot_q8_k = nullptr,
            .ops.pack_activations_f32 = CPU_ACT_PACK_F32(MARMOT_QUANT_KIND_Q8_1),
            .ops.pack_activations_f16 = CPU_ACT_PACK_F16(MARMOT_QUANT_KIND_Q8_1),
            .supports_fp16_input = true,
            .impl_name = "scalar:q8_1",
        },
    [MARMOT_QUANT_KIND_Q2_K] =
        {
            .format = CPU_QUANT_FORMAT_INFO(MARMOT_QUANT_KIND_Q2_K),
            .ops.dot_q8_0 = nullptr,
            .ops.dot_fp16 = (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q2_k_f16_scalar,
            .ops.dot_q8_k = (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q2_k_q8_k_scalar,
            .ops.pack_activations_f32 = CPU_ACT_PACK_F32(MARMOT_QUANT_KIND_Q2_K),
            .ops.pack_activations_f16 = CPU_ACT_PACK_F16(MARMOT_QUANT_KIND_Q2_K),
            .supports_fp16_input = true,
            .impl_name = "scalar:q2_k",
        },
    [MARMOT_QUANT_KIND_Q3_K] =
        {
            .format = CPU_QUANT_FORMAT_INFO(MARMOT_QUANT_KIND_Q3_K),
            .ops.dot_q8_0 = nullptr,
            .ops.dot_fp16 = (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q3_k_f16_scalar,
            .ops.dot_q8_k = (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q3_k_q8_k_scalar,
            .ops.pack_activations_f32 = CPU_ACT_PACK_F32(MARMOT_QUANT_KIND_Q3_K),
            .ops.pack_activations_f16 = CPU_ACT_PACK_F16(MARMOT_QUANT_KIND_Q3_K),
            .supports_fp16_input = true,
            .impl_name = "scalar:q3_k",
        },
    [MARMOT_QUANT_KIND_Q4_K] =
        {
            .format = CPU_QUANT_FORMAT_INFO(MARMOT_QUANT_KIND_Q4_K),
            .ops.dot_q8_0 = nullptr,
            .ops.dot_fp16 = (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q4_k_f16_scalar,
            .ops.dot_q8_k = (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q4_k_q8_k_scalar,
            .ops.pack_activations_f32 = CPU_ACT_PACK_F32(MARMOT_QUANT_KIND_Q4_K),
            .ops.pack_activations_f16 = CPU_ACT_PACK_F16(MARMOT_QUANT_KIND_Q4_K),
            .supports_fp16_input = true,
            .impl_name = "scalar:q4_k",
        },
    [MARMOT_QUANT_KIND_Q5_K] =
        {
            .format = CPU_QUANT_FORMAT_INFO(MARMOT_QUANT_KIND_Q5_K),
            .ops.dot_q8_0 = nullptr,
            .ops.dot_fp16 = (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q5_k_f16_scalar,
            .ops.dot_q8_k = (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q5_k_q8_k_scalar,
            .ops.pack_activations_f32 = CPU_ACT_PACK_F32(MARMOT_QUANT_KIND_Q5_K),
            .ops.pack_activations_f16 = CPU_ACT_PACK_F16(MARMOT_QUANT_KIND_Q5_K),
            .supports_fp16_input = true,
            .impl_name = "scalar:q5_k",
        },
    [MARMOT_QUANT_KIND_Q6_K] =
        {
            .format = CPU_QUANT_FORMAT_INFO(MARMOT_QUANT_KIND_Q6_K),
            .ops.dot_q8_0 = nullptr,
            .ops.dot_fp16 = (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q6_k_f16_scalar,
            .ops.dot_q8_k = (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q6_k_q8_k_scalar,
            .ops.pack_activations_f32 = CPU_ACT_PACK_F32(MARMOT_QUANT_KIND_Q6_K),
            .ops.pack_activations_f16 = CPU_ACT_PACK_F16(MARMOT_QUANT_KIND_Q6_K),
            .supports_fp16_input = true,
            .impl_name = "scalar:q6_k",
        },
    [MARMOT_QUANT_KIND_Q8_K] = {
        .format = CPU_QUANT_FORMAT_INFO(MARMOT_QUANT_KIND_Q8_K),
        .ops.dot_q8_0 = nullptr,
        .ops.dot_fp16 = (cpu_matmul_quant_vec_dot_fp16_fn)cpu_vec_dot_q8_k_f16_scalar,
        .ops.dot_q8_k = (cpu_matmul_quant_vec_dot_q8k_fn)cpu_vec_dot_q8_k_q8_k_scalar,
        .ops.pack_activations_f32 = CPU_ACT_PACK_F32(MARMOT_QUANT_KIND_Q8_K),
        .ops.pack_activations_f16 = CPU_ACT_PACK_F16(MARMOT_QUANT_KIND_Q8_K),
        .supports_fp16_input = true,
        .impl_name = "scalar:q8_k",
    },
};

const cpu_matmul_quant_kernel_t *cpu_matmul_quant_kernels_scalar(void) {
    return cpu_matmul_quant_kernels_scalar_table;
}
