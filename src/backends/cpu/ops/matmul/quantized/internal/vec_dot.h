#ifndef CPU_VEC_DOT_H
#define CPU_VEC_DOT_H

#include "marmot/quant_block.h"

#include <stddef.h>

#include "cpu_backend_internal.h"

#define CPU_VEC_DOT_VARIANT_DECL(suffix)                                                                               \
    float cpu_vec_dot_q4_0_q8_0_##suffix(                                                                              \
        const marmot_q4_0_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks                  \
    );                                                                                                                 \
    float cpu_vec_dot_q4_0_f16_##suffix(                                                                               \
        const marmot_q4_0_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,   \
        size_t K                                                                                                       \
    );                                                                                                                 \
    float cpu_vec_dot_q4_1_q8_0_##suffix(                                                                              \
        const marmot_q4_1_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks                  \
    );                                                                                                                 \
    float cpu_vec_dot_q4_1_f16_##suffix(                                                                               \
        const marmot_q4_1_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,   \
        size_t K                                                                                                       \
    );                                                                                                                 \
    float cpu_vec_dot_q5_0_q8_0_##suffix(                                                                              \
        const marmot_q5_0_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks                  \
    );                                                                                                                 \
    float cpu_vec_dot_q5_1_q8_0_##suffix(                                                                              \
        const marmot_q5_1_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks                  \
    );                                                                                                                 \
    float cpu_vec_dot_q5_0_f16_##suffix(                                                                               \
        const marmot_q5_0_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,   \
        size_t K                                                                                                       \
    );                                                                                                                 \
    float cpu_vec_dot_q5_1_f16_##suffix(                                                                               \
        const marmot_q5_1_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,   \
        size_t K                                                                                                       \
    );                                                                                                                 \
    float cpu_vec_dot_q8_0_f16_##suffix(                                                                               \
        const marmot_q8_0_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,   \
        size_t K                                                                                                       \
    );                                                                                                                 \
    float cpu_vec_dot_q8_0_q8_0_##suffix(                                                                              \
        const marmot_q8_0_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks                  \
    );                                                                                                                 \
    float cpu_vec_dot_q8_1_q8_0_##suffix(                                                                              \
        const marmot_q8_1_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks                  \
    );                                                                                                                 \
    float cpu_vec_dot_q8_1_f16_##suffix(                                                                               \
        const marmot_q8_1_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,   \
        size_t K                                                                                                       \
    );                                                                                                                 \
    float cpu_vec_dot_q2_k_q8_k_##suffix(                                                                              \
        const marmot_q2_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks                  \
    );                                                                                                                 \
    float cpu_vec_dot_q3_k_q8_k_##suffix(                                                                              \
        const marmot_q3_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks                  \
    );                                                                                                                 \
    float cpu_vec_dot_q4_k_q8_k_##suffix(                                                                              \
        const marmot_q4_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks                  \
    );                                                                                                                 \
    float cpu_vec_dot_q5_k_q8_k_##suffix(                                                                              \
        const marmot_q5_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks                  \
    );                                                                                                                 \
    float cpu_vec_dot_q6_k_q8_k_##suffix(                                                                              \
        const marmot_q6_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks                  \
    );                                                                                                                 \
    float cpu_vec_dot_q8_k_q8_k_##suffix(                                                                              \
        const marmot_q8_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks                  \
    );                                                                                                                 \
    float cpu_vec_dot_q2_k_f16_##suffix(                                                                               \
        const marmot_q2_k_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,   \
        size_t K                                                                                                       \
    );                                                                                                                 \
    float cpu_vec_dot_q3_k_f16_##suffix(                                                                               \
        const marmot_q3_k_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,   \
        size_t K                                                                                                       \
    );                                                                                                                 \
    float cpu_vec_dot_q4_k_f16_##suffix(                                                                               \
        const marmot_q4_k_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,   \
        size_t K                                                                                                       \
    );                                                                                                                 \
    float cpu_vec_dot_q5_k_f16_##suffix(                                                                               \
        const marmot_q5_k_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,   \
        size_t K                                                                                                       \
    );                                                                                                                 \
    float cpu_vec_dot_q6_k_f16_##suffix(                                                                               \
        const marmot_q6_k_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,   \
        size_t K                                                                                                       \
    );                                                                                                                 \
    float cpu_vec_dot_q8_k_f16_##suffix(                                                                               \
        const marmot_q8_k_block_t *weights, const marmot_float16_t *activations, size_t stride_k, size_t num_blocks,   \
        size_t K                                                                                                       \
    )

CPU_VEC_DOT_VARIANT_DECL(scalar);
CPU_VEC_DOT_VARIANT_DECL(neon);
CPU_VEC_DOT_VARIANT_DECL(avx2);

#undef CPU_VEC_DOT_VARIANT_DECL

#if MARMOT_ENABLE_NEON && defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
float cpu_vec_dot_q8_0_q8_0_neon_dotprod(
    const marmot_q8_0_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
);
float cpu_vec_dot_q8_1_q8_0_neon_dotprod(
    const marmot_q8_1_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
);
float cpu_vec_dot_q4_0_q8_0_neon_dotprod(
    const marmot_q4_0_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
);
float cpu_vec_dot_q4_1_q8_0_neon_dotprod(
    const marmot_q4_1_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
);
float cpu_vec_dot_q5_0_q8_0_neon_dotprod(
    const marmot_q5_0_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
);
float cpu_vec_dot_q5_1_q8_0_neon_dotprod(
    const marmot_q5_1_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
);
float cpu_vec_dot_q8_k_q8_k_neon_dotprod(
    const marmot_q8_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
);
float cpu_vec_dot_q2_k_q8_k_neon_dotprod(
    const marmot_q2_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
);
float cpu_vec_dot_q3_k_q8_k_neon_dotprod(
    const marmot_q3_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
);
float cpu_vec_dot_q4_k_q8_k_neon_dotprod(
    const marmot_q4_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
);
float cpu_vec_dot_q5_k_q8_k_neon_dotprod(
    const marmot_q5_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
);
float cpu_vec_dot_q6_k_q8_k_neon_dotprod(
    const marmot_q6_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
);
#endif

#if MARMOT_ENABLE_NEON && defined(__aarch64__) && (defined(__ARM_FEATURE_I8MM) || defined(__ARM_FEATURE_MATMUL_INT8))
float cpu_vec_dot_q8_0_q8_0_neon_i8mm(
    const marmot_q8_0_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
);
float cpu_vec_dot_q8_1_q8_0_neon_i8mm(
    const marmot_q8_1_block_t *weights, const marmot_q8_0_block_t *activations, size_t num_blocks
);
float cpu_vec_dot_q8_k_q8_k_neon_i8mm(
    const marmot_q8_k_block_t *weights, const marmot_q8_k_block_t *activations, size_t num_blocks
);
#endif

marmot_error_t cpu_vec_dot(const void *device_ctx, const marmot_vec_dot_descriptor_t *desc, float *result);

#endif
