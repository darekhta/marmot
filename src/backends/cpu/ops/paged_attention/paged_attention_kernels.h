#ifndef CPU_PAGED_ATTENTION_KERNELS_H
#define CPU_PAGED_ATTENTION_KERNELS_H

#include <stddef.h>

#include "cpu_backend_internal.h"

typedef float (*cpu_paged_attention_dot_f32_fn)(const float *a, const float *b, size_t n);
typedef void (*cpu_paged_attention_scale_f32_fn)(float *acc, size_t n, float scale);
typedef float (*cpu_paged_attention_block_sum_f32_fn)(
    float *out_vec, const float *v_block, const float *scores, size_t k_count, size_t head_dim, float next_max
);

typedef struct {
    cpu_paged_attention_dot_f32_fn dot_f32;
    cpu_paged_attention_scale_f32_fn scale_f32;
    cpu_paged_attention_block_sum_f32_fn block_sum_f32;
} cpu_paged_attention_f32_ops_t;

float cpu_paged_attention_dot_f32_scalar(const float *a, const float *b, size_t n);
void cpu_paged_attention_scale_f32_scalar(float *acc, size_t n, float scale);
float cpu_paged_attention_block_sum_f32_scalar(
    float *out_vec, const float *v_block, const float *scores, size_t k_count, size_t head_dim, float next_max
);

#if HAS_NEON
float cpu_paged_attention_dot_f32_neon(const float *a, const float *b, size_t n);
void cpu_paged_attention_scale_f32_neon(float *acc, size_t n, float scale);
float cpu_paged_attention_block_sum_f32_neon(
    float *out_vec, const float *v_block, const float *scores, size_t k_count, size_t head_dim, float next_max
);
#endif

#if HAS_AVX2
float cpu_paged_attention_dot_f32_avx2(const float *a, const float *b, size_t n);
void cpu_paged_attention_scale_f32_avx2(float *acc, size_t n, float scale);
float cpu_paged_attention_block_sum_f32_avx2(
    float *out_vec, const float *v_block, const float *scores, size_t k_count, size_t head_dim, float next_max
);
#endif

#endif // CPU_PAGED_ATTENTION_KERNELS_H
