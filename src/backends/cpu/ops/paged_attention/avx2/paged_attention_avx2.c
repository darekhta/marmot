#include "ops/paged_attention/paged_attention_kernels.h"

#if HAS_AVX2

#include <math.h>

#include "ops/softmax/avx2/avx2_math.h"

float cpu_paged_attention_dot_f32_avx2(const float *a, const float *b, size_t n) {
    size_t d = 0;
    __m256 acc = _mm256_setzero_ps();
    for (; d + 8 <= n; d += 8) {
        __m256 a_v = _mm256_loadu_ps(a + d);
        __m256 b_v = _mm256_loadu_ps(b + d);
        acc = _mm256_fmadd_ps(a_v, b_v, acc);
    }
    float sum = simd_reduce_sum_f32_avx2(acc);
    for (; d < n; ++d) {
        sum += a[d] * b[d];
    }
    return sum;
}

static inline void
cpu_paged_attention_update_f32_avx2(float *acc, const float *v, size_t n, float acc_scale, float v_scale) {
    size_t d = 0;
    __m256 acc_scale_v = _mm256_set1_ps(acc_scale);
    __m256 v_scale_v = _mm256_set1_ps(v_scale);
    for (; d + 8 <= n; d += 8) {
        __m256 acc_v = _mm256_loadu_ps(acc + d);
        __m256 v_v = _mm256_loadu_ps(v + d);
        acc_v = _mm256_fmadd_ps(v_v, v_scale_v, _mm256_mul_ps(acc_v, acc_scale_v));
        _mm256_storeu_ps(acc + d, acc_v);
    }
    for (; d < n; ++d) {
        acc[d] = acc[d] * acc_scale + v[d] * v_scale;
    }
}

void cpu_paged_attention_scale_f32_avx2(float *acc, size_t n, float scale) {
    size_t d = 0;
    __m256 scale_v = _mm256_set1_ps(scale);
    for (; d + 8 <= n; d += 8) {
        __m256 acc_v = _mm256_loadu_ps(acc + d);
        acc_v = _mm256_mul_ps(acc_v, scale_v);
        _mm256_storeu_ps(acc + d, acc_v);
    }
    for (; d < n; ++d) {
        acc[d] *= scale;
    }
}

float cpu_paged_attention_block_sum_f32_avx2(
    float *out_vec, const float *v_block, const float *scores, size_t k_count, size_t head_dim, float next_max
) {
    size_t kj = 0;
    __m256 max_vec = _mm256_set1_ps(next_max);
    __m256 sum_vec = _mm256_setzero_ps();
    for (; kj + 8 <= k_count; kj += 8) {
        __m256 logit_vec = _mm256_loadu_ps(scores + kj);
        __m256 weight_vec = cpu_avx2_exp_vec(_mm256_sub_ps(logit_vec, max_vec));
        sum_vec = _mm256_add_ps(sum_vec, weight_vec);

        float weights[8];
        _mm256_storeu_ps(weights, weight_vec);
        for (size_t lane = 0; lane < 8; ++lane) {
            float weight = weights[lane];
            if (weight == 0.0f) {
                continue;
            }
            const float *v_row = v_block + (kj + lane) * head_dim;
            cpu_paged_attention_update_f32_avx2(out_vec, v_row, head_dim, 1.0f, weight);
        }
    }
    float block_sum = simd_reduce_sum_f32_avx2(sum_vec);
    for (; kj < k_count; ++kj) {
        float weight = expf(scores[kj] - next_max);
        block_sum += weight;
        const float *v_row = v_block + kj * head_dim;
        cpu_paged_attention_update_f32_avx2(out_vec, v_row, head_dim, 1.0f, weight);
    }
    return block_sum;
}

#endif
