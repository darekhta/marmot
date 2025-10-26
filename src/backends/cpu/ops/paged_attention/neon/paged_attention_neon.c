#include "ops/paged_attention/paged_attention_kernels.h"

#if HAS_NEON

#include <math.h>

#include "ops/cpu_neon_math.h"

float cpu_paged_attention_dot_f32_neon(const float *a, const float *b, size_t n) {
    size_t d = 0;
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; d + 4 <= n; d += 4) {
        acc = vmlaq_f32(acc, vld1q_f32(a + d), vld1q_f32(b + d));
    }
    float sum = simd_reduce_sum_f32_neon(acc);
    for (; d < n; ++d) {
        sum += a[d] * b[d];
    }
    return sum;
}

static inline void
cpu_paged_attention_update_f32_neon(float *acc, const float *v, size_t n, float acc_scale, float v_scale) {
    size_t d = 0;
    float32x4_t acc_scale_v = vdupq_n_f32(acc_scale);
    float32x4_t v_scale_v = vdupq_n_f32(v_scale);
    for (; d + 4 <= n; d += 4) {
        float32x4_t acc_v = vld1q_f32(acc + d);
        float32x4_t v_v = vld1q_f32(v + d);
        acc_v = vmlaq_f32(vmulq_f32(acc_v, acc_scale_v), v_v, v_scale_v);
        vst1q_f32(acc + d, acc_v);
    }
    for (; d < n; ++d) {
        acc[d] = acc[d] * acc_scale + v[d] * v_scale;
    }
}

void cpu_paged_attention_scale_f32_neon(float *acc, size_t n, float scale) {
    size_t d = 0;
    float32x4_t scale_v = vdupq_n_f32(scale);
    for (; d + 4 <= n; d += 4) {
        float32x4_t acc_v = vld1q_f32(acc + d);
        acc_v = vmulq_f32(acc_v, scale_v);
        vst1q_f32(acc + d, acc_v);
    }
    for (; d < n; ++d) {
        acc[d] *= scale;
    }
}

float cpu_paged_attention_block_sum_f32_neon(
    float *out_vec, const float *v_block, const float *scores, size_t k_count, size_t head_dim, float next_max
) {
    size_t kj = 0;
    float32x4_t max_vec = vdupq_n_f32(next_max);
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (; kj + 4 <= k_count; kj += 4) {
        float32x4_t logit_vec = vld1q_f32(scores + kj);
        float32x4_t weight_vec = cpu_neon_exp_vec(vsubq_f32(logit_vec, max_vec));
        sum_vec = vaddq_f32(sum_vec, weight_vec);

        float weights[4];
        vst1q_f32(weights, weight_vec);
        for (size_t lane = 0; lane < 4; ++lane) {
            float weight = weights[lane];
            if (weight == 0.0f) {
                continue;
            }
            const float *v_row = v_block + (kj + lane) * head_dim;
            cpu_paged_attention_update_f32_neon(out_vec, v_row, head_dim, 1.0f, weight);
        }
    }
    float block_sum = simd_reduce_sum_f32_neon(sum_vec);
    for (; kj < k_count; ++kj) {
        float weight = expf(scores[kj] - next_max);
        block_sum += weight;
        const float *v_row = v_block + kj * head_dim;
        cpu_paged_attention_update_f32_neon(out_vec, v_row, head_dim, 1.0f, weight);
    }
    return block_sum;
}

#endif
