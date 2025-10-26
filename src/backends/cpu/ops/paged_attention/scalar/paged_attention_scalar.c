#include <math.h>

#include "ops/paged_attention/paged_attention_kernels.h"

float cpu_paged_attention_dot_f32_scalar(const float *a, const float *b, size_t n) {
    float sum = 0.0f;
    for (size_t d = 0; d < n; ++d) {
        sum += a[d] * b[d];
    }
    return sum;
}

void cpu_paged_attention_scale_f32_scalar(float *acc, size_t n, float scale) {
    for (size_t d = 0; d < n; ++d) {
        acc[d] *= scale;
    }
}

float cpu_paged_attention_block_sum_f32_scalar(
    float *out_vec, const float *v_block, const float *scores, size_t k_count, size_t head_dim, float next_max
) {
    float block_sum = 0.0f;
    for (size_t kj = 0; kj < k_count; ++kj) {
        float weight = expf(scores[kj] - next_max);
        block_sum += weight;
        const float *v_row = v_block + kj * head_dim;
        for (size_t d = 0; d < head_dim; ++d) {
            out_vec[d] += weight * v_row[d];
        }
    }
    return block_sum;
}
