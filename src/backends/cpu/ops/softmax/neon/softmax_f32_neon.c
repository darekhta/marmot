#if MARMOT_ENABLE_NEON

#include "ops/softmax/softmax_internal.h"
#include "ops/softmax/softmax_kernels.h"

static inline float softmax_find_max_neon(const float *data, size_t n) {
    float32x4_t max_vec = vdupq_n_f32(-FLT_MAX);
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        max_vec = vmaxq_f32(max_vec, v);
    }

    float max_val = softmax_neon_reduce_max_f32(max_vec);

    for (; i < n; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }

    return max_val;
}

static inline float softmax_exp_sum_neon(const float *x, float *out, size_t n, float max_val) {
    float32x4_t vmax = vdupq_n_f32(max_val);
    float32x4_t vsum = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vexp = softmax_exp_neon(vsubq_f32(vx, vmax));
        vst1q_f32(out + i, vexp);
        vsum = vaddq_f32(vsum, vexp);
    }
    float sum = softmax_neon_reduce_sum_f32(vsum);
    for (; i < n; i++) {
        float diff = x[i] - max_val;
        out[i] = softmax_safe_expf(diff);
        sum += out[i];
    }
    return sum;
}

static inline void softmax_normalize_neon(float *data, size_t n, float inv_sum) {
    float32x4_t inv_sum_vec = vdupq_n_f32(inv_sum);
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        v = vmulq_f32(v, inv_sum_vec);
        vst1q_f32(data + i, v);
    }

    for (; i < n; i++) {
        data[i] *= inv_sum;
    }
}

static inline void softmax_two_pass_neon(const float *x, float *out, size_t n) {
    softmax_accum_f32_t acc = softmax_accumulate_f32(x, n);
    float inv_sum = acc.sum_exp > 0.0f ? 1.0f / acc.sum_exp : 0.0f;
    float32x4_t max_vec = vdupq_n_f32(acc.max_val);
    float32x4_t inv_vec = vdupq_n_f32(inv_sum);
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vexp = softmax_exp_neon(vsubq_f32(vx, max_vec));
        vexp = vmulq_f32(vexp, inv_vec);
        vst1q_f32(out + i, vexp);
    }

    for (; i < n; i++) {
        float diff = x[i] - acc.max_val;
        out[i] = softmax_safe_expf(diff) * inv_sum;
    }
}

static marmot_error_t softmax_f32_neon(
    cpu_context_t *ctx, const marmot_tensor_t *x, const marmot_softmax_shape_t *shape, marmot_tensor_t *out
) {
    (void)ctx;
    const size_t softmax_size = shape->axis_size;
    const size_t inner_stride = shape->inner_stride;
    const size_t outer_size = shape->outer_size;
    const size_t row_count = shape->row_count;

    const float *x_data = (const float *)x->data;
    float *out_data = (float *)out->data;

    if (inner_stride == 1) {
        const bool use_two_pass = softmax_should_use_two_pass(softmax_size);
        for (size_t i = 0; i < outer_size; i++) {
            const float *x_row = x_data + i * softmax_size;
            float *out_row = out_data + i * softmax_size;

            if (use_two_pass) {
                softmax_two_pass_neon(x_row, out_row, softmax_size);
            } else {
                float max_val = softmax_find_max_neon(x_row, softmax_size);
                float sum = softmax_exp_sum_neon(x_row, out_row, softmax_size, max_val);
                softmax_normalize_neon(out_row, softmax_size, 1.0f / sum);
            }
        }
    } else {
        for (size_t row = 0; row < row_count; ++row) {
            size_t outer_idx = row / inner_stride;
            size_t inner_idx = row % inner_stride;
            size_t base = outer_idx * softmax_size * inner_stride + inner_idx;

            float max_val = -FLT_MAX;
            for (size_t j = 0; j < softmax_size; ++j) {
                float val = x_data[base + j * inner_stride];
                if (val > max_val) {
                    max_val = val;
                }
            }

            float sum = 0.0f;
            for (size_t j = 0; j < softmax_size; ++j) {
                float diff = x_data[base + j * inner_stride] - max_val;
                float exp_val = softmax_safe_expf(diff);
                sum += exp_val;
                out_data[base + j * inner_stride] = exp_val;
            }

            float inv_sum = 1.0f / sum;
            for (size_t j = 0; j < softmax_size; ++j) {
                out_data[base + j * inner_stride] *= inv_sum;
            }
        }
    }

    return MARMOT_SUCCESS;
}

const cpu_softmax_traits_t cpu_softmax_f32_neon_traits = {
    .dtype = MARMOT_DTYPE_FLOAT32,
    .impl_kind = SOFTMAX_IMPL_NEON,
    .fn = softmax_f32_neon,
    .impl_name = "f32_neon",
};
SOFTMAX_REGISTER_TRAITS(cpu_softmax_f32_neon_traits)

#endif
