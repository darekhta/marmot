#if MARMOT_ENABLE_NEON

#include "ops/softmax/softmax_internal.h"
#include "ops/softmax/softmax_kernels.h"

static inline float
softmax_exp_sum_f16_neon(const marmot_float16_t *x, marmot_float16_t *out, size_t n, float max_val) {
    float32x4_t max_vec = vdupq_n_f32(max_val);
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        uint16x8_t bits = vld1q_u16((const uint16_t *)(x + i));
        float16x8_t x_f16 = vreinterpretq_f16_u16(bits);

        float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(x_f16));
        float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(x_f16));

        float32x4_t exp_lo = softmax_exp_neon(vsubq_f32(x_lo, max_vec));
        float32x4_t exp_hi = softmax_exp_neon(vsubq_f32(x_hi, max_vec));

        sum_vec = vaddq_f32(sum_vec, exp_lo);
        sum_vec = vaddq_f32(sum_vec, exp_hi);

        float16x4_t out_lo = vcvt_f16_f32(exp_lo);
        float16x4_t out_hi = vcvt_f16_f32(exp_hi);
        float16x8_t packed = vcombine_f16(out_lo, out_hi);
        vst1q_u16((uint16_t *)(out + i), vreinterpretq_u16_f16(packed));
    }

    float sum = softmax_neon_reduce_sum_f32(sum_vec);
    for (; i < n; i++) {
        float diff = marmot_f16_to_f32_ref(x[i]) - max_val;
        float exp_val = softmax_safe_expf(diff);
        out[i] = marmot_f32_to_f16_ref(exp_val);
        sum += exp_val;
    }

    return sum;
}

static marmot_error_t softmax_f16_neon(
    cpu_context_t *ctx, const marmot_tensor_t *x, const marmot_softmax_shape_t *shape, marmot_tensor_t *out
) {
    const size_t softmax_size = shape->axis_size;
    const size_t inner_stride = shape->inner_stride;
    const size_t outer_size = shape->outer_size;
    const size_t row_count = shape->row_count;

    const marmot_float16_t *x_data = (const marmot_float16_t *)x->data;
    marmot_float16_t *out_data = (marmot_float16_t *)out->data;

    if (inner_stride == 1) {
        for (size_t i = 0; i < outer_size; i++) {
            const marmot_float16_t *x_row = x_data + i * softmax_size;
            marmot_float16_t *out_row = out_data + i * softmax_size;

            float max_val = softmax_find_max_f16(ctx, x_row, softmax_size);
            float sum = softmax_exp_sum_f16_neon(x_row, out_row, softmax_size, max_val);
            softmax_scale_f16(ctx, out_row, softmax_size, 1.0f / sum);
        }
    } else {
        for (size_t row = 0; row < row_count; ++row) {
            size_t outer_idx = row / inner_stride;
            size_t inner_idx = row % inner_stride;
            size_t base = outer_idx * softmax_size * inner_stride + inner_idx;

            float max_val = -FLT_MAX;
            for (size_t j = 0; j < softmax_size; ++j) {
                float val = marmot_f16_to_f32_ref(x_data[base + j * inner_stride]);
                if (val > max_val) {
                    max_val = val;
                }
            }

            float sum = 0.0f;
            for (size_t j = 0; j < softmax_size; ++j) {
                float diff = marmot_f16_to_f32_ref(x_data[base + j * inner_stride]) - max_val;
                float exp_val = softmax_safe_expf(diff);
                sum += exp_val;
                out_data[base + j * inner_stride] = marmot_f32_to_f16_ref(exp_val);
            }

            float inv_sum = 1.0f / sum;
            for (size_t j = 0; j < softmax_size; ++j) {
                float value = marmot_f16_to_f32_ref(out_data[base + j * inner_stride]);
                out_data[base + j * inner_stride] = marmot_f32_to_f16_ref(value * inv_sum);
            }
        }
    }

    return MARMOT_SUCCESS;
}

const cpu_softmax_traits_t cpu_softmax_f16_neon_traits = {
    .dtype = MARMOT_DTYPE_FLOAT16,
    .impl_kind = SOFTMAX_IMPL_NEON,
    .fn = softmax_f16_neon,
    .impl_name = "f16_neon",
};
SOFTMAX_REGISTER_TRAITS(cpu_softmax_f16_neon_traits)

#endif
