#include "ops/softmax/softmax_internal.h"
#include "ops/softmax/softmax_kernels.h"

static inline float softmax_find_max_f32(const float *data, size_t n) {
    float max_val = data[0];
    for (size_t i = 1; i < n; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    return max_val;
}

static inline float softmax_exp_sum_f32(const float *x, float *out, size_t n, float max_val) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = x[i] - max_val;
        out[i] = softmax_safe_expf(diff);
        sum += out[i];
    }
    return sum;
}

static inline void softmax_normalize_f32(float *data, size_t n, float inv_sum) {
    for (size_t i = 0; i < n; i++) {
        data[i] *= inv_sum;
    }
}

static marmot_error_t softmax_f32_scalar(
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
        for (size_t i = 0; i < outer_size; i++) {
            const float *x_row = x_data + i * softmax_size;
            float *out_row = out_data + i * softmax_size;

            float max_val = softmax_find_max_f32(x_row, softmax_size);
            float sum = softmax_exp_sum_f32(x_row, out_row, softmax_size, max_val);
            softmax_normalize_f32(out_row, softmax_size, 1.0f / sum);
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

const cpu_softmax_traits_t cpu_softmax_f32_scalar_traits = {
    .dtype = MARMOT_DTYPE_FLOAT32,
    .impl_kind = SOFTMAX_IMPL_SCALAR,
    .fn = softmax_f32_scalar,
    .impl_name = "f32_scalar",
};
SOFTMAX_REGISTER_TRAITS(cpu_softmax_f32_scalar_traits)
