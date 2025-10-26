#include "ops/softmax/softmax_internal.h"
#include "ops/softmax/softmax_kernels.h"

static inline double softmax_find_max_f64(const double *data, size_t n) {
    double max_val = data[0];
    for (size_t i = 1; i < n; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    return max_val;
}

static inline double softmax_exp_sum_f64(const double *x, double *out, size_t n, double max_val) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        out[i] = exp(x[i] - max_val);
        sum += out[i];
    }
    return sum;
}

static inline void softmax_normalize_f64(double *data, size_t n, double inv_sum) {
    for (size_t i = 0; i < n; i++) {
        data[i] *= inv_sum;
    }
}

static marmot_error_t softmax_f64_scalar(
    cpu_context_t *ctx, const marmot_tensor_t *x, const marmot_softmax_shape_t *shape, marmot_tensor_t *out
) {
    (void)ctx;
    const size_t softmax_size = shape->axis_size;
    const size_t inner_stride = shape->inner_stride;
    const size_t outer_size = shape->outer_size;
    const size_t row_count = shape->row_count;

    const double *x_data = (const double *)x->data;
    double *out_data = (double *)out->data;

    if (inner_stride == 1) {
        for (size_t i = 0; i < outer_size; i++) {
            const double *x_row = x_data + i * softmax_size;
            double *out_row = out_data + i * softmax_size;

            double max_val = softmax_find_max_f64(x_row, softmax_size);
            double sum = softmax_exp_sum_f64(x_row, out_row, softmax_size, max_val);
            softmax_normalize_f64(out_row, softmax_size, 1.0 / sum);
        }
    } else {
        for (size_t row = 0; row < row_count; ++row) {
            size_t outer_idx = row / inner_stride;
            size_t inner_idx = row % inner_stride;
            size_t base = outer_idx * softmax_size * inner_stride + inner_idx;

            double max_val = -DBL_MAX;
            for (size_t j = 0; j < softmax_size; ++j) {
                double val = x_data[base + j * inner_stride];
                if (val > max_val) {
                    max_val = val;
                }
            }

            double sum = 0.0;
            for (size_t j = 0; j < softmax_size; ++j) {
                double exp_val = exp(x_data[base + j * inner_stride] - max_val);
                sum += exp_val;
                out_data[base + j * inner_stride] = exp_val;
            }

            double inv_sum = 1.0 / sum;
            for (size_t j = 0; j < softmax_size; ++j) {
                out_data[base + j * inner_stride] *= inv_sum;
            }
        }
    }

    return MARMOT_SUCCESS;
}

const cpu_softmax_traits_t cpu_softmax_f64_scalar_traits = {
    .dtype = MARMOT_DTYPE_FLOAT64,
    .impl_kind = SOFTMAX_IMPL_SCALAR,
    .fn = softmax_f64_scalar,
    .impl_name = "f64_scalar",
};
SOFTMAX_REGISTER_TRAITS(cpu_softmax_f64_scalar_traits)
