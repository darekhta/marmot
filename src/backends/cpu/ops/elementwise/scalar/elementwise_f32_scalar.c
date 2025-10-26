#include "marmot/tensor.h"

#include <math.h>

#include "cpu_backend_internal.h"

// -----------------------------------------------------------------------------
// Scalar float32 kernels
// -----------------------------------------------------------------------------

static inline float cpu_silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

static inline float cpu_gelu_tanh_f32(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static inline bool cpu_row_strided_2d(
    const marmot_tensor_t *a, const marmot_tensor_t *b, const marmot_tensor_t *out, size_t *rows, size_t *cols,
    size_t *a_row_stride, size_t *b_row_stride, size_t *out_row_stride
) {
    if (a->shape.ndim != 2 || b->shape.ndim != 2 || out->shape.ndim != 2) {
        return false;
    }
    if (a->shape.shape[0] != b->shape.shape[0] || a->shape.shape[1] != b->shape.shape[1] ||
        a->shape.shape[0] != out->shape.shape[0] || a->shape.shape[1] != out->shape.shape[1]) {
        return false;
    }
    if (a->shape.strides[1] != 1 || b->shape.strides[1] != 1 || out->shape.strides[1] != 1) {
        return false;
    }
    if (a->shape.strides[0] < a->shape.shape[1] || b->shape.strides[0] < b->shape.shape[1] ||
        out->shape.strides[0] < out->shape.shape[1]) {
        return false;
    }
    *rows = a->shape.shape[0];
    *cols = a->shape.shape[1];
    *a_row_stride = a->shape.strides[0];
    *b_row_stride = b->shape.strides[0];
    *out_row_stride = out->shape.strides[0];
    return true;
}

marmot_error_t
cpu_add_f32_scalar(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    size_t rows = 0;
    size_t cols = 0;
    size_t a_row_stride = 0;
    size_t b_row_stride = 0;
    size_t out_row_stride = 0;
    if (cpu_row_strided_2d(a, b, out, &rows, &cols, &a_row_stride, &b_row_stride, &out_row_stride) &&
        (a_row_stride != cols || b_row_stride != cols || out_row_stride != cols)) {
        const float *lhs = (const float *)a->data;
        const float *rhs = (const float *)b->data;
        float *dst = (float *)out->data;
        for (size_t row = 0; row < rows; ++row) {
            const size_t a_base = row * a_row_stride;
            const size_t b_base = row * b_row_stride;
            const size_t out_base = row * out_row_stride;
            for (size_t col = 0; col < cols; ++col) {
                dst[out_base + col] = lhs[a_base + col] + rhs[b_base + col];
            }
        }
        return MARMOT_SUCCESS;
    }
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = lhs[i] + rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_sub_f32_scalar(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    size_t rows = 0;
    size_t cols = 0;
    size_t a_row_stride = 0;
    size_t b_row_stride = 0;
    size_t out_row_stride = 0;
    if (cpu_row_strided_2d(a, b, out, &rows, &cols, &a_row_stride, &b_row_stride, &out_row_stride) &&
        (a_row_stride != cols || b_row_stride != cols || out_row_stride != cols)) {
        const float *lhs = (const float *)a->data;
        const float *rhs = (const float *)b->data;
        float *dst = (float *)out->data;
        for (size_t row = 0; row < rows; ++row) {
            const size_t a_base = row * a_row_stride;
            const size_t b_base = row * b_row_stride;
            const size_t out_base = row * out_row_stride;
            for (size_t col = 0; col < cols; ++col) {
                dst[out_base + col] = lhs[a_base + col] - rhs[b_base + col];
            }
        }
        return MARMOT_SUCCESS;
    }
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = lhs[i] - rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_mul_f32_scalar(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    size_t rows = 0;
    size_t cols = 0;
    size_t a_row_stride = 0;
    size_t b_row_stride = 0;
    size_t out_row_stride = 0;
    if (cpu_row_strided_2d(a, b, out, &rows, &cols, &a_row_stride, &b_row_stride, &out_row_stride) &&
        (a_row_stride != cols || b_row_stride != cols || out_row_stride != cols)) {
        const float *lhs = (const float *)a->data;
        const float *rhs = (const float *)b->data;
        float *dst = (float *)out->data;
        for (size_t row = 0; row < rows; ++row) {
            const size_t a_base = row * a_row_stride;
            const size_t b_base = row * b_row_stride;
            const size_t out_base = row * out_row_stride;
            for (size_t col = 0; col < cols; ++col) {
                dst[out_base + col] = lhs[a_base + col] * rhs[b_base + col];
            }
        }
        return MARMOT_SUCCESS;
    }
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = lhs[i] * rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_swiglu_f32_scalar(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    size_t rows = 0;
    size_t cols = 0;
    size_t a_row_stride = 0;
    size_t b_row_stride = 0;
    size_t out_row_stride = 0;
    if (cpu_row_strided_2d(a, b, out, &rows, &cols, &a_row_stride, &b_row_stride, &out_row_stride) &&
        (a_row_stride != cols || b_row_stride != cols || out_row_stride != cols)) {
        const float *lhs = (const float *)a->data;
        const float *rhs = (const float *)b->data;
        float *dst = (float *)out->data;
        for (size_t row = 0; row < rows; ++row) {
            const size_t a_base = row * a_row_stride;
            const size_t b_base = row * b_row_stride;
            const size_t out_base = row * out_row_stride;
            for (size_t col = 0; col < cols; ++col) {
                const float x = lhs[a_base + col];
                const float y = rhs[b_base + col];
                dst[out_base + col] = cpu_silu_f32(x) * y;
            }
        }
        return MARMOT_SUCCESS;
    }
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = cpu_silu_f32(lhs[i]) * rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_geglu_f32_scalar(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    size_t rows = 0;
    size_t cols = 0;
    size_t a_row_stride = 0;
    size_t b_row_stride = 0;
    size_t out_row_stride = 0;
    if (cpu_row_strided_2d(a, b, out, &rows, &cols, &a_row_stride, &b_row_stride, &out_row_stride) &&
        (a_row_stride != cols || b_row_stride != cols || out_row_stride != cols)) {
        const float *lhs = (const float *)a->data;
        const float *rhs = (const float *)b->data;
        float *dst = (float *)out->data;
        for (size_t row = 0; row < rows; ++row) {
            const size_t a_base = row * a_row_stride;
            const size_t b_base = row * b_row_stride;
            const size_t out_base = row * out_row_stride;
            for (size_t col = 0; col < cols; ++col) {
                const float x = lhs[a_base + col];
                const float y = rhs[b_base + col];
                dst[out_base + col] = cpu_gelu_tanh_f32(x) * y;
            }
        }
        return MARMOT_SUCCESS;
    }
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = cpu_gelu_tanh_f32(lhs[i]) * rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_div_f32_scalar(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    size_t rows = 0;
    size_t cols = 0;
    size_t a_row_stride = 0;
    size_t b_row_stride = 0;
    size_t out_row_stride = 0;
    if (cpu_row_strided_2d(a, b, out, &rows, &cols, &a_row_stride, &b_row_stride, &out_row_stride) &&
        (a_row_stride != cols || b_row_stride != cols || out_row_stride != cols)) {
        const float *lhs = (const float *)a->data;
        const float *rhs = (const float *)b->data;
        float *dst = (float *)out->data;
        for (size_t row = 0; row < rows; ++row) {
            const size_t a_base = row * a_row_stride;
            const size_t b_base = row * b_row_stride;
            const size_t out_base = row * out_row_stride;
            for (size_t col = 0; col < cols; ++col) {
                dst[out_base + col] = lhs[a_base + col] / rhs[b_base + col];
            }
        }
        return MARMOT_SUCCESS;
    }
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = lhs[i] / rhs[i];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_min_f32_scalar(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    size_t rows = 0;
    size_t cols = 0;
    size_t a_row_stride = 0;
    size_t b_row_stride = 0;
    size_t out_row_stride = 0;
    if (cpu_row_strided_2d(a, b, out, &rows, &cols, &a_row_stride, &b_row_stride, &out_row_stride) &&
        (a_row_stride != cols || b_row_stride != cols || out_row_stride != cols)) {
        const float *lhs = (const float *)a->data;
        const float *rhs = (const float *)b->data;
        float *dst = (float *)out->data;
        for (size_t row = 0; row < rows; ++row) {
            const size_t a_base = row * a_row_stride;
            const size_t b_base = row * b_row_stride;
            const size_t out_base = row * out_row_stride;
            for (size_t col = 0; col < cols; ++col) {
                dst[out_base + col] = fminf(lhs[a_base + col], rhs[b_base + col]);
            }
        }
        return MARMOT_SUCCESS;
    }
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = fminf(lhs[i], rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_max_f32_scalar(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    size_t rows = 0;
    size_t cols = 0;
    size_t a_row_stride = 0;
    size_t b_row_stride = 0;
    size_t out_row_stride = 0;
    if (cpu_row_strided_2d(a, b, out, &rows, &cols, &a_row_stride, &b_row_stride, &out_row_stride) &&
        (a_row_stride != cols || b_row_stride != cols || out_row_stride != cols)) {
        const float *lhs = (const float *)a->data;
        const float *rhs = (const float *)b->data;
        float *dst = (float *)out->data;
        for (size_t row = 0; row < rows; ++row) {
            const size_t a_base = row * a_row_stride;
            const size_t b_base = row * b_row_stride;
            const size_t out_base = row * out_row_stride;
            for (size_t col = 0; col < cols; ++col) {
                dst[out_base + col] = fmaxf(lhs[a_base + col], rhs[b_base + col]);
            }
        }
        return MARMOT_SUCCESS;
    }
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = fmaxf(lhs[i], rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_pow_f32_scalar(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    size_t rows = 0;
    size_t cols = 0;
    size_t a_row_stride = 0;
    size_t b_row_stride = 0;
    size_t out_row_stride = 0;
    if (cpu_row_strided_2d(a, b, out, &rows, &cols, &a_row_stride, &b_row_stride, &out_row_stride) &&
        (a_row_stride != cols || b_row_stride != cols || out_row_stride != cols)) {
        const float *lhs = (const float *)a->data;
        const float *rhs = (const float *)b->data;
        float *dst = (float *)out->data;
        for (size_t row = 0; row < rows; ++row) {
            const size_t a_base = row * a_row_stride;
            const size_t b_base = row * b_row_stride;
            const size_t out_base = row * out_row_stride;
            for (size_t col = 0; col < cols; ++col) {
                dst[out_base + col] = powf(lhs[a_base + col], rhs[b_base + col]);
            }
        }
        return MARMOT_SUCCESS;
    }
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = powf(lhs[i], rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_mod_f32_scalar(const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {
    (void)device_ctx;
    size_t rows = 0;
    size_t cols = 0;
    size_t a_row_stride = 0;
    size_t b_row_stride = 0;
    size_t out_row_stride = 0;
    if (cpu_row_strided_2d(a, b, out, &rows, &cols, &a_row_stride, &b_row_stride, &out_row_stride) &&
        (a_row_stride != cols || b_row_stride != cols || out_row_stride != cols)) {
        const float *lhs = (const float *)a->data;
        const float *rhs = (const float *)b->data;
        float *dst = (float *)out->data;
        for (size_t row = 0; row < rows; ++row) {
            const size_t a_base = row * a_row_stride;
            const size_t b_base = row * b_row_stride;
            const size_t out_base = row * out_row_stride;
            for (size_t col = 0; col < cols; ++col) {
                dst[out_base + col] = fmodf(lhs[a_base + col], rhs[b_base + col]);
            }
        }
        return MARMOT_SUCCESS;
    }
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = fmodf(lhs[i], rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_eq_f32_scalar(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] == rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_ne_f32_scalar(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] != rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_lt_f32_scalar(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] < rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_le_f32_scalar(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] <= rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_gt_f32_scalar(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] > rhs[i]);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_compare_ge_f32_scalar(
    const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out
) {
    (void)device_ctx;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    uint8_t *dst = (uint8_t *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = (uint8_t)(lhs[i] >= rhs[i]);
    }
    return MARMOT_SUCCESS;
}
