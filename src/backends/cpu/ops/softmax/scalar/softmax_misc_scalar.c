#include "ops/softmax/softmax_internal.h"
#include "ops/softmax/softmax_kernels.h"

static marmot_error_t softmax_f16_scalar(
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
            float sum = 0.0f;
            for (size_t j = 0; j < softmax_size; ++j) {
                float diff = marmot_f16_to_f32_ref(x_row[j]) - max_val;
                float exp_val = softmax_safe_expf(diff);
                sum += exp_val;
                out_row[j] = marmot_f32_to_f16_ref(exp_val);
            }
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

static marmot_error_t softmax_bf16_scalar(
    cpu_context_t *ctx, const marmot_tensor_t *x, const marmot_softmax_shape_t *shape, marmot_tensor_t *out
) {
    const size_t softmax_size = shape->axis_size;
    const size_t inner_stride = shape->inner_stride;
    const size_t outer_size = shape->outer_size;
    const size_t row_count = shape->row_count;

    const marmot_bfloat16_t *x_data = (const marmot_bfloat16_t *)x->data;
    marmot_bfloat16_t *out_data = (marmot_bfloat16_t *)out->data;

    if (inner_stride == 1) {
        for (size_t i = 0; i < outer_size; i++) {
            const marmot_bfloat16_t *x_row = x_data + i * softmax_size;
            marmot_bfloat16_t *out_row = out_data + i * softmax_size;

            float max_val = softmax_find_max_bf16(ctx, x_row, softmax_size);
            float sum = 0.0f;
            for (size_t j = 0; j < softmax_size; ++j) {
                float diff = marmot_bf16_to_f32_ref(x_row[j]) - max_val;
                float exp_val = softmax_safe_expf(diff);
                sum += exp_val;
                out_row[j] = marmot_f32_to_bf16_ref(exp_val);
            }
            softmax_scale_bf16(ctx, out_row, softmax_size, 1.0f / sum);
        }
    } else {
        for (size_t row = 0; row < row_count; ++row) {
            size_t outer_idx = row / inner_stride;
            size_t inner_idx = row % inner_stride;
            size_t base = outer_idx * softmax_size * inner_stride + inner_idx;

            float max_val = -FLT_MAX;
            for (size_t j = 0; j < softmax_size; ++j) {
                float val = marmot_bf16_to_f32_ref(x_data[base + j * inner_stride]);
                if (val > max_val) {
                    max_val = val;
                }
            }

            float sum = 0.0f;
            for (size_t j = 0; j < softmax_size; ++j) {
                float diff = marmot_bf16_to_f32_ref(x_data[base + j * inner_stride]) - max_val;
                float exp_val = softmax_safe_expf(diff);
                sum += exp_val;
                out_data[base + j * inner_stride] = marmot_f32_to_bf16_ref(exp_val);
            }

            float inv_sum = 1.0f / sum;
            for (size_t j = 0; j < softmax_size; ++j) {
                float value = marmot_bf16_to_f32_ref(out_data[base + j * inner_stride]);
                out_data[base + j * inner_stride] = marmot_f32_to_bf16_ref(value * inv_sum);
            }
        }
    }

    return MARMOT_SUCCESS;
}

#if MARMOT_ENABLE_FP8
static void softmax_compute_fp8_e4m3_row(
    const marmot_float8_e4m3_t *x_data, marmot_float8_e4m3_t *out_data, size_t base, size_t softmax_size,
    size_t inner_stride
) {
    float max_val = -FLT_MAX;
    for (size_t j = 0; j < softmax_size; ++j) {
        float val = marmot_fp8_e4m3_to_native(x_data[base + j * inner_stride]);
        if (val > max_val) {
            max_val = val;
        }
    }
    float sum = 0.0f;
    for (size_t j = 0; j < softmax_size; ++j) {
        float diff = marmot_fp8_e4m3_to_native(x_data[base + j * inner_stride]) - max_val;
        float exp_val = softmax_safe_expf(diff);
        sum += exp_val;
        out_data[base + j * inner_stride] = marmot_native_to_fp8_e4m3((_Float16)exp_val);
    }
    float inv_sum = 1.0f / sum;
    for (size_t j = 0; j < softmax_size; ++j) {
        float value = marmot_fp8_e4m3_to_native(out_data[base + j * inner_stride]);
        out_data[base + j * inner_stride] = marmot_native_to_fp8_e4m3((_Float16)(value * inv_sum));
    }
}

static void softmax_compute_fp8_e5m2_row(
    const marmot_float8_e5m2_t *x_data, marmot_float8_e5m2_t *out_data, size_t base, size_t softmax_size,
    size_t inner_stride
) {
    float max_val = -FLT_MAX;
    for (size_t j = 0; j < softmax_size; ++j) {
        float val = marmot_fp8_e5m2_to_native(x_data[base + j * inner_stride]);
        if (val > max_val) {
            max_val = val;
        }
    }
    float sum = 0.0f;
    for (size_t j = 0; j < softmax_size; ++j) {
        float diff = marmot_fp8_e5m2_to_native(x_data[base + j * inner_stride]) - max_val;
        float exp_val = softmax_safe_expf(diff);
        sum += exp_val;
        out_data[base + j * inner_stride] = marmot_native_to_fp8_e5m2((_Float16)exp_val);
    }
    float inv_sum = 1.0f / sum;
    for (size_t j = 0; j < softmax_size; ++j) {
        float value = marmot_fp8_e5m2_to_native(out_data[base + j * inner_stride]);
        out_data[base + j * inner_stride] = marmot_native_to_fp8_e5m2((_Float16)(value * inv_sum));
    }
}

static marmot_error_t softmax_fp8_e4m3_scalar(
    cpu_context_t *ctx, const marmot_tensor_t *x, const marmot_softmax_shape_t *shape, marmot_tensor_t *out
) {
    (void)ctx;
    const size_t softmax_size = shape->axis_size;
    const size_t inner_stride = shape->inner_stride;
    const size_t outer_size = shape->outer_size;
    const size_t row_count = shape->row_count;

    const marmot_float8_e4m3_t *x_data = (const marmot_float8_e4m3_t *)x->data;
    marmot_float8_e4m3_t *out_data = (marmot_float8_e4m3_t *)out->data;

    if (inner_stride == 1) {
        for (size_t i = 0; i < outer_size; i++) {
            size_t base = i * softmax_size;
            softmax_compute_fp8_e4m3_row(x_data, out_data, base, softmax_size, inner_stride);
        }
    } else {
        for (size_t row = 0; row < row_count; ++row) {
            size_t outer_idx = row / inner_stride;
            size_t inner_idx = row % inner_stride;
            size_t base = outer_idx * softmax_size * inner_stride + inner_idx;
            softmax_compute_fp8_e4m3_row(x_data, out_data, base, softmax_size, inner_stride);
        }
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t softmax_fp8_e5m2_scalar(
    cpu_context_t *ctx, const marmot_tensor_t *x, const marmot_softmax_shape_t *shape, marmot_tensor_t *out
) {
    (void)ctx;
    const size_t softmax_size = shape->axis_size;
    const size_t inner_stride = shape->inner_stride;
    const size_t outer_size = shape->outer_size;
    const size_t row_count = shape->row_count;

    const marmot_float8_e5m2_t *x_data = (const marmot_float8_e5m2_t *)x->data;
    marmot_float8_e5m2_t *out_data = (marmot_float8_e5m2_t *)out->data;

    if (inner_stride == 1) {
        for (size_t i = 0; i < outer_size; i++) {
            size_t base = i * softmax_size;
            softmax_compute_fp8_e5m2_row(x_data, out_data, base, softmax_size, inner_stride);
        }
    } else {
        for (size_t row = 0; row < row_count; ++row) {
            size_t outer_idx = row / inner_stride;
            size_t inner_idx = row % inner_stride;
            size_t base = outer_idx * softmax_size * inner_stride + inner_idx;
            softmax_compute_fp8_e5m2_row(x_data, out_data, base, softmax_size, inner_stride);
        }
    }
    return MARMOT_SUCCESS;
}
#endif

const cpu_softmax_traits_t cpu_softmax_f16_scalar_traits = {
    .dtype = MARMOT_DTYPE_FLOAT16,
    .impl_kind = SOFTMAX_IMPL_SCALAR,
    .fn = softmax_f16_scalar,
    .impl_name = "f16_scalar",
};
SOFTMAX_REGISTER_TRAITS(cpu_softmax_f16_scalar_traits)

const cpu_softmax_traits_t cpu_softmax_bf16_scalar_traits = {
    .dtype = MARMOT_DTYPE_BFLOAT16,
    .impl_kind = SOFTMAX_IMPL_SCALAR,
    .fn = softmax_bf16_scalar,
    .impl_name = "bf16_scalar",
};
SOFTMAX_REGISTER_TRAITS(cpu_softmax_bf16_scalar_traits)

#if MARMOT_ENABLE_FP8
const cpu_softmax_traits_t cpu_softmax_fp8_e4m3_scalar_traits = {
    .dtype = MARMOT_DTYPE_FLOAT8_E4M3,
    .impl_kind = SOFTMAX_IMPL_SCALAR,
    .fn = softmax_fp8_e4m3_scalar,
    .impl_name = "fp8_e4m3_scalar",
};
SOFTMAX_REGISTER_TRAITS(cpu_softmax_fp8_e4m3_scalar_traits)

const cpu_softmax_traits_t cpu_softmax_fp8_e5m2_scalar_traits = {
    .dtype = MARMOT_DTYPE_FLOAT8_E5M2,
    .impl_kind = SOFTMAX_IMPL_SCALAR,
    .fn = softmax_fp8_e5m2_scalar,
    .impl_name = "fp8_e5m2_scalar",
};
SOFTMAX_REGISTER_TRAITS(cpu_softmax_fp8_e5m2_scalar_traits)
#endif
