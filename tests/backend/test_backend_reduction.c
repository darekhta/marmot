#include <math.h>
#include <string.h>

#include "backend/golden_data.h"
#include "backend/test_backend_utils.h"

[[maybe_unused]] static void
marmot_test_tensor_write_floats(const marmot_test_env_t *env, marmot_tensor_t *tensor, const float *values) {
    const size_t count = marmot_tensor_num_elements(tensor);
    marmot_test_convert_span(env, tensor, MARMOT_DTYPE_FLOAT32, values, count);
}

[[maybe_unused]] static void
marmot_test_tensor_read_floats(const marmot_test_env_t *env, const marmot_tensor_t *tensor, float *out_values) {
    const size_t count = marmot_tensor_num_elements(tensor);
    marmot_test_fetch_span(env, out_values, MARMOT_DTYPE_FLOAT32, (marmot_tensor_t *)tensor, count);
}

static void marmot_test_tensor_write_i32(marmot_tensor_t *tensor, const int32_t *values) {
    const size_t count = marmot_tensor_num_elements(tensor);
    marmot_int32_t *dst = (marmot_int32_t *)tensor->data;
    for (size_t i = 0; i < count; ++i) {
        dst[i].value = values[i];
    }
}

static void marmot_test_tensor_write_u32(marmot_tensor_t *tensor, const uint32_t *values) {
    const size_t count = marmot_tensor_num_elements(tensor);
    marmot_uint32_t *dst = (marmot_uint32_t *)tensor->data;
    for (size_t i = 0; i < count; ++i) {
        dst[i].value = values[i];
    }
}

static void marmot_test_tensor_write_i64(marmot_tensor_t *tensor, const int64_t *values) {
    const size_t count = marmot_tensor_num_elements(tensor);
    marmot_int64_t *dst = (marmot_int64_t *)tensor->data;
    for (size_t i = 0; i < count; ++i) {
        dst[i].value = values[i];
    }
}

static void marmot_test_tensor_write_i16(marmot_tensor_t *tensor, const int16_t *values) {
    const size_t count = marmot_tensor_num_elements(tensor);
    marmot_int16_t *dst = (marmot_int16_t *)tensor->data;
    for (size_t i = 0; i < count; ++i) {
        dst[i].value = values[i];
    }
}

static void marmot_test_tensor_write_u16(marmot_tensor_t *tensor, const uint16_t *values) {
    const size_t count = marmot_tensor_num_elements(tensor);
    marmot_uint16_t *dst = (marmot_uint16_t *)tensor->data;
    for (size_t i = 0; i < count; ++i) {
        dst[i].value = values[i];
    }
}

static void marmot_test_tensor_write_i8(marmot_tensor_t *tensor, const int8_t *values) {
    const size_t count = marmot_tensor_num_elements(tensor);
    marmot_int8_t *dst = (marmot_int8_t *)tensor->data;
    for (size_t i = 0; i < count; ++i) {
        dst[i].value = values[i];
    }
}

static void marmot_test_tensor_write_u8(marmot_tensor_t *tensor, const uint8_t *values) {
    const size_t count = marmot_tensor_num_elements(tensor);
    marmot_uint8_t *dst = (marmot_uint8_t *)tensor->data;
    for (size_t i = 0; i < count; ++i) {
        dst[i].value = values[i];
    }
}

#define DEFINE_SIGNED_REDUCTION_TEST(FN_NAME, DTYPE_ENUM, GOLDEN, WRITE_FN, STRUCT_T, C_T)                             \
    static void FN_NAME(const marmot_test_env_t *env) {                                                                \
        const size_t shape[] = {GOLDEN.shape[0], GOLDEN.shape[1]};                                                     \
        marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape, 2, DTYPE_ENUM);                                 \
        assert_non_null(input);                                                                                        \
        WRITE_FN(input, GOLDEN.input);                                                                                 \
        marmot_test_commit_tensor(env, input);                                                                         \
                                                                                                                       \
        const size_t scalar_shape[] = {1};                                                                             \
        marmot_tensor_t *scalar_out = marmot_tensor_create(env->ctx, scalar_shape, 1, DTYPE_ENUM);                     \
        marmot_tensor_t *indices_scalar = marmot_tensor_create(env->ctx, scalar_shape, 1, MARMOT_DTYPE_UINT64);        \
        assert_non_null(scalar_out);                                                                                   \
        assert_non_null(indices_scalar);                                                                               \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_sum(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}                \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        assert_int_equal(((const STRUCT_T *)marmot_test_tensor_data(env, scalar_out))[0].value, (C_T)GOLDEN.sum_all);  \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_prod(                                                                                        \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}                \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        assert_int_equal(((const STRUCT_T *)marmot_test_tensor_data(env, scalar_out))[0].value, (C_T)GOLDEN.prod_all); \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_max(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}                \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        assert_int_equal(((const STRUCT_T *)marmot_test_tensor_data(env, scalar_out))[0].value, (C_T)GOLDEN.max_all);  \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_min(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}                \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        assert_int_equal(((const STRUCT_T *)marmot_test_tensor_data(env, scalar_out))[0].value, (C_T)GOLDEN.min_all);  \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_argmax(                                                                                      \
                env->ctx,                                                                                              \
                &(marmot_reduction_desc_t){.input = input,                                                             \
                                           .out = scalar_out,                                                          \
                                           .indices_out = indices_scalar,                                              \
                                           .axes = nullptr,                                                            \
                                           .num_axes = 0,                                                              \
                                           .keepdims = false}                                                          \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        assert_int_equal(((const STRUCT_T *)marmot_test_tensor_data(env, scalar_out))[0].value, (C_T)GOLDEN.max_all);  \
        assert_int_equal(                                                                                              \
            ((const uint64_t *)marmot_test_tensor_data(env, indices_scalar))[0], (uint64_t)GOLDEN.argmax_all           \
        );                                                                                                             \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_argmin(                                                                                      \
                env->ctx,                                                                                              \
                &(marmot_reduction_desc_t){.input = input,                                                             \
                                           .out = scalar_out,                                                          \
                                           .indices_out = indices_scalar,                                              \
                                           .axes = nullptr,                                                            \
                                           .num_axes = 0,                                                              \
                                           .keepdims = false}                                                          \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        assert_int_equal(((const STRUCT_T *)marmot_test_tensor_data(env, scalar_out))[0].value, (C_T)GOLDEN.min_all);  \
        assert_int_equal(                                                                                              \
            ((const uint64_t *)marmot_test_tensor_data(env, indices_scalar))[0], (uint64_t)GOLDEN.argmin_all           \
        );                                                                                                             \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_any(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}                \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        assert_int_equal(((const STRUCT_T *)marmot_test_tensor_data(env, scalar_out))[0].value, (C_T)GOLDEN.any_all);  \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_all(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}                \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        assert_int_equal(((const STRUCT_T *)marmot_test_tensor_data(env, scalar_out))[0].value, (C_T)GOLDEN.all_all);  \
                                                                                                                       \
        const int32_t axis0[] = {0};                                                                                   \
        const size_t axis0_shape[] = {GOLDEN.shape[1]};                                                                \
        marmot_tensor_t *axis0_out = marmot_tensor_create(env->ctx, axis0_shape, 1, DTYPE_ENUM);                       \
        marmot_tensor_t *axis0_idx = marmot_tensor_create(env->ctx, axis0_shape, 1, MARMOT_DTYPE_UINT64);              \
        assert_non_null(axis0_out);                                                                                    \
        assert_non_null(axis0_idx);                                                                                    \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_sum(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = axis0_out, .axes = axis0, .num_axes = 1, .keepdims = false}                   \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        const STRUCT_T *axis0_sum = (const STRUCT_T *)marmot_test_tensor_data(env, axis0_out);                         \
        for (size_t i = 0; i < axis0_shape[0]; ++i) {                                                                  \
            assert_int_equal(axis0_sum[i].value, (C_T)GOLDEN.sum_axis0[i]);                                            \
        }                                                                                                              \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_argmax(                                                                                      \
                env->ctx,                                                                                              \
                &(marmot_reduction_desc_t){.input = input,                                                             \
                                           .out = axis0_out,                                                           \
                                           .indices_out = axis0_idx,                                                   \
                                           .axes = axis0,                                                              \
                                           .num_axes = 1,                                                              \
                                           .keepdims = false}                                                          \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        const STRUCT_T *axis0_max = (const STRUCT_T *)marmot_test_tensor_data(env, axis0_out);                         \
        const uint64_t *axis0_indices = (const uint64_t *)marmot_test_tensor_data(env, axis0_idx);                     \
        for (size_t i = 0; i < axis0_shape[0]; ++i) {                                                                  \
            assert_int_equal(axis0_max[i].value, (C_T)GOLDEN.max_axis0[i]);                                            \
            assert_int_equal(axis0_indices[i], (uint64_t)GOLDEN.argmax_axis0[i]);                                      \
        }                                                                                                              \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_argmin(                                                                                      \
                env->ctx,                                                                                              \
                &(marmot_reduction_desc_t){.input = input,                                                             \
                                           .out = axis0_out,                                                           \
                                           .indices_out = axis0_idx,                                                   \
                                           .axes = axis0,                                                              \
                                           .num_axes = 1,                                                              \
                                           .keepdims = false}                                                          \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        const STRUCT_T *axis0_min = (const STRUCT_T *)marmot_test_tensor_data(env, axis0_out);                         \
        axis0_indices = (const uint64_t *)marmot_test_tensor_data(env, axis0_idx);                                     \
        for (size_t i = 0; i < axis0_shape[0]; ++i) {                                                                  \
            assert_int_equal(axis0_min[i].value, (C_T)GOLDEN.min_axis0[i]);                                            \
            assert_int_equal(axis0_indices[i], (uint64_t)GOLDEN.argmin_axis0[i]);                                      \
        }                                                                                                              \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_any(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = axis0_out, .axes = axis0, .num_axes = 1, .keepdims = false}                   \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        const STRUCT_T *axis0_any = (const STRUCT_T *)marmot_test_tensor_data(env, axis0_out);                         \
        for (size_t i = 0; i < axis0_shape[0]; ++i) {                                                                  \
            assert_int_equal(axis0_any[i].value, (C_T)GOLDEN.any_axis0[i]);                                            \
        }                                                                                                              \
                                                                                                                       \
        const int32_t axis1[] = {1};                                                                                   \
        const size_t axis1_shape[] = {GOLDEN.shape[0]};                                                                \
        marmot_tensor_t *axis1_out = marmot_tensor_create(env->ctx, axis1_shape, 1, DTYPE_ENUM);                       \
        marmot_tensor_t *axis1_idx = marmot_tensor_create(env->ctx, axis1_shape, 1, MARMOT_DTYPE_UINT64);              \
        assert_non_null(axis1_out);                                                                                    \
        assert_non_null(axis1_idx);                                                                                    \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_sum(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = axis1_out, .axes = axis1, .num_axes = 1, .keepdims = false}                   \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        const STRUCT_T *axis1_sum = (const STRUCT_T *)marmot_test_tensor_data(env, axis1_out);                         \
        for (size_t i = 0; i < axis1_shape[0]; ++i) {                                                                  \
            assert_int_equal(axis1_sum[i].value, (C_T)GOLDEN.sum_axis1[i]);                                            \
        }                                                                                                              \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_argmax(                                                                                      \
                env->ctx,                                                                                              \
                &(marmot_reduction_desc_t){.input = input,                                                             \
                                           .out = axis1_out,                                                           \
                                           .indices_out = axis1_idx,                                                   \
                                           .axes = axis1,                                                              \
                                           .num_axes = 1,                                                              \
                                           .keepdims = false}                                                          \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        const STRUCT_T *axis1_max = (const STRUCT_T *)marmot_test_tensor_data(env, axis1_out);                         \
        const uint64_t *axis1_indices = (const uint64_t *)marmot_test_tensor_data(env, axis1_idx);                     \
        for (size_t i = 0; i < axis1_shape[0]; ++i) {                                                                  \
            assert_int_equal(axis1_max[i].value, (C_T)GOLDEN.max_axis1[i]);                                            \
            assert_int_equal(axis1_indices[i], (uint64_t)GOLDEN.argmax_axis1[i]);                                      \
        }                                                                                                              \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_argmin(                                                                                      \
                env->ctx,                                                                                              \
                &(marmot_reduction_desc_t){.input = input,                                                             \
                                           .out = axis1_out,                                                           \
                                           .indices_out = axis1_idx,                                                   \
                                           .axes = axis1,                                                              \
                                           .num_axes = 1,                                                              \
                                           .keepdims = false}                                                          \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        const STRUCT_T *axis1_min = (const STRUCT_T *)marmot_test_tensor_data(env, axis1_out);                         \
        axis1_indices = (const uint64_t *)marmot_test_tensor_data(env, axis1_idx);                                     \
        for (size_t i = 0; i < axis1_shape[0]; ++i) {                                                                  \
            assert_int_equal(axis1_min[i].value, (C_T)GOLDEN.min_axis1[i]);                                            \
            assert_int_equal(axis1_indices[i], (uint64_t)GOLDEN.argmin_axis1[i]);                                      \
        }                                                                                                              \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_all(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = axis1_out, .axes = axis1, .num_axes = 1, .keepdims = false}                   \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        const STRUCT_T *axis1_all = (const STRUCT_T *)marmot_test_tensor_data(env, axis1_out);                         \
        for (size_t i = 0; i < axis1_shape[0]; ++i) {                                                                  \
            assert_int_equal(axis1_all[i].value, (C_T)GOLDEN.all_axis1[i]);                                            \
        }                                                                                                              \
                                                                                                                       \
        const size_t axis1_keep_shape[] = {GOLDEN.shape[0], 1};                                                        \
        marmot_tensor_t *axis1_keep = marmot_tensor_create(env->ctx, axis1_keep_shape, 2, DTYPE_ENUM);                 \
        marmot_tensor_t *axis1_keep_idx = marmot_tensor_create(env->ctx, axis1_keep_shape, 2, MARMOT_DTYPE_UINT64);    \
        assert_non_null(axis1_keep);                                                                                   \
        assert_non_null(axis1_keep_idx);                                                                               \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_sum(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = axis1_keep, .axes = axis1, .num_axes = 1, .keepdims = true}                   \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        const STRUCT_T *axis1_keep_sum = (const STRUCT_T *)marmot_test_tensor_data(env, axis1_keep);                   \
        for (size_t i = 0; i < axis1_shape[0]; ++i) {                                                                  \
            assert_int_equal(axis1_keep_sum[i].value, (C_T)GOLDEN.sum_axis1[i]);                                       \
        }                                                                                                              \
        assert_int_equal(axis1_keep->shape.ndim, 2);                                                                   \
        assert_int_equal(axis1_keep->shape.shape[0], GOLDEN.shape[0]);                                                 \
        assert_int_equal(axis1_keep->shape.shape[1], 1);                                                               \
                                                                                                                       \
        marmot_tensor_destroy(axis1_keep_idx);                                                                         \
        marmot_tensor_destroy(axis1_keep);                                                                             \
        marmot_tensor_destroy(axis1_idx);                                                                              \
        marmot_tensor_destroy(axis1_out);                                                                              \
        marmot_tensor_destroy(axis0_idx);                                                                              \
        marmot_tensor_destroy(axis0_out);                                                                              \
        marmot_tensor_destroy(indices_scalar);                                                                         \
        marmot_tensor_destroy(scalar_out);                                                                             \
        marmot_tensor_destroy(input);                                                                                  \
    }

#define DEFINE_UNSIGNED_REDUCTION_TEST(FN_NAME, DTYPE_ENUM, GOLDEN, WRITE_FN, STRUCT_T, C_T)                           \
    static void FN_NAME(const marmot_test_env_t *env) {                                                                \
        const size_t shape[] = {GOLDEN.shape[0], GOLDEN.shape[1]};                                                     \
        marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape, 2, DTYPE_ENUM);                                 \
        assert_non_null(input);                                                                                        \
        WRITE_FN(input, GOLDEN.input);                                                                                 \
        marmot_test_commit_tensor(env, input);                                                                         \
                                                                                                                       \
        const size_t scalar_shape[] = {1};                                                                             \
        marmot_tensor_t *scalar_out = marmot_tensor_create(env->ctx, scalar_shape, 1, DTYPE_ENUM);                     \
        marmot_tensor_t *indices_scalar = marmot_tensor_create(env->ctx, scalar_shape, 1, MARMOT_DTYPE_UINT64);        \
        assert_non_null(scalar_out);                                                                                   \
        assert_non_null(indices_scalar);                                                                               \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_sum(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}                \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        assert_int_equal(((const STRUCT_T *)marmot_test_tensor_data(env, scalar_out))[0].value, (C_T)GOLDEN.sum_all);  \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_prod(                                                                                        \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}                \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        assert_int_equal(((const STRUCT_T *)marmot_test_tensor_data(env, scalar_out))[0].value, (C_T)GOLDEN.prod_all); \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_max(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}                \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        assert_int_equal(((const STRUCT_T *)marmot_test_tensor_data(env, scalar_out))[0].value, (C_T)GOLDEN.max_all);  \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_min(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}                \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        assert_int_equal(((const STRUCT_T *)marmot_test_tensor_data(env, scalar_out))[0].value, (C_T)GOLDEN.min_all);  \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_argmax(                                                                                      \
                env->ctx,                                                                                              \
                &(marmot_reduction_desc_t){.input = input,                                                             \
                                           .out = scalar_out,                                                          \
                                           .indices_out = indices_scalar,                                              \
                                           .axes = nullptr,                                                            \
                                           .num_axes = 0,                                                              \
                                           .keepdims = false}                                                          \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        assert_int_equal(((const STRUCT_T *)marmot_test_tensor_data(env, scalar_out))[0].value, (C_T)GOLDEN.max_all);  \
        assert_int_equal(                                                                                              \
            ((const uint64_t *)marmot_test_tensor_data(env, indices_scalar))[0], (uint64_t)GOLDEN.argmax_all           \
        );                                                                                                             \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_argmin(                                                                                      \
                env->ctx,                                                                                              \
                &(marmot_reduction_desc_t){.input = input,                                                             \
                                           .out = scalar_out,                                                          \
                                           .indices_out = indices_scalar,                                              \
                                           .axes = nullptr,                                                            \
                                           .num_axes = 0,                                                              \
                                           .keepdims = false}                                                          \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        assert_int_equal(((const STRUCT_T *)marmot_test_tensor_data(env, scalar_out))[0].value, (C_T)GOLDEN.min_all);  \
        assert_int_equal(                                                                                              \
            ((const uint64_t *)marmot_test_tensor_data(env, indices_scalar))[0], (uint64_t)GOLDEN.argmin_all           \
        );                                                                                                             \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_any(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}                \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        assert_int_equal(((const STRUCT_T *)marmot_test_tensor_data(env, scalar_out))[0].value, (C_T)GOLDEN.any_all);  \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_all(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}                \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        assert_int_equal(((const STRUCT_T *)marmot_test_tensor_data(env, scalar_out))[0].value, (C_T)GOLDEN.all_all);  \
                                                                                                                       \
        const int32_t axis0[] = {0};                                                                                   \
        const size_t axis0_shape[] = {GOLDEN.shape[1]};                                                                \
        marmot_tensor_t *axis0_out = marmot_tensor_create(env->ctx, axis0_shape, 1, DTYPE_ENUM);                       \
        marmot_tensor_t *axis0_idx = marmot_tensor_create(env->ctx, axis0_shape, 1, MARMOT_DTYPE_UINT64);              \
        assert_non_null(axis0_out);                                                                                    \
        assert_non_null(axis0_idx);                                                                                    \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_sum(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = axis0_out, .axes = axis0, .num_axes = 1, .keepdims = false}                   \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        const STRUCT_T *axis0_sum = (const STRUCT_T *)marmot_test_tensor_data(env, axis0_out);                         \
        for (size_t i = 0; i < axis0_shape[0]; ++i) {                                                                  \
            assert_int_equal(axis0_sum[i].value, (C_T)GOLDEN.sum_axis0[i]);                                            \
        }                                                                                                              \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_argmax(                                                                                      \
                env->ctx,                                                                                              \
                &(marmot_reduction_desc_t){.input = input,                                                             \
                                           .out = axis0_out,                                                           \
                                           .indices_out = axis0_idx,                                                   \
                                           .axes = axis0,                                                              \
                                           .num_axes = 1,                                                              \
                                           .keepdims = false}                                                          \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        const STRUCT_T *axis0_max = (const STRUCT_T *)marmot_test_tensor_data(env, axis0_out);                         \
        const uint64_t *axis0_indices = (const uint64_t *)marmot_test_tensor_data(env, axis0_idx);                     \
        for (size_t i = 0; i < axis0_shape[0]; ++i) {                                                                  \
            assert_int_equal(axis0_max[i].value, (C_T)GOLDEN.max_axis0[i]);                                            \
            assert_int_equal(axis0_indices[i], (uint64_t)GOLDEN.argmax_axis0[i]);                                      \
        }                                                                                                              \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_argmin(                                                                                      \
                env->ctx,                                                                                              \
                &(marmot_reduction_desc_t){.input = input,                                                             \
                                           .out = axis0_out,                                                           \
                                           .indices_out = axis0_idx,                                                   \
                                           .axes = axis0,                                                              \
                                           .num_axes = 1,                                                              \
                                           .keepdims = false}                                                          \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        const STRUCT_T *axis0_min = (const STRUCT_T *)marmot_test_tensor_data(env, axis0_out);                         \
        axis0_indices = (const uint64_t *)marmot_test_tensor_data(env, axis0_idx);                                     \
        for (size_t i = 0; i < axis0_shape[0]; ++i) {                                                                  \
            assert_int_equal(axis0_min[i].value, (C_T)GOLDEN.min_axis0[i]);                                            \
            assert_int_equal(axis0_indices[i], (uint64_t)GOLDEN.argmin_axis0[i]);                                      \
        }                                                                                                              \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_any(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = axis0_out, .axes = axis0, .num_axes = 1, .keepdims = false}                   \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        const STRUCT_T *axis0_any = (const STRUCT_T *)marmot_test_tensor_data(env, axis0_out);                         \
        for (size_t i = 0; i < axis0_shape[0]; ++i) {                                                                  \
            assert_int_equal(axis0_any[i].value, (C_T)GOLDEN.any_axis0[i]);                                            \
        }                                                                                                              \
                                                                                                                       \
        const int32_t axis1[] = {1};                                                                                   \
        const size_t axis1_shape[] = {GOLDEN.shape[0]};                                                                \
        marmot_tensor_t *axis1_out = marmot_tensor_create(env->ctx, axis1_shape, 1, DTYPE_ENUM);                       \
        marmot_tensor_t *axis1_idx = marmot_tensor_create(env->ctx, axis1_shape, 1, MARMOT_DTYPE_UINT64);              \
        assert_non_null(axis1_out);                                                                                    \
        assert_non_null(axis1_idx);                                                                                    \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_sum(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = axis1_out, .axes = axis1, .num_axes = 1, .keepdims = false}                   \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        const STRUCT_T *axis1_sum = (const STRUCT_T *)marmot_test_tensor_data(env, axis1_out);                         \
        for (size_t i = 0; i < axis1_shape[0]; ++i) {                                                                  \
            assert_int_equal(axis1_sum[i].value, (C_T)GOLDEN.sum_axis1[i]);                                            \
        }                                                                                                              \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_argmax(                                                                                      \
                env->ctx,                                                                                              \
                &(marmot_reduction_desc_t){.input = input,                                                             \
                                           .out = axis1_out,                                                           \
                                           .indices_out = axis1_idx,                                                   \
                                           .axes = axis1,                                                              \
                                           .num_axes = 1,                                                              \
                                           .keepdims = false}                                                          \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        const STRUCT_T *axis1_max = (const STRUCT_T *)marmot_test_tensor_data(env, axis1_out);                         \
        const uint64_t *axis1_indices = (const uint64_t *)marmot_test_tensor_data(env, axis1_idx);                     \
        for (size_t i = 0; i < axis1_shape[0]; ++i) {                                                                  \
            assert_int_equal(axis1_max[i].value, (C_T)GOLDEN.max_axis1[i]);                                            \
            assert_int_equal(axis1_indices[i], (uint64_t)GOLDEN.argmax_axis1[i]);                                      \
        }                                                                                                              \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_argmin(                                                                                      \
                env->ctx,                                                                                              \
                &(marmot_reduction_desc_t){.input = input,                                                             \
                                           .out = axis1_out,                                                           \
                                           .indices_out = axis1_idx,                                                   \
                                           .axes = axis1,                                                              \
                                           .num_axes = 1,                                                              \
                                           .keepdims = false}                                                          \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        const STRUCT_T *axis1_min = (const STRUCT_T *)marmot_test_tensor_data(env, axis1_out);                         \
        axis1_indices = (const uint64_t *)marmot_test_tensor_data(env, axis1_idx);                                     \
        for (size_t i = 0; i < axis1_shape[0]; ++i) {                                                                  \
            assert_int_equal(axis1_min[i].value, (C_T)GOLDEN.min_axis1[i]);                                            \
            assert_int_equal(axis1_indices[i], (uint64_t)GOLDEN.argmin_axis1[i]);                                      \
        }                                                                                                              \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_all(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = axis1_out, .axes = axis1, .num_axes = 1, .keepdims = false}                   \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        const STRUCT_T *axis1_all = (const STRUCT_T *)marmot_test_tensor_data(env, axis1_out);                         \
        for (size_t i = 0; i < axis1_shape[0]; ++i) {                                                                  \
            assert_int_equal(axis1_all[i].value, (C_T)GOLDEN.all_axis1[i]);                                            \
        }                                                                                                              \
                                                                                                                       \
        const size_t axis1_keep_shape[] = {GOLDEN.shape[0], 1};                                                        \
        marmot_tensor_t *axis1_keep = marmot_tensor_create(env->ctx, axis1_keep_shape, 2, DTYPE_ENUM);                 \
        marmot_tensor_t *axis1_keep_idx = marmot_tensor_create(env->ctx, axis1_keep_shape, 2, MARMOT_DTYPE_UINT64);    \
        assert_non_null(axis1_keep);                                                                                   \
        assert_non_null(axis1_keep_idx);                                                                               \
                                                                                                                       \
        assert_int_equal(                                                                                              \
            marmot_reduce_sum(                                                                                         \
                env->ctx,                                                                                              \
                &(                                                                                                     \
                    marmot_reduction_desc_t                                                                            \
                ){.input = input, .out = axis1_keep, .axes = axis1, .num_axes = 1, .keepdims = true}                   \
            ),                                                                                                         \
            MARMOT_SUCCESS                                                                                             \
        );                                                                                                             \
        const STRUCT_T *axis1_keep_sum = (const STRUCT_T *)marmot_test_tensor_data(env, axis1_keep);                   \
        for (size_t i = 0; i < axis1_shape[0]; ++i) {                                                                  \
            assert_int_equal(axis1_keep_sum[i].value, (C_T)GOLDEN.sum_axis1[i]);                                       \
        }                                                                                                              \
        assert_int_equal(axis1_keep->shape.ndim, 2);                                                                   \
        assert_int_equal(axis1_keep->shape.shape[0], GOLDEN.shape[0]);                                                 \
        assert_int_equal(axis1_keep->shape.shape[1], 1);                                                               \
                                                                                                                       \
        marmot_tensor_destroy(axis1_keep_idx);                                                                         \
        marmot_tensor_destroy(axis1_keep);                                                                             \
        marmot_tensor_destroy(axis1_idx);                                                                              \
        marmot_tensor_destroy(axis1_out);                                                                              \
        marmot_tensor_destroy(axis0_idx);                                                                              \
        marmot_tensor_destroy(axis0_out);                                                                              \
        marmot_tensor_destroy(indices_scalar);                                                                         \
        marmot_tensor_destroy(scalar_out);                                                                             \
        marmot_tensor_destroy(input);                                                                                  \
    }

DEFINE_SIGNED_REDUCTION_TEST(
    check_reduce_int32, MARMOT_DTYPE_INT32, g_reduction_i32, marmot_test_tensor_write_i32, marmot_int32_t, int32_t
)
DEFINE_SIGNED_REDUCTION_TEST(
    check_reduce_int16, MARMOT_DTYPE_INT16, g_reduction_i16, marmot_test_tensor_write_i16, marmot_int16_t, int16_t
)
DEFINE_SIGNED_REDUCTION_TEST(
    check_reduce_int8, MARMOT_DTYPE_INT8, g_reduction_i8, marmot_test_tensor_write_i8, marmot_int8_t, int8_t
)

DEFINE_UNSIGNED_REDUCTION_TEST(
    check_reduce_uint32, MARMOT_DTYPE_UINT32, g_reduction_u32, marmot_test_tensor_write_u32, marmot_uint32_t, uint32_t
)
DEFINE_UNSIGNED_REDUCTION_TEST(
    check_reduce_uint16, MARMOT_DTYPE_UINT16, g_reduction_u16, marmot_test_tensor_write_u16, marmot_uint16_t, uint16_t
)
DEFINE_UNSIGNED_REDUCTION_TEST(
    check_reduce_uint8, MARMOT_DTYPE_UINT8, g_reduction_u8, marmot_test_tensor_write_u8, marmot_uint8_t, uint8_t
)

static void marmot_tensor_write_floats_ctx(const marmot_context_t *ctx, marmot_tensor_t *tensor, const float *values) {
    marmot_test_env_t env = {.backend = ctx->backend_type, .ctx = (marmot_context_t *)ctx};
    const size_t count = marmot_tensor_num_elements(tensor);
    marmot_test_convert_span(&env, tensor, MARMOT_DTYPE_FLOAT32, values, count);
}

static void
marmot_tensor_read_floats_ctx(const marmot_context_t *ctx, const marmot_tensor_t *tensor, float *out_values) {
    marmot_test_env_t env = {.backend = ctx->backend_type, .ctx = (marmot_context_t *)ctx};
    const size_t count = marmot_tensor_num_elements(tensor);
    marmot_test_fetch_span(&env, out_values, MARMOT_DTYPE_FLOAT32, (marmot_tensor_t *)tensor, count);
}

static float marmot_reduction_value_tol(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT64:
        return 1e-9f;
    case MARMOT_DTYPE_FLOAT16:
        return 2e-2f;
    case MARMOT_DTYPE_BFLOAT16:
        return 7e-2f;
    default:
        return 1e-4f;
    }
}

static size_t marmot_compute_reduction_shape(
    const marmot_tensor_t *input, const int32_t *axes, size_t num_axes, bool keepdims, size_t *out_shape
) {
    const size_t ndim = input->shape.ndim;
    bool reduce_mask[MARMOT_MAX_DIMS] = {false};

    if (num_axes == 0) {
        for (size_t i = 0; i < ndim; ++i) {
            reduce_mask[i] = true;
        }
    } else {
        for (size_t i = 0; i < num_axes; ++i) {
            int32_t axis = axes[i];
            if (axis < 0) {
                axis += (int32_t)ndim;
            }
            assert_true(axis >= 0 && axis < (int32_t)ndim);
            reduce_mask[axis] = true;
        }
    }

    if (keepdims) {
        for (size_t i = 0; i < ndim; ++i) {
            out_shape[i] = reduce_mask[i] ? 1 : input->shape.shape[i];
        }
        return ndim;
    }

    size_t out_ndim = 0;
    for (size_t i = 0; i < ndim; ++i) {
        if (!reduce_mask[i]) {
            out_shape[out_ndim++] = input->shape.shape[i];
        }
    }

    if (out_ndim == 0) {
        out_shape[0] = 1;
        return 1;
    }

    return out_ndim;
}

static void marmot_assert_allclose(const float *a, const float *b, size_t count, float tol, const char *label) {
    for (size_t i = 0; i < count; ++i) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > tol) {
            printf("%s mismatch at %zu: a=%f b=%f tol=%f\n", label, i, a[i], b[i], tol);
        }
        assert_true(diff <= tol);
    }
}

static void marmot_assert_indices_equal(
    const marmot_uint64_t *cpu_indices, const marmot_uint64_t *metal_indices, size_t count, const char *label
) {
    for (size_t i = 0; i < count; ++i) {
        if (cpu_indices[i].value != metal_indices[i].value) {
            printf(
                "%s index mismatch at %zu: cpu=%llu metal=%llu\n", label, i, (unsigned long long)cpu_indices[i].value,
                (unsigned long long)metal_indices[i].value
            );
        }
        assert_int_equal((int)cpu_indices[i].value, (int)metal_indices[i].value);
    }
}

static void marmot_run_reduction_value_op(
    const char *label, marmot_error_t (*op)(const marmot_context_t *, const marmot_reduction_desc_t *),
    const marmot_context_t *cpu_ctx, const marmot_context_t *metal_ctx, const marmot_tensor_t *cpu_input,
    const marmot_tensor_t *metal_input, marmot_dtype_t dtype, const int32_t *axes, size_t num_axes, bool keepdims,
    const size_t *out_shape, size_t out_ndim, float tol
) {
    marmot_tensor_t *cpu_out = marmot_tensor_create(cpu_ctx, out_shape, out_ndim, dtype);
    marmot_tensor_t *metal_out = marmot_tensor_create(metal_ctx, out_shape, out_ndim, dtype);
    assert_non_null(cpu_out);
    assert_non_null(metal_out);

    assert_int_equal(
        op(cpu_ctx,
           &(
               marmot_reduction_desc_t
           ){.input = cpu_input, .out = cpu_out, .axes = axes, .num_axes = num_axes, .keepdims = keepdims}),
        MARMOT_SUCCESS
    );
    assert_int_equal(
        op(metal_ctx,
           &(
               marmot_reduction_desc_t
           ){.input = metal_input, .out = metal_out, .axes = axes, .num_axes = num_axes, .keepdims = keepdims}),
        MARMOT_SUCCESS
    );

    const size_t count = marmot_tensor_num_elements(cpu_out);
    float *cpu_vals = (float *)malloc(count * sizeof(float));
    float *metal_vals = (float *)malloc(count * sizeof(float));
    assert_non_null(cpu_vals);
    assert_non_null(metal_vals);

    marmot_tensor_read_floats_ctx(cpu_ctx, cpu_out, cpu_vals);
    marmot_tensor_read_floats_ctx(metal_ctx, metal_out, metal_vals);
    marmot_assert_allclose(cpu_vals, metal_vals, count, tol, label);

    free(metal_vals);
    free(cpu_vals);
    marmot_tensor_destroy(metal_out);
    marmot_tensor_destroy(cpu_out);
}

static void marmot_run_reduction_variance_op(
    const char *label, marmot_error_t (*op)(const marmot_context_t *, const marmot_reduction_desc_t *),
    const marmot_context_t *cpu_ctx, const marmot_context_t *metal_ctx, const marmot_tensor_t *cpu_input,
    const marmot_tensor_t *metal_input, marmot_dtype_t dtype, const int32_t *axes, size_t num_axes, bool keepdims,
    const size_t *out_shape, size_t out_ndim, float tol
) {
    marmot_tensor_t *cpu_out = marmot_tensor_create(cpu_ctx, out_shape, out_ndim, dtype);
    marmot_tensor_t *metal_out = marmot_tensor_create(metal_ctx, out_shape, out_ndim, dtype);
    assert_non_null(cpu_out);
    assert_non_null(metal_out);

    assert_int_equal(
        op(cpu_ctx,
           &(marmot_reduction_desc_t){.input = cpu_input,
                                      .out = cpu_out,
                                      .axes = axes,
                                      .num_axes = num_axes,
                                      .keepdims = keepdims,
                                      .unbiased = false,
                                      .epsilon = 0.0f}),
        MARMOT_SUCCESS
    );
    assert_int_equal(
        op(metal_ctx,
           &(marmot_reduction_desc_t){.input = metal_input,
                                      .out = metal_out,
                                      .axes = axes,
                                      .num_axes = num_axes,
                                      .keepdims = keepdims,
                                      .unbiased = false,
                                      .epsilon = 0.0f}),
        MARMOT_SUCCESS
    );

    const size_t count = marmot_tensor_num_elements(cpu_out);
    float *cpu_vals = (float *)malloc(count * sizeof(float));
    float *metal_vals = (float *)malloc(count * sizeof(float));
    assert_non_null(cpu_vals);
    assert_non_null(metal_vals);

    marmot_tensor_read_floats_ctx(cpu_ctx, cpu_out, cpu_vals);
    marmot_tensor_read_floats_ctx(metal_ctx, metal_out, metal_vals);
    marmot_assert_allclose(cpu_vals, metal_vals, count, tol, label);

    free(metal_vals);
    free(cpu_vals);
    marmot_tensor_destroy(metal_out);
    marmot_tensor_destroy(cpu_out);
}

static void marmot_run_reduction_arg_op(
    const char *label, marmot_error_t (*op)(const marmot_context_t *, const marmot_reduction_desc_t *),
    const marmot_context_t *cpu_ctx, const marmot_context_t *metal_ctx, const marmot_tensor_t *cpu_input,
    const marmot_tensor_t *metal_input, marmot_dtype_t dtype, const int32_t *axes, size_t num_axes, bool keepdims,
    const size_t *out_shape, size_t out_ndim, float tol
) {
    marmot_tensor_t *cpu_vals = marmot_tensor_create(cpu_ctx, out_shape, out_ndim, dtype);
    marmot_tensor_t *metal_vals = marmot_tensor_create(metal_ctx, out_shape, out_ndim, dtype);
    marmot_tensor_t *cpu_idx = marmot_tensor_create(cpu_ctx, out_shape, out_ndim, MARMOT_DTYPE_UINT64);
    marmot_tensor_t *metal_idx = marmot_tensor_create(metal_ctx, out_shape, out_ndim, MARMOT_DTYPE_UINT64);
    assert_non_null(cpu_vals);
    assert_non_null(metal_vals);
    assert_non_null(cpu_idx);
    assert_non_null(metal_idx);

    assert_int_equal(
        op(cpu_ctx,
           &(marmot_reduction_desc_t){.input = cpu_input,
                                      .out = cpu_vals,
                                      .indices_out = cpu_idx,
                                      .axes = axes,
                                      .num_axes = num_axes,
                                      .keepdims = keepdims}),
        MARMOT_SUCCESS
    );
    assert_int_equal(
        op(metal_ctx,
           &(marmot_reduction_desc_t){.input = metal_input,
                                      .out = metal_vals,
                                      .indices_out = metal_idx,
                                      .axes = axes,
                                      .num_axes = num_axes,
                                      .keepdims = keepdims}),
        MARMOT_SUCCESS
    );

    const size_t count = marmot_tensor_num_elements(cpu_vals);
    float *cpu_vals_buf = (float *)malloc(count * sizeof(float));
    float *metal_vals_buf = (float *)malloc(count * sizeof(float));
    assert_non_null(cpu_vals_buf);
    assert_non_null(metal_vals_buf);

    marmot_tensor_read_floats_ctx(cpu_ctx, cpu_vals, cpu_vals_buf);
    marmot_tensor_read_floats_ctx(metal_ctx, metal_vals, metal_vals_buf);
    marmot_assert_allclose(cpu_vals_buf, metal_vals_buf, count, tol, label);

    const marmot_uint64_t *cpu_indices = marmot_tensor_data_u64(cpu_ctx, cpu_idx);
    const marmot_uint64_t *metal_indices = marmot_tensor_data_u64(metal_ctx, metal_idx);
    assert_non_null(cpu_indices);
    assert_non_null(metal_indices);
    marmot_assert_indices_equal(cpu_indices, metal_indices, count, label);

    free(metal_vals_buf);
    free(cpu_vals_buf);
    marmot_tensor_destroy(metal_idx);
    marmot_tensor_destroy(cpu_idx);
    marmot_tensor_destroy(metal_vals);
    marmot_tensor_destroy(cpu_vals);
}

static void marmot_compare_reduction_case(
    const marmot_context_t *cpu_ctx, const marmot_context_t *metal_ctx, const marmot_tensor_t *cpu_input,
    const marmot_tensor_t *metal_input, marmot_dtype_t dtype, const int32_t *axes, size_t num_axes, bool keepdims
) {
    size_t out_shape[MARMOT_MAX_DIMS] = {0};
    const size_t out_ndim = marmot_compute_reduction_shape(cpu_input, axes, num_axes, keepdims, out_shape);
    const float base_tol = marmot_reduction_value_tol(dtype);
    const float prod_tol = base_tol * 3.0f + 1e-5f;
    const float variance_tol = base_tol * 10.0f + 1e-4f;
    const float std_tol = base_tol * 5.0f + 1e-4f;
    const float norm_tol = base_tol * 2.5f + 1e-4f;

    marmot_run_reduction_value_op(
        "sum", marmot_reduce_sum, cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axes, num_axes, keepdims,
        out_shape, out_ndim, base_tol
    );
    marmot_run_reduction_value_op(
        "mean", marmot_reduce_mean, cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axes, num_axes, keepdims,
        out_shape, out_ndim, base_tol
    );
    marmot_run_reduction_value_op(
        "prod", marmot_reduce_prod, cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axes, num_axes, keepdims,
        out_shape, out_ndim, prod_tol
    );
    marmot_run_reduction_value_op(
        "max", marmot_reduce_max, cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axes, num_axes, keepdims,
        out_shape, out_ndim, base_tol
    );
    marmot_run_reduction_value_op(
        "min", marmot_reduce_min, cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axes, num_axes, keepdims,
        out_shape, out_ndim, base_tol
    );
    marmot_run_reduction_arg_op(
        "argmax", marmot_reduce_argmax, cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axes, num_axes, keepdims,
        out_shape, out_ndim, base_tol
    );
    marmot_run_reduction_arg_op(
        "argmin", marmot_reduce_argmin, cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axes, num_axes, keepdims,
        out_shape, out_ndim, base_tol
    );
    marmot_run_reduction_value_op(
        "any", marmot_reduce_any, cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axes, num_axes, keepdims,
        out_shape, out_ndim, base_tol
    );
    marmot_run_reduction_value_op(
        "all", marmot_reduce_all, cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axes, num_axes, keepdims,
        out_shape, out_ndim, base_tol
    );
    marmot_run_reduction_variance_op(
        "variance", marmot_reduce_variance, cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axes, num_axes, keepdims,
        out_shape, out_ndim, variance_tol
    );
    marmot_run_reduction_variance_op(
        "std", marmot_reduce_std, cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axes, num_axes, keepdims,
        out_shape, out_ndim, std_tol
    );
    marmot_run_reduction_value_op(
        "norm_l1", marmot_reduce_norm_l1, cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axes, num_axes, keepdims,
        out_shape, out_ndim, norm_tol
    );
    marmot_run_reduction_value_op(
        "norm_l2", marmot_reduce_norm_l2, cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axes, num_axes, keepdims,
        out_shape, out_ndim, norm_tol
    );
}

static size_t marmot_tensor_element_count(const size_t *shape, size_t ndim) {
    size_t total = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total *= shape[i];
    }
    return total;
}

static void marmot_fill_random_values(uint32_t *state, float *values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        *state = (*state * 1664525u) + 1013904223u;
        float v = (float)(*state) / (float)UINT32_MAX;
        v = v * 2.0f - 1.0f;
        if ((i % 17) == 0) {
            v = 0.0f;
        }
        values[i] = v * 0.75f;
    }
}

static void check_reduce_float(const marmot_test_env_t *env, marmot_dtype_t dtype, float tol) {
    const size_t shape[] = {g_reduction_fp32.shape[0], g_reduction_fp32.shape[1]};
    marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape, 2, dtype);
    assert_non_null(input);
    const bool use_f64 = (dtype == MARMOT_DTYPE_FLOAT64);
    const marmot_dtype_t golden_dtype = use_f64 ? MARMOT_DTYPE_FLOAT64 : MARMOT_DTYPE_FLOAT32;
    const size_t elem_count = shape[0] * shape[1];
    const void *input_src = use_f64 ? (const void *)g_reduction_fp32.input_f64 : (const void *)g_reduction_fp32.input;
    marmot_test_convert_span(env, input, golden_dtype, input_src, elem_count);

    const size_t scalar_shape[] = {1};
    marmot_tensor_t *scalar_out = marmot_tensor_create(env->ctx, scalar_shape, 1, dtype);
    assert_non_null(scalar_out);
    marmot_tensor_t *indices_scalar = marmot_tensor_create(env->ctx, scalar_shape, 1, MARMOT_DTYPE_UINT64);
    assert_non_null(indices_scalar);

    double buffer[8] = {0.0};

#define EXPECT_SCALAR(field)                                                                                           \
    do {                                                                                                               \
        marmot_test_fetch_span(env, buffer, MARMOT_DTYPE_FLOAT64, scalar_out, 1);                                      \
        double expected = g_reduction_fp32.field##_f64;                                                                \
        double diff = fabs(buffer[0] - expected);                                                                      \
        if (diff > (double)tol) {                                                                                      \
            printf(                                                                                                    \
                "%s mismatch (dtype=%d): got=%f expected=%f tol=%f\n", #field, (int)dtype, buffer[0], expected,        \
                (double)tol                                                                                            \
            );                                                                                                         \
        }                                                                                                              \
        assert_true(diff <= (double)tol);                                                                              \
    } while (0)

#define EXPECT_ARRAY(field, tensor_ref, count)                                                                         \
    do {                                                                                                               \
        marmot_test_fetch_span(env, buffer, MARMOT_DTYPE_FLOAT64, tensor_ref, count);                                  \
        const double *expected = g_reduction_fp32.field##_f64;                                                         \
        for (size_t idx__ = 0; idx__ < (count); ++idx__) {                                                             \
            double diff__ = fabs(buffer[idx__] - expected[idx__]);                                                     \
            if (diff__ > (double)tol) {                                                                                \
                printf(                                                                                                \
                    "%s[%zu] mismatch (dtype=%d): got=%f expected=%f tol=%f\n", #field, idx__, (int)dtype,             \
                    buffer[idx__], expected[idx__], (double)tol                                                        \
                );                                                                                                     \
            }                                                                                                          \
            assert_true(diff__ <= (double)tol);                                                                        \
        }                                                                                                              \
    } while (0)

    assert_int_equal(
        marmot_reduce_sum(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_SCALAR(sum_all);

    assert_int_equal(
        marmot_reduce_mean(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_SCALAR(mean_all);

    assert_int_equal(
        marmot_reduce_prod(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_SCALAR(prod_all);

    assert_int_equal(
        marmot_reduce_max(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_SCALAR(max_all);

    assert_int_equal(
        marmot_reduce_min(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_SCALAR(min_all);

    assert_int_equal(
        marmot_reduce_argmax(
            env->ctx,
            &(marmot_reduction_desc_t){.input = input,
                                       .out = scalar_out,
                                       .indices_out = indices_scalar,
                                       .axes = nullptr,
                                       .num_axes = 0,
                                       .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_SCALAR(max_all);
    assert_int_equal(((const uint64_t *)marmot_test_tensor_data(env, indices_scalar))[0], g_reduction_fp32.argmax_all);

    assert_int_equal(
        marmot_reduce_argmin(
            env->ctx,
            &(marmot_reduction_desc_t){.input = input,
                                       .out = scalar_out,
                                       .indices_out = indices_scalar,
                                       .axes = nullptr,
                                       .num_axes = 0,
                                       .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_SCALAR(min_all);
    assert_int_equal(((const uint64_t *)marmot_test_tensor_data(env, indices_scalar))[0], g_reduction_fp32.argmin_all);

    assert_int_equal(
        marmot_reduce_any(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_SCALAR(any_all);

    assert_int_equal(
        marmot_reduce_all(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_SCALAR(all_all);

    assert_int_equal(
        marmot_reduce_variance(
            env->ctx,
            &(marmot_reduction_desc_t){.input = input,
                                       .out = scalar_out,
                                       .axes = nullptr,
                                       .num_axes = 0,
                                       .keepdims = false,
                                       .unbiased = false,
                                       .epsilon = 0.0f}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_SCALAR(variance_all);

    assert_int_equal(
        marmot_reduce_std(
            env->ctx,
            &(marmot_reduction_desc_t){.input = input,
                                       .out = scalar_out,
                                       .axes = nullptr,
                                       .num_axes = 0,
                                       .keepdims = false,
                                       .unbiased = false,
                                       .epsilon = 0.0f}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_SCALAR(std_all);

    assert_int_equal(
        marmot_reduce_norm_l1(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_SCALAR(norm_l1_all);

    assert_int_equal(
        marmot_reduce_norm_l2(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_SCALAR(norm_l2_all);

    const int32_t axis0[] = {0};
    const size_t axis0_shape[] = {g_reduction_fp32.shape[1]};
    marmot_tensor_t *axis0_out = marmot_tensor_create(env->ctx, axis0_shape, 1, dtype);
    marmot_tensor_t *axis0_idx = marmot_tensor_create(env->ctx, axis0_shape, 1, MARMOT_DTYPE_UINT64);
    assert_non_null(axis0_out);
    assert_non_null(axis0_idx);

    assert_int_equal(
        marmot_reduce_sum(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = axis0_out, .axes = axis0, .num_axes = 1, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_ARRAY(sum_axis0, axis0_out, axis0_shape[0]);

    assert_int_equal(
        marmot_reduce_mean(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = axis0_out, .axes = axis0, .num_axes = 1, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_ARRAY(mean_axis0, axis0_out, axis0_shape[0]);

    assert_int_equal(
        marmot_reduce_max(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = axis0_out, .axes = axis0, .num_axes = 1, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_ARRAY(max_axis0, axis0_out, axis0_shape[0]);

    assert_int_equal(
        marmot_reduce_min(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = axis0_out, .axes = axis0, .num_axes = 1, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_ARRAY(min_axis0, axis0_out, axis0_shape[0]);

    assert_int_equal(
        marmot_reduce_argmax(
            env->ctx,
            &(marmot_reduction_desc_t){.input = input,
                                       .out = axis0_out,
                                       .indices_out = axis0_idx,
                                       .axes = axis0,
                                       .num_axes = 1,
                                       .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    const uint64_t *axis0_indices = (const uint64_t *)marmot_test_tensor_data(env, axis0_idx);
    for (size_t i = 0; i < axis0_shape[0]; ++i) {
        assert_int_equal(axis0_indices[i], g_reduction_fp32.argmax_axis0[i]);
    }

    assert_int_equal(
        marmot_reduce_argmin(
            env->ctx,
            &(marmot_reduction_desc_t){.input = input,
                                       .out = axis0_out,
                                       .indices_out = axis0_idx,
                                       .axes = axis0,
                                       .num_axes = 1,
                                       .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    axis0_indices = (const uint64_t *)marmot_test_tensor_data(env, axis0_idx);
    for (size_t i = 0; i < axis0_shape[0]; ++i) {
        assert_int_equal(axis0_indices[i], g_reduction_fp32.argmin_axis0[i]);
    }

    assert_int_equal(
        marmot_reduce_any(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = axis0_out, .axes = axis0, .num_axes = 1, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_ARRAY(any_axis0, axis0_out, axis0_shape[0]);

    const int32_t axis1[] = {1};
    const size_t axis1_shape[] = {g_reduction_fp32.shape[0]};
    marmot_tensor_t *axis1_out = marmot_tensor_create(env->ctx, axis1_shape, 1, dtype);
    marmot_tensor_t *axis1_idx = marmot_tensor_create(env->ctx, axis1_shape, 1, MARMOT_DTYPE_UINT64);
    assert_non_null(axis1_out);
    assert_non_null(axis1_idx);

    assert_int_equal(
        marmot_reduce_sum(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = axis1_out, .axes = axis1, .num_axes = 1, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_ARRAY(sum_axis1, axis1_out, axis1_shape[0]);

    assert_int_equal(
        marmot_reduce_mean(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = axis1_out, .axes = axis1, .num_axes = 1, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_ARRAY(mean_axis1, axis1_out, axis1_shape[0]);

    assert_int_equal(
        marmot_reduce_max(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = axis1_out, .axes = axis1, .num_axes = 1, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_ARRAY(max_axis1, axis1_out, axis1_shape[0]);

    assert_int_equal(
        marmot_reduce_min(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = axis1_out, .axes = axis1, .num_axes = 1, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_ARRAY(min_axis1, axis1_out, axis1_shape[0]);

    assert_int_equal(
        marmot_reduce_argmax(
            env->ctx,
            &(marmot_reduction_desc_t){.input = input,
                                       .out = axis1_out,
                                       .indices_out = axis1_idx,
                                       .axes = axis1,
                                       .num_axes = 1,
                                       .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    const uint64_t *axis1_indices = (const uint64_t *)marmot_test_tensor_data(env, axis1_idx);
    for (size_t i = 0; i < axis1_shape[0]; ++i) {
        assert_int_equal(axis1_indices[i], g_reduction_fp32.argmax_axis1[i]);
    }

    assert_int_equal(
        marmot_reduce_argmin(
            env->ctx,
            &(marmot_reduction_desc_t){.input = input,
                                       .out = axis1_out,
                                       .indices_out = axis1_idx,
                                       .axes = axis1,
                                       .num_axes = 1,
                                       .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    axis1_indices = (const uint64_t *)marmot_test_tensor_data(env, axis1_idx);
    for (size_t i = 0; i < axis1_shape[0]; ++i) {
        assert_int_equal(axis1_indices[i], g_reduction_fp32.argmin_axis1[i]);
    }

    assert_int_equal(
        marmot_reduce_all(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = axis1_out, .axes = axis1, .num_axes = 1, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_ARRAY(all_axis1, axis1_out, axis1_shape[0]);

    const size_t axis1_keep_shape[] = {g_reduction_fp32.shape[0], 1};
    marmot_tensor_t *axis1_keep = marmot_tensor_create(env->ctx, axis1_keep_shape, 2, dtype);
    marmot_tensor_t *axis1_keep_idx = marmot_tensor_create(env->ctx, axis1_keep_shape, 2, MARMOT_DTYPE_UINT64);
    assert_non_null(axis1_keep);
    assert_non_null(axis1_keep_idx);

    assert_int_equal(
        marmot_reduce_sum(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = axis1_keep, .axes = axis1, .num_axes = 1, .keepdims = true}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_ARRAY(sum_axis1, axis1_keep, axis1_shape[0]);
    assert_int_equal(axis1_keep->shape.shape[0], g_reduction_fp32.shape[0]);
    assert_int_equal(axis1_keep->shape.shape[1], 1);

    assert_int_equal(
        marmot_reduce_argmax(
            env->ctx,
            &(marmot_reduction_desc_t){.input = input,
                                       .out = axis1_keep,
                                       .indices_out = axis1_keep_idx,
                                       .axes = axis1,
                                       .num_axes = 1,
                                       .keepdims = true}
        ),
        MARMOT_SUCCESS
    );
    EXPECT_ARRAY(max_axis1, axis1_keep, axis1_shape[0]);
    const uint64_t *axis1_keep_indices = (const uint64_t *)marmot_test_tensor_data(env, axis1_keep_idx);
    for (size_t i = 0; i < axis1_shape[0]; ++i) {
        assert_int_equal(axis1_keep_indices[i], g_reduction_fp32.argmax_axis1[i]);
    }

    marmot_test_tensor_destroy_all(7, axis1_keep_idx, axis1_keep, axis1_idx, axis1_out, axis0_idx, axis0_out, input);
#undef EXPECT_ARRAY
#undef EXPECT_SCALAR
}
static void check_reduce_int64(const marmot_test_env_t *env) {
    if (env->backend == MARMOT_BACKEND_METAL) {
        return;
    }
    const size_t shape[] = {g_reduction_i32.shape[0], g_reduction_i32.shape[1]};
    marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape, 2, MARMOT_DTYPE_INT64);
    assert_non_null(input);
    const size_t elem_count = sizeof(g_reduction_i32.input) / sizeof(g_reduction_i32.input[0]);
    int64_t input_vals[sizeof(g_reduction_i32.input) / sizeof(g_reduction_i32.input[0])];
    for (size_t i = 0; i < elem_count; ++i) {
        input_vals[i] = (int64_t)g_reduction_i32.input[i];
    }
    marmot_test_tensor_write_i64(input, input_vals);
    marmot_test_commit_tensor(env, input);

    const size_t scalar_shape[] = {1};
    marmot_tensor_t *scalar_out = marmot_tensor_create(env->ctx, scalar_shape, 1, MARMOT_DTYPE_INT64);
    assert_non_null(scalar_out);
    marmot_tensor_t *indices_scalar = marmot_tensor_create(env->ctx, scalar_shape, 1, MARMOT_DTYPE_UINT64);
    assert_non_null(indices_scalar);

    assert_int_equal(
        marmot_reduce_sum(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    assert_int_equal(
        ((const marmot_int64_t *)marmot_test_tensor_data(env, scalar_out))[0].value, (int64_t)g_reduction_i32.sum_all
    );

    assert_int_equal(
        marmot_reduce_prod(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    assert_int_equal(
        ((const marmot_int64_t *)marmot_test_tensor_data(env, scalar_out))[0].value, (int64_t)g_reduction_i32.prod_all
    );

    assert_int_equal(
        marmot_reduce_max(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    assert_int_equal(
        ((const marmot_int64_t *)marmot_test_tensor_data(env, scalar_out))[0].value, (int64_t)g_reduction_i32.max_all
    );

    assert_int_equal(
        marmot_reduce_min(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    assert_int_equal(
        ((const marmot_int64_t *)marmot_test_tensor_data(env, scalar_out))[0].value, (int64_t)g_reduction_i32.min_all
    );

    assert_int_equal(
        marmot_reduce_argmax(
            env->ctx,
            &(marmot_reduction_desc_t){.input = input,
                                       .out = scalar_out,
                                       .indices_out = indices_scalar,
                                       .axes = nullptr,
                                       .num_axes = 0,
                                       .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    assert_int_equal(
        ((const marmot_int64_t *)marmot_test_tensor_data(env, scalar_out))[0].value, (int64_t)g_reduction_i32.max_all
    );
    assert_int_equal(((const uint64_t *)marmot_test_tensor_data(env, indices_scalar))[0], g_reduction_i32.argmax_all);

    assert_int_equal(
        marmot_reduce_argmin(
            env->ctx,
            &(marmot_reduction_desc_t){.input = input,
                                       .out = scalar_out,
                                       .indices_out = indices_scalar,
                                       .axes = nullptr,
                                       .num_axes = 0,
                                       .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    assert_int_equal(
        ((const marmot_int64_t *)marmot_test_tensor_data(env, scalar_out))[0].value, (int64_t)g_reduction_i32.min_all
    );
    assert_int_equal(((const uint64_t *)marmot_test_tensor_data(env, indices_scalar))[0], g_reduction_i32.argmin_all);

    assert_int_equal(
        marmot_reduce_any(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    assert_int_equal(
        ((const marmot_int64_t *)marmot_test_tensor_data(env, scalar_out))[0].value, (int64_t)g_reduction_i32.any_all
    );

    assert_int_equal(
        marmot_reduce_all(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = scalar_out, .axes = nullptr, .num_axes = 0, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    assert_int_equal(
        ((const marmot_int64_t *)marmot_test_tensor_data(env, scalar_out))[0].value, (int64_t)g_reduction_i32.all_all
    );

    const int32_t axis0[] = {0};
    const size_t axis0_shape[] = {g_reduction_i32.shape[1]};
    marmot_tensor_t *axis0_out = marmot_tensor_create(env->ctx, axis0_shape, 1, MARMOT_DTYPE_INT64);
    marmot_tensor_t *axis0_idx = marmot_tensor_create(env->ctx, axis0_shape, 1, MARMOT_DTYPE_UINT64);
    assert_non_null(axis0_out);
    assert_non_null(axis0_idx);

    assert_int_equal(
        marmot_reduce_sum(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = axis0_out, .axes = axis0, .num_axes = 1, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    const marmot_int64_t *axis0_sum = (const marmot_int64_t *)marmot_test_tensor_data(env, axis0_out);
    for (size_t i = 0; i < axis0_shape[0]; ++i) {
        assert_int_equal(axis0_sum[i].value, (int64_t)g_reduction_i32.sum_axis0[i]);
    }

    assert_int_equal(
        marmot_reduce_argmax(
            env->ctx,
            &(marmot_reduction_desc_t){.input = input,
                                       .out = axis0_out,
                                       .indices_out = axis0_idx,
                                       .axes = axis0,
                                       .num_axes = 1,
                                       .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    const marmot_int64_t *axis0_max = (const marmot_int64_t *)marmot_test_tensor_data(env, axis0_out);
    const uint64_t *axis0_indices = (const uint64_t *)marmot_test_tensor_data(env, axis0_idx);
    for (size_t i = 0; i < axis0_shape[0]; ++i) {
        assert_int_equal(axis0_max[i].value, (int64_t)g_reduction_i32.max_axis0[i]);
        assert_int_equal(axis0_indices[i], g_reduction_i32.argmax_axis0[i]);
    }

    assert_int_equal(
        marmot_reduce_argmin(
            env->ctx,
            &(marmot_reduction_desc_t){.input = input,
                                       .out = axis0_out,
                                       .indices_out = axis0_idx,
                                       .axes = axis0,
                                       .num_axes = 1,
                                       .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    const marmot_int64_t *axis0_min = (const marmot_int64_t *)marmot_test_tensor_data(env, axis0_out);
    axis0_indices = (const uint64_t *)marmot_test_tensor_data(env, axis0_idx);
    for (size_t i = 0; i < axis0_shape[0]; ++i) {
        assert_int_equal(axis0_min[i].value, (int64_t)g_reduction_i32.min_axis0[i]);
        assert_int_equal(axis0_indices[i], g_reduction_i32.argmin_axis0[i]);
    }

    assert_int_equal(
        marmot_reduce_any(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = axis0_out, .axes = axis0, .num_axes = 1, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    const marmot_int64_t *axis0_any = (const marmot_int64_t *)marmot_test_tensor_data(env, axis0_out);
    for (size_t i = 0; i < axis0_shape[0]; ++i) {
        assert_int_equal(axis0_any[i].value, (int64_t)g_reduction_i32.any_axis0[i]);
    }

    const int32_t axis1[] = {1};
    const size_t axis1_shape[] = {g_reduction_i32.shape[0]};
    marmot_tensor_t *axis1_out = marmot_tensor_create(env->ctx, axis1_shape, 1, MARMOT_DTYPE_INT64);
    marmot_tensor_t *axis1_idx = marmot_tensor_create(env->ctx, axis1_shape, 1, MARMOT_DTYPE_UINT64);
    assert_non_null(axis1_out);
    assert_non_null(axis1_idx);

    assert_int_equal(
        marmot_reduce_sum(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = axis1_out, .axes = axis1, .num_axes = 1, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    const marmot_int64_t *axis1_sum = (const marmot_int64_t *)marmot_test_tensor_data(env, axis1_out);
    for (size_t i = 0; i < axis1_shape[0]; ++i) {
        assert_int_equal(axis1_sum[i].value, (int64_t)g_reduction_i32.sum_axis1[i]);
    }

    assert_int_equal(
        marmot_reduce_argmax(
            env->ctx,
            &(marmot_reduction_desc_t){.input = input,
                                       .out = axis1_out,
                                       .indices_out = axis1_idx,
                                       .axes = axis1,
                                       .num_axes = 1,
                                       .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    const marmot_int64_t *axis1_max = (const marmot_int64_t *)marmot_test_tensor_data(env, axis1_out);
    const uint64_t *axis1_indices = (const uint64_t *)marmot_test_tensor_data(env, axis1_idx);
    for (size_t i = 0; i < axis1_shape[0]; ++i) {
        assert_int_equal(axis1_max[i].value, (int64_t)g_reduction_i32.max_axis1[i]);
        assert_int_equal(axis1_indices[i], g_reduction_i32.argmax_axis1[i]);
    }

    assert_int_equal(
        marmot_reduce_argmin(
            env->ctx,
            &(marmot_reduction_desc_t){.input = input,
                                       .out = axis1_out,
                                       .indices_out = axis1_idx,
                                       .axes = axis1,
                                       .num_axes = 1,
                                       .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    const marmot_int64_t *axis1_min = (const marmot_int64_t *)marmot_test_tensor_data(env, axis1_out);
    axis1_indices = (const uint64_t *)marmot_test_tensor_data(env, axis1_idx);
    for (size_t i = 0; i < axis1_shape[0]; ++i) {
        assert_int_equal(axis1_min[i].value, (int64_t)g_reduction_i32.min_axis1[i]);
        assert_int_equal(axis1_indices[i], g_reduction_i32.argmin_axis1[i]);
    }

    assert_int_equal(
        marmot_reduce_all(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = axis1_out, .axes = axis1, .num_axes = 1, .keepdims = false}
        ),
        MARMOT_SUCCESS
    );
    const marmot_int64_t *axis1_all = (const marmot_int64_t *)marmot_test_tensor_data(env, axis1_out);
    for (size_t i = 0; i < axis1_shape[0]; ++i) {
        assert_int_equal(axis1_all[i].value, (int64_t)g_reduction_i32.all_axis1[i]);
    }

    const size_t axis1_keep_shape[] = {g_reduction_i32.shape[0], 1};
    marmot_tensor_t *axis1_keep = marmot_tensor_create(env->ctx, axis1_keep_shape, 2, MARMOT_DTYPE_INT64);
    marmot_tensor_t *axis1_keep_idx = marmot_tensor_create(env->ctx, axis1_keep_shape, 2, MARMOT_DTYPE_UINT64);
    assert_non_null(axis1_keep);
    assert_non_null(axis1_keep_idx);

    assert_int_equal(
        marmot_reduce_sum(
            env->ctx,
            &(
                marmot_reduction_desc_t
            ){.input = input, .out = axis1_keep, .axes = axis1, .num_axes = 1, .keepdims = true}
        ),
        MARMOT_SUCCESS
    );
    const marmot_int64_t *axis1_keep_sum = (const marmot_int64_t *)marmot_test_tensor_data(env, axis1_keep);
    for (size_t i = 0; i < axis1_shape[0]; ++i) {
        assert_int_equal(axis1_keep_sum[i].value, (int64_t)g_reduction_i32.sum_axis1[i]);
    }
    assert_int_equal(axis1_keep->shape.ndim, 2);
    assert_int_equal(axis1_keep->shape.shape[0], g_reduction_i32.shape[0]);
    assert_int_equal(axis1_keep->shape.shape[1], 1);

    marmot_test_tensor_destroy_all(
        9, axis1_keep_idx, axis1_keep, axis1_idx, axis1_out, axis0_idx, axis0_out, indices_scalar, scalar_out, input
    );
}
#if defined(__APPLE__) && MARMOT_ENABLE_METAL
static void test_reduction_random_cpu_vs_metal(void **state) {
    (void)state;

    marmot_context_t *cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(cpu_ctx);

    marmot_context_t *metal_ctx = marmot_init(MARMOT_BACKEND_METAL);
    if (metal_ctx == nullptr) {
        marmot_destroy(cpu_ctx);
        skip();
        return;
    }

    static const struct {
        size_t ndim;
        size_t shape[3];
    } kShapes[] = {
        {.ndim = 1, .shape = {128, 1, 1}},
        {.ndim = 2, .shape = {16, 9, 1}},
        {.ndim = 3, .shape = {4, 5, 7}},
    };

    static const marmot_dtype_t kDtypes[] = {
        MARMOT_DTYPE_FLOAT32,
        MARMOT_DTYPE_FLOAT16,
        MARMOT_DTYPE_BFLOAT16,
    };

    uint32_t rng_state = 0x13579BDFu;

    for (size_t d = 0; d < sizeof(kDtypes) / sizeof(kDtypes[0]); ++d) {
        marmot_dtype_t dtype = kDtypes[d];
        for (size_t s = 0; s < sizeof(kShapes) / sizeof(kShapes[0]); ++s) {
            const size_t ndim = kShapes[s].ndim;
            const size_t *shape = kShapes[s].shape;
            const size_t element_count = marmot_tensor_element_count(shape, ndim);

            float *values = (float *)malloc(element_count * sizeof(float));
            assert_non_null(values);
            marmot_fill_random_values(&rng_state, values, element_count);

            marmot_tensor_t *cpu_input = marmot_tensor_create(cpu_ctx, shape, ndim, dtype);
            marmot_tensor_t *metal_input = marmot_tensor_create(metal_ctx, shape, ndim, dtype);
            assert_non_null(cpu_input);
            assert_non_null(metal_input);

            marmot_tensor_write_floats_ctx(cpu_ctx, cpu_input, values);
            marmot_tensor_write_floats_ctx(metal_ctx, metal_input, values);

            marmot_compare_reduction_case(cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, nullptr, 0, false);
            marmot_compare_reduction_case(cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, nullptr, 0, true);

            const int32_t axis0[] = {0};
            marmot_compare_reduction_case(cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axis0, 1, false);
            marmot_compare_reduction_case(cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axis0, 1, true);

            if (ndim >= 2) {
                const int32_t axis1[] = {1};
                const int32_t axes01[] = {0, 1};
                marmot_compare_reduction_case(cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axis1, 1, false);
                marmot_compare_reduction_case(cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axis1, 1, true);
                marmot_compare_reduction_case(cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axes01, 2, false);
                marmot_compare_reduction_case(cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axes01, 2, true);
            }

            if (ndim >= 3) {
                const int32_t axis2[] = {2};
                const int32_t axes02[] = {0, 2};
                marmot_compare_reduction_case(cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axis2, 1, false);
                marmot_compare_reduction_case(cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axis2, 1, true);
                marmot_compare_reduction_case(cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axes02, 2, false);
                marmot_compare_reduction_case(cpu_ctx, metal_ctx, cpu_input, metal_input, dtype, axes02, 2, true);
            }

            marmot_tensor_destroy(metal_input);
            marmot_tensor_destroy(cpu_input);
            free(values);
        }
    }

    marmot_destroy(metal_ctx);
    marmot_destroy(cpu_ctx);
}
#else
static void test_reduction_random_cpu_vs_metal(void **state) {
    (void)state;
    skip();
}
#endif

static void run_reduction_suite(marmot_test_env_t *env) {
    check_reduce_float(env, MARMOT_DTYPE_FLOAT32, 1e-5f);
    if (env->backend != MARMOT_BACKEND_METAL) {
        check_reduce_float(env, MARMOT_DTYPE_FLOAT64, 1e-9f);
    }
    check_reduce_float(env, MARMOT_DTYPE_FLOAT16, 2e-2f);
    check_reduce_float(env, MARMOT_DTYPE_BFLOAT16, 7e-2f);
#if MARMOT_ENABLE_FP8
    const bool run_fp8 = env->backend != MARMOT_BACKEND_CPU && env->backend != MARMOT_BACKEND_METAL;
    if (run_fp8) {
        check_reduce_float(env, MARMOT_DTYPE_FLOAT8_E4M3, 5e-1f);
        check_reduce_float(env, MARMOT_DTYPE_FLOAT8_E5M2, 6e-1f);
    }
#endif
    check_reduce_int64(env);
    check_reduce_int32(env);
    check_reduce_int16(env);
    check_reduce_int8(env);
    check_reduce_uint32(env);
    check_reduce_uint16(env);
    check_reduce_uint8(env);
}

static void test_reduction_default(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    run_reduction_suite(env);
}

#if MARMOT_TEST_HAS_CPU_SCALAR_SUITE
static void test_reduction_scalar(void **state) {
    marmot_test_run_with_cpu_scalar((marmot_test_env_t *)(*state), run_reduction_suite);
}
#endif

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_reduction_random_cpu_vs_metal),
        cmocka_unit_test_setup_teardown(
            test_reduction_default, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
#if MARMOT_TEST_HAS_CPU_SCALAR_SUITE
        cmocka_unit_test_setup_teardown(
            test_reduction_scalar, marmot_test_backend_setup_scalar, marmot_test_backend_teardown
        ),
#endif
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
