#include <stddef.h>

#include <math.h>

#include "backend/golden_data.h"
#include "backend/test_backend_utils.h"
#include "utils/dtype_ref.h"

static void check_roundtrip_f16(marmot_test_env_t *env) {
    const size_t count = g_conv_f16.length;
    marmot_float16_t tmp[count];
    float recovered[count];

    assert_int_equal(marmot_convert_f32_to_f16(env->ctx, tmp, g_conv_f16.src, count), MARMOT_SUCCESS);
    for (size_t i = 0; i < count; ++i) {
        assert_int_equal(tmp[i].bits, g_conv_f16.f16_bits[i]);
    }
    assert_int_equal(marmot_convert_f16_to_f32(env->ctx, recovered, tmp, count), MARMOT_SUCCESS);
    for (size_t i = 0; i < count; ++i) {
        float diff = fabsf(recovered[i] - g_conv_f16.roundtrip[i]);
        assert_true(diff <= 6e-3f);
    }
}

static void check_roundtrip_bf16(marmot_test_env_t *env) {
    const size_t count = g_conv_bf16.length;
    marmot_bfloat16_t tmp[count];
    float recovered[count];

    assert_int_equal(marmot_convert_f32_to_bf16(env->ctx, tmp, g_conv_bf16.src, count), MARMOT_SUCCESS);
    for (size_t i = 0; i < count; ++i) {
        assert_int_equal(tmp[i].bits, g_conv_bf16.bf16_bits[i]);
    }
    assert_int_equal(marmot_convert_bf16_to_f32(env->ctx, recovered, tmp, count), MARMOT_SUCCESS);
    for (size_t i = 0; i < count; ++i) {
        float diff = fabsf(recovered[i] - g_conv_bf16.roundtrip[i]);
        assert_true(diff <= 1.6e-2f);
    }
}

static void check_f16_bf16_bridge(marmot_test_env_t *env) {
    const size_t count = g_conv_f16_bf16_bridge.length;
    marmot_float16_t tmp_f16[count];
    marmot_bfloat16_t tmp_bf16[count];
    float recovered[count];

    assert_int_equal(marmot_convert_f32_to_f16(env->ctx, tmp_f16, g_conv_f16_bf16_bridge.src, count), MARMOT_SUCCESS);
    assert_int_equal(marmot_convert_f16_to_bf16(env->ctx, tmp_bf16, tmp_f16, count), MARMOT_SUCCESS);
    for (size_t i = 0; i < count; ++i) {
        assert_int_equal(tmp_f16[i].bits, g_conv_f16_bf16_bridge.f16_bits[i]);
        assert_int_equal(tmp_bf16[i].bits, g_conv_f16_bf16_bridge.bf16_from_f16_bits[i]);
    }
    assert_int_equal(marmot_convert_bf16_to_f32(env->ctx, recovered, tmp_bf16, count), MARMOT_SUCCESS);
    for (size_t i = 0; i < count; ++i) {
        float diff = fabsf(recovered[i] - g_conv_f16_bf16_bridge.bf16_from_f16_roundtrip[i]);
        assert_true(diff <= 2.0e-2f);
    }

    // Reverse direction
    assert_int_equal(marmot_convert_f32_to_bf16(env->ctx, tmp_bf16, g_conv_f16_bf16_bridge.src, count), MARMOT_SUCCESS);
    assert_int_equal(marmot_convert_bf16_to_f16(env->ctx, tmp_f16, tmp_bf16, count), MARMOT_SUCCESS);
    for (size_t i = 0; i < count; ++i) {
        assert_int_equal(tmp_bf16[i].bits, g_conv_f16_bf16_bridge.bf16_bits[i]);
        assert_int_equal(tmp_f16[i].bits, g_conv_f16_bf16_bridge.f16_from_bf16_bits[i]);
    }
    assert_int_equal(marmot_convert_f16_to_f32(env->ctx, recovered, tmp_f16, count), MARMOT_SUCCESS);
    for (size_t i = 0; i < count; ++i) {
        float diff = fabsf(recovered[i] - g_conv_f16_bf16_bridge.f16_from_bf16_roundtrip[i]);
        assert_true(diff <= 2.0e-2f);
    }
}

#if MARMOT_ENABLE_FP8
static void check_fp8_conversions(marmot_test_env_t *env) {
    if (env->backend == MARMOT_BACKEND_METAL || env->backend == MARMOT_BACKEND_CPU) {
        return;
    }
    const size_t count = g_conv_fp8.length;
    marmot_float8_e4m3_t fp8_e4m3[count];
    marmot_float8_e5m2_t fp8_e5m2[count];
    float recovered[count];

    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT8_E4M3, fp8_e4m3, MARMOT_DTYPE_FLOAT32, g_conv_fp8.src, count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < count; ++i) {
        assert_int_equal(fp8_e4m3[i].bits, g_conv_fp8.e4m3_bits[i]);
    }
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT32, recovered, MARMOT_DTYPE_FLOAT8_E4M3, fp8_e4m3, count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < count; ++i) {
        float diff = fabsf(recovered[i] - g_conv_fp8.e4m3_roundtrip[i]);
        assert_true(diff <= 5e-1f);
    }

    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT8_E5M2, fp8_e5m2, MARMOT_DTYPE_FLOAT32, g_conv_fp8.src, count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < count; ++i) {
        assert_int_equal(fp8_e5m2[i].bits, g_conv_fp8.e5m2_bits[i]);
    }
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT32, recovered, MARMOT_DTYPE_FLOAT8_E5M2, fp8_e5m2, count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < count; ++i) {
        float diff = fabsf(recovered[i] - g_conv_fp8.e5m2_roundtrip[i]);
        assert_true(diff <= 6e-1f);
    }
}

static void check_fp8_half_bridge(marmot_test_env_t *env) {
    if (env->backend == MARMOT_BACKEND_METAL || env->backend == MARMOT_BACKEND_CPU) {
        return;
    }
    const size_t count = g_conv_fp8_half_bridge.length;
    marmot_float16_t f16[count];
    marmot_float16_t f16_roundtrip[count];
    marmot_float8_e4m3_t fp8[count];
    float recovered[count];

    assert_int_equal(marmot_convert_f32_to_f16(env->ctx, f16, g_conv_fp8_half_bridge.src, count), MARMOT_SUCCESS);
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT8_E4M3, fp8, MARMOT_DTYPE_FLOAT16, f16, count), MARMOT_SUCCESS
    );
    for (size_t i = 0; i < count; ++i) {
        assert_int_equal(fp8[i].bits, g_conv_fp8_half_bridge.fp8_bits[i]);
    }
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT16, f16_roundtrip, MARMOT_DTYPE_FLOAT8_E4M3, fp8, count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < count; ++i) {
        assert_int_equal(f16_roundtrip[i].bits, g_conv_fp8_half_bridge.f16_bits[i]);
    }
    assert_int_equal(marmot_convert_f16_to_f32(env->ctx, recovered, f16_roundtrip, count), MARMOT_SUCCESS);

    for (size_t i = 0; i < count; ++i) {
        float diff = fabsf(recovered[i] - g_conv_fp8_half_bridge.roundtrip[i]);
        assert_true(diff <= 7e-1f);
    }
}

static void check_fp8_bf16_bridge(marmot_test_env_t *env) {
    if (env->backend == MARMOT_BACKEND_METAL || env->backend == MARMOT_BACKEND_CPU) {
        return;
    }
    const size_t count = g_conv_fp8_bf16_bridge.length;
    marmot_bfloat16_t bf16[count];
    marmot_bfloat16_t bf16_roundtrip[count];
    marmot_float8_e5m2_t fp8[count];
    float recovered[count];

    assert_int_equal(marmot_convert_f32_to_bf16(env->ctx, bf16, g_conv_fp8_bf16_bridge.src, count), MARMOT_SUCCESS);
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT8_E5M2, fp8, MARMOT_DTYPE_BFLOAT16, bf16, count), MARMOT_SUCCESS
    );
    for (size_t i = 0; i < count; ++i) {
        assert_int_equal(fp8[i].bits, g_conv_fp8_bf16_bridge.fp8_bits[i]);
    }
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_BFLOAT16, bf16_roundtrip, MARMOT_DTYPE_FLOAT8_E5M2, fp8, count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < count; ++i) {
        assert_int_equal(bf16_roundtrip[i].bits, g_conv_fp8_bf16_bridge.bf16_bits[i]);
    }
    assert_int_equal(marmot_convert_bf16_to_f32(env->ctx, recovered, bf16_roundtrip, count), MARMOT_SUCCESS);

    for (size_t i = 0; i < count; ++i) {
        float diff = fabsf(recovered[i] - g_conv_fp8_bf16_bridge.roundtrip[i]);
        assert_true(diff <= 8e-1f);
    }
}
#endif

static void check_f64_golden_paths(marmot_test_env_t *env) {
    const size_t count = g_conv_f64_paths.length;
    marmot_float16_t f16[count];
    marmot_bfloat16_t bf16[count];
    marmot_int64_t ints[count];
    double recovered[count];

    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT16, f16, MARMOT_DTYPE_FLOAT64, g_conv_f64_paths.src, count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < count; ++i) {
        assert_int_equal(f16[i].bits, g_conv_f64_paths.f16_bits[i]);
    }
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT64, recovered, MARMOT_DTYPE_FLOAT16, f16, count), MARMOT_SUCCESS
    );
    for (size_t i = 0; i < count; ++i) {
        assert_true(fabs(recovered[i] - g_conv_f64_paths.f16_roundtrip[i]) <= 1e-9);
    }

    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_BFLOAT16, bf16, MARMOT_DTYPE_FLOAT64, g_conv_f64_paths.src, count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < count; ++i) {
        assert_int_equal(bf16[i].bits, g_conv_f64_paths.bf16_bits[i]);
    }
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT64, recovered, MARMOT_DTYPE_BFLOAT16, bf16, count), MARMOT_SUCCESS
    );
    for (size_t i = 0; i < count; ++i) {
        assert_true(fabs(recovered[i] - g_conv_f64_paths.bf16_roundtrip[i]) <= 1e-9);
    }

    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_INT64, ints, MARMOT_DTYPE_FLOAT64, g_conv_f64_paths.src, count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < count; ++i) {
        assert_int_equal(ints[i].value, g_conv_f64_paths.trunc_i64[i]);
    }
}

static void check_i64_golden_paths(marmot_test_env_t *env) {
    const size_t count = g_conv_i64_bridge.length;
    marmot_int64_t ints[count];
    float recovered_f32[count];
    double recovered_f64[count];

    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_INT64, ints, MARMOT_DTYPE_FLOAT32, g_conv_i64_bridge.f32_src, count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < count; ++i) {
        assert_int_equal(ints[i].value, g_conv_i64_bridge.f32_to_i64[i]);
    }

    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_INT64, ints, MARMOT_DTYPE_FLOAT64, g_conv_i64_bridge.f64_src, count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < count; ++i) {
        assert_int_equal(ints[i].value, g_conv_i64_bridge.f64_to_i64[i]);
    }

    for (size_t i = 0; i < count; ++i) {
        ints[i].value = g_conv_i64_bridge.i64_src[i];
    }
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT32, recovered_f32, MARMOT_DTYPE_INT64, ints, count), MARMOT_SUCCESS
    );
    for (size_t i = 0; i < count; ++i) {
        assert_true(fabsf(recovered_f32[i] - g_conv_i64_bridge.i64_to_f32[i]) <= 0.f);
    }
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT64, recovered_f64, MARMOT_DTYPE_INT64, ints, count), MARMOT_SUCCESS
    );
    for (size_t i = 0; i < count; ++i) {
        assert_true(fabs(recovered_f64[i] - g_conv_i64_bridge.i64_to_f64[i]) <= 0.0);
    }
}

static void check_convert_ctx_dispatch(marmot_test_env_t *env) {
    const size_t f16_count = g_conv_f16.length;
    const size_t bf16_count = g_conv_bf16.length;
#if MARMOT_ENABLE_FP8
    const size_t max_src = (g_conv_fp8.length > bf16_count ? g_conv_fp8.length : bf16_count);
#else
    const size_t max_src = bf16_count;
#endif
    double src_f64[max_src];
    double recovered_f64[max_src];

    for (size_t i = 0; i < f16_count; ++i) {
        src_f64[i] = (double)g_conv_f16.src[i];
    }
    marmot_float16_t dst_f16[f16_count];
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT16, dst_f16, MARMOT_DTYPE_FLOAT64, src_f64, f16_count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < f16_count; ++i) {
        assert_int_equal(dst_f16[i].bits, g_conv_f16.f16_bits[i]);
    }
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT64, recovered_f64, MARMOT_DTYPE_FLOAT16, dst_f16, f16_count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < f16_count; ++i) {
        assert_true(fabs(recovered_f64[i] - (double)g_conv_f16.roundtrip[i]) <= 1e-6);
    }

    for (size_t i = 0; i < bf16_count; ++i) {
        src_f64[i] = (double)g_conv_bf16.src[i];
    }
    marmot_bfloat16_t dst_bf16[bf16_count];
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_BFLOAT16, dst_bf16, MARMOT_DTYPE_FLOAT64, src_f64, bf16_count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < bf16_count; ++i) {
        assert_int_equal(dst_bf16[i].bits, g_conv_bf16.bf16_bits[i]);
    }
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT64, recovered_f64, MARMOT_DTYPE_BFLOAT16, dst_bf16, bf16_count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < bf16_count; ++i) {
        assert_true(fabs(recovered_f64[i] - (double)g_conv_bf16.roundtrip[i]) <= 1e-6);
    }

    static const double test_f64_vals[] = {-4.25, -1.0, -0.25, 0.0, 1.5, 2.75};
    static const float test_f32_vals[] = {-3.5f, -1.25f, -0.5f, 0.0f, 2.0f, 3.25f};
    static const int64_t test_i64_vals[] = {-7, -1, 0, 1, 3, 9};
    const size_t int_count = sizeof(test_i64_vals) / sizeof(test_i64_vals[0]);

    marmot_int64_t ints[int_count];
    for (size_t i = 0; i < int_count; ++i) {
        ints[i].value = test_i64_vals[i];
    }

    float from_i64_f32[int_count];
    double from_i64_f64[int_count];
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT32, from_i64_f32, MARMOT_DTYPE_INT64, ints, int_count),
        MARMOT_SUCCESS
    );
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT64, from_i64_f64, MARMOT_DTYPE_INT64, ints, int_count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < int_count; ++i) {
        assert_true(fabsf(from_i64_f32[i] - (float)test_i64_vals[i]) <= 0.f);
        assert_true(fabs(from_i64_f64[i] - (double)test_i64_vals[i]) <= 0.0);
    }

    marmot_int64_t back_from_f32[int_count];
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_INT64, back_from_f32, MARMOT_DTYPE_FLOAT32, test_f32_vals, int_count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < int_count; ++i) {
        assert_int_equal(back_from_f32[i].value, (int64_t)test_f32_vals[i]);
    }

    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_INT64, back_from_f32, MARMOT_DTYPE_FLOAT64, test_f64_vals, int_count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < int_count; ++i) {
        assert_int_equal(back_from_f32[i].value, (int64_t)test_f64_vals[i]);
    }

#if MARMOT_ENABLE_FP8
    const size_t fp8_count = g_conv_fp8.length;
    for (size_t i = 0; i < fp8_count; ++i) {
        src_f64[i] = (double)g_conv_fp8.src[i];
    }
    marmot_float8_e4m3_t fp8_e4m3[fp8_count];
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT8_E4M3, fp8_e4m3, MARMOT_DTYPE_FLOAT64, src_f64, fp8_count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < fp8_count; ++i) {
        assert_int_equal(fp8_e4m3[i].bits, g_conv_fp8.e4m3_bits[i]);
    }
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT64, recovered_f64, MARMOT_DTYPE_FLOAT8_E4M3, fp8_e4m3, fp8_count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < fp8_count; ++i) {
        assert_true(fabs(recovered_f64[i] - (double)g_conv_fp8.e4m3_roundtrip[i]) <= 1e-6);
    }
    marmot_float8_e5m2_t fp8_e5m2[fp8_count];
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT8_E5M2, fp8_e5m2, MARMOT_DTYPE_FLOAT64, src_f64, fp8_count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < fp8_count; ++i) {
        assert_int_equal(fp8_e5m2[i].bits, g_conv_fp8.e5m2_bits[i]);
    }
    assert_int_equal(
        marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT64, recovered_f64, MARMOT_DTYPE_FLOAT8_E5M2, fp8_e5m2, fp8_count),
        MARMOT_SUCCESS
    );
    for (size_t i = 0; i < fp8_count; ++i) {
        assert_true(fabs(recovered_f64[i] - (double)g_conv_fp8.e5m2_roundtrip[i]) <= 1e-6);
    }
#endif
}

static void expect_alias_error(
    marmot_test_env_t *env, marmot_dtype_t dst_dtype, void *dst, marmot_dtype_t src_dtype, const void *src, size_t n
) {
    marmot_clear_error();
    assert_int_equal(marmot_convert(env->ctx, dst_dtype, dst, src_dtype, src, n), MARMOT_ERROR_INVALID_ARGUMENT);
    assert_int_equal(marmot_get_last_error(), MARMOT_ERROR_INVALID_ARGUMENT);
}

static void check_conversion_alias_guards(marmot_test_env_t *env) {
    if (env->backend == MARMOT_BACKEND_METAL) {
        return;
    }
    _Alignas(max_align_t) uint8_t shared_f32[sizeof(float) * 8];
    float *src_f32 = (float *)shared_f32;
    for (size_t i = 0; i < 8; ++i) {
        src_f32[i] = (float)i;
    }
    marmot_float16_t *dst_alias = (marmot_float16_t *)(shared_f32 + sizeof(float));
    marmot_clear_error();
    assert_int_equal(marmot_convert_f32_to_f16(env->ctx, dst_alias, src_f32, 4), MARMOT_ERROR_INVALID_ARGUMENT);
    assert_int_equal(marmot_get_last_error(), MARMOT_ERROR_INVALID_ARGUMENT);
    expect_alias_error(env, MARMOT_DTYPE_FLOAT16, dst_alias, MARMOT_DTYPE_FLOAT32, src_f32, 4);

    _Alignas(max_align_t) uint8_t shared_half[sizeof(marmot_float16_t) * 16];
    marmot_float16_t *f16_src = (marmot_float16_t *)shared_half;
    marmot_bfloat16_t *bf16_dst = (marmot_bfloat16_t *)shared_half;
    expect_alias_error(env, MARMOT_DTYPE_BFLOAT16, bf16_dst, MARMOT_DTYPE_FLOAT16, f16_src, 8);

    marmot_float16_t f16_buf[16];
    expect_alias_error(env, MARMOT_DTYPE_FLOAT16, f16_buf + 1, MARMOT_DTYPE_FLOAT16, f16_buf, 8);
    expect_alias_error(env, MARMOT_DTYPE_FLOAT16, f16_buf, MARMOT_DTYPE_FLOAT16, f16_buf + 1, 8);

    double f64_buf[8];
    expect_alias_error(env, MARMOT_DTYPE_FLOAT64, f64_buf + 1, MARMOT_DTYPE_FLOAT64, f64_buf, 4);
}

static uint32_t conv_rng_next(uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float conv_rng_f32(uint32_t *state) {
    const float scale = 1.0f / 4294967296.0f;
    return ((float)conv_rng_next(state) * scale * 16.0f) - 8.0f;
}

static double conv_rng_f64(uint32_t *state) {
    uint64_t hi = conv_rng_next(state);
    uint64_t lo = conv_rng_next(state);
    uint64_t combined = (hi << 32) | lo;
    const double scale = 1.0 / (double)UINT64_MAX;
    return ((double)combined * scale * 32.0) - 16.0;
}

static void check_randomized_conversions(marmot_test_env_t *env) {
    if (env->backend == MARMOT_BACKEND_METAL) {
        return;
    }
    static const size_t sizes[] = {0, 1, 7, 256, 4096};
    const size_t max_size = sizes[sizeof(sizes) / sizeof(sizes[0]) - 1];

    float *src_f32 = (float *)malloc(max_size * sizeof(float));
    double *src_f64 = (double *)malloc(max_size * sizeof(double));
    marmot_float16_t *buf_f16 = (marmot_float16_t *)malloc(max_size * sizeof(marmot_float16_t));
    marmot_bfloat16_t *buf_bf16 = (marmot_bfloat16_t *)malloc(max_size * sizeof(marmot_bfloat16_t));
    marmot_int64_t *buf_i64 = (marmot_int64_t *)malloc(max_size * sizeof(marmot_int64_t));
    float *tmp_f32 = (float *)malloc(max_size * sizeof(float));
    double *tmp_f64 = (double *)malloc(max_size * sizeof(double));

    assert_non_null(src_f32);
    assert_non_null(src_f64);
    assert_non_null(buf_f16);
    assert_non_null(buf_bf16);
    assert_non_null(buf_i64);
    assert_non_null(tmp_f32);
    assert_non_null(tmp_f64);

    uint32_t rng_state = 0x12345678u;

    for (size_t idx = 0; idx < sizeof(sizes) / sizeof(sizes[0]); ++idx) {
        size_t count = sizes[idx];
        for (size_t i = 0; i < count; ++i) {
            src_f32[i] = conv_rng_f32(&rng_state);
            src_f64[i] = conv_rng_f64(&rng_state);
        }

        assert_int_equal(
            marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT16, buf_f16, MARMOT_DTYPE_FLOAT32, src_f32, count),
            MARMOT_SUCCESS
        );
        for (size_t i = 0; i < count; ++i) {
            marmot_float16_t expected = marmot_f32_to_f16_ref(src_f32[i]);
            uint16_t actual_bits = buf_f16[i].bits;
            uint16_t expected_bits = expected.bits;
            uint16_t diff =
                (actual_bits > expected_bits) ? (actual_bits - expected_bits) : (expected_bits - actual_bits);
            assert_true(diff <= 1);
        }
        assert_int_equal(
            marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT32, tmp_f32, MARMOT_DTYPE_FLOAT16, buf_f16, count),
            MARMOT_SUCCESS
        );
        for (size_t i = 0; i < count; ++i) {
            float expected = marmot_f16_to_f32_ref(buf_f16[i]);
            assert_true(fabsf(tmp_f32[i] - expected) <= 1e-6f);
        }

        assert_int_equal(
            marmot_convert(env->ctx, MARMOT_DTYPE_BFLOAT16, buf_bf16, MARMOT_DTYPE_FLOAT32, src_f32, count),
            MARMOT_SUCCESS
        );
        for (size_t i = 0; i < count; ++i) {
            marmot_bfloat16_t expected = marmot_f32_to_bf16_ref(src_f32[i]);
            uint16_t actual_bits = buf_bf16[i].bits;
            uint16_t expected_bits = expected.bits;
            uint16_t diff =
                (actual_bits > expected_bits) ? (actual_bits - expected_bits) : (expected_bits - actual_bits);
            assert_true(diff <= 1);
        }
        assert_int_equal(
            marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT32, tmp_f32, MARMOT_DTYPE_BFLOAT16, buf_bf16, count),
            MARMOT_SUCCESS
        );
        for (size_t i = 0; i < count; ++i) {
            float expected = marmot_bf16_to_f32_ref(buf_bf16[i]);
            assert_true(fabsf(tmp_f32[i] - expected) <= 1e-5f);
        }

        assert_int_equal(
            marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT16, buf_f16, MARMOT_DTYPE_FLOAT64, src_f64, count),
            MARMOT_SUCCESS
        );
        for (size_t i = 0; i < count; ++i) {
            marmot_float16_t expected = marmot_f32_to_f16_ref((float)src_f64[i]);
            uint16_t actual_bits = buf_f16[i].bits;
            uint16_t expected_bits = expected.bits;
            uint16_t diff =
                (actual_bits > expected_bits) ? (actual_bits - expected_bits) : (expected_bits - actual_bits);
            assert_true(diff <= 1);
        }
        assert_int_equal(
            marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT64, tmp_f64, MARMOT_DTYPE_FLOAT16, buf_f16, count),
            MARMOT_SUCCESS
        );
        for (size_t i = 0; i < count; ++i) {
            double expected = (double)marmot_f16_to_f32_ref(buf_f16[i]);
            assert_true(fabs(tmp_f64[i] - expected) <= 1e-9);
        }

        assert_int_equal(
            marmot_convert(env->ctx, MARMOT_DTYPE_BFLOAT16, buf_bf16, MARMOT_DTYPE_FLOAT64, src_f64, count),
            MARMOT_SUCCESS
        );
        for (size_t i = 0; i < count; ++i) {
            marmot_bfloat16_t expected = marmot_f32_to_bf16_ref((float)src_f64[i]);
            uint16_t actual_bits = buf_bf16[i].bits;
            uint16_t expected_bits = expected.bits;
            uint16_t diff =
                (actual_bits > expected_bits) ? (actual_bits - expected_bits) : (expected_bits - actual_bits);
            assert_true(diff <= 1);
        }
        assert_int_equal(
            marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT64, tmp_f64, MARMOT_DTYPE_BFLOAT16, buf_bf16, count),
            MARMOT_SUCCESS
        );
        for (size_t i = 0; i < count; ++i) {
            double expected = (double)marmot_bf16_to_f32_ref(buf_bf16[i]);
            assert_true(fabs(tmp_f64[i] - expected) <= 1e-6);
        }

        assert_int_equal(
            marmot_convert(env->ctx, MARMOT_DTYPE_INT64, buf_i64, MARMOT_DTYPE_FLOAT32, src_f32, count), MARMOT_SUCCESS
        );
        for (size_t i = 0; i < count; ++i) {
            int64_t expected = (int64_t)src_f32[i];
            assert_int_equal(buf_i64[i].value, expected);
        }
        assert_int_equal(
            marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT32, tmp_f32, MARMOT_DTYPE_INT64, buf_i64, count), MARMOT_SUCCESS
        );
        for (size_t i = 0; i < count; ++i) {
            float expected = (float)buf_i64[i].value;
            assert_true(fabsf(tmp_f32[i] - expected) <= 1e-6f);
        }

        assert_int_equal(
            marmot_convert(env->ctx, MARMOT_DTYPE_INT64, buf_i64, MARMOT_DTYPE_FLOAT64, src_f64, count), MARMOT_SUCCESS
        );
        for (size_t i = 0; i < count; ++i) {
            int64_t expected = (int64_t)src_f64[i];
            assert_int_equal(buf_i64[i].value, expected);
        }
        assert_int_equal(
            marmot_convert(env->ctx, MARMOT_DTYPE_FLOAT64, tmp_f64, MARMOT_DTYPE_INT64, buf_i64, count), MARMOT_SUCCESS
        );
        for (size_t i = 0; i < count; ++i) {
            double expected = (double)buf_i64[i].value;
            assert_true(fabs(tmp_f64[i] - expected) <= 1e-9);
        }
    }

    free(src_f32);
    free(src_f64);
    free(buf_f16);
    free(buf_bf16);
    free(buf_i64);
    free(tmp_f32);
    free(tmp_f64);
}

static void run_conversion_suite(marmot_test_env_t *env) {
    check_roundtrip_f16(env);
    check_roundtrip_bf16(env);
    check_f16_bf16_bridge(env);
    if (env->backend != MARMOT_BACKEND_METAL) {
        check_f64_golden_paths(env);
        check_i64_golden_paths(env);
        check_convert_ctx_dispatch(env);
    }
    check_conversion_alias_guards(env);
    check_randomized_conversions(env);
#if MARMOT_ENABLE_FP8
    check_fp8_conversions(env);
    check_fp8_half_bridge(env);
    check_fp8_bf16_bridge(env);
#endif
}

static void test_conversions_default(void **state) {
    run_conversion_suite((marmot_test_env_t *)(*state));
}

#if MARMOT_TEST_HAS_CPU_SCALAR_SUITE
static void test_conversions_scalar(void **state) {
    run_conversion_suite((marmot_test_env_t *)(*state));
}
#endif

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(
            test_conversions_default, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
#if MARMOT_TEST_HAS_CPU_SCALAR_SUITE
        cmocka_unit_test_setup_teardown(
            test_conversions_scalar, marmot_test_backend_setup_scalar, marmot_test_backend_teardown
        ),
#endif
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
