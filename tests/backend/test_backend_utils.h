#ifndef MARMOT_TEST_BACKEND_UTILS_H
#define MARMOT_TEST_BACKEND_UTILS_H

#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/ops/conversion.h"
#include "marmot/ops/elementwise.h"
#include "marmot/ops/manipulation.h"
#include "marmot/ops/matmul.h"
#include "marmot/ops/neural.h"
#include "marmot/ops/quantization.h"
#include "marmot/ops/reduction.h"
#include "marmot/ops/rope.h"
#include "marmot/ops/unary.h"
#include "marmot/tensor.h"
#include "marmot/types.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils/dtype_ref.h"

// clang-format off
#include <setjmp.h>  // Must be before cmocka.h for jmp_buf
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include <cmocka.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
// clang-format on
#include <ctype.h>
#include <math.h>
#include <string.h>

#if defined(__has_include)
#if __has_include("backends/cpu/cpu_backend_internal.h")
#include "backends/cpu/cpu_backend_internal.h"
#define MARMOT_TEST_HAS_CPU_INTERNALS 1
#endif
#endif

#ifndef MARMOT_TEST_HAS_CPU_INTERNALS
#define MARMOT_TEST_HAS_CPU_INTERNALS 0
#endif

#if MARMOT_TEST_HAS_CPU_INTERNALS
#if HAS_NEON || HAS_AVX2
#define MARMOT_TEST_HAS_CPU_SCALAR_SUITE 0
#else
#define MARMOT_TEST_HAS_CPU_SCALAR_SUITE 1
#endif
#else
#define MARMOT_TEST_HAS_CPU_SCALAR_SUITE 0
#endif

// -----------------------------------------------------------------------------
// Test harness configuration
//  - MARMOT_TEST_BACKEND selects the backend under test (defaults to CPU).
//    Accepts: "cpu", "metal" (when available), or "auto" (prefers GPU on macOS).
//  - Build with -Denable_simd=false to exercise scalar-only CPU fallbacks.
// -----------------------------------------------------------------------------

typedef struct marmot_test_env {
    marmot_backend_type_t backend;
    marmot_context_t *ctx;
} marmot_test_env_t;

static inline void
marmot_test_convert_f32_span(const marmot_test_env_t *env, marmot_tensor_t *tensor, const float *src, size_t count);

static inline void marmot_test_to_lower(char *dst, size_t dst_size, const char *src) {
    if (dst == nullptr || dst_size == 0) {
        return;
    }
    size_t i = 0;
    for (; src != nullptr && src[i] != '\0' && i + 1 < dst_size; ++i) {
        dst[i] = (char)tolower((unsigned char)src[i]);
    }
    dst[i] = '\0';
}

static inline marmot_backend_type_t marmot_test_parse_backend(const char *value) {
    if (value == nullptr || value[0] == '\0') {
        return MARMOT_BACKEND_CPU;
    }

    char lowered[32];
    marmot_test_to_lower(lowered, sizeof(lowered), value);

    if (strcmp(lowered, "cpu") == 0) {
        return MARMOT_BACKEND_CPU;
    }

#ifdef __APPLE__
    if (strcmp(lowered, "metal") == 0) {
        return MARMOT_BACKEND_METAL;
    }
#endif

    if (strcmp(lowered, "auto") == 0) {
#ifdef __APPLE__
        return MARMOT_BACKEND_METAL;
#else
        return MARMOT_BACKEND_CPU;
#endif
    }

    fail_msg("Unsupported backend specified via MARMOT_TEST_BACKEND: %s", value);
    return MARMOT_BACKEND_CPU;
}

static inline marmot_backend_type_t marmot_test_resolve_backend(void) {
    const char *env_value = getenv("MARMOT_TEST_BACKEND");
    marmot_backend_type_t backend = marmot_test_parse_backend(env_value);

    // For now default to CPU until other backends gain parity.
    if (env_value == nullptr || env_value[0] == '\0') {
        backend = MARMOT_BACKEND_CPU;
    }

    return backend;
}

#if MARMOT_TEST_HAS_CPU_INTERNALS
static inline bool marmot_test_scalar_build(void) {
#if HAS_NEON || HAS_AVX2
    return false;
#else
    return true;
#endif
}

static inline void marmot_test_run_with_cpu_scalar(marmot_test_env_t *env, void (*suite)(marmot_test_env_t *)) {
    if (env == nullptr || suite == nullptr) {
        return;
    }
    if (!marmot_test_scalar_build()) {
        skip();
        return;
    }

    if (env->backend != MARMOT_BACKEND_CPU) {
        marmot_context_t *cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
        if (cpu_ctx == nullptr) {
            fail_msg("%s", "Failed to initialize CPU context for scalar suite");
        }
        marmot_test_env_t cpu_env = {
            .backend = MARMOT_BACKEND_CPU,
            .ctx = cpu_ctx,
        };
        suite(&cpu_env);
        marmot_destroy(cpu_ctx);
        return;
    }

    suite(env);
}
#endif

static inline int marmot_test_backend_setup(void **state) {
    marmot_backend_type_t backend = marmot_test_resolve_backend();
    marmot_context_t *ctx = marmot_init(backend);
    if (ctx == nullptr) {
        marmot_error_t err = marmot_get_last_error();
        fail_msg("Failed to initialize backend (%d): %s", backend, marmot_error_string(err));
    }

    marmot_test_env_t *env = (marmot_test_env_t *)malloc(sizeof(marmot_test_env_t));
    assert_non_null(env);

    env->backend = backend;
    env->ctx = ctx;
    *state = env;
    return 0;
}

static inline int marmot_test_backend_setup_scalar(void **state) {
#if MARMOT_TEST_HAS_CPU_INTERNALS
    if (!marmot_test_scalar_build()) {
        skip();
        return 0;
    }
#endif
    return marmot_test_backend_setup(state);
}

static inline int marmot_test_backend_teardown(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env != nullptr) {
        if (env->ctx != nullptr) {
            marmot_destroy(env->ctx);
        }
        free(env);
    }
    return 0;
}

static inline marmot_tensor_t *
marmot_test_tensor_from_array(const marmot_test_env_t *env, const size_t *shape, size_t ndim, const float *values) {
    marmot_tensor_t *tensor = marmot_tensor_create(env->ctx, shape, ndim, MARMOT_DTYPE_FLOAT32);
    if (tensor == nullptr) {
        marmot_error_t last = marmot_get_last_error();
        fail_msg("tensor create failed (backend=%d, ndim=%zu): %s", env->backend, ndim, marmot_error_string(last));
    }
    if (values != nullptr) {
        marmot_test_convert_f32_span(env, tensor, values, marmot_tensor_num_elements(tensor));
    }
    return tensor;
}

static inline const void *marmot_test_tensor_data(const marmot_test_env_t *env, marmot_tensor_t *tensor) {
    const void *data = marmot_tensor_data(env->ctx, tensor);
    assert_non_null(data);
    return data;
}

static inline const float *marmot_test_tensor_f32_data(const marmot_test_env_t *env, marmot_tensor_t *tensor) {
    const float *data = marmot_tensor_data_f32(env->ctx, tensor);
    assert_non_null(data);
    return data;
}

static inline void marmot_test_commit_tensor(const marmot_test_env_t *env, marmot_tensor_t *tensor) {
    assert_non_null(env);
    assert_non_null(tensor);
    const marmot_dtype_traits_t *traits = marmot_get_dtype_traits(tensor->dtype);
    assert_non_null(traits);
    const size_t bytes = traits->storage_bytes * marmot_tensor_num_elements(tensor);
    assert_int_equal(marmot_tensor_copy_from_host_buffer(env->ctx, tensor, tensor->data, bytes), MARMOT_SUCCESS);
}

static inline void marmot_test_expect_close_array(const float *actual, const float *expected, size_t count, float tol) {
    for (size_t i = 0; i < count; ++i) {
        float diff = fabsf(actual[i] - expected[i]);
        if (diff > tol) {
            printf("Mismatch at index %zu: actual=%f expected=%f tol=%f\n", i, actual[i], expected[i], tol);
        }
        assert_true(diff <= tol);
    }
}

static inline void marmot_test_expect_equal_u32(const uint32_t *actual, const uint32_t *expected, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        assert_int_equal((int)actual[i], (int)expected[i]);
    }
}

static inline void marmot_test_tensor_destroy_all(size_t count, ...) {
    va_list args;
    va_start(args, count);
    for (size_t i = 0; i < count; ++i) {
        marmot_tensor_t *tensor = va_arg(args, marmot_tensor_t *);
        if (tensor != nullptr) {
            marmot_tensor_destroy(tensor);
        }
    }
    va_end(args);
}

static inline void marmot_test_convert_span(
    const marmot_test_env_t *env, marmot_tensor_t *tensor, marmot_dtype_t src_dtype, const void *src, size_t count
) {
    assert_non_null(env);
    assert_non_null(env->ctx);
    assert_non_null(tensor);
    assert_non_null(src);

    const marmot_dtype_traits_t *src_traits = marmot_get_dtype_traits(src_dtype);
    assert_non_null(src_traits);

    if (tensor->dtype == src_dtype) {
        const size_t bytes = src_traits->storage_bytes * count;
        assert_int_equal(marmot_tensor_copy_from_host_buffer(env->ctx, tensor, src, bytes), MARMOT_SUCCESS);
        return;
    }

    const marmot_dtype_traits_t *dst_traits = marmot_get_dtype_traits(tensor->dtype);
    assert_non_null(dst_traits);
    const size_t bytes = dst_traits->storage_bytes * count;
    void *tmp = malloc(bytes);
    assert_non_null(tmp);

    if (tensor->dtype == MARMOT_DTYPE_FLOAT16 && src_dtype == MARMOT_DTYPE_FLOAT32) {
        const float *src_f32 = (const float *)src;
        marmot_float16_t *dst = (marmot_float16_t *)tmp;
        for (size_t i = 0; i < count; ++i) {
            dst[i] = marmot_f32_to_f16_ref(src_f32[i]);
        }
    } else if (tensor->dtype == MARMOT_DTYPE_BFLOAT16 && src_dtype == MARMOT_DTYPE_FLOAT32) {
        const float *src_f32 = (const float *)src;
        marmot_bfloat16_t *dst = (marmot_bfloat16_t *)tmp;
        for (size_t i = 0; i < count; ++i) {
            dst[i] = marmot_f32_to_bf16_ref(src_f32[i]);
        }
#if MARMOT_ENABLE_FP8
    } else if (tensor->dtype == MARMOT_DTYPE_FLOAT8_E4M3 && src_dtype == MARMOT_DTYPE_FLOAT32) {
        const float *src_f32 = (const float *)src;
        marmot_float8_e4m3_t *dst = (marmot_float8_e4m3_t *)tmp;
        for (size_t i = 0; i < count; ++i) {
            dst[i] = marmot_f32_to_fp8_e4m3_ref(src_f32[i]);
        }
    } else if (tensor->dtype == MARMOT_DTYPE_FLOAT8_E5M2 && src_dtype == MARMOT_DTYPE_FLOAT32) {
        const float *src_f32 = (const float *)src;
        marmot_float8_e5m2_t *dst = (marmot_float8_e5m2_t *)tmp;
        for (size_t i = 0; i < count; ++i) {
            dst[i] = marmot_f32_to_fp8_e5m2_ref(src_f32[i]);
        }
#endif
    } else {
        assert_int_equal(marmot_convert(env->ctx, tensor->dtype, tmp, src_dtype, src, count), MARMOT_SUCCESS);
    }
    assert_int_equal(marmot_tensor_copy_from_host_buffer(env->ctx, tensor, tmp, bytes), MARMOT_SUCCESS);
    free(tmp);
}

static inline void marmot_test_fetch_span(
    const marmot_test_env_t *env, void *dst, marmot_dtype_t dst_dtype, marmot_tensor_t *tensor, size_t count
) {
    assert_non_null(env);
    assert_non_null(env->ctx);
    assert_non_null(tensor);
    assert_non_null(dst);

    const marmot_dtype_traits_t *dst_traits = marmot_get_dtype_traits(dst_dtype);
    assert_non_null(dst_traits);

    if (tensor->dtype == dst_dtype) {
        const size_t bytes = dst_traits->storage_bytes * count;
        assert_int_equal(marmot_tensor_copy_to_host_buffer(env->ctx, tensor, dst, bytes), MARMOT_SUCCESS);
        return;
    }

    const void *data = marmot_tensor_data(env->ctx, tensor);
    assert_non_null(data);

    // Use reference conversions on host for common test cases to avoid backend/device dependence.
    if (dst_dtype == MARMOT_DTYPE_FLOAT64) {
        double *dst_f64 = (double *)dst;
        switch (tensor->dtype) {
        case MARMOT_DTYPE_FLOAT32: {
            const float *src = (const float *)data;
            for (size_t i = 0; i < count; ++i) {
                dst_f64[i] = (double)src[i];
            }
            return;
        }
        case MARMOT_DTYPE_FLOAT16: {
            const marmot_float16_t *src = (const marmot_float16_t *)data;
            for (size_t i = 0; i < count; ++i) {
                dst_f64[i] = (double)marmot_f16_to_f32_ref(src[i]);
            }
            return;
        }
        case MARMOT_DTYPE_BFLOAT16: {
            const marmot_bfloat16_t *src = (const marmot_bfloat16_t *)data;
            for (size_t i = 0; i < count; ++i) {
                dst_f64[i] = (double)marmot_bf16_to_f32_ref(src[i]);
            }
            return;
        }
#if MARMOT_ENABLE_FP8
        case MARMOT_DTYPE_FLOAT8_E4M3: {
            const marmot_float8_e4m3_t *src = (const marmot_float8_e4m3_t *)data;
            for (size_t i = 0; i < count; ++i) {
                dst_f64[i] = (double)marmot_fp8_e4m3_to_f32_ref(src[i]);
            }
            return;
        }
        case MARMOT_DTYPE_FLOAT8_E5M2: {
            const marmot_float8_e5m2_t *src = (const marmot_float8_e5m2_t *)data;
            for (size_t i = 0; i < count; ++i) {
                dst_f64[i] = (double)marmot_fp8_e5m2_to_f32_ref(src[i]);
            }
            return;
        }
#endif
        default:
            break;
        }
    }

    if (dst_dtype == MARMOT_DTYPE_FLOAT32) {
        float *dst_f32 = (float *)dst;
        switch (tensor->dtype) {
        case MARMOT_DTYPE_FLOAT16: {
            const marmot_float16_t *src = (const marmot_float16_t *)data;
            for (size_t i = 0; i < count; ++i) {
                dst_f32[i] = marmot_f16_to_f32_ref(src[i]);
            }
            return;
        }
        case MARMOT_DTYPE_BFLOAT16: {
            const marmot_bfloat16_t *src = (const marmot_bfloat16_t *)data;
            for (size_t i = 0; i < count; ++i) {
                dst_f32[i] = marmot_bf16_to_f32_ref(src[i]);
            }
            return;
        }
#if MARMOT_ENABLE_FP8
        case MARMOT_DTYPE_FLOAT8_E4M3: {
            const marmot_float8_e4m3_t *src = (const marmot_float8_e4m3_t *)data;
            for (size_t i = 0; i < count; ++i) {
                dst_f32[i] = marmot_fp8_e4m3_to_f32_ref(src[i]);
            }
            return;
        }
        case MARMOT_DTYPE_FLOAT8_E5M2: {
            const marmot_float8_e5m2_t *src = (const marmot_float8_e5m2_t *)data;
            for (size_t i = 0; i < count; ++i) {
                dst_f32[i] = marmot_fp8_e5m2_to_f32_ref(src[i]);
            }
            return;
        }
#endif
        case MARMOT_DTYPE_FLOAT32: {
            const size_t bytes = dst_traits->storage_bytes * count;
            memcpy(dst_f32, data, bytes);
            return;
        }
        default:
            break;
        }
    }

    assert_int_equal(marmot_convert(env->ctx, dst_dtype, dst, tensor->dtype, data, count), MARMOT_SUCCESS);
}

static inline void
marmot_test_convert_f32_span(const marmot_test_env_t *env, marmot_tensor_t *tensor, const float *src, size_t count) {
    marmot_test_convert_span(env, tensor, MARMOT_DTYPE_FLOAT32, src, count);
}

static inline void
marmot_test_fetch_f32_span(const marmot_test_env_t *env, float *dst, marmot_tensor_t *tensor, size_t count) {
    marmot_test_fetch_span(env, dst, MARMOT_DTYPE_FLOAT32, tensor, count);
}

#endif
