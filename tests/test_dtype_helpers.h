#ifndef TEST_DTYPE_HELPERS_H
#define TEST_DTYPE_HELPERS_H

#include "marmot/tensor.h"
#include "marmot/types.h"

#include "utils/dtype_ref.h"

// Test helper functions: Vectorized conversions using reference implementations
// These wrap the scalar reference implementations for use in tests that don't have a context

static inline marmot_error_t
test_convert_f32_to_f16(const marmot_context_t *ctx, marmot_float16_t *dst, const float *src, size_t n) {
    (void)ctx;
    for (size_t i = 0; i < n; i++) {
        dst[i] = marmot_f32_to_f16_ref(src[i]);
    }
    return MARMOT_SUCCESS;
}

static inline marmot_error_t
test_convert_f16_to_f32(const marmot_context_t *ctx, float *dst, const marmot_float16_t *src, size_t n) {
    (void)ctx;
    for (size_t i = 0; i < n; i++) {
        dst[i] = marmot_f16_to_f32_ref(src[i]);
    }
    return MARMOT_SUCCESS;
}

static inline marmot_error_t
test_convert_f32_to_bf16(const marmot_context_t *ctx, marmot_bfloat16_t *dst, const float *src, size_t n) {
    (void)ctx;
    for (size_t i = 0; i < n; i++) {
        dst[i] = marmot_f32_to_bf16_ref(src[i]);
    }
    return MARMOT_SUCCESS;
}

static inline marmot_error_t
test_convert_bf16_to_f32(const marmot_context_t *ctx, float *dst, const marmot_bfloat16_t *src, size_t n) {
    (void)ctx;
    for (size_t i = 0; i < n; i++) {
        dst[i] = marmot_bf16_to_f32_ref(src[i]);
    }
    return MARMOT_SUCCESS;
}

static inline marmot_error_t
test_convert_f16_to_bf16(const marmot_context_t *ctx, marmot_bfloat16_t *dst, const marmot_float16_t *src, size_t n) {
    (void)ctx;
    for (size_t i = 0; i < n; i++) {
        dst[i] = marmot_f16_to_bf16_ref(src[i]);
    }
    return MARMOT_SUCCESS;
}

static inline marmot_error_t
test_convert_bf16_to_f16(const marmot_context_t *ctx, marmot_float16_t *dst, const marmot_bfloat16_t *src, size_t n) {
    (void)ctx;
    for (size_t i = 0; i < n; i++) {
        dst[i] = marmot_bf16_to_f16_ref(src[i]);
    }
    return MARMOT_SUCCESS;
}

// Macros to use in tests - maps old API names to helpers
#define marmot_convert_f32_to_f16 test_convert_f32_to_f16
#define marmot_convert_f16_to_f32 test_convert_f16_to_f32
#define marmot_convert_f32_to_bf16 test_convert_f32_to_bf16
#define marmot_convert_bf16_to_f32 test_convert_bf16_to_f32
#define marmot_convert_f16_to_bf16 test_convert_f16_to_bf16
#define marmot_convert_bf16_to_f16 test_convert_bf16_to_f16

// Scalar conversion macros
#define marmot_f32_to_f16 marmot_f32_to_f16_ref
#define marmot_f16_to_f32 marmot_f16_to_f32_ref
#define marmot_f32_to_bf16 marmot_f32_to_bf16_ref
#define marmot_bf16_to_f32 marmot_bf16_to_f32_ref
#define marmot_f16_to_bf16 marmot_f16_to_bf16_ref
#define marmot_bf16_to_f16 marmot_bf16_to_f16_ref

#endif // TEST_DTYPE_HELPERS_H
