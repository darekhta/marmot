#ifndef MARMOT_OPS_CONVERSION_H
#define MARMOT_OPS_CONVERSION_H

#include "../tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Scalar conversions
static inline marmot_float16_t marmot_f32_to_f16_scalar(float value);
static inline float marmot_f16_to_f32_scalar(marmot_float16_t value);
static inline marmot_bfloat16_t marmot_f32_to_bf16_scalar(float value);
static inline float marmot_bf16_to_f32_scalar(marmot_bfloat16_t value);

// Vectorized conversions
MARMOT_NODISCARD marmot_error_t
marmot_convert_f32_to_f16(const marmot_context_t *ctx, marmot_float16_t *dst, const float *src, size_t n);
MARMOT_NODISCARD marmot_error_t
marmot_convert_f16_to_f32(const marmot_context_t *ctx, float *dst, const marmot_float16_t *src, size_t n);
MARMOT_NODISCARD marmot_error_t
marmot_convert_f32_to_bf16(const marmot_context_t *ctx, marmot_bfloat16_t *dst, const float *src, size_t n);
MARMOT_NODISCARD marmot_error_t
marmot_convert_bf16_to_f32(const marmot_context_t *ctx, float *dst, const marmot_bfloat16_t *src, size_t n);
MARMOT_NODISCARD marmot_error_t
marmot_convert_f16_to_bf16(const marmot_context_t *ctx, marmot_bfloat16_t *dst, const marmot_float16_t *src, size_t n);
MARMOT_NODISCARD marmot_error_t
marmot_convert_bf16_to_f16(const marmot_context_t *ctx, marmot_float16_t *dst, const marmot_bfloat16_t *src, size_t n);
MARMOT_NODISCARD marmot_error_t marmot_convert(
    const marmot_context_t *ctx, marmot_dtype_t dst_dtype, void *dst, marmot_dtype_t src_dtype, const void *src,
    size_t n
);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_OPS_CONVERSION_H
