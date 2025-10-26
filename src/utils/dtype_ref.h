#ifndef MARMOT_DTYPE_REF_H
#define MARMOT_DTYPE_REF_H

#include "marmot/types.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===================================================================
// Scalar Reference Implementations - NO SIMD
// ===================================================================
// These are pure scalar implementations for:
// - Correctness testing
// - Fallback when SIMD is not available
// - Reference for backend implementations
//
// IMPORTANT: NO platform-specific code (#if HAS_NEON, etc.)
// IMPORTANT: NO SIMD optimizations - backends handle those
// ===================================================================

// FLOAT32 <-> FLOAT16
marmot_float16_t marmot_f32_to_f16_ref(float value);
float marmot_f16_to_f32_ref(marmot_float16_t value);

// FLOAT32 <-> BFLOAT16
marmot_bfloat16_t marmot_f32_to_bf16_ref(float value);
float marmot_bf16_to_f32_ref(marmot_bfloat16_t value);

// FLOAT16 <-> BFLOAT16 (via FLOAT32)
marmot_bfloat16_t marmot_f16_to_bf16_ref(marmot_float16_t value);
marmot_float16_t marmot_bf16_to_f16_ref(marmot_bfloat16_t value);

#if MARMOT_ENABLE_FP8
// FLOAT32 <-> FLOAT8 (E4M3 / E5M2)
marmot_float8_e4m3_t marmot_f32_to_fp8_e4m3_ref(float value);
float marmot_fp8_e4m3_to_f32_ref(marmot_float8_e4m3_t value);
marmot_float8_e5m2_t marmot_f32_to_fp8_e5m2_ref(float value);
float marmot_fp8_e5m2_to_f32_ref(marmot_float8_e5m2_t value);
#endif

#ifdef __cplusplus
}
#endif

#endif // MARMOT_DTYPE_REF_H
