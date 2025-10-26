#ifndef MARMOT_CONFIG_H
#define MARMOT_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

//-----------------------------------------------------------------------------
// Experimental feature flags
//-----------------------------------------------------------------------------
// Defaults can be overridden via compiler definitions (e.g. -DMARMOT_ENABLE_FP8=1).
//-----------------------------------------------------------------------------

#ifndef MARMOT_ENABLE_FP8
#define MARMOT_ENABLE_FP8 1
#endif

// SIMD/Accelerate flags are set by the build system (meson.build):
// - MARMOT_ENABLE_ACCELERATE: Apple Accelerate framework
// - MARMOT_ENABLE_NEON: ARM NEON SIMD
// - MARMOT_ENABLE_AVX2: x86 AVX2 SIMD
// - MARMOT_ENABLE_AVX512: x86 AVX512 SIMD
// These should always be defined by the build system; no fallback needed.

#ifndef MARMOT_ENABLE_METAL
#ifdef __APPLE__
#define MARMOT_ENABLE_METAL 1
#else
#define MARMOT_ENABLE_METAL 0
#endif
#endif

#ifdef __cplusplus
}
#endif

#endif // MARMOT_CONFIG_H
