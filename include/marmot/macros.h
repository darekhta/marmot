#ifndef MARMOT_MACROS_H
#define MARMOT_MACROS_H

#include <stdbool.h>
#include <stddef.h>

#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__has_c_attribute)
#define MARMOT_HAS_C_ATTRIBUTE(attr) __has_c_attribute(attr)
#else
#define MARMOT_HAS_C_ATTRIBUTE(attr) 0
#endif

#if defined(__has_builtin)
#define MARMOT_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#define MARMOT_HAS_BUILTIN(builtin) 0
#endif

#if !defined(MARMOT_HAS_BITINT)
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
#define MARMOT_HAS_BITINT 1
#elif defined(__has_extension)
#if __has_extension(c_bitint)
#define MARMOT_HAS_BITINT 1
#endif
#endif
#endif
#ifndef MARMOT_HAS_BITINT
#define MARMOT_HAS_BITINT 0
#endif

#ifndef __cplusplus
// Type-safe min/max using C23 static inline functions (portable, no GNU extensions)
static inline size_t min_size(size_t a, size_t b) {
    return a < b ? a : b;
}

static inline size_t max_size(size_t a, size_t b) {
    return a > b ? a : b;
}

static inline int min_int(int a, int b) {
    return a < b ? a : b;
}

static inline int max_int(int a, int b) {
    return a > b ? a : b;
}

static inline float min_float(float a, float b) {
    return a < b ? a : b;
}

static inline float max_float(float a, float b) {
    return a > b ? a : b;
}

// Type-safe MIN/MAX using _Generic (C11+) - dispatches to inline functions
// No double-evaluation, type-checked at compile time
#if defined(MIN)
#undef MIN
#endif
#define MIN(a, b) _Generic((a), size_t: min_size, int: min_int, float: min_float, default: min_int)((a), (b))

#if defined(MAX)
#undef MAX
#endif
#define MAX(a, b) _Generic((a), size_t: max_size, int: max_int, float: max_float, default: max_int)((a), (b))

// Note: Type-safe MIN/MAX macros use _Generic dispatch to inline functions
// This eliminates double-evaluation while providing convenient generic syntax
#endif

// Compile-time size assertions
#define ASSERT_SIZE(type, expected_size)                                                                               \
    static_assert(sizeof(type) == (expected_size), #type " must be " #expected_size " bytes")

// Array length macro (compile-time)
#define ARRAY_LENGTH(arr) (sizeof(arr) / sizeof((arr)[0]))

// Alignment helper macros
#define ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))
#define ALIGN_DOWN(x, align) ((x) & ~((align) - 1))
#define IS_ALIGNED(x, align) (((x) & ((align) - 1)) == 0)

// Cache line alignment (typically 64 bytes)
constexpr size_t MARMOT_CACHE_LINE_SIZE = 64;

#define CACHE_ALIGNED __attribute__((aligned(MARMOT_CACHE_LINE_SIZE)))

// Likely/unlikely hints for branch prediction using standard attributes where possible
static inline bool marmot_branch_likely(bool condition) {
#if MARMOT_HAS_C_ATTRIBUTE(likely)
    if (condition) [[likely]] {
        return true;
    }
    return false;
#elif MARMOT_HAS_BUILTIN(__builtin_expect)
    return __builtin_expect(condition, true);
#else
    return condition;
#endif
}

static inline bool marmot_branch_unlikely(bool condition) {
#if MARMOT_HAS_C_ATTRIBUTE(unlikely)
    if (condition) [[unlikely]] {
        return true;
    }
    return false;
#elif MARMOT_HAS_BUILTIN(__builtin_expect)
    return __builtin_expect(condition, false);
#else
    return condition;
#endif
}

#define likely(x) marmot_branch_likely(!!(x))
#define unlikely(x) marmot_branch_unlikely(!!(x))

#if MARMOT_HAS_C_ATTRIBUTE(likely)
#define MARMOT_IF_LIKELY(cond) if (cond) [[likely]]
#define MARMOT_IF_UNLIKELY(cond) if (cond) [[unlikely]]
#else
#define MARMOT_IF_LIKELY(cond) if (likely(cond))
#define MARMOT_IF_UNLIKELY(cond) if (unlikely(cond))
#endif

// Symbol visibility for shared library
// On Windows, dllexport/dllimport is required
// On Unix, all symbols are exported by default (no -fvisibility=hidden)
#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(MARMOT_BUILDING_LIBRARY)
#define MARMOT_API __declspec(dllexport)
#else
#define MARMOT_API __declspec(dllimport)
#endif
#else
#define MARMOT_API
#endif

#if MARMOT_HAS_C_ATTRIBUTE(nodiscard)
#define MARMOT_NODISCARD [[nodiscard]]
#elif defined(__cplusplus)
#define MARMOT_NODISCARD [[nodiscard]]
#elif defined(__GNUC__) || defined(__clang__)
#define MARMOT_NODISCARD __attribute__((warn_unused_result))
#else
#define MARMOT_NODISCARD
#endif

#if MARMOT_HAS_C_ATTRIBUTE(maybe_unused)
#define MARMOT_MAYBE_UNUSED [[maybe_unused]]
#elif defined(__GNUC__) || defined(__clang__)
#define MARMOT_MAYBE_UNUSED __attribute__((unused))
#else
#define MARMOT_MAYBE_UNUSED
#endif

#ifdef __cplusplus
}
#endif

#endif
