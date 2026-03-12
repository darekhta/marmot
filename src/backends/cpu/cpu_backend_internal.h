#ifndef CPU_BACKEND_INTERNAL_H
#define CPU_BACKEND_INTERNAL_H

// ==================================================================
// CPU Backend Internal Header
// ==================================================================
// Shared declarations, macros, and helpers for all CPU backend modules.
// This file is internal to the CPU backend and should NOT be included
// by code outside of src/backends/cpu/
// ==================================================================

#include "marmot/allocator.h"
#include "marmot/config.h"
#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/macros.h"
#include "marmot/ops/matmul.h"
#include "marmot/ops/paged_attention.h"
#include "marmot/ops_types.h"
#include "marmot/quant_traits.h"
#include "marmot/types.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <float.h>
#include <math.h>
#include <pthread.h>
#include <string.h>

#include "core/helpers/rope.h"
#include "ops/matmul/matmul_table.h"
#include "ops/matmul/neon/neon_matmul_scratch.h"
#include "ops/normalization/normalization_table.h"
#include "ops/reduction/reduction_table.h"
#include "ops/unary/unary_table.h"
#include "utils/dtype_ref.h"

// Prefetch helper (no-op when unsupported)
#if defined(__GNUC__) || defined(__clang__)
#define MARMOT_PREFETCH(addr) __builtin_prefetch((addr))
#else
#define MARMOT_PREFETCH(addr) (void)(addr)
#endif

static inline void cpu_sincosf(float x, float *sin_out, float *cos_out) {
    if (sin_out == nullptr && cos_out == nullptr) {
        return;
    }
#if defined(__has_builtin)
#if __has_builtin(__builtin_sincosf)
    float sin_val = 0.0f;
    float cos_val = 0.0f;
    __builtin_sincosf(x, &sin_val, &cos_val);
    if (sin_out != nullptr) {
        *sin_out = sin_val;
    }
    if (cos_out != nullptr) {
        *cos_out = cos_val;
    }
    return;
#endif
#endif
#if defined(__GNUC__) && !defined(__clang__)
    float sin_val = 0.0f;
    float cos_val = 0.0f;
    __builtin_sincosf(x, &sin_val, &cos_val);
    if (sin_out != nullptr) {
        *sin_out = sin_val;
    }
    if (cos_out != nullptr) {
        *cos_out = cos_val;
    }
    return;
#endif
    if (cos_out != nullptr) {
        *cos_out = cosf(x);
    }
    if (sin_out != nullptr) {
        *sin_out = sinf(x);
    }
}

static inline void cpu_sincos(double x, double *sin_out, double *cos_out) {
    if (sin_out == nullptr && cos_out == nullptr) {
        return;
    }
#if defined(__has_builtin)
#if __has_builtin(__builtin_sincos)
    double sin_val = 0.0;
    double cos_val = 0.0;
    __builtin_sincos(x, &sin_val, &cos_val);
    if (sin_out != nullptr) {
        *sin_out = sin_val;
    }
    if (cos_out != nullptr) {
        *cos_out = cos_val;
    }
    return;
#endif
#endif
#if defined(__GNUC__) && !defined(__clang__)
    double sin_val = 0.0;
    double cos_val = 0.0;
    __builtin_sincos(x, &sin_val, &cos_val);
    if (sin_out != nullptr) {
        *sin_out = sin_val;
    }
    if (cos_out != nullptr) {
        *cos_out = cos_val;
    }
    return;
#endif
    if (cos_out != nullptr) {
        *cos_out = cos(x);
    }
    if (sin_out != nullptr) {
        *sin_out = sin(x);
    }
}

#ifdef __cplusplus
extern "C" {
#endif

size_t cpu_allocator_current_usage(void);
size_t cpu_allocator_peak_usage(void);
void cpu_allocator_collect_usage(marmot_allocator_usage_t *usage);

// ===================================================================
// Platform Detection and SIMD Configuration
// ===================================================================

#if MARMOT_ENABLE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

// SIMD intrinsics configured at build time
#if MARMOT_ENABLE_NEON
#include <arm_neon.h>
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif

#if MARMOT_ENABLE_AVX2
#include <immintrin.h>
#define HAS_AVX2 1
#define HAS_F16C 1
#else
#define HAS_AVX2 0
#define HAS_F16C 0
#endif

// ===================================================================
// Memory Allocation Helpers
// ===================================================================

// Allocate aligned memory for SIMD operations
// Uses 64-byte alignment (MARMOT_CACHE_LINE_SIZE) for cache-line optimization
static inline void *marmot_aligned_alloc(size_t alignment, size_t size) {
    if (size == 0) {
        return nullptr;
    }

    // Round size up to alignment (required by aligned_alloc)
    size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    // C11+ aligned_alloc (standard)
    void *ptr = aligned_alloc(alignment, aligned_size);
    if (ptr != nullptr) {
        return ptr;
    }
#endif

    // Fallback to malloc if aligned_alloc fails or isn't available
    return malloc(size);
}

// ===================================================================
// CPU Context Structure
// ===================================================================

typedef struct cpu_capabilities {
    bool has_neon;
    bool has_avx2;
    bool has_f16c;
    bool has_accelerate;
} cpu_capabilities_t;

typedef struct cpu_allocation_entry {
    void *ptr;
    marmot_allocation_t info;
    struct cpu_allocation_entry *next;
} cpu_allocation_entry_t;

typedef struct cpu_allocator_tracker {
    pthread_mutex_t mutex;
    cpu_allocation_entry_t *head;
} cpu_allocator_tracker_t;

typedef struct cpu_rope_sincos_cache {
    float *sincos;
    size_t capacity_positions;
    size_t cached_positions;
    size_t pair_count;
    size_t dim;
    float theta;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    uint32_t orig_ctx_len;
    marmot_rope_scaling_type_t scaling_type;
    float attn_scale;
    bool owns_storage;
} cpu_rope_sincos_cache_t;

#define CPU_PACKED_WEIGHT_CACHE_SLOTS 128u

typedef enum cpu_packed_weight_layout {
    CPU_PACKED_WEIGHT_LAYOUT_RAW = 0,
    CPU_PACKED_WEIGHT_LAYOUT_ROW_PANEL = 1,
    CPU_PACKED_WEIGHT_LAYOUT_Q4_K_ROW_PANEL_DECODED = 2,
    CPU_PACKED_WEIGHT_LAYOUT_Q6_K_ROW_PANEL_DECODED = 3,
} cpu_packed_weight_layout_t;

typedef struct cpu_packed_weight_cache_entry {
    const void *src;
    uint8_t *packed;
    size_t bytes;
    size_t row_bytes;
    size_t rows;
    size_t block_bytes;
    size_t blocks_per_row;
    size_t panel_rows;
    size_t packed_bytes;
    cpu_packed_weight_layout_t layout;
    size_t capacity_bytes;
    uint64_t stamp;
    bool valid;
    bool sticky;
} cpu_packed_weight_cache_entry_t;

typedef struct cpu_packed_weight_cache_stats {
    uint64_t exact_lookups;
    uint64_t exact_hits;
    uint64_t range_lookups;
    uint64_t range_hits;
    uint64_t inserts;
    uint64_t evictions;
    uint64_t full_sticky_misses;
} cpu_packed_weight_cache_stats_t;

typedef struct cpu_packed_weight_cache {
    pthread_mutex_t mutex;
    cpu_packed_weight_cache_entry_t entries[CPU_PACKED_WEIGHT_CACHE_SLOTS];
    uint64_t stamp;
    cpu_packed_weight_cache_stats_t stats;
} cpu_packed_weight_cache_t;

typedef struct cpu_prepacked_weight_store {
    pthread_mutex_t mutex;
    cpu_packed_weight_cache_entry_t *entries;
    size_t count;
    size_t capacity;
    cpu_packed_weight_cache_stats_t stats;
} cpu_prepacked_weight_store_t;

typedef struct cpu_packed_weight_view {
    const uint8_t *data;
    size_t row_bytes;
    size_t rows;
    size_t block_bytes;
    size_t blocks_per_row;
    size_t panel_rows;
    size_t packed_bytes;
    cpu_packed_weight_layout_t layout;
} cpu_packed_weight_view_t;

#define CPU_QUANT_WORKSPACE_SLOTS 8u

typedef struct cpu_quant_workspace_slot {
    pthread_mutex_t mutex;
    uint8_t *activation_blocks;
    size_t activation_blocks_capacity;
    uint8_t *activation_panel;
    size_t activation_panel_capacity;
} cpu_quant_workspace_slot_t;

typedef struct cpu_quant_workspace_pool {
    cpu_quant_workspace_slot_t slots[CPU_QUANT_WORKSPACE_SLOTS];
} cpu_quant_workspace_pool_t;

// CPU backend context (shared across all modules)
typedef struct cpu_context {
    int initialized;

    size_t num_threads;
    bool thread_count_explicit;
    marmot_quant_activation_mode_t quant_activation_mode;
    bool force_q8_activations;

    cpu_capabilities_t runtime_caps;
    marmot_matmul_epilogue_t pending_matmul_epilogue;
    bool pending_matmul_epilogue_valid;

    marmot_rope_freq_cache_t rope_cache;
    cpu_rope_sincos_cache_t rope_sincos_cache;
    const marmot_allocator_ops_t *allocator_ops;
    cpu_allocator_tracker_t allocator_tracker;

    // Scratch buffer for RoPE positions (avoids malloc per dispatch)
    int32_t *rope_positions_scratch;
    size_t rope_positions_capacity;

    // Scratch buffer pool for GEMM operations (avoids malloc per matmul)
    marmot_neon_scratch_pool_t neon_scratch_pool;
    cpu_packed_weight_cache_t pinned_weight_cache;
    cpu_prepacked_weight_store_t prepacked_weight_store;
    cpu_packed_weight_cache_t packed_weight_cache;
    cpu_quant_workspace_pool_t quant_workspace_pool;
    bool profile_packed_weight_cache;

} cpu_context_t;

static inline bool cpu_matmul_take_pending_epilogue(cpu_context_t *ctx, marmot_matmul_epilogue_t *pending_epilogue) {
    if (ctx == nullptr || pending_epilogue == nullptr || !ctx->pending_matmul_epilogue_valid) {
        return false;
    }
    *pending_epilogue = ctx->pending_matmul_epilogue;
    ctx->pending_matmul_epilogue_valid = false;
    return true;
}

void cpu_packed_weight_cache_init(cpu_packed_weight_cache_t *cache);
void cpu_packed_weight_cache_destroy(cpu_packed_weight_cache_t *cache);
void cpu_prepacked_weight_store_init(cpu_prepacked_weight_store_t *store);
void cpu_prepacked_weight_store_destroy(cpu_prepacked_weight_store_t *store);
const uint8_t *
cpu_packed_weight_cache_get(cpu_context_t *ctx, const void *src, size_t bytes, size_t row_bytes, size_t rows);
const uint8_t *
cpu_prepacked_weight_lookup(cpu_context_t *ctx, const void *src, size_t bytes, size_t row_bytes, size_t rows);
const uint8_t *
cpu_pinned_weight_cache_lookup(cpu_context_t *ctx, const void *src, size_t bytes, size_t row_bytes, size_t rows);
const uint8_t *cpu_pinned_weight_cache_lookup_range(cpu_context_t *ctx, const void *src, size_t bytes);
bool cpu_pinned_weight_cache_pin(cpu_context_t *ctx, const void *src, size_t bytes, size_t row_bytes, size_t rows);
cpu_packed_weight_view_t cpu_packed_weight_cache_get_row_panel(
    cpu_context_t *ctx, const void *src, size_t rows, size_t row_bytes, size_t block_bytes, size_t blocks_per_row,
    size_t panel_rows
);
cpu_packed_weight_view_t cpu_packed_weight_cache_get_q4_k_row_panel_decoded(
    cpu_context_t *ctx, const void *src, size_t rows, size_t row_bytes, size_t blocks_per_row, size_t panel_rows
);
cpu_packed_weight_view_t cpu_packed_weight_cache_get_q6_k_row_panel_decoded(
    cpu_context_t *ctx, const void *src, size_t rows, size_t row_bytes, size_t blocks_per_row, size_t panel_rows
);
cpu_packed_weight_view_t cpu_packed_weight_cache_lookup_packed_range(
    cpu_context_t *ctx, const void *src, size_t rows, size_t row_bytes, size_t block_bytes, size_t blocks_per_row,
    size_t panel_rows, cpu_packed_weight_layout_t layout
);
cpu_packed_weight_view_t cpu_prepacked_weight_lookup_packed_range(
    cpu_context_t *ctx, const void *src, size_t rows, size_t row_bytes, size_t block_bytes, size_t blocks_per_row,
    size_t panel_rows, cpu_packed_weight_layout_t layout
);
bool cpu_prepacked_weight_store_put_raw(
    cpu_context_t *ctx, const void *src, size_t bytes, size_t row_bytes, size_t rows
);
cpu_packed_weight_view_t cpu_prepacked_weight_store_put_row_panel(
    cpu_context_t *ctx, const void *src, size_t rows, size_t row_bytes, size_t block_bytes, size_t blocks_per_row,
    size_t panel_rows
);
cpu_packed_weight_view_t cpu_prepacked_weight_store_put_q4_k_row_panel_decoded(
    cpu_context_t *ctx, const void *src, size_t rows, size_t row_bytes, size_t blocks_per_row, size_t panel_rows
);
cpu_packed_weight_view_t cpu_prepacked_weight_store_put_q6_k_row_panel_decoded(
    cpu_context_t *ctx, const void *src, size_t rows, size_t row_bytes, size_t blocks_per_row, size_t panel_rows
);
void cpu_packed_weight_cache_mark_sticky(
    cpu_context_t *ctx, const void *src, size_t bytes, size_t row_bytes, size_t rows, cpu_packed_weight_layout_t layout,
    size_t block_bytes, size_t blocks_per_row, size_t panel_rows
);
void cpu_packed_weight_cache_invalidate_ptr(cpu_context_t *ctx, const void *ptr);
void cpu_packed_weight_cache_invalidate_range(cpu_context_t *ctx, const void *start, size_t length);
void cpu_quant_workspace_pool_init(cpu_quant_workspace_pool_t *pool);
void cpu_quant_workspace_pool_destroy(cpu_quant_workspace_pool_t *pool);
cpu_quant_workspace_slot_t *cpu_quant_workspace_acquire(cpu_context_t *ctx);
void cpu_quant_workspace_release(cpu_quant_workspace_slot_t *slot);
void cpu_dispatch_set_thread_limit(size_t thread_limit);
size_t cpu_dispatch_get_thread_limit(void);
[[nodiscard]] marmot_error_t cpu_context_set_num_threads(void *device_ctx, size_t num_threads, bool explicit_override);
[[nodiscard]] size_t cpu_context_get_num_threads(const void *device_ctx);
[[nodiscard]] bool cpu_context_thread_count_is_explicit(const void *device_ctx);
bool cpu_quant_workspace_ensure_buffers(
    cpu_quant_workspace_slot_t *slot, size_t activation_blocks_bytes, size_t activation_panel_bytes
);
void cpu_on_host_ptr_freed(void *device_ctx, const void *ptr);
void cpu_on_host_range_freed(void *device_ctx, const void *start, size_t length);

// ===================================================================
// Validation Macros
// ===================================================================

// Validate that tensors are non-null
#define VALIDATE_TENSORS_2(a, b)                                                                                       \
    do {                                                                                                               \
        if (unlikely((a) == nullptr || (b) == nullptr)) {                                                              \
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensor pointer");                                    \
            return MARMOT_ERROR_INVALID_ARGUMENT;                                                                      \
        }                                                                                                              \
    } while (0)

#define VALIDATE_TENSORS_3(a, b, c)                                                                                    \
    do {                                                                                                               \
        if (unlikely((a) == nullptr || (b) == nullptr || (c) == nullptr)) {                                            \
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensor pointer");                                    \
            return MARMOT_ERROR_INVALID_ARGUMENT;                                                                      \
        }                                                                                                              \
    } while (0)

// Validate tensor dtype
#define VALIDATE_DTYPE(tensor, expected_dtype, op_name)                                                                \
    do {                                                                                                               \
        if (unlikely((tensor)->dtype != (expected_dtype))) {                                                           \
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, op_name " unsupported dtype");                            \
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;                                                                     \
        }                                                                                                              \
    } while (0)

// Validate shape compatibility
#define VALIDATE_SAME_SHAPE(a, b, op_name)                                                                             \
    do {                                                                                                               \
        if (unlikely((a)->shape.ndim != (b)->shape.ndim)) {                                                            \
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, op_name " dimension mismatch");                          \
            return MARMOT_ERROR_DIMENSION_MISMATCH;                                                                    \
        }                                                                                                              \
        for (size_t _i = 0; _i < (a)->shape.ndim; _i++) {                                                              \
            if (unlikely((a)->shape.shape[_i] != (b)->shape.shape[_i])) {                                              \
                marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, op_name " shape mismatch");                          \
                return MARMOT_ERROR_DIMENSION_MISMATCH;                                                                \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

// ===================================================================
// Via-F32 Conversion Helper
// ===================================================================

// Execute an F32 kernel on F16/BF16 data by converting to F32, operating, and converting back
// Used by operations that don't have native F16/BF16 implementations
typedef marmot_error_t (*f32_kernel_fn_t)(
    const void *device_ctx, const float *input, float *output, size_t n, void *extra_args
);

marmot_error_t cpu_op_via_f32_conversion(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output, marmot_dtype_t input_dtype,
    marmot_dtype_t output_dtype, f32_kernel_fn_t f32_kernel, void *extra_args
);

// ===================================================================
// SIMD Dispatch Helpers
// ===================================================================

// Get CPU context from device context
static inline cpu_context_t *get_cpu_context(const void *device_ctx) {
    return (cpu_context_t *)device_ctx;
}

// Check if NEON is available (compile-time flag)
static inline bool has_neon(const void *device_ctx) {
#if HAS_NEON
    const cpu_context_t *ctx = (const cpu_context_t *)device_ctx;
    if (ctx == nullptr) {
        return false;
    }
    return ctx->runtime_caps.has_neon;
#else
    (void)device_ctx;
    return false;
#endif
}

// Check if AVX2 is available (compile-time flag)
static inline bool has_avx2(const void *device_ctx) {
#if HAS_AVX2
    const cpu_context_t *ctx = (const cpu_context_t *)device_ctx;
    if (ctx == nullptr) {
        return false;
    }
    return ctx->runtime_caps.has_avx2;
#else
    (void)device_ctx;
    return false;
#endif
}

// Check if F16C is available (compile-time flag)
static inline bool has_f16c(const void *device_ctx) {
#if HAS_F16C
    const cpu_context_t *ctx = (const cpu_context_t *)device_ctx;
    if (ctx == nullptr) {
        return false;
    }
    return ctx->runtime_caps.has_f16c;
#else
    (void)device_ctx;
    return false;
#endif
}

static inline bool has_accelerate(const void *device_ctx) {
#if MARMOT_ENABLE_ACCELERATE
    const cpu_context_t *ctx = (const cpu_context_t *)device_ctx;
    if (ctx == nullptr) {
        return false;
    }
    return ctx->runtime_caps.has_accelerate;
#else
    (void)device_ctx;
    return false;
#endif
}

[[maybe_unused]] static inline const cpu_capabilities_t *cpu_compiled_capabilities(void) {
    static const cpu_capabilities_t caps = {
#if HAS_NEON
        .has_neon = true,
#else
        .has_neon = false,
#endif
#if HAS_AVX2
        .has_avx2 = true,
#else
        .has_avx2 = false,
#endif
#if HAS_F16C
        .has_f16c = true,
#else
        .has_f16c = false,
#endif
#if MARMOT_ENABLE_ACCELERATE
        .has_accelerate = true,
#else
        .has_accelerate = false,
#endif
    };
    return &caps;
}

// Compile-time helpers for cpu_context_t pointers
static inline bool cpu_ctx_has_neon(const cpu_context_t *ctx) {
    return has_neon((void *)ctx);
}

static inline bool cpu_ctx_has_avx2(const cpu_context_t *ctx) {
    return has_avx2((void *)ctx);
}

static inline bool cpu_ctx_has_f16c(const cpu_context_t *ctx) {
    return has_f16c((void *)ctx);
}

static inline float cpu_rope_position_as_f32(const marmot_tensor_t *positions, size_t index) {
    switch (positions->dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return ((const float *)positions->data)[index];
    case MARMOT_DTYPE_INT32:
        return (float)((const int32_t *)positions->data)[index];
    case MARMOT_DTYPE_INT64:
        return (float)((const int64_t *)positions->data)[index];
    default:
        return 0.0f;
    }
}

// ===================================================================
// SIMD Horizontal Reduction Helpers
// ===================================================================

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
static inline _Float16 marmot_f16_from_bits(uint16_t bits) {
    _Float16 value;
    memcpy(&value, &bits, sizeof(uint16_t));
    return value;
}

static inline uint16_t marmot_f16_to_bits(_Float16 value) {
    uint16_t bits;
    memcpy(&bits, &value, sizeof(uint16_t));
    return bits;
}
#else
#error "_Float16 support required for native half-precision kernels"
#endif

static inline _Float16 marmot_float16_to_native(marmot_float16_t value) {
    return marmot_f16_from_bits(value.bits);
}

static inline marmot_float16_t marmot_native_to_float16(_Float16 value) {
    marmot_float16_t out = {.bits = marmot_f16_to_bits(value)};
    return out;
}

static inline float marmot_bfloat16_to_native(marmot_bfloat16_t value) {
    return marmot_bf16_to_f32_ref(value);
}

static inline marmot_bfloat16_t marmot_native_to_bfloat16(float value) {
    return marmot_f32_to_bf16_ref(value);
}

#if MARMOT_ENABLE_FP8
static inline _Float16 marmot_fp8_e4m3_to_native(marmot_float8_e4m3_t value) {
    float v = marmot_fp8_e4m3_to_f32_ref(value);
    return (_Float16)v;
}

static inline _Float16 marmot_fp8_e5m2_to_native(marmot_float8_e5m2_t value) {
    float v = marmot_fp8_e5m2_to_f32_ref(value);
    return (_Float16)v;
}

static inline marmot_float8_e4m3_t marmot_native_to_fp8_e4m3(_Float16 value) {
    float promoted = (float)value;
    return marmot_f32_to_fp8_e4m3_ref(promoted);
}

static inline marmot_float8_e5m2_t marmot_native_to_fp8_e5m2(_Float16 value) {
    float promoted = (float)value;
    return marmot_f32_to_fp8_e5m2_ref(promoted);
}
#endif

static inline float cpu_load_as_f32(marmot_dtype_t dtype, const void *data, size_t index) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT64:
        return (float)((const double *)data)[index];
    case MARMOT_DTYPE_FLOAT32:
        return ((const float *)data)[index];
    case MARMOT_DTYPE_FLOAT16:
        return (float)marmot_float16_to_native(((const marmot_float16_t *)data)[index]);
    case MARMOT_DTYPE_BFLOAT16:
        return marmot_bfloat16_to_native(((const marmot_bfloat16_t *)data)[index]);
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
        return (float)marmot_fp8_e4m3_to_native(((const marmot_float8_e4m3_t *)data)[index]);
    case MARMOT_DTYPE_FLOAT8_E5M2:
        return (float)marmot_fp8_e5m2_to_native(((const marmot_float8_e5m2_t *)data)[index]);
#endif
    case MARMOT_DTYPE_INT64:
        return (float)((const marmot_int64_t *)data)[index].value;
    default:
        return 0.0f;
    }
}

static inline void cpu_store_from_f32(marmot_dtype_t dtype, void *data, size_t index, float value) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT64:
        ((double *)data)[index] = (double)value;
        break;
    case MARMOT_DTYPE_FLOAT32:
        ((float *)data)[index] = value;
        break;
    case MARMOT_DTYPE_FLOAT16:
        ((marmot_float16_t *)data)[index] = marmot_native_to_float16((_Float16)value);
        break;
    case MARMOT_DTYPE_BFLOAT16:
        ((marmot_bfloat16_t *)data)[index] = marmot_native_to_bfloat16(value);
        break;
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
        ((marmot_float8_e4m3_t *)data)[index] = marmot_native_to_fp8_e4m3((_Float16)value);
        break;
    case MARMOT_DTYPE_FLOAT8_E5M2:
        ((marmot_float8_e5m2_t *)data)[index] = marmot_native_to_fp8_e5m2((_Float16)value);
        break;
#endif
    case MARMOT_DTYPE_INT64: {
        ((marmot_int64_t *)data)[index].value = (int64_t)value;
        break;
    }
    default:
        break;
    }
}

static inline double cpu_load_as_f64(marmot_dtype_t dtype, const void *data, size_t index) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT64:
        return ((const double *)data)[index];
    case MARMOT_DTYPE_FLOAT32:
        return (double)((const float *)data)[index];
    case MARMOT_DTYPE_FLOAT16:
        return (double)marmot_float16_to_native(((const marmot_float16_t *)data)[index]);
    case MARMOT_DTYPE_BFLOAT16:
        return (double)marmot_bfloat16_to_native(((const marmot_bfloat16_t *)data)[index]);
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
        return (double)marmot_fp8_e4m3_to_native(((const marmot_float8_e4m3_t *)data)[index]);
    case MARMOT_DTYPE_FLOAT8_E5M2:
        return (double)marmot_fp8_e5m2_to_native(((const marmot_float8_e5m2_t *)data)[index]);
#endif
    case MARMOT_DTYPE_INT64:
        return (double)((const marmot_int64_t *)data)[index].value;
    default:
        return 0.0;
    }
}

static inline void cpu_store_from_f64(marmot_dtype_t dtype, void *data, size_t index, double value) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT64:
        ((double *)data)[index] = value;
        break;
    case MARMOT_DTYPE_FLOAT32:
        ((float *)data)[index] = (float)value;
        break;
    case MARMOT_DTYPE_FLOAT16:
        ((marmot_float16_t *)data)[index] = marmot_native_to_float16((_Float16)value);
        break;
    case MARMOT_DTYPE_BFLOAT16:
        ((marmot_bfloat16_t *)data)[index] = marmot_native_to_bfloat16((float)value);
        break;
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
        ((marmot_float8_e4m3_t *)data)[index] = marmot_native_to_fp8_e4m3((_Float16)value);
        break;
    case MARMOT_DTYPE_FLOAT8_E5M2:
        ((marmot_float8_e5m2_t *)data)[index] = marmot_native_to_fp8_e5m2((_Float16)value);
        break;
#endif
    case MARMOT_DTYPE_INT64:
        ((marmot_int64_t *)data)[index].value = (int64_t)value;
        break;
    default:
        break;
    }
}

#if HAS_NEON
// Horizontal sum of float32x4_t
static inline float simd_reduce_sum_f32_neon(float32x4_t v) {
    return vaddvq_f32(v);
}

// Horizontal max of float32x4_t
static inline float simd_reduce_max_f32_neon(float32x4_t v) {
    return vmaxvq_f32(v);
}
#endif

#if HAS_AVX2
// Horizontal sum of __m256
static inline float simd_reduce_sum_f32_avx2(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Horizontal max of __m256
static inline float simd_reduce_max_f32_avx2(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_max_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 maxs = _mm_max_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, maxs);
    maxs = _mm_max_ss(maxs, shuf);
    return _mm_cvtss_f32(maxs);
}
#endif

// ===================================================================
// Forward Declarations for All CPU Operations
// ===================================================================
// All compute operations are dispatched via universal dispatch.
// Backend helpers must not call back into C-API paths.

// Initialization and lifecycle
marmot_error_t cpu_init(void **device_ctx);
void cpu_destroy(const void *device_ctx);
marmot_error_t cpu_configure(const void *device_ctx, const marmot_context_t *ctx);
marmot_error_t cpu_synchronize(const void *device_ctx);

// Memory operations
marmot_error_t cpu_alloc(const void *device_ctx, size_t size, void **ptr);
void cpu_free(const void *device_ctx, void *ptr);
marmot_error_t cpu_memcpy_to_device(const void *device_ctx, void *dst, const void *src, size_t size);
marmot_error_t cpu_memcpy_from_device(const void *device_ctx, void *dst, const void *src, size_t size);
marmot_error_t cpu_allocator_usage(const void *device_ctx, marmot_allocator_usage_t *usage);

// Matrix operations
marmot_error_t cpu_matmul_quantized(
    const void *device_ctx, const marmot_tensor_t *weights, const marmot_tensor_t *activations,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);

// Per-quant-kind quantized matmul dispatch (compile-time selected)
marmot_error_t cpu_matmul_quantized_q2_k(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q3_k(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q4_k(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q5_k(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q6_k(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q8_k(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q4_0(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q4_1(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q5_0(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q5_1(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q8_0(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q8_1(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);

// Per-dtype QKV matmul dispatch (compile-time selected by codegen)
marmot_error_t cpu_matmul_qkv_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t cpu_matmul_qkv_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t cpu_matmul_qkv_bf16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);

// Per-quant-kind QKV matmul dispatch (compile-time selected by codegen)
marmot_error_t cpu_matmul_qkv_q2_k(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t cpu_matmul_qkv_q3_k(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t cpu_matmul_qkv_q4_k(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t cpu_matmul_qkv_q5_k(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t cpu_matmul_qkv_q6_k(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t cpu_matmul_qkv_q8_0(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t cpu_matmul_qkv_q8_k(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);

marmot_error_t cpu_embedding_gather(const void *device_ctx, const marmot_embedding_gather_desc_t *desc);

// Unary operations
marmot_error_t cpu_unary_apply(
    const void *device_ctx, marmot_device_unary_op_t op, const marmot_tensor_t *x,
    const marmot_activation_params_t *params, marmot_tensor_t *out
);

// Normalization operations
marmot_error_t cpu_layernorm_impl(const void *device_ctx, const marmot_layernorm_desc_t *desc);
marmot_error_t cpu_rmsnorm_impl(const void *device_ctx, const marmot_rmsnorm_desc_t *desc);
marmot_error_t cpu_rmsnorm_gemma_impl(const void *device_ctx, const marmot_rmsnorm_desc_t *desc);
marmot_error_t cpu_softmax_impl(const void *device_ctx, const marmot_softmax_desc_t *desc);
marmot_error_t cpu_topk_impl(const void *device_ctx, const marmot_topk_desc_t *desc);
marmot_error_t cpu_moe_experts_impl(const void *device_ctx, const marmot_moe_experts_desc_t *desc);
marmot_error_t cpu_rmsnorm(const void *device_ctx, const marmot_rmsnorm_desc_t *desc);

// Tensor manipulation operations
marmot_error_t cpu_reshape(
    const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const size_t *new_shape, size_t new_ndim
);
marmot_error_t cpu_contiguous(const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out);
marmot_error_t cpu_view(const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out, size_t byte_offset);
marmot_error_t cpu_transpose(const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const int *perm);
marmot_error_t cpu_concat(
    const void *device_ctx, const marmot_tensor_t *const *tensors, size_t num_tensors, marmot_tensor_t *out, int axis
);
marmot_error_t cpu_slice(
    const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const size_t *starts, const size_t *sizes
);
marmot_error_t cpu_gather_rows(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *indices, marmot_tensor_t *out
);

marmot_error_t cpu_scatter_u64_to_i32(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *indices, marmot_tensor_t *out
);

// Attention operations
marmot_error_t cpu_paged_attention_impl(const void *device_ctx, const marmot_paged_attention_desc_t *desc);
marmot_error_t
cpu_rope(const void *device_ctx, const marmot_tensor_t *x, const marmot_rope_params_t *params, marmot_tensor_t *out);

// Quantization operations (INT4 + INT8)
marmot_error_t cpu_quantize_q4_0(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_dequantize_q4_0(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_quantize_q4_1(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_dequantize_q4_1(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_quantize_q5_0(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_dequantize_q5_0(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_quantize_q5_1(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_dequantize_q5_1(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_quantize_q8_0(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_dequantize_q8_0(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_quantize_q8_1(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_dequantize_q8_1(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_quantize_q2_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_dequantize_q2_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_quantize_q3_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_dequantize_q3_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_quantize_q4_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_dequantize_q4_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_quantize_q5_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_dequantize_q5_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_quantize_q6_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_dequantize_q6_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_quantize_q8_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_dequantize_q8_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_compute_quant_params(
    const void *device_ctx, const marmot_tensor_t *tensor, marmot_dtype_t target_dtype, size_t block_size,
    marmot_quant_params_t *out_params
);
marmot_error_t cpu_quantize(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_quant_params_t *quant_params,
    marmot_tensor_t *output
);
marmot_error_t cpu_dequantize(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t
cpu_quantize_blockwise(const marmot_quant_traits_t *traits, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t
cpu_dequantize_blockwise(const marmot_quant_traits_t *traits, const marmot_tensor_t *input, marmot_tensor_t *output);
marmot_error_t cpu_quantize_with_kind(
    const void *device_ctx, marmot_quant_kind_t kind, marmot_quant_layout_t layout, const marmot_tensor_t *input,
    marmot_tensor_t *output
);
marmot_error_t cpu_dequantize_with_kind(
    const void *device_ctx, marmot_quant_kind_t kind, marmot_quant_layout_t layout, const marmot_tensor_t *input,
    marmot_tensor_t *output
);
// Reduction operations
marmot_error_t cpu_reduction(
    const void *device_ctx, marmot_device_reduction_op_t op, const marmot_tensor_t *input, marmot_tensor_t *out_values,
    marmot_tensor_t *out_indices, const marmot_reduction_params_t *params
);
marmot_error_t cpu_reduction_sum_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t cpu_reduction_mean_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t cpu_reduction_max_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t cpu_reduction_min_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t cpu_reduction_variance_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t cpu_reduction_std_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t cpu_reduction_norm_l1_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t cpu_reduction_norm_l2_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t cpu_reduction_prod_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t cpu_reduction_argmax_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t cpu_reduction_argmin_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t cpu_reduction_any_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t cpu_reduction_all_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);

// Vec dot operations (dispatches to individual cpu_vec_dot_* functions)
marmot_error_t cpu_vec_dot(const void *device_ctx, const marmot_vec_dot_descriptor_t *desc, float *result);

// Dtype conversion operations
void cpu_convert_f32_to_f16(const void *device_ctx, marmot_float16_t *dst, const float *src, size_t n);
void cpu_convert_f16_to_f32(const void *device_ctx, float *dst, const marmot_float16_t *src, size_t n);
void cpu_convert_f32_to_bf16(const void *device_ctx, marmot_bfloat16_t *dst, const float *src, size_t n);
void cpu_convert_bf16_to_f32(const void *device_ctx, float *dst, const marmot_bfloat16_t *src, size_t n);
void cpu_convert_f16_to_bf16(const void *device_ctx, marmot_bfloat16_t *dst, const marmot_float16_t *src, size_t n);
void cpu_convert_bf16_to_f16(const void *device_ctx, marmot_float16_t *dst, const marmot_bfloat16_t *src, size_t n);
void cpu_convert_f32_to_f64(const void *device_ctx, double *dst, const float *src, size_t n);
void cpu_convert_f64_to_f32(const void *device_ctx, float *dst, const double *src, size_t n);
void cpu_convert_f64_to_i64(const void *device_ctx, marmot_int64_t *dst, const double *src, size_t n);
void cpu_convert_i64_to_f64(const void *device_ctx, double *dst, const marmot_int64_t *src, size_t n);
void cpu_convert_f32_to_i64(const void *device_ctx, marmot_int64_t *dst, const float *src, size_t n);
void cpu_convert_i64_to_f32(const void *device_ctx, float *dst, const marmot_int64_t *src, size_t n);
MARMOT_NODISCARD marmot_error_t cpu_convert_dispatch(
    const void *device_ctx, marmot_dtype_t dst_dtype, void *dst, marmot_dtype_t src_dtype, const void *src, size_t n
);
#if MARMOT_ENABLE_FP8
void cpu_convert_f32_to_fp8_e4m3(const void *device_ctx, marmot_float8_e4m3_t *dst, const float *src, size_t n);
void cpu_convert_fp8_e4m3_to_f32(const void *device_ctx, float *dst, const marmot_float8_e4m3_t *src, size_t n);
void cpu_convert_f32_to_fp8_e5m2(const void *device_ctx, marmot_float8_e5m2_t *dst, const float *src, size_t n);
void cpu_convert_fp8_e5m2_to_f32(const void *device_ctx, float *dst, const marmot_float8_e5m2_t *src, size_t n);
void cpu_convert_f16_to_fp8_e4m3(
    const void *device_ctx, marmot_float8_e4m3_t *dst, const marmot_float16_t *src, size_t n
);
void cpu_convert_fp8_e4m3_to_f16(
    const void *device_ctx, marmot_float16_t *dst, const marmot_float8_e4m3_t *src, size_t n
);
void cpu_convert_f16_to_fp8_e5m2(
    const void *device_ctx, marmot_float8_e5m2_t *dst, const marmot_float16_t *src, size_t n
);
void cpu_convert_fp8_e5m2_to_f16(
    const void *device_ctx, marmot_float16_t *dst, const marmot_float8_e5m2_t *src, size_t n
);
void cpu_convert_bf16_to_fp8_e4m3(
    const void *device_ctx, marmot_float8_e4m3_t *dst, const marmot_bfloat16_t *src, size_t n
);
void cpu_convert_fp8_e4m3_to_bf16(
    const void *device_ctx, marmot_bfloat16_t *dst, const marmot_float8_e4m3_t *src, size_t n
);
void cpu_convert_bf16_to_fp8_e5m2(
    const void *device_ctx, marmot_float8_e5m2_t *dst, const marmot_bfloat16_t *src, size_t n
);
void cpu_convert_fp8_e5m2_to_bf16(
    const void *device_ctx, marmot_bfloat16_t *dst, const marmot_float8_e5m2_t *src, size_t n
);
#endif

void cpu_context_force_scalar(const void *device_ctx);
void cpu_context_use_compiled_capabilities(const void *device_ctx);

void cpu_rope_sincos_cache_init(cpu_rope_sincos_cache_t *cache);
void cpu_rope_sincos_cache_reset(cpu_rope_sincos_cache_t *cache);
void cpu_rope_sincos_cache_destroy(cpu_rope_sincos_cache_t *cache);
marmot_error_t cpu_rope_sincos_cache_ensure(
    cpu_context_t *ctx, const marmot_rope_freq_span_t *span, const marmot_tensor_t *positions, size_t count,
    bool *out_use_cache
);

// RoPE scratch buffer management (avoids malloc per dispatch)
int32_t *cpu_get_rope_positions_scratch(cpu_context_t *ctx, size_t seq_len);

// Element-wise operations

#ifdef __cplusplus
}
#endif

#endif // CPU_BACKEND_INTERNAL_H
