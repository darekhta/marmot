#ifndef MARMOT_TENSOR_H
#define MARMOT_TENSOR_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct marmot_context;
typedef struct marmot_context marmot_context_t;

typedef enum marmot_memory_location {
    MARMOT_MEMORY_HOST = 0,
    MARMOT_MEMORY_DEVICE = 1,
    MARMOT_MEMORY_UNIFIED = 2,
    MARMOT_MEMORY_UNKNOWN = 3
} marmot_memory_location_t;

struct marmot_tensor {
    marmot_shape_t shape;
    marmot_dtype_t dtype;
    void *data;
    size_t capacity_bytes;
    bool owns_data;
    marmot_quant_params_t *quant_params;
    marmot_quant_kind_t quant_kind;
    marmot_quant_layout_t quant_layout;
    marmot_backend_type_t backend;
    marmot_context_t *ctx;
    marmot_memory_location_t memory_location;
    bool needs_sync;
    void *packed_data;
    const void *packed_src_data;
    size_t packed_bytes;
    size_t packed_row_bytes;
    size_t packed_rows;
};

// Query functions
size_t marmot_dtype_size(marmot_dtype_t dtype);
size_t marmot_tensor_num_elements(const marmot_tensor_t *tensor);
size_t marmot_tensor_size_bytes(const marmot_tensor_t *tensor);
bool marmot_tensor_is_logical_quant_weight(const marmot_tensor_t *tensor);
MARMOT_NODISCARD size_t marmot_tensor_ndim(const marmot_tensor_t *tensor);
MARMOT_NODISCARD size_t marmot_tensor_shape_at(const marmot_tensor_t *tensor, size_t dim);
MARMOT_NODISCARD size_t marmot_tensor_stride_at(const marmot_tensor_t *tensor, size_t dim);
MARMOT_NODISCARD size_t marmot_tensor_numel(const marmot_tensor_t *tensor);

// Returns the physical storage size in bytes for block-quantized tensors
// using the tensor's quant metadata and shape. Returns 0 for non-quantized
// tensors or when the layout is not recognised.
size_t marmot_tensor_quant_storage_bytes(const marmot_tensor_t *tensor);

// Creation & destruction
// All creation functions take ctx as first parameter. When ctx is nullptr,
// MARMOT_BACKEND_CPU is used and no cleanup hooks are registered.
MARMOT_NODISCARD marmot_tensor_t *
marmot_tensor_create(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_dtype_t dtype);

MARMOT_NODISCARD marmot_tensor_t *marmot_tensor_create_quantized(
    const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_quant_kind_t quant_kind
);

MARMOT_NODISCARD marmot_tensor_t *
marmot_tensor_zeros(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_dtype_t dtype);

MARMOT_NODISCARD marmot_tensor_t *
marmot_tensor_ones(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_dtype_t dtype);

MARMOT_NODISCARD marmot_tensor_t *marmot_tensor_like(const marmot_tensor_t *tensor);
MARMOT_NODISCARD marmot_tensor_t *marmot_tensor_zeros_like(const marmot_tensor_t *tensor);
MARMOT_NODISCARD marmot_tensor_t *marmot_tensor_ones_like(const marmot_tensor_t *tensor);

void marmot_tensor_destroy(marmot_tensor_t *tensor);

//------------------------------------------------------------------------------
// Copy & clone
//------------------------------------------------------------------------------
MARMOT_NODISCARD marmot_error_t marmot_tensor_copy(marmot_tensor_t *dst, const marmot_tensor_t *src);
MARMOT_NODISCARD marmot_tensor_t *marmot_tensor_clone(const marmot_tensor_t *src);

//------------------------------------------------------------------------------
// Memory management
//------------------------------------------------------------------------------
MARMOT_NODISCARD marmot_error_t marmot_tensor_to_device(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD marmot_error_t marmot_tensor_to_host(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD marmot_error_t marmot_tensor_to_device_async(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD marmot_error_t marmot_tensor_to_host_async(const marmot_context_t *ctx, marmot_tensor_t *tensor);
marmot_memory_location_t marmot_tensor_memory_location(const marmot_tensor_t *tensor);
bool marmot_tensor_is_ready(const marmot_context_t *ctx, const marmot_tensor_t *tensor);
MARMOT_NODISCARD marmot_error_t
marmot_tensor_copy_to_host_buffer(const marmot_context_t *ctx, const marmot_tensor_t *tensor, void *dst, size_t bytes);
MARMOT_NODISCARD marmot_error_t marmot_tensor_copy_from_host_buffer(
    const marmot_context_t *ctx, marmot_tensor_t *tensor, const void *src, size_t bytes
);

//------------------------------------------------------------------------------
// Data access - read-only (implicit GPU->host sync)
//------------------------------------------------------------------------------
MARMOT_NODISCARD const void *marmot_tensor_data(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD const float *marmot_tensor_data_f32(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD const double *marmot_tensor_data_f64(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD const marmot_float16_t *marmot_tensor_data_f16(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD const marmot_bfloat16_t *marmot_tensor_data_bf16(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD const marmot_int8_t *marmot_tensor_data_i8(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD const marmot_int16_t *marmot_tensor_data_i16(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD const marmot_int32_t *marmot_tensor_data_i32(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD const marmot_int64_t *marmot_tensor_data_i64(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD const marmot_uint8_t *marmot_tensor_data_u8(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD const marmot_uint16_t *marmot_tensor_data_u16(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD const marmot_uint32_t *marmot_tensor_data_u32(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD const marmot_uint64_t *marmot_tensor_data_u64(const marmot_context_t *ctx, marmot_tensor_t *tensor);
#if MARMOT_ENABLE_FP8
MARMOT_NODISCARD const marmot_float8_e4m3_t *
marmot_tensor_data_fp8_e4m3(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD const marmot_float8_e5m2_t *
marmot_tensor_data_fp8_e5m2(const marmot_context_t *ctx, marmot_tensor_t *tensor);
#endif

//------------------------------------------------------------------------------
// Data access - mutable (implicit GPU->host sync, marks host-dirty)
//------------------------------------------------------------------------------
MARMOT_NODISCARD void *marmot_tensor_data_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD float *marmot_tensor_data_f32_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD double *marmot_tensor_data_f64_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD marmot_float16_t *marmot_tensor_data_f16_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD marmot_bfloat16_t *marmot_tensor_data_bf16_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD marmot_int8_t *marmot_tensor_data_i8_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD marmot_int16_t *marmot_tensor_data_i16_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD marmot_int32_t *marmot_tensor_data_i32_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD marmot_int64_t *marmot_tensor_data_i64_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD marmot_uint8_t *marmot_tensor_data_u8_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD marmot_uint16_t *marmot_tensor_data_u16_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD marmot_uint32_t *marmot_tensor_data_u32_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD marmot_uint64_t *marmot_tensor_data_u64_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor);
#if MARMOT_ENABLE_FP8
MARMOT_NODISCARD marmot_float8_e4m3_t *
marmot_tensor_data_fp8_e4m3_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor);
MARMOT_NODISCARD marmot_float8_e5m2_t *
marmot_tensor_data_fp8_e5m2_mut(const marmot_context_t *ctx, marmot_tensor_t *tensor);
#endif

//------------------------------------------------------------------------------
// Utilities
//------------------------------------------------------------------------------
void marmot_tensor_print(const marmot_tensor_t *tensor, const char *name);

//------------------------------------------------------------------------------
// Forward declarations for integer wrapper implementations used by generic macros
//------------------------------------------------------------------------------
marmot_tensor_t *
marmot_tensor_full_i32(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_int32_t value);
marmot_tensor_t *
marmot_tensor_full_i16(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_int16_t value);
marmot_tensor_t *
marmot_tensor_full_i8(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_int8_t value);
marmot_tensor_t *
marmot_tensor_full_u8(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_uint8_t value);
marmot_tensor_t *
marmot_tensor_full_u16(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_uint16_t value);
marmot_tensor_t *
marmot_tensor_full_u32(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_uint32_t value);
marmot_tensor_t *
marmot_tensor_full_u64(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_uint64_t value);
#if MARMOT_ENABLE_FP8
marmot_tensor_t *
marmot_tensor_full_fp8_e4m3(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_float8_e4m3_t value);
marmot_tensor_t *
marmot_tensor_full_fp8_e5m2(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_float8_e5m2_t value);
#endif
void marmot_tensor_fill_i32(marmot_tensor_t *tensor, marmot_int32_t value);
void marmot_tensor_fill_i16(marmot_tensor_t *tensor, marmot_int16_t value);
void marmot_tensor_fill_i8(marmot_tensor_t *tensor, marmot_int8_t value);
void marmot_tensor_fill_u8(marmot_tensor_t *tensor, marmot_uint8_t value);
void marmot_tensor_fill_u16(marmot_tensor_t *tensor, marmot_uint16_t value);
void marmot_tensor_fill_u32(marmot_tensor_t *tensor, marmot_uint32_t value);
void marmot_tensor_fill_u64(marmot_tensor_t *tensor, marmot_uint64_t value);
#if MARMOT_ENABLE_FP8
void marmot_tensor_fill_fp8_e4m3(marmot_tensor_t *tensor, marmot_float8_e4m3_t value);
void marmot_tensor_fill_fp8_e5m2(marmot_tensor_t *tensor, marmot_float8_e5m2_t value);
#endif

//==============================================================================
// PUBLIC API: Type-generic macros (type-safe entry points)
//==============================================================================

static inline marmot_tensor_t *
marmot_tensor_full_i32_from_scalar_impl(const marmot_context_t *ctx, const size_t *shape, size_t ndim, int32_t value) {
    return marmot_tensor_full_i32(ctx, shape, ndim, MARMOT_I32(value));
}

static inline marmot_tensor_t *
marmot_tensor_full_i16_from_scalar_impl(const marmot_context_t *ctx, const size_t *shape, size_t ndim, int16_t value) {
    return marmot_tensor_full_i16(ctx, shape, ndim, MARMOT_I16(value));
}

static inline marmot_tensor_t *
marmot_tensor_full_i8_from_scalar_impl(const marmot_context_t *ctx, const size_t *shape, size_t ndim, int8_t value) {
    return marmot_tensor_full_i8(ctx, shape, ndim, MARMOT_I8(value));
}

static inline marmot_tensor_t *
marmot_tensor_full_u8_from_scalar_impl(const marmot_context_t *ctx, const size_t *shape, size_t ndim, uint8_t value) {
    return marmot_tensor_full_u8(ctx, shape, ndim, MARMOT_U8(value));
}

static inline marmot_tensor_t *
marmot_tensor_full_u16_from_scalar_impl(const marmot_context_t *ctx, const size_t *shape, size_t ndim, uint16_t value) {
    return marmot_tensor_full_u16(ctx, shape, ndim, MARMOT_U16(value));
}

static inline marmot_tensor_t *
marmot_tensor_full_u32_from_scalar_impl(const marmot_context_t *ctx, const size_t *shape, size_t ndim, uint32_t value) {
    return marmot_tensor_full_u32(ctx, shape, ndim, MARMOT_U32(value));
}

static inline marmot_tensor_t *
marmot_tensor_full_u64_from_scalar_impl(const marmot_context_t *ctx, const size_t *shape, size_t ndim, uint64_t value) {
    return marmot_tensor_full_u64(ctx, shape, ndim, MARMOT_U64(value));
}

#if MARMOT_ENABLE_FP8
#define MARMOT_TENSOR_FULL_FP8_CASES                                                                                   \
    marmot_float8_e4m3_t:                                                                                              \
    marmot_tensor_full_fp8_e4m3, marmot_float8_e5m2_t : marmot_tensor_full_fp8_e5m2,
#define MARMOT_TENSOR_FILL_FP8_CASES                                                                                   \
    marmot_float8_e4m3_t:                                                                                              \
    marmot_tensor_fill_fp8_e4m3, marmot_float8_e5m2_t : marmot_tensor_fill_fp8_e5m2,
#else
#define MARMOT_TENSOR_FULL_FP8_CASES
#define MARMOT_TENSOR_FILL_FP8_CASES
#endif

#define marmot_tensor_full(ctx, shape, ndim, value)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            \
    _Generic((value), float: marmot_tensor_full_f32, marmot_float16_t: marmot_tensor_full_f16, marmot_bfloat16_t: marmot_tensor_full_bf16, marmot_int32_t: marmot_tensor_full_i32, marmot_int16_t: marmot_tensor_full_i16, marmot_int8_t: marmot_tensor_full_i8, marmot_uint8_t: marmot_tensor_full_u8, marmot_uint16_t: marmot_tensor_full_u16, marmot_uint32_t: marmot_tensor_full_u32, marmot_uint64_t: marmot_tensor_full_u64, MARMOT_TENSOR_FULL_FP8_CASES int32_t: marmot_tensor_full_i32_from_scalar_impl, int16_t: marmot_tensor_full_i16_from_scalar_impl, int8_t: marmot_tensor_full_i8_from_scalar_impl, uint8_t: marmot_tensor_full_u8_from_scalar_impl, uint16_t: marmot_tensor_full_u16_from_scalar_impl, uint32_t: marmot_tensor_full_u32_from_scalar_impl, uint64_t: marmot_tensor_full_u64_from_scalar_impl)( \
        ctx, shape, ndim, value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
    )

// Type-generic fill using _Generic
// Note: Use explicit float literals (e.g., 2.5f not 2.5) for FLOAT32 to avoid compile errors
static inline void marmot_tensor_fill_i32_scalar_impl(marmot_tensor_t *tensor, int32_t value) {
    marmot_tensor_fill_i32(tensor, MARMOT_I32(value));
}

static inline void marmot_tensor_fill_i16_scalar_impl(marmot_tensor_t *tensor, int16_t value) {
    marmot_tensor_fill_i16(tensor, MARMOT_I16(value));
}

static inline void marmot_tensor_fill_i8_scalar_impl(marmot_tensor_t *tensor, int8_t value) {
    marmot_tensor_fill_i8(tensor, MARMOT_I8(value));
}

static inline void marmot_tensor_fill_u8_scalar_impl(marmot_tensor_t *tensor, uint8_t value) {
    marmot_tensor_fill_u8(tensor, MARMOT_U8(value));
}

static inline void marmot_tensor_fill_u16_scalar_impl(marmot_tensor_t *tensor, uint16_t value) {
    marmot_tensor_fill_u16(tensor, MARMOT_U16(value));
}

static inline void marmot_tensor_fill_u32_scalar_impl(marmot_tensor_t *tensor, uint32_t value) {
    marmot_tensor_fill_u32(tensor, MARMOT_U32(value));
}

static inline void marmot_tensor_fill_u64_scalar_impl(marmot_tensor_t *tensor, uint64_t value) {
    marmot_tensor_fill_u64(tensor, MARMOT_U64(value));
}

#define marmot_tensor_fill(tensor, value)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   \
    _Generic((value), float: marmot_tensor_fill_f32, marmot_float16_t: marmot_tensor_fill_f16, marmot_bfloat16_t: marmot_tensor_fill_bf16, marmot_int32_t: marmot_tensor_fill_i32, marmot_int16_t: marmot_tensor_fill_i16, marmot_int8_t: marmot_tensor_fill_i8, marmot_uint8_t: marmot_tensor_fill_u8, marmot_uint16_t: marmot_tensor_fill_u16, marmot_uint32_t: marmot_tensor_fill_u32, marmot_uint64_t: marmot_tensor_fill_u64, MARMOT_TENSOR_FILL_FP8_CASES int32_t: marmot_tensor_fill_i32_scalar_impl, int16_t: marmot_tensor_fill_i16_scalar_impl, int8_t: marmot_tensor_fill_i8_scalar_impl, uint8_t: marmot_tensor_fill_u8_scalar_impl, uint16_t: marmot_tensor_fill_u16_scalar_impl, uint32_t: marmot_tensor_fill_u32_scalar_impl, uint64_t: marmot_tensor_fill_u64_scalar_impl)( \
        tensor, value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \
    )

// ===================================================================
// INTERNAL: Type-specific implementations
// These functions are called by the generic macros above.
// DO NOT call these directly - use the macros instead.
// ===================================================================

// Internal: fill functions
void marmot_tensor_fill_f32(marmot_tensor_t *tensor, float value);
void marmot_tensor_fill_f16(marmot_tensor_t *tensor, marmot_float16_t value);
void marmot_tensor_fill_bf16(marmot_tensor_t *tensor, marmot_bfloat16_t value);
void marmot_tensor_fill_i32(marmot_tensor_t *tensor, marmot_int32_t value);
void marmot_tensor_fill_i16(marmot_tensor_t *tensor, marmot_int16_t value);
void marmot_tensor_fill_i8(marmot_tensor_t *tensor, marmot_int8_t value);
void marmot_tensor_fill_u8(marmot_tensor_t *tensor, marmot_uint8_t value);
void marmot_tensor_fill_u16(marmot_tensor_t *tensor, marmot_uint16_t value);
void marmot_tensor_fill_u32(marmot_tensor_t *tensor, marmot_uint32_t value);
void marmot_tensor_fill_u64(marmot_tensor_t *tensor, marmot_uint64_t value);
#if MARMOT_ENABLE_FP8
void marmot_tensor_fill_fp8_e4m3(marmot_tensor_t *tensor, marmot_float8_e4m3_t value);
void marmot_tensor_fill_fp8_e5m2(marmot_tensor_t *tensor, marmot_float8_e5m2_t value);
#endif

// Internal: full tensor creation
marmot_tensor_t *marmot_tensor_full_f32(const marmot_context_t *ctx, const size_t *shape, size_t ndim, float value);
marmot_tensor_t *
marmot_tensor_full_f16(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_float16_t value);
marmot_tensor_t *
marmot_tensor_full_bf16(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_bfloat16_t value);
marmot_tensor_t *
marmot_tensor_full_i32(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_int32_t value);
marmot_tensor_t *
marmot_tensor_full_i16(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_int16_t value);
marmot_tensor_t *
marmot_tensor_full_i8(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_int8_t value);
marmot_tensor_t *
marmot_tensor_full_u8(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_uint8_t value);
marmot_tensor_t *
marmot_tensor_full_u16(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_uint16_t value);
marmot_tensor_t *
marmot_tensor_full_u32(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_uint32_t value);
marmot_tensor_t *
marmot_tensor_full_u64(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_uint64_t value);
#if MARMOT_ENABLE_FP8
marmot_tensor_t *
marmot_tensor_full_fp8_e4m3(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_float8_e4m3_t value);
marmot_tensor_t *
marmot_tensor_full_fp8_e5m2(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_float8_e5m2_t value);
#endif
// For higher-level tensor operations include <marmot/ops.h> or the specific headers in <marmot/ops/*.h>.

#ifdef __cplusplus
}
#endif

#endif // MARMOT_TENSOR_H
