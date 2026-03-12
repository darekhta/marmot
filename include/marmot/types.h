#ifndef MARMOT_TYPES_H
#define MARMOT_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "macros.h"

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// Fundamental constants
//==============================================================================

enum { MARMOT_MAX_DIMS = 8 };

//==============================================================================
// Scalar wrapper types (public API)
//==============================================================================
// Wrapping keeps the C23 _Generic API precise without introducing runtime cost.
//==============================================================================

typedef struct {
    uint16_t bits;
} marmot_float16_t;

typedef struct {
    uint16_t bits;
} marmot_bfloat16_t;

#if MARMOT_ENABLE_FP8
typedef struct {
    uint8_t bits;
} marmot_float8_e4m3_t;

typedef struct {
    uint8_t bits;
} marmot_float8_e5m2_t;
#endif

#if MARMOT_HAS_BITINT
typedef struct {
    union {
        signed _BitInt(4) value;
        uint8_t bits;
    };
} marmot_int4_t;

typedef struct {
    union {
        unsigned _BitInt(4) value;
        uint8_t bits;
    };
} marmot_uint4_t;
#else
typedef struct {
    uint8_t bits;
} marmot_int4_t;

typedef struct {
    uint8_t bits;
} marmot_uint4_t;
#endif

typedef struct {
    int32_t value;
} marmot_int32_t;

typedef struct {
    int16_t value;
} marmot_int16_t;

typedef struct {
    int8_t value;
} marmot_int8_t;

typedef struct {
    uint8_t value;
} marmot_uint8_t;

typedef struct {
    uint16_t value;
} marmot_uint16_t;

typedef struct {
    uint32_t value;
} marmot_uint32_t;

typedef struct {
    uint64_t value;
} marmot_uint64_t;

typedef struct {
    int64_t value;
} marmot_int64_t;

//==============================================================================
// Helper factories for wrapper literals
//==============================================================================

static inline marmot_float16_t marmot_make_f16(uint16_t bits) {
    marmot_float16_t v = {bits};
    return v;
}

static inline marmot_bfloat16_t marmot_make_bf16(uint16_t bits) {
    marmot_bfloat16_t v = {bits};
    return v;
}

#if MARMOT_ENABLE_FP8
static inline marmot_float8_e4m3_t marmot_make_fp8_e4m3(uint8_t bits) {
    marmot_float8_e4m3_t v = {bits};
    return v;
}

static inline marmot_float8_e5m2_t marmot_make_fp8_e5m2(uint8_t bits) {
    marmot_float8_e5m2_t v = {bits};
    return v;
}
#endif

static inline marmot_int32_t marmot_make_i32(int32_t value) {
    marmot_int32_t v;
    v.value = value;
    return v;
}

static inline marmot_int16_t marmot_make_i16(int16_t value) {
    marmot_int16_t v;
    v.value = value;
    return v;
}

static inline marmot_int8_t marmot_make_i8(int8_t value) {
    marmot_int8_t v;
    v.value = value;
    return v;
}

static inline marmot_uint8_t marmot_make_u8(uint8_t value) {
    marmot_uint8_t v;
    v.value = value;
    return v;
}

static inline marmot_uint16_t marmot_make_u16(uint16_t value) {
    marmot_uint16_t v;
    v.value = value;
    return v;
}

static inline marmot_uint32_t marmot_make_u32(uint32_t value) {
    marmot_uint32_t v;
    v.value = value;
    return v;
}

static inline marmot_uint64_t marmot_make_u64(uint64_t value) {
    marmot_uint64_t v;
    v.value = value;
    return v;
}

static inline marmot_int64_t marmot_make_i64(int64_t value) {
    marmot_int64_t v;
    v.value = value;
    return v;
}

#define MARMOT_F16(hex_bits) marmot_make_f16((uint16_t)(hex_bits))
#define MARMOT_BF16(hex_bits) marmot_make_bf16((uint16_t)(hex_bits))
#if MARMOT_ENABLE_FP8
#define MARMOT_FP8_E4M3(hex_bits) marmot_make_fp8_e4m3((uint8_t)(hex_bits))
#define MARMOT_FP8_E5M2(hex_bits) marmot_make_fp8_e5m2((uint8_t)(hex_bits))
#endif
#define MARMOT_I32(value) marmot_make_i32((int32_t)(value))
#define MARMOT_I16(value) marmot_make_i16((int16_t)(value))
#define MARMOT_I8(value) marmot_make_i8((int8_t)(value))
#define MARMOT_U8(value) marmot_make_u8((uint8_t)(value))
#define MARMOT_U16(value) marmot_make_u16((uint16_t)(value))
#define MARMOT_U32(value) marmot_make_u32((uint32_t)(value))
#define MARMOT_U64(value) marmot_make_u64((uint64_t)(value))
#define MARMOT_I64(value) marmot_make_i64((int64_t)(value))

static_assert(sizeof(marmot_float16_t) == sizeof(uint16_t), "marmot_float16_t must be 16 bits");
static_assert(sizeof(marmot_bfloat16_t) == sizeof(uint16_t), "marmot_bfloat16_t must be 16 bits");
#if MARMOT_ENABLE_FP8
static_assert(sizeof(marmot_float8_e4m3_t) == sizeof(uint8_t), "marmot_float8_e4m3_t must be 8 bits");
static_assert(sizeof(marmot_float8_e5m2_t) == sizeof(uint8_t), "marmot_float8_e5m2_t must be 8 bits");
#endif
static_assert(sizeof(marmot_int32_t) == sizeof(int32_t), "marmot_int32_t must match int32_t size");
static_assert(sizeof(marmot_int16_t) == sizeof(int16_t), "marmot_int16_t must match int16_t size");
static_assert(sizeof(marmot_int8_t) == sizeof(int8_t), "marmot_int8_t must match int8_t size");
static_assert(sizeof(marmot_uint8_t) == sizeof(uint8_t), "marmot_uint8_t must match uint8_t size");
static_assert(sizeof(marmot_uint16_t) == sizeof(uint16_t), "marmot_uint16_t must match uint16_t size");
static_assert(sizeof(marmot_uint32_t) == sizeof(uint32_t), "marmot_uint32_t must match uint32_t size");
static_assert(sizeof(marmot_uint64_t) == sizeof(uint64_t), "marmot_uint64_t must match uint64_t size");
static_assert(sizeof(marmot_int64_t) == sizeof(int64_t), "marmot_int64_t must match int64_t size");
static_assert(sizeof(marmot_int4_t) == 1, "marmot_int4_t must occupy 1 byte");
static_assert(sizeof(marmot_uint4_t) == 1, "marmot_uint4_t must occupy 1 byte");

//==============================================================================
// Error codes and enums
//==============================================================================

typedef enum {
    MARMOT_SUCCESS = 0,
    MARMOT_ERROR_INVALID_ARGUMENT = -1,
    MARMOT_ERROR_OUT_OF_MEMORY = -2,
    MARMOT_ERROR_DEVICE_NOT_AVAILABLE = -3,
    MARMOT_ERROR_BACKEND_INIT_FAILED = -4,
    MARMOT_ERROR_INVALID_OPERATION = -5,
    MARMOT_ERROR_UNSUPPORTED_DTYPE = -6,
    MARMOT_ERROR_DIMENSION_MISMATCH = -7,
    MARMOT_ERROR_NOT_IMPLEMENTED = -8,
} marmot_error_t;

typedef enum {
    MARMOT_BACKEND_CPU = 0,
    MARMOT_BACKEND_METAL = 1,
    MARMOT_BACKEND_CUDA = 2,
} marmot_backend_type_t;

enum { MARMOT_BACKEND_COUNT = 3 };

typedef enum {
    MARMOT_ROUTING_ALWAYS_CPU = 0,
    MARMOT_ROUTING_ALWAYS_GPU = 1,
    MARMOT_ROUTING_AUTO = 2,
} marmot_routing_policy_t;

typedef enum {
    MARMOT_PREFERENCE_DEFAULT = 0,
    MARMOT_PREFERENCE_ENABLE = 1,
    MARMOT_PREFERENCE_DISABLE = 2,
} marmot_preference_t;

static inline bool marmot_preference_resolve(marmot_preference_t preference, bool default_value) {
    switch (preference) {
    case MARMOT_PREFERENCE_ENABLE:
        return true;
    case MARMOT_PREFERENCE_DISABLE:
        return false;
    case MARMOT_PREFERENCE_DEFAULT:
    default:
        return default_value;
    }
}

typedef enum {
    MARMOT_ROPE_SCALING_UNSPECIFIED = -1,
    MARMOT_ROPE_SCALING_NONE = 0,
    MARMOT_ROPE_SCALING_LINEAR = 1,
    MARMOT_ROPE_SCALING_YARN = 2,
    MARMOT_ROPE_SCALING_LONGROPE = 3,
    MARMOT_ROPE_SCALING_MAX_VALUE = MARMOT_ROPE_SCALING_LONGROPE,
} marmot_rope_scaling_type_t;

typedef enum {
    MARMOT_ROPE_TYPE_NORM = 0,
    MARMOT_ROPE_TYPE_NEOX = 1,
    MARMOT_ROPE_TYPE_MAX_VALUE = MARMOT_ROPE_TYPE_NEOX,
} marmot_rope_type_t;

typedef enum {
    MARMOT_ARCH_UNKNOWN = 0,
    MARMOT_ARCH_LLAMA = 1,
    MARMOT_ARCH_MISTRAL = 2,
    MARMOT_ARCH_QWEN2 = 3,
    MARMOT_ARCH_PHI3 = 4,
    MARMOT_ARCH_GEMMA = 5,
    MARMOT_ARCH_QWEN3 = 6,
    MARMOT_ARCH_QWEN3MOE = 7,
    MARMOT_ARCH_COUNT,
} marmot_architecture_t;

typedef enum {
    MARMOT_FFN_SWIGLU = 0,
    MARMOT_FFN_GELU = 1,
    MARMOT_FFN_GEGLU = 2,
    MARMOT_FFN_COUNT,
} marmot_ffn_type_t;

typedef enum {
    MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED = 0,
    MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED_SCALED = 1,
    MARMOT_ROUTER_WEIGHT_POLICY_RENORMALIZE_SELECTED = 2,
    MARMOT_ROUTER_WEIGHT_POLICY_COUNT,
} marmot_router_weight_policy_t;

typedef enum {
    MARMOT_FAST_STAGE_HINT_NONE = 0,
    MARMOT_FAST_STAGE_HINT_ATTENTION = 1,
    MARMOT_FAST_STAGE_HINT_DENSE_FFN = 2,
    MARMOT_FAST_STAGE_HINT_MOE_FFN = 3,
    MARMOT_FAST_STAGE_HINT_LOGITS_TAIL = 4,
    MARMOT_FAST_STAGE_HINT_COUNT,
} marmot_fast_stage_hint_t;

typedef enum {
    MARMOT_FAST_NODE_ROLE_NONE = 0,
    MARMOT_FAST_NODE_ROLE_ATTN_NORM = 1,
    MARMOT_FAST_NODE_ROLE_ATTN_Q_PROJ = 2,
    MARMOT_FAST_NODE_ROLE_ATTN_K_PROJ = 3,
    MARMOT_FAST_NODE_ROLE_ATTN_V_PROJ = 4,
    MARMOT_FAST_NODE_ROLE_ATTN_Q_NORM_RESHAPE = 5,
    MARMOT_FAST_NODE_ROLE_ATTN_Q_NORM = 6,
    MARMOT_FAST_NODE_ROLE_ATTN_K_NORM_RESHAPE = 7,
    MARMOT_FAST_NODE_ROLE_ATTN_K_NORM = 8,
    MARMOT_FAST_NODE_ROLE_ATTN_Q_HEADS = 9,
    MARMOT_FAST_NODE_ROLE_ATTN_K_HEADS = 10,
    MARMOT_FAST_NODE_ROLE_ATTN_V_HEADS = 11,
    MARMOT_FAST_NODE_ROLE_ATTN_Q_ROPE = 12,
    MARMOT_FAST_NODE_ROLE_ATTN_K_ROPE = 13,
    MARMOT_FAST_NODE_ROLE_ATTN_Q_TOKENS = 14,
    MARMOT_FAST_NODE_ROLE_ATTN_K_TOKENS = 15,
    MARMOT_FAST_NODE_ROLE_ATTN_V_TOKENS = 16,
    MARMOT_FAST_NODE_ROLE_ATTN_PAGED = 17,
    MARMOT_FAST_NODE_ROLE_ATTN_OUT_RESHAPE = 18,
    MARMOT_FAST_NODE_ROLE_ATTN_OUT_PROJ = 19,
    MARMOT_FAST_NODE_ROLE_ATTN_RESIDUAL = 20,
    MARMOT_FAST_NODE_ROLE_FFN_NORM = 21,
    MARMOT_FAST_NODE_ROLE_FFN_GATE_PROJ = 22,
    MARMOT_FAST_NODE_ROLE_FFN_UP_PROJ = 23,
    MARMOT_FAST_NODE_ROLE_FFN_GELU = 24,
    MARMOT_FAST_NODE_ROLE_FFN_GLU = 25,
    MARMOT_FAST_NODE_ROLE_FFN_DOWN_PROJ = 26,
    MARMOT_FAST_NODE_ROLE_FFN_ROUTER = 27,
    MARMOT_FAST_NODE_ROLE_FFN_ROUTER_PROBS = 28,
    MARMOT_FAST_NODE_ROLE_FFN_TOPK = 29,
    MARMOT_FAST_NODE_ROLE_FFN_ROUTER_WEIGHTS = 30,
    MARMOT_FAST_NODE_ROLE_FFN_MOE_EXPERTS = 31,
    MARMOT_FAST_NODE_ROLE_FFN_RESIDUAL = 32,
    MARMOT_FAST_NODE_ROLE_LOGITS_NORM = 33,
    MARMOT_FAST_NODE_ROLE_LOGITS_GATHER = 34,
    MARMOT_FAST_NODE_ROLE_LOGITS_PROJECTION = 35,
    MARMOT_FAST_NODE_ROLE_COUNT,
} marmot_fast_node_role_t;

typedef enum {
    MARMOT_DTYPE_FLOAT32 = 0,
    MARMOT_DTYPE_FLOAT16 = 1,
    MARMOT_DTYPE_BFLOAT16 = 2,
    MARMOT_DTYPE_INT32 = 3,
    MARMOT_DTYPE_INT16 = 4,
    MARMOT_DTYPE_INT8 = 5,
    MARMOT_DTYPE_UINT8 = 6,
    MARMOT_DTYPE_UINT16 = 7,
    MARMOT_DTYPE_UINT32 = 8,
    MARMOT_DTYPE_UINT64 = 9,
    MARMOT_DTYPE_FLOAT64 = 10,
    MARMOT_DTYPE_INT64 = 11,
    MARMOT_DTYPE_INT4 = 12,
    MARMOT_DTYPE_UINT4 = 13,
#if MARMOT_ENABLE_FP8
    MARMOT_DTYPE_FLOAT8_E4M3 = 14,
    MARMOT_DTYPE_FLOAT8_E5M2 = 15,
#endif
    MARMOT_DTYPE_COUNT,
} marmot_dtype_t;

typedef enum {
    MARMOT_QUANT_LAYOUT_GENERIC = 0,
    MARMOT_QUANT_LAYOUT_GGUF = 1,
    MARMOT_QUANT_LAYOUT_COUNT,
} marmot_quant_layout_t;

typedef enum {
    MARMOT_QUANT_KIND_GENERIC = 0,
    MARMOT_QUANT_KIND_Q4_0 = 1,
    MARMOT_QUANT_KIND_Q4_1 = 2,
    MARMOT_QUANT_KIND_Q5_0 = 3,
    MARMOT_QUANT_KIND_Q5_1 = 4,
    MARMOT_QUANT_KIND_Q8_0 = 5,
    MARMOT_QUANT_KIND_Q8_1 = 6,
    // K‑Quant super‑blocks (256‑value groups)
    MARMOT_QUANT_KIND_Q2_K = 7,
    MARMOT_QUANT_KIND_Q3_K = 8,
    MARMOT_QUANT_KIND_Q4_K = 9,
    MARMOT_QUANT_KIND_Q5_K = 10,
    MARMOT_QUANT_KIND_Q6_K = 11,
    MARMOT_QUANT_KIND_Q8_K = 12,
    MARMOT_QUANT_KIND_COUNT,
} marmot_quant_kind_t;

typedef enum {
    MARMOT_CPU_UNKNOWN = 0,
    // Apple Silicon
    MARMOT_CPU_APPLE_M1 = 100,
    MARMOT_CPU_APPLE_M2 = 101,
    MARMOT_CPU_APPLE_M3 = 102,
    MARMOT_CPU_APPLE_M4 = 103,
    // ARM Cortex-A
    MARMOT_CPU_CORTEX_A53 = 200,
    MARMOT_CPU_CORTEX_A55 = 201,
    MARMOT_CPU_CORTEX_A57 = 202,
    MARMOT_CPU_CORTEX_A72 = 203,
    MARMOT_CPU_CORTEX_A76 = 204,
    MARMOT_CPU_CORTEX_X1 = 205,
    MARMOT_CPU_CORTEX_X2 = 206,
    // ARM Neoverse
    MARMOT_CPU_NEOVERSE_N1 = 300,
    MARMOT_CPU_NEOVERSE_N2 = 301,
    MARMOT_CPU_NEOVERSE_V1 = 302,
} marmot_cpu_microarch_t;

//==============================================================================
// Core structs
//==============================================================================

typedef struct {
    size_t ndim;
    size_t shape[MARMOT_MAX_DIMS];
    size_t strides[MARMOT_MAX_DIMS];
} marmot_shape_t;

typedef struct {
    float scale;
    float zero_point;
    size_t block_size; // 0 = per-tensor quantization
} marmot_quant_params_t;

typedef struct marmot_quant_kind_traits {
    marmot_quant_kind_t kind;
    marmot_dtype_t storage_dtype;
    size_t block_values;
    size_t header_bytes;
    size_t payload_bytes;
    bool payload_signed;
    bool is_block_quantized;
    bool is_bit_packed;
    marmot_quant_layout_t layout;
} marmot_quant_kind_traits_t;

MARMOT_NODISCARD const marmot_quant_kind_traits_t *marmot_get_quant_kind_traits(marmot_quant_kind_t kind);
MARMOT_NODISCARD bool marmot_quant_kind_is_block_quantized(marmot_quant_kind_t kind);

typedef struct marmot_context marmot_context_t;
typedef struct marmot_tensor marmot_tensor_t;
typedef struct marmot_device_ops marmot_device_ops_t;
typedef struct marmot_activation_params marmot_activation_params_t;
typedef struct marmot_matmul_epilogue marmot_matmul_epilogue_t;
typedef struct marmot_rope_params marmot_rope_params_t;
typedef struct marmot_matmul_qkv_desc marmot_matmul_qkv_desc_t;
typedef struct marmot_reduction_params marmot_reduction_params_t;
typedef struct marmot_vec_dot_descriptor marmot_vec_dot_descriptor_t;
typedef struct marmot_embedding_desc marmot_embedding_desc_t;
typedef struct marmot_embedding_gather_desc marmot_embedding_gather_desc_t;
typedef struct marmot_layernorm_desc marmot_layernorm_desc_t;
typedef struct marmot_rmsnorm_desc marmot_rmsnorm_desc_t;
typedef struct marmot_softmax_desc marmot_softmax_desc_t;
typedef struct marmot_topk_desc marmot_topk_desc_t;
typedef struct marmot_moe_experts_desc marmot_moe_experts_desc_t;
typedef struct marmot_reduction_desc marmot_reduction_desc_t;

//==============================================================================
// Dtype trait definitions
//==============================================================================

#if MARMOT_ENABLE_FP8
#define MARMOT_FOREACH_FP8_FLOATING_DTYPE(M)                                                                           \
    M(MARMOT_DTYPE_FLOAT8_E4M3, marmot_float8_e4m3_t, fp8_e4m3)                                                        \
    M(MARMOT_DTYPE_FLOAT8_E5M2, marmot_float8_e5m2_t, fp8_e5m2)
#else
#define MARMOT_FOREACH_FP8_FLOATING_DTYPE(M)
#endif

#define MARMOT_FOREACH_FLOATING_DTYPE(M)                                                                               \
    M(MARMOT_DTYPE_FLOAT32, float, f32)                                                                                \
    M(MARMOT_DTYPE_FLOAT16, marmot_float16_t, f16)                                                                     \
    M(MARMOT_DTYPE_BFLOAT16, marmot_bfloat16_t, bf16)                                                                  \
    MARMOT_FOREACH_FP8_FLOATING_DTYPE(M)

#define MARMOT_FOREACH_INTEGER_DTYPE(M)                                                                                \
    M(MARMOT_DTYPE_INT32, marmot_int32_t, i32)                                                                         \
    M(MARMOT_DTYPE_UINT32, marmot_uint32_t, u32)                                                                       \
    M(MARMOT_DTYPE_INT16, marmot_int16_t, i16)                                                                         \
    M(MARMOT_DTYPE_UINT16, marmot_uint16_t, u16)                                                                       \
    M(MARMOT_DTYPE_INT8, marmot_int8_t, i8)                                                                            \
    M(MARMOT_DTYPE_UINT8, marmot_uint8_t, u8)                                                                          \
    M(MARMOT_DTYPE_UINT64, marmot_uint64_t, u64)

#define MARMOT_FOREACH_PACKED_DTYPE(M)                                                                                 \
    M(MARMOT_DTYPE_INT4, marmot_int4_t, i4)                                                                            \
    M(MARMOT_DTYPE_UINT4, marmot_uint4_t, u4)

#define MARMOT_FOREACH_PRIMITIVE_DTYPE(M)                                                                              \
    MARMOT_FOREACH_FLOATING_DTYPE(M)                                                                                   \
    MARMOT_FOREACH_INTEGER_DTYPE(M)

#define MARMOT_FOREACH_STORAGE_DTYPE(M)                                                                                \
    MARMOT_FOREACH_PRIMITIVE_DTYPE(M)                                                                                  \
    MARMOT_FOREACH_PACKED_DTYPE(M)

typedef struct {
    marmot_dtype_t id;
    const char *name;
    size_t storage_bytes;
    size_t element_bits;
    size_t alignment;
    marmot_dtype_t compute_dtype;
    bool is_floating;
    bool is_signed;
    bool is_quantized;
    bool is_packed;
    bool has_cpu_support;
    bool has_metal_support;
    bool has_simd_support;
    bool supports_reduction;
    marmot_dtype_t reduction_accum_dtype;
} marmot_dtype_traits_t;

MARMOT_NODISCARD const marmot_dtype_traits_t *marmot_get_dtype_traits(marmot_dtype_t dtype);
MARMOT_NODISCARD bool marmot_dtype_is_floating(marmot_dtype_t dtype);
MARMOT_NODISCARD bool marmot_dtype_is_integer(marmot_dtype_t dtype);
MARMOT_NODISCARD bool marmot_dtype_is_signed(marmot_dtype_t dtype);
MARMOT_NODISCARD bool marmot_dtype_is_quantized(marmot_dtype_t dtype);
MARMOT_NODISCARD bool marmot_dtype_is_packed(marmot_dtype_t dtype);
MARMOT_NODISCARD size_t marmot_dtype_element_bits(marmot_dtype_t dtype);
MARMOT_NODISCARD size_t marmot_dtype_size(marmot_dtype_t dtype);
MARMOT_NODISCARD size_t marmot_dtype_alignment(marmot_dtype_t dtype);
MARMOT_NODISCARD marmot_dtype_t marmot_dtype_compute_dtype(marmot_dtype_t dtype);
MARMOT_NODISCARD bool marmot_dtype_has_cpu_support(marmot_dtype_t dtype);
MARMOT_NODISCARD bool marmot_dtype_has_metal_support(marmot_dtype_t dtype);
MARMOT_NODISCARD bool marmot_dtype_is_supported_on_backend(marmot_backend_type_t backend, marmot_dtype_t dtype);
MARMOT_NODISCARD bool marmot_dtype_supports_reduction(marmot_dtype_t dtype);
MARMOT_NODISCARD marmot_dtype_t marmot_dtype_reduction_accum_dtype(marmot_dtype_t dtype);

#if MARMOT_ENABLE_FP8
#define MARMOT_GENERIC_DTYPE_FP8_CASES                                                                                 \
    marmot_float8_e4m3_t:                                                                                              \
    MARMOT_DTYPE_FLOAT8_E4M3, marmot_float8_e5m2_t : MARMOT_DTYPE_FLOAT8_E5M2,
#define MARMOT_GENERIC_PTR_DTYPE_FP8_CASES                                                                             \
    marmot_float8_e4m3_t * : MARMOT_DTYPE_FLOAT8_E4M3,                                                                 \
                             const marmot_float8_e4m3_t * : MARMOT_DTYPE_FLOAT8_E4M3,                                  \
                                                            marmot_float8_e5m2_t * : MARMOT_DTYPE_FLOAT8_E5M2,         \
                                                                                     const marmot_float8_e5m2_t        \
                                                                                         * : MARMOT_DTYPE_FLOAT8_E5M2,
#define MARMOT_DISPATCH_PRIMITIVE_FP8_CASES(HANDLE_CASE)                                                               \
    case MARMOT_DTYPE_FLOAT8_E4M3: {                                                                                   \
        HANDLE_CASE(MARMOT_DTYPE_FLOAT8_E4M3, marmot_float8_e4m3_t, fp8_e4m3);                                         \
    } break;                                                                                                           \
    case MARMOT_DTYPE_FLOAT8_E5M2: {                                                                                   \
        HANDLE_CASE(MARMOT_DTYPE_FLOAT8_E5M2, marmot_float8_e5m2_t, fp8_e5m2);                                         \
    } break;
#else
#define MARMOT_GENERIC_DTYPE_FP8_CASES
#define MARMOT_GENERIC_PTR_DTYPE_FP8_CASES
#define MARMOT_DISPATCH_PRIMITIVE_FP8_CASES(HANDLE_CASE)
#endif

#define MARMOT_GENERIC_DTYPE(value)                                                                                    \
    _Generic(                                                                                                          \
        (value),                                                                                                       \
        float: MARMOT_DTYPE_FLOAT32,                                                                                   \
        marmot_float16_t: MARMOT_DTYPE_FLOAT16,                                                                        \
        marmot_bfloat16_t: MARMOT_DTYPE_BFLOAT16,                                                                      \
        MARMOT_GENERIC_DTYPE_FP8_CASES marmot_int32_t: MARMOT_DTYPE_INT32,                                             \
        marmot_int16_t: MARMOT_DTYPE_INT16,                                                                            \
        marmot_int8_t: MARMOT_DTYPE_INT8,                                                                              \
        marmot_uint64_t: MARMOT_DTYPE_UINT64,                                                                          \
        marmot_uint32_t: MARMOT_DTYPE_UINT32,                                                                          \
        marmot_uint16_t: MARMOT_DTYPE_UINT16,                                                                          \
        marmot_uint8_t: MARMOT_DTYPE_UINT8,                                                                            \
        marmot_int4_t: MARMOT_DTYPE_INT4,                                                                              \
        marmot_uint4_t: MARMOT_DTYPE_UINT4                                                                             \
    )

#define MARMOT_GENERIC_PTR_DTYPE(ptr)                                                                                  \
    _Generic(                                                                                                          \
        (ptr),                                                                                                         \
        float *: MARMOT_DTYPE_FLOAT32,                                                                                 \
        const float *: MARMOT_DTYPE_FLOAT32,                                                                           \
        marmot_float16_t *: MARMOT_DTYPE_FLOAT16,                                                                      \
        const marmot_float16_t *: MARMOT_DTYPE_FLOAT16,                                                                \
        marmot_bfloat16_t *: MARMOT_DTYPE_BFLOAT16,                                                                    \
        const marmot_bfloat16_t *: MARMOT_DTYPE_BFLOAT16,                                                              \
        MARMOT_GENERIC_PTR_DTYPE_FP8_CASES marmot_int32_t *: MARMOT_DTYPE_INT32,                                       \
        const marmot_int32_t *: MARMOT_DTYPE_INT32,                                                                    \
        marmot_int16_t *: MARMOT_DTYPE_INT16,                                                                          \
        const marmot_int16_t *: MARMOT_DTYPE_INT16,                                                                    \
        marmot_int8_t *: MARMOT_DTYPE_INT8,                                                                            \
        const marmot_int8_t *: MARMOT_DTYPE_INT8,                                                                      \
        marmot_uint64_t *: MARMOT_DTYPE_UINT64,                                                                        \
        const marmot_uint64_t *: MARMOT_DTYPE_UINT64,                                                                  \
        marmot_uint32_t *: MARMOT_DTYPE_UINT32,                                                                        \
        const marmot_uint32_t *: MARMOT_DTYPE_UINT32,                                                                  \
        marmot_uint16_t *: MARMOT_DTYPE_UINT16,                                                                        \
        const marmot_uint16_t *: MARMOT_DTYPE_UINT16,                                                                  \
        marmot_uint8_t *: MARMOT_DTYPE_UINT8,                                                                          \
        const marmot_uint8_t *: MARMOT_DTYPE_UINT8,                                                                    \
        marmot_int4_t *: MARMOT_DTYPE_INT4,                                                                            \
        const marmot_int4_t *: MARMOT_DTYPE_INT4,                                                                      \
        marmot_uint4_t *: MARMOT_DTYPE_UINT4,                                                                          \
        const marmot_uint4_t *: MARMOT_DTYPE_UINT4                                                                     \
    )

#define MARMOT_DISPATCH_PRIMITIVE(dtype_expr, HANDLE_CASE)                                                             \
    switch (dtype_expr) {                                                                                              \
    case MARMOT_DTYPE_FLOAT32: {                                                                                       \
        HANDLE_CASE(MARMOT_DTYPE_FLOAT32, float, f32);                                                                 \
    } break;                                                                                                           \
    case MARMOT_DTYPE_FLOAT16: {                                                                                       \
        HANDLE_CASE(MARMOT_DTYPE_FLOAT16, marmot_float16_t, f16);                                                      \
    } break;                                                                                                           \
    case MARMOT_DTYPE_BFLOAT16: {                                                                                      \
        HANDLE_CASE(MARMOT_DTYPE_BFLOAT16, marmot_bfloat16_t, bf16);                                                   \
    } break;                                                                                                           \
        MARMOT_DISPATCH_PRIMITIVE_FP8_CASES(HANDLE_CASE)                                                               \
    case MARMOT_DTYPE_INT32: {                                                                                         \
        HANDLE_CASE(MARMOT_DTYPE_INT32, marmot_int32_t, i32);                                                          \
    } break;                                                                                                           \
    case MARMOT_DTYPE_INT16: {                                                                                         \
        HANDLE_CASE(MARMOT_DTYPE_INT16, marmot_int16_t, i16);                                                          \
    } break;                                                                                                           \
    case MARMOT_DTYPE_INT8: {                                                                                          \
        HANDLE_CASE(MARMOT_DTYPE_INT8, marmot_int8_t, i8);                                                             \
    } break;                                                                                                           \
    case MARMOT_DTYPE_UINT64: {                                                                                        \
        HANDLE_CASE(MARMOT_DTYPE_UINT64, marmot_uint64_t, u64);                                                        \
    } break;                                                                                                           \
    case MARMOT_DTYPE_UINT32: {                                                                                        \
        HANDLE_CASE(MARMOT_DTYPE_UINT32, marmot_uint32_t, u32);                                                        \
    } break;                                                                                                           \
    case MARMOT_DTYPE_UINT16: {                                                                                        \
        HANDLE_CASE(MARMOT_DTYPE_UINT16, marmot_uint16_t, u16);                                                        \
    } break;                                                                                                           \
    case MARMOT_DTYPE_UINT8: {                                                                                         \
        HANDLE_CASE(MARMOT_DTYPE_UINT8, marmot_uint8_t, u8);                                                           \
    } break;                                                                                                           \
    default:                                                                                                           \
        break;                                                                                                         \
    }

#ifdef __cplusplus
}
#endif

#endif // MARMOT_TYPES_H
