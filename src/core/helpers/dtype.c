#include "marmot/device.h"
#include "marmot/quant_block.h"
#include "marmot/types.h"

#include <stddef.h>
#include <stdint.h>

#include "dtype_internal.h"

#define TRAIT_ENTRY(                                                                                                   \
    ID, NAME_STR, STORAGE_BYTES, ELEMENT_BITS, ALIGNMENT, COMPUTE_DTYPE, FLOATING, SIGNED, QUANTIZED, PACKED, CPU,     \
    METAL, SIMD, REDUCE, REDUCE_ACCUM                                                                                  \
)                                                                                                                      \
    [ID] = {                                                                                                           \
        .id = ID,                                                                                                      \
        .name = NAME_STR,                                                                                              \
        .storage_bytes = (STORAGE_BYTES),                                                                              \
        .element_bits = (ELEMENT_BITS),                                                                                \
        .alignment = (ALIGNMENT),                                                                                      \
        .compute_dtype = (COMPUTE_DTYPE),                                                                              \
        .is_floating = (FLOATING),                                                                                     \
        .is_signed = (SIGNED),                                                                                         \
        .is_quantized = (QUANTIZED),                                                                                   \
        .is_packed = (PACKED),                                                                                         \
        .has_cpu_support = (CPU),                                                                                      \
        .has_metal_support = (METAL),                                                                                  \
        .has_simd_support = (SIMD),                                                                                    \
        .supports_reduction = (REDUCE),                                                                                \
        .reduction_accum_dtype = (REDUCE_ACCUM),                                                                       \
    }

static const marmot_dtype_traits_t MARMOT_DTYPE_TABLE[MARMOT_DTYPE_COUNT] = {
    TRAIT_ENTRY(
        MARMOT_DTYPE_FLOAT32, "float32", sizeof(float), 32, _Alignof(float), MARMOT_DTYPE_FLOAT32, true, true, false,
        false, true, true, true, true, MARMOT_DTYPE_FLOAT32
    ),
    TRAIT_ENTRY(
        MARMOT_DTYPE_FLOAT16, "float16", sizeof(uint16_t), 16, _Alignof(marmot_float16_t), MARMOT_DTYPE_FLOAT16, true,
        true, false, false, true, true, true, true, MARMOT_DTYPE_FLOAT16
    ),
    TRAIT_ENTRY(
        MARMOT_DTYPE_BFLOAT16, "bfloat16", sizeof(uint16_t), 16, _Alignof(marmot_bfloat16_t), MARMOT_DTYPE_FLOAT32,
        true, true, false, false, true, true, true, true, MARMOT_DTYPE_FLOAT32
    ),
    TRAIT_ENTRY(
        MARMOT_DTYPE_INT32, "int32", sizeof(marmot_int32_t), 32, _Alignof(marmot_int32_t), MARMOT_DTYPE_INT32, false,
        true, false, false, true, true, true, true, MARMOT_DTYPE_INT32
    ),
    TRAIT_ENTRY(
        MARMOT_DTYPE_INT16, "int16", sizeof(marmot_int16_t), 16, _Alignof(marmot_int16_t), MARMOT_DTYPE_INT16, false,
        true, false, false, true, true, true, true, MARMOT_DTYPE_INT16
    ),
    TRAIT_ENTRY(
        MARMOT_DTYPE_INT8, "int8", sizeof(marmot_int8_t), 8, _Alignof(marmot_int8_t), MARMOT_DTYPE_INT8, false, true,
        false, false, true, true, true, true, MARMOT_DTYPE_INT8
    ),
    TRAIT_ENTRY(
        MARMOT_DTYPE_UINT8, "uint8", sizeof(marmot_uint8_t), 8, _Alignof(marmot_uint8_t), MARMOT_DTYPE_UINT8, false,
        false, false, false, true, true, true, true, MARMOT_DTYPE_UINT8
    ),
    TRAIT_ENTRY(
        MARMOT_DTYPE_UINT16, "uint16", sizeof(marmot_uint16_t), 16, _Alignof(marmot_uint16_t), MARMOT_DTYPE_UINT16,
        false, false, false, false, true, true, true, true, MARMOT_DTYPE_UINT16
    ),
    TRAIT_ENTRY(
        MARMOT_DTYPE_UINT32, "uint32", sizeof(marmot_uint32_t), 32, _Alignof(marmot_uint32_t), MARMOT_DTYPE_UINT32,
        false, false, false, false, true, true, true, true, MARMOT_DTYPE_UINT32
    ),
    TRAIT_ENTRY(
        MARMOT_DTYPE_UINT64, "uint64", sizeof(marmot_uint64_t), 64, _Alignof(marmot_uint64_t), MARMOT_DTYPE_UINT64,
        false, false, false, false, true, true, false, true, MARMOT_DTYPE_UINT64
    ),
    TRAIT_ENTRY(
        MARMOT_DTYPE_FLOAT64, "float64", sizeof(double), 64, _Alignof(double), MARMOT_DTYPE_FLOAT64, true, true, false,
        false, true, false, false, true, MARMOT_DTYPE_FLOAT64
    ),
    TRAIT_ENTRY(
        MARMOT_DTYPE_INT64, "int64", sizeof(marmot_int64_t), 64, _Alignof(marmot_int64_t), MARMOT_DTYPE_INT64, false,
        true, false, false, true, false, false, true, MARMOT_DTYPE_INT64
    ),
    TRAIT_ENTRY(
        MARMOT_DTYPE_INT4, "int4", 0, 4, 1, MARMOT_DTYPE_FLOAT32, false, true, true, true, true, false, false, false,
        MARMOT_DTYPE_FLOAT32
    ),
    TRAIT_ENTRY(
        MARMOT_DTYPE_UINT4, "uint4", 0, 4, 1, MARMOT_DTYPE_FLOAT32, false, false, true, true, true, false, false, false,
        MARMOT_DTYPE_FLOAT32
    ),
#if MARMOT_ENABLE_FP8
    TRAIT_ENTRY(
        MARMOT_DTYPE_FLOAT8_E4M3, "float8_e4m3", sizeof(marmot_float8_e4m3_t), 8, _Alignof(marmot_float8_e4m3_t),
        MARMOT_DTYPE_FLOAT16, true, true, false, false, true, true, true, true, MARMOT_DTYPE_FLOAT16
    ),
    TRAIT_ENTRY(
        MARMOT_DTYPE_FLOAT8_E5M2, "float8_e5m2", sizeof(marmot_float8_e5m2_t), 8, _Alignof(marmot_float8_e5m2_t),
        MARMOT_DTYPE_FLOAT16, true, true, false, false, true, true, true, true, MARMOT_DTYPE_FLOAT16
    ),
#endif
};

static const marmot_quant_kind_traits_t MARMOT_QUANT_KIND_TABLE[MARMOT_QUANT_KIND_COUNT] = {
    [MARMOT_QUANT_KIND_GENERIC] =
        {
            .kind = MARMOT_QUANT_KIND_GENERIC,
            .storage_dtype = MARMOT_DTYPE_COUNT,
            .block_values = 0,
            .header_bytes = 0,
            .payload_bytes = 0,
            .payload_signed = false,
            .is_block_quantized = false,
            .is_bit_packed = false,
            .layout = MARMOT_QUANT_LAYOUT_GENERIC,
        },
    [MARMOT_QUANT_KIND_Q4_0] =
        {
            .kind = MARMOT_QUANT_KIND_Q4_0,
            .storage_dtype = MARMOT_DTYPE_UINT8,
            .block_values = MARMOT_QUANT_BLOCK_SIZE,
            .header_bytes = sizeof(marmot_float16_t),
            .payload_bytes = MARMOT_Q4_PACKED_BYTES,
            .payload_signed = true,
            .is_block_quantized = true,
            .is_bit_packed = true,
            .layout = MARMOT_QUANT_LAYOUT_GGUF,
        },
    [MARMOT_QUANT_KIND_Q4_1] =
        {
            .kind = MARMOT_QUANT_KIND_Q4_1,
            .storage_dtype = MARMOT_DTYPE_UINT8,
            .block_values = MARMOT_QUANT_BLOCK_SIZE,
            .header_bytes = sizeof(marmot_float16_t) * 2,
            .payload_bytes = MARMOT_Q4_PACKED_BYTES,
            .payload_signed = false,
            .is_block_quantized = true,
            .is_bit_packed = true,
            .layout = MARMOT_QUANT_LAYOUT_GGUF,
        },
    [MARMOT_QUANT_KIND_Q5_0] =
        {
            .kind = MARMOT_QUANT_KIND_Q5_0,
            .storage_dtype = MARMOT_DTYPE_UINT8,
            .block_values = MARMOT_QUANT_BLOCK_SIZE,
            .header_bytes = sizeof(marmot_float16_t) + MARMOT_Q5_HIGH_BYTES,
            .payload_bytes = MARMOT_Q5_PACKED_BYTES,
            .payload_signed = true,
            .is_block_quantized = true,
            .is_bit_packed = true,
            .layout = MARMOT_QUANT_LAYOUT_GGUF,
        },
    [MARMOT_QUANT_KIND_Q5_1] =
        {
            .kind = MARMOT_QUANT_KIND_Q5_1,
            .storage_dtype = MARMOT_DTYPE_UINT8,
            .block_values = MARMOT_QUANT_BLOCK_SIZE,
            .header_bytes = sizeof(marmot_float16_t) * 2 + MARMOT_Q5_HIGH_BYTES,
            .payload_bytes = MARMOT_Q5_PACKED_BYTES,
            .payload_signed = false,
            .is_block_quantized = true,
            .is_bit_packed = true,
            .layout = MARMOT_QUANT_LAYOUT_GGUF,
        },
    [MARMOT_QUANT_KIND_Q8_0] =
        {
            .kind = MARMOT_QUANT_KIND_Q8_0,
            .storage_dtype = MARMOT_DTYPE_INT8,
            .block_values = MARMOT_QUANT_BLOCK_SIZE,
            .header_bytes = sizeof(marmot_float16_t),
            .payload_bytes = MARMOT_Q8_PACKED_BYTES,
            .payload_signed = true,
            .is_block_quantized = true,
            .is_bit_packed = false,
            .layout = MARMOT_QUANT_LAYOUT_GGUF,
        },
    [MARMOT_QUANT_KIND_Q8_1] =
        {
            .kind = MARMOT_QUANT_KIND_Q8_1,
            .storage_dtype = MARMOT_DTYPE_INT8,
            .block_values = MARMOT_QUANT_BLOCK_SIZE,
            .header_bytes = sizeof(marmot_float16_t) * 2, // scale + sum
            .payload_bytes = MARMOT_Q8_PACKED_BYTES,
            .payload_signed = true,
            .is_block_quantized = true,
            .is_bit_packed = false,
            .layout = MARMOT_QUANT_LAYOUT_GGUF,
        },
    [MARMOT_QUANT_KIND_Q2_K] =
        {
            .kind = MARMOT_QUANT_KIND_Q2_K,
            .storage_dtype = MARMOT_DTYPE_UINT8,
            .block_values = MARMOT_QK_K_VALUES,
            .header_bytes = MARMOT_QK_K_VALUES / 16 + MARMOT_QK_K_VALUES / 4,
            .payload_bytes = sizeof(marmot_float16_t) * 2,
            .payload_signed = false,
            .is_block_quantized = true,
            .is_bit_packed = true,
            .layout = MARMOT_QUANT_LAYOUT_GGUF,
        },
    [MARMOT_QUANT_KIND_Q3_K] =
        {
            .kind = MARMOT_QUANT_KIND_Q3_K,
            .storage_dtype = MARMOT_DTYPE_UINT8,
            .block_values = MARMOT_QK_K_VALUES,
            .header_bytes = MARMOT_QK_K_VALUES / 8 + MARMOT_QK_K_VALUES / 4 + MARMOT_QK_K_SCALES_BYTES,
            .payload_bytes = sizeof(marmot_float16_t),
            .payload_signed = false,
            .is_block_quantized = true,
            .is_bit_packed = true,
            .layout = MARMOT_QUANT_LAYOUT_GGUF,
        },
    [MARMOT_QUANT_KIND_Q4_K] =
        {
            .kind = MARMOT_QUANT_KIND_Q4_K,
            .storage_dtype = MARMOT_DTYPE_UINT8,
            .block_values = MARMOT_QK_K_VALUES,
            .header_bytes = sizeof(marmot_float16_t) * 2 + MARMOT_QK_K_SCALES_BYTES,
            .payload_bytes = MARMOT_QK_K_QS_BYTES,
            .payload_signed = false,
            .is_block_quantized = true,
            .is_bit_packed = true,
            .layout = MARMOT_QUANT_LAYOUT_GGUF,
        },
    [MARMOT_QUANT_KIND_Q5_K] =
        {
            .kind = MARMOT_QUANT_KIND_Q5_K,
            .storage_dtype = MARMOT_DTYPE_UINT8,
            .block_values = MARMOT_QK_K_VALUES,
            .header_bytes = sizeof(marmot_float16_t) * 2 + MARMOT_QK_K_SCALES_BYTES,
            .payload_bytes = MARMOT_QK_K_QS_BYTES + MARMOT_QK_K_QH_BYTES,
            .payload_signed = false,
            .is_block_quantized = true,
            .is_bit_packed = true,
            .layout = MARMOT_QUANT_LAYOUT_GGUF,
        },
    [MARMOT_QUANT_KIND_Q6_K] =
        {
            .kind = MARMOT_QUANT_KIND_Q6_K,
            .storage_dtype = MARMOT_DTYPE_UINT8,
            .block_values = MARMOT_QK_K_VALUES,
            // Q6_K has non-standard layout: ql[128] + qh[64] + scales[16] + d (scale at END, not front)
            .header_bytes = 0,
            .payload_bytes = 128 + 64 + 16 + sizeof(marmot_float16_t), // 210 bytes total
            .payload_signed = true,
            .is_block_quantized = true,
            .is_bit_packed = true,
            .layout = MARMOT_QUANT_LAYOUT_GGUF,
        },
    [MARMOT_QUANT_KIND_Q8_K] = {
        .kind = MARMOT_QUANT_KIND_Q8_K,
        .storage_dtype = MARMOT_DTYPE_INT8,
        .block_values = MARMOT_QK_K_VALUES,
        // Q8_K uses float (32-bit) scale, not FP16
        .header_bytes = sizeof(float),
        .payload_bytes = MARMOT_QK_K_VALUES + (MARMOT_QK_K_VALUES / 16) * sizeof(int16_t),
        .payload_signed = true,
        .is_block_quantized = true,
        .is_bit_packed = false,
        .layout = MARMOT_QUANT_LAYOUT_GGUF,
    },
};

const marmot_dtype_traits_t *marmot_get_dtype_traits(marmot_dtype_t dtype) {
    if (!marmot_dtype_valid(dtype)) {
        return nullptr;
    }
    return &MARMOT_DTYPE_TABLE[dtype];
}

bool marmot_dtype_is_floating(marmot_dtype_t dtype) {
    const marmot_dtype_traits_t *traits = marmot_get_dtype_traits(dtype);
    return traits && traits->is_floating;
}

bool marmot_dtype_is_integer(marmot_dtype_t dtype) {
    const marmot_dtype_traits_t *traits = marmot_get_dtype_traits(dtype);
    if (traits == nullptr) {
        return false;
    }
    return !traits->is_floating && !traits->is_quantized && !traits->is_packed;
}

bool marmot_dtype_is_signed(marmot_dtype_t dtype) {
    const marmot_dtype_traits_t *traits = marmot_get_dtype_traits(dtype);
    return traits && traits->is_signed;
}

bool marmot_dtype_is_quantized(marmot_dtype_t dtype) {
    const marmot_dtype_traits_t *traits = marmot_get_dtype_traits(dtype);
    return traits && traits->is_quantized;
}

bool marmot_dtype_is_packed(marmot_dtype_t dtype) {
    const marmot_dtype_traits_t *traits = marmot_get_dtype_traits(dtype);
    return traits && traits->is_packed;
}

size_t marmot_dtype_element_bits(marmot_dtype_t dtype) {
    const marmot_dtype_traits_t *traits = marmot_get_dtype_traits(dtype);
    return traits ? traits->element_bits : 0;
}

size_t marmot_dtype_size(marmot_dtype_t dtype) {
    const marmot_dtype_traits_t *traits = marmot_get_dtype_traits(dtype);
    return traits ? traits->storage_bytes : 0;
}

size_t marmot_dtype_alignment(marmot_dtype_t dtype) {
    const marmot_dtype_traits_t *traits = marmot_get_dtype_traits(dtype);
    return traits ? traits->alignment : 0;
}

marmot_dtype_t marmot_dtype_compute_dtype(marmot_dtype_t dtype) {
    const marmot_dtype_traits_t *traits = marmot_get_dtype_traits(dtype);
    return traits ? traits->compute_dtype : MARMOT_DTYPE_COUNT;
}

bool marmot_dtype_has_cpu_support(marmot_dtype_t dtype) {
    const marmot_dtype_traits_t *traits = marmot_get_dtype_traits(dtype);
    return traits ? traits->has_cpu_support : false;
}

bool marmot_dtype_has_metal_support(marmot_dtype_t dtype) {
    const marmot_dtype_traits_t *traits = marmot_get_dtype_traits(dtype);
    return traits ? traits->has_metal_support : false;
}

bool marmot_dtype_supports_reduction(marmot_dtype_t dtype) {
    const marmot_dtype_traits_t *traits = marmot_get_dtype_traits(dtype);
    return traits ? traits->supports_reduction : false;
}

marmot_dtype_t marmot_dtype_reduction_accum_dtype(marmot_dtype_t dtype) {
    const marmot_dtype_traits_t *traits = marmot_get_dtype_traits(dtype);
    return traits ? traits->reduction_accum_dtype : MARMOT_DTYPE_COUNT;
}

const marmot_quant_kind_traits_t *marmot_get_quant_kind_traits(marmot_quant_kind_t kind) {
    if (!marmot_quant_kind_valid(kind)) {
        return nullptr;
    }
    return &MARMOT_QUANT_KIND_TABLE[kind];
}

bool marmot_quant_kind_is_block_quantized(marmot_quant_kind_t kind) {
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(kind);
    return traits && traits->is_block_quantized;
}

bool marmot_dtype_is_supported_on_backend(marmot_backend_type_t backend, marmot_dtype_t dtype) {
    if (!marmot_dtype_valid(dtype)) {
        return false;
    }

    switch (backend) {
    case MARMOT_BACKEND_CPU:
        return marmot_dtype_has_cpu_support(dtype);
    case MARMOT_BACKEND_METAL:
        return marmot_dtype_has_metal_support(dtype);
    case MARMOT_BACKEND_CUDA:
        return false;
    default:
        return false;
    }
}
