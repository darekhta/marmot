#ifndef MARMOT_QUANT_BLOCK_H
#define MARMOT_QUANT_BLOCK_H

#include <stdint.h>

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MARMOT_QUANT_BLOCK_SIZE = 32U,
    MARMOT_Q4_PACKED_BYTES = 16U,
    MARMOT_Q5_PACKED_BYTES = 16U,
    MARMOT_Q5_HIGH_BYTES = 4U,
    MARMOT_Q8_PACKED_BYTES = 32U,
};

typedef struct {
    marmot_float16_t scale;
    uint8_t qs[MARMOT_Q4_PACKED_BYTES];
} marmot_q4_0_block_t;
static_assert(sizeof(marmot_q4_0_block_t) == 18, "Q4_0 block must be 18 bytes");

typedef struct {
    marmot_float16_t scale;
    marmot_float16_t min;
    uint8_t qs[MARMOT_Q4_PACKED_BYTES];
} marmot_q4_1_block_t;
static_assert(sizeof(marmot_q4_1_block_t) == 20, "Q4_1 block must be 20 bytes");

typedef struct {
    marmot_float16_t scale;
    uint8_t qh[MARMOT_Q5_HIGH_BYTES];
    uint8_t qs[MARMOT_Q5_PACKED_BYTES];
} marmot_q5_0_block_t;
static_assert(sizeof(marmot_q5_0_block_t) == 22, "Q5_0 block must be 22 bytes");

typedef struct {
    marmot_float16_t scale;
    marmot_float16_t min;
    uint8_t qh[MARMOT_Q5_HIGH_BYTES];
    uint8_t qs[MARMOT_Q5_PACKED_BYTES];
} marmot_q5_1_block_t;
static_assert(sizeof(marmot_q5_1_block_t) == 24, "Q5_1 block must be 24 bytes");

typedef struct {
    marmot_float16_t scale;
    int8_t qs[MARMOT_Q8_PACKED_BYTES];
} marmot_q8_0_block_t;
static_assert(sizeof(marmot_q8_0_block_t) == 34, "Q8_0 block must be 34 bytes");

// Q8_1: 32 int8 values with FP16 scale and FP16 sum (historically named "s").
// GGML stores two FP16 headers followed by 32 int8s.
typedef struct {
    marmot_float16_t scale;
    marmot_float16_t sum;
    int8_t qs[MARMOT_Q8_PACKED_BYTES];
} marmot_q8_1_block_t;
static_assert(sizeof(marmot_q8_1_block_t) == 36, "Q8_1 block must be 36 bytes");

// K‑Quant super‑block constants (GGUF / llama.cpp)
enum {
    MARMOT_QK_K_VALUES = 256U, // values per super‑block
    // Q4_K / Q5_K: 8 sub‑blocks × 32 values; packed per llama.cpp layout
    MARMOT_QK_K_QS_BYTES = 128U,    // 256 values at 4 bits → 128 bytes
    MARMOT_QK_K_QH_BYTES = 32U,     // Q5_K high bits per 64 values × 4 groups
    MARMOT_QK_K_SCALES_BYTES = 12U, // packed 6‑bit scales/mins (see llama.cpp get_scale_min_k4)
};

// Q2_K: 256 values per super-block with 2-bit quantization
typedef struct {
    uint8_t scales[MARMOT_QK_K_VALUES / 16]; // scales and mins, quantized with 4 bits
    uint8_t qs[MARMOT_QK_K_VALUES / 4];      // quants (2 bits per value)
    marmot_float16_t d;                      // super-block scale for quantized scales
    marmot_float16_t dmin;                   // super-block scale for quantized mins
} marmot_q2_k_block_t;
static_assert(sizeof(marmot_q2_k_block_t) == 84, "Q2_K block must be 84 bytes");

// Q3_K: 256 values per super-block with 3-bit quantization
typedef struct {
    uint8_t hmask[MARMOT_QK_K_VALUES / 8];    // quants - high bit
    uint8_t qs[MARMOT_QK_K_VALUES / 4];       // quants - low 2 bits
    uint8_t scales[MARMOT_QK_K_SCALES_BYTES]; // scales, quantized with 6 bits
    marmot_float16_t d;                       // super-block scale
} marmot_q3_k_block_t;
static_assert(sizeof(marmot_q3_k_block_t) == 110, "Q3_K block must be 110 bytes");

// Q4_K: 256 values per super-block with quantized scales
typedef struct {
    marmot_float16_t d;                       // super-block scale for quantized scales
    marmot_float16_t dmin;                    // super-block scale for quantized mins
    uint8_t scales[MARMOT_QK_K_SCALES_BYTES]; // scales and mins, quantized with 6 bits
    uint8_t qs[MARMOT_QK_K_QS_BYTES];         // 4-bit quants
} marmot_q4_k_block_t;
static_assert(sizeof(marmot_q4_k_block_t) == 144, "Q4_K block must be 144 bytes");

// Q5_K: 256 values per super-block with quantized scales and high bits
typedef struct {
    marmot_float16_t d;                       // super-block scale for quantized scales
    marmot_float16_t dmin;                    // super-block scale for quantized mins
    uint8_t scales[MARMOT_QK_K_SCALES_BYTES]; // scales and mins, quantized with 6 bits
    uint8_t qh[MARMOT_QK_K_QH_BYTES];         // quants, high bit
    uint8_t qs[MARMOT_QK_K_QS_BYTES];         // quants, low 4 bits
} marmot_q5_k_block_t;
static_assert(sizeof(marmot_q5_k_block_t) == 176, "Q5_K block must be 176 bytes");

// Q6_K: 256 values per super-block with 6-bit quantization
// NOTE: Unlike other K-quants, the scale 'd' is at the END of the struct
typedef struct {
    uint8_t ql[MARMOT_QK_K_QS_BYTES];       // quants, lower 4 bits
    uint8_t qh[MARMOT_QK_K_QS_BYTES / 2];   // quants, upper 2 bits
    int8_t scales[MARMOT_QK_K_VALUES / 16]; // scales, quantized with 8 bits
    marmot_float16_t d;                     // super-block scale (at END!)
} marmot_q6_k_block_t;
static_assert(sizeof(marmot_q6_k_block_t) == 210, "Q6_K block must be 210 bytes");

// Q8_K: 256 values per super-block with 8-bit quantization
// NOTE: Uses float (32-bit) for scale, not FP16
typedef struct {
    float d;                                // delta (32-bit float, not FP16!)
    int8_t qs[MARMOT_QK_K_VALUES];          // quants
    int16_t bsums[MARMOT_QK_K_VALUES / 16]; // sum of quants in groups of 16
} marmot_q8_k_block_t;
static_assert(sizeof(marmot_q8_k_block_t) == 292, "Q8_K block must be 292 bytes");

#ifdef __cplusplus
}
#endif

#endif
