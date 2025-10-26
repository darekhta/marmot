#include "marmot/tensor.h"
#include "marmot/types.h"

#include <stdint.h>

// ============================================================================
// INT4 Bit Packing (Signed: -8 to 7)
// ============================================================================
// Layout: [yyyy xxxx] where x=first value (LSB), y=second value (MSB)
// This follows ONNX/GGML standard for INT4 packing

#if MARMOT_HAS_BITINT
typedef signed _BitInt(4) marmot_int4_lane_t;
typedef unsigned _BitInt(4) marmot_uint4_lane_t;
#endif

static inline int8_t clamp_int4(int8_t value) {
#if MARMOT_HAS_BITINT
    marmot_int4_lane_t narrowed = value;
    if (narrowed < -7) {
        narrowed = -7;
    } else if (narrowed > 7) {
        narrowed = 7;
    }
    return (int8_t)narrowed;
#else
    return (value < -7) ? -7 : (value > 7 ? 7 : value);
#endif
}

uint8_t marmot_pack_int4(int8_t x, int8_t y) {
    x = clamp_int4(x);
    y = clamp_int4(y);
    return ((uint8_t)(y & 0x0F) << 4) | (uint8_t)(x & 0x0F);
}

void marmot_unpack_int4(uint8_t packed, int8_t *x, int8_t *y) {
#if MARMOT_HAS_BITINT
    marmot_int4_lane_t low = (marmot_int4_lane_t)(packed & 0x0F);
    marmot_int4_lane_t high = (marmot_int4_lane_t)((packed >> 4) & 0x0F);
    *x = (int8_t)low;
    *y = (int8_t)high;
#else
    *x = (int8_t)(packed & 0x0F);
    *y = (int8_t)((packed >> 4) & 0x0F);

    // Sign extend from 4-bit to 8-bit
    // If bit 3 (sign bit in 4-bit) is set, extend with 1s
    if (*x & 0x08) {
        *x |= (int8_t)0xF0;
    }
    if (*y & 0x08) {
        *y |= (int8_t)0xF0;
    }
#endif
}

// ============================================================================
// UINT4 Bit Packing (Unsigned: 0 to 15)
// ============================================================================

static inline uint8_t clamp_uint4(uint8_t value) {
#if MARMOT_HAS_BITINT
    marmot_uint4_lane_t narrowed = value;
    if (narrowed > 15) {
        narrowed = 15;
    }
    return (uint8_t)narrowed;
#else
    return (value > 15) ? 15 : value;
#endif
}

uint8_t marmot_pack_uint4(uint8_t x, uint8_t y) {
    x = clamp_uint4(x);
    y = clamp_uint4(y);
    return (y << 4) | x;
}

void marmot_unpack_uint4(uint8_t packed, uint8_t *x, uint8_t *y) {
    *x = packed & 0x0F;
    *y = (packed >> 4) & 0x0F;
}

// ============================================================================
// Block Packing/Unpacking (32 values per block)
// ============================================================================

void marmot_pack_int4_block(const int8_t *values, size_t count, uint8_t *packed) {
    size_t pairs = count / 2;
    for (size_t i = 0; i < pairs; i++) {
        packed[i] = marmot_pack_int4(values[i * 2], values[i * 2 + 1]);
    }

    // Handle odd count (pack last value with 0)
    if (count & 1) {
        packed[pairs] = marmot_pack_int4(values[count - 1], 0);
    }
}

void marmot_unpack_int4_block(const uint8_t *packed, size_t count, int8_t *values) {
    size_t pairs = (count + 1) / 2;
    for (size_t i = 0; i < pairs; i++) {
        int8_t x, y;
        marmot_unpack_int4(packed[i], &x, &y);
        values[i * 2] = x;
        if (i * 2 + 1 < count) {
            values[i * 2 + 1] = y;
        }
    }
}

void marmot_pack_uint4_block(const uint8_t *values, size_t count, uint8_t *packed) {
    size_t pairs = count / 2;
    for (size_t i = 0; i < pairs; i++) {
        packed[i] = marmot_pack_uint4(values[i * 2], values[i * 2 + 1]);
    }

    // Handle odd count (pack last value with 0)
    if (count & 1) {
        packed[pairs] = marmot_pack_uint4(values[count - 1], 0);
    }
}

void marmot_unpack_uint4_block(const uint8_t *packed, size_t count, uint8_t *values) {
    size_t pairs = (count + 1) / 2;
    for (size_t i = 0; i < pairs; i++) {
        uint8_t x, y;
        marmot_unpack_uint4(packed[i], &x, &y);
        values[i * 2] = x;
        if (i * 2 + 1 < count) {
            values[i * 2 + 1] = y;
        }
    }
}
