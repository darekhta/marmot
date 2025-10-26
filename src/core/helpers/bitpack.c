#include "bitpack.h"

#include <string.h>

void marmot_pack_int5_block(const int8_t *values, size_t count, uint8_t *packed) {
    if (count == 0) {
        return;
    }

    const size_t num_bytes = MARMOT_PACKED_5BIT_BYTES(count);
    memset(packed, 0, num_bytes);

    size_t bit_offset = 0;
    for (size_t i = 0; i < count; ++i, bit_offset += 5) {
        const uint8_t encoded = (uint8_t)(values[i] & 0x1FU);
        const size_t byte_index = bit_offset >> 3;
        const unsigned shift = (unsigned)(bit_offset & 7U);

        packed[byte_index] |= (uint8_t)(encoded << shift);
        if (shift > 3U && byte_index + 1U < num_bytes) {
            packed[byte_index + 1U] |= (uint8_t)(encoded >> (8U - shift));
        }
    }
}

void marmot_unpack_int5_block(const uint8_t *packed, size_t count, int8_t *values) {
    if (count == 0) {
        return;
    }

    const size_t num_bytes = MARMOT_PACKED_5BIT_BYTES(count);
    size_t bit_offset = 0;
    for (size_t i = 0; i < count; ++i, bit_offset += 5) {
        const size_t byte_index = bit_offset >> 3;
        const unsigned shift = (unsigned)(bit_offset & 7U);

        uint16_t chunk = (uint16_t)(packed[byte_index] >> shift);
        if (shift > 3U && byte_index + 1U < num_bytes) {
            chunk |= (uint16_t)packed[byte_index + 1U] << (8U - shift);
        }

        const uint8_t raw = (uint8_t)(chunk & 0x1FU);
        values[i] = marmot_sign_extend5(raw);
    }
}

void marmot_pack_uint5_block(const uint8_t *values, size_t count, uint8_t *packed) {
    if (count == 0) {
        return;
    }

    const size_t num_bytes = MARMOT_PACKED_5BIT_BYTES(count);
    memset(packed, 0, num_bytes);

    size_t bit_offset = 0;
    for (size_t i = 0; i < count; ++i, bit_offset += 5) {
        const uint8_t encoded = values[i] & 0x1FU;
        const size_t byte_index = bit_offset >> 3;
        const unsigned shift = (unsigned)(bit_offset & 7U);

        packed[byte_index] |= (uint8_t)(encoded << shift);
        if (shift > 3U && byte_index + 1U < num_bytes) {
            packed[byte_index + 1U] |= (uint8_t)(encoded >> (8U - shift));
        }
    }
}

void marmot_unpack_uint5_block(const uint8_t *packed, size_t count, uint8_t *values) {
    if (count == 0) {
        return;
    }

    const size_t num_bytes = MARMOT_PACKED_5BIT_BYTES(count);
    size_t bit_offset = 0;
    for (size_t i = 0; i < count; ++i, bit_offset += 5) {
        const size_t byte_index = bit_offset >> 3;
        const unsigned shift = (unsigned)(bit_offset & 7U);

        uint16_t chunk = (uint16_t)(packed[byte_index] >> shift);
        if (shift > 3U && byte_index + 1U < num_bytes) {
            chunk |= (uint16_t)packed[byte_index + 1U] << (8U - shift);
        }

        values[i] = (uint8_t)(chunk & 0x1FU);
    }
}
