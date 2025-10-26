#ifndef MARMOT_CORE_HELPERS_BITPACK_H
#define MARMOT_CORE_HELPERS_BITPACK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MARMOT_PACKED_5BIT_BYTES(count) (((count) * 5U + 7U) / 8U)

void marmot_pack_int5_block(const int8_t *values, size_t count, uint8_t *packed);
void marmot_unpack_int5_block(const uint8_t *packed, size_t count, int8_t *values);

void marmot_pack_uint5_block(const uint8_t *values, size_t count, uint8_t *packed);
void marmot_unpack_uint5_block(const uint8_t *packed, size_t count, uint8_t *values);

static inline int8_t marmot_sign_extend5(uint8_t value) {
    value &= 0x1F;
    return (value & 0x10U) ? (int8_t)(value | 0xE0U) : (int8_t)value;
}

#ifdef __cplusplus
}
#endif

#endif
