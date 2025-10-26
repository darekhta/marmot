#ifndef MARMOT_CORE_HELPERS_INT4_H
#define MARMOT_CORE_HELPERS_INT4_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===================================================================
// INTERNAL: INT4 Bit Packing Utilities
// These are low-level utilities for bit packing/unpacking. Higher level code should
// prefer tensor creation helpers or call into these routines directly when working
// with quantized weight blocks.
// ===================================================================

// Pack two INT4 values (signed: -8 to 7) into one byte [yyyy xxxx]
uint8_t marmot_pack_int4(int8_t x, int8_t y);

// Unpack one byte into two INT4 values (with sign extension)
void marmot_unpack_int4(uint8_t packed, int8_t *x, int8_t *y);

// Pack two UINT4 values (unsigned: 0 to 15) into one byte [yyyy xxxx]
uint8_t marmot_pack_uint4(uint8_t x, uint8_t y);

// Unpack one byte into two UINT4 values
void marmot_unpack_uint4(uint8_t packed, uint8_t *x, uint8_t *y);

// Block packing: pack array of INT4 values into byte array
void marmot_pack_int4_block(const int8_t *values, size_t count, uint8_t *packed);

// Block unpacking: unpack byte array into array of INT4 values
void marmot_unpack_int4_block(const uint8_t *packed, size_t count, int8_t *values);

// Block packing: pack array of UINT4 values into byte array
void marmot_pack_uint4_block(const uint8_t *values, size_t count, uint8_t *packed);

// Block unpacking: unpack byte array into array of UINT4 values
void marmot_unpack_uint4_block(const uint8_t *packed, size_t count, uint8_t *values);

#ifdef __cplusplus
}
#endif

#endif
