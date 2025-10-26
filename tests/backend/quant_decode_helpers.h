#ifndef TESTS_BACKEND_QUANT_DECODE_HELPERS_H
#define TESTS_BACKEND_QUANT_DECODE_HELPERS_H

#include "marmot/quant_block.h"

#include <string.h>

#include "utils/dtype_ref.h"

static inline void marmot_test_unpack_q4_0_block(const marmot_q4_0_block_t *block, float *dst) {
    const float scale = marmot_f16_to_f32_ref(block->scale);
    const size_t half = MARMOT_QUANT_BLOCK_SIZE / 2;
    for (size_t i = 0; i < MARMOT_Q4_PACKED_BYTES; ++i) {
        const uint8_t packed = block->qs[i];
        dst[i] = ((int32_t)(packed & 0x0f) - 8) * scale;
        dst[i + half] = ((int32_t)((packed >> 4) & 0x0f) - 8) * scale;
    }
}

static inline void marmot_test_unpack_q4_1_block(const marmot_q4_1_block_t *block, float *dst) {
    const float scale = marmot_f16_to_f32_ref(block->scale);
    const float min_value = marmot_f16_to_f32_ref(block->min);
    const size_t half = MARMOT_QUANT_BLOCK_SIZE / 2;
    for (size_t i = 0; i < MARMOT_Q4_PACKED_BYTES; ++i) {
        const uint8_t packed = block->qs[i];
        dst[i] = ((int32_t)(packed & 0x0f)) * scale + min_value;
        dst[i + half] = ((int32_t)((packed >> 4) & 0x0f)) * scale + min_value;
    }
}

static inline void marmot_test_unpack_q5_0_block(const marmot_q5_0_block_t *block, float *dst) {
    uint32_t qh;
    memcpy(&qh, block->qh, sizeof(qh));
    const float scale = marmot_f16_to_f32_ref(block->scale);
    const size_t half = MARMOT_QUANT_BLOCK_SIZE / 2;
    for (size_t j = 0; j < half; ++j) {
        const uint8_t packed = block->qs[j];
        uint8_t lo = packed & 0x0f;
        uint8_t hi = packed >> 4;
        lo |= (uint8_t)(((qh >> (j + 0)) & 0x1u) << 4);
        hi |= (uint8_t)(((qh >> (j + half)) & 0x1u) << 4);
        dst[j] = ((int32_t)lo - 16) * scale;
        dst[j + half] = ((int32_t)hi - 16) * scale;
    }
}

static inline void marmot_test_unpack_q5_1_block(const marmot_q5_1_block_t *block, float *dst) {
    uint32_t qh;
    memcpy(&qh, block->qh, sizeof(qh));
    const float scale = marmot_f16_to_f32_ref(block->scale);
    const float min_value = marmot_f16_to_f32_ref(block->min);
    const size_t half = MARMOT_QUANT_BLOCK_SIZE / 2;
    for (size_t j = 0; j < half; ++j) {
        const uint8_t packed = block->qs[j];
        uint8_t lo = packed & 0x0f;
        uint8_t hi = packed >> 4;
        lo |= (uint8_t)(((qh >> (j + 0)) & 0x1u) << 4);
        hi |= (uint8_t)(((qh >> (j + half)) & 0x1u) << 4);
        dst[j] = (float)lo * scale + min_value;
        dst[j + half] = (float)hi * scale + min_value;
    }
}

static inline void marmot_test_unpack_q8_0_block(const marmot_q8_0_block_t *block, float *dst) {
    const float scale = marmot_f16_to_f32_ref(block->scale);
    for (size_t i = 0; i < MARMOT_QUANT_BLOCK_SIZE; ++i) {
        dst[i] = (float)block->qs[i] * scale;
    }
}

#endif // TESTS_BACKEND_QUANT_DECODE_HELPERS_H
