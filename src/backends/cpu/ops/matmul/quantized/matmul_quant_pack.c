#include "marmot/quant_block.h"

#include <float.h>
#include <math.h>
#include <string.h>

#include "cpu_backend_internal.h"
#include "ops/matmul/quantized/matmul_quant_activation.h"
#include "quantization/common/block.h"

static inline int cpu_quant_matmul_nearest_int(float value) {
    if (value > 4194303.0f) {
        value = 4194303.0f;
    } else if (value < -4194303.0f) {
        value = -4194303.0f;
    }
    const float shifted = value + 12582912.0f;
    int bits = 0;
    memcpy(&bits, &shifted, sizeof(int));
    return (bits & 0x007fffff) - 0x00400000;
}

void cpu_matmul_quant_pack_q8_0_f32(
    const float *activations, size_t stride_k, size_t stride_n, size_t column_index, size_t K, size_t blocks_per_row,
    void *dst_blocks
) {
    marmot_q8_0_block_t *q8_blocks = (marmot_q8_0_block_t *)dst_blocks;
    float block_buffer[MARMOT_QUANT_BLOCK_SIZE];

    for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const size_t block_start = block_idx * MARMOT_QUANT_BLOCK_SIZE;
        size_t block_end = block_start + MARMOT_QUANT_BLOCK_SIZE;
        if (block_end > K) {
            block_end = K;
        }
        const size_t block_len = block_end - block_start;

        for (size_t i = 0; i < block_len; ++i) {
            const size_t k = block_start + i;
            block_buffer[i] = activations[column_index * stride_n + k * stride_k];
        }
        if (block_len < MARMOT_QUANT_BLOCK_SIZE) {
            memset(block_buffer + block_len, 0, (MARMOT_QUANT_BLOCK_SIZE - block_len) * sizeof(float));
        }

        float max_abs = cpu_quant_block_max_abs(block_buffer, block_len);
        float scale = max_abs / 127.0f;
        if (scale < FLT_MIN) {
            scale = 1.0f;
        }
        const float inv_scale = 1.0f / scale;

        marmot_q8_0_block_t *dst = &q8_blocks[block_idx];
        memset(dst, 0, sizeof(*dst));
        dst->scale = marmot_native_to_float16((_Float16)scale);
        cpu_quantize_q8_0_pack(block_buffer, block_len, inv_scale, dst->qs);

        if (block_len < MARMOT_QUANT_BLOCK_SIZE) {
            memset(dst->qs + block_len, 0, (MARMOT_QUANT_BLOCK_SIZE - block_len) * sizeof(int8_t));
        }
    }
}

void cpu_matmul_quant_pack_q8_0_f16(
    const marmot_float16_t *activations, size_t stride_k, size_t stride_n, size_t column_index, size_t K,
    size_t blocks_per_row, void *dst_blocks
) {
    marmot_q8_0_block_t *q8_blocks = (marmot_q8_0_block_t *)dst_blocks;
    float block_buffer[MARMOT_QUANT_BLOCK_SIZE];

    for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const size_t block_start = block_idx * MARMOT_QUANT_BLOCK_SIZE;
        size_t block_end = block_start + MARMOT_QUANT_BLOCK_SIZE;
        if (block_end > K) {
            block_end = K;
        }
        const size_t block_len = block_end - block_start;

        for (size_t i = 0; i < block_len; ++i) {
            const size_t k = block_start + i;
            const marmot_float16_t *elem = activations + k * stride_k + column_index * stride_n;
            block_buffer[i] = (float)marmot_float16_to_native(*elem);
        }
        if (block_len < MARMOT_QUANT_BLOCK_SIZE) {
            memset(block_buffer + block_len, 0, (MARMOT_QUANT_BLOCK_SIZE - block_len) * sizeof(float));
        }

        float max_abs = cpu_quant_block_max_abs(block_buffer, block_len);
        float scale = max_abs / 127.0f;
        if (scale < FLT_MIN) {
            scale = 1.0f;
        }
        const float inv_scale = 1.0f / scale;

        marmot_q8_0_block_t *dst = &q8_blocks[block_idx];
        memset(dst, 0, sizeof(*dst));
        dst->scale = marmot_native_to_float16((_Float16)scale);
        cpu_quantize_q8_0_pack(block_buffer, block_len, inv_scale, dst->qs);

        if (block_len < MARMOT_QUANT_BLOCK_SIZE) {
            memset(dst->qs + block_len, 0, (MARMOT_QUANT_BLOCK_SIZE - block_len) * sizeof(int8_t));
        }
    }
}

static void cpu_quant_matmul_pack_q8_k_block(const float *block_values, marmot_q8_k_block_t *dst) {
    float max_val = 0.0f;
    float abs_max = 0.0f;
    for (size_t i = 0; i < MARMOT_QK_K_VALUES; ++i) {
        const float ax = fabsf(block_values[i]);
        if (ax > abs_max) {
            abs_max = ax;
            max_val = block_values[i];
        }
    }

    if (abs_max <= 0.0f) {
        dst->d = 0.0f;
        memset(dst->qs, 0, sizeof(dst->qs));
        memset(dst->bsums, 0, sizeof(dst->bsums));
        return;
    }

    const float iscale = -127.0f / max_val;
    for (size_t i = 0; i < MARMOT_QK_K_VALUES; ++i) {
        int v = cpu_quant_matmul_nearest_int(iscale * block_values[i]);
        if (v > 127) {
            v = 127;
        }
        dst->qs[i] = (int8_t)v;
    }

    for (size_t block = 0; block < MARMOT_QK_K_VALUES / 16; ++block) {
        int sum = 0;
        for (size_t i = 0; i < 16; ++i) {
            sum += dst->qs[block * 16 + i];
        }
        dst->bsums[block] = (int16_t)sum;
    }

    dst->d = 1.0f / iscale;
}

void cpu_matmul_quant_pack_q8_k_f32(
    const float *activations, size_t stride_k, size_t stride_n, size_t column_index, size_t K, size_t blocks_per_row,
    void *dst_blocks
) {
    marmot_q8_k_block_t *q8_blocks = (marmot_q8_k_block_t *)dst_blocks;
    float block_buffer[MARMOT_QK_K_VALUES];

    for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const size_t block_start = block_idx * MARMOT_QK_K_VALUES;
        size_t block_end = block_start + MARMOT_QK_K_VALUES;
        if (block_end > K) {
            block_end = K;
        }
        const size_t block_len = block_end - block_start;

        for (size_t i = 0; i < block_len; ++i) {
            const size_t k = block_start + i;
            block_buffer[i] = activations[column_index * stride_n + k * stride_k];
        }
        if (block_len < MARMOT_QK_K_VALUES) {
            memset(block_buffer + block_len, 0, (MARMOT_QK_K_VALUES - block_len) * sizeof(block_buffer[0]));
        }

        cpu_quant_matmul_pack_q8_k_block(block_buffer, &q8_blocks[block_idx]);
    }
}

void cpu_matmul_quant_pack_q8_k_f16(
    const marmot_float16_t *activations, size_t stride_k, size_t stride_n, size_t column_index, size_t K,
    size_t blocks_per_row, void *dst_blocks
) {
    marmot_q8_k_block_t *q8_blocks = (marmot_q8_k_block_t *)dst_blocks;
    float block_buffer[MARMOT_QK_K_VALUES];

    for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const size_t block_start = block_idx * MARMOT_QK_K_VALUES;
        size_t block_end = block_start + MARMOT_QK_K_VALUES;
        if (block_end > K) {
            block_end = K;
        }
        const size_t block_len = block_end - block_start;

        for (size_t i = 0; i < block_len; ++i) {
            const size_t k = block_start + i;
            const marmot_float16_t *elem = activations + k * stride_k + column_index * stride_n;
            block_buffer[i] = (float)marmot_float16_to_native(*elem);
        }
        if (block_len < MARMOT_QK_K_VALUES) {
            memset(block_buffer + block_len, 0, (MARMOT_QK_K_VALUES - block_len) * sizeof(block_buffer[0]));
        }

        cpu_quant_matmul_pack_q8_k_block(block_buffer, &q8_blocks[block_idx]);
    }
}
