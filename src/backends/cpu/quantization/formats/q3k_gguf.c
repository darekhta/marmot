#include "marmot/error.h"
#include "marmot/macros.h"
#include "marmot/quant_block.h"
#include "marmot/quant_traits.h"
#include "marmot/tensor.h"

#include <float.h>
#include <math.h>
#include <string.h>

#include "../common/kquant_common.h"
#include "cpu_backend_internal.h"
#include "quantization/format_metadata.h"

static marmot_error_t q3_k_quantize_block(const float *values, uint32_t count, void *block_out, const void *params);
static marmot_error_t q3_k_dequantize_block(const void *block, uint32_t count, float *values_out, const void *params);

static const marmot_quant_traits_t q3_k_traits = {
    .kind = MARMOT_QUANT_KIND_Q3_K,
    .name = "Q3_K",
    .block_size = CPU_QUANT_FORMAT_BLOCK_VALUES(MARMOT_QUANT_KIND_Q3_K),
    .block_bytes = CPU_QUANT_FORMAT_BLOCK_BYTES(MARMOT_QUANT_KIND_Q3_K),
    .weight_bits = 3,
    .has_zero_point = true,
    .requires_calibration = false,
    .layout = CPU_QUANT_FORMAT_LAYOUT(MARMOT_QUANT_KIND_Q3_K),
    .compute_params = nullptr,
    .quantize_block = q3_k_quantize_block,
    .dequantize_block = q3_k_dequantize_block,
    .vec_dot_block = nullptr,
};

MARMOT_REGISTER_QUANT_SCHEME(q3_k_traits)

static marmot_error_t
q3_k_quantize_block(const float *values, uint32_t count, void *block_out, [[maybe_unused]] const void *params) {
    if (values == nullptr || block_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = qk_clamp_elems(count);
    float block_values[QK_K];
    qk_copy_and_pad(values, elems, block_values);

    int8_t L[QK_K];
    float scales[QK_K / 16];
    marmot_q3_k_block_t *block = (marmot_q3_k_block_t *)block_out;
    memset(block, 0, sizeof(*block));

    float max_scale = 0.0f;
    float amax = 0.0f;
    for (int j = 0; j < QK_K / 16; ++j) {
        scales[j] = make_q3_quants(16, 4, block_values + 16 * j, L + 16 * j, true);
        float scale = fabsf(scales[j]);
        if (scale > amax) {
            amax = scale;
            max_scale = scales[j];
        }
    }

    memset(block->scales, 0, 12);
    if (max_scale != 0.0f) {
        float iscale = -32.0f / max_scale;
        for (int j = 0; j < QK_K / 16; ++j) {
            int l = (int)nearbyintf(iscale * scales[j]);
            l = l > 31 ? 31 : (l < -32 ? -32 : l);
            int shifted = l + 32;
            if (j < 8) {
                block->scales[j] = (uint8_t)(shifted & 0xF);
            } else {
                block->scales[j - 8] |= (uint8_t)((shifted & 0xF) << 4);
            }
            shifted >>= 4;
            block->scales[j % 4 + 8] |= (uint8_t)(shifted << (2 * (j / 4)));
        }
        block->d = marmot_native_to_float16((_Float16)(1.0f / iscale));
    } else {
        block->d = marmot_native_to_float16((_Float16)0.0f);
    }

    for (int j = 0; j < QK_K / 16; ++j) {
        int sc = j < 8 ? (block->scales[j] & 0xF) : (block->scales[j - 8] >> 4);
        sc = (sc | (((block->scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32;
        float d = (float)marmot_float16_to_native(block->d) * sc;
        if (d == 0) {
            continue;
        }
        for (int ii = 0; ii < 16; ++ii) {
            int l = (int)nearbyintf(block_values[16 * j + ii] / d);
            l = l > 3 ? 3 : (l < -4 ? -4 : l);
            L[16 * j + ii] = (int8_t)(l + 4);
        }
    }

    memset(block->hmask, 0, QK_K / 8);
    int m = 0;
    uint8_t hm = 1;
    for (int j = 0; j < QK_K; ++j) {
        if (L[j] > 3) {
            block->hmask[m] |= hm;
            L[j] -= 4;
        }
        if (++m == QK_K / 8) {
            m = 0;
            hm <<= 1;
        }
    }

    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; ++l) {
            block->qs[j / 4 + l] =
                (uint8_t)(L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6));
        }
    }

    if (elems < QK_K) {
        size_t qs_offset = (elems + 3) / 4;
        size_t hmask_offset = (elems + 7) / 8;
        if (qs_offset < sizeof(block->qs)) {
            memset(block->qs + qs_offset, 0, sizeof(block->qs) - qs_offset);
        }
        if (hmask_offset < sizeof(block->hmask)) {
            memset(block->hmask + hmask_offset, 0, sizeof(block->hmask) - hmask_offset);
        }
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t
q3_k_dequantize_block(const void *block_in, uint32_t count, float *values_out, [[maybe_unused]] const void *params) {
    if (block_in == nullptr || values_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = qk_clamp_elems(count);
    const marmot_q3_k_block_t *block = (const marmot_q3_k_block_t *)block_in;
    const float d_all = (float)marmot_float16_to_native(block->d);
    const uint8_t *q = block->qs;
    const uint8_t *hm = block->hmask;

    uint32_t aux[4];
    memcpy(aux, block->scales, 12);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t *scales = (const int8_t *)aux;

    size_t written = 0;
    int is = 0;
    uint8_t m = 1;
    for (size_t n = 0; n < QK_K && written < elems; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4 && written < elems; ++j) {
            float dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16 && written < elems; ++l, ++written) {
                values_out[written] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            }

            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16 && written < elems; ++l, ++written) {
                values_out[written] = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            }

            shift += 2;
            m <<= 1;
        }
        q += 32;
    }

    return MARMOT_SUCCESS;
}

marmot_error_t cpu_quantize_q3_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = nullptr;
    marmot_error_t err = require_k_traits(MARMOT_QUANT_KIND_Q3_K, &traits, "Q3_K traits not registered");
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return cpu_quantize_blockwise(traits, input, output);
}

marmot_error_t cpu_dequantize_q3_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = nullptr;
    marmot_error_t err = require_k_traits(MARMOT_QUANT_KIND_Q3_K, &traits, "Q3_K traits not registered");
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return cpu_dequantize_blockwise(traits, input, output);
}
