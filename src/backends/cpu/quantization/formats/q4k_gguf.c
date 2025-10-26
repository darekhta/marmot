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

static marmot_error_t q4_k_quantize_block(const float *values, uint32_t count, void *block_out, const void *params);
static marmot_error_t q4_k_dequantize_block(const void *block, uint32_t count, float *values_out, const void *params);

static const marmot_quant_traits_t q4_k_traits = {
    .kind = MARMOT_QUANT_KIND_Q4_K,
    .name = "Q4_K",
    .block_size = CPU_QUANT_FORMAT_BLOCK_VALUES(MARMOT_QUANT_KIND_Q4_K),
    .block_bytes = CPU_QUANT_FORMAT_BLOCK_BYTES(MARMOT_QUANT_KIND_Q4_K),
    .weight_bits = 4,
    .has_zero_point = true,
    .requires_calibration = false,
    .layout = CPU_QUANT_FORMAT_LAYOUT(MARMOT_QUANT_KIND_Q4_K),
    .compute_params = nullptr,
    .quantize_block = q4_k_quantize_block,
    .dequantize_block = q4_k_dequantize_block,
    .vec_dot_block = nullptr,
};

MARMOT_REGISTER_QUANT_SCHEME(q4_k_traits)

static marmot_error_t
q4_k_quantize_block(const float *values, uint32_t count, void *block_out, [[maybe_unused]] const void *params) {
    if (values == nullptr || block_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = qk_clamp_elems(count);
    float block_values[QK_K];
    qk_copy_and_pad(values, elems, block_values);

    uint8_t L[QK_K];
    memset(L, 0, sizeof(L));
    uint8_t Laux[32];
    float weights[32];
    float scales[QK_K / 32];
    float mins[QK_K / 32];
    marmot_q4_k_block_t *block = (marmot_q4_k_block_t *)block_out;
    memset(block, 0, sizeof(*block));

    float max_scale = 0.0f;
    float max_min = 0.0f;
    for (int j = 0; j < QK_K / 32; ++j) {
        float sum_x2 = 0.0f;
        for (int l = 0; l < 32; ++l) {
            float val = block_values[32 * j + l];
            sum_x2 += val * val;
        }
        float av_x = sqrtf(sum_x2 / 32.0f);
        for (int l = 0; l < 32; ++l) {
            weights[l] = av_x + fabsf(block_values[32 * j + l]);
        }
        scales[j] = make_qkx2_quants(
            32, 15, block_values + 32 * j, weights, L + 32 * j, &mins[j], Laux, -1.0f, 0.1f, 20, false
        );
        if (scales[j] > max_scale) {
            max_scale = scales[j];
        }
        if (mins[j] > max_min) {
            max_min = mins[j];
        }
    }

    float inv_scale = max_scale > 0 ? 63.0f / max_scale : 0.0f;
    float inv_min = max_min > 0 ? 63.0f / max_min : 0.0f;
    for (int j = 0; j < QK_K / 32; ++j) {
        uint8_t ls = (uint8_t)nearbyintf(inv_scale * scales[j]);
        uint8_t lm = (uint8_t)nearbyintf(inv_min * mins[j]);
        ls = ls > 63 ? 63 : ls;
        lm = lm > 63 ? 63 : lm;
        if (j < 4) {
            block->scales[j] = ls;
            block->scales[j + 4] = lm;
        } else {
            block->scales[j + 4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4));
            block->scales[j - 4] |= (uint8_t)((ls >> 4) << 6);
            block->scales[j - 0] |= (uint8_t)((lm >> 4) << 6);
        }
    }
    block->d = marmot_native_to_float16((_Float16)(max_scale / 63.0f));
    block->dmin = marmot_native_to_float16((_Float16)(max_min / 63.0f));

    for (int j = 0; j < QK_K / 32; ++j) {
        uint8_t sc, m;
        if (j < 4) {
            sc = block->scales[j] & 63;
            m = block->scales[j + 4] & 63;
        } else {
            sc = (block->scales[j + 4] & 0xF) | ((block->scales[j - 4] >> 6) << 4);
            m = (block->scales[j + 4] >> 4) | ((block->scales[j - 0] >> 6) << 4);
        }
        float d = (float)marmot_float16_to_native(block->d) * sc;
        if (d == 0) {
            continue;
        }
        float dm = (float)marmot_float16_to_native(block->dmin) * m;
        for (int ii = 0; ii < 32; ++ii) {
            int l = (int)nearbyintf((block_values[32 * j + ii] + dm) / d);
            l = l > 15 ? 15 : (l < 0 ? 0 : l);
            L[32 * j + ii] = (uint8_t)l;
        }
    }

    for (int j = 0; j < QK_K; j += 64) {
        for (int l = 0; l < 32; ++l) {
            block->qs[(j / 2) + l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 4));
        }
    }

    if (elems < QK_K) {
        size_t qs_offset = (elems + 1) / 2;
        if (qs_offset < sizeof(block->qs)) {
            memset(block->qs + qs_offset, 0, sizeof(block->qs) - qs_offset);
        }
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t
q4_k_dequantize_block(const void *block_in, uint32_t count, float *values_out, [[maybe_unused]] const void *params) {
    if (block_in == nullptr || values_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = qk_clamp_elems(count);
    const marmot_q4_k_block_t *block = (const marmot_q4_k_block_t *)block_in;
    const uint8_t *q = block->qs;
    const float d = (float)marmot_float16_to_native(block->d);
    const float min = (float)marmot_float16_to_native(block->dmin);

    size_t written = 0;
    int is = 0;
    for (size_t j = 0; j < QK_K && written < elems; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, block->scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, block->scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = min * m;
        for (int l = 0; l < 32 && written < elems; ++l, ++written) {
            values_out[written] = d1 * (q[l] & 0xF) - m1;
        }
        for (int l = 0; l < 32 && written < elems; ++l, ++written) {
            values_out[written] = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }

    return MARMOT_SUCCESS;
}

marmot_error_t cpu_quantize_q4_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = nullptr;
    marmot_error_t err = require_k_traits(MARMOT_QUANT_KIND_Q4_K, &traits, "Q4_K traits not registered");
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return cpu_quantize_blockwise(traits, input, output);
}

marmot_error_t cpu_dequantize_q4_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = nullptr;
    marmot_error_t err = require_k_traits(MARMOT_QUANT_KIND_Q4_K, &traits, "Q4_K traits not registered");
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return cpu_dequantize_blockwise(traits, input, output);
}
