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

static marmot_error_t q2_k_quantize_block(const float *values, uint32_t count, void *block_out, const void *params);
static marmot_error_t q2_k_dequantize_block(const void *block, uint32_t count, float *values_out, const void *params);

static const marmot_quant_traits_t q2_k_traits = {
    .kind = MARMOT_QUANT_KIND_Q2_K,
    .name = "Q2_K",
    .block_size = CPU_QUANT_FORMAT_BLOCK_VALUES(MARMOT_QUANT_KIND_Q2_K),
    .block_bytes = CPU_QUANT_FORMAT_BLOCK_BYTES(MARMOT_QUANT_KIND_Q2_K),
    .weight_bits = 2,
    .has_zero_point = true,
    .requires_calibration = false,
    .layout = CPU_QUANT_FORMAT_LAYOUT(MARMOT_QUANT_KIND_Q2_K),
    .compute_params = nullptr,
    .quantize_block = q2_k_quantize_block,
    .dequantize_block = q2_k_dequantize_block,
    .vec_dot_block = nullptr,
};

MARMOT_REGISTER_QUANT_SCHEME(q2_k_traits)

static marmot_error_t
q2_k_quantize_block(const float *values, uint32_t count, void *block_out, [[maybe_unused]] const void *params) {
    if (values == nullptr || block_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = qk_clamp_elems(count);
    float block_values[QK_K];
    qk_copy_and_pad(values, elems, block_values);

    uint8_t L[QK_K];
    memset(L, 0, sizeof(L));
    uint8_t Laux[16];
    float weights[16];
    float scales[QK_K / 16];
    float mins[QK_K / 16];

    marmot_q2_k_block_t *block = (marmot_q2_k_block_t *)block_out;
    memset(block, 0, sizeof(*block));

    float max_scale = 0.0f;
    float max_min = 0.0f;
    for (int j = 0; j < QK_K / 16; ++j) {
        for (int l = 0; l < 16; ++l) {
            weights[l] = fabsf(block_values[16 * j + l]);
        }
        scales[j] =
            make_qkx2_quants(16, 3, block_values + 16 * j, weights, L + 16 * j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
        if (scales[j] > max_scale) {
            max_scale = scales[j];
        }
        if (mins[j] > max_min) {
            max_min = mins[j];
        }
    }

    const float inv_scale = max_scale > 0 ? 15.0f / max_scale : 0.0f;
    const float inv_min = max_min > 0 ? 15.0f / max_min : 0.0f;
    const float d = inv_scale > 0 ? 1.0f / inv_scale : 0.0f;
    const float dm = inv_min > 0 ? 1.0f / inv_min : 0.0f;
    block->d = marmot_native_to_float16((_Float16)d);
    block->dmin = marmot_native_to_float16((_Float16)dm);

    for (int j = 0; j < QK_K / 16; ++j) {
        uint8_t ls = (uint8_t)nearbyintf(inv_scale * scales[j]);
        uint8_t lm = (uint8_t)nearbyintf(inv_min * mins[j]);
        ls = ls > 15 ? 15 : ls;
        lm = lm > 15 ? 15 : lm;
        block->scales[j] = ls | (lm << 4);
    }

    for (int j = 0; j < QK_K / 16; ++j) {
        uint8_t sc = block->scales[j] & 0xF;
        uint8_t m = block->scales[j] >> 4;
        const float dl = d * sc;
        const float ml = dm * m;

        if (dl != 0.0f) {
            for (int ii = 0; ii < 16; ++ii) {
                int l = (int)nearbyintf((block_values[16 * j + ii] + ml) / dl);
                L[16 * j + ii] = (uint8_t)(l > 3 ? 3 : (l < 0 ? 0 : l));
            }
        }
    }

    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; ++l) {
            block->qs[j / 4 + l] =
                (uint8_t)(L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6));
        }
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t
q2_k_dequantize_block(const void *block_in, uint32_t count, float *values_out, [[maybe_unused]] const void *params) {
    if (block_in == nullptr || values_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = qk_clamp_elems(count);
    const marmot_q2_k_block_t *block = (const marmot_q2_k_block_t *)block_in;
    const float d = (float)marmot_float16_to_native(block->d);
    const float min = (float)marmot_float16_to_native(block->dmin);
    const uint8_t *q = block->qs;

    size_t written = 0;
    int is = 0;
    for (size_t n = 0; n < QK_K && written < elems; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4 && written < elems; ++j) {
            uint8_t sc = block->scales[is++];
            float dl = d * (sc & 0xF);
            float ml = min * (sc >> 4);
            for (int l = 0; l < 16 && written < elems; ++l, ++written) {
                values_out[written] = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
            }

            sc = block->scales[is++];
            dl = d * (sc & 0xF);
            ml = min * (sc >> 4);
            for (int l = 0; l < 16 && written < elems; ++l, ++written) {
                values_out[written] = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;
            }

            shift += 2;
        }
        q += 32;
    }

    return MARMOT_SUCCESS;
}

marmot_error_t cpu_quantize_q2_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = nullptr;
    marmot_error_t err = require_k_traits(MARMOT_QUANT_KIND_Q2_K, &traits, "Q2_K traits not registered");
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return cpu_quantize_blockwise(traits, input, output);
}

marmot_error_t cpu_dequantize_q2_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = nullptr;
    marmot_error_t err = require_k_traits(MARMOT_QUANT_KIND_Q2_K, &traits, "Q2_K traits not registered");
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return cpu_dequantize_blockwise(traits, input, output);
}
