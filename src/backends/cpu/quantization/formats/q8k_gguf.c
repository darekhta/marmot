#include "marmot/error.h"
#include "marmot/macros.h"
#include "marmot/quant_block.h"
#include "marmot/quant_traits.h"
#include "marmot/tensor.h"

#include <float.h>
#include <math.h>
#include <string.h>

#include "../common/kquant_common.h"
#include "../common/quant_utils.h"
#include "cpu_backend_internal.h"
#include "quantization/format_metadata.h"

static marmot_error_t q8_k_quantize_block(const float *values, uint32_t count, void *block_out, const void *params);
static marmot_error_t q8_k_dequantize_block(const void *block, uint32_t count, float *values_out, const void *params);

static const marmot_quant_traits_t q8_k_traits = {
    .kind = MARMOT_QUANT_KIND_Q8_K,
    .name = "Q8_K",
    .block_size = CPU_QUANT_FORMAT_BLOCK_VALUES(MARMOT_QUANT_KIND_Q8_K),
    .block_bytes = CPU_QUANT_FORMAT_BLOCK_BYTES(MARMOT_QUANT_KIND_Q8_K),
    .weight_bits = 8,
    .has_zero_point = false,
    .requires_calibration = false,
    .layout = CPU_QUANT_FORMAT_LAYOUT(MARMOT_QUANT_KIND_Q8_K),
    .compute_params = nullptr,
    .quantize_block = q8_k_quantize_block,
    .dequantize_block = q8_k_dequantize_block,
    .vec_dot_block = nullptr,
};

MARMOT_REGISTER_QUANT_SCHEME(q8_k_traits)

static marmot_error_t
q8_k_quantize_block(const float *values, uint32_t count, void *block_out, [[maybe_unused]] const void *params) {
    if (values == nullptr || block_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = qk_clamp_elems(count);
    float block_values[QK_K];
    qk_copy_and_pad(values, elems, block_values);

    marmot_q8_k_block_t *block = (marmot_q8_k_block_t *)block_out;
    memset(block, 0, sizeof(*block));

    const marmot_quant_signed_scale_t scale = marmot_quant_prepare_signed_scale(block_values, elems, 127.0f);
    if (scale.is_zero) {
        block->d = 0.0f;
        return MARMOT_SUCCESS;
    }

    block->d = scale.scale;
    marmot_quant_store_symmetric_int8(block_values, elems, scale.inv_scale, block->qs);

    for (uint32_t i = 0; i < QK_K / 16; ++i) {
        int sum = 0;
        for (uint32_t j = 0; j < 16; ++j) {
            sum += block->qs[i * 16 + j];
        }
        block->bsums[i] = (int16_t)sum;
    }

    if (elems < QK_K) {
        memset(block->qs + elems, 0, (QK_K - elems) * sizeof(int8_t));
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t
q8_k_dequantize_block(const void *block_in, uint32_t count, float *values_out, [[maybe_unused]] const void *params) {
    if (block_in == nullptr || values_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = qk_clamp_elems(count);
    const marmot_q8_k_block_t *block = (const marmot_q8_k_block_t *)block_in;
    const float d = block->d;
    for (uint32_t i = 0; i < elems; ++i) {
        values_out[i] = (float)block->qs[i] * d;
    }

    return MARMOT_SUCCESS;
}

marmot_error_t cpu_quantize_q8_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = nullptr;
    marmot_error_t err = require_k_traits(MARMOT_QUANT_KIND_Q8_K, &traits, "Q8_K traits not registered");
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return cpu_quantize_blockwise(traits, input, output);
}

marmot_error_t cpu_dequantize_q8_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = nullptr;
    marmot_error_t err = require_k_traits(MARMOT_QUANT_KIND_Q8_K, &traits, "Q8_K traits not registered");
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return cpu_dequantize_blockwise(traits, input, output);
}
