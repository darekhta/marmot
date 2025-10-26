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

static marmot_error_t q6_k_quantize_block(const float *values, uint32_t count, void *block_out, const void *params);
static marmot_error_t q6_k_dequantize_block(const void *block, uint32_t count, float *values_out, const void *params);

static const marmot_quant_traits_t q6_k_traits = {
    .kind = MARMOT_QUANT_KIND_Q6_K,
    .name = "Q6_K",
    .block_size = CPU_QUANT_FORMAT_BLOCK_VALUES(MARMOT_QUANT_KIND_Q6_K),
    .block_bytes = CPU_QUANT_FORMAT_BLOCK_BYTES(MARMOT_QUANT_KIND_Q6_K),
    .weight_bits = 6,
    .has_zero_point = false,
    .requires_calibration = false,
    .layout = CPU_QUANT_FORMAT_LAYOUT(MARMOT_QUANT_KIND_Q6_K),
    .compute_params = nullptr,
    .quantize_block = q6_k_quantize_block,
    .dequantize_block = q6_k_dequantize_block,
    .vec_dot_block = nullptr,
};

MARMOT_REGISTER_QUANT_SCHEME(q6_k_traits)

static marmot_error_t
q6_k_quantize_block(const float *values, uint32_t count, void *block_out, [[maybe_unused]] const void *params) {
    if (values == nullptr || block_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = qk_clamp_elems(count);
    float tmp[QK_K];
    qk_copy_and_pad(values, elems, tmp);

    marmot_q6_k_block_t *block = (marmot_q6_k_block_t *)block_out;
    memset(block, 0, sizeof(*block));
    quantize_row_q6_k_ref_single(tmp, block);

    if (elems < QK_K) {
        size_t ql_offset = (elems + 1) / 2;
        size_t qh_offset = (elems + 3) / 4;
        if (ql_offset < sizeof(block->ql)) {
            memset(block->ql + ql_offset, 0, sizeof(block->ql) - ql_offset);
        }
        if (qh_offset < sizeof(block->qh)) {
            memset(block->qh + qh_offset, 0, sizeof(block->qh) - qh_offset);
        }
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t
q6_k_dequantize_block(const void *block_in, uint32_t count, float *values_out, [[maybe_unused]] const void *params) {
    if (block_in == nullptr || values_out == nullptr || count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t elems = qk_clamp_elems(count);
    const marmot_q6_k_block_t *block = (const marmot_q6_k_block_t *)block_in;
    const float d = (float)marmot_float16_to_native(block->d);
    const uint8_t *ql = block->ql;
    const uint8_t *qh = block->qh;
    const int8_t *sc = block->scales;

    for (size_t n = 0; n < QK_K && n < elems; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

            if (n + l + 0 < elems)
                values_out[n + l + 0] = d * sc[is + 0] * q1;
            if (n + l + 32 < elems)
                values_out[n + l + 32] = d * sc[is + 2] * q2;
            if (n + l + 64 < elems)
                values_out[n + l + 64] = d * sc[is + 4] * q3;
            if (n + l + 96 < elems)
                values_out[n + l + 96] = d * sc[is + 6] * q4;
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }

    return MARMOT_SUCCESS;
}

marmot_error_t cpu_quantize_q6_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = nullptr;
    marmot_error_t err = require_k_traits(MARMOT_QUANT_KIND_Q6_K, &traits, "Q6_K traits not registered");
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return cpu_quantize_blockwise(traits, input, output);
}

marmot_error_t cpu_dequantize_q6_k(const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = nullptr;
    marmot_error_t err = require_k_traits(MARMOT_QUANT_KIND_Q6_K, &traits, "Q6_K traits not registered");
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return cpu_dequantize_blockwise(traits, input, output);
}
