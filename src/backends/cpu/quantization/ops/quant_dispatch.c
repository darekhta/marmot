#include "cpu_backend_internal.h"

static marmot_error_t
cpu_quant_validate_traits(const marmot_quant_traits_t *traits, marmot_quant_layout_t requested_layout) {
    if (traits == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Quantization kind is not registered on the CPU backend");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (traits->quantize_block == nullptr || traits->dequantize_block == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Quantization traits missing block handlers");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (requested_layout != MARMOT_QUANT_LAYOUT_GENERIC && requested_layout != traits->layout) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Quantization layout mismatch");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_quantize_with_kind(
    const void *device_ctx, marmot_quant_kind_t kind, marmot_quant_layout_t layout, const marmot_tensor_t *input,
    marmot_tensor_t *output
) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = marmot_get_quant_traits(kind);
    marmot_error_t status = cpu_quant_validate_traits(traits, layout);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return cpu_quantize_blockwise(traits, input, output);
}

marmot_error_t cpu_dequantize_with_kind(
    const void *device_ctx, marmot_quant_kind_t kind, marmot_quant_layout_t layout, const marmot_tensor_t *input,
    marmot_tensor_t *output
) {
    (void)device_ctx;
    const marmot_quant_traits_t *traits = marmot_get_quant_traits(kind);
    marmot_error_t status = cpu_quant_validate_traits(traits, layout);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return cpu_dequantize_blockwise(traits, input, output);
}
