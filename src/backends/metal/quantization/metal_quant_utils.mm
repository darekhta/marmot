#include "metal_quant_utils.h"

#include "core/helpers/quant.h"

#ifdef __APPLE__

void metal_quant_compute_row_config(const marmot_tensor_t *tensor, uint32_t block_size, metal_quant_row_config_t *out) {
    if (tensor == nullptr || out == nullptr) {
        return;
    }

    marmot_quant_row_config_t cfg;
    if (!marmot_quant_compute_row_config(tensor, block_size, &cfg)) {
        out->num_rows = 0;
        out->row_size = 0;
        out->blocks_per_row = 0;
        out->num_blocks = 0;
        out->num_elements = 0;
        return;
    }

    out->num_rows = (uint32_t)cfg.num_rows;
    out->row_size = (uint32_t)cfg.row_size;
    out->blocks_per_row = (uint32_t)cfg.blocks_per_row;
    out->num_blocks = (uint32_t)cfg.num_blocks;
    out->num_elements = cfg.num_elements;
}

marmot_error_t metal_quant_acquire_buffers(
    metal_context_t *ctx, const marmot_tensor_t *input, size_t input_bytes, const marmot_tensor_t *output,
    size_t output_bytes, metal_quant_buffer_bundle_t *bundle
) {
    if (ctx == nullptr || bundle == nullptr || input == nullptr || output == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    id<MTLBuffer> buffer_input = metal_residency_acquire_existing(ctx, input, input->dtype);
    if (buffer_input == nil) {
        buffer_input = metal_buffer_acquire(ctx, input->data, input_bytes);
    }

    bool out_private = false;
    bool out_is_staging = false;
    id<MTLBuffer> buffer_output = metal_residency_acquire_compute(ctx, output, output->dtype, &out_is_staging);
    if (buffer_output == nil) {
        buffer_output = metal_buffer_acquire(ctx, output->data, output_bytes);
    } else {
        out_private = true;
    }

    if (buffer_input == nil || buffer_output == nil) {
        if (buffer_input != nil) {
            [buffer_input release];
        }
        if (buffer_output != nil) {
            [buffer_output release];
        }
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    bundle->input = buffer_input;
    bundle->output = buffer_output;
    bundle->output_private = out_private;
    return MARMOT_SUCCESS;
}

void metal_quant_release_buffers(
    metal_context_t *ctx, const marmot_tensor_t *output, metal_quant_buffer_bundle_t *bundle
) {
    if (bundle == nullptr) {
        return;
    }

    if (bundle->input != nil) {
        [bundle->input release];
        bundle->input = nil;
    }
    if (bundle->output != nil) {
        [bundle->output release];
        if (bundle->output_private && ctx != nullptr && output != nullptr) {
            metal_residency_mark_dirty(ctx, output, output->dtype);
        }
        bundle->output = nil;
    }
    bundle->output_private = false;
}

#endif // __APPLE__
