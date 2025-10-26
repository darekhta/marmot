#include "quant.h"

#include "marmot/error.h"

bool marmot_quant_compute_row_config(
    const marmot_tensor_t *tensor, size_t block_size, marmot_quant_row_config_t *out_config
) {
    if (tensor == nullptr || out_config == nullptr || block_size == 0) {
        return false;
    }

    marmot_quant_row_config_t cfg = {
        .num_rows = 1,
        .row_size = marmot_tensor_num_elements(tensor),
        .blocks_per_row = 0,
        .num_blocks = 0,
        .num_elements = marmot_tensor_num_elements(tensor),
    };

    if (tensor->shape.ndim == 2) {
        cfg.num_rows = tensor->shape.shape[0];
        cfg.row_size = tensor->shape.shape[1];
    }

    cfg.blocks_per_row = (cfg.row_size + block_size - 1) / block_size;
    if (cfg.blocks_per_row == 0) {
        cfg.blocks_per_row = 1;
    }

    if (tensor->shape.ndim == 2) {
        cfg.num_blocks = cfg.blocks_per_row * cfg.num_rows;
    } else {
        cfg.num_blocks = (cfg.num_elements + block_size - 1) / block_size;
        if (cfg.num_blocks == 0) {
            cfg.num_blocks = 1;
        }
    }

    *out_config = cfg;
    return true;
}
