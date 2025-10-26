#ifndef MARMOT_CORE_HELPERS_QUANT_H
#define MARMOT_CORE_HELPERS_QUANT_H

#include "marmot/device.h"
#include "marmot/op_metadata.gen.h"
#include "marmot/quant_traits.h"
#include "marmot/tensor.h"
#include "marmot/traits_ids.gen.h"
#include "marmot/types.h"

#include <stdbool.h>
#include <stddef.h>

typedef struct {
    size_t num_rows;
    size_t row_size;
    size_t blocks_per_row;
    size_t num_blocks;
    size_t num_elements;
} marmot_quant_row_config_t;

#ifdef __cplusplus
extern "C" {
#endif

bool marmot_quant_compute_row_config(
    const marmot_tensor_t *tensor, size_t block_size, marmot_quant_row_config_t *out_config
);

#ifdef __cplusplus
}
#endif

static inline marmot_qscheme_id_t marmot_quant_kind_to_qscheme(marmot_quant_kind_t kind) {
    return marmot_op_metadata_quant_kind_to_qscheme(kind);
}

static inline marmot_quant_layout_t marmot_quant_kind_to_layout(marmot_quant_kind_t kind) {
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(kind);
    if (traits == nullptr) {
        return MARMOT_QUANT_LAYOUT_GENERIC;
    }
    return traits->layout;
}

static inline bool
marmot_quant_storage_dtype_compatible(const marmot_quant_kind_traits_t *traits, marmot_dtype_t dtype) {
    if (traits == nullptr) {
        return false;
    }
    if (dtype == traits->storage_dtype) {
        return true;
    }
    const bool traits_is_int8 =
        traits->storage_dtype == MARMOT_DTYPE_INT8 || traits->storage_dtype == MARMOT_DTYPE_UINT8;
    const bool dtype_is_int8 = dtype == MARMOT_DTYPE_INT8 || dtype == MARMOT_DTYPE_UINT8;
    return traits_is_int8 && dtype_is_int8;
}

static inline bool
marmot_matmul_quant_should_force_pack(marmot_quant_activation_mode_t activation_mode, bool backend_force_hint) {
    if (activation_mode == MARMOT_QUANT_ACTIVATION_FORCE_DIRECT) {
        return false;
    }
    return backend_force_hint;
}

#endif // MARMOT_CORE_HELPERS_QUANT_H
