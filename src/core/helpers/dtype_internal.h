#ifndef MARMOT_CORE_DTYPE_INTERNAL_H
#define MARMOT_CORE_DTYPE_INTERNAL_H

#include "marmot/device.h"
#include "marmot/quant_block.h"
#include "marmot/types.h"

#include <stddef.h>
#include <stdint.h>

static inline bool marmot_dtype_valid(marmot_dtype_t dtype) {
    return dtype >= 0 && dtype < MARMOT_DTYPE_COUNT;
}

static inline bool marmot_quant_kind_valid(marmot_quant_kind_t kind) {
    return kind >= 0 && kind < MARMOT_QUANT_KIND_COUNT;
}

#endif
