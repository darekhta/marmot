#include "graph_validator.hpp"

#include "marmot/tensor.h"

#include <cstddef>

namespace marmot::graph {

bool validate_desc(const marmot_graph_tensor_desc_t &desc) {
    if (desc.ndim == 0 || desc.ndim > MARMOT_MAX_DIMS)
        return false;
    if (desc.dtype < 0 || desc.dtype >= MARMOT_DTYPE_COUNT)
        return false;
    for (uint32_t i = 0; i < desc.ndim; ++i) {
        if (desc.shape[i] == 0)
            return false;
    }
    return true;
}

bool ensure_strides(marmot_graph_tensor_desc_t &desc) {
    if (desc.ndim == 0)
        return false;

    bool has_stride = false;
    bool has_zero = false;
    for (uint32_t i = 0; i < desc.ndim; ++i) {
        if (desc.strides[i] == 0) {
            has_zero = true;
        } else {
            has_stride = true;
        }
    }

    if (!has_stride) {
        desc.strides[desc.ndim - 1] = 1;
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(desc.ndim) - 2; d >= 0; --d) {
            desc.strides[d] = desc.strides[d + 1] * desc.shape[d + 1];
        }
        return true;
    }

    if (has_zero) {
        return false;
    }

    return true;
}

} // namespace marmot::graph
