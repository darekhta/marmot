#pragma once

#include "marmot/tensor.h"

#include <memory>

namespace marmot::inference {

struct TensorDeleter {
    void operator()(marmot_tensor_t *ptr) const noexcept;
};

using TensorPtr = std::unique_ptr<marmot_tensor_t, TensorDeleter>;

} // namespace marmot::inference
