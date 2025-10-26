#pragma once

#include "marmot/error.h"
#include "marmot/tensor.h"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace marmot::graph {

struct TensorDeleter {
    void operator()(marmot_tensor_t *ptr) const;
};

class BindingTable {
  public:
    explicit BindingTable(size_t size);

    [[nodiscard]] std::span<marmot_tensor_t *> bindings();
    void set(size_t index, marmot_tensor_t *ptr);
    [[nodiscard]] marmot_error_t emplace_owned(size_t index, marmot_tensor_t *ptr);

  private:
    std::vector<marmot_tensor_t *> bindings_;
    std::vector<std::unique_ptr<marmot_tensor_t, TensorDeleter>> owned_;
};

} // namespace marmot::graph
