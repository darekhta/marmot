#pragma once

#include "marmot/graph/graph_types.h"
#include "marmot/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace marmot::graph {

inline constexpr uint32_t kNodeIndexInvalid = UINT32_MAX;

struct GraphValue {
    marmot_graph_tensor_desc_t desc{};
    bool is_input{false};
    bool is_constant{false};
    std::string name{};
    marmot_tensor_t *constant_tensor{nullptr};
    uint32_t defining_node{kNodeIndexInvalid};
    std::vector<uint32_t> uses{};
};

} // namespace marmot::graph
