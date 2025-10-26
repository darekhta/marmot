#pragma once

#include <vector>

#include "graph_node.hpp"
#include "graph_value.hpp"

namespace marmot::graph {

[[nodiscard]] bool infer_matmul_signature(const std::vector<GraphValue> &values, GraphNode &node);
[[nodiscard]] bool populate_signature(const std::vector<GraphValue> &values, GraphNode &node);

} // namespace marmot::graph
