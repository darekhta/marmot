#pragma once

#include "marmot/graph/graph_types.h"

namespace marmot::graph {

[[nodiscard]] bool validate_desc(const marmot_graph_tensor_desc_t &desc);
[[nodiscard]] bool ensure_strides(marmot_graph_tensor_desc_t &desc);

} // namespace marmot::graph
