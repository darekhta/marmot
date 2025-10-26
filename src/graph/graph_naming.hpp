#pragma once

#include "marmot/graph/graph_types.h"

#include <string>

namespace marmot::graph {

[[nodiscard]] std::string generate_input_name(marmot_value_id_t id);
[[nodiscard]] std::string generate_output_name(size_t node_index, size_t out_index);

} // namespace marmot::graph
