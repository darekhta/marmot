#include "graph_naming.hpp"

#include <format>

namespace marmot::graph {

std::string generate_input_name(marmot_value_id_t id) {
    return std::format("input_{}", id);
}

std::string generate_output_name(size_t node_index, size_t out_index) {
    return std::format("node_{}_out_{}", node_index, out_index);
}

} // namespace marmot::graph
