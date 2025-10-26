#pragma once

#include "marmot/graph/graph_types.h"
#include "marmot/graph/op_signature.h"
#include "marmot/traits_ids.gen.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "core/bytecode/bytecode.h"

namespace marmot::graph {

struct GraphNode {
    std::string op_name{};
    marmot_op_signature_t signature{};
    std::vector<marmot_value_id_t> inputs{};
    std::vector<marmot_value_id_t> outputs{};
    marmot_kernel_id_t kernel_id{MARMOT_KERNEL_INVALID};
    uint16_t bc_op_index{MARMOT_BC_OP_INVALID};
    uint32_t rope_params_offset{MARMOT_BC_INVALID_OFFSET};
    float estimated_us{0.0f};
    bool skip{false};
    size_t view_byte_offset{0};
    uint32_t paged_attention_layer_idx{0};
    std::array<size_t, MARMOT_MAX_DIMS> slice_starts{};
};

struct ExecutionCommand {
    enum class Kind { Launch } kind = Kind::Launch;
    uint32_t node_index{0};
};

} // namespace marmot::graph
