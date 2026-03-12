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
    uint32_t fast_block_id{UINT32_MAX};
    uint32_t paged_attention_layer_idx{0};
    marmot_fast_stage_hint_t fast_stage_hint{MARMOT_FAST_STAGE_HINT_NONE};
    marmot_fast_node_role_t fast_node_role{MARMOT_FAST_NODE_ROLE_NONE};
    marmot_ffn_type_t moe_ffn_type{MARMOT_FFN_COUNT};
    float moe_weights_scale{1.0f};
    marmot_router_weight_policy_t moe_router_weight_policy{MARMOT_ROUTER_WEIGHT_POLICY_COUNT};
    std::array<size_t, MARMOT_MAX_DIMS> slice_starts{};
};

struct ExecutionCommand {
    enum class Kind { Launch } kind = Kind::Launch;
    uint32_t node_index{0};
};

} // namespace marmot::graph
